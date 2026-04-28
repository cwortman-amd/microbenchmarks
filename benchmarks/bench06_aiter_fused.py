"""Family 6 — AITER fused multi-GPU compute+collective kernels.

AITER-only benchmark for fused AG+MM / MM+RS kernels. This keeps the
legacy report path (`06_multigpu_fused/fused.json`) stable while
separating torch Symmetric Memory into `bench10_symm_fused.py`.

The probe order goes:

  1. Upstream AITER namespaces (``aiter.ops``, ``aiter.fused_collective``,
     ``aiter.distributed``) — preferred path once the kernels in
     ``benchmarks/aiter_kernels/`` are upstreamed into AITER itself.
  2. ``aiter.ops.triton.comms.fused.fused_*`` — same kernels, addressed at
     their canonical AITER path (this is the directory we wrote against).
  3. ``benchmarks.aiter_kernels`` — the vendored, AITER-conformant copy of
     the kernels in this repo. Always probed last so a real AITER install
     wins, but ensures the bench has a real fused-kernel impl to time
     even on hosts that haven't picked up the upstream yet.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


SHAPES = [
    (4096, 4096, 4096),
    (8192, 4096, 4096),
    (8192, 8192, 4096),
    (16384, 4096, 4096),
]


def _maybe_import(modname: str):
    try:
        return importlib.import_module(modname)
    except (ImportError, RuntimeError):
        return None


def _probe_aiter_api() -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve a fused AG+MM / MM+RS backend.

    Returns ``({source, ag_fn, rs_fn, call_kind}, label)`` or ``(None, reason)``.

    ``call_kind`` is either:

      * ``"symm_mem"`` — kernels accept the SymmMem signature
        ``ag(A_shard, [B], gather_dim, group_name)`` and
        ``rs(A, B, reduce_op, scatter_dim, group_name)``. This matches
        upstream torch and our new ``benchmarks.aiter_kernels`` module.
      * ``"legacy_positional"`` — kernels accept ``(A, B)`` positional only;
        used by the legacy ``aiter.ops`` / ``aiter.distributed`` candidates
        in case those ever ship.
    """
    tried: List[str] = []

    legacy_candidates: List[Tuple[str, str, str, str]] = [
        ("aiter.ops", "fused_all_gather_matmul", "fused_matmul_reduce_scatter", "aiter.ops"),
        ("aiter.fused_collective", "ag_matmul", "mm_reduce_scatter", "aiter.fused_collective"),
        ("aiter.distributed", "all_gather_matmul", "matmul_reduce_scatter", "aiter.distributed"),
    ]
    for modname, ag_name, rs_name, label in legacy_candidates:
        m = _maybe_import(modname)
        if m is None:
            tried.append(f"{modname}: import failed")
            continue
        ag_fn = getattr(m, ag_name, None)
        rs_fn = getattr(m, rs_name, None)
        if callable(ag_fn) and callable(rs_fn):
            return (
                {"source": label, "ag_fn": ag_fn, "rs_fn": rs_fn, "call_kind": "legacy_positional"},
                label,
            )
        tried.append(f"{modname}: ag={'ok' if ag_fn else 'missing'} rs={'ok' if rs_fn else 'missing'}")

    # Canonical AITER path that matches the directory we wrote against in
    # ``benchmarks/aiter_kernels/README.md §1.1``. Same signature as the
    # vendored kernels (SymmMem-style).
    ag_mod = _maybe_import("aiter.ops.triton.comms.fused.fused_all_gather_matmul")
    rs_mod = _maybe_import("aiter.ops.triton.comms.fused.fused_matmul_reduce_scatter")
    if ag_mod is not None and rs_mod is not None:
        ag_fn = getattr(ag_mod, "fused_all_gather_matmul", None)
        rs_fn = getattr(rs_mod, "fused_matmul_reduce_scatter", None)
        if callable(ag_fn) and callable(rs_fn):
            label = "aiter.ops.triton.comms.fused"
            return (
                {"source": label, "ag_fn": ag_fn, "rs_fn": rs_fn, "call_kind": "symm_mem"},
                label,
            )
        tried.append(
            f"aiter.ops.triton.comms.fused: ag={'ok' if ag_fn else 'missing'} "
            f"rs={'ok' if rs_fn else 'missing'}"
        )
    else:
        tried.append("aiter.ops.triton.comms.fused: import failed")

    # Vendored, AITER-conformant kernels in this repo. Always probed last so
    # a real upstream AITER install wins, but always available on CUDA hosts.
    local = _maybe_import("benchmarks.aiter_kernels")
    if local is not None:
        ag_fn = getattr(local, "fused_all_gather_matmul", None)
        rs_fn = getattr(local, "fused_matmul_reduce_scatter", None)
        if callable(ag_fn) and callable(rs_fn):
            return (
                {
                    "source": "benchmarks.aiter_kernels",
                    "ag_fn": ag_fn,
                    "rs_fn": rs_fn,
                    "call_kind": "symm_mem",
                },
                "benchmarks.aiter_kernels",
            )
        tried.append(
            f"benchmarks.aiter_kernels: ag={'ok' if ag_fn else 'missing'} "
            f"rs={'ok' if rs_fn else 'missing'}"
        )
    else:
        tried.append("benchmarks.aiter_kernels: import failed")

    return None, "; ".join(tried) or "no AITER candidate modules resolved"


def _is_distributed_env() -> bool:
    return all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"))


def _setup_distributed() -> Tuple[int, int, torch.device, str, bool]:
    has_gpu = torch.cuda.is_available()
    backend = "nccl" if has_gpu else "gloo"
    distributed = False
    if _is_distributed_env():
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        distributed = True
    else:
        rank = 0
        world = 1
    if has_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world, device, backend, distributed


def _bench_ag_mm(backend: Dict[str, Any], world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int, group_name: Optional[str]) -> Dict:
    """Bench fused AG+MM. ``M`` is the **global** M; we shard by ``world`` per rank.

    Shapes:
      A_shard:  [M / world, K]  per rank
      B:        [K, N]          replicated
      Y:        [M, N]          full output (each rank gets it)

    Throughput accounting:
      flops = 2 * M * K * N        (one full GEMM)
      ag_wire = (world-1) * (M/world) * K * dtype_bytes  (this rank's outbound bytes)
    """
    if M % world:
        M = (M // world) * world
        if M == 0:
            raise RuntimeError("AG+MM needs M >= world")
    M_shard = M // world
    A_shard = torch.empty(M_shard, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
    ag_fn = backend["ag_fn"]

    if backend["call_kind"] == "symm_mem":
        def fn():
            ag_fn(A_shard, [B], gather_dim=0, group_name=group_name)
    else:
        # Legacy positional API: caller signature was (A_local, B). We pass
        # the per-rank shard and the replicated B; legacy AITER ops are
        # responsible for their own AG.
        def fn():
            ag_fn(A_shard, B)

    dist.barrier()
    res = time_op(f"ag_mm_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K * N
    ag_wire_bytes = (world - 1) * M_shard * K * 2  # bf16 = 2B
    return {
        "op": "ag_mm",
        "world": world, "M": M, "M_shard": M_shard, "K": K, "N": N,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "ag_gb_s": ag_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def _bench_mm_rs(backend: Dict[str, Any], world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int, group_name: Optional[str]) -> Dict:
    """Bench fused MM+RS. ``M`` is the **global** M; output shard is M/world.

    Shapes:
      A:        [M, K]            replicated
      B:        [K, N]            replicated
      Y_shard:  [M / world, N]    per-rank output

    Throughput accounting:
      flops = 2 * M * K * N
      rs_wire = (world-1) * M_shard * N * dtype_bytes (per-rank inbound)
    """
    if M % world:
        M = (M // world) * world
        if M == 0:
            raise RuntimeError("MM+RS needs M >= world")
    M_shard = M // world
    A = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
    rs_fn = backend["rs_fn"]

    if backend["call_kind"] == "symm_mem":
        def fn():
            rs_fn(A, B, "avg", scatter_dim=0, group_name=group_name)
    else:
        def fn():
            rs_fn(A, B)

    dist.barrier()
    res = time_op(f"mm_rs_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K * N
    rs_wire_bytes = (world - 1) * M_shard * N * 2
    return {
        "op": "mm_rs",
        "world": world, "M": M, "M_shard": M_shard, "K": K, "N": N,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "rs_gb_s": rs_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def _write_skip(out_dir: Path, reason: str, *, world: int, backend: str,
                device_type: str) -> None:
    payload = {
        "available": False,
        "reason": reason,
        "world": world,
        "backend": backend,
        "device_type": device_type,
        "rows": [],
        "_note": (
            "AITER fused AG+MM / MM+RS not available in this build. "
            "Re-run after AITER ships fused-collectives support."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fused.json", payload)
    write_csv(out_dir / "fused.csv", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", type=str, default=None,
                    help="Optional override: 'M,K,N;M,K,N;...'")
    args = ap.parse_args()

    rank, world, device, backend, distributed = _setup_distributed()
    has_gpu = device.type == "cuda"
    out_dir = Path(args.out) / "06_multigpu_fused"

    def _maybe_barrier_and_destroy() -> None:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    if not has_gpu:
        if rank == 0:
            _write_skip(out_dir, "CPU host: no fused-collective+GEMM kernel exists.",
                        world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0

    if world < 2:
        if rank == 0:
            _write_skip(out_dir, f"world={world}: fused AG+MM / MM+RS need world>=2.",
                        world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0

    backend_info, source = _probe_aiter_api()
    if backend_info is None:
        if rank == 0:
            _write_skip(out_dir, f"no fused-collective API found ({source})",
                        world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0

    if args.shapes:
        try:
            shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";") if s]
        except ValueError as e:
            raise SystemExit(f"--shapes parse failed: {e}")
    else:
        shapes = SHAPES

    group_name = (
        dist.group.WORLD.group_name if (distributed and backend_info["call_kind"] == "symm_mem")  # type: ignore[union-attr]
        else None
    )

    rows: List[Dict] = []
    for (M, K, N) in shapes:
        for label in ("ag_mm", "mm_rs"):
            try:
                if label == "ag_mm":
                    row = _bench_ag_mm(backend_info, world, device, M, K, N,
                                       args.warmup, args.iters, group_name)
                else:
                    row = _bench_mm_rs(backend_info, world, device, M, K, N,
                                       args.warmup, args.iters, group_name)
                row["api_source"] = source
                row["call_kind"] = backend_info["call_kind"]
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "op": label, "world": world, "M": M, "K": K, "N": N,
                    "error": repr(e), "api_source": source,
                    "call_kind": backend_info["call_kind"],
                })

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "available": True,
            "api_source": source,
            "world": world,
            "backend": backend,
            "device_type": device.type,
            "rows": rows,
        }
        write_json(out_dir / "fused.json", payload)
        write_csv(out_dir / "fused.csv", rows)

    _maybe_barrier_and_destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

