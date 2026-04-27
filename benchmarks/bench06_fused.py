"""Family 6 — Fused multi-GPU compute+collective kernels (AG+MM, MM+RS).

The source PDF flags fused collective+GEMM kernels (``all_gather`` + matmul
and matmul + ``reduce_scatter``) as a future-work item that AITER is
expected to ship. This benchmark scaffolds the measurement now so:

  * if the fused API is **already** available in this build of AITER, we
    measure TFLOP/s and effective bytes/s and produce report-ready
    numbers;
  * if it's **not** available, we record ``available=False`` with the
    reason, the report shows a SKIP bullet, and the same script picks
    up the new kernels automatically the moment AITER lands them.

We probe for several plausible API surfaces in the AITER namespace:

  * ``aiter.ops.fused_all_gather_matmul`` (and ``fused_matmul_reduce_scatter``)
  * ``aiter.fused_collective.ag_matmul`` / ``mm_reduce_scatter``
  * ``aiter.distributed.all_gather_matmul`` / ``matmul_reduce_scatter``

If none of these resolve, we also probe the equivalent torch-native
candidates (``torch.distributed._functional_collectives.fused_*``) for
forward-compatibility with upstream PyTorch's fused-collective track.

Launch (GPU):
    torchrun --nproc_per_node=8 benchmarks/bench06_fused.py --out results/<id>/

Launch (CPU): no fused kernels exist on CPU; the script writes
``available=false`` and exits 0 so it never blocks a CPU campaign.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


# Micro-shapes representative of TP-sharded GEMMs in the workload. Each
# tuple is (M, K, N) — the per-rank matrix-multiply portion. The
# all-gather (AG+MM) variant gathers along the K dim across ranks, so
# the global K is K * world; the reduce-scatter (MM+RS) variant scatters
# along the M dim, so the local output is M / world. Same shape pattern
# the existing AITER fused-collective design docs reference.
SHAPES = [
    # (M, K, N)
    (4096,   4096,  4096),
    (8192,   4096,  4096),
    (8192,   8192,  4096),
    (16384,  4096,  4096),
]


def _maybe_import(modname: str):
    try:
        return importlib.import_module(modname)
    except (ImportError, RuntimeError):
        return None


def _probe_api() -> Tuple[Optional[Callable], Optional[Callable], str]:
    """Return (ag_mm_fn, mm_rs_fn, source_label) or (None, None, why).

    Walks the candidate API surfaces in priority order. Returns the
    first pair that resolves with **both** sides present — partial APIs
    are treated as not-available so the SC-row stays clean (we don't
    want a half-fused result snuck in as PASS).
    """
    candidates: List[Tuple[str, str, str, str]] = [
        ("aiter.ops",                  "fused_all_gather_matmul",
                                       "fused_matmul_reduce_scatter", "aiter.ops"),
        ("aiter.fused_collective",     "ag_matmul",
                                       "mm_reduce_scatter",          "aiter.fused_collective"),
        ("aiter.distributed",          "all_gather_matmul",
                                       "matmul_reduce_scatter",      "aiter.distributed"),
        # PyTorch native candidates (functional collectives track)
        ("torch.distributed._functional_collectives",
                                       "fused_all_gather_matmul",
                                       "fused_matmul_reduce_scatter",
                                       "torch._functional_collectives"),
    ]
    tried: List[str] = []
    for modname, ag_name, rs_name, label in candidates:
        m = _maybe_import(modname)
        if m is None:
            tried.append(f"{modname}: import failed")
            continue
        ag_fn = getattr(m, ag_name, None)
        rs_fn = getattr(m, rs_name, None)
        if callable(ag_fn) and callable(rs_fn):
            return ag_fn, rs_fn, label
        tried.append(
            f"{modname}: ag={'ok' if ag_fn else 'missing'} "
            f"rs={'ok' if rs_fn else 'missing'}"
        )
    return None, None, "; ".join(tried) or "no candidate modules resolved"


def _is_distributed_env() -> bool:
    """True iff a torchrun-style launcher set up the rendezvous env."""
    return all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"))


def _setup_distributed() -> Tuple[int, int, torch.device, str, bool]:
    """Returns (rank, world, device, backend, distributed_initialized)."""
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


def _bench_ag_mm(ag_mm_fn: Callable, world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int) -> Dict:
    """Each rank holds A_local[M, K]; the fused op gathers along K to make
    A[M, K*world] then matmuls with B[K*world, N] returning C[M, N]."""
    A_local = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K * world, N, dtype=torch.bfloat16, device=device).normal_()

    def fn():
        ag_mm_fn(A_local, B)

    dist.barrier()
    res = time_op(f"ag_mm_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * (K * world) * N
    # Bytes: A on the wire = (world-1)/world * M*K*world * 2 (gather), B local read,
    # C local write. We report the on-wire AG bytes because that's the bandwidth
    # the fused kernel is actually shrinking via overlap.
    ag_wire_bytes = (world - 1) * M * K * 2
    return {
        "op":           "ag_mm",
        "world":        world, "M": M, "K": K, "N": N,
        "t_ms":         res.median_ms,
        "tflops":       flops / (res.median_ms * 1e-3) / 1e12,
        "ag_gb_s":      ag_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def _bench_mm_rs(mm_rs_fn: Callable, world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int) -> Dict:
    """Each rank holds A_local[M, K]; the fused op matmuls A with B[K, N*world]
    then reduce-scatters along M, returning C_local[M/world, N*world]."""
    if M % world:
        # MM+RS conventionally needs M divisible by world; round down.
        M = (M // world) * world
        if M == 0:
            raise RuntimeError("MM+RS needs M >= world")
    A_local = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N * world, dtype=torch.bfloat16, device=device).normal_()

    def fn():
        mm_rs_fn(A_local, B)

    dist.barrier()
    res = time_op(f"mm_rs_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K * (N * world)
    # On-wire RS bytes per rank: (world-1)/world * M * N*world * 2 ≈ (world-1) * M * N * 2
    rs_wire_bytes = (world - 1) * M * N * 2
    return {
        "op":           "mm_rs",
        "world":        world, "M": M, "K": K, "N": N,
        "t_ms":         res.median_ms,
        "tflops":       flops / (res.median_ms * 1e-3) / 1e12,
        "rs_gb_s":      rs_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def _write_skip(out_dir: Path, reason: str, *, world: int, backend: str,
                device_type: str) -> None:
    payload = {
        "available":  False,
        "reason":     reason,
        "world":      world,
        "backend":    backend,
        "device_type": device_type,
        "rows":       [],
        "_note":      ("AITER fused AG+MM / MM+RS not available in this "
                       "build. Re-run after AITER ships the fused-"
                       "collectives API and this benchmark will produce "
                       "TFLOP/s and bytes/s numbers automatically."),
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
                    help="Optional override: 'M,K,N;M,K,N;...' for the "
                         "(M,K,N) micro-shape sweep. Default is the "
                         "workload-representative SHAPES list.")
    args = ap.parse_args()

    rank, world, device, backend, distributed = _setup_distributed()
    has_gpu = device.type == "cuda"
    out_dir = Path(args.out) / "06_multigpu_fused"

    def _maybe_barrier_and_destroy() -> None:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    # CPU short-circuit: no plausible fused-collective+GEMM kernels exist
    # on CPU. Write the not-available stub and exit 0.
    if not has_gpu:
        if rank == 0:
            _write_skip(out_dir,
                        reason="CPU host: no fused-collective+GEMM kernel exists.",
                        world=world, backend=backend, device_type=device.type)
            print("[06f] CPU host — writing not-available stub.")
        _maybe_barrier_and_destroy()
        return 0

    if world < 2:
        if rank == 0:
            _write_skip(out_dir,
                        reason=f"world={world}: fused AG+MM / MM+RS need world>=2.",
                        world=world, backend=backend, device_type=device.type)
            print(f"[06f] world={world} — writing not-available stub.")
        _maybe_barrier_and_destroy()
        return 0

    ag_fn, rs_fn, source = _probe_api()
    if ag_fn is None or rs_fn is None:
        if rank == 0:
            _write_skip(out_dir,
                        reason=f"no fused-collective API found ({source})",
                        world=world, backend=backend, device_type=device.type)
            print(f"[06f] not available: {source}")
        _maybe_barrier_and_destroy()
        return 0

    if args.shapes:
        try:
            shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";") if s]
        except ValueError as e:
            raise SystemExit(f"--shapes parse failed: {e}")
    else:
        shapes = SHAPES

    if rank == 0:
        print(f"[06f] backend={backend} world={world} device={device.type} "
              f"api_source={source} shapes={shapes}")

    rows: List[Dict] = []
    for (M, K, N) in shapes:
        for label, fn_pair in (("ag_mm", (_bench_ag_mm, ag_fn)),
                               ("mm_rs", (_bench_mm_rs, rs_fn))):
            bench_fn, kernel_fn = fn_pair
            try:
                row = bench_fn(kernel_fn, world, device, M, K, N,
                               args.warmup, args.iters)
                row["api_source"] = source
                rows.append(row)
                if rank == 0:
                    print(f"[06f] {label:5s} M={M:6d} K={K:5d} N={N:5d} "
                          f"t={row['t_ms']:7.2f} ms "
                          f"tflops={row['tflops']:7.1f} "
                          f"wirebw={(row.get('ag_gb_s') or row.get('rs_gb_s')):6.1f} GB/s")
            except Exception as e:  # noqa: BLE001
                if rank == 0:
                    print(f"[06f] {label} ({M},{K},{N}) failed: {e!r}")
                rows.append({
                    "op": label, "world": world, "M": M, "K": K, "N": N,
                    "error": repr(e), "api_source": source,
                })

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "available":   True,
            "api_source":  source,
            "world":       world,
            "backend":     backend,
            "device_type": device.type,
            "rows":        rows,
        }
        write_json(out_dir / "fused.json", payload)
        write_csv(out_dir / "fused.csv", rows)

    _maybe_barrier_and_destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
