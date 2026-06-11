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
  3. ``benchmarks.aiter_kernels.hipkittens`` — experimental HK extension when
     explicitly built/exported.
  4. ``benchmarks.aiter_kernels`` — the vendored, AITER-conformant copy of
     the kernels in this repo. Always probed last so a real AITER install
     wins, but ensures the bench has a real fused-kernel impl to time
     even on hosts that haven't picked up the upstream yet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

# Ensure Flash Attention 2 on ROCm uses the Triton backend rather than failing 
# when looking for the CUDA extension.
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.overlap import summarize_overlap
from benchmarks.common.shapes import pad_to_multiple, resolve_shapes
from benchmarks.common.timing import time_op


# GEMM (M, K, N) sweep. Definitions live in the central matrix-size file
# (configs/matrix_sizes.yaml) and are loaded via benchmarks.common.shapes, so
# shapes can be edited there or overridden per run with --shape-set / --shapes
# / --shapes-file. This module-level constant is only the import-time default
# (used by tooling that imports SHAPES); main() resolves the live sweep.
SHAPES = [s.as_tuple() for s in resolve_shapes()]


def _maybe_import(modname: str):
    try:
        return importlib.import_module(modname)
    except (ImportError, RuntimeError):
        return None


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower()


class _IrisCtx:
    """Adapter exposing the symmetric-memory contract our fused kernels expect.

    The vendored AG+MM / MM+RS wrappers (``benchmarks/aiter_kernels``) take the
    optimized Iris path only when an input tensor carries an ``_iris_ctx``
    attribute. That object must expose ``get_heap_bases()`` and an
    ``iris_ctx`` handle with ``.zeros(...)`` / ``.barrier()``. The raw Iris
    instance (``iris.iris(heap)``) already provides those, so this is a thin
    pass-through that keeps the attribute name stable regardless of the
    underlying Iris API surface.
    """

    def __init__(self, shmem: Any) -> None:
        self.iris_ctx = shmem
        self._iris_workspace: Dict[Tuple, torch.Tensor] = {}
        self.skip_ag_full_output = True

    def get_heap_bases(self):
        return self.iris_ctx.get_heap_bases()


def _make_iris_ctx(distributed: bool, world: int, want: bool) -> Tuple[Optional["_IrisCtx"], str]:
    """Create one Iris symmetric-memory context for the whole sweep.

    Returns ``(ctx, reason)``. ``ctx`` is ``None`` when Iris is disabled,
    unavailable, or the instance is missing the symmetric-memory API the
    fused kernels rely on — in which case the benchmark transparently falls
    back to the staged (non-Iris) path.
    """
    if not want:
        return None, "disabled (BENCH06_USE_IRIS=0)"
    if not (distributed and world >= 2):
        return None, "disabled (needs distributed world>=2)"
    try:
        import iris  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return None, f"iris import failed: {e!r}"

    heap_gib = float(os.environ.get("BENCH06_IRIS_HEAP_GIB", "8"))
    heap_bytes = int(heap_gib * (1 << 30))
    try:
        shmem = iris.iris(heap_bytes)
    except TypeError:
        try:
            shmem = iris.iris()
        except Exception as e:  # noqa: BLE001
            return None, f"iris.iris() init failed: {e!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"iris.iris(heap) init failed: {e!r}"

    for attr in ("get_heap_bases", "zeros", "barrier"):
        if not hasattr(shmem, attr):
            return None, f"iris instance missing .{attr}()"
    return _IrisCtx(shmem), f"iris (heap={heap_gib:g} GiB)"


def _symm_randn(ctx: "_IrisCtx", shape: Tuple[int, ...], dtype, device) -> torch.Tensor:
    """Allocate a symmetric-memory tensor and tag it for the Iris fused path."""
    shmem = ctx.iris_ctx
    if hasattr(shmem, "randn"):
        t = shmem.randn(shape, dtype=dtype)
    else:
        t = shmem.zeros(shape, dtype=dtype)
        t.normal_()
    if t.device != torch.device(device):  # best-effort; iris should already place on-device
        t = t.to(device)
    t._iris_ctx = ctx  # type: ignore[attr-defined]
    return t


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
    forced = os.environ.get("AITER_KERNELS_BACKEND")

    # Let the shared dispatcher own explicit backend pins, including the new
    # HipKittens/HK aliases. This keeps bench06 reporting aligned with the
    # reusable op surface in ``benchmarks.aiter_kernels``.
    if forced:
        local = _maybe_import("benchmarks.aiter_kernels")
        if local is None:
            return None, "benchmarks.aiter_kernels: import failed"
        select_backend = getattr(local, "select_backend", None)
        if not callable(select_backend):
            return None, "benchmarks.aiter_kernels: select_backend missing"
        try:
            info = select_backend(force=forced)
        except Exception as e:  # noqa: BLE001
            return None, f"forced backend {forced!r} unavailable ({e})"
        if info.source == "fallback.pure_torch":
            return (
                {"source": info.source, "ag_fn": info.ag_fn, "rs_fn": info.rs_fn, "call_kind": "symm_mem"},
                info.source,
            )
        return (
            {"source": info.source, "ag_fn": info.ag_fn, "rs_fn": info.rs_fn, "call_kind": "symm_mem"},
            info.source,
        )

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

    # Vendored, AITER-conformant kernels in this repo. The shared dispatcher
    # resolves the exact backend (HK if a fused extension exists, otherwise
    # local Triton/Iris, SymmMem, or fallback).
    local = _maybe_import("benchmarks.aiter_kernels")
    if local is not None:
        select_backend = getattr(local, "select_backend", None)
        try:
            info = select_backend() if callable(select_backend) else None
        except Exception as e:  # noqa: BLE001
            info = None
            tried.append(f"benchmarks.aiter_kernels dispatcher: {e!r}")
        if info is not None and info.source != "fallback.pure_torch":
            return (
                {
                    "source": info.source,
                    "ag_fn": info.ag_fn,
                    "rs_fn": info.rs_fn,
                    "call_kind": "symm_mem",
                },
                info.source,
            )
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
    if has_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    distributed = False
    if _is_distributed_env():
        if not dist.is_initialized():
            if has_gpu:
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        distributed = True
    else:
        rank = 0
        world = 1
        
    return rank, world, device, backend, distributed


def _bench_unfused_ag_mm(world: int, device, M: int, K: int, N: int, warmup: int, iters: int) -> Dict:
    """Baseline unfused AG+MM using PyTorch native gather and matmul."""
    # Pad M up to a multiple of world (preserve the production shape rather
    # than truncating it) so the collective shards evenly.
    M = pad_to_multiple(M, world)
    M_shard = M // world
    A_shard = torch.empty(M_shard, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
    
    A_full = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    A_full_pre = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    
    def fn_ag():
        dist.all_gather_into_tensor(A_full, A_shard)

    def fn_mm():
        torch.matmul(A_full_pre, B)
        
    def fn_full():
        dist.all_gather_into_tensor(A_full, A_shard)
        torch.matmul(A_full, B)

    dist.barrier()
    res_full = time_op(f"unfused_ag_mm_full_{M}_{K}_{N}", fn_full, warmup=warmup, iters=iters)
    res_ag = time_op(f"unfused_ag_mm_ag_{M}_{K}_{N}", fn_ag, warmup=warmup, iters=iters)
    res_mm = time_op(f"unfused_ag_mm_mm_{M}_{K}_{N}", fn_mm, warmup=warmup, iters=iters)

    flops = 2 * M * K * N
    ag_wire_bytes = (world - 1) * M_shard * K * 2
    return {
        "op":           "unfused_ag_mm",
        "world":        world,
        "M":            M,
        "K":            K,
        "N":            N,
        "t_ms":         res_full.median_ms,
        "t_ms_ag":      res_ag.median_ms,
        "t_ms_mm":      res_mm.median_ms,
        "tflops":       flops / (res_full.median_ms * 1e-3) / 1e12,
        "ag_gb_s":      ag_wire_bytes / (res_full.median_ms * 1e-3) / 1e9,
        "ag_wire_bytes": ag_wire_bytes,
    }


def _bench_unfused_mm_rs(world: int, device, M: int, K: int, N: int, warmup: int, iters: int) -> Dict:
    """Baseline unfused MM+RS using PyTorch native matmul and reduce_scatter."""
    # Pad M up to a multiple of world (preserve the production shape rather
    # than truncating it) so the collective shards evenly.
    M = pad_to_multiple(M, world)
    M_shard = M // world
    A = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
    Y_shard = torch.empty(M_shard, N, dtype=torch.bfloat16, device=device)
    
    Y_full_pre = torch.empty(M, N, dtype=torch.bfloat16, device=device).normal_()
    
    def fn_mm():
        torch.matmul(A, B)
        
    def fn_rs():
        dist.reduce_scatter_tensor(Y_shard, Y_full_pre, op=dist.ReduceOp.SUM)
    
    def fn_full():
        Y_full = torch.matmul(A, B)
        dist.reduce_scatter_tensor(Y_shard, Y_full, op=dist.ReduceOp.SUM)

    dist.barrier()
    res_full = time_op(f"unfused_mm_rs_full_{M}_{K}_{N}", fn_full, warmup=warmup, iters=iters)
    res_mm = time_op(f"unfused_mm_rs_mm_{M}_{K}_{N}", fn_mm, warmup=warmup, iters=iters)
    res_rs = time_op(f"unfused_mm_rs_rs_{M}_{K}_{N}", fn_rs, warmup=warmup, iters=iters)

    flops = 2 * M * K * N
    rs_wire_bytes = (world - 1) * M_shard * N * 2
    return {
        "op":           "unfused_mm_rs",
        "world":        world,
        "M":            M,
        "K":            K,
        "N":            N,
        "t_ms":         res_full.median_ms,
        "t_ms_mm":      res_mm.median_ms,
        "t_ms_rs":      res_rs.median_ms,
        "tflops":       flops / (res_full.median_ms * 1e-3) / 1e12,
        "rs_gb_s":      rs_wire_bytes / (res_full.median_ms * 1e-3) / 1e9,
        "rs_wire_bytes": rs_wire_bytes,
    }


def _bench_ag_mm(backend: Dict[str, Any], world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int, group_name: Optional[str],
                 iris_ctx: Optional["_IrisCtx"] = None) -> Dict:
    """Bench fused AG+MM. ``M`` is the **global** M; we shard by ``world`` per rank.

    Shapes:
      A_shard:  [M / world, K]  per rank
      B:        [K, N]          replicated
      Y:        [M, N]          full output (each rank gets it)

    Throughput accounting:
      flops = 2 * M * K * N        (one full GEMM)
      ag_wire = (world-1) * (M/world) * K * dtype_bytes  (this rank's outbound bytes)
    """
    # Pad M up to a multiple of world (preserve the production shape rather
    # than truncating it) so the collective shards evenly.
    M = pad_to_multiple(M, world)
    M_shard = M // world
    ag_fn = backend["ag_fn"]
    path = "staged"

    if backend["call_kind"] == "symm_mem":
        # AG+MM reads A_shard cross-rank, so A must live in symmetric memory for
        # the Iris fused path; tagging it with ``_iris_ctx`` selects that branch.
        if iris_ctx is not None:
            A_shard = _symm_randn(iris_ctx, (M_shard, K), torch.bfloat16, device)
            path = "iris"
        else:
            A_shard = torch.empty(M_shard, K, dtype=torch.bfloat16, device=device).normal_()
        B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
        def fn():
            ag_fn(A_shard, [B], gather_dim=0, group_name=group_name)
    else:
        # Legacy positional API: caller signature was (A_local, B) where
        # A_local is sharded on K, i.e., [M, K // world], and B is [K, N]
        if K % world:
            K = (K // world) * world
            if K == 0:
                raise RuntimeError("Legacy AG+MM needs K >= world")
        K_shard = K // world
        A_local = torch.empty(M, K_shard, dtype=torch.bfloat16, device=device).normal_()
        B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()
        def fn():
            ag_fn(A_local, B)

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
        "path": path,
    }


def _bench_mm_rs(backend: Dict[str, Any], world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int, group_name: Optional[str],
                 iris_ctx: Optional["_IrisCtx"] = None) -> Dict:
    """Bench fused MM+RS. ``M`` is the **global** M; output shard is M/world.

    Shapes:
      A:        [M, K]            replicated
      B:        [K, N]            replicated
      Y_shard:  [M / world, N]    per-rank output

    Throughput accounting:
      flops = 2 * M * K * N
      rs_wire = (world-1) * M_shard * N * dtype_bytes (per-rank inbound)
    """
    # Pad M up to a multiple of world (preserve the production shape rather
    # than truncating it) so the collective shards evenly.
    M = pad_to_multiple(M, world)
    M_shard = M // world
    rs_fn = backend["rs_fn"]
    path = "staged"

    if backend["call_kind"] == "symm_mem" and iris_ctx is not None:
        # MM+RS allocates its symmetric scratch internally; A only needs the
        # ``_iris_ctx`` tag to select the Iris fused path.
        A = _symm_randn(iris_ctx, (M, K), torch.bfloat16, device)
        path = "iris"
    else:
        A = torch.empty(M, K, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()

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
        "path": path,
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
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", type=str, default=None,
                    help="Override the sweep: 'M,K,N;M,K,N;...' (beats --shape-set).")
    ap.add_argument("--shape-set", type=str, default=None,
                    help="Named set from the matrix-size file (default: file's default_set).")
    ap.add_argument("--shapes-file", type=str, default=None,
                    help="Path to the matrix-size YAML (default: configs/matrix_sizes.yaml "
                         "or $BENCH_MATRIX_SIZES).")
    args = ap.parse_args()

    # Read methodology/config for timing
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    
    cfg_timing = {}
    if Path(args.config).is_file():
        try:
            cfg_timing = json.loads(Path(args.config).read_text()).get("timing", {})
        except Exception:  # noqa: BLE001
            pass
    
    t_cfg = m_cfg.get("timing", cfg_timing)
    
    def _is_set(opt): return any(o in a for a in sys.argv for o in (opt, opt.replace("--iters", "--iterations")))
    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 0)
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 0)

    # If both CLI and JSON are missing, use final hardcoded fallbacks
    if warmup_val == 0 and not _is_set("--warmup"): warmup_val = 5
    if iters_val == 0 and not _is_set("--iters"): iters_val = 20

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

    try:
        shapes = [s.as_tuple() for s in
                  resolve_shapes(args.shapes, args.shape_set, path=args.shapes_file)]
    except ValueError as e:
        raise SystemExit(f"shape resolution failed: {e}")

    group_name = (
        dist.group.WORLD.group_name if (distributed and backend_info["call_kind"] == "symm_mem")  # type: ignore[union-attr]
        else None
    )

    # Stand up a single Iris symmetric-memory context for the whole sweep so the
    # fused kernels exercise their optimized Iris path (Phase 1/2/3) instead of
    # the staged fallback. Creating Iris is a collective op; if any rank fails to
    # init we disable it everywhere to avoid a hang on the next collective.
    want_iris = (backend_info["call_kind"] == "symm_mem"
                 and _env_str("BENCH06_USE_IRIS", "auto") != "0")
    iris_ctx, iris_reason = _make_iris_ctx(distributed, world, want_iris)
    if distributed:
        ok = torch.tensor([1 if iris_ctx is not None else 0], device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() == 0 and iris_ctx is not None:
            iris_ctx = None
            iris_reason = "disabled (another rank failed Iris init)"
    if rank == 0:
        print(f"[06f] fused path: {'IRIS' if iris_ctx is not None else 'STAGED'} ({iris_reason})")

    rows: List[Dict] = []
    for (M, K, N) in shapes:
        for label in ("ag_mm", "mm_rs", "unfused_ag_mm", "unfused_mm_rs"):
            try:
                if label == "unfused_ag_mm":
                    row = _bench_unfused_ag_mm(world, device, M, K, N, warmup_val, iters_val)
                elif label == "unfused_mm_rs":
                    row = _bench_unfused_mm_rs(world, device, M, K, N, warmup_val, iters_val)
                elif label == "ag_mm":
                    row = _bench_ag_mm(backend_info, world, device, M, K, N,
                                       warmup_val, iters_val, group_name, iris_ctx)
                else:
                    row = _bench_mm_rs(backend_info, world, device, M, K, N,
                                       warmup_val, iters_val, group_name, iris_ctx)
                row["api_source"] = source
                row["call_kind"] = backend_info["call_kind"]
                rows.append(row)
                if rank == 0:
                    print(f"[06f] {label[:13]:13s} M={M:6d} K={K:5d} N={N:5d} "
                          f"t={row['t_ms']:7.2f} ms "
                          f"tflops={row['tflops']:7.1f} "
                          f"{row.get('path', ''):6s}")
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "op": label, "world": world, "M": M, "K": K, "N": N,
                    "error": repr(e), "api_source": source,
                    "call_kind": backend_info["call_kind"],
                })

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Flux-style overlap scorecard: ECT + overlap efficiency per shape and a
        # fitted collective cost model. This is the fair cross-kernel metric —
        # raw t_ms hides whether a "fused" kernel actually overlaps comm or just
        # runs a slower GEMM (overlap_efficiency < 0 flags the latter).
        overlap = summarize_overlap(rows)
        payload = {
            "available": True,
            "api_source": source,
            "world": world,
            "backend": backend,
            "device_type": device.type,
            "fused_path": "iris" if iris_ctx is not None else "staged",
            "fused_path_reason": iris_reason,
            "rows": rows,
            "overlap_summary": overlap,
        }
        write_json(out_dir / "fused.json", payload)
        write_csv(out_dir / "fused.csv", rows)
        write_csv(out_dir / "overlap.csv", overlap["rows"])

        for s in overlap["rows"]:
            eff = s.get("overlap_efficiency")
            spd = s.get("speedup_vs_unfused")
            ceil = s.get("ceiling_speedup")
            eff_str = f"{eff * 100:6.1f}%" if isinstance(eff, (int, float)) else "    n/a"
            spd_str = f"{spd:5.2f}x" if isinstance(spd, (int, float)) else "  n/a"
            ceil_str = f"{ceil:5.2f}x" if isinstance(ceil, (int, float)) else "  n/a"
            print(f"[06f] overlap {s['pattern']:6s} M={s['M']:6d} K={s['K']:5d} "
                  f"N={s['N']:5d}  eff={eff_str}  speedup={spd_str}  ceiling={ceil_str}")
        for op, model in overlap["cost_model"].items():
            bw = model.get("bandwidth_gb_s")
            bw_str = f"{bw:6.1f} GB/s" if isinstance(bw, (int, float)) else "n/a"
            print(f"[06f] cost-model {op:6s} comm ~= {model['alpha_ms']:.3f} ms + "
                  f"bytes/({bw_str})  (r2={model['r2']:.3f}, n={model['n_points']})")

    _maybe_barrier_and_destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

