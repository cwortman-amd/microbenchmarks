"""Experimental HipKittens backend adapter for fused collective+GEMM.

HipKittens is currently integrated here as an AITER-managed backend contract,
not as a completed fused AG+MM/MM+RS implementation. The public HK repository
contains the right CDNA4 building blocks and a distributed Iris GEMM example,
but the sample extension exposes ``dispatch_micro`` for local BF16 GEMM rather
than SymmMem-style fused collective entry points.

Selection contract
------------------
Set ``HIPKITTENS_FUSED_MODULE`` to a Python extension/module that exports:

  * ``fused_all_gather_matmul(A_shard, Bs, *, gather_dim, group_name)``
  * ``fused_matmul_reduce_scatter(A, B, reduce_op, *, scatter_dim, group_name)``

The repo-local native prototype builds as ``hk_iris_fused`` and may expose
lower-level dispatch functions first. Those are reported as prototypes and are
not selected by the public dispatcher until they implement the full API above.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.aiter_kernels._capabilities import HIPKITTENS_ROOT


@dataclass(frozen=True)
class HipKittensBackend:
    source: str
    ag_fn: Callable
    rs_fn: Callable


_LAST_REASON = "not probed"


def _iris_workspace(ctx, key: Tuple, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    cache = getattr(ctx, "_hipkittens_workspace", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_hipkittens_workspace", cache)
    full_key = ("hk", key, tuple(shape), str(dtype))
    t = cache.get(full_key)
    if t is None:
        t = ctx.iris_ctx.zeros(shape, dtype=dtype)
        cache[full_key] = t
    return t


def _candidate_roots() -> List[Path]:
    roots: List[Path] = []
    for var in ("HIPKITTENS_SRC", "HK_SRC"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value).expanduser())
    if HIPKITTENS_ROOT is not None:
        roots.append(HIPKITTENS_ROOT)
    roots.extend([Path.home() / ".cache" / "HipKittens", Path.cwd() / "HipKittens"])
    seen = set()
    out = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _add_hk_paths() -> None:
    """Expose built HK distributed extension directories to Python imports."""
    native = Path(__file__).with_name("hipkittens_native")
    if native.is_dir():
        native_s = str(native)
        if native_s not in sys.path:
            sys.path.insert(0, native_s)
    for root in _candidate_roots():
        distributed = root / "distributed-kernels"
        bf16_gemm = distributed / "bf16_gemm"
        for path in (distributed, bf16_gemm):
            if path.is_dir():
                path_s = str(path)
                if path_s not in sys.path:
                    sys.path.insert(0, path_s)


def _module_candidates() -> List[str]:
    env_mod = os.environ.get("HIPKITTENS_FUSED_MODULE")
    names = [env_mod] if env_mod else []
    names.extend([
        "hk_iris_fused",
        "hipkittens_aiter_fused",
        "hk_aiter_fused",
        "tk_fused_collective",
    ])
    return [name for name in names if name]


def availability_reason() -> str:
    return _LAST_REASON


def _get_iris_context_tensor(A: torch.Tensor, op_name: str) -> Tuple[object, torch.Tensor]:
    ctx = getattr(A, "_iris_ctx", None)
    if ctx is None:
        raise NotImplementedError(f"HipKittens {op_name} requires input allocated in Iris symmetric memory")
    raw_ctx = getattr(ctx, "iris_ctx", ctx)
    get_device_context = getattr(raw_ctx, "get_device_context", None)
    if not callable(get_device_context):
        raise NotImplementedError("Iris context does not expose get_device_context(); update ROCm/iris")
    return ctx, get_device_context()


def _make_ag_wrapper(mod) -> Callable:
    def _ag(
        A_shard: torch.Tensor,
        Bs: List[torch.Tensor],
        *,
        gather_dim: int,
        group_name: str,
    ):
        if gather_dim != 0:
            raise NotImplementedError("HipKittens AG+MM prototype currently supports gather_dim=0 only")
        if len(Bs) != 1:
            raise NotImplementedError("HipKittens AG+MM prototype currently supports one B matrix")
        if A_shard.dim() != 2 or Bs[0].dim() != 2:
            raise NotImplementedError("HipKittens AG+MM prototype currently supports 2-D [M,K] @ [K,N]")
        if A_shard.dtype != torch.bfloat16 or Bs[0].dtype != torch.bfloat16:
            raise NotImplementedError("HipKittens AG+MM prototype currently supports bf16 only")

        group = dist.distributed_c10d._resolve_process_group(group_name)
        world = group.size()
        rank = dist.get_rank(group)
        M_shard, K = A_shard.shape
        B = Bs[0]
        if B.shape[0] != K:
            raise ValueError(f"B must be [K,N], got {tuple(B.shape)} for K={K}")
        N = B.shape[1]
        M_global = M_shard * world
        if M_shard % 128 or M_global % 128 or N % 256 or K % 64:
            raise NotImplementedError(
                "HipKittens AG+MM prototype requires M_shard and M_global multiples of 128, "
                "N multiple of 256, and K multiple of 64"
            )

        ctx, device_context = _get_iris_context_tensor(A_shard, "AG+MM")
        # HK's MFMA path consumes B as [N,K] and uses mma_ABt.
        B_t = B.t().contiguous()
        Y = _iris_workspace(ctx, ("ag_mm_y", M_global, K, N), (M_global, N), A_shard.dtype)
        reuse_mode = os.environ.get("HIPKITTENS_AG_N_REUSE", "").strip().lower()
        auto_reuse = reuse_mode == "auto" and N % 512 == 0 and K >= 256
        if (
            (reuse_mode in {"2_spillfree", "2-sf", "spillfree"} or auto_reuse)
            and hasattr(mod, "dispatch_ag_mm_reuse_spillfree")
            and N % 512 == 0
        ):
            ag_dispatch = mod.dispatch_ag_mm_reuse_spillfree
        elif reuse_mode == "2" and hasattr(mod, "dispatch_ag_mm_reuse") and N % 512 == 0:
            ag_dispatch = mod.dispatch_ag_mm_reuse
        else:
            ag_dispatch = mod.dispatch_ag_mm
        ag_dispatch(
            A_shard,
            B_t,
            Y,
            int(device_context.data_ptr()),
            int(M_global),
            int(M_shard),
            int(N),
            int(K),
            int(rank * M_shard),
        )
        raw_ctx = getattr(ctx, "iris_ctx", ctx)
        raw_ctx.barrier()
        if not getattr(ctx, "skip_ag_full_output", False):
            A_full = _iris_workspace(ctx, ("ag_full", M_global, K), (M_global, K), A_shard.dtype)
            dist.all_gather_into_tensor(A_full, A_shard, group=group)
            return A_full, [Y]
        return A_shard, [Y]

    return _ag


def _make_rs_wrapper(mod) -> Callable:
    def _rs(A: torch.Tensor, B: torch.Tensor, reduce_op: str, *, scatter_dim: int, group_name: str):
        if reduce_op not in ("sum", "avg"):
            raise ValueError(f"reduce_op must be 'sum' or 'avg', got {reduce_op!r}")
        if scatter_dim != 0:
            raise NotImplementedError("HipKittens MM+RS prototype currently supports scatter_dim=0 only")
        if A.dim() != 2 or B.dim() != 2:
            raise NotImplementedError("HipKittens MM+RS prototype currently supports 2-D [M,K] @ [K,N]")
        if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
            raise NotImplementedError("HipKittens MM+RS prototype currently supports bf16 only")

        group = dist.distributed_c10d._resolve_process_group(group_name)
        world = group.size()
        M_global, K = A.shape
        if M_global % world != 0:
            raise ValueError(f"M_global={M_global} not divisible by world={world}")
        M_shard = M_global // world
        if B.shape[0] != K:
            raise ValueError(f"B must be [K,N], got {tuple(B.shape)} for K={K}")
        N = B.shape[1]
        if M_shard % 128 or M_global % 128 or N % 256 or K % 64:
            raise NotImplementedError(
                "HipKittens MM+RS prototype requires M_shard and M_global multiples of 128, "
                "N multiple of 256, and K multiple of 64"
            )

        ctx, device_context = _get_iris_context_tensor(A, "MM+RS")
        B_t = B.t().contiguous()
        rank = dist.get_rank(group)
        scratch_swizzle = 1 if os.environ.get("HIPKITTENS_MM_RS_SWIZZLE", "").strip().lower() in {"1", "true", "yes", "on", "swizzle"} else 0
        double_buffer = os.environ.get("HIPKITTENS_MM_RS_DOUBLE_BUFFER", "").strip().lower() in {"1", "true", "yes", "on"}
        # Device-side per-rank flag handoff replaces the post-writer host barrier
        # with a release/acquire flag protocol. It requires double-buffered
        # scratch so the next writer cannot clobber a buffer still being read.
        use_flags = os.environ.get("HIPKITTENS_MM_RS_FLAGS", "").strip().lower() in {"1", "true", "yes", "on"}
        if use_flags and not hasattr(mod, "dispatch_mm_rs_reduce_vec4_flags"):
            use_flags = False
        if use_flags:
            double_buffer = True
        raw_ctx = getattr(ctx, "iris_ctx", ctx)
        # Iris exposes a device-side, CUDA-graph-capturable barrier built on
        # sys-scope release/acquire atomics. Unlike barrier(), it does not do a
        # host torch.cuda.synchronize()+distributed_barrier() round-trip, so it
        # removes most of the post-writer barrier's host latency while keeping
        # Iris's own correctness guarantee for remote-write visibility. It
        # requires the writer to use write-through stores so the remote payload
        # lands before the device barrier is observed.
        use_device_barrier = (
            os.environ.get("HIPKITTENS_MM_RS_DEVICE_BARRIER", "").strip().lower() in {"1", "true", "yes", "on"}
            and hasattr(raw_ctx, "device_barrier")
        )
        # Write-through remote stores are needed whenever a device-side barrier
        # or the flag handoff replaces the host barrier's full device sync.
        writer_write_through = 1 if (use_flags or use_device_barrier) else 0
        if double_buffer:
            # Two symmetric scratch buffers alternated per call. The next writer
            # targets the *other* buffer, so it cannot clobber the buffer the
            # previous reducer is still reading. This removes the write-after-read
            # hazard that the post-reducer barrier otherwise guards, letting us
            # drop that barrier. Cross-rank reuse two calls later is ordered by
            # the following call's post-writer barrier.
            scratch_full = _iris_workspace(
                ctx,
                ("mm_rs_scratch_db", world, M_global, M_shard, K, N, scratch_swizzle),
                (2, world, M_shard, N),
                A.dtype,
            )
            buf_idx = getattr(ctx, "_mm_rs_db_idx", 0)
            ctx._mm_rs_db_idx = buf_idx ^ 1
            scratch = scratch_full[buf_idx]
        else:
            scratch = _iris_workspace(
                ctx,
                ("mm_rs_scratch", world, M_global, M_shard, K, N, scratch_swizzle),
                (world, M_shard, N),
                A.dtype,
            )
        Y_shard = _iris_workspace(
            ctx,
            ("mm_rs_y", world, M_global, M_shard, K, N),
            (M_shard, N),
            A.dtype,
        )
        if use_flags:
            flags = _iris_workspace(
                ctx,
                ("mm_rs_flags", world),
                (world,),
                torch.int32,
            )
            wg_counter = _iris_workspace(
                ctx,
                ("mm_rs_wg_counter", world),
                (world,),
                torch.int32,
            )
            generation = int(getattr(ctx, "_mm_rs_gen", 0)) + 1
            ctx._mm_rs_gen = generation
            flags_ptr = int(flags.data_ptr())
            counter_ptr = int(wg_counter.data_ptr())
        else:
            flags_ptr = 0
            counter_ptr = 0
            generation = 0
        mod.dispatch_mm_rs_write(
            A,
            B_t,
            scratch,
            int(device_context.data_ptr()),
            int(M_global),
            int(M_shard),
            int(N),
            int(K),
            int(scratch_swizzle),
            int(flags_ptr),
            int(counter_ptr),
            int(generation),
            int(1 if use_flags else 0),
            int(writer_write_through),
        )
        # ABLATION ONLY: HIPKITTENS_MM_RS_NO_BARRIER1 skips the post-writer
        # barrier to measure the timing floor of a future device-flag pipeline.
        # This is racy and numerically INCORRECT (the reducer may read scratch
        # before remote writes land); use only for ceiling measurement, never
        # for correctness or production.
        skip_barrier1 = os.environ.get("HIPKITTENS_MM_RS_NO_BARRIER1", "").strip().lower() in {"1", "true", "yes", "on"}

        def _barrier():
            if use_device_barrier:
                raw_ctx.device_barrier()
            else:
                raw_ctx.barrier()

        scale = (1.0 / world) if reduce_op == "avg" else 1.0
        if use_flags:
            # Device-side handoff: the writer's last workgroup per destination
            # does threadfence_system + a release store of the generation into
            # that rank's flags[source]; the flag-aware reducer acquire-waits on
            # every source slot before reducing. No host barrier is used between
            # the writer and the reducer.
            mod.dispatch_mm_rs_reduce_vec4_flags(
                scratch,
                Y_shard,
                int(flags.data_ptr()),
                int(M_shard),
                int(N),
                int(world),
                float(scale),
                int(rank),
                int(scratch_swizzle),
                int(generation),
            )
            return Y_shard
        if not skip_barrier1:
            _barrier()
        reducer_mode = os.environ.get("HIPKITTENS_MM_RS_REDUCER", "").strip().lower()
        if reducer_mode == "auto" and hasattr(mod, "dispatch_mm_rs_reduce_vec4"):
            reduce_dispatch = mod.dispatch_mm_rs_reduce_vec4
        elif reducer_mode in {"specialized", "world"} and hasattr(mod, "dispatch_mm_rs_reduce_specialized"):
            reduce_dispatch = mod.dispatch_mm_rs_reduce_specialized
        elif reducer_mode in {"vec4", "vector4"} and hasattr(mod, "dispatch_mm_rs_reduce_vec4"):
            reduce_dispatch = mod.dispatch_mm_rs_reduce_vec4
        else:
            reduce_dispatch = mod.dispatch_mm_rs_reduce
        reduce_dispatch(
            scratch,
            Y_shard,
            int(M_shard),
            int(N),
            int(world),
            float(scale),
            int(rank),
            int(scratch_swizzle),
        )
        if not double_buffer:
            # Post-reducer barrier guards scratch reuse by the next writer. With
            # double-buffering the next writer uses the other buffer, so this
            # barrier is unnecessary and is dropped (the measured ~35% phase).
            _barrier()
        return Y_shard

    return _rs


def _missing_mm_rs(*args, **kwargs):
    raise NotImplementedError("HipKittens module does not expose dispatch_mm_rs_write/dispatch_mm_rs_reduce")


def resolve_backend() -> Optional[HipKittensBackend]:
    """Resolve a built fused HK extension, if one is present."""
    global _LAST_REASON

    if not torch.cuda.is_available():
        _LAST_REASON = "HipKittens backend requires a CUDA/ROCm GPU device"
        return None
    if not _candidate_roots():
        _LAST_REASON = "HipKittens source checkout not found; set HIPKITTENS_SRC"
        return None

    _add_hk_paths()
    tried: List[str] = []
    for modname in _module_candidates():
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{modname}: import failed ({e!r})")
            continue
        ag_fn = getattr(mod, "fused_all_gather_matmul", None)
        rs_fn = getattr(mod, "fused_matmul_reduce_scatter", None)
        if callable(ag_fn) and callable(rs_fn):
            _LAST_REASON = f"resolved {modname}"
            return HipKittensBackend(source=f"hipkittens.{modname}", ag_fn=ag_fn, rs_fn=rs_fn)
        dispatch_ag = getattr(mod, "dispatch_ag_mm", None)
        dispatch_rs_write = getattr(mod, "dispatch_mm_rs_write", None)
        dispatch_rs_reduce = getattr(mod, "dispatch_mm_rs_reduce", None)
        if callable(dispatch_ag) and callable(dispatch_rs_write) and callable(dispatch_rs_reduce):
            _LAST_REASON = f"resolved {modname} native AG+MM and MM+RS prototypes"
            return HipKittensBackend(
                source=f"hipkittens.{modname}.native",
                ag_fn=_make_ag_wrapper(mod),
                rs_fn=_make_rs_wrapper(mod),
            )
        if callable(dispatch_ag):
            _LAST_REASON = (
                f"resolved {modname} AG+MM native prototype; "
                "MM+RS native dispatch is missing"
            )
            return HipKittensBackend(
                source=f"hipkittens.{modname}.ag_mm+triton_rs",
                ag_fn=_make_ag_wrapper(mod),
                rs_fn=_make_rs_wrapper(mod) if callable(dispatch_rs_write) and callable(dispatch_rs_reduce) else _missing_mm_rs,
            )
        low_level = []
        if callable(dispatch_ag):
            low_level.append("dispatch_ag_mm")
        if callable(getattr(mod, "dispatch_mm_rs", None)):
            low_level.append("dispatch_mm_rs")
        if callable(dispatch_rs_write):
            low_level.append("dispatch_mm_rs_write")
        if callable(dispatch_rs_reduce):
            low_level.append("dispatch_mm_rs_reduce")
        low_level_msg = f" low_level={','.join(low_level)}" if low_level else ""
        tried.append(
            f"{modname}: ag={'ok' if callable(ag_fn) else 'missing'} "
            f"rs={'ok' if callable(rs_fn) else 'missing'}{low_level_msg}"
        )

    _LAST_REASON = (
        "no fused HipKittens extension found; build/export a module with "
        "fused_all_gather_matmul and fused_matmul_reduce_scatter. "
        "Low-level HK dispatch prototypes are intentionally not selected until "
        "they implement the full collective+GEMM API "
        f"(tried: {'; '.join(tried) if tried else 'no module candidates'})"
    )
    return None


def fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    *,
    gather_dim: int,
    group_name: str,
):
    backend = resolve_backend()
    if backend is None:
        raise NotImplementedError(availability_reason())
    return backend.ag_fn(A_shard, Bs, gather_dim=gather_dim, group_name=group_name)


def fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    *,
    scatter_dim: int,
    group_name: str,
):
    backend = resolve_backend()
    if backend is None:
        raise NotImplementedError(availability_reason())
    return backend.rs_fn(A, B, reduce_op, scatter_dim=scatter_dim, group_name=group_name)
