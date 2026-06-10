"""AITER-style wrapper for fused matmul + reduce-scatter.

Public surface mirrors ``torch.ops.symm_mem.fused_matmul_reduce_scatter``.

Iris path (when available): tile-level push to peer-rank symmetric-memory
partials with ``iris.atomic_add`` followed by a barrier; optional ``avg``
scaling kernel.

Staged path (Iris missing): produce the full ``Y = A @ B`` with the local
GEMM kernel, then drive a real ``dist.reduce_scatter_tensor`` for transport.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import triton

from benchmarks.aiter_kernels._capabilities import IRIS_AVAILABLE, detect_arch
from benchmarks.aiter_kernels._config_loader import env_override, get_kernel_config
from benchmarks.aiter_kernels.triton._triton_kernels.fused_matmul_reduce_scatter import (
    _fused_mm_rs_staged_kernel,
    _reduce_partials_kernel,
    _reduce_partials_signal_kernel,
)

if IRIS_AVAILABLE:
    import iris  # type: ignore
    from benchmarks.aiter_kernels.triton._triton_kernels.fused_matmul_reduce_scatter import (
        _fused_mm_rs_iris_write_kernel,
    )


def _env_flag(name: str) -> bool:
    """Truthy env-var check (``1``/``true``/``yes``/``on``)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

logger = logging.getLogger("aiter_kernels.triton.fused_mm_rs")


def _resolve_config(M: int, N: int, K: int) -> dict:
    arch = detect_arch() or "gfx950"
    cfg, tuned = get_kernel_config(arch, "FUSED-MATMUL-RS", M=M, N=N, K=K)
    cfg = env_override("AITER_KERNELS_FUSED_MM_RS", cfg)
    if not tuned:
        logger.debug("FUSED-MATMUL-RS: using untuned config for arch=%s M=%d N=%d K=%d",
                     arch, M, N, K)
    return cfg


def _flatten_batch(t: torch.Tensor, scatter_dim: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pivot scatter_dim to dim 0 and flatten any further batch dims.

    ``rest_shape`` describes the *batch* dims to restore on the **output** —
    i.e. the dims strictly between the scattered row dim and the contraction
    (K) dim. For the common 2-D ``[M, K]`` case there are none, so we return an
    empty tuple: the output is ``[M_shard, N]`` and must NOT be reshaped using
    K. (Returning ``shape[1:]`` here was a bug — it tried to reshape the
    N-column result with K.)
    """
    permuted = t.movedim(scatter_dim, 0).contiguous()
    if permuted.dim() <= 2:
        return permuted, ()
    rest_shape = permuted.shape[1:-1]
    flat = permuted.reshape(permuted.shape[0], -1)
    return flat, rest_shape


def fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    *,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Fused ``A @ B`` followed by reduce-scatter along ``scatter_dim``.

    Args:
        A: ``[M, K]`` (or batched) replicated activation tensor.
        B: ``[K, N]`` weight matrix.
        reduce_op: ``"sum"`` or ``"avg"``.
        scatter_dim: Dim along which to scatter the result.
        group_name: Process-group name (``dist.group.WORLD.group_name``).

    Returns:
        ``Y_shard`` with shape matching ``Y[scatter_dim]/world`` along
        ``scatter_dim`` and otherwise identical to ``A @ B``.
    """
    if reduce_op not in ("sum", "avg"):
        raise ValueError(f"reduce_op must be 'sum' or 'avg', got {reduce_op!r}")

    group = dist.distributed_c10d._resolve_process_group(group_name)
    world = group.size()
    rank = dist.get_rank(group)

    A_perm, rest_shape = _flatten_batch(A, scatter_dim)
    M_global, K = A_perm.shape[0], A_perm.shape[1] if A_perm.dim() == 2 else A_perm.shape[-1]
    if M_global % world != 0:
        raise ValueError(
            f"fused_matmul_reduce_scatter: M_global={M_global} not divisible by world={world}"
        )
    M_shard = M_global // world
    assert B.dim() == 2 and B.shape[0] == K, f"B must be [K, N], got {B.shape}"
    N = B.shape[1]
    cfg = _resolve_config(M_global, N, K)

    if cfg["BLOCK_M"] > M_shard:
        # Tile bigger than per-rank shard would route an entire tile to one
        # destination, but if BLOCK_M > M_shard the assignment isn't single-
        # destination anymore. Fall back to a smaller BLOCK_M.
        cfg = dict(cfg)
        cfg["BLOCK_M"] = max(1, M_shard)

    if IRIS_AVAILABLE and isinstance(getattr(A, "_iris_ctx", None), object):
        ctx = A._iris_ctx  # type: ignore[attr-defined]
        heap_bases = ctx.get_heap_bases()
        inv_scale = (1.0 / world) if reduce_op == "avg" else 1.0
        use_signal = _env_flag("AITER_KERNELS_MM_RS_SIGNAL")

        # Phase 2/3: one symmetric per-source slot per rank — no atomic
        # contention on the destination. Each rank writes scratch[cur_rank].
        scratch = ctx.iris_ctx.zeros((world, M_shard, N), dtype=A.dtype)
        num_pid_m = triton.cdiv(M_shard, cfg["BLOCK_M"])
        num_pid_n = triton.cdiv(N, cfg["BLOCK_N"])
        # Per-tile arrival counters (Phase 3 signaling). Symmetric so peers can
        # bump them with iris.atomic_add; freshly zeroed each call.
        flags = ctx.iris_ctx.zeros((num_pid_m * num_pid_n,), dtype=torch.int32)

        grid = (cfg["NUM_SMS"],)
        _fused_mm_rs_iris_write_kernel[grid](
            A_perm, B, scratch, flags,
            M_global, M_shard, N, K,
            A_perm.stride(0), A_perm.stride(1),
            B.stride(0), B.stride(1),
            scratch.stride(0), scratch.stride(1), scratch.stride(2),
            cur_rank=rank, world_size=world, heap_bases=heap_bases,
            SIGNAL=use_signal,
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"], NUM_SMS=cfg["NUM_SMS"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
            waves_per_eu=cfg["waves_per_eu"],
        )

        Y_shard = torch.empty((M_shard, N), dtype=A.dtype, device=A.device)
        if use_signal:
            # No global barrier: per-tile signals carry the cross-rank
            # synchronization, overlapping reduce with in-flight writes.
            _reduce_partials_signal_kernel[grid](
                scratch, Y_shard, flags,
                M_shard, N,
                scratch.stride(0), scratch.stride(1), scratch.stride(2),
                Y_shard.stride(0), Y_shard.stride(1),
                inv_scale,
                world_size=world,
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
                NUM_SMS=cfg["NUM_SMS"],
            )
            # Ensure all ranks finished consuming their scratch before it is
            # freed / reused by the next op.
            ctx.iris_ctx.barrier()
        else:
            ctx.iris_ctx.barrier()
            reduce_grid = (num_pid_m, num_pid_n)
            _reduce_partials_kernel[reduce_grid](
                scratch, Y_shard,
                M_shard, N,
                scratch.stride(0), scratch.stride(1), scratch.stride(2),
                Y_shard.stride(0), Y_shard.stride(1),
                inv_scale,
                world_size=world,
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
            )

        if rest_shape:
            Y_shard = Y_shard.reshape(M_shard, *rest_shape).movedim(0, scatter_dim).contiguous()
        return Y_shard

    # Staged path: full GEMM + dist reduce_scatter.
    Y_full = torch.empty((M_global, N), dtype=A.dtype, device=A.device)
    grid = (cfg["NUM_SMS"],)
    _fused_mm_rs_staged_kernel[grid](
        A_perm, B, Y_full,
        M_global, N, K,
        A_perm.stride(0), A_perm.stride(1),
        B.stride(0), B.stride(1),
        Y_full.stride(0), Y_full.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"], NUM_SMS=cfg["NUM_SMS"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    Y_shard = torch.empty((M_shard, N), dtype=A.dtype, device=A.device)
    dist.reduce_scatter_tensor(Y_shard, Y_full, op=dist.ReduceOp.SUM, group=group)
    if reduce_op == "avg":
        Y_shard = Y_shard / world
    if rest_shape:
        Y_shard = Y_shard.reshape(M_shard, *rest_shape).movedim(0, scatter_dim).contiguous()
    return Y_shard
