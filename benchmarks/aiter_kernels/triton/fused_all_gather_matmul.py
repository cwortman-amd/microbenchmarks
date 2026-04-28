"""AITER-style wrapper for fused all-gather + matmul.

Public surface mirrors ``torch.ops.symm_mem.fused_all_gather_matmul`` so it
slots in as a column-parallel TP linear primitive at any call-site that
already uses SymmMem.

When Iris is available the wrapper routes to the Iris-aware Triton kernel
(``_fused_ag_mm_iris_kernel``) for full producer-consumer overlap. When
Iris is absent we still produce correct results by performing a real
``dist.all_gather_into_tensor`` and then running the staged GEMM kernel
(``_fused_ag_mm_staged_kernel``) — slower than the fused variant but
still measurable and useful as a Triton-only baseline for the bench.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.distributed as dist
import triton

from benchmarks.aiter_kernels._capabilities import IRIS_AVAILABLE, detect_arch
from benchmarks.aiter_kernels._config_loader import env_override, get_kernel_config
from benchmarks.aiter_kernels.triton._triton_kernels.fused_all_gather_matmul import (
    _fused_ag_mm_staged_kernel,
)

if IRIS_AVAILABLE:
    import iris  # type: ignore
    from benchmarks.aiter_kernels.triton._triton_kernels.fused_all_gather_matmul import (
        _fused_ag_mm_iris_kernel,
    )

logger = logging.getLogger("aiter_kernels.triton.fused_ag_mm")


def _resolve_config(M: int, N: int, K: int) -> dict:
    arch = detect_arch() or "gfx950"
    cfg, tuned = get_kernel_config(arch, "FUSED-AG-MATMUL", M=M, N=N, K=K)
    cfg = env_override("AITER_KERNELS_FUSED_AG_MM", cfg)
    if not tuned:
        logger.debug("FUSED-AG-MATMUL: using untuned config for arch=%s M=%d N=%d K=%d",
                     arch, M, N, K)
    return cfg


def _flatten_batch(t: torch.Tensor, gather_dim: int) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """Collapse leading batch dims so the kernel sees a 2-D problem.

    Matches what ``_symmetric_memory._fused_all_gather_matmul_fallback`` does
    internally: pivot the gather_dim to dim 0, flatten batch, run, restore.
    """
    permuted = t.movedim(gather_dim, 0).contiguous()
    rest_shape = permuted.shape[1:]
    flat = permuted.reshape(permuted.shape[0], -1) if permuted.dim() > 2 else permuted
    K_or_None = flat.shape[-1]
    return flat, rest_shape, K_or_None


def fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    *,
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Fused AG along ``gather_dim`` of ``A_shard`` followed by ``A_full @ B_i``.

    Args:
        A_shard: This rank's local shard of ``A``. Any leading dims are
            allowed; ``gather_dim`` selects the M-axis along which we
            all-gather.
        Bs: One or more weight matrices ``[K, N_i]``, each replicated
            across ranks.
        gather_dim: Dim of ``A_shard`` to all-gather along (matches
            SymmMem's argument by the same name).
        group_name: Process-group name (``dist.group.WORLD.group_name``).

    Returns:
        ``(A_full, [Y_i])`` with the same shape semantics as
        ``torch.ops.symm_mem.fused_all_gather_matmul``.
    """
    group = dist.distributed_c10d._resolve_process_group(group_name)
    world = group.size()
    rank = dist.get_rank(group)

    A_perm, rest_shape, K = _flatten_batch(A_shard, gather_dim)
    M_shard = A_perm.shape[0]
    M_global = M_shard * world

    if IRIS_AVAILABLE and isinstance(getattr(A_shard, "_iris_ctx", None), object):
        # Iris path: caller already allocated A_shard in symmetric memory.
        # We can launch the fused kernel directly without staging through
        # all_gather_into_tensor.
        ctx = A_shard._iris_ctx  # type: ignore[attr-defined]
        heap_bases = ctx.get_heap_bases()
        outputs: List[torch.Tensor] = []
        for B in Bs:
            assert B.dim() == 2 and B.shape[0] == K, (
                f"B must be [K, N], got {B.shape} (K={K})"
            )
            N = B.shape[1]
            cfg = _resolve_config(M_global, N, K)
            Y = ctx.iris_ctx.zeros((M_global, N), dtype=A_shard.dtype)
            grid = (cfg["NUM_SMS"],)
            _fused_ag_mm_iris_kernel[grid](
                A_perm, B, Y,
                M_global, M_shard, N, K,
                A_perm.stride(0), A_perm.stride(1),
                B.stride(0), B.stride(1),
                Y.stride(0), Y.stride(1),
                cur_rank=rank, world_size=world, heap_bases=heap_bases,
                BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                GROUP_SIZE_M=cfg["GROUP_SIZE_M"], NUM_SMS=cfg["NUM_SMS"],
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
                waves_per_eu=cfg["waves_per_eu"],
            )
            outputs.append(Y if not rest_shape else Y.reshape(M_global, *rest_shape).movedim(0, gather_dim).contiguous())
        ctx.iris_ctx.barrier()
        # A_full is the gathered tensor — we need to materialize it via
        # iris.put on the source side, but Iris already exposes the
        # symmetric tensor as logically global; the caller can read any
        # slice. For API parity with SymmMem we still return a torch view.
        A_full_perm = ctx.iris_ctx.zeros((M_global, K), dtype=A_shard.dtype)
        # Stage AG into A_full_perm via dist as a correctness fallback for
        # the AG-output position (downstream rarely uses it; SymmMem only
        # returns it for parity with the eager path).
        dist.all_gather_into_tensor(A_full_perm, A_perm, group=group)
        A_full = A_full_perm if not rest_shape else A_full_perm.reshape(M_global, *rest_shape).movedim(0, gather_dim).contiguous()
        return A_full, outputs

    # Staged path: run a real AG, then the staged GEMM kernel.
    A_full_perm = torch.empty((M_global, K), dtype=A_shard.dtype, device=A_shard.device)
    dist.all_gather_into_tensor(A_full_perm, A_perm, group=group)

    outputs = []
    for B in Bs:
        assert B.dim() == 2 and B.shape[0] == K, (
            f"B must be [K, N], got {B.shape} (K={K})"
        )
        N = B.shape[1]
        cfg = _resolve_config(M_global, N, K)
        Y = torch.empty((M_global, N), dtype=A_shard.dtype, device=A_shard.device)
        grid = (cfg["NUM_SMS"],)
        _fused_ag_mm_staged_kernel[grid](
            A_full_perm, B, Y,
            M_global, N, K,
            A_full_perm.stride(0), A_full_perm.stride(1),
            B.stride(0), B.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"], NUM_SMS=cfg["NUM_SMS"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
            waves_per_eu=cfg["waves_per_eu"],
        )
        if rest_shape:
            Y = Y.reshape(M_global, *rest_shape).movedim(0, gather_dim).contiguous()
        outputs.append(Y)

    A_full = (
        A_full_perm.reshape(M_global, *rest_shape).movedim(0, gather_dim).contiguous()
        if rest_shape
        else A_full_perm
    )
    return A_full, outputs
