"""Triton kernel: fused all-gather + matmul (column-parallel TP linear).

Schedule: **per-shard single-owner** (Phase 1 refactor; see
``docs/AITER_FUSED_KERNELS.md`` §"Refactor path").

The tile space is iterated as ``(shard, local_m_tile, n_tile)``. Each gathered
row is owned by exactly one shard, so each output tile is computed **once**
from its owner's symmetric-memory shard:

  for (shard, m_local_tile, n_tile):
    acc = 0
    for k_tile:
      a_tile = iris.load(A[shard][m_local_tile, k_tile])  # identity if shard==self
      b_tile = tl.load(B[k_tile, n_tile])
      acc   += mfma(a_tile, b_tile)
    store(Y[shard*M_shard + m_local_tile, n_tile], acc)

The earlier schedule iterated global-M tiles and looped over *all* ranks per
tile, running a full masked-to-zero K-loop for every non-owning rank —
``world_size×`` the necessary MFMA work. Resolving the owner from ``shard``
removes that redundancy and is correct for any ``M_shard`` (tiles never
straddle a shard boundary).

When Iris is not present at compile time we still produce correct results
by indexing into a pre-gathered ``A_full`` buffer (the wrapper performs the
gather via ``dist.all_gather_into_tensor`` first); the kernel signature is
unchanged so the dispatcher can swap backends without recompiling.

All kernel template choices (persistent grid, swizzled tile order,
fp32 accumulator, ``num_warps=16/num_stages=3/waves_per_eu=4``) match
``aiter/ops/triton/comms/all_gather.py`` and ``reduce_scatter.py`` so the
LDS/register pressure characteristics carry over.
"""

from __future__ import annotations

import triton
import triton.language as tl

try:  # Iris is optional; we fall through to staged-AG when missing.
    import iris  # type: ignore  # noqa: F401
    _IRIS_AVAILABLE = True
except (ImportError, RuntimeError):
    _IRIS_AVAILABLE = False


@triton.jit
def _fused_ag_mm_staged_kernel(
    A_full_ptr,
    B_ptr,
    Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Staged-AG variant: ``A_full`` is the result of a prior dist all-gather.

    This is the fallback path when Iris is unavailable. The wrapper has
    already paid for the gather (NCCL/RCCL); this kernel just does the
    persistent-tile GEMM that AITER would do for a normal A16W16 op.
    The schedule is identical to ``aiter/ops/triton/_triton_kernels/gemm/basic/``.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
        mask_m = rm < M
        mask_n = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mask_k = rk < K
            a_ptrs = A_full_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)

        y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.type.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# Iris-aware variant guarded by the Iris probe so we never reference an
# undefined symbol when Iris isn't installed.
if _IRIS_AVAILABLE:
    import iris  # type: ignore  # noqa: F811

    @triton.jit
    def _fused_ag_mm_iris_kernel(
        A_shard_ptr,
        B_ptr,
        Y_ptr,
        M_global, M_shard, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_ym, stride_yn,
        cur_rank: tl.constexpr,
        world_size: tl.constexpr,
        heap_bases: tl.tensor,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Iris variant: per-shard single-owner tiling.

        Every gathered row lives on exactly one rank (the shard that owns it),
        so each output row should be computed **once** from its owner's
        symmetric-memory shard. We therefore iterate the tile space as
        ``(shard, local_m_tile, n_tile)`` and resolve the owner directly from
        ``shard`` — instead of the previous schedule, which iterated global-M
        tiles and ran a full masked-to-zero K-loop for *every* rank
        (``world_size×`` the MFMA work; see ``docs/AITER_FUSED_KERNELS.md`` §13).

        For each tile we pull ``A`` from rank ``shard`` via ``iris.load`` (the
        pointer translation is identity when ``shard == cur_rank``, so one code
        path covers local and remote strips) and run a single K-loop. Iterating
        per shard also means the schedule is correct for any ``M_shard`` —
        tiles never straddle a shard boundary — so no ``BLOCK_M | M_shard``
        constraint is needed.

        ``M_global`` is retained in the signature for call-site parity; bounds
        come from ``M_shard``. ``heap_bases`` is the per-rank base-pointer table
        (see ``aiter/ops/triton/comms/iris.py``).
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M_shard, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        tiles_per_shard = num_pid_m * num_pid_n
        total_tiles = world_size * tiles_per_shard

        for tile_id in range(pid, total_tiles, NUM_SMS):
            shard = tile_id // tiles_per_shard
            local_id = tile_id % tiles_per_shard

            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = local_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((local_id % num_pid_in_group) % group_size_m)
            pid_n = (local_id % num_pid_in_group) // group_size_m

            rm_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            rm_local = tl.max_contiguous(tl.multiple_of(rm_local, BLOCK_M), BLOCK_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
            mask_m = rm_local < M_shard
            mask_n = rn < N
            rm_global = shard * M_shard + rm_local

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                rk = k0 + tl.arange(0, BLOCK_K)
                mask_k = rk < K

                a_ptrs = (
                    A_shard_ptr
                    + rm_local[:, None] * stride_am
                    + rk[None, :] * stride_ak
                )
                a_mask = mask_m[:, None] & mask_k[None, :]
                a = iris.load(a_ptrs, cur_rank, shard, heap_bases, mask=a_mask)

                b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                acc += tl.dot(a, b)

            y_ptrs = Y_ptr + rm_global[:, None] * stride_ym + rn[None, :] * stride_yn
            tl.store(y_ptrs, acc.to(Y_ptr.type.element_ty),
                     mask=mask_m[:, None] & mask_n[None, :])
