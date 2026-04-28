"""Triton kernel: fused all-gather + matmul (column-parallel TP linear).

Schedule: **pull-K-strip** (see ``aiter_kernels/README.md §2.1``).

For each output tile ``(m_tile, n_tile)``:

  acc = 0
  for r in [0, world):
    for k_tile:
      a_tile = iris.load(A_remote[r] @ (m_tile - r*M_shard), k_tile)   # masked
      b_tile = tl.load(B[k_tile, n_tile])
      acc   += mfma(a_tile, b_tile)
  store(Y[m_tile, n_tile], acc)

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
        """Iris variant: pull each rank's K-strip directly into the GEMM.

        Each output tile loads ``BLOCK_M * K * sizeof(dtype)`` from each peer
        rank's symmetric-memory shard via ``iris.load``. The K-loop interleaves
        comm and MFMA so the load latency overlaps the previous tile's MFMA.

        ``heap_bases`` is the per-rank base pointer table (see
        ``aiter/ops/triton/comms/iris.py``); ``iris.load(ptr, src, dst, heap_bases)``
        translates a virtual symmetric-memory pointer into the dst-rank's
        physical pointer at runtime.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M_global, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        total_tiles = num_pid_m * num_pid_n

        for tile_id in range(pid, total_tiles, NUM_SMS):
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            rm_global = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            rm_global = tl.max_contiguous(tl.multiple_of(rm_global, BLOCK_M), BLOCK_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
            mask_m = rm_global < M_global
            mask_n = rn < N

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for src_rank in tl.static_range(world_size):
                rm_local = rm_global - src_rank * M_shard
                in_rank = (rm_local >= 0) & (rm_local < M_shard)
                rm_local_clamped = tl.where(in_rank, rm_local, 0)

                for k0 in range(0, K, BLOCK_K):
                    rk = k0 + tl.arange(0, BLOCK_K)
                    mask_k = rk < K

                    a_ptrs = (
                        A_shard_ptr
                        + rm_local_clamped[:, None] * stride_am
                        + rk[None, :] * stride_ak
                    )
                    a_mask = (in_rank & mask_m)[:, None] & mask_k[None, :]
                    if src_rank == cur_rank:
                        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    else:
                        a = iris.load(a_ptrs, cur_rank, src_rank, heap_bases, mask=a_mask)

                    b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                    acc += tl.dot(a, b)

            y_ptrs = Y_ptr + rm_global[:, None] * stride_ym + rn[None, :] * stride_yn
            tl.store(y_ptrs, acc.to(Y_ptr.type.element_ty),
                     mask=mask_m[:, None] & mask_n[None, :])
