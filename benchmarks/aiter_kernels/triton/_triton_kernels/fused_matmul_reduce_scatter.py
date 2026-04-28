"""Triton kernel: fused matmul + reduce-scatter (row-parallel TP linear).

Schedule: **compute-then-push-reduce** (see ``aiter_kernels/README.md §3.1``).

For each output tile ``(m_tile, n_tile)`` produced locally:

  acc = 0
  for k_tile:
    acc += mfma(tl.load(A[m_tile, k_tile]), tl.load(B[k_tile, n_tile]))
  dst_rank = m_tile_global // M_shard
  iris.put(local_partial[m_tile_local, n_tile], acc, src=cur_rank,
           dst=dst_rank, reduce="add")

After every rank pushes, each destination's ``local_partial`` holds the
sum of contributions from all ``world_size`` ranks for the rows it owns.
A second mini-kernel (``_avg_kernel`` below) divides by ``world_size``
when ``reduce_op == "avg"`` and writes the final shard.

The K-loop is **completely local** — comm is moved out of the critical
GEMM path so the MFMA pipeline runs at near-peak TFLOPs. xGMI atomic-add
on MI300/MI355 is fast enough that this beats compute → all_reduce → scatter.
"""

from __future__ import annotations

import triton
import triton.language as tl

try:
    import iris  # type: ignore  # noqa: F401
    _IRIS_AVAILABLE = True
except (ImportError, RuntimeError):
    _IRIS_AVAILABLE = False


@triton.jit
def _fused_mm_rs_staged_kernel(
    A_ptr,
    B_ptr,
    Y_full_ptr,
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
    """Staged-RS variant: produce the full Y matrix; wrapper does the RS.

    Used when Iris isn't available. ``Y_full`` is later passed to
    ``dist.reduce_scatter_tensor`` by the wrapper, which is the same
    transport ``_fallback.fused_matmul_reduce_scatter_fallback`` uses.
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
            a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)

        y_ptrs = Y_full_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(Y_full_ptr.type.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


if _IRIS_AVAILABLE:
    import iris  # type: ignore  # noqa: F811

    @triton.jit
    def _fused_mm_rs_iris_kernel(
        A_ptr,
        B_ptr,
        partial_ptr,        # symmetric-memory partial-sum buffer, shape [M_shard, N], one per rank
        M_global, M_shard, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_pm, stride_pn,
        cur_rank: tl.constexpr,
        world_size: tl.constexpr,
        heap_bases: tl.tensor,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Iris variant: push tile partials to the destination rank with atomic add.

        The K-loop runs purely local. Once each output tile finishes, we
        compute its destination rank (``m_tile_global // M_shard``) and use
        ``iris.atomic_add`` (or ``iris.put`` with reduce="add" semantics) to
        accumulate the partial into the destination's symmetric ``partial_ptr``
        buffer. After all ranks complete, the per-rank ``partial_ptr`` already
        holds the reduce-summed shard; the wrapper scales by ``1/world`` for
        ``reduce_op == "avg"``.
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
            for k0 in range(0, K, BLOCK_K):
                rk = k0 + tl.arange(0, BLOCK_K)
                mask_k = rk < K
                a_ptrs = A_ptr + rm_global[:, None] * stride_am + rk[None, :] * stride_ak
                b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(a, b)

            # First row of this tile picks the destination shard. Whole tile
            # belongs to one dst rank because BLOCK_M divides M_shard (asserted
            # by the wrapper), which means we can do a single iris.atomic_add
            # call instead of per-row routing.
            tile_first_row = pid_m * BLOCK_M
            dst_rank = tile_first_row // M_shard
            rm_local = rm_global - dst_rank * M_shard
            mask_local = (rm_local >= 0) & (rm_local < M_shard)
            p_ptrs = partial_ptr + rm_local[:, None] * stride_pm + rn[None, :] * stride_pn

            iris.atomic_add(
                p_ptrs,
                acc.to(partial_ptr.type.element_ty),
                cur_rank,
                dst_rank,
                heap_bases,
                mask=(mask_local & mask_m)[:, None] & mask_n[None, :],
            )


@triton.jit
def _avg_kernel(
    Y_ptr,
    M_shard, N,
    stride_ym, stride_yn,
    inv_world,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scale the destination shard by ``1/world_size``.

    Used when ``reduce_op == "avg"``. Cheap (memory-bound, ~0.5% of MM+RS
    time) so we don't bother fusing this into the MM kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M_shard)[:, None] & (rn < N)[None, :]
    ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32) * inv_world
    tl.store(ptrs, val.to(Y_ptr.type.element_ty), mask=mask)
