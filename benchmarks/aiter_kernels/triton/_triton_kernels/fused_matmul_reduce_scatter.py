"""Triton kernel: fused matmul + reduce-scatter (row-parallel TP linear).

Schedule (Phase 2/3 refactor; see ``docs/AITER_FUSED_KERNELS.md`` §13):
**write-to-per-source-slot + local reduce**, with staggered communication
order and optional per-tile signaling.

The earlier Iris schedule pushed every output tile straight onto the owner's
single ``[M_shard, N]`` buffer with a fabric ``iris.atomic_add``. All
``world_size`` ranks accumulated into the *same* destination addresses, so the
reduction serialized on contended bf16 fabric atomics (see Flux / FlashOverlap
discussion in the refactor doc). The new schedule removes both the atomics and
the contention:

  Stage 1 — write (``_fused_mm_rs_iris_write_kernel``):
    for (dst_shard, m_local_tile, n_tile) in staggered order:
      acc = local GEMM of the rows owned by dst_shard
      iris.store(scratch_on_dst[cur_rank, m_local_tile, n_tile], acc)   # no atomic
      [if SIGNAL] iris.atomic_add(flags_on_dst[tile], 1, sem="release", scope="sys")

  Stage 2 — reduce (``_reduce_partials_kernel`` / ``_reduce_partials_signal_kernel``):
    Y_shard[m, n] = scale * sum_{s in world} scratch[s, m, n]   # local, contention-free

``scratch`` is a symmetric ``[world, M_shard, N]`` buffer: each source rank
owns a *distinct* slot on every destination, so the cross-fabric writes never
collide. The reduction is then a local, vectorized add-tree over the
``world_size`` slots (with the ``avg`` scale folded in — no separate pass).

Destination order is **staggered** per rank (peers first, self last, each rank
starting at a different peer) to spread fabric injection and avoid incast onto
one destination (Flux "communication order selection").

Two reduce variants:
  * default (``_reduce_partials_kernel``): a host ``barrier()`` separates the
    write and reduce stages — simple and robust.
  * opt-in (``_reduce_partials_signal_kernel``, env
    ``AITER_KERNELS_MM_RS_SIGNAL=1``): per-tile producer/consumer signals
    replace the global barrier so a destination can reduce early-arriving
    tiles while peers are still writing later ones. Deadlock-free by
    construction (every rank signals every tile exactly once). Experimental —
    validate on a real multi-GPU Iris node before relying on it.

When Iris is absent we fall back to ``_fused_mm_rs_staged_kernel`` + a real
``dist.reduce_scatter_tensor`` (unchanged).
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
    def _fused_mm_rs_iris_write_kernel(
        A_ptr,
        B_ptr,
        scratch_ptr,        # symmetric [world, M_shard, N], one source-slot per rank
        flags_ptr,          # symmetric [num_m_tiles * num_n_tiles] arrival counters
        M_global, M_shard, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_ss, stride_sm, stride_sn,
        cur_rank: tl.constexpr,
        world_size: tl.constexpr,
        heap_bases: tl.tensor,
        SIGNAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        """Phase 2/3 write stage: GEMM each tile, then ``iris.store`` it into the
        destination rank's per-source slot (no atomics, no contention).

        Tiling is *per destination shard* — the tile space is
        ``world_size * cdiv(M_shard, BLOCK_M) * cdiv(N, BLOCK_N)`` — so a tile
        never straddles two shards and routing is exact regardless of whether
        ``BLOCK_M`` divides ``M_shard``.

        Destination order is staggered: logical step ``i`` maps to physical
        destination ``(cur_rank + 1 + i) % world_size`` so peers are served
        before self and ranks start at different destinations (spreads incast).

        When ``SIGNAL`` is set, each completed tile bumps a per-tile arrival
        counter on the destination with a *release* system-scope atomic so the
        consumer's *acquire* read observes the data write that precedes it.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M_shard, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        tiles_per_shard = num_pid_m * num_pid_n
        total_tiles = world_size * tiles_per_shard

        for tile_id in range(pid, total_tiles, NUM_SMS):
            logical = tile_id // tiles_per_shard
            dst_rank = (cur_rank + 1 + logical) % world_size
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
            rm_global = dst_rank * M_shard + rm_local

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                rk = k0 + tl.arange(0, BLOCK_K)
                mask_k = rk < K
                a_ptrs = A_ptr + rm_global[:, None] * stride_am + rk[None, :] * stride_ak
                b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(a, b)

            tile_mask = mask_m[:, None] & mask_n[None, :]
            p_ptrs = (
                scratch_ptr
                + cur_rank * stride_ss
                + rm_local[:, None] * stride_sm
                + rn[None, :] * stride_sn
            )
            iris.store(
                p_ptrs,
                acc.to(scratch_ptr.type.element_ty),
                cur_rank,
                dst_rank,
                heap_bases,
                mask=tile_mask,
            )

            if SIGNAL:
                # Canonical (row-major) tile index — must match the consumer's
                # indexing in _reduce_partials_signal_kernel.
                flag_idx = pid_m * num_pid_n + pid_n
                iris.atomic_add(
                    flags_ptr + flag_idx,
                    1,
                    cur_rank,
                    dst_rank,
                    heap_bases,
                    sem="release",
                    scope="sys",
                )

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


@triton.jit
def _reduce_partials_kernel(
    scratch_ptr,        # [world, M_shard, N] symmetric, this rank's local view
    Y_ptr,              # [M_shard, N] final shard (local)
    M_shard, N,
    stride_ss, stride_sm, stride_sn,
    stride_ym, stride_yn,
    inv_scale,          # 1.0 for sum, 1/world for avg
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 2 reduce stage (barrier-gated): local add-tree over source slots.

    After the host ``barrier()`` every source rank has written its partial into
    ``scratch[source]``. Each program reduces one output tile by summing the
    ``world_size`` slots locally and folding in the ``avg`` scale.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M_shard)[:, None] & (rn < N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(world_size):
        p = scratch_ptr + s * stride_ss + rm[:, None] * stride_sm + rn[None, :] * stride_sn
        acc += tl.load(p, mask=mask, other=0.0).to(tl.float32)

    acc = acc * inv_scale
    y = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y, acc.to(Y_ptr.type.element_ty), mask=mask)


@triton.jit
def _reduce_partials_signal_kernel(
    scratch_ptr,        # [world, M_shard, N] symmetric, this rank's local view
    Y_ptr,              # [M_shard, N] final shard (local)
    flags_ptr,          # [num_m_tiles * num_n_tiles] arrival counters (local view)
    M_shard, N,
    stride_ss, stride_sm, stride_sn,
    stride_ym, stride_yn,
    inv_scale,
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Phase 3 reduce stage (per-tile signaling, no global barrier).

    Persistent over output tiles. For each tile we spin on its arrival counter
    until all ``world_size`` sources have signalled (acquire read pairs with the
    producer's release atomic), then reduce that tile. This lets a destination
    consume early-arriving tiles while peers are still writing later ones.

    Deadlock-free: each rank's write kernel signals every tile exactly once, so
    every counter is guaranteed to reach ``world_size``. Tile indexing here is
    canonical row-major to match the producer's ``flag_idx``.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_shard, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Wait until every source has written this tile. atomic_add(...,0)
        # returns the current value with acquire ordering at system scope.
        while tl.atomic_add(flags_ptr + tile_id, 0, sem="acquire", scope="sys") < world_size:
            pass

        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm < M_shard)[:, None] & (rn < N)[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(world_size):
            p = scratch_ptr + s * stride_ss + rm[:, None] * stride_sm + rn[None, :] * stride_sn
            acc += tl.load(p, mask=mask, other=0.0).to(tl.float32)

        acc = acc * inv_scale
        y = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
        tl.store(y, acc.to(Y_ptr.type.element_ty), mask=mask)
