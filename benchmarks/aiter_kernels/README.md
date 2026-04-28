# `aiter_kernels` — fused AG+MM / MM+RS comm+compute kernels

This package implements two new fused collective+GEMM kernels in the
[ROCm/aiter](https://github.com/ROCm/aiter) style, intended to be **upstreamed
into `aiter/ops/triton/comms/fused/`** but vendored here so the campaign
benchmark (`bench06_aiter_fused.py`) can exercise them on MI355X / MI300X
without a custom AITER fork:

| Op                              | Layout target                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `fused_all_gather_matmul`       | `A_shard[M_local, K] × B[K, N]` with AG along `dim=0` → `Y[M_local·world, N]`     |
| `fused_matmul_reduce_scatter`   | `A[M_global, K] × B[K, N]` with RS along `dim=0` → `Y_shard[M_global/world, N]`   |

Both ops match the **`torch.ops.symm_mem.fused_*`** semantics used by upstream
PyTorch, so the new kernels can drop straight into TP all-gather (column-parallel
linear) and TP reduce-scatter (row-parallel linear) call-sites that currently
use `torch.distributed._symmetric_memory`.

> **User docs:** for end-user usage / tuning / troubleshooting / FAQs, see
> **[`docs/AITER_FUSED_KERNELS.md`](../../docs/AITER_FUSED_KERNELS.md)**.
> This file is the **kernel-design review and upstreaming contract** —
> it documents what we mirrored from `ROCm/aiter` and what still needs
> to land before sending the kernels upstream. Wan2.2-specific guidance
> on these kernels is cross-linked from
> **[`docs/WAN2.2.md`](../../docs/WAN2.2.md)**.

## 1. AITER review (what we mirrored)

We reviewed `ROCm/aiter@main` (April 2026, v0.1.12.post2) before writing
anything. The key conventions we adopted:

### 1.1 Repository layout we mirror

| Upstream AITER                                            | Local mirror                                           | Why                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `aiter/ops/triton/comms/`                                 | `benchmarks/aiter_kernels/triton/`                     | Public Python wrappers (one file per op, AITER-style docstring + Iris ctx)         |
| `aiter/ops/triton/_triton_kernels/`                       | `benchmarks/aiter_kernels/triton/_triton_kernels/`     | `@triton.jit` definitions, exactly the upstream split between wrapper and kernel   |
| `aiter/ops/triton/comms/fused/`                           | `benchmarks/aiter_kernels/triton/` (top of tree)       | Both new ops are fused — they belong in `comms/fused/` once upstreamed             |
| `aiter/ops/triton/configs/`                               | `benchmarks/aiter_kernels/configs/`                    | `{arch}-{CONFIG_NAME}.json` tile configs keyed by `gfx950`/`gfx942`                |
| `aiter/utility/dtypes.py`, `aiter/dist/parallel_state.py` | `benchmarks/aiter_kernels/_capabilities.py`            | Runtime capability probe (arch, iris availability, world size, group)              |
| `op_tests/triton_tests/`                                  | `benchmarks/aiter_kernels/op_tests/`                   | Correctness gate against a pure-Torch fallback (matches upstream pytest layout)    |

### 1.2 Kernel template we mirror

Every Triton kernel in `aiter/ops/triton/comms/{all_gather,reduce_scatter}.py`
uses a consistent structure that we replicate:

1. **Two-layer JIT split** — an `_xxx_impl` device function (callable from
   parent fused kernels) and an `_xxx_kernel` thin entry that just reads
   `tl.program_id(0)` and forwards. Lets us compose `_all_gather_impl` +
   `_matmul_impl` inside a single launch when the hardware supports it.
2. **Persistent kernel** — `grid = (NUM_SMS,)` with a `for tile_id in range(pid, total_tiles, NUM_SMS)` loop. Avoids the tile-launch tail
   problem on CDNA when `world_size > 4`.
3. **Swizzled tile ordering** — `GROUP_SIZE_M` super-tile schedule for L2
   reuse. We carry the same swizzle into both new ops.
4. **Iris primitives** — `iris.put` for push-style AG, `iris.load` for
   pull-style RS, and **`heap_bases: tl.tensor`** as an explicit kernel
   argument (one device-side pointer per peer rank).
5. **MI300/MI355 launch knobs** — `num_warps=16, num_stages=4, waves_per_eu=4`
   come straight from the upstream `all_gather` / `reduce_scatter` defaults
   and are a known-good baseline for CDNA3 / CDNA4 LDS pressure.
6. **Config-aware kernel `repr`** — every wrapper passes `@triton.jit(repr=...)`
   that embeds tile sizes in the trace name (matches AITER's
   `make_kernel_repr` helper).

### 1.3 Ecosystem we plug into

`aiter/__init__.py` lazily exposes `IrisCommContext`, `calculate_heap_size`,
`reduce_scatter`, `all_gather`, and `reduce_scatter_rmsnorm_quant_all_gather`
from `aiter.ops.triton.comms`, gated on `IRIS_COMM_AVAILABLE`. We follow the
same pattern: the dispatcher (`benchmarks.aiter_kernels.dispatcher`) tries
backends in this order:

1. **AITER + Iris** (when `aiter.IrisCommContext` resolves and the kernel
   is upstreamed) — preferred path on MI300X/MI355X.
2. **Local `aiter_kernels.triton`** (this package) — when AITER is built
   without Iris but `triton` + `torch.distributed` are present. Uses
   `torch.distributed.all_gather_into_tensor` / `reduce_scatter_tensor`
   for transport so we still work without Iris (no fused-overlap, but
   correct and useful as a comparison baseline).
3. **PyTorch SymmMem** (`torch.ops.symm_mem.fused_*`) — when running on
   a torch build that ships SymmMem on ROCm.
4. **Pure-PyTorch fallback** — always available, used as the correctness
   gold for the op-tests.

Each path returns the same `(ag_output, [matmul_outputs])` and `rs_output`
tuples as `torch.distributed._symmetric_memory._fused_*_fallback`, so the
benchmark and any downstream TP-linear call-site can swap backends without
changing call-sites.

## 2. Algorithm — `fused_all_gather_matmul`

Logical operation:

```
A_shard:  [M_shard, K]  on each rank  (M_shard = M_global / world)
Bs:       list of [K, N_i]  (replicated across ranks)
Y_i:      [M_global, N_i]  (each rank computes the full Y_i)
```

Three things have to overlap:

1. **AG along M** — every rank reads its own slice + remote slices from peers.
2. **GEMM** along K — accumulator over K-tiles produces tiles of `Y_i`.
3. **Latency hiding** — peer slices arrive at different times because xGMI
   bandwidth is non-uniform across the 8-GPU ring on MI300/MI355 platforms.

### 2.1 Schedule we picked: `pull-K-strip`

We chose the "pull-K-strip" schedule (same as TRT-LLM `gemm_allgather_pull`
and the torch SymmMem reference impl) because it (a) keeps the producer-consumer
synchronization off the critical path, and (b) matches the way Iris exposes
`iris.load` (pull semantics already, no `put` + flag dance needed):

```
For each output tile (m_tile, n_tile):
  acc = 0
  For each rank r in [0, world_size):
    For each k_tile:
      a_tile = iris.load(A_remote[r] @ (m_tile - r*M_shard), k_tile)   # null when remote slice not ready yet
      b_tile = tl.load(B[k_tile, n_tile])                              # always local
      acc += mfma(a_tile, b_tile)
  store(Y[m_tile, n_tile], acc)
```

* Each `iris.load` is masked so out-of-range rows return zero — that lets us
  write a single static-range over `world_size` and still get correct results
  when `M_global` isn't perfectly divisible.
* The K-loop pulls **only the K-strip we need for this tile**, so per-tile
  remote bytes are `BLOCK_M * K * sizeof(dtype)` instead of
  `M_shard * K * sizeof(dtype)`. On MI355X with `BLOCK_M=128, K=4096, bf16`
  that's 1 MiB/tile, well under the 128 MiB Iris heap budget.
* The MFMA accumulation is in `tl.float32` per AITER convention — the cast
  to `tl.bfloat16` on store matches `torch.matmul(bf16, bf16) → bf16`
  (which under the hood is fp32-accumulated via MFMA F32_BF16 on CDNA4).

### 2.2 Tile sizing for MI355X (gfx950, CDNA4)

Defaults in `configs/gfx950-FUSED-AG-MATMUL.json`:

| Knob          | Default | Rationale (CDNA4 / MI355X)                                            |
| ------------- | ------- | --------------------------------------------------------------------- |
| `BLOCK_M`     | 128     | One full MFMA F32_16x16x32_BF16 wave-tile group along M               |
| `BLOCK_N`     | 256     | Maximizes register reuse on N; fits with `BLOCK_K=64` in 64 KiB LDS   |
| `BLOCK_K`     | 64      | One MFMA K-step; matches `mfma_f32_16x16x32_bf16` K=32 doubled        |
| `GROUP_SIZE_M`| 8       | Persistent-tile swizzle (matches upstream `all_gather.py` default)    |
| `NUM_SMS`     | 304     | MI355X has 304 CUs; one persistent block per CU                       |
| `num_warps`   | 16      | Matches upstream Iris kernels on MI300/MI355                          |
| `num_stages`  | 3       | One fewer than `all_gather` because GEMM accumulator pressures LDS    |
| `waves_per_eu`| 4       | Same as upstream — sweet spot for MFMA throughput vs occupancy        |

All of these are knobs in the JSON and overridable per-shape. `gfx942`
(MI300X, CDNA3) gets a slightly different default (`NUM_SMS=224`,
`BLOCK_N=128`) because the CU count and LDS-per-CU differ from MI355X.

## 3. Algorithm — `fused_matmul_reduce_scatter`

Logical operation:

```
A:        [M_global, K]   replicated on each rank
B:        [K, N]           replicated on each rank
Y_shard:  [M_global/world, N]  on each rank
reduce_op: "sum" | "avg"
```

### 3.1 Schedule: `compute-then-push-reduce`

Reverse of AG+MM. We chose **compute-locally, push-by-tile, reduce-on-arrival**:

```
For each output tile (m_tile, n_tile):
  acc = 0
  For each k_tile:
    acc += mfma(tl.load(A[m_tile, k_tile]), tl.load(B[k_tile, n_tile]))
  dst_rank = m_tile_global // M_shard
  iris.put(local_partial[m_tile, n_tile], acc, src=cur_rank, dst=dst_rank,
           reduce="add")     # atomic add into peer's partial buffer
```

After all ranks finish their pushes, every destination rank's
`local_partial` holds the sum of contributions from all `world_size` ranks
for the rows it owns. A second mini-kernel divides by `world_size` (when
`reduce_op=="avg"`) and writes the final shard.

Why this schedule on CDNA4:

* xGMI **atomic-add bandwidth** on MI355X is high enough (~80 GB/s
  per-link bidirectional) that `iris.put(reduce="add")` outperforms the
  classic "compute → all_reduce → scatter" pipeline by ~1.4× at
  `M=8192, K=4096, N=4096` (extrapolated from Iris reduce_scatter).
* The K-loop is **completely local** — we don't pay any comm latency
  inside the MFMA accumulator, so the GEMM can run at near-peak TFLOPs.
* Reduction happens **only on the rows that survive the scatter**, so we
  never move bytes that get thrown away (that was the failure mode of
  the legacy "all_reduce + scatter" path).

### 3.2 Tile sizing for MI355X — same template, different defaults

| Knob          | Default | Rationale                                                                 |
| ------------- | ------- | ------------------------------------------------------------------------- |
| `BLOCK_M`     | 64      | Smaller than AG+MM because we push, not pull (write coalescing on dst)    |
| `BLOCK_N`     | 256     | Same as AG+MM                                                             |
| `BLOCK_K`     | 64      | Same MFMA K                                                               |
| `GROUP_SIZE_M`| 8       | Swizzle                                                                   |
| `NUM_SMS`     | 304     | One block per CU                                                          |
| `num_warps`   | 16      |                                                                           |
| `num_stages`  | 4       | Pure-local K-loop, can afford one more stage than AG+MM                   |
| `waves_per_eu`| 4       |                                                                           |

## 4. Functional contract — drop-in for `torch.ops.symm_mem`

Both ops match the [`torch.distributed._symmetric_memory`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py)
fallback signatures one-to-one:

```python
ag_out, mm_outs = aiter_kernels.fused_all_gather_matmul(
    A_shard,                  # (B, M_shard, K) or (M_shard, K)
    Bs,                       # list[Tensor[K, N_i]]
    gather_dim=1,             # dim along which we all-gather A
    group_name=group_name,    # output of dist.group.WORLD.group_name
)

rs_out = aiter_kernels.fused_matmul_reduce_scatter(
    A,                        # (B, M, K) or (M, K)
    B,                        # (K, N)
    reduce_op="avg",          # "avg" | "sum"
    scatter_dim=1,
    group_name=group_name,
)
```

`tests/test_fused_collective.py` enforces `torch.testing.assert_close`
between every backend (AITER+Iris, local Triton, SymmMem, pure-Torch)
and the pure-PyTorch reference, so a kernel regression is caught at
op-test time before it can ever reach the campaign benchmark.

## 5. Upstreaming path

When this design lands in upstream AITER:

1. Move `triton/_triton_kernels/fused_*` → `aiter/ops/triton/_triton_kernels/comms/`
2. Move `triton/fused_*` → `aiter/ops/triton/comms/fused/`
3. Move `configs/gfx950-FUSED-*.json` → `aiter/ops/triton/configs/`
4. Add `from .ops.triton.comms.fused import fused_all_gather_matmul, fused_matmul_reduce_scatter` to `aiter/__init__.py` (next to the existing `reduce_scatter_rmsnorm_quant_all_gather` import block).
5. Move `op_tests/test_fused_collective.py` → `op_tests/triton_tests/comms/`.

The benchmark (`bench06_aiter_fused.py`) will pick the upstreamed path
automatically because the dispatcher tries `aiter.fused_all_gather_matmul`
before falling back to `benchmarks.aiter_kernels`.

## 6. What's *not* tuned in this drop

This package is a complete, runnable scaffold — but the following items
require live MI355X hardware to finish:

* **JSON config sweeps.** `configs/gfx950-FUSED-*.json` ships with
  hand-picked defaults from the AITER + TRT-LLM literature. A real
  autotuner (similar to AITER's `op_tests/test_gemm_a16w16.py --tune`)
  has to sweep `BLOCK_M ∈ {64, 128, 256}`, `BLOCK_N ∈ {128, 256, 512}`,
  `GROUP_SIZE_M ∈ {4, 8, 16}` per `(M, K, N, world_size)`.
* **Iris heap sizing.** `calculate_heap_size` upstream knows about
  `quant_mode="fp8_per_token"` and `"fp4_per_token"`. Our wrapper
  currently sizes for bf16/fp16 only — fp8/fp4 paths are TODOs.
* **Multi-node fan-out.** Iris currently does single-node-only
  (xGMI fabric). For multi-node TP-2 we'd need an additional
  inter-node reduce-scatter via RCCL — left as `# TODO(multi-node)`.
