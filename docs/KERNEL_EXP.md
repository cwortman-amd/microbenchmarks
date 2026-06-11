# Fusing Collectives and GEMM on MI355X: A Study of AG+MM and MM+RS Kernels

## Executive Summary

This study explored how far we could push fused collective + GEMM operators for tensor-parallel inference on AMD MI355X/CDNA4. The target operators were:

- `AG+MM`: all-gather an activation shard, then multiply by a weight matrix.
- `MM+RS`: multiply locally, then reduce-scatter the output shard.

The main finding is architectural: real overlap requires the communication and compute portions to live inside the same persistent kernel if they need to share LDS. LDS is scoped to a workgroup and a single kernel launch. It cannot be used as a handoff buffer between separately launched producer and consumer kernels.

The best long-term direction is therefore a single HipKittens/Iris persistent AG+MM kernel with producer waves, consumer waves, shared LDS staging, and double-buffered K-panel reuse. A first native HK/Iris backend was implemented, verified, and benchmarked. The default native AG+MM path produced a positive small-shape speedup in the 2-rank smoke. The first N-tile reuse variant was architecturally valid and correct, but slower because it introduced VGPR spilling.

The result is not yet a production-optimized fused kernel, but it is a useful map of what works, what does not, and why. For MM+RS specifically, split-phase instrumentation overturned the early assumption that the writer or reducer dominates: the two Iris host barriers are ~70% of MM+RS time, while the writer and reducer kernels are cheap. Acting on that, double-buffering the scratch to drop the post-reducer barrier was the largest MM+RS win **at the small 2-rank smoke shapes** (1.56x; 1.62x with the optional vec4+swizzle reducer). Removing the *remaining* host barrier is worth a further ~2.5x in principle, but every device-side attempt (hand-rolled flags, Iris's own `device_barrier()`, write-through stores) is intermittently incorrect: this is a framework-level gap — Iris exposes no device-side remote-write completion primitive usable from the native HK C++ writer — not a kernel-tuning problem. See the RFC / Blocker Note below. The reducer micro-optimizations (scalar-specialized, `vec4`) and destination swizzle are correct but only low-single-digit gains, consistent with the instrumentation.

> **DECISIVE UPDATE (production-accurate sharding).** The headline speedups above were measured with **toy shapes and a replicated-weight model** that overstated per-rank GEMM work by `tp×`. After fixing the benchmark to model **true Megatron TP sharding** (column-parallel AG+MM with weight `[K, N/tp]`; row-parallel MM+RS with weight `[K/tp, N]`) and re-running at the real Wan2.2 / Odyssey shapes at TP=8, the picture inverts: the fused kernels are **~0.35–0.45x of unfused PyTorch** on production shapes, the `double_buffer+vec4+swizzle` advantage over the default kernel shrinks to ~1.02–1.03x, and the **theoretical overlap ceiling collapses to ~1.1–1.6x** across *all three* patterns (AG+MM, MM+RS, **and MM+AR**, the dominant production pattern). No available kernel — HK native, the upstream Iris fused `matmul_all_reduce` (which does not even compile on this stack), or a portable pipelined overlap — beats unfused PyTorch + RCCL at these shapes. See **"Production-Accurate TP Sharding Regression"** immediately below; it supersedes the replicated-shape numbers throughout this document.

## Production-Accurate TP Sharding Regression (decisive; supersedes replicated-shape numbers)

Every earlier speedup in this document was measured either at tiny 2-rank smoke shapes
(e.g. `512,256,512`) or with a **replicated-weight** model where each rank held the full
`[K, N]` weight and redundantly computed the full `[M, N]`. That model overstates per-rank
FLOPs by `tp×` and therefore inflates every fused-vs-unfused ratio. The benchmark was fixed
to model real Megatron tensor-parallel linears, and the operators were re-run at the actual
customer shapes. This section is the trustworthy result.

### What the fix does (`benchmarks/bench06_aiter_fused.py`)

- `BENCH06_TP_SHARD=tp` (now default): **AG+MM is column-parallel** — weight sharded on `N`,
  so each rank holds `B[K, N/tp]` and computes `[M, K] @ [K, N/tp] = [M, N/tp]`.
  **MM+RS is row-parallel** — weight and activation sharded on the contraction `K`, so each
  rank computes `[M, K/tp] @ [K/tp, N] = [M, N]` partial, reduce-scattered to `[M/tp, N]`.
  Per-rank FLOPs become `2·M·K·(N/tp)` and `2·M·(K/tp)·N` respectively (not `2·M·K·N`).
  `BENCH06_TP_SHARD=replicated` restores the old model for A/B comparison only.
- `BENCH06_PAD_TO_KERNEL=1`: the HK fused prototype requires `M_global`/`M_shard` multiples of
  128, kernel-side `N` a multiple of 256, and kernel-side `K` a multiple of 64. The real shards
  (e.g. `M_shard = 4680/8 = 585`, column-parallel `N_shard = 13824/8 = 1728`) **violate these**,
  so unpadded the fused path errors out with `NotImplementedError`. This flag pads the global
  dims up so the per-rank shards satisfy the constraints, giving an **indicative (padded)** fused
  number. Rows are tagged `padded=True` + `req_shape` so a padded number is never mistaken for the
  true shape. (Wan2.2 pads `M 4680→5120`, AG `N 13824→14336`.)

### Production shapes — fused HK vs unfused PyTorch (TP=8, BF16, warmup=10/iters=50, padded)

MM+RS uses the documented best combo (`double_buffer` + `vec4` + `swizzle`); AG+MM uses the
default HK kernel (the env flags only affect MM+RS). Unfused is the layout-matched PyTorch baseline.

| pattern | requested (M,K,N) | run (padded) | fused ms | unfused ms | **speedup** |
|---|---|---|---|---|---|
| AG+MM | Wan2.2 4680,5120,13824 | 5120,5120,14336 | 0.710 | 0.251 | **0.35x** |
| AG+MM | Odyssey 1590,5120,13824 | 2048,5120,14336 | 0.330 | 0.139 | **0.42x** |
| AG+MM | Odyssey 4680,5120,13824 | 5120,5120,14336 | 0.698 | 0.264 | **0.38x** |
| MM+RS | Wan2.2 4680,5120,13824 | 5120,5120,13824 | 1.304 | 0.478 | **0.37x** |
| MM+RS | Odyssey 1590,5120,13824 | 2048,5120,13824 | 0.529 | 0.237 | **0.45x** |
| MM+RS | Odyssey 4680,5120,13824 | 5120,5120,13824 | 1.346 | 0.491 | **0.37x** |

The `double_buffer+vec4+swizzle` MM+RS that gave **1.56x** at `512,256,512` gives only
**~1.02–1.03x over the default kernel** here — the win essentially evaporates at production scale.
Best-case fused MM+RS is still **~0.37x of unfused** (≈2.7x slower).

### base2 proxy shapes — fused HK vs unfused (TP=8, no padding needed)

The base-2 shards happen to satisfy the tile constraints, so the fused path runs natively.
It still loses, confirming this is not shape-quantization noise:

| pattern | shape | speedup | | pattern | shape | speedup |
|---|---|---|---|---|---|---|
| AG+MM | 4096,4096,4096 | 0.52x | | MM+RS | 4096,4096,4096 | 0.50x |
| AG+MM | 8192,4096,4096 | 0.84x | | MM+RS | 8192,4096,4096 | 0.34x |
| AG+MM | 8192,8192,4096 | 0.85x | | MM+RS | 8192,8192,4096 | 0.38x |
| AG+MM | 16384,4096,4096 | 0.97x | | MM+RS | 16384,4096,4096 | 0.31x |

### MM+AR — the dominant production pattern (`benchmarks/bench13_iris_overlap.py`, TP=8)

Earlier profiling indicated Wan2.2 / SGLang TP inference is dominated by **MM+AR** (row-parallel
matmul + all-reduce), not MM+RS (which only appears under sequence parallelism). `bench13` scores
three implementations of the same row-parallel MM+AR against each other with the Flux ECT /
overlap-efficiency metric: an RCCL baseline (`unfused_mm_ar`), a FlashOverlap-style chunked
pipelined overlap (`pipelined_mm_ar`), and the upstream Iris fused kernel
(`iris.ops.matmul_all_reduce`).

| shape (M,K,N) | RCCL unfused ms | pipelined ms | **pipelined speedup** | overlap ceiling | Iris fused |
|---|---|---|---|---|---|
| Wan2.2 4680,5120,13824 | 0.76 | 0.88 | **0.87x** | 1.22x | **CompilationError** |
| Odyssey 1590,5120,13824 | 0.28 | 0.50 | **0.57x** | 1.16x | **CompilationError** |
| Odyssey 4680,5120,13824 | 0.76 | 0.87 | **0.88x** | 1.23x | **CompilationError** |
| base2 4096,4096,4096 | 0.23 | 0.46 | 0.51x | 1.14x | CompilationError |
| base2 8192,4096,4096 | 0.39 | 0.60 | 0.65x | 1.13x | CompilationError |
| base2 8192,8192,4096 | 0.44 | 0.61 | 0.72x | 1.16x | CompilationError |
| base2 16384,4096,4096 | 0.73 | 0.94 | 0.77x | 1.12x | CompilationError |

Two hard findings: (1) the portable pipelined overlap **loses to plain serial RCCL** everywhere,
and the overlap ceiling is the lowest of all three patterns (**~1.1–1.23x**); (2) the upstream
**Iris fused `matmul_all_reduce` does not compile** on this MI355X / ROCm / Triton build — every
shape fails in `_fused_matmul_all_reduce_kernel` (tritonblas `GemmContext` path). The framework's
own fused kernel for the *dominant* pattern is unavailable here, which is a concrete framework bug,
not something kernel tuning can address.

### Conclusions from the production-accurate regression

1. **No kernel-tuning path to parity exists at production shapes** for any of AG+MM, MM+RS, or
   MM+AR. Once sharding is honest the per-rank GEMM shrinks by `tp×`, comm dominates, and the
   *theoretical* perfect-overlap ceiling is only ~1.1–1.6x — a bar no available kernel reaches,
   while unfused PyTorch (hipBLASLt + RCCL) is very strong (AG+MM unfused hits ~374 TFLOP/s padded).
2. **The earlier large speedups were artifacts** of toy shapes plus the replicated-weight model.
   Keep `_kernel_pad_dims()` and the `padded`/`req_shape` tagging so this can never recur silently.
3. **Freeze the kernels as research baselines.** The shipped `double_buffer` MM+RS remains the
   correct MM+RS result and a useful framework stress test, but is not a production target.
4. **The leverage is elsewhere:** (a) confirm the customer's real pattern/layout (very likely
   MM+AR, possibly with sequence parallelism toggling AG+MM/MM+RS on); (b) the two pinned
   framework blockers — the missing device-side remote-write completion primitive (MM+RS,
   see RFC below) and the non-compiling Iris `matmul_all_reduce` (MM+AR); (c) RCCL/collective and
   GEMM tuning rather than bespoke fused kernels.

Artifacts: `results/prod-final/agmm_mmrs_*` (bench06) and `results/prod-final/mmar_*` (bench13).

## Background

Modern tensor-parallel transformer and diffusion blocks repeatedly execute linear layers that pair GEMM with a collective:

```text
Column-parallel linear:  all_gather(A_shard) -> A_full @ B
Row-parallel linear:     A @ B -> reduce_scatter(Y_full)
```

In the unfused implementation, each collective and GEMM is a separate operation. That has two immediate costs:

1. The collective latency is exposed because the GEMM cannot begin until the communication completes.
2. Intermediate tensors are written to and read from HBM, even when the data is only an implementation artifact.

On MI355X, this is especially painful because the dense BF16 MFMA units are fast enough that even small collective overheads become visible in end-to-end timing. The goal of fusion is to convert the collective from a serialized phase into a pipeline stage hidden behind the GEMM K-loop.

The study used three layers of implementation:

- A benchmark-facing AITER-style Python dispatcher in `benchmarks/aiter_kernels/`.
- Iris symmetric memory for GPU-initiated rank-to-rank data movement.
- HipKittens native CDNA4 tile primitives for LDS staging and BF16 MFMA.

## Operators Studied

### AG+MM

Logical operation:

```text
A_shard: [M / world, K] per rank
B:       [K, N]
A_full = all_gather(A_shard, dim=M)
Y      = A_full @ B
```

The optimization target is to avoid materializing `A_full` in HBM and instead pull each owner rank's A tile directly into the compute kernel.

### MM+RS

Logical operation:

```text
A:      [M, K]
B:      [K, N]
Y_full = A @ B
Y_out  = reduce_scatter(Y_full, dim=M)
```

The optimization target is to avoid writing a full unreduced `Y_full` and instead push or stage finalized output tiles directly to the destination rank.

## Baseline

The baseline used by `benchmarks/bench06_aiter_fused.py` is a sequential unfused implementation:

```text
AG+MM:  dist.all_gather_into_tensor(A_full, A_shard) -> torch.matmul(A_full, B)
MM+RS:  torch.matmul(A, B) -> dist.reduce_scatter_tensor(Y_shard, Y_full)
```

This baseline is simple and strong. On the Escher-like 8-rank shapes it already reaches hundreds of TFLOP/s for the GEMM portions, so a fused path has to both overlap communication and avoid adding scheduling overhead.

## Attempted Architectures

### 1. Staged Triton Path

The first runnable path used the existing AITER-style dispatcher and Triton kernels. This provided the API shape and benchmark integration before the native HK/Iris backend existed.

Characteristics:

- It exercised the same public fused API as the target backend.
- It could compare fused-style calls against the unfused baseline.
- It was useful for correctness and benchmarking plumbing.
- It did not provide the desired low-level persistent producer/consumer schedule.

The staged path was slower than the unfused baseline on large Escher-like shapes. That was expected once the kernel-level limitations became clear: the path added overhead without truly hiding communication behind MFMA work.

### 2. Iris Symmetric-Memory Path

The next step was to route tensors through Iris symmetric memory. Iris provides a symmetric heap and a device context that lets kernels translate a local pointer into the corresponding remote rank's heap address.

For AG+MM, the kernel can identify which rank owns a global M tile and load A from that rank's heap. For MM+RS, a writer kernel can store partial results into a symmetric scratch buffer on the destination rank.

This was a major architectural improvement, but the early kernels were still not optimized enough. Some runs were slower than staged or unfused baselines because they paid the complexity cost of Iris without enough compute reuse, coalescing, or persistent scheduling to compensate.

### 3. Native HipKittens/Iris Backend

The native backend moved the core compute path into `benchmarks/aiter_kernels/hipkittens_native/kernel.cpp`.

The key implementation pieces are:

- `iris_context_view`: lightweight device-side access to current rank, world size, and per-rank heap bases.
- Symmetric pointer translation: `A_shard` on the local rank can be translated into the owning remote rank's heap.
- HipKittens global-to-LDS tile loads for BF16 A and B tiles.
- Producer waves for global/symmetric loads.
- Consumer waves for LDS-to-register loads and BF16 MFMA.
- Double-buffering across K tiles with `tic`/`toc` LDS buffers.
- Chiplet-aware workgroup swizzling through `chiplet_transform_chunked`.

The native AG+MM kernel computes global M tiles directly from their owning rank's A shard:

```text
global row tile -> owner rank
owner A pointer -> symmetric heap translation
producer waves -> load A and B tiles into LDS
consumer waves -> run mma_ABt
store C tile
```

The native MM+RS path is split into two kernels:

```text
writer kernel:  compute GEMM tile and write to destination scratch[slot, row, col]
reduce kernel:  local rank reduces scratch slots into Y_shard
```

This MM+RS design is valid because the handoff is through global/symmetric memory, not LDS. It is also slower than the ideal persistent design because it introduces extra HBM traffic and a second kernel launch.

### 4. Rejected: Separate Producer and Consumer Kernels Sharing LDS

One proposed design had a collective producer kernel fill LDS and a separate GEMM consumer kernel read that LDS region. That design is invalid.

LDS is not a device-global scratchpad. It is:

- Per workgroup.
- Per kernel launch.
- Not visible to a later kernel.

Separate kernels can communicate through global memory, symmetric memory, or other persisted memory spaces. They cannot communicate through LDS.

This was the critical design correction in the study.

### 5. Valid but Slower: Separate Kernels With Global/Symmetric Staging

The valid split-kernel version is:

```text
kernel 1: collective/staging writes panel to global or symmetric memory
kernel 2: GEMM reads panel from global or symmetric memory into its own LDS
```

This is easier to debug, but it gives up the core performance advantage:

- The staged panel is written to HBM.
- The GEMM kernel reads it back from HBM.
- LDS bandwidth is only used inside each individual kernel, not as a producer/consumer handoff.
- Overlap depends on stream scheduling and resource availability rather than deterministic in-kernel coordination.

### 6. Best Direction: Single Persistent Producer/Consumer Kernel

The best AG+MM design is a single kernel with distinct producer and consumer waves in the same workgroup:

```text
for each K panel:
  producer waves load remote/local A into LDS buffer[next]
  producer waves load B tile(s) into LDS buffer[next]
  barrier
  consumer waves compute from LDS buffer[cur]
  ping-pong buffers
```

Why this is the right architecture:

- Producers and consumers share the same LDS address space.
- Barriers are meaningful because they synchronize waves in one kernel launch.
- A panel can be loaded once and reused across multiple N tiles.
- Remote A traffic can be reduced.
- The K-loop can hide remote load latency under MFMA work.

## Implementation Details

### Python Backend Adapter

`benchmarks/aiter_kernels/hipkittens.py` resolves the native extension and wraps lower-level dispatch functions into the public fused API.

Important behavior:

- It requires Iris symmetric tensors for native HK execution.
- It supports BF16 2-D tensors.
- AG+MM currently supports one B matrix and `gather_dim=0`.
- MM+RS supports `scatter_dim=0` and `reduce_op in {"sum", "avg"}`.
- Workspace tensors are allocated from the Iris symmetric heap and cached on the context.
- `B` is transposed to `[N, K]` because the HK kernel uses `mma_ABt`.

The latest adapter also includes an opt-in AG+MM reuse variant:

```bash
HIPKITTENS_AG_N_REUSE=2
```

When enabled and `N % 512 == 0`, the adapter dispatches `dispatch_ag_mm_reuse`; otherwise it uses the default `dispatch_ag_mm`.

### Native AG+MM Kernel

The default native AG+MM kernel uses:

- `BLOCK_SIZE = 64`
- `M_BLOCK = 2`, so the CTA covers 128 rows
- `N_BLOCK = 4`, so the CTA covers 256 columns
- 4 producer waves
- 8 consumer waves
- 12 total waves per CTA
- dynamic shared memory of 96 KiB

Each CTA maps to a global M/N tile, translates the A pointer to the owner rank's heap, and streams A and B through double-buffered LDS.

The default kernel built with:

```text
VGPRs: 135
ScratchSize: 0 bytes/lane
VGPR spills: 0
Occupancy: 3 waves/SIMD
```

### Native MM+RS Kernel

The native MM+RS writer reuses the same HK compute pattern, but its epilogue stores the computed tile into a destination rank's symmetric scratch buffer:

```text
scratch[cur_rank, local_row, col] on dest_rank
```

Then `mm_rs_reduce_kernel` reduces the `world` slots into the final local shard.

This prioritizes correctness and avoids remote atomic-add complexity. The tradeoff is that it is not yet the ideal overlapped MM+RS design.

### AG+MM N-Tile Reuse Variant

The first reuse experiment widened the CTA in the N dimension:

```text
default:  one CTA covers 128 x 256 output tile
reuse=2:  one CTA covers 128 x 512 output tile
```

The idea was to load the same A K-panel once and reuse it across two adjacent N blocks. This is the correct direction for reducing repeated remote A loads.

The result was slower because the widened accumulator footprint increased register pressure:

```text
VGPRs: 168
ScratchSize: 180 bytes/lane
VGPR spills: 108
Occupancy: 3 waves/SIMD
```

The experiment proved correctness and architectural validity, but not performance. It remains opt-in.

### Spill-Free AG+MM N-Reuse Variant

A second reuse variant was added after the first matrix. It is selected with:

```bash
HIPKITTENS_AG_N_REUSE=2_spillfree
```

This version keeps the "stage A once, reuse across two N blocks" idea, but avoids keeping both N-block accumulator sets live in the same thread. Instead, it reduces the CTA M height from 128 rows to 64 rows and splits the two N blocks across separate consumer wave groups. Each consumer group owns one 256-column N block, so the per-thread accumulator footprint is close to the default kernel while both consumer groups share the same staged A panel in LDS.

Compiler resource output for the new variant:

```text
VGPRs: 130
ScratchSize: 0 bytes/lane
SGPR spills: 0
VGPR spills: 0
Occupancy: 3 waves/SIMD
```

That confirms the variant is genuinely spill-free. It is not yet faster than the default HK/Iris kernel on the tiny `M=256,K=64,N=512` smoke shape, but it is faster than the original spilling reuse variant.

## Result Matrix

All times are wall-clock kernel benchmark times in milliseconds. Speedup is `comparison_baseline_ms / experiment_ms`; values below `1.0x` are slowdowns. The rows are grouped by experiment family because the shapes, rank counts, and backend maturity differ.

| Family | Backend / Architecture | Op | World | Shape `(M,K,N)` | Comparison Baseline | Baseline ms | Experiment ms | Speedup | Notes |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Baseline | Sequential unfused | AG+MM | 8 | `(1592,5120,13824)` | self | 0.297 | 0.297 | 1.00x | Escher-like small-M shape |
| Baseline | Sequential unfused | MM+RS | 8 | `(1592,5120,13824)` | self | 0.364 | 0.364 | 1.00x | Escher-like small-M shape |
| Baseline | Sequential unfused | AG+MM | 8 | `(4680,5120,13824)` | self | 0.719 | 0.719 | 1.00x | Escher-like larger-M shape |
| Baseline | Sequential unfused | MM+RS | 8 | `(4680,5120,13824)` | self | 0.910 | 0.910 | 1.00x | Escher-like larger-M shape |
| Early staged | Triton/staged, Iris unavailable | AG+MM | 8 | `(1592,5120,13824)` | sequential unfused | 0.298 | 0.550 | 0.54x | API and benchmark plumbing worked, but no real overlap |
| Early staged | Triton/staged, Iris unavailable | MM+RS | 8 | `(1592,5120,13824)` | sequential unfused | 0.364 | 0.924 | 0.39x | Extra staging dominated |
| Early staged | Triton/staged, Iris unavailable | AG+MM | 8 | `(4680,5120,13824)` | sequential unfused | 0.716 | 1.424 | 0.50x | Slower despite larger GEMM |
| Early staged | Triton/staged, Iris unavailable | MM+RS | 8 | `(4680,5120,13824)` | sequential unfused | 0.908 | 2.617 | 0.35x | Reduction path remained expensive |
| Iris symmetric | Iris workspace path | AG+MM | 8 | `(1592,5120,13824)` | sequential unfused | 0.297 | 2.052 | 0.14x | Symmetric path worked, but schedule was not efficient |
| Iris symmetric | Iris workspace path | MM+RS | 8 | `(1592,5120,13824)` | sequential unfused | 0.364 | 1.507 | 0.24x | Writer/reducer overhead exposed |
| Iris symmetric | Iris workspace path | AG+MM | 8 | `(4680,5120,13824)` | sequential unfused | 0.719 | 15.310 | 0.05x | Pathological AG schedule; strong evidence against this form |
| Iris symmetric | Iris workspace path | MM+RS | 8 | `(4680,5120,13824)` | sequential unfused | 0.910 | 3.211 | 0.28x | Valid but not competitive |
| Iris tuning | Base Iris tuned shape | AG+MM | 8 | `(8192,4096,4096)` | sequential unfused | 0.407 | 2.062 | 0.20x | Cost model showed overlap opportunity but kernel did not realize it |
| Iris tuning | Base Iris tuned shape | MM+RS | 8 | `(8192,4096,4096)` | sequential unfused | 0.421 | 1.610 | 0.26x | Same conclusion as AG+MM |
| HK native | HipKittens/Iris default | AG+MM | 2 | `(256,64,256)` | sequential unfused | 0.357 | 0.182 | 1.97x | First positive native AG+MM smoke |
| HK native | HipKittens/Iris default | MM+RS | 2 | `(256,64,256)` | sequential unfused | 0.056 | 0.194 | 0.29x | Correct, but reducer is intentionally simple |
| HK native | HipKittens/Iris default | AG+MM | 2 | `(256,64,512)` | sequential unfused | 0.092 | 0.111 | 0.83x | Default native kernel on N=512 smoke |
| HK native experiment | AG N-reuse = 2 | AG+MM | 2 | `(256,64,512)` | HK native default | 0.111 | 0.138 | 0.80x | Correct but slower due to VGPR spills |

## Requested Experiment Execution Matrix

After the initial study, the benchmark and test infrastructure was updated with:

- `benchmarks/aiter_kernels/hk_correctness.py`: distributed correctness smoke for default HK/Iris, `HIPKITTENS_AG_N_REUSE=2`, and `HIPKITTENS_AG_N_REUSE=2_spillfree`.
- `benchmarks/bench06_hk_experiments.py`: experiment matrix runner that invokes `bench06_aiter_fused.py`, records measured rows, and emits `hk_experiment_matrix.{csv,json}`.
- A hardened optional-backend probe in `benchmarks/aiter_kernels/_capabilities.py`, so an optional AITER import failure cannot make the HK path unimportable.

Correctness status:

```text
default HK/Iris:                  PASS
HIPKITTENS_AG_N_REUSE=2:          PASS
HIPKITTENS_AG_N_REUSE=2_spillfree: PASS
```

Run command:

```bash
python benchmarks/bench06_hk_experiments.py \
  --out results/hk-experiment-matrix-20260611 \
  --nproc 2 \
  --warmup 3 \
  --iters 10
```

Consolidated output:

```text
results/hk-experiment-matrix-20260611/hk_experiment_matrix.csv
results/hk-experiment-matrix-20260611/hk_experiment_matrix.json
```

| # | Requested Experiment | Status | Op | World | Shape `(M,K,N)` | Baseline | Baseline ms | Experiment ms | Speedup | Result |
| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Build best baseline inside AITER using HipKittens GEMM | measured | AG+MM | 2 | `(256,64,256)` | sequential unfused | 0.036 | 0.104 | 0.35x | Native HK/Iris GEMM path ran correctly, but this tiny shape is launch/overhead dominated and slower than unfused. |
| 2 | Add IRIS-backed remote A staging with panel reuse | measured | AG+MM | 2 | `(256,64,512)` | sequential unfused | 0.037 | 0.127 | 0.29x | Original `reuse=2` path is correct but spills 108 VGPRs and is slow. |
| 3 | Add double buffering | covered by default | AG+MM | 2 | `(256,64,512)` | sequential unfused | 0.038 | 0.107 | 0.35x | Default HK kernel already uses `tic/toc` LDS double buffering; no no-double-buffer variant was compiled for an isolated A/B. |
| 4 | Add LDS transpose on read-back path | unsupported | AG+MM | 2 | n/a | n/a | n/a | n/a | n/a | No LDS-transpose variant exists in the native extension yet. |
| 5 | Try chiplet-aware swizzling and grid mapping | covered by default | AG+MM | 2 | `(256,64,512)` | sequential unfused | 0.038 | 0.107 | 0.35x | Default HK kernels already call `chiplet_transform_chunked`; no no-swizzle variant was compiled for isolated A/B. |
| 6 | Rework MM+RS as chunked or hierarchical reduction | current split reducer | MM+RS | 2 | `(256,64,256)` | sequential unfused | 0.046 | 0.155 | 0.30x | Current implementation is writer + simple reducer, not a full hierarchical tiled reducer. |
| 7 | Evaluate FP8/FP6 variants | unsupported | AG+MM | 2 | n/a | n/a | n/a | n/a | n/a | Native HK/Iris extension currently exposes BF16 dispatch only. |
| 8 | Add spill-free AG+MM reuse design | measured | AG+MM | 2 | `(256,64,512)` | sequential unfused | 0.037 | 0.105 | 0.36x | New `2_spillfree` variant is correct and has zero spills; it beats spilling reuse but not default/unfused yet. |

Same-shape AG+MM comparison for the reuse experiments:

| Variant | Shape `(M,K,N)` | Time ms | Relative to Default HK | Compiler Spill Status |
| --- | --- | ---: | ---: | --- |
| Default HK/Iris | `(256,64,512)` | 0.107 | 1.00x | 0 VGPR spills |
| `HIPKITTENS_AG_N_REUSE=2` | `(256,64,512)` | 0.127 | 0.84x | 108 VGPR spills |
| `HIPKITTENS_AG_N_REUSE=2_spillfree` | `(256,64,512)` | 0.105 | 1.02x | 0 VGPR spills |

The most important result is the last table. The spill-free design does what it was supposed to do mechanically: it removes spills and recovers the loss from the original reuse implementation. On this small shape the gain over default HK is only about 2%, which is within normal benchmark noise; it should not be promoted to default based on this smoke alone. It should be carried into a larger K/N shape ladder where remote-A reuse has enough work to amortize the extra CTA scheduling.

### Follow-Up Shape Ladder Results

The next execution pass expanded the matrix runner to sweep a small shape ladder across four AG+MM variants:

- `default`: baseline HK/Iris native kernel.
- `reuse2`: original N-reuse implementation with known VGPR spills.
- `spillfree`: `HIPKITTENS_AG_N_REUSE=2_spillfree`.
- `auto`: new heuristic selector, currently choosing spill-free reuse when `N % 512 == 0` and `K >= 256`.

The updated runner writes:

```text
results/hk-shape-ladder-20260611/hk_shape_ladder.csv
results/hk-shape-ladder-20260611/hk_shape_ladder.json
```

Correctness status:

```text
default:   PASS
reuse2:    PASS
spillfree: PASS
auto:      PASS
```

AG+MM ladder results:

| Shape `(M,K,N)` | Variant | Time ms | Speedup vs Default HK | Speedup vs Unfused | VGPR Spills | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `(256,64,512)` | default | 0.130 | 1.00x | 0.34x | 0 | Default HK/Iris baseline |
| `(256,64,512)` | reuse2 | 0.134 | 0.96x | 0.33x | 108 | Still hurt by spills |
| `(256,64,512)` | spillfree | 0.108 | 1.19x | 0.41x | 0 | Best on this smoke shape |
| `(256,64,512)` | auto | 0.139 | 0.93x | 0.32x | 0 | Auto falls back to default for `K < 256`; timing variance dominated this tiny shape |
| `(512,256,512)` | default | 0.140 | 1.00x | 0.36x | 0 | Register-probe shape |
| `(512,256,512)` | reuse2 | 0.155 | 0.91x | 0.33x | 108 | Spilling reuse remains worse |
| `(512,256,512)` | spillfree | 0.120 | 1.17x | 0.43x | 0 | Spill-free reuse starts to pay off |
| `(512,256,512)` | auto | 0.113 | 1.24x | 0.45x | 0 | Auto selects spill-free and is best in this run |
| `(1024,1024,1024)` | default | 0.138 | 1.00x | 0.44x | 0 | Larger K-loop shape |
| `(1024,1024,1024)` | reuse2 | 0.215 | 0.64x | 0.29x | 108 | Register spills dominate |
| `(1024,1024,1024)` | spillfree | 0.140 | 0.99x | 0.44x | 0 | Neutral vs default |
| `(1024,1024,1024)` | auto | 0.135 | 1.03x | 0.46x | 0 | Slightly best, but still far from unfused |

The key finding is that spill-free reuse is now directionally useful: it beats the default HK/Iris kernel on the two N=512 ladder shapes and is neutral on the larger K-loop shape. The original spilling reuse is consistently worse. However, all native HK/Iris variants are still slower than the unfused PyTorch baseline on these short 2-rank tests. The fused kernels are not yet hiding enough communication or feeding MFMA efficiently enough to overcome launch/scheduling overhead.

ECT/overlap results on the middle ladder shape show the remaining gap:

| Shape `(M,K,N)` | Variant | Fused ms | ECT Fused ms | Overlap Efficiency | Speedup vs Unfused |
| --- | --- | ---: | ---: | ---: | ---: |
| `(512,256,512)` | default | 0.140 | 0.125 | -250.3% | 0.36x |
| `(512,256,512)` | spillfree | 0.120 | 0.104 | -201.6% | 0.42x |
| `(512,256,512)` | auto | 0.113 | 0.098 | -178.2% | 0.45x |

That is an improvement, but still a negative-overlap regime: the fused kernel is adding overhead beyond the unfused communication cost. The next kernel step should therefore focus on MFMA feed efficiency and per-panel scheduling, not only on dispatch selection.

Infrastructure updates from this pass:

- `bench06_hk_experiments.py` now accepts `--ladder-shapes`.
- The runner emits `hk_shape_ladder.{csv,json}`.
- The runner records compiler-resource metadata per variant: VGPRs, SGPRs, scratch bytes per lane, VGPR spills, and occupancy.
- `HIPKITTENS_AG_N_REUSE=auto` was added to the Python adapter.
- Future ladder rows now extract ECT and overlap fields correctly from `overlap.csv`.

## What the Numbers Mean

The early staged and Iris workspace rows are mostly negative results. They are still valuable because they ruled out two tempting but insufficient designs:

1. Merely wrapping collectives and GEMM in a fused-looking API does not create overlap.
2. Moving to symmetric memory is not enough if the kernel schedule repeats remote loads or adds extra HBM traffic.

The HK native AG+MM smoke is the first positive result. It is small, tile-aligned, and not representative of full production shapes, but it proves that a native producer/consumer HK kernel can beat the sequential unfused baseline when resource pressure is controlled.

The N-reuse row is the most important negative result. It used the right architectural idea, but the simple implementation expanded the accumulator footprint enough to spill VGPRs. The slowdown was therefore not an indictment of panel reuse. It was evidence that reuse has to be introduced with a more careful wave schedule, narrower live accumulator set, or additional CTA decomposition.

## Recommended Experiment Methodology

The next phase should be run like a kernel engineering study, not like a single benchmark chase. Each experiment should isolate one design variable, record both runtime and compiler/resource data, and compare against the right baseline.

### 1. Keep a Stable Baseline Ladder

Every new kernel variant should be compared against a fixed ladder of baselines:

| Baseline | Purpose | What It Answers |
| --- | --- | --- |
| Sequential unfused | End-to-end product relevance | Is fusion actually faster than the current safe path? |
| Default HK/Iris AG+MM | Native backend regression guard | Did the new variant improve the native kernel or just move work around? |
| Theoretical overlap ceiling | Opportunity sizing | How much speedup is physically available if communication is fully hidden? |
| PyTorch reference | Correctness | Did the kernel preserve numerical behavior? |

For AG+MM N-reuse experiments, the primary engineering baseline should be the default HK/Iris AG+MM kernel on the same shape. The product baseline remains sequential unfused. This distinction matters because a new native variant may improve the native backend while still not yet beating the production baseline.

### 2. Change One Kernel Variable at a Time

The highest-signal AG+MM variables are:

- N reuse factor: `1`, `2`, possibly `4` only after spills are eliminated.
- Accumulator live range: all N blocks live at once vs one N block at a time.
- Producer/consumer wave split: number of producer waves, consumer waves, and wave ownership of output fragments.
- LDS layout: double-buffered A/B panels, bank conflict behavior, and whether A is kept resident across multiple B tiles.
- CTA tile shape: `128x256`, `128x512`, and any narrower accumulator-preserving decomposition.
- Workgroup swizzle: M/N ordering, XCD/chiplet-aware mapping, and remote-rank distribution.

Avoid combining several of these in one change. If a result improves or regresses, the next action should be obvious.

### 3. Use a Shape Ladder Instead of One Shape

A single shape can lie. The recommended shape ladder is:

| Tier | Shape Pattern | Purpose |
| --- | --- | --- |
| Correctness smoke | Small tile-aligned shapes such as `(256,64,256)` and `(256,64,512)` on 2 ranks | Fast correctness and resource sanity |
| Register-pressure probe | Fixed M/K with increasing N, such as `N=256,512,1024` | Shows when accumulator footprint breaks occupancy or spills |
| K-loop steady state | Larger K, such as `K=1024,2048,4096` | Reduces launch overhead sensitivity and exposes pipeline quality |
| Production-like | Escher-like shapes such as `(1592,5120,13824)` and `(4680,5120,13824)` on 8 ranks | Measures relevance to real tensor-parallel layers |
| Stress | Large M/N/K with full world size | Tests scheduling, swizzle, heap use, and synchronization |

Each kernel variant should pass the correctness smoke before it is run on larger shapes. A variant that spills on the small shape should usually be rejected before spending time on full sweeps.

### 4. Gate on Correctness First

For every candidate:

- Run a 2-rank correctness smoke against PyTorch.
- Test both local-owned and remote-owned A tiles.
- Use at least one `N=512` shape for N-reuse variants.
- Keep BF16 tolerance explicit, for example comparing fp32-cast outputs with a tolerance appropriate for BF16 accumulation and conversion.
- Barrier after native dispatches that write symmetric memory, so validation does not race remote stores.

Correctness failures should be fixed before performance profiling. Profiling an incorrect kernel is almost always wasted time.

### 5. Record Compiler Resource Usage for Every Variant

Runtime alone is not enough. The build output should be treated as part of the benchmark result.

For each kernel, record:

- VGPR count.
- SGPR count.
- Scratch bytes per lane.
- VGPR and SGPR spills.
- Occupancy in waves/SIMD.
- Dynamic shared memory requested.
- Launch bounds.

For this study, the key example was the N-reuse variant. It had the right memory-reuse idea, but the compiler reported `VGPRs Spill: 108`, which explained the slower runtime. That signal should be an automatic red flag. A recommended acceptance rule is:

```text
Reject AG+MM performance variants with VGPR spills unless they show a large measured speedup anyway.
Prefer variants with zero spills, stable occupancy, and no scratch traffic.
```

### 6. Separate Bring-Up Runs From Measurement Runs

Use different benchmark modes for different purposes:

```bash
# Correctness / smoke
BENCH06_USE_IRIS=1 AITER_KERNELS_BACKEND=hipkittens \
  torchrun --nproc_per_node=2 benchmarks/bench06_aiter_fused.py \
  --out /tmp/hk_smoke --shapes "256,64,512" --warmup 2 --iters 5

# More stable timing
BENCH06_USE_IRIS=1 AITER_KERNELS_BACKEND=hipkittens \
  torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py \
  --out results/hk_sweep_name \
  --shapes "8192,4096,4096" --warmup 20 --iters 100

# Opt-in N-reuse experiment
BENCH06_USE_IRIS=1 AITER_KERNELS_BACKEND=hipkittens HIPKITTENS_AG_N_REUSE=2 \
  torchrun --nproc_per_node=2 benchmarks/bench06_aiter_fused.py \
  --out /tmp/hk_reuse2 --shapes "256,64,512" --warmup 5 --iters 50
```

Smoke runs should be fast and disposable. Measurement runs should use enough warmup and iterations to reduce noise, especially for sub-millisecond kernels.

## How to Measure Performance

The most useful performance view is not just "fused time". It is a decomposition of where the time went and how much communication was actually hidden.

### Primary Metrics

For each row, collect:

| Metric | Definition | Why It Matters |
| --- | --- | --- |
| `t_fused_ms` | Time for fused kernel path | Direct end-to-end result |
| `t_unfused_ms` | Sequential collective + GEMM time | Product baseline |
| `speedup_vs_unfused` | `t_unfused_ms / t_fused_ms` | Main pass/fail signal |
| `t_gemm_ms` | GEMM-only time from unfused breakdown | Estimate of compute floor |
| `t_comm_ms` | Collective-only time from unfused breakdown | Communication cost to hide |
| `overlap_ceiling_ms` | `max(t_gemm_ms, t_comm_ms)` | Best possible fused time if perfectly overlapped |
| `ceiling_speedup` | `t_unfused_ms / overlap_ceiling_ms` | Maximum realistic speedup |
| `ect_fused_ms` | `t_fused_ms - t_gemm_ms` for AG+MM-style analysis | Effective communication time left exposed |
| `overlap_efficiency` | Fraction of unfused communication hidden | Shows if fusion is doing its job |
| `TFLOP/s` | GEMM work divided by fused time | Compute utilization proxy |
| `GB/s` | Collective wire bytes divided by communication or fused time | Fabric utilization proxy |

The overlap ceiling is critical. If the ceiling speedup is only `1.1x`, a heroic kernel rewrite will not produce a `2x` win on that shape. If the ceiling speedup is `1.8x` and the fused kernel is slower than unfused, the schedule is leaving real performance on the table.

### Effective Communication Time

For AG+MM, a practical ECT estimate is:

```text
ECT_fused = t_fused - t_gemm_only
ECT_unfused = t_unfused - t_gemm_only
overlap_efficiency = 1 - (ECT_fused / ECT_unfused)
```

Interpretation:

- `ECT_fused <= 0`: communication is fully hidden or measurement noise dominates.
- `0 < ECT_fused < ECT_unfused`: partial overlap.
- `ECT_fused >= ECT_unfused`: fusion failed to hide communication.
- `ECT_fused >> ECT_unfused`: fused path added overhead beyond communication.

Negative overlap efficiency is not just "bad overlap"; it usually means the fused kernel introduced extra work, spills, poor memory access, or synchronization overhead.

### Measurement Hygiene

Sub-millisecond distributed kernels are easy to mismeasure. Use these rules:

- Pin the backend explicitly with `AITER_KERNELS_BACKEND=hipkittens` when measuring HK kernels.
- Pin experimental variants with explicit environment variables such as `HIPKITTENS_AG_N_REUSE=2`.
- Use the same shape, rank count, warmup count, and iteration count for baseline and experiment.
- Run enough iterations that launch jitter is small compared with total measured time.
- Ignore first-run import/JIT/build overhead; only timed benchmark regions should be compared.
- Keep GPU clocks and thermals stable. Record `rocm-smi` state for longer sweeps.
- Avoid comparing a 2-rank smoke directly to an 8-rank production run.
- Re-run surprising wins or losses at least once.
- Preserve `fused.csv`, `overlap.csv`, build logs, and the exact git diff for each experiment.

### Profiling Sequence

When a variant regresses, debug in this order:

1. Check correctness. Incorrect output can make performance meaningless.
2. Check compiler resource usage. Spills and scratch traffic are often decisive.
3. Compare `t_fused_ms` with `t_gemm_ms`. If fused is far above GEMM-only, the kernel is not hiding communication or is adding extra work.
4. Inspect LDS use and accumulator live ranges. Register pressure often comes from too many output fragments live at once.
5. Inspect global/symmetric memory traffic. Repeated remote A loads can erase the benefit of fusion.
6. Only then profile wave scheduling, barriers, and XCD swizzle.

### Acceptance Criteria for the Next AG+MM Kernel

A candidate AG+MM persistent kernel should be considered promising if it meets these gates:

- Correct on 2-rank BF16 tile-aligned smoke tests.
- No VGPR spills.
- No scratch traffic in compiler resource output.
- Runtime improves over default HK/Iris AG+MM on at least one N-reuse shape.
- Runtime does not regress badly on `N=256` where reuse has little opportunity.
- ECT improves versus the default HK/Iris path.
- The result moves toward the overlap ceiling on at least one larger K-loop shape.

Promotion to default should require more:

- Geomean speedup over default HK/Iris across the shape ladder.
- No correctness failures across 2-rank and 8-rank sweeps.
- No pathological slowdown on production-like Escher shapes.
- Clear fallback behavior for unsupported shapes.

## Insights and Takeaways

### 1. Fusion is a kernel scheduling property, not just an API property

A fused Python function can still launch two kernels sequentially. That may reduce framework overhead, but it does not hide xGMI latency behind MFMA work. The performance win comes when communication and compute are interleaved at the kernel schedule level.

### 2. LDS lifetime defines the architecture

The decisive correction was recognizing that LDS cannot cross kernel launches. Any design that relies on separate kernels sharing an LDS staging panel is invalid.

This leaves two valid patterns:

- Single persistent kernel with shared LDS between producer and consumer waves.
- Split kernels with global/symmetric memory staging.

The first is the performance target. The second is a debugging and bring-up tool.

### 3. AG+MM wants A-panel reuse across N tiles

The main AG+MM bottleneck is repeated remote A traffic. If each N tile independently loads the same A panel, the kernel wastes xGMI bandwidth and exposes remote latency. The right approach is to stage an A panel once and reuse it across multiple N tiles.

The first `AG_N_REUSE=2` experiment implemented that idea, but kept too much live state. The next version should preserve the reuse while reducing register pressure.

### 4. Register pressure can erase communication savings

The reuse kernel reduced conceptual remote A traffic, but spilled 108 VGPRs. On CDNA4, spilling is expensive enough to dominate the saved communication for the tested shape.

Future reuse designs should consider:

- Computing one reused N block at a time while keeping A resident in LDS.
- Splitting consumers into subgroups with fewer simultaneous accumulators.
- Increasing producer/consumer specialization without increasing per-thread accumulator footprint.
- Autotuning reuse factor by `(M,K,N,world)` rather than hardcoding it.

### 5. MM+RS needs a different optimization path

The native MM+RS writer + reducer is correct, but it is not yet a high-performance overlap design. It writes partial tiles to symmetric scratch and then launches a reducer. That is valid, but it adds HBM traffic and launch overhead.

A stronger MM+RS design likely needs either:

- A persistent single-kernel reduction schedule for local groups of contributors.
- A carefully optimized remote write/atomic epilogue.
- A hierarchical reduction strategy that avoids full `Y_full` materialization without serializing the reducer.

### 6. Small positive smokes matter, but production shapes need different tuning

The best positive result was on a small 2-rank tile-aligned shape. The production-like 8-rank Escher shapes were much less forgiving. Larger shapes require:

- Better CTA mapping.
- Larger and more stable steady-state K loops.
- Reduced launch overhead sensitivity.
- Careful chiplet/XCD swizzling.
- Avoidance of any redundant work across ranks or N tiles.

## Recommended Next Steps

The next phase should not treat every interesting kernel idea as equal. Based on the results so far and current CDNA4 guidance, the roadmap should prioritize the innovations that directly attack the measured AG+MM bottleneck: remote-A reuse without register spills. Three ideas are directly applicable and high value, two are targeted but secondary, and the rest should stay deferred until the BF16 persistent kernel is stronger.

### Directly Applicable, High Impact

#### 1. K-Splitting / Sequential N-Tile Accumulation

The highest-value AG+MM problem remains the accumulator scheduling around N reuse. The original `reuse=2` variant proved that stage-once/reuse-many is architecturally valid, but it kept too many N-block accumulators live and spilled 108 VGPRs. The `2_spillfree` variant fixed the spills by splitting work across consumer groups, but it only matched the default kernel on the tiny smoke shape.

The next design should preserve A-panel residency while sequencing N-tile accumulation so only one accumulator set is live per consumer at a time. The important constraint is that A must remain resident at the K-panel level. A naive "finish all K for N0, then finish all K for N1" loses the A reuse because the A panels have to be streamed again. The useful schedule is:

```text
for each K panel:
  producer waves load A_panel once into LDS
  producer waves load B panels for a small N group
  consumer group 0 computes N tile 0 with one accumulator footprint
  consumer group 1 computes N tile 1 with one accumulator footprint
  or time-slice one consumer group across N tiles without keeping both accumulators live
  advance to next K panel
```

This is related in spirit to Stream-K and split-K work-centric scheduling: reduce idle work and avoid pathological register pressure by changing how work is partitioned. The immediate goal is not a full Stream-K implementation; it is a spill-free A-panel reuse schedule that wins on larger K/N shapes.

Acceptance gates:

- Zero VGPR spills.
- No scratch traffic.
- Faster than default HK/Iris on at least one larger K/N shape.
- ECT improvement versus default HK/Iris, not just raw runtime noise.
- No severe regression at `N=256`.

#### 2. CDNA4 LDS Read-With-Transpose / Operand Layout

CDNA4 added wider LDS capacity and bandwidth, wider global-to-LDS movement, and read-with-transpose LDS support. This is directly relevant because the AG+MM kernel's inner loop is a global/symmetric memory -> LDS -> registers -> MFMA pipeline. Any register shuffle or inefficient LDS lane mapping that exists only to satisfy MFMA operand layout is a candidate for removal.

The next implementation should add a single LDS-layout experiment:

```text
stage A in LDS in a producer-friendly layout
read A from LDS with transpose or MFMA-friendly lane mapping
feed MFMA with fewer register reshapes
compare against default HK row-layout loads
```

This should be measured first as a GEMM-time optimization, then as a fused AG+MM optimization. If it increases VGPR pressure, creates bank conflicts, or lowers occupancy, reject it. The expected signal is lower GEMM time or better TFLOP/s at the same ECT.

### LDS Transpose-Read Investigation

Follow-up investigation found that the local HipKittens CDNA4 path already exposes the relevant hardware instruction. `include/cdna4/common/macros.cuh` defines `macros::ds_read_b64_tr_b16`, and `include/cdna4/ops/warp/memory/tile/shared_to_register.cuh` emits `ds_read_b64_tr_b16` for column-layout register tile loads. The blocker is therefore not instruction availability.

The blocker is operand layout. The current AG+MM inner loop uses:

```text
A_slice = rt_bf<32,64,row_l,rt_16x32_s>
B_slice = rt_bf<32,64,row_l,rt_16x32_s>
mma_ABt(C, A_row, B_row, C)
```

The transpose-read path naturally loads column-layout register tiles. In HK's CDNA4 API, the matching primitive is the transposed-A family:

```text
ColA = rt_bf<64,32,col_l,rt_16x32_s>
ColB = rt_bf<64,32,col_l,rt_16x32_s>
C    = rt_fl<32,32,col_l,rt_16x16_s>
mma_AtB(C, ColA, ColB, C)
```

`benchmarks/aiter_kernels/hipkittens_native/gemm_layout_probe.cuh` now records this contract as a non-runtime probe header. It is deliberately not wired into the extension yet; the next implementation should turn that contract into a local GEMM-only kernel first.

A correct fused experiment needs a coordinated change across:

- LDS staging shape/layout for A.
- shared-to-register load layout.
- MFMA primitive selection (`mma_ABt` vs an `At*` form).
- output register tile layout and store path.

So the next LDS-transpose task is not "replace one `load(a0, ...)` call." It is a small operand-layout variant. Do not inline assembly or `reinterpret_cast` row-layout tiles as column-layout tiles. The safest implementation path is:

```text
1. Build a local GEMM-only HK layout probe from gemm_layout_probe.cuh.
2. Verify correctness against the current row-layout kernel.
3. Confirm compiler resources stay spill-free: VGPR, SGPR, spills, occupancy.
4. Verify MFMA throughput is at least as good as the row-layout variant.
5. Only then port the layout to the fused AG+MM kernel.
```

Porting the layout into AG+MM means changing all of these together:

```text
LDS staging:     column-compatible A/B tiles
register loads:  column-layout RT loads, causing HK to emit ds_read_b64_tr_b16
MFMA primitive:  mma_AtB or the correct At* variant for the final B layout
store path:      layout-correct row-major global/symmetric output
```

This preserves the main lesson from the study: use the CDNA4 hardware feature, but do not mix an unsafe register reinterpretation into the fused kernel without a correctness-first layout probe.

#### 3. FP8 / FP6 Block-Scale GEMM Path

The current backend is BF16-only. CDNA4 has much stronger low-precision matrix-core capability, including FP8 and FP8/FP6/FP4 style MFMA families with block exponent scaling. If the model tolerates lower precision, this is the largest absolute throughput opportunity.

This should be staged carefully:

1. Add a local HK/AITER FP8 block-scale GEMM baseline, without collectives.
2. Validate numerical tolerance against the target model or representative layers.
3. Add fused AG+MM FP8 after the GEMM baseline is credible.
4. Consider FP6 only after packing, scaling, and model tolerance are clear.

The acceptance order is:

```text
numerical tolerance -> local GEMM TFLOP/s -> fused AG+MM ECT -> end-to-end speedup
```

Do not mix FP8/FP6 work with the BF16 scheduling experiments. First prove the scheduling pattern in BF16, then port the winning structure to low precision.

### Moderately Applicable / Targeted

#### 4. Adaptive Reuse Factor

Hardcoding `HIPKITTENS_AG_N_REUSE=2` is useful for bring-up but not a production strategy. Reuse should be selected by shape and world size. This is a methodology and dispatch improvement more than a kernel innovation.

Start with a simple heuristic:

```text
if N < 512:
  use default
elif K is small:
  use default
elif reuse_spillfree wins in the autotune cache:
  use reuse_spillfree
else:
  use default
```

Then replace the heuristic with an autotune cache keyed by:

```text
(M, K, N, world, dtype, arch)
```

This helps avoid wasting time on shapes where reuse is structurally unlikely to win. It does not replace kernel work; it selects among measured kernels.

#### 5. 4-Wave Interleave and DPP

DPP and finer-grained wave interleave are more relevant to reductions and register-level shuffles than to the core AG+MM bottleneck. For AG+MM, the dominant issue is remote-A traffic and accumulator scheduling. For MM+RS, DPP may help the reducer by keeping partial reductions in registers and avoiding unnecessary LDS traffic.

Use this as an MM+RS-targeted experiment:

- Replace scalar reducer pieces with wave-level DPP reductions where the data layout fits.
- Compare against an HK tiled reducer.
- Measure reducer time, LDS traffic proxy, HBM traffic, and ECT.

For AG+MM, only revisit DPP if profiling shows register shuffle overhead after LDS layout work.

### Lower Priority / Deferred

#### StragglAR-Style Scheduling

StragglAR is an AllReduce algorithm that exploits rank arrival skew before a collective. That is not the same problem as in-kernel AG+MM producer/consumer scheduling. It may inspire future collective scheduling work, especially for MM+RS or full model execution, but it should not be in the near-term AG+MM kernel plan.

#### Stream-K Hybrid for Edge Quantization

Stream-K style scheduling is useful for wave quantization and load balance. It is not the immediate fix for the measured AG+MM problem unless larger production shapes show poor CTA quantization across CUs/XCDs. Treat it as a later scheduling layer after the basic persistent reuse kernel is faster.

### MM+RS Track

The current split writer/reducer remains a correctness baseline. The next performance pass should focus on replacing the scalar reducer with a real HK reducer and reducing scratch traffic.

There is one important architectural constraint: MM+RS reduction is across source ranks, not just within a wave or workgroup. DPP is wave-local, LDS is workgroup-local, and a normal HIP kernel has no global cross-rank barrier that lets one CTA safely wait for every other rank's GEMM partials and then reduce them in registers. Therefore, a "single persistent kernel with in-register reduction across source ranks" is not valid unless the algorithm changes one of the following:

- Every destination rank's kernel computes all source-rank partials itself by remotely reading the needed inputs, which duplicates GEMM/input traffic and changes the parallel decomposition.
- All ranks write into the same destination using remote atomics, so the network/L2 path performs the reduction instead of a later reducer kernel.
- A cooperative/grid-level synchronization primitive exists and is safe across all participating CTAs/ranks, which this backend does not currently have.
- The "single kernel" only reduces within local wave/CTA fragments before writing a rank partial; a cross-rank reducer or atomic accumulation is still required.

So the next valid MM+RS plan is not to collapse the entire cross-rank reduce into DPP. It is to reduce the overhead of the valid handoff mechanisms.

#### Implemented Reducer Experiments

A first MM+RS improvement pass added three selectable reducer modes behind `HIPKITTENS_MM_RS_REDUCER`:

- `default`: the original scalar reducer. One thread reduces one output element across `scratch[slot,row,col]`.
- `specialized`: world-size-specialized reducers for world sizes 2, 4, and 8. This removes the dynamic slot loop and branch for common tensor-parallel sizes.
- `vec4`: one thread reduces four contiguous output elements. This reduces per-element indexing and launch scheduling overhead while preserving the same cross-rank scratch contract.
- `auto`: currently selects `vec4`, because it was the only variant with a repeatable positive signal.

The native extension exposes these as `dispatch_mm_rs_reduce_specialized` and `dispatch_mm_rs_reduce_vec4`. The benchmark matrix now writes `results/.../hk_mmrs_ladder.csv` with one row per MM+RS reducer variant, and the correctness smoke covers all four modes against PyTorch `reduce_scatter_tensor`.

Compiler resource summary for the reducer experiments:

```text
default reducer:        6 VGPR, 16 SGPR, 0 spills, 8 waves/SIMD
specialized world=2:    8 VGPR, 14 SGPR, 0 spills, 8 waves/SIMD
specialized world=4:   14 VGPR, 14 SGPR, 0 spills, 8 waves/SIMD
specialized world=8:   26 VGPR, 16 SGPR, 0 spills, 8 waves/SIMD
vec4 world=2:          10 VGPR, 17 SGPR, 0 spills, 8 waves/SIMD
vec4 world=4:          16 VGPR, 17 SGPR, 0 spills, 8 waves/SIMD
vec4 world=8:          28 VGPR, 17 SGPR, 0 spills, 8 waves/SIMD
```

Measured compact ladder, world=2, `warmup=3`, `iters=10`:

```text
shape          default ms  specialized  vec4       best
256,64,256     0.191721    0.188001     0.184161   vec4, 1.041x vs default
256,64,512     0.176781    0.187121     0.182062   default
512,256,512    0.197262    0.190001     0.186161   vec4, 1.060x vs default
```

Repeated larger-shape run, world=2, `warmup=10`, `iters=50`, shape `512,256,512`:

```text
default:      0.177762 ms
specialized:  0.181741 ms
vec4:         0.172821 ms
auto:         0.183401 ms before remapping auto to vec4
```

The repeat shows `vec4` at about 1.029x over default for this shape. The world-specialized scalar reducer did not survive the higher-iteration repeat, so it should remain an experiment rather than a default. `auto` was remapped to `vec4` after this measurement.

Interpretation:

- The reducer micro-optimization is correct and cheap, but the win is only low single digits on the repeated shape.
- Negative overlap remains large; MM+RS is still around 0.3x of the unfused PyTorch baseline on these small 2-rank shapes.
- The simple reducer loop is not the main bottleneck. (At this point the suspect was the writer + remote scratch handoff; the later split-phase instrumentation corrected this — the two **host barriers**, not the writer or reducer, dominate. See "Split-Phase Instrumentation (decisive)" below.)
- A real MM+RS step-function improvement still needs either a better scratch handoff, destination-side tiling/swizzling, or a remote atomic epilogue that removes the reducer launch without invalid cross-rank DPP assumptions.

The split path pays three major HBM-facing transactions for each output element:

```text
1. writer:  remote partial write into destination scratch
2. reducer: scratch read-back across source-rank slots
3. reducer: final output write
```

That traffic, plus the cross-rank barrier and second launch, is the architectural gap. More reducer micro-variants are not expected to move ECT materially.

#### Destination-Rank Swizzle Experiment

The next architecture experiment added an opt-in destination scratch swizzle:

```text
HIPKITTENS_MM_RS_SWIZZLE=1
```

The implementation keeps the same scratch tensor shape, but permutes destination row tiles per `(dest_rank, source_slot)`:

```text
stored_row_tile = (logical_row_tile + ((dest_rank + source_slot) % min(NUM_XCDS, row_tiles))) % row_tiles
```

The reducer receives `dest_rank` and applies the same permutation when reading each source slot, so the logical output is unchanged. This is deliberately a low-risk swizzle: it does not change the cross-rank protocol, scratch allocation size, or reducer launch structure.

Validation:

```text
build:       PASS
correctness: PASS for swizzle=0/1 and reducers default/specialized/vec4/auto
resources:  writer remains 0 spills; reducers remain 0 spills and 8 waves/SIMD
```

The address math increases reducer SGPR pressure modestly versus the earlier reducer-only kernels, but does not introduce spills.

Measured compact ladder, world=2, `warmup=3`, `iters=10`:

```text
shape          default ms  vec4 ms   swizzle ms  vec4+swizzle ms  best
256,64,256     0.224642    0.223862  0.166761    0.225902         swizzle, 1.347x
256,64,512     0.210222    0.212102  0.236263    0.226502         default
512,256,512    0.221222    0.239102  0.208542    0.192222         vec4+swizzle, 1.151x
```

Repeated larger-shape run, world=2, `warmup=10`, `iters=50`, shape `512,256,512`:

```text
default:       0.215782 ms
vec4:          0.214242 ms  1.007x vs default
swizzle:       0.212002 ms  1.018x vs default
vec4+swizzle:  0.209922 ms  1.028x vs default
```

Interpretation:

- Destination-rank swizzle is correct and can improve MM+RS, but the signal is shape-sensitive.
- The repeat confirms a modest 2.8% win for `vec4+swizzle` on `512,256,512`.
- The `256,64,512` regression means swizzle should remain opt-in or autotuned, not a global default.
- The improvement is still much smaller than the gap to unfused PyTorch, so the main architectural issue remains the scratch handoff plus barrier plus reducer launch.
- Remote atomic epilogue remains the next architecture experiment. It was not implemented in this pass because the native HK/Iris path does not yet expose a safe BF16 symmetric-memory atomic-add epilogue; HK has low-level buffer atomic BF16 assembly helpers, but using them correctly would require a separate packed-output protocol and contention/correctness test.

#### Split-Phase Instrumentation (decisive)

Before doing more writer/reducer work, the four MM+RS phases were timed separately with `benchmarks/aiter_kernels/hk_mmrs_profile.py`. Each phase is synchronized so the numbers attribute cost rather than measure overlap. Kernel phases use CUDA events; barrier phases use wall clock around a synchronized Iris barrier. world=2, `warmup=10`, `iters=50`, per-phase median reduced with MAX across ranks (ms):

```text
reducer=default swizzle=0
shape          writer   barrier1  reducer  barrier2  total
256,64,256     0.0282   0.0475    0.0116   0.0477    0.1349
256,64,512     0.0275   0.0468    0.0109   0.0469    0.1320
512,256,512    0.0293   0.0438    0.0088   0.0440    0.1259

reducer=vec4 swizzle=1
512,256,512    0.0286   0.0454    0.0082   0.0454    0.1277
```

This overturns the earlier assumption that the writer or reducer dominates:

- The **two Iris host barriers are ~70% of MM+RS time** (`barrier1 + barrier2` ≈ 0.088–0.091 ms of ~0.126–0.128 ms).
- The writer kernel is cheap (~0.029 ms) and is *not* the bottleneck, so vectorizing its remote stores has a low ceiling.
- The reducer is nearly free (~0.009 ms), confirming reducer micro-variants (vec4/specialized) were always going to be marginal.
- The phase split is invariant to reducer choice and swizzle, so those knobs cannot address the real cost.

The architectural cost is cross-rank synchronization plus the second kernel launch, not arithmetic or local HBM bandwidth at these shapes.

Specific MM+RS experiments:

- Add destination-rank swizzling so remote scratch writes distribute more evenly across XCDs and memory partitions. This is the next low-effort experiment because it directly targets the destination scratch write path without changing correctness semantics.
- Explore remote atomic accumulation where fabric atomics are competitive. This is the only true one-kernel way to remove the reducer kernel without duplicating computation, but it must be benchmarked because atomics are still read-modify-write operations and can lose under contention.
- Replace the simple scalar reducer with an HK tiled destination reducer only after swizzle/atomic experiments if scratch read-back remains the best valid handoff mechanism.
- Add a local pre-reduction inside the writer only where multiple wave/CTA fragments contribute to the same rank-local tile. DPP can help there, but it does not replace cross-rank reduction.
- Measure writer time, reducer time, scratch bytes, ECT, and HBM traffic.

Remote atomics should be tested, not assumed. They may win for small tiles or low contention; writer plus tiled local reduce may win for larger tiles.

Priority order for MM+RS (revised after split-phase instrumentation, then updated with execution outcomes):

The instrumentation shows the two host barriers, not the kernels, are the dominant cost. The next-level experiments therefore targeted synchronization and launch overhead. Outcomes are annotated below; full detail follows in the subsequent subsections.

1. **[DONE — shipped win]** Double-buffer the scratch to eliminate the post-reducer barrier (`barrier2`, ~35% of total). Alternate scratch buffers per iteration so the next writer cannot overwrite a buffer the previous reducer still reads, removing the write-after-read hazard the post-reducer barrier guarded. Measured **1.56x** (1.62x with vec4+swizzle), correct.
2. **[ATTEMPTED — blocked at framework level]** Replace the post-writer barrier (`barrier1`, ~35% of total) with device-side flags/barrier so the reducer no longer waits on a host barrier. Implemented as per-rank generation flags, then re-tested with Iris's own `device_barrier()` and with write-through stores. All are intermittently incorrect because Iris exposes no device-side remote-write completion primitive usable from the native HK C++ writer. Perf reaches the floor (~3.4x, first positive ECT) but correctness fails. See the Iris API audit and RFC / Blocker Note below.
3. **[BLOCKED — depends on #2]** Persistent flag-driven reducer (one launch). Cannot be correct until the #2 completion-semantics gap is resolved.
4. Remote atomic epilogue as an alternative to scratch + reducer — also depends on a verified remote-write completion/atomic-visibility guarantee, so it is gated behind the same framework gap.
5. Destination-rank scratch swizzle remains an opt-in, shape-sensitive tuning knob, not a primary lever.
6. Writer remote-store vectorization, chunked/tiled reduction, and FP8/FP6 are low-ceiling at these shapes and should follow the synchronization work, if at all.

#### Double-Buffered Scratch — Drop the Post-Reducer Barrier (executed, biggest MM+RS win)

Priority item #1 above was implemented behind `HIPKITTENS_MM_RS_DOUBLE_BUFFER=1`. This is a host-side scheduling change only; the writer and reducer kernels are unchanged.

- Allocate a 2-deep symmetric scratch buffer `(2, world, M_shard, N)` and alternate the active buffer per call via a counter on the Iris context (deterministic and identical across ranks, since all ranks run in lockstep).
- The next call's writer targets the *other* buffer, so it cannot clobber the buffer the previous reducer is still reading. This removes the write-after-read hazard that `barrier2` guarded, so the post-reducer barrier is dropped entirely.
- Correctness for two-call-back buffer reuse is preserved by the *following* call's post-writer barrier (`barrier1`): on rank `d`, that barrier cannot pass until every rank finished the intervening writer, which on rank `d`'s stream is ordered after the prior reducer. The final call needs no trailing barrier because no later writer reuses its buffer, and the local `Y_shard` is consumer-ordered on-stream.
- The post-writer barrier (`barrier1`) is retained; only `barrier2` is removed in this pass.

Correctness smoke now sweeps `double_buffer=0/1` and calls `rs_fn` three times per config when double-buffering to stress the two-call-back reuse ordering:

```text
HK/Iris correctness smoke OK (... swizzle=0/1, double_buffer=0/1)
```

Measured, world=2, `warmup=25`, `iters=200`, shape `512,256,512` (MM+RS time from `fused.csv`):

```text
variant                       mm_rs ms   vs default   speedup vs torch
default                       0.16374    1.000x       0.82x
double_buffer                 0.10508    1.558x        1.28x
vec4_swizzle                  0.15766    1.039x       0.85x
vec4_swizzle_double_buffer    0.10120    1.618x        1.33x
```

Interpretation:

- Dropping a single host barrier reduces MM+RS time by ~36% (1.56x), and combined with the opt-in vec4+swizzle reducer reaches ~38% (1.62x). This is the largest MM+RS improvement to date and it crosses parity with unfused PyTorch (speedup goes from 0.82x to 1.28–1.33x).
- This directly validates the split-phase instrumentation: removing `barrier2` recovered almost exactly its measured ~35% share, confirming the barriers — not the reducer read-back or writer stores — were the dominant cost.
- The reducer/swizzle micro-knobs remain marginal on top of the barrier win (~1.04x), consistent with the earlier finding that they cannot move the architectural cost.

Remaining MM+RS headroom is the second barrier (`barrier1`, the other ~35%). Priority item #2 — replacing it with device-side per-tile signaling — was attempted next and reaches the predicted perf floor, but it is **blocked by a framework-level gap** rather than landing as a win: Iris exposes no device-side remote-write completion primitive usable from the native HK C++ writer, so every device-side handoff variant is intermittently incorrect. The double-buffer result above therefore stands as the production MM+RS path. See the next three subsections (ablation floor, device-flag handoff, Iris API audit) and the RFC / Blocker Note for the full investigation and recommended framework-level next step.

#### Ablation: `barrier1` Removal Floor (methodology step before building flags)

Before implementing the device-flag pipeline (a correctness-sensitive cross-GPU change), a throwaway ablation bounds the achievable win. `HIPKITTENS_MM_RS_NO_BARRIER1=1` additionally skips the post-writer barrier. This is **racy and numerically incorrect** (the reducer may read scratch before remote writes land) and exists only to measure the timing floor. world=2, `warmup=25`, `iters=200`, shape `512,256,512`:

```text
config                          host barriers   mm_rs ms   vs db    vs default   vs torch
double_buffer                   barrier1 only    0.10320   1.00x    1.59x        1.30x
double_buffer + no_barrier1     none             0.04138   2.49x    3.96x        3.24x
```

The floor (0.04138 ms) shows `barrier1` removal is worth up to ~2.49x more — larger than the ≤1.5x the additive table predicted, because the host barrier wall-clock dwarfs the kernel work at these shapes. This empirically justifies building the device-flag pipeline. A correct flag version will land between the floor and current `db` (polling and imperfect overlap cost something), but even half the gap reaches ~0.07 ms (~1.45x over current).

Important implementation constraint surfaced while grounding the design: the device-side `iris_context_view` exposes only a plain `store` (a translated-pointer dereference) and `translate`/`get_heap_base`. It exposes **no fence, atomic, or acquire/release primitive**. Today the host barrier is what guarantees cross-GPU completion and visibility of the remote scratch writes. A correct device-flag protocol must therefore hand-roll the ordering: a `__threadfence_system()` after the tile payload stores, then a release-style flag store through the translated pointer, with an acquire-style poll on the reducer. Getting fabric-scope memory ordering wrong yields silent corruption that small smokes can miss, so this step needs a dedicated multi-shape stress and contention test, not just the existing smoke.

#### Device-Flag Handoff (executed): floor-level perf, but blocked on a missing Iris primitive

The flag handoff was implemented behind `HIPKITTENS_MM_RS_FLAGS=1` (implies double-buffered scratch, removes both host barriers): a per-rank symmetric `int32 flags[world]` with a monotonically increasing generation as the flag value (no reset, no stale-flag race), `__hip_atomic_*` release/acquire at `__HIP_MEMORY_SCOPE_SYSTEM`, and a `__threadfence_system()` push of the payload. Signaling was tried three ways (separate signal kernel; per-thread reducer acquire; in-writer grid-completion where the last workgroup per destination releases the flag), plus a coherent system-scope scratch read on the reducer.

Performance (world=2, `512,256,512`, `iters=200`):

```text
config        mm_rs ms   vs default   vs torch   ECT eff
default       0.16374    1.00x        0.82x      -302.7%
double_buffer 0.10320    1.59x        1.30x      -126.6%
flags         0.04814    3.40x        2.79x      +6.7%   (first positive ECT)
no_barrier1   0.04138    3.96x        3.24x      (racy floor)
```

The flag handoff reaches the racy floor (0.048 vs 0.041 ms) and is the first MM+RS configuration with **positive ECT (+6.7%)**. But it is **numerically incorrect**: a dedicated stress (3 shapes x 20 repeats) consistently fails with ~24-30 of 131072 elements wrong, scattered, each missing one source rank's contribution (relative diff up to `inf`). All four fix attempts produced the identical signature, which is itself the diagnosis: the failure is not in any mechanism that was changed, but in the one guarantee they all depend on and none can provide.

Root cause: `__threadfence_system()` orders the source rank's remote stores *at the source*, but does not guarantee the xGMI writes have *landed* in the destination's HBM before the fence returns. So the reducer can observe the release flag before a few payload writes arrive. The Iris **host** barrier works because it drains in-flight remote writes; Iris's **device-side** view exposes no equivalent quiet/drain primitive (only `store`/`translate`). This empirically confirms the constraint noted above: a correct one-launch-ish device handoff needs an Iris device-side remote-write-completion primitive (a `quiet`/drain or a system-scope fence with landing semantics). The flag path and its stress are kept opt-in (`HIPKITTENS_MM_RS_FLAGS`, `HIPKITTENS_MM_RS_FLAGS_STRESS`) and clearly marked experimental/incorrect so the default smoke stays green and the work documents both the achievable ceiling and the exact blocker.

Net for MM+RS: the shipped, correct win remains **double-buffering at 1.56x (1.62x with vec4+swizzle)**. The barrier1 removal is worth a further ~2.5x in principle and is reachable in perf, but is gated on Iris exposing a device-side remote-completion primitive; pursue that (or a verified gfx950 landing fence) before treating the flag handoff as correct.

#### Iris API audit + framework-level blocker (decisive, do not chase a guessed fence)

Rather than prototype a speculative gfx950 "landing fence", the Iris source was audited for any remote-write quiescence primitive. Findings:

- Iris's collectives (`ops/matmul_reduce_scatter.py`, `ops/matmul_all_reduce.py`, `ccl/triton/all_reduce.py`) use the same release/acquire flag pattern (`tl.atomic_xchg(..., sem="release", scope="sys")` / acquire spin), with the payload stored using a **write-through cache modifier** (`tl.store(..., cache_modifier=".wt")`) and a `tl.debug_barrier()` before the release. The release is done by the *same* program that wrote the tile (per-tile lock), so it genuinely covers that tile's payload.
- Iris exposes a **device-side barrier**: `shmem.device_barrier(group)` → `distributed_device_barrier(...)` → `_device_barrier_kernel` (a single-CTA Triton kernel using `atomic_add` release / `atomic_cas` acquire at `scope="sys"`, CUDA-graph capturable, no host `torch.cuda.synchronize()`).
- There is **no** documented device-side remote-write `quiet`/`drain`/`flush`/`complete` primitive. The only completion guarantee for the HK native writer's plain symmetric stores is the host `barrier()`'s `torch.cuda.synchronize()` (a full device idle that flushes caches to memory).

Tested the framework primitive directly (`HIPKITTENS_MM_RS_DEVICE_BARRIER=1`, opt-in), swapping the host `barrier()` for Iris's `device_barrier()` with the proven reducer:

- It reproduces the **same** intermittent race (~19-38 / 131072 elements) as the hand-rolled flag handoff.
- Adding write-through (nontemporal) stores to the HK writer (`store_wt`, gated by a writer `write_through` flag) did **not** fix it either.

Conclusion: the device-side path is **blocked at the framework/integration level**, not at the kernel level. Even Iris's own validated device barrier is insufficient for the HK native C++ writer's symmetric-store path; the working Iris collectives are end-to-end Triton with their own store/lock/reduce_scatter memory model. Making MM+RS barrier-free correctly requires either (a) Iris exposing a device-side remote-write-completion primitive usable from the C++/HK writer, or (b) porting the MM+RS writer+reducer into Iris's Triton store/lock model. Both are framework/runtime work, not kernel tuning. The experimental paths (`HIPKITTENS_MM_RS_FLAGS`, `HIPKITTENS_MM_RS_DEVICE_BARRIER`, and write-through) are kept opt-in and clearly marked incorrect; their stresses are gated behind `HIPKITTENS_MM_RS_FLAGS_STRESS` so the default smoke stays green. **Double-buffering (1.56-1.62x) remains the correct, shipped MM+RS win.**

### RFC / Blocker Note: MM+RS Barrier-Free

**Status.** The MM+RS barrier-free path is not correct in the native HK C++ writer model. The shipped `double_buffer` path remains the correct production result, and all device-flag / device-barrier / write-through variants remain experimental and incorrect under stress.

**What was verified.**

- Iris collectives use a release/acquire `scope="sys"` pattern with write-through payload stores, debug barriers, and per-tile locks in an end-to-end Triton model.
- Iris does expose a device-side barrier via `shmem.device_barrier()`.
- Iris does *not* expose a documented device-side remote-write completion primitive for the native HK symmetric-store writer path.
- Swapping the host barrier for Iris's device barrier reproduces the same race (~19-38 / 131072 elements).
- Adding write-through (nontemporal) stores does not fix the race.

**API audit (conclusive).** A full pass over the installed Iris package confirms there is no `quiet`/`drain`/`flush`/`quiesce`/`complete`/`fence` primitive. The public device-side surface (`iris.__all__`) is `load`, `store`, `copy`, `get`, `put`, and the atomics (`atomic_add/cas/xchg/xor/or/and/min/max`), plus `barrier()` and `device_barrier()` as context methods. The *only* remote-write completion lever Iris has is the `cache_modifier` argument on its Triton `store`/`load`/`copy`:

- `.wt` = write-through, documented as "bypasses L1 and L2 (coherent cache bypass), may hit in LLC with LRU" (`iris/mem/{triton,gluon}/context.py`).
- Completion is therefore *emergent* in the Triton path: `.wt` store + `tl.debug_barrier()` + release/acquire `scope="sys"` atomics. There is no standalone completion call.

This lever is a Triton store attribute and is **not reachable from the native HK C++ writer**, whose nearest equivalent (`__builtin_nontemporal_store`, an L2/streaming bypass) is not the same coherent L1+L2 write-through and did not fix the race. This is the precise, conclusive form of the gap.

**Conclusion.** The blocker is framework/runtime-level, not kernel-level. The native HK writer needs a real remote-write completion semantic, or the MM+RS logic must be ported into Iris's Triton store/lock model.

**Implication.** Further kernel tuning will not make the barrier-free path correct. The next milestone is one of:

1. Expose a device-side remote-write completion primitive usable from native HK/C++.
2. Port the MM+RS writer/reducer into Iris's Triton memory model.

The Triton port may be faster for a proof-of-concept; exposing completion semantics to native HK is the cleaner architectural fix for the current stack.

**Current baseline (frozen).**

- `double_buffer`: correct, shipped, best validated MM+RS result (1.56x; 1.62x with vec4+swizzle).
- `vec4 + swizzle`: modest incremental, opt-in, shape-sensitive improvement.
- `HIPKITTENS_MM_RS_FLAGS`, `HIPKITTENS_MM_RS_DEVICE_BARRIER`, and write-through variants: opt-in only, known incorrect under stress (gated behind `HIPKITTENS_MM_RS_FLAGS_STRESS`).

**Recommended ticket title.** Expose remote-write completion semantics for native HK MM+RS writers, or port MM+RS into Iris Triton memory model.

**Suggested acceptance criteria.**

- A native HK writer can signal completion of remote symmetric stores without a host barrier.
- Stress tests over repeated shapes and repeats pass with no missing contributions.
- Performance matches or exceeds the current `double_buffer` baseline.
- Correctness remains green under the default smoke.

**One-line summary.** Barrier-free MM+RS is blocked by missing runtime semantics, not by more kernel optimization.

### Documentation and Upstreaming

The current implementation is useful as an AITER-managed backend scaffold. Before upstreaming, it needs:

- Stable correctness tests across more shapes.
- Clear runtime fallback when shape constraints are not met.
- Autotuned kernel variants by architecture and shape.
- Documentation that distinguishes valid LDS-sharing designs from invalid split-kernel designs.

## Conclusion

The study changed direction from "make fused-looking collectives faster" to a more precise kernel design rule:

> If the optimization depends on LDS reuse between communication and compute, communication and compute must be in the same kernel launch.

That rule makes the single persistent HipKittens/Iris AG+MM kernel the right next step for MI355X. The current native backend proves the plumbing, correctness, symmetric-memory integration, and producer/consumer structure. The first N-reuse experiment shows the next bottleneck: register pressure. The path forward is not to abandon reuse, but to implement it with a schedule that keeps A resident in LDS while limiting live accumulators.

For AG+MM and MM+RS, the reducer/swizzle experiments plus split-phase instrumentation pinned the cost (the two Iris host barriers, ~70% of MM+RS time), and double-buffering recovered the post-reducer barrier — a real win **at the 2-rank smoke shapes** (1.56x; ~1.62x with vec4+swizzle). But the **production-accurate TP sharding regression (see the section near the top of this document) supersedes those numbers**: at the real Wan2.2 / Odyssey shapes at TP=8, the fused kernels run at **~0.35–0.45x of unfused PyTorch**, the best MM+RS combo beats the default kernel by only ~1.02–1.03x, and the theoretical overlap ceiling collapses to ~1.1–1.6x across AG+MM, MM+RS, **and MM+AR** (the dominant production pattern). The earlier optimism was an artifact of toy shapes and a replicated-weight model that overstated per-rank GEMM by `tp×`.

The decisive conclusion is therefore a **strategic pivot, not more kernel tuning**: there is no kernel-level path to parity at production shapes for any of the three patterns, while unfused PyTorch (hipBLASLt + RCCL) is strong. The shipped `double_buffer` MM+RS is frozen as a correct research baseline. The remaining leverage is (1) confirming the customer's true pattern/layout (likely MM+AR, with sequence parallelism toggling AG+MM/MM+RS on), and (2) the two pinned framework blockers — the missing device-side remote-write completion primitive for native HK MM+RS (see the RFC / Blocker Note below), and the upstream Iris `matmul_all_reduce` fused kernel that **does not compile** on this MI355X/ROCm/Triton stack — plus RCCL/collective and GEMM tuning. Those are framework/runtime work items, not kernel micro-optimizations.
