---
aliases: [kernels, Triton vs Mojo, custom kernels]
tags: [kernel, triton, mojo, MI355X, MFMA, wave64]
---
# Advanced Custom Kernels (Triton & Mojo)

While `microbenchmarks` recently integrated `aiter_kernels` for fused operations, it currently relies heavily on the standard PyTorch and Triton stacks. To push hardware to its absolute limit, we must look to the aggressive custom kernels.

This document clarifies two different mental models of the same hardware: **Triton abstracts around tiles of data**, while **Mojo/MAX stays closer to threads executing instructions**. That difference drives how you document memory layout, compute, and optimization.

**See also**: [[ROOFLINE]] (portable timing & memory formulas) · [[AITER_FUSED_KERNELS]] (fused comm+compute kernels that apply these models) · [[WAN2.2]] (Wan2.2 workload integration)

---

## 1. The Two Mental Models

### Triton: Block / Tile-Centric Model
*Key intuition: "I own a chunk of memory; I operate on it efficiently."*

**Memory Layout**
- Think in contiguous tiles of size `BLOCK_SIZE`.
- Each program instance (not a thread) operates on a vector slice: `offsets = pid * BLOCK_SIZE + [0...BLOCK_SIZE-1]`.
- Memory accesses are implicitly vectorized and coalesced if data is contiguous, `BLOCK_SIZE` aligns with cache lines (e.g., 128B → 32 FP32 elements), and masking is used to handle tail elements without branching.

**Compute Pattern**
- **SPMD over blocks**, not threads.
- Each "program" executes SIMD-style ops over a vector: `x = load(x_ptr + offsets)`.
- No explicit thread indexing—vector lanes are completely abstracted by the compiler.

**Performance Optimizations**
- `BLOCK_SIZE` tuning manages the occupancy vs register pressure tradeoff.
- Autotuning over `BLOCK_SIZE`, `num_warps`, and `num_stages` (pipelining).
- Software pipelining via `num_stages` to overlap: `load → compute → store`.
- Vectorized memory ops yield fewer instructions and better bandwidth utilization.

**What to Document Clearly**
- Mapping: `program_id` → data tile.
- Expected alignment and stride.
- `constexpr` parameters (`BLOCK_SIZE`, `num_warps`).
- Whether loads are coalesced / vector width.

### Mojo (MAX): Thread / Block-Centric Model
*Key intuition: "Each thread owns an element; performance comes from organizing threads."*

**Memory Layout**
- Flat 1D indexing: `idx = block_idx.x * block_dim.x + thread_idx.x`.
- Each thread handles one element (scalar view).
- Coalescing depends on adjacent threads accessing adjacent addresses; you must explicitly reason about warp-level access patterns.

**Compute Pattern**
- Classic CUDA-style **SIMT** (Single Instruction, Multiple Threads).
- One thread → one element, with warps executing in lockstep.
- Control flow (e.g., `if idx < n`) introduces divergence risk.

**Performance Optimizations**
- Tune `block_dim` (e.g., 128, 256, 512 threads) and `grid_dim` for full coverage.
- Ensure coalesced access by mapping `thread_idx.x` linearly to memory.
- Use explicit shared memory for reuse-heavy kernels.
- Minimize divergence from boundary checks.
- Occupancy tuning requires manual balancing of registers vs threads per block.

**What to Document Clearly**
- Grid/block mapping to data.
- Thread-to-element mapping.
- Memory access pattern (coalesced or strided).
- Boundary handling and divergence implications.

---

## 2. Case Study 1: GEMM (Matrix Multiplication)

GEMM is where the abstraction gap becomes highly apparent.

### Triton GEMM (Tile-first design)
**Memory Layout**: Operates on 2D tiles ($A_{tile} = [BLOCK_M, BLOCK_K]$). Access is block-contiguous and naturally coalesced.  
**Compute Pattern**: Loops over the K dimension: `C_tile += A_tile @ B_tile`. This is a vectorized FMA across the tile with implicit warp-level execution.  
**Optimizations**: Tuning block sizes, software pipelining (prefetching next tiles while computing the current), and maximizing SRAM (register/LDS) reuse across the K loop. High arithmetic intensity leads to near-roofline performance if tuned well.  
**Key Insight**: You explicitly design data reuse via tiling; compute follows memory.

### Mojo/MAX GEMM (Threadblock + shared memory)
**Memory Layout**: A threadblock computes a tile of $C$. Threads collaboratively load tiles into shared memory, and each thread computes one or more specific elements of $C$.  
**Compute Pattern**: Load $A_{tile}, B_{tile}$ into shared memory $\rightarrow$ synchronize threads $\rightarrow$ perform partial dot products $\rightarrow$ repeat over K tiles.  
**Optimizations**: Shared memory tiling is critical. Uses warp-level MMA instructions if the backend supports it. Carefully maps threads (e.g., each thread computes 4x4 outputs) to avoid shared memory bank conflicts and implements double buffering in shared memory for overlap.  
**Key Insight**: You explicitly orchestrate threads + shared memory to mimic hardware pipelines.

---

## 3. Case Study 2: Attention (Flash-style)

Attention is where Triton's block-centric abstractions truly shine.

### Triton Attention (FlashAttention-style)
**Memory Layout**: Operates on Query tiles ($Q_{tile}: [BLOCK_M, D]$), while Keys and Values are streamed over the sequence length ($N$).  
**Compute Pattern**: For each $Q_{tile}$, load it once, then stream over K/V tiles. Compute $QK^T$, apply scaling/masking, update the running softmax (online normalization), and accumulate the output. Avoids $O(N^2)$ memory by never materializing the full attention matrix.  
**Optimizations**: Keeps $Q_{tile}$ hot in SRAM (registers). Fuses `matmul + softmax + matmul(V)` into a single kernel. Autotunes tile sizes for specific sequence lengths and head dimensions.  
**Key Insight**: Triton expresses streaming dataflow + fusion naturally. *"Load a tile → keep it hot → stream the rest → fuse everything."*

### Mojo/MAX Attention
**Memory Layout**: Requires explicit shared memory staging of Q/K/V tiles. Threads cooperate on partial dot products.  
**Compute Pattern**: Similar K/V loop, but requires explicit thread synchronization barriers and explicit softmax reduction across threads. Often split into multiple kernels unless heavily and manually optimized.  
**Optimizations**: Requires warp-level reductions for softmax and careful synchronization (barrier costs matter). Kernel fusion is significantly harder because it requires manual coordination across threadblocks.  
**Key Insight**: You manually reconstruct FlashAttention behavior with thread primitives. *"Assign threads → coordinate them → stage memory → compute step-by-step."*

---

## 4. The MI355X Hardware Lens

To understand why these frameworks perform differently, you must map them to MI355X hardware realities:
- **Wavefront size**: 64 (not 32 like NVIDIA warps).
- **Large LDS**: High bandwidth, banked shared memory equivalent.
- **Strong L2 + HBM bandwidth**: Designed for streaming and reuse.
- **Matrix Cores**: MFMA-style instructions.
- **High Concurrency**: Occupancy vs register pressure tradeoffs matter significantly more across CUs than on smaller GPUs.

### Triton on MI355X
- **Memory + Tiling**: Triton tiles map cleanly to LDS (for staging) and Registers (for accumulation). `BLOCK` sizes should be multiples of 64 to align with waves. Coalescing is optimal when programs issue 128B+ contiguous loads to match HBM burst sizes.
- **Practical Tuning**: `BLOCK_M`/`BLOCK_N` should be multiples of 64. `BLOCK_K` must fit in LDS to avoid spilling. Set `num_warps` to 4–8 to align with MI3xx CU scheduling, and `num_stages` to 2–4 to overlap HBM latency.
- **GEMM**: Highly efficient if tiles maximize LDS reuse and the K-loop is long enough to amortize loads. Large tiles (128x128) and fused epilogues work best.
- **Attention**: Keeps Q tiles resident in large LDS and streams K/V. Tuning `BLOCK_M` balances register pressure vs occupancy.

### Mojo/MAX on MI355X
- **Thread/Block Mapping**: You must explicitly respect `wave64` behavior. `block_dim.x` must ideally be a multiple of 64. Thread layout explicitly determines MFMA utilization and LDS bank conflicts (MI3xx has 32 banks, making stride critical).
- **GEMM**: Threadblocks typically compute 128x128 tiles, with each wave computing a 64x64 sub-tile. Critical optimizations include careful LDS layout to avoid bank conflicts, register tiling, and double buffering.
- **Attention**: Much harder to get right than Triton. Requires explicit wave-level reductions for softmax (`max`, `sum(exp)`) and careful minimization of synchronization barriers inside the K-loop.

**Key Performance Pitfalls (MI355X-specific)**:
- Using non–wave-aligned sizes (e.g., 96 threads) causes severe underutilization.
- Naive layouts in Mojo cause LDS bank conflicts.
- Overly large tiles cause register spilling, which kills occupancy.
- Too many stages without enough work causes scheduler thrash.

---

## 5. MI355X Roofline Model (Applied)

The portable roofline formulas (per-op `t_flops`, `t_mem`, `t_bottleneck`) and memory budgeting equations (KV cache sizing, TP capacity) are documented in [[ROOFLINE]]. The canonical GPU spec registry (rated peak TFLOP/s, HBM BW, VRAM) lives in [[../configs/report_config.json|report_config.json]].

Here we apply the roofline model to the GEMM and Attention case studies above:

### GEMM on MI355X
For $C = A \times B$ with large $M, N, K$, the AI is roughly $\frac{2K}{bytes\_per\_element}$.
For BF16 (2 bytes), if $K=4096$: $AI \approx 4096$ — way above the ridge point ($\approx 50–100$ FLOPs/byte).
✅ **Compute-bound.** Performance limited by MFMA utilization and register/LDS reuse. Triton maximizes tile reuse automatically. Mojo can outperform *only* if MFMA scheduling is explicitly hand-tuned.

### Attention on MI355X
1. **$QK^T$ matmul**: High AI → compute-bound.
2. **Softmax + $V$ accumulation**:
   - **Without fusion**: Writing $QK^T$ to HBM and reloading → AI drops below ridge → **Memory-bound**.
   - **With Flash-style (Triton)**: No materialization → AI increases significantly → **Compute-bound**.

For sequences 4K–32K, naive attention leaves MI355X severely underutilized. Triton's fused attention increases AI enough to approach MFMA limits. Mojo can achieve this but requires manual streaming; if poorly tuned, it leaves 2–5× performance on the table.

---

## 6. Architectural Takeaways & The Path Forward

**The Quick Heuristics for MI355X**
- **GEMM**: Already compute-bound. The goal is optimizing MFMA usage.
- **Attention**: Starts memory-bound. Triton-style fusion makes it compute-efficient.
- **Left of Ridge Point**: Memory-bound (HBM limited).
- **Right of Ridge Point**: Compute-bound (MFMA limited). Triton pushes kernels to the right by reducing memory traffic.

**When deciding which framework to use:**
- **Fast time-to-performance / LLM inference**: Use Triton. It aligns well with wide vector units, high bandwidth systems, and compiler-driven scheduling.
- **Custom HPC / Legacy CUDA ports**: Use Mojo/MAX. It provides fine control over MFMA scheduling and explicit wave-level behavior for irregular or latency-sensitive kernels.

**Action Item**: Add Triton and Mojo kernels into `microbenchmarks/benchmarks/aiter_kernels/`. Benchmark them using these mental models to identify precisely where Mojo's fine-grained wave control beats Triton's compiler heuristics on AMD MI300X/MI355X hardware. See [[AITER_FUSED_KERNELS]] for the existing fused kernel API and dispatcher architecture.
