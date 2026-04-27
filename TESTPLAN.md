# TESTPLAN.md — `escher_14b_480p` on MI355X Benchmark Campaign

A detailed, technical, repeatable test plan to reproduce the PDF's methodology, tables, and result structure for the `escher_14b_480p` workload on AMD Instinct MI355X. The plan is organized as a single benchmark campaign that can be re-run after software changes to quantify regression or improvement.

---

## 1. Objective and Success Criteria

### 1.1 Primary Objective

Characterize GPU performance for the `escher_14b_480p` workload on MI355X across three axes:

1. **Compute** — BF16 matrix throughput.
2. **Memory bandwidth** — sustained HBM3E throughput.
3. **Memory capacity** — practical allocatable VRAM.

The plan must reproduce, on the reference platform:

- The PDF's BF16 dense compute result.
- The HBM bandwidth characterization.
- The roofline placement of `escher_14b_480p` ops.
- The MFU comparison between sum-of-ops, eager e2e, and `torch.compile` e2e.

### 1.2 Success Criteria

| ID | Criterion | Pass Condition |
|----|-----------|----------------|
| SC-1 | BF16 GEMM microbenchmarks approach MI355X dense peak | Largest square GEMM ≥ 90% of measured peak; ≥ 50% of rated peak (on the same spec basis used in the PDF) |
| SC-2 | HBM bandwidth converges near sustained ceiling | Streaming `copy_`/`add` plateaus within ±5% across 3 successive sizes near the ceiling |
| SC-3 | Roofline placement is correct | Large GEMMs and attention kernels sit in the compute-bound regime (right of the ridge); norms, GELU, small projections sit in the bandwidth-bound regime |
| SC-4 | Compiled e2e MFU ≥ eager e2e MFU ≥ sum-of-ops MFU | Same ordering as the PDF (≈77% / 93% / 99% on measured-chip-peak basis), within ±5 percentage points |
| SC-5 | All artifacts in §13 are produced | Every deliverable file exists and renders |

A run is considered **passing** only if SC-1 through SC-5 hold simultaneously.

### 1.3 Regression Definition

A re-run is considered a **regression** if any one of the following is true relative to the most recent passing baseline:

- BF16 peak drops by more than 3%.
- Sustained HBM drops by more than 3%.
- Any roofline op moves across the ridge in the wrong direction (compute → memory) without an environmental cause.
- E2E compiled MFU drops by more than 2 percentage points.

---

## 2. Hardware and Software Baseline

### 2.1 Reference Platform

| Property | Value |
|----------|-------|
| Node | 8× AMD Instinct MI355X |
| HBM per GPU | 288 GB HBM3E |
| Peak HBM BW per GPU | 8 TB/s |
| BF16 peak (dense) | 1.26 PFLOP/s — 2.5 PFLOP/s, depending on spec basis |
| Interconnect | Infinity Fabric (intra-node) |

The PDF's measured environment uses **ROCm 7.0 nightly PyTorch**, **Triton ROCm**, and **AITER attention kernels**. These software choices are part of the performance result and **must be recorded as a first-class data column**, not just an implementation detail.

### 2.2 Required Environment Capture

Each run must serialize the following metadata into `env.json` alongside results:

```jsonc
{
  "hardware": {
    "gpu_model": "...",          // e.g. "AMD Instinct MI355X"
    "gpu_count": 8,
    "rocm_smi_dump": "...",       // raw rocm-smi -a output path
    "sclk_state": "...",          // current/max core clock
    "mclk_state": "...",          // current/max memory clock
    "power_cap_w": 0,
    "edge_temp_c": 0,
    "junction_temp_c": 0,
    "mem_temp_c": 0
  },
  "software": {
    "rocm_version": "...",
    "hip_version": "...",
    "kernel_driver": "...",       // amdgpu version / dkms
    "torch_version": "...",
    "torch_build": "nightly|release",
    "triton_version": "...",
    "aiter_commit": "...",
    "flash_attn_backend": "aiter|sdpa|cudnn|xformers",
    "sdpa_kernel_path": "math|memory_efficient|flash|flash_attention",
    "compiler_mode": "eager|reduce-overhead|max-autotune|default",
    "miopen_version": "...",
    "rccl_version": "..."
  },
  "run": {
    "campaign_id": "...",
    "git_sha": "...",
    "host": "...",
    "timestamp_utc": "..."
  }
}
```

### 2.3 Determinism / Stability Pre-Run Checks

Before any benchmark family runs, capture and verify:

1. `rocm-smi -a` snapshot.
2. Confirm GPU at expected sclk/mclk DPM state (no thermal throttling).
3. Confirm power cap.
4. Confirm `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` matches intended scope.
5. Confirm no other compute jobs on the node.
6. Drop filesystem cache before any HBM-from-disk dependent step.

---

## 3. Benchmark Structure

The campaign is split into five families, run in order:

| # | Family | Purpose | Output |
|---|--------|---------|--------|
| 1 | BF16 compute microbenchmarks | Establish compute ceiling | TFLOP/s peak; size-sweep curve |
| 2 | HBM bandwidth microbenchmarks | Establish memory ceiling | GB/s plateau; access-pattern table |
| 3 | VRAM capacity & allocator | Verify usable memory | Max alloc; fragmentation profile |
| 4 | `escher_14b_480p` per-op accounting | Build roofline map | Per-op (FLOPs, bytes, AI, time) table |
| 5 | E2E execution + MFU comparison | Whole-program validation | Eager / compiled MFU table |

Family 1–3 establish the **rooflines and capacity envelope**. Family 4 builds the **per-op accounting**. Family 5 confirms that the modeled kernel-level picture matches the whole-program performance.

---

## 4. Timing Methodology

All families share one timing protocol so MFU comparisons across scopes are valid.

### 4.1 Required Properties of Every Timed Region

1. **Warmup** — kernel compile, autotune, and cache warm before timing starts.
2. **Device-side timing** — `torch.cuda.Event(enable_timing=True)` (HIP-backed on ROCm), not host wall-clock. `torch.cuda.synchronize()` immediately before `.elapsed_time()`.
3. **Frozen shapes** — all tensor shapes fixed at module load. No dynamic recompile inside the timed region.
4. **Repetitions and statistics** — multiple iterations; report distributional statistics, not a single best.

### 4.2 Recommended Iteration Counts

| Test | Warmup iters | Timed iters | Reported stats |
|------|--------------|-------------|----------------|
| Microbenchmarks (GEMM, BW) | 3–20 | 10–30 | median, p10, p90, min, max, std |
| Per-op (workload) | 5 | 20 | median, p10, p90 |
| E2E benchmark | 1 chunk | 24 chunks | mean, std (first chunk discarded) |
| Peak sweep (GEMM tight loop) | 3 | 50–200 | total elapsed / iter count |

### 4.3 Peak-Sweep Special Case

For the BF16 peak number reported on the chart, run a tight loop of repeated identical matmuls with **no Python overhead between iterations** (allocate tensors once, pre-bind handles, run inside a single launched loop where possible) and divide total elapsed time by iteration count. This is the apples-to-apples basis for the compute roof.

### 4.4 Why This Matters

The PDF's MFU numbers are derived from different scopes (sum-of-ops vs eager e2e vs compiled e2e). The cross-scope comparison only works if **each scope is measured with the same timing rules** above.

---

## 5. BF16 Compute Tests (Family 1)

### 5.1 Goal

Determine how close the device gets to peak matrix throughput when non-compute overhead is minimized. Use the result as the **compute ceiling** for the roofline.

### 5.2 Test Cases

#### 5.2.1 Square GEMM Sweep

`torch.matmul(A, B)` with `A, B ∈ bfloat16`, `M = N = K`:

```
sizes = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768]
```

#### 5.2.2 Rectangular GEMM

Shapes matching the workload's projection and FFN layers (extracted from the op table in §8):

- Self-attn QKV projections: `(B*S, d) × (d, 3d)` and `(B*S, d) × (d, d)` for the output projection.
- FFN: `(B*S, d) × (d, 4d)` then `(B*S, 4d) × (4d, d)`.
- Cross-attn Q and O.

#### 5.2.3 Batched GEMM and `addmm`

`torch.bmm` and `torch.addmm` at projection-like shapes to capture launch + epilogue overhead representative of inference layers.

#### 5.2.4 Optional Fused Paths

If a fused path is available for matmul + bias + activation in the active stack, run it and record the delta vs unfused. Document availability in `env.json`.

### 5.3 Metrics

For each test:

| Metric | Definition |
|--------|------------|
| `t_iter_ms` | median per-iter runtime (device events) |
| `tflops_achieved` | `2 * M * N * K / t_iter_s / 1e12` |
| `eff_vs_measured_peak` | `tflops_achieved / measured_peak` |
| `eff_vs_rated_peak` | `tflops_achieved / rated_peak` (1.26 and 2.5 PFLOP/s, both rows) |
| `size_skew_sensitivity` | TFLOP/s as M/K varies at fixed flop count |

### 5.4 Expected Shape

A steep size-dependent ramp; performance becomes near-peak once matrices are large enough to keep matrix units busy. Reproduce this curve in **Chart 1: BF16 GEMM Size Sweep** (§13).

---

## 6. HBM Bandwidth Tests (Family 2)

### 6.1 Goal

Isolate sustained memory traffic with minimal arithmetic to establish the **bandwidth ceiling** for the roofline.

### 6.2 Microbenchmarks

| ID | Op | Pattern | Bytes per element |
|----|-----|---------|-------------------|
| BW-1 | `tensor.copy_(src)` | read + write | 4 (BF16: 2R + 2W) |
| BW-2 | `out = a + b` | 2 reads + 1 write | 6 (BF16) |
| BW-3 | `out = a * b` | 2 reads + 1 write | 6 (BF16) |
| BW-4 | `out.add_(a, alpha=k)` (axpy) | 2 reads + 1 write | 6 (BF16) |
| BW-5 | `tensor.sum()` | read-only reduction | 2 (BF16) |
| BW-6 | `tensor.fill_(v)` | write-only | 2 (BF16) |
| BW-7 | Strided slice copy | non-contiguous read | 4 (BF16) — sensitivity test |

Tensor sizes: powers-of-two from 64 MiB up to a per-GPU cap that stays well clear of OOM (target ≥ 8 GiB per buffer for the plateau).

### 6.3 Bandwidth Formula

```
GB/s = bytes_moved / t_iter_s / 1e9
bytes_moved = num_elements * bytes_per_element_for_op
```

Always state the byte-counting convention explicitly in the artifact.

### 6.4 Metrics

- Sustained GB/s (median across iters, plateau region).
- Variance (p90 − p10).
- Sensitivity to size (curve from launch-bound to BW-bound).
- Sensitivity to access pattern (BW-7 vs BW-1).
- Effective sustained BW / 8 TB/s spec.

### 6.5 Expected Output

A monotonically rising curve flattening into the **plateau** that defines the roofline's bandwidth roof. The plateau, not the spec sheet, is what the roofline uses.

---

## 7. VRAM Capacity Tests (Family 3)

### 7.1 Goal

Determine **practical allocatable memory**, not nominal board capacity. The board spec is 288 GB; usable memory is reduced by driver reserve, framework overhead, fragmentation, and resident context buffers.

### 7.2 Methodology

#### 7.2.1 Single Large Tensor — Binary Search

Allocate one BF16 contiguous tensor; binary-search for the largest stable size that does not OOM after a `torch.cuda.synchronize()` and a no-op kernel launch. Record bytes and surrounding allocator state.

#### 7.2.2 Allocation Patterns

Repeat with:

- Contiguous allocation.
- Non-contiguous (multiple chunks summing to target).
- BF16 and FP16 (verify dtype scaling is exact).

#### 7.2.3 Realistic Headroom

Run with model weights for `escher_14b_480p` already loaded; binary-search remaining allocatable memory for activation/KV-cache style buffers.

### 7.3 Reported Metrics

| Metric | Description |
|--------|-------------|
| `max_alloc_bytes` | Largest single contiguous BF16 tensor |
| `eff_util_fraction` | `max_alloc_bytes / (288 * 1024**3)` |
| `frag_sensitivity` | Max contiguous / max non-contiguous |
| `headroom_after_model` | Remaining bytes after model load |
| `allocator_state` | `torch.cuda.memory_stats()` dump |

The headroom number is the **operationally relevant figure** for inference and diffusion workloads — report it separately and prominently.

---

## 8. `escher_14b_480p` Workload Decomposition (Family 4)

### 8.1 Op Categories

The workload must be decomposed into the same op families as the PDF's table:

| Family | Ops |
|--------|-----|
| Time embed | `time_proj`, `time_embed` |
| Self-attention | `self_attn.q`, `self_attn.k`, `self_attn.v`, `self_attn.o`, `self_attn.flash` |
| Cross-attention | `cross_attn.q`, `cross_attn.flash`, `cross_attn.o` |
| FFN | `ffn.linear1`, `ffn.gelu`, `ffn.linear2` |
| Norms / cache | `norm.*`, `kv_cache_write` |

### 8.2 Per-Op Required Columns

For each op:

| Column | Source |
|--------|--------|
| `op_name` | model trace |
| `category` | mapping above |
| `input_shape` / `output_shape` | model trace, frozen |
| `flops` | analytic from shape + op semantics |
| `bytes_hbm` | analytic from input + output + weight reads (no recompute assumption) |
| `arithmetic_intensity` | `flops / bytes_hbm` |
| `t_measured_ms` | device events, eager path |
| `t_measured_ms_optimized` | device events, fused/AITER path where applicable |
| `bound` | `compute` if `AI > ridge`, else `memory` |

### 8.3 Reference Totals

The PDF's op table totals:

- **4641.7 GFLOPs**
- **3070.2 MB HBM**

The implementation must preserve this accounting structure even if the exact kernels or fused paths differ in future software stacks. Any change to FLOP/byte counting must be a deliberate, reviewed change documented in the artifact.

### 8.4 FLOP Counting Conventions

Document and use exactly one convention throughout:

- GEMM: `2 * M * N * K`.
- Attention QK^T and AV: counted as standard GEMMs over `(B, H, S, D)` shapes.
- Softmax: counted as bandwidth-only (no FLOP credit beyond exp/normalize, which is negligible vs GEMM).
- GELU: bandwidth-only for accounting.
- Norms: bandwidth-only.
- Element-wise: bandwidth-only.

Any deviation must be flagged in the per-op CSV.

---

## 9. Roofline Analysis

### 9.1 Construction

- **X-axis:** arithmetic intensity (FLOP/B), log scale.
- **Y-axis:** achieved BF16 performance (TFLOP/s), log scale.
- **Compute roof:** measured BF16 GEMM peak from §5.
- **Bandwidth roof:** sustained HBM plateau from §6 (line `y = AI × BW_sustained`).
- **Ridge point:** `compute_peak / BW_sustained`. The PDF places the MI355X ridge near **210 FLOP/B**; this is the practical divider between bandwidth-bound and compute-bound.

### 9.2 Plotting Rules

- One marker per op from §8.
- Color-code by category: `time_proj`, `self_attn`, `cross_attn`, `ffn`, `norm/cache`.
- Marker shape encodes optimized-vs-default path.
- Compute and bandwidth roofs are drawn as a single piecewise envelope.
- Annotate the ridge point and the major outlier ops.

### 9.3 Expected Result

- Large GEMMs and attention kernels: high on the plot, **right of the ridge**, compute-bound.
- Norms, GELU, small projections: lower, **left of the ridge**, tracking the bandwidth roof.
- Anything sitting well below both roofs is a kernel-implementation issue, not a hardware limit, and should be flagged.

---

## 10. Per-Op Timing vs Theory

### 10.1 Theoretical Times

For each op:

```
t_compute_theoretical = flops / compute_peak
t_memory_theoretical  = bytes_hbm / bw_sustained
t_bottleneck          = max(t_compute_theoretical, t_memory_theoretical)
```

`compute_peak` and `bw_sustained` come from §5 and §6 respectively (measured, not rated).

### 10.2 Measured Times

For each op record:

- `t_measured_eager_default` — default torch path, no AITER, default SDPA backend.
- `t_measured_optimized` — AITER attention, fused paths where available.

### 10.3 Required Chart

A grouped bar chart, one group per op, three bars per group:

1. `t_bottleneck` (theory).
2. `t_measured_eager_default`.
3. `t_measured_optimized`.

This chart is where the PDF's **SDPA vs AITER** story shows up: default torch SDPA is much slower on the attention-heavy path, AITER closes most of the gap, and the remaining gap to theory is the implementation-quality tax.

### 10.4 Interpretation Rules

- `measured / theory ≤ 1.1` → kernel is at the hardware limit; further wins require algorithmic or hardware changes.
- `1.1 < measured / theory ≤ 1.5` → tunable: autotuning, fusion, layout.
- `measured / theory > 1.5` → likely kernel implementation issue; flag for follow-up.

---

## 11. End-to-End Benchmark and MFU

### 11.1 Run Configurations

Run the full `escher_14b_480p` transformer stack in three configurations, with the same input shape, weights, and FLOP basis:

1. **Sum-of-ops (isolated)** — sum of per-op measured runtimes from §8.
2. **Eager e2e** — model executed end-to-end without `torch.compile`.
3. **Compiled e2e** — model executed end-to-end with `torch.compile` (mode documented in `env.json`).

### 11.2 MFU Computation

For each scope:

```
flops_total       = sum(per_op_flops)            # same basis across all 3 scopes
t_total_s         = measured_runtime_s_for_scope
tflops_achieved   = flops_total / t_total_s / 1e12
mfu_measured_peak = tflops_achieved / measured_peak_tflops_from_section_5
mfu_rated_peak    = tflops_achieved / rated_peak_tflops    # report both 1.26 and 2.5 PFLOP/s rows
```

### 11.3 Reference Targets (from PDF)

On the **measured chip peak basis**:

| Scope | PDF target |
|-------|------------|
| Sum-of-ops | ≈ 77% |
| Eager e2e | ≈ 93% |
| Compiled e2e | ≈ 99% |

### 11.4 Interpretation

If compiled e2e MFU exceeds sum-of-ops MFU, that is **not inherently suspicious** — it indicates the compiled graph fuses work and removes dispatch / launch / framework overhead. If the e2e number looks too good, audit:

- FLOP accounting for missing ops.
- Bytes accounting for ignored loads/stores.
- Whether timed region excludes warmup, allocator churn, or async H2D copies.

Document the audit result alongside the MFU table.

### 11.5 E2E Timing Specifics

- 25 chunks, first discarded as warmup.
- Each chunk identical in shape and content.
- Synchronize at chunk boundaries only.
- Report `mean ± std` and the per-chunk distribution.

---

## 12. Multi-GPU Extension (Optional)

### 12.1 Scope

Tensor-parallel communication tests at world sizes 2, 4, 8 on the 8-GPU node. Focus is the all-gather / reduce-scatter pattern that pairs with TP projection and reduction phases.

### 12.2 Test Cases

| Test | Op | Payloads |
|------|-----|----------|
| TP-1 | `all_gather` BW | sizes matching TP activations: ranges from 1 MiB up to several hundred MiB |
| TP-2 | `reduce_scatter` BW | same payloads as TP-1 |
| TP-3 | Strong scaling | fixed problem size, world ∈ {1, 2, 4, 8} |
| TP-4 | Per-rank timing | barrier before and after each measured interval |

### 12.3 Metrics

- Communication BW plateau (GB/s).
- Payload-dependent achieved bus BW (algorithmic vs bus distinction documented).
- Speedup vs ideal linear scaling.
- Communication / computation ratio at each world size — **must remain compute-favorable** for the workload to scale.

### 12.4 Interpretation

The PDF's conclusion is that the communication path is promising but current fused kernels still need work. Treat this section as both a perf measurement and a **kernel readiness check**: if fused projection+communication kernels are unavailable in the current stack, record that explicitly, since it is the dominant headroom item for TP scaling.

---

## 13. Required Output Artifacts

Every campaign run must produce, under `results/<campaign_id>/`:

| # | Artifact | Format | Source |
|---|----------|--------|--------|
| A1 | Hardware summary (BF16 peak, HBM BW, VRAM cap) | table + chart | §5, §6, §7 |
| A2 | BF16 GEMM size-sweep chart | png/svg | §5 |
| A3 | HBM bandwidth chart | png/svg | §6 |
| A4 | VRAM capacity report | md + json | §7 |
| A5 | `escher_14b_480p` op table (FLOPs, bytes, AI, time) | csv + md | §8 |
| A6 | Roofline plot with MI355X compute and BW ceilings | png/svg | §9 |
| A7 | Per-op theory-vs-measured chart | png/svg | §10 |
| A8 | MFU table/chart for sum-of-ops, eager e2e, compiled e2e | csv + chart | §11 |
| A9 | Multi-GPU communication and scaling charts (optional) | png/svg | §12 |
| A10 | `env.json` | json | §2 |
| A11 | `summary.md` (auto-generated, links A1–A10) | md | aggregator |

`summary.md` is the human-readable entry point and must lead with the SC-1…SC-5 pass/fail badge from §1.2.

---

## 14. Acceptance and Interpretation

### 14.1 What a "Good Run" Looks Like

- Large BF16 GEMMs sit close to peak.
- The workload's major GEMM and attention ops sit **above the roofline ridge**, confirming a compute-dominant transformer stack.
- Memory-bound ops (norms, GELU, small projections) sit below the ridge and track the bandwidth ceiling.
- Eager e2e MFU > sum-of-ops MFU; compiled e2e MFU > eager e2e MFU.

### 14.2 What Is Not Suspicious

Compiled e2e MFU exceeding sum-of-ops MFU is **expected**, because the compiled graph reduces dispatch, fusion gaps, and launch overhead that the per-op accounting bakes in.

### 14.3 What Is Suspicious

- E2E MFU > 100% on rated peak (basis problem, not real).
- Compiled e2e dramatically faster than the *theoretical bottleneck sum* of its ops.
- BW microbench above the spec sheet.

When these appear, **audit accounting first** (missing FLOPs, double-counted overhead, untimed sync points). The PDF itself flags this as the most likely explanation.

### 14.4 Sign-Off Rule

A campaign is signed off when:

1. SC-1 … SC-5 all pass.
2. `env.json` is complete.
3. `summary.md` audit section is filled in (even if "no anomalies").
4. The artifact set A1 … A10 (A9 if multi-GPU) is committed.

---

## 15. Implementation Priorities (Execution Order)

Run in this order. Each step's output anchors the next; skipping or reordering breaks the chain that connects raw measurements to MFU.

1. **Verify environment and device state** (§2.3) — clocks, power cap, thermals, env capture.
2. **BF16 compute ceiling** (§5) — establishes compute roof for §9, denominator for §11 MFU.
3. **HBM bandwidth ceiling** (§6) — establishes bandwidth roof for §9 and `t_memory_theoretical` for §10.
4. **VRAM capacity** (§7) — confirms the workload and timing buffers fit; informs e2e batch sizing.
5. **Op-level FLOP and byte accounting for `escher_14b_480p`** (§8) — produces the table that everything downstream cites.
6. **Roofline plot** (§9) — first integration check across §5/§6/§8.
7. **E2E eager and compiled** (§11) — measures the end-to-end story.
8. **Compute MFU and compare against PDF targets** (§11.3) — the campaign-level pass/fail step.
9. *(Optional)* **Multi-GPU communication and scaling** (§12).

Each derived metric in this plan is anchored in a direct measurement from an earlier step. When a result diverges from the PDF or from a prior baseline, this ordering also gives a clean attribution path:

- Diverged at step 2 → hardware/clock/thermal regression.
- Diverged at step 3 → memory subsystem or driver regression.
- Diverged at step 5 → model graph or shape change (recheck FLOP/byte accounting).
- Diverged at step 6 only → ceiling vs op metadata mismatch.
- Diverged at step 7 only → compiler / fusion / launch-overhead change.
- Diverged at step 8 only → MFU basis change (rated vs measured peak, FLOP convention).

This is the audit ladder. Use it to decide whether a discrepancy comes from hardware behavior, kernel implementation, compiler behavior, or FLOP accounting **before** investing in deeper investigation.

---

## 16. Collectability Audit

This section audits the test plan against the implementation in this repo
(`benchmarks/`, `validation/`, `scripts/`) to confirm that every metric named
in §1–§14 is actually emitted by some script, and that every external-tool
cross-check named in this repo is wired into `validation/compare.py`.

### 16.1 What each TESTPLAN section is collected by

| § | Plan item | Implementation | Output artifact |
|---|-----------|----------------|-----------------|
| 2.2 | env.json schema (hardware, software, run) | `benchmarks/common/env.py` | `env.json` |
| 2.3 | rocm-smi snapshot, clock state, power cap | `env.py` (`rocm-smi -a`, `rocminfo`) | `env.json.hardware.rocm_smi_dump` |
| 4 | warmup, device events, repetitions, distributional stats | `benchmarks/common/timing.py` (`time_op`, `time_tight_loop`) | per-test JSON |
| 4.3 | tight-loop peak measurement | `time_tight_loop` | `01_bf16_compute/peak.json` |
| 5.2.1 | square GEMM sweep up to 32768 | `bench01.square_sweep` | `01_bf16_compute/sweep.{json,csv}` |
| 5.2.2 | rectangular GEMM at projection / FFN shapes | `bench01.rectangular_sweep` | same |
| 5.2.3 | batched GEMM and addmm | `bench01.addmm_and_bmm` | same |
| 5.2.4 | optional fused matmul+bias+activation | **not implemented** — backend-dependent; document availability in `env.json` and add ad-hoc when stack supports it |
| 5.3 | TFLOP/s, eff vs measured, eff vs rated | `bench01` `_row` + `summary.json` |
| 6.2 | copy_, add, mul, axpy, sum, fill_, strided | `bench02` BW-1…BW-7 | `02_hbm_bandwidth/bandwidth.{csv,json}` |
| 6.4 | sustained GB/s, variance, plateau | `bench02` + `summary.json.plateau_gb_s_per_op` | same |
| 7.2 | binary search bf16 + fp16 | `bench03.binary_search_max_contig` | `03_vram_capacity/summary.json` |
| 7.2.2 | non-contiguous fragmentation | `bench03.fragmentation_probe` | same |
| 7.2.3 | headroom with model loaded | **partial** — `bench05` allocates the model, allocator stats are in `env.json`; an explicit "headroom-after-load binary search" is not run separately to keep the campaign single-pass. Add a flag if a hard headroom number is required. |
| 8 | per-op decomposition + analytic FLOP/byte | `benchmarks/common/flop_accounting.py` + `bench04` | `04_workload_ops/ops.{csv,json,md}` |
| 8.3 | calibration drift vs reference totals | `bench04.calibration_drift` | `04_workload_ops/ops.json.calibration_drift` |
| 8.4 | FLOP convention (GEMM = 2·M·N·K, softmax/GELU/norm = bandwidth-only) | `flop_accounting.gemm_op` + `attention_flash` + `elementwise_op` | enforced by code |
| 9 | roofline construction + ridge | `bench04` (computes ridge, classifies bound) + `scripts/plot_results.plot_roofline` | `plots/A6_roofline.png` |
| 10 | theory + measured-default + measured-optimized per op | `bench04._runner_for_op` runs `default` (no AITER, math+mem-efficient SDPA) and `optimized` (AITER → flash_attn → SDPA flash) | `04_workload_ops/ops.json.t_*` + `plots/A7_per_op_theory_vs_meas.png` |
| 10.4 | meas/theory threshold buckets | computed as `meas_over_theory_*`; thresholds in plan, not auto-flagged |
| 11.1 | three scopes: sum-of-ops, eager, compiled | `bench05` | `05_e2e_mfu/mfu.{csv,json,md}` |
| 11.2 | MFU on measured + 1.26 PF + 2.5 PF bases | `bench05._mfu_row` | same |
| 11.4 | audit prose for too-good-to-be-true e2e | author-supplied in `summary.md` (not auto-generated) |
| 11.5 | 25 chunks, first discarded as warmup | `cfg.timing.e2e_chunks` honored in `bench05` | same |
| 12 | all-gather, reduce-scatter, all-reduce + bus BW | `bench06` (torchrun) | `06_multigpu_comm/comm.{csv,json}` |
| 12.2 (TP-3) | strong scaling at world ∈ {2,4,8} | **partial** — `bench06` runs at the world size torchrun was launched with. To collect the full strong-scaling sweep, run the campaign three times with `NPROC=2,4,8` or extend `bench06` to subdivide and call into multiple sub-process-groups. Documented as a known operational caveat. |
| 13 | A1…A11 artifacts | `bench0x` + `scripts/plot_results.py` + `summary.md` | `results/<id>/` |
| 1.2 SC-1…SC-5 | pass/fail scorecard | `scripts/score_campaign.py` | `results/<id>/scorecard.{md,json}` |

### 16.2 Validation cross-checks (PyTorch vs ground truth)

| PyTorch metric | Ground-truth tool | Tolerance | Wired in |
|----------------|-------------------|-----------|----------|
| BF16 compute peak (`bench01`) | RVS `gst` (`validation/rvs/gst_bf16_mi355x.conf`) — bf16 GEMM stress, reports `gflops_actual` | ±10% | `validation/compare.py:parse_rvs_gst` |
| HBM bandwidth roof (`bench02` plateau) | `rocm-bandwidth-test -e <gpu> -m ...` D2D max GB/s | ±15% | `validation/compare.py:parse_rocm_bw` |
| HBM integrity under load | RVS `mem` (`mem_mi355x.conf`) — pass/fail integrity, NOT a BW measurement | n/a (binary) | run only; integrity log in `validation/rvs/mem.stdout.log` |
| PCIe H2D / D2H | RVS `pebb` (`pebb_pcie.conf`) | n/a — informational | run only |
| `all_gather` / `reduce_scatter` / `all_reduce` busbw (`bench06`) | `rccl-tests all_{gather,reduce,reduce_scatter}_perf` | ±10% per payload | `validation/compare.py:parse_rccl_log` |

`validation/compare.py` writes `validation.{md,json}` with PASS / FAIL / SKIP
per row. `SKIP` means the ground-truth tool was not installed; the campaign
proceeds but the cross-check is recorded as SKIP, not as PASS.

### 16.3 Known operational caveats

1. **RVS `gst` BF16 path** depends on the local RVS build's `data_type` support
   (`rvs_blas_types_bf16`). On builds without bf16 wired into GST, the conf
   falls back to `hgemm` (fp16) — record this in `env.json.software.rvs_*`
   and treat the cross-check as a soft check.
2. **`torch.compile` mode availability** varies across ROCm PyTorch nightlies;
   `bench05` tries `max-autotune` → `reduce-overhead` → `default` and records
   which mode ran. SC-4 is computed against whichever mode succeeded.
3. **Multi-GPU strong scaling** (TESTPLAN §12.2 TP-3) collects only the
   world size torchrun launched with. Run `NPROC=2 ./scripts/run_campaign.sh ...`,
   `NPROC=4 ...`, `NPROC=8 ...` to populate the full scaling table, or extend
   `bench06` to launch sub-process-groups in a single run.
4. **AITER attention path** is best-effort: `bench04.attention_optimized`
   probes `aiter` then `flash_attn` then SDPA-flash. The path actually used
   is recorded per-op via the optimized timing + the `env.json` import probe;
   if neither AITER nor flash_attn imports succeed, the optimized column equals
   SDPA-flash, which understates §11's optimized-vs-default delta.
5. **Headroom-after-model** (§7.2.3) is observed indirectly via `bench05`
   model allocation + `torch.cuda.memory_stats`. For a hard number, add a
   `--measure-headroom` flag to `bench03` that loads the DiT and then
   binary-searches remaining capacity. Not run by default to keep the
   campaign single-pass.
6. **FLOP/byte accounting policy** (§8.4) counts every weight read once per
   op call, with no L2 reuse credit. This deliberately upper-bounds HBM bytes
   and therefore lower-bounds arithmetic intensity, which is why our default
   per-block byte total can run higher than the PDF's. The convention is
   correct for the roofline (worst-case bandwidth pressure). Switch to a
   reuse-aware counter only with explicit policy review.
7. **MFU rated bases**: every MFU row reports both 1.26 PF and 2.5 PF
   denominators. The PDF's headline numbers (77 / 99) use the measured
   chip peak, which is what `mfu_measured_peak` expresses; the rated columns
   are present so external readers can re-derive on their preferred basis.

### 16.4 Unified runner (`test.sh`)

`test.sh` is the GuideLLM-style entrypoint that wraps every benchmark family,
the validators, and the report generator behind named `--testcase` /
`--workload` selectors. The same set of testcase keys is recognized on the
command line:

| `-t` value | What it runs |
|------------|--------------|
| `compute` `bandwidth` `vram` `workload` `e2e` `multigpu` | the matching `bench0X` family |
| `validation` | `env.py` + RVS + `rocm-bandwidth-test` + `rccl-tests` + `compare.py` |
| `plot` `score` `report` | regenerate the post-processing artifacts only |
| `campaign` | full TESTPLAN §15 sequence (delegates to `scripts/run_campaign.sh`) |

`-w` selects a workload variant. `escher_14b_480p` keeps `configs/escher_14b_480p.json`
verbatim; `smoke` and `big` materialize a derived JSON in
`results/<ts>-<tc>-<wl>-rN/derived_config.json` with depth/seq overrides applied
(no in-place mutation of the canonical config). `-i N` repeats each
`(testcase, workload)` pair N times so that a regression-averaging campaign
can be assembled without re-typing arguments.

### 16.5 Per-job runner (`run.sh`) and report shortcut (`report.py`)

`run.sh` is the per-job runner that `test.sh` delegates to. Invoked
directly it accepts the same testcase / workload / iteration options and
adds the system-profiling layer (Phoronix-style):

| Flag | Behavior |
|------|----------|
| `--prepare-sys` | `numa_balancing=0`, transparent hugepages always, `vm.compact_memory=1`, `drop_caches=3`, `rocm-smi --setperflevel high`. Requires `sudo`; failures non-fatal. Off by default. |
| `--no-stat` / `STAT=off` | Disable the live telemetry poller. |
| `--stat-interval N` | Poller interval seconds (default 1). |
| `--numactl "<spec>"` | Wrap each non-distributed iteration with `numactl <spec>`. Ignored for `torchrun` jobs. |

The poller auto-selects `amd-smi metric --watch` →
`rocm-smi --csv` loop → `nvidia-smi --query-gpu=...` → disabled. Sidecar
files are written next to the main log under `results/_logs/`.

`report.py` (top-level) is a thin shortcut over `scripts/report.py`. With
no `--out`, it picks the most recent `${RESULTS_DIR:-results}/*-campaign-*`
directory (mtime sort) and forwards everything else verbatim. Falls back
to `<repo>/runs/` for backward compatibility with trees from before the
`runs/` → `results/` rename.

### 16.6 CPU-host (no accelerator) behavior

The campaign is fully runnable on a CPU-only host so the analytic and
post-processing layers can be exercised in CI without a GPU:

| Step | Behavior on CPU |
|------|-----------------|
| `setup.sh` | Installs CPU `torch + torchvision` from `https://download.pytorch.org/whl/cpu`; creates `.microbenchmarks-cpu-venv/`. |
| `bench01..03`, `bench05`, `bench06` | Skipped explicitly with `[campaign] benchXX: skipped (DEVICE=cpu)` log lines. |
| `bench04_workload_ops` | Runs the analytic FLOP/byte/AI table; `t_ms_default` / `t_ms_optimized` columns are NaN by design. |
| External validators (RVS / `rocm-bandwidth-test` / `rccl-tests`) | Skipped. |
| `compare.py` | PASS=0 FAIL=0 SKIP=2 (BF16 vs RVS, BW vs rocm-bw both SKIP, no FAILs). |
| `score_campaign.py` | Inserts a `HOST CPU` row at the top; SC-1, SC-2, SC-4 → `SKIP reason=missing <gpu artifact>`; SC-5 → `SKIP reason=no GPU detected`. SC-3 (workload op taxonomy) is graded normally. |
| `plot_results.py`, `report.py` | Run end-to-end; plots show only the per-op theory chart (A7) since GPU artifacts are absent. |
| `run_campaign.sh` exit code | `0` on a CPU host (GPU steps intentionally skipped) so CI smoke gates pass. |

This guarantees that doc-pipeline regressions, scorecard plumbing,
calibration drift logic, and per-op accounting are caught in CI even when
the runner has no accelerator.

### 16.7 What the campaign WILL NOT collect

These are explicitly out of scope for a single run and require manual ops:

- Sustained / 24-hour stability (§ Future Work in PDF).
- Power draw, thermal trajectory beyond the start-of-run snapshot in `env.json`.
- VAE encoder / decoder performance (PDF: "Scope: Only transformer stack
  optimized, VAE untouched").
- Multi-node scaling (the rccl path here is intra-node Infinity Fabric).
- Fused AG+MM and MM+RS kernel evaluation (PDF flags these as "not yet
  optimal" — when AITER ships them, add to `bench06` as new ops).

