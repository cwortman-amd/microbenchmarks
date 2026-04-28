# TESTPLAN.md — `escher_14b_480p` on MI355X Benchmark Campaign

A detailed, technical, repeatable test plan to reproduce the PDF's methodology, tables, and result structure for the `escher_14b_480p` workload on AMD Instinct MI355X. The plan is organized as a single benchmark campaign that can be re-run after software changes to quantify regression or improvement.

---

## 1. Objective and Success Criteria

### 1.1 Primary Objective

Characterize GPU performance for the `escher_14b_480p` workload on MI355X across three axes:

1. **Compute** — BF16 matrix throughput.
2. **Memory bandwidth** — sustained HBM3E throughput.
3. **Memory capacity** — practical allocatable device DRAM (HBM3E on MI355X, system DDR on CPU hosts).

The plan must reproduce, on the reference platform:

- The PDF's BF16 dense compute result.
- The HBM bandwidth characterization.
- The roofline placement of `escher_14b_480p` ops.
- The MFU comparison between sum-of-ops, eager e2e, and `torch.compile` e2e.

### 1.2 Success Criteria

The campaign grades twelve criteria. SC-1 … SC-6 are **gating** (all must
hold for sign-off); SC-7 … SC-12 are **opt-in** — they `SKIP` when their
underlying probe was not exercised, but `FAIL` if it was and produced a
regressing number. Every numeric threshold below is the default in
`scripts/score_campaign.py` and the report-side classifier in
`configs/report_config.json` — both files are the single auditable source.

| ID | Criterion | Pass Condition | Role |
|----|-----------|----------------|------|
| SC-1 | BF16 GEMM microbenchmarks approach the target's dense peak | Largest square GEMM ≥ 90 % of measured peak AND ≥ 50 % of rated peak (basis matches the PDF) | gating |
| SC-2 | Memory plateau stability | Streaming `copy_`/`add` plateaus within ±5 % across 3 successive sizes near the ceiling (±15 % on CPU host's DDR) | gating |
| SC-3 | Roofline placement is correct | Large GEMMs and attention sit AI > ridge (compute-bound); norms / GELU / small projections sit AI < ridge (memory-bound) | gating |
| SC-4 | MFU ordering matches the PDF | Compiled E2E MFU ≥ eager E2E MFU ≥ sum-of-ops MFU within ±`mfu_pdf_tolerance_pp` (default 5 pp) of the workload's `source_pilot_reference.pdf_reference_targets_pct` | gating |
| SC-5 | All required artifacts in §13 are produced | Every A1 … A11 deliverable exists, including `report.{md,html,pdf}` | gating |
| SC-6 | BF16 GEMM is numerically equivalent to FP32 | All `bench01.correctness_check` rows ≤ analytic `5·√K·2⁻⁸` rel-error bound | gating |
| SC-7 | Sustained throughput is stable over the `bench07` window | head→tail drift < 5 %, σ growth < 3×, no clock drop > 10 % | opt-in (skips when bench07 didn't run) |
| SC-8 | Across-invocation variability is bounded | Cross-run CV % on the primary throughput metric ≤ threshold (default 10 %) — fed by `scripts/across_run_variability.py` | opt-in |
| SC-9 | Ground-truth validation | `validation/compare.py` produces no `FAIL` rows: PyTorch microbench numbers agree with RVS / `rocm-bandwidth-test` / `rccl-tests` within their per-metric tolerance | opt-in (skips when the external tools are not installed) |
| SC-10 | Numerical-stability sweep | Every `(dtype, K)` row in `bench09` lands inside its analytic error bound | opt-in |
| SC-11 | Post-model-load residual capacity | `bench03 --measure-headroom` reports residual ≥ workload's per-block activation budget; emits `WARN_CPU` on a CPU host where the DiT does not fit in RAM | opt-in |
| SC-12 | Fused AG+MM / MM+RS kernels | At least one fused backend resolves AND its end-to-end time is faster than the AG-then-MM (or MM-then-RS) sequential reference. The vendored `benchmarks.aiter_kernels` backend always resolves on a CUDA host with triton, so SC-12 grades to PASS / FAIL (not SKIP) on a GPU node. `SKIP` only on CPU hosts and on hosts where neither triton, AITER+Iris, nor `torch.ops.symm_mem` is present. | opt-in |

A run is considered **passing** only if SC-1 through SC-6 hold
simultaneously and `compare.py` shows no `FAIL` rows. SC-7 … SC-12 are
recorded in `scorecard.json` but do not gate the orchestrator's exit code
unless the matching probe ran and produced a `FAIL`.

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

The campaign is split into nine families. Families 1–6 are the **default
single-pass campaign** wired into `test.sh` / `run.sh` /
`scripts/run_campaign.sh`. Families 6-fused / 7 / 8 / 9 are **opt-in
probes**: invoked directly when the matching SC needs collecting, but
not part of the default `-t campaign` flow (so a CI smoke run stays cheap).

| # | Family | Purpose | Output | Default |
|---|--------|---------|--------|---------|
| 1 | BF16 compute microbenchmarks (`bench01_bf16_compute`) | Establish compute ceiling + BF16-vs-FP32 correctness | TFLOP/s peak; size-sweep curve; per-component GEMM table; correctness rows | yes (gates SC-1, SC-6) |
| 2 | Memory bandwidth microbenchmarks (`bench02_hbm_bandwidth`) | Establish memory ceiling | GB/s plateau; access-pattern table | yes (gates SC-2) |
| 3 | DRAM capacity & allocator (`bench03_dram_capacity`) | Verify usable memory; optional residual-after-model headroom | Max alloc; fragmentation profile; `headroom_after_model` (with `--measure-headroom`) | yes (gates SC-11 when `--measure-headroom`) |
| 4 | `escher_14b_480p` per-op accounting (`bench04_workload_ops`) | Build roofline map | Per-op (FLOPs, bytes, AI, time) table; ops.{csv,json,md} | yes (gates SC-3) |
| 5 | E2E execution + MFU comparison (`bench05_e2e_mfu`) | Whole-program validation | Eager / compiled MFU table; per-chunk distribution | yes (gates SC-4) |
| 6 | Multi-GPU collectives (`bench06_multigpu_comm`) | TP collective busbw | `comm.{csv,json}` per (op, payload, world) | yes |
| 6f | Fused multi-GPU compute+collective AITER side (`bench06_aiter_fused`, legacy alias `bench06_fused`) | AG+MM and MM+RS fused-kernel availability + speedup vs sequential reference, sourced from upstream AITER first then `benchmarks/aiter_kernels/` (vendored, AITER-conformant) | `06_multigpu_fused/fused.{json,csv}` with `api_source`, `call_kind`, fused-vs-sequential ratio | opt-in (gates SC-12) |
| 10 | Fused multi-GPU compute+collective torch SymmMem side (`bench10_symm_fused`) | Same shape sweep as 6f against `torch.ops.symm_mem.fused_*` after a runtime correctness gate vs the SymmMem fallback helpers | `10_symm_fused/fused.{json,csv}` | opt-in (also feeds SC-12) |
| 7 | Sustained throughput / thermal-drift (`bench07_sustained`) | Detect head→tail drift, σ growth, clock drop on a long run | `sustained.json` per-window throughput + paired `telemetry.json` (power, temp, clocks) | opt-in (gates SC-7) |
| 8 | Topology / inter-device bandwidth (`bench08_topology_bw`) | All-pairs D2D (or inter-CCD/inter-socket) BW matrix | `topology.json` matrix + per-pair fabric labelling | opt-in |
| 9 | Numerical-stability sweep (`bench09_numerical_stability`) | Per-(dtype, K) GEMM error distribution vs FP32 reference | `stability.{csv,json}` | opt-in (gates SC-10) |

Family 1–3 establish the **rooflines and capacity envelope**. Family 4
builds the **per-op accounting**. Family 5 confirms that the modeled
kernel-level picture matches the whole-program performance. Family 6 (and
6-fused) cover the multi-GPU TP path. Families 7–9 fill in the long-run /
topology / numerical envelope of the campaign so the report's executive
summary can carry a `PASS` rather than a `SKIP` against those rows.

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

## 7. DRAM Capacity Tests (Family 3)

### 7.1 Goal

Determine **practical allocatable device memory**, not nominal board capacity. The board spec is 288 GB (MI355X HBM3E); usable memory is reduced by driver reserve, framework overhead, fragmentation, and resident context buffers. Named generically (DRAM) so the same accounting applies on CPU hosts where the bench measures system DDR instead of HBM.

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

### 12.5 CPU host analogue: multi-CCD / multi-socket

When the campaign is run on a CPU host (no CUDA/HIP device visible), bench06's "multi-GPU" sweep is automatically remapped onto the CPU's hardware topology so the same JSON schema and the same payload sweep still produce a meaningful interconnect measurement:

- **Backend:** `gloo` over loopback (instead of `nccl` / `rccl`).
- **Rank → hardware mapping** is selected at launch time:
  - `ccd` (default on AMD): one rank per Linux `die_id`, i.e. one rank per CCD. On EPYC / Threadripper PRO this is the **Infinity Fabric** boundary inside a socket. Each rank is pinned with `os.sched_setaffinity()` to the CPU set returned by sysfs (`/sys/devices/system/cpu/cpu*/topology/{die_id,physical_package_id}`) and `torch.set_num_threads()` is set to that CCD's physical-core count (8 cores / 16 threads is the AMD canonical CCD).
  - `socket`: one rank per `physical_package_id`. Crosses the **inter-socket interconnect** (xGMI on AMD, UPI on Intel) — this is the dual-socket case.
  - `split`: ignore topology and slice the union of online CPUs into `world` equal-sized contiguous groups. Used as a fallback when the kernel / hypervisor flattens `die_id` (common in WSL2 and some VM types).
  - `auto` (orchestrator default): try `ccd`, then `socket`, then `split`.
- **World-size selection:** the campaign orchestrator picks `WORLD = max(#CCDs, #sockets)` automatically; override with `WORLD=N` and `CPU_TOPOLOGY=ccd|socket|split|auto`.
- **JSON output:** `06_multigpu_comm/comm.json` gains a `cpu_topology` block with the resolved mode, sockets / dies / cores-per-die / threads-per-core, and a `rank_pinning` array of `(rank, n_cpus, cpus[])`. The `device_type` field is `"cpu"` so report.py knows to retitle the section "Multi-CCD / Multi-Socket Communication".
- **Caveats:** numbers reflect gloo + loopback TCP plus the inter-CCD or inter-socket fabric, *not* RCCL/NCCL. They are intended for regression detection and CI smoke tests, not as a substitute for the MI355X TP scaling number. The ridge between intra-CCD memcpy and Infinity Fabric still appears in the busbw curve, which is the actionable signal.

---

## 13. Required Output Artifacts

Every campaign run must produce, under `results/<campaign_id>/`:

| # | Artifact | Format | Source |
|---|----------|--------|--------|
| A1 | Hardware summary (BF16 peak, HBM BW, DRAM cap) | table + chart | §5, §6, §7 |
| A1b | **Component GEMMs** — per-decomposition (M, K, N) shape table + measured BF16 TFLOP/s | json + csv + md table | §5, §8 (`bench01.component_gemm_sweep`, `flop_accounting.gemm_inventory`) |
| A2 | BF16 GEMM size-sweep chart | png/svg | §5 |
| A3 | HBM bandwidth chart | png/svg | §6 |
| A4 | DRAM capacity report | md + json | §7 |
| A5 | `escher_14b_480p` op table (FLOPs, bytes, AI, time) | csv + md | §8 |
| A6 | Roofline plot with MI355X compute and BW ceilings | png/svg | §9 |
| A7 | Per-op theory-vs-measured chart | png/svg | §10 |
| A8 | MFU table/chart for sum-of-ops, eager e2e, compiled e2e — grouped bars across measured-peak and the target's two rated-peak rows, with PDF reference-target overlay sourced from the workload's `source_pilot_reference.pdf_reference_targets_pct` | csv + json + chart | §11 |
| A8b | Per-chunk e2e timing distribution (boxplot + strip plot) — reproduces the PDF's "compiled e2e is more stable" finding | chart (png) | §11.5 |
| A9 | Multi-GPU communication and scaling charts (optional) | png/svg | §12 |
| A10 | `env.json` | json | §2 |
| A11 | `summary.md` (auto-generated, links A1–A10) | md | aggregator |
| A12 | `report.{md,html,pdf}` (data-driven, mirrors the source PDF) — `pdf` is generated by default via `wkhtmltopdf` (preferred) or `pandoc + xelatex`; suppress with `--no-pdf` | md + html + pdf | `scripts/report.py` |
| A13 | `scorecard.{md,json}` — SC-1 … SC-12 rollup with a HOST row at the top | md + json | `scripts/score_campaign.py` |
| A14 | `validation.{md,json}` — PyTorch vs RVS / `rocm-bandwidth-test` / `rccl-tests` (gates SC-9) | md + json | `validation/compare.py` |
| A15 | `06_multigpu_fused/fused.{json,csv}` — fused AG+MM / MM+RS kernel availability, dispatcher pick (`api_source`, `call_kind`), per-shape `t_ms` / `tflops` / `ag_gb_s` / `rs_gb_s`, plus the fused-vs-sequential speedup ratio (gates SC-12; opt-in) | json + csv | `bench06_aiter_fused` (legacy entry: `bench06_fused`) |
| A15b | `10_symm_fused/fused.{json,csv}` — same schema as A15 but fixed to `torch.ops.symm_mem` after a runtime correctness gate against the SymmMem fallback helpers (also feeds SC-12) | json + csv | `bench10_symm_fused` |
| A16 | `07_sustained/{sustained.json,telemetry.json}` — per-window throughput, σ, paired SMI telemetry (gates SC-7; opt-in) | json | `bench07_sustained` |
| A17 | `08_topology/topology.json` — all-pairs device-to-device (or inter-CCD/inter-socket) BW matrix with per-pair fabric label (opt-in) | json | `bench08_topology_bw` |
| A18 | `09_stability/stability.{csv,json}` — per-(dtype, K) GEMM error distribution (gates SC-10; opt-in) | csv + json | `bench09_numerical_stability` |

`summary.md` is the human-readable entry point and must lead with the
SC-1 … SC-12 pass/fail badge from §1.2 (gating row first, opt-in rows
collapsed beneath). The full narrative lives in `report.{md,html,pdf}`,
which is data-driven from `configs/report_config.json` (target registry,
classification thresholds, glossary) and the workload's
`source_pilot_reference` block.

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

1. **All gating SCs pass** — SC-1 … SC-6 (see §1.2). Opt-in SC-7 … SC-12
   are recorded as `PASS` / `FAIL` / `SKIP` in `scorecard.json`; only the
   `FAIL` case blocks sign-off, and only when the matching probe was
   actually exercised.
2. `env.json` is complete.
3. `summary.md` audit section is filled in (even if "no anomalies").
4. The artifact set A1 … A14 is committed (A15 … A18 only if the matching
   opt-in probe was exercised).
5. `validation.{md,json}` shows no `FAIL` rows (gates SC-9).
6. `report.pdf` renders (gates SC-5).

---

## 15. Implementation Priorities (Execution Order)

Run in this order. Each step's output anchors the next; skipping or reordering breaks the chain that connects raw measurements to MFU.

1. **Verify environment and device state** (§2.3) — clocks, power cap, thermals, env capture.
2. **BF16 compute ceiling** (§5) — establishes compute roof for §9, denominator for §11 MFU.
3. **HBM bandwidth ceiling** (§6) — establishes bandwidth roof for §9 and `t_memory_theoretical` for §10.
4. **DRAM capacity** (§7) — confirms the workload and timing buffers fit; informs e2e batch sizing.
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
| 5.2.2 | rectangular GEMM at projection / FFN shapes (fused QKV/KV variants) | `bench01.rectangular_sweep` | same |
| 5.2.2b | per-component GEMM throughput (one row per GEMM in the per-block decomposition; names line up 1:1 with `bench04`'s op table) | `bench01.component_gemm_sweep` driven by `flop_accounting.gemm_inventory` | `01_bf16_compute/component_gemms.{json,csv}` |
| 5.2.3 | batched GEMM and addmm | `bench01.addmm_and_bmm` | same |
| 5.2.4 | optional fused matmul+bias+activation | **not implemented** — backend-dependent; document availability in `env.json` and add ad-hoc when stack supports it |
| 5.3 | TFLOP/s, eff vs measured, eff vs rated | `bench01` `_row` + `summary.json` |
| 6.2 | copy_, add, mul, axpy, sum, fill_, strided | `bench02` BW-1…BW-7 | `02_hbm_bandwidth/bandwidth.{csv,json}` |
| 6.4 | sustained GB/s, variance, plateau | `bench02` + `summary.json.plateau_gb_s_per_op` | same |
| 7.2 | binary search bf16 + fp16 | `bench03.binary_search_max_contig` | `03_dram_capacity/summary.json` |
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
| 12.2 (TP-3) | strong scaling at world ∈ {2,4,8} | `scripts/strong_scaling.sh` drives `run_campaign.sh` once per world; `scripts/strong_scaling_table.py` rolls the per-world artifacts into the TP-3 table | `<sweep>/world_{2,4,8}/` + `<sweep>/strong_scaling.{md,json}` |
| 12 fused TP linears | fused AG+MM / MM+RS kernels | `bench06_aiter_fused` (AITER side) + `bench10_symm_fused` (torch SymmMem side, with a runtime correctness gate against the fallback helpers). The AITER-side bench's dispatcher tries upstream AITER first, then `aiter.ops.triton.comms.fused.*`, then the vendored `benchmarks.aiter_kernels` (always available on a CUDA host with triton). Both benches emit `api_source`, `call_kind`, fused-vs-sequential ratio (gates SC-12). Detailed kernel design in `benchmarks/aiter_kernels/README.md`; user-facing usage / tuning / troubleshooting in `docs/AITER_FUSED_KERNELS.md`. | `06_multigpu_fused/fused.{json,csv}` and `10_symm_fused/fused.{json,csv}` |
| sustained / thermal | head→tail drift, σ growth, clock drop on a long run | `bench07_sustained` runs the workload for `--duration` seconds with paired SMI poller (gates SC-7) | `07_sustained/sustained.json` + `telemetry.json` |
| topology | all-pairs D2D / inter-CCD / inter-socket BW matrix | `bench08_topology_bw` (works on GPU and CPU; emits per-pair fabric label) | `08_topology/topology.json` |
| numerical envelope | per-(dtype, K) GEMM error distribution vs FP32 | `bench09_numerical_stability` extends `bench01.correctness_check` from a binary gate into a full sweep (gates SC-10) | `09_stability/stability.{csv,json}` |
| inter-run variance | cross-run CV % on the headline metric | `scripts/across_run_variability.py` walks N campaign dirs, computes CV % per metric (gates SC-8) | `<root>/across_run_variability.{md,json}` |
| 13 | A1…A18 artifacts | `bench0x` + `scripts/plot_results.py` + `scripts/report.py` + `summary.md` | `results/<id>/` |
| 1.2 SC-1…SC-12 | pass/fail scorecard (gating SC-1…SC-6, opt-in SC-7…SC-12) | `scripts/score_campaign.py` | `results/<id>/scorecard.{md,json}` |
| report config | target registry (AMD / NVIDIA / CPU rated specs), classification thresholds, status pills, glossary, project metadata | `configs/report_config.json` consumed by `scripts/report.py` (override via `--report-config`) | input config (no per-run artifact) |
| source-pilot reference | "the source PDF says X" numbers (peak TFLOPs, ICI ring BW, MFU targets 77 / 93 / 99 %) | `configs/escher_14b_480p.json` `source_pilot_reference` block; `bench05` reads it into `mfu.json`; `report.py` reads it into the executive summary + reference-vs-observed table | input config (no per-run artifact) |

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
| `compute` `bandwidth` `dram` `workload` `e2e` `multigpu` | the matching `bench0X` family |
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
no `--out`, it picks the most recent campaign directory under
`${RESULTS_DIR:-results}/` that contains `env.json` and whose name either
includes `-campaign-` (legacy) or matches `<model>-YYYYMMDD-HHMMSS` (mtime
sort), then forwards everything else verbatim. Falls back
to `<repo>/runs/` for backward compatibility with trees from before the
`runs/` → `results/` rename.

### 16.6 CPU-host (no accelerator) behavior

The campaign is fully runnable on a CPU-only host so the analytic and
post-processing layers can be exercised in CI without a GPU:

| Step | Behavior on CPU |
|------|-----------------|
| `setup.sh` | Installs CPU `torch + torchvision` from `https://download.pytorch.org/whl/cpu`; creates `.microbenchmarks-cpu-venv/`. |
| `bench01..05` | Run normally; each script auto-detects CPU and falls back to a CPU-tractable size grid. See §12.5 for `bench06`'s CPU multi-CCD / multi-socket path. |
| `bench06` | Auto-targets `WORLD = max(#CCDs, #sockets)` with `CPU_TOPOLOGY=auto` (CCD → socket → split). Override via `WORLD=N CPU_TOPOLOGY=ccd|socket|split|auto`. Single-CCD / single-socket VMs degrade to `split` automatically. |
| `bench04_workload_ops` | Runs the analytic FLOP/byte/AI table; `t_ms_default` / `t_ms_optimized` columns are NaN by design. |
| `bench06_aiter_fused` (legacy `bench06_fused`), `bench10_symm_fused`, `bench07_sustained`, `bench08_topology_bw`, `bench09_numerical_stability` | Run on CPU when invoked directly. `bench06_aiter_fused` reports `SKIP reason=no AITER candidate modules resolved` because the vendored `benchmarks.aiter_kernels` triton path requires a CUDA host (the dispatcher's CUDA gate refuses to launch the kernels without `torch.cuda.is_available()`); the pure-Torch fallback is intentionally not selected by the dispatcher in benchmark mode so the SC-12 row stays honest. `bench10_symm_fused` reports `SKIP reason=SymmMem unavailable` for the same CPU-not-CUDA reason (`torch.ops.symm_mem.fused_*` raises `NotImplementedError` on CPU; `_capabilities.SYMM_MEM_AVAILABLE` checks `torch.cuda.is_available()` precisely so we don't fail loudly on host-only nodes). `bench07_sustained` substitutes a CPU FFN forward for the DiT. `bench08_topology_bw` produces a CCD/socket BW matrix. `bench09` runs the dtype × K sweep at CPU-tractable sizes. |
| External validators (RVS / `rocm-bandwidth-test` / `rccl-tests`) | Skipped. |
| `compare.py` | PASS=0 FAIL=0 SKIP=2 (BF16 vs RVS, BW vs rocm-bw both SKIP, no FAILs). |
| `score_campaign.py` | Inserts a `HOST CPU` row at the top. SC-1 drops the absolute rated-peak target (still enforces `best ≥ 0.9 × measured`). SC-2 widens the plateau spread to ±15 %. SC-3 (op taxonomy) graded normally. SC-4 → `SKIP` when MFU rows incomplete. SC-5 graded normally (the report PDF must still exist). SC-6 (BF16 vs FP32 correctness) graded normally. SC-7..SC-9 → `SKIP` until the matching probe runs. SC-10 graded if `bench09` ran. SC-11 emits `WARN_CPU` rather than `FAIL` when the DiT does not fit in host RAM. SC-12 `SKIP` on CPU. |
| `plot_results.py`, `report.py` | Run end-to-end; plots show only the per-op theory chart (A7) since GPU artifacts are absent. The report retitles itself "CPU host campaign report" and pulls the device profile from `configs/report_config.json`'s registry — no MI355X-specific prose appears. |
| `run_campaign.sh` exit code | `0` on a CPU host (GPU steps intentionally skipped) so CI smoke gates pass. |

This guarantees that doc-pipeline regressions, scorecard plumbing,
calibration drift logic, and per-op accounting are caught in CI even when
the runner has no accelerator.

### 16.7 What the campaign WILL NOT collect by default

These are explicitly out of scope for the *default* `-t campaign` flow.
Some now have dedicated opt-in probes (called out below) — invoke them
directly when the matching SC needs to graduate from `SKIP` to `PASS`.

- **Sustained / long-window stability**. Out of the default flow because
  it adds 30+ minutes per run. Opt-in probe:
  `python -m benchmarks.bench07_sustained --duration <sec>` (gates SC-7).
- **Power draw, thermal trajectory beyond the start-of-run snapshot in
  `env.json`**. The opt-in `bench07_sustained` run pairs each window with
  `telemetry.json` (power / temp / clocks per polling interval); the
  per-job `run.sh --stat` poller writes a `*.stat.log` sidecar per job.
- **VAE encoder / decoder performance**. PDF scope: "Only transformer
  stack optimized, VAE untouched." Out of scope for this campaign.
- **Multi-node scaling**. The RCCL / NCCL path here is intra-node
  Infinity Fabric / NVLink only.
- **Fused AG+MM and MM+RS kernel evaluation**. Now collected by two
  opt-in probes that together gate SC-12:
  - `bench06_aiter_fused` (legacy entry: `bench06_fused`) — AITER side.
    Dispatcher tries upstream AITER → `aiter.ops.triton.comms.fused.*` →
    vendored `benchmarks.aiter_kernels` (which is itself an
    AITER-conformant Triton kernel set with per-arch JSON tile configs
    in `benchmarks/aiter_kernels/configs/gfx{950,942}-*.json`, an
    Iris-aware fast path, and a staged fallback). On a CUDA host with
    triton the vendored backend always resolves, so SC-12 grades to
    PASS / FAIL rather than SKIP. See `docs/AITER_FUSED_KERNELS.md` for
    user-facing usage / tuning / troubleshooting and
    `benchmarks/aiter_kernels/README.md` for the kernel design + the
    upstream-to-`aiter.ops.triton.comms.fused/` path.
  - `bench10_symm_fused` — torch SymmMem side. Same shape sweep against
    `torch.ops.symm_mem.fused_*` after a runtime correctness gate vs the
    SymmMem fallback helpers. Records `SKIP` only when SymmMem isn't
    registered (e.g. CPU host).
- **Alternative TP topology (A2A in place of AG+RS)**. Now collected by
  the opt-in `bench08_topology_bw` probe — it doesn't substitute `A2A`
  for the algorithm under test, but it produces the all-pairs BW matrix
  that an A2A-style topology decision needs.
- **Numerical-precision envelope across dtypes**. Default `bench01`
  correctness check is a binary gate at K = 256 / 1024. The opt-in
  `bench09_numerical_stability` runs the full per-(dtype, K) error
  distribution (gates SC-10).
- **Cross-run / inter-process variance**. A single campaign captures
  intra-run σ but not the cold-cache / scheduler-state delta between
  runs. Opt-in: collect three or more campaigns, then run
  `scripts/across_run_variability.py` (gates SC-8).

### 16.8 Data-driven report config (`configs/report_config.json`)

Every tunable that the report itself enforces lives in
`configs/report_config.json`. `scripts/report.py` loads this file once at
startup, caches it on the module, and resolves every threshold / device
profile / glossary entry through that cache. The file is **required** —
if it goes missing the renderer fails fast with a pointer to the default
path rather than silently swapping in a hardcoded fallback. Override the
default with `python scripts/report.py --report-config <path>`; this
makes campaign-specific overrides (e.g. tightened thresholds for a
sign-off run) a one-flag affair.

The schema is intentionally narrow:

| Key | Purpose | Consumers |
|-----|---------|-----------|
| `project.name`, `project.default_workload_label`, `project.unknown_device_fallback_short` | cover-page, header, fallback labelling | `section_cover_page` (incl. optional Hugging Face README embed), header banner |
| `target_registry[]` | one entry per supported device family. Each entry carries `pattern` (regex matched against `env.json.hardware.gpu_model`), `short` / `name_template` (display name), `vendor` (`amd` / `nvidia`), `rated_bf16_low`, `rated_bf16_high`, `rated_bw_gb_s`, `rated_mem_gib`. Add a new GPU here and the report retitles itself automatically. | `_target_profile`, every reference-vs-rated table |
| `thresholds.meas_over_theory_at_hw_limit` (default 1.10) and `meas_over_theory_tunable` (default 1.50) | bucket per-op `measured / theory` ratio into "at hardware limit" / "tunable" / "kernel issue" | `section_per_op_default_vs_optimized`, recommendations |
| `thresholds.calibration_drift_pct` (default 5.0) | per-block GFLOP / MB drift gate before the report flags shape-config drift | `section_workload_roofline`, recommendations |
| `thresholds.mfu_pdf_tolerance_pp` (default 5.0) | ±pp tolerance for MFU comparison vs the workload's `pdf_reference_targets_pct`; also the `likely cause` classifier in `section_reference_vs_observed` | `section_mfu`, `section_reference_vs_observed` |
| `thresholds.gemm_sweep_peak_fraction_high` (0.9) / `gemm_sweep_peak_fraction_low` (0.5) | bucket the per-shape GEMM TFLOP/s sweep into "near peak" / "ramp" / "launch-bound" | `section_relevant_shapes` |
| `thresholds.fragmentation_warning_ratio` (0.95) | flag fragmentation when contiguous max < `ratio · non-contiguous max` | `section_dram` |
| `thresholds.topology_decisive_advantage_pct` (5.0) | declare an A2A vs AG+RS topology verdict only when the faster path beats the slower one by more than this margin | `section_multigpu` |
| `thresholds.rated_peak_overshoot_message_at` (1.0) | trigger the "audit your FLOP accounting" callout when measured TFLOP/s ≥ this fraction of rated peak | `section_relevant_shapes` |
| `pdf_reference_targets_pct` | generic fallback MFU targets used only when a workload config has no `source_pilot_reference.pdf_reference_targets_pct` of its own | `section_mfu` |
| `status_pills` | CSS class per scorecard status (`PASS`, `FAIL`, `WARN`, `WARN_CPU`, `SKIP`, `PARTIAL_PASS`, `$default`) | `_status_pill` |
| `glossary[]` | acronym list rendered into the appendix; one `{term, definition}` per entry | `section_glossary` |

The report-side classifier uses these values *exclusively* — there are no
hardcoded numeric literals left in `scripts/report.py`. Re-tune any
threshold, the JSON, and the next render picks it up; nothing else has to
change.

### 16.9 Source-pilot reference values (`configs/<workload>.json`)

Workload-specific reference numbers — the ones a reader will paraphrase as
"the source PDF says X" — live in the workload config's
`source_pilot_reference` block, **not** in `report_config.json`. The block
exists once per workload and is the single source for:

| Field | Used by | Where it surfaces |
|-------|---------|-------------------|
| `peak_tflops` | `report.py` reference-vs-observed | "Reproduction vs Source PDF" delta column |
| `ici_ring_gb_s` | `report.py` multi-GPU section | ICI / NVLink ring BW reference |
| `memory_bw_to_ici_ratio` | `report.py` reference-vs-observed | "ratio of memory BW to ICI ring BW" line |
| `pdf_reference_targets_pct.{sum_of_ops_optimized, sum_of_ops_default, eager_e2e, compiled_e2e}` | `bench05_e2e_mfu` (writes them into `mfu.json`) and `report.py` (renders them as the dashed reference line on the MFU chart) | MFU table headline column + reference line on A8 |

`bench05_e2e_mfu` reads `pdf_reference_targets_pct` from the workload
config when present and falls back to the legacy hardcoded 77 / 93 / 99
defaults only for backwards compatibility with older configs. Any new
workload config should declare its own `source_pilot_reference` block so
the report's "the source PDF says X" lines are auditable end-to-end.

### 16.10 Report generator — Hugging Face model card & `REFERENCE_MODEL`

`scripts/report.py` builds the campaign narrative from JSON artifacts plus
optional **Hugging Face** context (no Physics-leaderboard table in the report).

**Resolving a Hub repo id** (first match wins):

1. Workload JSON: `huggingface_id` or `huggingface_model_id`.
2. `results/<id>/campaign_meta.json` → `reference_model` → row in
   `configs/reference_video_models.json` → `huggingface_id`.
3. Workload JSON `name` equal to a registry row `id` with `huggingface_id`.

**0. Cover** — Adds a **Hugging Face** row (markdown link / HTML anchor).
When `README.md` can be fetched from
`https://huggingface.co/<repo>/raw/{main,master}/README.md`, the README is
embedded on the cover (size-capped). Offline builds keep the link only.

**Model Description** (renamed from *Workload Description*) — When a README
is available, an **Architecture from Hugging Face model card** subsection
embeds a digest: top-level `##` sections from the README with license /
citation-style headings removed, then **Benchmark configuration** documents
the instrumented `bench04` / `bench05` parameters from the workload JSON.

**`REFERENCE_MODEL` in `run.sh`** — For `testcase=campaign`, when the
registry row defines `workload_config`, the campaign benches use that JSON
path (and `test.sh` aligns `--config` / `out_id` with the same resolution).
`campaign_meta.json` records `reference_model` for the report resolver only.

**CLI** — `python scripts/report.py --list-reference-models` prints registry
ids (and optional `--reference-models-config` for an alternate JSON).

### 16.11 AITER comm+compute kernels (`benchmarks/aiter_kernels/`)

`benchmarks/aiter_kernels/` is a self-contained, AITER-conformant
implementation of two new fused collective+GEMM kernels:

* `fused_all_gather_matmul` — column-parallel TP linear (AG along M
  followed by `A_full @ B`).
* `fused_matmul_reduce_scatter` — row-parallel TP linear (`A @ B`
  followed by reduce-scatter along M).

Both ops match the `torch.ops.symm_mem.fused_*` signatures, so any
TP-linear call-site that today uses `torch.distributed._symmetric_memory`
can swap in the new kernels without changes. The package mirrors the
upstream AITER directory layout (`aiter/ops/triton/comms/fused/`,
`aiter/ops/triton/_triton_kernels/`, `aiter/ops/triton/configs/`,
`op_tests/triton_tests/`) so each file translates one-to-one when
upstreamed; see `benchmarks/aiter_kernels/README.md` for the design
review and the exact upstreaming map.

**Backend dispatcher** (`benchmarks.aiter_kernels.dispatcher`):

| Order | Backend label                       | When picked                                                  |
|-------|-------------------------------------|--------------------------------------------------------------|
| 1     | `aiter.upstream`                    | `aiter.fused_all_gather_matmul` resolves (post-upstream)     |
| 2     | `aiter.ops.triton.comms.fused`      | Canonical AITER path resolves (kernels merged into AITER)    |
| 3     | `benchmarks.aiter_kernels.triton`   | Vendored kernels available (triton + CUDA host)              |
| 4     | `torch.ops.symm_mem`                | Torch-native SymmMem available on the current device         |
| 5     | `fallback.pure_torch`               | Always available; correctness gold for the op-tests          |

Pin a backend with `AITER_KERNELS_BACKEND={aiter, aiter_triton_comms,
local_triton, symm_mem, fallback}` for A/B comparison runs.

**Triton kernel template** mirrors `aiter/ops/triton/comms/{all_gather,
reduce_scatter}.py`:

* Two-layer JIT split (`_xxx_impl` device function + `_xxx_kernel` entry).
* Persistent grid (`grid = (NUM_SMS,)`) with `GROUP_SIZE_M` swizzle.
* `fp32` MFMA accumulator; `num_warps=16, num_stages=3-4, waves_per_eu=4`.
* Iris-aware path (`iris.load` / `iris.atomic_add`) with a staged-AG /
  staged-RS fallback when Iris isn't available.
* Per-arch JSON tile configs in `configs/{gfx950, gfx942}-FUSED-*.json`
  loaded by `benchmarks/aiter_kernels/_config_loader.py` using the same
  `M_LEQ_<N>` / `M_GEQ_<N>` / `any` selection rules as AITER's
  `aiter/ops/triton/utils/gemm_config_utils.py`. Per-key env overrides
  via `AITER_KERNELS_FUSED_{AG_MM,MM_RS}_{BLOCK_M,BLOCK_N,...}`.

**Wiring**:

* `bench06_aiter_fused.py` probes upstream AITER first, then the
  canonical AITER-comms-fused path, and finally the vendored
  `benchmarks.aiter_kernels` module — so a campaign report on
  MI300X / MI355X without an upstream AITER fused kernel still
  produces real fused-AG+MM / MM+RS numbers and can compare them
  against `bench10_symm_fused.py` (torch-native SymmMem).
* SC-12 in §1.2 grades the speedup ratio of the fused path against
  the unfused AG-then-MM and MM-then-RS reference; with the vendored
  kernels in place it never reports `SKIP reason=fused API not available`
  on a CUDA host that has triton.

**Op-tests** (mirrors `op_tests/triton_tests/comms/`):

```
torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective
```

Asserts `torch.testing.assert_close(out, fallback_out, rtol=1e-2,
atol=1e-2)` against the pure-Torch reference for every available
backend. A backend that's not runnable on the current device (e.g.
`torch.ops.symm_mem` on a CPU host) is treated as SKIP, not FAIL.
