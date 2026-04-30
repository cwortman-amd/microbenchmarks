# microbenchmarks — `escher_14b_480p`

Reproduces the Odyssey AMD Inference Pilot (April 2026) benchmark methodology
end-to-end and cross-validates the PyTorch numbers against **ROCm Validation
Suite (RVS)**, **`rocm-bandwidth-test`**, and **rccl-tests**. AMD Instinct
MI355X is the *reference* target — every threshold and ceiling cited in the
PDF is captured against it — but the same toolchain runs unmodified on:

- **AMD Instinct** MI355X / MI325X / MI300X / MI250X
- **AMD Radeon** R9700
- **NVIDIA** B200 / H200 / H100 (SXM, PCIe) / L40S / A100 (80 / 40 GB) / V100
- **CPU host** (Linux, AMD or Intel) for CI / dev-laptop validation runs

`setup.sh` auto-detects the device, picks the matching PyTorch wheel
(ROCm / CUDA / CPU), and creates a per-device venv. The report renderer reads
the device profile out of `configs/report_config.json` so headlines, deltas,
and prose retitle themselves automatically — no MI355X-specific text appears
when the run lands on a different target.

The full methodology lives in [`TESTPLAN.md`](docs/TESTPLAN.md). The data-driven
report knobs (target registry, classification thresholds, glossary) live in
[`configs/report_config.json`](configs/report_config.json).

**Wan2.2 (720p / real video DiT):** this repo does not vendor upstream Wan
weights or `generate.py`, but you can run **[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)**
alongside the same hardware benchmark (ceilings + comm) and optionally add a
workload JSON that approximates token geometry. See
[`docs/WAN2.2.md`](docs/WAN2.2.md) for install links, Hugging Face checkpoints,
`--size 1280*720` / `--convert_model_dtype`, and how `bench04`/`bench05` relate
to the real Wan graph.

## Layout

```
setup.sh                  one-shot env provisioner (CPU/CUDA/ROCm; auto-detects)
test.sh                   top-level test runner (registry of testcase × workload)
run.sh                    per-job runner; system probe, profiling, iterations, parsing
report.py                 thin shortcut → scripts/report.py (auto-picks latest benchmark)
benchmarks/
  common/                 timing, env capture, stats, IO, FLOP/byte accounting,
                          telemetry (SMI poller), topology (CCD/socket/NUMA helpers)
  bench01_bf16_compute.py        Family 1 — BF16 GEMM peak, size sweep, per-component GEMMs, BF16-vs-FP32 correctness
  bench02_hbm_bandwidth.py       Family 2 — HBM / DDR streaming microbenchmarks
  bench03_dram_capacity.py       Family 3 — practical allocatable memory (HBM on GPU, DDR on CPU); --measure-headroom for post-model residual capacity
  bench04_workload_ops.py        Family 4 — escher_14b_480p per-op decomposition (theory vs default vs optimized)
  bench05_e2e_mfu.py             Family 5 — eager vs torch.compile end-to-end MFU + per-chunk stability
  bench12_multigpu_comm.py       Family 6 — all-gather / reduce-scatter / all-reduce (torchrun)
  bench06_fused.py               Family 6 — compatibility shim to bench06_aiter_fused.py
  bench06_aiter_fused.py         Family 6 — AITER fused AG+MM / MM+RS availability + speedup probe
                                 (probes upstream aiter.* first, then aiter.ops.triton.comms.fused.*,
                                 finally falls through to the vendored AITER-conformant kernels in
                                 benchmarks/aiter_kernels/)
  aiter_kernels/                 New AITER-style fused collective+GEMM kernels (AG+MM and MM+RS) —
                                 SymmMem-compatible API, Iris-aware Triton kernels (with staged
                                 fallback), per-arch JSON tile configs (gfx950/gfx942), pure-Torch
                                 reference, op-tests. See docs/AITER_FUSED_KERNELS.md for user-
                                 facing usage / tuning / troubleshooting and benchmarks/aiter_kernels/
                                 README.md for the kernel-design review + upstream-to-aiter/ops/triton/
                                 comms/fused/ path. Dispatcher selection:
                                 aiter.upstream → aiter.ops.triton.comms.fused →
                                 benchmarks.aiter_kernels.triton → torch.ops.symm_mem → pure-Torch fallback.
  bench10_symm_fused.py          Family 10 — torch SymmMem fused AG+MM / MM+RS probe + correctness gate
  bench07_sustained.py           Family 7 — sustained-throughput / thermal-drift probe with paired SMI telemetry
  bench08_topology_bw.py         Family 8 — all-pairs device-to-device (or inter-CCD/inter-socket) bandwidth matrix
  bench09_numerical_stability.py Family 9 — full per-(dtype, K) GEMM error distribution vs FP32 reference
configs/
  escher_14b_480p.json    workload shape spec + source-pilot reference numbers
                          (peak TFLOPs, ICI BW, MFU targets) — single source of truth for "the PDF says X"
  report_config.json      data-driven report knobs: target registry (AMD/NVIDIA/CPU specs),
                          classification thresholds, status-pill mapping, glossary, project metadata
  reference_video_models.json  optional registry for video foundation models (ids,
                          optional ``huggingface_id``, optional ``workload_config``).
                          For ``testcase=benchmark``, ``REFERENCE_MODEL=<id>`` in ``run.sh``
                          selects ``workload_config`` when present, writes
                          ``results/<out>/benchmark_meta.json`` (``reference_model``, metadata),
                          and lets ``scripts/report.py`` resolve the Hub repo for the cover +
                          **Model Description** section. List ids: ``python scripts/report.py
                          --list-reference-models`` (optional ``--reference-models-config``).
validation/
  rvs/                    ROCm Validation Suite configs (gst, mem, pebb) + runner
  rocm_bw/                rocm-bandwidth-test runner (HBM ground truth)
  rccl/                   rccl-tests runner + parser (RCCL ground truth)
  compare.py              PyTorch vs RVS vs rocm-bandwidth-test vs rccl-tests
scripts/
  run_benchmark.sh             orchestrator (TESTPLAN §15 sequence)
  strong_scaling.sh           sweep WORLD ∈ {2,4,8} -> TP-3 strong-scaling table (§16.3)
  strong_scaling_table.py     aggregator: per-world busbw / efficiency / MFU table
  across_run_variability.py   inter-run CV%% on the headline metrics (feeds SC-8)
  plot_results.py             charts A2/A3/A6/A7/A8 from TESTPLAN §13
  score_benchmark.py           SC-1…SC-12 pass/fail scorecard
  report.py                   data-driven report.md / report.html / report.pdf (cover +
                              Model Description may embed Hugging Face README when resolvable)
results/                  per-benchmark output directories (override via $RESULTS_DIR)
```

## Quick start

```bash
# 1. One-time setup: auto-detects rocm / cuda / cpu, creates a venv at
#    .microbenchmarks-<device>-venv/, installs the matching PyTorch wheel
#    (and torchvision), probes optional backends (AITER, flash_attn, triton)
#    on GPU hosts, and reports calibration drift against TESTPLAN §1.4.
./setup.sh
source .microbenchmarks-rocm-venv/bin/activate     # or -cuda-venv / -cpu-venv

# 2. Run a full single-node 8-GPU benchmark
./test.sh -t benchmark

# 3. Inspect the latest benchmark (dirs are <model>-YYYYMMDD-HHMMSS; legacy
#    runs may still be *-benchmark-*)
LATEST=$(ls -1td results/*/ 2>/dev/null | while read -r d; do
  [[ -f "${d}env.json" ]] || continue
  b=$(basename "$d")
  [[ "$b" == *-benchmark-* || "$b" =~ -[0-9]{8}-[0-9]{6}$ ]] && echo "$d"
done | head -1)
cat $LATEST/summary.md
cat $LATEST/scorecard.md
cat $LATEST/validation.md

# 4. Open the auto-generated report (data-driven, mirrors the source PDF).
#    `./report.py` with no args picks the most recent full benchmark directory
#    and emits report.md, report.html and report.pdf side-by-side.
./report.py
xdg-open $LATEST/report.pdf
```

The report is regenerable on demand from the JSON outputs alone:

```bash
./report.py                                          # auto-pick latest benchmark dir; emits md+html+pdf
./report.py --out results/<id>/                      # explicit dir
./report.py --out results/<id>/ --format md          # md only
./report.py --out results/<id>/ --format html        # html only
./report.py --out results/<id>/ --format pdf         # pdf only
./report.py --out results/<id>/ --no-pdf             # md+html, skip pdf
./report.py --out results/<id>/ --no-embed           # link plots/ instead of base64
./report.py --out results/<id>/ --target h100        # force the device profile
                                                     #   (mi355x | mi325x | mi300x | mi250x |
                                                     #    r9700 | b200 | h200 | h100 | l40s |
                                                     #    a100 | v100, or any registry pattern)
./report.py --out results/<id>/ --report-config configs/report_config.json
```

PDF generation prefers `wkhtmltopdf` (clean HTML→PDF), falls back to `pandoc`
with a LaTeX engine, and finally to a Markdown→PDF pandoc path. For full PDF 
formatting—including the `[AMD Confidential - Distribution Under NDA]` running
header and page-numbered footers—a patched Qt engine is required. Install the 
static `wkhtmltox` binary from the official releases rather than using the 
unpatched `apt install wkhtmltopdf`.

The generated `report.md` includes Obsidian-compliant YAML frontmatter (author, 
sensitivity, title) for easy knowledge base integration.

`configs/report_config.json` is the **single tunable surface** for the
report itself: the target-hardware registry (rated peaks, HBM size, vendor),
classification thresholds (e.g. `meas_over_theory_at_hw_limit = 1.10`,
`meas_over_theory_tunable = 1.50`, `mfu_pdf_tolerance_pp = 5.0`,
`calibration_drift_pct = 5.0`), status-pill CSS mapping, glossary, and
project metadata all live there. Re-tune any of those without touching code,
or pass `--report-config <path>` to swap in a benchmark-specific override.

**Hugging Face on the report:** `scripts/report.py` resolves a repo id from
the workload JSON (`huggingface_id` / `huggingface_model_id`), from
`benchmark_meta.json` + `reference_video_models.json`, or when the workload
`name` matches a registry `id`. **0. Cover** can embed **README.md** (model
card) and adds a Hugging Face table row. **Model Description** (instrumented
DiT parameters + shapes + op mix) may prepend an **Architecture from Hugging
Face model card** subsection: filtered `##` sections from that README (license
/ citation blocks skipped). Fetching the README needs outbound HTTPS when the
report is built; if it fails, the cover still links to the Hub.

Source-pilot reference values (peak TFLOPs, ICI ring BW, MFU targets
77 / 93 / 99 %) live next to the workload spec in the
`source_pilot_reference` block of `configs/escher_14b_480p.json`. `bench05`
reads MFU targets from there, and the report reads everything else from
there, so any line in the artifacts that says "the source PDF says X"
traces back to that one block.

`setup.sh` auto-detects the device (`rocm`/`cuda`/`cpu`) and creates a
correspondingly-named venv. To force a specific stack:

```bash
DEVICE=rocm TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 ./setup.sh
DEVICE=cuda ./setup.sh                               # default torch+torchvision channel
DEVICE=cpu  ./setup.sh                               # https://download.pytorch.org/whl/cpu
```

Each family can also be run individually (substitute your benchmark id):

```bash
# Core 1..6 — wired into test.sh / run.sh / run_benchmark.sh
python -m benchmarks.bench01_bf16_compute        --out results/$BENCHMARK_ID/
python -m benchmarks.bench02_hbm_bandwidth       --out results/$BENCHMARK_ID/
python -m benchmarks.bench03_dram_capacity       --out results/$BENCHMARK_ID/
python -m benchmarks.bench04_workload_ops        --out results/$BENCHMARK_ID/ --config configs/escher_14b_480p.json
python -m benchmarks.bench05_e2e_mfu             --out results/$BENCHMARK_ID/ --config configs/escher_14b_480p.json
torchrun --nproc_per_node=8 benchmarks/bench12_multigpu_comm.py --out results/$BENCHMARK_ID/

# Optional fused/sustained/topology/stability probes invoked directly when needed
torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py    --out results/$BENCHMARK_ID/   # AITER AG+MM / MM+RS availability + speedup
torchrun --nproc_per_node=8 benchmarks/bench10_symm_fused.py     --out results/$BENCHMARK_ID/   # torch SymmMem AG+MM / MM+RS probe + fallback correctness check

# Standalone correctness gate for the new AITER-style fused kernels (bench06's probe target):
torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective         # asserts every available backend (aiter / local triton / symm_mem) matches the pure-Torch reference
python -m benchmarks.bench07_sustained           --out results/$BENCHMARK_ID/ --duration 1800   # 30-min sustained throughput + telemetry
python -m benchmarks.bench08_topology_bw         --out results/$BENCHMARK_ID/                   # all-pairs D2D / inter-CCD BW matrix
python -m benchmarks.bench09_numerical_stability --out results/$BENCHMARK_ID/                   # per-(dtype, K) GEMM error distribution
```

### Unified test runner

`test.sh` is the GuideLLM-style top-level entrypoint that wraps the same
scripts with named testcases and workload variants. It does **not** run the
benchmarks itself — it enumerates `(testcase × workload)` combinations and
delegates each to `run.sh`, which is the per-job runner. The two-script split
mirrors the Partition project layout:

| Script  | Role |
|---------|------|
| `test.sh` | Registry of testcases + workloads, argv parser, batch driver. Calls `run.sh` once per `(testcase, workload)` combination. |
| `run.sh`  | Single-job runner. Probes the system (`amd-smi` partition mode, GPU count), materializes a derived config when shape/depth overrides are set, dispatches to the right `bench0X` / validator / report script, runs `ITERATIONS` iterations, parses each iteration's JSON outputs into a one-line `RESULT:` record, and at the end aggregates mean / median across iterations. |

```bash
./test.sh --list                       # registered testcases & workloads
./test.sh -t compute                   # only Family 1 (BF16 GEMM)
./test.sh -t e2e -w smoke              # quick MFU sanity run
./test.sh -t benchmark -i 3             # full benchmark 3× (regression averaging)
./test.sh -a                           # every testcase × every workload
```

`run.sh` can also be invoked directly when you already know the exact
parameters and don't want the test.sh registry layer:

```bash
./run.sh --testcase compute   --iterations 5
./run.sh --testcase e2e       --workload smoke      --iterations 3
./run.sh --testcase multigpu  --nproc 8             --iterations 3
./run.sh --testcase benchmark  --benchmark-id $(date +%Y%m%d-%H%M%S)
```

Each `run.sh` invocation tees its full console output to
`results/_logs/<testcase>-<workload>-px<P>-nps<M>-g<GPUS>-tp<NPROC>.<ts>.log` and
appends one `RESULT:` line per iteration to `results/<out-id>/iterations.results`.
When `--iterations > 1` the aggregate block at the end prints
`mean / median / values=(...)` for each headline metric of the chosen testcase
(e.g. `peak_tflops` for compute, `mfu_eager`/`mfu_compiled` for e2e,
`all_reduce_busbw_gb_s` for multigpu).

#### Profiling flags

`run.sh` ships with Phoronix-style profiling hooks that capture the host /
GPU state for every job. The behaviour mirrors what `turbostat` does for CPU
work, but uses the GPU vendor's SMI tool:

| Flag | What it does |
|------|--------------|
| `--prepare-sys` | Run kernel/VM tuning (`numa_balancing=0`, transparent hugepages always, `vm.compact_memory=1`, `drop_caches=3`, `rocm-smi --setperflevel high`). Requires `sudo`; failures are non-fatal so unprivileged hosts keep going. Off by default. |
| `--no-stat` / `STAT=off` | Disable the live telemetry poller. Default: on. |
| `--stat-interval N` | Poller interval in seconds. Default: `1`. |
| `--numactl "<spec>"` | Wrap the per-iteration command with `numactl <spec>` (only honored for non-distributed testcases; `torchrun` jobs handle their own affinity). Example: `--numactl "--physcpubind=0-15 --localalloc"`. |

The poller auto-selects in this order: `amd-smi metric --watch`, then a
`rocm-smi --showpower --showtemp --showuse --showmemuse --showclocks --csv`
loop, then `nvidia-smi --query-gpu=...`, then disabled. Output goes to a
sidecar file paired with the main log:

```
results/_logs/<testcase>-<workload>-px<P>-nps<M>-g<G>-tp<N>.<ts>.log         # console
results/_logs/<testcase>-<workload>-px<P>-nps<M>-g<G>-tp<N>.<ts>.stat.log    # GPU telemetry
```

A one-shot system fingerprint (`uname`, `lscpu`, `rocminfo`, `rocm-smi`,
`amd-smi static`, `/sys` knobs, relevant env vars, key pip packages) is
written to the main log at the start of every run, so a single `.log` is
enough to reproduce the hardware/software baseline of any reported number.

### CPU-host runs (CI / dev laptops)

The benchmark is fully runnable on a CPU-only host. Every `bench0X` script
detects the device internally and produces real measured numbers under the
same JSON schema as the GPU path, so report.py and score_benchmark.py can
diff CPU and GPU runs against each other without special-casing. On a CPU
host:

- `./setup.sh` installs `torch` + `torchvision` from the CPU wheel index
  and creates `.microbenchmarks-cpu-venv/`.
- `./test.sh -t benchmark` runs **the core six families** end-to-end:
  - `bench01..03` measure CPU BF16 GEMM peak, system DDR bandwidth, and
    `psutil`-bounded DRAM capacity.
  - `bench04` analytically accounts every op and selectively measures
    sub-`--cpu-budget-gflops` ops (default 5 GFLOP); heavy GEMMs / attention
    are intentionally `NaN` to keep the run tractable.
  - `bench05` auto-downscales `depth` / `seq_image` / `seq_text`, runs
    eager-only (no `torch.compile`), and suppresses sum-of-ops MFU when
    bench04's measurement coverage falls below 95%.
  - `bench06` is the **CPU multi-CCD / multi-socket** path: gloo over
    loopback with one rank per Linux `die_id` (Infinity Fabric boundary)
    and `os.sched_setaffinity` pinning. World size is auto-set to
    `max(#CCDs, #sockets)`. Override with `WORLD=N CPU_TOPOLOGY=ccd|socket|split|auto`
    or `./run.sh --testcase multigpu --cpu-topology socket`.
- The opt-in probes (`bench06_aiter_fused`, `bench10_symm_fused`, `bench07_sustained`,
  `bench08_topology_bw`, `bench09_numerical_stability`) also run on a CPU
  host: the fused-kernel probe always reports `not available` (which is
  itself the report-relevant data point), the sustained probe drives the
  CPU FFN forward instead of the DiT, and the topology probe falls back to
  a CCD/socket bandwidth matrix. Wire them in as needed; they are not part
  of the default `-t benchmark` flow.
- `scorecard.md` shows a `HOST CPU` row at the top. The CPU host treats
  GPU-bound criteria as `SKIP` rather than `FAIL`:
  - `SC-1` drops the absolute rated-peak TFLOP/s target (no GPU to compare
    against) but still enforces the `best_sweep ≥ 0.9 × measured peak`
    consistency check.
  - `SC-2` widens the bandwidth-spread threshold to 15 % to match DDR
    variance.
  - `SC-3` (op taxonomy) is graded normally.
  - `SC-4` SKIPs when bench05's MFU rows are incomplete (no
    `torch.compile` on CPU).
  - `SC-5` (artifact set) is graded normally — the `*.pdf` and `*.html`
    are required outputs even on CPU.
  - `SC-6` (BF16-vs-FP32 numerical correctness) is graded normally.
  - `SC-7..SC-9` are opt-in: `SKIP` until the matching probe runs.
  - `SC-10` (numerical-stability sweep) graded if bench09 ran.
  - `SC-11` (post-model-load residual capacity) emits `WARN_CPU` when the
    DiT does not fit in host RAM — informational, not a failure.
  - `SC-12` (fused AG+MM / MM+RS) is `SKIP` on CPU. On GPU it grades to
    `PASS` / `FAIL` rather than `SKIP`: the dispatcher in
    `benchmarks/aiter_kernels/` always resolves a vendored Triton
    backend on a CUDA host with triton, so the speedup ratio over the
    sequential reference can always be measured. See
    `docs/AITER_FUSED_KERNELS.md`.
- The report retitles itself "CPU host benchmark report", swaps
  "HBM bandwidth" → "Memory bandwidth", compares DRAM against host RAM,
  and renames the multi-GPU section to "Multi-CCD / Multi-Socket
  Communication" with a topology-pinning table. No MI355X-specific prose
  appears unless the run actually lands on an MI355X — the device profile
  is looked up dynamically from `configs/report_config.json`'s target
  registry.

The benchmark's exit code is 0 on a CPU host: the per-SC details are
recorded in `scorecard.{md,json}` for inspection, but absolute thresholds
against the GPU target are not enforced as pass/fail.

### Result location

Per-benchmark artifacts land under `results/<benchmark-id>/` and the
per-iteration logs under `results/_logs/`. Override the root with
`$RESULTS_DIR`:

```bash
RESULTS_DIR=/data/odyssey-mi355x ./test.sh -t benchmark
./report.py                                  # honors $RESULTS_DIR
```

`./report.py` falls back to `<repo>/runs/` when `<repo>/results/` is
absent, so existing working trees from before the rename keep working
without manual migration.

## Validation tools

External validators are shelled out, not vendored. Install separately:

| Tool | Provides | Install |
|------|----------|---------|
| ROCm Validation Suite (`rvs`) | BF16 peak (`gst`), memory integrity (`mem`), PCIe BW (`pebb`) | shipped with ROCm |
| `rocm-bandwidth-test` | HBM D2D / H2D / D2H bandwidth ground truth | shipped with ROCm |
| `rccl-tests` | RCCL all-reduce / all-gather / reduce-scatter ground truth | <https://github.com/ROCm/rccl-tests> |

Cross-validation is run by `validation/compare.py`, which loads the JSON outputs from
each family and from each external tool, then writes `results/<id>/validation.md`
flagging any disagreement larger than the per-metric tolerance.

## Pass / fail

A benchmark signs off when **all gating SCs (SC-1 … SC-6)** in
[`TESTPLAN.md §1.2`](docs/TESTPLAN.md) hold AND the cross-validation report
shows no `FAIL` rows. SC-7 … SC-12 are opt-in / informational:

| SC | Role | When it must PASS |
|----|------|-------------------|
| SC-1 | BF16 compute peak vs measured + rated | gating |
| SC-2 | HBM / memory plateau stability | gating |
| SC-3 | Roofline placement (compute vs memory bound) | gating |
| SC-4 | MFU ordering: compiled ≥ eager ≥ sum-of-ops (±5 pp) | gating |
| SC-5 | All required artifacts (incl. `report.pdf`) exist | gating |
| SC-6 | BF16 vs FP32 numerical correctness | gating |
| SC-7 | Sustained throughput stable over `bench07` window | opt-in (skips if not run) |
| SC-8 | Cross-run CV %% on headline metrics ≤ threshold | opt-in (needs `across_run_variability.py`) |
| SC-9 | RVS / `rocm-bandwidth-test` / `rccl-tests` agreement | opt-in (skips if external tools absent) |
| SC-10 | `bench09` per-(dtype, K) error distribution within bound | opt-in |
| SC-11 | `bench03 --measure-headroom` post-model-load residual | opt-in (`WARN_CPU` on host) |
| SC-12 | Fused AG+MM / MM+RS kernels available + faster than unfused — sourced from upstream AITER, the canonical `aiter.ops.triton.comms.fused.*` path, the vendored `benchmarks.aiter_kernels` Triton backend, or `torch.ops.symm_mem` (in priority order) | opt-in — `SKIP` only on CPU and on hosts with no triton + no SymmMem; otherwise grades `PASS` / `FAIL` |

The orchestrator exits non-zero only when a gating SC fails or
`compare.py` flags a `FAIL`, so this can run in CI as a regression gate
without forcing the opt-in probes to be wired in everywhere.
