# microbenchmarks — `escher_14b_480p` on MI355X

Reproduces the Odyssey AMD Inference Pilot (April 2026) benchmark methodology on the
AMD Instinct MI355X platform, and cross-validates the PyTorch results against
**ROCm Validation Suite (RVS)**, **`rocm-bandwidth-test`**, and **rccl-tests**.

The full methodology lives in [`TESTPLAN.md`](TESTPLAN.md).

## Layout

```
setup.sh                  one-shot env provisioner (CPU/CUDA/ROCm; auto-detects)
test.sh                   top-level test runner (registry of testcase × workload)
run.sh                    per-job runner; system probe, profiling, iterations, parsing
report.py                 thin shortcut → scripts/report.py (auto-picks latest campaign)
benchmarks/
  common/                 timing, env capture, stats, IO, FLOP/byte accounting
  bench01_bf16_compute.py    Family 1 — BF16 GEMM peak + size sweep
  bench02_hbm_bandwidth.py   Family 2 — HBM streaming microbenchmarks
  bench03_vram_capacity.py   Family 3 — practical allocatable memory
  bench04_workload_ops.py    Family 4 — escher_14b_480p per-op decomposition
  bench05_e2e_mfu.py         Family 5 — eager vs torch.compile end-to-end MFU
  bench06_multigpu_comm.py   Family 6 — all-gather / reduce-scatter (torch.distributed)
configs/
  escher_14b_480p.json    workload shape spec (parameterizable)
validation/
  rvs/                    ROCm Validation Suite configs (gst, mem, pebb) + runner
  rocm_bw/                rocm-bandwidth-test runner (HBM ground truth)
  rccl/                   rccl-tests runner + parser (RCCL ground truth)
  compare.py              PyTorch vs RVS vs rocm-bandwidth-test vs rccl-tests
scripts/
  run_campaign.sh         orchestrator (TESTPLAN §15 sequence)
  plot_results.py         charts A2/A3/A6/A7/A8 from TESTPLAN §13
  score_campaign.py       SC-1…SC-5 pass/fail scorecard
  report.py               data-driven report.md + report.html generator
results/                  per-campaign output directories (override via $RESULTS_DIR)
```

## Quick start

```bash
# 1. One-time setup: auto-detects rocm / cuda / cpu, creates a venv at
#    .microbenchmarks-<device>-venv/, installs the matching PyTorch wheel
#    (and torchvision), probes optional backends (AITER, flash_attn, triton)
#    on GPU hosts, and reports calibration drift against TESTPLAN §1.4.
./setup.sh
source .microbenchmarks-rocm-venv/bin/activate     # or -cuda-venv / -cpu-venv

# 2. Run a full single-node 8-GPU campaign
./test.sh -t campaign

# 3. Inspect the latest campaign
LATEST=$(ls -1dt results/*-campaign-* | head -1)
cat $LATEST/summary.md
cat $LATEST/scorecard.md
cat $LATEST/validation.md

# 4. Open the auto-generated report (data-driven, mirrors the source PDF).
#    `./report.py` with no args picks the most recent results/*-campaign-*.
./report.py
xdg-open $LATEST/report.html

# or convert to PDF / PPTX with pandoc
pandoc $LATEST/report.md -o $LATEST/report.pdf
pandoc $LATEST/report.md -o $LATEST/report.pptx
```

The report is regenerable on demand from the JSON outputs alone:

```bash
./report.py                                          # auto-pick latest results/*-campaign-*
./report.py --out results/<id>/                      # explicit dir, both formats
./report.py --out results/<id>/ --format md          # md only
./report.py --out results/<id>/ --format html        # html only
./report.py --out results/<id>/ --no-embed           # link plots/ instead of base64
```

`setup.sh` auto-detects the device (`rocm`/`cuda`/`cpu`) and creates a
correspondingly-named venv. To force a specific stack:

```bash
DEVICE=rocm TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 ./setup.sh
DEVICE=cuda ./setup.sh                               # default torch+torchvision channel
DEVICE=cpu  ./setup.sh                               # https://download.pytorch.org/whl/cpu
```

Each family can also be run individually (substitute your campaign id):

```bash
python -m benchmarks.bench01_bf16_compute  --out results/$CAMPAIGN_ID/
python -m benchmarks.bench02_hbm_bandwidth --out results/$CAMPAIGN_ID/
python -m benchmarks.bench03_vram_capacity --out results/$CAMPAIGN_ID/
python -m benchmarks.bench04_workload_ops  --out results/$CAMPAIGN_ID/ --config configs/escher_14b_480p.json
python -m benchmarks.bench05_e2e_mfu       --out results/$CAMPAIGN_ID/ --config configs/escher_14b_480p.json
torchrun --nproc_per_node=8 benchmarks/bench06_multigpu_comm.py --out results/$CAMPAIGN_ID/
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
./test.sh -t campaign -i 3             # full campaign 3× (regression averaging)
./test.sh -a                           # every testcase × every workload
```

`run.sh` can also be invoked directly when you already know the exact
parameters and don't want the test.sh registry layer:

```bash
./run.sh --testcase compute   --iterations 5
./run.sh --testcase e2e       --workload smoke      --iterations 3
./run.sh --testcase multigpu  --nproc 8             --iterations 3
./run.sh --testcase campaign  --campaign-id $(date +%Y%m%d-%H%M%S)
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

The campaign is fully runnable on a CPU-only host so that the analytic
parts of TESTPLAN §1.2 (op-table accounting, calibration drift, scorecard
plumbing, report rendering) can be exercised without an accelerator.
On a CPU host:

- `./setup.sh` installs `torch` + `torchvision` from the CPU wheel index
  and creates `.microbenchmarks-cpu-venv/`.
- `./test.sh -t campaign` runs `bench04_workload_ops` (analytic table only;
  `t_ms_*` columns are NaN by design) and skips `bench01..03`, `bench05`,
  `bench06`, and the external validators with explicit `skipped (DEVICE=cpu)`
  log lines.
- `scorecard.md` shows a `HOST CPU` row at the top, marks SC-1, SC-2, SC-4
  with `SKIP reason=missing <gpu artifact>`, and SC-5 with
  `SKIP reason=no GPU detected` (instead of `FAIL`). SC-3 (workload op
  taxonomy) is graded normally.
- `report.{md,html}` and `validation.md` are still generated end-to-end so
  the doc pipeline is regression-tested in CI.

The campaign's exit code is 0 on a CPU host as long as the analytic
artifacts are produced, so this can run as a smoke gate in CI even when
no GPU is attached.

### Result location

Per-campaign artifacts land under `results/<campaign-id>/` and the
per-iteration logs under `results/_logs/`. Override the root with
`$RESULTS_DIR`:

```bash
RESULTS_DIR=/data/odyssey-mi355x ./test.sh -t campaign
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

A campaign signs off only if SC-1 … SC-5 in `TESTPLAN.md §1.2` all hold AND the
cross-validation report shows no `FAIL` rows. The orchestrator exits non-zero
if either check fails, so this can run in CI as a regression gate.
