#!/bin/bash
#===============================================================================
# Project: microbenchmarks (escher_14b_480p / MI355X)
# Run script: dispatches a single (testcase, workload, N-iteration) job.
# Invoked by test.sh; can also be called standalone.
#===============================================================================

# Configuration
PROJECT="microbenchmarks"
[[ -f .env ]] && . .env

# Auto-detect DEVICE if not explicitly set (mirrors setup.sh's autodetect):
#   rocm if rocm-smi or amd-smi reports a GPU
#   cuda if nvidia-smi reports a GPU
#   cpu  otherwise
# Override by exporting DEVICE=rocm|cuda|cpu before invocation.
if [[ -z "${DEVICE:-}" ]]; then
  if   command -v rocm-smi   >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1 \
       && [[ $(rocm-smi --showid 2>/dev/null | grep -c '^GPU') -gt 0 ]]; then
    DEVICE="rocm"
  elif command -v amd-smi    >/dev/null 2>&1 && [[ $(amd-smi list 2>/dev/null | grep -c 'GPU:') -gt 0 ]]; then
    DEVICE="rocm"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
       && [[ $(nvidia-smi -L 2>/dev/null | wc -l) -gt 0 ]]; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

# Environment Setup — find the matching venv, or fall back to whichever
# *-venv exists (e.g. ./run.sh on a CPU host with the rocm venv removed).
VENV=".${PROJECT}-${DEVICE}-venv"
if [[ ! $VIRTUAL_ENV =~ $VENV ]]; then
  if [[ -d "$VENV" ]]; then
    . "$VENV/bin/activate"
  else
    # Look for any sibling venv (cpu/cuda/rocm) created by setup.sh.
    fallback=""
    for d in .${PROJECT}-cpu-venv .${PROJECT}-cuda-venv .${PROJECT}-rocm-venv; do
      [[ -d "$d" ]] && { fallback="$d"; break; }
    done
    if [[ -n "$fallback" ]]; then
      echo "WARN: venv $VENV not found — falling back to $fallback (run ./setup.sh DEVICE=$DEVICE for a matching one)."
      . "$fallback/bin/activate"
    else
      echo "WARN: venv $VENV not found — run ./setup.sh first."
    fi
  fi
fi

# Environment Configuration — partition mode + GPU count
GPUS=$(amd-smi list 2>/dev/null | grep -c "GPU:")
[[ -z "$GPUS" || "$GPUS" =~ [^0-9] ]] && GPUS=0
if [[ "$GPUS" -le 0 ]]; then
  GPUS=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU')
  [[ -z "$GPUS" || "$GPUS" =~ [^0-9] ]] && GPUS=0
fi
if [[ "$GPUS" -le 0 ]]; then
  GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
  [[ -z "$GPUS" || "$GPUS" =~ [^0-9] ]] && GPUS=0
fi
PX=$(amd-smi static --gpu 0 2>/dev/null | grep "COMPUTE_PARTITION:" | awk '{print $2}')
NPS=$(amd-smi static --gpu 0 2>/dev/null | grep "MEMORY_PARTITION:" | awk '{print $2}')
PX=${PX:-"NA"}
NPS=${NPS:-"NA"}

# Results and Logging Setup
RESULTS_DIR=${RESULTS_DIR:-"$PWD/results"}
LOG_DIR=${LOG_DIR:-"$RESULTS_DIR/_logs"}
export RESULTS_DIR LOG_DIR
mkdir -p "$LOG_DIR"

# --- Argument parsing ---------------------------------------------------------
usage() {
  cat <<'USAGE'
Usage: ./run.sh [options]

Single-job benchmark runner. Normally invoked by test.sh, but can also be
called directly to run a specific testcase. Probes the system, materializes
a derived config when shape/depth overrides are present, dispatches to the
right bench0X / validator / report script, runs N iterations, parses each
iteration's JSON outputs into a single-line RESULT: record, and aggregates
mean/median across iterations.

Required (when invoked manually, otherwise defaulted):
  --testcase NAME      compute|bandwidth|dram|workload|e2e|multigpu|
                       validation|plot|score|report|campaign
  --workload NAME      escher_14b_480p|smoke|big  (or any registered key)

Optional:
  --config PATH        Workload JSON spec (default: configs/escher_14b_480p.json)
  --depth N            Override model.depth in a derived config copy
  --seq-img N          Override shapes.seq_image in a derived config copy
  --seq-txt N          Override shapes.seq_text  in a derived config copy
  --out-id ID          Per-job output directory name under results/
  --campaign-id ID     Group identifier for the surrounding campaign
  --iterations N       Repeat the testcase N times and aggregate (default: 1)
  --nproc N            Distributed world size (default: detected GPU count,
                       or #CCDs on CPU for the multigpu testcase)
  --dist 0|1           Force distributed mode (torchrun) on/off
  --cpu-topology MODE  ccd|socket|split|auto — how the multigpu testcase
                       maps gloo ranks to host CPUs (default: auto)

Profiling:
  --prepare-sys        Apply kernel/VM tunings (numa_balancing, hugepages,
                       drop_caches, perflevel high). Requires sudo.
  --no-stat            Disable background telemetry (amd-smi/rocm-smi/nvidia-smi)
  --stat-interval N    Telemetry sample interval in seconds (default: 1)
  --numactl SPEC       Wrap the bench command with `numactl SPEC ...`

Environment overrides:
  DEVICE=rocm|cuda|cpu  (auto-detected when unset)
  RESULTS_DIR=PATH      (default: \$PWD/results)
  LOG_DIR=PATH          (default: \$RESULTS_DIR/_logs)

  -h, --help           Show this help and exit.
USAGE
}

arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --testcase)    TESTCASE="$2"; shift 2 ;;
      --workload)    WORKLOAD="$2"; shift 2 ;;
      --config)      CONFIG="$2"; shift 2 ;;
      --depth)       DEPTH="$2"; shift 2 ;;
      --seq-img)     SEQ_IMG="$2"; shift 2 ;;
      --seq-txt)     SEQ_TXT="$2"; shift 2 ;;
      --out-id)      OUT_ID="$2"; shift 2 ;;
      --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
      --iterations)  ITERATIONS="$2"; shift 2 ;;
      --nproc)       NPROC="$2"; shift 2 ;;
      --dist)        DIST="$2"; shift 2 ;;
      --cpu-topology) CPU_TOPOLOGY="$2"; shift 2 ;;
      --prepare-sys) PREPARE_SYS=1; shift ;;
      --no-stat)     STAT="off"; shift ;;
      --stat-interval) STAT_INTERVAL="$2"; shift 2 ;;
      --numactl)     NUMACTL_SPEC="$2"; shift 2 ;;
      -h|--help)     usage; exit 0 ;;
      *) echo "ERROR: Unknown argument $1"; usage; exit 1 ;;
    esac
  done

  TESTCASE=${TESTCASE:-"compute"}
  WORKLOAD=${WORKLOAD:-"escher_14b_480p"}
  CONFIG=${CONFIG:-"configs/escher_14b_480p.json"}
  ITERATIONS=${ITERATIONS:-1}
  CPU_TOPOLOGY=${CPU_TOPOLOGY:-"auto"}
  NPROC=${NPROC:-$GPUS}
  [[ -z "$NPROC" || "$NPROC" =~ [^0-9] || "$NPROC" -le 0 ]] && NPROC=1
  # On a CPU host the multigpu testcase auto-targets one rank per CCD so the
  # gloo collective sweep actually crosses the Infinity Fabric boundary
  # rather than memcpy'ing within a single CCD. Honor an explicit --nproc.
  if [[ "$DEVICE" == "cpu" && "$TESTCASE" == "multigpu" && "$NPROC" -le 1 ]]; then
    AUTO_W=$(python3 -c "from benchmarks.common.topology import detect_cpu_topology as t; \
import os; topo=t(); \
print(max(1, min(int(topo.get('dies') or 1), int(os.cpu_count() or 1))))" 2>/dev/null || echo 1)
    if [[ "$AUTO_W" -gt 1 ]]; then
      echo "INFO: CPU host: auto-selecting NPROC=$AUTO_W (one rank per CCD; override with --nproc)"
      NPROC="$AUTO_W"
    fi
  fi
  DIST=${DIST:-0}
  CAMPAIGN_ID=${CAMPAIGN_ID:-$(date +%Y%m%d-%H%M%S)}
  OUT_ID=${OUT_ID:-"${CAMPAIGN_ID}-${TESTCASE}-${WORKLOAD}"}

  PREPARE_SYS=${PREPARE_SYS:-0}
  STAT=${STAT:-"on"}
  STAT_INTERVAL=${STAT_INTERVAL:-1}
  NUMACTL_SPEC=${NUMACTL_SPEC:-""}
  if [[ -n "$NUMACTL_SPEC" ]] && command -v numactl >/dev/null 2>&1; then
    NUMACTL="numactl $NUMACTL_SPEC"
  else
    NUMACTL=""
    if [[ -n "$NUMACTL_SPEC" ]]; then
      echo "WARN: numactl not installed; ignoring --numactl '$NUMACTL_SPEC'"
    fi
  fi

  # Resolve TESTCASE -> SCRIPT_KEY (the canonical bench0X module / pseudo-target).
  case "$TESTCASE" in
    compute)    SCRIPT_KEY="bench01_bf16_compute" ;;
    bandwidth)  SCRIPT_KEY="bench02_hbm_bandwidth" ;;
    dram)       SCRIPT_KEY="bench03_dram_capacity" ;;
    workload)   SCRIPT_KEY="bench04_workload_ops" ;;
    e2e)        SCRIPT_KEY="bench05_e2e_mfu" ;;
    multigpu)   SCRIPT_KEY="bench06_multigpu_comm"; DIST=1 ;;
    validation) SCRIPT_KEY="VALIDATE" ;;
    plot)       SCRIPT_KEY="PLOT" ;;
    score)      SCRIPT_KEY="SCORE" ;;
    report)     SCRIPT_KEY="REPORT" ;;
    campaign)   SCRIPT_KEY="CAMPAIGN" ;;
    *) echo "ERROR: Unknown testcase '$TESTCASE'"; exit 1 ;;
  esac

  RUN_OUT="${RESULTS_DIR}/${OUT_ID}"
  mkdir -p "$RUN_OUT"

  export CPU_TOPOLOGY
  export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$PWD"

  # Materialize derived config when shape/depth overrides are present so that
  # configs/escher_14b_480p.json itself is never mutated.
  if [[ -n "$DEPTH" || -n "$SEQ_IMG" || -n "$SEQ_TXT" ]]; then
    DERIVED="$RUN_OUT/derived_config.json"
    python3 - "$CONFIG" "$DERIVED" "${DEPTH:-}" "${SEQ_IMG:-}" "${SEQ_TXT:-}" <<'PY'
import json, pathlib, sys
src, dst, depth, si, st = sys.argv[1:6]
j = json.loads(pathlib.Path(src).read_text())
if depth: j.setdefault("model", {})["depth"] = int(depth)
if si:    j.setdefault("shapes", {})["seq_image"] = int(si)
if st:    j.setdefault("shapes", {})["seq_text"]  = int(st)
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(dst).write_text(json.dumps(j, indent=2))
PY
    CONFIG_EFFECTIVE="$DERIVED"
  else
    CONFIG_EFFECTIVE="$CONFIG"
  fi

  CMD="(set per-iteration in run_benchmark)"
  echo "INFO: Run: testcase=$TESTCASE workload=$WORKLOAD script=$SCRIPT_KEY"
  echo "INFO: Run: nproc=$NPROC dist=$DIST iterations=$ITERATIONS"
  echo "INFO: Run: config=$CONFIG_EFFECTIVE out=$RUN_OUT"
}

# --- Stat helpers (Partition style) -------------------------------------------
median() {
  local sorted=($(printf "%s\n" "$@" | sort -n))
  local count=${#sorted[@]}
  ((count == 0)) && { echo "n/a"; return; }
  if ((count % 2)); then
    echo "${sorted[$((count / 2))]}"
  else
    echo "scale=4; (${sorted[$((count / 2 - 1))]} + ${sorted[$((count / 2))]}) / 2" | bc -l
  fi
}
mean() {
  (( $# == 0 )) && { echo "n/a"; return; }
  local sum=0
  for n in "$@"; do sum=$(echo "scale=6; $sum + $n" | bc -l); done
  echo "scale=4; $sum / $#" | bc -l
}
total() {
  local sum=0
  for n in "$@"; do sum=$(echo "scale=6; $sum + $n" | bc -l); done
  echo "scale=4; $sum" | bc -l
}

# --- Profiling: live GPU/CPU telemetry -----------------------------------------
# start_stat / end_stat run a background poller for the duration of the
# iteration loop, writing time-series telemetry to $STAT_LOG. Tries amd-smi,
# falls back to rocm-smi polling, then nvidia-smi, then disables itself.
# Disable explicitly with --no-stat or STAT=off. Interval is in seconds.
start_stat() {
  STAT_PID=""
  if [[ "$STAT" == "off" ]]; then
    echo "INFO: stat: disabled (--no-stat)"
    return 0
  fi
  : > "$STAT_LOG"
  if command -v amd-smi >/dev/null 2>&1; then
    # amd-smi metric streams a snapshot every --watch seconds.
    amd-smi metric --watch "$STAT_INTERVAL" >> "$STAT_LOG" 2>&1 &
    STAT_PID=$!
    echo "INFO: stat: amd-smi metric --watch ${STAT_INTERVAL}s (pid=$STAT_PID -> $STAT_LOG)"
  elif command -v rocm-smi >/dev/null 2>&1; then
    (
      while :; do
        echo "--- $(date +%s.%N) ---"
        rocm-smi --showpower --showtemp --showuse --showmemuse --showclocks --csv 2>/dev/null
        sleep "$STAT_INTERVAL"
      done
    ) >> "$STAT_LOG" 2>&1 &
    STAT_PID=$!
    echo "INFO: stat: rocm-smi polling @ ${STAT_INTERVAL}s (pid=$STAT_PID -> $STAT_LOG)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem \
      --format=csv -lms $((STAT_INTERVAL * 1000)) >> "$STAT_LOG" 2>&1 &
    STAT_PID=$!
    echo "INFO: stat: nvidia-smi @ ${STAT_INTERVAL}s (pid=$STAT_PID -> $STAT_LOG)"
  else
    echo "WARN: stat: no telemetry tool found (amd-smi/rocm-smi/nvidia-smi); skipping."
  fi
}
end_stat() {
  if [[ -n "${STAT_PID:-}" ]]; then
    kill "$STAT_PID" 2>/dev/null || true
    wait "$STAT_PID" 2>/dev/null || true
    echo "INFO: stat: stopped (pid=$STAT_PID, log=$STAT_LOG)"
    STAT_PID=""
  fi
}

# --- Profiling: kernel/VM tuning (Phoronix-style) ------------------------------
# Disabled by default because every operation needs sudo. Enable with
# --prepare-sys when the runner is on a dedicated benchmark host. Failures
# are non-fatal so the rest of the run continues even when sudo is denied.
prepare_sys() {
  if [[ "$PREPARE_SYS" != "1" ]]; then
    echo "INFO: prepare_sys: skipped (set --prepare-sys to enable; needs sudo)"
    return 0
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "WARN: prepare_sys: sudo not available; skipping kernel tuning."
    return 0
  fi
  echo "INFO: prepare_sys: numa_balancing=0"
  echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1 || true
  echo "INFO: prepare_sys: transparent_hugepage/enabled=always"
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
  echo "INFO: prepare_sys: transparent_hugepage/defrag=always"
  echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
  echo "INFO: prepare_sys: vm/compact_memory=1"
  echo 1 | sudo tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
  echo "INFO: prepare_sys: vm/drop_caches=3"
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "INFO: prepare_sys: rocm-smi --setperflevel high (lock SCLK/MCLK to top P-state)"
    sudo rocm-smi --setperflevel high >/dev/null 2>&1 || true
  fi
}

# --- Profiling: one-shot system metadata dump ---------------------------------
# Writes hardware/software fingerprint to the main LOG so every run is
# self-describing without needing the env.json sidecar.
log_sysinfo() {
  echo "================================================"
  echo "INFO: System Settings"
  echo "================================================"
  echo "--- uname -a ---";        uname -a 2>/dev/null
  echo "--- lscpu (head) ---";    lscpu 2>/dev/null | head -40
  if command -v lsb_release >/dev/null 2>&1; then
    echo "--- lsb_release -a ---"; lsb_release -a 2>&1
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    echo "--- rocminfo (head) ---"
    rocminfo 2>/dev/null | sed -n '1,80p'
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "--- rocm-smi --showversion --showvbios --showid ---"
    rocm-smi --showversion --showvbios --showid 2>/dev/null
    echo "--- rocm-smi --showclocks --showtemp --showpower --showmeminfo vram ---"
    rocm-smi --showclocks --showtemp --showpower --showmeminfo vram 2>/dev/null
    echo "--- rocm-smi --showperflevel ---"
    rocm-smi --showperflevel 2>/dev/null
  fi
  if command -v amd-smi >/dev/null 2>&1; then
    echo "--- amd-smi static --gpu all (head) ---"
    amd-smi static --gpu all 2>/dev/null | sed -n '1,80p'
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi -q (head) ---"
    nvidia-smi -q 2>/dev/null | sed -n '1,40p'
  fi
  echo "--- /sys knobs ---"
  for f in \
    /proc/sys/kernel/numa_balancing \
    /sys/kernel/mm/transparent_hugepage/enabled \
    /sys/kernel/mm/transparent_hugepage/defrag \
    /sys/devices/system/cpu/smt/control \
    /sys/devices/system/cpu/cpufreq/boost; do
    if [[ -r "$f" ]]; then
      printf '%s = %s\n' "$f" "$(cat "$f" 2>/dev/null)"
    fi
  done
  echo "--- env (relevant) ---"
  env | grep -E '^(ROC|HSA|HIP|HF_|CUDA|TORCH|PYT|VLLM|OMP|MIOPEN|NCCL|RCCL)' | sort
  echo "--- pip freeze (key pkgs) ---"
  pip freeze 2>/dev/null | grep -iE '^(torch|triton|aiter|flash|numpy|matplotlib|pandas|jq)' || echo "(pip not available)"
}

# --- Per-iteration dispatcher -------------------------------------------------
# Use python3 directly: when no venv is active, the bare `python` symlink may
# not exist (Ubuntu 24.04 ships only python3). When a venv IS active, python3
# resolves into the venv's bin/ as well.
PY="${PY:-python3}"

# Which testcases unconditionally require a GPU. The bench0* scripts now
# detect the device internally and run a CPU fallback when no accelerator is
# present, so only the external validators (rvs / rocm-bw / rccl) and any
# multi-rank GPU collective bench remain GPU-only.
gpu_required() {
  # bench06 now has a CPU multi-CCD / multi-socket gloo path; it is GPU-only
  # only when DIST=1 was forced on a host with no accelerator AND no CPU
  # topology to spread across. The orchestrator handles that case explicitly.
  case "$1" in
    VALIDATE)
      return 0 ;;
    *) return 1 ;;
  esac
}

run_benchmark() {
  if gpu_required "$SCRIPT_KEY" && [[ "$DEVICE" == "cpu" ]]; then
    echo "INFO: $TESTCASE ($SCRIPT_KEY): skipped — DEVICE=cpu and this testcase needs a GPU."
    echo "      To run on a GPU host: export DEVICE=rocm (or cuda) and re-run."
    return 0
  fi

  case "$SCRIPT_KEY" in
    CAMPAIGN)
      CMD="bash scripts/run_campaign.sh $OUT_ID"
      ;;
    VALIDATE)
      "$PY" -m benchmarks.common.env --out "$RUN_OUT" --campaign-id "$OUT_ID" || true
      CMD="bash validation/rvs/run_rvs.sh $RUN_OUT \
        && bash validation/rocm_bw/run_rocm_bw.sh $RUN_OUT"
      [[ "$NPROC" -gt 1 ]] && CMD+=" && bash validation/rccl/run_rccl_tests.sh $RUN_OUT $NPROC"
      CMD+=" && $PY validation/compare.py --out $RUN_OUT"
      ;;
    REPORT)
      CMD="$PY scripts/report.py --out $RUN_OUT --config $CONFIG_EFFECTIVE --format both"
      ;;
    SCORE)
      CMD="$PY scripts/score_campaign.py --out $RUN_OUT"
      ;;
    PLOT)
      CMD="$PY scripts/plot_results.py --out $RUN_OUT"
      ;;
    bench0*)
      "$PY" -m benchmarks.common.env --out "$RUN_OUT" --campaign-id "$OUT_ID" || true
      if [[ "$DIST" == "1" ]]; then
        if [[ "$NPROC" -lt 2 ]]; then
          echo "WARN: $TESTCASE needs multi-rank but NPROC=$NPROC; skipping iteration."
          return 0
        fi
        CMD="torchrun --nproc_per_node=$NPROC benchmarks/${SCRIPT_KEY}.py --out $RUN_OUT"
      else
        CMD="$PY -m benchmarks.${SCRIPT_KEY} --out $RUN_OUT"
      fi
      case "$SCRIPT_KEY" in
        bench04_workload_ops|bench05_e2e_mfu)
          CMD+=" --config $CONFIG_EFFECTIVE"
          ;;
        bench06_multigpu_comm)
          if [[ "$DEVICE" == "cpu" ]]; then
            CMD+=" --cpu-topology $CPU_TOPOLOGY"
          fi
          ;;
      esac
      ;;
    *)
      echo "ERROR: unknown SCRIPT '$SCRIPT_KEY'"
      return 1
      ;;
  esac

  # Prefix CPU pinning if requested (no-op when --numactl was not passed
  # or numactl is not installed). Only meaningful for non-distributed runs;
  # torchrun multi-process launches handle their own affinity.
  if [[ -n "$NUMACTL" && "$DIST" != "1" ]]; then
    CMD="$NUMACTL $CMD"
  fi

  echo "INFO: Executing -> $CMD"
  eval "$CMD"
}

# --- Result extraction --------------------------------------------------------
# Pulls the headline JSON metrics for the testcase and prints a single-line
# RESULT: ... record per iteration. This is the analogue of get_results() in
# the Partition project, but reads bench-script JSONs instead of jsonl.
get_results() {
  local r="${RUN_OUT}"
  local RESULT="iteration=${ITERATION} testcase=${TESTCASE}"
  echo "========================================"
  echo "JSON outputs (testcase=$TESTCASE, iter=$ITERATION):"
  case "$TESTCASE" in
    compute)
      local f="$r/01_bf16_compute/peak.json"
      local s="$r/01_bf16_compute/summary.json"
      if [[ -f "$f" && -f "$s" ]]; then
        local peak=$(jq -r '.tflops // empty' "$f")
        local roof=$(jq -r '.compute_roof_tflops // empty' "$s")
        local best_name=$(jq -r '.best_sweep.name // empty' "$s")
        local best_tflops=$(jq -r '.best_sweep.tflops // empty' "$s")
        RESULT+=" peak_tflops=${peak:-na} compute_roof_tflops=${roof:-na} best_sweep=${best_name:-na} best_sweep_tflops=${best_tflops:-na}"
      else
        RESULT+=" peak_tflops=na (missing $f or $s)"
      fi
      ;;
    bandwidth)
      local f="$r/02_hbm_bandwidth/summary.json"
      if [[ -f "$f" ]]; then
        local roof=$(jq -r '.bandwidth_roof_gb_s // empty' "$f")
        local copy=$(jq -r '.plateau_gb_s_per_op["copy_"] // empty' "$f")
        local sumv=$(jq -r '.plateau_gb_s_per_op["sum"] // empty' "$f")
        RESULT+=" hbm_roof_gb_s=${roof:-na} copy_plateau_gb_s=${copy:-na} sum_plateau_gb_s=${sumv:-na}"
      else
        RESULT+=" hbm_roof_gb_s=na (missing $f)"
      fi
      ;;
    dram)
      local f="$r/03_dram_capacity/summary.json"
      if [[ -f "$f" ]]; then
        local bf16=$(jq -r '.max_alloc_bf16_gib // empty' "$f")
        local util=$(jq -r '.eff_util_fraction_bf16 // empty' "$f")
        local frag=$(jq -r '.frag_sensitivity_ratio // empty' "$f")
        RESULT+=" max_alloc_bf16_gib=${bf16:-na} eff_util_fraction=${util:-na} frag_ratio=${frag:-na}"
      else
        RESULT+=" max_alloc_bf16_gib=na (missing $f)"
      fi
      ;;
    workload)
      local f="$r/04_workload_ops/ops.json"
      if [[ -f "$f" ]]; then
        local gflops=$(jq -r '.totals.total_gflops // empty' "$f")
        local mb=$(jq -r '.totals.total_mb_hbm // empty' "$f")
        local ai=$(jq -r '.totals.avg_arithmetic_intensity // empty' "$f")
        local drift_g=$(jq -r '.calibration_drift.gflops_drift_pct // empty' "$f")
        local drift_b=$(jq -r '.calibration_drift.mb_hbm_drift_pct // empty' "$f")
        local ridge=$(jq -r '.ridge_flop_per_byte // empty' "$f")
        RESULT+=" total_gflops=${gflops:-na} total_mb_hbm=${mb:-na} avg_AI=${ai:-na} ridge_flop_per_byte=${ridge:-na} drift_gflops_pct=${drift_g:-na} drift_hbm_pct=${drift_b:-na}"
      else
        RESULT+=" total_gflops=na (missing $f)"
      fi
      ;;
    e2e)
      local f="$r/05_e2e_mfu/mfu.json"
      if [[ -f "$f" ]]; then
        local mode=$(jq -r '.compile_mode_used // "n/a"' "$f")
        local mfu_eag=$(jq -r '.rows[] | select(.scope=="eager_e2e") | .mfu_measured_peak // empty' "$f")
        local mfu_cmp=$(jq -r '.rows[] | select(.scope=="compiled_e2e") | .mfu_measured_peak // empty' "$f")
        local mfu_sop=$(jq -r '.rows[] | select(.scope=="sum_of_ops_optimized") | .mfu_measured_peak // empty' "$f")
        local tps_eag=$(jq -r '.rows[] | select(.scope=="eager_e2e") | .tflops_achieved // empty' "$f")
        local tps_cmp=$(jq -r '.rows[] | select(.scope=="compiled_e2e") | .tflops_achieved // empty' "$f")
        RESULT+=" compile_mode=${mode} mfu_sum_of_ops=${mfu_sop:-na} mfu_eager=${mfu_eag:-na} mfu_compiled=${mfu_cmp:-na} tflops_eager=${tps_eag:-na} tflops_compiled=${tps_cmp:-na}"
      else
        RESULT+=" mfu=na (missing $f)"
      fi
      ;;
    multigpu)
      local f="$r/06_multigpu_comm/comm.json"
      if [[ -f "$f" ]]; then
        local world=$(jq -r '.world // empty' "$f")
        local ar_max=$(jq -r '[.rows[] | select(.op=="all_reduce") | .busbw_gb_s] | max // empty' "$f")
        local ag_max=$(jq -r '[.rows[] | select(.op=="all_gather") | .busbw_gb_s] | max // empty' "$f")
        local rs_max=$(jq -r '[.rows[] | select(.op=="reduce_scatter") | .busbw_gb_s] | max // empty' "$f")
        RESULT+=" world=${world:-na} all_reduce_busbw_gb_s=${ar_max:-na} all_gather_busbw_gb_s=${ag_max:-na} reduce_scatter_busbw_gb_s=${rs_max:-na}"
      else
        RESULT+=" busbw=na (missing $f)"
      fi
      ;;
    validation)
      local f="$r/validation.json"
      if [[ -f "$f" ]]; then
        local pass=$(jq -r '[.[] | select(.status=="PASS")] | length' "$f")
        local fail=$(jq -r '[.[] | select(.status=="FAIL")] | length' "$f")
        local skip=$(jq -r '[.[] | select(.status=="SKIP")] | length' "$f")
        RESULT+=" pass=${pass} fail=${fail} skip=${skip}"
      else
        RESULT+=" validation=na (missing $f)"
      fi
      ;;
    score)
      local f="$r/scorecard.json"
      if [[ -f "$f" ]]; then
        local pass=$(jq -r '[.[] | select(.status=="PASS")] | length' "$f")
        local fail=$(jq -r '[.[] | select(.status=="FAIL")] | length' "$f")
        local skip=$(jq -r '[.[] | select(.status=="SKIP")] | length' "$f")
        RESULT+=" SC_pass=${pass} SC_fail=${fail} SC_skip=${skip}"
      else
        RESULT+=" scorecard=na (missing $f)"
      fi
      ;;
    campaign)
      local s="$r/scorecard.json"
      local v="$r/validation.json"
      local sp=$([[ -f "$s" ]] && jq -r '[.[] | select(.status=="PASS")] | length' "$s" || echo na)
      local sf=$([[ -f "$s" ]] && jq -r '[.[] | select(.status=="FAIL")] | length' "$s" || echo na)
      local vp=$([[ -f "$v" ]] && jq -r '[.[] | select(.status=="PASS")] | length' "$v" || echo na)
      local vf=$([[ -f "$v" ]] && jq -r '[.[] | select(.status=="FAIL")] | length' "$v" || echo na)
      RESULT+=" SC_pass=${sp} SC_fail=${sf} valid_pass=${vp} valid_fail=${vf}"
      ;;
    plot|report)
      RESULT+=" status=ok"
      ;;
  esac
  echo "RESULT: $RESULT"
  if [[ -n "$RESULT_LOG" ]]; then
    # Re-create the parent dir defensively. The campaign itself (or an
    # external cleanup) can have rmtree'd $RUN_OUT between iterations;
    # we'd rather lose the sidecar than abort the wrapper.
    mkdir -p "$(dirname "$RESULT_LOG")" 2>/dev/null
    echo "$RESULT" >> "$RESULT_LOG" 2>/dev/null \
      || echo "WARN: could not append RESULT to $RESULT_LOG"
  fi
}

# --- Cross-iteration aggregation ---------------------------------------------
# After all ITERATIONS run, pull a single headline metric from each iteration's
# RESULT line and compute mean/median. RESULT lines are appended to a sidecar
# file ($RESULT_LOG) by get_results() so we don't race with the |& tee pipeline.
aggregate_iterations() {
  local fields="$1"
  local sidecar="$2"
  echo "========================================"
  echo "AGGREGATE across $ITERATIONS iterations:"
  if [[ ! -s "$sidecar" ]]; then
    echo "  no iteration results recorded"
    return 0
  fi
  for field in $fields; do
    local vals=()
    while IFS= read -r line; do
      local v
      v=$(echo "$line" | grep -oE "(^| )${field}=[^ ]+" | head -n1 | cut -d= -f2)
      if [[ -n "$v" && "$v" != "na" ]]; then vals+=("$v"); fi
    done < "$sidecar"
    if (( ${#vals[@]} == 0 )); then
      echo "  $field: no numeric samples"
    else
      echo "  $field: n=${#vals[@]} mean=$(mean "${vals[@]}") median=$(median "${vals[@]}") values=(${vals[*]})"
    fi
  done
}

# --- Main --------------------------------------------------------------------
main() {
  echo "========================================"
  echo "INFO: Project: $PROJECT-$DEVICE"
  echo "========================================"

  arguments "$@"

  DATETIME=$(date +'%Y%m%d-%H%M%S')
  LOG_ID="${TESTCASE}-${WORKLOAD}-px${PX}-nps${NPS}-g${GPUS}-tp${NPROC}"
  LOG="${LOG_DIR}/${LOG_ID}.${DATETIME}.log"
  STAT_LOG="${LOG_DIR}/${LOG_ID}.${DATETIME}.stat.log"
  RESULT_LOG="${RUN_OUT}/iterations.results"
  # Defensive: $RUN_OUT may have been wiped between arguments() and main()
  # by an external cleanup; recreate before opening the sidecar.
  mkdir -p "$RUN_OUT"
  : > "$RESULT_LOG"
  trap 'end_stat' EXIT INT TERM

  # Headline fields per testcase used by aggregate_iterations.
  local AGG_FIELDS
  case "$TESTCASE" in
    compute)    AGG_FIELDS="peak_tflops compute_roof_tflops best_sweep_tflops" ;;
    bandwidth)  AGG_FIELDS="hbm_roof_gb_s copy_plateau_gb_s sum_plateau_gb_s" ;;
    dram)       AGG_FIELDS="max_alloc_bf16_gib eff_util_fraction frag_ratio" ;;
    workload)   AGG_FIELDS="total_gflops total_mb_hbm avg_AI ridge_flop_per_byte drift_gflops_pct drift_hbm_pct" ;;
    e2e)        AGG_FIELDS="mfu_eager mfu_compiled mfu_sum_of_ops tflops_eager tflops_compiled" ;;
    multigpu)   AGG_FIELDS="all_reduce_busbw_gb_s all_gather_busbw_gb_s reduce_scatter_busbw_gb_s" ;;
    validation) AGG_FIELDS="pass fail skip" ;;
    score)      AGG_FIELDS="SC_pass SC_fail SC_skip" ;;
    campaign)   AGG_FIELDS="SC_pass SC_fail valid_pass valid_fail" ;;
    *) AGG_FIELDS="" ;;
  esac

  {
    echo "INFO: Test:    $LOG_ID"
    echo "INFO: Date:    $DATETIME"
    echo "INFO: Out dir: $RUN_OUT"
    echo "INFO: Log:     $LOG"
    echo "INFO: Stat:    $STAT_LOG"
    echo "========================================"
    echo "SYSTEM: compute_partition=${PX}, memory_partition=${NPS}, gpus_available=${GPUS}, nproc=${NPROC}"
    echo "CONFIG: project=${PROJECT}, device=${DEVICE}, testcase=${TESTCASE}, workload=${WORKLOAD}, dist=${DIST}, script=${SCRIPT_KEY}"
    echo "PARAMS: config=${CONFIG_EFFECTIVE}, depth=${DEPTH:-keep}, seq_image=${SEQ_IMG:-keep}, seq_text=${SEQ_TXT:-keep}, iterations=${ITERATIONS}, prepare_sys=${PREPARE_SYS}, stat=${STAT}, stat_interval=${STAT_INTERVAL}, numactl='${NUMACTL_SPEC}'"
    echo "========================================"

    echo "INFO: Preparing System"
    prepare_sys
    log_sysinfo
    echo "========================================"
    echo "INFO: Starting telemetry"
    start_stat
    echo "========================================"

    start_time=$(date +%s)
    for ITERATION in $(seq 1 "$ITERATIONS"); do
      echo "INFO: Iteration: $ITERATION of $ITERATIONS"
      run_benchmark
      get_results
    done
    end_time=$(date +%s)

    end_stat

    if [[ -n "$AGG_FIELDS" && "$ITERATIONS" -gt 1 ]]; then
      aggregate_iterations "$AGG_FIELDS" "$RESULT_LOG"
    fi

    echo "========================================"
    echo "INFO: Date: $(date)"
    echo "INFO: Total Elapsed Time: $((end_time - start_time)) seconds"
    echo "INFO: Stat File:  $STAT_LOG"
    echo "INFO: Output Dir: $RUN_OUT"
    echo "INFO: Log File:   $LOG"
    echo "========================================"
  } |& tee "$LOG"
}

main "$@"
