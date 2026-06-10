#!/bin/bash
# microbenchmarks Unified Test Runner
# Executes benchmark families and workload variants for the
# escher_14b_480p / MI355X benchmark described in docs/TESTPLAN.md.

PROJECT="microbenchmarks"
[[ -f .env ]] && . .env

# Auto-detect DEVICE if not explicitly set (mirrors setup.sh / run.sh):
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
export DEVICE

VENV=".${PROJECT}-${DEVICE}-venv"
if [[ ! "${VIRTUAL_ENV:-}" =~ $VENV ]]; then
  if [[ -d "$VENV" ]]; then
    . $VENV/bin/activate
  else
    fallback=""
    for d in .${PROJECT}-cpu-venv .${PROJECT}-cuda-venv .${PROJECT}-rocm-venv; do
      [[ -d "$d" ]] && { fallback="$d"; break; }
    done
    if [[ -n "$fallback" ]]; then
      echo "WARN: venv $VENV not found — falling back to $fallback (run DEVICE=$DEVICE ./setup.sh for a matching one)."
      . "$fallback/bin/activate"
    else
      echo "WARN: venv $VENV not found — run ./setup.sh first."
    fi
  fi
fi

# --- Test case definitions -----------------------------------------------------
# Each entry maps a short name to a `KEY: value; KEY: value` spec.
# SCRIPT special values:
#   bench0X_*  -> python -m benchmarks.<SCRIPT>
#   BENCHMARK   -> bash scripts/run_benchmark.sh <id>
#   VALIDATE   -> external validators only (RVS, rocm-bw, rccl-tests, compare)
#   REPORT     -> python scripts/report.py
#   SCORE      -> python scripts/score_benchmark.py
#   PLOT       -> python scripts/plot_results.py
# DIST=1 marks the testcase as multi-process (torchrun -nproc_per_node=$NPROC).

declare -A TESTCASE
TESTCASE[compute]="DESC: Family 1 — BF16 compute peak + size sweep; SCRIPT: bench01_bf16_compute"
TESTCASE[bandwidth]="DESC: Family 2 — HBM streaming microbenchmarks; SCRIPT: bench02_hbm_bandwidth"
TESTCASE[dram]="DESC: Family 3 — DRAM capacity binary search; SCRIPT: bench03_dram_capacity"
TESTCASE[workload]="DESC: Family 4 — escher_14b_480p per-op decomposition; SCRIPT: bench04_workload_ops"
TESTCASE[e2e]="DESC: Family 5 — eager + torch.compile MFU; SCRIPT: bench05_e2e_mfu"
TESTCASE[multigpu]="DESC: Family 6 — RCCL collectives via torchrun; SCRIPT: bench12_multigpu_comm; DIST: 1"
TESTCASE[validation]="DESC: External validators (RVS, rocm-bandwidth-test, rccl-tests, compare); SCRIPT: VALIDATE"
TESTCASE[plot]="DESC: Regenerate plots A2/A3/A6/A7/A8; SCRIPT: PLOT"
TESTCASE[score]="DESC: Score SC-1…SC-5; SCRIPT: SCORE"
TESTCASE[report]="DESC: Build report.md / report.html; SCRIPT: REPORT"
TESTCASE[benchmark]="DESC: Full single-node benchmark (TESTPLAN §15 order); SCRIPT: BENCHMARK"
TESTCASE[fused_aiter]="DESC: Family 6f — AITER fused kernels; SCRIPT: bench06_aiter_fused; DIST: 1"
TESTCASE[fused_symm]="DESC: Family 10 — SymmMem fused kernels; SCRIPT: bench10_symm_fused; DIST: 1"
TESTCASE[iris_overlap]="DESC: Family 13 — MM+AllReduce overlap (Iris vs pipelined vs RCCL); SCRIPT: bench13_iris_overlap; DIST: 1"
TESTCASE[sustained]="DESC: Family 7 — Sustained throughput / thermal drift; SCRIPT: bench07_sustained"
TESTCASE[topology]="DESC: Family 8 — All-pairs D2D bandwidth matrix; SCRIPT: bench08_topology_bw"
TESTCASE[stability]="DESC: Family 9 — Numerical stability sweep; SCRIPT: bench09_numerical_stability"
TESTCASE[quality]="DESC: Family 11 — Perceptual Quality Benchmarking (VBench); SCRIPT: bench11_quality"

# --- Workload (config) definitions --------------------------------------------
# Each workload pins a config file and optional shape/depth overrides.
# Empty override fields keep the value from the config file unchanged.

declare -A WORKLOAD
WORKLOAD[escher_14b_480p]="CONFIG: configs/escher_14b_480p.json; DEPTH: ; SEQ_IMG: ; SEQ_TXT: "
WORKLOAD[smoke]="CONFIG: configs/escher_14b_480p.json; DEPTH: 4; SEQ_IMG: 2048; SEQ_TXT: 256"
WORKLOAD[big]="CONFIG: configs/escher_14b_480p.json; DEPTH: 60; SEQ_IMG: 16384; SEQ_TXT: 1024"

if [[ -f "configs/reference_video_models.json" ]]; then
  while IFS='=' read -r key val; do
    WORKLOAD[$key]="$val"
  done < <(python3 -c '
import json
try:
  with open("configs/reference_video_models.json") as f:
    for m in json.load(f).get("models", []):
      wid = m.get("id")
      wc = m.get("workload_config")
      if wid and wc:
        print(f"{wid}=CONFIG: {wc}; DEPTH: ; SEQ_IMG: ; SEQ_TXT: ")
except Exception:
  pass
' 2>/dev/null)
fi

# --- Environment --------------------------------------------------------------
RESULTS_DIR=${RESULTS_DIR:-"$PWD/results"}
LOG_DIR=${LOG_DIR:-"$RESULTS_DIR/_logs"}
export RESULTS_DIR LOG_DIR
mkdir -p "$LOG_DIR"

DATETIME=$(date +'%Y%m%d-%H%M%S')
TEST_LOG=$LOG_DIR/test.${DATETIME}.log
# ITERATIONS and WARMUP are empty by default to allow Methodology JSON to be the source of truth.
# If not provided on CLI, run.sh and benchmarks fallback to JSON or hardcoded defaults (1 and 5).
ITERATIONS=${ITERATIONS:-""}
WARMUP=${WARMUP:-""}
detect_nproc() {
  local n
  # Primary: torch knows the real device count (not XCC/GCD count)
  n=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
  [[ -z "$n" || "$n" =~ [^0-9] ]] && n=0
  # Fallback: rocm-smi unique GPU IDs (deduped — avoids XCC overcount)
  if [[ "$n" -le 0 ]] && command -v rocm-smi &>/dev/null; then
    n=$(rocm-smi --showuniqueid 2>/dev/null | grep -c 'Unique ID')
    [[ -z "$n" || "$n" =~ [^0-9] ]] && n=0
  fi
  # Fallback: nvidia-smi
  if [[ "$n" -le 0 ]] && command -v nvidia-smi &>/dev/null; then
    n=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | wc -l)
    [[ -z "$n" || "$n" =~ [^0-9] ]] && n=0
  fi
  echo "$n"
}
NPROC=${NPROC:-"$(detect_nproc)"}

# --- Helpers ------------------------------------------------------------------
parse_kv() {
  # parse_kv "$spec_string" "KEY"  -> trimmed value, "" if absent
  local spec="$1" key="$2"
  awk -v k="$key" '
    BEGIN { RS=";" }
    {
      sub(/^[ \t]+/, ""); sub(/[ \t]+$/, "")
      if (substr($0, 1, length(k)+1) == k ":") {
        v = substr($0, length(k)+2)
        sub(/^[ \t]+/, "", v); sub(/[ \t]+$/, "", v)
        print v
        exit
      }
    }
  ' <<< "$spec"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -t, --testcase NAME      run a single testcase (default: benchmark)
  -w, --workload NAME      run with a single workload spec (default: escher_14b_480p)
  -m, --methodology PATH   test methodology JSON (default: configs/test_methodology.json)
  -i, --iterations N       repeat each (testcase, workload) N times (default: 1)
  -W, --warmup N           warmup iterations (default: 5)
  -a, --all                run every testcase × every workload
  -l, --list               list available testcases and workloads, then exit
  -h, --help               this message

Environment:
  DEVICE        rocm | cuda | cpu                (default: rocm)
  NPROC         GPU count for multi-GPU step    (default: auto-detected)
  RESULTS_DIR   benchmark root                   (default: ./results)
  METHODOLOGY   test methodology JSON           (default: configs/test_methodology.json)
  ITERATIONS    repeat count                    (default: 1)
  WARMUP        warmup iterations               (default: 5)
  REFERENCE_MODEL  optional id from configs/reference_video_models.json; written to
                benchmark_meta.json under each results dir for report.py

Examples:
  $0 -t compute                       # only BF16 GEMM
  $0 -t e2e -w smoke                  # quick MFU smoke test
  $0 -t benchmark -i 3                 # full benchmark 3× (regression averaging)
  $0 -a                               # everything × every workload variant
EOF
}

list_options() {
  echo "Test cases:"
  for k in $(echo "${!TESTCASE[@]}" | tr ' ' '\n' | sort); do
    printf "  %-12s %s\n" "$k" "$(parse_kv "${TESTCASE[$k]}" DESC)"
  done
  echo
  echo "Workloads:"
  for k in $(echo "${!WORKLOAD[@]}" | tr ' ' '\n' | sort); do
    cfg=$(parse_kv "${WORKLOAD[$k]}" CONFIG)
    d=$(parse_kv "${WORKLOAD[$k]}" DEPTH)
    si=$(parse_kv "${WORKLOAD[$k]}" SEQ_IMG)
    st=$(parse_kv "${WORKLOAD[$k]}" SEQ_TXT)
    printf "  %-18s config=%s depth=%s seq_img=%s seq_txt=%s\n" \
      "$k" "$cfg" "${d:-keep}" "${si:-keep}" "${st:-keep}"
  done
}

# --- Test runner --------------------------------------------------------------
# Delegates a single (testcase, workload) job to run.sh. run.sh owns the
# per-iteration loop, system probing, derived-config materialization, log
# emission and per-iteration result aggregation. test.sh stays focused on
# enumerating combinations and ordering the benchmark.
run_benchmark() {
  local testcase=$1
  local workload=$2

  local desc=$(parse_kv "${TESTCASE[$testcase]}" DESC)
  local dist=$(parse_kv "${TESTCASE[$testcase]}" DIST)
  local base_cfg=$(parse_kv "${WORKLOAD[$workload]}" CONFIG)
  local depth=$(parse_kv "${WORKLOAD[$workload]}" DEPTH)
  local seq_img=$(parse_kv "${WORKLOAD[$workload]}" SEQ_IMG)
  local seq_txt=$(parse_kv "${WORKLOAD[$workload]}" SEQ_TXT)
  # Benchmark + REFERENCE_MODEL: match run.sh — use registry workload_config for
  # out_id slug and --config so results/ names align with the Physics reference.
  local eff_cfg="$base_cfg"
  if [[ "$testcase" == "benchmark" && -n "${REFERENCE_MODEL:-}" ]]; then
    local _ref_wc
    _ref_wc=$(REFERENCE_MODEL="$REFERENCE_MODEL" python3 - <<'PY' 2>/dev/null || true
import json, os, pathlib
rid = (os.environ.get("REFERENCE_MODEL") or "").strip()
if not rid:
    raise SystemExit(0)
p = pathlib.Path("configs/reference_video_models.json")
if not p.is_file():
    raise SystemExit(0)
for m in json.loads(p.read_text()).get("models", []):
    if m.get("id") == rid and m.get("workload_config"):
        print(m["workload_config"])
        break
PY
)
    if [[ -n "$_ref_wc" ]]; then
      eff_cfg="$_ref_wc"
    fi
  fi
  # Results dir: <model>-<YYYYMMDD>-<HHMMSS>[-<testcase>].  "model" is JSON
  # ``name`` when present (same as run.sh), else the workload registry key.
  local ts_date="${DATETIME%-*}"
  local ts_time="${DATETIME#*-}"
  local model_slug
  model_slug=$(python3 -c "
import json, pathlib, re, sys
cfg, wl = pathlib.Path(sys.argv[1]), sys.argv[2]
try:
    name = json.loads(cfg.read_text()).get('name') or wl
except Exception:
    name = wl
slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('._-') or wl
print(slug)
" "$eff_cfg" "$workload" 2>/dev/null || echo "$workload")
  local out_id
  if [[ "$testcase" == "benchmark" ]]; then
    out_id="${model_slug}-${ts_date}-${ts_time}"
  else
    out_id="${model_slug}-${ts_date}-${ts_time}-${testcase}"
  fi

  echo "================================================"
  echo "Testcase: $testcase ($desc)"
  echo "Workload: $workload (config=$eff_cfg)"
  echo "Iters:    $ITERATIONS    NPROC=$NPROC    out_id=$out_id"
  echo "================================================"

  local cmd=(bash run.sh
    --testcase "$testcase"
    --workload "$workload"
    --config      "$eff_cfg"
    --methodology "${METHODOLOGY:-configs/test_methodology.json}"
    --out-id      "$out_id"
    --benchmark-id "$out_id"
    --nproc      "$NPROC"
    --dist       "${dist:-0}"
  )
  [[ -n "$ITERATIONS" ]] && cmd+=(--iterations "$ITERATIONS")
  [[ -n "$WARMUP" ]]     && cmd+=(--warmup "$WARMUP")
  [[ -n "$depth"   ]] && cmd+=(--depth   "$depth")
  [[ -n "$seq_img" ]] && cmd+=(--seq-img "$seq_img")
  [[ -n "$seq_txt" ]] && cmd+=(--seq-txt "$seq_txt")

  echo "INFO: ${cmd[*]}"
  "${cmd[@]}"
}

# --- Main ---------------------------------------------------------------------
main() {
  local testcase="benchmark"
  local workload="escher_14b_480p"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--testcase) testcase="$2"; shift 2 ;;
      -m|--methodology) METHODOLOGY="$2"; shift 2 ;;
      -w|--workload) workload="$2"; shift 2 ;;
      -i|--iterations) ITERATIONS="$2"; shift 2 ;;
      -W|--warmup)     WARMUP="$2"; shift 2 ;;
      -a|--all) testcase="all"; workload="all"; shift ;;
      -l|--list) list_options; exit 0 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "ERROR: Unknown option $1"; usage; exit 1 ;;
    esac
  done

  # Stable execution order matching TESTPLAN §15 (each ceiling anchored before
  # the metrics derived from it). When --all is requested we run every testcase
  # in this order; "benchmark" already does this internally so we exclude it
  # from the explicit sweep.
  local ALL_ORDER=(compute bandwidth dram workload e2e multigpu validation plot score report)

  if [[ "$testcase" == "all" ]]; then
    echo "INFO: Running every testcase × ${workload}"
    for tc in "${ALL_ORDER[@]}"; do
      if [[ "$workload" == "all" ]]; then
        for wl in "${!WORKLOAD[@]}"; do
          run_benchmark "$tc" "$wl"
        done
      else
        if [[ -z ${WORKLOAD[$workload]} ]]; then
          echo "ERROR: Unknown workload '$workload'"
          exit 1
        fi
        run_benchmark "$tc" "$workload"
      fi
    done
  else
    if [[ -z ${TESTCASE[$testcase]} ]]; then
      echo "ERROR: Unknown testcase '$testcase'"
      list_options
      exit 1
    fi
    if [[ "$workload" == "all" ]]; then
      for wl in "${!WORKLOAD[@]}"; do
        run_benchmark "$testcase" "$wl"
      done
    else
      if [[ -z ${WORKLOAD[$workload]} ]]; then
        echo "ERROR: Unknown workload '$workload'"
        list_options
        exit 1
      fi
      run_benchmark "$testcase" "$workload"
    fi
  fi
}

main "$@" |& tee "$TEST_LOG"
MAIN_RC=${PIPESTATUS[0]}

echo "INFO: Test Log: $TEST_LOG"
# Best-effort copy to a stable name. If the per-invocation log is missing
# (e.g. $RESULTS_DIR was wiped mid-run by a concurrent cleanup), don't
# explode the wrapper — the per-iteration logs in results/_logs/ are still
# the source of truth.
if [[ -f "$TEST_LOG" ]]; then
  cp "$TEST_LOG" "$PWD/test.log"
else
  echo "WARN: $TEST_LOG missing — skipping copy to ./test.log"
fi
exit "$MAIN_RC"
