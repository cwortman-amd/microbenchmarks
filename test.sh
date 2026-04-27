#!/bin/bash
# microbenchmarks Unified Test Runner
# Executes benchmark families and workload variants for the
# escher_14b_480p / MI355X campaign described in TESTPLAN.md.

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
#   CAMPAIGN   -> bash scripts/run_campaign.sh <id>
#   VALIDATE   -> external validators only (RVS, rocm-bw, rccl-tests, compare)
#   REPORT     -> python scripts/report.py
#   SCORE      -> python scripts/score_campaign.py
#   PLOT       -> python scripts/plot_results.py
# DIST=1 marks the testcase as multi-process (torchrun -nproc_per_node=$NPROC).

declare -A TESTCASE
TESTCASE[compute]="DESC: Family 1 — BF16 compute peak + size sweep; SCRIPT: bench01_bf16_compute"
TESTCASE[bandwidth]="DESC: Family 2 — HBM streaming microbenchmarks; SCRIPT: bench02_hbm_bandwidth"
TESTCASE[vram]="DESC: Family 3 — VRAM capacity binary search; SCRIPT: bench03_vram_capacity"
TESTCASE[workload]="DESC: Family 4 — escher_14b_480p per-op decomposition; SCRIPT: bench04_workload_ops"
TESTCASE[e2e]="DESC: Family 5 — eager + torch.compile MFU; SCRIPT: bench05_e2e_mfu"
TESTCASE[multigpu]="DESC: Family 6 — RCCL collectives via torchrun; SCRIPT: bench06_multigpu_comm; DIST: 1"
TESTCASE[validation]="DESC: External validators (RVS, rocm-bandwidth-test, rccl-tests, compare); SCRIPT: VALIDATE"
TESTCASE[plot]="DESC: Regenerate plots A2/A3/A6/A7/A8; SCRIPT: PLOT"
TESTCASE[score]="DESC: Score SC-1…SC-5; SCRIPT: SCORE"
TESTCASE[report]="DESC: Build report.md / report.html; SCRIPT: REPORT"
TESTCASE[campaign]="DESC: Full single-node campaign (TESTPLAN §15 order); SCRIPT: CAMPAIGN"

# --- Workload (config) definitions --------------------------------------------
# Each workload pins a config file and optional shape/depth overrides.
# Empty override fields keep the value from the config file unchanged.

declare -A WORKLOAD
WORKLOAD[escher_14b_480p]="CONFIG: configs/escher_14b_480p.json; DEPTH: ; SEQ_IMG: ; SEQ_TXT: "
WORKLOAD[smoke]="CONFIG: configs/escher_14b_480p.json; DEPTH: 4; SEQ_IMG: 2048; SEQ_TXT: 256"
WORKLOAD[big]="CONFIG: configs/escher_14b_480p.json; DEPTH: 60; SEQ_IMG: 16384; SEQ_TXT: 1024"

# --- Environment --------------------------------------------------------------
RESULTS_DIR=${RESULTS_DIR:-"$PWD/results"}
LOG_DIR=${LOG_DIR:-"$RESULTS_DIR/_logs"}
export RESULTS_DIR LOG_DIR
mkdir -p "$LOG_DIR"

DATETIME=$(date +'%Y%m%d-%H%M%S')
TEST_LOG=$LOG_DIR/test.${DATETIME}.log
ITERATIONS=${ITERATIONS:-"1"}
detect_nproc() {
  local n
  n=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU')
  [[ -z "$n" || "$n" =~ [^0-9] ]] && n=0
  if [[ "$n" -le 0 ]]; then
    n=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
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
  -t, --testcase NAME      run a single testcase (default: campaign)
  -w, --workload NAME      run with a single workload spec (default: escher_14b_480p)
  -i, --iterations N       repeat each (testcase, workload) N times (default: 1)
  -a, --all                run every testcase × every workload
  -l, --list               list available testcases and workloads, then exit
  -h, --help               this message

Environment:
  DEVICE        rocm | cuda | cpu                (default: rocm)
  NPROC         GPU count for multi-GPU step    (default: auto-detected)
  RESULTS_DIR   campaign root                   (default: ./results)
  ITERATIONS    repeat count                    (default: 1)

Examples:
  $0 -t compute                       # only BF16 GEMM
  $0 -t e2e -w smoke                  # quick MFU smoke test
  $0 -t campaign -i 3                 # full campaign 3× (regression averaging)
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
# enumerating combinations and ordering the campaign.
run_benchmark() {
  local testcase=$1
  local workload=$2

  local desc=$(parse_kv "${TESTCASE[$testcase]}" DESC)
  local dist=$(parse_kv "${TESTCASE[$testcase]}" DIST)
  local base_cfg=$(parse_kv "${WORKLOAD[$workload]}" CONFIG)
  local depth=$(parse_kv "${WORKLOAD[$workload]}" DEPTH)
  local seq_img=$(parse_kv "${WORKLOAD[$workload]}" SEQ_IMG)
  local seq_txt=$(parse_kv "${WORKLOAD[$workload]}" SEQ_TXT)
  local out_id="${DATETIME}-${testcase}-${workload}"

  echo "================================================"
  echo "Testcase: $testcase ($desc)"
  echo "Workload: $workload (config=$base_cfg)"
  echo "Iters:    $ITERATIONS    NPROC=$NPROC    out_id=$out_id"
  echo "================================================"

  local cmd=(bash run.sh
    --testcase "$testcase"
    --workload "$workload"
    --config   "$base_cfg"
    --out-id   "$out_id"
    --campaign-id "$DATETIME"
    --iterations "$ITERATIONS"
    --nproc      "$NPROC"
    --dist       "${dist:-0}"
  )
  [[ -n "$depth"   ]] && cmd+=(--depth   "$depth")
  [[ -n "$seq_img" ]] && cmd+=(--seq-img "$seq_img")
  [[ -n "$seq_txt" ]] && cmd+=(--seq-txt "$seq_txt")

  echo "INFO: ${cmd[*]}"
  "${cmd[@]}"
}

# --- Main ---------------------------------------------------------------------
main() {
  local testcase="campaign"
  local workload="escher_14b_480p"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--testcase) testcase="$2"; shift 2 ;;
      -w|--workload) workload="$2"; shift 2 ;;
      -i|--iterations) ITERATIONS="$2"; shift 2 ;;
      -a|--all) testcase="all"; workload="all"; shift ;;
      -l|--list) list_options; exit 0 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "ERROR: Unknown option $1"; usage; exit 1 ;;
    esac
  done

  # Stable execution order matching TESTPLAN §15 (each ceiling anchored before
  # the metrics derived from it). When --all is requested we run every testcase
  # in this order; "campaign" already does this internally so we exclude it
  # from the explicit sweep.
  local ALL_ORDER=(compute bandwidth vram workload e2e multigpu validation plot score report)

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
