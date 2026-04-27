#!/usr/bin/env bash
# rccl-tests runner — RCCL collective ground truth.
#
# Cross-validates benchmarks/06_multigpu_comm.py. Same payload range, same
# op kinds (all_gather, reduce_scatter, all_reduce). rccl-tests reports
# algbw / busbw which our PyTorch script also computes, so the comparison
# is apples-to-apples.
#
# Expects rccl-tests built and on PATH (or pointed to by RCCL_TESTS_DIR).
# Build instructions: https://github.com/ROCm/rccl-tests
#
# Usage:
#   bash validation/rccl/run_rccl_tests.sh results/<campaign-id>/ [world]
# Default world = $(rocm-smi --showid | wc -l) or 8.

set -uo pipefail
OUT="${1:?usage: run_rccl_tests.sh <campaign-out-dir> [world]}"
WORLD="${2:-8}"
OUT_DIR="${OUT}/validation/rccl"
mkdir -p "$OUT_DIR"

# Locate rccl-tests binaries.
RCCL_TESTS_DIR="${RCCL_TESTS_DIR:-/opt/rccl-tests/build}"
find_bin() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then echo "$(command -v "$name")"; return; fi
  if [[ -x "$RCCL_TESTS_DIR/$name" ]]; then echo "$RCCL_TESTS_DIR/$name"; return; fi
  echo ""
}

ALLREDUCE=$(find_bin all_reduce_perf)
ALLGATHER=$(find_bin all_gather_perf)
REDUCESCATTER=$(find_bin reduce_scatter_perf)

if [[ -z "$ALLREDUCE$ALLGATHER$REDUCESCATTER" ]]; then
  echo "[rccl-tests] not found on PATH or in $RCCL_TESTS_DIR ; skipping" \
    | tee "$OUT_DIR/SKIPPED.txt"
  exit 0
fi

# Common args: bf16 dtype (-d), -b begin size, -e end size, -f factor, -g GPUs, -n iters, -w warmup
COMMON=(-b 1M -e 1G -f 4 -d half -g "$WORLD" -n 20 -w 5)

run_one() {
  local bin="$1" label="$2"
  if [[ -z "$bin" ]]; then
    echo "[rccl-tests] missing $label binary; skipping"
    return
  fi
  echo "[rccl-tests] running $label ..."
  "$bin" "${COMMON[@]}" > "$OUT_DIR/${label}.log" 2>&1 || \
    echo "[rccl-tests] $label exited non-zero (see $OUT_DIR/${label}.log)"
}

run_one "$ALLREDUCE"      all_reduce
run_one "$ALLGATHER"      all_gather
run_one "$REDUCESCATTER"  reduce_scatter

echo "[rccl-tests] done -> $OUT_DIR"
