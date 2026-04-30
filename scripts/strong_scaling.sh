#!/usr/bin/env bash
# Strong-scaling sweep across WORLD ∈ {2, 4, 8}  (TESTPLAN §16.3).
#
# Drives `run_benchmark.sh` once per world size, collects each benchmark
# under its own subdirectory, then invokes `scripts/strong_scaling_table.py`
# to roll the per-world artifacts up into the TP-3 table the testplan
# specifies.
#
# Usage:
#   bash scripts/strong_scaling.sh                    # default worlds: 2 4 8
#   WORLDS="2 4 8" bash scripts/strong_scaling.sh     # explicit set
#   WORLDS="1 2 4 8" SWEEP_ID=foo bash scripts/strong_scaling.sh
#
# Each per-world benchmark writes to:
#   results/<sweep_id>/world_<N>/
# The aggregated table goes to:
#   results/<sweep_id>/tp3_table.{md,json,csv}
#
# Notes:
#   * On GPU hosts, NPROC is overridden per world so torchrun launches the
#     right number of ranks. CUDA_VISIBLE_DEVICES is not set here — if the
#     host has more GPUs than the largest WORLD, the lower-numbered GPUs
#     are used. Set CUDA_VISIBLE_DEVICES upstream to control pinning.
#   * On CPU hosts, WORLD is passed through to the benchmark's CPU
#     multi-rank path (gloo + sched_setaffinity per CCD/socket). Hosts
#     with fewer CCDs than the requested WORLD will fall back to the
#     `split` topology mode (see bench12).
#
# Exit status:
#   0 — every per-world benchmark exited 0 AND the aggregator wrote a table
#   1 — at least one per-world benchmark failed; table still attempted

set -uo pipefail

WORLDS="${WORLDS:-2 4 8}"
SWEEP_ID="${SWEEP_ID:-$(date -u +%Y%m%d-%H%M%S)-strong-scaling}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
SWEEP_DIR="${RESULTS_ROOT}/${SWEEP_ID}"
CONFIG="${CONFIG:-configs/escher_14b_480p.json}"
COMM_ONLY="${COMM_ONLY:-0}"

mkdir -p "$SWEEP_DIR"
echo "[strong-scaling] sweep_id=$SWEEP_ID worlds={$WORLDS} dir=$SWEEP_DIR"

OVERALL_STATUS=0
SUCCESSFUL_WORLDS=()

for W in $WORLDS; do
  if ! [[ "$W" =~ ^[0-9]+$ ]]; then
    echo "[strong-scaling] skipping non-integer world: $W"
    continue
  fi
  BENCHMARK_ID="world_${W}"
  BENCHMARK_DIR="${SWEEP_DIR}/${BENCHMARK_ID}"
  echo "[strong-scaling] === WORLD=$W -> $BENCHMARK_DIR ==="

  if [[ "$COMM_ONLY" == "1" ]]; then
    # Lighter mode: skip the full benchmark and just run the multi-rank
    # comm sweep. Useful when we only want the TP-3 collective rows
    # without the 30-min per-world bench05/bench04 overhead.
    mkdir -p "$BENCHMARK_DIR"
    python -m benchmarks.common.env --out "$BENCHMARK_DIR" --benchmark-id "$BENCHMARK_ID" \
      || echo "[strong-scaling] env capture failed for WORLD=$W (continuing)"
    if command -v torchrun >/dev/null 2>&1 && [[ "$W" -gt 1 ]]; then
      MASTER_PORT_W="${MASTER_PORT_BASE:-29550}"
      MASTER_PORT_W="$((MASTER_PORT_W + W))"
      PYTHONPATH="${PYTHONPATH:-.}:." WORLD="$W" \
        torchrun --nproc_per_node="$W" \
        --master_addr=127.0.0.1 --master_port="$MASTER_PORT_W" \
        benchmarks/bench12_multigpu_comm.py --out "$BENCHMARK_DIR" \
          --cpu-topology "${CPU_TOPOLOGY:-auto}"
      RC=$?
    else
      echo "[strong-scaling] WORLD=$W needs torchrun and W>1; skipping"
      RC=2
    fi
  else
    NPROC="$W" WORLD="$W" RESULTS_DIR="$SWEEP_DIR" \
      bash scripts/run_benchmark.sh "$BENCHMARK_ID"
    RC=$?
  fi

  if [[ "$RC" -eq 0 ]]; then
    SUCCESSFUL_WORLDS+=("$W")
  else
    echo "[strong-scaling] WORLD=$W exited $RC"
    OVERALL_STATUS=1
  fi
done

echo "[strong-scaling] === aggregating TP-3 table ==="
python scripts/strong_scaling_table.py --sweep-dir "$SWEEP_DIR" \
  || { echo "[strong-scaling] table generation failed"; OVERALL_STATUS=1; }

echo "[strong-scaling] done. successful_worlds=${SUCCESSFUL_WORLDS[*]:-} overall=$OVERALL_STATUS"
exit "$OVERALL_STATUS"
