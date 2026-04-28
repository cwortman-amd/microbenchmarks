#!/usr/bin/env bash
# Single-node 8-GPU benchmark campaign orchestrator (TESTPLAN §15).
#
# Order matches the implementation priorities in TESTPLAN §15:
#   1) env capture
#   2) BF16 compute ceiling
#   3) HBM bandwidth ceiling
#   4) DRAM capacity
#   5) Per-op accounting
#   6) Roofline plot
#   7) E2E eager + compiled
#   8) MFU comparison
#   9) (optional) multi-GPU comm
# Then external validators (RVS, rocm-bandwidth-test, rccl-tests) and the
# cross-validation comparison.
#
# Exit status:
#   0 — all SC pass and validation has no FAIL rows
#   1 — any SC fail or any validation row FAIL
#   2 — fatal error before all benchmarks ran
#
# Usage:
#   bash scripts/run_campaign.sh [campaign_id]

set -uo pipefail

CAMPAIGN_ID="${1:-$(date -u +%Y%m%d-%H%M%S)}"
OUT="${RESULTS_DIR:-results}/${CAMPAIGN_ID}"
# Honor CONFIG from the parent (run.sh passes CONFIG_EFFECTIVE); default for standalone runs.
CONFIG="${CONFIG:-configs/escher_14b_480p.json}"

# --- GPU detection (sanitized) ------------------------------------------------
# `grep -c` exits 1 with output "0" when there are no matches, so a naive
# `grep -c | echo 8` produces "0\n8". Detect explicitly and clamp to 0/N.
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
NPROC="${NPROC:-$(detect_nproc)}"
[[ -z "$NPROC" || "$NPROC" =~ [^0-9] ]] && NPROC=0

# DEVICE classification: pick first matching family. Used to skip GPU-only
# benches cleanly when no accelerator is present (so the campaign can still
# run report/score over JSON outputs collected elsewhere).
if   [[ "$NPROC" -gt 0 ]] && command -v rocm-smi   >/dev/null 2>&1; then DEVICE="${DEVICE:-rocm}"
elif [[ "$NPROC" -gt 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then DEVICE="${DEVICE:-cuda}"
else                                                                    DEVICE="${DEVICE:-cpu}"
fi

mkdir -p "$OUT"
echo "[campaign] id=$CAMPAIGN_ID  out=$OUT  nproc=$NPROC  device=$DEVICE  config=$CONFIG"

# 1) env capture
python -m benchmarks.common.env --out "$OUT" --campaign-id "$CAMPAIGN_ID" \
  || { echo "[campaign] env capture failed"; exit 2; }

# Each bench detects the device internally and runs a CPU fallback when no
# accelerator is present, so the orchestrator no longer skips them by DEVICE.
# The benches return non-zero only on real failures, not on missing GPU.

# 2) BF16 compute ceiling
python -m benchmarks.bench01_bf16_compute  --out "$OUT" --config "$CONFIG" \
  || echo "[campaign] bench01_bf16_compute had errors (continuing)"
# 3) HBM bandwidth ceiling (system DRAM on CPU host)
python -m benchmarks.bench02_hbm_bandwidth --out "$OUT" \
  || echo "[campaign] bench02_hbm_bandwidth had errors (continuing)"
# 4) DRAM capacity (system DDR on CPU host) + post-model-load headroom
python -m benchmarks.bench03_dram_capacity --out "$OUT" \
  --measure-headroom --config "$CONFIG" \
  || echo "[campaign] bench03_dram_capacity had errors (continuing)"
# 5) Per-op accounting — analytic always; CPU adds lite measured timings
#    for ops below the FLOP budget.
python -m benchmarks.bench04_workload_ops --out "$OUT" --config "$CONFIG" \
  || echo "[campaign] bench04_workload_ops had errors (continuing)"
# 7+8) E2E + MFU (uses 01 + 04 outputs); auto-downscales on CPU.
python -m benchmarks.bench05_e2e_mfu       --out "$OUT" --config "$CONFIG" \
  || echo "[campaign] bench05_e2e_mfu had errors (continuing)"
# Numerical-stability sweep (bf16/fp16/fp8 vs fp32 across K). Cheap and
# device-agnostic; informs the report's reduced-precision section.
python -m benchmarks.bench09_numerical_stability --out "$OUT" \
  || echo "[campaign] bench09_numerical_stability had errors (continuing)"
# 9) Multi-GPU comm.
#    GPU path: one rank per GPU via NCCL/RCCL.
#    CPU path: the analogue is one rank per CCD (Infinity Fabric) or per
#      socket (xGMI/UPI). We auto-pick WORLD = #CCDs so the gloo collective
#      sweep actually exercises the inter-CCD interconnect rather than
#      memcpy within a single CCD. Override with `WORLD=N` and choose how
#      ranks map onto cores with `CPU_TOPOLOGY=ccd|socket|split|auto`.
CPU_TOPOLOGY="${CPU_TOPOLOGY:-auto}"
# torchrun spawns workers in fresh subprocesses; export PYTHONPATH so the
# `benchmarks` package resolves from CWD without requiring an editable install.
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$PWD"
if [[ "$NPROC" -gt 1 && "$DEVICE" != "cpu" ]]; then
  torchrun --nproc_per_node="$NPROC" benchmarks/bench06_multigpu_comm.py --out "$OUT" \
    || echo "[campaign] multi-GPU comm step had errors (continuing)"
  torchrun --nproc_per_node="$NPROC" benchmarks/bench06_fused.py --out "$OUT" \
    || echo "[campaign] multi-GPU fused-collective probe had errors (continuing)"
elif [[ "$DEVICE" == "cpu" ]]; then
  if [[ -z "${WORLD:-}" ]]; then
    WORLD=$(python -c "from benchmarks.common.topology import detect_cpu_topology as t; \
import os; topo=t(); \
dies = int(topo.get('dies') or 1); \
sockets = int(topo.get('sockets') or 1); \
print(max(1, min(max(dies, sockets), int(os.cpu_count() or 1))))" 2>/dev/null || echo 1)
  fi
  if [[ "${WORLD:-1}" -gt 1 ]]; then
    echo "[campaign] CPU multi-rank: WORLD=$WORLD CPU_TOPOLOGY=$CPU_TOPOLOGY"
    MICROBENCH_CPU_TOPOLOGY="$CPU_TOPOLOGY" \
      torchrun --nproc_per_node="${WORLD}" benchmarks/bench06_multigpu_comm.py \
        --out "$OUT" --cpu-topology "$CPU_TOPOLOGY" \
      || echo "[campaign] multi-rank gloo step had errors (continuing)"
    # Fused-kernel probe runs on CPU too — it just writes the
    # not-available stub since no fused-collective+GEMM kernel exists
    # for CPU. Keeps the SC-row consistent across hosts.
    MICROBENCH_CPU_TOPOLOGY="$CPU_TOPOLOGY" \
      torchrun --nproc_per_node="${WORLD}" benchmarks/bench06_fused.py \
        --out "$OUT" \
      || echo "[campaign] fused-collective probe had errors (continuing)"
  else
    echo "[campaign] bench06_multigpu_comm: skipped (single-CCD CPU host, WORLD=$WORLD)"
    # Still write the not-available stub so report.py and score_campaign.py
    # have a consistent input shape regardless of world size.
    python -m benchmarks.bench06_fused --out "$OUT" \
      || echo "[campaign] fused-collective stub write failed (continuing)"
  fi
else
  echo "[campaign] bench06_multigpu_comm: skipped (NPROC=$NPROC, DEVICE=$DEVICE)"
fi

# External validators (skip cleanly if tools missing)
if [[ "$DEVICE" != "cpu" ]]; then
  bash validation/rvs/run_rvs.sh "$OUT"
  bash validation/rocm_bw/run_rocm_bw.sh "$OUT"
  if [[ "$NPROC" -gt 1 ]]; then
    bash validation/rccl/run_rccl_tests.sh "$OUT" "$NPROC"
  fi
else
  echo "[campaign] external validators (rvs/rocm-bw/rccl): skipped (DEVICE=cpu)"
fi

# Cross-validate
python validation/compare.py --out "$OUT"
VAL_STATUS=$?

# 6) Plots
python scripts/plot_results.py --out "$OUT"

# Defensive: $OUT may have been pruned by an external cleanup between the
# initial mkdir and now (we've seen this in CI). Re-create before writing
# any of the post-processing artifacts so we don't crash the campaign.
mkdir -p "$OUT"

# summary.md is required artifact A11 in TESTPLAN §13, so write it BEFORE
# score_campaign.py (which checks for A11) runs. Otherwise SC-5 always reports
# A11 missing even when the campaign successfully completes.
cat > "$OUT/summary.md" <<EOF
# Campaign $CAMPAIGN_ID — summary

See:

- env: \`env.json\`
- compute: \`01_bf16_compute/summary.json\`
- bandwidth: \`02_hbm_bandwidth/summary.json\`
- DRAM: \`03_dram_capacity/summary.json\`
- per-op: \`04_workload_ops/ops.md\`
- E2E + MFU: \`05_e2e_mfu/mfu.md\`
- multi-GPU: \`06_multigpu_comm/comm.csv\`
- cross-validation: \`validation.md\`
- charts: \`plots/\`

EOF

# Score against TESTPLAN §1.2 success criteria SC-1...SC-5
python scripts/score_campaign.py --out "$OUT"
SCORE_STATUS=$?

# Generate report (Markdown + HTML + PDF). PDF is auto-skipped if no
# wkhtmltopdf / LaTeX engine is present; the campaign continues either way.
python scripts/report.py --out "$OUT" --config "$CONFIG" \
  || echo "[campaign] report generation had errors (continuing)"

FINAL=0
if [[ "$DEVICE" == "cpu" ]]; then
  # CPU campaign produces real measurements (compute peak, DRAM bandwidth,
  # capacity, etc.) but the absolute thresholds in scorecard.md and the
  # external validator suites only make sense on the MI355X target.
  # Don't fail the campaign on threshold misses; the report still surfaces
  # numbers and SC rows for inspection.
  echo "[campaign] DEVICE=cpu — exit code 0 (CPU thresholds advisory only)"
elif [[ "$VAL_STATUS" -ne 0 || "$SCORE_STATUS" -ne 0 ]]; then
  FINAL=1
fi
echo "[campaign] done. device=$DEVICE validation=$VAL_STATUS score=$SCORE_STATUS exit=$FINAL"
exit $FINAL
