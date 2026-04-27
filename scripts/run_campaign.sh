#!/usr/bin/env bash
# Single-node 8-GPU benchmark campaign orchestrator (TESTPLAN §15).
#
# Order matches the implementation priorities in TESTPLAN §15:
#   1) env capture
#   2) BF16 compute ceiling
#   3) HBM bandwidth ceiling
#   4) VRAM capacity
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
CONFIG="configs/escher_14b_480p.json"

# --- GPU detection (sanitized) ------------------------------------------------
# `grep -c` exits 1 with output "0" when there are no matches, so a naive
# `grep -c | echo 8` produces "0\n8". Detect explicitly and clamp to 0/N.
detect_nproc() {
  local n
  n=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU')
  [[ -z "$n" || "$n" =~ [^0-9] ]] && n=0
  if [[ "$n" -le 0 ]]; then
    n=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
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
echo "[campaign] id=$CAMPAIGN_ID  out=$OUT  nproc=$NPROC  device=$DEVICE"

# 1) env capture
python -m benchmarks.common.env --out "$OUT" --campaign-id "$CAMPAIGN_ID" \
  || { echo "[campaign] env capture failed"; exit 2; }

run_or_skip() {
  # run_or_skip <label> <cmd...>
  # On a CPU host the GPU benches exit non-zero by design ("CUDA/HIP device
  # required"). Tag the skip explicitly instead of letting it look like a fault.
  local label="$1"; shift
  if [[ "$DEVICE" == "cpu" ]]; then
    echo "[campaign] $label: skipped (DEVICE=cpu)"
    return 0
  fi
  "$@"
}

# 2) BF16 compute ceiling
run_or_skip bench01_bf16_compute  python -m benchmarks.bench01_bf16_compute  --out "$OUT" --config "$CONFIG"
# 3) HBM bandwidth ceiling
run_or_skip bench02_hbm_bandwidth python -m benchmarks.bench02_hbm_bandwidth --out "$OUT"
# 4) VRAM capacity
run_or_skip bench03_vram_capacity python -m benchmarks.bench03_vram_capacity --out "$OUT"
# 5) Per-op accounting — analytic table is CPU-friendly; the python script
# itself decides whether to add measured timings (skipped on cpu).
python -m benchmarks.bench04_workload_ops --out "$OUT" --config "$CONFIG"
# 7+8) E2E + MFU (uses 01 + 04 outputs)
run_or_skip bench05_e2e_mfu       python -m benchmarks.bench05_e2e_mfu       --out "$OUT" --config "$CONFIG"
# 9) Multi-GPU (only if multiple GPUs)
if [[ "$NPROC" -gt 1 && "$DEVICE" != "cpu" ]]; then
  torchrun --nproc_per_node="$NPROC" benchmarks/bench06_multigpu_comm.py --out "$OUT" \
    || echo "[campaign] multi-GPU comm step had errors (continuing)"
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
- VRAM: \`03_vram_capacity/summary.json\`
- per-op: \`04_workload_ops/ops.md\`
- E2E + MFU: \`05_e2e_mfu/mfu.md\`
- multi-GPU: \`06_multigpu_comm/comm.csv\`
- cross-validation: \`validation.md\`
- charts: \`plots/\`

EOF

# Score against TESTPLAN §1.2 success criteria SC-1...SC-5
python scripts/score_campaign.py --out "$OUT"
SCORE_STATUS=$?

# Generate report (Markdown + HTML). Convert to PDF/PPTX with pandoc.
python scripts/report.py --out "$OUT" --config "$CONFIG" --format both \
  || echo "[campaign] report generation had errors (continuing)"

FINAL=0
if [[ "$DEVICE" == "cpu" ]]; then
  # Nothing GPU-bound ran, so the SC fail / validation skip rows are expected.
  # Don't fail the campaign just because there is no accelerator — the report
  # and score artifacts are still useful for inspecting the analytic accounting
  # path on CPU-only hosts (CI, dev laptops).
  echo "[campaign] DEVICE=cpu — exit code 0 (GPU steps intentionally skipped)"
elif [[ "$VAL_STATUS" -ne 0 || "$SCORE_STATUS" -ne 0 ]]; then
  FINAL=1
fi
echo "[campaign] done. device=$DEVICE validation=$VAL_STATUS score=$SCORE_STATUS exit=$FINAL"
exit $FINAL
