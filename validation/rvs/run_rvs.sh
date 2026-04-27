#!/usr/bin/env bash
# Runs the three RVS configurations and stores raw logs under
# <out>/validation/rvs/.
#
# Usage:
#   bash validation/rvs/run_rvs.sh results/<campaign-id>/

set -uo pipefail
OUT="${1:?usage: run_rvs.sh <campaign-out-dir>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT}/validation/rvs"
mkdir -p "$OUT_DIR"

if ! command -v rvs >/dev/null 2>&1; then
  echo "[rvs] rvs not on PATH; trying /opt/rocm/bin/rvs"
  RVS=/opt/rocm/bin/rvs
else
  RVS=rvs
fi

if [[ ! -x "${RVS}" && ! "$(command -v "${RVS}")" ]]; then
  echo "[rvs] rvs not installed; skipping (install with the ROCm rocm-validation-suite package)" \
    | tee "$OUT_DIR/SKIPPED.txt"
  exit 0
fi

run_one() {
  local conf="$1"
  local label="$2"
  echo "[rvs] running $label ..."
  "${RVS}" -c "$DIR/$conf" -d 3 -j "$OUT_DIR/${label}.json" \
    > "$OUT_DIR/${label}.stdout.log" 2> "$OUT_DIR/${label}.stderr.log" || \
    echo "[rvs] $label exited non-zero (see $OUT_DIR/${label}.stderr.log)"
}

run_one gst_bf16_mi355x.conf gst_bf16
run_one mem_mi355x.conf       mem
run_one pebb_pcie.conf        pebb

echo "[rvs] done -> $OUT_DIR"
