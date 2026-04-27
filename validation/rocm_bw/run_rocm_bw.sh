#!/usr/bin/env bash
# rocm-bandwidth-test runner — HBM bandwidth ground truth.
#
# RVS does not directly measure HBM bandwidth; rocm-bandwidth-test is the
# canonical AMD validation tool for it. We run:
#   -e   sweep all topologies (H2D, D2H, D2D)
#   -t   topology summary
#   -a   all sizes including ladder up to large transfers
#
# The cross-validator (validation/compare.py) reads the device-to-device and
# host-to-device numbers and compares them against the plateau reported by
# benchmarks/02_hbm_bandwidth.py.
#
# Usage:
#   bash validation/rocm_bw/run_rocm_bw.sh results/<campaign-id>/

set -uo pipefail
OUT="${1:?usage: run_rocm_bw.sh <campaign-out-dir>}"
OUT_DIR="${OUT}/validation/rocm_bw"
mkdir -p "$OUT_DIR"

BIN="rocm-bandwidth-test"
if ! command -v "${BIN}" >/dev/null 2>&1; then
  if [[ -x /opt/rocm/bin/rocm-bandwidth-test ]]; then
    BIN=/opt/rocm/bin/rocm-bandwidth-test
  else
    echo "[rocm-bw] rocm-bandwidth-test not installed; skipping" | tee "$OUT_DIR/SKIPPED.txt"
    exit 0
  fi
fi

# Topology + all-sizes sweep (default device list = all GPUs)
"${BIN}" -t > "$OUT_DIR/topology.log" 2>&1 || true
"${BIN}" -a > "$OUT_DIR/all_sizes.log" 2>&1 || true

# Targeted unidirectional D->D copies on each GPU (best HBM proxy)
for gpu in $(seq 0 7); do
  "${BIN}" -e "$gpu" -m 1024,16384,65536,262144,1048576,4194304,16777216,67108864,268435456 \
    > "$OUT_DIR/d2d_gpu${gpu}.log" 2>&1 || true
done

echo "[rocm-bw] done -> $OUT_DIR"
