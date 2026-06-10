#!/usr/bin/env bash
# rocm-bandwidth-test runner — HBM bandwidth ground truth.
#
# RVS does not directly measure HBM bandwidth; rocm-bandwidth-test is the
# canonical AMD validation tool for it. The CLI changed across ROCm releases:
#
#   * Legacy (<= ROCm 6.x): flag-driven
#       -e   sweep all topologies (H2D, D2H, D2D)
#       -t   topology summary
#       -a   all sizes including ladder up to large transfers
#       -m   explicit transfer sizes
#
#   * Modern (ROCm 7.x, rocm-bandwidth-test 2.6+): subcommand-driven
#       `rocm-bandwidth-test plugin --run tb ...` (TransferBench engine).
#     The TransferBench engine ships as a standalone binary too, with an `hbm`
#     preset that directly reports per-GPU HBM bandwidth — the cleanest D2D /
#     HBM-roof ground truth — so we drive that when the modern CLI is detected.
#
# The cross-validator (validation/compare.py :: parse_rocm_bw) reads the
# resulting logs (legacy `<num> GB/s` rows and/or the TransferBench `hbm`
# table) and compares the max device bandwidth against the plateau reported by
# benchmarks/02_hbm_bandwidth.py.
#
# Usage:
#   bash validation/rocm_bw/run_rocm_bw.sh results/<benchmark-id>/

set -uo pipefail
OUT="${1:?usage: run_rocm_bw.sh <benchmark-out-dir>}"
OUT_DIR="${OUT}/validation/rocm_bw"
mkdir -p "$OUT_DIR"

BIN="rocm-bandwidth-test"
if ! command -v "${BIN}" >/dev/null 2>&1; then
  if [[ -x /opt/rocm/bin/rocm-bandwidth-test ]]; then
    BIN=/opt/rocm/bin/rocm-bandwidth-test
  else
    BIN=""
  fi
fi

# Detect CLI generation. The modern (2.6+) CLI advertises "SUBCOMMAND" in its
# top-level help; the legacy CLI is purely flag-driven.
NEW_CLI=0
if [[ -n "${BIN}" ]] && "${BIN}" --help 2>&1 | grep -qiE 'SUBCOMMAND'; then
  NEW_CLI=1
fi

run_legacy() {
  # Topology + all-sizes sweep (default device list = all GPUs)
  "${BIN}" -t > "$OUT_DIR/topology.log" 2>&1 || true
  "${BIN}" -a > "$OUT_DIR/all_sizes.log" 2>&1 || true
  # Targeted unidirectional D->D copies on each GPU (best HBM proxy)
  for gpu in $(seq 0 7); do
    "${BIN}" -e "$gpu" -m 1024,16384,65536,262144,1048576,4194304,16777216,67108864,268435456 \
      > "$OUT_DIR/d2d_gpu${gpu}.log" 2>&1 || true
  done
}

run_transferbench() {
  # `hbm` preset reports per-GPU MaxBw/AvgBw/MinBw (GB/s) — the HBM roof.
  local TB="$1"
  "${TB}" hbm > "$OUT_DIR/hbm.log" 2>&1 || true
}

if [[ "$NEW_CLI" -eq 0 && -n "${BIN}" ]]; then
  run_legacy
else
  # Modern CLI (or rocm-bandwidth-test absent): prefer the standalone
  # TransferBench `hbm` preset, which is the same engine the modern plugin uses.
  TB=""
  if command -v TransferBench >/dev/null 2>&1; then
    TB="$(command -v TransferBench)"
  elif [[ -x /usr/local/bin/TransferBench ]]; then
    TB=/usr/local/bin/TransferBench
  fi

  if [[ -n "${TB}" ]]; then
    run_transferbench "${TB}"
  elif [[ -n "${BIN}" ]]; then
    # Last resort: drive the bundled TransferBench plugin in legacy-output mode.
    # Requires a transfer config; generate one local HBM copy per GPU.
    CFG="$OUT_DIR/tb_hbm.cfg"
    {
      for gpu in $(seq 0 7); do
        # 1 transfer: GPU<gpu> reads + writes its own HBM (local copy).
        echo "1 (G${gpu}->G${gpu})"
      done
    } > "$CFG"
    "${BIN}" plugin --legacy --run tb "$CFG" > "$OUT_DIR/hbm.log" 2>&1 || true
  else
    echo "[rocm-bw] neither rocm-bandwidth-test nor TransferBench installed; skipping" \
      | tee "$OUT_DIR/SKIPPED.txt"
    exit 0
  fi
fi

echo "[rocm-bw] done -> $OUT_DIR"
