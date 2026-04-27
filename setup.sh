#!/bin/bash
export PROJECT="microbenchmarks"

if [[ -z $DEVICE ]] ; then
  if command -v rocm-smi &>/dev/null && rocm-smi --showuniqueid 2>/dev/null | grep -q "ID:"; then
    export DEVICE="rocm"
  elif command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | grep -q .; then
    export DEVICE="cuda"
  else
    export DEVICE="cpu"
    echo "Note: No AMD or NVIDIA GPU detected — running in CPU-only mode (peak/BW/MFU steps will skip)."
  fi
fi

# Ensure no environment is active
[ -n "$VIRTUAL_ENV" ] && deactivate
while [ -n "$CONDA_DEFAULT_ENV" ]; do conda deactivate ; done

echo "========================================"
echo "Setup: ${PROJECT}-${DEVICE}"
echo "========================================"
echo "Setup Python Virtual Environment"
VENV=".${PROJECT}-${DEVICE}-venv"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
[ -f .env ] && source .env

export WORKSPACE=${WORKSPACE:-"$HOME/workspace"}
export WORKDIR="$PWD"

mkdir -p runs

echo "========================================"
echo "Project Setup"

pip install -U pip wheel setuptools
pip install -r requirements.txt

echo "--- PyTorch ($DEVICE)"
case "$DEVICE" in
  rocm)
    # TESTPLAN §2.1 reference stack is the ROCm 7.x PyTorch wheels. Override
    # TORCH_INDEX_URL to pin a different ROCm channel (e.g. rocm6.2 or a
    # nightly index).
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}"
    echo "  using index: ${TORCH_INDEX_URL}"
    pip3 install torch torchvision --index-url "${TORCH_INDEX_URL}"
    ;;
  cuda)
    # Default PyPI wheel on Linux x86_64 already ships with CUDA runtime
    # bundled. Override TORCH_INDEX_URL only if you need a specific CUDA
    # channel (e.g. https://download.pytorch.org/whl/cu124 for cu12.4).
    if [[ -n "${TORCH_INDEX_URL}" ]]; then
      echo "  using index: ${TORCH_INDEX_URL}"
      pip3 install torch torchvision --index-url "${TORCH_INDEX_URL}"
    else
      pip3 install torch torchvision
    fi
    ;;
  cpu)
    # Pull the CPU-only wheels from PyTorch's CPU index. The default PyPI
    # `torch` wheel on Linux x86_64 ships CUDA runtime libs (~2 GB) which we
    # don't want when no GPU is present and which can confuse runtime device
    # autodetect. Override with TORCH_INDEX_URL when you need a specific CPU
    # build (e.g. a nightly).
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    echo "  using index: ${TORCH_INDEX_URL}"
    pip3 install torch torchvision --index-url "${TORCH_INDEX_URL}"
    ;;
esac

echo "--- Optional attention backends (probed, not required)"
python3 - <<'PY'
def probe(mod):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod:12s} present (v{v})")
    except Exception as e:
        print(f"  {mod:12s} not present ({type(e).__name__})")
for m in ("torch", "triton", "aiter", "flash_attn"):
    probe(m)
PY

echo "--- ROCm tools (validation/ground truth)"
for tool in rocm-smi rocminfo rocm-bandwidth-test rvs amd-smi; do
  if command -v "$tool" &>/dev/null; then
    echo "  $tool: $(command -v $tool)"
  elif [ -x "/opt/rocm/bin/$tool" ]; then
    echo "  $tool: /opt/rocm/bin/$tool — adding /opt/rocm/bin to PATH"
    export PATH="/opt/rocm/bin:$PATH"
  else
    echo "  $tool: not found (cross-validation rows depending on it will SKIP)"
  fi
done

if [ -d "/opt/rocm/bin" ] && ! grep -q "/opt/rocm/bin" "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="/opt/rocm/bin:$PATH"' >> "$HOME/.bashrc"
  echo "  Added /opt/rocm/bin to ~/.bashrc PATH"
fi

echo "--- ROCm Validation Suite (rvs)"
if command -v rvs &>/dev/null || [ -x "/opt/rocm/bin/rvs" ]; then
  echo "  rvs already installed"
elif [[ "$DEVICE" == "rocm" ]]; then
  echo "  rvs missing — attempting apt install of rocm-validation-suite ..."
  sudo apt-get install -y -qq rocm-validation-suite 2>/dev/null || \
    echo "  apt install failed — install manually from https://github.com/ROCm/ROCmValidationSuite"
fi

echo "--- rccl-tests (multi-GPU collective ground truth)"
RCCL_TESTS_SRC="${RCCL_TESTS_SRC:-$HOME/.cache/rccl-tests}"
if command -v all_reduce_perf &>/dev/null; then
  echo "  rccl-tests: $(command -v all_reduce_perf)"
elif [ -x "${RCCL_TESTS_SRC}/build/all_reduce_perf" ]; then
  echo "  rccl-tests: ${RCCL_TESTS_SRC}/build"
  export RCCL_TESTS_DIR="${RCCL_TESTS_SRC}/build"
elif [[ "$DEVICE" == "rocm" ]]; then
  echo "  rccl-tests: not found — cloning + building (MPI=0) ..."
  if [ ! -d "$RCCL_TESTS_SRC" ]; then
    git clone --depth=1 https://github.com/ROCm/rccl-tests "$RCCL_TESTS_SRC" 2>/dev/null || \
      echo "  git clone failed — multi-GPU validation will SKIP rccl rows"
  fi
  if [ -d "$RCCL_TESTS_SRC" ]; then
    (cd "$RCCL_TESTS_SRC" && make MPI=0 HIP_HOME=/opt/rocm -j"$(nproc)") 2>&1 | tail -5
    if [ -x "${RCCL_TESTS_SRC}/build/all_reduce_perf" ]; then
      echo "  rccl-tests built at ${RCCL_TESTS_SRC}/build"
      export RCCL_TESTS_DIR="${RCCL_TESTS_SRC}/build"
      if ! grep -q "RCCL_TESTS_DIR" "$HOME/.bashrc" 2>/dev/null; then
        echo "export RCCL_TESTS_DIR=${RCCL_TESTS_SRC}/build" >> "$HOME/.bashrc"
      fi
    else
      echo "  rccl-tests build failed — multi-GPU validation will SKIP rccl rows"
    fi
  fi
fi

echo "========================================"
echo "Checking Installation"
python3 -V
pip check || true

DEVICE="$DEVICE" python3 - <<'PY'
import os
import torch
device = os.environ.get("DEVICE", "cpu")
print(f"torch:           {torch.__version__}")
print(f"  cuda/HIP avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device count:   {torch.cuda.device_count()}")
    print(f"  device 0:       {torch.cuda.get_device_name(0)}")
    print(f"  HIP version:    {getattr(torch.version, 'hip', None)}")
    print(f"  CUDA version:   {getattr(torch.version, 'cuda', None)}")
elif device == "cpu":
    print("  CPU-only build (expected for DEVICE=cpu).")
    print(f"  threads:        {torch.get_num_threads()}")
    print(f"  CPU caps:       {torch.backends.cpu.get_cpu_capability()}")
else:
    print(f"  WARNING: DEVICE={device} but torch.cuda.is_available() is False.")
    print( "  This usually means PyTorch could not initialize the GPU runtime.")
    print( "  Re-run with DEVICE=cpu, or fix the driver/library before benchmarking.")
PY

echo "--- escher_14b_480p analytic FLOP/byte calibration drift"
python3 - <<'PY'
import json
from benchmarks.common.flop_accounting import WorkloadConfig, per_block_ops, totals
cfg = json.load(open("configs/escher_14b_480p.json"))
ops = per_block_ops(WorkloadConfig.from_json(cfg))
t = totals(ops)
ref = cfg["reference_totals_per_block"]
print(f"  computed/block: {t['total_gflops']:.1f} GFLOPs / {t['total_mb_hbm']:.1f} MB")
print(f"  reference:      {ref['gflops']:.1f} GFLOPs / {ref['hbm_mb']:.1f} MB")
print(f"  drift:          GFLOPs {(t['total_gflops']/ref['gflops']-1)*100:+.1f}%  HBM {(t['total_mb_hbm']/ref['hbm_mb']-1)*100:+.1f}%")
PY

echo "--- Validation tools summary"
for tool in rvs rocm-bandwidth-test all_reduce_perf; do
  if command -v "$tool" &>/dev/null; then
    echo "  $tool: AVAILABLE"
  elif [ "$tool" = "all_reduce_perf" ] && [ -n "${RCCL_TESTS_DIR}" ] && [ -x "${RCCL_TESTS_DIR}/$tool" ]; then
    echo "  $tool: AVAILABLE (${RCCL_TESTS_DIR})"
  else
    echo "  $tool: missing — that cross-validation row will SKIP, not fail"
  fi
done

echo "========================================"
echo "Environment setup completed."
echo "========================================"
case "$DEVICE" in
  rocm|cuda)
    echo "Next:"
    echo "  source ${VENV}/bin/activate"
    echo "  bash scripts/run_campaign.sh"
    ;;
  cpu)
    echo "Note: CPU-only environment."
    echo "  GPU-bound testcases will exit early on this host:"
    echo "    bench01_bf16_compute, bench02_hbm_bandwidth, bench03_dram_capacity,"
    echo "    bench05_e2e_mfu, bench06_multigpu_comm,"
    echo "    validation/rvs/*, validation/rccl/*, validation/rocm_bw/*"
    echo "  These still work without a GPU:"
    echo "    bench04_workload_ops    (analytic FLOP/byte accounting)"
    echo "    scripts/score_campaign.py / scripts/report.py / scripts/plot_results.py"
    echo "    (over JSON outputs collected on a GPU host)"
    echo
    echo "Next:"
    echo "  source ${VENV}/bin/activate"
    echo "  python -m benchmarks.bench04_workload_ops --out results/cpu-smoke --config configs/escher_14b_480p.json"
    ;;
esac
echo "========================================"
