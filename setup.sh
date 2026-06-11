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

echo "--- AITER (optional ROCm attention backend)"
# Defaults:
#   DEVICE=rocm -> attempt install when import is missing
#   DEVICE!=rocm -> skip unless AITER_INSTALL=1
# Overrides:
#   AITER_INSTALL=1   force install attempt
#   AITER_INSTALL=0   disable install attempt
#   AITER_SRC=<path>  clone/build directory (default: ~/.cache/aiter)
AITER_SRC="${AITER_SRC:-$HOME/.cache/aiter}"
_aiter_present=0
python3 - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("aiter") else 1)
PY
if [[ $? -eq 0 ]]; then
  _aiter_present=1
fi

_want_aiter=0
case "${AITER_INSTALL:-auto}" in
  1|true|yes|on) _want_aiter=1 ;;
  0|false|no|off) _want_aiter=0 ;;
  auto|AUTO|"")
    [[ "$DEVICE" == "rocm" ]] && _want_aiter=1
    ;;
  *)
    echo "  AITER_INSTALL='${AITER_INSTALL}' not recognized; treating as auto"
    [[ "$DEVICE" == "rocm" ]] && _want_aiter=1
    ;;
esac

if [[ "$_aiter_present" -eq 1 ]]; then
  echo "  aiter already importable in this venv"
elif [[ "$_want_aiter" -eq 1 ]]; then
  echo "  aiter missing — cloning/building from ROCm/aiter ..."
  if [[ ! -d "$AITER_SRC/.git" ]]; then
    mkdir -p "$(dirname "$AITER_SRC")"
    git clone --recursive https://github.com/ROCm/aiter.git "$AITER_SRC" 2>/dev/null \
      || echo "  git clone failed — continuing without aiter"
  else
    (cd "$AITER_SRC" && git submodule update --init --recursive) 2>/dev/null \
      || echo "  submodule update failed — continuing with existing checkout"
  fi
  if [[ -d "$AITER_SRC" ]]; then
    # Use pip install -e (not setup.py develop) so the install targets the
    # active venv rather than system site-packages.
    pip install -e "$AITER_SRC" 2>&1 | tail -5 \
      || echo "  aiter build/install failed — continuing without aiter"
  fi
fi

echo "--- Iris (GPU-initiated multi-GPU comm — fused AG+MM / MM+RS Iris path)"
# Iris (ROCm/iris) is the symmetric-memory RMA layer that powers the fused
# collective+GEMM kernels exercised by bench06_aiter_fused and
# bench13_iris_overlap. Without it those benches silently fall back to the
# staged (non-overlapped) path, so the Phase 1/2/3 fused kernels never run and
# the fused-vs-unfused comparison cannot show their speedup. Pure Python +
# Triton; needs ROCm 6.3.1+, Python 3.10+, Triton (MI300X/MI350X/MI355X).
# Defaults mirror the AITER block:
#   DEVICE=rocm  -> attempt install when import is missing
#   DEVICE!=rocm -> skip unless IRIS_INSTALL=1
# Overrides:
#   IRIS_INSTALL=1   force install attempt
#   IRIS_INSTALL=0   disable install attempt
#   IRIS_SRC=<path>  clone dir for the editable install (default: ~/.cache/iris)
IRIS_SRC="${IRIS_SRC:-$HOME/.cache/iris}"
_iris_present=0
python3 - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("iris") else 1)
PY
if [[ $? -eq 0 ]]; then
  _iris_present=1
fi

_want_iris=0
case "${IRIS_INSTALL:-auto}" in
  1|true|yes|on) _want_iris=1 ;;
  0|false|no|off) _want_iris=0 ;;
  auto|AUTO|"")
    [[ "$DEVICE" == "rocm" ]] && _want_iris=1
    ;;
  *)
    echo "  IRIS_INSTALL='${IRIS_INSTALL}' not recognized; treating as auto"
    [[ "$DEVICE" == "rocm" ]] && _want_iris=1
    ;;
esac

if [[ "$_iris_present" -eq 1 ]]; then
  echo "  iris already importable in this venv"
elif [[ "$_want_iris" -eq 1 ]]; then
  echo "  iris missing — cloning/installing from ROCm/iris ..."
  if [[ ! -d "$IRIS_SRC/.git" ]]; then
    mkdir -p "$(dirname "$IRIS_SRC")"
    git clone https://github.com/ROCm/iris.git "$IRIS_SRC" 2>/dev/null \
      || echo "  git clone failed — will try a direct pip install from git"
  else
    (cd "$IRIS_SRC" && git pull --ff-only) 2>/dev/null \
      || echo "  git pull failed — continuing with existing checkout"
  fi
  if [[ -d "$IRIS_SRC/.git" ]]; then
    # Editable install (matches the aiter pattern) so the checkout can be
    # hacked on locally; targets the active venv, not system site-packages.
    pip install -e "$IRIS_SRC" 2>&1 | tail -5 \
      || echo "  iris editable install failed — fused Iris path will fall back to staged"
  else
    # Clone failed (offline/proxy) — last-ditch direct install from git.
    pip install "git+https://github.com/ROCm/iris.git" 2>&1 | tail -5 \
      || echo "  iris pip install failed — fused Iris path will fall back to staged"
  fi
else
  echo "  skipped (set IRIS_INSTALL=1 to force install attempt)"
fi

echo "--- HipKittens (experimental CDNA4 AITER backend)"
# HipKittens provides the CDNA3/CDNA4 tile/MFMA primitives and distributed Iris
# prototype we use for the next AITER-managed backend. The source checkout is
# useful immediately for kernel development; compiling the distributed example
# is opt-in because it requires MPI, CMake, pybind11, hipcc, and may fetch Iris
# through CMake.
# Defaults:
#   DEVICE=rocm  -> clone/update when missing
#   DEVICE!=rocm -> skip unless HIPKITTENS_INSTALL=1
# Overrides:
#   HIPKITTENS_INSTALL=1      force clone/update
#   HIPKITTENS_INSTALL=0      disable clone/update
#   HIPKITTENS_SRC=<path>     checkout directory (default: ~/.cache/HipKittens)
#   HIPKITTENS_BUILD_DK=1     build distributed-kernels/bf16_gemm prototype
#   HIPKITTENS_BUILD_FUSED=1  build this repo's hk_iris_fused native prototype
HIPKITTENS_SRC="${HIPKITTENS_SRC:-$HOME/.cache/HipKittens}"
_want_hk=0
case "${HIPKITTENS_INSTALL:-auto}" in
  1|true|yes|on) _want_hk=1 ;;
  0|false|no|off) _want_hk=0 ;;
  auto|AUTO|"")
    [[ "$DEVICE" == "rocm" ]] && _want_hk=1
    ;;
  *)
    echo "  HIPKITTENS_INSTALL='${HIPKITTENS_INSTALL}' not recognized; treating as auto"
    [[ "$DEVICE" == "rocm" ]] && _want_hk=1
    ;;
esac

if [[ "$_want_hk" -eq 1 ]]; then
  if [[ ! -d "$HIPKITTENS_SRC/.git" ]]; then
    echo "  cloning HipKittens into ${HIPKITTENS_SRC} ..."
    mkdir -p "$(dirname "$HIPKITTENS_SRC")"
    git clone https://github.com/HazyResearch/HipKittens.git "$HIPKITTENS_SRC" 2>/dev/null \
      || echo "  HipKittens clone failed — set HIPKITTENS_SRC to an existing checkout"
  else
    echo "  updating HipKittens checkout at ${HIPKITTENS_SRC} ..."
    (cd "$HIPKITTENS_SRC" && git pull --ff-only) 2>/dev/null \
      || echo "  HipKittens git pull failed — continuing with existing checkout"
  fi
  if [[ -d "$HIPKITTENS_SRC" ]]; then
    pip install pybind11 mpi4py 2>&1 | tail -5 \
      || echo "  pybind11/mpi4py install failed — HK distributed build may fail"
    if ! grep -q "HIPKITTENS_SRC=" "$VENV/bin/activate"; then
      echo "export HIPKITTENS_SRC=\"$HIPKITTENS_SRC\"" >> "$VENV/bin/activate"
    fi
    export HIPKITTENS_SRC
  fi

  case "${HIPKITTENS_BUILD_DK:-0}" in
    1|true|yes|on)
      if [[ -d "$HIPKITTENS_SRC/distributed-kernels" ]]; then
        echo "  building HipKittens distributed-kernels/bf16_gemm prototype ..."
        (
          cd "$HIPKITTENS_SRC/distributed-kernels" &&
          cmake -B build -DDK_BUILD=bf16_gemm -DGPU_TARGET=CDNA4 &&
          cmake --build build -j "${MAX_JOBS:-16}"
        ) || echo "  HipKittens distributed-kernels build failed — fused HK backend remains unavailable"
      fi
      ;;
    *)
      echo "  distributed prototype build skipped (set HIPKITTENS_BUILD_DK=1 to build)"
      ;;
  esac

  case "${HIPKITTENS_BUILD_FUSED:-0}" in
    1|true|yes|on)
      echo "  building microbenchmarks hk_iris_fused native prototype ..."
      (
        cmake -S benchmarks/aiter_kernels/hipkittens_native \
              -B benchmarks/aiter_kernels/hipkittens_native/build \
              -DHIPKITTENS_ROOT="$HIPKITTENS_SRC" \
              -DGPU_TARGET=CDNA4 &&
        cmake --build benchmarks/aiter_kernels/hipkittens_native/build -j "${MAX_JOBS:-16}"
      ) || echo "  hk_iris_fused build failed — AITER_KERNELS_BACKEND=hipkittens remains unavailable"
      ;;
    *)
      echo "  fused native prototype build skipped (set HIPKITTENS_BUILD_FUSED=1 to build)"
      ;;
  esac
else
  echo "  skipped (set HIPKITTENS_INSTALL=1 to force clone/update)"
fi

echo "--- Flash Attention (ROCm fork)"
FLASH_ATTN_INSTALL="${FLASH_ATTN_INSTALL:-auto}"
_want_flash=0
case "${FLASH_ATTN_INSTALL}" in
  1|true|yes|on) _want_flash=1 ;;
  0|false|no|off) _want_flash=0 ;;
  auto|AUTO|"")
    # By default, install if on ROCM and not already present
    if [[ "$DEVICE" == "rocm" ]] && ! python3 -c "import flash_attn" 2>/dev/null; then
      _want_flash=1
    fi
    ;;
esac
if [[ "$_want_flash" -eq 1 ]]; then
  echo "  Installing Flash Attention from ROCm fork (tridao branch)..."
  # Build dependencies required by flash-attention setup.py
  pip install packaging psutil ninja 2>/dev/null
  # Use optimized build flags for MI300 (gfx942) and MI355 (gfx950).
  # Triton backend is recommended for CDNA performance.
  # Limit MAX_JOBS to avoid OOM during heavy ninja build.
  export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
  export GPU_ARCHS="gfx942;gfx950"
  export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  
  if pip install --no-build-isolation git+https://github.com/ROCm/flash-attention.git@tridao 2>&1 | tail -20; then
    echo "  Flash Attention installed successfully"
    if ! grep -q "FLASH_ATTENTION_TRITON_AMD_ENABLE" "$VENV/bin/activate"; then
      echo 'export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"' >> "$VENV/bin/activate"
    fi
  else
    echo "  Flash Attention install failed - check HIP/ROCm environment"
  fi
else
  echo "  skipped (set FLASH_ATTN_INSTALL=1 to force install attempt)"
fi

echo "--- VBench (Perceptual Quality Benchmarking)"
VBENCH_INSTALL="${VBENCH_INSTALL:-auto}"
_want_vbench=0
case "${VBENCH_INSTALL}" in
  1|true|yes|on) _want_vbench=1 ;;
  0|false|no|off) _want_vbench=0 ;;
  auto|AUTO|"")
    # By default, attempt to install VBench unless explicitly disabled
    _want_vbench=1
    ;;
esac
if [[ "$_want_vbench" -eq 1 ]]; then
  echo "  Installing VBench from GitHub..."
  pip3 install git+https://github.com/Vchitect/VBench.git 2>/dev/null || echo "  VBench install failed — bench11_quality will skip"
else
  echo "  skipped (set VBENCH_INSTALL=1 to force install attempt)"
fi


echo "--- PDF report tooling (wkhtmltopdf / pandoc)"
PDF_TOOLS_INSTALL="${PDF_TOOLS_INSTALL:-auto}"
_want_pdf=0
case "${PDF_TOOLS_INSTALL}" in
  1|true|yes|on) _want_pdf=1 ;;
  0|false|no|off) _want_pdf=0 ;;
  auto|AUTO|"")
    # Auto-install if neither tool is present
    if ! command -v wkhtmltopdf &>/dev/null && ! command -v pandoc &>/dev/null; then
      _want_pdf=1
    fi
    ;;
esac
if command -v wkhtmltopdf &>/dev/null; then
  echo "  wkhtmltopdf: $(command -v wkhtmltopdf) ($(wkhtmltopdf --version 2>&1 | head -1))"
elif command -v pandoc &>/dev/null; then
  echo "  pandoc: $(command -v pandoc) ($(pandoc --version | head -1))"
  echo "  wkhtmltopdf: not found (pandoc will be used as fallback)"
elif [[ "$_want_pdf" -eq 1 ]]; then
  echo "  No PDF tools found — attempting install..."
  # Try wkhtmltopdf first (preferred — embeds plots inline)
  if sudo apt-get install -y -qq wkhtmltopdf 2>/dev/null; then
    echo "  wkhtmltopdf installed successfully"
  else
    echo "  wkhtmltopdf install failed — trying pandoc..."
    if sudo apt-get install -y -qq pandoc 2>/dev/null; then
      echo "  pandoc installed successfully"
    else
      echo "  PDF tools install failed — report.pdf will be skipped"
      echo "  Install manually: sudo apt install wkhtmltopdf  OR  sudo apt install pandoc"
    fi
  fi
else
  echo "  skipped (set PDF_TOOLS_INSTALL=1 to force)"
fi

echo "--- Optional attention backends (probed, not required)"
python3 - <<'PY'
def probe(mod):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod:12s} present (v{v})")
    except Exception as e:
        print(f"  {mod:12s} not present ({type(e).__name__})")
for m in ("torch", "triton", "aiter", "flash_attn", "iris"):
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

print_activate_hint() {
  # Avoid redundant instructions when this shell is already in the target venv.
  local active_basename=""
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    active_basename="$(basename "$VIRTUAL_ENV")"
  fi
  if [[ "$active_basename" != "$VENV" ]]; then
    echo "  source ${VENV}/bin/activate"
  else
    echo "  (already active: ${VENV})"
  fi
}

case "$DEVICE" in
  rocm|cuda)
    echo "Next:"
    print_activate_hint
    echo "  ./test.sh -t benchmark"
    echo "    # High-level orchestrator: runs the curated full benchmark sequence"
    echo "    # (testcase/workload registry + stable order + benchmark defaults)."
    echo "  ./run.sh --testcase <name> [--workload <name>] [--iterations N]"
    echo "    # Low-level single-job runner: use when you want direct control over"
    echo "    # one testcase/workload invocation and its knobs."
    ;;
  cpu)
    echo "Note: CPU-only environment."
    echo "  GPU-bound testcases will exit early on this host:"
    echo "    bench01_bf16_compute, bench02_hbm_bandwidth, bench03_dram_capacity,"
    echo "    bench05_e2e_mfu, bench12_multigpu_comm,"
    echo "    validation/rvs/*, validation/rccl/*, validation/rocm_bw/*"
    echo "  These still work without a GPU:"
    echo "    bench04_workload_ops    (analytic FLOP/byte accounting)"
    echo "    scripts/score_benchmark.py / scripts/report.py / scripts/plot_results.py"
    echo "    (over JSON outputs collected on a GPU host)"
    echo
    echo "Next:"
    print_activate_hint
    echo "  ./test.sh -t benchmark"
    echo "    # High-level orchestrator: runs the benchmark flow with CPU-aware"
    echo "    # fallbacks/skips handled by each benchmark."
    echo "  ./run.sh --testcase workload --workload escher_14b_480p --iterations 1"
    echo "    # Low-level single-job runner: direct one-off control (quick local"
    echo "    # smoke/development loops without the full benchmark)."
    ;;
esac
echo "========================================"
