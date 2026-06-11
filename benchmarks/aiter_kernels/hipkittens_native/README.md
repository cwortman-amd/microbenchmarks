# HipKittens/Iris Native Backend

This directory contains the repo-local native extension for the experimental
`AITER_KERNELS_BACKEND=hipkittens` path.

The intended layering is:

- Python dispatcher: keeps the AITER/SymmMem-compatible public API.
- Iris: owns symmetric allocations, barriers, and remote transport metadata.
- HipKittens: owns CDNA4 tile scheduling, LDS staging, and MFMA execution.

The first native target is a BF16 AG+MM producer/consumer GEMM kernel adapted
from `HipKittens/distributed-kernels/bf16_gemm`. It currently exposes a low
level `dispatch_ag_mm(...)` entry point; the Python adapter wraps this into the
public `fused_all_gather_matmul(...)` signature when a compatible Iris device
view is available.

Build manually:

```bash
cmake -S benchmarks/aiter_kernels/hipkittens_native \
      -B benchmarks/aiter_kernels/hipkittens_native/build \
      -DHIPKITTENS_ROOT=$HOME/.cache/HipKittens \
      -DGPU_TARGET=CDNA4
cmake --build benchmarks/aiter_kernels/hipkittens_native/build -j 16
```

Or through setup:

```bash
HIPKITTENS_BUILD_FUSED=1 ./setup.sh
```

