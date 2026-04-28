"""AITER-style fused collective+GEMM kernels.

Public surface for the new comm+compute kernels we developed against the
ROCm/aiter conventions:

  * ``fused_all_gather_matmul`` — column-parallel TP linear (AG along M)
  * ``fused_matmul_reduce_scatter`` — row-parallel TP linear (RS along M)

Both ops match the ``torch.ops.symm_mem.fused_*`` signatures so they are
drop-in replacements at TP linear call-sites.

The actual backend selection (AITER+Iris / local Triton / torch SymmMem /
pure-Torch reference) is delegated to ``benchmarks.aiter_kernels.dispatcher``.
That separation is intentional: it lets the benchmark benchmark instrument
each backend independently and emits a clean op-test gate against the
pure-Torch reference.
"""

from __future__ import annotations

from benchmarks.aiter_kernels._capabilities import (
    AITER_AVAILABLE,
    IRIS_AVAILABLE,
    SYMM_MEM_AVAILABLE,
    TRITON_AVAILABLE,
    detect_arch,
    probe_backends,
)
from benchmarks.aiter_kernels.dispatcher import (
    BackendInfo,
    fused_all_gather_matmul,
    fused_matmul_reduce_scatter,
    select_backend,
)

__all__ = [
    "AITER_AVAILABLE",
    "BackendInfo",
    "IRIS_AVAILABLE",
    "SYMM_MEM_AVAILABLE",
    "TRITON_AVAILABLE",
    "detect_arch",
    "fused_all_gather_matmul",
    "fused_matmul_reduce_scatter",
    "probe_backends",
    "select_backend",
]
