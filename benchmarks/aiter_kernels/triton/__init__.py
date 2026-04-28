"""Triton-backed implementations of the fused AG+MM and MM+RS ops.

Layout mirrors ``aiter/ops/triton/comms/fused/`` — the wrapper modules
(``fused_all_gather_matmul.py`` / ``fused_matmul_reduce_scatter.py``) are
the public surface, and the device kernels live under ``_triton_kernels/``.

Both wrappers are no-ops when triton is not importable on the host: the
dispatcher falls through to ``torch.ops.symm_mem`` or the pure-Torch
fallback in that case.
"""

from __future__ import annotations

from benchmarks.aiter_kernels._capabilities import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from benchmarks.aiter_kernels.triton.fused_all_gather_matmul import (  # noqa: F401
        fused_all_gather_matmul,
    )
    from benchmarks.aiter_kernels.triton.fused_matmul_reduce_scatter import (  # noqa: F401
        fused_matmul_reduce_scatter,
    )
