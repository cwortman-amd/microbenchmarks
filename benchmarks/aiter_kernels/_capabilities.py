"""Runtime capability probes mirroring ``aiter/__init__.py``'s try-import gates.

Why this lives separately from ``dispatcher.py``: the dispatcher imports
this module unconditionally at package-load time, but the capability flags
need to be cheap and importable on any host (including a CPU CI box that
has no triton, no aiter, no iris). We therefore wrap every probe in a
try/except — never a hard import — and expose the result as module-level
booleans.

Detection order matches what ``aiter/__init__.py`` itself does:

  1. ``triton`` — required for any GPU-accelerated path.
  2. ``aiter`` — preferred backend (carries the upstreamed kernel + Iris).
  3. ``iris`` — required for AITER's GPU-initiated communication ops.
  4. ``torch.distributed._symmetric_memory`` — torch-native fallback path.

Anything that isn't available stays ``False`` and the dispatcher routes
around it. None of these probes are allowed to raise at import time.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

logger = logging.getLogger("aiter_kernels")


def _try_import(modname: str):
    try:
        return importlib.import_module(modname)
    except (ImportError, RuntimeError, OSError):
        return None


_triton_mod = _try_import("triton")
TRITON_AVAILABLE = _triton_mod is not None

_aiter_mod = _try_import("aiter")
AITER_AVAILABLE = _aiter_mod is not None

_iris_mod = _try_import("iris")
IRIS_AVAILABLE = _iris_mod is not None and TRITON_AVAILABLE

_symm_mem_mod = _try_import("torch.distributed._symmetric_memory")
# SymmMem fused ops are only registered for CUDA — calling them on a CPU
# tensor raises NotImplementedError. We require both the ops to be present
# AND the host to expose CUDA, so the dispatcher doesn't pick a backend
# that will explode the moment we hand it a tensor.
SYMM_MEM_AVAILABLE = (
    _symm_mem_mod is not None
    and torch.cuda.is_available()
    and hasattr(torch.ops, "symm_mem")
    and hasattr(getattr(torch.ops, "symm_mem", None), "fused_all_gather_matmul")
    and hasattr(getattr(torch.ops, "symm_mem", None), "fused_matmul_reduce_scatter")
)


def detect_arch() -> Optional[str]:
    """Return the gfx arch string for the current device (e.g. ``gfx950``).

    Mirrors AITER's ``DEVICE_ARCH`` check used in ``aiter/ops/triton/``: we key
    config files by gfx code (``gfx950`` for MI350 / MI355X CDNA4, ``gfx942``
    for MI300X / MI325X CDNA3) rather than product names.

    Returns None on a CPU-only host.
    """
    forced = os.environ.get("AITER_KERNELS_ARCH")
    if forced:
        return forced.lower()
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None
    arch = getattr(props, "gcnArchName", None) or ""
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8", errors="ignore")
    arch = arch.strip().split(":")[0].lower()  # gfx950:sramecc+:xnack- -> gfx950
    return arch or None


@dataclass(frozen=True)
class BackendCapabilities:
    """Capability snapshot used by the dispatcher and the benchmark report."""

    triton: bool
    aiter: bool
    iris: bool
    symm_mem: bool
    arch: Optional[str]
    device: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "triton": self.triton,
            "aiter": self.aiter,
            "iris": self.iris,
            "symm_mem": self.symm_mem,
            "arch": self.arch,
            "device": self.device,
        }


def probe_backends() -> BackendCapabilities:
    """One-shot probe of every backend the dispatcher knows how to route to.

    Returned dataclass is what ``dispatcher.select_backend()`` keys off.
    Logged once at INFO level so the bench JSON makes the choice auditable
    without re-running the probe.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    caps = BackendCapabilities(
        triton=TRITON_AVAILABLE,
        aiter=AITER_AVAILABLE,
        iris=IRIS_AVAILABLE,
        symm_mem=SYMM_MEM_AVAILABLE,
        arch=detect_arch(),
        device=device,
    )
    logger.info("aiter_kernels capabilities: %s", caps.as_dict())
    return caps
