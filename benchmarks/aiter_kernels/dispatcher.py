"""Backend dispatcher for the fused AG+MM and MM+RS ops.

Selection order (mirrors what we'd want upstream in ``aiter/__init__.py``):

  1. ``aiter.fused_all_gather_matmul`` / ``aiter.fused_matmul_reduce_scatter``
     — preferred path once the kernels in this package are upstreamed into
     AITER and re-exported from the top-level namespace.
  2. ``aiter.ops.triton.comms.fused.fused_all_gather_matmul`` / ``fused_matmul_reduce_scatter``
     — same kernels, addressed by their canonical AITER path. Tried before
     the local copy so an installed AITER always wins over a vendored mirror.
  3. ``benchmarks.aiter_kernels.hipkittens`` — experimental AITER-managed
     CDNA4-native backend. Selectable when a built HK fused extension exports
     the SymmMem-style AG+MM and MM+RS entry points.
  4. ``benchmarks.aiter_kernels.triton`` — the vendored copy of the kernels
     in this repo. Used when AITER is built without Iris (CDNA3 hosts that
     skipped the ``[triton_comms]`` extra) or when running directly out of
     this microbench repo.
  5. ``torch.ops.symm_mem.fused_*`` — the upstream PyTorch native path.
  6. Pure-PyTorch fallback (``_fallback.py``) — always available, used as
     correctness gold by the op-tests.

The dispatcher is deliberately *explicit*: every selection records which
backend won (``BackendInfo.source``) so the benchmark benchmark can attribute
TFLOPs and bandwidth numbers to the right impl.

A user can pin the backend with the ``backend`` argument or by setting
``AITER_KERNELS_BACKEND={aiter,aiter_triton_comms,hipkittens,hk,local_triton,symm_mem,fallback}``.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.aiter_kernels._capabilities import (
    AITER_AVAILABLE,
    HIPKITTENS_AVAILABLE,
    IRIS_AVAILABLE,
    SYMM_MEM_AVAILABLE,
    TRITON_AVAILABLE,
    BackendCapabilities,
    probe_backends,
)
from benchmarks.aiter_kernels._fallback import (
    fused_all_gather_matmul_fallback,
    fused_matmul_reduce_scatter_fallback,
)

logger = logging.getLogger("aiter_kernels")


@dataclass(frozen=True)
class BackendInfo:
    """Describes which backend the dispatcher picked for a given call.

    Carried into the bench JSON so a row's TFLOPs number can be attributed
    to the exact impl that produced it (e.g. ``aiter.upstream`` vs
    ``benchmarks.aiter_kernels.triton`` vs ``torch.ops.symm_mem``).
    """

    source: str
    ag_fn: Optional[Callable[..., Any]] = None
    rs_fn: Optional[Callable[..., Any]] = None
    capabilities: Optional[BackendCapabilities] = None


def _try_get(modname: str, attr: str) -> Optional[Callable[..., Any]]:
    try:
        m = importlib.import_module(modname)
    except (ImportError, RuntimeError, OSError):
        return None
    fn = getattr(m, attr, None)
    return fn if callable(fn) else None


def _resolve_aiter_upstream() -> Tuple[Optional[Callable], Optional[Callable]]:
    """Look for the kernels at the top of the upstream ``aiter`` namespace.

    This is where the kernels would land after the upstreaming PR described
    in ``aiter_kernels/README.md §5``.
    """
    if not AITER_AVAILABLE:
        return None, None
    ag = _try_get("aiter", "fused_all_gather_matmul")
    rs = _try_get("aiter", "fused_matmul_reduce_scatter")
    return ag, rs


def _resolve_aiter_triton_comms() -> Tuple[Optional[Callable], Optional[Callable]]:
    """Look for the kernels at their canonical AITER path."""
    if not (AITER_AVAILABLE and IRIS_AVAILABLE):
        return None, None
    ag = _try_get("aiter.ops.triton.comms.fused.fused_all_gather_matmul",
                  "fused_all_gather_matmul")
    rs = _try_get("aiter.ops.triton.comms.fused.fused_matmul_reduce_scatter",
                  "fused_matmul_reduce_scatter")
    return ag, rs


def _resolve_hipkittens() -> Tuple[Optional[Callable], Optional[Callable]]:
    """Resolve a built HipKittens fused-collective extension."""
    if not HIPKITTENS_AVAILABLE:
        return None, None
    try:
        from benchmarks.aiter_kernels import hipkittens
    except (ImportError, RuntimeError, OSError):
        return None, None
    backend = hipkittens.resolve_backend()
    if backend is None:
        return None, None
    return backend.ag_fn, backend.rs_fn


def _resolve_local_triton() -> Tuple[Optional[Callable], Optional[Callable]]:
    """The vendored kernels in this microbench repo."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return None, None
    ag = _try_get("benchmarks.aiter_kernels.triton.fused_all_gather_matmul",
                  "fused_all_gather_matmul")
    rs = _try_get("benchmarks.aiter_kernels.triton.fused_matmul_reduce_scatter",
                  "fused_matmul_reduce_scatter")
    return ag, rs


def _resolve_symm_mem() -> Tuple[Optional[Callable], Optional[Callable]]:
    """The torch.ops.symm_mem path."""
    if not SYMM_MEM_AVAILABLE:
        return None, None
    ns = torch.ops.symm_mem
    return ns.fused_all_gather_matmul, ns.fused_matmul_reduce_scatter


_RESOLVERS = [
    ("aiter.upstream", _resolve_aiter_upstream),
    ("aiter.ops.triton.comms.fused", _resolve_aiter_triton_comms),
    ("benchmarks.aiter_kernels.hipkittens", _resolve_hipkittens),
    ("benchmarks.aiter_kernels.triton", _resolve_local_triton),
    ("torch.ops.symm_mem", _resolve_symm_mem),
]


def _fallback_backend(caps: BackendCapabilities) -> BackendInfo:
    return BackendInfo(
        source="fallback.pure_torch",
        ag_fn=fused_all_gather_matmul_fallback,
        rs_fn=fused_matmul_reduce_scatter_fallback,
        capabilities=caps,
    )


def select_backend(*, force: Optional[str] = None) -> BackendInfo:
    """Pick the highest-priority backend whose probes succeeded.

    ``force`` (or env ``AITER_KERNELS_BACKEND``) pins selection. Recognized
    values: ``aiter``, ``aiter_triton_comms``, ``hipkittens``/``hk``,
    ``local_triton``, ``symm_mem``, ``fallback``. An unknown value raises
    ``ValueError`` so a typo doesn't silently fall back to pure-Torch.
    """
    caps = probe_backends()
    pin = force or os.environ.get("AITER_KERNELS_BACKEND")
    forced_alias = {
        "aiter": "aiter.upstream",
        "aiter_triton_comms": "aiter.ops.triton.comms.fused",
        "hipkittens": "benchmarks.aiter_kernels.hipkittens",
        "hk": "benchmarks.aiter_kernels.hipkittens",
        "local_triton": "benchmarks.aiter_kernels.triton",
        "symm_mem": "torch.ops.symm_mem",
        "fallback": "fallback.pure_torch",
    }
    if pin is not None:
        if pin not in forced_alias:
            raise ValueError(
                f"AITER_KERNELS_BACKEND={pin!r} not recognized. "
                f"Choose one of {sorted(forced_alias)}."
            )
        wanted = forced_alias[pin]
        if wanted == "fallback.pure_torch":
            return _fallback_backend(caps)
        for label, resolver in _RESOLVERS:
            if label != wanted:
                continue
            ag, rs = resolver()
            if ag is None or rs is None:
                extra = ""
                if wanted == "benchmarks.aiter_kernels.hipkittens":
                    try:
                        from benchmarks.aiter_kernels.hipkittens import availability_reason
                        extra = f" HipKittens: {availability_reason()}."
                    except Exception:  # noqa: BLE001
                        extra = ""
                raise RuntimeError(
                    f"Forced backend {pin!r} not available on this host. "
                    f"Capabilities: {caps.as_dict()}.{extra}"
                )
            return BackendInfo(source=label, ag_fn=ag, rs_fn=rs, capabilities=caps)
        raise RuntimeError(f"Forced backend {pin!r} resolver missing")

    for label, resolver in _RESOLVERS:
        ag, rs = resolver()
        if ag is not None and rs is not None:
            logger.info("aiter_kernels dispatcher selected %s", label)
            return BackendInfo(source=label, ag_fn=ag, rs_fn=rs, capabilities=caps)

    logger.info("aiter_kernels dispatcher: no GPU backend resolved, using pure-Torch fallback")
    return _fallback_backend(caps)


def fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    *,
    gather_dim: int = 0,
    group_name: Optional[str] = None,
    backend: Optional[str] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Public AG+MM entry point matching ``torch.ops.symm_mem.fused_all_gather_matmul``.

    ``A_shard`` is the local M-shard of a tensor-parallel column-parallel
    weight. ``Bs`` is one or more replicated weight matrices. Returns
    ``(A_full, [Y_i])`` where ``A_full`` is the gathered activation and
    ``Y_i = A_full @ B_i``. Falls through to whichever backend
    ``select_backend()`` picks.
    """
    if group_name is None:
        if not dist.is_initialized():
            raise RuntimeError(
                "fused_all_gather_matmul needs a process group. "
                "Pass group_name= or initialize torch.distributed first."
            )
        group_name = dist.group.WORLD.group_name  # type: ignore[union-attr]
    info = select_backend(force=backend)
    return info.ag_fn(A_shard, Bs, gather_dim=gather_dim, group_name=group_name)  # type: ignore[misc]


def fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str = "avg",
    *,
    scatter_dim: int = 0,
    group_name: Optional[str] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Public MM+RS entry point matching ``torch.ops.symm_mem.fused_matmul_reduce_scatter``."""
    if group_name is None:
        if not dist.is_initialized():
            raise RuntimeError(
                "fused_matmul_reduce_scatter needs a process group. "
                "Pass group_name= or initialize torch.distributed first."
            )
        group_name = dist.group.WORLD.group_name  # type: ignore[union-attr]
    info = select_backend(force=backend)
    return info.rs_fn(A, B, reduce_op, scatter_dim=scatter_dim, group_name=group_name)  # type: ignore[misc]
