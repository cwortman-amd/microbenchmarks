"""Tile-config loader matching AITER's ``get_gemm_config`` selection rules.

Upstream reference: ``aiter/ops/triton/utils/gemm_config_utils.py``.

Key contract we preserve so the JSON files in ``configs/`` are wire-compatible
with what AITER expects (and so our configs can be lifted into upstream
without translation):

  * Files are named ``{arch}-{CONFIG_NAME}.json`` (e.g. ``gfx950-FUSED-AG-MATMUL.json``).
  * Top-level keys are one of ``M_LEQ_<N>``, ``M_GEQ_<N>``, or ``any``
    (deprecated ``"large"``/``"small"`` keys are not accepted).
  * ``M_LEQ_x`` keys are searched in ascending order using the standard
    bounds list ``[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]``,
    falling through to ``M_GEQ_x`` (descending) and finally ``any``.
  * Returned dict is the kernel constexpr table — each kernel is responsible
    for unpacking the keys it understands.

We deliberately re-implement (rather than depend on) AITER's loader so this
package is importable on a host that has not built AITER yet.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

_DEFAULT_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _load_json_for(arch: str, config_name: str) -> Optional[dict]:
    path = _CONFIGS_DIR / f"{arch}-{config_name}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _bound_keys(prefix: str, payload: dict) -> Dict[int, str]:
    """Extract ``{prefix}_<N>`` keys from payload mapped to their integer N."""
    out: Dict[int, str] = {}
    pat = re.compile(rf"^{prefix}_(\d+)$")
    for k in payload:
        m = pat.match(k)
        if m:
            out[int(m.group(1))] = k
    return out


def get_kernel_config(
    arch: str,
    config_name: str,
    M: int,
    N: Optional[int] = None,
    K: Optional[int] = None,
    bounds: Optional[Tuple[int, ...]] = None,
) -> Tuple[dict, bool]:
    """Return ``(config_dict, is_tuned)`` for the given (arch, config_name, M).

    ``is_tuned`` is True when a real ``M_LEQ_x``/``M_GEQ_x`` bucket matched
    and False when we fell through to ``any`` (or to the built-in default).
    The bench JSON records this so a row can be marked "untuned" without
    having to re-derive it from the shape.
    """
    bounds = bounds or _DEFAULT_BOUNDS

    fname_specialized = (
        f"{arch}-{config_name}-N={N}-K={K}.json" if (N is not None and K is not None) else None
    )
    payload: Optional[dict] = None
    if fname_specialized:
        spath = _CONFIGS_DIR / fname_specialized
        if spath.is_file():
            try:
                payload = json.loads(spath.read_text())
            except (OSError, json.JSONDecodeError):
                payload = None
    if payload is None:
        payload = _load_json_for(arch, config_name)
    if payload is None:
        return _builtin_default(config_name), False

    leq = _bound_keys("M_LEQ", payload)
    for thr in sorted(set(leq) | set(bounds)):
        if thr in leq and M <= thr:
            return _strip_meta(payload[leq[thr]]), True

    geq = _bound_keys("M_GEQ", payload)
    for thr in sorted(geq.keys(), reverse=True):
        if M >= thr:
            return _strip_meta(payload[geq[thr]]), True

    if "any" in payload:
        return _strip_meta(payload["any"]), False

    return _builtin_default(config_name), False


def _strip_meta(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _builtin_default(config_name: str) -> dict:
    """Last-resort default if the JSON for the arch is missing entirely.

    Documented in ``README.md §2.2 / §3.2`` so the report can footnote
    it explicitly when this branch fires.
    """
    is_rs = "MATMUL-RS" in config_name
    return {
        "BLOCK_M": 64 if is_rs else 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_SMS": 256,
        "num_warps": 16,
        "num_stages": 4 if is_rs else 3,
        "waves_per_eu": 4,
    }


def env_override(env_prefix: str, config: dict) -> dict:
    """Allow per-key env overrides for ad-hoc shape sweeps.

    e.g. ``AITER_KERNELS_FUSED_AG_MM_BLOCK_M=256 ./test.sh -t multigpu``.
    Only the keys already present in ``config`` are honored — typos go to
    ``ValueError`` so a misnamed env var can't silently no-op.
    """
    out = dict(config)
    for key in list(out):
        env = f"{env_prefix}_{key}".upper()
        if env in os.environ:
            try:
                out[key] = int(os.environ[env])
            except ValueError as e:
                raise ValueError(f"{env}={os.environ[env]!r} must be int") from e
    return out
