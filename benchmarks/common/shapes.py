"""Centralized GEMM matrix-size definitions and generic shape helpers.

Benchmarks load their ``(M, K, N)`` sweep from a single YAML definition file
(default ``configs/matrix_sizes.yaml``) instead of hard-coding shapes, so the
sizes can be edited in one place — or swapped per run via ``--shape-set`` /
``--shapes`` / ``--shapes-file`` — without touching benchmark code.

This module also provides :func:`pad_to_multiple`, the generic padding used by
the collective benchmarks (AG+MM / MM+RS) which need the GEMM M dimension to
divide the tensor-parallel world size. We pad UP (never truncate) so the
production shape is preserved rather than shrunk.

The loader degrades gracefully: if the YAML file or PyYAML is unavailable
(e.g. a minimal CPU CI box), it falls back to a built-in default set so the
benchmarks still run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Canonical default file, relative to the repo root (cwd of the runner).
DEFAULT_SHAPES_FILE = "configs/matrix_sizes.yaml"

# Built-in fallback, mirrors the ``odyssey_production`` set in the YAML so the
# benchmarks have valid shapes even when the file / PyYAML can't be loaded.
_BUILTIN_DEFAULT: List[Tuple[str, int, int, int]] = [
    ("odyssey_1frame", 1590, 5120, 13824),
    ("odyssey_3frame", 4680, 5120, 13824),
]


@dataclass(frozen=True)
class Shape:
    """A single GEMM problem ``A[M, K] @ B[K, N]`` with an optional label."""

    M: int
    K: int
    N: int
    name: str = ""

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.M, self.K, self.N)


def pad_to_multiple(value: int, multiple: int) -> int:
    """Round ``value`` UP to the nearest positive multiple of ``multiple``.

    Used to pad the GEMM M dimension so it divides the TP world size for the
    AG/RS collective benchmarks. Padding (not truncation) keeps the requested
    production shape intact. ``multiple <= 1`` returns ``value`` unchanged;
    ``value <= 0`` returns 0.
    """
    if multiple <= 1:
        return value
    if value <= 0:
        return 0
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_path(path: Optional[str]) -> Optional[Path]:
    """Find the shapes file: explicit arg, then env var, then repo default."""
    if path:
        p = Path(path)
        return p if p.is_file() else None
    env = os.environ.get("BENCH_MATRIX_SIZES")
    if env and Path(env).is_file():
        return Path(env)
    p = Path(DEFAULT_SHAPES_FILE)
    return p if p.is_file() else None


def _builtin_shapes() -> List[Shape]:
    return [Shape(M=m, K=k, N=n, name=nm) for (nm, m, k, n) in _BUILTIN_DEFAULT]


def available_sets(path: Optional[str] = None) -> List[str]:
    """List the named sets defined in the YAML file (empty if unavailable)."""
    resolved = _resolve_path(path)
    if resolved is None:
        return []
    try:
        import yaml  # local import: module stays importable without PyYAML

        doc = yaml.safe_load(resolved.read_text()) or {}
        return sorted((doc.get("sets") or {}).keys())
    except Exception:  # noqa: BLE001
        return []


def load_shapes(set_name: Optional[str] = None, *, path: Optional[str] = None) -> List[Shape]:
    """Load a named shape set from the YAML definition file.

    ``set_name=None`` uses the file's ``default_set``. Falls back to the
    built-in Odyssey-production default when the file/PyYAML is unavailable or
    the requested set is missing/empty.
    """
    resolved = _resolve_path(path)
    if resolved is not None:
        try:
            import yaml  # local import so this module imports without PyYAML

            doc = yaml.safe_load(resolved.read_text()) or {}
            sets = doc.get("sets") or {}
            chosen = set_name or doc.get("default_set")
            if chosen and chosen in sets:
                raw = (sets[chosen] or {}).get("shapes", []) or []
                shapes = [
                    Shape(
                        M=int(s["M"]),
                        K=int(s["K"]),
                        N=int(s["N"]),
                        name=str(s.get("name", chosen)),
                    )
                    for s in raw
                ]
                if shapes:
                    return shapes
        except Exception:  # noqa: BLE001
            pass
    return _builtin_shapes()


def parse_shapes_arg(arg: str) -> List[Shape]:
    """Parse a ``'M,K,N;M,K,N;...'`` CLI override string into Shapes."""
    out: List[Shape] = []
    idx = 0
    for chunk in arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"shape {chunk!r} must be 'M,K,N' (3 ints)")
        try:
            M, K, N = (int(p) for p in parts)
        except ValueError as e:
            raise ValueError(f"shape {chunk!r} must be 3 ints: {e}") from e
        out.append(Shape(M=M, K=K, N=N, name=f"cli{idx}"))
        idx += 1
    if not out:
        raise ValueError(f"no shapes parsed from {arg!r}")
    return out


def resolve_shapes(
    cli_shapes: Optional[str] = None,
    set_name: Optional[str] = None,
    *,
    path: Optional[str] = None,
) -> List[Shape]:
    """Resolve the shape sweep for a run.

    Precedence: explicit ``--shapes`` CLI string  >  named YAML set  >
    built-in default. CLI parse errors raise ``ValueError`` (callers should
    surface them as a clean exit rather than a silent fallback).
    """
    if cli_shapes:
        return parse_shapes_arg(cli_shapes)
    return load_shapes(set_name, path=path)
