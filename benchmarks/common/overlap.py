"""Communication-overlap scorecard for fused compute+collective benchmarks.

Raw ``t_ms`` tells you which kernel was faster on the day, but not *why* — and
in particular not whether a "fused" kernel is actually hiding communication
behind compute or just running a slower GEMM. This module implements the
standard way the overlap literature answers that question.

Effective Communication Time (ECT), from Flux (Chang et al., 2024,
arXiv:2406.06858):

    ECT = overall_time - best_non_split_GEMM_time

For a non-overlapping (serial) baseline, ECT is just the exposed collective
time. For an overlapping kernel, ECT collapses any non-overlapped comm *plus*
any GEMM-efficiency loss into a single number. A perfect overlap has ECT == 0.

Overlap Efficiency:

    E_overlap = 1 - ECT_overlap / ECT_non_overlap

0% means "no better than running comm and GEMM serially"; 100% means "comm is
fully hidden"; a **negative** value means the fused kernel is *slower* than the
unfused baseline (exactly the failure mode the audit found in the Iris AG+MM /
MM+RS kernels). This is the only fair cross-kernel metric because the GEMM term
cancels.

We also fit a simple linear collective cost model (TokenWeave / FiCCO style):

    t_comm(bytes) ~= alpha + beta * GB

``alpha`` is the fixed launch/latency floor (ms) and ``1/beta`` is the achieved
fabric bandwidth (GB/s). Together with the per-shape ``comm_proportion`` this
tells you, before writing a single fused kernel, whether overlap can pay off at
all for a given shape (below ~10% comm proportion it generally cannot).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# Below this comm fraction of serial runtime, overlap generally is not worth the
# kernel-fusion complexity / GEMM-efficiency risk (TokenWeave, FiCCO).
WORTHWHILE_COMM_PROPORTION = 0.10


def effective_comm_time(t_overall_ms: float, t_gemm_ms: float) -> float:
    """ECT = overall - best non-split GEMM time (Flux Eq. 1)."""
    return t_overall_ms - t_gemm_ms


def overlap_efficiency(ect_overlap_ms: float, ect_baseline_ms: float) -> Optional[float]:
    """1 - ECT_overlap / ECT_baseline (Flux Eq. 2).

    Returns ``None`` when the baseline has no measurable exposed comm
    (``ECT_baseline <= 0``), since the ratio is then undefined / pure noise.
    """
    if ect_baseline_ms is None or ect_baseline_ms <= 0:
        return None
    return 1.0 - (ect_overlap_ms / ect_baseline_ms)


def _linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    """Ordinary least-squares ``y = a + b x``. Returns ``(a, b, r2)`` or None.

    Needs at least two points with non-zero spread in ``x``.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot <= 0:
        r2 = 1.0
    else:
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


def fit_comm_cost_model(samples: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    """Fit ``t_comm_ms = alpha + beta * GB`` over ``(wire_bytes, t_comm_ms)``.

    ``alpha`` is the latency floor (ms); ``beta`` is ms per GB so the achieved
    bandwidth is ``1/beta`` GB/s. Returns ``None`` if there isn't enough spread
    to fit (e.g. a single shape).
    """
    pts = [(b / 1e9, t) for (b, t) in samples if b and b > 0 and t is not None and t > 0]
    fit = _linear_fit([x for x, _ in pts], [y for _, y in pts])
    if fit is None:
        return None
    alpha, beta, r2 = fit
    out = {"alpha_ms": alpha, "beta_ms_per_gb": beta, "r2": r2, "n_points": len(pts)}
    if beta > 0:
        out["bandwidth_gb_s"] = 1.0 / beta
    return out


# Pattern wiring. Each pattern pairs one non-overlapping baseline (which carries
# the standalone GEMM time ``t_ms_mm`` and the standalone collective time) with
# one or more overlap *candidate* ops to score against it. A shape only
# contributes the patterns whose rows are actually present, so the same scorecard
# serves bench06 (ag_mm / mm_rs) and the MM+AR bench (pipelined / iris).
_PATTERNS = [
    {"name": "ag_mm", "unfused": "unfused_ag_mm", "comm": "t_ms_ag",
     "bytes": "ag_wire_bytes", "candidates": ["ag_mm"]},
    {"name": "mm_rs", "unfused": "unfused_mm_rs", "comm": "t_ms_rs",
     "bytes": "rs_wire_bytes", "candidates": ["mm_rs"]},
    {"name": "mm_ar", "unfused": "unfused_mm_ar", "comm": "t_ms_ar",
     "bytes": "ar_wire_bytes", "candidates": ["pipelined_mm_ar", "iris_mm_ar"]},
]


def _key(row: Dict) -> Tuple:
    return (row.get("M"), row.get("K"), row.get("N"))


def _num(row: Optional[Dict], field: str) -> Optional[float]:
    if not row:
        return None
    v = row.get(field)
    return float(v) if isinstance(v, (int, float)) else None


def summarize_overlap(rows: Sequence[Dict]) -> Dict[str, object]:
    """Build the overlap scorecard from a bench06 ``rows`` list.

    Pairs each fused row with its unfused counterpart and the standalone GEMM
    time (the unfused row's ``t_ms_mm``) by ``(M, K, N)``, then computes ECT,
    overlap efficiency, speedup, comm proportion, and the theoretical overlap
    ceiling. Also fits a per-pattern collective cost model across shapes.

    Returns ``{"rows": [...summary rows...], "cost_model": {pattern: {...}}}``.
    """
    by_op: Dict[str, Dict[Tuple, Dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        by_op.setdefault(r.get("op", ""), {})[_key(r)] = r

    summary: List[Dict] = []
    cost_samples: Dict[str, List[Tuple[float, float]]] = {p["name"]: [] for p in _PATTERNS}

    for pat in _PATTERNS:
        unfused_rows = by_op.get(pat["unfused"], {})
        candidate_rows = [(c, by_op.get(c, {})) for c in pat["candidates"]]
        for key, u in unfused_rows.items():
            t_gemm = _num(u, "t_ms_mm")
            t_comm = _num(u, pat["comm"])
            t_unfused = _num(u, "t_ms")
            wire_bytes = _num(u, pat["bytes"])
            if wire_bytes and t_comm:
                cost_samples[pat["name"]].append((wire_bytes, t_comm))

            M, K, N = key
            # Baseline-only fields, shared across all candidates for this shape.
            base: Dict[str, object] = {
                "world": u.get("world"),
                "M": M, "K": K, "N": N,
                "t_gemm_ms": t_gemm,
                "t_comm_ms": t_comm,
                "t_unfused_ms": t_unfused,
            }
            ect_unfused = None
            if t_gemm is not None and t_unfused is not None:
                ect_unfused = effective_comm_time(t_unfused, t_gemm)
                base["ect_unfused_ms"] = ect_unfused
                if ect_unfused > 0 and t_unfused > 0:
                    base["comm_proportion"] = ect_unfused / t_unfused
                    base["overlap_worthwhile"] = (
                        ect_unfused / t_unfused >= WORTHWHILE_COMM_PROPORTION
                    )
                # Theoretical best overlapped time: comm fully hidden behind the
                # larger of the two. Independent of any candidate impl.
                if t_comm is not None:
                    ceiling = max(t_gemm, t_comm)
                    base["t_overlap_ceiling_ms"] = ceiling
                    if ceiling > 0:
                        base["ceiling_speedup"] = t_unfused / ceiling

            for cand_op, cand_rows in candidate_rows:
                f = cand_rows.get(key)
                if f is None and cand_op not in by_op:
                    continue  # this bench didn't run this candidate at all
                t_fused = _num(f, "t_ms")
                entry: Dict[str, object] = {"pattern": cand_op, **base, "t_fused_ms": t_fused}
                if (
                    t_fused is not None and t_fused > 0
                    and t_gemm is not None and ect_unfused is not None
                ):
                    ect_fused = effective_comm_time(t_fused, t_gemm)
                    entry["ect_fused_ms"] = ect_fused
                    entry["overlap_efficiency"] = overlap_efficiency(ect_fused, ect_unfused)
                    entry["speedup_vs_unfused"] = t_unfused / t_fused
                summary.append(entry)

    cost_model = {
        name: model
        for name, samples in cost_samples.items()
        if (model := fit_comm_cost_model(samples)) is not None
    }

    return {"rows": summary, "cost_model": cost_model}
