"""Generate TESTPLAN §13 charts from a benchmark's outputs.

  A2  bf16_gemm_sweep.png      — TFLOP/s vs square size
  A3  hbm_bandwidth.png        — GB/s vs size, per op
  A6  roofline.png             — per-op AI vs achieved TFLOP/s with rooflines
  A7  per_op_theory_vs_meas.png — grouped bars: theory / default / optimized
  A8  mfu.png                  — grouped bars: MFU per scope on three FLOP
                                 bases (measured peak / 1.26 PF rated /
                                 2.5 PF rated) with PDF reference-target
                                 overlay on the measured-peak basis
  A8b mfu_per_chunk.png        — per-chunk timing distribution for the timed
                                 e2e scopes (eager / compiled), reproducing
                                 the source PDF's "compiled e2e is faster
                                 *and* more stable" finding

Skips any chart whose source JSON is missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plot_gemm_sweep(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "01_bf16_compute" / "sweep.json")
    if not j:
        return
    sq = j.get("square", [])
    if not sq:
        return
    sizes = [r["M"] for r in sq]
    tf = [r["tflops"] for r in sq]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, tf, "-o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("TFLOP/s (bf16)")
    ax.set_title("BF16 GEMM size sweep")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A2_bf16_gemm_sweep.png", dpi=120)
    plt.close(fig)


def plot_bandwidth(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "02_hbm_bandwidth" / "bandwidth.json")
    if not j:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ops = sorted({r["op"] for r in j})
    for op in ops:
        rs = sorted([r for r in j if r["op"] == op], key=lambda r: r["bytes"])
        ax.plot([r["bytes"] / 1e9 for r in rs], [r["gb_s"] for r in rs], "-o", label=op)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("buffer size (GB)")
    ax.set_ylabel("achieved GB/s")
    ax.set_title("HBM bandwidth microbenchmarks")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A3_hbm_bandwidth.png", dpi=120)
    plt.close(fig)


_CAT_COLORS = {
    "time": "#888888",
    "self_attn": "#1f77b4",
    "cross_attn": "#ff7f0e",
    "ffn": "#2ca02c",
    "norm": "#d62728",
}


def plot_stability(out_dir: Path, plots_dir: Path) -> None:
    """Numerical-stability sweep: max relative error vs K, per dtype.

    Two panels:
      (a) max relative error vs K, log-log, with the analytic
          ``5·√K·2^-mantissa`` bound dashed in for each dtype.
      (b) per-element relative-error histogram for the largest K, so the
          reader can see how many elements actually live near the worst
          case (the answer is "very few" — the worst case is a tail).
    """
    j = _load(out_dir / "09_numerical_stability" / "stability.json")
    if not j or not j.get("rows"):
        return
    rows = [r for r in j["rows"] if "rel_err" in r]
    if not rows:
        return

    dtypes = sorted({r["dtype"] for r in rows})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color = {dt: palette[i % len(palette)] for i, dt in enumerate(dtypes)}

    # Panel (a): max rel err vs K, with bounds
    for dt in dtypes:
        per = sorted([r for r in rows if r["dtype"] == dt],
                     key=lambda r: r["K"])
        ks = [r["K"] for r in per]
        maxs = [r["rel_err"]["max"] for r in per]
        bounds = [r["rel_err_bound"] for r in per]
        ax_a.plot(ks, maxs, "-o", label=f"{dt} (measured)",
                  color=color[dt])
        ax_a.plot(ks, bounds, "--", label=f"{dt} (5·√K·2⁻ᵐ bound)",
                  color=color[dt], alpha=0.5)
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("K  (M = K = N)")
    ax_a.set_ylabel("max relative error vs FP32 (matrix-scaled)")
    ax_a.set_title("Reduced-precision GEMM error vs K")
    ax_a.legend(loc="best", fontsize=7, ncol=2)
    ax_a.grid(True, which="both", ls=":")

    # Panel (b): pointwise rel-err histogram at largest K, per dtype
    largest_k = max(r["K"] for r in rows)
    largest = [r for r in rows if r["K"] == largest_k]
    for r in largest:
        hist = r.get("rel_err_hist") or {}
        edges = hist.get("edges") or []
        counts = hist.get("counts") or []
        if not edges or not counts:
            continue
        centers = [(edges[i] * edges[i + 1]) ** 0.5
                   for i in range(len(counts))]
        total = sum(counts) or 1
        frac = [c / total for c in counts]
        ax_b.plot(centers, frac, "-", label=r["dtype"], color=color[r["dtype"]])
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("per-element relative error (|err| / |Z_ref|)")
    ax_b.set_ylabel("fraction of elements")
    ax_b.set_title(f"Per-element error distribution (K = {largest_k})")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(True, which="both", ls=":")

    fig.tight_layout()
    fig.savefig(plots_dir / "A9_stability.png", dpi=120)
    plt.close(fig)


def plot_cache_curve(out_dir: Path, plots_dir: Path) -> None:
    """Cache-hierarchy bandwidth curve from `bench02.cache_curve`.

    Plots GB/s vs working-set size with cache-tier guidelines pulled
    from sysfs (CPU) or torch device props (GPU). The expected stepwise
    descent — L1 -> L2 -> L3/Infinity -> DRAM/HBM — is what the eye
    should be looking for.
    """
    j = _load(out_dir / "02_hbm_bandwidth" / "cache_curve.json")
    if not j or not j.get("rows"):
        return
    rows = sorted(j["rows"], key=lambda r: r["working_set_bytes"])
    xs = [r["working_set_bytes"] / 1024 for r in rows]  # KiB
    ys = [r["gb_s"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, "-o", color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("working-set size (KiB, log scale)")
    ax.set_ylabel("achieved GB/s (copy_)")
    ax.set_title("Cache-hierarchy bandwidth curve")

    # Draw dashed verticals at known cache boundaries so the steps line
    # up with hardware tiers rather than just visual heuristics.
    seen_levels: set = set()
    for tier in (j.get("cpu_caches") or []) + (j.get("gpu_caches") or []):
        ws_kib = tier["size_bytes"] / 1024
        key = (tier["level"], int(ws_kib))
        if key in seen_levels:
            continue
        seen_levels.add(key)
        ax.axvline(ws_kib, color="#888888", ls="--", lw=0.8, alpha=0.7)
        label = f"L{tier['level']}"
        if tier.get("type") == "InfinityCache":
            label = "InfinityCache"
        ax.text(ws_kib, max(ys) * 0.96, label,
                rotation=90, fontsize=8, va="top", ha="right",
                color="#555555")

    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A3b_cache_curve.png", dpi=120)
    plt.close(fig)


def plot_roofline(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "04_workload_ops" / "ops.json")
    if not j:
        return
    rows = j["rows"]
    peak = j.get("compute_roof_tflops")
    bw = j.get("bandwidth_roof_gb_s")
    if not peak or not bw:
        return
    peak_flops = peak * 1e12
    bw_bps = bw * 1e9
    ridge = peak_flops / bw_bps

    ai_grid = [10 ** (i / 10) for i in range(0, 41)]
    bw_line = [min(bw_bps * ai, peak_flops) / 1e12 for ai in ai_grid]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ai_grid, bw_line, "k-", lw=2, label=f"Roof (peak={peak:.0f} TFLOP/s, BW={bw:.0f} GB/s)")
    ax.axvline(ridge, color="grey", ls="--", alpha=0.7, label=f"Ridge ≈ {ridge:.0f} FLOP/B")
    plotted = set()
    for r in rows:
        if r.get("flops", 0) <= 0:
            continue
        cat = r.get("category", "other")
        color = _CAT_COLORS.get(cat, "#444")
        # Use measured optimized time if available, else default, else theory
        t_ms = r.get("t_ms_optimized") or r.get("t_ms_default")
        if t_ms is None or (isinstance(t_ms, float) and math.isnan(t_ms)) or t_ms == 0:
            tflops = r.get("flops", 0) / max(r.get("t_bottleneck_theory_ms", 1) * 1e-3, 1e-9) / 1e12
        else:
            tflops = r["flops"] / (t_ms * 1e-3) / 1e12
        ai = r.get("arithmetic_intensity")
        ax.scatter([ai], [tflops], color=color, s=40,
                   label=cat if cat not in plotted else "")
        plotted.add(cat)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/B)")
    ax.set_ylabel("Achieved TFLOP/s (bf16)")
    ax.set_title("escher_14b_480p roofline")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "A6_roofline.png", dpi=120)
    plt.close(fig)


def plot_theory_vs_meas(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "04_workload_ops" / "ops.json")
    if not j:
        return
    rows = [r for r in j["rows"] if r.get("flops", 0) > 0 or r.get("bytes_hbm", 0) > 0]
    if not rows:
        return
    names: List[str] = [r["op_name"] for r in rows]
    theory = [r.get("t_bottleneck_theory_ms", 0) or 0 for r in rows]
    meas_def = [r.get("t_ms_default") or 0 for r in rows]
    meas_opt = [r.get("t_ms_optimized") or 0 for r in rows]
    x = list(range(len(names)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.45), 4.5))
    ax.bar([i - w for i in x], theory, w, label="theory")
    ax.bar(x,                meas_def, w, label="measured (default)")
    ax.bar([i + w for i in x], meas_opt, w, label="measured (optimized)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("time per call (ms)")
    ax.set_title("Per-op theory vs measured (default vs optimized)")
    ax.legend()
    ax.grid(True, axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A7_per_op_theory_vs_meas.png", dpi=120)
    plt.close(fig)


_PRETTY_SCOPE = {
    "sum_of_ops_default":   "sum-of-ops (default)",
    "sum_of_ops_optimized": "sum-of-ops (optimized)",
    "eager_e2e":            "eager e2e",
    "compiled_e2e":         "compiled e2e",
    "compiled_e2e_unavailable": "compiled e2e (n/a)",
}


def _scope_label(scope: str) -> str:
    return _PRETTY_SCOPE.get(scope, scope)


def plot_mfu(out_dir: Path, plots_dir: Path) -> None:
    """Grouped MFU bars (one cluster per scope; bars per FLOP basis) plus
    PDF reference-target overlay on the measured-peak basis.

    Reproduces the comparison the source PDF makes for the
    sum-of-ops / eager / compiled scopes: every scope's MFU is shown on three
    bases (measured chip peak, 1.26 PF rated, 2.5 PF rated) so the reader can
    see both the *ordering* (sum-of-ops < eager < compiled) and the
    *gap to the rated spec*. The PDF's headline numbers (77 / 93 / 99 %) are
    overlaid as horizontal target markers on the measured-peak bars.
    """
    j = _load(out_dir / "05_e2e_mfu" / "mfu.json")
    if not j:
        return
    rows = j.get("rows") or []
    if not rows:
        return

    bases = [
        ("mfu_measured_peak", "% of measured peak", "#1f77b4"),
        ("mfu_rated_1_26pf",  "% of 1.26 PF rated", "#ff7f0e"),
        ("mfu_rated_2_5pf",   "% of 2.5 PF rated",  "#2ca02c"),
    ]
    scopes = [r["scope"] for r in rows]
    labels = [_scope_label(s) for s in scopes]
    targets = j.get("pdf_reference_targets_pct") or {}

    n_scopes = len(scopes)
    n_bases = len(bases)
    width = 0.8 / n_bases
    x = list(range(n_scopes))

    fig, ax = plt.subplots(figsize=(max(9, n_scopes * 2.0), 5.0))
    any_value = False
    for bi, (key, base_label, color) in enumerate(bases):
        vals = [(r.get(key) or 0) * 100 for r in rows]
        if any(v > 0 for v in vals):
            any_value = True
        offsets = [xi - 0.4 + width * (bi + 0.5) for xi in x]
        bars = ax.bar(offsets, vals, width, color=color, label=base_label)
        for b, v, r in zip(bars, vals, rows):
            if r.get(key) is None:
                # Mark suppressed / unavailable rows so the empty bar is
                # disambiguated from "really 0".
                ax.text(b.get_x() + b.get_width() / 2, 1, "n/a",
                        ha="center", va="bottom", fontsize=7,
                        color="#666", rotation=90)
            elif v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=8)

    # PDF reference-target overlay (always drawn on the measured-peak band, bi=0).
    overlay_drawn = False
    for xi, scope in zip(x, scopes):
        # match e.g. "compiled_e2e_unavailable" against "compiled_e2e"
        key = scope
        if key not in targets:
            for k in targets:
                if scope.startswith(k):
                    key = k
                    break
        if key in targets:
            t = float(targets[key])
            x_left = xi - 0.4
            x_right = xi - 0.4 + width  # span only the measured-peak (first) bar
            ax.hlines(t, x_left, x_right, colors="black", lw=2.0,
                       label="PDF target" if not overlay_drawn else None,
                       zorder=5)
            ax.text(xi - 0.4 + width / 2, t + 1.5, f"PDF≈{t:.0f}%",
                    ha="center", va="bottom", fontsize=7, color="black",
                    fontweight="bold")
            overlay_drawn = True

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Model FLOPs Utilization (%)")
    ax.set_title("MFU: sum-of-ops vs eager e2e vs compiled e2e")
    if j.get("device_type") == "cpu":
        ax.text(0.5, -0.22,
                "CPU host — rated-peak (1.26 PF / 2.5 PF) bars compare a CPU number "
                "to an MI355X spec; only the measured-peak basis is meaningful.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc",
                          ec="#aaa", lw=0.5))
    ymax = max(110.0, max(((r.get("mfu_measured_peak") or 0) * 100 for r in rows),
                         default=110.0) + 12)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", ls=":")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "A8_mfu.png", dpi=120)
    plt.close(fig)
    if not any_value:
        # No scope produced a usable MFU number; leave the empty figure on
        # disk so the report still has the placeholder, but flag it.
        return


def plot_mfu_per_chunk(out_dir: Path, plots_dir: Path) -> None:
    """Per-chunk timing distribution for the timed e2e scopes.

    The PDF makes the point that compiled e2e is not just faster on the
    median but *more stable* across chunks (lower std, tighter p10..p90).
    This second figure surfaces that distribution as a strip + boxplot per
    e2e scope. Sum-of-ops scopes have no per-chunk distribution (they're a
    sum across ops, not a repeated measurement) so they're omitted here.
    """
    j = _load(out_dir / "05_e2e_mfu" / "mfu.json")
    if not j:
        return
    rows = j.get("rows") or []
    series = []
    for r in rows:
        scope = r.get("scope", "")
        if not (scope.startswith("eager") or scope.startswith("compiled")):
            continue
        times = r.get("times_ms") or []
        if not times:
            continue
        series.append({
            "scope": _scope_label(scope),
            "times": times,
            "p10": r.get("p10_ms"),
            "p90": r.get("p90_ms"),
            "median": r.get("t_total_ms"),
            "std": r.get("std_ms"),
        })
    if not series:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(series) * 3.0), 5.0))
    box_data = [s["times"] for s in series]
    positions = list(range(1, len(series) + 1))
    bp = ax.boxplot(box_data, positions=positions, widths=0.45,
                    showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], ("#1f77b4", "#2ca02c", "#9467bd")):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    # Strip plot of individual chunk times for transparency.
    y_top = max((max(s["times"]) for s in series if s["times"]), default=0)
    headroom = y_top * 0.10 if y_top else 1.0
    for i, s in enumerate(series, start=1):
        xs = [i + (k - len(s["times"]) / 2) * 0.02 for k in range(len(s["times"]))]
        ax.scatter(xs, s["times"], s=18, color="#333", alpha=0.7,
                   zorder=3)
        annot = (f"med={s['median']:.1f} ms"
                 + (f"\nσ={s['std']:.2f} ms" if s.get("std") is not None else ""))
        ax.text(i, (max(s["times"]) if s["times"] else 0) + headroom * 0.15,
                annot, ha="center", va="bottom", fontsize=8)
    if y_top:
        ax.set_ylim(top=y_top + headroom)

    ax.set_xticks(positions)
    ax.set_xticklabels([s["scope"] for s in series], rotation=10, ha="right")
    ax.set_ylabel("per-chunk forward time (ms)")
    ax.set_title("E2E per-chunk timing distribution: eager vs compiled")
    ax.grid(True, axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A8b_mfu_per_chunk.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_gemm_sweep(args.out, plots)
    plot_bandwidth(args.out, plots)
    plot_cache_curve(args.out, plots)
    plot_stability(args.out, plots)
    plot_roofline(args.out, plots)
    plot_theory_vs_meas(args.out, plots)
    plot_mfu(args.out, plots)
    plot_mfu_per_chunk(args.out, plots)
    print(f"[plots] -> {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
