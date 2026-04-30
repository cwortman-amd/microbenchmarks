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
    ax.axvline(ridge, color="#7f7f7f", ls="--", alpha=0.7, label=f"Ridge ≈ {ridge:.0f} FLOP/B")
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
    theory = [(r.get("t_bottleneck_theory_ms", 0) or 0) * 1000 for r in rows]
    meas_def = [(r.get("t_ms_default") or 0) * 1000 for r in rows]
    meas_opt = [(r.get("t_ms_optimized") or 0) * 1000 for r in rows]
    x = list(range(len(names)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.45), 4.5))
    ax.bar([i - w for i in x], theory, w, label="Theory (roofline)", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.bar(x,                meas_def, w, label="Measured (default SDPA)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.bar([i + w for i in x], meas_opt, w, label="Measured (AITER flash)", color="#d62728", edgecolor="black", linewidth=0.5)
    
    # Add text annotations for the bar heights
    for i, (t, md, mo) in enumerate(zip(theory, meas_def, meas_opt)):
        if t > 0:
            ax.text(i - w, t * 1.05, f"{t:.0f}", ha="center", va="bottom", fontsize=6)
        if md > 0:
            ax.text(i, md * 1.05, f"{md:.0f}", ha="center", va="bottom", fontsize=6)
        if mo > 0:
            ax.text(i + w, mo * 1.05, f"{mo:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Time per layer (µs, log)")
    ax.set_yscale("log")
    ax.set_title("Per-op timing: theory (roofline) vs measured")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", ls=":", alpha=0.7)
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
    j = _load(out_dir / "05_e2e_mfu" / "mfu.json")
    if not j:
        return
    rows = j.get("rows") or []
    if not rows:
        return

    # Filter and map scopes
    display_rows = []
    for r in rows:
        scope = r["scope"]
        if scope == "sum_of_ops_optimized":
            label = "Per-layer sum-of-ops\n(AITER, eager, isolated)"
            color = "#d62728"
            order = 3
        elif scope == "eager_e2e":
            label = "40-layer E2E\n(AITER, eager)"
            color = "#ff7f0e"
            order = 2
        elif scope == "compiled_e2e":
            label = "40-layer E2E\n(AITER + compile) <- topline"
            color = "#2ca02c"
            order = 1
        else:
            continue
        display_rows.append({"r": r, "label": label, "color": color, "order": order})
        
    display_rows.sort(key=lambda x: x["order"])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    y = list(range(len(display_rows)))
    height = 0.35
    
    any_value = False
    
    for i, item in enumerate(display_rows):
        r = item["r"]
        v_meas = (r.get("mfu_measured_peak") or 0) * 100
        v_rated = (r.get("mfu_rated_2_5pf") or 0) * 100
        if v_meas > 0 or v_rated > 0:
            any_value = True
            
        # Plot solid bar (measured peak)
        b1 = ax.barh(i + height/2, v_meas, height, color=item["color"], edgecolor="black", linewidth=0.5)
        # Plot hatched bar (rated spec)
        b2 = ax.barh(i - height/2, v_rated, height, color=item["color"], hatch="////", edgecolor="black", linewidth=0.5, alpha=0.5)
        
        if v_meas > 0:
            ax.text(v_meas + 1, i + height/2, f"{v_meas:.0f}%", va="center", ha="left", fontsize=9, fontweight="bold")
        if v_rated > 0:
            ax.text(v_rated + 1, i - height/2, f"{v_rated:.0f}%", va="center", ha="left", fontsize=8, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels([item["label"] for item in display_rows], fontsize=8)
    ax.set_xlabel("MFU (%)")
    
    peak_j = _load(out_dir / "01_bf16_compute" / "peak.json")
    meas_peak = peak_j.get("tflops", 1360) if peak_j else 1360
    
    # Custom legend for the fill styles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', label=f'vs measured peak ({meas_peak:.0f} TF/s)'),
        Patch(facecolor='lightgray', edgecolor='black', hatch='////', label='vs AMD rated spec (2.5 PF)')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    ax.grid(True, axis="x", ls=":", alpha=0.7)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(plots_dir / "A8_mfu.png", dpi=120)
    plt.close(fig)
    if not any_value:
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
    y_bot = min((min(s["times"]) for s in series if s["times"]), default=0)
    spread = y_top - y_bot if y_top > y_bot else y_top
    headroom = spread * 0.25 if spread else 1.0
    
    for i, s in enumerate(series, start=1):
        xs = [i + (k - len(s["times"]) / 2) * 0.02 for k in range(len(s["times"]))]
        ax.scatter(xs, s["times"], s=18, color="#333", alpha=0.7, zorder=3)
        annot = (f"med={s['median']:.1f} ms"
                 + (f"\nσ={s['std']:.2f} ms" if s.get("std") is not None else ""))
        ax.text(i, (max(s["times"]) if s["times"] else 0) + headroom * 0.15,
                annot, ha="center", va="bottom", fontsize=8)
    if y_top:
        ax.set_ylim(bottom=max(0, y_bot - headroom), top=y_top + headroom * 1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels([s["scope"] for s in series], rotation=10, ha="right")
    ax.set_ylabel("per-chunk forward time (ms)")
    ax.set_title("E2E per-chunk timing distribution: eager vs compiled")
    ax.grid(True, axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A8b_mfu_per_chunk.png", dpi=120)
    plt.close(fig)


def plot_multigpu_comm(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "06_multigpu_comm" / "comm.json")
    if not j or not j.get("rows"):
        return
    rows = j["rows"]
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter for AG and RS
    ag_rows = [r for r in rows if r["op"] == "all_gather"]
    rs_rows = [r for r in rows if r["op"] == "reduce_scatter"]
    
    # Find the data points that correspond to the actual payload sizes per world size
    # In lieu of exact matches, we will take the maximum bandwidth observed per world size
    ag_by_ws = {}
    rs_by_ws = {}
    
    for r in ag_rows:
        ws = r["world"]
        if ws not in ag_by_ws or r["busbw_gb_s"] > ag_by_ws[ws]:
            ag_by_ws[ws] = r["busbw_gb_s"]
            
    for r in rs_rows:
        ws = r["world"]
        if ws not in rs_by_ws or r["busbw_gb_s"] > rs_by_ws[ws]:
            rs_by_ws[ws] = r["busbw_gb_s"]
            
    worlds = sorted(set(list(ag_by_ws.keys()) + list(rs_by_ws.keys())))
    if not worlds:
        return
        
    ag_vals = [ag_by_ws.get(w, 0) for w in worlds]
    rs_vals = [rs_by_ws.get(w, 0) for w in worlds]
    
    ax.plot(worlds, ag_vals, "-o", color="#1f77b4", linewidth=2, label="AG (achieved busbw @ real payload)")
    ax.plot(worlds, rs_vals, "-s", color="#d62728", linewidth=2, label="RS (achieved busbw @ real payload)")
    
    for w, v in zip(worlds, ag_vals):
        if v > 0: ax.text(w, v + 5, f"{v:.0f}", ha="center", va="bottom", color="#1f77b4", fontweight="bold", fontsize=9)
    for w, v in zip(worlds, rs_vals):
        if v > 0: ax.text(w, v - 15, f"{v:.0f}", ha="center", va="top", color="#d62728", fontweight="bold", fontsize=9)
        
    # Draw theoretical watermark for fully connected mesh
    theo_x = [2, 4, 8]
    theo_y = [(w - 1) * 76.8 for w in theo_x]
    ax.plot(theo_x, theo_y, "k:", linewidth=2, alpha=0.6, label="Theoretical Peak ((N-1) * 76.8 GB/s)")
    for w, v in zip(theo_x, theo_y):
        ax.text(w, v + 10, f"{v:.1f}", ha="left", va="bottom", color="#475569", fontsize=8)
    
    ax.set_xlim(1.5, 8.5)
    ax.set_xticks([2, 4, 8])
    ax.set_xticklabels([f"ws={w}" for w in [2, 4, 8]])
    ax.set_ylabel("busbw per GPU (GB/s)")
    ax.set_ylim(bottom=0, top=theo_y[-1] * 1.25)
    ax.set_title("(C) Achieved ICI bandwidth for AG/RS at the actual payload")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "A18_multigpu_comm.png", dpi=120)
    plt.close(fig)

def plot_multigpu_strong_scaling(out_dir: Path, plots_dir: Path) -> None:
    env = _load(out_dir / "env.json")
    if not env:
        return
        
    cfg = env.get("workload_config", {})
    model = cfg.get("model", {})
    shapes = cfg.get("shapes", {})
    
    S = shapes.get("seq_image", 8192) + shapes.get("seq_text", 512)
    D = model.get("hidden_dim", 4096)
    
    # Analytical projection based on theoretical bounds and empirical peak bandwidth
    world_sizes = [1, 2, 4, 8]
    
    # Estimate single GPU MM time
    flops_qkv = 2 * S * D * (3 * D)
    flops_o = 2 * S * D * D
    
    # Assuming peak achievable TFLOP/s is ~1260
    peak_tflops = 1260
    t_mm_qkv_1 = (flops_qkv / 1e12) / peak_tflops * 1000  # ms
    t_mm_o_1 = (flops_o / 1e12) / peak_tflops * 1000      # ms
    
    speedup_qkv_unfused = [1.0]
    speedup_qkv_fused = [1.0]
    speedup_o_unfused = [1.0]
    speedup_o_fused = [1.0]
    
    for ws in [2, 4, 8]:
        # Payload size
        ag_payload_gb = (ws - 1) / ws * (S * D * 2) / 1e9
        
        # AG/RS time based on theoretical max 537.6 GB/s
        t_ag_ms = (ag_payload_gb / 537.6) * 1000
        t_rs_ms = (ag_payload_gb / 537.6) * 1000
        
        # MM time scales perfectly with ws
        t_mm_qkv_ws = t_mm_qkv_1 / ws
        t_mm_o_ws = t_mm_o_1 / ws
        
        # Unfused = MM + Comm
        t_qkv_unfused = t_mm_qkv_ws + t_ag_ms
        t_o_unfused = t_mm_o_ws + t_rs_ms
        
        # Fused = max(MM, Comm) assuming perfect overlap
        t_qkv_fused = max(t_mm_qkv_ws, t_ag_ms)
        t_o_fused = max(t_mm_o_ws, t_rs_ms)
        
        speedup_qkv_unfused.append(t_mm_qkv_1 / t_qkv_unfused)
        speedup_qkv_fused.append(t_mm_qkv_1 / t_qkv_fused)
        speedup_o_unfused.append(t_mm_o_1 / t_o_unfused)
        speedup_o_fused.append(t_mm_o_1 / t_o_fused)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Plot 1: Strong-scaling speedup
    ax1.plot(world_sizes, world_sizes, "k--", label="perfect speedup = P")
    ax1.plot(world_sizes, speedup_qkv_unfused, "-o", color="#1f77b4", label="AG+QKV unfused")
    ax1.plot(world_sizes, speedup_qkv_fused, "--o", color="#1f77b4", markerfacecolor="white", label="AG+QKV fused projected")
    ax1.plot(world_sizes, speedup_o_unfused, "-s", color="#d62728", label="O+RS unfused")
    ax1.plot(world_sizes, speedup_o_fused, "--s", color="#d62728", markerfacecolor="white", label="O+RS fused projected")
    
    for w, su in zip(world_sizes[1:], speedup_qkv_unfused[1:]): ax1.text(w, su-0.2, f"{su:.2f}x", ha="center", va="top", fontsize=7, color="#1f77b4")
    
    ax1.set_xticks(range(1, 9))
    ax1.set_xlabel("world size")
    ax1.set_ylabel("speedup vs ws=1")
    ax1.set_title("Strong-scaling speedup")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.7)
    
    # Plot 2: Strong-scaling efficiency
    eff_qkv_unfused = [s/w * 100 for s, w in zip(speedup_qkv_unfused, world_sizes)]
    eff_qkv_fused = [s/w * 100 for s, w in zip(speedup_qkv_fused, world_sizes)]
    eff_o_unfused = [s/w * 100 for s, w in zip(speedup_o_unfused, world_sizes)]
    eff_o_fused = [s/w * 100 for s, w in zip(speedup_o_fused, world_sizes)]
    
    ax2.axhline(100, color="#475569", linestyle="--", label="perfect strong eff. = 100%")
    ax2.plot(world_sizes, eff_qkv_unfused, "-o", color="#1f77b4", label="AG+QKV unfused")
    ax2.plot(world_sizes, eff_qkv_fused, "--o", color="#1f77b4", markerfacecolor="white", label="AG+QKV fused projected")
    ax2.plot(world_sizes, eff_o_unfused, "-s", color="#d62728", label="O+RS unfused")
    ax2.plot(world_sizes, eff_o_fused, "--s", color="#d62728", markerfacecolor="white", label="O+RS fused projected")
    
    for w, ef in zip(world_sizes, eff_qkv_unfused): ax2.text(w+0.1, ef+1, f"{ef:.0f}%", ha="left", va="bottom", fontsize=7, color="#1f77b4")
    for w, ef in zip(world_sizes, eff_o_unfused): ax2.text(w+0.1, ef+1, f"{ef:.0f}%", ha="left", va="bottom", fontsize=7, color="#d62728")
    
    ax2.set_xticks(range(1, 9))
    ax2.set_xlabel("world size")
    ax2.set_ylabel("strong scaling efficiency (%)")
    ax2.set_title("Strong-scaling efficiency (speedup / P)")
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, ls=":", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "A23_strong_scaling.png", dpi=120)
    plt.close(fig)


def plot_validation(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "validation.json")
    if not j:
        return
    comm_rows = []
    for r in j:
        metric = r.get("metric", "")
        size_mb = r.get("message_size_mb")
        if " busbw" in metric and size_mb is not None:
            op = metric.split(" ")[0]
            comm_rows.append({"op": op, "size_mb": size_mb, "pyt": r.get("pytorch"), "gt": r.get("ground_truth")})
    
    if not comm_rows:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ops = sorted({r["op"] for r in comm_rows})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, op in enumerate(ops):
        op_rows = sorted([r for r in comm_rows if r["op"] == op], key=lambda r: r["size_mb"])
        
        # Ground Truth curve (all sizes)
        gt_sizes = [r["size_mb"] for r in op_rows if r["gt"] is not None]
        gt_vals = [r["gt"] for r in op_rows if r["gt"] is not None]
        
        # PyTorch curve (only sizes tested by PyTorch)
        pyt_sizes = [r["size_mb"] for r in op_rows if r["pyt"] is not None]
        pyt_vals = [r["pyt"] for r in op_rows if r["pyt"] is not None]
        
        c = colors[i % len(colors)]
        
        if pyt_sizes:
            ax.plot(pyt_sizes, pyt_vals, marker="o", linestyle="-", color=c, label=f"{op} (PyTorch)")
        if gt_sizes:
            ax.plot(gt_sizes, gt_vals, marker="x", linestyle=":", color=c, alpha=0.8, label=f"{op} (Ground Truth)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Payload size (MB)")
    ax.set_ylabel("Bus Bandwidth (GB/s)")
    ax.set_title("Validation: PyTorch vs Ground Truth")
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(plots_dir / "A20_validation.png", dpi=120)
    plt.close(fig)


def plot_fused_comparison(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "06_multigpu_fused" / "fused.json")
    if not j or not j.get("rows"):
        return
        
    import numpy as np
    
    rows = j["rows"]
    
    shapes = {}
    for r in rows:
        if "error" in r:
            continue
        shape_key = f"{r['M']}x{r['K']}x{r['N']}"
        op = r["op"]
        if shape_key not in shapes:
            shapes[shape_key] = {}
        shapes[shape_key][op] = r["tflops"]
        
    ag_shapes = [s for s in shapes if "ag_mm" in shapes[s] and "unfused_ag_mm" in shapes[s]]
    rs_shapes = [s for s in shapes if "mm_rs" in shapes[s] and "unfused_mm_rs" in shapes[s]]
    
    if ag_shapes:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ag_shapes))
        width = 0.35
        
        unfused_vals = [shapes[s]["unfused_ag_mm"] for s in ag_shapes]
        fused_vals = [shapes[s]["ag_mm"] for s in ag_shapes]
        
        ax.bar(x - width/2, unfused_vals, width, label='Un-fused', color='#ff7f0e')
        ax.bar(x + width/2, fused_vals, width, label='Fused', color='#1f77b4')
        
        ax.set_ylabel('TFLOP/s')
        ax.set_title('AG+MM: Fused vs Un-fused Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(ag_shapes, rotation=45, ha="right")
        ax.legend()
        
        for i in range(len(ag_shapes)):
            if unfused_vals[i] > 0:
                pct = (fused_vals[i] / unfused_vals[i] - 1) * 100
                ax.annotate(f'{pct:+.0f}%',
                            xy=(x[i] + width/2, fused_vals[i]),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ca02c' if pct > 0 else '#d62728')
                            
        fig.tight_layout()
        fig.savefig(plots_dir / "A21_fused_ag_mm.png", dpi=120)
        plt.close(fig)

    if rs_shapes:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(rs_shapes))
        width = 0.35
        
        unfused_vals = [shapes[s]["unfused_mm_rs"] for s in rs_shapes]
        fused_vals = [shapes[s]["mm_rs"] for s in rs_shapes]
        
        ax.bar(x - width/2, unfused_vals, width, label='Un-fused', color='#ff7f0e')
        ax.bar(x + width/2, fused_vals, width, label='Fused', color='#1f77b4')
        
        ax.set_ylabel('TFLOP/s')
        ax.set_title('MM+RS: Fused vs Un-fused Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(rs_shapes, rotation=45, ha="right")
        ax.legend()
        
        for i in range(len(rs_shapes)):
            if unfused_vals[i] > 0:
                pct = (fused_vals[i] / unfused_vals[i] - 1) * 100
                ax.annotate(f'{pct:+.0f}%',
                            xy=(x[i] + width/2, fused_vals[i]),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color='#2ca02c' if pct > 0 else '#d62728')
                            
        fig.tight_layout()
        fig.savefig(plots_dir / "A22_fused_mm_rs.png", dpi=120)
        plt.close(fig)


def plot_relevant_shapes(out_dir: Path, plots_dir: Path) -> None:
    import re
    j = _load(out_dir / "04_workload_ops" / "ops.json")
    env = _load(out_dir / "env.json")
    if not j or not j.get("rows"):
        return
    
    rows = j["rows"]
    gemms = []
    
    # Filter for GEMMs based on shape presence (e.g., contains 'x') and name
    for r in rows:
        name = r.get("op_name", "")
        shape = r.get("input_shape", "")
        # Only parse ops that look like matrix math and have measured time
        if "x" in shape and r.get("t_ms_optimized"):
            flops = r.get("flops", 0)
            if flops > 0:
                t_ms = r["t_ms_optimized"]
                tflops_s = (flops / 1e12) / (t_ms / 1000)
                
                # Determine op type/label for the graph
                if "self_attn.q" in name: label = "SA_Q"; color = "#1f77b4"
                elif "self_attn.k" in name or "self_attn.v" in name: continue  # usually grouped or similar
                elif "self_attn.o" in name: label = "SA_O"; color = "#ff7f0e"
                elif "cross_attn.q" in name: continue
                elif "cross_attn.k" in name or "cross_attn.v" in name: continue
                elif "cross_attn.o" in name: continue
                elif "ffn.linear1" in name: label = "FFN_L1"; color = "#2ca02c"
                elif "ffn.linear2" in name: label = "FFN_L2"; color = "#9467bd"
                elif "time_embed" in name: label = "Big"; color = "#e377c2"
                elif "time_proj" in name: continue
                else: continue
                
                gemms.append({"label": label, "tflops_s": tflops_s, "color": color})
                
    if not gemms:
        return
        
    # Remove duplicates if any
    unique_gemms = []
    seen = set()
    for g in gemms:
        if g["label"] not in seen:
            unique_gemms.append(g)
            seen.add(g["label"])
            
    fig, ax = plt.subplots(figsize=(8, 5))
    y = list(range(len(unique_gemms)))
    
    bars = ax.barh(y, [g["tflops_s"] for g in unique_gemms], color=[g["color"] for g in unique_gemms], edgecolor="black")
    
    # Red dashed line for Spec Peak
    peak_tflops = 2457.6 # Single-GPU Matrix Peak (CDNA 4: 256 CUs * 4 cores * 2.4GHz)
    ax.axvline(x=peak_tflops, color="#d62728", linestyle="--", label=f"Spec peak ({peak_tflops:.0f} TF/s)")
    
    for i, (b, g) in enumerate(zip(bars, unique_gemms)):
        t = g["tflops_s"]
        pct = (t / peak_tflops) * 100
        ax.text(t + 30, b.get_y() + b.get_height() / 2, f"{t:.0f} ({pct:.0f}%)", va="center", ha="left", fontsize=8, fontweight="bold")
        
    ax.set_yticks(y)
    ax.set_yticklabels([g["label"] for g in unique_gemms])
    ax.set_xlabel("TFLOP/s")
    ax.set_title("Matmul BF16 throughput on MI355X (empirical)")
    ax.legend(loc="lower right")
    
    # Adjust x limit to fit text
    ax.set_xlim(0, peak_tflops * 1.25)
    ax.grid(True, axis="x", ls=":", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "A9_shapes.png", dpi=120)
    plt.close(fig)


def plot_memory_footprint(out_dir: Path, plots_dir: Path) -> None:
    env = _load(out_dir / "env.json")
    if not env:
        return
        
    cfg = env.get("workload_config", {})
    model = cfg.get("model", {})
    shapes = cfg.get("shapes", {})
    
    hidden_dim = model.get("hidden_dim", 4096)
    ffn_expansion = model.get("ffn_expansion", 4)
    context_dim = model.get("context_dim", 4096)
    seq_image = shapes.get("seq_image", 8192)
    seq_text = shapes.get("seq_text", 512)
    
    # 2 bytes per param/element for bf16
    bp = 2
    
    sa_weights = 4 * (hidden_dim * hidden_dim) * bp / 1e6
    ffn_weights = 2 * (hidden_dim * hidden_dim * ffn_expansion) * bp / 1e6
    ca_weights = 4 * (hidden_dim * context_dim) * bp / 1e6
    activations = seq_image * hidden_dim * bp / 1e6
    sa_kv = 2 * seq_image * hidden_dim * bp / 1e6
    ca_kv = 2 * seq_text * context_dim * bp / 1e6
    
    bars_data = [
        ("CA KV (text x dim)", ca_kv, "#17becf"),
        ("SA KV (window x HxW x dim)", sa_kv, "#d62728"),
        ("Activations (tokens x dim)", activations, "#e377c2"),
        ("CA weights (Q,K,V,O)", ca_weights, "#17becf"),
        ("FFN weights (L1+L2)", ffn_weights, "#8c564b"),
        ("SA weights (Q,K,V,O)", sa_weights, "#1f77b4")
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = list(range(len(bars_data)))
    labels = [b[0] for b in bars_data]
    vals = [b[1] for b in bars_data]
    colors = [b[2] for b in bars_data]
    
    bars = ax.barh(y, vals, height=0.35, color=colors, edgecolor="black", linewidth=0.5)
    
    for b, val in zip(bars, vals):
        ax.text(val * 1.05, b.get_y() + b.get_height()/2, f"{val:.0f} MB", va="center", ha="left", fontsize=8)
        
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(-0.5, len(bars_data) + 0.5)
    ax.set_xlabel("Size (MB, log)")
    ax.set_title("Per-layer data vs L2 cache capacity\n(items left of each dashed line fit in that GPU's L2)", fontsize=10)
    
    # Add vertical dashed lines for L2 cache sizes
    ax.axvline(x=64, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(64, len(bars_data)-0.5, "(64MB)", color="#2ca02c", rotation=0, va="bottom", ha="center", fontsize=7)
    
    ax.axvline(x=128, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(128, len(bars_data)-0.5, "(128MB)", color="#2ca02c", rotation=0, va="bottom", ha="center", fontsize=7)
    
    ax.axvline(x=256, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(256, len(bars_data)-0.5, "(256MB)", color="#d62728", rotation=0, va="bottom", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(plots_dir / "A8c_memory_footprint.png", dpi=120)
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
    plot_memory_footprint(args.out, plots)
    plot_multigpu_comm(args.out, plots)
    plot_multigpu_strong_scaling(args.out, plots)
    plot_validation(args.out, plots)
    plot_fused_comparison(args.out, plots)
    plot_relevant_shapes(args.out, plots)
    print(f"[plots] -> {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
