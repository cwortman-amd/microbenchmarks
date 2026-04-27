"""Generate TESTPLAN §13 charts from a campaign's outputs.

  A2  bf16_gemm_sweep.png      — TFLOP/s vs square size
  A3  hbm_bandwidth.png        — GB/s vs size, per op
  A6  roofline.png             — per-op AI vs achieved TFLOP/s with rooflines
  A7  per_op_theory_vs_meas.png — grouped bars: theory / default / optimized
  A8  mfu.png                  — MFU bars across scopes

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


def plot_mfu(out_dir: Path, plots_dir: Path) -> None:
    j = _load(out_dir / "05_e2e_mfu" / "mfu.json")
    if not j:
        return
    rows = j["rows"]
    scopes = [r["scope"] for r in rows]
    mfu = [(r.get("mfu_measured_peak") or 0) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(scopes, mfu, color="#1f77b4")
    for b, v in zip(bars, mfu):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("MFU (% of measured peak)")
    ax.set_title("MFU comparison: sum-of-ops vs eager vs compiled")
    ax.set_ylim(0, max(110, max(mfu) + 10) if mfu else 110)
    ax.grid(True, axis="y", ls=":")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "A8_mfu.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_gemm_sweep(args.out, plots)
    plot_bandwidth(args.out, plots)
    plot_roofline(args.out, plots)
    plot_theory_vs_meas(args.out, plots)
    plot_mfu(args.out, plots)
    print(f"[plots] -> {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
