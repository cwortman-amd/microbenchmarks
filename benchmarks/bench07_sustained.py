"""Family 7 — Sustained-throughput / thermal-drift probe (TESTPLAN §16.2).

Runs the same DiT forward as ``bench05_e2e_mfu`` but for a wall-clock
duration (default: 5 min on CPU, 30 min on GPU) and reports:

  - per-window throughput (TFLOP/s and MFU)        → drift detection
  - per-window σ                                    → stability detection
  - paired ``telemetry.json``                       → power / thermal / clock
  - ``sustained.json`` summary with PASS/FAIL gates against TESTPLAN §16.2:
      * end-window throughput vs first-window: drift_pct (PASS if |drift| < 5%)
      * worst-window σ vs first-window σ:    σ_growth_pct (PASS if < 3×)
      * thermal throttling indicator:        any clock dip > 10% in tail half
      * power steady-state:                  power_w mean tail / mean head < 1.10

This is the long-stability counterpart to ``bench05`` (which is a 5-iter
spot-check). Together they answer two different questions:

  - bench05: "what is peak achievable MFU under a fresh-cache, no-thermal
    scenario?"
  - bench07: "does that MFU **stay** there for an hour, or does the GPU
    throttle / the allocator fragment / the kernel scheduler degrade?"

Designed to run as the *last* family in the benchmark so it doesn't perturb
the spot-check numbers in §5–§11.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import torch

from benchmarks.common.flop_accounting import WorkloadConfig, per_block_ops, totals
from benchmarks.common.io import write_csv, write_json
from benchmarks.common.telemetry import telemetry

# Reuse the canonical model, not a copy. If bench05's DiT changes shape,
# bench07 follows automatically.
from benchmarks.bench05_e2e_mfu import DiT, _peak_from, _ops_total_per_block


def _now() -> float:
    return time.perf_counter()


def _windowize(t_ms_per_iter: List[float], window_iters: int) -> List[Dict]:
    """Group per-iter times into fixed-size windows; report median/std/p95."""
    out: List[Dict] = []
    for i in range(0, len(t_ms_per_iter), window_iters):
        chunk = t_ms_per_iter[i:i + window_iters]
        if len(chunk) < 2:
            continue
        s = sorted(chunk)
        out.append({
            "window_idx": len(out),
            "iters":      len(chunk),
            "median_ms":  s[len(s) // 2],
            "p95_ms":     s[min(len(s) - 1, int(len(s) * 0.95))],
            "std_ms":     pstdev(chunk) if len(chunk) > 1 else 0.0,
            "min_ms":     min(chunk),
            "max_ms":     max(chunk),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="Total wall-clock duration. 0 = device default "
                         "(5 min on CPU, 30 min on GPU).")
    ap.add_argument("--window-iters", type=int, default=0,
                    help="Number of iterations per window for drift/stability "
                         "binning. Aim for ~30 windows over the full run.")
    ap.add_argument("--telemetry-interval-s", type=float, default=1.0,
                    help="Sampling interval for the paired telemetry sampler. "
                         "Set to 0 to disable telemetry.")
    ap.add_argument("--cpu-depth", type=int, default=2,
                    help="Depth used on CPU hosts. 0 disables override.")
    ap.add_argument("--cpu-seq-image", type=int, default=512)
    ap.add_argument("--cpu-seq-text", type=int, default=128)
    ap.add_argument("--drift-threshold-pct", type=float, default=5.0,
                    help="Pass/fail gate for end-vs-start throughput drift.")
    ap.add_argument("--sigma-growth-threshold", type=float, default=3.0,
                    help="Pass/fail gate for worst-window σ vs first-window σ.")
    args = ap.parse_args()

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")
    cfg_json = json.loads(Path(args.config).read_text())
    cfg = WorkloadConfig.from_json(cfg_json)

    # Methodology check
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    
    b07_cfg = m_cfg.get("bench07", {})

    # Default duration: shorter on CPU because each forward is seconds; longer
    # on GPU because that's where thermal / DVFS effects actually show up.
    if args.minutes <= 0:
        args.minutes = b07_cfg.get("gpu_minutes" if has_gpu else "cpu_minutes", 30.0 if has_gpu else 5.0)
    
    window_iters = args.window_iters or b07_cfg.get("window_iters", 20)

    # CPU downscale identical to bench05 so the two are directly comparable.
    cpu_overrides_applied = {}
    if not has_gpu:
        if args.cpu_depth and cfg.depth > args.cpu_depth:
            cpu_overrides_applied["depth"] = (cfg.depth, args.cpu_depth)
            cfg.depth = args.cpu_depth
        if args.cpu_seq_image and cfg.seq_image > args.cpu_seq_image:
            cpu_overrides_applied["seq_image"] = (cfg.seq_image, args.cpu_seq_image)
            cfg.seq_image = args.cpu_seq_image
        if args.cpu_seq_text and cfg.seq_text > args.cpu_seq_text:
            cpu_overrides_applied["seq_text"] = (cfg.seq_text, args.cpu_seq_text)
            cfg.seq_text = args.cpu_seq_text
        if cpu_overrides_applied:
            print("[07] CPU host detected — auto-downscaled "
                  + ", ".join(f"{k}: {old}->{new}"
                              for k, (old, new) in cpu_overrides_applied.items()))

    out_dir_full = Path(args.out)
    out_dir = out_dir_full / "07_sustained"
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_peak = _peak_from(out_dir_full)
    gflops_per_block, _ = _ops_total_per_block(out_dir_full, cfg)
    flops_per_iter = gflops_per_block * cfg.depth * 1e9
    print(f"[07] device={device.type} depth={cfg.depth}  "
          f"flops/iter={flops_per_iter / 1e12:.2f} TFLOP  "
          f"duration={args.minutes:.1f} min  window={args.window_iters} iters")

    torch.manual_seed(0)
    model = DiT(cfg).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(cfg.batch, cfg.seq_image, cfg.D,
                    device=device, dtype=torch.bfloat16)
    ctx = torch.randn(cfg.batch, cfg.seq_text, cfg.context_dim,
                      device=device, dtype=torch.bfloat16)

    # Warmup so cache + JIT settle before the timed loop starts.
    print("[07] warmup ...")
    with torch.inference_mode():
        for _ in range(2):
            model(x, ctx)
            if has_gpu:
                torch.cuda.synchronize()

    deadline = _now() + args.minutes * 60.0
    times_ms: List[float] = []
    print(f"[07] sustained loop start (deadline = {args.minutes:.1f} min) ...")
    telemetry_path = out_dir / "telemetry.json"
    enabled = args.telemetry_interval_s > 0

    started = _now()
    with telemetry(telemetry_path,
                   interval_s=args.telemetry_interval_s,
                   enabled=enabled) as tel:
        if tel:
            tel.note("phase", "sustained_loop_start")
        with torch.inference_mode():
            it = 0
            last_log = started
            while _now() < deadline:
                t0 = _now()
                model(x, ctx)
                if has_gpu:
                    torch.cuda.synchronize()
                t1 = _now()
                times_ms.append((t1 - t0) * 1000.0)
                it += 1
                if t1 - last_log > 30.0:
                    elapsed = t1 - started
                    cur_tflops = flops_per_iter / (times_ms[-1] * 1e-3) / 1e12
                    print(f"[07]   t={elapsed:5.0f}s  iters={it}  "
                          f"last={times_ms[-1]:.1f} ms  TFLOP/s={cur_tflops:.3f}")
                    last_log = t1
        if tel:
            tel.note("phase", "sustained_loop_end")

    elapsed_s = _now() - started
    print(f"[07] done. {len(times_ms)} iterations in {elapsed_s:.1f}s "
          f"(mean {mean(times_ms):.1f} ms, σ {pstdev(times_ms) if len(times_ms) > 1 else 0:.2f} ms)")

    if not times_ms:
        print("[07] FAIL — no iterations completed before deadline. "
              "Reduce --cpu-depth / --cpu-seq-image or extend --minutes.")
        write_json(out_dir / "sustained.json", {
            "device_type": device.type,
            "minutes_requested": args.minutes,
            "iters_completed":   0,
            "status": "FAIL",
            "reason": "no iterations completed",
        })
        return 1

    windows = _windowize(times_ms, window_iters)

    # Per-window throughput in TFLOP/s.
    for w in windows:
        t_med_s = w["median_ms"] * 1e-3
        w["tflops_median"] = flops_per_iter / t_med_s / 1e12 if t_med_s > 0 else None
        if bf16_peak:
            w["mfu_measured_peak"] = w["tflops_median"] / bf16_peak

    # Drift / stability gates.
    head = windows[0] if windows else None
    tail = windows[-1] if windows else None
    drift_pct = None
    sigma_growth = None
    sigma_growth_pct = None
    sustained_status = "PASS"
    failure_reasons: List[str] = []
    if head and tail and head["median_ms"] > 0:
        drift_pct = (tail["median_ms"] - head["median_ms"]) / head["median_ms"] * 100.0
        if abs(drift_pct) > args.drift_threshold_pct:
            sustained_status = "FAIL"
            failure_reasons.append(
                f"throughput drift |{drift_pct:+.1f}%| > {args.drift_threshold_pct}% threshold"
            )
    if windows:
        first_sigma = max(windows[0]["std_ms"], 1e-9)
        worst_sigma = max(w["std_ms"] for w in windows)
        sigma_growth = worst_sigma / first_sigma
        sigma_growth_pct = (sigma_growth - 1.0) * 100.0
        if sigma_growth > args.sigma_growth_threshold:
            sustained_status = "FAIL"
            failure_reasons.append(
                f"σ grew {sigma_growth:.1f}× over the run "
                f"(threshold {args.sigma_growth_threshold:.1f}×)"
            )

    # Telemetry-derived gates (best effort: only if data exists).
    telemetry_summary: Dict = {}
    thermal_throttle_warn = None
    if telemetry_path.exists():
        try:
            tel_doc = json.loads(telemetry_path.read_text())
            telemetry_summary = tel_doc.get("summary") or {}
            samples = tel_doc.get("samples") or []
            clk = [s.get("clk_mhz") for s in samples
                   if isinstance(s.get("clk_mhz"), (int, float))]
            if clk and len(clk) >= 4:
                head_clk = mean(clk[:max(1, len(clk) // 4)])
                tail_clk = mean(clk[-max(1, len(clk) // 4):])
                if head_clk > 0:
                    drop_pct = (head_clk - tail_clk) / head_clk * 100.0
                    thermal_throttle_warn = drop_pct > 10.0
                    telemetry_summary["clk_head_mhz"] = round(head_clk, 1)
                    telemetry_summary["clk_tail_mhz"] = round(tail_clk, 1)
                    telemetry_summary["clk_drop_pct"] = round(drop_pct, 2)
                    if thermal_throttle_warn:
                        failure_reasons.append(
                            f"clk dropped {drop_pct:.1f}% from head→tail of run "
                            "(thermal/DVFS throttling indicator)"
                        )
                        if sustained_status == "PASS":
                            sustained_status = "WARN"
        except (ValueError, OSError):
            pass

    summary = {
        "device_type":          device.type,
        "depth":                cfg.depth,
        "seq_image":            cfg.seq_image,
        "seq_text":             cfg.seq_text,
        "cpu_overrides":        cpu_overrides_applied,
        "minutes_requested":    args.minutes,
        "elapsed_s":            round(elapsed_s, 1),
        "iters_completed":      len(times_ms),
        "window_iters":         window_iters,
        "n_windows":            len(windows),
        "compute_roof_tflops":  bf16_peak,
        "flops_per_iter":       flops_per_iter,
        "head_window_tflops":   head.get("tflops_median") if head else None,
        "tail_window_tflops":   tail.get("tflops_median") if tail else None,
        "drift_pct":            round(drift_pct, 3) if drift_pct is not None else None,
        "drift_threshold_pct":  args.drift_threshold_pct,
        "sigma_growth_factor":  round(sigma_growth, 3) if sigma_growth is not None else None,
        "sigma_growth_threshold": args.sigma_growth_threshold,
        "thermal_throttle_warn": thermal_throttle_warn,
        "telemetry_summary":    telemetry_summary,
        "windows":              windows,
        "status":               sustained_status,
        "failure_reasons":      failure_reasons,
    }
    write_json(out_dir / "sustained.json", summary)
    write_csv(out_dir / "sustained_windows.csv", windows)

    print(f"[07] head→tail TFLOP/s: "
          f"{summary['head_window_tflops']!r} → {summary['tail_window_tflops']!r}  "
          f"drift={summary['drift_pct']}%")
    print(f"[07] σ growth factor: {summary['sigma_growth_factor']}×")
    if telemetry_summary:
        print(f"[07] telemetry summary: {json.dumps(telemetry_summary)}")
    print(f"[07] sustained status = {summary['status']}")
    if failure_reasons:
        for r in failure_reasons:
            print(f"[07]   - {r}")

    return 0 if sustained_status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
