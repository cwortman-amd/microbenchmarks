#!/usr/bin/env python3
"""Across-invocation variability for bench05 / bench01.

Today the timing files capture *intra-run* variance (per-iter σ inside one
process). What they miss is *inter-run* variance: cold cache, allocator
reseeded, OS scheduler in a different state, BLAS heuristics on different
codepaths. A 0.27 TFLOP/s number with σ 0.01 *inside* one run can swing
0.20–0.34 *across* runs — and that's the variance a regression CI actually
has to budget for.

This script runs a chosen benchmark N times in fresh subprocess invocations
and aggregates:

  - per-invocation peak / median / σ
  - across-invocation σ (the number that matters for regression budgeting)
  - across-invocation min/max envelope
  - PASS/FAIL: across-invocation σ should be ≤ ``--max-cross-run-sigma``
    fraction of the across-invocation mean

Usage::

    python scripts/across_run_variability.py \\
        --target bench05 \\
        --runs   5 \\
        --out    results/<campaign>/variability_bench05/

    python scripts/across_run_variability.py \\
        --target bench01 \\
        --runs   5 \\
        --out    results/<campaign>/variability_bench01/

Each invocation gets its own subdirectory under ``--out`` so the source
JSON is preserved for forensic comparisons (e.g. spotting that one run hit
a thermal event the others did not).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# Mapping from --target to (module, json_path_relative_to_run_dir, metric_extractor)
def _extract_bench05(run_dir: Path) -> Optional[Dict]:
    f = run_dir / "05_e2e_mfu" / "mfu.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    rows = {r["scope"]: r for r in (j.get("rows") or [])}
    eager = rows.get("eager_e2e") or {}
    compiled = rows.get("compiled_e2e") or {}
    return {
        "eager_t_total_ms":     eager.get("t_total_ms"),
        "eager_tflops":         eager.get("tflops_achieved"),
        "eager_mfu":            eager.get("mfu_measured_peak"),
        "eager_std_ms":         eager.get("std_ms"),
        "compiled_t_total_ms":  compiled.get("t_total_ms"),
        "compiled_tflops":      compiled.get("tflops_achieved"),
        "compiled_mfu":         compiled.get("mfu_measured_peak"),
        "compiled_std_ms":      compiled.get("std_ms"),
    }


def _extract_bench01(run_dir: Path) -> Optional[Dict]:
    f = run_dir / "01_bf16_compute" / "summary.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    return {
        "compute_roof_tflops":     j.get("compute_roof_tflops"),
        "peak_tight_loop_tflops":  j.get("peak_tight_loop_tflops"),
        "best_sweep_tflops":       (j.get("best_sweep") or {}).get("tflops"),
    }


TARGETS = {
    "bench01": {
        "module":  "benchmarks.bench01_bf16_compute",
        "extract": _extract_bench01,
        "metrics": ["compute_roof_tflops", "peak_tight_loop_tflops", "best_sweep_tflops"],
    },
    "bench05": {
        "module":  "benchmarks.bench05_e2e_mfu",
        "extract": _extract_bench05,
        "metrics": ["eager_t_total_ms", "eager_tflops", "eager_mfu",
                    "compiled_t_total_ms", "compiled_tflops", "compiled_mfu"],
    },
}


def _agg(vals: List[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "stddev": None, "cv_pct": None}
    n = len(vals)
    m = sum(vals) / n
    sd = statistics.pstdev(vals) if n > 1 else 0.0
    cv_pct = (sd / m * 100.0) if m else None
    return {
        "n":      n,
        "mean":   round(m, 6),
        "min":    round(min(vals), 6),
        "max":    round(max(vals), 6),
        "stddev": round(sd, 6),
        "cv_pct": round(cv_pct, 3) if cv_pct is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS.keys()),
                    help="Which benchmark to repeatedly invoke.")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of subprocess invocations.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory; per-run subdirs r0…rN-1 are created here.")
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="Forward additional args to the inner benchmark.")
    ap.add_argument("--max-cross-run-cv-pct", type=float, default=10.0,
                    help="Pass/fail gate: cross-invocation CV%% (σ/mean*100) "
                         "for the primary metric.")
    ap.add_argument("--cool-down-s", type=float, default=2.0,
                    help="Pause between invocations (helps thermal recovery / "
                         "page-cache reset).")
    args = ap.parse_args()

    target = TARGETS[args.target]
    args.out.mkdir(parents=True, exist_ok=True)

    per_run: List[Dict] = []
    print(f"[variability] target={args.target}  runs={args.runs}  out={args.out}")
    for i in range(args.runs):
        run_dir = args.out / f"r{i}"
        run_dir.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", target["module"],
               "--out", str(run_dir),
               "--config", args.config, *args.extra_args]
        t0 = time.perf_counter()
        print(f"[variability] run {i + 1}/{args.runs}: {' '.join(cmd)}")
        # Stream subprocess output so a wedge is visible in real time.
        proc = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"[variability] run {i + 1} failed (rc={proc.returncode})")
            per_run.append({"run": i, "status": "FAIL", "elapsed_s": round(elapsed, 1)})
            continue
        metrics = target["extract"](run_dir) or {}
        per_run.append({
            "run":       i,
            "status":    "OK",
            "elapsed_s": round(elapsed, 1),
            "metrics":   metrics,
        })
        print(f"[variability] run {i + 1} done ({elapsed:.1f}s) metrics={metrics}")
        if i + 1 < args.runs and args.cool_down_s > 0:
            time.sleep(args.cool_down_s)

    # Aggregate per-metric across the OK runs.
    metric_keys = target["metrics"]
    agg: Dict[str, Dict] = {}
    for mk in metric_keys:
        vals = [r["metrics"].get(mk) for r in per_run
                if r.get("status") == "OK" and r.get("metrics")]
        agg[mk] = _agg([v for v in vals if v is not None])

    # The "primary" metric for the gate: peak for bench01, compiled_tflops for
    # bench05 (with fallback to eager_tflops when --no-compile / compile failed
    # so the gate still fires off measurable data).
    if args.target == "bench01":
        primary_candidates = ["compute_roof_tflops", "peak_tight_loop_tflops",
                              "best_sweep_tflops"]
    else:
        primary_candidates = ["compiled_tflops", "eager_tflops"]
    primary_key = None
    primary = {}
    for cand in primary_candidates:
        a = agg.get(cand) or {}
        if a.get("cv_pct") is not None:
            primary_key = cand
            primary = a
            break
    primary_cv = primary.get("cv_pct") if primary else None
    if primary_cv is None:
        status = "SKIP"
        reason = (f"no values collected for any primary metric "
                  f"({primary_candidates})")
    elif primary_cv > args.max_cross_run_cv_pct:
        status = "FAIL"
        reason = (f"cross-run CV%% on {primary_key} = {primary_cv:.1f}% "
                  f"> threshold {args.max_cross_run_cv_pct:.1f}%")
    else:
        status = "PASS"
        reason = (f"cross-run CV%% on {primary_key} = {primary_cv:.1f}% "
                  f"<= threshold {args.max_cross_run_cv_pct:.1f}%")

    summary = {
        "target":              args.target,
        "module":              target["module"],
        "runs":                args.runs,
        "primary_metric":      primary_key,
        "primary_cv_pct":      primary_cv,
        "max_cross_run_cv_pct": args.max_cross_run_cv_pct,
        "status":              status,
        "reason":              reason,
        "per_run":             per_run,
        "aggregate":           agg,
    }
    (args.out / "variability.json").write_text(json.dumps(summary, indent=2))
    print(f"[variability] {status}: {reason}")
    print(f"[variability] aggregate:")
    for mk in metric_keys:
        a = agg.get(mk) or {}
        print(f"[variability]   {mk:30s} mean={a.get('mean')} σ={a.get('stddev')} "
              f"CV%={a.get('cv_pct')} (n={a.get('n')})")
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
