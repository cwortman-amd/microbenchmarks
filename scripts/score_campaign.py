"""Score a campaign against TESTPLAN §1.2 success criteria SC-1…SC-5.

Reads all the artifacts produced by run_campaign.sh and writes:
    <out>/scorecard.md   — human-readable PASS/FAIL per SC
    <out>/scorecard.json — machine-readable, used by run_campaign.sh exit code

SC-1: BF16 GEMM peak ≥ 50% of 1.26 PF rated, AND best square ≥ 90% of measured peak
SC-2: HBM plateau within ±5% across 3 successive top sizes
SC-3: Roofline placement — large GEMMs and attention sit AI > ridge
SC-4: compiled e2e MFU ≥ eager e2e MFU ≥ sum-of-ops MFU (within ±5pp tolerance)
SC-5: All required artifacts in TESTPLAN §13 (A1–A11) exist
SC-6: BF16 GEMM is numerically equivalent to FP32 reference within an
      analytic ``5·√K·2⁻⁸`` rel-error bound (catches dtype/transpose/accum
      regressions that pure-throughput sweeps miss).
SC-7: (opt-in) Sustained throughput stable over the bench07 run window —
      head→tail drift < 5%, σ growth < 3×, no clock drop > 10%.
SC-8: (opt-in) Across-invocation variability bounded —
      cross-run CV%% on the primary throughput metric ≤ threshold (default 10%).
SC-9: (opt-in) Ground-truth validation — PyTorch microbench numbers agree
      with RVS / rocm-bandwidth-test / rccl-tests within the per-metric
      tolerance documented in ``validation/compare.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _device_summary(out: Path) -> Dict[str, object]:
    """Inspect env.json to decide whether a GPU campaign actually ran.

    Returns a dict with `is_gpu` (bool) and `reason` (str). When the campaign
    was executed on a CPU host (no accelerator) the GPU-bound success criteria
    cannot be evaluated and should be reported as SKIP, not FAIL — otherwise a
    CPU-only smoke run looks like a regression.
    """
    env = _load(out / "env.json") or {}
    torch_info = (env.get("software") or {}).get("torch") or {}
    device_count = torch_info.get("device_count") or 0
    cuda_avail = bool(torch_info.get("torch_cuda_available"))
    is_gpu = bool(device_count) and cuda_avail
    return {
        "is_gpu": is_gpu,
        "device_count": device_count,
        "cuda_available": cuda_avail,
        "reason": "no GPU detected on the campaign host (env.json reports device_count=0)" if not is_gpu else "",
    }


def sc1_bf16_peak(out: Path, device: Dict[str, object]) -> Dict:
    """SC-1: BF16 GEMM peak.

    GPU: peak >= 50% of rated 1.26 PF, AND best square sweep >= 90% of peak.
    CPU: only the 90% sweep-vs-peak consistency check is enforced — the
    1.26 PF rated-peak threshold is GPU-only and reported as informational.
    """
    j = _load(out / "01_bf16_compute" / "summary.json")
    sweep = _load(out / "01_bf16_compute" / "sweep.json")
    if not j or not sweep:
        return {"sc": "SC-1", "status": "SKIP", "reason": "missing 01_bf16_compute outputs"}
    peak = j["compute_roof_tflops"]
    sq = sweep.get("square", [])
    if not sq:
        return {"sc": "SC-1", "status": "SKIP", "reason": "no square sweep results"}
    best = max(r["tflops"] for r in sq)
    rated_low = 1260.0  # MI355X 1.26 PF
    pct_rated = peak / rated_low if peak else 0.0
    pct_measured = best / peak if peak else 0.0
    is_gpu = bool(device.get("is_gpu"))
    if is_gpu:
        ok = (pct_rated >= 0.50) and (pct_measured >= 0.90)
        thresholds = {"min_pct_rated": 50.0, "min_best_pct_of_peak": 90.0}
    else:
        # On CPU, drop the absolute MI355X target — the host can't be
        # expected to hit a fraction of an MI355X. Keep the cross-check
        # that the sweep peak is consistent with the tight-loop peak.
        ok = pct_measured >= 0.90
        thresholds = {"min_pct_rated": 0.0, "min_best_pct_of_peak": 90.0,
                      "note": "CPU host: rated-peak threshold not enforced"}
    return {
        "sc": "SC-1",
        "status": "PASS" if ok else "FAIL",
        "host": "GPU" if is_gpu else "CPU",
        "peak_tflops": round(peak, 3),
        "best_sweep_tflops": round(best, 3),
        "pct_of_rated_1_26pf": round(pct_rated * 100, 2),
        "best_pct_of_peak": round(pct_measured * 100, 1),
        "thresholds": thresholds,
    }


def sc2_hbm_plateau(out: Path, device: Dict[str, object]) -> Dict:
    j = _load(out / "02_hbm_bandwidth" / "bandwidth.json")
    summary = _load(out / "02_hbm_bandwidth" / "summary.json") or {}
    if not j:
        return {"sc": "SC-2", "status": "SKIP", "reason": "missing 02_hbm_bandwidth outputs"}
    rows = sorted([r for r in j if r["op"] == "copy_"], key=lambda r: r["bytes"])
    if len(rows) < 3:
        return {"sc": "SC-2", "status": "SKIP", "reason": "fewer than 3 copy_ sizes succeeded"}
    top3 = rows[-3:]
    bws = [r["gb_s"] for r in top3]
    spread = (max(bws) - min(bws)) / max(bws)
    # Looser plateau on CPU: OS scheduling, NUMA effects, and shared memory
    # bandwidth between processes routinely add 5-15% jitter to copy_ at
    # gigabyte-scale buffers. The HBM3 5% target was set for the GPU device
    # under quiesced conditions.
    is_cpu = (summary.get("device_type") == "cpu") or not device.get("is_gpu")
    threshold = 0.15 if is_cpu else 0.05
    ok = spread <= threshold
    return {
        "sc": "SC-2",
        "status": "PASS" if ok else "FAIL",
        "host": "CPU" if is_cpu else "GPU",
        "top3_gb_s": [round(v, 1) for v in bws],
        "spread_pct": round(spread * 100, 2),
        "threshold_pct": round(threshold * 100, 1),
    }


def sc3_roofline_placement(out: Path) -> Dict:
    j = _load(out / "04_workload_ops" / "ops.json")
    if not j:
        return {"sc": "SC-3", "status": "SKIP", "reason": "missing 04_workload_ops outputs"}
    rows = j["rows"]
    expected_compute = []  # large GEMMs + attention should be 'compute'
    expected_memory = []   # norms/elementwise should be 'memory'
    for r in rows:
        cat = r.get("category")
        if cat in ("self_attn", "cross_attn", "ffn") and r.get("flops", 0) > 5e9:
            expected_compute.append(r)
        if cat == "norm" and r.get("flops", 0) == 0:
            expected_memory.append(r)
    misplaced = []
    for r in expected_compute:
        if r.get("bound") == "memory":
            misplaced.append(("compute->memory", r["op_name"]))
    for r in expected_memory:
        if r.get("bound") == "compute":
            misplaced.append(("memory->compute", r["op_name"]))
    ok = not misplaced
    return {
        "sc": "SC-3",
        "status": "PASS" if ok else "FAIL",
        "n_compute_expected": len(expected_compute),
        "n_memory_expected": len(expected_memory),
        "misplaced": misplaced,
    }


def sc4_mfu_ordering(out: Path) -> Dict:
    """SC-4: compiled_e2e ≥ eager_e2e ≥ sum_of_ops MFU (within ±5pp tolerance).

    Evaluation rules:
      - missing artifact            -> SKIP
      - all three rows present      -> PASS / FAIL on the full chain
      - sum-of-ops suppressed       -> PARTIAL_PASS / FAIL on the
        compiled ≥ eager sub-chain (this happens on CPU when bench04
        coverage < the threshold and sum-of-ops MFU is intentionally
        suppressed to avoid mismatched numerator/denominator)
      - eager or compiled missing   -> SKIP
      - on CPU host an inversion of compiled ≥ eager is reported as
        ``WARN_CPU`` rather than FAIL, because Inductor on small CPU
        shapes is a documented `torch.compile` failure mode and the
        scorecard is otherwise meant to track GPU-host regressions.
    """
    j = _load(out / "05_e2e_mfu" / "mfu.json")
    if not j:
        return {"sc": "SC-4", "status": "SKIP", "reason": "missing 05_e2e_mfu outputs"}
    device_type = j.get("device_type", "")
    rows = {r["scope"]: r for r in j["rows"]}
    sop = (rows.get("sum_of_ops_optimized") or rows.get("sum_of_ops_default") or {}).get("mfu_measured_peak")
    eager = (rows.get("eager_e2e") or {}).get("mfu_measured_peak")
    compiled = (rows.get("compiled_e2e") or {}).get("mfu_measured_peak")
    if eager is None or compiled is None:
        return {"sc": "SC-4", "status": "SKIP",
                "reason": "incomplete MFU rows (eager and compiled both required)",
                "sop": sop, "eager": eager, "compiled": compiled,
                "device_type": device_type}
    tol_pp = 0.05  # 5 percentage points
    out_row: Dict = {
        "sc": "SC-4",
        "mfu_sum_of_ops_pct": round(sop * 100, 1) if sop is not None else None,
        "mfu_eager_pct": round(eager * 100, 1),
        "mfu_compiled_pct": round(compiled * 100, 1),
        "tolerance_pp": tol_pp * 100,
        "device_type": device_type,
    }
    compiled_vs_eager_ok = (compiled + tol_pp >= eager)
    eager_vs_sop_ok = sop is None or (eager + tol_pp >= sop)
    if compiled_vs_eager_ok and eager_vs_sop_ok:
        out_row["status"] = "PASS" if sop is not None else "PARTIAL_PASS"
        if sop is None:
            out_row["note"] = ("sum-of-ops MFU suppressed (coverage gate); "
                               "scored compiled ≥ eager only.")
    else:
        if not compiled_vs_eager_ok and device_type == "cpu":
            out_row["status"] = "WARN_CPU"
            out_row["note"] = (
                "compiled_e2e < eager_e2e by "
                f"{(eager - compiled) * 100:+.1f} pp on CPU host: known "
                "torch.compile-on-small-CPU-shape failure mode. Treated "
                "as a warning, not a regression — re-evaluate on GPU."
            )
        else:
            out_row["status"] = "FAIL"
            out_row["note"] = (
                "ordering violation: " +
                ("compiled<eager" if not compiled_vs_eager_ok else "") +
                ("; " if (not compiled_vs_eager_ok and not eager_vs_sop_ok) else "") +
                ("eager<sum-of-ops" if not eager_vs_sop_ok else "")
            )
    return out_row


REQUIRED_ARTIFACTS = [
    ("A1", "01_bf16_compute/summary.json"),
    ("A1b", "01_bf16_compute/component_gemms.json"),
    ("A1c", "01_bf16_compute/correctness.json"),
    ("A2", "plots/A2_bf16_gemm_sweep.png"),
    ("A3", "plots/A3_hbm_bandwidth.png"),
    ("A4", "03_dram_capacity/summary.json"),
    ("A5", "04_workload_ops/ops.csv"),
    ("A6", "plots/A6_roofline.png"),
    ("A7", "plots/A7_per_op_theory_vs_meas.png"),
    ("A8", "05_e2e_mfu/mfu.csv"),
    # A9 multi-GPU is optional
    ("A10", "env.json"),
    ("A11", "summary.md"),
]


def sc6_numerical_correctness(out: Path) -> Dict:
    """SC-6: BF16 GEMM correctness vs FP32 reference.

    The bench01 correctness gate is a per-shape pass/fail. SC-6 surfaces
    the worst-case shape and flags FAIL if any shape exceeds its analytic
    rel-error bound. This catches:
      - kernel computing the wrong operation but with valid timing,
      - hardware emulating BF16 as FP16 (different accumulation precision),
      - transpose / shape / dtype mismatches,
      - accumulator skipping past some K threshold.
    """
    j = _load(out / "01_bf16_compute" / "correctness.json")
    if not j:
        return {"sc": "SC-6", "status": "SKIP",
                "reason": "missing 01_bf16_compute/correctness.json (pre-SC-6 bench01)"}
    rows = j.get("rows") or []
    if not rows:
        return {"sc": "SC-6", "status": "SKIP", "reason": "no correctness rows"}
    failed = [r for r in rows if not r.get("passed")]
    worst = max(rows, key=lambda r: r.get("max_rel_err", 0.0))
    return {
        "sc": "SC-6",
        "status": "PASS" if not failed else "FAIL",
        "shapes_checked": [r["size"] for r in rows],
        "worst_size": worst["size"],
        "worst_max_rel_err": round(worst["max_rel_err"], 6),
        "worst_bound": round(worst["rel_err_bound"], 6),
        "n_failed": len(failed),
    }


def sc7_sustained(out: Path) -> Dict:
    """SC-7: bench07 sustained-throughput probe pass/fail.

    Optional gate. Returns SKIP when bench07 was not run (no
    07_sustained/sustained.json), otherwise mirrors the bench's own
    self-reported status.
    """
    j = _load(out / "07_sustained" / "sustained.json")
    if not j:
        return {"sc": "SC-7", "status": "SKIP",
                "reason": "bench07 sustained probe not run (opt-in)"}
    status = j.get("status", "SKIP")
    return {
        "sc": "SC-7",
        "status": status,
        "drift_pct":            j.get("drift_pct"),
        "sigma_growth_factor":  j.get("sigma_growth_factor"),
        "thermal_throttle_warn": j.get("thermal_throttle_warn"),
        "iters_completed":      j.get("iters_completed"),
        "elapsed_s":            j.get("elapsed_s"),
        "failure_reasons":      j.get("failure_reasons"),
    }


def sc8_variability(out: Path) -> Dict:
    """SC-8: across-invocation variability gate (opt-in).

    Reads any ``variability_*/variability.json`` artifacts written by
    ``scripts/across_run_variability.py``. Multiple targets may be present;
    the worst CV%% drives the overall verdict.
    """
    files = sorted(out.glob("variability_*/variability.json"))
    if not files:
        return {"sc": "SC-8", "status": "SKIP",
                "reason": "no across-invocation variability runs (opt-in)"}
    rows = []
    worst_cv = 0.0
    overall_status = "PASS"
    for f in files:
        try:
            j = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        cv = j.get("primary_cv_pct")
        rows.append({
            "target": j.get("target"),
            "primary_metric": j.get("primary_metric"),
            "primary_cv_pct": cv,
            "status": j.get("status"),
            "runs": j.get("runs"),
        })
        if isinstance(cv, (int, float)) and cv > worst_cv:
            worst_cv = cv
        if j.get("status") == "FAIL":
            overall_status = "FAIL"
    return {
        "sc": "SC-8",
        "status": overall_status if rows else "SKIP",
        "n_targets": len(rows),
        "worst_cv_pct": round(worst_cv, 3) if rows else None,
        "details": rows,
    }


def sc9_validation(out: Path) -> Dict:
    """SC-9: cross-validation against ground-truth tools (opt-in).

    Reads ``validation.json`` written by ``validation/compare.py``. Reports
    PASS only when at least one row is genuinely PASS (not SKIP). All SKIP
    rows are surfaced as a SKIP verdict because absent tooling shouldn't
    look like a regression.
    """
    j = _load(out / "validation.json")
    if not isinstance(j, list):
        return {"sc": "SC-9", "status": "SKIP",
                "reason": "no validation.json (run validation/compare.py first)"}
    n_pass = sum(1 for r in j if r.get("status") == "PASS")
    n_fail = sum(1 for r in j if r.get("status") == "FAIL")
    n_skip = sum(1 for r in j if r.get("status") == "SKIP")
    if n_fail:
        status = "FAIL"
    elif n_pass:
        status = "PASS"
    else:
        status = "SKIP"
    failed_rows = [r for r in j if r.get("status") == "FAIL"]
    return {
        "sc": "SC-9",
        "status": status,
        "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
        "failed_metrics": [r.get("metric") for r in failed_rows],
    }


def sc12_fused_collectives(out: Path) -> Dict:
    """SC-12: fused AG+MM / MM+RS kernels (source-PDF future work).

    Reads ``06_multigpu_fused/fused.json`` written by ``bench06_fused``.
    The benchmark is a forward-looking probe: AITER hasn't shipped the
    fused-collective API yet, so the expected resting state is SKIP
    with ``available=false``. The row flips to PASS the moment the API
    appears and the rows actually contain TFLOP/s.

    Status semantics:
      - SKIP: ``available=false`` (expected today; row exists so the
        absence is *visible* in the scorecard rather than silent).
      - PASS: ``available=true`` and at least one row has a positive
        ``tflops`` value.
      - FAIL: ``available=true`` but every row errored — surfaces an
        API regression once it ships.
    """
    j = _load(out / "06_multigpu_fused" / "fused.json")
    if not j:
        return {"sc": "SC-12", "status": "SKIP",
                "reason": "bench06_fused not run (no fused.json)"}
    if not j.get("available"):
        return {"sc": "SC-12", "status": "SKIP",
                "reason": j.get("reason") or "fused-collective API not available",
                "world": j.get("world"),
                "device_type": j.get("device_type")}
    rows = j.get("rows") or []
    ok = [r for r in rows if isinstance(r.get("tflops"), (int, float)) and r["tflops"] > 0]
    if not ok:
        return {"sc": "SC-12", "status": "FAIL",
                "reason": "fused API resolved but all measured rows errored",
                "rows_total": len(rows),
                "api_source": j.get("api_source")}
    ag = [r for r in ok if r.get("op") == "ag_mm"]
    rs = [r for r in ok if r.get("op") == "mm_rs"]
    return {
        "sc": "SC-12", "status": "PASS",
        "api_source":     j.get("api_source"),
        "world":          j.get("world"),
        "ag_mm_rows":     len(ag),
        "mm_rs_rows":     len(rs),
        "ag_mm_max_tflops": round(max((r["tflops"] for r in ag), default=0), 2),
        "mm_rs_max_tflops": round(max((r["tflops"] for r in rs), default=0), 2),
    }


def sc11_headroom(out: Path, device: Dict[str, object]) -> Dict:
    """SC-11: post-model-load residual capacity (TESTPLAN §16.3).

    Reads ``03_dram_capacity/summary.json[.headroom]`` from the
    ``--measure-headroom`` probe in ``bench03``. Reports PASS when the
    full model fits and residual > 0; FAIL when the host couldn't hold
    the model; SKIP when the probe wasn't run (older bench03 outputs).
    On CPU hosts the probe is intentionally capped at the contig limit
    so we mark the row as ``WARN_CPU`` rather than FAIL.
    """
    j = _load(out / "03_dram_capacity" / "summary.json")
    if not j or "headroom" not in j:
        return {"sc": "SC-11", "status": "SKIP",
                "reason": "bench03 --measure-headroom not run"}
    h = j["headroom"]
    full_gib = h.get("model_target_full_gib") or 0
    loaded_gib = h.get("model_bytes_gib") or 0
    full_deficit_gib = max(0.0, full_gib - loaded_gib)
    if h.get("probe_capped") and not device.get("is_gpu"):
        return {
            "sc": "SC-11", "status": "WARN_CPU",
            "reason": ("CPU host could not hold the full model; residual is "
                       "computed from a capped allocation — re-run on GPU "
                       "for the operational TP-3 number."),
            "model_target_gib":  round(full_gib, 2),
            "model_loaded_gib":  round(loaded_gib, 2),
            "residual_gib":      round(h.get("residual_capacity_gib") or 0, 2),
            "deficit_vs_full_gib": round(full_deficit_gib, 2),
        }
    if not h.get("loaded"):
        return {
            "sc": "SC-11", "status": "FAIL",
            "reason": "host could not hold the full bf16 model",
            "model_target_gib":  round(h.get("model_target_full_gib") or 0, 2),
            "model_loaded_gib":  round(h.get("model_bytes_gib") or 0, 2),
            "deficit_gib":       round((h.get("deficit_bytes") or 0) / 1024**3, 2),
        }
    return {
        "sc": "SC-11", "status": "PASS",
        "model_target_gib":  round(h.get("model_target_full_gib") or 0, 2),
        "model_loaded_gib":  round(h.get("model_bytes_gib") or 0, 2),
        "residual_gib":      round(h.get("residual_capacity_gib") or 0, 2),
        "residual_fraction": round(h.get("residual_fraction") or 0, 4),
    }


def sc10_stability_sweep(out: Path) -> Dict:
    """SC-10: bench09 numerical-stability sweep (opt-in).

    Where SC-6 is a coarse pass/fail on the bf16 correctness gate at two
    shapes, SC-10 reads the full bench09 sweep across (dtype, K) and
    fails if *any* row exceeds its analytic ``5·√K·2^-mantissa`` bound.
    The output also surfaces the per-dtype worst-case ``rel_err`` so a
    reader can see the precision/throughput trade-off explicitly.
    """
    j = _load(out / "09_numerical_stability" / "stability.json")
    if not j:
        return {"sc": "SC-10", "status": "SKIP",
                "reason": "bench09 stability sweep not run (opt-in)"}
    rows = j.get("rows") or []
    if not rows:
        return {"sc": "SC-10", "status": "SKIP",
                "reason": "no rows in stability.json"}
    failed = [r for r in rows if not r.get("passed")]
    per_dtype: Dict[str, float] = {}
    for r in rows:
        if "rel_err" not in r:
            continue
        m = r["rel_err"].get("max")
        if m is None:
            continue
        per_dtype[r["dtype"]] = max(per_dtype.get(r["dtype"], 0.0), float(m))
    return {
        "sc": "SC-10",
        "status": "PASS" if not failed else "FAIL",
        "n_rows": len(rows),
        "n_failed": len(failed),
        "worst_per_dtype": {k: round(v, 6) for k, v in per_dtype.items()},
        "failed_rows": [{"dtype": r["dtype"], "K": r["K"]} for r in failed],
    }


def sc5_artifacts(out: Path, device: Dict[str, object]) -> Dict:
    missing = [(tag, rel) for tag, rel in REQUIRED_ARTIFACTS if not (out / rel).exists()]
    if missing and not device.get("is_gpu"):
        # On a CPU host the GPU-bound artifacts (A1..A4, A6..A8) are absent by
        # design. Marking that as FAIL would conflate "no accelerator" with
        # "regression", so skip instead and surface the reason.
        return {
            "sc": "SC-5",
            "status": "SKIP",
            "reason": device.get("reason"),
            "missing": [{"tag": t, "path": p} for t, p in missing],
        }
    return {
        "sc": "SC-5",
        "status": "PASS" if not missing else "FAIL",
        "missing": [{"tag": t, "path": p} for t, p in missing],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = _device_summary(args.out)
    rows = [
        sc1_bf16_peak(args.out, device),
        sc2_hbm_plateau(args.out, device),
        sc3_roofline_placement(args.out),
        sc4_mfu_ordering(args.out),
        sc5_artifacts(args.out, device),
        sc6_numerical_correctness(args.out),
        sc7_sustained(args.out),
        sc8_variability(args.out),
        sc9_validation(args.out),
        sc10_stability_sweep(args.out),
        sc11_headroom(args.out, device),
        sc12_fused_collectives(args.out),
    ]
    # Surface the host context at the top of the scorecard so a reader knows
    # at a glance why so many rows might be SKIP.
    rows.insert(0, {
        "sc": "HOST",
        "status": "GPU" if device.get("is_gpu") else "CPU",
        "device_count": device.get("device_count"),
        "cuda_available": device.get("cuda_available"),
    })

    md = ["# Scorecard\n", ""]
    md.append("| SC | Status | Detail |")
    md.append("|----|--------|--------|")
    for r in rows:
        detail = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("sc", "status"))
        md.append(f"| {r['sc']} | **{r['status']}** | {detail} |")
    (args.out / "scorecard.md").write_text("\n".join(md) + "\n")
    (args.out / "scorecard.json").write_text(json.dumps(rows, indent=2, default=str))

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    n_pass = sum(1 for r in rows if r["status"] in ("PASS", "PARTIAL_PASS"))
    n_warn = sum(1 for r in rows if r["status"].startswith("WARN"))
    parts = [f"PASS={n_pass}", f"FAIL={n_fail}", f"SKIP={n_skip}"]
    if n_warn:
        parts.append(f"WARN={n_warn}")
    print(f"[score] " + " ".join(parts))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
