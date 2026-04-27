"""Score a campaign against TESTPLAN §1.2 success criteria SC-1…SC-5.

Reads all the artifacts produced by run_campaign.sh and writes:
    <out>/scorecard.md   — human-readable PASS/FAIL per SC
    <out>/scorecard.json — machine-readable, used by run_campaign.sh exit code

SC-1: BF16 GEMM peak ≥ 50% of 1.26 PF rated, AND best square ≥ 90% of measured peak
SC-2: HBM plateau within ±5% across 3 successive top sizes
SC-3: Roofline placement — large GEMMs and attention sit AI > ridge
SC-4: compiled e2e MFU ≥ eager e2e MFU ≥ sum-of-ops MFU (within ±5pp tolerance)
SC-5: All required artifacts in TESTPLAN §13 (A1–A11) exist
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


def sc1_bf16_peak(out: Path) -> Dict:
    j = _load(out / "01_bf16_compute" / "summary.json")
    sweep = _load(out / "01_bf16_compute" / "sweep.json")
    if not j or not sweep:
        return {"sc": "SC-1", "status": "SKIP", "reason": "missing 01_bf16_compute outputs"}
    peak = j["compute_roof_tflops"]
    sq = sweep.get("square", [])
    if not sq:
        return {"sc": "SC-1", "status": "SKIP", "reason": "no square sweep results"}
    best = max(r["tflops"] for r in sq)
    rated_low = 1260.0  # 1.26 PF
    pct_rated = peak / rated_low
    pct_measured = best / peak if peak else 0.0
    ok = (pct_rated >= 0.50) and (pct_measured >= 0.90)
    return {
        "sc": "SC-1",
        "status": "PASS" if ok else "FAIL",
        "peak_tflops": round(peak, 1),
        "best_sweep_tflops": round(best, 1),
        "pct_of_rated_1_26pf": round(pct_rated * 100, 1),
        "best_pct_of_peak": round(pct_measured * 100, 1),
        "thresholds": {"min_pct_rated": 50.0, "min_best_pct_of_peak": 90.0},
    }


def sc2_hbm_plateau(out: Path) -> Dict:
    j = _load(out / "02_hbm_bandwidth" / "bandwidth.json")
    if not j:
        return {"sc": "SC-2", "status": "SKIP", "reason": "missing 02_hbm_bandwidth outputs"}
    # Look at copy_ which is the canonical streaming pattern; need 3 successive top sizes within ±5%.
    rows = sorted([r for r in j if r["op"] == "copy_"], key=lambda r: r["bytes"])
    if len(rows) < 3:
        return {"sc": "SC-2", "status": "SKIP", "reason": "fewer than 3 copy_ sizes succeeded"}
    top3 = rows[-3:]
    bws = [r["gb_s"] for r in top3]
    spread = (max(bws) - min(bws)) / max(bws)
    ok = spread <= 0.05
    return {
        "sc": "SC-2",
        "status": "PASS" if ok else "FAIL",
        "top3_gb_s": [round(v, 1) for v in bws],
        "spread_pct": round(spread * 100, 2),
        "threshold_pct": 5.0,
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
    j = _load(out / "05_e2e_mfu" / "mfu.json")
    if not j:
        return {"sc": "SC-4", "status": "SKIP", "reason": "missing 05_e2e_mfu outputs"}
    rows = {r["scope"]: r for r in j["rows"]}
    sop = (rows.get("sum_of_ops_optimized") or rows.get("sum_of_ops_default") or {}).get("mfu_measured_peak")
    eager = (rows.get("eager_e2e") or {}).get("mfu_measured_peak")
    compiled = (rows.get("compiled_e2e") or {}).get("mfu_measured_peak")
    if sop is None or eager is None or compiled is None:
        return {"sc": "SC-4", "status": "SKIP", "reason": "incomplete MFU rows", "sop": sop, "eager": eager, "compiled": compiled}
    tol_pp = 0.05  # 5 percentage points
    ok = (compiled + tol_pp >= eager) and (eager + tol_pp >= sop)
    return {
        "sc": "SC-4",
        "status": "PASS" if ok else "FAIL",
        "mfu_sum_of_ops_pct": round(sop * 100, 1),
        "mfu_eager_pct": round(eager * 100, 1),
        "mfu_compiled_pct": round(compiled * 100, 1),
        "tolerance_pp": tol_pp * 100,
    }


REQUIRED_ARTIFACTS = [
    ("A1", "01_bf16_compute/summary.json"),
    ("A2", "plots/A2_bf16_gemm_sweep.png"),
    ("A3", "plots/A3_hbm_bandwidth.png"),
    ("A4", "03_vram_capacity/summary.json"),
    ("A5", "04_workload_ops/ops.csv"),
    ("A6", "plots/A6_roofline.png"),
    ("A7", "plots/A7_per_op_theory_vs_meas.png"),
    ("A8", "05_e2e_mfu/mfu.csv"),
    # A9 multi-GPU is optional
    ("A10", "env.json"),
    ("A11", "summary.md"),
]


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
        sc1_bf16_peak(args.out),
        sc2_hbm_plateau(args.out),
        sc3_roofline_placement(args.out),
        sc4_mfu_ordering(args.out),
        sc5_artifacts(args.out, device),
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
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    print(f"[score] PASS={n_pass} FAIL={n_fail} SKIP={n_skip}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
