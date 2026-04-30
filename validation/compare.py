"""Cross-validation: PyTorch microbenchmarks vs RVS, rocm-bandwidth-test, rccl-tests.

For each PyTorch metric that has a corresponding ground-truth tool,
compute the ratio and PASS/FAIL it against a per-metric tolerance:

  | PyTorch metric                       | Ground truth         | Tolerance |
  |--------------------------------------|----------------------|-----------|
  | 01 BF16 peak TFLOP/s                  | RVS gst gflops_actual | ±10%      |
  | 02 HBM plateau GB/s                   | rocm-bandwidth-test D2D max | ±15% |
  | 06 all_gather busbw (per payload)     | rccl-tests all_gather_perf  | ±10% |
  | 06 reduce_scatter busbw (per payload) | rccl-tests reduce_scatter_perf | ±10% |
  | 06 all_reduce busbw (per payload)     | rccl-tests all_reduce_perf | ±10% |

Outputs:
  <out>/validation.md   — human-readable PASS/FAIL table
  <out>/validation.json — machine-readable, used by run_benchmark.sh exit code
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Parsers for external tool outputs.
# ---------------------------------------------------------------------------


def parse_rvs_gst(rvs_dir: Path) -> Optional[float]:
    """Return max gflops_actual reported by RVS gst across all GPUs.

    RVS JSON output shape varies by version; we also fall back to the stdout
    log, scanning for `gflops_actual: <num>` patterns.
    """
    json_path = rvs_dir / "gst_bf16.json"
    log_path = rvs_dir / "gst_bf16.stdout.log"
    vals: list[float] = []
    if json_path.exists():
        try:
            j = json.loads(json_path.read_text())
            # Walk the structure for any 'gflops_actual' or 'gflops_target' fields.
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k in ("gflops_actual", "gflops") and isinstance(v, (int, float)):
                            vals.append(float(v))
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for x in o:
                        walk(x)
            walk(j)
        except Exception:  # noqa: BLE001
            pass
    if not vals and log_path.exists():
        for m in re.finditer(r"gflops[_ ]?actual[:= ]+([0-9.eE+]+)", log_path.read_text(), re.I):
            try:
                vals.append(float(m.group(1)))
            except Exception:  # noqa: BLE001
                pass
    return max(vals) if vals else None


def parse_rocm_bw(rocm_bw_dir: Path) -> Optional[float]:
    """Max device-to-device unidirectional bandwidth observed (GB/s)."""
    best = 0.0
    found = False
    for log in sorted(rocm_bw_dir.glob("d2d_gpu*.log")):
        text = log.read_text(errors="ignore")
        # rocm-bandwidth-test prints lines like:
        #   "D2D Bandwidth ... XX.XXX GB/s" or table rows with bandwidth columns.
        for m in re.finditer(r"([0-9]+\.[0-9]+)\s*GB/s", text):
            try:
                v = float(m.group(1))
                if v > best:
                    best = v
                    found = True
            except Exception:  # noqa: BLE001
                pass
    if not found:
        # try all_sizes.log
        all_sizes = rocm_bw_dir / "all_sizes.log"
        if all_sizes.exists():
            for m in re.finditer(r"([0-9]+\.[0-9]+)\s*GB/s", all_sizes.read_text(errors="ignore")):
                v = float(m.group(1))
                if v > best:
                    best = v
                    found = True
    return best if found else None


_RCCL_LINE = re.compile(
    r"^\s*([0-9]+)\s+([0-9]+)\s+\S+\s+\S+\s+\S+\s+"
    r"([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)"
)


def parse_rccl_log(path: Path) -> list[dict]:
    """Parse rccl-tests perf log. Returns rows of {bytes, time_us, algbw, busbw}.

    rccl-tests perf log format (out-of-place section):
        size  count  type  ... time  algbw  busbw  #wrong
    Columns vary slightly between versions; we anchor on `size` (bytes) and the
    last two numeric columns (algbw, busbw in GB/s).
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("size"):
            continue
        parts = s.split()
        # Need at least: size count type op root time algbw busbw
        if len(parts) < 8:
            continue
        try:
            size = int(parts[0])
        except Exception:  # noqa: BLE001
            continue
        # try to find two trailing numerics in the last 4 fields = algbw, busbw
        numerics: list[float] = []
        for tok in parts[-6:]:
            try:
                numerics.append(float(tok))
            except Exception:  # noqa: BLE001
                pass
        if len(numerics) < 3:
            continue
        algbw, busbw = numerics[-3], numerics[-2]
        rows.append({"bytes": size, "algbw_gb_s": algbw, "busbw_gb_s": busbw})
    return rows


# ---------------------------------------------------------------------------
# PyTorch metric loaders.
# ---------------------------------------------------------------------------


def load_pytorch_compute_peak(out: Path) -> Optional[float]:
    p = out / "01_bf16_compute" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["compute_roof_tflops"]


def load_pytorch_bw_roof(out: Path) -> Optional[float]:
    p = out / "02_hbm_bandwidth" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["bandwidth_roof_gb_s"]


def load_pytorch_comm(out: Path) -> list[dict]:
    p = out / "06_multigpu_comm" / "comm.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["rows"]


# ---------------------------------------------------------------------------

def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return abs(a - b) / b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="benchmark output dir, e.g. results/<id>/")
    ap.add_argument("--bw-tol", type=float, default=0.15)
    ap.add_argument("--compute-tol", type=float, default=0.10)
    ap.add_argument("--comm-tol", type=float, default=0.10)
    args = ap.parse_args()

    out: Path = args.out
    rows: list[dict] = []

    # 1) BF16 compute peak vs RVS gst
    pyt_peak = load_pytorch_compute_peak(out)
    rvs_gflops = parse_rvs_gst(out / "validation" / "rvs")
    if pyt_peak is not None and rvs_gflops is not None:
        rvs_tflops = rvs_gflops / 1e3
        diff = _pct_diff(pyt_peak, rvs_tflops)
        rows.append({
            "metric": "BF16 compute peak (TFLOP/s)",
            "pytorch": round(pyt_peak, 2),
            "ground_truth": round(rvs_tflops, 2),
            "tool": "RVS gst",
            "abs_pct_diff": round(diff * 100, 2),
            "tolerance_pct": args.compute_tol * 100,
            "status": "PASS" if diff <= args.compute_tol else "FAIL",
        })
    else:
        rows.append({
            "metric": "BF16 compute peak (TFLOP/s)",
            "pytorch": pyt_peak, "ground_truth": rvs_gflops,
            "tool": "RVS gst", "abs_pct_diff": None,
            "tolerance_pct": args.compute_tol * 100,
            "status": "SKIP" if (pyt_peak is None or rvs_gflops is None) else "PASS",
        })

    # 2) HBM bandwidth plateau vs rocm-bandwidth-test D2D max
    pyt_bw = load_pytorch_bw_roof(out)
    rbw = parse_rocm_bw(out / "validation" / "rocm_bw")
    if pyt_bw is not None and rbw is not None:
        diff = _pct_diff(pyt_bw, rbw)
        rows.append({
            "metric": "Memory bandwidth roof (GB/s)",
            "pytorch": round(pyt_bw, 1),
            "ground_truth": round(rbw, 1),
            "tool": "rocm-bandwidth-test (D2D max)",
            "abs_pct_diff": round(diff * 100, 2),
            "tolerance_pct": args.bw_tol * 100,
            "status": "PASS" if diff <= args.bw_tol else "FAIL",
        })
    else:
        rows.append({
            "metric": "Memory bandwidth roof (GB/s)",
            "pytorch": pyt_bw, "ground_truth": rbw,
            "tool": "rocm-bandwidth-test (D2D max)",
            "abs_pct_diff": None,
            "tolerance_pct": args.bw_tol * 100,
            "status": "SKIP" if (pyt_bw is None or rbw is None) else "PASS",
        })

    # 3) Per-payload comm ops vs rccl-tests
    pyt_comm = load_pytorch_comm(out)
    rccl_dir = out / "validation" / "rccl"
    rccl_logs = {
        "all_gather":     parse_rccl_log(rccl_dir / "all_gather.log"),
        "reduce_scatter": parse_rccl_log(rccl_dir / "reduce_scatter.log"),
        "all_reduce":     parse_rccl_log(rccl_dir / "all_reduce.log"),
        "all_to_all":     parse_rccl_log(rccl_dir / "all_to_all.log"),
    }
    # Keep track of which rccl-tests sizes were matched
    matched_rccl = {"all_gather": set(), "reduce_scatter": set(), "all_reduce": set(), "all_to_all": set()}

    for prow in pyt_comm:
        op = prow["op"]
        rccl_rows = rccl_logs.get(op, [])
        size_mb = int(round(prow["bytes"] / 1e6))
        
        # Match by closest payload size
        if not rccl_rows:
            rows.append({"metric": f"{op} busbw",
                         "message_size_mb": size_mb,
                         "pytorch": prow["busbw_gb_s"], "ground_truth": None,
                         "tool": "rccl-tests", "abs_pct_diff": None,
                         "tolerance_pct": args.comm_tol * 100, "status": "SKIP"})
            continue
        # find closest size
        match = min(rccl_rows, key=lambda r: abs(r["bytes"] - prow["bytes"]))
        matched_rccl[op].add(match["bytes"])
        diff = _pct_diff(prow["busbw_gb_s"], match["busbw_gb_s"])
        rows.append({
            "metric": f"{op} busbw",
            "message_size_mb": size_mb,
            "pytorch": round(prow["busbw_gb_s"], 1),
            "ground_truth": round(match["busbw_gb_s"], 1),
            "tool": f"rccl-tests ({int(round(match['bytes'] / 1e6))} MB)",
            "abs_pct_diff": round(diff * 100, 2),
            "tolerance_pct": args.comm_tol * 100,
            "status": "PASS" if diff <= args.comm_tol else "FAIL",
        })

    # Add unmatched rccl-tests rows for the smooth baseline curve in the plot
    for op, rccl_rows in rccl_logs.items():
        for r in rccl_rows:
            if r["bytes"] not in matched_rccl.get(op, set()):
                size_mb = int(round(r["bytes"] / 1e6))
                rows.append({
                    "metric": f"{op} busbw",
                    "message_size_mb": size_mb,
                    "pytorch": None,
                    "ground_truth": round(r["busbw_gb_s"], 1),
                    "tool": "rccl-tests",
                    "abs_pct_diff": None,
                    "tolerance_pct": args.comm_tol * 100,
                    "status": "PLOT_ONLY"
                })

    # Write artifacts.
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(rows, indent=2, default=str))
    md = ["# Cross-validation report\n", "", "PyTorch microbenchmarks vs ground-truth tools.\n", ""]
    if rows:
        keys = list(rows[0].keys())
        md.append("| " + " | ".join(keys) + " |")
        md.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    (out / "validation.md").write_text("\n".join(md) + "\n")

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    print(f"[validate] PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
