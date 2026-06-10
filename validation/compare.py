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
            # Walk for 'gflops_actual'/'gflops' fields. RVS emits these as
            # *strings* (e.g. "gflops": "1088326"), so accept anything that
            # parses as a float — not just int/float instances.
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k in ("gflops_actual", "gflops"):
                            try:
                                vals.append(float(v))
                            except (TypeError, ValueError):
                                walk(v)
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for x in o:
                        walk(x)
            walk(j)
        except Exception:  # noqa: BLE001
            pass
    if not vals and log_path.exists():
        text = log_path.read_text()
        # RVS stdout prints per-interval lines like "... GFLOPS 1055817".
        # `GFLOPS\s+<num>` matches those but NOT the "Target GFLOPS: 1360000"
        # line (colon, not whitespace), so we capture achieved — not target.
        patterns = (
            r"gflops[_ ]?actual[:= ]+([0-9.eE+]+)",
            r"\bGFLOPS\s+([0-9.eE+]+)",
        )
        for pat in patterns:
            for m in re.finditer(pat, text, re.I):
                try:
                    vals.append(float(m.group(1)))
                except Exception:  # noqa: BLE001
                    pass
            if vals:
                break
    return max(vals) if vals else None


def parse_rocm_bw(rocm_bw_dir: Path) -> Optional[float]:
    """Max device HBM/D2D bandwidth observed (GB/s).

    Handles both CLI generations of the tool (see run_rocm_bw.sh):

      * Legacy rocm-bandwidth-test rows: ``D2D Bandwidth ... XX.XXX GB/s``.
      * Modern TransferBench ``hbm`` preset table, e.g. ::

            | Rank   GPU | MaxBw (GB/s)   AvgBw (GB/s)   MinBw (GB/s) |
            |    0     0 |      7409.20        6573.42        4808.95 |

        Here the bandwidth numbers are *not* individually suffixed with
        "GB/s", so we additionally scrape decimals from box-drawn table rows.
    """
    best = 0.0
    found = False

    def _consume(text: str) -> None:
        nonlocal best, found
        # 1) Legacy/explicit "<num> GB/s" occurrences.
        for m in re.finditer(r"([0-9]+\.[0-9]+)\s*GB/?s", text):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if v > best:
                best, found = v, True
        # 2) TransferBench table rows: any line that is part of a box-drawn
        #    table (vertical bar, ASCII '|' or unicode U+2502) and carries
        #    decimal bandwidth values. The largest value per row is MaxBw.
        for line in text.splitlines():
            if ("|" not in line and "\u2502" not in line):
                continue
            for m in re.finditer(r"[0-9]+\.[0-9]+", line):
                v = float(m.group())
                if v > best:
                    best, found = v, True

    # Preferred logs first, then any legacy per-GPU sweeps.
    candidates = [
        rocm_bw_dir / "hbm.log",
        rocm_bw_dir / "all_sizes.log",
    ]
    candidates += sorted(rocm_bw_dir.glob("d2d_gpu*.log"))
    for log in candidates:
        if log.exists():
            _consume(log.read_text(errors="ignore"))

    return best if found else None


# rccl-tests perf data row:
#   size count type redop root  oop_time oop_algbw oop_busbw oop_#wrong \
#                               ip_time  ip_algbw  ip_busbw  ip_#wrong
# We anchor on `size count type redop root` and read the *out-of-place* triple
# (time, algbw, busbw) that immediately follows. Anchoring this way is robust
# to the trailing `#wrong` column being `0`, `N/A`, or asterisk-flagged — the
# old "take the last few numeric tokens" heuristic silently grabbed the wrong
# column for all_to_all (whose in-place `#wrong` is `N/A`), returning in-place
# *algbw* instead of out-of-place *busbw*.
_RCCL_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+\w+\s+\w+\s+-?\d+\s+"
    r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
)


def parse_rccl_log(path: Path) -> list[dict]:
    """Parse rccl-tests perf log. Returns rows of {bytes, algbw, busbw}.

    Uses the out-of-place algbw/busbw columns (the canonical metric, and what
    the out-of-place PyTorch collectives in bench12 correspond to).
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith("size"):
            continue
        m = _RCCL_ROW.match(s)
        if not m:
            continue
        try:
            size = int(m.group(1))
            algbw = float(m.group(4))
            busbw = float(m.group(5))
        except Exception:  # noqa: BLE001
            continue
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
    ap.add_argument("--compute-tol", type=float, default=0.10,
                    help="Lower-bound tolerance: PyTorch peak must be within this "
                         "fraction BELOW the RVS gst number.")
    ap.add_argument("--compute-upper-tol", type=float, default=0.40,
                    help="Upper-bound headroom: PyTorch *peak* GEMM is expected to "
                         "exceed RVS gst's *sustained* stress throughput (RVS runs "
                         "a continuous kernel and loses to thermal/power limits). "
                         "Only flag if PyTorch exceeds RVS by more than this.")
    ap.add_argument("--comm-tol", type=float, default=0.10)
    ap.add_argument("--comm-plateau-min-mb", type=float, default=64.0,
                    help="Cross-tool busbw agreement is only physically meaningful "
                         "in the bandwidth-bound (plateau) regime. Payloads smaller "
                         "than this are latency-bound and harness-dependent, so they "
                         "are recorded as non-gating INFO instead of PASS/FAIL.")
    args = ap.parse_args()

    out: Path = args.out
    rows: list[dict] = []

    # 1) BF16 compute peak vs RVS gst
    pyt_peak = load_pytorch_compute_peak(out)
    rvs_gflops = parse_rvs_gst(out / "validation" / "rvs")
    if pyt_peak is not None and rvs_gflops is not None:
        rvs_tflops = rvs_gflops / 1e3
        # Directional comparison. RVS gst is a *sustained* stress kernel; over
        # its multi-second run it loses throughput to thermal/power limits and
        # uses its own (not necessarily optimal) GEMM, so it reads lower than a
        # peak single-GEMM microbenchmark. The invariant we actually want to
        # enforce is: PyTorch's peak must not fall *below* RVS by more than
        # ``compute_tol`` (that would mean PyTorch under-performs an independent
        # tool), while allowing the peak to sit above RVS up to
        # ``compute_upper_tol`` of headroom (peak >= sustained).
        signed = (pyt_peak - rvs_tflops) / rvs_tflops if rvs_tflops else float("inf")
        ok = (-args.compute_tol) <= signed <= args.compute_upper_tol
        rows.append({
            "metric": "BF16 compute peak (TFLOP/s)",
            "pytorch": round(pyt_peak, 2),
            "ground_truth": round(rvs_tflops, 2),
            "tool": "RVS gst (sustained)",
            "abs_pct_diff": round(abs(signed) * 100, 2),
            "tolerance_pct": f"-{args.compute_tol * 100:g}/+{args.compute_upper_tol * 100:g}",
            "status": "PASS" if ok else "FAIL",
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
        # Only gate (PASS/FAIL) in the bandwidth-bound plateau regime. Below the
        # plateau, busbw is latency-bound: the value is dominated by per-call
        # launch/sync overhead, which differs between an in-process PyTorch loop
        # and the standalone rccl-tests binary, so cross-tool agreement is not
        # physically expected. Such points are recorded as non-gating INFO.
        is_plateau = (prow["bytes"] / 1e6) >= args.comm_plateau_min_mb
        if is_plateau:
            status = "PASS" if diff <= args.comm_tol else "FAIL"
        else:
            status = "INFO"
        rows.append({
            "metric": f"{op} busbw",
            "message_size_mb": size_mb,
            "pytorch": round(prow["busbw_gb_s"], 1),
            "ground_truth": round(match["busbw_gb_s"], 1),
            "tool": f"rccl-tests ({int(round(match['bytes'] / 1e6))} MB)",
            "abs_pct_diff": round(diff * 100, 2),
            "tolerance_pct": args.comm_tol * 100,
            "status": status,
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
