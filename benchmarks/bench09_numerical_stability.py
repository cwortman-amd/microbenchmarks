"""Family 9 — Numerical-stability sweep across reduced-precision GEMMs.

Where ``bench01.correctness_check`` is a binary gate ("did BF16 matmul stay
inside the analytic bound at K=256, K=1024?"), this benchmark is the
*detailed* picture: per-dtype, per-K, the full distribution of GEMM error
versus an FP32 reference. The output feeds the report's roofline-adjacent
narrative — "given that BF16 saves 2× memory, how much accuracy did we
trade for it?" — with concrete numbers rather than spec-sheet claims.

Methodology
-----------
For each (dtype, K):

  1. Sample ``A, B ∈ R^(K×K)`` from ``N(0, 1/sqrt(K))``. The ``1/sqrt(K)``
     scaling keeps ``A @ B`` at unit-variance regardless of K, which is
     the regime LLM weights actually live in and means ``rel_err`` is a
     fair number to compare across sizes.
  2. Compute ``Z_ref = A @ B`` in FP32 — that's our ground truth.
  3. Cast inputs to the test dtype, perform the matmul, cast result back
     to FP32. For FP8 this is necessarily ``cast -> bf16 matmul -> cast``
     on backends without a native FP8 GEMM path (i.e. CPU + most GPUs in
     PyTorch today); we record this fact in the row.
  4. Element-wise error: ``abs_err``, ``rel_err = abs_err / max(|Z_ref|, eps)``.
     Histogram log-binned (10 bins / decade) across the full element set.
  5. Compare ``max_rel`` against the analytic bound
     ``5 · √K · 2⁻mantissa_bits`` (5× safety factor over the standard
     IEEE GEMM accumulation bound, same convention as bench01).

Outputs
-------
``results/<benchmark>/09_numerical_stability/stability.json`` — full row
data including per-bin histograms; ``summary.json`` — compact roll-up
that ``score_benchmark`` and ``report`` can consume directly. CSV export
of the row table for downstream analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from benchmarks.common.io import write_csv, write_json


# Mantissa bit counts drive the analytic error bound. fp8 numbers come
# from IEEE 754 binary8 conventions: e4m3 has 3 mantissa bits, e5m2 has
# 2. (Note `e4m3fn` swaps the inf encoding for an extra finite value; the
# mantissa width is still 3.)
DTYPE_MANTISSA_BITS: Dict[str, int] = {
    "fp32":     23,
    "fp16":     10,
    "bf16":      7,
    "fp8_e4m3":  3,
    "fp8_e5m2":  2,
}


def _supported_test_dtypes() -> List[Tuple[str, torch.dtype]]:
    """Every reduced-precision dtype this PyTorch build advertises.

    fp32 is the reference and is *not* in the test set. We deliberately
    don't try-and-catch for whether the backend can actually matmul each
    dtype here; that check happens inline so the artifact records the
    backend's reason for skipping (matches `bench01.dtype_sweep`).
    """
    candidates = [
        ("fp16",     getattr(torch, "float16",       None)),
        ("bf16",     getattr(torch, "bfloat16",      None)),
        ("fp8_e4m3", getattr(torch, "float8_e4m3fn", None)),
        ("fp8_e5m2", getattr(torch, "float8_e5m2",   None)),
    ]
    return [(lbl, dt) for lbl, dt in candidates if dt is not None]


def _safe_matmul(a_lp: torch.Tensor, b_lp: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Matmul in `dtype`, casting up to bf16 if the backend lacks a native path.

    Returns ``(result_fp32, note)``. ``note`` is empty when the matmul
    ran natively in the requested dtype and a human-readable string
    when an upcast was needed (so the reported error reflects the
    upcasted compute, not a real low-precision tensor-core path).
    """
    try:
        z = (a_lp @ b_lp).to(torch.float32)
        return z, ""
    except (RuntimeError, NotImplementedError, TypeError):
        a_bf16 = a_lp.to(torch.bfloat16)
        b_bf16 = b_lp.to(torch.bfloat16)
        z = (a_bf16 @ b_bf16).to(torch.float32)
        return z, "upcast to bf16 for matmul (no native GEMM on this backend)"


def _log_histogram(values: torch.Tensor, *,
                   floor: float = 1e-12,
                   bins_per_decade: int = 10,
                   n_decades: int = 12) -> Dict[str, list]:
    """Log-binned histogram of ``|values|``.

    Bins span ``[floor, floor * 10^n_decades]`` with `bins_per_decade`
    bins per decade. We use log bins because GEMM relative error spans
    many decades (1e-8 for FP32-vs-FP32 sanity, 1e-2 for FP8) and a
    linear histogram would collapse interesting structure into one bin.
    """
    edges_log = [math.log10(floor) + i / bins_per_decade
                 for i in range(bins_per_decade * n_decades + 1)]
    edges = [10 ** e for e in edges_log]
    abs_values = values.flatten().abs().to(torch.float32).cpu()
    abs_values = abs_values.clamp_min(floor / 10)
    counts = torch.histogram(
        abs_values,
        bins=torch.tensor(edges, dtype=torch.float32),
    ).hist
    return {
        "edges":  [float(e) for e in edges],
        "counts": [int(c) for c in counts.tolist()],
    }


def _percentiles(values: torch.Tensor) -> Dict[str, float]:
    flat = values.flatten().abs()
    n = flat.numel()
    if n == 0:
        return {"p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "p999": float("nan"),
                "mean": float("nan"), "max": float("nan")}
    sorted_v = flat.sort().values
    return {
        "p50":  float(sorted_v[int(0.50 * (n - 1))].item()),
        "p90":  float(sorted_v[int(0.90 * (n - 1))].item()),
        "p99":  float(sorted_v[int(0.99 * (n - 1))].item()),
        "p999": float(sorted_v[min(int(0.999 * (n - 1)), n - 1)].item()),
        "mean": float(flat.mean().item()),
        "max":  float(sorted_v[-1].item()),
    }


def stability_row(device: torch.device,
                  label: str,
                  dtype: torch.dtype,
                  K: int,
                  seed: int) -> dict:
    """Compute one (dtype, K) stability row vs FP32 reference."""
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(K)
    a = torch.randn(K, K, dtype=torch.float32, device=device) * scale
    b = torch.randn(K, K, dtype=torch.float32, device=device) * scale
    z_ref = a @ b

    a_lp = a.to(dtype)
    b_lp = b.to(dtype)
    z_lp, note = _safe_matmul(a_lp, b_lp)

    abs_err = (z_lp - z_ref).abs()
    # Two relative-error definitions, both useful but for different audiences:
    #
    #   * `rel_err_pointwise`  — per-element abs_err / |Z_ref|. Great for
    #     histogramming the *shape* of the error distribution, but the
    #     `max` is dominated by elements where Z_ref ≈ 0 and is therefore
    #     useless as a gate (a single near-zero entry blows it up).
    #
    #   * `rel_err_scaled`     — abs_err / max(|Z_ref|). This matches the
    #     standard GEMM accuracy convention (and bench01's gate) — it
    #     compares each element's absolute error to the matrix's overall
    #     scale, which is what tolerances like `assert_close(rtol, atol)`
    #     and the analytic √K · 2⁻mant bound are written against.
    ref_max = z_ref.abs().max().clamp_min(1e-30)
    rel_err_scaled = abs_err / ref_max
    ref_pointwise = z_ref.abs().clamp_min(1e-12)
    rel_err_pointwise = abs_err / ref_pointwise

    abs_stats = _percentiles(abs_err)
    rel_stats = _percentiles(rel_err_scaled)
    rel_pointwise_stats = _percentiles(rel_err_pointwise)

    mantissa_bits = DTYPE_MANTISSA_BITS.get(label, 0)
    bound = 5.0 * math.sqrt(K) * (2.0 ** -mantissa_bits) if mantissa_bits else float("inf")
    passed = rel_stats["max"] <= bound and not math.isnan(rel_stats["max"])

    abs_hist = _log_histogram(abs_err, floor=1e-10, bins_per_decade=4, n_decades=10)
    rel_hist = _log_histogram(rel_err_pointwise, floor=1e-10, bins_per_decade=4, n_decades=10)

    del (a, b, z_ref, a_lp, b_lp, z_lp,
         abs_err, rel_err_scaled, rel_err_pointwise,
         ref_max, ref_pointwise)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "dtype":              label,
        "K":                  K,
        "M":                  K,
        "N":                  K,
        "mantissa_bits":      mantissa_bits,
        "abs_err":            abs_stats,
        "rel_err":            rel_stats,            # matrix-scaled; gate uses this
        "rel_err_pointwise":  rel_pointwise_stats,  # per-element; for histograms
        "rel_err_bound":      bound,
        "passed":             bool(passed),
        "note":               note,
        "abs_err_hist":       abs_hist,
        "rel_err_hist":       rel_hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=None,
                    help="Inner dimensions to sweep. Default: 256/512/1024 "
                         "on CPU, 256/512/1024/2048/4096 on GPU.")
    args = ap.parse_args()

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")
    if args.ks:
        ks = list(args.ks)
    elif has_gpu:
        ks = [256, 512, 1024, 2048, 4096]
    else:
        # CPU: 4096^2 fp32 reference matmul is ~1 minute; cap at 1024 so
        # the bench finishes inside ~30s on a laptop without sacrificing
        # the K-scaling visibility (error scales as √K, so three points
        # already pin the slope down).
        ks = [256, 512, 1024]

    out_dir = Path(args.out) / "09_numerical_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for k in ks:
        for label, dtype in _supported_test_dtypes():
            try:
                row = stability_row(device, label, dtype, k, seed=args.seed)
            except Exception as e:  # noqa: BLE001 — capture-and-continue
                row = {
                    "dtype":         label,
                    "K":             k,
                    "M":             k,
                    "N":             k,
                    "mantissa_bits": DTYPE_MANTISSA_BITS.get(label, 0),
                    "passed":        False,
                    "error":         f"{type(e).__name__}: {e}",
                }
            rows.append(row)
            if "error" in row:
                print(f"[09] {label:9s} K={k:5d}  ERROR: {row['error']}")
            else:
                print(f"[09] {label:9s} K={k:5d}  "
                      f"max_rel={row['rel_err']['max']:.3e}  "
                      f"p99_rel={row['rel_err']['p99']:.3e}  "
                      f"bound={row['rel_err_bound']:.3e}  "
                      f"[{'ok' if row['passed'] else 'FAIL'}]")

    write_json(out_dir / "stability.json", {
        "device":       str(device),
        "device_type":  device.type,
        "rows":         rows,
        "ks":           ks,
        "seed":         args.seed,
    })

    # Compact CSV for spreadsheet-style review.
    csv_rows: List[dict] = []
    for r in rows:
        if "error" in r:
            csv_rows.append({
                "dtype": r["dtype"], "K": r["K"],
                "max_rel_err": "",
                "p99_rel_err": "",
                "bound":       "",
                "passed":      False,
                "note":        r.get("error", ""),
            })
        else:
            csv_rows.append({
                "dtype":       r["dtype"],
                "K":           r["K"],
                "max_rel_err": r["rel_err"]["max"],
                "p99_rel_err": r["rel_err"]["p99"],
                "p50_rel_err": r["rel_err"]["p50"],
                "bound":       r["rel_err_bound"],
                "passed":      r["passed"],
                "note":        r.get("note", ""),
            })
    write_csv(out_dir / "stability.csv", csv_rows)

    summary = {
        "device":      str(device),
        "device_type": device.type,
        "all_passed":  all(r.get("passed") for r in rows),
        "per_dtype": {
            label: {
                "max_rel_err_overall": max(
                    (r["rel_err"]["max"]
                     for r in rows
                     if r["dtype"] == label and "rel_err" in r),
                    default=None,
                ),
                "all_passed_overall": all(
                    r.get("passed")
                    for r in rows if r["dtype"] == label
                ),
            }
            for label, _ in _supported_test_dtypes()
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[09] all_passed={summary['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
