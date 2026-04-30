"""Family 4 — escher_14b_480p per-op decomposition (TESTPLAN §8).

Two outputs per op:

  1. Analytic FLOP and HBM-byte counts (from common.flop_accounting)
  2. Measured runtime under
       a) eager-default path (no AITER, default SDPA)
       b) optimized path (AITER if importable; else SDPA flash; else best available)

Combined into a single per-op CSV/JSON with the columns that §8.2 prescribes.
The analytic totals are also compared against `reference_totals_per_block` in
the config so calibration drift is caught early.
"""

from __future__ import annotations

import argparse
import json
import os

# Ensure Flash Attention 2 on ROCm uses the Triton backend rather than failing 
# when looking for the CUDA extension.
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from benchmarks.common.flop_accounting import (
    OpAcct,
    WorkloadConfig,
    per_block_ops,
    totals,
)
from benchmarks.common.io import write_csv, write_json, write_md_table
from benchmarks.common.timing import time_op


# ---------------------------------------------------------------------------
# Optimized attention backend selection.
# ---------------------------------------------------------------------------

def _try_import_aiter():
    try:
        import aiter  # noqa: F401
        return aiter
    except Exception:  # noqa: BLE001
        return None


def _try_import_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return flash_attn
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def _sdpa_kernel(flash: bool = True, mem: bool = False, math: bool = False):
    """Best-effort SDPA backend lock. No-op on older torch."""
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backends = []
        if flash: backends.append(SDPBackend.FLASH_ATTENTION)
        if mem: backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if math: backends.append(SDPBackend.MATH)
        try:
            with sdpa_kernel(backends):
                yield
            return
        except Exception:
            pass

    backends = getattr(torch.backends.cuda, "sdp_kernel", None)
    if backends is None:
        yield
        return
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=flash, enable_mem_efficient=mem, enable_math=math
        ):
            yield
    except Exception:  # noqa: BLE001
        yield


def attention_default(q, k, v) -> torch.Tensor:
    """Default torch SDPA, math+mem_efficient backends only (NO flash)."""
    with _sdpa_kernel(flash=False, mem=True, math=True):
        return F.scaled_dot_product_attention(q, k, v)


def attention_optimized(q, k, v) -> torch.Tensor:
    """Best available: AITER -> flash_attn -> SDPA flash."""
    aiter = _try_import_aiter()
    if aiter is not None:
        # AITER's interface evolves; try a few common entry points.
        for name in ("flash_attn_varlen_func", "flash_attn_func", "scaled_dot_product_attention"):
            fn = getattr(aiter, name, None)
            if fn is None:
                continue
            try:
                return fn(q, k, v)
            except Exception:  # noqa: BLE001
                continue
    fa = _try_import_flash_attn()
    if fa is not None:
        try:
            from flash_attn import flash_attn_func  # type: ignore
            return flash_attn_func(q, k, v)
        except Exception:  # noqa: BLE001
            pass
    with _sdpa_kernel(flash=True, mem=False, math=False):
        return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# Op runners — each returns (closure, allocator). Closures only run the op,
# allocations are out of the timed region.
# ---------------------------------------------------------------------------

def _gemm_runner(M: int, K: int, N: int, device) -> Callable:
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    def fn():
        torch.matmul(a, b, out=out)
    fn._refs = (a, b, out)  # type: ignore[attr-defined]
    return fn


def _attn_runner(B: int, H: int, S_q: int, S_kv: int, Dh: int, device, optimized: bool) -> Callable:
    q = torch.randn(B, H, S_q, Dh, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, S_kv, Dh, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, S_kv, Dh, dtype=torch.bfloat16, device=device)
    fn_attn = attention_optimized if optimized else attention_default
    def fn():
        _ = fn_attn(q, k, v)
    fn._refs = (q, k, v)  # type: ignore[attr-defined]
    return fn


def _elementwise_runner(n_elements: int, kind: str, device) -> Callable | None:
    if n_elements == 0:
        return None
    a = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
    if kind == "norm":
        b = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
        gamma = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
        beta = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
        # Use a coarse stand-in for layer/RMS norm bandwidth: two passes over a, plus affine
        def fn():
            mean = a.mean()
            x = (a - mean) * gamma + beta
            b.copy_(x)
        fn._refs = (a, b, gamma, beta)  # type: ignore[attr-defined]
        return fn
    if kind == "gelu":
        out = torch.empty_like(a)
        def fn():
            F.gelu(a, out=out)
        fn._refs = (a, out)  # type: ignore[attr-defined]
        return fn
    if kind == "residual":
        b = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(a)
        def fn():
            torch.add(a, b, out=out)
        fn._refs = (a, b, out)  # type: ignore[attr-defined]
        return fn
    raise ValueError(f"unknown elementwise kind: {kind}")


def _runner_for_op(op: OpAcct, cfg: WorkloadConfig, device, optimized: bool) -> Callable | None:
    n = op.op_name
    # Time embed & projections & QKV / O / FFN linears -> GEMM
    if n in ("time_proj", "time_embed") or n.endswith((".q", ".k", ".v", ".o")) \
       or n in ("ffn.linear1", "ffn.linear2"):
        # parse shapes from accounting
        # input_shape format "(M,K)x(K,N)"
        try:
            left, right = op.input_shape.split("x")
            M, K = [int(s) for s in left.strip("()").split(",")]
            _, N = [int(s) for s in right.strip("()").split(",")]
        except Exception:  # noqa: BLE001
            return None
        return _gemm_runner(M, K, N, device)
    if n.endswith(".flash"):
        if "self_attn" in n:
            return _attn_runner(cfg.batch, cfg.n_heads, cfg.seq_image, cfg.seq_image,
                                cfg.head_dim, device, optimized=optimized)
        if "cross_attn" in n:
            return _attn_runner(cfg.batch, cfg.n_heads, cfg.seq_image, cfg.seq_text,
                                cfg.head_dim, device, optimized=optimized)
    if n.startswith("norm_"):
        # take element count from accounting (divide by stream count)
        n_elem = max(op.bytes_hbm // (4 * 2), 1)  # 4 streams, 2 B/elt
        return _elementwise_runner(n_elem, "norm", device)
    if n.startswith("residual_"):
        n_elem = max(op.bytes_hbm // (3 * 2), 1)
        return _elementwise_runner(n_elem, "residual", device)
    if n == "ffn.gelu":
        n_elem = max(op.bytes_hbm // (2 * 2), 1)
        return _elementwise_runner(n_elem, "gelu", device)
    if n == "kv_cache_write":
        return None
    return None


# ---------------------------------------------------------------------------

def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cpu-budget-gflops", type=float, default=5.0,
                    help="On CPU, skip per-op timing for ops whose analytic "
                         "FLOPs exceed this budget. 0 disables CPU timing.")
    args = ap.parse_args()

    # bench04 has two collection paths:
    #   (a) the analytic op table (FLOPs / HBM bytes / AI per op)  — pure math,
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")
    cfg_json = json.loads(Path(args.config).read_text())
    cfg = WorkloadConfig.from_json(cfg_json)

    # Methodology check
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    t_cfg = m_cfg.get("timing", cfg_json.get("timing", {}))

    out_dir = Path(args.out) / "04_workload_ops"
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = per_block_ops(cfg)
    tot = totals(ops)

    # Calibration check vs reference totals.
    ref = cfg_json.get("reference_totals_per_block", {})
    cal_drift = {}
    if ref:
        cal_drift["gflops_drift_pct"] = (tot["total_gflops"] / ref["gflops"] - 1) * 100 if ref.get("gflops") else None
        cal_drift["mb_hbm_drift_pct"] = (tot["total_mb_hbm"] / ref["hbm_mb"] - 1) * 100 if ref.get("hbm_mb") else None

    # Read measured ceilings from earlier families if present.
    bf16_peak_tflops = None
    hbm_roof_gb_s = None
    try:
        bf16_peak_tflops = json.loads((Path(args.out) / "01_bf16_compute" / "summary.json").read_text())["compute_roof_tflops"]
    except Exception:  # noqa: BLE001
        pass
    try:
        hbm_roof_gb_s = json.loads((Path(args.out) / "02_hbm_bandwidth" / "summary.json").read_text())["bandwidth_roof_gb_s"]
    except Exception:  # noqa: BLE001
        pass

    # Keep iteration cadence aligned with the benchmark timing methodology:
    # warmup in [3, 20], timed iterations in [10, 30]. CPU remains tractable
    # because heavy ops are gated by --cpu-budget-gflops.
    def _is_set(opt): return any(o in a for a in sys.argv for o in (opt, opt.replace("--iters", "--iterations")))
    
    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 0)
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 0)

    # If both CLI and JSON are missing, use final hardcoded fallbacks
    if warmup_val == 0 and not _is_set("--warmup"): warmup_val = 5
    if iters_val == 0 and not _is_set("--iters"): iters_val = 20

    if has_gpu:
        warmup, iters = warmup_val, iters_val
    else:
        warmup = max(3, min(warmup_val, 20))
        iters = max(10, min(iters_val, 30))

    cpu_budget_flops = args.cpu_budget_gflops * 1e9 if not has_gpu else float("inf")

    rows: List[Dict] = []
    for op in ops:
        row = op.as_row()

        # On CPU, gate measurement by analytic FLOPs so the bench stays cheap.
        skip_measure = (not has_gpu) and (
            op.flops > cpu_budget_flops or args.cpu_budget_gflops <= 0
        )

        if skip_measure:
            row["t_ms_default"] = float("nan")
            row["t_ms_optimized"] = float("nan")
        else:
            for label, optimized in (("default", False), ("optimized", True)):
                fn = _runner_for_op(op, cfg, device, optimized=optimized)
                if fn is None:
                    row[f"t_ms_{label}"] = float("nan")
                    continue
                try:
                    r = time_op(f"{op.op_name}_{label}", fn, warmup=warmup, iters=iters)
                    row[f"t_ms_{label}"] = r.median_ms
                    row[f"t_ms_{label}_p10"] = r.p10_ms
                    row[f"t_ms_{label}_p90"] = r.p90_ms
                except RuntimeError as e:
                    if not _is_oom(e):
                        raise
                    row[f"t_ms_{label}"] = float("nan")
                finally:
                    if hasattr(fn, "_refs"):
                        fn._refs = None  # type: ignore[attr-defined]
                    _empty_cache()

        # Roofline classification
        ridge = (bf16_peak_tflops * 1e12 / (hbm_roof_gb_s * 1e9)
                 if (bf16_peak_tflops and hbm_roof_gb_s) else None)
        row["ridge_flop_per_byte"] = ridge
        if ridge is not None:
            row["bound"] = "compute" if op.arithmetic_intensity > ridge else "memory"
        else:
            row["bound"] = ""

        # Theoretical times (TESTPLAN §10.1)
        if bf16_peak_tflops:
            row["t_compute_theory_ms"] = (op.flops / (bf16_peak_tflops * 1e12)) * 1e3
        if hbm_roof_gb_s:
            row["t_memory_theory_ms"] = (op.bytes_hbm / (hbm_roof_gb_s * 1e9)) * 1e3
        if "t_compute_theory_ms" in row and "t_memory_theory_ms" in row:
            row["t_bottleneck_theory_ms"] = max(
                row["t_compute_theory_ms"], row["t_memory_theory_ms"]
            )
            for label in ("default", "optimized"):
                t_meas = row.get(f"t_ms_{label}")
                if t_meas and not (isinstance(t_meas, float) and t_meas != t_meas):  # not NaN
                    row[f"meas_over_theory_{label}"] = t_meas / row["t_bottleneck_theory_ms"]

        rows.append(row)
        t_def = row.get('t_ms_default', float('nan'))
        t_opt = row.get('t_ms_optimized', float('nan'))
        annotation = "" if has_gpu else (
            "  (analytic+measured)" if not skip_measure else "  (analytic-only)"
        )
        print(f"[04] {op.op_name:32s} AI={op.arithmetic_intensity:8.1f} FLOP/B  "
              f"def={t_def if isinstance(t_def, float) else float('nan'):7.3f}ms  "
              f"opt={t_opt if isinstance(t_opt, float) else float('nan'):7.3f}ms"
              f"{annotation}")

    write_csv(out_dir / "ops.csv", rows)
    write_json(out_dir / "ops.json", {
        "config": cfg_json,
        "totals": tot,
        "calibration_drift": cal_drift,
        "ridge_flop_per_byte": rows[0].get("ridge_flop_per_byte") if rows else None,
        "compute_roof_tflops": bf16_peak_tflops,
        "bandwidth_roof_gb_s": hbm_roof_gb_s,
        "rows": rows,
    })
    write_md_table(out_dir / "ops.md", rows, title="escher_14b_480p per-op accounting")

    print(f"[04] totals/block: {tot['total_gflops']:.1f} GFLOP, "
          f"{tot['total_mb_hbm']:.1f} MB HBM, AI={tot['avg_arithmetic_intensity']:.1f}")
    if cal_drift:
        print(f"[04] calibration drift: GFLOPs {cal_drift.get('gflops_drift_pct', 0):.1f}% "
              f"HBM {cal_drift.get('mb_hbm_drift_pct', 0):.1f}%")
    if not has_gpu:
        n_measured = sum(1 for r in rows
                         if isinstance(r.get("t_ms_default"), float)
                         and r["t_ms_default"] == r["t_ms_default"])  # not NaN
        print(f"[04] DEVICE=cpu — measured {n_measured}/{len(rows)} ops "
              f"(budget={args.cpu_budget_gflops} GFLOP); heavy GEMMs/attention "
              f"intentionally NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
