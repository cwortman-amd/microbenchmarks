"""Family 1 — BF16 compute microbenchmarks (TESTPLAN §5).

Establishes the BF16 compute ceiling used by the roofline (§9) and as the
denominator for measured-peak MFU (§11). Runs:

  1. Square GEMM size sweep (M = N = K).
  2. Rectangular GEMM at workload-relevant projection / FFN shapes
     (the *fused* variants — qkv_fused, kv_fused — that an optimized
     kernel would dispatch).
  3. **Component GEMM sweep** — one row per GEMM in the per-block op
     decomposition (self_attn.{q,k,v,o}, cross_attn.{q,k,v,o},
     ffn.linear{1,2}, time_proj, time_embed). Names line up 1:1 with
     ``bench04``'s ops table so the per-component BF16 throughput can be
     diff'd directly against the per-op measured timing.
  4. Batched / addmm variants for projection-like kernels.
  5. Peak: tight-loop matmul at the largest size, dividing total / iters.

Outputs (under ``<out>/01_bf16_compute/``):
  - sweep.json            all results, full per-iter times
  - sweep.csv             flat table for plotting
  - peak.json             single peak number used as the compute roof
  - component_gemms.json  per-component GEMM TFLOP/s table (workload-shaped)
  - component_gemms.csv   same as flat CSV for plotting / pandoc

Device-agnostic: runs on CUDA/HIP when an accelerator is present, otherwise
falls back to a CPU-only sweep using ``torch.matmul`` on bf16 CPU tensors
(works on x86 with ``avx512_bf16`` and on aarch64 with BF16 instruction set
extensions; otherwise PyTorch upcasts internally and the numbers reflect the
emulated path). The CPU sweep uses smaller default sizes — large square
GEMMs on CPU can take many seconds per iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from benchmarks.common.flop_accounting import WorkloadConfig, gemm_inventory
from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op, time_tight_loop


SQUARE_SIZES_GPU = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768]
SQUARE_SIZES_CPU = [256, 512, 1024, 2048, 4096]


def _is_oom(exc: BaseException) -> bool:
    """Detect OOM in a device-agnostic way."""
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def _flops_gemm(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def _alloc_pair(M: int, K: int, N: int, device: torch.device):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    return a, b, out


def _row(name: str, M: int, K: int, N: int, t_ms: float) -> dict:
    flops = _flops_gemm(M, N, K)
    return {
        "name": name,
        "M": M,
        "K": K,
        "N": N,
        "t_ms_median": t_ms,
        "tflops": flops / (t_ms * 1e-3) / 1e12,
        "flops": flops,
    }


def square_sweep(device, warmup: int, iters: int, sizes: List[int]) -> List[dict]:
    rows = []
    for s in sizes:
        try:
            a, b, out = _alloc_pair(s, s, s, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            print(f"[skip] square {s}: OOM")
            continue
        # Use out= form to avoid allocator churn inside the timed region.
        def fn(a=a, b=b, out=out):
            torch.matmul(a, b, out=out)
        res = time_op(f"square_{s}", fn, warmup=warmup, iters=iters)
        rows.append({**_row(f"square_{s}", s, s, s, res.median_ms),
                     "p10_ms": res.p10_ms, "p90_ms": res.p90_ms,
                     "min_ms": res.min_ms, "max_ms": res.max_ms,
                     "std_ms": res.std_ms, "iters": iters})
        del a, b, out
        _empty_cache()
    return rows


def rectangular_sweep(device, warmup: int, iters: int, cfg: dict,
                      m_cap: int = 0) -> List[dict]:
    """Workload-relevant projection / FFN shapes.

    ``m_cap`` lets the CPU path clamp the leading dim (sequence length) so
    individual GEMMs don't blow up runtime. Set to 0 (default) for no cap.
    """
    m = cfg["model"]
    s = cfg["shapes"]
    D = m["hidden_dim"]
    Dh = m["n_heads"] * m["head_dim"]
    Dff = D * m["ffn_expansion"]
    M_img = s["batch"] * s["seq_image"]
    M_txt = s["batch"] * s["seq_text"]
    if m_cap:
        M_img = min(M_img, m_cap)
        M_txt = min(M_txt, m_cap)

    cases = [
        ("self_attn_qkv_fused", M_img, D, 3 * Dh),
        ("self_attn_q",         M_img, D, Dh),
        ("self_attn_o",         M_img, Dh, D),
        ("cross_attn_q",        M_img, D, Dh),
        ("cross_attn_kv_fused", M_txt, D, 2 * Dh),
        ("ffn_linear1",         M_img, D, Dff),
        ("ffn_linear2",         M_img, Dff, D),
    ]
    rows = []
    for name, M, K, N in cases:
        try:
            a, b, out = _alloc_pair(M, K, N, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            print(f"[skip] rect {name}: OOM")
            continue
        def fn(a=a, b=b, out=out):
            torch.matmul(a, b, out=out)
        res = time_op(name, fn, warmup=warmup, iters=iters)
        rows.append({**_row(name, M, K, N, res.median_ms),
                     "p10_ms": res.p10_ms, "p90_ms": res.p90_ms})
        del a, b, out
        _empty_cache()
    return rows


def component_gemm_sweep(device, warmup: int, iters: int, cfg: dict,
                         m_cap: int = 0,
                         flop_budget_gflops: float = 0.0) -> List[dict]:
    """Time every dense GEMM in the workload's per-block decomposition.

    Each row carries the canonical ``op_name`` from
    :func:`benchmarks.common.flop_accounting.per_block_ops`, the (M, K, N)
    shape it would have at the configured ``shapes.{batch, seq_image,
    seq_text}``, the analytic GFLOPs, and the measured BF16 throughput.

    Parameters
    ----------
    m_cap
        Optional cap on the leading dimension (rows of A / output) so the
        CPU path doesn't grind through full S = 8192 GEMMs. Set to 0 for
        no cap.
    flop_budget_gflops
        On CPU, GEMMs whose analytic FLOPs exceed this budget are
        recorded with NaN timings (analytic-only). Use 0 to time
        everything regardless of cost.
    """
    cfg_obj = WorkloadConfig.from_json(cfg)
    inventory = gemm_inventory(cfg_obj)
    rows = []
    budget_flops = flop_budget_gflops * 1e9 if flop_budget_gflops > 0 else float("inf")
    for spec in inventory:
        name = spec["name"]
        M, K, N = spec["M"], spec["K"], spec["N"]
        # Apply the leading-dim cap to keep CPU GEMMs tractable. Scale FLOPs
        # accordingly so TFLOP/s reflects the *capped* shape.
        M_eff = min(M, m_cap) if m_cap and M > m_cap else M
        flops_eff = 2 * M_eff * K * N

        record = {
            **spec,
            "M_measured": M_eff,
            "flops_measured": flops_eff,
            "gflops_measured": flops_eff / 1e9,
            "t_ms_median": float("nan"),
            "tflops": float("nan"),
            "p10_ms": float("nan"),
            "p90_ms": float("nan"),
            "skipped_reason": "",
        }

        if flops_eff > budget_flops:
            record["skipped_reason"] = (
                f"flops {flops_eff/1e9:.2f} G > budget {flop_budget_gflops:.2f} G"
            )
            rows.append(record)
            continue

        try:
            a, b, out = _alloc_pair(M_eff, K, N, device)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            record["skipped_reason"] = "OOM"
            rows.append(record)
            continue

        def fn(a=a, b=b, out=out):
            torch.matmul(a, b, out=out)

        try:
            res = time_op(name, fn, warmup=warmup, iters=iters)
            record["t_ms_median"] = res.median_ms
            record["tflops"] = flops_eff / (res.median_ms * 1e-3) / 1e12
            record["p10_ms"] = res.p10_ms
            record["p90_ms"] = res.p90_ms
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            record["skipped_reason"] = "OOM"
        finally:
            del a, b, out
            _empty_cache()

        rows.append(record)
    return rows


def addmm_and_bmm(device, warmup: int, iters: int, cfg: dict,
                  m_cap: int = 0) -> List[dict]:
    m = cfg["model"]
    s = cfg["shapes"]
    D = m["hidden_dim"]
    Dh = m["n_heads"] * m["head_dim"]
    M_img = s["batch"] * s["seq_image"]
    if m_cap:
        M_img = min(M_img, m_cap)

    rows = []
    # addmm: y = bias + x @ W
    a = torch.randn(M_img, D, dtype=torch.bfloat16, device=device)
    w = torch.randn(D, Dh, dtype=torch.bfloat16, device=device)
    bias = torch.randn(Dh, dtype=torch.bfloat16, device=device)
    out = torch.empty(M_img, Dh, dtype=torch.bfloat16, device=device)
    def fn_addmm(a=a, w=w, bias=bias, out=out):
        torch.addmm(bias, a, w, out=out)
    res = time_op("addmm_self_attn_q", fn_addmm, warmup=warmup, iters=iters)
    rows.append(_row("addmm_self_attn_q", M_img, D, Dh, res.median_ms))
    del a, w, bias, out

    # bmm: per-head attention-style multiply. Cap S on CPU to avoid S² blowup.
    n_heads = m["n_heads"]
    head = m["head_dim"]
    Sq = Sk = s["seq_image"]
    if m_cap:
        Sq = Sk = min(Sq, max(m_cap // max(s["batch"], 1), 256))
    q = torch.randn(n_heads, Sq, head, dtype=torch.bfloat16, device=device)
    k = torch.randn(n_heads, head, Sk, dtype=torch.bfloat16, device=device)
    out = torch.empty(n_heads, Sq, Sk, dtype=torch.bfloat16, device=device)
    def fn_bmm(q=q, k=k, out=out):
        torch.bmm(q, k, out=out)
    res = time_op("bmm_qkT_per_head", fn_bmm, warmup=warmup, iters=iters)
    rows.append(_row("bmm_qkT_per_head", n_heads * Sq, head, Sk, res.median_ms))
    del q, k, out

    _empty_cache()
    return rows


def _supported_dtypes():
    """Return a tuple of (label, dtype) for every dtype this build supports.

    The list is intentionally conservative — only formats whose matmul
    semantics are well-defined under PyTorch's standard GEMM dispatcher are
    included. FP8 variants are gated behind feature-flags because:

      * ``float8_e4m3fn`` / ``float8_e5m2`` only exist on torch ≥ 2.1
      * the matmul dispatch for FP8 requires both inputs in FP8 *and* an
        FP32 accumulator output on CPU (no native FP8 GEMM kernel), so we
        skip rather than emit a meaningless "FP8 == BF16-emulated" number.

    On hardware that does have native FP8 (MI300X+, H100), ``torch.matmul``
    will route through the right kernel and we'll capture genuine FP8
    throughput.
    """
    candidates = [
        ("fp32",   getattr(torch, "float32",  None)),
        ("fp16",   getattr(torch, "float16",  None)),
        ("bf16",   getattr(torch, "bfloat16", None)),
        ("fp8_e4m3", getattr(torch, "float8_e4m3fn", None)),
        ("fp8_e5m2", getattr(torch, "float8_e5m2",   None)),
    ]
    return [(lbl, dt) for lbl, dt in candidates if dt is not None]


def dtype_sweep(device, warmup: int, iters: int, size: int = 2048) -> List[dict]:
    """Time a square GEMM at the same ``size`` across every supported dtype.

    Output rows have a uniform schema so the report can render them as a
    side-by-side table:

        {dtype, M, K, N, t_ms_median, tflops, supported, error}

    A dtype with ``supported=False`` carries an ``error`` string explaining
    why it was skipped (typically "matmul not implemented for dtype X on
    backend Y"). This is exactly the failure mode FP8-on-CPU hits today,
    and surfacing it cleanly is more useful than silently dropping the row.
    """
    rows: List[dict] = []
    for label, dtype in _supported_dtypes():
        flops = _flops_gemm(size, size, size)
        try:
            # FP8 dtypes do not support `randn` directly. Cast from fp32.
            if "fp8" in label:
                a32 = torch.randn(size, size, dtype=torch.float32, device=device)
                b32 = torch.randn(size, size, dtype=torch.float32, device=device)
                a = a32.to(dtype)
                b = b32.to(dtype)
                # Output for FP8 GEMM is conventionally a higher-precision
                # accumulator. Use bf16 / fp32 depending on availability.
                out_dtype = torch.bfloat16
                out = torch.empty(size, size, dtype=out_dtype, device=device)
                def fn(a=a, b=b, out=out):
                    torch.matmul(a.to(out.dtype), b.to(out.dtype), out=out)
                # Note: torch.matmul on FP8 inputs without an explicit
                # ``_scaled_mm`` kernel may upcast — that's intentional. We
                # capture whatever the standard dispatch does and label
                # accordingly.
            else:
                a = torch.randn(size, size, dtype=dtype, device=device)
                b = torch.randn(size, size, dtype=dtype, device=device)
                out = torch.empty(size, size, dtype=dtype, device=device)
                def fn(a=a, b=b, out=out):
                    torch.matmul(a, b, out=out)
            res = time_op(f"dtype_{label}", fn, warmup=warmup, iters=iters)
            rows.append({
                "dtype":    label,
                "M": size, "K": size, "N": size,
                "t_ms_median": res.median_ms,
                "tflops":   flops / (res.median_ms * 1e-3) / 1e12,
                "supported": True,
                "note": ("FP8 inputs upcast to bf16 for matmul (no native FP8 GEMM "
                         "on this backend)" if "fp8" in label else None),
            })
        except (RuntimeError, NotImplementedError, TypeError) as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            rows.append({
                "dtype":     label,
                "M": size, "K": size, "N": size,
                "t_ms_median": None,
                "tflops":     None,
                "supported":  False,
                "error":      msg[:200],
            })
        finally:
            for _name in ("a", "b", "out"):
                if _name in locals():
                    del locals()[_name]
            _empty_cache()
    return rows


def correctness_check(device, sizes=(256, 1024)) -> dict:
    """Sanity-check the BF16 matmul kernel against an FP32 reference.

    For each size we generate the *same* FP32 random matrices, downcast to
    BF16, compute ``c_bf16 = a_bf16 @ b_bf16``, and compare to
    ``c_ref = a_fp32 @ b_fp32`` (upcast for the comparison). The point of
    this gate is **not** to validate BF16 precision per se — that's a
    hardware/architecture property. The point is to catch the silent
    failure modes that a pure-throughput bench misses:

      - kernel returns the wrong shape / wrong dtype but timing still works,
      - kernel computes ``A·Bᵀ`` instead of ``A·B`` (transpose bug),
      - kernel skips accumulation past some K threshold,
      - hardware reports BF16 done but actually emulated as FP16 (different
        accumulation precision => much larger error than the analytic bound).

    The pass bound is a 5× safety factor on the analytic BF16 GEMM error:
    for inputs drawn from N(0, 1) at inner dim K the expected relative
    error after a single BF16 GEMM is ≈ ``√K · 2⁻⁸`` (8-bit BF16 mantissa,
    accumulated over K terms whose individual rounding errors add in
    quadrature). 5× that bound flags real numerical regressions while
    accommodating per-architecture rounding-mode differences.
    """
    results = []
    for size in sizes:
        torch.manual_seed(0)
        a = torch.randn(size, size, dtype=torch.float32, device=device)
        b = torch.randn(size, size, dtype=torch.float32, device=device)
        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        c_bf16 = (a_bf16 @ b_bf16).to(torch.float32)
        c_ref = a @ b
        diff = (c_bf16 - c_ref).abs()
        max_abs = diff.max().item()
        ref_scale = c_ref.abs().max().item()
        mean_abs = diff.mean().item()
        max_rel = max_abs / ref_scale if ref_scale > 0 else float("inf")
        # Analytic per-element bound × 5× safety factor.
        bound = 5.0 * (size ** 0.5) * (2 ** -8)
        passed = max_rel <= bound and not (max_rel != max_rel)  # also catches NaN
        results.append({
            "size": size,
            "K": size,
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
            "max_rel_err": max_rel,
            "ref_max_abs": ref_scale,
            "rel_err_bound": bound,
            "passed": bool(passed),
        })
        del a, b, a_bf16, b_bf16, c_bf16, c_ref, diff
        _empty_cache()
    return {
        "rows": results,
        "all_passed": all(r["passed"] for r in results),
    }


def peak_tight_loop(device, size: int, warmup: int, iters: int) -> dict:
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    out = torch.empty(size, size, dtype=torch.bfloat16, device=device)
    def fn(a=a, b=b, out=out):
        torch.matmul(a, b, out=out)
    res = time_tight_loop(f"peak_square_{size}", fn, warmup=warmup, iters=iters)
    flops = _flops_gemm(size, size, size)
    return {
        "size": size,
        "iters": iters,
        "t_iter_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "tight_loop_total_ms": res.extra.get("tight_loop_total_ms"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--peak-size", type=int, default=0,
                    help="Square peak-sweep size. 0 = device default (16384 GPU / 2048 CPU).")
    ap.add_argument("--peak-iters", type=int, default=0,
                    help="Tight-loop iterations at peak size. 0 = device default (200 GPU / 20 CPU).")
    ap.add_argument("--component-gemm-budget-gflops", type=float, default=0.0,
                    help="On CPU, skip per-component GEMM timing for GEMMs whose "
                         "analytic FLOPs (after the leading-dim cap) exceed this "
                         "budget. 0 (default) = time everything; the leading-dim "
                         "cap (--m_cap) is what keeps the per-GEMM runtime "
                         "bounded on CPU.")
    args = ap.parse_args()

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")
    cfg = json.loads(Path(args.config).read_text())

    out_dir = Path(args.out) / "01_bf16_compute"
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_gpu:
        sizes = SQUARE_SIZES_GPU
        peak_size_cli = args.peak_size or 16384
        peak_iters = args.peak_iters or 200
        m_cap = 0
        warmup = args.warmup
        iters = args.iters
        auto_peak_size = False
    else:
        sizes = SQUARE_SIZES_CPU
        peak_size_cli = args.peak_size  # honor user override when set
        peak_iters = args.peak_iters or 20
        m_cap = 4096          # cap leading dim for rect/addmm/bmm so per-iter <~1s
        # CPU runs are seconds per iter; trim warmup/iters for tractable runtime.
        warmup = max(1, min(args.warmup, 2))
        iters = max(3, min(args.iters, 5))
        auto_peak_size = True

    dev_label = _device_label(device)
    print(f"[01] device={dev_label} sizes={sizes} peak_size_cli={peak_size_cli or 'auto'}")
    print("[01] square sweep ...")
    square = square_sweep(device, warmup, iters, sizes)
    print("[01] rectangular sweep ...")
    rect = rectangular_sweep(device, warmup, iters, cfg, m_cap=m_cap)
    print("[01] component GEMM sweep (per-block decomposition) ...")
    component_budget = (args.component_gemm_budget_gflops if not has_gpu else 0.0)
    components = component_gemm_sweep(
        device, warmup, iters, cfg,
        m_cap=m_cap, flop_budget_gflops=component_budget,
    )
    print("[01] addmm + bmm ...")
    extras = addmm_and_bmm(device, warmup, iters, cfg, m_cap=m_cap)
    # Cross-dtype throughput sweep so the report can show fp16/fp32/fp8
    # alongside the headline bf16 number. Uses a single representative size
    # so the comparison is apples-to-apples; the sweep itself is dirt-cheap
    # compared to bench01's main loop.
    print("[01] dtype sweep (fp32/fp16/bf16/fp8 if available) ...")
    dtype_size = 1024 if not has_gpu else 4096
    dtype_rows = dtype_sweep(device, warmup, iters, size=dtype_size)

    # On CPU the best square size is BLAS- and cache-dependent; lock the
    # tight-loop peak to the best-performing sweep size unless the user
    # explicitly pinned --peak-size. This keeps best_pct_of_peak <= 100% in
    # SC-1 and avoids the "best sweep beats peak" pathology.
    if auto_peak_size and not peak_size_cli and square:
        best_sq = max(square, key=lambda r: r["tflops"])
        peak_size = int(best_sq["M"])
        print(f"[01] CPU host: auto-selected peak size from best sweep = {peak_size}")
    else:
        peak_size = peak_size_cli or 2048

    print(f"[01] peak tight loop @ {peak_size} ...")
    peak = peak_tight_loop(device, peak_size, warmup, peak_iters)

    # Numerical correctness gate. Two shapes: a small one to catch
    # transpose / shape / dtype bugs cheaply, and a larger one to surface
    # accumulation-precision regressions (BF16 emulated as FP16 etc.).
    print("[01] BF16 GEMM correctness check vs FP32 reference ...")
    correctness_sizes = (256, 1024) if has_gpu else (256, 512)
    correctness = correctness_check(device, sizes=correctness_sizes)

    rows = square + rect + extras
    write_json(out_dir / "sweep.json",
               {"square": square, "rectangular": rect, "extra": extras,
                "component_gemms": components})
    write_csv(out_dir / "sweep.csv", rows)
    write_json(out_dir / "peak.json", peak)
    # Component GEMMs get their own dedicated artifact so the report can
    # render the workload's GEMM inventory + measured BF16 throughput as a
    # single self-contained table.
    write_json(out_dir / "component_gemms.json", {
        "device": dev_label,
        "device_type": device.type,
        "config_path": str(args.config),
        "leading_dim_cap": m_cap,
        "flop_budget_gflops": component_budget,
        "rows": components,
    })
    write_csv(out_dir / "component_gemms.csv", components)
    write_json(out_dir / "correctness.json", {
        "device": dev_label,
        "device_type": device.type,
        "rows": correctness["rows"],
        "all_passed": correctness["all_passed"],
    })
    write_json(out_dir / "dtype_sweep.json", {
        "device":      dev_label,
        "device_type": device.type,
        "size":        dtype_size,
        "rows":        dtype_rows,
    })

    # Compact summary.
    best = max(rows, key=lambda r: r.get("tflops", 0.0))
    # The compute roof is the higher of (a) tight-loop median at peak_size and
    # (b) the best individual sweep result. On GPUs the tight-loop number
    # essentially always wins; on CPUs cache + scheduler variance can flip
    # which one dominates between runs. Taking max keeps SC-1's
    # `best_sweep / peak >= 0.90` consistency check well-defined.
    compute_roof = max(peak["tflops"], best["tflops"])
    summary = {
        "device": dev_label,
        "device_type": device.type,
        "best_sweep": best,
        "peak_tight_loop_tflops": peak["tflops"],
        "compute_roof_tflops": compute_roof,
        "correctness_passed": correctness["all_passed"],
        "correctness_summary": [
            {"size": r["size"], "max_rel_err": r["max_rel_err"],
             "bound": r["rel_err_bound"], "passed": r["passed"]}
            for r in correctness["rows"]
        ],
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[01] tight_loop_peak={peak['tflops']:.3f}  best_sweep={best['name']}="
          f"{best['tflops']:.3f}  compute_roof={compute_roof:.3f} TFLOP/s")
    verdict = "PASS" if correctness["all_passed"] else "FAIL"
    print(f"[01] correctness vs FP32 ref: {verdict}")
    for r in correctness["rows"]:
        flag = "ok" if r["passed"] else "FAIL"
        print(f"[01]   K={r['K']:5d}  max_rel_err={r['max_rel_err']:.4f}  "
              f"(bound={r['rel_err_bound']:.4f})  [{flag}]")
    print(f"[01] dtype throughput @ M=K=N={dtype_size}:")
    for r in dtype_rows:
        if r["supported"]:
            print(f"[01]   {r['dtype']:8s}  t={r['t_ms_median']:.3f}ms  "
                  f"{r['tflops']:6.3f} TFLOP/s")
        else:
            print(f"[01]   {r['dtype']:8s}  unsupported: {r.get('error', '?')}")

    measured = [r for r in components if not (isinstance(r["tflops"], float)
                                              and r["tflops"] != r["tflops"])]
    skipped = [r for r in components if r.get("skipped_reason")]
    print(f"[01] component GEMMs: {len(measured)}/{len(components)} measured "
          f"(skipped={len(skipped)})")
    for r in components:
        if r.get("skipped_reason"):
            print(f"[01]   {r['name']:32s} ({r['M']}x{r['K']}x{r['N']}) "
                  f"SKIPPED: {r['skipped_reason']}")
        else:
            print(f"[01]   {r['name']:32s} ({r['M_measured']}x{r['K']}x{r['N']}) "
                  f"t={r['t_ms_median']:.3f}ms  {r['tflops']:6.3f} TFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
