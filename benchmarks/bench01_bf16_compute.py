"""Family 1 — BF16 compute microbenchmarks (TESTPLAN §5).

Establishes the BF16 compute ceiling used by the roofline (§9) and as the
denominator for measured-peak MFU (§11). Runs:

  1. Square GEMM size sweep (M = N = K).
  2. Rectangular GEMM at workload-relevant projection / FFN shapes.
  3. Batched / addmm variants for projection-like kernels.
  4. Peak: tight-loop matmul at the largest size, dividing total / iters.

Outputs (under <out>/01_bf16_compute/):
  - sweep.json     all results, full per-iter times
  - sweep.csv      flat table for plotting
  - peak.json      single peak number used as the compute roof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op, time_tight_loop


SQUARE_SIZES = [1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768]


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


def square_sweep(device, warmup: int, iters: int) -> List[dict]:
    rows = []
    for s in SQUARE_SIZES:
        try:
            a, b, out = _alloc_pair(s, s, s, device)
        except torch.cuda.OutOfMemoryError:
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
        torch.cuda.empty_cache()
    return rows


def rectangular_sweep(device, warmup: int, iters: int, cfg: dict) -> List[dict]:
    """Workload-relevant projection / FFN shapes."""
    m = cfg["model"]
    s = cfg["shapes"]
    D = m["hidden_dim"]
    Dh = m["n_heads"] * m["head_dim"]
    Dff = D * m["ffn_expansion"]
    M_img = s["batch"] * s["seq_image"]
    M_txt = s["batch"] * s["seq_text"]

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
        except torch.cuda.OutOfMemoryError:
            print(f"[skip] rect {name}: OOM")
            continue
        def fn(a=a, b=b, out=out):
            torch.matmul(a, b, out=out)
        res = time_op(name, fn, warmup=warmup, iters=iters)
        rows.append({**_row(name, M, K, N, res.median_ms),
                     "p10_ms": res.p10_ms, "p90_ms": res.p90_ms})
        del a, b, out
        torch.cuda.empty_cache()
    return rows


def addmm_and_bmm(device, warmup: int, iters: int, cfg: dict) -> List[dict]:
    m = cfg["model"]
    s = cfg["shapes"]
    D = m["hidden_dim"]
    Dh = m["n_heads"] * m["head_dim"]
    M_img = s["batch"] * s["seq_image"]

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

    # bmm: per-head attention-style multiply
    n_heads = m["n_heads"]
    head = m["head_dim"]
    Sq = s["seq_image"]
    Sk = s["seq_image"]
    q = torch.randn(n_heads, Sq, head, dtype=torch.bfloat16, device=device)
    k = torch.randn(n_heads, head, Sk, dtype=torch.bfloat16, device=device)
    out = torch.empty(n_heads, Sq, Sk, dtype=torch.bfloat16, device=device)
    def fn_bmm(q=q, k=k, out=out):
        torch.bmm(q, k, out=out)
    res = time_op("bmm_qkT_per_head", fn_bmm, warmup=warmup, iters=iters)
    rows.append(_row("bmm_qkT_per_head", n_heads * Sq, head, Sk, res.median_ms))
    del q, k, out

    torch.cuda.empty_cache()
    return rows


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
    ap.add_argument("--peak-size", type=int, default=16384)
    ap.add_argument("--peak-iters", type=int, default=200)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = torch.device("cuda:0")
    cfg = json.loads(Path(args.config).read_text())

    out_dir = Path(args.out) / "01_bf16_compute"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01] device={torch.cuda.get_device_name(0)}")
    print("[01] square sweep ...")
    square = square_sweep(device, args.warmup, args.iters)
    print("[01] rectangular sweep ...")
    rect = rectangular_sweep(device, args.warmup, args.iters, cfg)
    print("[01] addmm + bmm ...")
    extras = addmm_and_bmm(device, args.warmup, args.iters, cfg)
    print(f"[01] peak tight loop @ {args.peak_size} ...")
    peak = peak_tight_loop(device, args.peak_size, args.warmup, args.peak_iters)

    rows = square + rect + extras
    write_json(out_dir / "sweep.json", {"square": square, "rectangular": rect, "extra": extras})
    write_csv(out_dir / "sweep.csv", rows)
    write_json(out_dir / "peak.json", peak)

    # Compact summary.
    best = max(rows, key=lambda r: r.get("tflops", 0.0))
    summary = {
        "device": torch.cuda.get_device_name(0),
        "best_sweep": best,
        "peak_tight_loop_tflops": peak["tflops"],
        "compute_roof_tflops": peak["tflops"],
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[01] peak {peak['tflops']:.2f} TFLOP/s @ {args.peak_size} ; best sweep "
          f"{best['name']}={best['tflops']:.2f} TFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
