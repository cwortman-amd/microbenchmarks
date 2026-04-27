"""Family 2 — HBM bandwidth microbenchmarks (TESTPLAN §6).

Establishes the bandwidth ceiling used by the roofline (§9) and the
`t_memory_theoretical` denominator in §10.

Each micro-op moves a known number of HBM bytes per element. We use bf16
tensors to match the workload dtype.

  BW-1  copy_           2 R + 2 W (4 B/elt)
  BW-2  add             2 R + 1 W (6 B/elt)
  BW-3  mul             2 R + 1 W (6 B/elt)
  BW-4  axpy (add_)     2 R + 1 W (6 B/elt)   [in-place: dst is read+written]
  BW-5  sum reduction   1 R       (2 B/elt)
  BW-6  fill_           1 W       (2 B/elt)
  BW-7  strided copy    1 R + 1 W (4 B/elt)   [non-contiguous read]

Sizes are powers-of-two across a wide range so the launch-bound -> BW-bound
plateau is visible in the resulting curve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


SIZES_BYTES = [
    64 * 1024 * 1024,            # 64 MiB
    256 * 1024 * 1024,           # 256 MiB
    1 * 1024 * 1024 * 1024,      # 1 GiB
    2 * 1024 * 1024 * 1024,      # 2 GiB
    4 * 1024 * 1024 * 1024,      # 4 GiB
    8 * 1024 * 1024 * 1024,      # 8 GiB
]


def _alloc_bf16(n_bytes: int, device) -> torch.Tensor:
    n_elem = n_bytes // 2
    return torch.empty(n_elem, dtype=torch.bfloat16, device=device)


def _try_alloc(n_bytes: int, n: int, device) -> List[torch.Tensor]:
    """Try to allocate `n` buffers of size `n_bytes` each. Raises OOM if can't."""
    return [_alloc_bf16(n_bytes, device) for _ in range(n)]


def bench_copy(n_bytes: int, device, warmup: int, iters: int) -> dict:
    src, dst = _try_alloc(n_bytes, 2, device)
    src.normal_()
    def fn(src=src, dst=dst):
        dst.copy_(src)
    r = time_op(f"copy_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (2 * n_bytes) / (r.median_ms * 1e-3) / 1e9  # 1R + 1W
    return {"op": "copy_", "bytes": n_bytes, "bytes_per_elt": 4, "t_ms": r.median_ms,
            "gb_s": bw, "p10_ms": r.p10_ms, "p90_ms": r.p90_ms, "std_ms": r.std_ms}


def bench_add(n_bytes: int, device, warmup: int, iters: int) -> dict:
    a, b, out = _try_alloc(n_bytes, 3, device)
    a.normal_(); b.normal_()
    def fn(a=a, b=b, out=out):
        torch.add(a, b, out=out)
    r = time_op(f"add_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (3 * n_bytes) / (r.median_ms * 1e-3) / 1e9  # 2R + 1W
    return {"op": "add", "bytes": n_bytes, "bytes_per_elt": 6, "t_ms": r.median_ms, "gb_s": bw}


def bench_mul(n_bytes: int, device, warmup: int, iters: int) -> dict:
    a, b, out = _try_alloc(n_bytes, 3, device)
    a.normal_(); b.normal_()
    def fn(a=a, b=b, out=out):
        torch.mul(a, b, out=out)
    r = time_op(f"mul_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (3 * n_bytes) / (r.median_ms * 1e-3) / 1e9
    return {"op": "mul", "bytes": n_bytes, "bytes_per_elt": 6, "t_ms": r.median_ms, "gb_s": bw}


def bench_axpy(n_bytes: int, device, warmup: int, iters: int) -> dict:
    y, x = _try_alloc(n_bytes, 2, device)
    y.normal_(); x.normal_()
    def fn(y=y, x=x):
        y.add_(x, alpha=1.5)  # in-place: 1R(x) + 1R+1W(y)
    r = time_op(f"axpy_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (3 * n_bytes) / (r.median_ms * 1e-3) / 1e9
    return {"op": "axpy", "bytes": n_bytes, "bytes_per_elt": 6, "t_ms": r.median_ms, "gb_s": bw}


def bench_sum(n_bytes: int, device, warmup: int, iters: int) -> dict:
    x = _alloc_bf16(n_bytes, device)
    x.normal_()
    def fn(x=x):
        _ = x.sum()
    r = time_op(f"sum_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (1 * n_bytes) / (r.median_ms * 1e-3) / 1e9
    return {"op": "sum", "bytes": n_bytes, "bytes_per_elt": 2, "t_ms": r.median_ms, "gb_s": bw}


def bench_fill(n_bytes: int, device, warmup: int, iters: int) -> dict:
    x = _alloc_bf16(n_bytes, device)
    def fn(x=x):
        x.fill_(1.0)
    r = time_op(f"fill_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (1 * n_bytes) / (r.median_ms * 1e-3) / 1e9
    return {"op": "fill_", "bytes": n_bytes, "bytes_per_elt": 2, "t_ms": r.median_ms, "gb_s": bw}


def bench_strided_copy(n_bytes: int, device, warmup: int, iters: int) -> dict:
    """Strided read: take every other element. Effective bytes touched = n_bytes."""
    src = _alloc_bf16(n_bytes * 2, device)
    src.normal_()
    dst = _alloc_bf16(n_bytes, device)
    src_strided = src[::2]
    def fn(src=src_strided, dst=dst):
        dst.copy_(src)
    r = time_op(f"strided_copy_{n_bytes}", fn, warmup=warmup, iters=iters)
    bw = (2 * n_bytes) / (r.median_ms * 1e-3) / 1e9
    return {"op": "strided_copy", "bytes": n_bytes, "bytes_per_elt": 4,
            "t_ms": r.median_ms, "gb_s": bw}


BENCHES = [
    ("BW-1", bench_copy, 2),       # buffers needed
    ("BW-2", bench_add, 3),
    ("BW-3", bench_mul, 3),
    ("BW-4", bench_axpy, 2),
    ("BW-5", bench_sum, 1),
    ("BW-6", bench_fill, 1),
    ("BW-7", bench_strided_copy, 3),  # 2x src + 1x dst => effectively 3
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = torch.device("cuda:0")

    out_dir = Path(args.out) / "02_hbm_bandwidth"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for n_bytes in SIZES_BYTES:
        for bid, fn, n_buf in BENCHES:
            try:
                row = fn(n_bytes, device, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                print(f"[02] OOM at {bid} {n_bytes/1e9:.1f} GB — skipping")
                torch.cuda.empty_cache()
                continue
            row["bench_id"] = bid
            rows.append(row)
            print(f"[02] {bid} {row['op']:>14s} {n_bytes/1e9:6.2f} GB -> {row['gb_s']:8.1f} GB/s")
            torch.cuda.empty_cache()

    write_csv(out_dir / "bandwidth.csv", rows)
    write_json(out_dir / "bandwidth.json", rows)

    # Plateau = max of last-two-size sustained for each op.
    plateau = {}
    for op in {r["op"] for r in rows}:
        per_op = sorted([r for r in rows if r["op"] == op], key=lambda r: r["bytes"])
        plateau[op] = max((r["gb_s"] for r in per_op[-2:]), default=0.0)
    bw_roof = max(plateau.values()) if plateau else 0.0
    summary = {"plateau_gb_s_per_op": plateau, "bandwidth_roof_gb_s": bw_roof}
    write_json(out_dir / "summary.json", summary)
    print(f"[02] bandwidth roof = {bw_roof:.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
