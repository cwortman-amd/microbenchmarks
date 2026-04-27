"""Family 3 — VRAM capacity (TESTPLAN §7).

Determines practical allocatable memory via binary search. Reports:

  - max single contiguous bf16 allocation
  - effective utilization vs nominal 288 GB
  - fragmentation sensitivity (contig vs many-chunk)
  - dtype scaling (bf16 vs fp16 should be ~equal)
  - allocator state (torch.cuda.memory_stats dump)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.common.io import write_json


NOMINAL_BYTES = 288 * 1024 ** 3  # MI355X spec


def _try_alloc_contig(n_bytes: int, device, dtype) -> bool:
    n_elem = n_bytes // dtype.itemsize if hasattr(dtype, "itemsize") else n_bytes // 2
    try:
        t = torch.empty(n_elem, dtype=dtype, device=device)
        # Force a real residency probe by writing one element.
        t.fill_(0)
        torch.cuda.synchronize()
        del t
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


def binary_search_max_contig(device, dtype, lo: int, hi: int, tol_bytes: int = 64 * 1024 * 1024) -> int:
    """Binary search the largest single contiguous allocation that succeeds."""
    # Ensure hi fails before search.
    while _try_alloc_contig(hi, device, dtype):
        lo = hi
        hi = int(hi * 1.5)
        if hi > NOMINAL_BYTES * 2:
            return lo  # absurd, return last good
    while hi - lo > tol_bytes:
        mid = (lo + hi) // 2
        if _try_alloc_contig(mid, device, dtype):
            lo = mid
        else:
            hi = mid
    return lo


def fragmentation_probe(device, target_bytes: int, chunk_bytes: int) -> dict:
    """Try to allocate target_bytes as many chunks of chunk_bytes; report success."""
    n_chunks = target_bytes // chunk_bytes
    chunks = []
    allocated = 0
    try:
        for _ in range(n_chunks):
            t = torch.empty(chunk_bytes // 2, dtype=torch.bfloat16, device=device)
            t.fill_(0)
            chunks.append(t)
            allocated += chunk_bytes
    except torch.cuda.OutOfMemoryError:
        pass
    finally:
        del chunks
        torch.cuda.empty_cache()
    return {"target_bytes": target_bytes, "chunk_bytes": chunk_bytes,
            "allocated_bytes": allocated,
            "frac_of_target": allocated / target_bytes if target_bytes else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frag-chunk-mb", type=int, default=512)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = torch.device("cuda:0")

    out_dir = Path(args.out) / "03_vram_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[03] binary search bf16 contiguous max ...")
    bf16_max = binary_search_max_contig(device, torch.bfloat16, lo=8 * 1024 ** 3, hi=64 * 1024 ** 3)
    print(f"[03]   bf16: {bf16_max/1024**3:.2f} GiB")
    print("[03] binary search fp16 contiguous max ...")
    fp16_max = binary_search_max_contig(device, torch.float16, lo=8 * 1024 ** 3, hi=64 * 1024 ** 3)
    print(f"[03]   fp16: {fp16_max/1024**3:.2f} GiB")

    print("[03] fragmentation probe (many small chunks) ...")
    frag = fragmentation_probe(device, target_bytes=bf16_max,
                               chunk_bytes=args.frag_chunk_mb * 1024 * 1024)
    print(f"[03]   chunked allocated {frag['allocated_bytes']/1024**3:.2f} GiB "
          f"({frag['frac_of_target']*100:.1f}% of contig)")

    mem_stats = {k: v for k, v in torch.cuda.memory_stats(device).items()
                 if isinstance(v, (int, float))}
    free, total = torch.cuda.mem_get_info(device)
    summary = {
        "device_total_bytes": total,
        "device_free_bytes_pre_test": free,
        "nominal_bytes": NOMINAL_BYTES,
        "max_alloc_bf16_bytes": bf16_max,
        "max_alloc_fp16_bytes": fp16_max,
        "max_alloc_bf16_gib": bf16_max / 1024 ** 3,
        "eff_util_fraction_bf16": bf16_max / NOMINAL_BYTES,
        "eff_util_fraction_vs_total": bf16_max / total,
        "frag_sensitivity_ratio": frag["frac_of_target"],
        "frag_chunk_mb": args.frag_chunk_mb,
        "memory_stats": mem_stats,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[03] bf16 effective util vs spec: {summary['eff_util_fraction_bf16']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
