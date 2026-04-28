"""Family 2 — HBM / DRAM bandwidth microbenchmarks (TESTPLAN §6).

Establishes the bandwidth ceiling used by the roofline (§9) and the
``t_memory_theoretical`` denominator in §10.

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

Device-agnostic: HBM on CUDA/HIP, system DDR on CPU. The ``bandwidth_roof_gb_s``
field in summary.json carries whichever sustained number this host produced;
``device_type`` annotates the source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import torch

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


SIZES_BYTES_GPU = [
    64 * 1024 * 1024,            # 64 MiB
    256 * 1024 * 1024,           # 256 MiB
    1 * 1024 * 1024 * 1024,      # 1 GiB
    2 * 1024 * 1024 * 1024,      # 2 GiB
    4 * 1024 * 1024 * 1024,      # 4 GiB
    8 * 1024 * 1024 * 1024,      # 8 GiB
]
# CPU caps out far below GPU; large sizes thrash the OS page cache and add
# noise without revealing the plateau. 16/64/256 MB and 1 GiB cover the
# launch-bound -> BW-bound transition for most x86 / aarch64 hosts.
SIZES_BYTES_CPU = [
    16 * 1024 * 1024,            # 16 MiB
    64 * 1024 * 1024,            # 64 MiB
    256 * 1024 * 1024,           # 256 MiB
    1 * 1024 * 1024 * 1024,      # 1 GiB
]


def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _alloc_bf16(n_bytes: int, device) -> torch.Tensor:
    n_elem = n_bytes // 2
    return torch.empty(n_elem, dtype=torch.bfloat16, device=device)


def _try_alloc(n_bytes: int, n: int, device) -> List[torch.Tensor]:
    """Try to allocate ``n`` buffers of size ``n_bytes`` each. Raises OOM if can't."""
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


# ---------------------------------------------------------------------------
# Cache-hierarchy bandwidth curve (BW-8)
# ---------------------------------------------------------------------------
#
# Methodology:
#   Repeated `copy_` over the *same* buffer pair, swept across working-set
#   sizes from a few KiB up through DRAM. After the first warm-up, hot
#   data lives in whichever cache level the working set fits into; the
#   measured GB/s therefore plateaus at L1, L2, L3 / Infinity Cache, and
#   finally DRAM. Plotted vs working-set size, the curve produces the
#   classic stepwise descent that pins down each cache tier's sustained
#   bandwidth.
#
# Working set definition:
#   `copy_` touches two buffers of size W each. The "working set" reported
#   below is therefore `2 * W`, which is what must fit in cache to stay
#   hot. This matches the standard cache-curve convention used by STREAM /
#   tinymembench and lets the resulting plateaus be compared directly to
#   advertised cache capacities.
#
# CUDA / HIP note:
#   On GPUs the hierarchy is L1 (per-CU) -> L2 -> Infinity Cache (MI300A/X,
#   MI355X). The same buffer-reuse trick exposes those tiers, though the
#   smallest sizes will be launch-bound rather than cache-bound; we
#   annotate that in the output.

CPU_CACHE_SIZES_KIB = [
    4, 8, 16, 32, 64,                   # L1d (~32-64 KiB)
    128, 256, 512, 1024,                # L2 (~512 KiB - 1 MiB per core)
    2 * 1024, 4 * 1024, 8 * 1024,       # L3 lower edge
    16 * 1024, 32 * 1024, 64 * 1024,    # L3 upper edge / shared
    128 * 1024, 256 * 1024,             # DRAM
    512 * 1024, 1024 * 1024,            # DRAM (1 GiB)
]

GPU_CACHE_SIZES_KIB = [
    32, 64, 128, 256,                   # GPU L1 (per-CU/SM)
    512, 1024, 2 * 1024, 4 * 1024,      # GPU L2
    8 * 1024, 16 * 1024, 32 * 1024,     # GPU L2 -> Infinity Cache
    64 * 1024, 128 * 1024, 256 * 1024,  # Infinity Cache (~256 MiB on MI355X)
    512 * 1024, 1024 * 1024,            # HBM
    2 * 1024 * 1024, 4 * 1024 * 1024,   # HBM
]


def _detect_cpu_caches() -> List[dict]:
    """Read `/sys/devices/system/cpu/cpu0/cache/index*/{level,type,size}`.

    Returns a sorted list of cache tiers with their reported size in bytes.
    On environments without `sysfs` (rare WSL, containers), returns []
    and the curve is annotated without level boundaries.
    """
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.is_dir():
        return []
    tiers: List[dict] = []
    for idx_dir in sorted(base.glob("index*")):
        try:
            level = int((idx_dir / "level").read_text().strip())
            ctype = (idx_dir / "type").read_text().strip()
            size_s = (idx_dir / "size").read_text().strip()
        except OSError:
            continue
        m = re.match(r"(\d+)\s*([KMG]?)", size_s)
        if not m:
            continue
        n = int(m.group(1))
        unit = m.group(2)
        size_b = n * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]
        if ctype == "Instruction":
            continue
        tiers.append({"level": level, "type": ctype, "size_bytes": size_b})
    tiers.sort(key=lambda t: (t["level"], t["size_bytes"]))
    return tiers


def _detect_gpu_caches() -> List[dict]:
    """Best-effort GPU cache topology.

    PyTorch only exposes L2 (`l2CacheSize`); L1 / Infinity Cache aren't
    in the device props, so we annotate what we *do* know and fill the
    rest with literature-derived estimates for visualisation. The
    measured curve is the source of truth — these are landmarks only.
    """
    if not torch.cuda.is_available():
        return []
    props = torch.cuda.get_device_properties(0)
    name = (props.name or "").lower()
    tiers: List[dict] = []
    l2 = getattr(props, "L2CacheSize", None) or getattr(props, "l2CacheSize", None) or 0
    if l2:
        tiers.append({"level": 2, "type": "Data", "size_bytes": int(l2)})
    if "mi300" in name or "mi355" in name or "mi325" in name:
        tiers.append({"level": 3, "type": "InfinityCache",
                      "size_bytes": 256 * 1024 * 1024})
    return tiers


def cache_curve(device, warmup: int, iters: int) -> dict:
    """Sweep `copy_` working-set size to expose the cache hierarchy."""
    has_gpu = device.type == "cuda"
    sizes_kib = GPU_CACHE_SIZES_KIB if has_gpu else CPU_CACHE_SIZES_KIB
    rows: List[dict] = []
    for kib in sizes_kib:
        n_bytes = kib * 1024
        try:
            src, dst = _try_alloc(n_bytes, 2, device)
        except RuntimeError as e:
            if _is_oom(e):
                _empty_cache()
                continue
            raise
        src.normal_()
        # Pre-fault dst on CPU so we measure steady-state bandwidth, not
        # first-touch zero-fill cost.
        dst.zero_()

        def fn(src=src, dst=dst):
            dst.copy_(src)

        try:
            r = time_op(f"cache_curve_{n_bytes}", fn,
                        warmup=warmup,
                        iters=iters)
        except RuntimeError as e:
            del src, dst
            _empty_cache()
            if _is_oom(e):
                continue
            raise
        bw = (2 * n_bytes) / (r.median_ms * 1e-3) / 1e9  # 1R + 1W
        rows.append({
            "working_set_bytes": 2 * n_bytes,
            "buffer_bytes":      n_bytes,
            "working_set_kib":   2 * kib,
            "t_ms_median":       r.median_ms,
            "t_ms_p10":          r.p10_ms,
            "t_ms_p90":          r.p90_ms,
            "gb_s":              bw,
        })
        del src, dst
        _empty_cache()
        print(f"[02] cache_curve  {2*kib:>10d} KiB working set -> "
              f"{bw:8.1f} GB/s  (median {r.median_ms:.3f} ms)")

    cpu_tiers = _detect_cpu_caches()
    gpu_tiers = _detect_gpu_caches()
    return {
        "device_type":   device.type,
        "rows":          rows,
        "cpu_caches":    cpu_tiers,
        "gpu_caches":    gpu_tiers,
        "methodology": (
            "copy_ over reused buffer pair; working_set = 2 * buffer_size. "
            "Plateaus identify successive cache tiers; final descent is DRAM "
            "/ HBM steady state."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")

    # Read methodology/config for timing
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    
    cfg_timing = {}
    if Path(args.config).is_file():
        try:
            cfg_timing = json.loads(Path(args.config).read_text()).get("timing", {})
        except Exception:  # noqa: BLE001
            pass
    
    t_cfg = m_cfg.get("timing", cfg_timing)
    
    def _is_set(opt): return any(o in a for a in sys.argv for o in (opt, opt.replace("--iters", "--iterations")))
    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 0)
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 0)

    # If both CLI and JSON are missing, use final hardcoded fallbacks
    if warmup_val == 0 and not _is_set("--warmup"): warmup_val = 5
    if iters_val == 0 and not _is_set("--iters"): iters_val = 20

    b02_cfg = m_cfg.get("bench02", {})
    if has_gpu:
        sizes = b02_cfg.get("sizes_gpu", SIZES_BYTES_GPU)
        warmup, iters = warmup_val, iters_val
    else:
        sizes = b02_cfg.get("sizes_cpu", SIZES_BYTES_CPU)
        # CPU memops are still fast (1 GiB copy ~30 ms on dual-channel DDR5),
        # but 20 iters x 7 ops x 4 sizes = 560 ops; trim warmup so the bench
        # finishes inside ~30s on a laptop. Plateau still emerges from the
        # last two sizes.
        warmup = max(1, min(warmup_val, 2))
        iters = max(5, min(iters_val, 10))

    out_dir = Path(args.out) / "02_hbm_bandwidth"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for n_bytes in sizes:
        for bid, fn, n_buf in BENCHES:
            try:
                row = fn(n_bytes, device, warmup, iters)
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                print(f"[02] OOM at {bid} {n_bytes/1e9:.1f} GB — skipping")
                _empty_cache()
                continue
            row["bench_id"] = bid
            rows.append(row)
            print(f"[02] {bid} {row['op']:>14s} {n_bytes/1e9:6.2f} GB -> {row['gb_s']:8.1f} GB/s")
            _empty_cache()

    write_csv(out_dir / "bandwidth.csv", rows)
    write_json(out_dir / "bandwidth.json", rows)

    print("[02] cache hierarchy curve ...")
    curve = cache_curve(device, warmup=warmup, iters=iters)
    write_json(out_dir / "cache_curve.json", curve)
    if curve["rows"]:
        peak = max(curve["rows"], key=lambda r: r["gb_s"])
        floor = min(curve["rows"][-3:], key=lambda r: r["gb_s"]) \
            if len(curve["rows"]) >= 3 else curve["rows"][-1]
        print(f"[02] cache_curve peak  : {peak['gb_s']:.1f} GB/s "
              f"@ {peak['working_set_kib']} KiB working set")
        print(f"[02] cache_curve floor : {floor['gb_s']:.1f} GB/s "
              f"@ {floor['working_set_kib']} KiB (DRAM / HBM steady state)")

    # Plateau = max of last-two-size sustained for each op.
    plateau = {}
    for op in {r["op"] for r in rows}:
        per_op = sorted([r for r in rows if r["op"] == op], key=lambda r: r["bytes"])
        plateau[op] = max((r["gb_s"] for r in per_op[-2:]), default=0.0)
    bw_roof = max(plateau.values()) if plateau else 0.0
    summary = {
        "device_type": device.type,
        "plateau_gb_s_per_op": plateau,
        "bandwidth_roof_gb_s": bw_roof,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"[02] bandwidth roof = {bw_roof:.1f} GB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
