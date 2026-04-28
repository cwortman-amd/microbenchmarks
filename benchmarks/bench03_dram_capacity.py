"""Family 3 — DRAM capacity (TESTPLAN §7).

Determines practical allocatable device memory via binary search. The bench
is named generically (DRAM) because the same accounting applies whether the
device is a GPU (HBM3e on MI355X) or a CPU host (system DDR). Reports:

  - max single contiguous bf16 allocation
  - effective utilization vs the device's nominal capacity
  - fragmentation sensitivity (contig vs many-chunk)
  - dtype scaling (bf16 vs fp16 should be ~equal)
  - allocator state (torch.cuda.memory_stats dump on GPU; psutil dump on CPU)

The MI355X HBM3e nominal is 288 GB. On CPU we use ``psutil`` to read total /
available system RAM at probe time so the same JSON schema carries a
meaningful "nominal" baseline for each host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from benchmarks.common.flop_accounting import WorkloadConfig, model_param_bytes
from benchmarks.common.io import write_json


GPU_NOMINAL_BYTES = 288 * 1024 ** 3  # MI355X HBM3e spec sheet


def _is_oom(exc: BaseException) -> bool:
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _try_alloc_contig(n_bytes: int, device, dtype) -> bool:
    n_elem = n_bytes // dtype.itemsize if hasattr(dtype, "itemsize") else n_bytes // 2
    try:
        t = torch.empty(n_elem, dtype=dtype, device=device)
        # Force a real residency probe by writing one element.
        t.fill_(0)
        _sync()
        del t
        _empty_cache()
        return True
    except RuntimeError as e:
        if _is_oom(e):
            _empty_cache()
            return False
        raise
    except MemoryError:
        _empty_cache()
        return False


def binary_search_max_contig(device, dtype, lo: int, hi: int,
                             tol_bytes: int = 64 * 1024 * 1024,
                             absolute_cap: int | None = None) -> int:
    """Binary search the largest single contiguous allocation that succeeds."""
    cap = absolute_cap if absolute_cap is not None else GPU_NOMINAL_BYTES * 2
    # Ensure hi fails before search.
    while _try_alloc_contig(hi, device, dtype):
        lo = hi
        hi = int(hi * 1.5)
        if hi > cap:
            return lo  # absurd, return last good
    while hi - lo > tol_bytes:
        mid = (lo + hi) // 2
        if _try_alloc_contig(mid, device, dtype):
            lo = mid
        else:
            hi = mid
    return lo


def headroom_after_model_load(
    device,
    weight_bytes: int,
    *,
    chunk_bytes: int = 512 * 1024 * 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Allocate ``weight_bytes`` of ``dtype`` tensors then report free residual.

    Models the post-load steady state: persistent weight tensors held
    resident on the device, with the rest of HBM / DRAM available for
    activations, KV caches, and gradients. The probe allocates
    ``weight_bytes`` as a sequence of ``chunk_bytes`` tensors (default
    512 MiB) so we can keep the actual allocations resident for the
    free-memory query while staying inside any allocator-imposed
    largest-block limit.

    Returns a dict with:
      - ``model_bytes``       — bytes the probe successfully allocated
      - ``model_target_bytes`` — bytes requested
      - ``device_free_after_load_bytes`` — free memory while weights resident
      - ``residual_capacity_bytes``       — same number, more user-friendly name
      - ``residual_fraction``            — residual_capacity / nominal
      - ``loaded`` — True iff the full target allocated; False = host can't
        hold this model and the residual figure is computed from a partial
        allocation (we still report it to surface the deficit).

    On exit the probe drops every allocation and empties the cache so the
    host's measured state is unchanged for the rest of the benchmark.
    """
    chunks: List[torch.Tensor] = []
    n_full = weight_bytes // chunk_bytes
    tail_bytes = weight_bytes - n_full * chunk_bytes
    allocated = 0
    loaded = True
    last_error: str = ""

    item_bytes = dtype.itemsize if hasattr(dtype, "itemsize") else 2
    try:
        for _ in range(n_full):
            t = torch.empty(chunk_bytes // item_bytes,
                            dtype=dtype, device=device)
            t.fill_(0)
            chunks.append(t)
            allocated += chunk_bytes
        if tail_bytes:
            t = torch.empty(tail_bytes // item_bytes,
                            dtype=dtype, device=device)
            t.fill_(0)
            chunks.append(t)
            allocated += tail_bytes
        _sync()
    except (RuntimeError, MemoryError) as e:
        if not _is_oom(e):
            for t in chunks:
                del t
            chunks.clear()
            _empty_cache()
            raise
        loaded = False
        last_error = f"{type(e).__name__}: {str(e)[:200]}"

    if device.type == "cuda":
        free_after, total_after = torch.cuda.mem_get_info(device)
        free_after = int(free_after)
        total_after = int(total_after)
    else:
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            free_after = int(vm.available)
            total_after = int(vm.total)
        except Exception:  # noqa: BLE001
            free_after = 0
            total_after = 0

    for t in chunks:
        del t
    chunks.clear()
    _empty_cache()
    _sync()

    return {
        "loaded":                          bool(loaded),
        "model_target_bytes":              int(weight_bytes),
        "model_bytes":                     int(allocated),
        "chunk_bytes":                     int(chunk_bytes),
        "device_total_bytes":              int(total_after),
        "device_free_after_load_bytes":    int(free_after),
        "residual_capacity_bytes":         int(free_after),
        "residual_capacity_gib":           free_after / 1024 ** 3,
        "model_bytes_gib":                 allocated / 1024 ** 3,
        "residual_fraction":               (free_after / total_after) if total_after else 0.0,
        "deficit_bytes":                   max(0, int(weight_bytes - allocated)),
        "last_error":                      last_error,
    }


def fragmentation_probe(device, target_bytes: int, chunk_bytes: int) -> dict:
    """Try to allocate target_bytes as many chunks of chunk_bytes; report success."""
    n_chunks = target_bytes // chunk_bytes if chunk_bytes else 0
    chunks = []
    allocated = 0
    try:
        for _ in range(n_chunks):
            t = torch.empty(chunk_bytes // 2, dtype=torch.bfloat16, device=device)
            t.fill_(0)
            chunks.append(t)
            allocated += chunk_bytes
    except RuntimeError as e:
        if not _is_oom(e):
            raise
    except MemoryError:
        pass
    finally:
        del chunks
        _empty_cache()
    return {"target_bytes": target_bytes, "chunk_bytes": chunk_bytes,
            "allocated_bytes": allocated,
            "frac_of_target": allocated / target_bytes if target_bytes else 0.0}


def _device_setup() -> Tuple[torch.device, str, int, int, dict]:
    """Returns (device, label, total_bytes, free_bytes, mem_stats_snapshot)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        stats = {k: v for k, v in torch.cuda.memory_stats(device).items()
                 if isinstance(v, (int, float))}
        return device, torch.cuda.get_device_name(device), int(total), int(free), stats
    # CPU
    device = torch.device("cpu")
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        total = int(vm.total)
        free = int(vm.available)
        stats = {k: int(v) for k, v in vm._asdict().items()}
    except Exception:  # noqa: BLE001
        total = free = 0
        stats = {}
    return device, "cpu", total, free, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frag-chunk-mb", type=int, default=512)
    ap.add_argument("--cpu-headroom-fraction", type=float, default=0.5,
                    help="Fraction of free system RAM to probe on CPU. Keeps "
                         "the host responsive (default 0.5).")
    ap.add_argument("--measure-headroom", action="store_true",
                    help="After the contig / fragmentation probes, allocate "
                         "the full bf16 model weight tensor (sized analytically "
                         "from --config) and report residual free capacity. "
                         "Answers TESTPLAN §16.3's headroom-after-model-load.")
    ap.add_argument("--config", type=Path, default=None,
                    help="Workload config JSON. Required when "
                         "--measure-headroom is set; the parameter count "
                         "comes from this config.")
    ap.add_argument("--headroom-chunk-mb", type=int, default=512,
                    help="Chunk size for the headroom probe's weight "
                         "allocation. Larger = fewer allocator bookkeeping "
                         "rows; smaller = closer to the largest-block "
                         "limit on fragmented hosts.")
    args = ap.parse_args()

    device, dev_label, total, free, mem_stats = _device_setup()
    has_gpu = device.type == "cuda"

    out_dir = Path(args.out) / "03_dram_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_gpu:
        nominal = GPU_NOMINAL_BYTES
        cap = total or GPU_NOMINAL_BYTES * 2
        lo = 8 * 1024 ** 3
        hi = 64 * 1024 ** 3
    else:
        # On CPU, "nominal" is total system RAM at probe time. Cap probing at
        # a fraction of currently-free RAM so the OS doesn't OOM-kill the
        # process. The fragmentation probe re-uses the same headroom.
        nominal = total or 1
        headroom = int(free * max(0.05, min(args.cpu_headroom_fraction, 0.95))) if free else 1 * 1024 ** 3
        cap = max(headroom, 64 * 1024 ** 2)
        lo = min(64 * 1024 * 1024, cap // 4)
        hi = max(lo * 2, min(cap, 4 * 1024 ** 3))

    print(f"[03] device={dev_label} total={total/1024**3:.1f} GiB "
          f"free={free/1024**3:.1f} GiB nominal={nominal/1024**3:.1f} GiB "
          f"probe_range=[{lo/1024**3:.2f},{hi/1024**3:.2f}] GiB cap={cap/1024**3:.2f} GiB")

    print("[03] binary search bf16 contiguous max ...")
    bf16_max = binary_search_max_contig(device, torch.bfloat16, lo=lo, hi=hi,
                                        absolute_cap=cap)
    print(f"[03]   bf16: {bf16_max/1024**3:.3f} GiB")
    print("[03] binary search fp16 contiguous max ...")
    fp16_max = binary_search_max_contig(device, torch.float16, lo=lo, hi=hi,
                                        absolute_cap=cap)
    print(f"[03]   fp16: {fp16_max/1024**3:.3f} GiB")

    # Frag probe chunk size: scale down on tiny CPU targets so we get >= a few chunks.
    chunk_mb = args.frag_chunk_mb
    if not has_gpu and bf16_max < args.frag_chunk_mb * 1024 * 1024 * 4:
        chunk_mb = max(8, bf16_max // (1024 * 1024) // 8)
    print(f"[03] fragmentation probe (chunk={chunk_mb} MB) ...")
    frag = fragmentation_probe(device, target_bytes=bf16_max,
                               chunk_bytes=chunk_mb * 1024 * 1024)
    print(f"[03]   chunked allocated {frag['allocated_bytes']/1024**3:.3f} GiB "
          f"({frag['frac_of_target']*100:.1f}% of contig)")

    summary = {
        "device_type": device.type,
        "device_label": dev_label,
        "device_total_bytes": total,
        "device_free_bytes_pre_test": free,
        "nominal_bytes": nominal,
        "max_alloc_bf16_bytes": bf16_max,
        "max_alloc_fp16_bytes": fp16_max,
        "max_alloc_bf16_gib": bf16_max / 1024 ** 3,
        "eff_util_fraction_bf16": (bf16_max / nominal) if nominal else 0.0,
        "eff_util_fraction_vs_total": (bf16_max / total) if total else 0.0,
        "frag_sensitivity_ratio": frag["frac_of_target"],
        "frag_chunk_mb": chunk_mb,
        "memory_stats": mem_stats,
    }

    if args.measure_headroom:
        if not args.config:
            print("[03] --measure-headroom requires --config; skipping headroom probe.")
        else:
            cfg_json = json.loads(args.config.read_text())
            cfg = WorkloadConfig.from_json(cfg_json)
            mb = model_param_bytes(cfg)
            requested = mb["total_weight_bytes"]
            # On CPU hosts the full 14B bf16 model is ~28 GiB and almost
            # always exceeds the available RAM (and would OOM-kill the
            # process). Cap the probe at the same headroom as the contig
            # search so we stay within OS-safe bounds; the deficit field
            # will surface that we can't hold the full weights here.
            if has_gpu:
                target = requested
            else:
                target = min(requested, cap)
            print(f"[03] headroom probe: model={mb['params']/1e9:.2f}B params "
                  f"({requested/1024**3:.2f} GiB bf16, target probe="
                  f"{target/1024**3:.2f} GiB)")
            headroom = headroom_after_model_load(
                device,
                weight_bytes=target,
                chunk_bytes=args.headroom_chunk_mb * 1024 * 1024,
                dtype=torch.bfloat16,
            )
            headroom["model_params"] = mb["params"]
            headroom["model_params_per_block"] = mb["params_per_block"]
            headroom["model_target_full_bytes"] = requested
            headroom["model_target_full_gib"] = requested / 1024 ** 3
            headroom["probe_capped"] = target < requested
            summary["headroom"] = headroom
            print(f"[03]   loaded={headroom['loaded']} model="
                  f"{headroom['model_bytes_gib']:.2f} GiB residual="
                  f"{headroom['residual_capacity_gib']:.2f} GiB "
                  f"({headroom['residual_fraction']*100:.1f}% of total)")
            if not headroom["loaded"]:
                print(f"[03]   deficit: "
                      f"{headroom['deficit_bytes']/1024**3:.2f} GiB short — "
                      f"{headroom['last_error']}")

    write_json(out_dir / "summary.json", summary)
    print(f"[03] bf16 effective util vs nominal: {summary['eff_util_fraction_bf16']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
