"""Family 13 — MM + AllReduce overlap: Iris fused vs signaling vs RCCL.

This benchmark targets the **GEMM + AllReduce** pattern (row-parallel TP second
GEMM), which is the dominant collective in the Wan2.2 / Odyssey proxy and the
shared target of two independent overlap strategies we want to compare head to
head on MI355X:

  * ``unfused_mm_ar`` — the RCCL baseline: ``Y = all_reduce(A_local @ B_local)``
    run serially (matmul, then a blocking ``dist.all_reduce``). This is the bar
    every overlap method must beat.

  * ``pipelined_mm_ar`` — **Track B**, a FlashOverlap-style signaling overlap
    expressed in portable PyTorch: split the GEMM into row-chunks and, as each
    chunk's output is produced, fire its ``all_reduce`` on a separate
    communication stream so it overlaps the next chunk's matmul. Communication-
    agnostic (plain RCCL), GEMM stays per-chunk dense. The cross-stream event is
    the "signal". (FlashOverlap, Hong et al., arXiv:2504.19519.)

  * ``iris_mm_ar`` — **Track A**, AMD's upstream fused kernel
    ``iris.ops.matmul_all_reduce`` (monolithic tile-level fusion, the Flux /
    Iris approach). Probed at runtime; skipped cleanly if Iris isn't installed.

All three are scored with the same Flux ECT / overlap-efficiency metric
(``benchmarks.common.overlap``) against the *same* RCCL baseline, so a negative
overlap efficiency immediately flags a method that is slower than not
overlapping at all.

Sharding convention (row-parallel / Megatron RowParallelLinear): the contraction
dim ``K`` is sharded across ranks. Each rank holds ``A_local[M, K/world]`` and
``B_local[K/world, N]``, computes a partial ``[M, N]``, and the all-reduce sums
the partials. ``K`` is padded UP to a multiple of ``world`` (never truncated).
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.overlap import summarize_overlap
from benchmarks.common.shapes import pad_to_multiple, resolve_shapes
from benchmarks.common.timing import time_op

SHAPES = [s.as_tuple() for s in resolve_shapes()]

_DTYPE = torch.bfloat16
_DTYPE_BYTES = 2


def _is_distributed_env() -> bool:
    return all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"))


def _setup_distributed() -> Tuple[int, int, torch.device, str, bool]:
    has_gpu = torch.cuda.is_available()
    backend = "nccl" if has_gpu else "gloo"
    if has_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    distributed = False
    if _is_distributed_env():
        if not dist.is_initialized():
            if has_gpu:
                dist.init_process_group(backend=backend, device_id=device)
            else:
                dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        distributed = True
    else:
        rank, world = 0, 1
    return rank, world, device, backend, distributed


def _ar_wire_bytes(world: int, M: int, N: int) -> int:
    """Ring all-reduce moves ~2(world-1)/world * payload per rank."""
    payload = M * N * _DTYPE_BYTES
    return int(2 * (world - 1) / world * payload)


def _shard_K(K: int, world: int) -> Tuple[int, int]:
    K_pad = pad_to_multiple(K, world)
    return K_pad, K_pad // world


def _bench_unfused_mm_ar(world: int, device, M: int, K: int, N: int,
                         warmup: int, iters: int) -> Dict:
    """RCCL baseline: matmul then blocking all_reduce, plus standalone splits."""
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=_DTYPE, device=device).normal_()
    B = torch.empty(K_local, N, dtype=_DTYPE, device=device).normal_()
    Y_pre = torch.empty(M, N, dtype=_DTYPE, device=device).normal_()

    def fn_mm():
        torch.matmul(A, B)

    def fn_ar():
        dist.all_reduce(Y_pre, op=dist.ReduceOp.SUM)

    def fn_full():
        Y = torch.matmul(A, B)
        dist.all_reduce(Y, op=dist.ReduceOp.SUM)

    dist.barrier()
    res_full = time_op(f"unfused_mm_ar_full_{M}_{K_pad}_{N}", fn_full, warmup=warmup, iters=iters)
    res_mm = time_op(f"unfused_mm_ar_mm_{M}_{K_pad}_{N}", fn_mm, warmup=warmup, iters=iters)
    res_ar = time_op(f"unfused_mm_ar_ar_{M}_{K_pad}_{N}", fn_ar, warmup=warmup, iters=iters)

    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N)
    return {
        "op": "unfused_mm_ar",
        "world": world, "M": M, "K": K_pad, "K_local": K_local, "N": N,
        "t_ms": res_full.median_ms,
        "t_ms_mm": res_mm.median_ms,
        "t_ms_ar": res_ar.median_ms,
        "tflops": flops / (res_full.median_ms * 1e-3) / 1e12,
        "ar_gb_s": wire / (res_full.median_ms * 1e-3) / 1e9,
        "ar_wire_bytes": wire,
    }


def _bench_pipelined_mm_ar(world: int, device, M: int, K: int, N: int,
                           warmup: int, iters: int, n_chunks: int) -> Dict:
    """Track B: chunked row-wise GEMM with per-chunk all_reduce on a comm stream.

    The output is split into ``n_chunks`` row-blocks. Chunk i's matmul runs on
    the compute stream; once it completes (signalled by a CUDA event) chunk i's
    all_reduce is issued on the comm stream, overlapping chunk i+1's matmul.
    """
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=_DTYPE, device=device).normal_()
    B = torch.empty(K_local, N, dtype=_DTYPE, device=device).normal_()
    Y = torch.empty(M, N, dtype=_DTYPE, device=device)

    n_chunks = max(1, min(n_chunks, M))
    bounds: List[Tuple[int, int]] = []
    step = (M + n_chunks - 1) // n_chunks
    for r0 in range(0, M, step):
        bounds.append((r0, min(r0 + step, M)))

    s_compute = torch.cuda.Stream()
    s_comm = torch.cuda.Stream()
    evts = [torch.cuda.Event() for _ in bounds]

    def fn():
        cur = torch.cuda.current_stream()
        # Make the compute stream start from a consistent point each iter.
        s_compute.wait_stream(cur)
        for i, (r0, r1) in enumerate(bounds):
            with torch.cuda.stream(s_compute):
                torch.matmul(A[r0:r1], B, out=Y[r0:r1])
                evts[i].record(s_compute)
            with torch.cuda.stream(s_comm):
                s_comm.wait_event(evts[i])
                dist.all_reduce(Y[r0:r1], op=dist.ReduceOp.SUM)
        cur.wait_stream(s_compute)
        cur.wait_stream(s_comm)

    dist.barrier()
    res = time_op(f"pipelined_mm_ar_{M}_{K_pad}_{N}_c{n_chunks}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N)
    return {
        "op": "pipelined_mm_ar",
        "world": world, "M": M, "K": K_pad, "K_local": K_local, "N": N,
        "n_chunks": len(bounds),
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
    }


def _make_iris(heap_bytes: int):
    """Create an Iris instance, or return ``(None, reason)`` if unavailable."""
    try:
        import iris  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return None, f"iris import failed: {e!r}"
    try:
        shmem = iris.iris(heap_bytes)
    except TypeError:
        try:
            shmem = iris.iris()
        except Exception as e:  # noqa: BLE001
            return None, f"iris.iris() init failed: {e!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"iris.iris(heap) init failed: {e!r}"
    if not hasattr(shmem, "ops") or not hasattr(shmem.ops, "matmul_all_reduce"):
        return None, "iris instance has no ops.matmul_all_reduce"
    return shmem, "iris.ops.matmul_all_reduce"


def _bench_iris_mm_ar(shmem, world: int, device, M: int, K: int, N: int,
                      warmup: int, iters: int) -> Dict:
    """Track A: upstream fused ``iris.ops.matmul_all_reduce(out, A, B)``."""
    K_pad, K_local = _shard_K(K, world)
    A = shmem.randn((M, K_local), dtype=_DTYPE)
    B = shmem.randn((K_local, N), dtype=_DTYPE)
    out = shmem.zeros((M, N), dtype=_DTYPE)

    def fn():
        shmem.ops.matmul_all_reduce(out, A, B)

    shmem.barrier() if hasattr(shmem, "barrier") else dist.barrier()
    res = time_op(f"iris_mm_ar_{M}_{K_pad}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N)
    return {
        "op": "iris_mm_ar",
        "world": world, "M": M, "K": K_pad, "K_local": K_local, "N": N,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
    }


def _write_skip(out_dir: Path, reason: str, *, world: int, backend: str, device_type: str) -> None:
    payload = {
        "available": False, "reason": reason, "world": world,
        "backend": backend, "device_type": device_type, "rows": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "overlap.json", payload)
    write_csv(out_dir / "overlap.csv", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ar-chunks", type=int, default=4,
                    help="Row-chunks for the pipelined (FlashOverlap-style) overlap.")
    ap.add_argument("--iris-heap-gb", type=float, default=8.0,
                    help="Iris symmetric-heap size in GiB for Track A.")
    ap.add_argument("--shapes", type=str, default=None)
    ap.add_argument("--shape-set", type=str, default=None)
    ap.add_argument("--shapes-file", type=str, default=None)
    args = ap.parse_args()

    m_cfg = {}
    if Path(args.methodology).is_file():
        try:
            m_cfg = json.loads(Path(args.methodology).read_text())
        except Exception:  # noqa: BLE001
            pass
    t_cfg = m_cfg.get("timing", {})

    def _is_set(opt): return any(opt in a for a in sys.argv)
    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 0) or 5
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 0) or 20

    rank, world, device, backend, distributed = _setup_distributed()
    has_gpu = device.type == "cuda"
    out_dir = Path(args.out) / "13_iris_overlap"

    def _teardown() -> None:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    if not has_gpu:
        if rank == 0:
            _write_skip(out_dir, "CPU host: MM+AllReduce overlap needs GPUs.",
                        world=world, backend=backend, device_type=device.type)
        _teardown()
        return 0
    if world < 2:
        if rank == 0:
            _write_skip(out_dir, f"world={world}: MM+AllReduce overlap needs world>=2.",
                        world=world, backend=backend, device_type=device.type)
        _teardown()
        return 0

    try:
        shapes = [s.as_tuple() for s in
                  resolve_shapes(args.shapes, args.shape_set, path=args.shapes_file)]
    except ValueError as e:
        raise SystemExit(f"shape resolution failed: {e}")

    # Track A backend (probed once; may be unavailable).
    shmem, iris_label = _make_iris(int(args.iris_heap_gb * (1 << 30)))
    if rank == 0:
        print(f"[13] Track A (iris): {iris_label}")
        print(f"[13] Track B (pipelined): {args.ar_chunks} chunks")

    rows: List[Dict] = []
    for (M, K, N) in shapes:
        plan = [("unfused_mm_ar", None), ("pipelined_mm_ar", None)]
        if shmem is not None:
            plan.append(("iris_mm_ar", None))
        for label, _ in plan:
            try:
                if label == "unfused_mm_ar":
                    row = _bench_unfused_mm_ar(world, device, M, K, N, warmup_val, iters_val)
                elif label == "pipelined_mm_ar":
                    row = _bench_pipelined_mm_ar(world, device, M, K, N, warmup_val,
                                                 iters_val, args.ar_chunks)
                else:
                    row = _bench_iris_mm_ar(shmem, world, device, M, K, N, warmup_val, iters_val)
                rows.append(row)
                if rank == 0:
                    print(f"[13] {label:16s} M={M:6d} K={K:5d} N={N:5d} "
                          f"t={row['t_ms']:7.2f} ms tflops={row['tflops']:7.1f}")
            except Exception as e:  # noqa: BLE001
                rows.append({"op": label, "world": world, "M": M, "K": K, "N": N,
                             "error": repr(e)})
                if rank == 0:
                    print(f"[13] {label:16s} M={M:6d} K={K:5d} N={N:5d} ERROR: {e!r}")

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        overlap = summarize_overlap(rows)
        payload = {
            "available": True,
            "world": world, "backend": backend, "device_type": device.type,
            "iris_backend": iris_label, "ar_chunks": args.ar_chunks,
            "rows": rows, "overlap_summary": overlap,
        }
        write_json(out_dir / "overlap.json", payload)
        write_csv(out_dir / "overlap.csv", rows)
        write_csv(out_dir / "overlap_summary.csv", overlap["rows"])

        for s in overlap["rows"]:
            eff = s.get("overlap_efficiency")
            spd = s.get("speedup_vs_unfused")
            ceil = s.get("ceiling_speedup")
            eff_str = f"{eff * 100:6.1f}%" if isinstance(eff, (int, float)) else "    n/a"
            spd_str = f"{spd:5.2f}x" if isinstance(spd, (int, float)) else "  n/a"
            ceil_str = f"{ceil:5.2f}x" if isinstance(ceil, (int, float)) else "  n/a"
            print(f"[13] overlap {s['pattern']:16s} M={s['M']:6d} N={s['N']:5d}  "
                  f"eff={eff_str}  speedup={spd_str}  ceiling={ceil_str}")
        for name, model in overlap["cost_model"].items():
            bw = model.get("bandwidth_gb_s")
            bw_str = f"{bw:6.1f} GB/s" if isinstance(bw, (int, float)) else "n/a"
            print(f"[13] cost-model {name:6s} comm ~= {model['alpha_ms']:.3f} ms + "
                  f"bytes/({bw_str})  (r2={model['r2']:.3f}, n={model['n_points']})")

    _teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
