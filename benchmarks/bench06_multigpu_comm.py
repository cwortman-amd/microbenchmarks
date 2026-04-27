"""Family 6 — Multi-GPU communication (TESTPLAN §12).

Runs under torchrun. Times all-gather and reduce-scatter across a payload
sweep representative of TP activations, then a strong-scaling sweep at fixed
problem size for world ∈ {2, 4, 8} (limited to actually-available ranks).

The PyTorch numbers are intentionally side-by-side comparable to the
`rccl-tests` numbers — same payload sizes, same op kinds, same timing
methodology (device events, multiple iters, distributional stats).

Launch:
    torchrun --nproc_per_node=8 benchmarks/06_multigpu_comm.py --out results/<id>/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


PAYLOAD_BYTES = [
    1 * 1024 * 1024,       # 1 MiB
    8 * 1024 * 1024,       # 8 MiB
    32 * 1024 * 1024,      # 32 MiB
    128 * 1024 * 1024,     # 128 MiB
    512 * 1024 * 1024,     # 512 MiB
    1 * 1024 * 1024 * 1024,  # 1 GiB
]


def _setup() -> tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # rccl on ROCm
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world, device


def _busbw_factor_allgather(world: int) -> float:
    """rccl-tests bus-bandwidth factor for AllGather: (n-1)/n."""
    return (world - 1) / world if world > 0 else 0.0


def _busbw_factor_reducescatter(world: int) -> float:
    return (world - 1) / world if world > 0 else 0.0


def bench_all_gather(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    """Each rank contributes n_bytes/world bytes; output is n_bytes total."""
    chunk = n_bytes // world
    elems = chunk // 2  # bf16
    in_buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()
    out_buf = torch.empty(world * elems, dtype=torch.bfloat16, device=device)
    out_list = list(out_buf.chunk(world))

    def fn():
        dist.all_gather(out_list, in_buf)

    dist.barrier()
    res = time_op(f"all_gather_{n_bytes}", fn, warmup=warmup, iters=iters)
    # rccl-tests algbw = bytes / time; busbw = algbw * (n-1)/n
    algbw = (chunk * world) / (res.median_ms * 1e-3) / 1e9
    busbw = algbw * _busbw_factor_allgather(world)
    return {"op": "all_gather", "world": world, "bytes": n_bytes,
            "t_ms": res.median_ms, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def bench_reduce_scatter(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    elems = (n_bytes // 2)  # bf16, total elements across all input chunks
    in_buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()
    in_list = list(in_buf.chunk(world))
    out_buf = torch.empty(elems // world, dtype=torch.bfloat16, device=device)

    def fn():
        dist.reduce_scatter(out_buf, in_list, op=dist.ReduceOp.SUM)

    dist.barrier()
    res = time_op(f"reduce_scatter_{n_bytes}", fn, warmup=warmup, iters=iters)
    algbw = n_bytes / (res.median_ms * 1e-3) / 1e9
    busbw = algbw * _busbw_factor_reducescatter(world)
    return {"op": "reduce_scatter", "world": world, "bytes": n_bytes,
            "t_ms": res.median_ms, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def bench_all_reduce(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    elems = n_bytes // 2
    buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()

    def fn():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    dist.barrier()
    res = time_op(f"all_reduce_{n_bytes}", fn, warmup=warmup, iters=iters)
    algbw = n_bytes / (res.median_ms * 1e-3) / 1e9
    busbw = algbw * (2 * (world - 1) / world)  # rccl-tests AR busbw
    return {"op": "all_reduce", "world": world, "bytes": n_bytes,
            "t_ms": res.median_ms, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    rank, world, device = _setup()
    rows: List[dict] = []
    for n_bytes in PAYLOAD_BYTES:
        try:
            rows.append(bench_all_gather(world, device, n_bytes, args.warmup, args.iters))
            rows.append(bench_reduce_scatter(world, device, n_bytes, args.warmup, args.iters))
            rows.append(bench_all_reduce(world, device, n_bytes, args.warmup, args.iters))
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(f"[06] {n_bytes/1e6:.0f} MB failed: {e!r}")

    if rank == 0:
        out_dir = Path(args.out) / "06_multigpu_comm"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "comm.csv", rows)
        write_json(out_dir / "comm.json", {"world": world, "rows": rows})
        for r in rows:
            print(f"[06] world={world} {r['op']:16s} {r['bytes']/1e6:7.0f} MB "
                  f"alg={r['algbw_gb_s']:7.1f} bus={r['busbw_gb_s']:7.1f} GB/s")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
