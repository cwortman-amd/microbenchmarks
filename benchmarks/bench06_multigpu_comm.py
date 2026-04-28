"""Family 6 — Multi-GPU communication (TESTPLAN §12).

Runs under torchrun. Times all-gather, reduce-scatter, and all-reduce across
a payload sweep representative of TP activations.

The PyTorch numbers are intentionally side-by-side comparable to the
``rccl-tests`` numbers — same payload sizes, same op kinds, same timing
methodology (device events, multiple iters, distributional stats), with the
reported collective latency reduced to the **median across ranks**.

Backend selection:

  * GPU host (CUDA / HIP available): ``nccl`` (RCCL on ROCm). Each rank pins
    a GPU via ``LOCAL_RANK``.
  * CPU host: ``gloo`` over loopback. The CPU analogue of "multi-GPU" is
    **multi-CCD** within a socket (Infinity Fabric) and **multi-socket**
    across sockets (xGMI / UPI). Each rank is pinned to its CCD or socket
    via ``os.sched_setaffinity`` and ``torch.set_num_threads`` so that the
    measurement reflects the inter-CCD or inter-socket interconnect rather
    than memcpy within a single CCD.

      mode=ccd     : 1 rank per CCD (default; AMD ``die_id`` from sysfs)
      mode=socket  : 1 rank per socket
      mode=split   : carve all online CPUs into N equal-sized groups
                     (fallback for hypervisor / VM topologies that flatten
                     ``die_id``)
      mode=auto    : try ccd, then socket, then split

Launch (GPU):
    torchrun --nproc_per_node=8 benchmarks/bench06_multigpu_comm.py --out results/<id>/

Launch (CPU, multi-CCD):
    torchrun --nproc_per_node=<dies> benchmarks/bench06_multigpu_comm.py \\
      --out results/<id>/ --cpu-topology ccd

Launch (CPU, multi-socket):
    torchrun --nproc_per_node=<sockets> benchmarks/bench06_multigpu_comm.py \\
      --out results/<id>/ --cpu-topology socket
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op
from benchmarks.common.topology import (
    describe_partition,
    detect_cpu_topology,
    partition_cpus,
    pin_to_cpus,
)


PAYLOAD_BYTES_GPU = [
    1 * 1024 * 1024,       # 1 MiB
    8 * 1024 * 1024,       # 8 MiB
    32 * 1024 * 1024,      # 32 MiB
    128 * 1024 * 1024,     # 128 MiB
    512 * 1024 * 1024,     # 512 MiB
    1 * 1024 * 1024 * 1024,  # 1 GiB
]
PAYLOAD_BYTES_CPU = [
    1 * 1024 * 1024,       # 1 MiB
    8 * 1024 * 1024,       # 8 MiB
    32 * 1024 * 1024,      # 32 MiB
    128 * 1024 * 1024,     # 128 MiB
]


def _rank_median_ms(local_ms: float, device: torch.device) -> float:
    """Median of per-rank median timings for the current collective sample."""
    t = torch.tensor([float(local_ms)], dtype=torch.float32, device=device)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    vals = [float(x.item()) for x in gathered]
    return float(statistics.median(vals))


def _setup(cpu_topology_mode: str) -> Tuple[int, int, torch.device, str, Optional[dict]]:
    """Initialise the process group and (on CPU) pin the rank to its CCD/socket.

    Returns (rank, world, device, backend, topology_block) where
    ``topology_block`` is a JSON-serialisable summary on CPU and ``None`` on
    GPU.
    """
    has_gpu = torch.cuda.is_available()
    backend = "nccl" if has_gpu else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    topo_block: Optional[dict] = None
    if has_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        topology = detect_cpu_topology()
        try:
            rank_cpus = partition_cpus(topology, world, mode=cpu_topology_mode)
        except ValueError as e:
            if rank == 0:
                print(f"[06] topology partition failed ({e}); running unpinned")
            rank_cpus = []
        if rank_cpus:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            my_cpus = rank_cpus[local_rank]
            pinned = pin_to_cpus(my_cpus)
            # Use one BLAS thread per physical core in the CCD slice; SMT
            # siblings rarely help for BLAS on AMD parts and tend to hurt
            # bandwidth-bound microkernels.
            tpc = max(1, int(topology.get("threads_per_core") or 1))
            n_threads = max(1, len(my_cpus) // tpc)
            torch.set_num_threads(n_threads)
            if rank == 0:
                resolved = topology.get("_resolved_partition_mode",
                                        cpu_topology_mode)
                print(
                    f"[06] CPU topology mode={resolved} "
                    f"sockets={topology.get('sockets')} "
                    f"dies={topology.get('dies')} "
                    f"cores_per_die={topology.get('cores_per_die')} "
                    f"threads_per_core={tpc} "
                    f"world={world} pinned_per_rank={len(my_cpus)} "
                    f"torch_threads_per_rank={n_threads} "
                    f"sched_setaffinity={'ok' if pinned else 'noop'}"
                )
            topo_block = describe_partition(
                topology,
                mode=topology.get("_resolved_partition_mode", cpu_topology_mode),
                world=world,
                rank_cpus=rank_cpus,
            )
        else:
            # Topology probe failed entirely; degrade to single-thread/rank.
            topo_block = {
                "topology_source": "unavailable",
                "topology_mode": "none",
                "world": world,
                "rank_pinning": [],
            }
    return rank, world, device, backend, topo_block


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
    t_ms_rank_median = _rank_median_ms(res.median_ms, device)
    # rccl-tests algbw = bytes / time; busbw = algbw * (n-1)/n
    algbw = (chunk * world) / (t_ms_rank_median * 1e-3) / 1e9
    busbw = algbw * _busbw_factor_allgather(world)
    return {"op": "all_gather", "world": world, "bytes": n_bytes,
            "t_ms": t_ms_rank_median, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def bench_reduce_scatter(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    elems = (n_bytes // 2)  # bf16, total elements across all input chunks
    in_buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()
    in_list = list(in_buf.chunk(world))
    out_buf = torch.empty(elems // world, dtype=torch.bfloat16, device=device)

    def fn():
        dist.reduce_scatter(out_buf, in_list, op=dist.ReduceOp.SUM)

    dist.barrier()
    res = time_op(f"reduce_scatter_{n_bytes}", fn, warmup=warmup, iters=iters)
    t_ms_rank_median = _rank_median_ms(res.median_ms, device)
    algbw = n_bytes / (t_ms_rank_median * 1e-3) / 1e9
    busbw = algbw * _busbw_factor_reducescatter(world)
    return {"op": "reduce_scatter", "world": world, "bytes": n_bytes,
            "t_ms": t_ms_rank_median, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def _busbw_factor_alltoall(world: int) -> float:
    """rccl-tests bus-bandwidth factor for AllToAll: (n-1)/n.

    Each rank sends `n_bytes/world` to every other rank, so total bytes
    on the wire = `n_bytes * (world-1)/world` (the local 1/world stays
    on-rank). Same convention rccl-tests / nccl-tests use for
    cross-comparison.
    """
    return (world - 1) / world if world > 0 else 0.0


def bench_all_to_all(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    """All-to-all (A2A) collective.

    Each rank's input tensor of `n_bytes` is split into `world` chunks
    of `n_bytes/world`; chunk `j` from rank `i` goes to rank `j`'s slot
    `i`. The output is the same shape as the input. Same payload
    convention as the other ops so the GB/s columns are
    apples-to-apples comparable.

    A2A is the alternative TP topology the source PDF flags as future
    work — it replaces the AG+RS pair with a single round of
    all-to-all, which can be cheaper on dense full-mesh interconnects
    (xGMI 4.0 / NVLink 4) and more expensive on hierarchical fabrics.
    Measuring it head-to-head against AR / AG / RS makes the trade-off
    visible.
    """
    elems = n_bytes // 2  # bf16
    if elems % world:
        elems -= elems % world
    in_buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()
    out_buf = torch.empty_like(in_buf)

    def fn():
        dist.all_to_all_single(out_buf, in_buf)

    dist.barrier()
    res = time_op(f"all_to_all_{n_bytes}", fn, warmup=warmup, iters=iters)
    t_ms_rank_median = _rank_median_ms(res.median_ms, device)
    actual_bytes = elems * 2
    algbw = actual_bytes / (t_ms_rank_median * 1e-3) / 1e9
    busbw = algbw * _busbw_factor_alltoall(world)
    return {"op": "all_to_all", "world": world, "bytes": actual_bytes,
            "t_ms": t_ms_rank_median, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def bench_all_reduce(world: int, device, n_bytes: int, warmup: int, iters: int) -> dict:
    elems = n_bytes // 2
    buf = torch.empty(elems, dtype=torch.bfloat16, device=device).normal_()

    def fn():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    dist.barrier()
    res = time_op(f"all_reduce_{n_bytes}", fn, warmup=warmup, iters=iters)
    t_ms_rank_median = _rank_median_ms(res.median_ms, device)
    algbw = n_bytes / (t_ms_rank_median * 1e-3) / 1e9
    busbw = algbw * (2 * (world - 1) / world)  # rccl-tests AR busbw
    return {"op": "all_reduce", "world": world, "bytes": n_bytes,
            "t_ms": t_ms_rank_median, "algbw_gb_s": algbw, "busbw_gb_s": busbw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--cpu-topology",
        choices=("auto", "ccd", "socket", "split"),
        default=os.environ.get("MICROBENCH_CPU_TOPOLOGY", "auto"),
        help=(
            "On CPU hosts, how to map ranks to hardware. "
            "'ccd' = one rank per AMD CCD (die_id); "
            "'socket' = one rank per physical socket; "
            "'split' = ignore topology and slice all online CPUs evenly; "
            "'auto' = ccd → socket → split."
        ),
    )
    args = ap.parse_args()

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
    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 5)
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 20)

    rank, world, device, backend, topo_block = _setup(args.cpu_topology)
    has_gpu = device.type == "cuda"
    payloads = PAYLOAD_BYTES_GPU if has_gpu else PAYLOAD_BYTES_CPU
    if has_gpu:
        warmup, iters = warmup_val, iters_val
    else:
        warmup = max(1, min(warmup_val, 2))
        iters = max(3, min(iters_val, 5))

    if rank == 0:
        topo_label = ""
        if topo_block:
            topo_label = (f" topology={topo_block.get('topology_mode')}"
                          f"({topo_block.get('dies') or '?'} dies / "
                          f"{topo_block.get('sockets') or '?'} sockets)")
        print(f"[06] backend={backend} world={world} device={device.type}"
              f"{topo_label} payloads={[p//(1024*1024) for p in payloads]} MiB")

    rows: List[dict] = []
    for n_bytes in payloads:
        try:
            rows.append(bench_all_gather(world, device, n_bytes, warmup, iters))
            rows.append(bench_reduce_scatter(world, device, n_bytes, warmup, iters))
            rows.append(bench_all_reduce(world, device, n_bytes, warmup, iters))
            rows.append(bench_all_to_all(world, device, n_bytes, warmup, iters))
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                print(f"[06] {n_bytes/1e6:.0f} MB failed: {e!r}")

    if rank == 0:
        out_dir = Path(args.out) / "06_multigpu_comm"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "comm.csv", rows)
        payload = {
            "backend": backend,
            "device_type": device.type,
            "world": world,
            "rows": rows,
        }
        if topo_block is not None:
            payload["cpu_topology"] = topo_block
        write_json(out_dir / "comm.json", payload)
        for r in rows:
            print(f"[06] world={world} {r['op']:16s} {r['bytes']/1e6:7.0f} MB "
                  f"alg={r['algbw_gb_s']:7.1f} bus={r['busbw_gb_s']:7.1f} GB/s")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
