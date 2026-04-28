"""Family 10 — PyTorch Symmetric Memory fused collective+GEMM probe.

SymmMem-only benchmark for:
  * torch.ops.symm_mem.fused_all_gather_matmul
  * torch.ops.symm_mem.fused_matmul_reduce_scatter

Before measuring throughput, this benchmark validates functional support on
the current ROCm/PyTorch stack by checking:
  1) op registration in torch.ops.symm_mem
  2) enable_symm_mem_for_group(group_name)
  3) fused op execution without runtime error
  4) fused output correctness vs fallback helper outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.timing import time_op


SHAPES = [
    (4096, 4096, 4096),
    (8192, 4096, 4096),
    (8192, 8192, 4096),
    (16384, 4096, 4096),
]


def _is_distributed_env() -> bool:
    return all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"))


def _setup_distributed() -> Tuple[int, int, torch.device, str, bool]:
    has_gpu = torch.cuda.is_available()
    backend = "nccl" if has_gpu else "gloo"
    distributed = False
    if _is_distributed_env():
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        distributed = True
    else:
        rank = 0
        world = 1
    if has_gpu:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world, device, backend, distributed


def _write_skip(out_dir: Path, reason: str, *, world: int, backend: str,
                device_type: str) -> None:
    payload = {
        "available": False,
        "reason": reason,
        "world": world,
        "backend": backend,
        "device_type": device_type,
        "rows": [],
        "_note": "SymmMem fused ops unavailable or failed functional probe.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fused.json", payload)
    write_csv(out_dir / "fused.csv", [])


def _probe_symm_mem(world: int, device) -> Tuple[Optional[Dict], str]:
    try:
        import torch.distributed._symmetric_memory as symm_mem  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, f"symm_mem import failed: {e!r}"

    has_ns = hasattr(torch.ops, "symm_mem")
    has_agmm = has_ns and hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")
    has_mmrs = has_ns and hasattr(torch.ops.symm_mem, "fused_matmul_reduce_scatter")
    if not (has_ns and has_agmm and has_mmrs):
        return None, (
            f"torch.ops.symm_mem present={has_ns} "
            f"ag_mm={has_agmm} mm_rs={has_mmrs}"
        )

    group = dist.group.WORLD
    group_name = getattr(group, "group_name", None)
    if not group_name:
        return None, "WORLD group has no group_name"

    try:
        symm_mem.enable_symm_mem_for_group(group_name)
    except Exception as e:  # noqa: BLE001
        return None, f"enable_symm_mem_for_group failed: {e!r}"

    dtype = torch.bfloat16
    batch, M, K, N = 2, max(1024, world * 128), 1024, 512
    M = (M // world) * world
    if M <= 0:
        return None, f"invalid probe shape with world={world}"

    try:
        A_shard = torch.randn(batch, M // world, K, device=device, dtype=dtype)
        if hasattr(symm_mem, "restride_A_shard_for_fused_all_gather_matmul"):
            A_shard = symm_mem.restride_A_shard_for_fused_all_gather_matmul(A_shard, dim=1)
        Bs = [torch.randn(K, N, device=device, dtype=dtype)]
        ag_ref, mm_ref = symm_mem._fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim=1, group_name=group_name
        )
        ag_out, mm_out = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, Bs, gather_dim=1, group_name=group_name
        )
        torch.testing.assert_close(ag_ref, ag_out)
        torch.testing.assert_close(mm_ref[0], mm_out[0])
    except Exception as e:  # noqa: BLE001
        return None, f"fused_all_gather_matmul probe failed: {e!r}"

    try:
        A = torch.randn(batch, M, K, device=device, dtype=dtype)
        if hasattr(symm_mem, "restride_A_for_fused_matmul_reduce_scatter"):
            A = symm_mem.restride_A_for_fused_matmul_reduce_scatter(A, dim=1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        rs_ref = symm_mem._fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=1, group_name=group_name
        )
        rs_out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, "avg", scatter_dim=1, group_name=group_name
        )
        torch.testing.assert_close(rs_ref, rs_out)
    except Exception as e:  # noqa: BLE001
        return None, f"fused_matmul_reduce_scatter probe failed: {e!r}"

    return {"symm_mem": symm_mem, "group_name": group_name, "source": "torch.ops.symm_mem"}, "torch.ops.symm_mem"


def _bench_ag_mm(symm_mem, group_name: str, world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int) -> Dict:
    if M % world:
        M = (M // world) * world
        if M == 0:
            raise RuntimeError("SymmMem AG+MM needs M >= world")
    batch = 2
    A_shard = torch.empty(batch, M // world, K, dtype=torch.bfloat16, device=device).normal_()
    if hasattr(symm_mem, "restride_A_shard_for_fused_all_gather_matmul"):
        A_shard = symm_mem.restride_A_shard_for_fused_all_gather_matmul(A_shard, dim=1)
    Bs = [torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()]

    def fn():
        torch.ops.symm_mem.fused_all_gather_matmul(A_shard, Bs, gather_dim=1, group_name=group_name)

    dist.barrier()
    res = time_op(f"ag_mm_symm_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * batch * M * K * N
    ag_wire_bytes = (world - 1) * batch * M * K * 2
    return {
        "op": "ag_mm",
        "world": world, "M": M, "K": K, "N": N, "batch": batch,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "ag_gb_s": ag_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def _bench_mm_rs(symm_mem, group_name: str, world: int, device, M: int, K: int, N: int,
                 warmup: int, iters: int) -> Dict:
    if M % world:
        M = (M // world) * world
        if M == 0:
            raise RuntimeError("SymmMem MM+RS needs M >= world")
    batch = 2
    A = torch.empty(batch, M, K, dtype=torch.bfloat16, device=device).normal_()
    if hasattr(symm_mem, "restride_A_for_fused_matmul_reduce_scatter"):
        A = symm_mem.restride_A_for_fused_matmul_reduce_scatter(A, dim=1)
    B = torch.empty(K, N, dtype=torch.bfloat16, device=device).normal_()

    def fn():
        torch.ops.symm_mem.fused_matmul_reduce_scatter(A, B, "avg", scatter_dim=1, group_name=group_name)

    dist.barrier()
    res = time_op(f"mm_rs_symm_{M}_{K}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * batch * M * K * N
    rs_wire_bytes = (world - 1) * batch * M * N * 2
    return {
        "op": "mm_rs",
        "world": world, "M": M, "K": K, "N": N, "batch": batch,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "rs_gb_s": rs_wire_bytes / (res.median_ms * 1e-3) / 1e9,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shapes", type=str, default=None, help="Optional override: 'M,K,N;M,K,N;...'")
    args = ap.parse_args()

    rank, world, device, backend, distributed = _setup_distributed()
    has_gpu = device.type == "cuda"
    out_dir = Path(args.out) / "10_symm_fused"

    def _maybe_barrier_and_destroy() -> None:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    if not has_gpu:
        if rank == 0:
            _write_skip(out_dir, "CPU host: SymmMem fused kernels require GPU.", world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0
    if world < 2:
        if rank == 0:
            _write_skip(out_dir, f"world={world}: SymmMem fused kernels need world>=2.", world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0

    backend_info, reason = _probe_symm_mem(world, device)
    if backend_info is None:
        if rank == 0:
            _write_skip(out_dir, reason, world=world, backend=backend, device_type=device.type)
        _maybe_barrier_and_destroy()
        return 0

    if args.shapes:
        try:
            shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";") if s]
        except ValueError as e:
            raise SystemExit(f"--shapes parse failed: {e}")
    else:
        shapes = SHAPES

    rows: List[Dict] = []
    for (M, K, N) in shapes:
        for label in ("ag_mm", "mm_rs"):
            try:
                if label == "ag_mm":
                    row = _bench_ag_mm(backend_info["symm_mem"], backend_info["group_name"], world, device, M, K, N, args.warmup, args.iters)
                else:
                    row = _bench_mm_rs(backend_info["symm_mem"], backend_info["group_name"], world, device, M, K, N, args.warmup, args.iters)
                row["api_source"] = backend_info["source"]
                rows.append(row)
            except Exception as e:  # noqa: BLE001
                rows.append({
                    "op": label, "world": world, "M": M, "K": K, "N": N,
                    "error": repr(e), "api_source": backend_info["source"],
                })

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "available": True,
            "api_source": backend_info["source"],
            "world": world,
            "backend": backend,
            "device_type": device.type,
            "rows": rows,
        }
        write_json(out_dir / "fused.json", payload)
        write_csv(out_dir / "fused.csv", rows)

    _maybe_barrier_and_destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

