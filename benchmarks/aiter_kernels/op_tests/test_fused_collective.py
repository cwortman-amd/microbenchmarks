"""Correctness + functional gate for the fused AG+MM and MM+RS kernels.

Mirrors what AITER's ``op_tests/triton_tests/comms/`` does upstream: every
available backend is run against the same input, and every result is
compared against the pure-PyTorch fallback (the "gold" reference).

Run with torchrun (multi-rank correctness needs a real process group):

  torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective

Single-rank dry-run (skips the multi-rank tests but checks dispatcher /
import surface):

  python -m benchmarks.aiter_kernels.op_tests.test_fused_collective
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

from benchmarks.aiter_kernels._capabilities import (
    AITER_AVAILABLE,
    HIPKITTENS_AVAILABLE,
    IRIS_AVAILABLE,
    SYMM_MEM_AVAILABLE,
    TRITON_AVAILABLE,
    detect_arch,
)
from benchmarks.aiter_kernels._fallback import (
    fused_all_gather_matmul_fallback,
    fused_matmul_reduce_scatter_fallback,
)
from benchmarks.aiter_kernels.dispatcher import select_backend


@dataclass
class TestShape:
    M_shard: int
    K: int
    N: int
    dtype: torch.dtype = torch.bfloat16


SHAPES = [
    TestShape(M_shard=256, K=512, N=512),
    TestShape(M_shard=512, K=1024, N=1024),
]


def _setup_distributed() -> Tuple[int, int, torch.device]:
    if not all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    rank = int(os.environ.get("RANK", 0))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        if torch.cuda.is_available():
            dist.init_process_group(backend=backend, device_id=device)
        else:
            dist.init_process_group(backend=backend)
            
    rank = dist.get_rank()
    world = dist.get_world_size()
    return rank, world, device


def _backends_available() -> List[str]:
    """Backend labels we can probe at runtime (matches dispatcher order)."""
    out: List[str] = []
    if AITER_AVAILABLE:
        out.append("aiter")
    if AITER_AVAILABLE and IRIS_AVAILABLE:
        out.append("aiter_triton_comms")
    if HIPKITTENS_AVAILABLE:
        out.append("hipkittens")
    if TRITON_AVAILABLE and torch.cuda.is_available():
        out.append("local_triton")
    if SYMM_MEM_AVAILABLE:
        out.append("symm_mem")
    out.append("fallback")  # always available
    return out


def _make_inputs(shape: TestShape, world: int, rank: int, device: torch.device):
    torch.manual_seed(1234 + rank)
    M_global = shape.M_shard * world
    A_full = torch.randn(M_global, shape.K, dtype=shape.dtype, device=device)
    A_shard = A_full[rank * shape.M_shard:(rank + 1) * shape.M_shard].contiguous()
    B = torch.randn(shape.K, shape.N, dtype=shape.dtype, device=device)
    # broadcast B from rank 0 so every rank has the same weight (replicated)
    if dist.is_initialized():
        dist.broadcast(B, src=0)
        # also broadcast A_full to keep the gold reference identical on all ranks
        dist.broadcast(A_full, src=0)
        A_shard = A_full[rank * shape.M_shard:(rank + 1) * shape.M_shard].contiguous()
    return A_shard, B, A_full


def _check_ag_mm(rank: int, world: int, device: torch.device, group_name: str) -> int:
    failures = 0
    for shape in SHAPES:
        A_shard, B, A_full_ref = _make_inputs(shape, world, rank, device)
        # Gold reference: pure-Torch fallback.
        ag_ref, mm_ref = fused_all_gather_matmul_fallback(
            A_shard, [B], gather_dim=0, group_name=group_name
        )
        torch.testing.assert_close(ag_ref, A_full_ref, msg="AG fallback != broadcast A_full")

        for label in _backends_available():
            try:
                info = select_backend(force=label)
            except RuntimeError as e:
                if rank == 0:
                    print(f"  [skip] backend {label}: {e}")
                continue
            try:
                ag_out, mm_outs = info.ag_fn(
                    A_shard, [B], gather_dim=0, group_name=group_name
                )
            except NotImplementedError as e:
                # Backend exists but doesn't have a kernel for this device
                # (e.g. SymmMem on CPU). Treat as SKIP, not FAIL.
                if rank == 0:
                    msg = str(e).splitlines()[0]
                    print(f"  [skip] {label}: shape={shape}: NotImplementedError: {msg}")
                continue
            except Exception as e:  # noqa: BLE001
                failures += 1
                if rank == 0:
                    print(f"  [FAIL] {label}: shape={shape}: {e!r}")
                continue
            try:
                torch.testing.assert_close(ag_out, ag_ref, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(mm_outs[0], mm_ref[0], rtol=1e-2, atol=1e-2)
            except AssertionError as e:
                failures += 1
                if rank == 0:
                    print(f"  [FAIL] {label}: shape={shape}: assert_close: {e}")
            else:
                if rank == 0:
                    print(f"  [ok]   {label}: shape={shape}")
    return failures


def _check_mm_rs(rank: int, world: int, device: torch.device, group_name: str) -> int:
    failures = 0
    for shape in SHAPES:
        torch.manual_seed(4321 + rank)
        M_global = shape.M_shard * world
        A = torch.randn(M_global, shape.K, dtype=shape.dtype, device=device)
        B = torch.randn(shape.K, shape.N, dtype=shape.dtype, device=device)
        if dist.is_initialized():
            dist.broadcast(A, src=0)
            dist.broadcast(B, src=0)
        rs_ref = fused_matmul_reduce_scatter_fallback(
            A, B, "avg", scatter_dim=0, group_name=group_name
        )
        for label in _backends_available():
            try:
                info = select_backend(force=label)
            except RuntimeError as e:
                if rank == 0:
                    print(f"  [skip] backend {label}: {e}")
                continue
            try:
                rs_out = info.rs_fn(
                    A, B, "avg", scatter_dim=0, group_name=group_name
                )
            except NotImplementedError as e:
                if rank == 0:
                    msg = str(e).splitlines()[0]
                    print(f"  [skip] {label}: shape={shape}: NotImplementedError: {msg}")
                continue
            except Exception as e:  # noqa: BLE001
                failures += 1
                if rank == 0:
                    print(f"  [FAIL] {label}: shape={shape}: {e!r}")
                continue
            try:
                torch.testing.assert_close(rs_out, rs_ref, rtol=1e-2, atol=1e-2)
            except AssertionError as e:
                failures += 1
                if rank == 0:
                    print(f"  [FAIL] {label}: shape={shape}: assert_close: {e}")
            else:
                if rank == 0:
                    print(f"  [ok]   {label}: shape={shape}")
    return failures


def main() -> int:
    rank, world, device = _setup_distributed()
    if rank == 0:
        print(f"=== aiter_kernels op_tests (world={world}, device={device}) ===")
        print(f"capabilities: triton={TRITON_AVAILABLE} aiter={AITER_AVAILABLE} "
              f"hipkittens={HIPKITTENS_AVAILABLE} "
              f"iris={IRIS_AVAILABLE} symm_mem={SYMM_MEM_AVAILABLE} "
              f"arch={detect_arch()}")

    if world < 2:
        if rank == 0:
            print("[skip] need world>=2 for collective correctness; "
                  "single-rank smoke check passed (dispatcher imports + capability probe).")
        return 0

    group_name = dist.group.WORLD.group_name  # type: ignore[union-attr]
    fails = 0
    if rank == 0:
        print("\n--- fused_all_gather_matmul ---")
    fails += _check_ag_mm(rank, world, device, group_name)
    if rank == 0:
        print("\n--- fused_matmul_reduce_scatter ---")
    fails += _check_mm_rs(rank, world, device, group_name)

    fails_t = torch.tensor([fails], dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(fails_t, op=dist.ReduceOp.SUM)
    total = int(fails_t.item())

    if rank == 0:
        if total == 0:
            print("\nALL OK.")
        else:
            print(f"\n{total} failure(s) across all ranks/backends.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
