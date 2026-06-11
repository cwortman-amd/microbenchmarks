"""Distributed correctness smoke for the HipKittens/Iris fused backend.

Run with:

    BENCH06_USE_IRIS=1 AITER_KERNELS_BACKEND=hipkittens \
      torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.hk_correctness

Set ``HIPKITTENS_AG_N_REUSE`` to select AG+MM variants, for example ``2`` or
``2_spillfree``.
"""

from __future__ import annotations

import os

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.distributed as dist

from benchmarks.aiter_kernels.dispatcher import select_backend
from benchmarks.bench06_aiter_fused import _IrisCtx, _symm_randn


def _assert_close_bf16(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-1)


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError("HK correctness smoke requires world >= 2")

    import iris

    ctx = _IrisCtx(iris.iris(1 << 30))
    ctx.skip_ag_full_output = False
    backend = select_backend(force="hipkittens")
    group_name = dist.group.WORLD.group_name

    # Include N=512 so both reuse=2 variants exercise their widened N path.
    for M, K, N in ((256, 64, 256), (256, 64, 512)):
        M_shard = M // world
        A_shard = _symm_randn(ctx, (M_shard, K), torch.bfloat16, device)
        B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        dist.broadcast(B, src=0)

        A_ref = torch.empty(M, K, dtype=torch.bfloat16, device=device)
        dist.all_gather_into_tensor(A_ref, A_shard)
        ag_out, y_outs = backend.ag_fn(A_shard, [B], gather_dim=0, group_name=group_name)
        torch.testing.assert_close(ag_out, A_ref, rtol=0, atol=0)
        _assert_close_bf16(y_outs[0], A_ref @ B)

    M, K, N = 256, 64, 256
    for double_buffer in ("0", "1"):
        os.environ["HIPKITTENS_MM_RS_DOUBLE_BUFFER"] = double_buffer
        # With double-buffering the post-reducer barrier is dropped and scratch
        # reuse is ordered by the following call's post-writer barrier, so we
        # invoke rs_fn repeatedly to exercise the two-call-back buffer reuse.
        repeats = 3 if double_buffer == "1" else 1
        for swizzle in ("0", "1"):
            os.environ["HIPKITTENS_MM_RS_SWIZZLE"] = swizzle
            for reducer_mode in ("default", "specialized", "vec4", "auto"):
                if reducer_mode == "default":
                    os.environ.pop("HIPKITTENS_MM_RS_REDUCER", None)
                else:
                    os.environ["HIPKITTENS_MM_RS_REDUCER"] = reducer_mode
                A = _symm_randn(ctx, (M, K), torch.bfloat16, device)
                B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
                dist.broadcast(B, src=0)
                for _ in range(repeats):
                    y_shard = backend.rs_fn(A, B, "sum", scatter_dim=0, group_name=group_name)
                y_full = A @ B
                chunks = list(y_full.chunk(world, dim=0))
                expected = torch.empty_like(chunks[rank])
                dist.reduce_scatter_tensor(expected, y_full, op=dist.ReduceOp.SUM)
                _assert_close_bf16(y_shard, expected)
    os.environ.pop("HIPKITTENS_MM_RS_DOUBLE_BUFFER", None)

    # Iris device-side barrier replacing the host post-writer barrier. EXPERIMENTAL
    # and currently numerically incorrect with the HK native writer: even Iris's
    # own validated device_barrier (and write-through stores) leaves a small,
    # intermittent set of elements stale, the same signature as the flag handoff.
    # This confirms the blocker is framework/integration-level remote-write
    # completion, not the barrier mechanism. Opt-in via the same stress env.
    device_barrier_ok = False
    if os.environ.get("HIPKITTENS_MM_RS_FLAGS_STRESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["HIPKITTENS_MM_RS_DEVICE_BARRIER"] = "1"
        for double_buffer in ("0", "1"):
            os.environ["HIPKITTENS_MM_RS_DOUBLE_BUFFER"] = double_buffer
            for M, K, N in ((256, 64, 256), (512, 256, 512)):
                A = _symm_randn(ctx, (M, K), torch.bfloat16, device)
                B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
                dist.broadcast(B, src=0)
                y_full = A @ B
                expected = torch.empty_like(list(y_full.chunk(world, dim=0))[rank])
                dist.reduce_scatter_tensor(expected, y_full, op=dist.ReduceOp.SUM)
                for _ in range(10):
                    y_shard = backend.rs_fn(A, B, "sum", scatter_dim=0, group_name=group_name)
                    _assert_close_bf16(y_shard, expected)
        os.environ.pop("HIPKITTENS_MM_RS_DOUBLE_BUFFER", None)
        os.environ.pop("HIPKITTENS_MM_RS_DEVICE_BARRIER", None)
        device_barrier_ok = True

    # Dedicated stress for the device-side flag handoff (replaces barrier1).
    # Cross-GPU release/acquire ordering bugs are intermittent, so use multiple
    # shapes and many back-to-back repeats per config to exercise generation
    # monotonicity and double-buffered scratch reuse under the flag protocol.
    #
    # NOTE: the flag handoff is EXPERIMENTAL and currently numerically incorrect.
    # Without an Iris device-side remote-write-completion ("quiet"/drain)
    # primitive, threadfence_system does not guarantee the source's xGMI scratch
    # writes have landed in the destination's HBM before the flag is observed,
    # so the reducer occasionally reads stale data for a few elements. The stress
    # is therefore opt-in via HIPKITTENS_MM_RS_FLAGS_STRESS=1 and is expected to
    # fail until that primitive exists; it is kept to detect regressions/progress.
    flags_ok = False
    if hasattr(backend, "rs_fn") and os.environ.get("HIPKITTENS_MM_RS_FLAGS_STRESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["HIPKITTENS_MM_RS_FLAGS"] = "1"
        flag_repeats = 20
        for M, K, N in ((256, 64, 256), (256, 64, 512), (512, 256, 512)):
            for swizzle in ("0", "1"):
                os.environ["HIPKITTENS_MM_RS_SWIZZLE"] = swizzle
                A = _symm_randn(ctx, (M, K), torch.bfloat16, device)
                B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
                dist.broadcast(B, src=0)
                y_full = A @ B
                expected = torch.empty_like(list(y_full.chunk(world, dim=0))[rank])
                dist.reduce_scatter_tensor(expected, y_full, op=dist.ReduceOp.SUM)
                for _ in range(flag_repeats):
                    y_shard = backend.rs_fn(A, B, "sum", scatter_dim=0, group_name=group_name)
                    _assert_close_bf16(y_shard, expected)
        os.environ.pop("HIPKITTENS_MM_RS_FLAGS", None)
        os.environ.pop("HIPKITTENS_MM_RS_SWIZZLE", None)
        flags_ok = True

    if rank == 0:
        mode = os.environ.get("HIPKITTENS_AG_N_REUSE", "default") or "default"
        flag_note = ", flag_handoff_stress=PASS(3 shapes x 20 repeats)" if flags_ok else ""
        dbar_note = ", device_barrier_stress=PASS" if device_barrier_ok else ""
        print(f"HK/Iris correctness smoke OK (AG mode={mode}, MM+RS reducers=default/specialized/vec4/auto, swizzle=0/1, double_buffer=0/1{dbar_note}{flag_note})")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
