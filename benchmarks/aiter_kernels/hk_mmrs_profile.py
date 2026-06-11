"""Split-phase timing for the HipKittens/Iris MM+RS path.

This instruments the four MM+RS phases separately so we can attribute the
fused MM+RS cost instead of guessing:

  1. writer kernel       (dispatch_mm_rs_write)
  2. post-writer barrier  (raw_ctx.barrier)
  3. reducer kernel       (dispatch_mm_rs_reduce*)
  4. post-reducer barrier (raw_ctx.barrier)

Phases are synchronized between each other so the printed numbers attribute
cost cleanly; this intentionally removes any writer/reducer overlap so we can
see where the ~0.17 ms architectural overhead actually goes. Kernel phases use
CUDA events; barrier phases use wall clock around a synchronized barrier.

Run with:

    BENCH06_USE_IRIS=1 AITER_KERNELS_BACKEND=hipkittens TORCHDYNAMO_DISABLE=1 \
      torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.hk_mmrs_profile \
      --shapes "256,64,256;256,64,512;512,256,512" --warmup 10 --iters 50

Honors ``HIPKITTENS_MM_RS_REDUCER`` and ``HIPKITTENS_MM_RS_SWIZZLE`` so each
variant can be profiled with the same per-phase breakdown.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Dict, List, Tuple

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.distributed as dist

from benchmarks.aiter_kernels import hipkittens as hk
from benchmarks.bench06_aiter_fused import _IrisCtx, _symm_randn


def _parse_shapes(shapes: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for item in shapes.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(p) for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"shape must be M,K,N; got {item!r}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _resolve_module():
    hk._add_hk_paths()
    import importlib

    for name in hk._module_candidates():
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise RuntimeError("could not import the native hk_iris_fused module")


def _select_reducer(mod):
    reducer_mode = os.environ.get("HIPKITTENS_MM_RS_REDUCER", "").strip().lower()
    if reducer_mode == "auto" and hasattr(mod, "dispatch_mm_rs_reduce_vec4"):
        return mod.dispatch_mm_rs_reduce_vec4, "vec4(auto)"
    if reducer_mode in {"specialized", "world"} and hasattr(mod, "dispatch_mm_rs_reduce_specialized"):
        return mod.dispatch_mm_rs_reduce_specialized, "specialized"
    if reducer_mode in {"vec4", "vector4"} and hasattr(mod, "dispatch_mm_rs_reduce_vec4"):
        return mod.dispatch_mm_rs_reduce_vec4, "vec4"
    return mod.dispatch_mm_rs_reduce, "default"


def _event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _barrier_ms(raw_ctx) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    raw_ctx.barrier()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _profile_shape(mod, ctx, device, M: int, K: int, N: int, warmup: int, iters: int) -> Dict[str, float]:
    group = dist.group.WORLD
    world = dist.get_world_size()
    rank = dist.get_rank()
    M_shard = M // world

    swizzle = 1 if os.environ.get("HIPKITTENS_MM_RS_SWIZZLE", "").strip().lower() in {"1", "true", "yes", "on", "swizzle"} else 0
    reduce_dispatch, _ = _select_reducer(mod)

    A = _symm_randn(ctx, (M, K), torch.bfloat16, device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    dist.broadcast(B, src=0)
    B_t = B.t().contiguous()
    _, device_context = hk._get_iris_context_tensor(A, "MM+RS")
    raw_ctx = getattr(ctx, "iris_ctx", ctx)
    scratch = raw_ctx.zeros((world, M_shard, N), dtype=torch.bfloat16)
    Y_shard = raw_ctx.zeros((M_shard, N), dtype=torch.bfloat16)
    scale = 1.0 / world

    def writer():
        mod.dispatch_mm_rs_write(
            A, B_t, scratch, int(device_context.data_ptr()),
            int(M), int(M_shard), int(N), int(K), int(swizzle),
        )

    def reducer():
        reduce_dispatch(
            scratch, Y_shard, int(M_shard), int(N), int(world), float(scale),
            int(rank), int(swizzle),
        )

    for _ in range(warmup):
        writer()
        raw_ctx.barrier()
        reducer()
        raw_ctx.barrier()
    torch.cuda.synchronize()

    samples = {"writer": [], "barrier1": [], "reducer": [], "barrier2": []}
    for _ in range(iters):
        samples["writer"].append(_event_ms(writer))
        samples["barrier1"].append(_barrier_ms(raw_ctx))
        samples["reducer"].append(_event_ms(reducer))
        samples["barrier2"].append(_barrier_ms(raw_ctx))

    local = {k: statistics.median(v) for k, v in samples.items()}
    # Reduce across ranks with MAX so the reported phase cost reflects the
    # slowest rank, which is what the host-visible barrier actually waits on.
    out: Dict[str, float] = {}
    for k, v in local.items():
        t = torch.tensor([v], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        out[k] = float(t.item())
    out["total"] = sum(out.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="256,64,256;256,64,512;512,256,512")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    if world < 2:
        raise RuntimeError("MM+RS profile requires world >= 2")

    import iris

    ctx = _IrisCtx(iris.iris(1 << 31))
    mod = _resolve_module()

    reducer_label = os.environ.get("HIPKITTENS_MM_RS_REDUCER", "default") or "default"
    swizzle_label = os.environ.get("HIPKITTENS_MM_RS_SWIZZLE", "0") or "0"

    rows = []
    for (M, K, N) in _parse_shapes(args.shapes):
        rows.append(((M, K, N), _profile_shape(mod, ctx, device, M, K, N, args.warmup, args.iters)))

    if rank == 0:
        print(f"[mmrs-profile] reducer={reducer_label} swizzle={swizzle_label} world={world} "
              f"warmup={args.warmup} iters={args.iters}")
        print(f"{'shape':>14}  {'writer':>9}  {'barrier1':>9}  {'reducer':>9}  {'barrier2':>9}  {'total':>9}")
        for (shape, p) in rows:
            stag = ",".join(str(x) for x in shape)
            print(f"{stag:>14}  {p['writer']:9.4f}  {p['barrier1']:9.4f}  "
                  f"{p['reducer']:9.4f}  {p['barrier2']:9.4f}  {p['total']:9.4f}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
