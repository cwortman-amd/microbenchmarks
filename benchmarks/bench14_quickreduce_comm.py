"""Family 14 — QuickReduce communication backend experiment.

This benchmark is the first evaluation harness for the post-HK strategy:
optimize the **MM + AllReduce** path by swapping the communication backend
rather than writing another fused AG+MM/MM+RS kernel.

Why this exists:

* Production-accurate TP sharding showed AG+MM/MM+RS fusion has a small ceiling
  and loses to hipBLASLt + RCCL at Wan2.2 / Odyssey shapes.
* Wan2.2/SGLang-style TP appears dominated by row-parallel MM+AR.
* QuickReduce claims faster ROCm all-reduce on 2/4 GPUs and supports compressed
  line codecs. This benchmark tests whether that moves end-to-end MM+AR.

The benchmark is optional and defensive: if ``quickreduce`` is not installed or
cannot initialize on this node, it still writes RCCL baseline rows and a skip
reason for the QuickReduce track.

Sharding convention (row-parallel / Megatron RowParallelLinear): the contraction
dim ``K`` is sharded across ranks. Each rank holds ``A_local[M, K/world]`` and
``B_local[K/world, N]``, computes a partial ``[M, N]``, and then all-reduces the
partials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

import torch
import torch.distributed as dist

from benchmarks.common.io import write_csv, write_json
from benchmarks.common.shapes import pad_to_multiple, resolve_shapes
from benchmarks.common.timing import time_op

_DTYPE_BYTES = {"fp16": 2, "bf16": 2}

# QuickReduce line-codec profile ids. fp16..q4 match the archived
# mk1-project/quickreduce build; fp4 is the MI355/CDNA4 codec introduced in the
# AMD ROCm blog (QuickReduce FP4 on MI355) and added by the local gfx950 patch.
_CODECS = {
    "fp16": 1,
    "fp8": 2,
    "q8": 3,
    "q6": 4,
    "q4": 5,
    "fp4": 6,
}

# Transport bit-width per codec. QuickReduce codecs compress only the all-reduce
# *transport* (the matmul/compute stays fp16); this drives the effective
# (compressed) wire-bytes figure so a codec's bandwidth advantage is visible
# instead of being hidden behind the uncompressed fp16 "logical" payload.
_CODEC_BITS = {
    "fp16": 16,
    "fp8": 8,
    "q8": 8,
    "q6": 6,
    "q4": 4,
    "fp4": 4,
}


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


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _ar_wire_bytes(world: int, M: int, N: int, dtype_name: str) -> int:
    """Ring all-reduce moves ~2(world-1)/world * payload per rank."""
    payload = M * N * _DTYPE_BYTES[dtype_name]
    return int(2 * (world - 1) / world * payload)


def _shard_K(K: int, world: int) -> Tuple[int, int]:
    K_pad = pad_to_multiple(K, world)
    return K_pad, K_pad // world


def _make_quickreduce(world: int, rank: int) -> Tuple[Optional[Any], str]:
    """Initialize QuickReduce and exchange IPC handles with torch.distributed."""
    try:
        import quickreduce as qr  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, f"quickreduce import failed: {e!r}"

    try:
        qr.init(world, rank)
        local_handle = qr.get_comm_handle()
        handles: List[Any] = [None for _ in range(world)]
        dist.all_gather_object(handles, local_handle)
        qr.set_comm_handles(handles)
    except Exception as e:  # noqa: BLE001
        return None, f"quickreduce init/handle exchange failed: {e!r}"
    return qr, "quickreduce"


def _make_custom_all_reduce(device: torch.device) -> Tuple[Optional[Any], str]:
    """Construct vLLM's Custom AllReduce (CR), defensively.

    CR is the real low-volume fallback in vLLM/SGLang and beats both RCCL and
    QuickReduce below the activation threshold (~512KB-1MB per the AMD blog), so
    it is the correct small-message baseline. This is optional: on any failure
    (vLLM missing, unsupported topology, no P2P) we return ``None`` and record a
    skip reason rather than aborting the RCCL/QuickReduce comparison.
    """
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import (  # type: ignore
            CustomAllreduce,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"custom_all_reduce import failed: {e!r}"
    try:
        group = dist.group.WORLD
        ca = CustomAllreduce(group=group, device=device)
        if not getattr(ca, "disabled", False):
            return ca, "custom_all_reduce"
        return None, "custom_all_reduce disabled (unsupported topology/P2P)"
    except Exception as e:  # noqa: BLE001
        return None, f"custom_all_reduce init failed: {e!r}"


def _cr_all_reduce(ca: Any, x: torch.Tensor) -> torch.Tensor:
    """Invoke CR; raise if this size is not CR-eligible (so caller can skip)."""
    out = None
    for name in ("custom_all_reduce", "all_reduce"):
        fn = getattr(ca, name, None)
        if fn is not None:
            out = fn(x)
            break
    if out is None:
        raise RuntimeError("custom all-reduce returned None (size not CR-eligible)")
    return out


def _codec_rel_l2(qr: Any, profile: int, device: torch.device, M: int, N: int,
                  dtype: torch.dtype) -> float:
    """Relative L2 error of a QuickReduce codec vs an fp32 reference all-reduce.

    Characterizes codec fidelity at microbench level (independent of model
    quality). Computed on fresh clones outside the timed loop, so it is not
    perturbed by in-place RCCL accumulation. Returns NaN if QR errors out.
    """
    try:
        x = torch.empty(M, N, dtype=dtype, device=device).normal_()
        ref = x.float().clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        out = qr.allreduce(profile, x.clone()).float()
        denom = ref.norm().item()
        if denom == 0.0:
            return float("nan")
        return (out - ref).norm().item() / denom
    except Exception:  # noqa: BLE001
        return float("nan")


def _qr_allreduce(qr: Any, profile: int, x: torch.Tensor, grid_cap: int = 0) -> torch.Tensor:
    return qr.allreduce(profile, x, grid_cap) if grid_cap else qr.allreduce(profile, x)


def _qr_allreduce_bf16(qr: Any, profile: int, x: torch.Tensor, grid_cap: int = 0) -> torch.Tensor:
    if not hasattr(qr, "allreduce_bf16"):
        raise RuntimeError("quickreduce build does not expose allreduce_bf16")
    return qr.allreduce_bf16(profile, x, grid_cap) if grid_cap else qr.allreduce_bf16(profile, x)


def _qr_allreduce_out(qr: Any, profile: int, x: torch.Tensor, out: torch.Tensor, grid_cap: int = 0) -> torch.Tensor:
    if hasattr(qr, "allreduce_out"):
        qr.allreduce_out(profile, x, out, grid_cap)
        return out
    return _qr_allreduce(qr, profile, x, grid_cap)


def _qr_allreduce_bf16_out(qr: Any, profile: int, x: torch.Tensor, out: torch.Tensor,
                           grid_cap: int = 0) -> torch.Tensor:
    if hasattr(qr, "allreduce_bf16_out"):
        qr.allreduce_bf16_out(profile, x, out, grid_cap)
        return out
    return _qr_allreduce_bf16(qr, profile, x, grid_cap)


def _bench_rccl_all_reduce(
    world: int,
    device: torch.device,
    M: int,
    N: int,
    dtype_name: str,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    dtype = _dtype(dtype_name)
    x = torch.empty(M, N, dtype=dtype, device=device).normal_()

    def fn():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    dist.barrier()
    res = time_op(f"rccl_all_reduce_{M}_{N}_{dtype_name}", fn, warmup=warmup, iters=iters)
    wire = _ar_wire_bytes(world, M, N, dtype_name)
    return {
        "op": "rccl_all_reduce",
        "world": world,
        "M": M,
        "N": N,
        "dtype": dtype_name,
        "t_ms": res.median_ms,
        "ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "ar_wire_bytes": wire,
    }


def _probe_native_integration(qr: Any, device: torch.device) -> Dict[str, object]:
    """Validate the native-PyTorch integration contract for QuickReduce.

    The production engine is native ``torch.distributed`` (no vLLM), so the only
    integration surface is: pass a torch CUDA tensor to ``qr.allreduce`` and use
    the returned tensor. This collective probe (all ranks must call it) records
    the contract a proprietary engine must satisfy:

    * out-of-place: ``allreduce`` returns a NEW tensor; the input is unmodified
      (so the engine cannot assume in-place semantics like ``dist.all_reduce``).
    * dtype checked: the legacy fp16 API rejects bf16; the fused bf16 API can be
      used when available.
    * world-size is restricted to {2, 4, 8}.
    """
    info: Dict[str, object] = {}
    try:
        world = dist.get_world_size()
        info["world_size"] = world
        info["world_size_supported"] = world in (2, 4, 8)
        info["dtype_required"] = "fp16 for allreduce; bf16 for allreduce_bf16"
        info["semantics"] = "out_of_place_returns_new_tensor"
        info["max_problem_bytes"] = 536870912  # kMaxProblemSize in quickreduce.h

        x = torch.empty(4096, dtype=torch.float16, device=device).normal_()
        x_before = x.clone()
        out = qr.allreduce(_CODECS["fp16"], x)
        info["input_unmodified"] = bool(torch.equal(x, x_before))
        info["returns_new_tensor"] = out.data_ptr() != x.data_ptr()

        ref = x.float().clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        denom = ref.norm().item()
        info["fp16_rel_l2_err"] = (out.float() - ref).norm().item() / denom if denom else float("nan")

        # Existing fp16 API should reject bf16 instead of silently corrupting it.
        xb = torch.empty(4096, dtype=torch.bfloat16, device=device).normal_()
        refb = xb.float().clone()
        dist.all_reduce(refb, op=dist.ReduceOp.SUM)
        try:
            qr.allreduce(_CODECS["fp16"], xb)
            info["legacy_allreduce_rejects_bf16"] = False
        except Exception as e:  # noqa: BLE001
            info["legacy_allreduce_rejects_bf16"] = True
            info["legacy_bf16_reject_reason"] = repr(e)
        if hasattr(qr, "allreduce_bf16"):
            outb = qr.allreduce_bf16(_CODECS["fp16"], xb).float()
            dn = refb.norm().item()
            info["bf16_native_rel_l2_err"] = (outb - refb).norm().item() / dn if dn else float("nan")
            info["bf16_native_safe"] = bool(info["bf16_native_rel_l2_err"] < 1e-1)
        else:
            info["bf16_native_safe"] = False
            info["bf16_native_error"] = "quickreduce build does not expose allreduce_bf16"
    except Exception as e:  # noqa: BLE001
        info["error"] = repr(e)
    return info


def _bench_cr_all_reduce(
    ca: Any,
    world: int,
    device: torch.device,
    M: int,
    N: int,
    dtype_name: str,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    dtype = _dtype(dtype_name)
    x = torch.empty(M, N, dtype=dtype, device=device).normal_()
    holder: Dict[str, torch.Tensor] = {}

    # Probe once so an ineligible size raises before the timed loop.
    holder["out"] = _cr_all_reduce(ca, x)

    def fn():
        holder["out"] = _cr_all_reduce(ca, x)

    dist.barrier()
    res = time_op(f"cr_all_reduce_{M}_{N}_{dtype_name}", fn, warmup=warmup, iters=iters)
    wire = _ar_wire_bytes(world, M, N, dtype_name)
    return {
        "op": "cr_all_reduce",
        "world": world,
        "M": M,
        "N": N,
        "dtype": dtype_name,
        "t_ms": res.median_ms,
        "ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "ar_wire_bytes": wire,
    }


def _bench_quickreduce_all_reduce(
    qr: Any,
    world: int,
    device: torch.device,
    M: int,
    N: int,
    dtype_name: str,
    codec: str,
    warmup: int,
    iters: int,
    grid_cap: int = 0,
) -> Dict[str, object]:
    dtype = _dtype(dtype_name)
    profile = _CODECS[codec]
    bits = _CODEC_BITS[codec]
    # Codec fidelity is always measured in fp16 (QuickReduce's only supported
    # transport dtype, and what the bf16 path feeds after casting). Measuring in
    # bf16 would report the dtype-corruption error (~1e2), not codec error.
    rel_l2 = _codec_rel_l2(qr, profile, device, M, N, torch.float16)
    x = torch.empty(M, N, dtype=dtype, device=device).normal_()
    out = torch.empty_like(x)
    holder: Dict[str, torch.Tensor] = {}

    def fn():
        holder["out"] = _qr_allreduce_out(qr, profile, x, out, grid_cap)

    dist.barrier()
    res = time_op(f"quickreduce_ar_{codec}_{M}_{N}_{dtype_name}", fn, warmup=warmup, iters=iters)
    wire = _ar_wire_bytes(world, M, N, dtype_name)
    eff_wire = int(wire * bits / 16)
    return {
        "op": "quickreduce_all_reduce",
        "codec": codec,
        "profile": profile,
        "codec_bits": bits,
        "qr_grid_cap": grid_cap,
        "world": world,
        "M": M,
        "N": N,
        "dtype": dtype_name,
        "t_ms": res.median_ms,
        "rel_l2_err": rel_l2,
        "logical_ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "logical_ar_wire_bytes": wire,
        "effective_ar_gb_s": eff_wire / (res.median_ms * 1e-3) / 1e9,
        "effective_ar_wire_bytes": eff_wire,
    }


def _bench_unfused_mm_ar(
    world: int,
    device: torch.device,
    M: int,
    K: int,
    N: int,
    dtype_name: str,
    warmup: int,
    iters: int,
) -> Dict[str, object]:
    dtype = _dtype(dtype_name)
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=dtype, device=device).normal_()
    B = torch.empty(K_local, N, dtype=dtype, device=device).normal_()
    Y_pre = torch.empty(M, N, dtype=dtype, device=device).normal_()

    def fn_mm():
        torch.matmul(A, B)

    def fn_ar():
        dist.all_reduce(Y_pre, op=dist.ReduceOp.SUM)

    def fn_full():
        Y = torch.matmul(A, B)
        dist.all_reduce(Y, op=dist.ReduceOp.SUM)

    dist.barrier()
    res_full = time_op(f"unfused_mm_ar_full_{M}_{K_pad}_{N}_{dtype_name}", fn_full, warmup=warmup, iters=iters)
    res_mm = time_op(f"unfused_mm_ar_mm_{M}_{K_pad}_{N}_{dtype_name}", fn_mm, warmup=warmup, iters=iters)
    res_ar = time_op(f"unfused_mm_ar_ar_{M}_{K_pad}_{N}_{dtype_name}", fn_ar, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N, dtype_name)
    return {
        "op": "unfused_mm_ar",
        "world": world,
        "M": M,
        "K": K_pad,
        "K_local": K_local,
        "N": N,
        "dtype": dtype_name,
        "t_ms": res_full.median_ms,
        "t_ms_mm": res_mm.median_ms,
        "t_ms_ar": res_ar.median_ms,
        "tflops": flops / (res_full.median_ms * 1e-3) / 1e12,
        "ar_gb_s": wire / (res_full.median_ms * 1e-3) / 1e9,
        "ar_wire_bytes": wire,
    }


def _bench_quickreduce_mm_ar(
    qr: Any,
    world: int,
    device: torch.device,
    M: int,
    K: int,
    N: int,
    dtype_name: str,
    codec: str,
    warmup: int,
    iters: int,
    grid_cap: int = 0,
) -> Dict[str, object]:
    dtype = _dtype(dtype_name)
    profile = _CODECS[codec]
    bits = _CODEC_BITS[codec]
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=dtype, device=device).normal_()
    B = torch.empty(K_local, N, dtype=dtype, device=device).normal_()
    out = torch.empty(M, N, dtype=dtype, device=device)
    holder: Dict[str, torch.Tensor] = {}

    def fn():
        Y = torch.matmul(A, B)
        holder["out"] = _qr_allreduce_out(qr, profile, Y, out, grid_cap)

    dist.barrier()
    res = time_op(f"quickreduce_mm_ar_{codec}_{M}_{K_pad}_{N}_{dtype_name}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N, dtype_name)
    eff_wire = int(wire * bits / 16)
    return {
        "op": "quickreduce_mm_ar",
        "codec": codec,
        "profile": profile,
        "codec_bits": bits,
        "qr_grid_cap": grid_cap,
        "world": world,
        "M": M,
        "K": K_pad,
        "K_local": K_local,
        "N": N,
        "dtype": dtype_name,
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "logical_ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "logical_ar_wire_bytes": wire,
        "effective_ar_gb_s": eff_wire / (res.median_ms * 1e-3) / 1e9,
        "effective_ar_wire_bytes": eff_wire,
    }


def _bench_quickreduce_mm_ar_bf16cast(
    qr: Any,
    world: int,
    device: torch.device,
    M: int,
    K: int,
    N: int,
    codec: str,
    warmup: int,
    iters: int,
    grid_cap: int = 0,
) -> Dict[str, object]:
    """Realistic bf16-engine path: bf16 GEMM -> cast fp16 -> QR -> cast bf16.

    QuickReduce operates on fp16 only, so a native bf16 engine must wrap the
    collective in casts. This measures the full exposed path (including both
    casts) so the uplift is not overstated by the pure-fp16 number. The matching
    baseline is ``unfused_mm_ar`` run with ``--dtype bf16`` (bf16 GEMM + bf16
    RCCL all-reduce).
    """
    profile = _CODECS[codec]
    bits = _CODEC_BITS[codec]
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K_local, N, dtype=torch.bfloat16, device=device).normal_()
    qr_out = torch.empty(M, N, dtype=torch.float16, device=device)
    holder: Dict[str, torch.Tensor] = {}

    def fn():
        Y = torch.matmul(A, B)               # bf16 GEMM
        Yh = Y.to(torch.float16)             # cast down for QuickReduce
        red = _qr_allreduce_out(qr, profile, Yh, qr_out, grid_cap)  # fp16 all-reduce
        holder["out"] = red.to(torch.bfloat16)  # cast back to engine dtype

    # Isolate the cast overhead (both directions) for attribution.
    Yc = torch.empty(M, N, dtype=torch.bfloat16, device=device).normal_()

    def fn_cast():
        h = Yc.to(torch.float16)
        holder["c"] = h.to(torch.bfloat16)

    dist.barrier()
    res = time_op(f"quickreduce_mm_ar_bf16cast_{codec}_{M}_{K_pad}_{N}", fn, warmup=warmup, iters=iters)
    res_cast = time_op(f"qr_bf16cast_castonly_{codec}_{M}_{K_pad}_{N}", fn_cast, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N, "fp16")
    eff_wire = int(wire * bits / 16)
    return {
        "op": "quickreduce_mm_ar_bf16cast",
        "codec": codec,
        "profile": profile,
        "codec_bits": bits,
        "qr_grid_cap": grid_cap,
        "world": world,
        "M": M,
        "K": K_pad,
        "K_local": K_local,
        "N": N,
        "dtype": "bf16",
        "t_ms": res.median_ms,
        "t_ms_cast": res_cast.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "logical_ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "logical_ar_wire_bytes": wire,
        "effective_ar_gb_s": eff_wire / (res.median_ms * 1e-3) / 1e9,
        "effective_ar_wire_bytes": eff_wire,
    }


def _bench_quickreduce_mm_ar_bf16native(
    qr: Any,
    world: int,
    device: torch.device,
    M: int,
    K: int,
    N: int,
    codec: str,
    warmup: int,
    iters: int,
    grid_cap: int = 0,
) -> Dict[str, object]:
    """bf16 GEMM followed by QuickReduce's fused bf16 I/O collective."""
    profile = _CODECS[codec]
    bits = _CODEC_BITS[codec]
    K_pad, K_local = _shard_K(K, world)
    A = torch.empty(M, K_local, dtype=torch.bfloat16, device=device).normal_()
    B = torch.empty(K_local, N, dtype=torch.bfloat16, device=device).normal_()
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    holder: Dict[str, torch.Tensor] = {}

    def fn():
        Y = torch.matmul(A, B)
        holder["out"] = _qr_allreduce_bf16_out(qr, profile, Y, out, grid_cap)

    dist.barrier()
    res = time_op(f"quickreduce_mm_ar_bf16native_{codec}_{M}_{K_pad}_{N}", fn, warmup=warmup, iters=iters)
    flops = 2 * M * K_local * N
    wire = _ar_wire_bytes(world, M, N, "fp16")
    eff_wire = int(wire * bits / 16)
    return {
        "op": "quickreduce_mm_ar_bf16native",
        "codec": codec,
        "profile": profile,
        "codec_bits": bits,
        "qr_grid_cap": grid_cap,
        "world": world,
        "M": M,
        "K": K_pad,
        "K_local": K_local,
        "N": N,
        "dtype": "bf16",
        "t_ms": res.median_ms,
        "tflops": flops / (res.median_ms * 1e-3) / 1e12,
        "logical_ar_gb_s": wire / (res.median_ms * 1e-3) / 1e9,
        "logical_ar_wire_bytes": wire,
        "effective_ar_gb_s": eff_wire / (res.median_ms * 1e-3) / 1e9,
        "effective_ar_wire_bytes": eff_wire,
    }


def _summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    rccl_ar = {
        (r.get("M"), r.get("N"), r.get("dtype")): r
        for r in rows
        if r.get("op") == "rccl_all_reduce" and "error" not in r
    }
    cr_ar = {
        (r.get("M"), r.get("N"), r.get("dtype")): r
        for r in rows
        if r.get("op") == "cr_all_reduce" and "error" not in r
    }
    qr_ar_by_key: Dict[Any, List[Dict[str, object]]] = {}
    for r in rows:
        if r.get("op") == "quickreduce_all_reduce" and "error" not in r:
            qr_ar_by_key.setdefault((r.get("M"), r.get("N"), r.get("dtype")), []).append(r)
    rccl_mm_ar = {
        (r.get("M"), r.get("K"), r.get("N"), r.get("dtype")): r
        for r in rows
        if r.get("op") == "unfused_mm_ar" and "error" not in r
    }
    for r in rows:
        if "error" in r:
            continue
        if r.get("op") == "quickreduce_all_reduce":
            base = rccl_ar.get((r.get("M"), r.get("N"), r.get("dtype")))
            if base:
                out.append({
                    "comparison": "quickreduce_ar_vs_rccl_ar",
                    "codec": r.get("codec"),
                    "world": r.get("world"),
                    "M": r.get("M"),
                    "N": r.get("N"),
                    "dtype": r.get("dtype"),
                    "candidate_ms": r.get("t_ms"),
                    "baseline_ms": base.get("t_ms"),
                    "speedup": float(base["t_ms"]) / float(r["t_ms"]),
                })
        elif r.get("op") == "cr_all_reduce":
            base = rccl_ar.get((r.get("M"), r.get("N"), r.get("dtype")))
            if base:
                out.append({
                    "comparison": "cr_ar_vs_rccl_ar",
                    "world": r.get("world"),
                    "M": r.get("M"),
                    "N": r.get("N"),
                    "dtype": r.get("dtype"),
                    "candidate_ms": r.get("t_ms"),
                    "baseline_ms": base.get("t_ms"),
                    "speedup": float(base["t_ms"]) / float(r["t_ms"]),
                })
        elif r.get("op") == "quickreduce_mm_ar":
            base = rccl_mm_ar.get((r.get("M"), r.get("K"), r.get("N"), r.get("dtype")))
            if base:
                out.append({
                    "comparison": "quickreduce_mm_ar_vs_rccl_mm_ar",
                    "codec": r.get("codec"),
                    "world": r.get("world"),
                    "M": r.get("M"),
                    "K": r.get("K"),
                    "N": r.get("N"),
                    "dtype": r.get("dtype"),
                    "candidate_ms": r.get("t_ms"),
                    "baseline_ms": base.get("t_ms"),
                    "speedup": float(base["t_ms"]) / float(r["t_ms"]),
                    "baseline_mm_ms": base.get("t_ms_mm"),
                    "baseline_ar_ms": base.get("t_ms_ar"),
                })
        elif r.get("op") == "quickreduce_mm_ar_bf16cast":
            base = rccl_mm_ar.get((r.get("M"), r.get("K"), r.get("N"), r.get("dtype")))
            if base:
                out.append({
                    "comparison": "quickreduce_mm_ar_bf16cast_vs_rccl_mm_ar",
                    "codec": r.get("codec"),
                    "world": r.get("world"),
                    "M": r.get("M"),
                    "K": r.get("K"),
                    "N": r.get("N"),
                    "dtype": r.get("dtype"),
                    "candidate_ms": r.get("t_ms"),
                    "cast_ms": r.get("t_ms_cast"),
                    "baseline_ms": base.get("t_ms"),
                    "speedup": float(base["t_ms"]) / float(r["t_ms"]),
                    "baseline_mm_ms": base.get("t_ms_mm"),
                    "baseline_ar_ms": base.get("t_ms_ar"),
                })
        elif r.get("op") == "quickreduce_mm_ar_bf16native":
            base = rccl_mm_ar.get((r.get("M"), r.get("K"), r.get("N"), r.get("dtype")))
            if base:
                out.append({
                    "comparison": "quickreduce_mm_ar_bf16native_vs_rccl_mm_ar",
                    "codec": r.get("codec"),
                    "world": r.get("world"),
                    "M": r.get("M"),
                    "K": r.get("K"),
                    "N": r.get("N"),
                    "dtype": r.get("dtype"),
                    "qr_grid_cap": r.get("qr_grid_cap"),
                    "candidate_ms": r.get("t_ms"),
                    "baseline_ms": base.get("t_ms"),
                    "speedup": float(base["t_ms"]) / float(r["t_ms"]),
                    "baseline_mm_ms": base.get("t_ms_mm"),
                    "baseline_ar_ms": base.get("t_ms_ar"),
                })

    # Engine selection policy: a real proprietary engine dispatches the *best*
    # available all-reduce per message size (RCCL/CR below the activation
    # threshold, QuickReduce above it). Model the uplift as
    # best_of(RCCL, CR, QR_codec*) vs the no-QuickReduce fallback best_of(RCCL, CR).
    for key, qr_list in qr_ar_by_key.items():
        rccl = rccl_ar.get(key)
        if not rccl:
            continue
        cr = cr_ar.get(key)
        M, N, dtype_name = key
        fallback_candidates = [("rccl", float(rccl["t_ms"]))]
        if cr:
            fallback_candidates.append(("cr", float(cr["t_ms"])))
        fb_method, fb_ms = min(fallback_candidates, key=lambda kv: kv[1])
        engine_candidates = list(fallback_candidates)
        best_qr = min(qr_list, key=lambda r: float(r["t_ms"]))
        engine_candidates.append((f"qr_{best_qr.get('codec')}", float(best_qr["t_ms"])))
        eng_method, eng_ms = min(engine_candidates, key=lambda kv: kv[1])
        out.append({
            "comparison": "engine_best_of_ar",
            "world": rccl.get("world"),
            "M": M,
            "N": N,
            "dtype": dtype_name,
            "fallback_method": fb_method,
            "fallback_ms": fb_ms,
            "engine_method": eng_method,
            "engine_ms": eng_ms,
            "best_qr_codec": best_qr.get("codec"),
            "speedup": fb_ms / eng_ms,
            "quickreduce_selected": eng_method.startswith("qr_"),
        })
    return out


def _write_skip(out_dir: Path, reason: str, *, world: int, backend: str, device_type: str) -> None:
    payload = {
        "available": False,
        "reason": reason,
        "world": world,
        "backend": backend,
        "device_type": device_type,
        "rows": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "quickreduce.json", payload)
    write_csv(out_dir / "quickreduce.csv", [])
    write_csv(out_dir / "quickreduce_summary.csv", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16",
                    help="QuickReduce README documents FP16 codecs; BF16 is allowed but may fail.")
    ap.add_argument("--codecs", default="fp16",
                    help="Comma-separated QuickReduce codecs: fp16,fp8,q8,q6,q4,fp4.")
    ap.add_argument("--with-cr", action="store_true",
                    help="Add the vLLM Custom AllReduce (CR) baseline as an EXTERNAL "
                         "reference. Off by default: the production engine is native "
                         "PyTorch and does not use vLLM, so the native baseline is "
                         "torch.distributed.all_reduce (RCCL).")
    ap.add_argument("--qr-grid-cap", type=int, default=0,
                    help="Override QuickReduce max grid blocks (0 keeps library default).")
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

    def _is_set(opt: str) -> bool:
        return any(opt in a for a in sys.argv)

    warmup_val = args.warmup if _is_set("--warmup") else t_cfg.get("warmup_iters", 0) or 5
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 0) or 20
    codecs = [c.strip().lower() for c in args.codecs.split(",") if c.strip()]
    unknown = [c for c in codecs if c not in _CODECS]
    if unknown:
        raise SystemExit(f"unknown codec(s): {unknown}; valid={sorted(_CODECS)}")

    rank, world, device, backend, distributed = _setup_distributed()
    has_gpu = device.type == "cuda"
    out_dir = Path(args.out) / "14_quickreduce_comm"

    def _teardown() -> None:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    if not has_gpu:
        if rank == 0:
            _write_skip(out_dir, "CPU host: QuickReduce/RCCL comparison needs GPUs.",
                        world=world, backend=backend, device_type=device.type)
        _teardown()
        return 0
    if world < 2:
        if rank == 0:
            _write_skip(out_dir, f"world={world}: all-reduce comparison needs world>=2.",
                        world=world, backend=backend, device_type=device.type)
        _teardown()
        return 0

    try:
        shapes = [s.as_tuple() for s in
                  resolve_shapes(args.shapes, args.shape_set, path=args.shapes_file)]
    except ValueError as e:
        raise SystemExit(f"shape resolution failed: {e}")

    qr, qr_label = _make_quickreduce(world, rank)
    if args.with_cr:
        ca, cr_label = _make_custom_all_reduce(device)
    else:
        ca, cr_label = None, "custom_all_reduce not requested (native-PyTorch engine; pass --with-cr for vLLM reference)"
    integration = _probe_native_integration(qr, device) if qr is not None else None
    if rank == 0:
        print(f"[14] QuickReduce: {qr_label}")
        print(f"[14] CustomAllReduce (external ref): {cr_label}")
        print(f"[14] codecs: {','.join(codecs)} dtype={args.dtype} qr_grid_cap={args.qr_grid_cap}")
        if integration is not None:
            print(f"[14] native-pytorch integration: {integration}")

    rows: List[Dict[str, object]] = []
    for (M, K, N) in shapes:
        for label in ("rccl_all_reduce", "unfused_mm_ar"):
            try:
                if label == "rccl_all_reduce":
                    row = _bench_rccl_all_reduce(world, device, M, N, args.dtype, warmup_val, iters_val)
                else:
                    row = _bench_unfused_mm_ar(world, device, M, K, N, args.dtype, warmup_val, iters_val)
                rows.append(row)
                if rank == 0:
                    print(f"[14] {label:18s} M={M:6d} K={K:5d} N={N:5d} "
                          f"t={row['t_ms']:7.3f} ms")
            except Exception as e:  # noqa: BLE001
                rows.append({"op": label, "world": world, "M": M, "K": K, "N": N,
                             "dtype": args.dtype, "error": repr(e)})
                if rank == 0:
                    print(f"[14] {label:18s} M={M:6d} K={K:5d} N={N:5d} ERROR: {e!r}")

        if ca is None:
            rows.append({"op": "cr_all_reduce", "world": world, "M": M, "K": K, "N": N,
                         "dtype": args.dtype, "error": cr_label})
        else:
            try:
                row = _bench_cr_all_reduce(ca, world, device, M, N, args.dtype, warmup_val, iters_val)
                rows.append(row)
                if rank == 0:
                    print(f"[14] {'cr_all_reduce':18s} M={M:6d} K={K:5d} N={N:5d} "
                          f"t={row['t_ms']:7.3f} ms")
            except Exception as e:  # noqa: BLE001
                rows.append({"op": "cr_all_reduce", "world": world, "M": M, "K": K, "N": N,
                             "dtype": args.dtype, "error": repr(e)})
                if rank == 0:
                    print(f"[14] {'cr_all_reduce':18s} M={M:6d} K={K:5d} N={N:5d} ERROR: {e!r}")

        if qr is None:
            for codec in codecs:
                rows.append({
                    "op": "quickreduce_all_reduce",
                    "codec": codec,
                    "world": world,
                    "M": M,
                    "K": K,
                    "N": N,
                    "dtype": args.dtype,
                    "error": qr_label,
                })
                rows.append({
                    "op": "quickreduce_mm_ar",
                    "codec": codec,
                    "world": world,
                    "M": M,
                    "K": K,
                    "N": N,
                    "dtype": args.dtype,
                    "error": qr_label,
                })
            continue

        for codec in codecs:
            for label in ("quickreduce_all_reduce", "quickreduce_mm_ar"):
                try:
                    if label == "quickreduce_all_reduce":
                        row = _bench_quickreduce_all_reduce(
                            qr, world, device, M, N, args.dtype, codec, warmup_val, iters_val,
                            args.qr_grid_cap
                        )
                    else:
                        row = _bench_quickreduce_mm_ar(
                            qr, world, device, M, K, N, args.dtype, codec, warmup_val, iters_val,
                            args.qr_grid_cap
                        )
                    rows.append(row)
                    if rank == 0:
                        print(f"[14] {label:18s} codec={codec:4s} M={M:6d} K={K:5d} N={N:5d} "
                              f"t={row['t_ms']:7.3f} ms")
                except Exception as e:  # noqa: BLE001
                    rows.append({"op": label, "codec": codec, "world": world, "M": M, "K": K,
                                 "N": N, "dtype": args.dtype, "error": repr(e)})
                    if rank == 0:
                        print(f"[14] {label:18s} codec={codec:4s} M={M:6d} K={K:5d} N={N:5d} "
                              f"ERROR: {e!r}")

            if args.dtype == "bf16":
                try:
                    row = _bench_quickreduce_mm_ar_bf16cast(
                        qr, world, device, M, K, N, codec, warmup_val, iters_val,
                        args.qr_grid_cap
                    )
                    rows.append(row)
                    if rank == 0:
                        print(f"[14] {'qr_mm_ar_bf16cast':18s} codec={codec:4s} M={M:6d} "
                              f"K={K:5d} N={N:5d} t={row['t_ms']:7.3f} ms "
                              f"(cast={row['t_ms_cast']:.3f} ms)")
                except Exception as e:  # noqa: BLE001
                    rows.append({"op": "quickreduce_mm_ar_bf16cast", "codec": codec,
                                 "world": world, "M": M, "K": K, "N": N, "dtype": "bf16",
                                 "error": repr(e)})
                    if rank == 0:
                        print(f"[14] {'qr_mm_ar_bf16cast':18s} codec={codec:4s} M={M:6d} "
                              f"K={K:5d} N={N:5d} ERROR: {e!r}")
                try:
                    row = _bench_quickreduce_mm_ar_bf16native(
                        qr, world, device, M, K, N, codec, warmup_val, iters_val,
                        args.qr_grid_cap
                    )
                    rows.append(row)
                    if rank == 0:
                        print(f"[14] {'qr_mm_ar_bf16native':18s} codec={codec:4s} M={M:6d} "
                              f"K={K:5d} N={N:5d} t={row['t_ms']:7.3f} ms")
                except Exception as e:  # noqa: BLE001
                    rows.append({"op": "quickreduce_mm_ar_bf16native", "codec": codec,
                                 "world": world, "M": M, "K": K, "N": N, "dtype": "bf16",
                                 "qr_grid_cap": args.qr_grid_cap, "error": repr(e)})
                    if rank == 0:
                        print(f"[14] {'qr_mm_ar_bf16native':18s} codec={codec:4s} M={M:6d} "
                              f"K={K:5d} N={N:5d} ERROR: {e!r}")

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = _summary(rows)
        payload = {
            "available": True,
            "world": world,
            "backend": backend,
            "device_type": device.type,
            "quickreduce": qr_label,
            "custom_all_reduce": cr_label,
            "native_pytorch_integration": integration,
            "dtype": args.dtype,
            "codecs": codecs,
            "qr_grid_cap": args.qr_grid_cap,
            "rows": rows,
            "summary": summary,
        }
        write_json(out_dir / "quickreduce.json", payload)
        write_csv(out_dir / "quickreduce.csv", rows)
        write_csv(out_dir / "quickreduce_summary.csv", summary)
        for s in summary:
            if s["comparison"] == "engine_best_of_ar":
                print(
                    f"[14] summary {s['comparison']:32s} M={s.get('M', '')} N={s.get('N', '')} "
                    f"fallback={s.get('fallback_method')}({float(s['fallback_ms']):.3f}ms) "
                    f"engine={s.get('engine_method')}({float(s['engine_ms']):.3f}ms) "
                    f"speedup={float(s['speedup']):.3f}x"
                )
            else:
                print(
                    f"[14] summary {s['comparison']:32s} codec={s.get('codec', ''):4s} "
                    f"M={s.get('M', '')} N={s.get('N', '')} speedup={float(s['speedup']):.3f}x"
                )

    _teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
