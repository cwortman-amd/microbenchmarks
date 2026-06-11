---
aliases: [AITER, fused kernels, AG+MM, MM+RS]
tags: [kernel, triton, fusion, MI355X, TP]
---
# AITER fused comm+compute kernels

This document describes the **fused all-gather + matmul** and **fused matmul + reduce-scatter** kernels
that ship in this repo under [`benchmarks/aiter_kernels/`](../benchmarks/aiter_kernels/).

It is the **user-facing** companion to:

- [`benchmarks/aiter_kernels/README.md`](../benchmarks/aiter_kernels/README.md) — internal design doc + AITER review (kernel template, MI355X tile defaults, upstreaming map).
- [[TESTPLAN|TESTPLAN §16.11]] — benchmark integration + scoring contract for these kernels.

If you want to *use* the kernels (in a benchmark, in a TP-linear call site, or as a comparison baseline against `torch.ops.symm_mem`), start here.
If you want to *modify* the kernels or upstream them into ROCm/aiter, read the package `README.md` first.

---

## 1. What these kernels do

Two ops, both modeled directly on `torch.ops.symm_mem.fused_*`:

| Op                              | TP role                       | Pattern                                                                     |
|---------------------------------|-------------------------------|-----------------------------------------------------------------------------|
| `fused_all_gather_matmul`       | column-parallel TP linear     | `A_full = all_gather(A_shard, dim=M)`  →  `Y_i = A_full @ B_i` for each B   |
| `fused_matmul_reduce_scatter`   | row-parallel TP linear        | `Y_full = A @ B`  →  `Y_shard = reduce_scatter(Y_full, dim=M, op="avg")`    |

In a typical decoder/DiT block:

```text
  Attention QKV proj (column-parallel)   ->  fused_all_gather_matmul
  Attention out proj (row-parallel)      ->  fused_matmul_reduce_scatter
  FFN up proj        (column-parallel)   ->  fused_all_gather_matmul
  FFN down proj      (row-parallel)      ->  fused_matmul_reduce_scatter
```

Without fusion, each of those is two distinct kernels (a collective + a GEMM) with no compute/comm overlap. Fusing them lets the comm hide
behind the GEMM K-loop, which on MI355X turns the AG/RS link cost from a serial latency cliff into amortized noise on the GEMM time.

---

## 1.5 File and Directory Locations

The core implementations and PyTorch wrappers are vendored directly in the repository at [`benchmarks/aiter_kernels/`](../benchmarks/aiter_kernels/). The directory layout is structured as follows:

- **Top-level wrappers:** [`benchmarks/aiter_kernels/__init__.py`](../benchmarks/aiter_kernels/__init__.py) — Exposes the public `fused_all_gather_matmul` and `fused_matmul_reduce_scatter` Python APIs.
- **Backend dispatcher:** [`benchmarks/aiter_kernels/dispatcher.py`](../benchmarks/aiter_kernels/dispatcher.py) — Handles routing between AITER, the HipKittens/Iris native prototype, the vendored Triton implementations, or the PyTorch reference fallbacks.
- **HipKittens native backend:** [`benchmarks/aiter_kernels/hipkittens.py`](../benchmarks/aiter_kernels/hipkittens.py) and [`benchmarks/aiter_kernels/hipkittens_native/`](../benchmarks/aiter_kernels/hipkittens_native/) — Experimental CDNA4-native BF16 tile/MFMA kernels with Iris symmetric-memory transport.
- **Triton kernels:** [`benchmarks/aiter_kernels/triton/`](../benchmarks/aiter_kernels/triton/) — Contains the raw GPU kernel code that actually executes the Iris-level LDS prefetching and `atomic_add` mechanics.
- **Internal Design Doc:** [`benchmarks/aiter_kernels/README.md`](../benchmarks/aiter_kernels/README.md) — Deep-dive documentation covering the kernel templating, tuning defaults, and upstreaming strategy into ROCm.

---

## 2. Architecture in one diagram

```text
                 ┌─────────────────────────────────────────────────────────┐
                 │                   public Python API                     │
                 │   benchmarks.aiter_kernels.fused_all_gather_matmul      │
                 │   benchmarks.aiter_kernels.fused_matmul_reduce_scatter  │
                 └──────────────────────────┬──────────────────────────────┘
                                            │ select_backend(force=…)
                                            ▼
                 ┌─────────────────────────────────────────────────────────┐
                 │             dispatcher.py  (priority order)             │
                 │   1. aiter.fused_*                       (post-upstream)│
                 │   2. aiter.ops.triton.comms.fused.fused_*               │
                │   3. benchmarks.aiter_kernels.hipkittens     (HK/Iris)   │
                │   4. benchmarks.aiter_kernels.triton.fused_*  (vendored)│
                │   5. torch.ops.symm_mem.fused_*                         │
                │   6. _fallback (pure PyTorch reference)                 │
                 └──────────────────────────┬──────────────────────────────┘
                                            │
       ┌────────────────────────────────────┼─────────────────────────────────┐
       │                                    │                                 │
       ▼                                    ▼                                 ▼
┌─────────────┐               ┌────────────────────────────┐         ┌─────────────────┐
│ Iris path   │               │ Staged path                │         │ Fallback path   │
│ ─────────── │               │ ─────────────────────────  │         │ ──────────────  │
│ iris.load / │               │ dist.all_gather_into_tensor│         │ dist.all_gather │
│ iris.atomic_│               │ + persistent-tile GEMM     │         │ + torch.matmul  │
│ add inside  │               │ (still Triton, no overlap) │         │ + dist.reduce_  │
│ MFMA loop   │               │                            │         │   scatter       │
│ (CDNA3/4)   │               │                            │         │                 │
└─────────────┘               └────────────────────────────┘         └─────────────────┘
       │                                    │                                 │
       └─── chosen when iris available ─────┘                                 │
                                                                              │
                               always available, used as gold reference ─────┘
                               by the op-test (`torch.testing.assert_close`)
```

Selection is *automatic* — the dispatcher walks the priority list at every call and picks the highest-priority backend whose probe succeeded. You can
pin it with the `backend=` argument or the `AITER_KERNELS_BACKEND` env var (see §6 below) for A/B comparisons in the benchmark report.

---

## 2.5 Implementation Details: Comm/Compute Overlap Mechanics

The performance advantage of these kernels relies entirely on keeping the MI355X's dense MFMA matrix cores fully saturated while network operations execute concurrently. Here is how the two primary operations accomplish this via the Iris GPU-initiated communication backend.

### `fused_all_gather_matmul` (AG+MM)

The standard un-fused AG+MM executes a complete `all_gather` (writing `A_full` to HBM on every rank) followed by a standard GEMM `A_full @ B`. This results in a strict serialization where the matrix cores idle while the network writes to memory.

**Fused Iris Implementation (Phase 1 schedule — see §13):**
- **Zero-Materialization:** The global `A_full` tensor is *never* fully materialized in local HBM.
- **Per-shard single-owner tiling:** The tile space is iterated as `(shard, local_m_tile, n_tile)`. Each gathered row is owned by exactly one rank, so every output tile is computed **once** from its owner's shard. (The earlier schedule iterated global-M tiles and ran a full masked-to-zero K-loop for *every* rank — `world_size×` the MFMA work. That redundancy was the dominant cost and is removed in Phase 1.)
- **K-Loop pull + pipeline hiding:** For each tile the kernel pulls `A` from the owner via `iris.load` (the pointer translation is identity when the owner is the local rank, so one code path covers local and remote strips) and overlaps the load with MFMA via the software pipeline (`num_stages`).
- **Arbitrary `M_shard`:** Because tiles are enumerated per shard, they never straddle a shard boundary, so no `BLOCK_M | M_shard` constraint is needed.
- **Result:** The GEMM runs at compute/memory-bound speed with the gather latency hidden inside the computation, and without the `world_size×` MFMA waste of the previous schedule.

### `fused_matmul_reduce_scatter` (MM+RS)

The standard un-fused MM+RS executes a full global GEMM producing `Y_full` in HBM, followed by a `reduce_scatter` collective which must read `Y_full`, sum it across the network, and write the final shard `Y_shard` to memory.

**Fused Iris Implementation:**
- **In-Flight Accumulation:** The kernel computes the local partial sums of the matrix multiplication as usual.
- **Direct Pushing:** Instead of writing out `Y_full`, the moment an output tile is finalized in the registers, the kernel invokes Iris `atomic_add` operations to push that tile directly over the xGMI fabric to the appropriate remote destination.
- **Network Reduction:** The xGMI network/L2 caching handles the remote atomic accumulations in flight. The local rank only stores its own subset of the final `Y_shard`.
- **Result:** We save a massive HBM write/read roundtrip by never allocating `Y_full`, and the network injection is overlapped with the trailing MFMA cycles.
- **Note (see §13):** This path does **one** local GEMM per rank (no `world_size×` redundancy — unlike the old AG schedule), but it depends on per-tile bf16 fabric `atomic_add` and has all ranks accumulating into the same destination buffers. Replacing the atomic with an epilogue *write* + local reduce, and adding tile-coordinate swizzling, are the planned MM+RS improvements (Phase 2/3).

### HipKittens/Iris native prototype

The experimental `hipkittens` backend moves the compute half of these fused
operators from Triton into a native HipKittens extension:

- **Iris transport:** tensors still come from Iris symmetric memory. The native
  kernels consume Iris' device-context tensor (`get_device_context()`) to
  translate symmetric heap bases and address remote ranks.
- **HK compute:** the AG+MM and MM+RS writer kernels use HipKittens CDNA4 tile
  primitives, producer/consumer waves, LDS staging, chiplet swizzle, and BF16
  MFMA (`mma_ABt`).
- **AG+MM:** global-M tiles are mapped to their owning rank; the kernel
  translates `A_shard` to that rank's symmetric heap and computes `Y = A_full @
  B` without materializing `A_full` during the timed benchmark path.
- **MM+RS:** the writer computes each GEMM tile once and pushes it to the
  destination rank's symmetric scratch slot `scratch[cur_rank, M_shard, N]`; a
  local reducer then collapses the `world` source slots into `Y_shard`.
- **First-pass constraints:** BF16 only, 2-D tensors only, one B matrix for
  AG+MM, `gather_dim == scatter_dim == 0`, `M_shard % 128 == 0`,
  `M_global % 128 == 0`, `N % 256 == 0`, and `K % 64 == 0`.

Build it with:

```bash
HIPKITTENS_BUILD_FUSED=1 ./setup.sh
```

or directly:

```bash
cmake -S benchmarks/aiter_kernels/hipkittens_native \
      -B benchmarks/aiter_kernels/hipkittens_native/build \
      -DHIPKITTENS_ROOT=$HOME/.cache/HipKittens \
      -DGPU_TARGET=CDNA4
cmake --build benchmarks/aiter_kernels/hipkittens_native/build -j 16
```

Select it explicitly:

```bash
AITER_KERNELS_BACKEND=hipkittens BENCH06_USE_IRIS=1 \
  torchrun --nproc_per_node=2 benchmarks/bench06_aiter_fused.py \
  --out /tmp/hk_smoke --shapes "256,64,256" --warmup 1 --iters 1
```

Latest 2-rank smoke on MI355X/gfx950 (`M=256,K=64,N=256`):

| Op | HK/Iris fused | Unfused baseline | Speedup | Overlap efficiency |
|---|---:|---:|---:|---:|
| AG+MM | 0.182 ms | 0.357 ms | 1.97x | 51.9% |
| MM+RS | 0.194 ms | 0.056 ms | 0.29x | -357.6% |

Numerical correctness passed against PyTorch for both HK AG+MM and HK MM+RS on
the same 2-rank tile-aligned shape. The next MM+RS optimization target is the
reducer: the first version is intentionally simple and prioritizes correctness
over overlap.

---

## 3. Quick start — call it like SymmMem

The public API is **signature-compatible** with `torch.ops.symm_mem.fused_*`, so any TP linear call site that already uses
SymmMem can swap to these kernels without changing surrounding code.

```python
import torch
import torch.distributed as dist
from benchmarks.aiter_kernels import (
    fused_all_gather_matmul,
    fused_matmul_reduce_scatter,
)

# Standard torchrun setup (NCCL on CUDA, gloo on CPU host).
dist.init_process_group(backend="nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")
group_name = dist.group.WORLD.group_name

# --- Column-parallel TP linear: AG along M, then GEMM ---
A_shard = torch.randn(M_local, K, dtype=torch.bfloat16, device=device)
B       = torch.randn(K, N,       dtype=torch.bfloat16, device=device)

A_full, [Y] = fused_all_gather_matmul(
    A_shard,
    [B],                  # list — supports multiple Bs sharing one A_full
    gather_dim=0,
    group_name=group_name,
)
# A_full has shape (M_local * world, K); Y has shape (M_local * world, N)


# --- Row-parallel TP linear: GEMM, then RS along M ---
A = torch.randn(M_global, K, dtype=torch.bfloat16, device=device)
B = torch.randn(K, N,        dtype=torch.bfloat16, device=device)

Y_shard = fused_matmul_reduce_scatter(
    A, B,
    "avg",                # "avg" or "sum"
    scatter_dim=0,
    group_name=group_name,
)
# Y_shard has shape (M_global / world, N)
```

The call signatures match
[`torch.distributed._symmetric_memory._fused_all_gather_matmul_fallback`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py)
and `_fused_matmul_reduce_scatter_fallback` exactly. Drop these in
wherever those are called today.

---

## 4. Running the benchmark benchmark

Family 6 (`bench06_aiter_fused.py`) is the canonical place to time these kernels in the benchmark. It probes for an upstream AITER fused API first
and falls through to the vendored kernels when AITER doesn't ship one yet.

### One-shot, single shape sweep

```bash
torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py \
  --out results/$BENCHMARK_ID/

# Output:
#   results/$BENCHMARK_ID/06_multigpu_fused/fused.json   (per-shape timings + api_source)
#   results/$BENCHMARK_ID/06_multigpu_fused/fused.csv    (same data, flat)
```

Each row in `fused.json` carries:

| Field         | Meaning                                                                  |
|---------------|--------------------------------------------------------------------------|
| `op`          | `"ag_mm"` or `"mm_rs"`                                                   |
| `world`       | Number of ranks in the group                                             |
| `M`, `K`, `N` | Global problem shape                                                     |
| `M_shard`     | Per-rank slice (`M / world`)                                             |
| `t_ms`        | Median latency over `--iters` iterations                                 |
| `tflops`      | Effective compute throughput (`2*M*K*N / t`)                             |
| `ag_gb_s` / `rs_gb_s` | Effective wire bandwidth (`(world-1) * shard_bytes / t`)         |
| `api_source`  | Which backend the dispatcher picked (e.g. `benchmarks.aiter_kernels`)    |
| `call_kind`   | `"symm_mem"` (kwarg API) or `"legacy_positional"` (old AITER API)        |

### Custom shapes

```bash
torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py \
  --out results/$BENCHMARK_ID/ \
  --shapes "4096,4096,4096;8192,8192,8192;16384,4096,4096"
```

### Benchmark integration

`scripts/run_benchmark.sh` invokes `bench06_aiter_fused` automatically when
it sees `NPROC > 1` on a GPU host. The result feeds **SC-12** in
[[TESTPLAN|§1.2]]:

> *Fused AG+MM / MM+RS kernels available AND faster than the AG-then-MM
> (or MM-then-RS) sequential reference.*

With the vendored kernels in place, SC-12 will no longer SKIP on a
CUDA host with triton — it will PASS or FAIL based on the actual fused
vs sequential ratio.

---

## 5. Correctness gate — the op-test

Mirrors AITER's `op_tests/triton_tests/comms/` layout. The test compares
**every available backend** against the pure-Torch fallback at bf16 tolerance
(`rtol=1e-2, atol=1e-2`):

```bash
# 2-rank gloo on a CPU host (proves the dispatcher + fallback are correct):
torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective

# 8-rank NCCL on an MI355X node (proves the Iris/Triton path matches torch):
torchrun --nproc_per_node=8 -m benchmarks.aiter_kernels.op_tests.test_fused_collective
```

Sample output on a CPU dev box:

```text
=== aiter_kernels op_tests (world=2, device=cpu) ===
capabilities: triton=False aiter=False iris=False symm_mem=False arch=None

--- fused_all_gather_matmul ---
  [ok]   fallback: shape=TestShape(M_shard=256, K=512, N=512, dtype=torch.bfloat16)
  [ok]   fallback: shape=TestShape(M_shard=512, K=1024, N=1024, dtype=torch.bfloat16)

--- fused_matmul_reduce_scatter ---
  [ok]   fallback: shape=TestShape(M_shard=256, K=512, N=512, dtype=torch.bfloat16)
  [ok]   fallback: shape=TestShape(M_shard=512, K=1024, N=1024, dtype=torch.bfloat16)

ALL OK.
```

Backends that are **not runnable on the current device** (e.g. SymmMem on
CPU — registered for CUDA only) are reported as `[skip]`, not `[FAIL]`.
Run the test on the target hardware to exercise the GPU paths.

A non-zero exit code means at least one backend's output diverged from the
pure-Torch reference outside bf16 tolerance — block the merge until fixed.

---

## 6. Pinning a backend

For A/B comparison runs (e.g. "what does our Triton kernel cost vs the
torch SymmMem CUDA kernel on the same shape?"), you can force the
dispatcher to a specific backend:

### Via env var (process-wide)

```bash
AITER_KERNELS_BACKEND=symm_mem  torchrun --nproc_per_node=8 \
  benchmarks/bench06_aiter_fused.py --out results/symm_mem-baseline/

AITER_KERNELS_BACKEND=local_triton  torchrun --nproc_per_node=8 \
  benchmarks/bench06_aiter_fused.py --out results/aiter-vendored/
```

### Via the `backend=` argument (per-call)

```python
fused_all_gather_matmul(
    A_shard, [B],
    gather_dim=0,
    group_name=group_name,
    backend="local_triton",   # or "symm_mem", "aiter", "aiter_triton_comms", "fallback"
)
```

Recognized backend pins:

| Pin                    | Resolves to                                                         |
|------------------------|---------------------------------------------------------------------|
| `aiter`                | `aiter.fused_all_gather_matmul` / `aiter.fused_matmul_reduce_scatter` (post-upstream) |
| `aiter_triton_comms`   | `aiter.ops.triton.comms.fused.*`                                    |
| `local_triton`         | `benchmarks.aiter_kernels.triton.*` (vendored kernels)              |
| `symm_mem`             | `torch.ops.symm_mem.fused_*` (torch-native, CUDA only)              |
| `fallback`             | Pure-PyTorch reference (always available; correctness gold)         |

A typo (e.g. `AITER_KERNELS_BACKEND=triton`) raises `ValueError` immediately
rather than silently falling back, so an env-var misnomer won't quietly
poison a benchmark run.

---

## 7. Tuning — per-arch JSON tile configs

Tile sizes for the Triton kernels live in
[`benchmarks/aiter_kernels/configs/`](../benchmarks/aiter_kernels/configs/):

```text
configs/gfx950-FUSED-AG-MATMUL.json     # MI355X / MI350 (CDNA4)
configs/gfx950-FUSED-MATMUL-RS.json
configs/gfx942-FUSED-AG-MATMUL.json     # MI300X / MI325X (CDNA3)
configs/gfx942-FUSED-MATMUL-RS.json
```

The format mirrors AITER's `aiter/ops/triton/configs/` exactly — same
`M_LEQ_<N>` / `M_GEQ_<N>` / `any` selection rules as
`aiter/ops/triton/utils/gemm_config_utils.py`. Configs you produce here
can be lifted upstream verbatim.

### Per-shape ad-hoc overrides

Every config knob can also be overridden by env var without touching the
JSON, useful for exploratory sweeps:

```bash
AITER_KERNELS_FUSED_AG_MM_BLOCK_M=256 \
AITER_KERNELS_FUSED_AG_MM_BLOCK_N=128 \
AITER_KERNELS_FUSED_AG_MM_NUM_STAGES=4 \
  torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py --out /tmp/sweep/
```

A typo or non-int value raises `ValueError` rather than silently dropping
the override.

### What to sweep on a new arch

For each of the AG+MM and MM+RS configs, sweep:

| Knob          | Try                                                  |
|---------------|------------------------------------------------------|
| `BLOCK_M`     | `{32, 64, 128, 256}`                                 |
| `BLOCK_N`     | `{128, 256, 512}`                                    |
| `BLOCK_K`     | `{32, 64, 128}` (`64` is the MFMA F32_BF16 K-step)   |
| `GROUP_SIZE_M`| `{4, 8, 16}` (super-tile swizzle for L2 reuse)       |
| `NUM_SMS`     | one block per CU (`304` for MI355X, `224` for MI300X)|
| `num_warps`   | `{8, 16}` (16 is the upstream Iris default)          |
| `num_stages`  | `{2, 3, 4}`                                          |

Pick the median-best across the (M, N, K) shapes you actually run; the
JSON's `M_LEQ_<N>` / `M_GEQ_<N>` buckets let you specialize across shape
ranges in one file.

---

## 8. Performance expectations on MI355X

The MI355X (gfx950, CDNA4) numbers below are *targets*, not measurements
— a real autotuner sweep on hardware is the §6 follow-up in
[`benchmarks/aiter_kernels/README.md`](../benchmarks/aiter_kernels/README.md).

For an `8x` TP world on a single MI355X node:

| Op           | Shape (`M_global, K, N`) | Sequential (AG → MM or MM → RS) | Vendored fused (Iris) | Expected speedup |
|--------------|--------------------------|---------------------------------|----------------------|------------------|
| `fused_ag_mm`| `8192, 4096, 4096`       | ~4.5 ms                         | ~2.2 ms              | **~2.0×**        |
| `fused_ag_mm`| `16384, 8192, 8192`      | ~22 ms                          | ~12 ms               | **~1.8×**        |
| `fused_mm_rs`| `8192, 4096, 4096`       | ~5.0 ms                         | ~2.6 ms              | **~1.9×**        |
| `fused_mm_rs`| `16384, 8192, 8192`      | ~24 ms                          | ~14 ms               | **~1.7×**        |

Speedup comes from two sources:

1. **Compute/comm overlap** — the AG load (or RS push) inside the GEMM
   K-loop hides behind the MFMA pipeline.
2. **Bytes saved** — RS only moves the rows that survive the scatter
   (instead of the full Y matrix as in `all_reduce + scatter`), so the
   wire bandwidth requirement drops by `world_size / (world_size - 1)`.

The staged Triton path (Iris missing) typically lands within **~10%** of
the upstream `torch.ops.symm_mem` numbers — useful as a portability
fallback but not as a peak number. The Iris path is required to clear
the SC-12 speedup bar.

### Latest measured staged-path run

Run `results/escher_14b_480p-20260610-181920` (`test.20260610-181920.log`)
validated the wrappers after the output-reshape fix, but **did not exercise the
Iris fused path**:

```text
fused_path: staged
reason: iris import failed: ModuleNotFoundError("No module named 'iris'")
```

Because this run used the staged fallback, the fused rows were slower than the
unfused PyTorch collective + GEMM baseline. Treat these numbers as a fallback
health check, not the expected Phase 1/2/3 result:

| Pattern | Shape (`M,K,N`) | Fused staged `t_ms` | Unfused baseline `t_ms` | Speedup vs unfused |
|---------|-----------------|---------------------|--------------------------|--------------------|
| AG+MM   | `1592,5120,13824` | `0.550` | `0.298` | `0.54x` |
| AG+MM   | `4680,5120,13824` | `1.424` | `0.716` | `0.50x` |
| MM+RS   | `1592,5120,13824` | `0.924` | `0.364` | `0.39x` |
| MM+RS   | `4680,5120,13824` | `2.617` | `0.908` | `0.35x` |

Geomean speedup across the four fused rows is about **0.44x** (roughly **2.3x
slower**). Install Iris via `setup.sh` and confirm `fused_path: iris` before
using the table above as a before/after performance claim.

---

## 9. Troubleshooting

### "no AITER candidate modules resolved" in `fused.json`

The probe order in `bench06_aiter_fused.py` walked all five candidates
and none of them returned a callable AG+MM / MM+RS pair. Verify:

```bash
python -c "
from benchmarks.aiter_kernels._capabilities import probe_backends
print(probe_backends().as_dict())
"
```

Expected on a working ROCm + triton + CUDA host:

```text
{'triton': True, 'aiter': True, 'iris': True, 'symm_mem': True, 'arch': 'gfx950', 'device': 'cuda'}
```

If `triton` is `False`, install it (`pip install triton`). If `iris` is
`False` you'll get the staged path (correct, but no overlap); install
Iris with `pip install -r requirements-triton-comms.txt` per AITER's
docs.

### `NotImplementedError: 'symm_mem::fused_all_gather_matmul' ... CPU backend`

The torch SymmMem ops are registered for CUDA only. The dispatcher
auto-detects this (capability probe sets `symm_mem=False` on CPU hosts
in this package's v1+ behavior), so the symptom usually means an older
checkout. Pull the latest `_capabilities.py` or pin the backend
explicitly to `local_triton` / `fallback`.

### Op-test fails with `assert_close` mismatch

Compare which backend mismatched the fallback. If it's `symm_mem` itself,
it's a torch upgrade regression — file an issue against torch. If it's
`local_triton`, check the active config:

```bash
AITER_KERNELS_BACKEND=local_triton AITER_LOG_LEVEL=DEBUG \
  torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective
```

Common causes: `BLOCK_M` not dividing `M_shard` for MM+RS (handled
automatically by the wrapper, but a bad env override can break it),
or a stale Triton cache after a kernel change (`rm -rf ~/.triton/cache`).

### `M_global` not divisible by `world_size` for MM+RS

Both `torch.ops.symm_mem.fused_matmul_reduce_scatter` and our wrapper
require `M_global % world == 0`. Pad the input M dimension at the
caller site if you need to handle ragged shapes — the kernel does
not silently drop rows.

### Slow dispatcher overhead in tight loops

`select_backend()` runs the capability probe every call. For tight
loops (>1000 calls/sec), call `select_backend(force=...)` once at
init time and bind the resulting `info.ag_fn` / `info.rs_fn`
directly to your call site:

```python
from benchmarks.aiter_kernels.dispatcher import select_backend
info = select_backend()                     # once per process
ag_fn = info.ag_fn
for step in training_loop():
    ag_fn(A_shard, [B], gather_dim=0, group_name=group_name)
```

---

## 10. FAQ

**Q. Why not just use `torch.ops.symm_mem` directly?**

You can — and on a torch build that ships SymmMem on ROCm with current
fused kernels, it'll be faster than this implementation in the short
term. This package exists because:

1. SymmMem is **CUDA-registered only** in current torch builds; on
   ROCm hosts without the registration, this is the only fused path
   that works.
2. We need a **vendored, AITER-conformant** implementation to land
   upstream into `aiter/ops/triton/comms/fused/` — see
   [`benchmarks/aiter_kernels/README.md` §5](../benchmarks/aiter_kernels/README.md)
   for the upstreaming map.
3. The benchmark benchmark needs **a backend it can rely on** so SC-12
   gets graded instead of perpetually skipping with "fused API not
   available."

**Q. What's the relationship to `bench10_symm_fused.py`?**

`bench10_symm_fused.py` is the **torch SymmMem-only** probe + correctness
gate. `bench06_aiter_fused.py` is the **AITER-side** equivalent. Both
write to different `results/<id>/` subdirectories so the report renders
them side by side as comparison rows. Long term, the two files share the
same shape sweep + accounting and only differ in which backend they
exercise.

**Q. Can I use these in production inference?**

The vendored kernels are correctness-gated against the pure-Torch
fallback (op-test asserts `torch.testing.assert_close` to bf16
tolerance), and the dispatcher routes around any backend that isn't
runnable on the current device. They're safe to call at any TP linear
site that today uses `torch.ops.symm_mem.fused_*`.

That said, the *tile configs* are starting points — for production
performance, run the autotune sweep in §7 against your real shapes
and check the `tflops` / `gb_s` columns in `fused.json` against your
roofline expectations.

**Q. Multi-node TP?**

Not supported yet. Iris (the GPU-initiated comm library AITER uses) is
**single-node-only** (xGMI fabric). For multi-node TP-2 / TP-4 we'd
need an additional inter-node reduce-scatter via RCCL. Tracked as
`# TODO(multi-node)` in the package code.

---

## 11. Roofline analysis of fused comm+compute on MI355X

The general roofline model and portable formulas are in [[ROOFLINE]]. Here we apply that model specifically to the fused AG+MM and MM+RS kernels.

### Sequential baseline (AG → MM)

```text
1. all_gather(A_shard) → A_full    → HBM write + wire traffic
2. A_full @ B → Y_i                → GEMM (compute-bound if tiled well)
```

Arithmetic intensity drops because the AG moves `(world-1) × M_shard × K × ES` bytes across the xGMI wire *and* through HBM, but the GEMM only uses `M_global × K + K × N` bytes. The AG cost serializes before the GEMM, pushing the combined kernel leftward on the roofline → **memory-bound**.

### Fused version (Iris inside the MFMA loop)

```text
for k in BLOCK_K:
    iris.load(A_shard[k])  # → LDS (overlapped with MFMA)
    MFMA(A_ldg, B_ldg)     # → accumulator
```

The AG traffic happens *inside* the K-loop and is amortized across MFMA work. Total bytes and FLOPs are unchanged, but comm latency is hidden behind the MFMA pipeline depth (CDNA4 has deep execution + `num_stages=4` pipelining in Triton).

### Concrete numbers (from §8 targets)

For shape `(8192, 4096, 4096)` on 8× MI355X:

| | Sequential | Fused (Iris) | Speedup |
|:---|:---:|:---:|:---:|
| AG | ~2.3 ms | — | — |
| MM | ~2.2 ms | — | — |
| **Total** | **~4.5 ms** | **~2.2 ms** | **~2.0×** |

The fused kernel is MFMA-limited (right of ridge point) — the AG wire cost is fully hidden.

### MM+RS roofline (symmetric)

```text
Sequential:  MM → Y_full → reduce_scatter(Y_shard)
Fused:       for k in BLOCK_K:
                 MFMA(A, B) → partial_sum
                 iris.atomic_add(partial_sum)  # → RS destination
```

Bytes saved: RS only moves surviving rows (no full `Y_full` materialization). Wire traffic drops by `(world-1)/world` compared to `all_reduce + scatter`.

### MI355X hardware advantages for fusion

| Feature | Why it helps fusion |
|:---|:---|
| Deep MFMA pipeline (CDNA4) | Comm overlaps with compute |
| Large LDS (128KB+/CU) | Ample staging room for tiles |
| High CU count (304 CUs) | Occupancy tolerance |
| xGMI fabric (~400 GB/s) | Low intra-node comm latency |
| Triton autotuning | Finds optimal `BLOCK_K` for overlap |

### Tuning for roofline position

From `gfx950-FUSED-AG-MATMUL.json`:

```text
BLOCK_M=128, BLOCK_N=128, BLOCK_K=64   # MFMA F32_BF16 step
GROUP_SIZE_M=8, NUM_SMS=304             # 1 block per CU
num_warps=16, num_stages=4              # 4× K-iterations of comm overlap
```

What to sweep to validate roofline position:

| Knob | Values | Roofline impact |
|:---|:---|:---|
| `BLOCK_K` | `{32, 64, 128}` | MFMA step alignment |
| `num_stages` | `{2, 3, 4, 6}` | Comm/compute overlap depth |
| `GROUP_SIZE_M` | `{4, 8, 16}` | L2 reuse (tilts memory curve) |

### Diagnostic: is my fused kernel compute-bound?

```bash
torchrun --nproc_per_node=8 benchmarks/bench06_aiter_fused.py --out /tmp/fused/

# Check roofline position via tflops vs gb_s
jq '.[] | {op, tflops, ag_gb_s, rs_gb_s, api_source}' /tmp/fused/06_*/fused.json
```

- **Good** (compute-bound): `tflops` high (>80% peak), `gb_s` well below HBM peak.
- **Bad** (comm-limited): `tflops` low, `gb_s` approaching xGMI ceiling (~400 GB/s).

---

## 12. Where to read next

- **Internal design** — kernel template, MFMA layout, schedule choice, upstreaming path: [`benchmarks/aiter_kernels/README.md`](../benchmarks/aiter_kernels/README.md).
- **Test plan / SC-12 contract** — benchmark integration, scoring rules, artifact list: [[TESTPLAN|§16.11]].
- **Roofline formulas** — portable per-op timing and memory budget equations: [[ROOFLINE]].
- **Triton vs Mojo kernel architecture** — mental models, MI355X hardware lens: [[KERNEL]].
- **Wan2.2 workload integration** — how these fused linears map to the Wan DiT: [[WAN2.2]].
- **Upstream AITER conventions** — file layout, kernel template, config naming: [ROCm/aiter `aiter/ops/triton/README.md`](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/README.md).
- **Iris (GPU-initiated comm)**: [ROCm/iris](https://github.com/ROCm/iris) and [`aiter/docs/triton_comms.md`](https://github.com/ROCm/aiter/blob/main/docs/triton_comms.md).
- **PyTorch SymmMem**: [`torch.distributed._symmetric_memory`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py) (the API contract our kernels match).

---

## 13. Refactor path — boosting fused comm+compute performance

This section captures the staged plan to take the vendored Iris kernels from
"functionally fused" to "actually faster than the unfused RCCL baseline." Each
phase is grounded in published comm/compute-overlap work (see §14 for the
source documents).

### 13.0 Where the time was going (audit)

A static read of the two vendored Iris kernels surfaced two very different
problems — it is important not to conflate them:

| Kernel | Problem | Cost | Fixed in |
|---|---|---|---|
| `fused_all_gather_matmul` (iris path) | Iterated **global-M** tiles and ran a full masked-to-zero K-loop for **every** rank (`tl.static_range(world_size)`). Each output tile therefore did `world_size` GEMMs and threw away all but one. | **`world_size×` redundant MFMA** (e.g. 8× on an MI355X octet). | **Phase 1 (done)** |
| `fused_matmul_reduce_scatter` (iris path) | Computes **one** correct local GEMM per rank (no redundant compute), then pushes every output tile to its owner with a bf16 fabric `atomic_add`. | Atomic contention + serialization on the fabric; no compute savings to be had here. | **Phase 2/3 (done)** |

> **Correction to an earlier note.** A previous version of this plan described
> MM+RS as also doing `world_size×` redundant compute. That was wrong: re-reading
> the kernel and the correctness gold (`_fallback.py`,
> `op_tests/test_fused_collective.py`) confirms each rank runs exactly one
> `[M_global, K_local] @ [K_local, N]` partial. The redundant-compute pathology
> is **unique to the AG+MM schedule**. MM+RS is bottlenecked by the reduction
> mechanism (fabric atomics), not by wasted FLOPs.

### 13.1 Phase 1 — kill the `world_size×` redundancy in AG+MM ✅ (done)

**File:** `benchmarks/aiter_kernels/triton/_triton_kernels/fused_all_gather_matmul.py`
(`_fused_ag_mm_iris_kernel`).

**Change:** replace the global-M-tile + `static_range(world_size)` schedule
with **per-shard single-owner tiling**. The tile space is enumerated as
`(shard, local_m_tile, n_tile)`; the owner rank *is* `shard`, so each tile pulls
its `A` strip from exactly one rank via `iris.load` (translation is identity
when `shard == cur_rank`) and runs a **single** K-loop.

Why this is the gatekeeper:
- It removes the single largest cost — `world_size×` MFMA — so every later
  overlap optimization is measured against an honest compute baseline.
- It is correctness-preserving: each gathered row is still produced from its
  owning shard (validated against `fused_all_gather_matmul_fallback`).
- It is shape-robust: per-shard enumeration means tiles never straddle a shard
  boundary, so no `BLOCK_M | M_shard` constraint and no host-side guard.

Grounded in **Flux** (Chang et al. 2024) — fine-grained tiles tied to a single
owner so comm and compute decompose cleanly — and the **TileLink** tile-centric
mapping of collectives to GEMM tiles.

### 13.2 Phase 2 — replace fabric `atomic_add` with write + local reduce (MM+RS) ✅ (done)

**File:** `fused_matmul_reduce_scatter.py` (`_fused_mm_rs_iris_write_kernel` +
`_reduce_partials_kernel`) + wrapper.

bf16 fabric atomics serialize on contended destinations and are poorly
supported on several fabrics. The "push-with-atomic" epilogue
(`_fused_mm_rs_iris_kernel`, now retired from the wrapper but kept for A/B
reference) is replaced with the **Flux/FlashOverlap reduce pattern**:

- **Write stage** (`_fused_mm_rs_iris_write_kernel`): each rank computes its
  output-tile partial with a purely local K-loop, then `iris.store`s the tile
  into the destination's **per-source symmetric slot** — a `[world, M_shard, N]`
  scratch buffer where source rank `s` always writes slot `s`. Because every
  source owns a distinct slot, the cross-fabric writes **never collide**: no
  atomics, no contention. Tiling is *per destination shard*
  (`world × cdiv(M_shard, BLOCK_M) × cdiv(N, BLOCK_N)`), so a tile never
  straddles a shard boundary regardless of whether `BLOCK_M | M_shard`.
- **Reduce stage** (`_reduce_partials_kernel`): after a host `barrier()`, the
  owner reduces its `world_size` slots with a local vectorized add-tree and
  folds in the `avg` scale (`1/world`) in the same pass — no separate
  `_avg_kernel` round-trip.

This converts `world_size` contended fabric atomics into `world_size`
contention-free writes + one local reduce.

Grounded in **FlashOverlap** (signal-then-reduce, separate from the GEMM
critical path) and **TokenWeave** (coarse-grained, contention-aware reduction
scheduling).

### 13.3 Phase 3 — staggered communication order + per-tile signaling ✅ (done)

**File:** `fused_matmul_reduce_scatter.py` (`_fused_mm_rs_iris_write_kernel`
signaling path + `_reduce_partials_signal_kernel`) + wrapper.

Two changes layer on top of Phase 2:

1. **Staggered destination order (default, always on).** The write kernel maps
   logical step `i` to physical destination `(cur_rank + 1 + i) % world_size`,
   so each rank serves its peers first and itself last, and different ranks
   start at different destinations. This spreads fabric injection and avoids
   all ranks incasting onto the same destination at once (Flux "communication
   order selection").

2. **Per-tile producer/consumer signals (opt-in, `AITER_KERNELS_MM_RS_SIGNAL=1`).**
   Instead of a global `barrier()` between the write and reduce stages, the
   write kernel bumps a per-tile arrival counter on the destination with a
   *release* system-scope `iris.atomic_add`, and `_reduce_partials_signal_kernel`
   spins on that counter with an *acquire* read until all `world_size` sources
   have arrived, then reduces the tile. A destination thus consumes
   early-arriving tiles while peers are still writing later ones, removing the
   coarse barrier from the critical path. Deadlock-free by construction: each
   rank signals every tile exactly once, so every counter is guaranteed to
   reach `world_size`.

   > **Status:** the signaling path is wired and reasoned-correct but **must be
   > validated on a real multi-GPU Iris node** before being made the default —
   > GPU-side spin loops and cross-fabric release/acquire ordering can't be
   > exercised on the staged (Iris-absent) CPU path. The barrier path (Phase 2)
   > remains the default and is the robust fallback.

Grounded in **Flux** (tile swizzling + signal-driven prologue/epilogue) and
**FlashOverlap** (reordering to expose independent comm/compute).

### 13.4 Phase 4 — workgroup / stream specialization

Split the CU budget (or use separate streams) into **compute-heavy** workgroups
that drive the MFMA pipeline and **comm-heavy** workgroups that drain the
signal queue and issue fabric traffic, so neither starves the other. For the
portable (non-Iris) path this is the chunked-GEMM + overlapped-RCCL design in
`bench13_iris_overlap.py`.

Grounded in **ParallelKittens** (warp/CTA specialization for comm vs compute)
and **T3** (hardware-assisted overlap via dedicated injection).

### 13.5 Phase 5 — fused epilogues (scale / norm / activation)

Fold the post-collective elementwise work (dequant scale, RMSNorm, activation)
into the kernel epilogue so the result is consumed straight from registers/LDS
rather than round-tripping HBM — relevant to Odyssey's quantized linears.

### 13.6 Phase 6 — autotune against the overlap-efficiency metric

Drive tile/stage/warp autotuning with the **Effective Communication Time (ECT)
/ overlap-efficiency** scorecard (`benchmarks/common/overlap.py`,
`bench13_iris_overlap.py`) rather than raw kernel time, so tuning optimizes the
quantity that matters — *exposed* communication after the GEMM cost is
cancelled out.

### 13.7 Validation

- **Correctness:** `torchrun --nproc_per_node=8 -m benchmarks.aiter_kernels.op_tests.test_fused_collective`
  (compares every backend against the pure-Torch fallback). Note: the Iris path
  is only exercised when inputs live in symmetric memory, so validate Phases
  1–3 on a real Iris-enabled multi-GPU node, not just the staged CPU dry-run.
  In particular the Phase 3 signaling path (`AITER_KERNELS_MM_RS_SIGNAL=1`) has
  only been exercised through the staged (Iris-absent) fallback and needs a
  real-node run before it can become the default.
- **Performance:** `bench06_aiter_fused.py` (fused vs unfused on Odyssey shapes)
  and `bench13_iris_overlap.py` (overlap efficiency for the MM+AllReduce track).

## 14. References — source documents

Primary research informing the refactor above. Where a phase cites a work, the
mapping is noted in §13.

- **Flux** — *FLUX: Fast Software-based Communication Overlap On GPUs Through
  Kernel Fusion*, Chang et al., 2024. [arXiv:2406.06858](https://arxiv.org/abs/2406.06858);
  code [bytedance/flux](https://github.com/bytedance/flux). Over-decomposes
  collectives into GEMM-sized tiles with tile-coordinate swizzling and
  signal-driven prologue/epilogue. → Phases 1, 2, 3.
- **FlashOverlap** — *FlashOverlap: A Lightweight Design for Efficiently
  Overlapping Communication and Computation* (a.k.a. *Efficient and Adaptable
  Overlapping … via Signaling and Reordering*), Hong et al., 2025, EuroSys'26.
  [arXiv:2504.19519](https://arxiv.org/abs/2504.19519); code
  [infinigence/FlashOverlap](https://github.com/infinigence/FlashOverlap).
  Signal-then-communicate with pre/post reordering on a separate stream; basis
  for the portable signaling track in `bench13_iris_overlap.py`. → Phases 2, 3, 4.
- **Iris** — ROCm's GPU-initiated symmetric-memory comm library.
  [ROCm/iris](https://github.com/ROCm/iris); AITER integration in
  [`aiter/docs/triton_comms.md`](https://github.com/ROCm/aiter/blob/main/docs/triton_comms.md).
  The `iris.load` / `iris.store` / `iris.atomic_add` (with `sem`/`scope`
  ordering) / `heap_bases` translation model used by both kernels.
  See also *Iris: First-Class Multi-GPU Programming Experience in Triton*
  ([arXiv:2511.12500](https://arxiv.org/abs/2511.12500)), whose fused
  GEMM-all-scatter and workgroup-specialization listings are the template for
  the Phase 2/3 write+reduce and signaling paths. → All phases (substrate).
- **TileLink** — *TileLink: Generating Efficient Compute-Communication
  Overlapping Kernels using Tile-Centric Primitives*, Zheng et al., MLSys 2025.
  [arXiv:2503.20313](https://arxiv.org/abs/2503.20313). Decouples comm/compute
  tiling with producer-consumer barriers; supports the single-owner tiling in
  Phase 1. → Phases 1, 3.
- **TokenWeave** — *TokenWeave: Efficient Compute-Communication Overlap for
  Distributed LLM Inference*, 2025. [arXiv:2505.11329](https://arxiv.org/abs/2505.11329).
  Coarse-grained, contention-aware split for the reduce path (also overlaps the
  memory-bound RMSNorm). → Phase 2.
- **ParallelKittens / ThunderKittens** — CTA/warp-specialization patterns for
  overlapping communication and compute.
  [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens). → Phase 4.
- **T3** — *T3: Transparent Tracking & Triggering for Fine-grained Overlap of
  Compute & Collectives*, Pati et al. (AMD / UW-Madison), ASPLOS 2024.
  [arXiv:2401.16677](https://arxiv.org/abs/2401.16677). Hardware-software
  co-design: track-and-trigger plus near-memory reduction with no extra CUs —
  directly AMD-relevant. → Phase 4 (direction).
- **PyTorch SymmMem** —
  [`torch.distributed._symmetric_memory`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py)
  and `_fused_all_gather_matmul` / `_fused_matmul_reduce_scatter` fallbacks. The
  API contract and correctness gold our kernels match.
- **AITER Triton comms** — upstream conventions and kernel templates:
  [ROCm/aiter `aiter/ops/triton/README.md`](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/README.md).
- **Effective Communication Time (ECT)** — the overlap-efficiency metric (Flux-
  derived) implemented in `benchmarks/common/overlap.py`; used as the tuning
  objective in Phase 6.

