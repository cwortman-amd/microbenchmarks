# Communication Backend Experiments for Production TP: QuickReduce and MM+AR

## Executive Summary

The HK/Iris fused AG+MM and MM+RS study ended with a production-accurate result:
once the benchmark models real Megatron tensor-parallel sharding, the current fused
kernels are not production wins. At Wan2.2 / Odyssey shapes, unfused
hipBLASLt/rocBLAS GEMM + RCCL is stronger than the available fused paths, and the
theoretical overlap ceiling is small (~1.1-1.6x).

The next plausible performance lever is therefore **not another AG+MM/MM+RS fused
kernel**, but a better communication backend for the likely mainstream production
pattern:

```text
row-parallel GEMM -> all_reduce
```

That is the MM+AR path. `bench13_iris_overlap.py` already showed that portable
chunked overlap is near parity at TP=2/4 but not a clear win, and the upstream Iris
`matmul_all_reduce` fused kernel does not compile on this MI355X/ROCm/Triton stack.
This document covers the QuickReduce experiment track: evaluate QuickReduce-style
all-reduce as a replacement for RCCL in MM+AR. **This experiment has now been run
end-to-end on this node** (`bench14_quickreduce_comm.py`); the verdict is in
"Bottom Line" below and the data in "Measured Results".

QuickReduce is a ROCm all-reduce library that reports up to 2.25x faster
performance than RCCL on 2x/4x MI300X configurations, with optional inline
compression codecs (FP16, FP8, Q8, Q6, Q4). See
[mk1-project/quickreduce](https://github.com/mk1-project/quickreduce). The
upstream repo is archived/read-only and targets CDNA3 (`gfx942`); AMD's
[QuickReduce FP4 on MI355](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html)
blog (May 2026) extends it to MI355/CDNA4 (`gfx950`) and adds an FP4 codec, but
through the vLLM/SGLang integration. Public vLLM currently exposes the AMD
QuickReduce integration and bf16/threshold policy, but not a published FP4 codec
source path. We built the **standalone** library for `gfx950` on this node with a
local native-PyTorch patch that now includes an experimental FP4 profile, and
drive it from **native `torch.distributed`** — matching the production engine,
which is a proprietary native-PyTorch implementation, not vLLM.

### Bottom Line

For the production **bf16 native-PyTorch engine** on MI355X, against the engine's
real fallback (RCCL `torch.distributed.all_reduce`), end-to-end MM+AR at the
Wan2.2/Odyssey GEMM shape:

| TP | Verdict after fused bf16 I/O | Best codec | MM+AR speedup |
| --- | --- | --- | --- |
| 2 | **Clear win** | Q6 (or Q4) | ~2.0x (Q6), ~2.5x (Q4) |
| 4 | **Clear win** | Q6/Q4 | ~1.6x (Q6), ~2.0x (Q4) |
| 8 | **Now viable** | Q4 | ~1.4x (Q4), ~1.2x (Q6) |

Latest FP4 update: after reviewing the public AMD/vLLM integration and adding a
standalone gfx950 FP4 profile, the TP=2 smoke shape (`M=16,K=4096,N=4096`) shows
the expected true 4-bit behavior: standalone FP4 QR all-reduce is **2.14x** faster
than RCCL with `rel_l2_err ~= 0.143`, and fused-bf16 FP4 MM+AR is **2.41x** faster
than RCCL MM+AR. This confirms the codec path is real, not the earlier FP16 alias;
it is still a smoke result, not a full Wan2.2 TP sweep.

Drivers and caveats:

- The win is **entirely from compression**; FP16 QuickReduce is at parity (or a
  loss once casts are added). The compression codecs reduce the all-reduce
  transport only — the GEMM stays fp16.
- The original standalone QuickReduce was **fp16-only and out-of-place**; a bf16
  engine had to cast around the collective, costing ~0.08-0.25 ms. We added a
  fused bf16 I/O path (`allreduce_bf16`) that converts bf16<->fp16 inside the QR
  kernel, eliminating the external cast kernels and materially improving TP=4/8.
- The **codec frontier is Q8 (safest) / Q6 (balanced) / Q4/FP4 (fastest)**; FP8
  is dominated. FP4 is newly available in the local standalone patch as an
  experimental gfx950-only profile (`profile=6`).
- Remaining gate: **Wan2.2 model-quality** validation of Q6/Q4/FP4. The
  `rel_l2_err` here is transport-level (Q6 ~3%, Q4/FP4 ~12-14%), not output
  fidelity.

## Why This Path Is Worth Testing

The latest TP sweep showed:

- AG+MM remains far below parity at TP=2/4/8.
- MM+RS can approach parity at TP=2 but still loses and remains blocked for
  barrier-free variants by missing remote-write completion semantics.
- MM+AR is the likely dominant Wan2.2/SGLang TP pattern.
- MM+AR portable overlap is roughly parity at TP=2/4 for the large 4680-token
  shape, but the ceiling is only ~1.14-1.23x.

QuickReduce is aligned with this result because it attacks the all-reduce backend
itself rather than trying to hide RCCL behind a slower fused GEMM. Its README also
claims its strongest uncompressed/FP16 advantage at 2 and 4 GPUs, which matches the
TP sizes where our MM+AR data looked most promising.

The question for this track is narrow:

> Can QuickReduce reduce the exposed all-reduce cost enough to move end-to-end
> row-parallel MM+AR at Wan2.2/Odyssey shapes on MI355X?

## New Benchmark: `bench14_quickreduce_comm.py`

`benchmarks/bench14_quickreduce_comm.py` adds Family 14. It compares:

- `rccl_all_reduce`: standalone RCCL `dist.all_reduce` on `[M, N]`.
- `cr_all_reduce`: standalone vLLM Custom AllReduce on `[M, N]` — an **external
  reference only**, opt-in via `--with-cr`. The production engine is native
  PyTorch and does **not** use vLLM, so this is not part of the engine; the
  native baseline is `torch.distributed.all_reduce` (RCCL).
- `unfused_mm_ar`: row-parallel GEMM followed by RCCL all-reduce.
- `quickreduce_all_reduce`: standalone QuickReduce all-reduce for selected codecs.
- `quickreduce_mm_ar`: row-parallel GEMM followed by QuickReduce all-reduce.

Codecs accepted by the CLI: `fp16, fp8, q8, q6, q4, fp4`. The blog's INT8/INT6/INT4
map to `q8/q6/q4`. The local standalone gfx950 patch supports
`fp16/fp8/q8/q6/q4/fp4`; FP4 is profile 6 and uses the MI355/gfx950 FP4 conversion
instructions with block-size-32 scaling. Compression applies to the
**all-reduce transport only** — the GEMM stays fp16. Each QuickReduce row reports
both the uncompressed
`logical_ar_gb_s` and an `effective_ar_gb_s` scaled by the codec bit-width, plus
`rel_l2_err`, the relative L2 error of the codec versus an fp32 reference
reduction (microbench-level fidelity, computed outside the timed loop).

It uses the same production row-parallel sharding convention as `bench13`:

```text
A_local: [M, K/tp]
B_local: [K/tp, N]
Y_partial = A_local @ B_local
Y = all_reduce(Y_partial)
```

The benchmark is optional and defensive. If `quickreduce` is not installed or cannot
initialize, it still writes RCCL baseline rows and records QuickReduce rows with an
error/skip reason.

**Status: QuickReduce built and validated on MI355X (`gfx950`).** The standalone
`mk1-project/quickreduce` library compiles and runs on this CDNA4 node after the
gfx950/native-PyTorch patch (see "Building QuickReduce on gfx950" below), and the
benchmark produces real numbers (see "Measured Results"). The archived build
exposes profiles 1-5 only (FP16/FP8/Q8/Q6/Q4); the local patch adds profile 6
(`FP4`) using ROCm gfx950 FP4 intrinsics. Public vLLM/AMD QuickReduce integration
adds bf16, thresholds, max-buffer sizing, and native output-buffer semantics, but
as of this review its public `csrc/quickreduce` still exposes only
`F16/INT8/INT6/INT4`; the FP4 kernel source referenced by the MI355 blog is not
publicly available in upstream vLLM.

### Building QuickReduce on gfx950

The upstream `setup.py` defaults to `GPU_ARCHS=gfx942`. Building for `gfx950`
needs the arch override plus a small patch. The first required fix is that
torch's `cpp_extension` hipify pass rewrites `HIP_CHECK` -> `CUDA_CHECK` in
`csrc/quickreduce.hip` but leaves the header (which only defines `HIP_CHECK`)
untouched, so add a `CUDA_CHECK` alias to `csrc/quickreduce.h`:

```cpp
#ifndef CUDA_CHECK
#define CUDA_CHECK(err) HIP_CHECK(err)
#endif
```

For production bf16, the local patch also adds:

- `allreduce_bf16` / `allreduce_bf16_out` with bf16 load/store conversion fused
  into `AllReduceTwoshot<LineCodec, Bf16IO>`.
- `allreduce_out` for workspace reuse.
- per-call current-stream lookup in the PyTorch binding.
- dtype/contiguity checks.
- optional `grid_cap` argument for MI355 tuning.
- experimental FP4 profile 6 for gfx950, using block-scaled FP4 transport with
  the same block-size-32 scale layout as Q4.

The exact patch is preserved in-repo at
`patches/quickreduce-gfx950-bf16-native.patch`.

Then build/install:

```bash
git clone https://github.com/mk1-project/quickreduce
# apply the preserved patch from this repo
git -C quickreduce apply /home/amd/workspace/microbenchmarks/patches/quickreduce-gfx950-bf16-native.patch
cd quickreduce
GPU_ARCHS=gfx950 .microbenchmarks-rocm-venv/bin/pip install --no-build-isolation ./quickreduce
```

The unused `MUBUF_ACQUIRE/RELEASE` macros and the `gfx942`-only `set_fp16_ovfl`
asm are not referenced / degrade to a no-op on `gfx950`, so no further porting
was needed for the FP16/Q8/Q6/Q4 codecs. CDNA4-specific correctness was spot
checked via `rel_l2_err`.

### Native PyTorch Integration Model

The production flow is a **proprietary native-PyTorch engine, not vLLM/SGLang**,
so QuickReduce must drop into plain `torch.distributed`. It does: the upstream
demo uses Ray, but QuickReduce itself only needs a torch CUDA tensor and an IPC
handle exchange, which we do with `torch.distributed` (no vLLM, no Ray):

1. `quickreduce.init(world, rank)`
2. `local_handle = quickreduce.get_comm_handle()`
3. `dist.all_gather_object(handles, local_handle)`
4. `quickreduce.set_comm_handles(handles)`
5. `out = quickreduce.allreduce(profile, x)`

The benchmark records a `native_pytorch_integration` probe (measured on this
node, TP=2) capturing the contract a proprietary engine must satisfy:

| Property | Value | Engine implication |
| --- | --- | --- |
| `returns_new_tensor` / `input_unmodified` | True / True | **Out-of-place.** Unlike `dist.all_reduce`, the input is not reduced in place; the engine must consume the returned tensor. |
| `dtype_required` | `fp16` | Kernel reinterprets the data pointer as `half` regardless of torch dtype. |
| `bf16_rel_l2_err` / `bf16_safe` | ~103 / False | **Critical: bf16 input is silently corrupted.** A bf16 engine MUST cast to fp16 before the all-reduce (and back after), or results are garbage. |
| `fp16_rel_l2_err` | ~2e-4 | fp16 path is correct (rounding-level error). |
| `world_size_supported` | {2, 4, 8} only | TP must be 2, 4, or 8. |
| `max_problem_bytes` | 536,870,912 | Per-call payload capped at 512 MB. |

The bf16 caveat is the most important integration finding: most DiT/LLM engines
run bf16, so adopting QuickReduce requires an fp16 cast around the collective
(which also slightly changes numerics versus a bf16 RCCL all-reduce).

This lets the benchmark run under the same `torch.distributed.run` launcher used by
the rest of the repo.

## How To Run

`torchrun` spawns workers in fresh subprocesses, so export `PYTHONPATH=$PWD` so
the `benchmarks` package resolves from the repo root.

Smoke test (runs even without QuickReduce installed):

```bash
PYTHONPATH=$PWD .microbenchmarks-rocm-venv/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  benchmarks/bench14_quickreduce_comm.py \
  --out results/quickreduce-smoke \
  --shapes 128,128,128 \
  --warmup 1 \
  --iters 2 \
  --codecs fp16
```

Production-like Wan2.2 / Odyssey sweep (codecs the mk1 build supports;
add `fp4` only against an FP4-capable build):

```bash
for TP in 2 4 8; do
  for SET in wan2_2 odyssey_production; do
    PYTHONPATH=$PWD .microbenchmarks-rocm-venv/bin/python -m torch.distributed.run \
      --nproc_per_node=${TP} \
      benchmarks/bench14_quickreduce_comm.py \
      --out results/quickreduce-prod/tp${TP}_${SET} \
      --shape-set ${SET} \
      --warmup 10 \
      --iters 50 \
      --dtype bf16 \
      --codecs fp16,fp8,q8,q6,q4
  done
done
```

Decode-regime sweep (small/low-volume payloads; sub-100us and noisy):

```bash
PYTHONPATH=$PWD .microbenchmarks-rocm-venv/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  benchmarks/bench14_quickreduce_comm.py \
  --out results/quickreduce-prod/tp2_decode \
  --shape-set quickreduce_decode \
  --warmup 10 --iters 50 --dtype bf16 --codecs fp16,q6,q4
```

The vLLM Custom AllReduce baseline is **off by default** (the native engine does
not use vLLM). Pass `--with-cr` only if you want it as an external reference.

## Measured Results (MI355X / gfx950, ROCm 7.2)

First real runs on this node (8x MI355X, ROCm 7.2, torch 2.12+rocm7.2),
warmup=10 iters=30, fp16, `odyssey_production` shapes. The baseline is RCCL
(`torch.distributed.all_reduce`), which is the correct fallback for the native
PyTorch engine. "AR" is standalone all-reduce, "MM+AR" is the end-to-end
row-parallel linear.

### TP=2

| Shape (M, N=13824) | codec | AR speedup | MM+AR speedup | rel_l2_err |
| --- | --- | --- | --- | --- |
| 1590 | fp16 | 1.01x | 1.00x | 2.3e-4 |
| 1590 | q8   | 1.76x | 1.59x | 7.5e-3 |
| 1590 | q6   | 2.22x | 1.90x | 3.0e-2 |
| 1590 | q4   | 3.06x | 2.40x | 1.2e-1 |
| 4680 | q8   | 1.78x | 1.61x | 7.5e-3 |
| 4680 | q6   | 2.27x | 1.95x | 3.0e-2 |
| 4680 | q4   | 3.15x | 2.47x | 1.2e-1 |

### TP=4

| Shape (M, N=13824) | codec | AR speedup | MM+AR speedup |
| --- | --- | --- | --- |
| 1590 | fp16 | 0.96x | 0.96x |
| 1590 | q6   | 1.61x | 1.47x |
| 1590 | q4   | 2.16x | 1.86x |
| 4680 | q8   | 1.52x | 1.41x |
| 4680 | q6   | 1.89x | 1.69x |
| 4680 | q4   | 2.59x | 2.15x |

### Realistic bf16 path (bf16 GEMM -> cast fp16 -> QR -> cast bf16)

The pure-fp16 numbers above do not include the `bf16<->fp16` casts a native bf16
engine needs (QuickReduce is fp16-only). The `quickreduce_mm_ar_bf16cast` variant
measures the full exposed path versus the **bf16** RCCL MM+AR baseline.

| TP | Shape M | codec | MM+AR speedup (bf16 cast) | cast overhead |
| --- | --- | --- | --- | --- |
| 2 | 1590 | fp16 | 0.93x | 0.077 ms |
| 2 | 1590 | q6   | 1.65x | 0.077 ms |
| 2 | 1590 | q4   | 2.01x | 0.077 ms |
| 2 | 4680 | q6   | 1.66x | 0.255 ms |
| 2 | 4680 | q4   | 2.01x | 0.255 ms |
| 4 | 1590 | q6   | 1.15x | 0.077 ms |
| 4 | 1590 | q4   | 1.39x | 0.077 ms |
| 4 | 4680 | q6   | 1.23x | 0.254 ms |
| 4 | 4680 | q4   | 1.45x | 0.253 ms |
| 8 | 1590 | q6   | 0.83x | 0.078 ms |
| 8 | 1590 | q4   | 0.95x | 0.078 ms |
| 8 | 4680 | q6   | 0.88x | 0.255 ms |
| 8 | 4680 | q4   | 0.95x | 0.253 ms |

The casts cost ~0.08 ms (M=1590) / ~0.25 ms (M=4680) round-trip and erode the
uplift relative to the pure-fp16 numbers:

- **TP=2**: Q6 drops from ~1.9-2.0x (fp16-only) to **~1.65x** (bf16 cast); Q4 stays
  ~2.0x. Still a clear win.
- **TP=4**: Q6 drops from ~1.7x to **~1.15-1.23x**; Q8 collapses to ~parity
  (~1.0-1.08x); only Q4 keeps a meaningful **~1.4-1.45x**.
- **TP=8: net loss for every codec** (0.69-0.95x), even Q4. RCCL all-reduce is
  already fast at TP=8, the exposed-AR fraction of MM+AR is smaller, and the cast
  tax dominates. Note that the *standalone fp16 AR* still wins at TP=8
  (`engine_best_of_ar` ~1.5-1.7x) — this is exactly the trap the end-to-end bf16
  gate is designed to catch: a standalone-AR win does not survive the GEMM + cast
  round-trip.
- **FP16 codec is a net loss with casts** at every TP (0.69-0.93x): no compression
  benefit to pay back the cast cost.

Implication: for a bf16 engine, QuickReduce is compelling at **TP=2 with Q6/Q4**,
narrows to **Q4-only at TP=4**, and is **not worth it at TP=8** once the cast tax
is included. The cast could be eliminated only by an fp16/bf16-native QuickReduce
build (e.g. AMD's fork); that is the prerequisite for any TP=4/TP=8 case.

### Wan2.2 GEMM shape, all codecs incl. FP8 (bf16 cast path)

Tested against the `wan2_2` shape set (M=4680, K=5120, N=13824 — the Wan2.2
I2V/T2V-A14B FFN GEMM; identical to `odyssey_3frame`, results match). bf16 cast
path, MM+AR speedup vs bf16 RCCL:

| codec | rel_l2_err | TP=2 | TP=4 | TP=8 |
| --- | --- | --- | --- | --- |
| fp16 | 2.3e-4 | 0.90x | 0.81x | 0.68x |
| fp8  | 3.3e-2 | 1.40x | 1.09x | 0.79x |
| q8   | 7.5e-3 | 1.40x | 1.08x | 0.79x |
| q6   | 3.0e-2 | 1.65x | 1.23x | 0.88x |
| q4   | 1.2e-1 | 2.01x | 1.46x | 0.95x |

FP8-specific finding: **FP8 is dominated** on this hardware. It matches Q8 on
speed (both 8-bit transport, ~1.40x at TP=2) but is ~4x less accurate
(fp8 3.3% vs q8 0.75% rel_l2), and Q6 matches FP8's accuracy (~3.0%) while being
faster (1.65x vs 1.40x). So the useful frontier is **Q8 (safest), Q6 (balanced),
Q4/FP4 (fastest)** — FP8 is not worth selecting. The standalone build's INT and
FP4 codecs use block-size-32 quantization; FP4 is experimental and currently
validated only by smoke tests in this repo.

### FP4 smoke after AMD-fork review

Public vLLM/AMD QuickReduce was reviewed for the MI355 blog enhancements. The
available source includes bf16 integration, output-buffer semantics, ROCm arch
gating, max-buffer sizing, and min-size/codec-threshold policy, but still exposes
only `F16/INT8/INT6/INT4` in `csrc/quickreduce`. The local standalone patch
therefore ports FP4 from the public ROCm MI355 FP4 intrinsic path rather than from
a published vLLM FP4 codec source.

Smoke command shape: TP=2, `M=16,K=4096,N=4096`, `--qr-grid-cap 1024`,
`warmup=1,iters=3`.

| dtype/path | FP4 row | FP4 time | baseline | speedup | rel_l2_err |
| --- | --- | --- | --- | --- | --- |
| fp16 AR | `quickreduce_all_reduce` | 0.026 ms | RCCL AR 0.056 ms | 2.14x | 0.143 |
| fp16 MM+AR | `quickreduce_mm_ar` | 0.047 ms | RCCL MM+AR 0.215 ms | 4.57x | transport row above |
| bf16 MM+AR, external casts | `quickreduce_mm_ar_bf16cast` | 0.046 ms | RCCL MM+AR 0.092 ms | 1.98x | transport row above |
| bf16 MM+AR, fused I/O | `quickreduce_mm_ar_bf16native` | 0.038 ms | RCCL MM+AR 0.092 ms | 2.41x | transport row above |

The latest FP4 smoke also confirms the fused-bf16 path avoids the measured
external-cast overhead (`0.016 ms`) and improves the same FP4 bf16 MM+AR case from
**1.98x** to **2.41x**. The absolute latencies are small and the shape is not the
full Wan2.2 prefill shape, so use this as codec validation before a full
TP=2/4/8 sweep.

Artifacts:
`results/quickreduce-fp4-smoke/14_quickreduce_comm/quickreduce.csv` and
`results/quickreduce-fp4-bf16-smoke/14_quickreduce_comm/quickreduce.csv`.

### Fused bf16 I/O implementation (`allreduce_bf16`)

The initial bf16 path used external cast kernels:

```text
bf16 GEMM -> bf16_to_fp16 kernel -> QR(fp16) -> fp16_to_bf16 kernel
```

That cast tax was the main reason TP=4 was marginal and TP=8 was a loss. The
standalone QuickReduce extension was updated with:

- `allreduce_bf16(profile, A_bf16, grid_cap=0)`: QR loads bf16, converts to fp16
  in registers, performs the existing fp16 codec/reduce path, then stores bf16.
- `allreduce_out` / `allreduce_bf16_out`: output-buffer reuse for engine
  workspaces and benchmark timing without allocator noise.
- Per-call current-stream lookup instead of capturing the stream at `init()`.
- Dtype/contiguity checks so bf16 passed to legacy `allreduce` is rejected rather
  than silently corrupted.

`bench14` adds `quickreduce_mm_ar_bf16native`, which measures:

```text
bf16 GEMM -> QR allreduce_bf16
```

Wan2.2 shape (M=4680, K=5120, N=13824), bf16 MM+AR speedup vs bf16 RCCL:

| TP | codec | external-cast QR | fused-bf16 QR | improvement |
| --- | --- | --- | --- | --- |
| 2 | q6 | 1.64x | 1.97x | +20% |
| 2 | q4 | 2.01x | 2.51x | +25% |
| 4 | q6 | 1.23x | 1.63x | +33% |
| 4 | q4 | 1.44x | 1.97x | +37% |
| 8 | q6 | 0.87x | 1.24x | flips to win |
| 8 | q4 | 0.95x | 1.39x | flips to win |

This materially changes the deployment recommendation: with fused bf16 I/O, Q6
is viable through TP=4 and Q4 remains a win even at TP=8. The remaining TP=8
speedup is modest enough that it should still be policy-gated by shape and codec.

### MI355 grid-cap sweep

The upstream kernel caps grid blocks at `304*4`, inherited from MI300X. MI355
reports 256 CUs, so the benchmark now accepts `--qr-grid-cap` and the extension
passes that cap to the kernel launcher. Sweep on Wan2.2 bf16-native Q6/Q4:

| TP | Best cap observed | q6 speedup | q4 speedup | Note |
| --- | --- | --- | --- | --- |
| 2 | 1024-2048 | ~1.98-1.99x | ~2.51-2.53x | insensitive |
| 4 | 2048 | ~1.65x | ~2.04x | small gain over default |
| 8 | 1536 | ~1.27x | ~1.47x | best measured TP=8 |

The default 1216-block cap is not badly wrong, but TP=8 benefits from a higher
cap (~1536). Production should either expose this as a tuning knob or auto-set a
larger cap for TP=8 on MI355.

### Findings

- **FP16 QuickReduce is at parity with RCCL** at these large prefill shapes
  (~1.0x at TP=2, ~0.96x at TP=4) — the win comes entirely from compression.
- **Compression delivers the uplift, scaling with aggressiveness.** With fused
  bf16 I/O, Q6 gives ~2.0x TP=2, ~1.6x TP=4, and ~1.2x TP=8; Q4 reaches ~2.5x,
  ~2.0x, and ~1.4-1.5x respectively.
- **The end-to-end MM+AR speedup is smaller than the standalone-AR speedup**, as
  expected, because the fp16 GEMM is a fixed cost the codec cannot compress —
  validating the doc's insistence on gating on MM+AR, not AR alone.
- **`rel_l2_err` grows steeply**: q8 ~0.8%, q6 ~3%, q4 ~12% transport error.
  Q4's 12% is large; Q6 is the defensible production candidate pending Wan2.2
  quality validation. (This is transport error, not model-quality.)
- **Decode regime** (`quickreduce_decode`, M=1..64, payload 27KB-1.7MB): QR edged
  RCCL here too. Because the native engine's fallback is RCCL (not a custom
  all-reduce), QR-vs-RCCL is the correct comparison at these sizes — but the
  timings are sub-100us and noisy, so treat the decode numbers as indicative
  only and re-measure with higher iteration counts if decode latency matters.

These results clear success criteria 1, 2, and 4 for Q6/Q4 at TP=2/4 and Q4 at
TP=8 against RCCL, the native engine's real fallback. Criterion 3 (codec quality)
still needs Wan2.2 model-quality validation; the latency half is met with fused
bf16 I/O.

Artifacts:

- `14_quickreduce_comm/quickreduce.csv`: raw timing rows.
- `14_quickreduce_comm/quickreduce.json`: full payload with run metadata.
- `14_quickreduce_comm/quickreduce_summary.csv`: speedups versus RCCL baselines,
  plus `cr_ar_vs_rccl_ar` and the `engine_best_of_ar` selection-policy rows
  (see Metrics).

## Metrics and Interpretation

The primary decision metrics are:

- `engine_best_of_ar`: the headline number. Models how the native-PyTorch engine
  dispatches: `best_of(RCCL, QuickReduce_codec*)` versus the no-QuickReduce
  fallback (RCCL). `quickreduce_selected=True` means QuickReduce won at that size.
  Because the engine is native PyTorch, **RCCL is the correct fallback** — there
  is no vLLM CR in the loop. (If `--with-cr` is passed, CR is folded in purely as
  an external reference.)
- `quickreduce_all_reduce` vs `rccl_all_reduce` and `cr_ar_vs_rccl_ar`: the
  component speedups that feed the selection policy.
- `quickreduce_mm_ar` vs `unfused_mm_ar`: does the communication improvement move
  the end-to-end row-parallel linear?
- Codec sensitivity: does FP16 already win? If not, do Q8/Q6/Q4/FP4 win enough to
  justify quality validation? Cross-check each codec's `rel_l2_err`.

Do **not** claim success from standalone all-reduce alone. The production gate is
end-to-end MM+AR because GEMM remains a large part of the row-parallel linear, and
for bf16 the collective must use the fused-bf16 path (`quickreduce_mm_ar_bf16native`)
rather than external cast kernels. The older external-cast TP=8 result is the
cautionary case: standalone fp16 AR won (~1.5-1.7x) but end-to-end bf16 MM+AR lost
until the cast tax was fused into QR. The `quickreduce_decode` set characterizes
small/low-volume payloads (sub-100us, noisy); treat those as indicative only.

## Caveats

- The upstream `mk1-project/quickreduce` repo is archived/read-only and targets
  ROCm/CDNA3 (`gfx942`). It has now been **built and validated on MI355X/CDNA4
  (`gfx950`)** on this node (see "Building QuickReduce on gfx950"), so the
  standalone module and its native-PyTorch API are confirmed for FP16/BF16 I/O
  with FP8/Q8/Q6/Q4 codecs. FP4 is available only through the local experimental
  patch; the public AMD/vLLM integration reviewed here does not yet publish its
  MI355 FP4 QuickReduce codec source.
- The original README reports MI300X / ROCm 6.2; the AMD MI355 results use
  ROCm 7.2.2. Match the installed stack when interpreting absolute latencies.
- Compression codecs affect numerical accuracy and model quality. The benchmark's
  `rel_l2_err` is a transport-level proxy only; Q6/Q4/FP4 are **not** drop-in
  inference-safe without Wan2.2 quality validation (the blog's GSM8K recovery is
  an LLM result, not a Wan2.2/DiT result).
- The production engine is **native PyTorch, not vLLM**, so RCCL
  (`torch.distributed.all_reduce`) is the correct baseline and `engine_best_of_ar`
  uses RCCL as the only fallback by design. The vLLM CR baseline is opt-in
  (`--with-cr`) and external-reference only. Note that on this ROCm stack torch's
  native symmetric-memory custom all-reduce ops (`torch.ops.symm_mem`) are not
  registered (NVSHMEM/CUDA-gated), so there is no native sub-RCCL fast path to
  compare against — RCCL is genuinely the engine's status-quo collective.
- Legacy `allreduce` is fp16-only and now rejects bf16 with a dtype check. Use
  `allreduce_bf16` / `allreduce_bf16_out` for bf16 tensors; this is the production
  path measured by `quickreduce_mm_ar_bf16native`.
- QuickReduce solves **all-reduce**, not AG+MM or MM+RS. That is appropriate only if
  the customer path is MM+AR or if all-reduce remains the dominant exposed
  collective after layout confirmation.

## Success Criteria

QuickReduce is worth a deeper integration only if:

1. `quickreduce_mm_ar_bf16native` shows a real speedup over RCCL
   (`torch.distributed.all_reduce`, the native engine's actual fallback) for the
   production `[M, N]` payloads. **Met** for Q6/Q4 at TP=2/4 and Q4 at TP=8.
2. End-to-end MM+AR improves, not just standalone AR. **Met** with fused bf16 I/O;
   the older external-cast path is retained only as a cautionary baseline.
3. A compressed codec (Q8/Q6/Q4/FP4) improves end-to-end latency further with
   acceptable `rel_l2_err` and, ultimately, Wan2.2 model-quality impact.
   **Met for latency** (Q6 ~2.0x TP=2, ~1.6x TP=4, ~1.2x TP=8; Q4 ~2.5x,
   ~2.0x, ~1.4-1.5x — see Fused bf16 I/O; FP4 smoke TP=2 shows the expected
   4-bit transport behavior and needs full TP sweep/model-quality validation).
   Wan2.2 model-quality validation is the remaining open item.
4. The result holds on MI355X (`gfx950`) via the standalone module. **Met** — the
   module is built and validated on this node.

If those criteria fail, the right production path remains unfused hipBLASLt/rocBLAS
GEMM + RCCL tuning, while the framework issues documented in `KERNEL_EXP.md` remain
separate blockers.

## Relationship to `KERNEL_EXP.md`

`KERNEL_EXP.md` answers: "Should we keep tuning fused AG+MM/MM+RS kernels?"

The production-accurate answer is no: current fused kernels are not competitive at
the real TP layouts, and the remaining barrier-free MM+RS route is blocked by
missing framework semantics.

`COMM_EXP.md` answers the next question: "If the mainstream production pattern is
MM+AR, can a better all-reduce backend beat RCCL enough to matter?"

That is the QuickReduce experiment.
