# Wan2.2 and this repository

This document ties **[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)** (official inference code and `wan/` package) to the **microbenchmarks** campaign (ceilings, roofline-style accounting, scorecard, report).

## What lives where

| Concern | Use this repo | Use Wan2.2 upstream |
|--------|----------------|---------------------|
| BF16 GEMM peak, memory BW, DRAM headroom, multigpu collectives, scorecard shell | `microbenchmarks` (`bench01`–`bench03`, `bench06`, …) | — |
| Real Wan2.2 forward (MoE DiT, VAE, T5, 720p, FSDP / Ulysses) | — | [Wan2.2](https://github.com/Wan-Video/Wan2.2) `generate.py` + `wan/` |
| Per-op FLOP table + synthetic DiT-block E2E MFU (`bench04` / `bench05`) | Same **JSON-driven** DiT *template* as `escher_14b_480p` | Not a substitute for the Wan graph until you add a Wan-specific benchmark module |

Upstream documents installation in **[INSTALL.md](https://github.com/Wan-Video/Wan2.2/blob/main/INSTALL.md)** and generation in **[README.md](https://github.com/Wan-Video/Wan2.2/blob/main/README.md)**.

## Quick install (Wan2.2)

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt   # torch >= 2.4; install flash_attn last if it fails
```

Download a checkpoint (examples from upstream README):

| Variant | 720P-oriented | Hugging Face |
|--------|----------------|--------------|
| T2V-A14B | `--size 1280*720` | [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |
| I2V-A14B | `--size 1280*720` | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) |
| TI2V-5B | `--size 1280*704` or `704*1280` | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
```

## 720p-style run (BF16-friendly dtype)

Upstream recommends **`--convert_model_dtype`** so weights match `config.param_dtype` (typically BF16 for speed). Example (single GPU, from Wan README):

```bash
python generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --prompt "Your prompt here."
```

Multi-GPU (FSDP + DeepSpeed Ulysses) is documented upstream with `torchrun --nproc_per_node=8 generate.py ...`.

For **~10 s of video at 24 fps**, derive the frame count from your pipeline (`num_frames` / segment length); Wan2.2 uses clip-oriented generation—see upstream `generate.py` and task configs for the exact flags.

## Campaign report (this repo)

When you point a campaign at a Wan-style workload and the Hub repo is known
(`huggingface_id` on the registry row for `wan2_2_*` ids, or
`REFERENCE_MODEL=wan2_2_i2v_a14b` so `campaign_meta.json` + registry resolve
the repo), **`scripts/report.py`** does **not** embed the README. It only
adds a single row on **0. Cover** linking to the Hugging Face model card,
and a matching link plus a **Model definition (benchmarked)** summary table
(precision, depth, dimensions, parameter counts, weight footprint) at the
top of **5. Model Description**, followed by the **Input shape definition
(fixed during timing)** and **Op mix (per block)** tables that come from
the `bench04` / `bench05` analytic shapes. The Hugging Face README is the
canonical source of truth — the report just links to it so the campaign
artifact stays small and reproducible.

## How to “leverage” both in one evaluation

1. **Hardware + comm baseline (this repo)**  
   Run `./test.sh -t campaign` (or individual families) on the same node and PyTorch/ROCm or CUDA build you use for Wan. That produces comparable ceilings for MFU denominators and multigpu tables.

2. **Real Wan2.2 throughput (Wan2.2 clone)**  
   Time full `generate.py` runs or add `torch.cuda.Event` timing around the DiT step loop in a small fork—keep the same driver/torch build as step 1 for apples-to-apples.

3. **Optional surrogate workload (this repo)**  
   Add `configs/wan2_2_720p_10s_bf16.json` (copy schema from `configs/escher_14b_480p.json`) with `model` / `shapes` chosen to approximate **token counts** relevant to your Wan step—not the MoE graph itself. Point `bench04` / `bench05` at it via `--config` to reuse roofline + MFU machinery for **comparative** analysis, clearly labeled as surrogate.

4. **Diffusers path**  
   Wan2.2 is also available via **Diffusers** (see news section in upstream README). You can profile `WanPipeline`-style code the same way as `generate.py`, still separate from `bench05`’s synthetic stack.

## Wan2.2 TP linears and the vendored fused AITER kernels

The Wan2.2 DiT is dominated by a stack of `(qkv_proj, o_proj, ffn_w1, ffn_w2)`
linears. Under tensor parallelism the column-parallel halves (e.g.
`qkv_proj`, `ffn_w1`) want a **`all_gather` of activations → `matmul`**,
and the row-parallel halves (e.g. `o_proj`, `ffn_w2`) want a **`matmul →
reduce_scatter`**. These two patterns are exactly what the new vendored
fused kernels in **[`benchmarks/aiter_kernels/`](../benchmarks/aiter_kernels/)**
target — see **[docs/AITER_FUSED_KERNELS.md](AITER_FUSED_KERNELS.md)** for
the user-facing usage / tuning / troubleshooting guide and
**[benchmarks/aiter_kernels/README.md](../benchmarks/aiter_kernels/README.md)**
for the kernel-design review and the upstream-to-`aiter.ops.triton.comms.fused/`
path.

Practical implications when benchmarking a Wan2.2 surrogate workload:

* `bench06_aiter_fused.py` (legacy alias `bench06_fused.py`) sweeps the
  `(M, N, K)` shapes that the surrogate workload registers (the same
  shapes used by `bench04_workload_ops` for the per-op accounting) and
  reports the fused-vs-sequential speedup ratio. The dispatcher tries
  upstream AITER first, then the canonical
  `aiter.ops.triton.comms.fused.*` path, then the vendored
  `benchmarks.aiter_kernels` Triton backend (which always resolves on a
  CUDA host with triton). See `bench10_symm_fused.py` for the matching
  torch SymmMem-side measurement plus a runtime correctness gate.
* SC-12 in `scorecard.{md,json}` therefore grades to `PASS` / `FAIL`
  (not `SKIP`) on a GPU node for Wan2.2 campaigns — the report will
  show whether the fused TP linears actually beat the AG-then-MM /
  MM-then-RS sequential reference at the workload's shapes.
* The kernels expose the SymmMem-compatible signature
  `fused_all_gather_matmul(A_shard, [B], gather_dim, group_name, ...)`
  and `fused_matmul_reduce_scatter(A, B, reduce_op, scatter_dim,
  group_name, ...)`, so a future Wan2.2 forward-bench can drop them
  into the model code as a one-line replacement for
  `torch.ops.symm_mem.fused_*`.

## Next step (code integration)

A dedicated **`bench11_wan_forward.py`** (or similar) would: add `Wan2.2`
to `PYTHONPATH`, load `ckpt_dir`, run one denoising step (or full
sampling) under the same timing helpers as `bench05`, and emit JSON into
`results/<id>/11_wan_forward/` for the report to ingest. That is not
implemented yet; this file is the contract for how upstream fits in
until that bridge exists. (Family 10 is reserved for `bench10_symm_fused`,
the torch SymmMem fused-collective probe — pick the next free family
number for any Wan-specific bench so the artifact tree stays unambiguous.)
