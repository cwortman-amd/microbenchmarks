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
the repo), **`scripts/report.py`** embeds the Hugging Face **README** (model
card) on **0. Cover** and pulls a filtered **##**-section digest into **Model
Description** → *Architecture from Hugging Face model card*, ahead of the
**Benchmark configuration** tables (`bench04` / `bench05` analytic shapes).
Regenerating the report may fetch README over HTTPS.

## How to “leverage” both in one evaluation

1. **Hardware + comm baseline (this repo)**  
   Run `./test.sh -t campaign` (or individual families) on the same node and PyTorch/ROCm or CUDA build you use for Wan. That produces comparable ceilings for MFU denominators and multigpu tables.

2. **Real Wan2.2 throughput (Wan2.2 clone)**  
   Time full `generate.py` runs or add `torch.cuda.Event` timing around the DiT step loop in a small fork—keep the same driver/torch build as step 1 for apples-to-apples.

3. **Optional surrogate workload (this repo)**  
   Add `configs/wan2_2_720p_10s_bf16.json` (copy schema from `configs/escher_14b_480p.json`) with `model` / `shapes` chosen to approximate **token counts** relevant to your Wan step—not the MoE graph itself. Point `bench04` / `bench05` at it via `--config` to reuse roofline + MFU machinery for **comparative** analysis, clearly labeled as surrogate.

4. **Diffusers path**  
   Wan2.2 is also available via **Diffusers** (see news section in upstream README). You can profile `WanPipeline`-style code the same way as `generate.py`, still separate from `bench05`’s synthetic stack.

## Next step (code integration)

A dedicated **`bench10_wan_forward.py`** (or similar) would: add `Wan2.2` to `PYTHONPATH`, load `ckpt_dir`, run one denoising step (or full sampling) under the same timing helpers as `bench05`, and emit JSON into `results/<id>/10_wan_forward/` for the report to ingest. That is not implemented yet; this file is the contract for how upstream fits in until that bridge exists.
