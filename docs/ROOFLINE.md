---
aliases: [roofline, cost model, MFU, capacity planning]
tags: [roofline, MI355X, HBM, KV-cache, TP]
---
# Deep Roofline & Cost Models

The `microbenchmarks` test suite provides empirical telemetry and benchmark results. 

For the MI355X-specific application of these models to GEMM and Attention kernels, see [[KERNEL#5. MI355X Roofline Model (Applied)]]. The canonical GPU spec registry (rated peaks, HBM BW) lives in [[../configs/report_config.json|report_config.json]].

## The Roofline Cost Model

The analytical model breaks down inference time per-operation by calculating both a theoretical compute time and memory time, ultimately taking the maximum of the two to establish the hardware limit. 

### Core Timing Formulas
To implement this outside of a notebook, you can apply the following straightforward logic for each operation (e.g., a linear layer, an attention block):

1. **Calculate Compute Time (`t_flops`)**:
   `t_flops (us) = (FLOPs / GPU_BF16_FLOPS_S) * 1e6`
2. **Calculate Memory Time (`t_mem`)**:
   First, determine the effective HBM bytes transferred, accounting for an estimated L2 Cache hit rate:
   `effective_bytes = total_read_write_bytes * (1.0 - L2_hit_rate)`
   Then, calculate the time to transfer those bytes:
   `t_mem (us) = (effective_bytes / GPU_MEM_BW_BYTES_S) * 1e6`
3. **Determine the Bottleneck**:
   The predicted time for the operation is simply the bottlenecked time:
   `t_bottleneck = max(t_flops, t_mem)`
   The operation is considered **compute-bound** if `t_flops >= t_mem`, and **memory-bound** otherwise.

### Per-Op Efficiency Breakdown
The cost models demonstrate the ability to predict per-layer theoretical limits versus measured efficiency. For instance, the notebooks reveal:
- `self_attn.QKV` achieves ~92% efficiency.
- `self_attn.flash` is typically the bottleneck at stock settings, but achieves ~87% efficiency with optimized AITER kernels.
- **VAE Decode** emerges as the dominant bottleneck for real-time video generation once the transformer operations are pushed to the hardware ceiling.

## Memory Budgeting & Tensor Parallelism (TP)

Predictive models are critical for determining when Tensor Parallelism (TP) or sequence parallel methods become necessary to avoid Out-Of-Memory (OOM) errors. To estimate VRAM requirements mathematically:

### KV Cache Memory Formulas
For BF16 inference (2 bytes per element):
- **Self-Attention KV Cache (Bytes)**:
  `SA_KV = 2 * (window_size * H * W) * dim * ES * num_layers`
- **Cross-Attention KV Cache (Bytes)**:
  `CA_KV = 2 * text_len * dim * ES * num_layers`

Where `ES` = element size in bytes (2 for BF16, matching the notebook's convention).

By summing `Weights_GB + SA_KV_GB + CA_KV_GB + Activations_GB`, you can predict exact memory usage for a specific sequence dimension ($H \times W$) and sequence length, dynamically plotting the point at which $VRAM_{required} > VRAM_{available}$, necessitating TP.

## Future Porting Opportunity

Currently, these deep predictive models exist solely within the exploratory Jupyter notebooks. 

**Recommendation:** You could port this mathematical logic directly into the `scripts/plot_results.py` reporting tool within the `microbenchmarks` suite. 

By doing so, the benchmark report could automatically generate **predictive scaling curves**, answering hypothetical questions like:
- *"What happens to MFU if we increase the sequence length to 100K?"*
- *"At what sequence length does Tensor Parallelism (TP=2 or TP=4) become strictly necessary to avoid HBM capacity exhaustion?"*

Integrating the notebook's theoretical cost formulas with the empirical measurements captured by `run.sh` would transform the `microbenchmarks` suite from a pure reporting tool into a forward-looking capacity planning engine.

---

## See also

- [[KERNEL]] — Triton vs Mojo mental models, GEMM/Attention case studies, MI355X hardware lens
- [[AITER_FUSED_KERNELS]] — Fused comm+compute kernels (AG+MM / MM+RS) with roofline analysis
- [[WAN2.2]] — Wan2.2 integration guide and surrogate workload setup
