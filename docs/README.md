---
aliases: [docs index, documentation]
tags: [index, navigation]
---
# Documentation Index

This directory contains architecture and design documentation for the `microbenchmarks` suite.

| Document | Purpose | Start here if… |
|:---|:---|:---|
| [[TESTPLAN]] | Full benchmark methodology: success criteria (SC-1…SC-12), benchmark families 1–11, timing protocol, roofline construction, MFU computation, artifact manifest, and sign-off rules. | You need to understand the benchmark structure, pass/fail criteria, or how families relate to each other. |
| [[AITER_FUSED_KERNELS]] | User-facing guide for the fused All-Gather+MatMul and MatMul+Reduce-Scatter kernels in `benchmarks/aiter_kernels/`. API, dispatcher, tuning, troubleshooting, roofline analysis. | You need to use, tune, or debug the fused TP-linear kernels. |
| [[KERNEL]] | Deep comparison of Triton (tile-centric) vs Mojo/MAX (thread-centric) kernel architectures, with GEMM and Attention case studies mapped to MI355X hardware (wave64, LDS, MFMA, roofline). | You are porting or writing custom kernels and need to understand the programming model tradeoffs on AMD hardware. |
| [[ROOFLINE]] | Portable formulas for the analytical roofline cost model and memory budgeting (KV cache sizing, TP capacity planning). | You want to implement predictive latency/scaling models in Python scripts. |
| [[WAN2.2]] | Integration guide tying [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) upstream to the benchmark scorecard and report. Install, run, and surrogate-workload instructions. | You are evaluating Wan2.2 inference alongside hardware benchmarks. |

## See also

- [[../README|README]] — top-level repo overview, quick start, layout, and pass/fail criteria.
- `benchmarks/aiter_kernels/README.md` — internal kernel design doc and upstream review.
- `configs/report_config.json` — canonical GPU spec registry and report tuning knobs.
