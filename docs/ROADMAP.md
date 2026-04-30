# Microbenchmark Framework Roadmap

This roadmap captures the planned enhancements to transition the framework to an expert-level technical benchmark suite for the MI355X hardware.

## 1. GPU-Native Profiling
* Transition from CPU-fallback testing to active MI355X hardware testing.
* Introduce a sustained workload loop (`bench07_sustained.py`) designed to run for several minutes to induce thermal states.
* Deepen `rocm-smi` integration to sample real-time power draw (W), thermal throttling flags, and sustained clock frequencies.
* Implement a `rocprof` wrapper capability to capture hardware performance counters.

## 2. Advanced Datatype Scaling (MXFP6 / MXFP4)
* Rename `bench01_bf16_compute.py` to reflect multi-datatype support.
* Add GEMM sweeps for MXFP6 and MXFP4 datatypes.
* Target and validate the 20.1 PFLOPS peak potential for FP4.
* Update arithmetic intensity and FLOP accounting in `bench04_workload_ops.py` to support mixed-precision workload analysis.

## 3. Topology-Aware Benchmarking
* Expand `bench08_topology_bw.py` to explicitly map the 8-GPU UBB (Universal Baseboard) configuration.
* Add direct Peer-to-Peer (P2P) transfers across the 7 Infinity Fabric links per GPU.
* Target the 153 GB/s theoretical per-link bandwidth and emit a full N x N bandwidth matrix.
