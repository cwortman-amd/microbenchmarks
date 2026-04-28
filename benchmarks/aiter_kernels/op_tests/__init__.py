"""Operator tests for aiter_kernels — mirror of ``op_tests/triton_tests/comms/``.

Run via:

  torchrun --nproc_per_node=2 -m benchmarks.aiter_kernels.op_tests.test_fused_collective

The test harness picks up every backend the dispatcher knows about (local
Triton, AITER+Iris, torch SymmMem) and asserts they all match the pure-Torch
fallback within bf16 tolerance. A backend that's not available on the host
is skipped (logged) rather than failed.
"""
