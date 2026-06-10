"""Family 6 — Fused multi-GPU compute+collective kernels (AG+MM, MM+RS).

Back-compat shim. The implementation now lives in
``benchmarks/bench06_aiter_fused.py`` (AITER fused path), which carries the
apples-to-apples unfused baselines and the Odyssey production shape sweep.
This module remains only so existing call-sites that reference
``benchmarks/bench06_fused.py`` (e.g. ``scripts/run_benchmark.sh``) keep
working; it delegates straight to the canonical entry point.

Launch (GPU):
    torchrun --nproc_per_node=8 benchmarks/bench06_fused.py --out results/<id>/

The legacy in-file AG+MM / MM+RS benchmark functions were removed because
they used a misaligned unfused baseline (gather-on-K with ``B[K*world, N]``
and transpose-based RS) that overstated fused speedups. See
``bench06_aiter_fused.py`` for the corrected, contract-matched versions.
"""

from __future__ import annotations

from benchmarks.bench06_aiter_fused import main

if __name__ == "__main__":
    raise SystemExit(main())
