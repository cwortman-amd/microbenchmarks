"""Device-side timing primitives.

All campaign measurements MUST go through this module so that timing rules
(TESTPLAN §4) are uniform: warmup, device events (HIP-backed on ROCm), frozen
shapes, multiple repetitions, distributional stats.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class TimedResult:
    name: str
    iters: int
    warmup: int
    times_ms: List[float]
    median_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    extra: Dict[str, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        # Round long arrays for readable JSON; full data still in CSV.
        d["times_ms"] = [round(t, 6) for t in self.times_ms]
        return d


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def time_op(
    name: str,
    fn: Callable[[], None],
    warmup: int = 5,
    iters: int = 20,
    extra: Optional[Dict[str, float]] = None,
) -> TimedResult:
    """Time a callable using torch.cuda.Event (HIP-backed on ROCm).

    The callable receives no args and must perform exactly the work to be timed.
    Tensors should be allocated outside `fn` so allocation cost is excluded.

    The first `warmup` iterations are not timed but their results synchronize
    the device, triggering kernel compile / autotune / cache warm. Then `iters`
    iterations are timed individually and summary stats are reported.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for timing")

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Per-iter device events.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return TimedResult(
        name=name,
        iters=iters,
        warmup=warmup,
        times_ms=times_ms,
        median_ms=statistics.median(times_ms),
        p10_ms=_percentile(times_ms, 0.10),
        p90_ms=_percentile(times_ms, 0.90),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        std_ms=statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        extra=extra or {},
    )


def time_tight_loop(
    name: str,
    fn: Callable[[], None],
    warmup: int = 5,
    iters: int = 100,
    extra: Optional[Dict[str, float]] = None,
) -> TimedResult:
    """Tight-loop variant for peak-sweep measurements (TESTPLAN §4.3).

    Runs `iters` calls between a single pair of device events, then divides
    elapsed time by iters. Reduces Python+launch jitter for the peak number.
    Per-iter distribution is not available; we synthesize a flat list of
    `iters` copies of the mean for stat compatibility.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required for timing")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    per_ms = total_ms / iters

    times_ms = [per_ms] * iters
    return TimedResult(
        name=name,
        iters=iters,
        warmup=warmup,
        times_ms=times_ms,
        median_ms=per_ms,
        p10_ms=per_ms,
        p90_ms=per_ms,
        min_ms=per_ms,
        max_ms=per_ms,
        std_ms=0.0,
        extra={**(extra or {}), "tight_loop_total_ms": total_ms},
    )
