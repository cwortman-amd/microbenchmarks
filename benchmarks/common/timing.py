"""Device-side timing primitives.

All benchmark measurements MUST go through this module so that timing rules
(TESTPLAN §4) are uniform: warmup, device events (HIP-backed on ROCm), frozen
shapes, multiple repetitions, distributional stats.

The primitives are device-agnostic:

  * On a GPU host (CUDA / HIP available) we use ``torch.cuda.Event`` for
    sub-microsecond accurate device-side timing.
  * On a CPU host we fall back to ``time.perf_counter_ns()``. Wall-clock is
    sufficient because the kernels are synchronous on CPU — there is no
    queue-and-stream model to race with — and the benchmark on CPU is meant
    for infrastructure validation / regression smoke tests, not for the
    headline MI355X numbers.

The same TimedResult is produced either way so downstream parsers stay
oblivious to which timer ran.
"""

from __future__ import annotations

import statistics
import time
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


def _sync() -> None:
    """Synchronize the active device. No-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize(name: str, iters: int, warmup: int, times_ms: List[float],
               extra: Optional[Dict[str, float]] = None) -> TimedResult:
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


def time_op(
    name: str,
    fn: Callable[[], None],
    warmup: int = 5,
    iters: int = 20,
    extra: Optional[Dict[str, float]] = None,
) -> TimedResult:
    """Time a callable using torch.cuda.Event (GPU) or perf_counter_ns (CPU).

    The callable receives no args and must perform exactly the work to be timed.
    Tensors should be allocated outside ``fn`` so allocation cost is excluded.

    The first ``warmup`` iterations are not timed but do trigger compile /
    autotune / cache warm. Then ``iters`` iterations are timed individually
    and summary stats are reported.
    """
    print(f"[bench] {name:32s} warmup={warmup:2d} iters={iters:3d} ...", end="", flush=True)
    t_start = time.perf_counter()
    for _ in range(warmup):
        fn()
    _sync()

    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        times_ms = []
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
            # Methodology: synchronize after each timed iteration so each sample
            # is a fully completed kernel path, not queued work.
            torch.cuda.synchronize()
            times_ms.append(starts[i].elapsed_time(ends[i]))
    else:
        times_ms = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1e6)

    res = _summarize(name, iters, warmup, times_ms, extra)
    t_end = time.perf_counter()
    print(f" done ({res.median_ms:8.3f} ms, {(t_end - t_start):.1f}s total)")
    return res


def time_tight_loop(
    name: str,
    fn: Callable[[], None],
    warmup: int = 5,
    iters: int = 100,
    extra: Optional[Dict[str, float]] = None,
) -> TimedResult:
    """Tight-loop variant for peak-sweep measurements (TESTPLAN §4.3).

    Runs ``iters`` calls between a single timing pair, then divides elapsed
    time by iters. Reduces Python+launch jitter for the peak number.
    Per-iter distribution is not available; we synthesize a flat list of
    ``iters`` copies of the mean for stat compatibility.
    """
    print(f"[bench] {name:32s} warmup={warmup:2d} iters={iters:3d} (tight-loop) ...", end="", flush=True)
    t_start = time.perf_counter()
    for _ in range(warmup):
        fn()
    _sync()

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter_ns()
        total_ms = (t1 - t0) / 1e6

    per_ms = total_ms / iters
    times_ms = [per_ms] * iters
    res = TimedResult(
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
    t_end = time.perf_counter()
    print(f" done ({res.median_ms:8.3f} ms/iter, {(t_end - t_start):.1f}s total)")
    return res
