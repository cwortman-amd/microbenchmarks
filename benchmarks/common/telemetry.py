"""Telemetry sampler for hardware health and power monitoring during a benchmark run.

A lightweight, dependency-free sampler that runs in a background thread
alongside any benchmark and writes a per-bench ``telemetry.json``. The
sampler picks the best available source for the host:

  - GPU (AMD ROCm):  ``amd-smi`` (preferred) or ``rocm-smi``  → power, temp, clk
  - GPU (NVIDIA):    ``nvidia-smi``                            → power, temp, clk, util
  - CPU host:        ``turbostat`` (root) or ``/sys/class/powercap`` (RAPL)
                     plus ``/proc/cpuinfo`` for clock and ``psutil`` if
                     available for utilization.

Use as a context manager::

    from benchmarks.common.telemetry import telemetry

    with telemetry(out_dir / "telemetry.json", interval_s=1.0) as tel:
        run_my_benchmark()
        # tel.note("phase", "warmup")  # optional event annotations

The output JSON has the schema::

    {
      "started_iso": "...",
      "ended_iso":   "...",
      "interval_s":  1.0,
      "source":      "amd-smi" | "rocm-smi" | "nvidia-smi" | "rapl" | "none",
      "samples": [
        {"t_s": 0.0,  "device_idx": 0, "power_w": 145.2, "temp_c": 52.1,
         "clk_mhz": 1955, "util_pct": 99.0, ...},
        ...
      ],
      "events": [{"t_s": 1.7, "key": "phase", "value": "warmup"}, ...],
      "summary": {
        "power_w":    {"mean": ..., "p95": ..., "max": ...},
        "temp_c":     {"mean": ..., "p95": ..., "max": ...},
        "clk_mhz":    {"mean": ..., "min": ..., "max": ...},
      }
    }

The sampler is best-effort: if a source fails partway through (e.g.
``amd-smi`` becomes unavailable mid-run) the affected fields are simply
omitted from subsequent samples. The sampler **never** raises into the
benchmark thread.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _safe_run(cmd: List[str], timeout: float = 2.0) -> Optional[str]:
    """Run a short shell command; return stdout or None on any failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# ---------------------------------------------------------------------------
# Source detection + per-source samplers


def _detect_source() -> str:
    """Return the best available telemetry source for this host."""
    if _which("amd-smi"):
        return "amd-smi"
    if _which("rocm-smi"):
        return "rocm-smi"
    if _which("nvidia-smi"):
        return "nvidia-smi"
    if Path("/sys/class/powercap/intel-rapl:0/energy_uj").exists():
        return "rapl"
    if Path("/proc/cpuinfo").exists():
        return "procfs"
    return "none"


def _sample_amd_smi() -> List[Dict]:
    """One sample per visible AMD GPU via ``amd-smi monitor -p -t -c``.

    ``amd-smi monitor`` output is column-oriented and varies by amd-smi
    version. We parse the JSON output via ``--json`` if supported, falling
    back to a regex over the text columns.
    """
    j = _safe_run(["amd-smi", "monitor", "-p", "-t", "-c", "-u", "--json"])
    rows: List[Dict] = []
    if j:
        try:
            data = json.loads(j)
            for entry in data:
                idx = entry.get("gpu", entry.get("GPU", 0))
                rows.append({
                    "device_idx": int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else idx,
                    "power_w":    _to_float(entry.get("power") or entry.get("Power") or entry.get("power_w")),
                    "temp_c":     _to_float(entry.get("temperature") or entry.get("edge_temperature") or entry.get("temp")),
                    "clk_mhz":    _to_float(entry.get("clock_gfx") or entry.get("gpu_clock") or entry.get("clk_mhz")),
                    "util_pct":   _to_float(entry.get("gfx") or entry.get("utilization") or entry.get("util_pct")),
                })
            if rows:
                return rows
        except (ValueError, TypeError):
            pass
    # Text fallback: amd-smi monitor table parsing
    out = _safe_run(["amd-smi", "monitor", "-p", "-t", "-c", "-u"])
    if not out:
        return rows
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if not m:
            continue
        rows.append({
            "device_idx": int(m.group(1)),
            "power_w":    _to_float(m.group(2)),
            "temp_c":     _to_float(m.group(3)),
            "clk_mhz":    _to_float(m.group(4)),
            "util_pct":   _to_float(m.group(5)),
        })
    return rows


def _sample_rocm_smi() -> List[Dict]:
    j = _safe_run(["rocm-smi", "--showpower", "--showtemp", "--showclocks", "--json"])
    rows: List[Dict] = []
    if not j:
        return rows
    try:
        data = json.loads(j)
    except ValueError:
        return rows
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        m = re.match(r"card(\d+)", k)
        if not m:
            continue
        idx = int(m.group(1))
        rows.append({
            "device_idx": idx,
            "power_w":    _to_float(v.get("Average Graphics Package Power (W)")
                                    or v.get("Current Socket Graphics Package Power (W)")),
            "temp_c":     _to_float(v.get("Temperature (Sensor edge) (C)")
                                    or v.get("Temperature (Sensor junction) (C)")),
            "clk_mhz":    _parse_rocm_clk(v.get("sclk clock speed:")
                                         or v.get("sclk clock level: 0")),
            "util_pct":   None,
        })
    return rows


def _parse_rocm_clk(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*Mhz", s, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _sample_nvidia_smi() -> List[Dict]:
    out = _safe_run([
        "nvidia-smi",
        "--query-gpu=index,power.draw,temperature.gpu,clocks.current.graphics,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows: List[Dict] = []
    if not out:
        return rows
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append({
            "device_idx": int(parts[0]) if parts[0].isdigit() else 0,
            "power_w":    _to_float(parts[1]),
            "temp_c":     _to_float(parts[2]),
            "clk_mhz":    _to_float(parts[3]),
            "util_pct":   _to_float(parts[4]),
        })
    return rows


def _sample_rapl() -> List[Dict]:
    """Per-package CPU power via Intel/AMD RAPL.

    Power is computed from energy deltas across two reads spaced by ~50 ms
    so a single call returns instantaneous-ish power rather than zero. This
    is good enough for a 1 Hz sampler.
    """
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    pkgs = sorted(p for p in base.iterdir() if p.name.startswith("intel-rapl:"))
    if not pkgs:
        return []

    def _read_energy_uj(pkg: Path) -> Optional[int]:
        try:
            return int((pkg / "energy_uj").read_text().strip())
        except (OSError, ValueError):
            return None

    t0 = time.perf_counter()
    e0 = [_read_energy_uj(p) for p in pkgs]
    time.sleep(0.05)
    t1 = time.perf_counter()
    e1 = [_read_energy_uj(p) for p in pkgs]
    rows: List[Dict] = []
    for idx, (pkg, a, b) in enumerate(zip(pkgs, e0, e1)):
        if a is None or b is None:
            continue
        de = b - a
        # Wraparound guard: max_energy_range_uj exists on most platforms.
        if de < 0:
            try:
                m = int((pkg / "max_energy_range_uj").read_text().strip())
                de += m
            except (OSError, ValueError):
                de = 0
        dt = max(t1 - t0, 1e-6)
        power_w = (de * 1e-6) / dt
        # Per-package temperature (best effort).
        temp = _read_cpu_temp(idx)
        rows.append({
            "device_idx": idx,
            "package":    pkg.name,
            "power_w":    round(power_w, 2),
            "temp_c":     temp,
            "clk_mhz":    None,
            "util_pct":   None,
        })
    return rows


def _sample_procfs() -> List[Dict]:
    """No-power fallback: at minimum surface the per-CPU clock from
    ``/proc/cpuinfo`` and any hwmon temperature node we can find.

    Useful in containers/WSL where neither amd-smi/rocm-smi/nvidia-smi nor
    Intel RAPL is exposed, so the sampler still has a story to tell about
    clock frequency and temperature trends across a long run.
    """
    rows: List[Dict] = []
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        return rows
    mhz_vals: List[float] = []
    for m in re.finditer(r"^cpu MHz\s*:\s*([\d.]+)", info, re.MULTILINE):
        try:
            mhz_vals.append(float(m.group(1)))
        except ValueError:
            continue
    avg_mhz = sum(mhz_vals) / len(mhz_vals) if mhz_vals else None
    temp = _read_cpu_temp(0)
    rows.append({
        "device_idx": 0,
        "power_w":    None,
        "temp_c":     temp,
        "clk_mhz":    round(avg_mhz, 1) if avg_mhz is not None else None,
        "util_pct":   None,
        "n_cpus":     len(mhz_vals) or None,
    })
    return rows


def _read_cpu_temp(pkg_idx: int) -> Optional[float]:
    # Look for a coretemp / k10temp hwmon node whose ``name`` matches.
    base = Path("/sys/class/hwmon")
    if not base.exists():
        return None
    for h in base.iterdir():
        try:
            name = (h / "name").read_text().strip()
        except OSError:
            continue
        if name not in ("coretemp", "k10temp", "zenpower"):
            continue
        # Use temp1_input as the package temperature (Tdie / Package).
        f = h / "temp1_input"
        try:
            return int(f.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sampler

_SAMPLERS = {
    "amd-smi":    _sample_amd_smi,
    "rocm-smi":   _sample_rocm_smi,
    "nvidia-smi": _sample_nvidia_smi,
    "rapl":       _sample_rapl,
    "procfs":     _sample_procfs,
    "none":       lambda: [],
}


class TelemetryRecorder:
    """Background sampler. Use via the ``telemetry()`` context manager."""

    def __init__(self, out_path: Path, interval_s: float = 1.0,
                 source: Optional[str] = None) -> None:
        self.out_path = Path(out_path)
        self.interval_s = float(interval_s)
        self.source = source or _detect_source()
        self._sampler = _SAMPLERS.get(self.source, _SAMPLERS["none"])
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict] = []
        self._events: List[Dict] = []
        self._t0: Optional[float] = None
        self.started_iso: Optional[str] = None
        self.ended_iso: Optional[str] = None

    # -- public API --
    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_iso = _now_iso()
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="telemetry",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 2 + 1.0)
        self.ended_iso = _now_iso()
        self._thread = None
        self._write()

    def note(self, key: str, value: object) -> None:
        """Annotate the timeline with a free-form event."""
        if self._t0 is None:
            return
        self._events.append({
            "t_s":   round(time.perf_counter() - self._t0, 3),
            "key":   str(key),
            "value": value,
        })

    # -- internals --
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rows = self._sampler() or []
            except Exception:  # noqa: BLE001 — never raise into the bench thread
                rows = []
            t = round(time.perf_counter() - (self._t0 or 0), 3)
            for r in rows:
                self._samples.append({"t_s": t, **r})
            # If we got nothing back, still emit a heartbeat so the user can
            # tell the sampler is alive but the source returned no data.
            if not rows:
                self._samples.append({"t_s": t, "device_idx": -1,
                                      "power_w": None, "temp_c": None,
                                      "clk_mhz": None, "util_pct": None,
                                      "no_data": True})
            self._stop.wait(self.interval_s)

    def _summarize(self) -> Dict:
        """Per-field aggregate: mean / p95 / max / min."""
        out: Dict = {}
        for field in ("power_w", "temp_c", "clk_mhz", "util_pct"):
            vals = [s.get(field) for s in self._samples
                    if isinstance(s.get(field), (int, float))]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            p95 = vals_sorted[min(n - 1, int(n * 0.95))]
            out[field] = {
                "mean":  round(sum(vals) / n, 3),
                "p95":   round(p95, 3),
                "max":   round(max(vals), 3),
                "min":   round(min(vals), 3),
                "count": n,
            }
        return out

    def _write(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_iso": self.started_iso,
            "ended_iso":   self.ended_iso,
            "interval_s":  self.interval_s,
            "source":      self.source,
            "n_samples":   len(self._samples),
            "samples":     self._samples,
            "events":      self._events,
            "summary":     self._summarize(),
        }
        self.out_path.write_text(json.dumps(payload, indent=2))


@contextmanager
def telemetry(out_path: Path, interval_s: float = 1.0,
              source: Optional[str] = None,
              enabled: bool = True):
    """Context manager: start / stop a TelemetryRecorder around a benchmark.

    When ``enabled`` is False this is a no-op stub so callers can wire the
    sampler in unconditionally and gate it via a CLI flag.
    """
    if not enabled:
        yield None
        return
    rec = TelemetryRecorder(out_path, interval_s=interval_s, source=source)
    rec.start()
    try:
        yield rec
    finally:
        rec.stop()


# Allow ``python -m benchmarks.common.telemetry --probe`` for ad-hoc inspection.
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Detect source and emit a single sample to stdout.")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="Sample for N seconds and write to /tmp/telemetry.json.")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    if args.probe:
        src = _detect_source()
        sampler = _SAMPLERS[src]
        print(f"source={src}")
        print(json.dumps(sampler(), indent=2))
    if args.seconds > 0:
        out = Path(os.environ.get("TELEMETRY_OUT", "/tmp/telemetry.json"))
        with telemetry(out, interval_s=args.interval) as rec:
            time.sleep(args.seconds)
        print(f"wrote {out} (n_samples={len(rec._samples)})")
