"""Family 8 — Topology / inter-device bandwidth matrix (TESTPLAN §16.4).

All-pairs device-to-device (GPU) or inter-CCD/inter-socket (CPU) bandwidth
matrix. The matrix tells you which pairs of devices share a fast fabric
(NVLink, xGMI, Infinity Fabric within a socket) vs a slow one (PCIe peer,
UPI cross-socket, no peering). On modern AMD MI355X this is what surfaces
the 8-GPU-per-node ring topology that bench06 implicitly relies on.

Output: ``08_topology_bw/topology.json`` with::

    {
      "device_type":  "cuda" | "cpu",
      "n_devices":    N,
      "device_names": ["MI355X 0", ...],
      "matrix_gb_s":  N×N float matrix (None on diagonal),
      "summary": {
         "max_pair_gb_s":    ...,
         "min_offdiag_gb_s": ...,
         "asymmetry_ratio":  max(|m[i][j] - m[j][i]|) / max(m),
      },
      "pair_classes": [{"pair": [i, j], "class": "intra-socket"|"inter-socket"|...}],
    }

CPU notes:

- On a single-CCD WSL/container host the matrix collapses to a 1×1 cell
  and the bench reports "no fabric crossing to measure".
- For multi-CCD / multi-socket hosts, the bench pins the timing thread to
  the producer CCD, allocates the destination buffer while pinned to the
  consumer CCD (Linux first-touch policy is what places it there), then
  times a copy that crosses the Infinity Fabric. This is a *proxy* for
  inter-CCD bandwidth — exact numbers depend on the kernel's NUMA policy.

GPU notes:

- Uses ``torch.Tensor.copy_`` between bf16 tensors of size ``--bytes-mib``
  (default 256 MiB). Times per-pair via ``torch.cuda.Event``.
- Diagonal elements are intentionally ``None`` (intra-device copy is HBM
  bandwidth, already covered by bench02).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from benchmarks.common.io import write_json
from benchmarks.common.topology import detect_cpu_topology, pin_to_cpus


def _cuda_event_time_ms(fn) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def _gpu_pairwise(bytes_per_buf: int, iters: int) -> Dict:
    n = torch.cuda.device_count()
    if n == 0:
        return {"reason": "no CUDA/HIP devices visible", "matrix_gb_s": None}
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    # bf16 = 2 B/elem
    elems = bytes_per_buf // 2
    bf16 = torch.bfloat16

    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    for src in range(n):
        a = torch.empty(elems, dtype=bf16, device=f"cuda:{src}")
        a.fill_(1.0)
        for dst in range(n):
            if dst == src:
                continue
            b = torch.empty(elems, dtype=bf16, device=f"cuda:{dst}")
            b.fill_(0.0)
            torch.cuda.synchronize()
            # warmup
            for _ in range(2):
                b.copy_(a, non_blocking=True)
            torch.cuda.synchronize()
            samples = []
            for _ in range(iters):
                t = _cuda_event_time_ms(lambda: b.copy_(a, non_blocking=True))
                samples.append(t)
            samples.sort()
            t_med = samples[len(samples) // 2]
            gb = bytes_per_buf / 1e9
            matrix[src][dst] = round(gb / (t_med * 1e-3), 2)
            del b
        del a
    return {
        "device_type": "cuda",
        "n_devices":   n,
        "device_names": names,
        "matrix_gb_s":  matrix,
    }


def _cpu_pairwise(bytes_per_buf: int, iters: int) -> Dict:
    """Inter-CCD / inter-socket bandwidth proxy for CPU hosts."""
    topo = detect_cpu_topology()
    groups = topo.get("ccd_groups") or topo.get("socket_groups") or []
    n = len(groups)
    base = {
        "device_type": "cpu",
        "topology": {
            "source":          topo.get("source"),
            "sockets":         topo.get("sockets"),
            "dies":            topo.get("dies"),
            "dies_per_socket": topo.get("dies_per_socket"),
            "cores_per_die":   topo.get("cores_per_die"),
            "total_cpus":      topo.get("total_cpus"),
        },
        "n_devices": n,
        "device_names": [f"CCD{i} (cpus={len(g)})" for i, g in enumerate(groups)],
    }
    if n < 2:
        return {**base, "reason":
                "single-CCD/single-socket host; no fabric crossing to measure",
                "matrix_gb_s": None}

    elems = bytes_per_buf // 4  # int32 elements
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    for src in range(n):
        for dst in range(n):
            if dst == src:
                continue
            # Allocate destination buffer while pinned to its CCD so Linux
            # first-touch places its pages on dst's local memory.
            pin_to_cpus(groups[dst])
            b = torch.zeros(elems, dtype=torch.int32)
            b.fill_(0)
            # Allocate source buffer pinned to src.
            pin_to_cpus(groups[src])
            a = torch.empty(elems, dtype=torch.int32)
            a.fill_(1)
            # Warmup
            for _ in range(2):
                b.copy_(a)
            # Time the cross-fabric copy from src's perspective.
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                b.copy_(a)
                t1 = time.perf_counter()
                samples.append((t1 - t0) * 1000.0)
            samples.sort()
            t_med = samples[len(samples) // 2]
            gb = bytes_per_buf / 1e9
            matrix[src][dst] = round(gb / (t_med * 1e-3), 2)
            del a, b
    return {**base, "matrix_gb_s": matrix}


def _summarize(matrix: Optional[List[List[Optional[float]]]]) -> Dict:
    if not matrix or not matrix[0]:
        return {}
    n = len(matrix)
    offdiag = [matrix[i][j] for i in range(n) for j in range(n)
               if i != j and matrix[i][j] is not None]
    if not offdiag:
        return {}
    asymmetry = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = matrix[i][j], matrix[j][i]
            if a is None or b is None:
                continue
            asymmetry = max(asymmetry, abs(a - b))
    max_pair = max(offdiag)
    return {
        "max_pair_gb_s":     round(max_pair, 2),
        "min_offdiag_gb_s":  round(min(offdiag), 2),
        "mean_offdiag_gb_s": round(sum(offdiag) / len(offdiag), 2),
        "asymmetry_gb_s":    round(asymmetry, 2),
        "asymmetry_ratio":   round(asymmetry / max_pair, 3) if max_pair else None,
        "n_pairs_measured":  len(offdiag),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--bytes-mib", type=int, default=256,
                    help="Buffer size per-pair (MiB). Big enough to leave L3.")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    # Methodology check
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    t_cfg = m_cfg.get("timing", {})
    
    def _is_set(opt): return any(o in a for a in sys.argv for o in (opt, opt.replace("--iters", "--iterations")))
    iters_val = args.iters if _is_set("--iters") else t_cfg.get("timed_iters", 5)

    out_dir = Path(args.out) / "08_topology_bw"
    out_dir.mkdir(parents=True, exist_ok=True)
    bytes_per_buf = args.bytes_mib * 1024 * 1024

    if torch.cuda.is_available():
        result = _gpu_pairwise(bytes_per_buf, iters_val)
    else:
        result = _cpu_pairwise(bytes_per_buf, iters_val)

    result["bytes_per_buf"]     = bytes_per_buf
    result["iters"]              = iters_val
    result["summary"]            = _summarize(result.get("matrix_gb_s"))

    write_json(out_dir / "topology.json", result)

    matrix = result.get("matrix_gb_s")
    print(f"[08] device_type={result.get('device_type')} "
          f"n_devices={result.get('n_devices')} "
          f"buf={args.bytes_mib} MiB")
    if "reason" in result:
        print(f"[08] note: {result['reason']}")
    elif matrix:
        print("[08] all-pairs bandwidth (GB/s):")
        names = result.get("device_names") or []
        header = "src/dst |" + " | ".join(f"{i:>8d}" for i in range(len(matrix)))
        print(f"[08]   {header}")
        for i, row in enumerate(matrix):
            cells = " | ".join(
                f"{v:>8.1f}" if isinstance(v, (int, float)) else f"{'-':>8}"
                for v in row
            )
            print(f"[08]   {i:7d} | {cells}    ({names[i] if i < len(names) else ''})")
        s = result["summary"]
        print(f"[08] summary: max={s.get('max_pair_gb_s')}  "
              f"min={s.get('min_offdiag_gb_s')}  "
              f"asym_ratio={s.get('asymmetry_ratio')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
