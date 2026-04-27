"""CPU topology detection and rank partitioning.

This module is the CPU analogue of multi-GPU enumeration for bench06. Modern
AMD EPYC and Ryzen Threadripper / PRO parts ship as multiple CCDs (Core
Complex Dies) connected via Infinity Fabric to a central I/O die, and dual-
socket boards add a second tier of inter-socket interconnect. When we run
bench06's collective sweep on a CPU host, "world size" should match that
hardware-meaningful boundary so that the all-reduce / all-gather numbers
reflect *Infinity Fabric* (intra-socket) or *xGMI / UPI* (inter-socket)
bandwidth, not just memcpy within a single CCD.

Topology source: ``/sys/devices/system/cpu/cpuN/topology/{physical_package_id,
die_id, core_id}`` — the Linux kernel exposes the CCD as the "die" on AMD
parts. Anything that doesn't expose a die_id (e.g. flattened hypervisor
topology, older kernels) falls through to a "split" mode that just carves
the visible CPU set into N equal groups so the bench is still useful.

Usage::

    from benchmarks.common.topology import (
        detect_cpu_topology, partition_cpus, pin_to_cpus,
    )

    topo = detect_cpu_topology()
    cpus = partition_cpus(topo, world=topo["dies"], mode="ccd")[rank]
    pin_to_cpus(cpus)

The returned dict is JSON-serialisable so it can be embedded directly into
bench06's ``comm.json``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


SYS_CPU = Path("/sys/devices/system/cpu")


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def detect_cpu_topology() -> Dict:
    """Probe sysfs and return a JSON-serialisable topology dict.

    Keys:
      sockets, dies, cores_per_die, threads_per_core, total_cpus,
      ccd_groups (list of CPU-id lists, one per CCD),
      socket_groups (list of CPU-id lists, one per socket),
      online_cpus, source ("sysfs" | "fallback").
    """
    online_cpus: List[int] = []
    if SYS_CPU.exists():
        for p in sorted(SYS_CPU.glob("cpu[0-9]*"), key=lambda x: int(x.name[3:])):
            try:
                online_cpus.append(int(p.name[3:]))
            except ValueError:
                continue

    if not online_cpus:
        # Fallback: assume a flat CPU set sized to os.cpu_count().
        n = os.cpu_count() or 1
        online_cpus = list(range(n))
        return _fallback_topology(online_cpus)

    cpu_info = []
    for c in online_cpus:
        topo = SYS_CPU / f"cpu{c}" / "topology"
        socket = _read_int(topo / "physical_package_id")
        die = _read_int(topo / "die_id")
        core = _read_int(topo / "core_id")
        if socket is None or die is None or core is None:
            return _fallback_topology(online_cpus)
        # WSL/hypervisor sometimes returns die_id == -1; treat as "no CCD".
        if die < 0:
            die = 0
        cpu_info.append((c, socket, die, core))

    by_die: Dict[tuple, List[int]] = defaultdict(list)
    by_socket: Dict[int, List[int]] = defaultdict(list)
    cores_per_die_set: Dict[tuple, set] = defaultdict(set)
    for c, s, d, core in cpu_info:
        by_die[(s, d)].append(c)
        by_socket[s].append(c)
        cores_per_die_set[(s, d)].add(core)

    ccd_groups = [sorted(by_die[k]) for k in sorted(by_die)]
    socket_groups = [sorted(by_socket[k]) for k in sorted(by_socket)]
    sockets = len(by_socket)
    dies = len(by_die)
    cores_per_die = max(len(v) for v in cores_per_die_set.values()) if cores_per_die_set else 0
    threads_per_core = (len(cpu_info) // dies // cores_per_die) if (dies and cores_per_die) else 1

    return {
        "source": "sysfs",
        "sockets": sockets,
        "dies": dies,
        "dies_per_socket": dies // sockets if sockets else 0,
        "cores_per_die": cores_per_die,
        "threads_per_core": threads_per_core,
        "total_cpus": len(cpu_info),
        "ccd_groups": ccd_groups,
        "socket_groups": socket_groups,
        "online_cpus": online_cpus,
    }


def _fallback_topology(online_cpus: List[int]) -> Dict:
    """When sysfs topology is unavailable, treat the CPU set as a single
    CCD and let the caller fall back to ``mode='split'``."""
    return {
        "source": "fallback",
        "sockets": 1,
        "dies": 1,
        "dies_per_socket": 1,
        "cores_per_die": len(online_cpus),
        "threads_per_core": 1,
        "total_cpus": len(online_cpus),
        "ccd_groups": [list(online_cpus)],
        "socket_groups": [list(online_cpus)],
        "online_cpus": list(online_cpus),
    }


def partition_cpus(topology: Dict, world: int, mode: str = "auto") -> List[List[int]]:
    """Return ``world`` lists of logical CPU ids; one per rank.

    Modes:
      * ``"ccd"`` — one rank per CCD (Linux ``die_id``); requires
        ``world <= topology["dies"]`` and even divisibility.
      * ``"socket"`` — one rank per socket; requires
        ``world <= topology["sockets"]``.
      * ``"split"`` — ignore topology and carve the union of all online
        CPUs into ``world`` equal-sized contiguous slices. Useful in VMs /
        hypervisors that flatten topology, or when the user wants more
        ranks than physical CCDs for stress / oversubscription tests.
      * ``"auto"`` — try ``ccd``; if that doesn't fit, try ``socket``;
        finally fall through to ``split``.

    Raises ``ValueError`` when no valid partition exists.
    """
    if world < 1:
        raise ValueError(f"world must be >=1, got {world}")
    chosen = mode
    groups: List[List[int]] = []

    def _try_grouped(key: str) -> Optional[List[List[int]]]:
        gs = topology.get(f"{key}_groups") or []
        if not gs or world > len(gs) or len(gs) % world != 0:
            return None
        # Pack adjacent groups together when world < #groups so that
        # rank-0 lands on a physically-contiguous slice (CCDs 0..k, then
        # CCDs k..2k, etc.). This matches how torchrun assigns LOCAL_RANK.
        per_rank = len(gs) // world
        out = []
        for r in range(world):
            cpus: List[int] = []
            for j in range(per_rank):
                cpus.extend(gs[r * per_rank + j])
            out.append(sorted(cpus))
        return out

    if mode in ("ccd", "auto"):
        groups = _try_grouped("ccd") or []
        if groups:
            chosen = "ccd"
    if not groups and mode in ("socket", "auto"):
        groups = _try_grouped("socket") or []
        if groups:
            chosen = "socket"
    if not groups and mode in ("split", "auto"):
        all_cpus = sorted(topology.get("online_cpus") or [])
        if not all_cpus or world > len(all_cpus):
            raise ValueError(
                f"split partition needs >=1 CPU per rank "
                f"(world={world}, cpus={len(all_cpus)})"
            )
        per_rank = len(all_cpus) // world
        groups = [
            sorted(all_cpus[r * per_rank: (r + 1) * per_rank])
            for r in range(world)
        ]
        chosen = "split"
    if not groups:
        raise ValueError(
            f"no partition of topology={topology.get('source')} "
            f"into world={world} ranks under mode={mode}"
        )
    # Side-channel for callers: stash the resolved mode on the topology
    # dict so JSON output can record what we actually did.
    topology["_resolved_partition_mode"] = chosen
    return groups


def pin_to_cpus(cpus: List[int]) -> bool:
    """Pin the calling thread to ``cpus``. Returns True on success."""
    if not cpus:
        return False
    try:
        os.sched_setaffinity(0, set(cpus))
        return True
    except (OSError, AttributeError):
        return False


def describe_partition(topology: Dict, mode: str, world: int,
                       rank_cpus: List[List[int]]) -> Dict:
    """Build a JSON-friendly summary block for bench output."""
    return {
        "topology_source": topology.get("source"),
        "topology_mode": mode,
        "sockets": topology.get("sockets"),
        "dies": topology.get("dies"),
        "dies_per_socket": topology.get("dies_per_socket"),
        "cores_per_die": topology.get("cores_per_die"),
        "threads_per_core": topology.get("threads_per_core"),
        "total_cpus": topology.get("total_cpus"),
        "world": world,
        "rank_pinning": [
            {"rank": r, "n_cpus": len(cpus), "cpus": cpus}
            for r, cpus in enumerate(rank_cpus)
        ],
    }
