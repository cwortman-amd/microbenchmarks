"""Aggregate per-world strong-scaling benchmarks into a TP-3 table.

Walks ``<sweep>/world_*/`` looking for ``06_multigpu_comm/comm.json`` and
``05_e2e_mfu/mfu.json``. For each world, extracts:

  - per-collective plateau busbw (largest payload of each op),
  - per-collective time at a fixed reference payload (default 134 MiB),
  - end-to-end MFU (eager, compiled) where available,
  - strong-scaling efficiency relative to the smallest world.

Writes ``tp3_table.md``, ``tp3_table.json``, and ``tp3_table.csv`` under
the sweep directory. The Markdown table is what TESTPLAN §16.3 wants
shipped alongside the per-world reports; the JSON / CSV are for
downstream regression analysis.

Usage::

    python scripts/strong_scaling_table.py --sweep-dir results/<sweep>/

The script never fails when individual worlds are missing — partial
sweeps still produce a partial table with explicit ``n/a`` rows so the
gap is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# Reference payload for the "time @ ref payload" column. 128 MiB matches
# the largest CPU-path payload (and one tier below the GPU max), which
# keeps the column meaningful across host classes.
REF_PAYLOAD_BYTES = 128 * 1024 * 1024


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def _world_from_dirname(name: str) -> Optional[int]:
    m = re.match(r"world_(\d+)$", name)
    return int(m.group(1)) if m else None


def _collect_per_world(sweep_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        w = _world_from_dirname(child.name)
        if w is None:
            continue
        rows.append(_world_metrics(child, w))
    rows.sort(key=lambda r: r["world"])
    return rows


def _nearest_payload(comm_rows: List[Dict], op: str, ref_bytes: int) -> Optional[Dict]:
    """Pick the comm row for `op` whose `bytes` is closest to `ref_bytes`."""
    candidates = [r for r in comm_rows if r.get("op") == op]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs((r.get("bytes") or 0) - ref_bytes))


def _plateau(comm_rows: List[Dict], op: str) -> Optional[float]:
    """Largest measured payload, return its busbw_gb_s — i.e., the asymptote."""
    candidates = [r for r in comm_rows if r.get("op") == op]
    if not candidates:
        return None
    big = max(candidates, key=lambda r: r.get("bytes") or 0)
    return big.get("busbw_gb_s")


def _world_metrics(world_dir: Path, world: int) -> Dict:
    comm = _load(world_dir / "06_multigpu_comm" / "comm.json") or {}
    mfu  = _load(world_dir / "05_e2e_mfu" / "mfu.json") or {}
    rows = comm.get("rows") or []

    out: Dict = {
        "world":           world,
        "benchmark_dir":    str(world_dir),
        "backend":         comm.get("backend"),
        "device_type":     comm.get("device_type"),
        "n_collective_rows": len(rows),
    }

    for op in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all"):
        plateau = _plateau(rows, op)
        ref = _nearest_payload(rows, op, REF_PAYLOAD_BYTES)
        out[f"{op}_plateau_gb_s"] = plateau
        out[f"{op}_t_ref_ms"]     = ref.get("t_ms") if ref else None
        out[f"{op}_ref_bytes"]    = ref.get("bytes") if ref else None

    # MFU comes from bench05; rank-0 record is what we want.
    rank0_mfu = None
    if isinstance(mfu, dict):
        for r in (mfu.get("rows") or []):
            if r.get("rank") in (None, 0):
                rank0_mfu = r
                break
        rank0_mfu = rank0_mfu or (mfu.get("rows") or [None])[0]
    if rank0_mfu:
        out["mfu_eager_pct"]      = rank0_mfu.get("mfu_eager_pct")
        out["mfu_compiled_pct"]   = rank0_mfu.get("mfu_compiled_pct")
        out["e2e_eager_ms"]       = rank0_mfu.get("eager_e2e_ms")
        out["e2e_compiled_ms"]    = rank0_mfu.get("compiled_e2e_ms")

    return out


def _add_scaling_efficiency(rows: List[Dict]) -> None:
    """Compute strong-scaling efficiency vs the smallest world.

    For each collective, ``efficiency(W) = (W_min / W) * (busbw(W) /
    busbw(W_min))``. A value of 1.0 means the busbw scaled linearly with
    world (perfect scaling — every added rank delivers proportionally
    more aggregate bandwidth). On real interconnects efficiency drops
    below 1 because the per-link bandwidth doesn't grow linearly.
    """
    if not rows:
        return
    base = rows[0]
    base_w = base["world"]
    for op in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all"):
        bp = base.get(f"{op}_plateau_gb_s")
        for r in rows:
            if r is base:
                r[f"{op}_efficiency"] = 1.0 if bp else None
                continue
            bp_r = r.get(f"{op}_plateau_gb_s")
            w = r.get("world") or 0
            if bp and bp_r and w:
                r[f"{op}_efficiency"] = (base_w / w) * (bp_r / bp)
            else:
                r[f"{op}_efficiency"] = None


def _fmt(v, places: int = 2, na: str = "n/a") -> str:
    if v is None:
        return na
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return na
        return f"{v:.{places}f}"
    return str(v)


def _write_md(rows: List[Dict], dst: Path) -> None:
    lines: List[str] = ["# TP-3 strong-scaling table\n"]
    if not rows:
        lines.append("_(no per-world benchmarks found in this sweep)_\n")
        dst.write_text("\n".join(lines))
        return

    sample = rows[0]
    lines.append(
        f"Backend: `{sample.get('backend')}` — "
        f"device: `{sample.get('device_type')}`. "
        f"Reference payload: **{REF_PAYLOAD_BYTES // (1024 * 1024)} MiB**.\n"
    )

    lines.append("## Plateau bus-bandwidth per collective (GB/s)\n")
    lines.append("| world | AG | RS | AR | A2A |")
    lines.append("|------:|----:|----:|----:|----:|")
    for r in rows:
        lines.append("| {w} | {ag} | {rs} | {ar} | {a2a} |".format(
            w=r["world"],
            ag=_fmt(r.get("all_gather_plateau_gb_s"), 1),
            rs=_fmt(r.get("reduce_scatter_plateau_gb_s"), 1),
            ar=_fmt(r.get("all_reduce_plateau_gb_s"), 1),
            a2a=_fmt(r.get("all_to_all_plateau_gb_s"), 1),
        ))

    lines.append("\n## Strong-scaling efficiency vs smallest world\n")
    lines.append(
        "Efficiency = `(W_min / W) · (busbw(W) / busbw(W_min))`. "
        "1.0 = perfect linear scaling.\n"
    )
    lines.append("| world | AG | RS | AR | A2A |")
    lines.append("|------:|----:|----:|----:|----:|")
    for r in rows:
        lines.append("| {w} | {ag} | {rs} | {ar} | {a2a} |".format(
            w=r["world"],
            ag=_fmt(r.get("all_gather_efficiency"), 2),
            rs=_fmt(r.get("reduce_scatter_efficiency"), 2),
            ar=_fmt(r.get("all_reduce_efficiency"), 2),
            a2a=_fmt(r.get("all_to_all_efficiency"), 2),
        ))

    lines.append(f"\n## Time at reference payload "
                 f"({REF_PAYLOAD_BYTES // (1024 * 1024)} MiB) (ms)\n")
    lines.append("| world | AG | RS | AR | A2A |")
    lines.append("|------:|----:|----:|----:|----:|")
    for r in rows:
        lines.append("| {w} | {ag} | {rs} | {ar} | {a2a} |".format(
            w=r["world"],
            ag=_fmt(r.get("all_gather_t_ref_ms"), 2),
            rs=_fmt(r.get("reduce_scatter_t_ref_ms"), 2),
            ar=_fmt(r.get("all_reduce_t_ref_ms"), 2),
            a2a=_fmt(r.get("all_to_all_t_ref_ms"), 2),
        ))

    if any(r.get("mfu_eager_pct") is not None for r in rows):
        lines.append("\n## End-to-end MFU per world (rank 0)\n")
        lines.append(
            "Single-rank MFU; identical across worlds when bench05 doesn't "
            "shard the model. Differences here would surface multi-rank "
            "framework overhead leaking into the single-rank E2E measurement.\n"
        )
        lines.append("| world | MFU eager (%) | MFU compiled (%) | "
                     "e2e eager (ms) | e2e compiled (ms) |")
        lines.append("|------:|--------------:|----------------:|"
                     "---------------:|-----------------:|")
        for r in rows:
            lines.append("| {w} | {me} | {mc} | {ee} | {ec} |".format(
                w=r["world"],
                me=_fmt(r.get("mfu_eager_pct"), 1),
                mc=_fmt(r.get("mfu_compiled_pct"), 1),
                ee=_fmt(r.get("e2e_eager_ms"), 2),
                ec=_fmt(r.get("e2e_compiled_ms"), 2),
            ))

    dst.write_text("\n".join(lines) + "\n")


def _write_csv(rows: List[Dict], dst: Path) -> None:
    if not rows:
        dst.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    keys = ["world"] + [k for k in keys if k != "world"]
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        raise SystemExit(f"sweep dir not found: {args.sweep_dir}")

    rows = _collect_per_world(args.sweep_dir)
    _add_scaling_efficiency(rows)

    json_path = args.sweep_dir / "tp3_table.json"
    md_path   = args.sweep_dir / "tp3_table.md"
    csv_path  = args.sweep_dir / "tp3_table.csv"

    json_path.write_text(json.dumps({
        "ref_payload_bytes": REF_PAYLOAD_BYTES,
        "rows":              rows,
    }, indent=2, default=str))
    _write_md(rows, md_path)
    _write_csv(rows, csv_path)
    print(f"[strong-scaling-table] wrote "
          f"{md_path.name} / {json_path.name} / {csv_path.name} "
          f"({len(rows)} world(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
