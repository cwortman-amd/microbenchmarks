"""IO helpers: write JSON / CSV / Markdown artifacts under a campaign dir."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def write_csv(path: Path, rows: Sequence[Mapping]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return path


def write_md_table(path: Path, rows: Sequence[Mapping], title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    if not rows:
        lines.append("_(empty)_\n")
        path.write_text("\n".join(lines))
        return path
    keys = list(rows[0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
    for r in rows:
        vals: list[str] = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path
