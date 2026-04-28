#!/usr/bin/env python3
"""Top-level shortcut for the benchmark report generator.

This is a thin wrapper around `scripts/report.py` so users can invoke the
report generator with the same surface as `./test.sh` / `./run.sh` /
`./setup.sh` from the project root:

    ./report.py                             # uses the most recent benchmark results dir
    ./report.py --out results/<id>/         # explicit benchmark dir
    ./report.py --format md                 # md only
    ./report.py --format html --no-embed    # html with linked plots
    ./report.py --output-name myreport      # myreport.md / myreport.html

When `--out` is omitted, the wrapper picks the newest directory under
`$RESULTS_DIR` (default ``results/``) that looks like a benchmark output
(contains ``env.json`` and either ``-benchmark-`` in the name for legacy runs,
or matches ``<model>-YYYYMMDD-HHMMSS`` from the current naming scheme) and
forwards it to the underlying generator. All other arguments are forwarded
verbatim — see ``scripts/report.py --help`` for the full surface.
"""

from __future__ import annotations

import runpy
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts" / "report.py"

_NEW_BENCHMARK_DIR = re.compile(
    r"^[\w.-]+-\d{8}-\d{6}$"  # <model>-YYYYMMDD-HHMMSS (see test.sh / run.sh)
)


def _latest_benchmark_dir() -> Path | None:
    """Find the most recent benchmark output directory.

    Honors $RESULTS_DIR (test.sh / run.sh export it). Falls back to
    `<repo>/results` and, for backward compatibility with older runs,
    `<repo>/runs` if `results/` does not exist yet.
    """
    import os
    env_dir = os.environ.get("RESULTS_DIR")
    search_roots: list[Path] = []
    if env_dir:
        search_roots.append(Path(env_dir))
    search_roots.append(ROOT / "results")
    search_roots.append(ROOT / "runs")
    for root in search_roots:
        if not root.is_dir():
            continue
        candidates = [
            p for p in root.iterdir()
            if p.is_dir()
            and (p / "env.json").is_file()
            and not p.name.startswith("_")
            and (
                "benchmark" in p.name
                or _NEW_BENCHMARK_DIR.fullmatch(p.name) is not None
            )
        ]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def main() -> int:
    if not SCRIPTS.is_file():
        sys.stderr.write(f"ERROR: cannot find {SCRIPTS}\n")
        return 2

    argv = sys.argv[1:]

    # Auto-resolve --out to the most recent benchmark dir when omitted.
    has_out = any(a == "--out" or a.startswith("--out=") for a in argv)
    wants_help = any(a in ("-h", "--help") for a in argv)
    if not has_out and not wants_help:
        latest = _latest_benchmark_dir()
        if latest is None:
            sys.stderr.write(
                "ERROR: --out not provided and no benchmark directories found "
                "under results/ (or $RESULTS_DIR). Run `./test.sh -t benchmark` "
                "first, or pass `--out results/<id>/` explicitly.\n"
            )
            return 2
        sys.stderr.write(f"[report] auto-selected latest benchmark: {latest}\n")
        argv = ["--out", str(latest)] + argv

    # Delegate to scripts/report.py with the (possibly augmented) argv.
    sys.argv = [str(SCRIPTS)] + argv
    runpy.run_path(str(SCRIPTS), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
