"""Run and summarize the HipKittens/Iris fused-kernel experiment matrix.

This wrapper intentionally separates *measured* experiments from variants that
are already baked into the default kernel or not supported by the current native
extension. The output is a single matrix suitable for `KERNEL_EXP.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".microbenchmarks-rocm-venv" / "bin" / "python"

VARIANT_ENVS = {
    "default": {},
    "reuse2": {"HIPKITTENS_AG_N_REUSE": "2"},
    "spillfree": {"HIPKITTENS_AG_N_REUSE": "2_spillfree"},
    "auto": {"HIPKITTENS_AG_N_REUSE": "auto"},
}

MMRS_VARIANT_ENVS = {
    "default": {},
    "specialized": {"HIPKITTENS_MM_RS_REDUCER": "specialized"},
    "vec4": {"HIPKITTENS_MM_RS_REDUCER": "vec4"},
    "auto": {"HIPKITTENS_MM_RS_REDUCER": "auto"},
    "swizzle": {"HIPKITTENS_MM_RS_SWIZZLE": "1"},
    "vec4_swizzle": {"HIPKITTENS_MM_RS_REDUCER": "vec4", "HIPKITTENS_MM_RS_SWIZZLE": "1"},
    "auto_swizzle": {"HIPKITTENS_MM_RS_REDUCER": "auto", "HIPKITTENS_MM_RS_SWIZZLE": "1"},
    "double_buffer": {"HIPKITTENS_MM_RS_DOUBLE_BUFFER": "1"},
    "vec4_double_buffer": {"HIPKITTENS_MM_RS_REDUCER": "vec4", "HIPKITTENS_MM_RS_DOUBLE_BUFFER": "1"},
    "vec4_swizzle_double_buffer": {"HIPKITTENS_MM_RS_REDUCER": "vec4", "HIPKITTENS_MM_RS_SWIZZLE": "1", "HIPKITTENS_MM_RS_DOUBLE_BUFFER": "1"},
}

RESOURCE_SUMMARY = {
    "default": {"vgpr": 135, "sgpr": 57, "scratch_bytes_per_lane": 0, "vgpr_spills": 0, "occupancy_waves_per_simd": 3},
    "reuse2": {"vgpr": 168, "sgpr": 73, "scratch_bytes_per_lane": 180, "vgpr_spills": 108, "occupancy_waves_per_simd": 3},
    "spillfree": {"vgpr": 130, "sgpr": 73, "scratch_bytes_per_lane": 0, "vgpr_spills": 0, "occupancy_waves_per_simd": 3},
    "auto": {"vgpr": 130, "sgpr": 73, "scratch_bytes_per_lane": 0, "vgpr_spills": 0, "occupancy_waves_per_simd": 3},
}


def _python() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        value = row.get(key)
        return None if value in (None, "") else float(value)
    except ValueError:
        return None


def _row(rows: Iterable[Dict[str, str]], op: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("op") == op:
            return row
    return None


def _pattern_row(rows: Iterable[Dict[str, str]], pattern: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("pattern") == pattern:
            return row
    return None


def _parse_shapes(shapes: str) -> List[str]:
    out: List[str] = []
    for item in shapes.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"shape must be M,K,N; got {item!r}")
        out.append(",".join(parts))
    return out


def _run_bench(out: Path, shape: str, nproc: int, warmup: int, iters: int, extra_env: Dict[str, str]) -> Dict[str, object]:
    env = os.environ.copy()
    env.update({
        "BENCH06_USE_IRIS": "1",
        "AITER_KERNELS_BACKEND": "hipkittens",
    })
    env.update(extra_env)
    cmd = [
        _python(),
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "benchmarks/bench06_aiter_fused.py",
        "--out",
        str(out),
        "--shapes",
        shape,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    completed = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    log = out / "bench06.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + "\n" + completed.stderr)
    rows = _read_csv(out / "06_multigpu_fused" / "fused.csv")
    overlap = _read_csv(out / "06_multigpu_fused" / "overlap.csv")
    return {
        "returncode": completed.returncode,
        "rows": rows,
        "overlap": overlap,
        "log": str(log),
    }


def _measured_row(
    *,
    experiment_id: str,
    experiment: str,
    op: str,
    run: Dict[str, object],
    baseline_op: str,
    notes: str,
    status: str = "measured",
) -> Dict[str, object]:
    rows = run["rows"] if isinstance(run["rows"], list) else []
    exp = _row(rows, op)
    base = _row(rows, baseline_op)
    exp_ms = _float(exp or {}, "t_ms")
    base_ms = _float(base or {}, "t_ms")
    speedup = (base_ms / exp_ms) if exp_ms and base_ms else None
    return {
        "experiment_id": experiment_id,
        "experiment": experiment,
        "status": status if run.get("returncode") == 0 else "failed",
        "op": op,
        "world": (exp or base or {}).get("world", ""),
        "shape": ",".join((exp or base or {}).get(k, "") for k in ("M", "K", "N")),
        "baseline": baseline_op,
        "baseline_ms": f"{base_ms:.6f}" if base_ms is not None else "",
        "experiment_ms": f"{exp_ms:.6f}" if exp_ms is not None else "",
        "speedup": f"{speedup:.4f}" if speedup is not None else "",
        "api_source": (exp or {}).get("api_source", ""),
        "path": (exp or {}).get("path", ""),
        "log": run.get("log", ""),
        "notes": notes,
    }


def _shape_ladder_rows(runs: Dict[tuple, Dict[str, object]], shapes: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for shape in shapes:
        default_run = runs[(shape, "default")]
        default_rows = default_run["rows"] if isinstance(default_run["rows"], list) else []
        default_ag = _row(default_rows, "ag_mm")
        default_unfused = _row(default_rows, "unfused_ag_mm")
        default_ms = _float(default_ag or {}, "t_ms")
        unfused_ms = _float(default_unfused or {}, "t_ms")
        for variant in VARIANT_ENVS:
            run = runs[(shape, variant)]
            fused_rows = run["rows"] if isinstance(run["rows"], list) else []
            overlap_rows = run["overlap"] if isinstance(run["overlap"], list) else []
            ag = _row(fused_rows, "ag_mm")
            overlap = _pattern_row(overlap_rows, "ag_mm")
            t_ms = _float(ag or {}, "t_ms")
            vs_default = (default_ms / t_ms) if default_ms and t_ms else None
            vs_unfused = (unfused_ms / t_ms) if unfused_ms and t_ms else None
            resource = RESOURCE_SUMMARY.get(variant, {})
            rows.append({
                "shape": shape,
                "variant": variant,
                "status": "measured" if run.get("returncode") == 0 else "failed",
                "world": (ag or default_ag or {}).get("world", ""),
                "t_ms": f"{t_ms:.6f}" if t_ms is not None else "",
                "default_ms": f"{default_ms:.6f}" if default_ms is not None else "",
                "unfused_ms": f"{unfused_ms:.6f}" if unfused_ms is not None else "",
                "speedup_vs_default_hk": f"{vs_default:.4f}" if vs_default is not None else "",
                "speedup_vs_unfused": f"{vs_unfused:.4f}" if vs_unfused is not None else "",
                "tflops": (ag or {}).get("tflops", ""),
                "ect_fused_ms": (overlap or {}).get("ect_fused_ms", ""),
                "overlap_efficiency": (overlap or {}).get("overlap_efficiency", ""),
                "vgpr": resource.get("vgpr", ""),
                "sgpr": resource.get("sgpr", ""),
                "scratch_bytes_per_lane": resource.get("scratch_bytes_per_lane", ""),
                "vgpr_spills": resource.get("vgpr_spills", ""),
                "occupancy_waves_per_simd": resource.get("occupancy_waves_per_simd", ""),
                "log": run.get("log", ""),
            })
    return rows


def _mmrs_ladder_rows(runs: Dict[tuple, Dict[str, object]], shapes: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for shape in shapes:
        default_run = runs[(shape, "default")]
        default_rows = default_run["rows"] if isinstance(default_run["rows"], list) else []
        default_mmrs = _row(default_rows, "mm_rs")
        default_unfused = _row(default_rows, "unfused_mm_rs")
        default_ms = _float(default_mmrs or {}, "t_ms")
        unfused_ms = _float(default_unfused or {}, "t_ms")
        for variant in MMRS_VARIANT_ENVS:
            run = runs[(shape, variant)]
            fused_rows = run["rows"] if isinstance(run["rows"], list) else []
            overlap_rows = run["overlap"] if isinstance(run["overlap"], list) else []
            mmrs = _row(fused_rows, "mm_rs")
            overlap = _pattern_row(overlap_rows, "mm_rs")
            t_ms = _float(mmrs or {}, "t_ms")
            vs_default = (default_ms / t_ms) if default_ms and t_ms else None
            vs_unfused = (unfused_ms / t_ms) if unfused_ms and t_ms else None
            rows.append({
                "shape": shape,
                "variant": variant,
                "status": "measured" if run.get("returncode") == 0 else "failed",
                "world": (mmrs or default_mmrs or {}).get("world", ""),
                "t_ms": f"{t_ms:.6f}" if t_ms is not None else "",
                "default_ms": f"{default_ms:.6f}" if default_ms is not None else "",
                "unfused_ms": f"{unfused_ms:.6f}" if unfused_ms is not None else "",
                "speedup_vs_default_hk": f"{vs_default:.4f}" if vs_default is not None else "",
                "speedup_vs_unfused": f"{vs_unfused:.4f}" if vs_unfused is not None else "",
                "tflops": (mmrs or {}).get("tflops", ""),
                "ect_fused_ms": (overlap or {}).get("ect_fused_ms", ""),
                "overlap_efficiency": (overlap or {}).get("overlap_efficiency", ""),
                "log": run.get("log", ""),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/hk-experiment-matrix"))
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--ladder-shapes",
        default="256,64,512",
        help="Semicolon-separated shape ladder, e.g. '256,64,512;512,256,512'.",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    shapes = _parse_shapes(args.ladder_shapes)
    runs: Dict[tuple, Dict[str, object]] = {}
    for shape in shapes:
        shape_tag = shape.replace(",", "_")
        for variant, env in VARIANT_ENVS.items():
            runs[(shape, variant)] = _run_bench(
                out / f"shape_{shape_tag}_{variant}",
                shape,
                args.nproc,
                args.warmup,
                args.iters,
                env,
            )
    mmrs_runs: Dict[tuple, Dict[str, object]] = {}
    for shape in shapes:
        shape_tag = shape.replace(",", "_")
        for variant, env in MMRS_VARIANT_ENVS.items():
            mmrs_runs[(shape, variant)] = _run_bench(
                out / f"mmrs_shape_{shape_tag}_{variant}",
                shape,
                args.nproc,
                args.warmup,
                args.iters,
                env,
            )

    default_n256 = _run_bench(out / "exp01_hk_default_n256", "256,64,256", args.nproc, args.warmup, args.iters, {})
    default_n512 = runs.get(("256,64,512", "default")) or _run_bench(out / "exp_default_n512", "256,64,512", args.nproc, args.warmup, args.iters, {})
    reuse2 = runs.get(("256,64,512", "reuse2")) or _run_bench(out / "exp02_reuse2", "256,64,512", args.nproc, args.warmup, args.iters, {"HIPKITTENS_AG_N_REUSE": "2"})
    spillfree = runs.get(("256,64,512", "spillfree")) or _run_bench(out / "exp08_reuse2_spillfree", "256,64,512", args.nproc, args.warmup, args.iters, {"HIPKITTENS_AG_N_REUSE": "2_spillfree"})
    mmrs_specialized = mmrs_runs.get(("256,64,256", "specialized")) or _run_bench(out / "exp06_mmrs_specialized", "256,64,256", args.nproc, args.warmup, args.iters, {"HIPKITTENS_MM_RS_REDUCER": "specialized"})

    matrix: List[Dict[str, object]] = [
        _measured_row(
            experiment_id="1",
            experiment="HK GEMM baseline inside AITER/HipKittens",
            op="ag_mm",
            run=default_n256,
            baseline_op="unfused_ag_mm",
            notes="Default native HK/Iris AG+MM path; establishes CDNA4 tile/MFMA baseline.",
        ),
        _measured_row(
            experiment_id="2",
            experiment="Iris-backed remote A staging with N-panel reuse",
            op="ag_mm",
            run=reuse2,
            baseline_op="unfused_ag_mm",
            notes="Original reuse=2 implementation; architecturally valid but known to spill VGPRs.",
        ),
        _measured_row(
            experiment_id="3",
            experiment="Double buffering",
            op="ag_mm",
            run=default_n512,
            baseline_op="unfused_ag_mm",
            status="covered_by_default",
            notes="Default HK kernel uses tic/toc LDS double buffering; no no-double-buffer variant is currently compiled.",
        ),
        {
            "experiment_id": "4",
            "experiment": "LDS transpose on read-back path",
            "status": "unsupported",
            "op": "ag_mm",
            "world": args.nproc,
            "shape": "",
            "baseline": "",
            "baseline_ms": "",
            "experiment_ms": "",
            "speedup": "",
            "api_source": "hk_iris_fused",
            "path": "",
            "log": "",
            "notes": "No LDS-transpose variant exists in the native extension yet; current operands use HK row-layout register tiles.",
        },
        _measured_row(
            experiment_id="5",
            experiment="Chiplet-aware swizzling and grid mapping",
            op="ag_mm",
            run=default_n512,
            baseline_op="unfused_ag_mm",
            status="covered_by_default",
            notes="Default HK kernels call chiplet_transform_chunked; no no-swizzle variant is currently compiled.",
        ),
        _measured_row(
            experiment_id="6",
            experiment="MM+RS world-specialized reducer",
            op="mm_rs",
            run=mmrs_specialized,
            baseline_op="unfused_mm_rs",
            notes="First valid reducer experiment: keep writer+scratch handoff, remove scalar slot loop/branching for world=2/4/8.",
        ),
        {
            "experiment_id": "7",
            "experiment": "FP8/FP6 variants",
            "status": "unsupported",
            "op": "ag_mm",
            "world": args.nproc,
            "shape": "",
            "baseline": "",
            "baseline_ms": "",
            "experiment_ms": "",
            "speedup": "",
            "api_source": "hk_iris_fused",
            "path": "",
            "log": "",
            "notes": "Native HK/Iris extension currently exposes BF16 dispatch only; no FP8/FP6 AITER/HK fused entry point is available.",
        },
        _measured_row(
            experiment_id="8",
            experiment="Spill-free AG+MM reuse design",
            op="ag_mm",
            run=spillfree,
            baseline_op="unfused_ag_mm",
            notes="New reuse=2_spillfree design splits N blocks across consumer groups and compiles with zero VGPR spills.",
        ),
    ]

    _write_csv(out / "hk_experiment_matrix.csv", matrix)
    ladder = _shape_ladder_rows(runs, shapes)
    _write_csv(out / "hk_shape_ladder.csv", ladder)
    mmrs_ladder = _mmrs_ladder_rows(mmrs_runs, shapes)
    _write_csv(out / "hk_mmrs_ladder.csv", mmrs_ladder)
    (out / "hk_experiment_matrix.json").write_text(json.dumps(matrix, indent=2))
    (out / "hk_shape_ladder.json").write_text(json.dumps(ladder, indent=2))
    (out / "hk_mmrs_ladder.json").write_text(json.dumps(mmrs_ladder, indent=2))
    print(f"wrote {out / 'hk_experiment_matrix.csv'}")
    print(f"wrote {out / 'hk_shape_ladder.csv'}")
    print(f"wrote {out / 'hk_mmrs_ladder.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
