"""Family 5 — End-to-end DiT block + MFU comparison (TESTPLAN §11).

Builds a `escher_14b_480p`-shaped DiT stack out of vanilla PyTorch modules and
runs three timed scopes:

  1. sum-of-ops    -> reuses 04_workload_ops measurements, summed * depth
  2. eager e2e     -> full forward, no torch.compile
  3. compiled e2e  -> full forward under torch.compile(mode=max-autotune)

For each, computes:

  flops_total = total_gflops_per_block * depth * 1e9
  tflops_achieved = flops_total / t_total_s / 1e12
  mfu_measured_peak = tflops_achieved / measured_peak_tflops    [from 01_]
  mfu_rated_peak_low  = tflops_achieved / 1.26e3
  mfu_rated_peak_high = tflops_achieved / 2.5e3

Outputs `mfu.json`, `mfu.csv`, and `mfu.md` per TESTPLAN §13 A8.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common.flop_accounting import WorkloadConfig, per_block_ops, totals
from benchmarks.common.io import write_csv, write_json, write_md_table
from benchmarks.common.timing import time_op


# ---------------------------------------------------------------------------
# Minimal DiT block (image self-attn + cross-attn + FFN), bf16 throughout.
# Shapes and weight count are approximate; the benchmark cares about FLOPs/sec
# under the consistent FLOP basis from §8.4, not parameter count.
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, cfg: WorkloadConfig):
        super().__init__()
        D = cfg.D
        Dh = cfg.n_heads * cfg.head_dim
        Dctx = cfg.context_dim
        Dff = D * cfg.ffn_expansion
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.norm1 = nn.LayerNorm(D)
        self.q1 = nn.Linear(D, Dh, bias=False)
        self.k1 = nn.Linear(D, Dh, bias=False)
        self.v1 = nn.Linear(D, Dh, bias=False)
        self.o1 = nn.Linear(Dh, D, bias=False)
        self.norm2 = nn.LayerNorm(D)
        self.q2 = nn.Linear(D, Dh, bias=False)
        self.k2 = nn.Linear(Dctx, Dh, bias=False)
        self.v2 = nn.Linear(Dctx, Dh, bias=False)
        self.o2 = nn.Linear(Dh, D, bias=False)
        self.norm3 = nn.LayerNorm(D)
        self.lin1 = nn.Linear(D, Dff, bias=False)
        self.lin2 = nn.Linear(Dff, D, bias=False)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, S, _ = t.shape
        return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        B, H, S, D = t.shape
        return t.transpose(1, 2).contiguous().view(B, S, H * D)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # self-attn
        h = self.norm1(x)
        q = self._split(self.q1(h))
        k = self._split(self.k1(h))
        v = self._split(self.v1(h))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o1(self._merge(a))
        # cross-attn
        h = self.norm2(x)
        q = self._split(self.q2(h))
        k = self._split(self.k2(ctx))
        v = self._split(self.v2(ctx))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o2(self._merge(a))
        # ffn
        h = self.norm3(x)
        x = x + self.lin2(F.gelu(self.lin1(h)))
        return x


class DiT(nn.Module):
    def __init__(self, cfg: WorkloadConfig):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.depth)])

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, ctx)
        return x


# ---------------------------------------------------------------------------

def _peak_from(out_dir: Path) -> Optional[float]:
    try:
        return json.loads((out_dir / "01_bf16_compute" / "summary.json").read_text())["compute_roof_tflops"]
    except Exception:  # noqa: BLE001
        return None


def _ops_total_per_block(out_dir: Path, cfg: WorkloadConfig) -> tuple[float, float]:
    """Returns (total_gflops_per_block, total_mb_per_block) — analytic, not measured."""
    ops = per_block_ops(cfg)
    t = totals(ops)
    return t["total_gflops"], t["total_mb_hbm"]


def _sum_of_ops_total_ms(out_dir: Path, optimized: bool, depth: int) -> tuple[Optional[float], float]:
    """Return (total_ms, coverage_fraction).

    ``coverage_fraction`` is the share of analytic per-block FLOPs that have a
    corresponding measured t_ms. When < 1.0 the sum-of-ops total is incomplete
    and downstream MFU computation should skip the row to avoid the
    "FLOPs from all ops divided by time from a few ops" mismatch.
    """
    p = out_dir / "04_workload_ops" / "ops.json"
    if not p.exists():
        return None, 0.0
    j = json.loads(p.read_text())
    key = "t_ms_optimized" if optimized else "t_ms_default"
    per_block = 0.0
    measured_flops = 0
    total_flops = 0
    for r in j["rows"]:
        f = float(r.get("flops") or 0)
        total_flops += f
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        per_block += float(v)
        measured_flops += f
    coverage = measured_flops / total_flops if total_flops else 0.0
    return per_block * depth, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--methodology", default="configs/test_methodology.json")
    ap.add_argument("--depth-override", type=int, default=None,
                    help="reduce blocks for memory-constrained smoke tests")
    ap.add_argument("--cpu-depth", type=int, default=2,
                    help="depth used on CPU hosts (full-shape depth is intractable). 0 disables override.")
    ap.add_argument("--cpu-seq-image", type=int, default=512,
                    help="seq_image used on CPU hosts. 0 disables override.")
    ap.add_argument("--cpu-seq-text", type=int, default=128,
                    help="seq_text used on CPU hosts. 0 disables override.")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile path entirely (default: try modes; "
                         "runs on both GPU and CPU hosts).")
    ap.add_argument("--compile-budget-s", type=float, default=180.0,
                    help="hard wall-clock budget per torch.compile mode attempt, "
                         "in seconds. If a mode does not produce a timed result "
                         "within this budget the next mode is tried. 0 disables "
                         "the budget.")
    args = ap.parse_args()

    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if has_gpu else torch.device("cpu")
    cfg_json = json.loads(Path(args.config).read_text())
    cfg = WorkloadConfig.from_json(cfg_json)

    # CPU host: downscale depth/seq aggressively before allocating the model.
    # The full escher_14b_480p shape is multi-minute per chunk on CPU; the
    # benchmark is here to validate the timing infra and produce real MFU
    # numbers, not to grind through 40 layers of a 14B-parameter model.
    cpu_overrides_applied = {}
    if not has_gpu:
        if args.cpu_depth and cfg.depth > args.cpu_depth:
            cpu_overrides_applied["depth"] = (cfg.depth, args.cpu_depth)
            cfg.depth = args.cpu_depth
        if args.cpu_seq_image and cfg.seq_image > args.cpu_seq_image:
            cpu_overrides_applied["seq_image"] = (cfg.seq_image, args.cpu_seq_image)
            cfg.seq_image = args.cpu_seq_image
        if args.cpu_seq_text and cfg.seq_text > args.cpu_seq_text:
            cpu_overrides_applied["seq_text"] = (cfg.seq_text, args.cpu_seq_text)
            cfg.seq_text = args.cpu_seq_text
        if cpu_overrides_applied:
            print("[05] CPU host detected — auto-downscaled "
                  + ", ".join(f"{k}: {old}->{new}"
                              for k, (old, new) in cpu_overrides_applied.items()))

    if args.depth_override:
        cfg.depth = args.depth_override

    # Methodology check
    m_cfg = {}
    if Path(args.methodology).is_file():
        m_cfg = json.loads(Path(args.methodology).read_text())
    timing = m_cfg.get("timing", cfg_json.get("timing", {}))
    
    # Methodology: fixed N=25 chunks, with first chunk warmup-only.
    chunks = int(timing.get("e2e_chunks", 25))
    warmup_chunks = int(timing.get("e2e_warmup_chunks", 1))

    out_dir_full = Path(args.out)
    out_dir = out_dir_full / "05_e2e_mfu"
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_peak = _peak_from(out_dir_full)
    gflops_per_block, mb_per_block = _ops_total_per_block(out_dir_full, cfg)
    flops_total = gflops_per_block * cfg.depth * 1e9
    print(f"[05] depth={cfg.depth}  flops_total={flops_total/1e12:.2f} TFLOP  "
          f"bytes_total={mb_per_block * cfg.depth / 1024:.1f} GB")

    # Build model and inputs.
    torch.manual_seed(0)
    model = DiT(cfg).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(cfg.batch, cfg.seq_image, cfg.D, device=device, dtype=torch.bfloat16)
    ctx = torch.randn(cfg.batch, cfg.seq_text, cfg.context_dim, device=device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def fwd_eager():
        model(x, ctx)

    # Eager E2E
    print("[05] timing eager e2e ...")
    res_eager = time_op("eager_e2e", fwd_eager, warmup=warmup_chunks, iters=chunks - warmup_chunks)

    # Compiled E2E (best-effort across torch.compile modes). Runs on both GPU
    # and CPU hosts so the MFU table can populate the compiled_e2e row that
    # the source PDF uses for its 99% reference target. CPU prefers the
    # cheaper modes first because Inductor's max-autotune routinely spends
    # tens of seconds in autotune on CPU for a marginal win.
    #
    # Each mode attempt is wrapped in a SIGALRM-based wall-clock budget so a
    # wedged Inductor autotune cannot hang the benchmark indefinitely. SIGALRM
    # is Linux-only and only fires while the main thread is in Python — a
    # native-code stall inside Inductor will still block past the alarm. Treat
    # the budget as best-effort, not a hard cap.
    res_compiled = None
    compile_mode_used = None

    if not args.no_compile:
        # Cheap modes first on CPU; max-autotune first on GPU where the win
        # justifies the autotune cost.
        compile_modes = (
            ("max-autotune", "reduce-overhead", "default")
            if has_gpu
            else ("default", "reduce-overhead")
        )

        # SIGALRM is POSIX-only; on Windows / non-main-thread contexts it
        # raises, in which case we fall back to "no budget" rather than
        # silently skipping the compile path.
        import signal as _signal  # local import — keep top-of-file lean

        class _CompileBudgetExceeded(Exception):
            pass

        def _alarm_handler(_signum, _frame):  # noqa: ANN001
            raise _CompileBudgetExceeded(
                f"torch.compile attempt exceeded {args.compile_budget_s:.0f}s budget"
            )

        budget_active = (
            args.compile_budget_s > 0
            and hasattr(_signal, "SIGALRM")
            and hasattr(_signal, "alarm")
        )

        for mode in compile_modes:
            try:
                print(f"[05] compiling mode={mode} ...")
                if budget_active:
                    prev = _signal.signal(_signal.SIGALRM, _alarm_handler)
                    _signal.alarm(int(args.compile_budget_s))
                compiled = torch.compile(model, mode=mode, fullgraph=False)
                @torch.inference_mode()
                def fwd_compiled(_c=compiled):
                    _c(x, ctx)
                print(f"[05] timing compiled ({mode}) e2e ...")
                res_compiled = time_op(
                    f"compiled_e2e_{mode}", fwd_compiled,
                    warmup=warmup_chunks, iters=chunks - warmup_chunks,
                )
                compile_mode_used = mode
                break
            except _CompileBudgetExceeded as e:
                print(f"[05] mode={mode} aborted: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"[05] mode={mode} failed: {e!r}")
            finally:
                if budget_active:
                    _signal.alarm(0)
                    _signal.signal(_signal.SIGALRM, prev)
    else:
        print("[05] --no-compile passed; skipping torch.compile path")

    # Sum of ops (from Family 4). Coverage < 1.0 means bench04 only timed a
    # subset (e.g. CPU fast-path that skips the heavy GEMMs); in that case
    # the sum-of-ops total is incomplete and MFU must be suppressed.
    sum_default_ms, coverage_default = _sum_of_ops_total_ms(out_dir_full, optimized=False, depth=cfg.depth)
    sum_optim_ms,   coverage_optim   = _sum_of_ops_total_ms(out_dir_full, optimized=True, depth=cfg.depth)

    rated_low_tflops = 1.26e3
    rated_high_tflops = 2.5e3
    coverage_threshold = 0.95  # require at least 95% of FLOPs covered

    def _mfu_row(name: str, t_total_ms: Optional[float],
                 coverage: float = 1.0,
                 timed: Optional["object"] = None) -> dict:
        # Always include the per-chunk distribution (times_ms + p10/p90/std)
        # when we actually timed this scope. plot_results.py uses it for the
        # per-chunk timing stability chart that mirrors the PDF's "compiled
        # e2e is faster *and* more stable" finding.
        dist: Dict[str, object] = {}
        if timed is not None:
            dist = {
                "times_ms": list(getattr(timed, "times_ms", []) or []),
                "p10_ms":   getattr(timed, "p10_ms", None),
                "p90_ms":   getattr(timed, "p90_ms", None),
                "min_ms":   getattr(timed, "min_ms", None),
                "max_ms":   getattr(timed, "max_ms", None),
                "std_ms":   getattr(timed, "std_ms", None),
                "iters":    getattr(timed, "iters", None),
                "warmup":   getattr(timed, "warmup", None),
            }
        if t_total_ms is None or (isinstance(t_total_ms, float) and t_total_ms != t_total_ms):
            return {"scope": name, "t_total_ms": None, "tflops_achieved": None,
                    "coverage_fraction": coverage, **dist}
        if coverage < coverage_threshold:
            return {
                "scope": name,
                "t_total_ms": t_total_ms,
                "tflops_achieved": None,
                "mfu_measured_peak": None,
                "mfu_rated_1_26pf": None,
                "mfu_rated_2_5pf": None,
                "coverage_fraction": coverage,
                "note": (f"sum-of-ops covers only {coverage*100:.1f}% of "
                         f"analytic FLOPs; MFU suppressed to avoid mismatched "
                         f"numerator/denominator"),
                **dist,
            }
        tflops = (flops_total / (t_total_ms * 1e-3)) / 1e12
        return {
            "scope": name,
            "t_total_ms": t_total_ms,
            "tflops_achieved": tflops,
            "mfu_measured_peak": (tflops / bf16_peak) if bf16_peak else None,
            "mfu_rated_1_26pf": tflops / rated_low_tflops,
            "mfu_rated_2_5pf":  tflops / rated_high_tflops,
            "coverage_fraction": coverage,
            **dist,
        }

    rows = [
        _mfu_row("sum_of_ops_default", sum_default_ms, coverage=coverage_default),
        _mfu_row("sum_of_ops_optimized", sum_optim_ms, coverage=coverage_optim),
        _mfu_row("eager_e2e", res_eager.median_ms, timed=res_eager),
        _mfu_row("compiled_e2e" if compile_mode_used else "compiled_e2e_unavailable",
                 res_compiled.median_ms if res_compiled else None,
                 timed=res_compiled),
    ]

    write_json(out_dir / "mfu.json", {
        "device_type": device.type,
        "depth": cfg.depth,
        "seq_image": cfg.seq_image,
        "seq_text": cfg.seq_text,
        "cpu_overrides": cpu_overrides_applied,
        "flops_total": flops_total,
        "compute_roof_tflops": bf16_peak,
        "rated_peak_tflops_low": rated_low_tflops,
        "rated_peak_tflops_high": rated_high_tflops,
        "compile_mode_used": compile_mode_used,
        # PDF reference targets on the measured-chip-peak basis (TESTPLAN §11.3).
        # Plot / report overlay these against the benchmark's measured MFU bars
        # so the comparison the PDF makes is reproduced visually. The values
        # come from the workload config's ``source_pilot_reference`` block —
        # a single auditable home for every "the source PDF says X" number,
        # alongside ``reference_totals_per_block``. We keep a defensive
        # fallback (the documented Odyssey targets) so older workload configs
        # still produce a populated mfu.json.
        "pdf_reference_targets_pct":
            ((cfg_json.get("source_pilot_reference") or {})
                     .get("pdf_reference_targets_pct")
             or {"sum_of_ops_optimized": 77,
                 "sum_of_ops_default":   77,
                 "eager_e2e":            93,
                 "compiled_e2e":         99}),
        "rows": rows,
    })
    write_csv(out_dir / "mfu.csv", rows)
    write_md_table(out_dir / "mfu.md", rows, title="MFU comparison: sum-of-ops vs eager vs compiled")

    for r in rows:
        print(f"[05] {r['scope']:32s} t={r.get('t_total_ms')}  "
              f"TFLOP/s={r.get('tflops_achieved')}  "
              f"MFU(measured)={r.get('mfu_measured_peak')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
