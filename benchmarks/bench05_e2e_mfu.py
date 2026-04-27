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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.common.flop_accounting import WorkloadConfig, per_block_ops, totals
from benchmarks.common.io import write_csv, write_json, write_md_table
from benchmarks.common.timing import time_op


# ---------------------------------------------------------------------------
# Minimal DiT block (image self-attn + cross-attn + FFN), bf16 throughout.
# Shapes and weight count are approximate; the campaign cares about FLOPs/sec
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


def _sum_of_ops_total_ms(out_dir: Path, optimized: bool, depth: int) -> Optional[float]:
    p = out_dir / "04_workload_ops" / "ops.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    key = "t_ms_optimized" if optimized else "t_ms_default"
    per_block = 0.0
    for r in j["rows"]:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        per_block += float(v)
    return per_block * depth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default="configs/escher_14b_480p.json")
    ap.add_argument("--depth-override", type=int, default=None,
                    help="reduce blocks for memory-constrained smoke tests")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required")
    device = torch.device("cuda:0")
    cfg_json = json.loads(Path(args.config).read_text())
    cfg = WorkloadConfig.from_json(cfg_json)
    if args.depth_override:
        cfg.depth = args.depth_override

    timing = cfg_json.get("timing", {})
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

    # Compiled E2E (best-effort across torch.compile modes)
    res_compiled = None
    compile_mode_used = None
    for mode in ("max-autotune", "reduce-overhead", "default"):
        try:
            print(f"[05] compiling mode={mode} ...")
            compiled = torch.compile(model, mode=mode, fullgraph=False)
            @torch.inference_mode()
            def fwd_compiled(_c=compiled):
                _c(x, ctx)
            print(f"[05] timing compiled ({mode}) e2e ...")
            res_compiled = time_op(f"compiled_e2e_{mode}", fwd_compiled,
                                    warmup=warmup_chunks + 1, iters=chunks - warmup_chunks)
            compile_mode_used = mode
            break
        except Exception as e:  # noqa: BLE001
            print(f"[05] mode={mode} failed: {e!r}")

    # Sum of ops (from Family 4)
    sum_of_ops_default_total_ms = _sum_of_ops_total_ms(out_dir_full, optimized=False, depth=cfg.depth)
    sum_of_ops_optim_total_ms = _sum_of_ops_total_ms(out_dir_full, optimized=True, depth=cfg.depth)

    rated_low_tflops = 1.26e3
    rated_high_tflops = 2.5e3

    def _mfu_row(name: str, t_total_ms: Optional[float]) -> dict:
        if t_total_ms is None or (isinstance(t_total_ms, float) and t_total_ms != t_total_ms):
            return {"scope": name, "t_total_ms": None, "tflops_achieved": None}
        tflops = (flops_total / (t_total_ms * 1e-3)) / 1e12
        return {
            "scope": name,
            "t_total_ms": t_total_ms,
            "tflops_achieved": tflops,
            "mfu_measured_peak": (tflops / bf16_peak) if bf16_peak else None,
            "mfu_rated_1_26pf": tflops / rated_low_tflops,
            "mfu_rated_2_5pf":  tflops / rated_high_tflops,
        }

    rows = [
        _mfu_row("sum_of_ops_default", sum_of_ops_default_total_ms),
        _mfu_row("sum_of_ops_optimized", sum_of_ops_optim_total_ms),
        _mfu_row("eager_e2e", res_eager.median_ms),
        _mfu_row("compiled_e2e" if compile_mode_used else "compiled_e2e_unavailable",
                 res_compiled.median_ms if res_compiled else None),
    ]

    write_json(out_dir / "mfu.json", {
        "depth": cfg.depth,
        "flops_total": flops_total,
        "compute_roof_tflops": bf16_peak,
        "compile_mode_used": compile_mode_used,
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
