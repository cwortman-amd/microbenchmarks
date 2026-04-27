"""Analytic FLOP / HBM-byte accounting for the escher_14b_480p workload.

The conventions here MUST match TESTPLAN §8.4:

  - GEMM: 2 * M * N * K  (multiply+add counted, no fused-multiply-add halving)
  - QK^T and AV in attention: counted as standard GEMMs over (B, H, S, D)
  - softmax / GELU / norms / element-wise: bandwidth-only (no FLOP credit
    beyond the negligible scalar work)

Bytes are HBM bytes assuming bf16 (2 B/element), no recompute, no L2 reuse,
counting weight reads + activation reads + activation writes for each op.

The aggregator returns the same column set §8.2 prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


BYTES_PER_BF16 = 2


@dataclass
class OpAcct:
    op_name: str
    category: str
    input_shape: str
    output_shape: str
    flops: int
    bytes_hbm: int

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / max(self.bytes_hbm, 1)

    def as_row(self) -> Dict:
        d = asdict(self)
        d["arithmetic_intensity"] = round(self.arithmetic_intensity, 4)
        return d


def gemm_op(
    name: str,
    category: str,
    M: int,
    K: int,
    N: int,
    *,
    weight_bytes: bool = True,
    bias_bytes: bool = False,
) -> OpAcct:
    """A single dense GEMM C[M,N] = A[M,K] @ B[K,N].

    HBM bytes = read(A) + read(B if weight_bytes) + write(C) + read(bias if bias_bytes).
    """
    flops = 2 * M * N * K
    b = (M * K) + (M * N)
    if weight_bytes:
        b += K * N
    if bias_bytes:
        b += N
    return OpAcct(
        op_name=name,
        category=category,
        input_shape=f"({M},{K})x({K},{N})",
        output_shape=f"({M},{N})",
        flops=flops,
        bytes_hbm=b * BYTES_PER_BF16,
    )


def attention_flash(
    name: str,
    category: str,
    B: int,
    H: int,
    S_q: int,
    S_kv: int,
    D: int,
) -> OpAcct:
    """Flash-attention compute: softmax(QK^T / sqrt(D)) V.

    FLOPs = 2 * B * H * S_q * S_kv * D     # QK^T
          + 2 * B * H * S_q * S_kv * D     # AV
          ≈ 4 * B * H * S_q * S_kv * D

    HBM bytes (no-recompute, on-chip softmax) ≈ Q + K + V + O reads/writes:
          (B*H*S_q*D) + 2*(B*H*S_kv*D) + (B*H*S_q*D) elements.
    """
    flops = 4 * B * H * S_q * S_kv * D
    elements = (B * H * S_q * D) + 2 * (B * H * S_kv * D) + (B * H * S_q * D)
    return OpAcct(
        op_name=name,
        category=category,
        input_shape=f"Q({B},{H},{S_q},{D})/K({B},{H},{S_kv},{D})/V({B},{H},{S_kv},{D})",
        output_shape=f"({B},{H},{S_q},{D})",
        flops=flops,
        bytes_hbm=elements * BYTES_PER_BF16,
    )


def elementwise_op(name: str, category: str, n_elements: int, n_streams: int) -> OpAcct:
    """Bandwidth-only op (norms, GELU, residual add).

    `n_streams` = number of HBM read+write passes (e.g. add(a,b) -> 3,
    gelu(x) inplace -> 2, norm(x) ≈ 4 with mean+var two-pass + write).
    """
    return OpAcct(
        op_name=name,
        category=category,
        input_shape=f"({n_elements},)",
        output_shape=f"({n_elements},)",
        flops=0,
        bytes_hbm=n_elements * n_streams * BYTES_PER_BF16,
    )


@dataclass
class WorkloadConfig:
    depth: int
    hidden_dim: int
    n_heads: int
    head_dim: int
    ffn_expansion: int
    context_dim: int
    batch: int
    seq_image: int
    seq_text: int

    @classmethod
    def from_json(cls, j) -> "WorkloadConfig":
        m = j["model"]
        s = j["shapes"]
        return cls(
            depth=m["depth"],
            hidden_dim=m["hidden_dim"],
            n_heads=m["n_heads"],
            head_dim=m["head_dim"],
            ffn_expansion=m["ffn_expansion"],
            context_dim=m.get("context_dim", m["hidden_dim"]),
            batch=s["batch"],
            seq_image=s["seq_image"],
            seq_text=s["seq_text"],
        )

    @property
    def D(self) -> int:
        return self.hidden_dim

    @property
    def Dh(self) -> int:
        return self.n_heads * self.head_dim


def per_block_ops(cfg: WorkloadConfig) -> List[OpAcct]:
    """Op decomposition per transformer block (TESTPLAN §8.1)."""
    B = cfg.batch
    S = cfg.seq_image
    L = cfg.seq_text
    D = cfg.D
    n = cfg.n_heads
    h = cfg.head_dim
    Dh = n * h
    Dctx = cfg.context_dim
    Dff = D * cfg.ffn_expansion

    M_image = B * S
    M_text = B * L

    ops: List[OpAcct] = [
        # Time embedding (small projections; bandwidth-bound)
        gemm_op("time_proj",     "time",       M=B,         K=256,  N=D),
        gemm_op("time_embed",    "time",       M=B,         K=D,    N=D),
        # Pre-norm before attention
        elementwise_op("norm_pre_attn", "norm", n_elements=M_image * D, n_streams=4),
        # Self-attention QKV
        gemm_op("self_attn.q",   "self_attn",  M=M_image,   K=D,    N=Dh),
        gemm_op("self_attn.k",   "self_attn",  M=M_image,   K=D,    N=Dh),
        gemm_op("self_attn.v",   "self_attn",  M=M_image,   K=D,    N=Dh),
        attention_flash("self_attn.flash", "self_attn", B=B, H=n, S_q=S, S_kv=S, D=h),
        gemm_op("self_attn.o",   "self_attn",  M=M_image,   K=Dh,   N=D),
        elementwise_op("residual_post_self_attn", "norm", n_elements=M_image * D, n_streams=3),
        # Pre-norm before cross-attention
        elementwise_op("norm_pre_cross_attn", "norm", n_elements=M_image * D, n_streams=4),
        # Cross-attention: Q from image, K/V from text context (ctx is precomputed; we still read it)
        gemm_op("cross_attn.q",  "cross_attn", M=M_image,   K=D,    N=Dh),
        gemm_op("cross_attn.k",  "cross_attn", M=M_text,    K=Dctx, N=Dh),
        gemm_op("cross_attn.v",  "cross_attn", M=M_text,    K=Dctx, N=Dh),
        attention_flash("cross_attn.flash", "cross_attn", B=B, H=n, S_q=S, S_kv=L, D=h),
        gemm_op("cross_attn.o",  "cross_attn", M=M_image,   K=Dh,   N=D),
        elementwise_op("residual_post_cross_attn", "norm", n_elements=M_image * D, n_streams=3),
        # Pre-norm before FFN
        elementwise_op("norm_pre_ffn", "norm", n_elements=M_image * D, n_streams=4),
        # FFN
        gemm_op("ffn.linear1",   "ffn",        M=M_image,   K=D,    N=Dff),
        elementwise_op("ffn.gelu", "ffn",      n_elements=M_image * Dff, n_streams=2),
        gemm_op("ffn.linear2",   "ffn",        M=M_image,   K=Dff,  N=D),
        elementwise_op("residual_post_ffn", "norm", n_elements=M_image * D, n_streams=3),
        # Optional KV cache write (most diffusion DiTs don't cache; track 0 if N/A)
        elementwise_op("kv_cache_write", "norm", n_elements=0, n_streams=0),
    ]
    return ops


def totals(ops: List[OpAcct]) -> Dict[str, float]:
    flops = sum(o.flops for o in ops)
    bytes_ = sum(o.bytes_hbm for o in ops)
    return {
        "total_flops": flops,
        "total_gflops": flops / 1e9,
        "total_bytes_hbm": bytes_,
        "total_mb_hbm": bytes_ / 1e6,
        "avg_arithmetic_intensity": flops / max(bytes_, 1),
    }
