"""Pure-PyTorch reference implementations of the fused ops.

These are the **correctness gold** for every other backend in this package.
They mirror the semantics of:

  * ``torch.distributed._symmetric_memory._fused_all_gather_matmul_fallback``
  * ``torch.distributed._symmetric_memory._fused_matmul_reduce_scatter_fallback``

so a kernel that passes ``torch.testing.assert_close`` against these
fallbacks is guaranteed to be a drop-in replacement at any TP-linear
call-site that today calls ``torch.ops.symm_mem.fused_*``.

The fallbacks are intentionally written without optimization tricks:

  * AG is staged through a real ``dist.all_gather_into_tensor``;
  * RS is staged through a real ``dist.reduce_scatter_tensor`` with the
    matmul done locally first;
  * dim handling matches the upstream helper exactly (gather/scatter on
    any dim, not just dim=0).

Anything more clever belongs in the Triton backend.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist


def _gather_along_dim(local: torch.Tensor, dim: int, world: int) -> torch.Tensor:
    """All-gather ``local`` along ``dim`` and return the concatenated tensor.

    Uses ``all_gather_into_tensor`` when the input is dim-0 contiguous (the
    fast path); otherwise transposes, gathers along dim 0, and transposes
    back. Matches the dim-handling in
    ``torch.distributed._symmetric_memory._fused_all_gather_matmul_fallback``.
    """
    if not local.is_contiguous():
        local = local.contiguous()
    world_size = world
    if dim == 0:
        out_shape = list(local.shape)
        out_shape[0] *= world_size
        out = torch.empty(out_shape, dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(out, local)
        return out
    permuted = local.movedim(dim, 0).contiguous()
    out_shape = list(permuted.shape)
    out_shape[0] *= world_size
    out_perm = torch.empty(out_shape, dtype=permuted.dtype, device=permuted.device)
    dist.all_gather_into_tensor(out_perm, permuted)
    return out_perm.movedim(0, dim).contiguous()


def _reduce_scatter_along_dim(
    full: torch.Tensor, dim: int, world: int, reduce_op: str
) -> torch.Tensor:
    """Reduce-scatter ``full`` along ``dim``, returning the local shard.

    ``reduce_op`` ∈ {"sum", "avg"} matches what
    ``torch.ops.symm_mem.fused_matmul_reduce_scatter`` accepts.
    """
    op = dist.ReduceOp.SUM
    if not full.is_contiguous():
        full = full.contiguous()
    if dim == 0:
        out_shape = list(full.shape)
        out_shape[0] //= world
        out = torch.empty(out_shape, dtype=full.dtype, device=full.device)
        dist.reduce_scatter_tensor(out, full, op=op)
    else:
        permuted = full.movedim(dim, 0).contiguous()
        out_shape = list(permuted.shape)
        out_shape[0] //= world
        out_perm = torch.empty(out_shape, dtype=permuted.dtype, device=permuted.device)
        dist.reduce_scatter_tensor(out_perm, permuted, op=op)
        out = out_perm.movedim(0, dim).contiguous()
    if reduce_op == "avg":
        out = out / world
    elif reduce_op != "sum":
        raise ValueError(f"reduce_op must be 'sum' or 'avg', got {reduce_op!r}")
    return out


def fused_all_gather_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    *,
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Pure-Torch reference for ``torch.ops.symm_mem.fused_all_gather_matmul``.

    Returns ``(A_full, [Y_i])`` where:

      * ``A_full = all_gather(A_shard, dim=gather_dim)``
      * ``Y_i = A_full @ B_i`` for every ``B_i`` in ``Bs``

    Output dtype mirrors the matmul: ``Y_i.dtype == result_type(A_full, B_i)``.
    """
    group = dist.distributed_c10d._resolve_process_group(group_name)
    world = group.size()
    A_full = _gather_along_dim(A_shard, dim=gather_dim, world=world)
    outputs = [torch.matmul(A_full, B) for B in Bs]
    return A_full, outputs


def fused_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    *,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Pure-Torch reference for ``torch.ops.symm_mem.fused_matmul_reduce_scatter``.

    Computes ``Y = A @ B`` and then reduce-scatters ``Y`` along
    ``scatter_dim`` using ``reduce_op``.
    """
    group = dist.distributed_c10d._resolve_process_group(group_name)
    world = group.size()
    Y_full = torch.matmul(A, B)
    return _reduce_scatter_along_dim(Y_full, dim=scatter_dim, world=world, reduce_op=reduce_op)
