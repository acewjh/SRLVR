"""Tensor-parallel collectives for the exact SRLVR scoring/loss paths.

The output layer and the fused LoRA layers are tensor-parallel when TP > 1.
Every full-vocabulary reduction in this package must then be an exact sharded
reduction: local max/sum (or partial expectation) followed by a TP collective.
No code here ever all-gathers a ``[T, V]`` tensor.

Autograd rules used below:
  * sum reductions that feed a differentiable expression use
    ``ReduceFromTensorParallel`` (forward sum, backward passes
    the upstream gradient to every rank's local contribution).
  * the global max used for numerical stability is a *shift*; it is detached and
    non-differentiable.  ``logsumexp`` is shift invariant, so detaching is exact.
  * replicated parameters need their gradients summed across TP; apply
    ``AllReduceGradSum`` to the parameter before its local use.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def tp_info():
    """(size, rank, group) of the tensor-parallel group, or (1, 0, None)."""
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return 1, 0, None
        from megatron.core import mpu

        size = mpu.get_tensor_model_parallel_world_size()
        rank = mpu.get_tensor_model_parallel_rank()
        return int(size), int(rank), mpu.get_tensor_model_parallel_group()
    except Exception:
        return 1, 0, None


def vocab_start(local_vocab: int, rank: int, size: int) -> int:
    if size <= 1:
        return 0
    try:
        from megatron.core.tensor_parallel.utils import VocabUtility

        start, _ = VocabUtility.vocab_range_from_per_partition_vocab_size(
            int(local_vocab), int(rank), int(size)
        )
        return int(start)
    except Exception:
        return int(rank) * int(local_vocab)


def tp_sum(value: torch.Tensor, group, differentiable: bool = True) -> torch.Tensor:
    """All-reduce SUM across TP. Differentiable when a graph is being built."""
    if group is None or (dist.is_available() and dist.is_initialized()
                         and dist.get_world_size(group) == 1):
        return value
    if differentiable and torch.is_grad_enabled() and value.requires_grad:
        return ReduceFromTensorParallel.apply(value, group)
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


class SumGradFromTensorParallel(torch.autograd.Function):
    """all_reduce SUM forward, all_reduce SUM backward.

    Needed for a *shared* reduction whose result every rank then uses to build
    its own local term -- the canonical case is the global log-partition of a
    vocabulary-parallel softmax::

        gsum   = all_reduce(local_sum_r)          # identical on every rank
        logp_r = s_r - log(gsum)                  # each rank's local term
        loss_r = kl(logp_r, teacher_r)

    ``ReduceFromTensorParallel`` (pass-through backward) is right for the final
    total, which is summed *and* consumed once per rank. It is wrong for
    ``gsum``: every rank's ``loss_r`` depends on the same ``gsum``, so the true
    derivative accumulates all ranks' partials. Using the pass-through version
    there leaves the *value* exact and the *gradient* wrong -- measured on
    2026-10-28: a 2-rank SRLVR-v4 KL matched dense to 4e-16 in value while the
    hidden-state gradient was off by 49%, and the resulting full-RL run collapsed
    (action-valid 0.34 -> 0.00 by rollout 20) even though ``grad_norm`` looked
    healthy.
    """

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        out = value.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        if ctx.group is None:
            return grad, None
        out = grad.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def tp_sum_shared(value: torch.Tensor, group) -> torch.Tensor:
    """all_reduce SUM for a value that EVERY rank then consumes (see above)."""
    if group is None or (dist.is_available() and dist.is_initialized()
                         and dist.get_world_size(group) == 1):
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        return SumGradFromTensorParallel.apply(value, group)
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def tp_max_detached(value: torch.Tensor, group) -> torch.Tensor:
    if group is None:
        return value.detach()
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.MAX, group=group)
    return out


class ReduceFromTensorParallel(torch.autograd.Function):
    """Sum local contributions; replicated loss supplies one upstream gradient.

    All-reducing backward here would multiply the gradient by TP size. The
    complementary grad_sum at the replicated hidden input performs the one
    required sum of the vocabulary shards' input gradients.
    """

    @staticmethod
    def forward(ctx, value, group):
        out = value.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class AllReduceGradSum(torch.autograd.Function):
    """Identity forward; SUM the gradient across TP in backward.

    Used for replicated adapter matrices (column-parallel A, row-parallel B):
    each rank sees the same matrix but a different local output/input shard, so
    the global gradient is the sum of the local contributions.
    """

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, grad):
        if ctx.group is None:
            return grad, None
        out = grad.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def grad_sum(value: torch.Tensor, group) -> torch.Tensor:
    if group is None:
        return value
    return AllReduceGradSum.apply(value, group)


def local_vocab_valid(local_vocab: int, start: int, vocab_size: int, device) -> torch.Tensor:
    """Boolean mask over a rank's vocabulary shard for the model's real vocab."""
    idx = torch.arange(local_vocab, device=device) + start
    return idx < int(vocab_size)
