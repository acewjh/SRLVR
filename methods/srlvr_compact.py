"""Memory-exact SRLVR activation paths: row-length padding + response-only lm head.

Two independent costs are removed here; neither changes the objective.

1. Row-length padding.  In bshd training every row of a local step was padded to
   the longest row in the whole step (``actor._get_rollout_data``), and the raw
   probes padded every turn to the longest turn of their trajectory.  With
   ``micro_batch_size == 1`` each microbatch contains exactly one row, so the
   padding buys nothing.  ``padded_lengths`` returns the per-row alignment that
   the caller installs in ``max_seq_lens``; the framework's own bshd slicing
   (``loss.get_responses``: ``max_seq_len * i + total_length``) already indexes
   one row per micro.

2. Response-only output projection.  The stock forward builds logits for every
   prompt and padding position (``[T, V]`` bf16, 2.25 GiB at T=7936,
   V=151936) even though the loss only reads the response slice.  When this
   module is enabled the model is run with ``post_process = False`` (hidden
   states ``[S, B, H]``), and log-probabilities are computed by projecting only
   the positions whose ``full_loss_masks`` entry is set, in bounded chunks.
   Each chunk is recomputed during backward with ``torch.utils.checkpoint`` so
   the cross-entropy softmax transient never exceeds one chunk.

The projection uses the model's own output weight, the same full-vocabulary
softmax (``fused_vocab_parallel_cross_entropy`` on device, TP=1 -> full vocab),
the same temperature scaling and the same float32 response slice as
``get_responses``.  Prompt/padding logits are not materialised because their
gradient is exactly zero: the loss never reads them.  No top-k truncation, no
context shortening, no response-cap change.

Scope: static microbatches of one row (the SRLVR configuration validates
``micro_batch_size >= 1``; this path requires 1 and the launchers set 1).  Callers
outside that scope keep the original whole-sequence path.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from methods import tp_utils

ENABLE_ENV = "OPSD_SRLVR_SELECTIVE_LOGITS"
CHUNK_ENV = "OPSD_SRLVR_SELECTIVE_CHUNK"
VOCAB_CHUNK_ENV = "OPSD_SRLVR_SELECTIVE_VOCAB_CHUNK"
DEFAULT_CHUNK = 128
DEFAULT_VOCAB_CHUNK = 8192
_DIAG = {"projection": False, "chunk": 0}


def enabled(args) -> bool:
    """Single switch for both the hidden-only forward and row padding.

    It is intentionally an environment variable: ``args`` is rebuilt from the
    argv in several places (actor, ref, probes), while the process environment is
    the one fact every one of them sees.
    """
    if os.environ.get(ENABLE_ENV, "0") not in ("1", "true", "True"):
        return False
    return int(getattr(args, "micro_batch_size", 1)) == 1


def chunk_size() -> int:
    return max(1, int(os.environ.get(CHUNK_ENV, str(DEFAULT_CHUNK))))


def vocab_chunk_size() -> int:
    return max(1, int(os.environ.get(VOCAB_CHUNK_ENV, str(DEFAULT_VOCAB_CHUNK))))


def memory_text() -> str:
    """Allocator state on NPU; empty string on a CPU-only process."""
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            return ""
        free, _ = torch.npu.mem_get_info()
        return (
            f"alloc={torch.npu.memory_allocated() / 2**30:.2f}GiB "
            f"reserved={torch.npu.memory_reserved() / 2**30:.2f}GiB "
            f"free={free / 2**30:.2f}GiB"
        )
    except Exception:
        return ""


def memory_report(logger, tag: str) -> None:
    text = memory_text()
    if text:
        logger.info("[srlvr-mem] %s %s", tag, text)


def peak_memory_text() -> str:
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            return ""
        return (
            f"peak_alloc={torch.npu.max_memory_allocated() / 2**30:.2f}GiB "
            f"peak_reserved={torch.npu.max_memory_reserved() / 2**30:.2f}GiB"
        )
    except Exception:
        return ""


def reset_peak_memory() -> None:
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            torch.npu.reset_peak_memory_stats()
    except Exception:
        pass


def empty_cache() -> None:
    """Release cached segments at phase boundaries; never touches live tensors."""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def pad_multiple(args) -> int:
    return int(getattr(args, "data_pad_size_multiplier", 128)) * int(
        getattr(args, "tensor_model_parallel_size", 1)
    )


def padded_lengths(args, total_lengths) -> list[int] | None:
    """Per-row bshd alignment for micro_batch_size == 1, else None.

    Returning None keeps the historical step-wide maximum on every other path.
    """
    row_padding = (
        os.environ.get("OPSD_SRLVR", "0") == "1"
        and os.environ.get("OPSD_SRLVR_ROW_PADDING", "1") == "1"
        and int(getattr(args, "micro_batch_size", 1)) == 1
        and getattr(args, "qkv_format", "bshd") == "bshd"
        and int(getattr(args, "context_parallel_size", 1)) == 1
        and int(getattr(args, "pipeline_model_parallel_size", 1)) == 1
    )
    if not (enabled(args) or row_padding):
        return None
    multiple = pad_multiple(args)
    return [((int(t) + multiple - 1) // multiple) * multiple for t in total_lengths]


def _unwrap_gpt(model):
    module = model
    while hasattr(module, "module"):
        module = module.module
    return module


class HeadRef:
    """The exact output projection the stock forward would have used.

    With TP > 1 ``weight`` is the rank's contiguous vocabulary shard.  The
    reduction helpers in ``tp_utils`` turn local statistics into the exact
    global full-vocabulary distribution without all-gathering ``[T, V]``.
    """

    __slots__ = ("weight", "bias", "tp_size", "tp_rank", "vocab_start", "tp_group", "vocab_size")

    def __init__(self, weight, bias, tp_size=1, tp_rank=0, vocab_start=0,
                 tp_group=None, vocab_size=None):
        self.weight = weight
        self.bias = bias
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.vocab_start = int(vocab_start)
        self.tp_group = tp_group
        self.vocab_size = int(vocab_size if vocab_size is not None else weight.size(0) * self.tp_size)


def hidden_forward(model, forward_kwargs):
    """Run the decoder without its output layer; return hidden states and head."""
    gpt = _unwrap_gpt(model)
    # Bridge VL models compose a GPT decoder with a trainable visual tower.
    # Keep the outer visual forward intact; defer only the decoder's lm_head.
    gpt = getattr(gpt, "language_model", gpt)
    if not hasattr(gpt, "post_process"):
        raise RuntimeError("selective SRLVR needs a GPTModel chunk with post_process")
    previous = gpt.post_process
    gpt.post_process = False
    try:
        hidden = model(**forward_kwargs)
    finally:
        gpt.post_process = previous
    if not torch.is_tensor(hidden) or hidden.dim() != 3:
        raise RuntimeError(f"hidden-only forward returned {type(hidden)}/{getattr(hidden, 'shape', None)}")
    if gpt.share_embeddings_and_output_weights:
        weight = gpt.shared_embedding_or_output_weight()
    else:
        weight = gpt.output_layer.weight
    bias = getattr(gpt.output_layer, "bias", None)
    if weight is None:
        raise RuntimeError("selective SRLVR could not resolve the output-layer weight")
    tp_size, tp_rank, tp_group = tp_utils.tp_info()
    vstart = tp_utils.vocab_start(weight.size(0), tp_rank, tp_size)
    return hidden, HeadRef(weight, bias, tp_size, tp_rank, vstart, tp_group)


def _response_selection(batch, hidden):
    """Positions to project, their targets, and their response-index mapping.

    ``full_loss_masks`` is built by ``get_batch`` as
    ``F.pad(loss_mask, (prompt_length - 1, 1))``, so a set entry at position p
    means response token ``p - (prompt_length - 1)`` is trained and its target is
    ``tokens[p + 1]`` -- exactly the slice ``get_responses`` would have read.
    """
    masks = batch["full_loss_masks"]
    tokens = batch["tokens"]
    if masks.dim() != 2 or tokens.dim() != 2:
        raise RuntimeError("selective SRLVR expects bshd tensors [B, S]")
    if masks.size(0) != 1:
        raise RuntimeError(
            "selective SRLVR requires micro_batch_size=1 (one row per microbatch); "
            f"got {masks.size(0)} rows"
        )
    if hidden.dim() != 3 or hidden.size(1) != 1:
        raise RuntimeError(f"hidden states must be [S, 1, H]; got {tuple(hidden.shape)}")
    total = int(batch["total_lengths"][0])
    response = int(batch["response_lengths"][0])
    prompt = total - response
    if prompt < 1 or response <= 0:
        raise RuntimeError(f"invalid prompt/response split {prompt}/{response}")
    if hidden.size(0) < total:
        raise RuntimeError(f"hidden length {hidden.size(0)} < total_length {total}")
    mask = masks[0].to(dtype=torch.bool)
    positions = mask.nonzero().flatten()
    positions = positions[(positions >= prompt - 1) & (positions <= total - 2)]
    if positions.numel() and int(positions.min()) < prompt - 1:
        raise RuntimeError("loss-mask position precedes the response slice")
    row_mask = batch["loss_masks"][0]
    row_mask = torch.as_tensor(row_mask, dtype=torch.bool, device=masks.device)
    if row_mask.numel() != response:
        raise RuntimeError("row loss mask and response length disagree")
    response_index = positions - (prompt - 1)
    if response_index.numel() and int(response_index.max()) >= response:
        raise RuntimeError("loss-mask position exceeds the response slice")
    if not torch.equal(row_mask[response_index], torch.ones_like(response_index, dtype=torch.bool)):
        raise RuntimeError("full loss mask and row loss mask disagree")
    targets = tokens[0][positions + 1]
    return positions, targets, response_index, response


def _log_probs_vocab_chunked(x, targets, weight, bias, temperature, vocab_chunk, head):
    """Exact full-vocabulary log-softmax over a TP-sharded head.

    The local vocabulary is scanned in row chunks while a running global max and
    sum-of-exp are merged exactly.  The max is only a numerical shift and is
    detached; the sum is all-reduced with autograd so each rank's local shard
    contributes its gradient.  The chosen logit is gathered from the shard that
    owns the global target and summed across TP.  The hidden-state gradient is
    summed across TP (the hidden input is replicated; each shard contributes a
    partial log-probability gradient).
    """
    if weight.dim() != 2 or x.size(-1) != weight.size(1):
        raise RuntimeError(f"projection shape mismatch x={tuple(x.shape)} w={tuple(weight.shape)}")
    if x.dtype != weight.dtype:
        # The model can emit fp32 hidden states while the frozen head is bf16.
        # Cast the small [C, H] chunk explicitly; never cast the full weight.
        x = x.to(weight.dtype)
    group = head.tp_group
    if head.tp_size > 1:
        x = tp_utils.grad_sum(x, group)
    local_vocab = weight.size(0)
    rows = x.size(0)
    device = x.device
    valid_rows = max(0, min(local_vocab, head.vocab_size - head.vocab_start))
    neg_inf = float("-inf")
    running_max = torch.full((rows,), neg_inf, device=device, dtype=torch.float32)
    running_sum = torch.zeros((rows,), device=device, dtype=torch.float32)
    chosen = torch.zeros((rows,), device=device, dtype=torch.float32)
    for start in range(0, local_vocab, vocab_chunk):
        stop = min(start + vocab_chunk, local_vocab)
        logits = F.linear(x, weight[start:stop]).float()
        if bias is not None:
            logits = logits + bias[start:stop].float()
        if temperature != 1.0:
            logits = logits.div(temperature)
        # Shard bounds are host integers. Testing a device mask with bool()
        # would synchronize every vocabulary chunk, including unpadded shards.
        if stop > valid_rows:
            invalid = torch.arange(start, stop, device=device) >= valid_rows
            logits = logits.masked_fill(invalid.unsqueeze(0), neg_inf)
        chunk_max = logits.max(dim=-1).values.detach()
        new_max = tp_utils.tp_max_detached(torch.maximum(running_max, chunk_max), group)
        increment = torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1)
        # This selected-token log-probability (and its loss) is replicated on
        # every TP rank. Pass its upstream gradient through once, as for the
        # chosen logit below. A SUM backward here would multiply only the
        # log-partition gradient by TP size; grad_sum(x) already combines the
        # vocabulary shards' hidden-state gradients.
        increment = tp_utils.tp_sum(increment, group, differentiable=True)
        running_sum = running_sum * torch.exp(running_max - new_max) + increment
        running_max = new_max
        local = targets - head.vocab_start
        in_chunk = (local >= start) & (local < stop) & (targets < head.vocab_size)
        picked = logits.gather(1, (local - start).clamp(0, stop - start - 1).unsqueeze(1)).squeeze(1)
        chosen = chosen + torch.where(in_chunk, picked, torch.zeros_like(picked))
        if _DIAG["projection"] and _DIAG["chunk"] < 3:
            import logging

            logging.getLogger(__name__).info(
                "[srlvr-proj-chunk] tp=%d local_vocab=[%d,%d) x=%s/%s logits=%s/%s %s",
                head.tp_size, start, stop, tuple(x.shape), x.dtype,
                tuple(logits.shape), logits.dtype, memory_text(),
            )
            _DIAG["chunk"] += 1
    chosen = tp_utils.tp_sum(chosen, group, differentiable=True)
    return chosen - (running_max + torch.log(running_sum))


def _scatter(values, response_index, response_length):
    out = torch.zeros(response_length, dtype=values.dtype, device=values.device)
    if response_index.numel():
        out = out.index_put((response_index,), values)
    return out


def log_probs_from_hidden(args, batch, hidden, head=None, with_entropy: bool = False):
    """Differentiable response log-probs from ``[S, 1, H]`` hidden states.

    Returns ``{"log_probs": [tensor of length response_length]}``; entries at
    zero-mask response positions are exactly zero, which is what the masked loss
    uses them for.  Chunks are checkpointed so only one cross-entropy softmax
    chunk is alive at a time.
    """
    if with_entropy:
        raise RuntimeError("selective SRLVR loss path does not request entropy")
    head = head or getattr(args, "_srlvr_head", None)
    if head is None:
        raise RuntimeError("selective SRLVR loss has no captured output head")
    positions, targets, response_index, response = _response_selection(batch, hidden)
    temperature = float(getattr(args, "rollout_temperature", 1.0))
    flat = hidden.reshape(-1, hidden.size(-1))
    step = chunk_size()
    vocab_step = vocab_chunk_size()
    parts = []
    from torch.utils.checkpoint import checkpoint
    import logging

    logger = logging.getLogger(__name__)
    if not _DIAG["projection"]:
        _DIAG["projection"] = True
        logger.info(
            "[srlvr-proj] hidden=%s/%s contiguous=%s head.weight=%s/%s contiguous=%s "
            "requires_grad=%s bias=%s positions=%d pos_chunk=%d vocab_chunk=%d vocab=%d tp=%d/%d vstart=%d %s",
            tuple(hidden.shape), hidden.dtype, hidden.is_contiguous(),
            tuple(head.weight.shape), head.weight.dtype, head.weight.is_contiguous(),
            head.weight.requires_grad,
            None if head.bias is None else (tuple(head.bias.shape), head.bias.dtype),
            positions.numel(), step, vocab_step, head.vocab_size, head.tp_size, head.tp_rank, head.vocab_start, memory_text(),
        )
    if os.environ.get("OPSD_SRLVR_MEM_TRACE") == "1":
        _DIAG["calls"] = _DIAG.get("calls", 0) + 1
        if _DIAG["calls"] <= 5 or _DIAG["calls"] % 10 == 0:
            logger.info(
                "[srlvr-proj-call] n=%d positions=%d hidden=%s/%s %s",
                _DIAG["calls"], positions.numel(), tuple(hidden.shape), hidden.dtype, memory_text(),
            )

    # Pass bias as a dense checkpoint input (empty sentinel when the model has
    # none) so a trainable bias would receive its gradient; closures are not
    # differentiated by checkpoint.
    bias_arg = head.bias if head.bias is not None else torch.empty(
        0, device=head.weight.device, dtype=head.weight.dtype
    )
    for start in range(0, positions.numel(), step):
        chunk_positions = positions[start:start + step]
        chunk_targets = targets[start:start + step]

        def terms(x, tgt, w, b):
            return _log_probs_vocab_chunked(
                x, tgt, w, None if b.numel() == 0 else b, temperature, vocab_step, head
            )

        parts.append(
            checkpoint(
                terms,
                flat.index_select(0, chunk_positions),
                chunk_targets,
                head.weight,
                bias_arg,
                use_reentrant=False,
            )
        )
    values = torch.cat(parts) if parts else torch.zeros(0, device=hidden.device)
    output = _scatter(values, response_index, response)
    if not parts:
        # Padding/invalid-only rows still participate in a distributed backward.
        # Preserve zero gradients through both decoder and output parameters.
        output = output + hidden.sum() * 0 + head.weight.sum() * 0
        if head.bias is not None:
            output = output + head.bias.sum() * 0
    return {"log_probs": [output]}


def collect_log_probs(hidden, *, args, unconcat_tokens, total_lengths, response_lengths,
                      with_entropy=False, max_seq_lens=None):
    """``forward_only`` callback with the same return shape as the stock one."""
    head = getattr(args, "_srlvr_head", None)
    if head is None:
        raise RuntimeError("selective SRLVR forward has no captured output head")
    masks = args._srlvr_full_loss_masks
    batch = {
        "full_loss_masks": masks,
        "tokens": unconcat_tokens[0].unsqueeze(0) if torch.is_tensor(unconcat_tokens[0])
        else torch.tensor([unconcat_tokens[0]], dtype=torch.long, device=hidden.device),
        "loss_masks": [torch.ones(int(response_lengths[0]), dtype=torch.bool, device=hidden.device)],
        "total_lengths": [int(total_lengths[0])],
        "response_lengths": [int(response_lengths[0])],
    }
    with torch.no_grad():
        result = log_probs_from_hidden(args, batch, hidden, head=head, with_entropy=False)
    if with_entropy:
        raise RuntimeError("selective SRLVR reference path does not request entropy")
    return torch.empty((0,), device=hidden.device), result


def attach_full_mask(args, full_loss_masks):
    """``forward_only`` builds the batch inside its forward_step; hand it over."""
    args._srlvr_full_loss_masks = full_loss_masks
