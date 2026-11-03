"""Frozen initial-policy teacher: load, forward over the shared span, slice.

Extracted verbatim from ``math_opsd/opsd_loss.py`` (was ``_get_teacher``,
``teacher_forward_completion_logits``, ``slice_response_logits``). Those three
names are imported by the vendored framework and by every task line, so
``math_opsd/opsd_loss.py`` re-exports them unchanged. R7: the same slicing formula was re-derived in
``null_control_kl``, ``outcome_diagnostic``, ``offline_posterior_gate``,
``check_gt_privilege_gap`` and ``hsi2_jsd``. The first four were archived to
``_retired/`` (2026-10-15, methods not in the paper); ``hsi2_jsd`` remains the
live caller and should keep calling in here.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace

import torch

from .batch import build_right_padded_batch

logger = logging.getLogger(__name__)

_TEACHER_STATE: dict[str, object] = {"model": None, "path": None}


def get_teacher(args: Namespace):
    """Lazily load the frozen initial-policy teacher on the current NPU."""
    if _TEACHER_STATE["model"] is not None:
        return _TEACHER_STATE["model"]

    import torch_npu  # noqa: F401  (registers the NPU device)

    from transformers import AutoModelForCausalLM

    path = args.hf_checkpoint
    logger.info("[opsd] loading frozen initial-policy teacher from %s", path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to("npu")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    _TEACHER_STATE["model"] = model
    _TEACHER_STATE["path"] = path
    logger.info("[opsd] frozen teacher ready (vocab=%d)", model.config.vocab_size)
    return model


# Historical name kept: methods.opsd_loss._get_teacher is imported directly by
# tasks.alfworld/{scorer,candidate_center,diagnostic}.py and tasks.alfworld.v1/loss.py.
_get_teacher = get_teacher


@torch.no_grad()
def teacher_forward_completion_logits(
    args: Namespace,
    teacher_tokens: list[torch.Tensor],
    teacher_prompt_lengths: list[int],
    response_lengths: list[int],
) -> torch.Tensor:
    """Run the frozen teacher over [privileged prompt | completion] and return
    the logits that evaluate the shared completion tokens, concatenated as
    [sum(R_i), V] (fp32), aligned with the student response logits.

    The teacher batch is right-padded per chunk (same convention as the official
    data collator), so logits[P_i-1 : P_i-1+R_i] predict completion tokens.
    """
    model = get_teacher(args)
    chunk_size = int(os.environ.get("OPSD_TEACHER_CHUNK", "2"))

    device = next(model.parameters()).device
    chunks_out: list[torch.Tensor] = []
    for start in range(0, len(teacher_tokens), chunk_size):
        tok_chunk = teacher_tokens[start : start + chunk_size]
        plen_chunk = teacher_prompt_lengths[start : start + chunk_size]
        rlen_chunk = response_lengths[start : start + chunk_size]

        input_ids, attention_mask = build_right_padded_batch(tok_chunk)
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )
        logits = outputs.logits  # [B, T, V] bf16
        for i, (plen, rlen) in enumerate(zip(plen_chunk, rlen_chunk, strict=True)):
            # positions [P-1, P-1+R) predict the shared completion tokens
            chunks_out.append(logits[i, plen - 1 : plen - 1 + rlen, :].float())
    if len(chunks_out) == 1:
        # Memory no-op, NOT a numerics change: torch.cat on a single tensor
        # allocates a second full copy of it. This is the exact allocation that
        # OOMed the 4B run (594 MB = ~977 response tokens x 151936 vocab x 4 B,
        # with 82 MiB free). Returning the tensor itself is bit-identical --
        # .float() already produced a fresh, contiguous fp32 tensor.
        return chunks_out[0]
    return torch.cat(chunks_out, dim=0)


def slice_response_logits(
    args: Namespace,
    logits: torch.Tensor,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> torch.Tensor:
    """Extract response-aligned logits from the thd-packed stream as [sum(R), V].

    Same shift-by-one convention as slime's get_responses: for packed offsets
    [start, end) per sample, logits[start-1 : end-1] predict tokens[start:end).
    """
    assert logits.dim() == 3, f"{logits.shape}"

    chunks = []
    if logits.size(0) == 1 and getattr(args, "qkv_format", "thd") == "thd":
        # thd-packed stream: cumulative offsets per sample (mirror slime get_responses)
        logits = logits.squeeze(0)
        end = 0
        for _tokens, total_length, response_length in zip(
            unconcat_tokens, total_lengths, response_lengths, strict=True
        ):
            end += total_length
            start = end - response_length
            chunks.append(logits[start - 1 : end - 1, :])
    else:
        # bshd padded batch: sample i occupies rows [i*max_seq_len, i*max_seq_len + total_length)
        # (mirror slime get_responses bshd branch)
        logits = logits.reshape(-1, logits.size(-1))
        max_seq_lens = args._current_max_seq_lens  # set by opsd_loss_function per microbatch
        for i, (_tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=True)
        ):
            end = max_seq_lens * i + total_length
            start = end - response_length
            chunks.append(logits[start - 1 : end - 1, :])
    return torch.cat(chunks, dim=0)
