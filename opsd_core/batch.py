"""Micro-batch utilities for the (frozen) teacher forward.

Holds the shared completion-span contract used by both the rollout and the loss:

    teacher_token_ids  =  [teacher prompt (P_t)]  +  [completion (R)]
    student token_ids  =  [student prompt (P_s)]  +  [completion (R)]

so the identical completion token span is evaluated at
    student logits positions [P_s-1, P_s-1+R)   (blind context)
    teacher logits positions [P_t-1, P_t-1+R)   (privileged context)

Extracted verbatim from ``math_opsd/opsd_batch.py``.
"""

import torch


def build_right_padded_batch(
    sequences: list[torch.Tensor],
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack per-sample sequences into one right-padded (input_ids, attention_mask).

    The pad id is inert: pad positions are attention-masked and, with causal
    attention, cannot influence the earlier completion positions that the loss
    reads (the official collator uses the same right-padding convention).
    """
    max_len = max(t.size(0) for t in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, tokens in enumerate(sequences):
        input_ids[i, : tokens.size(0)] = tokens
        attention_mask[i, : tokens.size(0)] = 1
    return input_ids, attention_mask


def assert_alignment(
    teacher_tokens: list[torch.Tensor],
    teacher_prompt_lengths: list[int],
    student_tokens: list[torch.Tensor],
    response_lengths: list[int],
) -> None:
    """Assert the shared-completion contract between teacher and student streams."""
    for t_tokens, plen, s_tokens, rlen in zip(
        teacher_tokens, teacher_prompt_lengths, student_tokens, response_lengths, strict=True
    ):
        assert rlen > 0, "empty completion span"
        assert t_tokens.size(0) == plen + rlen, (
            f"teacher sequence length {t_tokens.size(0)} != P_t {plen} + R {rlen}"
        )
        assert torch.equal(t_tokens[plen:], s_tokens[-rlen:]), (
            "teacher completion tokens differ from student completion tokens"
        )
