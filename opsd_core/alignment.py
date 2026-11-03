"""Loss-side completion-span alignment assertion. R5: the same check was open-coded in eight loss modules, each
with a different error string. One implementation now; the message carries a
``tag`` so the caller's identity survives in the traceback.
"""

from __future__ import annotations

import torch


def assert_completion_alignment(
    completion_tokens: list[torch.Tensor],
    prompt_lengths: list[int],
    response_lengths: list[int],
    unconcat_tokens: list[torch.Tensor],
    tag: str = "completion",
) -> None:
    """The teacher's completion span must equal the student's, token by token.

    Both streams evaluate the SAME completion positions; a mismatch means the
    teacher prompt collator and the student packing disagree, and the KL below
    would silently compare different tokens.
    """
    for tokens, plen, rlen, ucon in zip(
        completion_tokens, prompt_lengths, response_lengths, unconcat_tokens, strict=True
    ):
        assert list(tokens[plen:].cpu()) == list(ucon[-rlen:].cpu()), (
            f"{tag} alignment violated: teacher completion tokens differ from "
            f"student completion tokens"
        )
