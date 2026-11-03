"""Target-span-only lm_head log-prob scoring.

Extracted verbatim from ``math_opsd/hpi.py`` (was ``_forward_last_hidden`` and
``score_view_logprobs``), which ALFWorld v1/v2 and code already imported. The
underlying idea — never materialise a [T, V] tensor for the whole sequence, only
project the scored span — is shared by every hindsight scorer.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

DEFAULT_CHUNK = 4
DEFAULT_POS_CHUNK = 0


def configured_chunk_size() -> int:
    return int(os.environ.get("OPSD_HPI_CHUNK", str(DEFAULT_CHUNK)))


def pos_chunk() -> int:
    """Positions of the teacher-forced target span scored per forward.

    ``0`` (default) keeps the shipped single-shot behaviour: the whole target
    span is projected through the lm_head at once, materialising a
    ``[tlen, vocab]`` fp32 tensor (up to ~2.5 GiB at tlen=4096, doubled by
    ``log_softmax``). That is the peak that OOMs a 32 GiB card once the student,
    the frozen teacher and the training batch are all resident.

    A positive value caps the span per forward. ``log_softmax`` is per-position,
    so slicing the span is mathematically EXACT -- the returned log-probs are
    identical, only the transient peak changes. Gated by an env var so every
    existing caller is bit-for-bit unchanged.
    """
    raw = os.environ.get("OPSD_HPI_POS_CHUNK", "")
    return max(int(raw), 0) if raw.strip() else 0


def forward_last_hidden(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Hidden states of the base transformer (avoids materialising [B,T,V] logits)."""
    base = getattr(model, "model", None)
    if base is not None:
        return base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    raise AttributeError("model has no .model; use the full-logits fallback")


# Historical name kept: `_retired/methods/candidate_credit.py` imports this directly.
_forward_last_hidden = forward_last_hidden


def score_view_logprobs(
    model,
    views: list[tuple[list[int], int, int]],
    chunk_size: int | None = None,
) -> list[torch.Tensor]:
    """Sampled-token log-probabilities of each view's target span.

    `views` is a list of (input_ids, prompt_length, target_length). Returns one
    fp32 CPU tensor of length target_length per view. Views are length-sorted
    and grouped so padding waste is bounded; only the [target, V] slice is ever
    materialised (the lm_head is applied to the target span alone), so peak
    memory is independent of the full-vocab logits of the whole sequence.
    """
    if chunk_size is None:
        chunk_size = configured_chunk_size()
    device = next(model.parameters()).device
    order = sorted(
        (i for i, (ids, _p, t) in enumerate(views) if t > 0),
        key=lambda i: len(views[i][0]),
    )
    results: list[torch.Tensor | None] = [None] * len(views)
    for v in [i for i, (_ids, _p, t) in enumerate(views) if t == 0]:
        results[v] = torch.zeros(0, dtype=torch.float32)

    for start in range(0, len(order), chunk):
        group = order[start : start + chunk]
        max_len = max(len(views[i][0]) for i in group)
        b = len(group)
        input_ids = torch.zeros((b, max_len), dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        for r, i in enumerate(group):
            ids = views[i][0]
            input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[r, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        try:
            hidden = forward_last_hidden(model, input_ids, attention_mask)
            use_hidden = hidden is not None
        except AttributeError:
            use_hidden = False

        pc = pos_chunk()

        def _span_logp(row_slice_of, ids, plen, tlen):
            """Sampled-token log-probs of one view's target span.

            ``row_slice_of(s, e)`` returns the fp32 ``[e-s, V]`` logits for target
            positions ``[s, e)``. With ``OPSD_HPI_POS_CHUNK`` unset that is the
            original single slice; with it set the span is walked in pieces so the
            transient ``[n, V]`` tensor stays bounded. ``log_softmax`` acts on the
            last dim only, so the pieces are exactly the whole.
            """
            step = pc if pc > 0 else tlen
            out = torch.empty(tlen, dtype=torch.float32)
            for s in range(0, tlen, step):
                e = min(s + step, tlen)
                sl = row_slice_of(s, e)
                targets = torch.tensor(ids[plen + s : plen + e], device=device, dtype=torch.long)
                lp = F.log_softmax(sl, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                out[s:e] = lp.cpu()
                del sl, lp, targets
            return out

        if use_hidden:
            lm_head = model.lm_head
            for r, i in enumerate(group):
                _ids, plen, tlen = views[i]
                results[i] = _span_logp(
                    lambda s, e, r=r, plen=plen: lm_head(
                        hidden[r, plen - 1 + s : plen - 1 + e, :]
                    ).float(),
                    _ids,
                    plen,
                    tlen,
                )
            del hidden
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for r, i in enumerate(group):
                _ids, plen, tlen = views[i]
                results[i] = _span_logp(
                    lambda s, e, r=r, plen=plen: logits[r, plen - 1 + s : plen - 1 + e, :].float(),
                    _ids,
                    plen,
                    tlen,
                )
            del logits
        del input_ids, attention_mask

    return [r if r is not None else torch.zeros(0, dtype=torch.float32) for r in results]
