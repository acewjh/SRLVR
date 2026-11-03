"""Conditional stop loss for the repeat-stop method (frozen spec section 6).

At a GATED position the ordinary PPO term is **replaced** -- not added to -- by a
one-shot conditional cross-entropy that asks the policy to prefer ``</think>``
over the tokens it historically repeated there::

    S      = repeat_candidates[j]                       # historical successors
    ce_j   = logsumexp(scores[[think_stop_id] + S]) - scores[think_stop_id]
    loss_j = stop_gradient(|A_j|) * ce_j

The gradient properties the spec pins down follow directly from that form:
``d loss_j / d z_stop = -|A_j| < 0`` and, for a repeated candidate ``r``,
``d loss_j / d z_r = |A_j| * softmax_candidate(r) > 0``; any logit outside
``[think_stop_id] + S`` has no gradient path at all.

Everything outside the gate is untouched: the caller keeps the original PPO/HSI
surrogate there, so turning the feature off (or gating nothing) is bit-identical
to the baseline.

Gate
----
``policy_loss_mask & think_content_mask & repeat_mask & (R == 0) & (A_token < 0)``

Memory
------
Only the gated positions' rows are ever read, in chunks of ``ALFWORLD_V2_SCORE_CHUNK``
(256, then 128, then 64 on OOM). No full-vocab supervision tensor is built: the
per-position candidate columns are extracted from the row immediately and the row
is dropped, so peak extra memory is ``chunk * vocab`` for one chunk at a time.
"""

from __future__ import annotations

import os

import torch

from .repeat import THINK_CLOSE_ID, repeat_signals_for_sample

__all__ = [
    "GATE_ENV",
    "REPEAT_STOP_ENV",
    "SCORE_CHUNK_ENV",
    "build_gate",
    "gated_positions",
    "maybe_replace",
    "repeat_stop_replace",
    "score_chunk",
]

#: Master switch. Off == the loss is exactly the shipped baseline.
REPEAT_STOP_ENV = "ALFWORLD_V2_REPEAT_STOP"
#: Extra oracle for tests: force the gate on regardless of R/A, or off.
GATE_ENV = "ALFWORLD_V2_REPEAT_STOP_GATE"
#: Row-gather chunk for the sparse CE. Falls back 256 -> 128 -> 64 on OOM.
SCORE_CHUNK_ENV = "ALFWORLD_V2_SCORE_CHUNK"
DEFAULT_SCORE_CHUNK = 256
SCORE_CHUNK_LADDER = (256, 128, 64)


def enabled() -> bool:
    return os.environ.get(REPEAT_STOP_ENV, "0") == "1"


def score_chunk() -> int:
    raw = os.environ.get(SCORE_CHUNK_ENV, "")
    if raw.strip():
        return max(1, int(raw))
    return DEFAULT_SCORE_CHUNK


def _response_slice(
    logits: torch.Tensor,
    args,
    total_lengths,
    response_lengths,
    max_seq_lens,
    sample_index: int,
):
    """``(start, end)`` into the flattened ``[T, V]`` logits for one sample's response.

    ``flat[start:end]`` are exactly the rows whose next token is the response --
    i.e. row ``start + j`` predicts response token ``j``.

    Mirrors ``get_responses`` (bshd, no context parallel -- the only geometry this
    harness launches). Reproducing the indexing instead of calling that helper is
    what lets us gather ONLY the candidate columns: calling it would materialise a
    second full ``[T, V]`` fp32 tensor, which the spec forbids.
    """
    if getattr(args, "qkv_format", "bshd") != "bshd":
        raise RuntimeError("repeat-stop sparse scoring assumes --qkv-format bshd")
    max_seq_len = int(max_seq_lens[sample_index])
    total_length = int(total_lengths[sample_index])
    response_length = int(response_lengths[sample_index])
    end = max_seq_len * sample_index + total_length
    start = end - response_length
    return start - 1, end - 1


def build_gate(
    think_masks,
    repeat_masks,
    advantages,
    rewards,
    *,
    force: bool | None = None,
):
    """The section-6 gate, per response position, over the concatenated rows.

    ``think_masks`` / ``repeat_masks`` / ``advantages`` are already concatenated
    over samples in row order; ``rewards`` is per sample and is broadcast.
    """
    gate = []
    for think, repeat, adv, reward in zip(think_masks, repeat_masks, advantages, rewards, strict=False):
        if force is None:
            gated = bool(think) and bool(repeat) and float(reward) == 0.0 and float(adv) < 0.0
        else:
            gated = bool(think) and bool(repeat) and bool(force)
        gate.append(gated)
    return gate


def gated_positions(gate, device=None):
    """Indices (into the concatenated response) where the gate is on."""
    idx = [i for i, on in enumerate(gate) if on]
    if device is None:
        return idx
    return torch.tensor(idx, dtype=torch.long, device=device)


def _conditional_ce(
    rows: torch.Tensor, candidate_index: torch.Tensor, *, stop_id: int = THINK_CLOSE_ID
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse conditional CE and the full-vocab stopping probability.

    ``rows``            -- ``[m, V]`` temperature-scaled logits at gated positions.
    ``candidate_index`` -- ``[m, C]`` long tensor, column 0 is the stop token
                           (``</think>`` in production), the rest are the
                           historical successors; padded with -1.

    Returns ``(ce, q_stop)``, each ``[m]``. ``ce`` is differentiable;
    ``q_stop`` is a no-grad diagnostic and is the spec's ratio
    ``p_stop / (p_stop + sum_S p)`` -- NOT a full-vocab stopping probability.
    """
    valid = candidate_index >= 0
    safe = candidate_index.clamp_min(0)
    cand_scores = rows.gather(1, safe).masked_fill(~valid, float("-inf"))

    stop_scores = cand_scores[:, 0]
    # logsumexp over each position's candidate set; a single candidate makes this
    # exactly -log 1 = 0, which is the correct degenerate answer.
    lse_cand = torch.logsumexp(cand_scores, dim=1)
    ce = lse_cand - stop_scores

    with torch.no_grad():
        row_lse = torch.logsumexp(rows, dim=1)
        p_stop = torch.exp(stop_scores - row_lse)
        repeat_scores = cand_scores[:, 1:]
        repeat_scores = repeat_scores.masked_fill(~valid[:, 1:], float("-inf"))
        p_repeat = torch.exp(repeat_scores - row_lse.unsqueeze(1)).sum(dim=1)
        denom = p_stop + p_repeat
        q_stop = torch.where(denom > 0, p_stop / denom, torch.zeros_like(p_stop))
    return ce, q_stop


def _gather_rows(logits_flat: torch.Tensor, row_index: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled logit rows for ``row_index`` (into the flat ``[T, V]``).

    Scaled exactly the way ``get_responses`` scales them, so the candidate
    distribution here is the same one the PPO surrogate is computed against.
    """
    rows = logits_flat.index_select(0, row_index)
    if rows.dtype != torch.float32:
        rows = rows.float()
    if temperature != 1.0:
        rows = rows / temperature
    return rows


def maybe_replace(per_token: torch.Tensor, **kwargs):
    """The seam ``loss.alfworld_v2_loss`` calls.

    With the feature off this returns the surrogate **unchanged and by identity**,
    which is what makes "feature OFF == baseline" a property rather than a hope.
    With it on and nothing gated, ``repeat_stop_replace`` also returns the
    surrogate unchanged.
    """
    if not enabled():
        return per_token, {}
    return repeat_stop_replace(per_token, **kwargs)


def repeat_stop_replace(
    per_token: torch.Tensor,
    *,
    args,
    logits: torch.Tensor,
    batch: dict,
    response_lengths,
    total_lengths,
    advantages: torch.Tensor,
    rewards,
    stop_id: int = THINK_CLOSE_ID,
):
    """Replace the surrogate at gated positions. Returns ``(new_per_token, metrics)``.

    ``per_token`` and ``advantages`` are the concatenated response-position
    tensors in row order; ``rewards`` is per sample.
    """
    response_lengths = [int(r) for r in response_lengths]
    total_lengths = [int(t) for t in total_lengths]
    unconcat_tokens = batch["unconcat_tokens"]
    max_seq_lens = batch.get("max_seq_lens")
    if max_seq_lens is None:
        raise RuntimeError("repeat-stop requires batch['max_seq_lens']")
    if rewards is None:
        # The gate is `(R == 0) & ...`. Defaulting a missing reward to 0 would
        # silently gate EVERY failed-looking position -- a wrong method that looks
        # like a working one. Fail closed instead.
        raise ValueError("repeat-stop requires batch['rewards'] for the (R == 0) gate")
    temperature = float(getattr(args, "rollout_temperature", 1.0))

    force_env = os.environ.get(GATE_ENV, "").strip().lower()
    force = None
    if force_env in ("1", "on", "true"):
        force = True
    elif force_env in ("0", "off", "false"):
        force = False

    # ---- per-position signals, row by row -----------------------------------
    think_flags: list[bool] = []
    repeat_flags: list[bool] = []
    successor_sets: list[frozenset[int]] = []
    adv_list: list[float] = []
    reward_per_position: list[float] = []
    for sample_index, rlen in enumerate(response_lengths):
        tokens = unconcat_tokens[sample_index]
        ids = [int(t) for t in (tokens[-rlen:] if rlen > 0 else [])]
        think_mask, repeat_mask, successors = repeat_signals_for_sample(ids)
        think_flags.extend(think_mask)
        repeat_flags.extend(repeat_mask)
        successor_sets.extend(successors)
        reward_per_position.extend([float(rewards[sample_index])] * rlen)
    adv_list = [float(a) for a in advantages.detach().cpu()]

    gate = build_gate(think_flags, repeat_flags, adv_list, reward_per_position, force=force)
    positions = [i for i, on in enumerate(gate) if on]

    metrics = {
        "v2/repeat_count": float(sum(1 for f in repeat_flags if f)),
        "v2/gated_count": float(len(positions)),
        "v2/think_tokens": float(sum(1 for f in think_flags if f)),
    }
    if not positions:
        metrics["v2/gate_fraction"] = 0.0
        return per_token, metrics

    flat = logits.view(-1, logits.size(-1))

    chunk = score_chunk()
    if chunk not in SCORE_CHUNK_LADDER:
        chunk = min(SCORE_CHUNK_LADDER, key=lambda c: abs(c - chunk))

    ce_values: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    replacement: list[torch.Tensor] = []
    # Map each gated position to the absolute row that predicts it.
    row_of_position: dict[int, int] = {}
    offset = 0
    for sample_index, rlen in enumerate(response_lengths):
        if rlen <= 0:
            continue
        start, _end = _response_slice(
            logits, args, total_lengths, response_lengths, max_seq_lens, sample_index
        )  # noqa: F841 - `end` is documented by the helper's contract
        for local in range(rlen):
            row_of_position[offset + local] = start + local
        offset += rlen

    for chunk_start in range(0, len(positions), chunk):
        block = positions[chunk_start : chunk_start + chunk]
        rows_idx = torch.tensor([row_of_position[p] for p in block], dtype=torch.long, device=flat.device)
        rows = _gather_rows(flat, rows_idx, temperature)

        # Candidate matrix: [stop] + sorted(historical successors).
        cand_lists = [[stop_id] + sorted(successor_sets[p]) for p in block]
        width = max(len(c) for c in cand_lists)
        candidate_index = torch.full((len(block), width), -1, dtype=torch.long, device=flat.device)
        for r, cands in enumerate(cand_lists):
            candidate_index[r, : len(cands)] = torch.tensor(cands, dtype=torch.long, device=flat.device)

        ce, q_stop = _conditional_ce(rows, candidate_index, stop_id=stop_id)
        abs_adv = advantages.index_select(0, torch.tensor(block, dtype=torch.long, device=advantages.device)).abs()
        replacement.append(torch.stack([(abs_adv[r].detach() * ce[r]) for r in range(len(block))]))
        ce_values.append(ce.detach())
        q_values.append(q_stop)
        del rows, candidate_index, ce, q_stop

    if not replacement:
        metrics["v2/gate_fraction"] = 0.0
        return per_token, metrics

    new_per_token = per_token.clone()
    new_per_token[torch.tensor(positions, dtype=torch.long, device=per_token.device)] = torch.cat(replacement)

    ce_all = torch.cat(ce_values)
    q_all = torch.cat(q_values)
    gate_pos = torch.tensor(positions, dtype=torch.long, device=per_token.device)
    metrics.update(
        {
            "v2/gate_fraction": float(len(positions)) / max(len(gate), 1),
            "v2/ce_mean": float(ce_all.mean()) if ce_all.numel() else 0.0,
            "v2/q_stop_mean": float(q_all.mean()) if q_all.numel() else 0.0,
            "v2/gate_abs_adv_mean": float(advantages.index_select(0, gate_pos).abs().mean()),
        }
    )
    return new_per_token, metrics
