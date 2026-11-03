"""Repeat-stop detector -- the ``n = 8`` streaming n-gram loop sensor.

Method (frozen; see the repeat-stop spec section 4)
---------------------------------------------------
Every assistant THINK span is treated independently. Inside one span we stream
over the sampled tokens and maintain

    previous-7-token prefix  ->  deduplicated set of historical successor tokens

At position ``j`` (0-based, inside the span):

1. the query may only use 8-grams that completed **strictly before** ``j``;
2. if fewer than 7 tokens precede ``j``, the successor set is empty;
3. look up the current last-7 prefix, ``span_ids[j-7:j]``;
4. if the sampled token ``span_ids[j]`` is in that set, ``repeat[j] = True``;
5. **only after the query** is the 8-gram ending at ``j`` inserted.

State is per call, so a span never sees another span's n-grams -- the spec's
"never cross prompt / history / observation / action / answer / turn" holds by
construction, because the caller hands this module exactly one think span.

``</think>`` and EOS are structural, not repetition, so they are never flagged.

Why this returns the successor SET and not just a flag
------------------------------------------------------
The conditional stop loss needs the candidate distribution at a gated position:
``[think_stop_id] + S_j``. Returning the set here keeps the detector the single
source of truth for it, so the loss and the detector cannot drift apart.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_IGNORE_IDS",
    "N_GRAM",
    "THINK_CLOSE_ID",
    "THINK_OPEN_ID",
    "detect_repeat_positions",
    "find_think_content_span",
    "repeat_signals_for_sample",
]

#: Fixed by the spec; changing it changes the method, not a tuning knob.
N_GRAM = 8

#: Qwen3 tag tokens, verified by encoding: ``<think>`` and ``</think>`` are each a
#: single token, which is what makes a "stop inside think" target expressible as
#: one logit. ``</think>`` is the stop target of the repeat-stop loss.
THINK_OPEN_ID = 151667
THINK_CLOSE_ID = 151668

#: Structural tokens that must never be counted as a repetition even if the
#: prefix lookup happens to contain them.
DEFAULT_IGNORE_IDS = frozenset({THINK_CLOSE_ID, 151645, 151643})  # </think>, <|im_end|>, <|endoftext|>


def find_think_content_span(
    response_ids,
    *,
    open_id: int = THINK_OPEN_ID,
    close_id: int = THINK_CLOSE_ID,
) -> tuple[int, int] | None:
    """``[start, end)`` of the FIRST think span's CONTENT inside ``response_ids``.

    ``start`` is the token after ``<think>``; ``end`` is the index of the closing
    ``</think>`` when present, else the end of the response (a turn truncated
    mid-think still has repetition worth stopping). The tags themselves are
    outside the span. Returns ``None`` when the response has no think content.
    """
    ids = [int(t) for t in response_ids]
    if open_id not in ids:
        return None
    start = ids.index(open_id) + 1
    if start >= len(ids):
        return None
    if close_id in ids[start:]:
        end = start + ids[start:].index(close_id)
    else:
        end = len(ids)
    if end <= start:
        return None
    return start, end


def detect_repeat_positions(
    span_ids,
    *,
    n: int = N_GRAM,
    ignore_ids=frozenset(),
) -> tuple[list[bool], list[frozenset[int]]]:
    """Streaming repeat detector over ONE think span.

    Returns ``(repeat_flags, successor_sets)``, both of length ``len(span_ids)``.
    ``successor_sets[j]`` is the deduplicated set of tokens that historically
    followed the 7-token prefix ending at ``j`` -- i.e. the state of the table at
    query time, BEFORE the 8-gram ending at ``j`` is inserted. It is empty for
    ``j < n - 1``.
    """
    ids = [int(t) for t in span_ids]
    length = len(ids)
    k = n - 1
    flags = [False] * length
    successor_sets: list[frozenset[int]] = [frozenset()] * length
    table: dict[tuple[int, ...], set[int]] = {}

    for j in range(length):
        prefix_start = j - k
        key: tuple[int, ...] | None = None
        if prefix_start >= 0:
            key = tuple(ids[prefix_start:j])
            historical = table.get(key)
        else:
            historical = None

        if historical:
            successor_sets[j] = frozenset(historical)
            token = ids[j]
            # `</think>` / EOS close the span by definition -- not a repetition.
            if token in historical and token not in ignore_ids:
                flags[j] = True

        # Insert AFTER the query, and only once a full 8-gram exists. This is what
        # makes a position unable to predict itself (rule 1 / rule 5).
        if key is not None:
            table.setdefault(key, set()).add(ids[j])

    return flags, successor_sets


def repeat_signals_for_sample(
    response_ids,
    *,
    n: int = N_GRAM,
    ignore_ids=DEFAULT_IGNORE_IDS,
    open_id: int = THINK_OPEN_ID,
    close_id: int = THINK_CLOSE_ID,
) -> tuple[list[bool], list[frozenset[int]], list[bool]]:
    """Per-RESPONSE-position signals, aligned to the whole generated turn.

    Returns ``(think_content_mask, repeat_flags, successor_sets)``, each of length
    ``len(response_ids)``. Positions outside the think content get
    ``think_content_mask = False`` and empty successor sets, so a caller can gate
    on the mask without any further bookkeeping.
    """
    ids = [int(t) for t in response_ids]
    length = len(ids)
    think_mask = [False] * length
    repeat_flags = [False] * length
    successor_sets: list[frozenset[int]] = [frozenset()] * length

    span = find_think_content_span(ids, open_id=open_id, close_id=close_id)
    if span is None:
        return think_mask, repeat_flags, successor_sets
    start, end = span

    span_flags, span_sets = detect_repeat_positions(ids[start:end], n=n, ignore_ids=ignore_ids)
    for offset, (flag, candidates) in enumerate(zip(span_flags, span_sets, strict=True)):
        position = start + offset
        think_mask[position] = True
        repeat_flags[position] = flag
        successor_sets[position] = candidates
    return think_mask, repeat_flags, successor_sets
