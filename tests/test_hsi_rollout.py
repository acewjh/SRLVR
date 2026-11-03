"""CPU-only contract tests for the math HSI rollout (methods/hsi_rollout.py).

These pin the parts of the task book the module is responsible for, so a future edit
cannot quietly break them:

  H1  the hindsight block carries the outcome descriptor, the realised future and the
      problem, and an empty future renders as a placeholder (never as empty text)
  H2  the scored span is exactly the target chunk: the view's target tokens equal the
      completion's chunk tokens, and prompt_len indexes their first position
  H3  the future is a strict SUFFIX of the completion, so the target chunk never
      appears twice (section 5's leak rule)
  H4  one row is emitted per chunk, with tokens / response_length / rollout_log_probs
      aligned and the shared turn-id episode so trajectory weighting treats one
      prompt as one trajectory
  H5  positions with no engine candidates are carried as empty rows (the scorer masks
      them) rather than being dropped or invented
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods import hsi_rollout as hsi  # noqa: E402


class _FakeTokenizer:
    """Minimal stand-in: chat template is a fixed 3-token prefix per turn."""

    PREFIX = [900, 901, 902]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        # Deterministic, content-length dependent, but always starts with PREFIX.
        return list(self.PREFIX) + [10 + (ord(c) % 7) for c in text[:8]]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"<{int(i)}>" for i in ids)


def _fake_decode_roundtrip(ids):
    """The fake decode never round-trips to real text; build tokens so that the
    future text is distinguishable from the target text by construction."""
    return " ".join(f"<{int(i)}>" for i in ids)


# ---------------------------------------------------------------------------
# H1 -- the block
# ---------------------------------------------------------------------------


def test_block_carries_outcome_future_and_problem():
    text = hsi.render_hindsight_block(
        "What is 2+2?", "then <5> <6>", "<outcome>The verifier judged it correct.</outcome>"
    )
    assert "<outcome>" in text
    assert "then <5> <6>" in text
    assert "What is 2+2?" in text


def test_block_renders_placeholder_for_an_empty_future():
    text = hsi.render_hindsight_block("P", "   ", "<outcome>x</outcome>")
    assert hsi._NO_FUTURE in text
    # The placeholder must not be an empty string, or the block silently loses a line.
    assert "FUTURE REASONING AND ANSWER:\n\n" not in text


# ---------------------------------------------------------------------------
# H2 / H3 -- the view
# ---------------------------------------------------------------------------


def test_view_target_span_is_exactly_the_chunk():
    tok = _FakeTokenizer()
    completion = [11, 12, 13, 14, 15, 16, 17, 18]
    chunk = (2, 5)
    view_ids, plen, tlen = hsi.build_hsi_view(tok, "P", completion, chunk, "<outcome>ok</outcome>")
    assert tlen == 3
    # The target tokens sit at [plen, plen+tlen) and equal completion[start:end].
    assert view_ids[plen : plen + tlen] == completion[2:5]
    # The raw prefix completion[:start] rides along immediately before the target,
    # so prompt_len = len(user_ids) + start and the view is role-faithful.
    start = chunk[0]
    assert view_ids[plen - start : plen] == completion[:start]


def test_future_is_a_strict_suffix_so_target_never_repeats():
    tok = _FakeTokenizer()
    completion = [11, 12, 13, 14, 15, 16]
    for start, end in [(0, 2), (2, 4), (4, 6)]:
        hsi.assert_no_leak((start, end), len(completion))  # must not raise
    with pytest.raises(AssertionError):
        hsi.assert_no_leak((3, 2), 6)  # inverted
    with pytest.raises(AssertionError):
        hsi.assert_no_leak((0, 7), 6)  # past the end


def test_future_text_excludes_the_target_chunk():
    tok = _FakeTokenizer()
    completion = [11, 12, 13, 14, 15, 16]
    chunk = (0, 3)
    # The block embeds decode(completion[3:]) == the future; the target ids 11..13
    # must not appear in it.
    block = hsi.render_hindsight_block(
        "P", _fake_decode_roundtrip(completion[3:]), "<outcome>ok</outcome>"
    )
    assert "<14>" in block and "<16>" in block
    for target_id in completion[:3]:
        assert f"<{target_id}>" not in block


# ---------------------------------------------------------------------------
# H4 -- chunk rows
# ---------------------------------------------------------------------------


def _make_rows(monkeypatch, rlen=8, chunks=4):
    monkeypatch.setenv("OPSD_HSI_CHUNKS", str(chunks))
    tok = _FakeTokenizer()
    completion = list(range(100, 100 + rlen))
    logp = [-0.1 * (i + 1) for i in range(rlen)]
    cand_ids = [[100 + i, 200 + i] for i in range(rlen)]
    cand_lps = [[-0.5, -1.5] for _ in range(rlen)]
    rows = hsi._chunk_rows(
        tok,
        "P",
        completion,
        logp,
        cand_ids,
        cand_lps,
        [1, 2, 3],
        "<outcome>ok</outcome>",
        1.0,
        None,
        group_index=0,
        index_start=0,
        stride=10000,
    )
    return rows, completion, logp, tok


def test_one_row_per_nonempty_chunk_with_aligned_fields(monkeypatch):
    rows, completion, logp, tok = _make_rows(monkeypatch, rlen=8, chunks=4)
    assert len(rows) == 4
    seen = []
    for r in rows:
        assert len(r.rollout_log_probs) == r.response_length
        assert len(r.loss_mask) == r.response_length
        assert len(r.opsd_advantages) == r.response_length
        assert len(r.turn_ids) == r.response_length
        # tokens = student prompt + completion prefix through this chunk
        assert r.tokens[:3] == [1, 2, 3]
        seen.append(r.response_length)
    assert sum(seen) == 8
    # Every chunk of one prompt shares the episode id -> one trajectory.
    assert {r.turn_ids[0] // 10000 for r in rows} == {0}


def test_row_log_probs_are_the_chunk_slice(monkeypatch):
    rows, completion, logp, tok = _make_rows(monkeypatch, rlen=8, chunks=4)
    # chunk boundaries for rlen=8, k=4 -> [0,2),[2,4),[4,6),[6,8)
    assert rows[0].rollout_log_probs == logp[0:2]
    assert rows[1].rollout_log_probs == logp[2:4]
    assert rows[3].rollout_log_probs == logp[6:8]


def test_short_completion_yields_fewer_rows(monkeypatch):
    rows, *_ = _make_rows(monkeypatch, rlen=2, chunks=4)
    # split_chunks(2, 4) -> two non-empty + two empty; empties are skipped.
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# H5 -- candidate table
# ---------------------------------------------------------------------------


def test_candidate_table_preserves_empty_positions():
    tok = _FakeTokenizer()
    ids = [[5, 6], [], [7]]
    lps = [[-0.5, -1.5], [], [-0.2]]
    table = hsi._candidate_table(ids, lps, [5, 9, 7], 0, 3)
    assert len(table) == 3
    assert table[0][0] == [5, 6]
    assert table[1] == ([], [])  # no candidates -> masked, not invented
    assert table[2][0] == [7]
    # probs are a normalised distribution over each non-empty row
    for row_ids, probs in table:
        if row_ids:
            assert abs(sum(probs) - 1.0) < 1e-9


def test_candidate_table_mismatched_lengths_are_empty():
    table = hsi._candidate_table([[5, 6]], [[-0.5]], [5], 0, 1)
    assert table == [([], [])]
