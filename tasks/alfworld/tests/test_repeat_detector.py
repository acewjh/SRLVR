"""Repeat detector contract (spec section 4): n=8, per think span, query-before-insert.

Every assertion here is one of the preflight cases the launch spec makes
mandatory. CPU-only: no tokenizer, no NPU, no ALFWorld.
"""

import sys
from pathlib import Path

import pytest

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from tasks.alfworld.repeat import (  # noqa: E402
    DEFAULT_IGNORE_IDS,
    N_GRAM,
    THINK_CLOSE_ID,
    THINK_OPEN_ID,
    detect_repeat_positions,
    find_think_content_span,
    repeat_signals_for_sample,
)


#: An 8-token block that will be repeated verbatim to build a genuine 8-gram loop.
BLOCK = [10, 11, 12, 13, 14, 15, 16, 17]
#: A second block used where the successor SET must hold more than one token.
BLOCK_B = [20, 21, 22, 23, 24, 25, 26, 27]


def test_n_gram_is_frozen_at_eight():
    # Arrange / Act / Assert -- the spec fixes n; a drift here changes the method.
    assert N_GRAM == 8


def test_first_occurrence_is_not_a_repeat():
    # Arrange
    span = BLOCK
    # Act
    flags, successors = detect_repeat_positions(span)
    # Assert
    assert not any(flags)
    assert sum(flags) == 0
    # The prefix of the would-be 8-gram must not have predicted itself.
    assert successors[N_GRAM - 1] == frozenset()


def test_second_occurrence_is_a_repeat():
    # Arrange -- the same 8-token block twice: the loop the method exists to catch.
    span = BLOCK + BLOCK
    # Act
    flags, successors = detect_repeat_positions(span)
    # Assert -- only the final token, and only because 17 followed {10..16} before.
    assert flags.index(True) == len(span) - 1
    assert sum(flags) == 1
    assert successors[-1] == frozenset({17})


def test_overlapping_blocks_are_detected_per_position():
    # Arrange -- a period-4 loop, so consecutive 8-grams OVERLAP. The first time an
    # 8-gram's 7-prefix appears its set is empty; the overlapping recurrence is
    # where the loop becomes visible.
    span = [1, 2, 3, 4] * 4
    # Act
    flags, successors = detect_repeat_positions(span)
    # Assert -- index 7 is the first completed 8-gram (nothing yet to match);
    # index 11's prefix [1,2,3,4,1,2,3] was seen at index 7, whose successor was 4.
    assert flags[7] is False
    assert successors[7] == frozenset()
    assert flags[11] is True
    assert successors[11] == frozenset({4})


def test_successor_set_is_deduplicated_across_occurrences():
    # Arrange -- prefix P = BLOCK[:7] appears three times. Its first two
    # occurrences carry DIFFERENT successors, so the table must hold a set; the
    # third occurrence carries the FIRST successor and must fire on it.
    prefix = BLOCK[:7]
    span = prefix + [17] + prefix + [18] + prefix + [17]
    # Act
    flags, successors = detect_repeat_positions(span)
    # Assert
    assert sum(flags) == 1
    assert flags[-1] is True
    assert successors[-1] == frozenset({17, 18})


def test_query_uses_only_8grams_completed_before_the_position():
    # Arrange -- exactly one 8-gram. If the detector inserted before querying, the
    # position would match its own successor and report a spurious repeat.
    span = BLOCK
    # Act
    flags, _ = detect_repeat_positions(span)
    # Assert
    assert not any(flags)


def test_prefix_shorter_than_seven_has_empty_successor_set():
    # Arrange
    span = [1, 2, 3, 4, 5, 6]
    # Act
    flags, successors = detect_repeat_positions(span)
    # Assert
    assert not any(flags)
    assert all(s == frozenset() for s in successors)


def test_think_close_and_eos_are_never_repeats():
    # Arrange -- the closing tag occupies the successor slot a real loop would use.
    span = BLOCK[:7] + [THINK_CLOSE_ID] + BLOCK[:7] + [THINK_CLOSE_ID]
    # Act
    flags, _ = detect_repeat_positions(span, ignore_ids=DEFAULT_IGNORE_IDS)
    # Assert
    assert not any(flags)
    assert THINK_CLOSE_ID in DEFAULT_IGNORE_IDS


def test_state_does_not_cross_turn_boundaries():
    # Arrange -- two independent spans, each a first occurrence internally.
    span = BLOCK + BLOCK[:1]
    # Act -- separate calls stand in for separate turns.
    first, _ = detect_repeat_positions(span)
    second, _ = detect_repeat_positions(span)
    # Assert -- identical, because no state survives a call.
    assert first == second
    assert not any(first)


def test_find_think_content_span_excludes_the_tags():
    # Arrange
    response = [THINK_OPEN_ID] + BLOCK + [THINK_CLOSE_ID] + [522, 1311, 29]
    # Act
    span = find_think_content_span(response)
    # Assert
    assert span == (1, 1 + len(BLOCK))


def test_find_think_content_span_extends_to_end_when_unclosed():
    # Arrange -- a truncated turn never emitted </think>; the content still runs on.
    response = [THINK_OPEN_ID] + BLOCK
    # Act
    span = find_think_content_span(response)
    # Assert
    assert span == (1, 1 + len(BLOCK))


def test_find_think_content_span_is_none_without_a_think_tag():
    # Arrange / Act / Assert
    assert find_think_content_span([1, 2, 3]) is None
    assert find_think_content_span([THINK_OPEN_ID]) is None


def test_sample_signals_align_to_the_full_response():
    # Arrange
    action = [522, 1311, 29]
    response = [THINK_OPEN_ID] + BLOCK + BLOCK + [THINK_CLOSE_ID] + action
    # Act
    think_mask, repeat_flags, successors = repeat_signals_for_sample(response)
    # Assert -- masks are response-aligned, and nothing outside the content fires.
    assert len(think_mask) == len(repeat_flags) == len(successors) == len(response)
    assert think_mask[0] is False and think_mask[-1] is False
    assert repeat_flags[-1] is False
    assert [i for i, m in enumerate(think_mask) if m] == list(range(1, 1 + 2 * len(BLOCK)))
    # exactly one repeat: the last content token
    assert [i for i, f in enumerate(repeat_flags) if f] == [1 + 2 * len(BLOCK) - 1]
    # the successor set travels with the flag, for the conditional stop loss
    assert successors[1 + 2 * len(BLOCK) - 1] == frozenset({BLOCK[-1]})


def test_sample_signals_are_empty_without_a_think_span():
    # Arrange / Act
    think_mask, repeat_flags, _ = repeat_signals_for_sample([1, 2, 3])
    # Assert
    assert not any(think_mask)
    assert not any(repeat_flags)
