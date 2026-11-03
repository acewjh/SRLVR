"""Information-isolation tests for the HPI privilege builder.

These are the tests that decide whether the paper claim is interpretable, so they
assert on exact index boundaries rather than on "looks right".
"""

import pytest

from tasks.alfworld.v1.privilege import (
    MODE_FUTURE_OUTCOME,
    MODE_NONE,
    MODE_OUTCOME_ONLY,
    PrivilegeConfig,
    assert_no_current_response_leak,
    assert_privilege_isolated,
    build_hpi_privilege,
)
from tasks.alfworld.v1.trajectory import FORBIDDEN_HISTORY_MARKERS

from conftest import make_trajectory


# ---------------------------------------------------------------------------
# 14. outcome_only privilege
# ---------------------------------------------------------------------------


def test_outcome_only_contains_success():
    trajectory = make_trajectory(success=True)
    text = build_hpi_privilege(trajectory, 0, MODE_OUTCOME_ONLY, future_gap=2)
    assert "Final episode outcome:" in text
    assert "SUCCESS" in text
    assert "FAILURE" not in text


def test_outcome_only_contains_failure():
    trajectory = make_trajectory(success=False)
    text = build_hpi_privilege(trajectory, 0, MODE_OUTCOME_ONLY, future_gap=2)
    assert "FAILURE" in text
    assert "SUCCESS" not in text


def test_outcome_only_has_no_future_trace():
    trajectory = make_trajectory(num_turns=5)
    text = build_hpi_privilege(trajectory, 0, MODE_OUTCOME_ONLY, future_gap=1)
    assert "Delayed future interaction trace" not in text
    for index in range(5):
        assert f"obs{index}" not in text
        assert f"act{index}" not in text


def test_outcome_only_is_identical_for_every_turn():
    trajectory = make_trajectory(num_turns=5)
    texts = {build_hpi_privilege(trajectory, t, MODE_OUTCOME_ONLY, 2) for t in range(5)}
    assert len(texts) == 1


# ---------------------------------------------------------------------------
# 15. future_outcome privilege + 12. future_gap indexing
# ---------------------------------------------------------------------------


def test_mode_none_renders_nothing():
    trajectory = make_trajectory()
    assert build_hpi_privilege(trajectory, 0, MODE_NONE, 2) == ""


def test_future_gap_2_starts_at_transition_t_plus_2():
    """Exact boundary: turn 0 with gap 2 -> steps are transitions 2 and 3 only."""
    trajectory = make_trajectory(num_turns=4)
    text = build_hpi_privilege(trajectory, 0, MODE_FUTURE_OUTCOME, future_gap=2)
    # transitions 0 and 1 are inside the gap and must not be rendered
    assert "obs0" not in text
    assert "obs1" not in text
    assert "(turn 0)" not in text
    assert "(turn 1)" not in text
    # transitions 2 and 3 are rendered
    assert "(turn 2)" in text
    assert "(turn 3)" in text
    # and the terminal observation (the environment's answer to the last action)
    assert "obs4" in text


def test_future_gap_1_withholds_only_the_current_transition():
    trajectory = make_trajectory(num_turns=4)
    text = build_hpi_privilege(trajectory, 0, MODE_FUTURE_OUTCOME, future_gap=1)
    assert "(turn 0)" not in text, "the current action's own result must be withheld"
    assert "(turn 1)" in text
    assert "(turn 2)" in text


def test_future_gap_0_includes_the_current_transition():
    """gap=0 is the leaky ablation; it must still be selectable and well-defined."""
    trajectory = make_trajectory(num_turns=3)
    text = build_hpi_privilege(trajectory, 1, MODE_FUTURE_OUTCOME, future_gap=0)
    assert "(turn 1)" in text


def test_future_gap_shifts_with_the_turn_index():
    """A later turn has a shorter remaining horizon, not a shifted window."""
    trajectory = make_trajectory(num_turns=6)
    text = build_hpi_privilege(trajectory, 3, MODE_FUTURE_OUTCOME, future_gap=2)
    assert "(turn 3)" not in text
    assert "(turn 4)" not in text
    assert "(turn 5)" in text


def test_future_window_is_empty_near_the_episode_end():
    trajectory = make_trajectory(num_turns=3)
    text = build_hpi_privilege(trajectory, 2, MODE_FUTURE_OUTCOME, future_gap=2)
    assert "(no further interaction was observed)" in text
    # the outcome is still present -- it is never gated by the gap
    assert "Final episode outcome:" in text


def test_gap_zero_index_is_not_off_by_one():
    """Turn t with gap 0 must render transition t, never t-1 or t+1."""
    trajectory = make_trajectory(num_turns=6)
    for turn in range(6):
        text = build_hpi_privilege(trajectory, turn, MODE_FUTURE_OUTCOME, future_gap=0)
        for other in range(6):
            marker = f"(turn {other})"
            if other >= turn:
                assert marker in text, f"turn {turn} gap 0 should include {other}"
            else:
                assert marker not in text, f"turn {turn} gap 0 should exclude {other}"


def test_max_future_steps_caps_rendering_but_keeps_the_outcome():
    trajectory = make_trajectory(num_turns=10)
    text = build_hpi_privilege(trajectory, 0, MODE_FUTURE_OUTCOME, future_gap=2, max_future_steps=2)
    assert "(turn 2)" in text and "(turn 3)" in text
    assert "(turn 4)" not in text
    assert "Final episode outcome:" in text


# ---------------------------------------------------------------------------
# 10 + 11. past and future chain-of-thought stripping
# ---------------------------------------------------------------------------


def test_future_trace_never_contains_future_reasoning():
    """Future <think> must be structurally absent, not merely unlikely."""
    trajectory = make_trajectory(num_turns=6)
    for turn in range(6):
        text = build_hpi_privilege(trajectory, turn, MODE_FUTURE_OUTCOME, future_gap=1)
        lowered = text.lower()
        for marker in FORBIDDEN_HISTORY_MARKERS:
            assert marker not in lowered, f"turn {turn} leaked {marker!r}"
        for index in range(6):
            assert f"reasoning {index}" not in text


def test_privilege_uses_only_environment_fields():
    """The builder must read actions/observations, never generated_text/think_text."""
    trajectory = make_trajectory(num_turns=5)
    text = build_hpi_privilege(trajectory, 0, MODE_FUTURE_OUTCOME, future_gap=1)
    for index in range(5):
        turn = trajectory.turns[index]
        assert turn.generated_text not in text
        if turn.think_text:
            assert turn.think_text not in text


# ---------------------------------------------------------------------------
# 13. current-action leakage assertion
# ---------------------------------------------------------------------------


def test_current_response_is_not_in_its_own_privilege():
    """For every gap >= 1 the privilege must not restate the evaluated turn."""
    trajectory = make_trajectory(num_turns=5)
    for gap in (1, 2, 3):
        for turn in range(5):
            text = build_hpi_privilege(trajectory, turn, MODE_FUTURE_OUTCOME, future_gap=gap)
            assert_no_current_response_leak(text, trajectory.turns[turn])


def test_gap_zero_does_leak_and_the_explicit_gate_says_so():
    """gap=0 renders transition t, so it restates the current action by design.

    The builder skips its self-check there (otherwise the ablation arm would be
    unusable), which is exactly why `assert_privilege_isolated` must still catch it.
    """
    trajectory = make_trajectory(num_turns=5)
    text = build_hpi_privilege(trajectory, 2, MODE_FUTURE_OUTCOME, future_gap=0)
    with pytest.raises(AssertionError, match="renders the evaluated turn itself"):
        assert_no_current_response_leak(text, trajectory.turns[2])


def test_leak_detector_rejects_the_evaluated_turn_being_rendered():
    """The evaluated turn must not appear as one of its own future steps."""
    trajectory = make_trajectory(num_turns=4)
    turn = trajectory.turns[1]
    poisoned = f"Delayed future interaction trace:\n[Future step 1] (turn 1)\nExecuted action: x"
    with pytest.raises(AssertionError, match="renders the evaluated turn itself"):
        assert_no_current_response_leak(poisoned, turn)


def test_a_repeated_later_action_is_not_a_leak():
    """Regression: an agent that repeats a command must not trip the guard.

    The first real HPI training run died here. Turn 2's action was repeated at a
    later turn, and the old string-based check called that a leak even though the
    evaluated turn was correctly absent from the window.
    """
    trajectory = make_trajectory(num_turns=6)
    turn = trajectory.turns[1]
    text = (
        "Delayed future interaction trace:\n"
        "[Future step 1] (turn 3)\nExecuted action: act1\n"  # same action, later turn
        "[Future step 2] (turn 4)\nExecuted action: other"
    )
    assert_no_current_response_leak(text, turn)  # must not raise


def test_leak_detector_rejects_reasoning_markers():
    trajectory = make_trajectory(num_turns=4)
    with pytest.raises(AssertionError, match="forbidden reasoning marker"):
        assert_privilege_isolated(trajectory, 1, "<think>future plan</think>", future_gap=2)


def test_gap_zero_is_flagged_by_the_isolation_gate():
    """A gap=0 trace judged under the production gap must be rejected.

    Either detection is a correct rejection -- the current-action check fires
    first because a gap-0 trace necessarily restates the evaluated action, and the
    window check would fire otherwise.
    """
    trajectory = make_trajectory(num_turns=5)
    text = build_hpi_privilege(trajectory, 2, MODE_FUTURE_OUTCOME, future_gap=0)
    with pytest.raises(AssertionError, match="inside the gap window|renders the evaluated turn itself"):
        assert_privilege_isolated(trajectory, 2, text, future_gap=2)


def test_default_gap_isolates_every_turn_of_a_realistic_episode():
    trajectory = make_trajectory(num_turns=8)
    for turn in range(8):
        text = build_hpi_privilege(trajectory, turn, MODE_FUTURE_OUTCOME, future_gap=2)
        assert_privilege_isolated(trajectory, turn, text, future_gap=2)


# ---------------------------------------------------------------------------
# Structural contract
# ---------------------------------------------------------------------------


def test_privilege_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown privilege mode"):
        PrivilegeConfig(mode="not_a_mode")


def test_privilege_config_rejects_negative_gap():
    with pytest.raises(ValueError, match="future_gap must be >= 0"):
        PrivilegeConfig(mode=MODE_OUTCOME_ONLY, future_gap=-1)


def test_turn_index_must_be_in_range():
    trajectory = make_trajectory(num_turns=3)
    with pytest.raises(IndexError):
        build_hpi_privilege(trajectory, 3, MODE_OUTCOME_ONLY, 2)


def test_future_outcome_differs_from_outcome_only_by_the_trace():
    """The Outcome-only vs Future-HPI arms must differ in exactly one region."""
    trajectory = make_trajectory(num_turns=5)
    outcome = build_hpi_privilege(trajectory, 1, MODE_OUTCOME_ONLY, future_gap=2)
    future = build_hpi_privilege(trajectory, 1, MODE_FUTURE_OUTCOME, future_gap=2)

    outcome_block = "Final episode outcome:\nSUCCESS"
    assert outcome_block in outcome
    assert outcome_block in future, "the future arm must inherit the same outcome block verbatim"
    # and only the future variant carries interaction text
    assert "(turn 3)" in future and "(turn 3)" not in outcome
    assert "Delayed future interaction trace" in future
    assert "Delayed future interaction trace" not in outcome
