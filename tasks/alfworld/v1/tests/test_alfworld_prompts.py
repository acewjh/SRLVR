"""Prompt construction, action parsing and past-CoT stripping tests."""

import re

import pytest

from tasks.alfworld.v1.action_parser import ParseStatus, normalize_action, parse_turn_output
from tasks.alfworld.v1.prompts import (
    ADMISSIBLE_ACTIONS_HEADER,
    TaskView,
    build_history_entries,
    build_turn_prompt_ids,
    build_turn_user_message,
    render_chat,
)
from tasks.alfworld.v1.trajectory import FORBIDDEN_HISTORY_MARKERS, assert_no_reasoning_in_history

from conftest import make_trajectory


# ---------------------------------------------------------------------------
# 8. action parser
# ---------------------------------------------------------------------------


def test_parses_a_well_formed_turn():
    text = "<think>\nI should head to the shelf.\n</think>\n<action>go to shelf 1</action>"
    parsed = parse_turn_output(text, ["go to shelf 1", "look"])
    assert parsed.status is ParseStatus.OK
    assert parsed.action == "go to shelf 1"
    assert parsed.think_text == "I should head to the shelf."
    assert parsed.is_valid


def test_parses_when_thinking_is_absent():
    parsed = parse_turn_output("<action>look</action>", ["look"])
    assert parsed.is_valid
    assert parsed.has_think is False
    assert parsed.think_text == ""


def test_reports_no_action_when_the_tag_is_missing():
    parsed = parse_turn_output("<think>hmm</think> I will look around.", ["look"])
    assert parsed.status is ParseStatus.NO_ACTION
    assert parsed.action is None
    assert not parsed.is_valid


def test_reports_empty_action():
    parsed = parse_turn_output("<action>   </action>", ["look"])
    assert parsed.status is ParseStatus.EMPTY_ACTION
    assert parsed.action is None


def test_reports_inadmissible_action_without_crashing():
    parsed = parse_turn_output("<action>go to the moon</action>", ["go to shelf 1", "look"])
    assert parsed.status is ParseStatus.NOT_ADMISSIBLE
    assert parsed.action == "go to the moon", "the raw command is kept for diagnostics"
    assert not parsed.is_valid


def test_normalises_whitespace_case_and_punctuation():
    parsed = parse_turn_output("<action>  Go To Shelf 1.  </action>", ["go to shelf 1"])
    assert parsed.is_valid
    assert parsed.action == "go to shelf 1"


def test_prefers_the_last_non_empty_action_block():
    """Models sometimes emit an aborted tag before the real one."""
    parsed = parse_turn_output("<action></action>\n<action>look</action>", ["look"])
    assert parsed.is_valid
    assert parsed.action == "look"


def test_truncated_turn_is_reported_as_truncated():
    parsed = parse_turn_output("<think>x</think>\n<action>go to sh", ["go to shelf 1"], truncated=True)
    assert parsed.status is ParseStatus.TRUNCATED
    assert not parsed.is_valid


def test_empty_admissible_list_does_not_invalidate_everything():
    """A degenerate env state must not turn every turn into an invalid action."""
    parsed = parse_turn_output("<action>look</action>", [])
    assert parsed.is_valid


def test_parser_never_raises_on_hostile_input():
    for text in ("", "<action", "</action>", "<action>" * 50, "no tags at all", "\x00\x01"):
        parse_turn_output(text, ["look"])  # must not raise


def test_normalize_action_handles_empty_input():
    assert normalize_action("") == ""
    assert normalize_action("   ") == ""


# ---------------------------------------------------------------------------
# 7. prompt construction
# ---------------------------------------------------------------------------


def _view():
    return TaskView(
        task_description="put a mug on the table",
        current_observation="You are in a kitchen. You see a mug 1.",
        admissible_actions=["go to table 1", "take mug 1 from countertop 1"],
        action_history=["look", "go to countertop 1"],
    )


def test_prompt_contains_task_observation_and_actions():
    message = build_turn_user_message(_view())
    assert "put a mug on the table" in message
    assert "You are in a kitchen." in message
    assert ADMISSIBLE_ACTIONS_HEADER in message
    assert "  - take mug 1 from countertop 1" in message


def test_prompt_instructs_exactly_one_action():
    assert "exactly one admissible action" in build_turn_user_message(_view()).lower()


def test_prompt_without_actions_uses_the_placeholder():
    view = TaskView("t", "o", [], [])
    assert "(none available)" in build_turn_user_message(view)


def test_prompt_ids_are_deterministic(tokenizer):
    assert build_turn_prompt_ids(tokenizer, _view()) == build_turn_prompt_ids(tokenizer, _view())


def test_chat_rendering_enables_thinking_explicitly(tokenizer):
    text = render_chat(tokenizer, [{"role": "user", "content": "hi"}])
    assert text.endswith("<|assistant|>\n")


def test_custom_action_history_limit_bounds_the_transcript():
    trajectory = make_trajectory(num_turns=12)
    full = build_history_entries(trajectory, 11, limit=0)
    capped = build_history_entries(trajectory, 11, limit=3)
    def action_lines(entries):
        return [line for line in entries if re.match(r"^  \d+\. ", line)]

    assert action_lines(full) == []
    assert len(action_lines(capped)) == 3
    assert "earlier actions omitted" in " ".join(capped)


# ---------------------------------------------------------------------------
# 10. past CoT stripping
# ---------------------------------------------------------------------------


def test_history_entries_exclude_past_reasoning():
    trajectory = make_trajectory(num_turns=6)
    for turn in range(6):
        entries = build_history_entries(trajectory, turn, limit=8)
        joined = "\n".join(entries).lower()
        for marker in FORBIDDEN_HISTORY_MARKERS:
            assert marker not in joined
        for index in range(6):
            assert f"reasoning {index}" not in joined


def test_history_entries_contain_past_actions_only():
    """History keeps past ACTIONS; the current turn's action is not yet taken."""
    trajectory = make_trajectory(num_turns=5)
    entries = build_history_entries(trajectory, 3, limit=8)
    joined = "\n".join(entries)
    assert "act0" in joined and "act1" in joined and "act2" in joined
    assert "act3" not in joined, "the action about to be generated must not be in its own context"


def test_no_reasoning_assertion_rejects_a_leak():
    with pytest.raises(AssertionError, match="reasoning leaked"):
        assert_no_reasoning_in_history([{"role": "user", "content": "<think>oops</think>"}])


def test_no_reasoning_assertion_rejects_template_control_tokens():
    with pytest.raises(AssertionError, match="reasoning leaked"):
        assert_no_reasoning_in_history(["<|im_start|>assistant"])


def test_no_reasoning_assertion_accepts_clean_history():
    assert_no_reasoning_in_history([{"role": "user", "content": "Observation: a mug."}])
    assert_no_reasoning_in_history([])


def test_trajectory_history_messages_is_reasoning_free():
    trajectory = make_trajectory(num_turns=4)
    for turn in range(4):
        assert_no_reasoning_in_history(trajectory.history_messages(turn))


def test_history_rejects_out_of_range_turn():
    trajectory = make_trajectory(num_turns=3)
    with pytest.raises(IndexError):
        build_history_entries(trajectory, 3)
