"""The 13 v2 correctness contracts, in the order the spec lists them.

Each of these is a statement the paper depends on. If one breaks, the HPI signal
stops meaning what it is claimed to mean, so they assert on exact structure rather
than on plausibility.
"""

import math

import pytest

from tasks.alfworld import config, loss as v2_loss
from tasks.alfworld.advantages import compute_group_advantages
from tasks.alfworld.privilege import (
    NO_FUTURE_PLACEHOLDER,
    action_token_span,
    build_full_control_ids,
    has_future,
    render_hpi_block,
)
from tasks.alfworld.prompts import TaskView

from conftest import make_trajectory


def _view(turn=0):
    trajectory = make_trajectory(num_turns=4)
    return TaskView(
        task_description=trajectory.task_description,
        current_observation=trajectory.turns[turn].observation,
        admissible_actions=list(trajectory.turns[turn].admissible_actions),
        history=[],
    )


def _pair(tokenizer, turn=0, trajectory=None, gap=config.HPI_FUTURE_GAP):
    trajectory = trajectory or make_trajectory(num_turns=4)
    generated_text = trajectory.turns[turn].generated_text
    generated_ids = tokenizer.encode(generated_text)
    return build_full_control_ids(
        tokenizer, _view(turn), trajectory, turn, generated_ids, generated_text, gap
    )


# --- 1: Full/Control teacher-force identical target token ids ---------------


def test_full_and_control_force_identical_targets(tokenizer):
    pair = _pair(tokenizer)
    assert pair.full_ids[pair.full_prompt_len :] == pair.control_ids[pair.control_prompt_len :]
    assert pair.target_len == len(pair.full_ids) - pair.full_prompt_len


# --- 2: action span identical in both views ---------------------------------


def test_action_span_is_shared_and_correct(tokenizer):
    trajectory = make_trajectory(num_turns=4)
    text = trajectory.turns[0].generated_text
    ids = tokenizer.encode(text)
    span = action_token_span(tokenizer, text, ids)
    assert span is not None
    start, end = span
    # FakeTokenizer maps one char to one token, so the span must cover exactly the
    # characters between the tags.
    content_start = text.index("<action>") + len("<action>")
    content_end = text.rindex("</action>")
    assert (start, end) == (content_start, content_end)
    # and it is the same span for both views of the pair
    pair = _pair(tokenizer, 0, trajectory)
    assert pair.action_span == (start, end)


# --- 3: outcome block identical in both views -------------------------------


def test_outcome_block_identical(tokenizer):
    trajectory = make_trajectory(num_turns=4, success=False)
    full = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    control = render_hpi_block(trajectory, 0, with_future=False, future_gap=1)
    assert "Final episode outcome:\nFAILURE" in full
    assert "Final episode outcome:\nFAILURE" in control
    assert full.count("FAILURE") == control.count("FAILURE") == 1


# --- 4: the ONLY intended difference is the future trace --------------------


def test_only_difference_is_the_future_trace(tokenizer):
    trajectory = make_trajectory(num_turns=4)
    full = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    control = render_hpi_block(trajectory, 0, with_future=False, future_gap=1)
    assert NO_FUTURE_PLACEHOLDER in control

    # Normalise away the trace body and the two blocks must be identical.
    def strip_trace(text):
        head, _, tail = text.partition("Future interaction trace:\n")
        _, _, end = tail.partition("</privileged_hindsight>")
        return head + "<TRACE>" + end

    assert strip_trace(full) == strip_trace(control)

    # ...and the trace must actually carry the later interaction.
    assert "[Future transition 1]" in full
    assert "Executed action: act1" in full
    # the withheld transition's RESULT must not appear
    assert "obs1" not in full, "o_{t+1} is the direct result of a_t and must be withheld"


# --- 5: recorded reasoning passes through; future think_text never does -----


def test_future_trace_preserves_recorded_reasoning_but_not_think_text():
    trajectory = make_trajectory(num_turns=5)
    for turn in range(5):
        block = render_hpi_block(trajectory, turn, with_future=True, future_gap=1)
        for index in range(turn + 1, 5):
            assert f"[Future transition {index}]" in block
        for index in range(5):
            assert f"r{index}" not in block, "future think_text leaked"


@pytest.mark.parametrize("field", ["action_text", "next_observation"])
@pytest.mark.parametrize(
    "marker", ["<think>future plan</think>", "<think>", "</think>", "reasoning_content"]
)
def test_recorded_reasoning_markers_pass_through_verbatim(field, marker):
    from copy import deepcopy

    from tasks.alfworld.privilege import assert_transition_provenance

    trajectory = make_trajectory(num_turns=4)
    source = trajectory.turns[2]
    setattr(source, field, f"{getattr(source, field)} {marker}")
    before = deepcopy(trajectory.to_dict())

    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert marker in block
    assert_transition_provenance(trajectory, 0, 1, block)
    assert trajectory.to_dict() == before, "rendering must not rewrite the trajectory"

    # Content tampering must still be caught, marker or not.
    tampered = block.replace(getattr(source, field), getattr(source, field) + "_CHANGED")
    with pytest.raises(AssertionError):
        assert_transition_provenance(trajectory, 0, 1, tampered)


def test_rejected_action_echo_with_reasoning_is_preserved():
    """The failure mode from the 2026-10-25 runs: the env echoes raw output back."""
    from tasks.alfworld.privilege import assert_transition_provenance

    trajectory = make_trajectory(num_turns=4)
    echoed = "<think>I should look around</think>\n<action>not a real command</action>"
    trajectory.turns[2].next_observation = (
        f"Your previous action {echoed!r} was rejected: it is not one of the "
        f"valid commands.\n\n{trajectory.turns[2].observation}"
    )

    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert "</think>" in block
    assert "<action>not a real command</action>" in block
    assert_transition_provenance(trajectory, 0, 1, block)


# --- 6: transition t never enters the Full privilege ------------------------


def test_current_transition_never_in_full_privilege():
    trajectory = make_trajectory(num_turns=5)
    full = render_hpi_block(trajectory, 2, with_future=True, future_gap=1)
    assert "[Future transition 2]" not in full, "the evaluated turn's own transition must be withheld"
    assert "[Future transition 3]" in full


# --- 7: future_gap=1 indexing ----------------------------------------------


def test_future_gap_one_indexing():
    trajectory = make_trajectory(num_turns=4)
    for turn in range(4):
        block = render_hpi_block(trajectory, turn, with_future=True, future_gap=1)
        for other in range(4):
            marker = f"[Future transition {other}]"
            marker = f"[Future transition {other}]"
            if other >= turn + 1:
                assert marker in block, f"turn {turn} gap1 should include {other}"
            else:
                assert marker not in block, f"turn {turn} gap1 should exclude {other}"


def test_last_turn_has_no_future():
    trajectory = make_trajectory(num_turns=3)
    assert has_future(trajectory, 0, 1)
    assert not has_future(trajectory, 2, 1)


# --- 8/9: the retired HPI scorer's per-turn log-ratio and weight maths were
# removed with the arm (2026-10-15); see _retired/methods/scorer.py.


def test_trajectory_mean_reduce_matches_the_explicit_reference():
    import torch

    # Two trajectories: one of 3 tokens, one of 2.
    per_token = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0])
    got = v2_loss.trajectory_mean_reduce(per_token, [3, 2], [0, 1])
    explicit = ((1.0 + 2.0 + 3.0) / 3 + (10.0 + 20.0) / 2) / 2
    assert float(got) == pytest.approx(explicit)


def test_long_episode_does_not_outweigh_a_short_one():
    """The whole reason for the reduction: equal per-trajectory weight."""
    import torch

    # trajectory 0 has 4 tokens of loss 1.0; trajectory 1 has 1 token of loss 0.0
    per_token = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    got = float(v2_loss.trajectory_mean_reduce(per_token, [4, 1], [0, 1]))
    assert got == pytest.approx(0.5)  # not 0.8, which a flat token mean would give


def _t(*values):
    import torch

    return torch.tensor(values)


# --- 10: (retired HPI weight-normalisation tests removed 2026-10-15) --------


# --- 11/12: no teacher forward for zero-advantage groups or future-less turns


def test_futureless_turn_gets_no_views(tokenizer):
    """The last turn has no future to condition on, so it is not scored."""
    advantages = compute_group_advantages([1.0, 1.0, 0.0, 0.0])
    rows = _rows_for(tokenizer, advantages)
    # 3-turn synthetic episodes: turns 0,1 have a future, turn 2 does not
    last_turn_rows = [r for r in rows if r.turn_ids[0] % config.TURN_ID_STRIDE == 2]
    assert last_turn_rows
    assert all("hsi_teacher_view" not in r.metadata for r in last_turn_rows)


# --- 13: recompute-loss-function cannot add teacher forwards ----------------


def test_loss_module_never_touches_the_teacher():
    assert v2_loss.teacher_forward_calls() == 0
    import inspect

    source = inspect.getsource(v2_loss)
    assert "_get_teacher" not in source
    assert "score_view_logprobs" not in source
    assert "from math_opsd" not in source


def _rows_for(tokenizer, advantages):
    """Build v2 rows for a synthetic 2-episode group without touching an env."""
    from tasks.alfworld.rollout import build_group_samples

    class _Episode:
        _next_id = 0

        def __init__(self, success):
            self.trajectory = make_trajectory(num_turns=3, success=success)
            base = _Episode._next_id
            _Episode._next_id += 1
            self.turns = [_FakeTurn(i, base) for i in range(3)]
            self.reward = 1.0 if success else 0.0

    class _FakeTurn:
        def __init__(self, index, episode_id):
            text = f"<think>r{index}</think>\n<action>act{index}</action>"
            self.episode_id = episode_id
            self.group_id = 0
            self.task_id = "t"
            self.turn_idx = index
            self.prompt_token_ids = [1, 2, 3]
            self.generated_token_ids = tokenizer.encode(text)
            self.generated_text = text
            self.view = _view(index)
            self.trajectory_success = False
            # Candidate-Centered HSI: the sampler's own log-probs and top-k table.
            self.generated_log_probs = [-0.5] * len(self.generated_token_ids)
            self.candidate_ids = [
                [int(token), (int(token) + 1) % 1000]
                for token in self.generated_token_ids
            ]
            self.candidate_log_probs = [[-0.2, -1.9]] * len(self.generated_token_ids)

    class _Game:
        game_id = "t"
        task_type = "pick_and_place_simple"
        split = "train"

    # A same-outcome sibling per trajectory is REQUIRED: a shuffled control may
    # never pair an episode with an opposite terminal outcome (the outcome block is
    # shared scaffold, and a foreign outcome would leak into the trace's content).
    episodes = [_Episode(True), _Episode(True), _Episode(False), _Episode(False)]
    pool = [e.trajectory for e in episodes]
    return build_group_samples(
        tokenizer, _Game(), episodes, advantages, 0, with_hpi=True, future_gap=1, sibling_pool=pool
    )


# ---------------------------------------------------------------------------
# Sentinel provenance test (spec section 2)
#
# The one off-by-one that matters: for turn t, transition i = (action_i,
# next_observation_i). With gap=1 the trace may start at i = t+1, so the FIRST
# thing it prints is (ACTION_1, OBS_2). OBS_1 is the direct result of ACTION_0 and
# must never appear -- not even though it is also turn1.observation_before.
# ---------------------------------------------------------------------------


def _sentinel_trajectory():
    from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord

    def turn(i):
        return TurnRecord(
            turn_idx=i,
            observation=f"OBS_{i}",
            admissible_actions=[f"ACTION_{i}"],
            student_prompt="",
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_text=f"<think>think{i}</think>\n<action>ACTION_{i}</action>",
            think_text=f"think{i}",
            action_text=f"ACTION_{i}",
            action_valid=True,
            parse_status="ok",
            next_observation=f"OBS_{i + 1}",
            env_reward=0.0,
            done=(i == 2),
        )

    return Trajectory(
        task_id="sentinel",
        task_type="pick_and_place_simple",
        split="train",
        task_description="sentinel task",
        seed=0,
        success=True,
        terminal_reward=1.0,
        num_turns=3,
        truncated=False,
        invalid_action_count=0,
        turns=[turn(0), turn(1), turn(2)],
    )


def test_sentinel_provenance_turn0_gap1():
    """Exact sentinel from the spec: t=0, gap=1."""
    from tasks.alfworld.privilege import assert_transition_provenance

    trajectory = _sentinel_trajectory()
    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)

    # FORBIDDEN: the evaluated action and its direct result
    assert "ACTION_0" not in block, "the current action leaked into the privilege"
    assert "OBS_1" not in block, "o_{t+1} (direct result of a_t) leaked into the privilege"

    # ALLOWED: transitions >= t+1, each as (action, resulting observation)
    assert "ACTION_1" in block
    assert "OBS_2" in block
    assert "ACTION_2" in block
    assert "OBS_3" in block

    # the structural gate agrees
    assert_transition_provenance(trajectory, 0, 1, block)


@pytest.mark.parametrize("field", ["action_text", "next_observation"])
@pytest.mark.parametrize("marker", ["<|im_end|>", "<|im_start|>"])
def test_provenance_matches_renderer_template_cleanup(field, marker):
    from copy import deepcopy
    from tasks.alfworld.privilege import assert_transition_provenance

    trajectory = _sentinel_trajectory()
    turn = trajectory.turns[1]
    original = getattr(turn, field)
    setattr(turn, field, original + marker)
    before = deepcopy(trajectory.to_dict())

    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert marker not in block
    assert original in block
    assert_transition_provenance(trajectory, 0, 1, block)
    assert trajectory.to_dict() == before

    # Removing template noise must not hide a changed action or observation.
    tampered = block.replace(original, original + "_CHANGED")
    with pytest.raises(AssertionError):
        assert_transition_provenance(trajectory, 0, 1, tampered)


def test_sentinel_holds_for_every_turn_of_a_long_episode():
    """Same check swept over turns, so an off-by-one cannot hide at one index."""
    trajectory = _sentinel_trajectory()
    for turn in range(len(trajectory.turns)):
        block = render_hpi_block(trajectory, turn, with_future=True, future_gap=1)
        for other in range(len(trajectory.turns)):
            if other < turn + 1:
                assert f"ACTION_{other}" not in block
                assert f"OBS_{other + 1}" not in block, f"o_{other + 1} leaked at turn {turn}"
            else:
                assert f"ACTION_{other}" in block
                assert f"OBS_{other + 1}" in block


def test_last_turn_full_equals_control():
    """With no allowed transition the two views must be literally identical."""
    trajectory = _sentinel_trajectory()
    full = render_hpi_block(trajectory, 2, with_future=True, future_gap=1)
    control = render_hpi_block(trajectory, 2, with_future=False, future_gap=1)
    assert full == control


def test_observation_before_next_action_is_not_reintroduced():
    """The specific trap: OBS_1 is also turn1.observation_before, and must stay out."""
    trajectory = _sentinel_trajectory()
    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert trajectory.turns[1].observation == "OBS_1"
    assert "OBS_1" not in block and "OBS_0" not in block


def test_repeated_observations_do_not_trip_the_gate():
    """Regression from real data: ALFWorld repeats text ("Nothing happens.").

    A substring check on the withheld observation fired on correct output, because
    a LATER transition legitimately printed the same string. The gate is now
    structural, so identical text in a later step is fine.
    """
    from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord
    from tasks.alfworld.privilege import assert_transition_provenance

    def turn(i, obs, nxt):
        return TurnRecord(
            turn_idx=i, observation=obs, admissible_actions=["look"], student_prompt="",
            prompt_token_ids=[1], generated_token_ids=[2],
            generated_text=f"<think>t</think><action>look</action>", think_text="t",
            action_text="look", action_valid=True, parse_status="ok",
            next_observation=nxt, env_reward=0.0, done=(i == 3),
        )

    trajectory = Trajectory(
        task_id="rep", task_type="look_at_obj_in_light", split="train",
        task_description="t", seed=0, success=False, terminal_reward=0.0,
        num_turns=4, truncated=False, invalid_action_count=0,
        # every observation is the same string, as ALFWorld does after a no-op
        turns=[turn(0, "obs", "Nothing happens."), turn(1, "Nothing happens.", "Nothing happens."),
               turn(2, "Nothing happens.", "Nothing happens."), turn(3, "Nothing happens.", "You win.")],
    )
    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert "Nothing happens." in block          # it IS present, from a later transition
    assert_transition_provenance(trajectory, 0, 1, block)  # and must not raise


def test_multiline_observation_is_compared_whole():
    """Regression: the inadmissible-action notice is multi-line.

    A line-by-line parse kept only its first line, so the gate compared a truncated
    observation against the full one and fired on correct output.
    """
    from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord
    from tasks.alfworld.privilege import assert_transition_provenance

    notice = "Your previous action 'x' was rejected.\\n\\nYou are in a room.\\nYou see a mug."

    def turn(i, nxt, done=False):
        return TurnRecord(
            turn_idx=i, observation="o", admissible_actions=["look"], student_prompt="",
            prompt_token_ids=[1], generated_token_ids=[2],
            generated_text="<think>t</think><action>look</action>", think_text="t",
            action_text="look", action_valid=True, parse_status="ok",
            next_observation=nxt, env_reward=0.0, done=done,
        )

    trajectory = Trajectory(
        task_id="ml", task_type="look_at_obj_in_light", split="train", task_description="t",
        seed=0, success=False, terminal_reward=0.0, num_turns=3, truncated=False,
        invalid_action_count=0,
        turns=[turn(0, notice), turn(1, notice), turn(2, "final", done=True)],
    )
    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert "You are in a room." in block
    assert_transition_provenance(trajectory, 0, 1, block)


# ---------------------------------------------------------------------------
# Matched shuffled sibling (frozen method, spec section 2)
# ---------------------------------------------------------------------------


def _traj(task_id, task_type, success, n):
    from tasks.alfworld.v1.trajectory import Trajectory
    from conftest import make_trajectory

    t = make_trajectory(num_turns=n, success=success, task_type=task_type)
    t.task_id = task_id
    return t


def test_sibling_match_prefers_same_category_and_outcome():
    from tasks.alfworld.privilege import matched_sibling

    me = _traj("me", "look_at_obj_in_light", True, 6)
    same_cat_other_outcome = _traj("a", "look_at_obj_in_light", False, 6)
    other_cat_same_outcome = _traj("b", "pick_two_obj_and_place", True, 6)
    same_cat_same_outcome = _traj("c", "look_at_obj_in_light", True, 7)
    pool = [me, same_cat_other_outcome, other_cat_same_outcome, same_cat_same_outcome]
    assert matched_sibling(me, pool, remaining=6).task_id == "c"


def test_sibling_match_relaxes_to_same_outcome_when_category_absent():
    from tasks.alfworld.privilege import matched_sibling

    me = _traj("me", "look_at_obj_in_light", False, 6)
    pool = [me, _traj("x", "pick_two_obj_and_place", False, 8)]
    assert matched_sibling(me, pool, remaining=6).task_id == "x"


def test_sibling_match_is_deterministic_and_can_be_absent():
    from tasks.alfworld.privilege import matched_sibling

    me = _traj("me", "look_at_obj_in_light", True, 6)
    pool = [me, _traj("a", "look_at_obj_in_light", True, 7), _traj("b", "look_at_obj_in_light", True, 5)]
    first = matched_sibling(me, pool, remaining=6)
    for _ in range(5):
        assert matched_sibling(me, pool, remaining=6).task_id == first.task_id
    # no sibling at all -> None (caller leaves the weight at 1, must not crash)
    assert matched_sibling(me, [me], remaining=6) is None


# ---------------------------------------------------------------------------
# Single-rollout HSI (spec section 10): the outcome anchor alone supervises.
# (The scorer's credit maths tests were removed with the arm, 2026-10-15.)
# ---------------------------------------------------------------------------


def test_g1_group_still_carries_signal(tokenizer):
    """Case C: G=1 -- the outcome anchor alone must be non-zero for both outcomes."""
    from tasks.alfworld.rollout import OutcomeAnchor

    for reward, expected in ((1.0, 1.0), (0.0, -1.0)):
        anchor = OutcomeAnchor([2.0 * reward - 1.0], [reward])
        assert anchor.advantages == [expected]
        assert anchor.mixed is True, "must not re-enable the removed mixed gate"
