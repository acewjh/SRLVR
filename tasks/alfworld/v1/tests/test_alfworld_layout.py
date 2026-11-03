"""Sequence layout, loss-mask, rollout-contract and token-alignment tests.

The layout is the correctness core: if ``loss_mask`` or the teacher-forced target
span is off by one token, training silently optimises the wrong thing.
"""

import asyncio

import pytest

from tasks.alfworld.v1.rollout import EpisodeLayout
from tasks.alfworld.v1.train_data import validate_layout
from tasks.alfworld.v1.reward import terminal_reward
from tasks.alfworld.v1.trajectory import Trajectory

from conftest import make_trajectory

from slime.utils.types import Sample


class _FakeView:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"view({self.name})"


def _layout_with_turns():
    """Build a layout directly, bypassing the engine.

    prefix ``[1, 2, 3]`` (loss_mask 0), then turn 0 generates ``[10, 11]``, then
    prompt ``[4, 5]`` (0), turn 1 generates ``[12]``.
    """
    layout = EpisodeLayout(prefix_ids=[1, 2, 3])
    layout.turns.append(([1, 2, 3], [10, 11], _FakeView(0)))
    layout.add_generation(0, [10, 11], [-0.1, -0.2])
    layout.turns.append(([4, 5], [12], _FakeView(1)))
    layout.add_prompt([4, 5])
    layout.add_generation(1, [12], [-0.3])
    return layout


# ---------------------------------------------------------------------------
# 16. loss-mask construction
# ---------------------------------------------------------------------------


def test_layout_tokens_and_loss_mask_are_index_aligned():
    layout = _layout_with_turns()
    assert layout.tokens == [1, 2, 3, 10, 11, 4, 5, 12]
    assert len(layout.tokens) == layout.prefix_len + layout.response_length
    assert len(layout.loss_mask) == layout.response_length
    assert len(layout.turn_ids) == layout.response_length
    assert len(layout.rollout_log_probs) == layout.response_length


def test_loss_mask_is_one_exactly_on_model_tokens():
    layout = _layout_with_turns()
    assert layout.loss_mask == [1, 1, 0, 0, 1]
    assert layout.turn_ids == [0, 0, -1, -1, 1]


def test_loss_mask_agrees_with_turn_ids():
    layout = _layout_with_turns()
    for mask, turn in zip(layout.loss_mask, layout.turn_ids, strict=True):
        assert (turn >= 0) == bool(mask)


def test_turn_spans_point_at_the_generated_tokens():
    layout = _layout_with_turns()
    assert layout.turn_spans == [(0, 2), (4, 5)]
    response = layout.response_tokens
    assert response[0:2] == [10, 11]
    assert response[4:5] == [12]


def test_prefix_tokens_are_outside_the_response_region():
    """The first turn's prompt is the sample's prompt; it carries no loss mask."""
    layout = _layout_with_turns()
    assert layout.prefix_len == 3
    assert layout.response_length == 5
    assert layout.response_tokens == [10, 11, 4, 5, 12]
    assert layout.tokens[: layout.prefix_len] == [1, 2, 3]
    assert 0 not in layout.response_tokens


def test_rollout_log_probs_are_zero_on_prompt_positions():
    layout = _layout_with_turns()
    assert layout.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3]


# ---------------------------------------------------------------------------
# 17. rollout object / Sample contract
# ---------------------------------------------------------------------------


def _sample_from(layout, trajectory, with_views=True):
    sample = Sample(index=0, prompt="")
    sample.tokens = list(layout.tokens)
    sample.response_length = layout.response_length
    sample.loss_mask = list(layout.loss_mask)
    sample.turn_ids = list(layout.turn_ids)
    sample.rollout_log_probs = list(layout.rollout_log_probs)
    sample.reward = trajectory.terminal_reward
    sample.metadata["turn_spans"] = [list(s) for s in layout.turn_spans]
    if with_views:
        views, plens, tlens = [], [], []
        for (start, end), (prompt_ids, y_ids, _view) in zip(
            layout.turn_spans, layout.turns, strict=True
        ):
            assert list(layout.response_tokens[start:end]) == list(y_ids)
            full_prefix = [900] + list(prompt_ids)
            views.append(list(full_prefix) + list(y_ids))
            views.append(list(prompt_ids) + list(y_ids))
            plens.extend([len(full_prefix), len(prompt_ids)])
            tlens.extend([len(y_ids), len(y_ids)])
        sample.metadata["hpi_views"] = views
        sample.metadata["hpi_prompt_lengths"] = plens
        sample.metadata["hpi_target_lengths"] = tlens
    return sample


def test_sample_contract_satisfies_convert_validation():
    layout = _layout_with_turns()
    trajectory = make_trajectory(num_turns=2)
    validate_layout(_sample_from(layout, trajectory))


def test_validation_rejects_a_loss_mask_length_mismatch():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    sample.loss_mask = sample.loss_mask[:-1]
    with pytest.raises(ValueError, match="loss_mask length"):
        validate_layout(sample)


def test_validation_rejects_loss_mask_turn_id_disagreement():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    sample.loss_mask = [1, 1, 0, 0, 1]
    sample.turn_ids = [0, 0, -1, 0, 1]  # position 3 disagrees
    with pytest.raises(ValueError, match="disagree at response position 3"):
        validate_layout(sample)


# ---------------------------------------------------------------------------
# 11. teacher/student token alignment
# ---------------------------------------------------------------------------


def test_view_targets_equal_the_student_tokens_token_for_token():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    views = sample.metadata["hpi_views"]
    plens = sample.metadata["hpi_prompt_lengths"]
    tlens = sample.metadata["hpi_target_lengths"]
    for index, (start, end) in enumerate(layout.turn_spans):
        expected = layout.response_tokens[start:end]
        for which in (0, 1):
            view = views[2 * index + which]
            assert list(view[plens[2 * index + which] :]) == expected
            assert tlens[2 * index + which] == len(expected)


def test_alignment_check_catches_a_shifted_target():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    # shift the full view's target by one token
    view = sample.metadata["hpi_views"][0]
    sample.metadata["hpi_views"][0] = view[:-1] + [999]
    with pytest.raises(AssertionError, match="target tokens differ"):
        validate_layout(sample)


def test_alignment_check_catches_a_wrong_target_length():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    sample.metadata["hpi_prompt_lengths"][0] += 1
    sample.metadata["hpi_target_lengths"][0] -= 1
    with pytest.raises((ValueError, AssertionError), match="target"):
        validate_layout(sample)


def test_views_must_come_in_pairs():
    layout = _layout_with_turns()
    sample = _sample_from(layout, make_trajectory(num_turns=2))
    sample.metadata["hpi_views"] = sample.metadata["hpi_views"][:-1]
    sample.metadata["hpi_prompt_lengths"] = sample.metadata["hpi_prompt_lengths"][:-1]
    sample.metadata["hpi_target_lengths"] = sample.metadata["hpi_target_lengths"][:-1]
    with pytest.raises(ValueError, match="pairs"):
        validate_layout(sample)


# ---------------------------------------------------------------------------
# 12 + 9. trajectory serialisation and privilege index convention
# ---------------------------------------------------------------------------


def test_trajectory_round_trips_through_json():
    trajectory = make_trajectory(num_turns=5, success=False, truncated=True, invalid_actions=2)
    restored = Trajectory.from_dict(trajectory.to_dict())
    assert restored.to_dict() == trajectory.to_dict()


def test_trajectory_jsonl_line_is_single_line():
    import json

    trajectory = make_trajectory(num_turns=3)
    line = trajectory.to_jsonl_line()
    assert "\n" not in line
    assert json.loads(line)["task_id"] == trajectory.task_id


def test_transitions_are_action_observation_pairs():
    trajectory = make_trajectory(num_turns=4)
    assert trajectory.num_transitions == 4
    assert trajectory.transition(0) == ("act0", "obs1")
    assert trajectory.transition(2) == ("act2", "obs3")


def test_a_turn_without_a_resulting_observation_has_no_transition():
    """Defensive: the rollout always records an observation, but a truncated or
    malformed record must raise rather than silently return a stale one."""
    trajectory = make_trajectory(num_turns=3)
    trajectory.turns[1].next_observation = None
    with pytest.raises(IndexError, match="no resulting observation"):
        trajectory.transition(1)


def test_iter_tail_transitions_yields_action_observation_pairs():
    trajectory = make_trajectory(num_turns=4)
    tail = list(trajectory.iter_tail_transitions(2))
    assert [t[0] for t in tail] == [2, 3]
    assert tail[0] == (2, "act2", "obs3")
    assert tail[1] == (3, "act3", "obs4")


def test_iter_tail_transitions_past_the_end_is_empty():
    trajectory = make_trajectory(num_turns=3)
    assert list(trajectory.iter_tail_transitions(10)) == []


def test_trajectory_summary_fields():
    trajectory = make_trajectory(num_turns=4, success=False, truncated=True, invalid_actions=3)
    summary = trajectory.summary()
    assert summary["success"] is False
    assert summary["num_turns"] == 4
    assert summary["invalid_action_count"] == 3


# ---------------------------------------------------------------------------
# 13. reward
# ---------------------------------------------------------------------------


def test_reward_is_terminal_success_only():
    assert terminal_reward(True) == 1.0
    assert terminal_reward(False) == 0.0


def test_truncated_episode_rewards_zero():
    trajectory = make_trajectory(num_turns=3, success=False, truncated=True)
    assert trajectory.terminal_reward == 0.0


# ---------------------------------------------------------------------------
# Regression: turn_ids arrive as torch tensors on the training side.
#
# Iterating a tensor yields 0-dim tensor scalars, and a tensor used as a dict key
# does NOT dedupe by value -- every position becomes its own key. A 5-turn episode
# was read as "1613 turns" and the HPI loss aborted on its first step. Everything
# that indexes turns must coerce to int first.
# ---------------------------------------------------------------------------


def test_compute_hindsight_weights_accepts_tensor_turn_ids():
    import torch

    from tasks.alfworld.v1.loss import compute_hindsight_weights

    # same layout as _layout_with_turns(): 2 tokens turn 0, 2 prompt, 1 token turn 1
    turn_ids = [torch.tensor(t) for t in ([0, 0, -1, -1, 1],)]
    e_per_turn = [[torch.tensor([3.0, 0.5]), torch.tensor([0.25])]]
    weights, stats = compute_hindsight_weights(e_per_turn, turn_ids, [5], alpha=0.5, scale=1.0)

    assert weights.numel() == 5
    assert stats["scored_turns"] == 2, "tensor turn ids must dedupe by VALUE, not by object"


def test_compute_hindsight_weights_matches_int_and_tensor_input():
    import torch

    from tasks.alfworld.v1.loss import compute_hindsight_weights

    e_per_turn = [[torch.tensor([3.0, 0.5]), torch.tensor([0.25])]]
    as_ints = compute_hindsight_weights(e_per_turn, [[0, 0, -1, -1, 1]], [5], 0.5, 1.0)[0]
    as_tensors = compute_hindsight_weights(
        e_per_turn, [torch.tensor([0, 0, -1, -1, 1])], [5], 0.5, 1.0
    )[0]
    assert torch.allclose(as_ints, as_tensors)
