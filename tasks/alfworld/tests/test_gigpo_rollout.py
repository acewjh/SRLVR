"""Rollout wiring for GiGPO: per-turn advantages must reach the emitted rows.

The loss consumes one scalar per token from `Sample.opsd_advantages`. GiGPO's
whole contribution is that the scalar CHANGES between turns of one trajectory,
so the failure this test rules out is silent: a group object that computes the
right numbers but is only read for its episode term would train plain GRPO and
every guard elsewhere would still pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.alfworld import config
from tasks.alfworld.advantages import (
    GigpoGroupAdvantages,
    compute_gigpo_advantages,
    compute_group_advantages,
)
from tasks.alfworld.episode import TurnSample
from tasks.alfworld.rollout import build_group_samples
from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord


def _trajectory(observations: list[str], *, success: bool, task_id: str = "task-0") -> Trajectory:
    rewards = [0.0] * len(observations)
    if success and observations:
        rewards[-1] = 1.0
    turns = [
        TurnRecord(
            turn_idx=index,
            observation=observation,
            admissible_actions=["look"],
            student_prompt="",
            prompt_token_ids=[100 + index],
            generated_token_ids=[200 + index],
            generated_text="<action>look</action>",
            think_text="",
            action_text="look",
            action_valid=True,
            parse_status="ok",
            next_observation=f"next-{index}",
            env_reward=rewards[index],
            done=index == len(observations) - 1,
        )
        for index, observation in enumerate(observations)
    ]
    return Trajectory(
        task_id=task_id,
        task_type="pick_and_place_simple",
        split="train",
        task_description="put a mug on the table",
        seed=0,
        success=success,
        terminal_reward=1.0 if success else 0.0,
        num_turns=len(turns),
        truncated=False,
        invalid_action_count=0,
        turns=turns,
    )


def _episode(trajectory: Trajectory, episode_id: int) -> SimpleNamespace:
    turns = [
        TurnSample(
            episode_id=episode_id,
            group_id=0,
            task_id=trajectory.task_id,
            turn_idx=index,
            prompt_token_ids=[100 + index],
            generated_token_ids=[200 + index, 201 + index],
            generated_text="<action>look</action>",
            view=None,
            generated_log_probs=[0.0, 0.0],
            harness_source="structured",
        )
        for index in range(len(trajectory.turns))
    ]
    return SimpleNamespace(
        turns=turns, trajectory=trajectory, reward=trajectory.terminal_reward
    )


def _build(episodes, advantages, tokenizer):
    return build_group_samples(
        tokenizer,
        game=None,
        episodes=episodes,
        advantages=advantages,
        group_id=7,
        with_hpi=False,
        future_gap=1,
        rollout_id=3,
    )


@pytest.mark.unit
def test_emitted_rows_carry_per_turn_advantages(tokenizer):
    # Arrange: same fixture as the advantage tests -- turn 0 shares an anchor,
    # turn 1 splits into two branches.
    trajectories = [
        _trajectory(["s0", "s1a", "s2"], success=True),
        _trajectory(["s0", "s1a", "s2"], success=False),
        _trajectory(["s0", "s1b"], success=True),
        _trajectory(["s0", "s1b"], success=False),
    ]
    episodes = [_episode(t, 700 + index) for index, t in enumerate(trajectories)]
    advantages = compute_gigpo_advantages(
        trajectories, gamma=1.0, step_w=1.0, eps=1e-12
    )

    # Act
    rows = _build(episodes, advantages, tokenizer)

    # Assert: one row per turn, and each row broadcasts ITS OWN combined advantage.
    assert len(rows) == sum(len(e.turns) for e in episodes)
    position = 0
    for episode_index, episode in enumerate(episodes):
        combined = advantages.combined_advantages[episode_index]
        for turn_index, turn in enumerate(episode.turns):
            row = rows[position]
            position += 1
            assert row.response_length == len(turn.generated_token_ids)
            assert row.opsd_advantages == [combined[turn_index]] * row.response_length
            assert row.turn_ids == [
                turn.episode_id * config.TURN_ID_STRIDE + turn.turn_idx
            ] * row.response_length

    # The step term really differs across turns (guard against a constant row
    # that would pass the per-row assertion above).
    assert len({row.opsd_advantages[0] for row in rows[:3]}) > 1


@pytest.mark.unit
def test_plain_grpo_groups_keep_a_constant_per_episode_advantage(tokenizer):
    # Arrange: a plain GroupAdvantages object (the matched control) must still
    # produce the historical constant-per-trajectory rows.
    trajectories = [
        _trajectory(["s0", "s1"], success=True),
        _trajectory(["s0", "s1"], success=False),
    ]
    episodes = [_episode(t, 800 + index) for index, t in enumerate(trajectories)]
    advantages = compute_group_advantages([1.0, 0.0])

    # Act
    rows = _build(episodes, advantages, tokenizer)

    # Assert
    first_two = rows[: len(episodes[0].turns)]
    assert {row.opsd_advantages[0] for row in first_two} == {advantages.advantages[0]}
    assert {row.opsd_advantages[0] for row in rows[len(episodes[0].turns):]} == {
        advantages.advantages[1]
    }


@pytest.mark.unit
def test_convert_preserves_per_turn_advantages(tokenizer):
    """The train-data conversion must not collapse the per-turn credit.

    `convert_samples_to_train_data` is shared with GRPO and only rewrites the
    batch-level channels; this pins that it forwards `opsd_advantages` verbatim
    and still builds a well-formed one-step batch.
    """
    from tasks.alfworld.train_data import convert_samples_to_train_data

    trajectories = [
        _trajectory(["s0", "s1a", "s2"], success=True),
        _trajectory(["s0", "s1a", "s2"], success=False),
        _trajectory(["s0", "s1b"], success=True),
        _trajectory(["s0", "s1b"], success=False),
    ]
    episodes = [_episode(t, 700 + index) for index, t in enumerate(trajectories)]
    advantages = compute_gigpo_advantages(
        trajectories, gamma=1.0, step_w=1.0, eps=1e-12
    )
    rows = _build(episodes, advantages, tokenizer)
    args = SimpleNamespace(rollout_batch_size=len(episodes), n_samples_per_prompt=1)

    # Act
    train_data = convert_samples_to_train_data(args, rows)

    # Assert
    assert train_data["dynamic_global_batch_size"] == len(rows)
    for row, emitted in zip(rows, train_data["opsd_advantages"], strict=True):
        assert emitted == row.opsd_advantages
    # Turn 0 and turn 1 of the same trajectory carry different credit.
    assert train_data["opsd_advantages"][0][0] != train_data["opsd_advantages"][1][0]


@pytest.mark.unit
def test_a_length_mismatch_fails_closed(tokenizer):
    # Arrange: a group object whose per-turn table is one entry short. Silently
    # indexing would shift the last turn's credit onto the wrong row.
    trajectories = [_trajectory(["s0", "s1"], success=True)]
    episodes = [_episode(trajectories[0], 900)]
    good = compute_gigpo_advantages(trajectories, gamma=1.0)
    broken = GigpoGroupAdvantages(
        episode_advantages=good.episode_advantages,
        step_advantages=[good.step_advantages[0][:1]],
        combined_advantages=[good.combined_advantages[0][:1]],
        mean_reward=good.mean_reward,
        std_reward=good.std_reward,
        mixed=good.mixed,
        num_episodes=good.num_episodes,
        step_w=good.step_w,
        gamma=good.gamma,
        mode=good.mode,
        diagnostics=good.diagnostics,
    )

    # Act / Assert
    with pytest.raises(ValueError, match="per-turn"):
        _build(episodes, broken, tokenizer)
