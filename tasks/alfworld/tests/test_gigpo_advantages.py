"""GiGPO advantage contracts (CPU only).

GiGPO = GRPO's episode-level advantage PLUS a step-level advantage computed
inside an "anchor state group": turns from different trajectories that stand in
the SAME environment state. The tests pin the three things that can silently
turn this into plain GRPO or into noise:

* the episode term is byte-for-byte the existing GRPO group advantage;
* anchoring is exact and scoped to one task -- a shared observation string from
  a different task must never join the group, and a singleton anchor must be
  exactly 0 (not a mean-subtracted value that a std of 1 would let through);
* the step return is the discounted return-to-go over the episode's per-turn
  environment rewards, so a terminal success at turn 3 is worth gamma**2 of the
  same success at turn 1 when compared inside one anchor group.
"""

from __future__ import annotations

import math

import pytest

from tasks.alfworld.advantages import (
    GigpoGroupAdvantages,
    compute_gigpo_advantages,
    compute_group_advantages,
)
from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord


def _turn(
    index: int,
    observation: str,
    *,
    reward: float = 0.0,
    done: bool = False,
    anchor_observation: str = "",
) -> TurnRecord:
    return TurnRecord(
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
        env_reward=reward,
        done=done,
        anchor_observation=anchor_observation,
    )


def _trajectory(
    observations: list[str],
    *,
    success: bool,
    task_id: str = "task-0",
    rewards: list[float] | None = None,
    truncated: bool = False,
) -> Trajectory:
    """One trajectory; a successful episode pays its reward on the last turn."""
    if rewards is None:
        rewards = [0.0] * len(observations)
        if success and observations:
            rewards[-1] = 1.0
    turns = [
        _turn(
            index,
            observation,
            reward=rewards[index],
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
        truncated=truncated,
        invalid_action_count=0,
        turns=turns,
    )


#: Four trajectories in ONE task. Turn 0 is a shared anchor; turn 1 splits into
#: two branch anchors (``s1a`` reached by two, ``s1b`` by two); turn 2 is a
#: singleton anchor in each branch.
FIXTURE = [
    _trajectory(["s0", "s1a", "s2"], success=True),
    _trajectory(["s0", "s1a", "s2"], success=False),
    _trajectory(["s0", "s1b"], success=True),
    _trajectory(["s0", "s1b"], success=False),
]


def _assert_close(left: float, right: float, tol: float = 1e-9) -> None:
    assert abs(left - right) <= tol, f"{left} != {right}"


@pytest.mark.unit
def test_episode_term_is_exactly_the_existing_grpo_group_advantage():
    # Arrange
    rewards = [float(t.terminal_reward) for t in FIXTURE]
    expected = compute_group_advantages(rewards)

    # Act
    result = compute_gigpo_advantages(FIXTURE, gamma=1.0, step_w=1.0)

    # Assert
    assert result.episode_advantages == pytest.approx(expected.advantages)
    assert result.mixed is expected.mixed
    assert result.mean_reward == pytest.approx(expected.mean_reward)
    assert result.std_reward == pytest.approx(expected.std_reward)


@pytest.mark.unit
def test_anchor_groups_are_exact_state_groups_per_branch():
    # Arrange / Act
    result = compute_gigpo_advantages(FIXTURE, gamma=1.0, step_w=1.0, mode="mean_std_norm", eps=1e-12)

    # Assert — s0 is shared by all four; s1a / s1b / s2 are each shared by the
    # two trajectories that took the same action sequence.
    s0 = 0.5 / math.sqrt(1.0 / 3.0)  # (1 - .5) / std, std = sqrt(1/3) with n-1
    branch = (1.0 - 0.5) / math.sqrt(0.5)  # two members, rewards [1, 0]

    assert result.step_advantages[0] == pytest.approx([s0, branch, branch])
    assert result.step_advantages[1] == pytest.approx([-s0, -branch, -branch])
    assert result.step_advantages[2] == pytest.approx([s0, branch])
    assert result.step_advantages[3] == pytest.approx([-s0, -branch])

    # Combined = episode + w * step, per turn.
    ep0 = result.episode_advantages[0]
    assert result.combined_advantages[0] == pytest.approx(
        [ep0 + s0, ep0 + branch, ep0 + branch]
    )
    assert result.advantage_for(0, 2) == pytest.approx(ep0 + branch)


@pytest.mark.unit
def test_step_return_uses_discounted_return_to_go():
    """A success is worth gamma**2 at turn 0 and gamma at turn 1."""
    # Arrange: one success over three turns, one failure over two; both start
    # at the same anchor and share no later state.
    trajectories = [
        _trajectory(["s0", "s1", "s2"], success=True),
        _trajectory(["s0", "x1"], success=False),
    ]

    # Act
    result = compute_gigpo_advantages(trajectories, gamma=0.5, step_w=1.0, eps=1e-12)

    # Assert: the anchor group at s0 holds [0.25, 0.0]; mean 0.125,
    # sample std sqrt(0.03125) = 0.1767767.
    std = math.sqrt(0.03125)
    expected = 0.125 / std
    assert result.step_advantages[0][0] == pytest.approx(expected)
    assert result.step_advantages[1][0] == pytest.approx(-expected)
    # A singleton anchor contributes nothing regardless of its return.
    assert result.step_advantages[0][1] == 0.0
    assert result.step_advantages[0][2] == 0.0


@pytest.mark.unit
def test_anchor_groups_never_cross_task_boundaries():
    # Arrange: the two trajectories share every observation string but belong to
    # different ALFWorld tasks, so their turn-0 states are NOT the same anchor.
    trajectories = [
        _trajectory(["s0", "s1"], success=True, task_id="task-A"),
        _trajectory(["s0", "s1"], success=False, task_id="task-B"),
    ]

    # Act
    result = compute_gigpo_advantages(trajectories, gamma=1.0, eps=1e-12)

    # Assert: no step-level signal at all; only the episode term remains.
    assert [list(row) for row in result.step_advantages] == [[0.0, 0.0], [0.0, 0.0]]
    for episode, row in enumerate(result.combined_advantages):
        assert row == pytest.approx([result.episode_advantages[episode]] * 2)


@pytest.mark.unit
def test_mean_norm_mode_subtracts_the_anchor_mean_without_std():
    # Arrange
    trajectories = [
        _trajectory(["s0"], success=True),
        _trajectory(["s0"], success=False),
    ]

    # Act
    result = compute_gigpo_advantages(trajectories, gamma=1.0, mode="mean_norm")

    # Assert: anchor group is [1, 0] -> mean .5, so step advantages are +/- .5.
    assert result.step_advantages[0] == pytest.approx([0.5])
    assert result.step_advantages[1] == pytest.approx([-0.5])


@pytest.mark.unit
def test_equal_length_uniform_rewards_produce_no_signal():
    # Arrange: every episode succeeds AND every trajectory has the same length,
    # so every anchor group is constant and the combined advantage must vanish.
    trajectories = [_trajectory(["s0", "s1"], success=True) for _ in range(4)]

    # Act
    result = compute_gigpo_advantages(trajectories, gamma=1.0)

    # Assert
    assert result.mixed is False
    assert result.has_signal is False
    assert all(value == 0.0 for row in result.combined_advantages for value in row)


@pytest.mark.unit
def test_all_success_with_unequal_lengths_still_has_step_signal():
    """`mixed` describes the episode term, not the step term.

    With gamma < 1, a success that took 3 turns and one that took 2 have
    different return-to-go at their shared first anchor. Gating GiGPO on
    `mixed` would silently discard that signal.
    """
    # Arrange
    trajectories = [
        _trajectory(["s0", "s1", "s2"], success=True),
        _trajectory(["s0", "x1"], success=True),
    ]

    # Act
    result = compute_gigpo_advantages(trajectories, gamma=0.5, eps=1e-12)

    # Assert: episode term 0 (uniform rewards), step term nonzero at s0.
    assert result.mixed is False
    assert result.has_signal is True
    assert result.step_advantages[0][0] != 0.0
    assert result.combined_advantages[0][0] != 0.0


def test_rejected_attempts_from_the_same_state_share_an_anchor():
    """The anchor is the raw state, not the rejection text.

    Two trajectories rejected from the same room attempted different commands,
    so their `observation` strings differ. The underlying state is identical and
    must form one anchor group.
    """
    # Arrange
    success = _trajectory(["s0"], success=True)
    success.turns[0].anchor_observation = "s0"
    rejected = _trajectory(["ignored"], success=False)
    rejected.turns[0].observation = (
        "Your previous action 'take mug' was rejected: it is not one of the valid "
        "commands. You are still in the same situation.\n\ns0"
    )
    rejected.turns[0].anchor_observation = "s0"

    # Act
    grouped = compute_gigpo_advantages([success, rejected], gamma=1.0, eps=1e-12)
    rejected.turns[0].anchor_observation = ""
    ungrouped = compute_gigpo_advantages([success, rejected], gamma=1.0, eps=1e-12)

    # Assert
    assert grouped.step_advantages[0][0] == pytest.approx(0.5 / math.sqrt(0.5))
    assert ungrouped.step_advantages[0][0] == 0.0


def test_step_weight_scales_only_the_step_term():
    # Arrange
    base = compute_gigpo_advantages(FIXTURE, gamma=1.0, step_w=0.0)
    weighted = compute_gigpo_advantages(FIXTURE, gamma=1.0, step_w=2.0)

    # Act / Assert
    for episode in range(len(FIXTURE)):
        for turn in range(len(FIXTURE[episode].turns)):
            expected = (
                base.episode_advantages[episode]
                + 2.0 * base.step_advantages[episode][turn]
            )
            assert weighted.advantage_for(episode, turn) == pytest.approx(expected)


@pytest.mark.unit
def test_diagnostics_describe_the_anchor_structure():
    # Arrange / Act
    result = compute_gigpo_advantages(FIXTURE, gamma=1.0)

    # Assert — four anchors: s0 x4, s1a x2, s1b x2, s2 x2.
    diagnostics = result.diagnostics
    assert diagnostics["anchor_groups"] == 4.0
    assert diagnostics["turns"] == 10.0
    assert diagnostics["singleton_turns"] == 0.0
    assert diagnostics["singleton_fraction"] == pytest.approx(0.0)
    assert diagnostics["mean_group_size"] == pytest.approx(2.5)
    assert diagnostics["max_group_size"] == 4.0
    assert diagnostics["nonzero_step_fraction"] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "nope"}, "mode"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": 1.5}, "gamma"),
        ({"step_w": -1.0}, "step_w"),
        ({"step_w": math.inf}, "step_w"),
        ({"step_w": math.nan}, "step_w"),
    ],
)
def test_invalid_settings_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_gigpo_advantages(FIXTURE, **kwargs)


@pytest.mark.unit
def test_empty_group_is_rejected():
    with pytest.raises(ValueError):
        compute_gigpo_advantages([])


@pytest.mark.unit
def test_result_is_a_frozen_dataclass_with_grpo_compatible_view():
    result = compute_gigpo_advantages(FIXTURE, gamma=1.0)
    assert isinstance(result, GigpoGroupAdvantages)
    # `advantages` keeps the rollout's existing per-episode contract.
    assert result.advantages == result.episode_advantages
    assert result.num_episodes == len(FIXTURE)
    with pytest.raises(Exception):
        result.step_w = 3.0  # frozen
