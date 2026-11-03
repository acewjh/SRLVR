"""Trajectory-level GRPO advantage, computed once per episode.

Why this lives in the rollout
-----------------------------
v2 emits one training row per turn. slime's stock GRPO normalises *positionally*
over the flattened sample list (`slime/ray/rollout.py:717-728`), so with per-turn
rows it would compute the group statistics over TURNS rather than over episodes --
a long episode would silently get more weight than a short one.

The advantage therefore has to be computed where the episode structure is still
known: inside the rollout, over the ``n_samples_per_prompt`` episodes of one task.
It is then handed to the trainer through ``Sample.opsd_advantages`` (with
``--use-opsd``), which slime passes through verbatim, bypassing its estimator and
all of its normalisation. That keeps the semantics exactly "GRPO over episodes".

Zero-advantage groups
---------------------
When every episode in a group has the same reward the advantage is identically 0,
so ``w * pg_loss == 0`` for every token no matter what the HPI weight is. Those
groups are reported as ``mixed=False`` and the scorer skips them entirely -- no
teacher forward is ever run for them (see ``scorer.py``).
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import config

__all__ = [
    "GIGPO_MODES",
    "GigpoGroupAdvantages",
    "GroupAdvantages",
    "compute_gigpo_advantages",
    "compute_group_advantages",
]

#: Step-level normalisation modes from the official `algorithm.gigpo.mode`.
GIGPO_MODES = ("mean_std_norm", "mean_norm")


@dataclass(frozen=True)
class GroupAdvantages:
    """Per-episode advantage for one task group, plus the mixed/skip decision."""

    advantages: list[float]
    mean_reward: float
    std_reward: float
    mixed: bool
    num_episodes: int

    @property
    def has_signal(self) -> bool:
        return self.mixed


def compute_group_advantages(rewards: list[float], *, eps: float = 1e-6) -> GroupAdvantages:
    """GRPO group-relative advantage: ``(R_i - mean) / (std + eps)``.

    Matches slime's own GRPO normalisation (population mean, ``unbiased=True`` std)
    so the statistic is the one the framework would have produced had the rows been
    one-per-episode.

    ``mixed`` is False when every reward is equal -- in that case the advantage is
    exactly 0 and no HPI scoring can change the update, so callers must skip it.
    """
    if not rewards:
        raise ValueError("compute_group_advantages needs at least one reward")
    n = len(rewards)
    mean = sum(rewards) / n
    if n > 1:
        variance = sum((r - mean) ** 2 for r in rewards) / (n - 1)
        std = variance**0.5
    else:
        std = 0.0
    mixed = any(reward != rewards[0] for reward in rewards)
    if not mixed or std == 0.0:
        return GroupAdvantages([0.0] * n, mean, std, False, n)
    advantages = [(reward - mean) / (std + eps) for reward in rewards]
    return GroupAdvantages(advantages, mean, std, True, n)


# --------------------------------------------------------------------------- #
# GiGPO: episode GRPO advantage + anchor-state step advantage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GigpoGroupAdvantages:
    """Per-turn advantages for one task group under GiGPO.

    ``A_{i,t} = A^E_i + w * A^S_{i,t}`` (arXiv:2505.10978, Eq. 8), where

    * ``A^E_i`` is the ordinary GRPO episode advantage of trajectory ``i`` --
      byte-for-byte what the matched GRPO arm computes;
    * ``A^S_{i,t}`` standardises the discounted return-to-go inside the ANCHOR
      STATE GROUP of turn ``t``: every turn of this task group (all G
      trajectories) whose raw environment state is identical. A group of one,
      or a group whose returns are all equal, contributes exactly 0.

    ``mixed`` describes the EPISODE term (whether the group's rewards vary).
    ``has_signal`` describes the COMBINED advantage: with ``gamma < 1`` two
    successful trajectories of different lengths have different return-to-go at
    a shared anchor, so the step term can carry signal even when every episode
    succeeded. Do not gate GiGPO on ``mixed``.

    ``advantages`` exposes the per-episode term so this object is a drop-in for
    ``GroupAdvantages`` in the rollout; consumers that want the method's actual
    per-turn credit call :meth:`advantage_for`.
    """

    episode_advantages: list[float]
    step_advantages: list[list[float]]
    combined_advantages: list[list[float]]
    mean_reward: float
    std_reward: float
    mixed: bool
    num_episodes: int
    step_w: float
    gamma: float
    mode: str
    diagnostics: Mapping[str, float]

    @property
    def advantages(self) -> list[float]:
        return self.episode_advantages

    @property
    def has_signal(self) -> bool:
        return any(
            value != 0.0 for row in self.combined_advantages for value in row
        )

    def advantage_for(self, episode_index: int, turn_index: int) -> float:
        return self.combined_advantages[episode_index][turn_index]


def _trajectory_reward(trajectory: Any) -> float:
    """Terminal reward of one trajectory (0/1 for ALFWorld)."""
    reward = getattr(trajectory, "terminal_reward", None)
    if reward is None:
        return 1.0 if getattr(trajectory, "success", False) else 0.0
    return float(reward)


def _step_returns(turns: Sequence[Any], gamma: float) -> list[float]:
    """Discounted return-to-go ``sum_{k>=t} gamma^(k-t) r_k`` per turn.

    ALFWorld pays no intermediate reward, so this is ``gamma^(T-1-t)`` for a
    successful episode and 0 otherwise. Keeping the general recurrence means the
    step term stays correct if the environment ever reports dense reward.
    """
    returns = [0.0] * len(turns)
    running = 0.0
    for index in range(len(turns) - 1, -1, -1):
        running = float(getattr(turns[index], "env_reward", 0.0) or 0.0) + gamma * running
        returns[index] = running
    return returns


def _anchor_state(turn: Any) -> str:
    """The state a turn acted on, excluding the invalid-action rejection wrapper.

    ``TurnRecord.anchor_observation`` is the raw environment observation (see
    ``tasks.alfworld.episode.run_episode``). Records written before that field
    existed fall back to ``observation``; for a VALID turn the two are identical,
    and for a rejected one the fallback is the best available key.
    """
    anchor = str(getattr(turn, "anchor_observation", "") or "")
    return anchor or str(getattr(turn, "observation", ""))


def _anchor_groups(trajectories: Sequence[Any]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Map ``(task_id, anchor state)`` to the turns standing in that state.

    History and thinking are excluded, exactly as the reference implementation's
    ``anchor`` observation. The task id is part of the key so two different
    ALFWorld games that happen to share a room description can never pool their
    returns.
    """
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for episode_index, trajectory in enumerate(trajectories):
        task_id = str(getattr(trajectory, "task_id", ""))
        for turn_index, turn in enumerate(getattr(trajectory, "turns", [])):
            groups[(task_id, _anchor_state(turn))].append((episode_index, turn_index))
    return groups


def compute_gigpo_advantages(
    trajectories: Sequence[Any],
    *,
    step_w: float | None = None,
    gamma: float | None = None,
    mode: str | None = None,
    eps: float = 1e-6,
) -> GigpoGroupAdvantages:
    """GiGPO advantages for ONE task group (all G episodes of one game).

    Defaults come from the environment (``ALFWORLD_V2_GIGPO_*``) so the rollout
    reads each knob in one place; explicit arguments exist for tests.
    """
    if not trajectories:
        raise ValueError("compute_gigpo_advantages needs at least one trajectory")
    step_w = config.gigpo_step_weight() if step_w is None else float(step_w)
    gamma = config.gigpo_gamma() if gamma is None else float(gamma)
    mode = config.gigpo_mode() if mode is None else str(mode).strip().lower()
    if mode not in GIGPO_MODES:
        raise ValueError(f"unknown GiGPO mode {mode!r}; expected one of {GIGPO_MODES}")
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if not math.isfinite(step_w) or step_w < 0.0:
        raise ValueError(f"step_w must be finite and >= 0, got {step_w}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    rewards = [_trajectory_reward(trajectory) for trajectory in trajectories]
    episode = compute_group_advantages(rewards, eps=eps)
    episode_advantages = list(episode.advantages)

    returns = [
        _step_returns(getattr(trajectory, "turns", []), gamma) for trajectory in trajectories
    ]
    step_advantages = [
        [0.0] * len(getattr(trajectory, "turns", [])) for trajectory in trajectories
    ]

    groups = _anchor_groups(trajectories)
    sizes = [len(members) for members in groups.values()]
    singleton_turns = sum(size for size in sizes if size == 1)
    nonzero_steps = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [returns[episode_index][turn_index] for episode_index, turn_index in members]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = variance**0.5
        if mode == "mean_std_norm":
            scale = std + eps
            if std == 0.0:
                continue
        else:  # mean_norm
            scale = 1.0
        for (episode_index, turn_index), value in zip(members, values, strict=True):
            step = (value - mean) / scale
            step_advantages[episode_index][turn_index] = step
            nonzero_steps += int(step != 0.0)

    combined = [
        [
            episode_advantages[episode_index] + step_w * step
            for step in step_advantages[episode_index]
        ]
        for episode_index in range(len(trajectories))
    ]

    turns = sum(len(getattr(trajectory, "turns", [])) for trajectory in trajectories)
    abs_steps = [abs(value) for row in step_advantages for value in row]
    diagnostics = {
        "anchor_groups": float(len(groups)),
        "turns": float(turns),
        "singleton_turns": float(singleton_turns),
        "singleton_fraction": float(singleton_turns) / max(turns, 1),
        "mean_group_size": float(turns) / max(len(groups), 1),
        "max_group_size": float(max(sizes, default=0)),
        "nonzero_step_fraction": float(nonzero_steps) / max(turns, 1),
        "step_adv_mean_abs": sum(abs_steps) / max(len(abs_steps), 1),
        "episode_adv_mean_abs": sum(abs(value) for value in episode_advantages)
        / max(len(episode_advantages), 1),
    }
    return GigpoGroupAdvantages(
        episode_advantages=episode_advantages,
        step_advantages=step_advantages,
        combined_advantages=combined,
        mean_reward=episode.mean_reward,
        std_reward=episode.std_reward,
        mixed=episode.mixed,
        num_episodes=len(trajectories),
        step_w=step_w,
        gamma=gamma,
        mode=mode,
        diagnostics=diagnostics,
    )
