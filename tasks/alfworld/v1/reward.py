"""ALFWorld reward -- terminal environment success ONLY.

Deliberately minimal::

    success  -> 1.0
    failure  -> 0.0

No action-format bonus, no turn penalty, no progress shaping. The main GRPO
result must be readable as "did the agent complete the task", and every shaping
term silently redefines that. Format and admissibility problems are recorded as
*diagnostics* on the trajectory (``invalid_action_count``, ``truncated``) and
reported in the eval table, but never enter the reward.

An episode that ran out of turns without terminating is a failure (0.0), not a
separate reward level; it is additionally flagged ``truncated`` so the eval can
report truncation rate separately.
"""

__all__ = ["terminal_reward", "reward_from_trajectory"]


def terminal_reward(success: bool) -> float:
    """The one and only reward function for the main result."""
    return 1.0 if success else 0.0


def reward_from_trajectory(trajectory) -> float:
    """Reward for a finished episode (``--custom-rm-path`` compatible)."""
    return terminal_reward(bool(trajectory.success))
