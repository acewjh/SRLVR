"""Per-TOKEN hindsight credit -- the "full think/action token-level HSI" advantage.

Frozen formula (launch spec section 3)::

    d_j = log q_true(y_j) - log q_matched(y_j)
    A_j = (2R - 1) + trajectory_normalize((2R - 1) d_j)

``j`` indexes the tokens of the generated turn -- the whole ``think + action``
span, not a per-turn mean over a chosen sub-span. ``trajectory_normalize`` is the
per-trajectory standardisation ``(x - mu_i) / (std_i + eps)`` taken over the
trajectory's own scored tokens, so the mean of ``c_hat`` inside a trajectory is
~0 and ``mean_j A_j ~= 2R_i - 1``: the verifier still owns the trajectory's
direction and hindsight only redistributes credit *within* it.

Differences from the per-turn variant this replaces
---------------------------------------------------
The shipped v2 scorer collapsed each turn to one scalar (a mean over the action
or think span) and broadcast it back over the turn's tokens. That is the
"action-only HSI" the launch spec forbids for this arm: it discards the per-token
variation inside a turn, which is exactly the signal the repeat-stop gate and the
conditional stop loss are indexed against. Here every scored token keeps its own
``d_j``.

A turn with fewer than one scored token, or a trajectory whose ``d`` has no
spread, contributes ``c_hat = 0`` rather than a NaN -- the same degenerate-case
contract as :func:`tasks.alfworld.scorer.hindsight_credit`.
"""

from __future__ import annotations

__all__ = ["CreditStats", "standardize_within_trajectory", "token_advantages"]

_DEFAULT_EPS = 1e-6


class CreditStats:
    """Running distribution diagnostics for the per-token credit."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values) -> None:
        for value in values:
            value = float(value)
            self.count += 1
            self.total += value
            self.total_sq += value * value
            self.minimum = min(self.minimum, value)
            self.maximum = max(self.maximum, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean
        return max(self.total_sq / self.count - mean * mean, 0.0) ** 0.5

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_min": self.minimum if self.count else 0.0,
            f"{prefix}_max": self.maximum if self.count else 0.0,
            f"{prefix}_n": float(self.count),
        }


def standardize_within_trajectory(
    deltas: dict[int, "object"],
    episode_of: dict[int, int],
    anchors: dict[int, float],
    *,
    eps: float = _DEFAULT_EPS,
) -> dict[int, "object"]:
    """``c_hat`` per sample, standardised over the trajectory's scored tokens.

    ``deltas[i]`` is the per-token ``d_j`` tensor for sample ``i`` (length =
    that turn's generated-token count). Returns one float tensor per sample, the
    same length as its ``deltas`` entry.
    """
    import torch

    by_episode: dict[int, list[int]] = {}
    for sample_index, episode in episode_of.items():
        if sample_index in deltas:
            by_episode.setdefault(episode, []).append(sample_index)

    out: dict[int, "object"] = {}
    for episode, sample_indices in by_episode.items():
        anchor_value = float(anchors.get(episode, 0.0))
        sign = 1.0 if anchor_value >= 0.0 else -1.0
        pieces = [torch.as_tensor(deltas[i], dtype=torch.float64) * sign for i in sample_indices]
        stacked = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)
        if stacked.numel() < 2:
            for i in sample_indices:
                out[i] = torch.zeros_like(torch.as_tensor(deltas[i], dtype=torch.float32))
            continue
        mean = stacked.mean()
        std = stacked.std(unbiased=False)
        if float(std) < eps:
            for i in sample_indices:
                out[i] = torch.zeros_like(torch.as_tensor(deltas[i], dtype=torch.float32))
            continue
        for i in sample_indices:
            value = (torch.as_tensor(deltas[i], dtype=torch.float64) * sign - mean) / (std + eps)
            out[i] = value.to(torch.float32)
    return out


def token_advantages(
    deltas: dict[int, "object"],
    episode_of: dict[int, int],
    anchors: dict[int, float],
    response_lengths: dict[int, int],
    *,
    eps: float = _DEFAULT_EPS,
):
    """``A_j`` per sample: ``(2R - 1) + c_hat_j``, length = the turn's tokens.

    Samples with no scored ``d`` (an unscored turn, or a leak-guard skip) receive
    the flat anchor ``2R - 1`` on every token -- the method's own fallback, not a
    silent zero.
    """
    import torch

    credit = standardize_within_trajectory(deltas, episode_of, anchors, eps=eps)
    advantages: dict[int, "object"] = {}
    stats = CreditStats()
    for sample_index, rlen in response_lengths.items():
        rlen = int(rlen)
        if rlen <= 0:
            advantages[sample_index] = torch.zeros(0, dtype=torch.float32)
            continue
        anchor_value = float(anchors.get(episode_of.get(sample_index, -1), 0.0))
        row = torch.full((rlen,), anchor_value, dtype=torch.float32)
        if sample_index in credit:
            piece = torch.as_tensor(credit[sample_index], dtype=torch.float32)[:rlen]
            row[: piece.numel()] = row[: piece.numel()] + piece
            stats.update(piece.tolist())
        advantages[sample_index] = row
    return advantages, stats
