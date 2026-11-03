"""Optimizer weight must depend on the TRAJECTORY, never on its length.

The collapse this fixes: a trajectory is one row per turn, every row carries its
episode's GRPO advantage, and equal row weighting let a long failing trajectory
outvote a short successful one purely by having more turns. Measured on a real 8B
rollout: 36 positive rows against 572 negative (15.9:1), so ~94% of the gradient
said "push these tokens down" and the policy bought silence by destroying its own
``<think>``/``</think>`` structure.

These tests use a deliberately extreme imbalance -- one short successful
trajectory and several long failing ones -- and assert the aggregate weight each
trajectory receives is equal. The assertions are on OPTIMIZER CONTRIBUTION, not on
`N_i` terms, so they hold through the whole chain
`train_data.trajectory_weighting` -> `loss.trajectory_balanced_reduce`.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from slime.utils.types import Sample  # noqa: E402

from tasks.alfworld import config  # noqa: E402
from tasks.alfworld.loss import trajectory_balanced_reduce  # noqa: E402
from tasks.alfworld.train_data import (  # noqa: E402
    _has_trainable_rows,
    convert_samples_to_train_data,
    outcome_composition,
    trajectory_weighting,
)


class _Row:
    """Minimal stand-in for a rollout Sample: only what the weighting reads."""

    def __init__(self, episode: int, turn: int, tokens: int):
        self.turn_ids = [episode * config.TURN_ID_STRIDE + turn] * tokens
        self.loss_mask = [1] * tokens
        self.response_length = tokens
        self.fit_turn_weights = [1.0] * tokens
        # GRPO advantage, constant across the row's tokens
        self.opsd_advantages = [0.0] * tokens

    def with_advantage(self, value: float):
        self.opsd_advantages = [value] * self.response_length
        return self


def _imbalanced_batch():
    """1 success: 2 short turns. 3 failures: 5 long turns each.

    Rows 3 vs 15 (5:1) and tokens 60 vs 1500 (25:1) -- both imbalances point the
    same way, which is exactly the real failure shape.
    """
    rows = []
    for turn, tokens in enumerate([20, 40]):                  # episode 0, SUCCESS
        rows.append(_Row(0, turn, tokens).with_advantage(+0.75))
    for episode in (1, 2, 3):                                 # episodes 1-3, FAILURE
        for turn, tokens in enumerate([80, 100, 120, 100, 100]):
            rows.append(_Row(episode, turn, tokens).with_advantage(-0.25))
    return rows


def _row_weight_sums(rows):
    """Aggregate reducer weight per trajectory, from the weighting itself."""
    weights, _scale, _diag = trajectory_weighting(rows)
    sums = {}
    for row, per_token in zip(rows, weights, strict=True):
        episode = row.turn_ids[0] // config.TURN_ID_STRIDE
        sums[episode] = sums.get(episode, 0.0) + per_token[0]
    return sums


def test_every_trajectory_receives_equal_total_weight_despite_25x_token_ratio():
    # Arrange: 60 tokens in the success vs 500 each in the three failures.
    rows = _imbalanced_batch()

    # Act
    sums = _row_weight_sums(rows)

    # Assert
    assert len(sums) == 4
    for episode, total in sums.items():
        assert total == pytest.approx(1.0), f"trajectory {episode} weight {total}"


def test_the_short_success_is_not_swamped_by_the_long_failures():
    # Arrange
    rows = _imbalanced_batch()

    # Act
    sums = _row_weight_sums(rows)

    # Assert: the 2-row success gets the SAME total weight as each 5-row failure.
    # This is the whole point -- before the fix its share was 2/17 of the rows.
    assert sums[0] == pytest.approx(sums[1])
    assert sums[0] == pytest.approx(sums[2])
    assert sums[0] == pytest.approx(sums[3])
    naive_row_share = 2 / len(rows)
    assert sums[0] != pytest.approx(naive_row_share), "equal-row weighting is the bug"


def test_reducer_output_equals_per_trajectory_token_mean():
    # Arrange: distinct per-token losses per row so a wrong weighting cannot hide.
    rows = _imbalanced_batch()
    _, scale, _ = trajectory_weighting(rows)
    lengths = [r.response_length for r in rows]
    per_token = torch.cat(
        [torch.full((length,), 1.0 + 0.5 * i) for i, length in enumerate(lengths)]
    )
    weights, _s, _d = trajectory_weighting(rows)
    batch = {
        "fit_turn_weights": weights,
        "traj_norm_scale": [[scale] * length for length in lengths],
    }

    # Act
    loss, _mult, _w = trajectory_balanced_reduce(per_token, lengths, batch)

    # Assert: hand-compute mean_i (1/N_i) sum_tokens l -- the objective the spec
    # asks for. `loss` already carries `scale`, and the framework then divides by
    # N_global == sum(N_i) == total_tokens, so what the optimizer sees is exactly
    # `loss / total_tokens`. (Multiplying by `scale` a second time here is the
    # easy mistake; this test caught it.)
    expected = 0.0
    offset = 0
    episodes = {}
    for i, length in enumerate(lengths):
        segment = per_token[offset : offset + length]
        offset += length
        episode = rows[i].turn_ids[0] // config.TURN_ID_STRIDE
        episodes.setdefault(episode, []).append(segment)
    for segments in episodes.values():
        joined = torch.cat(segments)
        expected += joined.sum() / joined.numel()
    expected /= len(episodes)

    total_tokens = float(sum(lengths))
    assert float(loss.item()) / total_tokens == pytest.approx(float(expected), rel=1e-6)


def test_weighting_is_invariant_to_adding_turns_to_one_trajectory():
    # Arrange: same as baseline, then pad ONE failing trajectory with 5 more turns.
    base = _imbalanced_batch()
    padded = _imbalanced_batch()
    for turn in range(5, 10):
        padded.append(_Row(1, turn, 100).with_advantage(-0.25))

    # Act
    base_sums = _row_weight_sums(base)
    padded_sums = _row_weight_sums(padded)

    # Assert: trajectory 1's total weight is unchanged even though it now owns
    # twice the rows -- a trajectory cannot buy influence by rambling.
    assert padded_sums[1] == pytest.approx(1.0)
    assert padded_sums[1] == pytest.approx(base_sums[1])
    assert padded_sums[0] == pytest.approx(base_sums[0])


def test_outcome_composition_reports_the_imbalance_that_caused_the_collapse():
    # Arrange
    rows = _imbalanced_batch()

    # Act
    composition = outcome_composition([(r.opsd_advantages[0], r.response_length) for r in rows])

    # Assert
    assert composition["rows_pos"] == 2
    assert composition["rows_neg"] == 15
    assert composition["row_neg_over_pos"] == pytest.approx(7.5)
    assert composition["token_neg_over_pos"] == pytest.approx(1500 / 60)


def test_missing_weights_is_an_error_not_a_silent_fallback():
    # Arrange: a batch without the weighting must fail loudly. Falling back to
    # equal rows IS the original bug, so it must never be the quiet path.
    per_token = torch.ones(10)
    batch = {}

    # Act / Assert
    with pytest.raises(ValueError, match="traj_norm_scale"):
        trajectory_balanced_reduce(per_token, [10], batch)


def test_length_mismatch_is_rejected():
    # Arrange
    per_token = torch.ones(10)
    batch = {"fit_turn_weights": [[1.0] * 4], "traj_norm_scale": [[5.0] * 4]}

    # Act / Assert
    with pytest.raises(ValueError, match="shorter than its response length"):
        trajectory_balanced_reduce(per_token, [10], batch)


class _ConvertibleSample:
    """A rollout Sample with exactly the fields the conversion reads."""

    def __init__(self, index: int, episode: int, turn_tokens: list[int], reward: float = 0.0):
        self.index = index
        self.turn_ids = [
            episode * config.TURN_ID_STRIDE + turn
            for turn, n in enumerate(turn_tokens)
            for _ in range(n)
        ]
        self.response_length = len(self.turn_ids)
        self.tokens = [7] * (self.response_length + 4)  # 4 prompt tokens + response
        self.loss_mask = [1] * self.response_length
        self.opsd_advantages = [0.0] * self.response_length
        self.fit_turn_weights = [1.0] * self.response_length
        self.reward = reward
        self.status = Sample.Status.COMPLETED
        self.metadata = {}
        self.rollout_log_probs = None

    @classmethod
    def generated_nothing(cls, index: int, episode: int) -> "_ConvertibleSample":
        """What a rollout that produced no tokens looks like: the PROMPT only.

        `tokens` is NOT empty -- that is the whole trap -- but `response_length`
        and `turn_ids` are both empty.
        """
        sample = cls.__new__(cls)
        sample.index = index
        sample.turn_ids = []
        sample.response_length = 0
        sample.tokens = [7] * 4
        sample.loss_mask = []
        sample.opsd_advantages = []
        sample.fit_turn_weights = []
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        sample.metadata = {}
        sample.rollout_log_probs = None
        return sample


def test_has_trainable_rows_needs_a_response_not_just_tokens():
    # Arrange: both samples have non-empty `tokens`; only one has turn-rows.
    good = _ConvertibleSample(1, 1, [10])
    empty = _ConvertibleSample.generated_nothing(0, 0)

    # Act / Assert
    assert _has_trainable_rows(good) is True
    assert _has_trainable_rows(empty) is False


def test_a_rollout_that_generated_nothing_is_dropped_not_fatal():
    """Regression -- 2026-10-06 the 1.7B alfworld HSI cell died at rollout 55/100.

    One sample came back PROMPT-only, and `trajectory_weighting` (train_data.py:72)
    then `advantage_diagnostics` (:155) both index `turn_ids[0]` and raised
    IndexError, killing the trainer mid-cell (rc=1) after ~9 h. A degenerate single
    sample must cost one row, not the run.
    """
    # Arrange: a full batch for GBS=4, with the degenerate sample placed FIRST so
    # the pre-fix code path is exercised before any GBS trim can mask it.
    empty = _ConvertibleSample.generated_nothing(0, 0)
    good = [_ConvertibleSample(i, 1, [10, 10], reward=1.0) for i in range(1, 5)]

    # Act
    data = convert_samples_to_train_data(SimpleNamespace(global_batch_size=4), [empty, *good])

    # Assert: the degenerate sample is gone, the four real rows survive intact.
    assert data["sample_indices"] == [1, 2, 3, 4]
    assert len(data["tokens"]) == 4
    assert all(len(row) > 0 for row in data["turn_ids"])
    assert all(len(row) > 0 for row in data["fit_turn_weights"])


def test_a_batch_of_only_empty_responses_fails_loudly():
    # Arrange: dropping must not quietly turn into an empty training batch.
    samples = [_ConvertibleSample.generated_nothing(i, i) for i in range(3)]

    # Act / Assert
    with pytest.raises(RuntimeError, match="no usable samples"):
        convert_samples_to_train_data(SimpleNamespace(global_batch_size=4), samples)


def test_full_batch_preserves_tail_and_padding_has_exact_zero_gradient(monkeypatch):
    monkeypatch.setenv("ALFWORLD_V2_HPI", "0")
    args = SimpleNamespace(global_batch_size=4, rollout_batch_size=2,
                           n_samples_per_prompt=2, actor_num_gpus_per_node=2,
                           actor_num_nodes=1)
    rows = [_ConvertibleSample(i, ep, [n]) for i, (ep, n) in
            enumerate([(0, 2), (0, 3), (1, 7), (2, 11), (3, 5)])]
    data = convert_samples_to_train_data(args, rows)
    assert data["sample_indices"] == [0, 1, 2, 3, 4, -1]
    assert data["dynamic_global_batch_size"] == 6
    assert data["loss_masks"][-1] == [0, 0]
    assert data["fit_turn_weights"][-1] == [0, 0]
    assert all(s.loss_mask == [1] * s.response_length for s in rows)
    values = torch.arange(1., 7., requires_grad=True)
    losses = torch.cat([values[i].expand(n) for i, n in enumerate(data["response_lengths"])])
    summed, _, _ = trajectory_balanced_reduce(losses, data["response_lengths"], data)
    denominator = sum(max(sum(m), 1) for m in data["loss_masks"])
    actual = summed / denominator
    expected = ((2 * values[0] + 3 * values[1]) / 5 + values[2] + values[3] + values[4]) / 4
    assert actual.item() == pytest.approx(expected.item())
    assert torch.allclose(torch.autograd.grad(actual, values)[0],
                          torch.autograd.grad(expected, values)[0])


def test_full_batch_rejects_a_missing_trajectory(monkeypatch):
    monkeypatch.setenv("ALFWORLD_V2_HPI", "0")
    args = SimpleNamespace(global_batch_size=4, rollout_batch_size=2,
                           n_samples_per_prompt=2)
    rows = [_ConvertibleSample(i, i, [3]) for i in range(3)]
    with pytest.raises(RuntimeError, match="lost trajectories"):
        convert_samples_to_train_data(args, rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
