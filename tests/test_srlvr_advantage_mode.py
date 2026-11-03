"""SRLVR_ADVANTAGE_MODE: history (default) vs REINFORCE++-style batch statistics.

The SRLVR actor advantage is ``A = (R - b - 1_U * lambda * B) / s``. The published
method takes ``b`` and ``s`` from the calibration history window. With one
trajectory per task position there is no group to normalise against, so the batch
is the only other reference set available -- that is the option added here.

These tests pin three things: the helper's arithmetic, its refusal to invent a
statistic for a degenerate batch, and that the default mode is untouched.
"""

import math
import os
from types import SimpleNamespace

import pytest

from opsd_core.srlvr_settings import (
    ADVANTAGE_MODES,
    Settings,
    advantage_mode,
    batch_baseline_and_scale,
)
from methods.srlvr_hsi import (
    advance_fallbacks,
    batch_rewards,
    clip_lambda,
    convert,
    effective_advantage_stats,
    frozen_history,
    frozen_lambda_for_batch,
    next_history,
    prepare_history,
    probe_advantage,
    score_norm,
    validate_batch_mode,
)


def test_modes_are_the_documented_pair():
    assert ADVANTAGE_MODES == ("history", "batch")


def test_default_is_history(monkeypatch):
    monkeypatch.delenv("SRLVR_ADVANTAGE_MODE", raising=False)
    assert advantage_mode() == "history"


def test_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("SRLVR_ADVANTAGE_MODE", "group")
    with pytest.raises(ValueError, match="SRLVR_ADVANTAGE_MODE"):
        advantage_mode()
    assert "group" not in ADVANTAGE_MODES


def test_batch_statistics_use_the_same_form_as_history():
    """b = clipped mean, s = sqrt(delta_s^2 + mean((r - mean)^2)) -- same shape."""
    rewards = [0.0, 0.0, 1.0, 1.0]
    delta_s = 0.1
    b, s, how = batch_baseline_and_scale(rewards, 0.5, 1.0, delta_s)
    mean = 0.5
    assert how == "batch"
    assert b == pytest.approx(mean)
    assert s == pytest.approx(math.sqrt(delta_s ** 2 + sum((r - mean) ** 2 for r in rewards) / 4))
    # The delta_s floor is what keeps s away from zero on a near-uniform batch.
    assert s >= delta_s


def test_baseline_is_clipped_away_from_the_boundaries():
    b, _, _ = batch_baseline_and_scale([0.0] * 9 + [1.0], 0.5, 1.0, 0.1)
    assert 0.0 < b < 1.0


@pytest.mark.parametrize("rewards,reason", [
    ([], "empty_batch"),
    ([1.0], "degenerate_batch"),
    ([0.0, 0.0, 0.0], "degenerate_batch"),
])
def test_degenerate_batches_fall_back_to_history_and_say_so(rewards, reason):
    """No silent statistics: a batch that cannot support one reports the fallback."""
    b, s, how = batch_baseline_and_scale(rewards, 0.37, 0.83, 0.1)
    assert how == reason
    assert (b, s) == (0.37, 0.83)


# --------------------------------------------------------------------------- #
# SRLVR-v2 batch path: one reward per trajectory, effective statistics in every
# consumer, lambda clipped against the baseline in use, mode recorded on convert.
# --------------------------------------------------------------------------- #

def _turn_row(position, turn, reward):
    return {"meta": {"position": position, "turn": turn, "terminal_reward": reward}}


def test_batch_rewards_count_each_trajectory_once_not_once_per_turn():
    """A multi-turn episode repeats its reward on every row; the batch must not."""
    trajectories = {
        position: [_turn_row(position, turn, float(position % 2))
                   for turn in range(position % 4 + 1)]
        for position in range(6)
    }
    # 6 trajectories with 1..4 turns each -> 13 rows, but exactly 6 rewards.
    assert sum(len(rows) for rows in trajectories.values()) == 13
    assert batch_rewards(trajectories, 6) == [0., 1., 0., 1., 0., 1.]


def test_batch_rewards_fail_closed_on_missing_or_extra_trajectory():
    trajectories = {position: [_turn_row(position, 0, 0.)] for position in (0, 2)}
    with pytest.raises(RuntimeError, match=r"missing=\[1\]"):
        batch_rewards(trajectories, 3)
    with pytest.raises(RuntimeError, match=r"extra=\[2\]"):
        batch_rewards(trajectories, 2)


def test_batch_mode_with_eps_b0_is_used_and_recorded_as_source():
    cfg = Settings(variant="v2", gamma=.99, calibration=2, eps_b0=.02)
    rewards = [0.] * 99 + [1.]
    mean = sum(rewards) / len(rewards)
    b, s, source = effective_advantage_stats("batch", rewards, .5, 1.7, cfg)
    assert source == "batch"
    # Mean == eps_b0 exactly, so the clip is observable and eps_b0 was plumbed.
    assert b == pytest.approx(.02)
    assert s == pytest.approx(
        math.sqrt(cfg.delta_s ** 2 + sum((r - mean) ** 2 for r in rewards) / len(rewards))
    )


def test_history_mode_keeps_published_statistics_and_batch_fallback_counts():
    cfg = Settings(variant="v2", gamma=.99, calibration=2)
    assert effective_advantage_stats("history", [0., 1.], .37, .83, cfg) == (.37, .83, "history")
    b, s, source = effective_advantage_stats("batch", [1., 1.], .37, .83, cfg)
    assert (b, s, source) == (.37, .83, "degenerate_batch")
    assert advance_fallbacks(prior=2, source=source, mode="batch") == 3
    assert advance_fallbacks(prior=2, source="history", mode="history") == 2


def test_probe_advantages_and_score_norm_share_the_effective_baseline_and_scale():
    b_eff, s_eff = .25, .55
    # Z removes this batch's baseline; Y isolates B; neither uses history values.
    assert probe_advantage(1., .4, b_eff, s_eff, "Z") == pytest.approx(-(.75 / .55))
    assert probe_advantage(1., .4, b_eff, s_eff, "Y") == pytest.approx(.4 / .55)
    # W = ||S||^2 recovered from ||Z||^2 with the same effective scale.
    assert score_norm(4., 1., b_eff, s_eff) == pytest.approx(4. * (.55 / .75) ** 2)
    with pytest.raises(RuntimeError, match="reward equals the baseline"):
        score_norm(1., b_eff, b_eff, s_eff)


def test_lambda_bound_uses_the_baseline_actually_in_use():
    cfg = Settings(variant="v2", gamma=.99, calibration=2, rho=.9)
    assert clip_lambda(5., .2, cfg) == pytest.approx(.9 * .2)
    assert clip_lambda(-5., .8, cfg) == pytest.approx(-.9 * .2)
    assert clip_lambda(.1, .5, cfg) == pytest.approx(.1)


def test_next_batch_lambda_is_frozen_from_calibration_and_clipped(monkeypatch):
    cfg = Settings(variant="v2", gamma=.99, calibration=2, rho=.9)
    state = dict(b=.4, s=.5, history=[], last_rollout=4, actor_optimizer_steps=5,
                 reduction="trajectory_token_mean_v1", advantage_mode="history",
                 lambda_raw=.99)
    # The coefficient in force for rollout 5 was frozen at the end of rollout 4.
    assert frozen_history(state, cfg, 5)[2] == pytest.approx(.9 * .4)
    # Rollout 5's calibration updates the NEXT coefficient, not this batch's.
    updated = next_history(state, [], cfg, c_hat=.9, d_hat=1., rollout_id=5)
    assert state["lambda_raw"] == .99
    assert updated["lambda_raw"] == pytest.approx(.9)
    assert updated["lambda_next"] == pytest.approx(.9 * .4)
    assert updated["advantage_mode"] == "history"


def test_batch_lambda_is_clipped_once_at_the_effective_baseline():
    """History b=.01 must not narrow a batch-b=.5 run: raw .3 stays .3, not .009."""
    cfg = Settings(variant="v2", gamma=.99, calibration=2, rho=.9)
    state = dict(b=.01, s=1., history=[], last_rollout=4, actor_optimizer_steps=5,
                 reduction="trajectory_token_mean_v1", advantage_mode="batch",
                 lambda_raw=.3)
    # The published history bound alone would give .009...
    assert frozen_history(state, cfg, 5)[2] == pytest.approx(.009)
    # ... but the effective batch baseline .5 allows rho*min(.5,.5) = .45.
    lam, raw, limit = frozen_lambda_for_batch(state, cfg, 5, .5)
    assert (lam, raw, limit) == pytest.approx((.3, .3, .45))
    # Reverse direction, and a tighter effective baseline still binds.
    assert frozen_lambda_for_batch({**state, "lambda_raw": -.3}, cfg, 5, .5)[0] == pytest.approx(-.3)
    assert frozen_lambda_for_batch(state, cfg, 5, .1)[0] == pytest.approx(.09)
    # history mode and the degenerate fallback pass the historical b, unchanged.
    assert frozen_lambda_for_batch(state, cfg, 5, .01)[0] == pytest.approx(.009)


def test_train_batch_reads_the_unclipped_ratio_and_clips_once():
    """Wiring guard: the real path must not pre-clip with the history bound."""
    import inspect

    from methods import srlvr_hsi

    source = inspect.getsource(srlvr_hsi.train_batch)
    assert "frozen_lambda_for_batch(" in source
    assert "clip_lambda(lam, eff_b" not in source
    assert source.index("effective_advantage_stats(") < source.index("frozen_lambda_for_batch(")
    assert "lambda_raw_frozen=lambda_raw" in inspect.getsource(srlvr_hsi.train_batch)


def test_resume_rejects_a_changed_advantage_mode(monkeypatch):
    cfg = Settings(variant="v2", gamma=.99, calibration=2)
    state = dict(b=.5, s=1., history=[], last_rollout=6, actor_optimizer_steps=7,
                 reduction="trajectory_token_mean_v1", advantage_mode="batch",
                 batch_fallbacks=3)
    monkeypatch.setenv("SRLVR_ADVANTAGE_MODE", "history")
    with pytest.raises(RuntimeError, match="changed advantage mode"):
        prepare_history(state, cfg, 7)
    monkeypatch.setenv("SRLVR_ADVANTAGE_MODE", "batch")
    assert prepare_history(state, cfg, 7)["batch_fallbacks"] == 3


def test_validate_batch_mode_fails_closed_on_a_changed_mode():
    validate_batch_mode("batch", "batch")
    with pytest.raises(RuntimeError, match="changed after sampling"):
        validate_batch_mode("history", "batch")


def _convert_args(monkeypatch):
    monkeypatch.setenv("SRLVR_VARIANT", "v2")
    monkeypatch.setenv("SRLVR_CALIBRATION", "2")
    monkeypatch.setenv("SRLVR_ADVANTAGE_MODE", "batch")
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    monkeypatch.delenv("OPSD_LORA_DROPOUT", raising=False)
    return SimpleNamespace(
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, tensor_model_parallel_size=2,
        actor_num_nodes=1, actor_num_gpus_per_node=2,
        n_samples_per_prompt=1, rollout_batch_size=4,
        rollout_top_p=1., rollout_top_k=-1, rollout_temperature=1.,
        attention_dropout=0., hidden_dropout=0., entropy_coef=0.,
        calculate_per_token_loss=False, micro_batch_size=1,
        use_dynamic_batch_size=False, update_weights_interval=1,
        keep_old_actor=False,
    )


def _sample(position, turn, reward):
    return SimpleNamespace(
        tokens=[1, 2, 3], response_length=2, loss_mask=[1, 1],
        rollout_log_probs=[-.1, -.2], turn_ids=[position * 1000 + turn] * 2,
        metadata={"srlvr": {"position": position, "turn": turn}},
        reward=reward, status=SimpleNamespace(name="COMPLETED"),
    )


def test_convert_records_mode_and_counts_each_real_trajectory(monkeypatch):
    args = _convert_args(monkeypatch)
    samples = [_sample(position, turn, float(position % 2))
               for position in range(4) for turn in range(2)]
    converted = convert(args, samples)
    header = converted["srlvr_batch"]
    assert header["mode"] == "batch"
    assert (header["N"], header["real_rows"], header["rows"]) == (4, 8, 8)
    # Two turns x two actor tokens -> L_i=4 per trajectory, counted once.
    assert {row["actor_token_count"] for row in converted["srlvr_rows"]} == {4}

