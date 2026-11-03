"""v2 sample -> train-data conversion.

Forwards exactly the keys slime already whitelists for the DP split and forwards
into the training batch -- ``tokens``, ``response_lengths``, ``loss_masks``,
``rollout_log_probs``, ``turn_ids``, ``opsd_advantages``, ``fit_turn_weights`` and
the four ``hpi_*`` arrays. No framework change is needed for any of them.

Every real turn is retained. Zero-mask padding aligns the rows to DP, and the
entire rollout is accumulated before one optimizer update.
"""

import logging
import math
import os
from copy import deepcopy
from typing import Any

import torch

from slime.utils.types import Sample

from . import config

logger = logging.getLogger(__name__)

__all__ = ["convert_samples_to_train_data"]


def _validate(sample: Sample) -> None:
    rlen = int(sample.response_length)
    for name, value in (
        ("loss_mask", sample.loss_mask),
        ("turn_ids", sample.turn_ids),
        ("opsd_advantages", sample.opsd_advantages),
        ("fit_turn_weights", sample.fit_turn_weights),
    ):
        if value is None:
            raise ValueError(f"v2 sample {sample.index} is missing {name}")
        if len(value) != rlen:
            raise ValueError(
                f"v2 sample {sample.index}: {name} length {len(value)} != response_length {rlen}"
            )


def _has_trainable_rows(sample: Sample) -> bool:
    """True when `sample` carries at least one trainable turn-row.

    NOT the same as "has tokens": a rollout whose generation came back empty is
    returned as the PROMPT with `response_length == 0` -- non-zero `tokens`, but an
    empty `turn_ids` table. Both `trajectory_weighting` and `advantage_diagnostics`
    index `turn_ids[0]`, so such a sample raised IndexError and killed the whole
    run (2026-10-06: 1.7B alfworld HSI cell, rc=1 at rollout 55 of 100).
    """
    turn_ids = getattr(sample, "turn_ids", None)
    return len(sample.tokens) > 0 and turn_ids is not None and len(turn_ids) > 0


def trajectory_weighting(samples: list) -> tuple[list[list[float]], float, dict[str, float]]:
    """Per-turn-row weights that give every TRAJECTORY the same total weight.

    THE BUG THIS FIXES. A trajectory is emitted as one row per turn, and every row
    carries its episode's GRPO advantage. Equal row weighting therefore makes a
    long trajectory shout louder than a short one: measured on a real 8B rollout
    the batch was 36 positive rows against 572 negative ones (15.9:1), and the
    cheap way to obey a gradient that is 94% "push these down" is to stop emitting
    coherent text. That is what destroyed the `</think>` structure and collapsed
    the run.

    The fix weights row `t` of trajectory `i` by its share of that trajectory's
    training tokens, `w = N_it / N_i`, so `sum_t w_it == 1` for every trajectory
    regardless of how many turns or tokens it took.

    Must run HERE, before the DP split and the micro-batch slicing: `N_i` is a sum
    over the trajectory's rows, and after the split no rank can see them all.

    Returns `(per_row_weights, scale, diagnostics)` where `scale` is the step
    constant the loss must multiply by -- see `loss.alfworld_v2_loss` for why it
    cannot be derived from a single micro-batch.
    """
    episode_of = config.TURN_ID_STRIDE
    row_tokens: list[float] = []
    row_episode: list[int] = []
    for sample in samples:
        first = sample.turn_ids[0]
        first = int(first.item()) if hasattr(first, "item") else int(first)
        row_episode.append(first // episode_of)
        row_tokens.append(float(len(sample.loss_mask)))

    per_episode_tokens: dict[int, float] = {}
    for episode, tokens in zip(row_episode, row_tokens, strict=True):
        per_episode_tokens[episode] = per_episode_tokens.get(episode, 0.0) + tokens

    n_trajectories = len(per_episode_tokens)
    total_tokens = sum(row_tokens)
    # N_global / (T*G): the framework will divide the accumulated gradient by
    # N_global, so the loss has to carry this to land on a per-trajectory mean.
    # Both sums are over exactly the rows that will train, which is why it is exact.
    scale = total_tokens / max(n_trajectories, 1)

    weights: list[list[float]] = []
    for episode, tokens in zip(row_episode, row_tokens, strict=True):
        n_i = per_episode_tokens[episode]
        w = tokens / n_i if n_i > 0 else 0.0
        weights.append([w] * int(tokens))

    # Assertion B of the fix spec: weights within one trajectory sum to 1, so a
    # trajectory's total optimizer weight cannot grow with its turn count.
    sums: dict[int, float] = {}
    for episode, row in zip(row_episode, weights, strict=True):
        sums[episode] = sums.get(episode, 0.0) + (row[0] if row else 0.0)
    worst = max((abs(v - 1.0) for v in sums.values()), default=0.0)

    diagnostics = {
        "n_trajectories": float(n_trajectories),
        "n_rows": float(len(samples)),
        "total_tokens": float(total_tokens),
        "scale": float(scale),
        "row_weight_sum_err": float(worst),
        "mean_rows_per_trajectory": float(len(samples)) / max(n_trajectories, 1),
    }
    return weights, scale, diagnostics


def outcome_composition(rows: list[tuple[float, int]]) -> dict[str, float]:
    """Positive/negative row and token counts, for the imbalance diagnostic.

    `rows` is `(advantage, token_count)` per turn-row.
    """
    pos_rows = sum(1 for a, _ in rows if a > 0)
    neg_rows = sum(1 for a, _ in rows if a < 0)
    zero_rows = sum(1 for a, _ in rows if a == 0)
    pos_tokens = sum(n for a, n in rows if a > 0)
    neg_tokens = sum(n for a, n in rows if a < 0)
    return {
        "rows_pos": float(pos_rows),
        "rows_neg": float(neg_rows),
        "rows_zero": float(zero_rows),
        "tokens_pos": float(pos_tokens),
        "tokens_neg": float(neg_tokens),
        "row_neg_over_pos": float(neg_rows) / max(pos_rows, 1),
        "token_neg_over_pos": float(neg_tokens) / max(pos_tokens, 1),
    }


def advantage_diagnostics(samples: list, row_weights: list[list[float]]) -> dict[str, float]:
    """Correctness checks on the advantage before it reaches the optimizer.

    Three sums over the same rows, which is the whole point of the fix:

    * `raw_row_adv_sum`   -- sum_r A_r. What equal-row weighting optimises. In a
      healthy-looking 8B rollout this is dominated by the failing turns simply
      because failures last longer, so it drifts far from 0.
    * `raw_token_adv_sum` -- sum_r A_r * N_r. Same bias, amplified by length.
    * `traj_normalized_adv_sum` -- sum_r A_r * w_r. Each trajectory now appears
      exactly once regardless of length, so within a group whose rewards have any
      variance this must sit near 0, which is what GRPO's group centring promises.
      This is assertion A of the fix spec, and it is the number that should be
      believed; the two raw sums above are expected to be badly off.

    Group identity comes from the episode id, which the rollout packs as
    `group_id * 1000 + episode_index` (rollout.py:build_group_samples).
    """
    row_adv: list[float] = []
    row_tokens: list[float] = []
    row_episode: list[int] = []
    for sample in samples:
        first = sample.turn_ids[0]
        first = int(first.item()) if hasattr(first, "item") else int(first)
        row_episode.append(first // config.TURN_ID_STRIDE)
        row_adv.append(float(sample.opsd_advantages[0]) if sample.opsd_advantages else 0.0)
        row_tokens.append(float(len(sample.loss_mask)))

    raw_row = sum(row_adv)
    raw_token = sum(a * n for a, n in zip(row_adv, row_tokens, strict=True))
    normalized = sum(
        a * (row[0] if row else 0.0) for a, row in zip(row_adv, row_weights, strict=True)
    )

    # Assertion A. A group with no reward variance has all-zero advantages by
    # construction, so the residual there is trivially 0 and carries no
    # information; only groups that actually had something to learn from count.
    per_group: dict[int, list[tuple[int, float]]] = {}
    for episode, advantage in zip(row_episode, row_adv, strict=True):
        per_group.setdefault(episode // 1000, []).append((episode, advantage))
    worst_group_residual = 0.0
    groups_with_variance = 0
    for members in per_group.values():
        by_episode: dict[int, float] = {}
        for episode, advantage in members:
            by_episode[episode] = advantage
        values = list(by_episode.values())
        if len(values) < 2 or max(values) == min(values):
            continue
        groups_with_variance += 1
        worst_group_residual = max(worst_group_residual, abs(sum(values)))

    return {
        "raw_row_adv_sum": raw_row,
        "raw_token_adv_sum": raw_token,
        "traj_normalized_adv_sum": normalized,
        "groups_with_reward_variance": float(groups_with_variance),
        "worst_group_adv_residual": worst_group_residual,
    }


def convert_samples_to_train_data(args, samples) -> dict[str, Any]:
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]

    # Drop samples with no turn-rows BEFORE the GBS trim below, so the
    # "multiple of global_batch_size" invariant the trim restores still holds.
    # See `_has_trainable_rows` for why "has tokens" is not sufficient.
    num_before = len(samples)
    samples = [s for s in samples if _has_trainable_rows(s)]
    if len(samples) < num_before:
        logger.warning(
            "dropped %d sample(s) with no trainable turn-rows "
            "(zero tokens, or a generated-empty response)",
            num_before - len(samples),
        )
    if not samples:
        raise RuntimeError("v2 rollout produced no usable samples")

    for sample in samples:
        _validate(sample)
        if sample.metadata.get("hsi_teacher_view") and sample.metadata.get("srlvr_v4") is None:
            # A row that HAS a teacher view must have one candidate entry per
            # generated token: a short table would silently shift the credit onto
            # the wrong positions rather than fail. SRLVR-v4 does not consume
            # candidates: its transport is the per-turn span in
            # `hpi_user_lengths`, so the table is deliberately absent there.
            candidates = sample.metadata.get("hsi_candidates") or []
            if len(candidates) != int(sample.response_length):
                raise ValueError(
                    f"v2 sample {sample.index}: {len(candidates)} candidate rows for "
                    f"response_length {sample.response_length}"
                )

    expected = getattr(args, "rollout_batch_size", 0) * getattr(args, "n_samples_per_prompt", 1)
    episodes = {int(s.turn_ids[0]) // config.TURN_ID_STRIDE for s in samples}
    if expected and len(episodes) != expected:
        raise RuntimeError(f"ALFWorld batch lost trajectories: {len(episodes)} != {expected}")
    expected = expected or len(episodes)

    # ---- advantage estimator -------------------------------------------------
    # `group` (default) is GRPO: the rollout already centred each group of
    # trajectories and divided by that group's std, so nothing happens here.
    #
    # `batch` is REINFORCE++: there is no group structure at all -- the baseline
    # and the whitening are computed over EVERY trajectory in the batch. With
    # 4 task slots x G=8 that pools 32 trajectories instead of 8, so a task whose
    # own 8 samples all failed no longer contributes a zero advantage; it is
    # judged against the whole batch. Everything else (clipped surrogate, initial
    # model KL, trajectory-balanced reduction) stays byte-identical to the GRPO
    # arm, so the only variable between the two arms is the advantage estimator.
    advantage_mode = os.environ.get("ALFWORLD_V2_ADVANTAGE_MODE", "group").lower()
    if advantage_mode not in ("group", "batch"):
        raise ValueError(
            f"ALFWORLD_V2_ADVANTAGE_MODE must be 'group' or 'batch', got {advantage_mode!r}"
        )
    if advantage_mode == "batch" and config.gigpo_enabled():
        raise ValueError(
            "GiGPO emits per-turn advantages; ALFWORLD_V2_ADVANTAGE_MODE=batch would "
            "overwrite them with one batch-level scalar and silently reduce the arm to "
            "REINFORCE++"
        )
    if advantage_mode == "batch":
        per_traj_reward: dict[int, float] = {}
        for sample in samples:
            episode = int(sample.turn_ids[0]) // config.TURN_ID_STRIDE
            per_traj_reward.setdefault(episode, float(sample.reward or 0.0))
        values = [per_traj_reward[e] for e in sorted(per_traj_reward)]
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n) if n else 0.0
        logger.info(
            "[v2-adv] REINFORCE++ batch-level baseline: trajectories=%d mean=%.4f "
            "std=%.4f (group centring disabled)",
            n, mean, std,
        )
        for sample in samples:
            episode = int(sample.turn_ids[0]) // config.TURN_ID_STRIDE
            scalar = 0.0 if std == 0.0 else (per_traj_reward[episode] - mean) / std
            sample.opsd_advantages = [float(scalar)] * int(sample.response_length)

    # Count all real rows before DP padding.
    row_weights, traj_scale, weight_diag = trajectory_weighting(samples)

    # Outcome composition, for the imbalance diagnostic the collapse was found
    # through: a healthy-looking rollout can still be 16:1 negative by row.
    composition = outcome_composition(
        [
            (float(s.opsd_advantages[0]) if s.opsd_advantages else 0.0, len(s.loss_mask))
            for s in samples
        ]
    )
    logger.info(
        "[v2-traj] rows=%d trajectories=%d tokens=%d scale=%.1f "
        "w_sum_err=%.2e | rows +/%d -/%d 0/%d  ratio -/+ rows=%.2f tokens=%.2f",
        int(weight_diag["n_rows"]),
        int(weight_diag["n_trajectories"]),
        int(weight_diag["total_tokens"]),
        traj_scale,
        weight_diag["row_weight_sum_err"],
        int(composition["rows_pos"]),
        int(composition["rows_neg"]),
        int(composition["rows_zero"]),
        composition["row_neg_over_pos"],
        composition["token_neg_over_pos"],
    )

    adv_diag = advantage_diagnostics(samples, row_weights)
    logger.info(
        "[v2-traj] adv sums: raw_row=%.3f raw_token=%.0f traj_normalized=%.3f "
        "(groups_with_variance=%d worst_group_residual=%.2e)",
        adv_diag["raw_row_adv_sum"],
        adv_diag["raw_token_adv_sum"],
        adv_diag["traj_normalized_adv_sum"],
        int(adv_diag["groups_with_reward_variance"]),
        adv_diag["worst_group_adv_residual"],
    )
    if config.gigpo_enabled():
        # `advantage_diagnostics` was written for a constant-per-trajectory
        # advantage. Under GiGPO the step term varies by turn, so the residual
        # above describes the LAST row of each trajectory and is not a centering
        # assertion for the step term. The anchor-group centring is asserted
        # inside compute_gigpo_advantages instead.
        logger.info(
            "[v2-traj] GiGPO per-turn advantages active: worst_group_adv_residual "
            "is diagnostic only (last row per trajectory), not a centering check"
        )

    train_data: dict[str, Any] = {
        "tokens": [list(s.tokens) for s in samples],
        "response_lengths": [int(s.response_length) for s in samples],
        "rewards": [0.0 if s.reward is None else float(s.reward) for s in samples],
        "raw_reward": [0.0 if s.reward is None else float(s.reward) for s in samples],
        "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in samples],
        "sample_indices": [s.index for s in samples],
        "loss_masks": [list(s.loss_mask) for s in samples],
        "turn_ids": [list(s.turn_ids) for s in samples],
        "opsd_advantages": [list(s.opsd_advantages) for s in samples],
        # Trajectory-balanced per-token weight (sums to 1 within each trajectory).
        # Replaces the rollout's 1.0 placeholder; see trajectory_weighting().
        "fit_turn_weights": row_weights,
        # Step constant N_global/(T*G) that the loss must carry, because the
        # framework divides by N_global and a single micro-batch cannot see it.
        "traj_norm_scale": [[traj_scale] * int(s.response_length) for s in samples],
    }
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [list(s.rollout_log_probs) for s in samples]

    if any(s.metadata.get("hsi_teacher_view") for s in samples):
        # Candidate-Centered HSI rides in the four `hpi_*` keys, which are already
        # whitelisted by slime's `_split_train_data_by_dp` -- adding a new key would
        # mean editing shared source this node must not touch.
        #
        #   hpi_views[s]          = [teacher_view_ids]   (ONE view: true future)
        #   hpi_prompt_lengths[s] = [prompt_length]
        #   hpi_target_lengths[s] = [target_length]
        #   hpi_user_lengths[s]   = flat integer candidate payload
        #
        # Every sample gets an entry (empty when its turn has no future or the leak
        # guard dropped it) so the per-sample partition in the split stays aligned.
        from .hsi_shared import encode_candidates

        views, prompt_lengths, target_lengths, payloads = [], [], [], []
        for sample in samples:
            view = sample.metadata.get("hsi_teacher_view")
            if view:
                ids, plen, tlen = view
                views.append([list(ids)])
                prompt_lengths.append([int(plen)])
                target_lengths.append([int(tlen)])
                v4 = sample.metadata.get("srlvr_v4")
                if v4 is not None:
                    # SRLVR-v4: (span_start, span_len) of this turn's generated
                    # tokens inside the shared trajectory continuation. The loss
                    # slices the teacher hidden states at prompt_len + span_start - 1.
                    # (-1, 0) marks an overlong teacher view: row kept, loss zero.
                    if v4.get("skipped_overlong"):
                        payloads.append([-1, 0])
                    else:
                        payloads.append([int(v4["span_start"]), int(v4["span_len"])])
                else:
                    payloads.append(encode_candidates(sample.metadata.get("hsi_candidates") or []))
            else:
                views.append([])
                prompt_lengths.append([])
                target_lengths.append([])
                payloads.append([])
        train_data["hpi_views"] = views
        train_data["hpi_prompt_lengths"] = prompt_lengths
        train_data["hpi_target_lengths"] = target_lengths
        train_data["hpi_user_lengths"] = payloads
    if samples:
        world = getattr(args, "actor_num_gpus_per_node", 1) * getattr(args, "actor_num_nodes", 1)
        parallel = (getattr(args, "tensor_model_parallel_size", 1)
                    * getattr(args, "pipeline_model_parallel_size", 1)
                    * getattr(args, "context_parallel_size", 1))
        dp = world // parallel
        real_rows = len(samples)
        padding = (-real_rows) % dp
        source = min(range(real_rows), key=lambda i: len(train_data["tokens"][i]))
        response_len = train_data["response_lengths"][source]
        for _ in range(padding):
            for key, values in train_data.items():
                if key in ("loss_masks", "opsd_advantages", "fit_turn_weights", "traj_norm_scale"):
                    values.append([0.0] * response_len)
                elif key in ("rewards", "raw_reward", "truncated"):
                    values.append(0)
                elif key == "sample_indices":
                    values.append(-1)
                else:
                    values.append(deepcopy(values[source]))
        # finalize_model_grads counts each all-zero padding mask as one token.
        # Compensate that exact denominator; padding has zero reducer weight.
        scale = (weight_diag["total_tokens"] + padding) / expected
        train_data["traj_norm_scale"] = [
            [scale] * int(n) for n in train_data["response_lengths"]
        ]
        train_data["dynamic_global_batch_size"] = real_rows + padding
        logger.info(
            "[alfworld-batch] trajectories=%d real_rows=%d padding_rows=%d "
            "dynamic_global_batch_size=%d optimizer_steps=1 norm_scale=%.6f",
            expected, real_rows, padding, real_rows + padding, scale,
        )
    return train_data


