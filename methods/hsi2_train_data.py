"""Sample -> train-data conversion for the HSI v2 (JSD) estimator.

Thin wrapper over ``tasks.alfworld.train_data``: the load-bearing pieces
(``trajectory_weighting`` -- the per-trajectory weights that keep a long
trajectory from shouting louder than a short one -- and the ``traj_norm_scale``
constant the custom loss must carry) are imported, not re-implemented, so they
cannot drift from the arm the loss was validated against.

The only difference from the shared converter is what it *refuses*: this
estimator has no candidate set, so a row is required to carry its hindsight
**view** (``hsi_teacher_view``) instead of a top-K candidate table. Keeping this
in its own module means the shared file -- which the ALFWorld campaign on
another pod is still running -- stays untouched (see
``shared-nfs-multi-agent-repo``).
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.types import Sample

from tasks.alfworld.train_data import (  # re-exported for callers/tests
    advantage_diagnostics,
    outcome_composition,
    trajectory_weighting,
)

logger = logging.getLogger(__name__)

__all__ = ["convert_samples_to_train_data"]


def _validate(sample) -> None:
    rlen = int(sample.response_length)
    for name, value in (
        ("loss_mask", sample.loss_mask),
        ("turn_ids", sample.turn_ids),
        ("opsd_advantages", sample.opsd_advantages),
        ("fit_turn_weights", sample.fit_turn_weights),
    ):
        if value is None:
            raise ValueError(f"hsi2 sample {sample.index} is missing {name}")
        if len(value) != rlen:
            raise ValueError(
                f"hsi2 sample {sample.index}: {name} length {len(value)} != response_length {rlen}"
            )


def convert_samples_to_train_data(args, samples) -> dict[str, Any]:
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]

    num_before = len(samples)
    samples = [s for s in samples if len(s.tokens) > 0]
    if len(samples) < num_before:
        logger.warning("dropped %d zero-token sample(s)", num_before - len(samples))
    if not samples:
        raise RuntimeError("hsi2 rollout produced no usable samples")

    for sample in samples:
        _validate(sample)

    gbs = int(args.global_batch_size)
    if len(samples) % gbs != 0:
        keep = (len(samples) // gbs) * gbs
        if keep == 0:
            raise RuntimeError(
                f"only {len(samples)} usable rows for global_batch_size {gbs}; "
                "increase --rollout-batch-size or lower --global-batch-size"
            )
        logger.warning(
            "trimming %d -> %d rows (multiple of GBS %d); trailing turns dropped",
            len(samples),
            keep,
            gbs,
        )
        samples = samples[:keep]

    # Computed after the trim, so `N_i` counts only rows that will actually train.
    row_weights, traj_scale, weight_diag = trajectory_weighting(samples)

    composition = outcome_composition(
        [
            (float(s.opsd_advantages[0]) if s.opsd_advantages else 0.0, len(s.loss_mask))
            for s in samples
        ]
    )
    logger.info(
        "[hsi2-traj] rows=%d trajectories=%d tokens=%d scale=%.1f w_sum_err=%.2e | "
        "rows +/%d -/%d 0/%d  ratio -/+ rows=%.2f tokens=%.2f",
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
        "fit_turn_weights": row_weights,
        "traj_norm_scale": [[traj_scale] * int(s.response_length) for s in samples],
    }
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [list(s.rollout_log_probs) for s in samples]

    # The hindsight view rides in the four already-whitelisted `hpi_*` keys, so no
    # framework change is needed. `hpi_user_lengths` is emitted EMPTY: this
    # estimator has no candidate payload (the WHERE is a full-vocab JSD, not a
    # top-K projection), and the key is kept present only so the per-sample
    # partition in the DP split stays aligned for consumers that expect it.
    views, prompt_lengths, target_lengths, payloads = [], [], [], []
    for sample in samples:
        view = sample.metadata.get("hsi_teacher_view")
        if view:
            ids, plen, tlen = view
            views.append([list(ids)])
            prompt_lengths.append([int(plen)])
            target_lengths.append([int(tlen)])
        else:
            views.append([])
            prompt_lengths.append([])
            target_lengths.append([])
        payloads.append([])
    train_data["hpi_views"] = views
    train_data["hpi_prompt_lengths"] = prompt_lengths
    train_data["hpi_target_lengths"] = target_lengths
    train_data["hpi_user_lengths"] = payloads
    return train_data
