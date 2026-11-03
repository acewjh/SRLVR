"""``--custom-convert-samples-to-train-data-path`` for ALFWorld.

Forwards exactly the keys slime's ``_split_train_data_by_dp`` already whitelists
(``hpi_views`` / ``hpi_prompt_lengths`` / ``hpi_target_lengths`` / ``turn_ids``),
so no change to the vendored slime tree is required.

Layout contract, asserted here rather than assumed:

* ``len(loss_mask) == response_length`` (slime asserts this too; failing early
  with a clearer message saves a confusing downstream crash);
* ``loss_mask == 1`` exactly on the turn id spans and ``0`` elsewhere -- i.e.
  ``turn_ids >= 0 iff loss_mask == 1``;
* every HPI view's target span equals the corresponding turn's generated token
  ids, token-for-token. This is the CPU-testable half of the "teacher scores the
  SAME tokens" guarantee; the loss re-checks it against the engine's batch.
"""

import logging
from typing import Any

import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["convert_samples_to_train_data", "validate_layout"]


def validate_layout(sample: Sample) -> None:
    """Assert the sequence/loss/turn-id contract for one sample."""
    rlen = sample.response_length
    loss_mask = sample.loss_mask
    if loss_mask is None:
        raise ValueError(f"sample {sample.index}: loss_mask is required for the ALFWorld harness")
    if len(loss_mask) != rlen:
        raise ValueError(
            f"sample {sample.index}: loss_mask length {len(loss_mask)} != response_length {rlen}"
        )

    turn_ids = sample.turn_ids
    if turn_ids is None or len(turn_ids) != rlen:
        raise ValueError(
            f"sample {sample.index}: turn_ids must be present and match response_length "
            f"(got {None if turn_ids is None else len(turn_ids)} vs {rlen})"
        )

    for position, (masked, turn) in enumerate(zip(loss_mask, turn_ids, strict=True)):
        if (turn >= 0) != bool(masked):
            raise ValueError(
                f"sample {sample.index}: loss_mask/turn_ids disagree at response position "
                f"{position} (loss_mask={masked}, turn_id={turn})"
            )

    views = sample.metadata.get("hpi_views")
    if views is None:
        return
    plens = sample.metadata["hpi_prompt_lengths"]
    tlens = sample.metadata["hpi_target_lengths"]
    spans = sample.metadata.get("turn_spans") or []
    if not (len(views) == len(plens) == len(tlens)):
        raise ValueError(
            f"sample {sample.index}: hpi view arrays disagree in length "
            f"({len(views)}, {len(plens)}, {len(tlens)})"
        )
    if len(views) % 2 != 0:
        raise ValueError(f"sample {sample.index}: hpi views must come in (full, control) pairs")
    if len(views) // 2 != len(spans):
        raise ValueError(
            f"sample {sample.index}: {len(views) // 2} view pairs for {len(spans)} turns"
        )

    response_tokens = sample.tokens[len(sample.tokens) - rlen :]
    for pair in range(len(views) // 2):
        start, end = spans[pair]
        expected = list(response_tokens[start:end])
        for which in (0, 1):
            index = 2 * pair + which
            view, plen, tlen = views[index], plens[index], tlens[index]
            if len(view) != plen + tlen:
                raise ValueError(
                    f"sample {sample.index}: view {index} length {len(view)} != "
                    f"prompt {plen} + target {tlen}"
                )
            if tlen != len(expected):
                raise ValueError(
                    f"sample {sample.index}: view {index} target length {tlen} != turn span "
                    f"length {len(expected)}"
                )
            if list(view[plen:]) != expected:
                raise AssertionError(
                    f"sample {sample.index}: view {index} target tokens differ from the "
                    "student's generated tokens for that turn -- teacher/student alignment violated"
                )


def convert_samples_to_train_data(args, samples) -> dict[str, Any]:
    """Custom convert entry point (see ``--custom-convert-samples-to-train-data-path``)."""
    if samples and isinstance(samples[0], list):
        samples = [sample for group in samples for sample in group]

    num_before = len(samples)
    samples = [s for s in samples if len(s.tokens) > 0]
    if len(samples) < num_before:
        logger.warning("Dropped %d zero-token sample(s)", num_before - len(samples))
    if not samples:
        raise RuntimeError("ALFWorld rollout produced no usable samples")

    for sample in samples:
        validate_layout(sample)

    gbs = args.global_batch_size
    if len(samples) % gbs != 0:
        keep = (len(samples) // gbs) * gbs
        if keep == 0:
            raise RuntimeError(
                f"Only {len(samples)} usable samples, fewer than global_batch_size {gbs}; "
                "cannot form a training step."
            )
        logger.warning(
            "Trimming %d usable samples to %d (multiple of GBS %d)", len(samples), keep, gbs
        )
        samples = samples[:keep]

    train_data: dict[str, Any] = {
        "tokens": [list(s.tokens) for s in samples],
        "response_lengths": [s.response_length for s in samples],
        "rewards": [0.0 if s.reward is None else float(s.reward) for s in samples],
        "raw_reward": [0.0 if s.reward is None else float(s.reward) for s in samples],
        "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in samples],
        "sample_indices": [s.index for s in samples],
        "loss_masks": [list(s.loss_mask) for s in samples],
        "turn_ids": [list(s.turn_ids) for s in samples],
    }
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [list(s.rollout_log_probs) for s in samples]

    if samples[0].metadata.get("hpi_views") is not None:
        missing = [i for i, s in enumerate(samples) if s.metadata.get("hpi_views") is None]
        if missing:
            raise ValueError(
                f"hpi_views present on only {len(samples) - len(missing)}/{len(samples)} samples "
                f"(missing at indices {missing[:5]}); a mixed batch cannot be scored consistently"
            )
        train_data["hpi_views"] = [s.metadata["hpi_views"] for s in samples]
        train_data["hpi_prompt_lengths"] = [s.metadata["hpi_prompt_lengths"] for s in samples]
        train_data["hpi_target_lengths"] = [s.metadata["hpi_target_lengths"] for s in samples]

    return train_data
