"""``--custom-loss-function-path`` for the ALFWorld HPI arms.

The objective is slime's **stock GRPO policy loss**, re-weighted per token:

    L = sum_of_sample_mean( w_t * pg_loss_t )

where ``pg_loss_t`` is ``compute_policy_loss(ppo_kl, advantages, eps_clip,
eps_clip_high)`` -- the exact function the stock ``policy_loss_function`` calls --
and ``w_t`` comes from the hindsight privileged teacher.

Both arms share this file:
  * ``--alfworld-hpi-mode none``        -> w == 1 exactly (must reduce bit-for-bit
    to stock GRPO; asserted in the smoke gate, not merely intended)
  * ``--alfworld-hpi-mode outcome_only``-> w from the outcome-conditioned teacher
  * ``--alfworld-hpi-mode future_outcome`` -> w from the outcome+future teacher

so ``outcome_only`` vs ``future_outcome`` differ in exactly one place: whether the
privileged context contains the delayed future interaction trace.

Hindsight score
---------------
For each turn ``t`` the rollout stored two views that share an identical scaffold:

    full     = h_t + P_t            (privileged block per mode)
    control  = h_t                  (the student's own context, verbatim)

Both are teacher-forced on the *same* ``y_t`` token ids, and only those sampled
token log-probs are read::

    e_t = log q_teacher(y_t | h_t, P_t) - log q_teacher(y_t | h_t)

The control view is the frozen initial policy on the student-visible context --
i.e. the same "reference" the repo's HPI uses, and the reason a plain
"presence of any hindsight" bias cancels instead of being counted as signal.

Weight (inherited unchanged from ``math_opsd.hpi_v15``)
-------------------------------------------------------
    u_t     = tanh(|e_t| / scale)          scale frozen at calibration
    raw_w_t = 1 + alpha * u_t
    w       = raw_w / mean(raw_w)          normalised per trajectory
    w       = clamp(w, 0.70, 1.30)
    w       = stopgrad(w)

``alpha = 0`` gives ``w == 1`` exactly, so this file degenerates to stock GRPO.
The weight is built from ``|e_t|`` and never from the sign of ``e_t`` or from R:
direction is the policy gradient's job, the hindsight only re-allocates credit.
"""

import logging
import os
from argparse import Namespace
from collections.abc import Callable

import torch

from . import config
from .privilege import MODE_NONE

logger = logging.getLogger(__name__)

__all__ = ["alfworld_hpi_loss", "compute_hindsight_weights", "WEIGHT_MIN", "WEIGHT_MAX"]

WEIGHT_MIN = 0.70
WEIGHT_MAX = 1.30
SCALE_FLOOR = 0.01
DEFAULT_ALPHA = 0.5
DEFAULT_SCALE = 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Hindsight weight
# ---------------------------------------------------------------------------


def compute_hindsight_weights(
    e_per_turn: list[list[torch.Tensor]],
    turn_ids_per_sample: list[list[int]],
    response_lengths: list[int],
    alpha: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-response-token weights, concatenated across the batch.

    ``e_per_turn[i][k]`` is the fp32 ``|e|`` vector for turn ``k`` of sample ``i``
    (empty when that turn was unscored). Positions not covered by any scored turn
    keep raw weight 1.
    """
    scale = max(float(scale), SCALE_FLOOR)
    parts: list[torch.Tensor] = []
    all_abs_e: list[torch.Tensor] = []
    scored_turns = 0
    skipped_turns = 0

    for sample_index, (turn_ids, rlen) in enumerate(zip(turn_ids_per_sample, response_lengths, strict=True)):
        raw = torch.ones(int(rlen), dtype=torch.float32)
        spans: dict[int, tuple[int, int]] = {}
        # `int(turn)` is load-bearing: a tensor key would not dedupe by value.
        for position, turn in enumerate(turn_ids):
            turn = int(turn)
            if turn < 0:
                continue
            start, end = spans.get(turn, (position, position))
            spans[turn] = (min(start, position), max(end, position) + 1)

        for turn, (start, end) in spans.items():
            turn_abs_e = e_per_turn[sample_index][turn] if turn < len(e_per_turn[sample_index]) else None
            if turn_abs_e is None or turn_abs_e.numel() != (end - start):
                skipped_turns += 1
                continue
            raw[start:end] = 1.0 + alpha * torch.tanh(turn_abs_e.float() / scale)
            all_abs_e.append(turn_abs_e.float().detach())
            scored_turns += 1

        mean = raw.mean()
        if not torch.isfinite(mean) or mean <= 0:
            parts.append(torch.ones(rlen, dtype=torch.float32))
            continue
        # stopgrad: the weight allocates credit, it is never a gradient path.
        parts.append((raw / mean).clamp(WEIGHT_MIN, WEIGHT_MAX).detach())

    weights = torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.float32)
    stats = {
        "scored_turns": float(scored_turns),
        "skipped_turns": float(skipped_turns),
        "mean_weight": float(weights.mean()) if weights.numel() else 1.0,
    }
    if all_abs_e:
        cat = torch.cat(all_abs_e)
        quantiles = torch.quantile(cat, torch.tensor([0.50, 0.75, 0.90]))
        stats["abs_e_p50"] = float(quantiles[0])
        stats["abs_e_p75"] = float(quantiles[1])
        stats["abs_e_p90"] = float(quantiles[2])
    else:
        stats["abs_e_p50"] = stats["abs_e_p75"] = stats["abs_e_p90"] = 0.0
    return weights, stats


#: Frozen-teacher score cache, keyed on the view content. See
#: `_score_views_with_teacher` for why this is safe and why it matters.
_TEACHER_CACHE: dict = {}
_TEACHER_CACHE_MAX = int(os.environ.get("ALFWORLD_HPI_TEACHER_CACHE", "256"))


def _teacher_offload_enabled() -> bool:
    return os.environ.get("ALFWORLD_HPI_TEACHER_OFFLOAD", "1").strip().lower() not in {"0", "false", "no"}


def _park_teacher(teacher) -> None:
    """Move the frozen teacher off the accelerator and release its cached blocks.

    Best-effort: a failure here must not take the training step down, because the
    teacher is only a scoring device and the next call re-materialises it anyway.
    """
    try:
        # Deliberately NO empty_cache(): moving the weights off the device already
        # returns their blocks to PyTorch's NPU pool, which the backward then reuses.
        # Calling empty_cache() forces them back to the driver and re-requests them
        # on the next microbatch -- pure thrash on a fragmented 25 GB pool.
        teacher.to("cpu")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[alfworld] teacher offload failed (continuing on-device): %s", exc)


def _score_views_with_teacher(args: Namespace, views, prompt_lengths, target_lengths) -> list[torch.Tensor]:
    """Teacher-force every view on the frozen initial policy, returning |e| per turn.

    Reuses ``math_opsd`` verbatim: the same frozen-HF teacher loader and the same
    sampled-token-only scorer, so the teacher is the *initial policy* the actor
    started from and the scoring path is the one already validated in this repo.

    The teacher is parked on CPU for the rest of the step (see ``_park_teacher``):
    it is a second 1.7B resident on the same NPU, and the actor's BACKWARD -- which
    runs after this function returns -- is what actually OOMed. Scoring finishes
    before the backward starts, so moving it off costs a transfer but frees
    ~3.4 GB exactly where it was needed. Without this the HPI arms sit ~3.4 GB
    below GRPO's ceiling and OOM at sequence lengths GRPO handles fine.
    """
    from opsd_core.scoring import score_view_logprobs
    from methods.opsd_loss import _get_teacher

    teacher = _get_teacher(args)
    device = next(teacher.parameters()).device
    offload = _teacher_offload_enabled() and str(device).startswith("npu")
    if offload:
        teacher.to(device)  # no-op if already resident

    flat = [(list(v), int(p), int(t)) for v, p, t in zip(views, prompt_lengths, target_lengths, strict=True)]

    # Memoise on the view content. `--recompute-loss-function` re-executes the
    # whole loss during the backward, so without this the teacher re-scores every
    # view a second time in the same step -- measured as the dominant cost of an
    # HPI step. The cache is semantically safe because the teacher is FROZEN: the
    # same view always yields the same log-probs. Bounded so a long run cannot
    # grow without limit.
    # The view token-ids are lists, and a tuple of lists is still unhashable,
    # so flatten each view to a tuple for the key.
    key = tuple((tuple(ids), plen, tlen) for ids, plen, tlen in flat)
    hit = _TEACHER_CACHE.get(key)
    if hit is not None:
        return hit

    try:
        logprobs = score_view_logprobs(teacher, flat)
    finally:
        if offload:
            _park_teacher(teacher)
    if len(_TEACHER_CACHE) >= _TEACHER_CACHE_MAX:
        _TEACHER_CACHE.clear()
    _TEACHER_CACHE[key] = logprobs

    per_turn: list[list[torch.Tensor]] = []
    cursor = 0
    n_turns = len(flat) // 2
    for _turn in range(n_turns):
        full_lp, ctrl_lp = logprobs[cursor], logprobs[cursor + 1]
        cursor += 2
        if full_lp.numel() == 0 or ctrl_lp.numel() == 0 or full_lp.numel() != ctrl_lp.numel():
            per_turn.append(torch.zeros(0, dtype=torch.float32))
        else:
            per_turn.append((full_lp - ctrl_lp).abs())
    return per_turn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def alfworld_hpi_loss(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """slime custom loss (see module docstring)."""
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    from slime.utils.ppo_utils import compute_policy_loss

    from methods.opsd_loss import slice_response_logits

    args._current_max_seq_lens = batch.get("max_seq_lens", [0])[0] if batch.get("max_seq_lens") else 0
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    unconcat_tokens = batch["unconcat_tokens"]
    loss_masks = batch["loss_masks"]

    mode = config.privilege_mode(args)

    # ---- student log-probs: identical to the stock GRPO path ----------------
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=getattr(args, "entropy_coef", 0.0) != 0,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    log_probs = log_probs_and_entropy["log_probs"]

    advantages = torch.cat(batch["advantages"], dim=0)
    if getattr(args, "use_rollout_logprobs", False):
        if "rollout_log_probs" not in batch:
            raise ValueError(
                "use_rollout_logprobs is set but the ALFWorld rollout did not provide "
                "rollout_log_probs; the PPO ratio would be undefined."
            )
        old_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
    else:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
    log_probs = torch.cat(log_probs, dim=0)

    ppo_kl = old_log_probs - log_probs
    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl, advantages, args.eps_clip, args.eps_clip_high
    )

    # ---- hindsight re-weighting --------------------------------------------
    stats: dict[str, float] = {}
    if mode == MODE_NONE:
        weights = torch.ones(pg_loss.numel(), dtype=torch.float32, device=pg_loss.device)
        stats["hpi/scored_turns"] = 0.0
    else:
        views = batch.get("hpi_views")
        if views is None:
            raise ValueError(
                f"--alfworld-hpi-mode={mode} requires hpi_views in the batch; the rollout "
                "did not attach them (was it run with the same mode?)."
            )
        flat_views, flat_plens, flat_tlens = [], [], []
        # Coerce to plain ints FIRST.
        #
        # `batch["turn_ids"]` entries are torch tensors on the training side.
        # Iterating one yields 0-dim tensor scalars, and using a tensor as a dict
        # key does not dedupe by value -- every position becomes a distinct key.
        # That turned a 5-turn episode into "1613 turns" and killed the run.
        turn_ids_per_sample = [
            [int(t) for t in sample_turn_ids] for sample_turn_ids in batch["turn_ids"]
        ]
        for index, sample_turn_ids in enumerate(turn_ids_per_sample):
            if len(sample_turn_ids) != int(response_lengths[index]):
                raise AssertionError(
                    f"ALFWorld HPI: turn_ids for sample {index} has {len(sample_turn_ids)} "
                    f"entries but response_length is {int(response_lengths[index])}"
                )
        for sample_views, sample_plens, sample_tlens in zip(
            views, batch["hpi_prompt_lengths"], batch["hpi_target_lengths"], strict=True
        ):
            flat_views.extend(sample_views)
            flat_plens.extend(sample_plens)
            flat_tlens.extend(sample_tlens)

        # Alignment gate.
        #
        # It deliberately does NOT compare against `unconcat_tokens`. That vector
        # belongs to the training batch, where the pipeline is free to pad / slice
        # it, so `tokens[-response_length:]` is not guaranteed to still be the
        # rollout's response -- which is exactly how the first HPI run died: the
        # rollout dump verified equal (checked offline), yet the in-loss comparison
        # against the batch tokens failed.
        #
        # What actually has to hold, and is checkable from the payload alone:
        #   * full view and control view teacher-force the SAME target tokens
        #     (they differ only by the privileged block);
        #   * that target has the length of the turn's span in `turn_ids`, and the
        #     view is exactly prompt + target.
        # A mis-specified span or a swapped target breaks one of these.
        cursor = 0
        for sample_index, turn_ids in enumerate(turn_ids_per_sample):
            spans: dict[int, tuple[int, int]] = {}
            for position, turn in enumerate(turn_ids):
                if turn < 0:
                    continue
                start, end = spans.get(turn, (position, position))
                spans[turn] = (min(start, position), max(end, position) + 1)

            if len(spans) != (len(views[sample_index]) // 2):
                raise AssertionError(
                    f"ALFWorld HPI alignment: sample {sample_index} has {len(spans)} turns in "
                    f"turn_ids but {len(views[sample_index]) // 2} (full, control) view pairs"
                )

            for turn in sorted(spans):
                span_len = spans[turn][1] - spans[turn][0]
                full = flat_views[cursor + 2 * turn]
                control = flat_views[cursor + 2 * turn + 1]
                full_plen = flat_plens[cursor + 2 * turn]
                ctrl_plen = flat_plens[cursor + 2 * turn + 1]
                full_tlen = flat_tlens[cursor + 2 * turn]
                ctrl_tlen = flat_tlens[cursor + 2 * turn + 1]

                if not (full_tlen == ctrl_tlen == span_len):
                    raise AssertionError(
                        f"ALFWorld HPI alignment violated at sample {sample_index} turn {turn}: "
                        f"target lengths full={full_tlen} control={ctrl_tlen} do not all equal the "
                        f"turn_ids span {span_len}"
                    )
                if list(full[full_plen:]) != list(control[ctrl_plen:]):
                    raise AssertionError(
                        f"ALFWorld HPI alignment violated at sample {sample_index} turn {turn}: the "
                        "full and control views teacher-force DIFFERENT target tokens; they must "
                        "differ only by the privileged block."
                    )
                if len(full) != full_plen + full_tlen or len(control) != ctrl_plen + ctrl_tlen:
                    raise AssertionError(
                        f"ALFWorld HPI alignment violated at sample {sample_index} turn {turn}: "
                        "a view is not exactly prompt + target"
                    )
            cursor += 2 * len(spans)

        per_turn_flat = _score_views_with_teacher(args, flat_views, flat_plens, flat_tlens)

        # regroup the flat |e| list back per sample
        e_per_turn: list[list[torch.Tensor]] = []
        cursor = 0
        for turn_ids in turn_ids_per_sample:
            n_turns = max(int(t) for t in turn_ids) + 1 if any(int(t) >= 0 for t in turn_ids) else 0
            e_per_turn.append(per_turn_flat[cursor : cursor + n_turns])
            cursor += n_turns

        weights, stats = compute_hindsight_weights(
            e_per_turn,
            turn_ids_per_sample,
            response_lengths,
            config.hpi_alpha(args),
            config.hpi_scale(args),
        )
        weights = weights.to(pg_loss.device, dtype=pg_loss.dtype)
        if weights.numel() != pg_loss.numel():
            raise ValueError(
                f"hindsight weight length {weights.numel()} != response token count {pg_loss.numel()}"
            )

    per_token = pg_loss * weights
    loss = sum_of_sample_mean(per_token)
    if per_token.numel() == 0:
        loss = loss + 0 * logits.sum()

    metrics = {
        "loss": loss.clone().detach(),
        "pg_loss": (pg_loss.detach().mean() if pg_loss.numel() else torch.zeros((), device=logits.device)),
        "pg_clipfrac": (
            pg_clipfrac.detach().mean() if pg_clipfrac.numel() else torch.zeros((), device=logits.device)
        ),
        "ppo_kl": (ppo_kl.detach().mean() if ppo_kl.numel() else torch.zeros((), device=logits.device)),
        "hpi/mean_weight": torch.tensor(float(stats.get("mean_weight", 1.0)), device=logits.device),
        "hpi/scored_turns": torch.tensor(float(stats.get("scored_turns", 0.0)), device=logits.device),
        "hpi/skipped_turns": torch.tensor(float(stats.get("skipped_turns", 0.0)), device=logits.device),
        "hpi/abs_e_p50": torch.tensor(float(stats.get("abs_e_p50", 0.0)), device=logits.device),
        "hpi/abs_e_p90": torch.tensor(float(stats.get("abs_e_p90", 0.0)), device=logits.device),
    }
    return loss, metrics
