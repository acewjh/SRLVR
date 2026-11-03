"""v2 loss -- trajectory-level GRPO, optionally HPI-reweighted.

    L_i        = (1 / M_i) * sum_t sum_j  w_it * loss_grpo(i, t, j)
    L_GRPO/HPI = mean_i L_i

``M_i`` is the number of trainable tokens of trajectory ``i``, ``w_it`` is the HPI
weight of turn ``t`` (1.0 for the GRPO arm and for every unscored turn).

Why the reduction is not slime's default
----------------------------------------
``sum_of_sample_mean`` averages per *sample*. With one sample per turn that would
weight a long episode more than a short one simply because it has more turns. The
reduction here groups the rows back into trajectories and takes each trajectory's
own token mean first, then averages over trajectories -- which is what v1's
one-row-per-episode layout computed implicitly, and what makes the two arms
comparable.

Trajectory identity comes from ``turn_ids``, which the rollout packs as
``episode_id * TURN_ID_STRIDE + turn_idx``.

Advantage estimator
-------------------
``args.advantage_estimator`` selects the *surrogate*, and only ``gspo`` changes
it: slime's sequence-level KL (:func:`compute_gspo_kl`) replaces the per-token
``old - new``, so ``ratio = exp(-ppo_kl)`` is one ratio per sampled sequence
instead of one per token. Mirroring slime's contract, the sequence is the ROW --
i.e. one turn here, since v2 emits one row per turn. Anything else (including a
missing attribute, and every arm launched before this existed) takes the plain
token-level path unchanged.

That branch is only meaningful when old and new are different policies. With the
old-policy source this launcher uses (the recomputed forward, detached) they are
the same by construction, ``ppo_kl`` is 0 on both branches, and GSPO collapses
onto GRPO exactly -- same loss, same gradient. That is a property of the launch
config, not of the estimator, so it is detected at run time
(:func:`detect_degenerate_sequence_kl`), warned about once, and reported per step
as ``v2/gspo_degenerate_kl``. Do not read a GSPO number off a run whose log says
``kl=sequence-degenerate``.

Teacher
-------
This module does NOT import the teacher. ``teacher_forward_calls`` stays 0 no matter
what ``--recompute-loss-function`` does, because the privileged scoring happened
once, before the step, in ``scorer.postprocess``.
"""

import logging
import os
from argparse import Namespace
from collections.abc import Callable

import torch

from . import config

logger = logging.getLogger(__name__)

__all__ = [
    "alfworld_v2_loss",
    "apply_framework_normalization",
    "compute_ppo_kl",
    "detect_degenerate_sequence_kl",
    "episode_of_turn_id",
    "estimator_name",
    "gather_full_sequence",
    "is_gspo",
    "log_probs_and_entropy",
    "norm_mode",
    "resolve_old_log_probs",
    "teacher_forward_calls",
    "trajectory_balanced_reduce",
    "trajectory_mean_reduce",
]

#: Provably zero: nothing in this module can call the teacher.
_TEACHER_FORWARD_CALLS = 0
_LOSS_CALLS = 0
#: One-shot latch for the "gspo is running as grpo" warning (see
#: :func:`detect_degenerate_sequence_kl`); a per-step warning would drown the log.
_DEGENERATE_SEQUENCE_KL_WARNED = False

#: slime's own default for ``--advantage-estimator``; also what a namespace that
#: predates the flag means.
DEFAULT_ADVANTAGE_ESTIMATOR = "grpo"
#: The one estimator whose surrogate is sequence-level rather than token-level.
GSPO_ADVANTAGE_ESTIMATOR = "gspo"


def teacher_forward_calls() -> int:
    return _TEACHER_FORWARD_CALLS


def reference_kl(args, log_probs, ref_log_probs):
    """Initial-model low-variance KL, shared by both custom actor objectives."""
    if not getattr(args, "use_kl_loss", False):
        return torch.zeros_like(log_probs)
    if ref_log_probs is None:
        raise ValueError("Initial-model KL enabled but ref_log_probs are missing")
    if getattr(args, "kl_loss_type", "low_var_kl") != "low_var_kl":
        raise ValueError("ALFWorld initial-model KL requires low_var_kl")
    # Same k3 estimator and stability clamp as slime.utils.ppo_utils.
    delta = ref_log_probs.detach().float() - log_probs.float()
    if not getattr(args, "alfworld_reference_kl_logged", False):
        logger.info("[initial-reference-kl] enabled=True type=low_var_kl coef=%s ref=%s tokens=%d",
                    getattr(args, "kl_loss_coef", None), getattr(args, "ref_load", None), delta.numel())
        args.alfworld_reference_kl_logged = True
    return (delta.exp() - 1 - delta).clamp(min=-10, max=10)


def estimator_name(args: Namespace) -> str:
    """The estimator this step actually uses; a namespace without the flag means
    slime's default, never an implicit switch to GSPO."""
    return getattr(args, "advantage_estimator", DEFAULT_ADVANTAGE_ESTIMATOR)


def is_gspo(args: Namespace) -> bool:
    """True when the surrogate must be sequence-level (GSPO)."""
    return estimator_name(args) == GSPO_ADVANTAGE_ESTIMATOR


#: Flipped by :func:`verify_old_policy_contract` after the first call, so the check
#: costs one comparison on one micro-batch per process and is silent thereafter.
_OLD_POLICY_CHECKED = False

#: Per-micro-batch biases making up the current window. The statistic the docstring
#: argues for is a MEAN over many tokens, but with ``--micro-batch-size 1`` one call
#: sees ONE turn (~hundreds of tokens) -- few enough that a handful of near-tie
#: tokens, where the engine's ascend kernels and the trainer's flash attention
#: reconstruct the same bf16 softmax differently by up to ~3 nats, can move that
#: single sample's signed mean to ~0.06. That is the tail of the noise distribution,
#: not a convention break. Measured 2026-10-04: this exact assertion killed the
#: ALFWorld HSI arm at rollout 8 (bias 0.0571) and the math HSI arm five times
#: (0.0501-0.0699), while every healthy step sits at 1e-5..1e-2.
_OLD_POLICY_BIAS_WINDOW: list[float] = []

#: Set once the first window has been judged. Only that window may carry the
#: tolerance gate -- see the comment at the check.
_OLD_POLICY_FIRST_WINDOW_DONE = False

#: Micro-batches per window. 32 == one optimizer step of turn-rows for both ALFWorld
#: arms, so the window is exactly the batch the docstring's mean is taken over.
_OLD_POLICY_WINDOW = int(os.environ.get("ALFWORLD_V2_OLDPOLICY_WINDOW", "32") or 32)

#: A sampling-convention mismatch shifts EVERY token, so its per-micro-batch mean is
#: O(1) nats. This ceiling is ~7x above the worst observed kernel-tail event (0.0699)
#: and an order below such a mismatch, so a genuine break is still fatal at once
#: instead of waiting for the window to fill.
_OLD_POLICY_HARD_CEIL = float(
    os.environ.get("ALFWORLD_V2_OLDPOLICY_HARD_CEIL", "0.5") or 0.5
)


def verify_old_policy_contract(
    args: Namespace,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> float:
    """Section 6's explicit sampling-convention assertion.

    The rollout stores the sampler's own log-prob, ``--use-rollout-logprobs`` feeds
    it to PPO as ``old``, and the trainer's first forward of the step still holds
    ``theta_old``. So on the very first micro-batch the two sides must agree token
    for token, up to bf16 noise. If they do not, the two sides are in different
    temperature/support conventions -- the failure mode that already wrecked one
    run on this stack (response length 800 -> 25, reward 0.58 -> 0.0) when
    ``SGLANG_RETURN_ORIGINAL_LOGPROB`` flipped the engine to pre-temperature values.

    WHY THE STATISTIC IS THE MEAN AND NOT THE MAX. The first version of this check
    asserted on ``max|current - rollout|`` and aborted a healthy run: two bf16
    implementations of the same softmax (the engine's ascend backend against the
    trainer's flash attention) disagree by ~0.35 nats on the WORST of a few hundred
    tokens, simply because a near-tie between the top two logits is reconstructed
    differently by the two kernels. A convention mismatch is not like that: it is a
    SYSTEMATIC shift, so it moves the MEAN of ``current - rollout`` by O(1) nats.
    The mean is therefore the statistic that separates the failure mode from the
    noise, and the max is reported alongside it as information.

    WHEN THE CHECK APPLIES, AND WHEN IT MUST NOT. "The trainer's first forward of the
    step still holds theta_old" is true of exactly ONE step per rollout: the weights
    are synced to the rollout engine before the rollout is generated, so the first
    optimizer step of the run runs on theta_old, and every later step of every rollout
    runs on weights that have moved. There, ``current - rollout`` IS the PPO ratio and
    is *supposed* to be non-zero. The tolerance gate is therefore applied to the FIRST
    window only (``_OLD_POLICY_FIRST_WINDOW_DONE``); the failure mode it exists to
    catch -- ``SGLANG_RETURN_ORIGINAL_LOGPROB`` flipping the engine to pre-temperature
    log-probs -- is fixed by launch flags for the whole run and is present from token
    one, so one certification is enough.

    Asserting on later micro-batches is not merely conservative, it is wrong, and it
    cost this campaign a run: the ``raise`` originally sat outside the ``if first:``
    guard, so it asserted on every micro-batch, and on 2026-10-04 17:58 it killed the
    ALFWorld HSI arm at rollout 8 with ``|mean delta| = 0.0571`` -- which is
    ``ratio = exp(-0.057) = 0.945``, deep inside ``clip_eps = 0.2``. The gate was
    asserting that the policy never moves.

    WHY A WINDOW AND NOT ONE MICRO-BATCH. With ``--micro-batch-size 1`` a call sees
    one turn, and the tail of the kernel-noise distribution is wide enough to push a
    single turn's mean past the tolerance (see ``_OLD_POLICY_BIAS_WINDOW`` for the
    measured events). The first window therefore averages ``_OLD_POLICY_WINDOW``
    micro-batches -- one optimizer step of turn-rows, all at theta_old -- and a single
    micro-batch at or above ``_OLD_POLICY_HARD_CEIL``, which only a real mismatch
    reaches, fails on the spot during initial certification.

    Measured on the retired candidate arm (same ``--use-rollout-logprobs`` path,
    same stack): step-0 ``ppo_kl`` -- which IS this mean, negated -- was -2.3e-05.
    The default tolerance of 0.05 nats is three orders of magnitude above that and
    three below a real mismatch.
    """
    global _OLD_POLICY_CHECKED, _OLD_POLICY_FIRST_WINDOW_DONE
    with torch.no_grad():
        delta = current_log_probs.float() - old_log_probs.float()
        if delta.numel():
            bias = float(delta.mean().abs().item())
            worst = float(delta.abs().max().item())
            mean_abs = float(delta.abs().mean().item())
        else:
            bias = worst = mean_abs = 0.0
    # Escape hatch for unit tests that deliberately pair an old policy with a
    # different current one to exercise the surrogate maths. Production runs leave
    # it unset, i.e. the contract is checked.
    if os.environ.get("ALFWORLD_V2_OLDPOLICY_ASSERT", "1") != "1":
        return bias
    tolerance = float(os.environ.get("ALFWORLD_V2_OLDPOLICY_TOL", "0.05"))
    first = not _OLD_POLICY_CHECKED
    _OLD_POLICY_CHECKED = True
    if first:
        logger.info(
            "[v2-contract] old-policy check: |mean(current - rollout)| = %.6f, "
            "mean|delta| = %.6f, max|delta| = %.6f over %d tokens "
            "(tol=%.3f on the MEAN, roll_temperature=%.2f, use_rollout_logprobs=%s)",
            bias,
            mean_abs,
            worst,
            int(delta.numel()),
            tolerance,
            float(getattr(args, "rollout_temperature", 1.0) or 1.0),
            bool(getattr(args, "use_rollout_logprobs", False)),
        )
    # Certify the initial policy only. Later optimizer steps intentionally move
    # current away from old; their PPO ratio must not trigger this startup gate.
    if not _OLD_POLICY_FIRST_WINDOW_DONE and bias >= _OLD_POLICY_HARD_CEIL:
        raise AssertionError(
            f"old-policy contract violated: one micro-batch's rollout log-probs and the "
            f"trainer's forward differ by {bias:.4f} nats ON AVERAGE (hard ceiling "
            f"{_OLD_POLICY_HARD_CEIL}, tol {tolerance}; max|delta|={worst:.4f}). An offset that "
            f"large on every token of one turn is a sampling-convention mismatch, not "
            f"numerical noise -- check SGLANG_RETURN_ORIGINAL_LOGPROB and "
            f"--rollout-temperature before training on this batch."
        )
    _OLD_POLICY_BIAS_WINDOW.append(bias)
    if len(_OLD_POLICY_BIAS_WINDOW) >= _OLD_POLICY_WINDOW:
        window = _OLD_POLICY_BIAS_WINDOW[:]
        del _OLD_POLICY_BIAS_WINDOW[:]
        window_mean = sum(window) / len(window)
        # The tolerance gate applies ONLY to the first window. That window is the
        # first optimizer step of the run, i.e. the only place where `current` and
        # `rollout` provably share theta: the weights are synced to the rollout
        # engine before the rollout is generated, so the first step's forward still
        # holds theta_old. Every later step of a rollout runs on weights that have
        # moved, so there `current - rollout` IS the PPO ratio and is *supposed* to
        # be non-zero -- asserting on it would assert that the policy never moves.
        # That is not hypothetical: it killed the ALFWorld HSI arm at rollout 8
        # (2026-10-04 17:58, bias 0.0571 == ratio 0.945, deep inside clip_eps 0.2).
        if not _OLD_POLICY_FIRST_WINDOW_DONE:
            _OLD_POLICY_FIRST_WINDOW_DONE = True
            if window_mean > tolerance:
                raise AssertionError(
                    f"old-policy contract violated: on the first {len(window)} micro-batches "
                    f"(theta_old, before any optimizer step) the rollout log-probs and the "
                    f"trainer's forward differ by {window_mean:.4f} nats ON AVERAGE "
                    f"(tol {tolerance}; this batch's max|delta|={worst:.4f}). A systematic offset "
                    f"of that size cannot be kernel noise -- check "
                    f"SGLANG_RETURN_ORIGINAL_LOGPROB and --rollout-temperature before training "
                    f"on this run."
                )
            logger.info(
                "[v2-contract] certified over first window: |mean(current - rollout)| = %.6f "
                "over %d micro-batches (tol %.3f)",
                window_mean,
                len(window),
                tolerance,
            )
        elif window_mean > tolerance:
            # Legitimate policy drift (this window is a later optimizer step), or a
            # tail event. Visible, never fatal after startup certification.
            logger.warning(
                "[v2-contract] window mean %.4f > tol %.3f over %d micro-batches "
                "-> post-first-window drift, not fatal",
                window_mean,
                tolerance,
                len(window),
            )
    elif bias > tolerance:
        # A single overflowing micro-batch inside a healthy window: report it so the
        # phenomenon stays visible without concluding anything from one sample.
        logger.warning(
            "[v2-contract] micro-batch bias %.4f > tol %.3f (window %d/%d, mean so far %.5f) "
            "-> tail event, not fatal",
            bias,
            tolerance,
            len(_OLD_POLICY_BIAS_WINDOW),
            _OLD_POLICY_WINDOW,
            sum(_OLD_POLICY_BIAS_WINDOW) / len(_OLD_POLICY_BIAS_WINDOW),
        )
    return bias


def detect_degenerate_sequence_kl(
    args: Namespace,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> bool:
    """True when the sequence-level KL would collapse onto the token-level one.

    That happens exactly when the old-policy source IS this forward -- i.e. the
    harness's ``log_probs.detach()`` -- because then ``old - new`` is identically
    zero for every token and the sequence mean of zeros is the token value. The
    GSPO surrogate is then *numerically GRPO*: same loss, same gradient, ratio == 1
    on both arms. This is not detectable from the launch flags, so it is detected
    here and said out loud (warning once, plus a per-step metric) instead of
    quietly shipping a GSPO label on GRPO numbers.

    Always False for the token-level estimators: they read the same source by
    design and are not claiming anything else.
    """
    if not is_gspo(args):
        return False
    degenerate = torch.equal(old_log_probs.detach(), new_log_probs.detach())
    if degenerate:
        global _DEGENERATE_SEQUENCE_KL_WARNED
        if not _DEGENERATE_SEQUENCE_KL_WARNED:
            _DEGENERATE_SEQUENCE_KL_WARNED = True
            logger.warning(
                "[v2-loss] advantage_estimator=gspo, but the old-policy source is this "
                "very forward (ppo_kl == 0 on every token). The sequence-level KL then "
                "equals the token-level one: this step is numerically GRPO. Give the arm "
                "a real old policy -- --use-rollout-logprobs with non-placeholder "
                "rollout log-probs, or slime's batch['log_probs'] -- before publishing "
                "any GSPO number."
            )
    return degenerate


def log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool,
    max_seq_lens: list[int] | None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """``get_log_probs_and_entropy`` behind a seam.

    That forward needs Megatron's parallel state, which a CPU test process does
    not have (and which ``slime.backends.megatron_utils`` cannot even be imported
    without -- see the CPU test notes). Same call, same arguments as before; only
    the import site moved so the tests can supply fixed log-probs.
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    return get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
    )


def gather_full_sequence(tensor: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    """``all_gather_with_cp`` behind a seam.

    GSPO takes its mean over the whole sequence, so slime gathers each sample's
    log-probs across the context-parallel group before averaging. At
    ``--context-parallel-size 1`` -- the only setting this harness launches with
    -- that function is the identity; the seam exists so the CPU tests can run
    the real sequence-level arithmetic without a Megatron process group.
    """
    from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

    return all_gather_with_cp(tensor, total_length, response_length)


def resolve_old_log_probs(
    args: Namespace,
    batch: dict,
    new_log_probs_per_sample: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Old-policy log-probs: per sample, and concatenated.

    Same source rule as slime's ``policy_loss_function``: the rollout's own
    log-probs when ``--use-rollout-logprobs`` is set, else ``batch["log_probs"]``.

    THE BUG THIS FIXES. ``batch["log_probs"]`` is the pass slime runs BEFORE any
    optimizer step (`actor.py:476-489`): it switches to the ``old_actor`` weights
    when ``--keep-old-actor`` is set, otherwise to the not-yet-updated ``actor``,
    and stores the result. That is the behaviour policy of this batch. The v2 loss
    used to ignore it and pass ``log_probs.detach()`` -- the loss's OWN forward.
    New and old were then literally the same tensor, so ``ratio == 1`` exactly,
    ``ppo_kl == 0`` and ``pg_clipfrac == 0`` on every step (measured across whole
    runs): PPO's clipping branch could never execute, and the objective silently
    degraded to a plain policy gradient with no trust region.

    Falling back to the loss's own forward is deliberately NOT done: if
    ``log_probs`` is missing that is a configuration error, and silently returning
    a ratio of exactly 1 is the failure mode this function exists to prevent.
    """
    if getattr(args, "use_rollout_logprobs", False) and "rollout_log_probs" in batch:
        per_sample = list(batch["rollout_log_probs"])
    else:
        stored = batch.get("log_probs")
        if stored is None:
            raise ValueError(
                "no old-policy log-probs in the batch: expected batch['log_probs'] from "
                "slime's pre-train old-actor pass. Using this forward's own log-probs "
                "as 'old' makes the PPO ratio identically 1."
            )
        per_sample = [log_probs.detach() for log_probs in stored]
    return per_sample, torch.cat(per_sample, dim=0)


def compute_ppo_kl(
    args: Namespace,
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs_per_sample: list[torch.Tensor],
    old_log_probs_per_sample: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> torch.Tensor:
    """``ppo_kl`` in slime's sign convention (``old - new``).

    Token-level by default. Under ``--advantage-estimator gspo`` this is slime's
    sequence-level construction (``megatron_utils/loss.py``, the
    ``compute_gspo_kl`` branch): every token carries its own sequence's mean
    ``old - new``, so ``ratio = exp(-ppo_kl)`` is one ratio per sequence. The
    arguments mirror that call site exactly.
    """
    if not is_gspo(args):
        return old_log_probs - new_log_probs

    from slime.utils.ppo_utils import compute_gspo_kl

    full_log_probs = [
        gather_full_sequence(log_probs, total_length, response_length)
        for log_probs, total_length, response_length in zip(
            new_log_probs_per_sample, total_lengths, response_lengths, strict=False
        )
    ]
    full_old_log_probs = [
        gather_full_sequence(log_probs, total_length, response_length)
        for log_probs, total_length, response_length in zip(
            old_log_probs_per_sample, total_lengths, response_lengths, strict=False
        )
    ]
    return compute_gspo_kl(
        full_log_probs=full_log_probs,
        full_old_log_probs=full_old_log_probs,
        local_log_probs=new_log_probs_per_sample,
        loss_masks=loss_masks,
    )


def episode_of_turn_id(turn_id: int) -> int:
    return int(turn_id) // config.TURN_ID_STRIDE


def trajectory_mean_reduce(
    per_token: torch.Tensor,
    response_lengths: list[int],
    episode_ids: list[int],
) -> torch.Tensor:
    """``mean_i (1/M_i) sum_tokens`` over the concatenated per-token losses.

    ``per_token`` is the concatenation of every sample's response positions;
    ``episode_ids[i]`` is the trajectory sample ``i`` belongs to.
    """
    if per_token.numel() == 0:
        return per_token.sum() * 0.0

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    offset = 0
    for sender_index, rlen in enumerate(response_lengths):
        rlen = int(rlen)
        if rlen <= 0:
            continue
        segment = per_token[offset : offset + rlen]
        offset += rlen
        episode = episode_ids[sender_index]
        sums[episode] = sums.get(episode, torch.zeros((), device=per_token.device)) + segment.sum()
        counts[episode] = counts.get(episode, 0) + rlen

    if not sums:
        return per_token.sum() * 0.0
    per_episode = torch.stack([sums[e] / max(counts[e], 1) for e in sums])
    return per_episode.mean()


#: `legacy` (default) or `sum`. See `apply_framework_normalization`.
_NORM_ENV = "OPSD_V2_LOSS_NORM"


def norm_mode() -> str:
    return os.environ.get(_NORM_ENV, "legacy").strip().lower()


def _local_tokens(loss_masks: list[torch.Tensor]) -> float:
    """This rank's token count for the current microbatch.

    Mirrors `num_tokens` in `megatron_utils/loss.py::loss_function`, and
    deliberately does NOT do a collective: the framework accumulates this value
    over microbatches and all-reduces it over the dp_cp group itself
    (`Megatron-LM/megatron/core/distributed/finalize_model_grads.py:492`), then
    divides the gradient by the result. Reproducing that sum here would double it.
    """
    return float(sum(float(torch.clamp_min(torch.as_tensor(m).sum(), 1).item()) for m in loss_masks))


def trajectory_balanced_reduce(
    per_token: torch.Tensor,
    response_lengths: list[int],
    batch: dict,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """``sum_rows w_r sum_tokens l`` with every trajectory's weights summing to 1.

    THE BUG THIS FIXES. A trajectory is emitted as one row per turn and every row
    carries its episode's GRPO advantage; weighting rows equally lets a long
    trajectory shout louder than a short one. Measured on a real 8B rollout the
    batch was 36 positive rows against 572 negative (15.9:1), so the gradient was
    ~94% "push these tokens down" -- and the cheapest way to obey that is to stop
    emitting coherent text, which is exactly what destroyed the ``</think>``
    structure and collapsed the run.

    `w` is the per-token trajectory-balanced weight computed in
    `train_data.trajectory_weighting` (constant within a row, summing to 1 across
    each trajectory's rows). `scale` is the step-level ``N_global/(T*G)`` that
    cancels the framework's own ``1/num_tokens`` division; it must come from the
    batch because one micro-batch cannot see the whole step.

    Both are required rather than derived, and both are validated: a missing or
    length-mismatched weight is an error, not a silent fallback to equal rows --
    a silent fallback here is the original bug.
    """
    row_weights = batch.get("fit_turn_weights")
    scales = batch.get("traj_norm_scale")
    if row_weights is None or scales is None:
        raise ValueError(
            "OPSD_V2_LOSS_NORM=traj requires batch['fit_turn_weights'] and "
            "batch['traj_norm_scale']; launch with the v2 convert "
            "(tasks.alfworld.train_data.convert_samples_to_train_data)."
        )

    expected = int(sum(int(r) for r in response_lengths))
    if int(per_token.numel()) != expected:
        raise ValueError(f"per_token has {per_token.numel()} entries, expected {expected}")

    def _flat(values, name):
        pieces = []
        for value, rlen in zip(values, response_lengths, strict=False):
            piece = torch.as_tensor(value, dtype=per_token.dtype, device=per_token.device)
            if int(piece.numel()) < int(rlen):
                raise ValueError(f"{name} row is shorter than its response length")
            pieces.append(piece[: int(rlen)])
        return torch.cat(pieces) if pieces else per_token.new_zeros(0)

    scale_tokens = _flat(scales, "traj_norm_scale")
    scale = float(scale_tokens[0].item()) if scale_tokens.numel() else 1.0

    # `w` is a weight on each ROW's token-mean, not on each token. Weighting the
    # per-token losses directly would give sum_r w_r N_r mean_r -- length back in,
    # through the back door, which is the bug. So reduce each row to its own mean
    # first, then take the weighted sum. Row means and the emitted weights are
    # built from the same `response_lengths`, so the row order is identical.
    w_tokens = _flat(row_weights, "fit_turn_weights")
    total = per_token.new_zeros(())
    offset = 0
    for sender, rlen in enumerate(response_lengths):
        rlen = int(rlen)
        if rlen <= 0:
            continue
        row_mean = per_token[offset : offset + rlen].mean()
        total = total + w_tokens[offset : offset + rlen][0] * row_mean
        offset += rlen
    return total * scale, scale, w_tokens


def apply_framework_normalization(
    loss: torch.Tensor,
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, float, float]:
    """Put the loss back into the units slime's contract expects. (loss, mult, n_tokens).

    THE BUG THIS EXISTS TO FIX. Under ``--calculate-per-token-loss`` slime hands
    `sum_of_token` in as the `sum_of_sample_mean` argument (`cp_utils.py:120`),
    i.e. **a custom loss is contracted to return a SUM over tokens** -- the one
    normalization is the framework's, and it happens once, later, in
    `finalize_model_grads`. `trajectory_mean_reduce` returns a per-trajectory
    MEAN instead, so the framework divided a second time: the gradient came out
    smaller by the microbatch's token count. Worse than a constant offset, that
    factor moves with response length, so the effective learning rate was coupled
    to how long the policy happened to ramble -- the two arms, whose lengths
    diverge, stopped being comparable at all.

    The fix multiplies back by exactly this rank's token count, which makes the
    custom loss arithmetically identical to slime's own `policy_loss` under
    ``--calculate-per-token-loss`` (a token-mean over the global batch). This is
    exact at ``--micro-batch-size 1``, where `trajectory_mean_reduce` reduces to
    the single row's token-mean and multiplying by that row's token count is the
    token sum.

    Note what this does NOT do: re-weight tokens per trajectory. `M_i` in
    `trajectory_mean_reduce` is a single turn's response length at this batch
    size, so the per-trajectory grouping in that function is inert here. A
    trajectory-balanced objective would be a method change, and belongs in the
    per-token weights, not in the normalization.

    `legacy` keeps the shipped value so historical runs stay reproducible.
    """
    if norm_mode() != "sum":
        return loss, 1.0, 0.0
    n_tokens = _local_tokens(loss_masks)
    if n_tokens <= 0.0:
        return loss, 1.0, 0.0
    return loss * n_tokens, n_tokens, n_tokens


def alfworld_v2_loss(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """slime custom loss (``--custom-loss-function-path``)."""
    from slime.utils.ppo_utils import compute_policy_loss

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    unconcat_tokens = batch["unconcat_tokens"]
    loss_masks = batch["loss_masks"]

    _, forward = log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=getattr(args, "entropy_coef", 0.0) != 0,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    log_probs_per_sample = forward["log_probs"]
    log_probs = torch.cat(log_probs_per_sample, dim=0)

    advantages = batch.get("opsd_advantages")
    if advantages is None:
        raise ValueError(
            "v2 requires the rollout's trajectory-level advantages; launch with --use-opsd "
            "(or OPSD_ENABLED=1) so Sample.opsd_advantages reaches the loss."
        )
    adv = torch.cat([torch.as_tensor(a, dtype=log_probs.dtype, device=log_probs.device) for a in advantages])

    old_log_probs_per_sample, old_log_probs = resolve_old_log_probs(args, batch, log_probs_per_sample)
    # Section 6 of the HSI protocol: assert ONCE that the stored rollout log-probs
    # and the trainer's first forward of this step agree on the same tokens.
    old_policy_bias = verify_old_policy_contract(args, log_probs, old_log_probs)
    # A GSPO arm whose old policy is this same forward is a GRPO arm wearing a
    # GSPO label; say so rather than let it train quietly (see the function).
    degenerate_kl = detect_degenerate_sequence_kl(args, old_log_probs, log_probs)
    ppo_kl = compute_ppo_kl(
        args,
        new_log_probs=log_probs,
        old_log_probs=old_log_probs,
        new_log_probs_per_sample=log_probs_per_sample,
        old_log_probs_per_sample=old_log_probs_per_sample,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, adv, args.eps_clip, args.eps_clip_high)

    # No loss-side re-weighting.
    #
    # The hindsight term now lives INSIDE the advantage (the rollout anchors on
    # 2R-1 and the scorer adds the standardised credit), so the surrogate is the
    # plain clipped PPO/GRPO form. Multiplying here as well would double-apply it,
    # and would re-introduce the zero-advantage annihilation the redesign removed.
    w = torch.ones_like(pg_loss)
    per_token = pg_loss
    refs = batch.get("ref_log_probs")
    kl = reference_kl(args, log_probs, torch.cat(refs) if refs is not None else None)
    per_token = per_token + getattr(args, "kl_loss_coef", 0.0) * kl

    # ---- repeat-stop conditional CE ------------------------------------------
    #
    # Feature-gated: with ALFWORLD_V2_REPEAT_STOP unset this is an identity return
    # and every downstream number is bit-identical to the baseline. Where the
    # section-6 gate fires, the PPO term at that position is REPLACED (not added
    # to) by a conditional cross-entropy that prefers `</think>` over the tokens
    # the policy historically repeated there. The replacement happens BEFORE the
    # trajectory reduction so the gated tokens ride the same weights, the same
    # denominator and the same all-reduce as everything else.
    from . import repeat_stop

    per_token, repeat_metrics = repeat_stop.maybe_replace(
        per_token,
        args=args,
        logits=logits,
        batch=batch,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        advantages=adv,
        rewards=batch.get("rewards"),
    )

    # ---- trajectory-level reduction -----------------------------------------
    turn_ids = batch["turn_ids"]
    episode_ids = []
    for sample_turn_ids in turn_ids:
        first = sample_turn_ids[0]
        first = int(first.item()) if hasattr(first, "item") else int(first)
        episode_ids.append(episode_of_turn_id(first))
    if norm_mode() == "traj":
        loss, norm_mult, w_tokens = trajectory_balanced_reduce(per_token, response_lengths, batch)
        n_global = 0.0
        # Assertion B: the reducer weights of each trajectory sum to 1, so a
        # trajectory's optimizer weight cannot grow with its turn or token count.
        # Grouped by episode exactly as the weights were built.
        with torch.no_grad():
            per_episode = {}
            offset = 0
            for sender, rlen in enumerate(response_lengths):
                rlen = int(rlen)
                if rlen <= 0:
                    continue
                segment = w_tokens[offset : offset + rlen]
                offset += rlen
                key = episode_ids[sender]
                per_episode[key] = per_episode.get(key, torch.zeros((), device=w_tokens.device)) + segment.sum()
            if per_episode:
                sums = torch.stack(list(per_episode.values()))
                weight_sum_err = float((sums - 1.0).abs().max().item())
            else:
                weight_sum_err = float("nan")
            n_rows_here = sum(1 for r in response_lengths if int(r) > 0)
            w_partial_sum = float(w_tokens.sum().item()) if w_tokens.numel() else float("nan")
    else:
        loss = trajectory_mean_reduce(per_token, response_lengths, episode_ids)
        loss, norm_mult, n_global = apply_framework_normalization(loss, loss_masks)
        weight_sum_err = float("nan")
        w_tokens = None
        n_rows_here = 0
        w_partial_sum = float("nan")

    with torch.no_grad():
        mask = torch.cat(
            [torch.as_tensor(m[: int(r)], device=pg_loss.device) for m, r in zip(loss_masks, response_lengths)]
        )
        n_masked = float(mask.sum().item()) if mask.numel() else 0.0
        metrics = {
            "loss": loss.clone().detach(),
            "v2/reference_kl_mean": kl.detach().mean(),
            "v2/pg_loss_mean": pg_loss.detach().mean() if pg_loss.numel() else torch.zeros((), device=logits.device),
            "v2/pg_clipfrac": pg_clipfrac.detach().mean() if pg_clipfrac.numel() else torch.zeros((), device=logits.device),
            "v2/ppo_kl": ppo_kl.detach().mean() if ppo_kl.numel() else torch.zeros((), device=logits.device),
            "v2/weight_mean": w.detach().mean() if w.numel() else torch.ones((), device=logits.device),
            "v2/weight_min": w.detach().min() if w.numel() else torch.ones((), device=logits.device),
            "v2/weight_max": w.detach().max() if w.numel() else torch.ones((), device=logits.device),
            "v2/n_trajectories": torch.tensor(float(len(set(episode_ids))), device=logits.device),
            # Nonzero only under OPSD_V2_LOSS_NORM=sum. slime rescales every value
            # in this dict by the batch token count, so read `norm_applied`, not the
            # magnitude of `norm_multiplier`, to tell which path ran. Under the fix
            # `norm_multiplier` is this rank's token count, i.e. `n_response_tokens`.
            "v2/norm_applied": torch.tensor(float(norm_mult > 1.0), device=logits.device),
            "v2/norm_multiplier": torch.tensor(float(norm_mult), device=logits.device),
            "v2/norm_tokens": torch.tensor(float(n_global), device=logits.device),
            # DELIBERATELY NOT an absolute check. At `--micro-batch-size 1` a
            # microbatch holds ONE turn-row, so the per-trajectory weight sum
            # cannot reach 1 here no matter how correct the reducer is -- an
            # earlier version of this metric was named
            # `traj_weight_sum_err` and documented as "must be ~0", which was
            # false by construction (measured 0.08) and would send a reader
            # hunting a bug that does not exist. The authoritative assertion B
            # is computed at convert time over whole trajectories and logged as
            # `[v2-traj] ... w_sum_err=...`.
            "v2/traj_rows_in_microbatch": torch.tensor(float(n_rows_here), device=logits.device),
            "v2/traj_w_partial_sum": torch.tensor(float(w_partial_sum), device=logits.device),
            "v2/traj_scale": torch.tensor(float(norm_mult), device=logits.device),
            "v2/traj_w_mean": (
                w_tokens.detach().mean() if w_tokens is not None and w_tokens.numel() else torch.zeros((), device=logits.device)
            ),
            "v2/traj_w_max": (
                w_tokens.detach().max() if w_tokens is not None and w_tokens.numel() else torch.zeros((), device=logits.device)
            ),
            # 1.0 exactly when the sequence-level branch ran. Zero-vs-nonzero
            # survives the token-count rescaling slime applies to this dict, so a
            # log reader can tell a GSPO step from a GRPO step without trusting
            # the launch flags.
            "v2/gspo_sequence_kl": torch.tensor(float(is_gspo(args)), device=logits.device),
            # 1.0 when that branch degenerated back onto the token-level one
            # (old policy == this forward). A GSPO run that reports this is a GRPO
            # run; it is the one thing the launch flags cannot tell you.
            "v2/gspo_degenerate_kl": torch.tensor(float(degenerate_kl), device=logits.device),
            "v2/n_response_tokens": torch.tensor(n_masked, device=logits.device),
            "v2/teacher_forward_calls": torch.tensor(float(teacher_forward_calls()), device=logits.device),
            # Section 6's contract residual, per step. Zero would be too strong a
            # claim across a bf16 forward; a value that jumps to O(1) is the
            # convention-mismatch signature.
            "v2/oldpolicy_mean_delta": torch.tensor(float(old_policy_bias), device=logits.device),
        }
        # Repeat-stop diagnostics. Empty when the feature is off, so the baseline
        # metric set is unchanged. Every key is deliberately added to the
        # whitelist-free path (plain floats -> tensors) so no new batch key and no
        # new skip-table entry is needed.
        for key, value in repeat_metrics.items():
            metrics[key] = torch.tensor(float(value), device=logits.device)
    # slime RESCALES everything in the returned metric dict by the batch token
    # count (measured: `n_response_tokens` came back as 1.0 and every other entry
    # as the same small scalar, because they were all divided by the token total).
    # The metrics dict is therefore unusable for reading absolute quantities, so the
    # numbers that matter are logged directly instead.
    if int(os.environ.get("ALFWORLD_V2_LOG_EVERY_N_CALLS", "1")) > 0 and w.numel():
        _LOG_EVERY = int(os.environ.get("ALFWORLD_V2_LOG_EVERY_N_CALLS", "1"))
        global _LOSS_CALLS
        _LOSS_CALLS += 1
        if _LOSS_CALLS % _LOG_EVERY == 0:
            logger.info(
                "[v2-loss] est=%s kl=%s rows=%d tokens=%d trajectories=%d "
                "w(mean=%.6f min=%.6f max=%.6f) pg_loss_mean=%.6g grad_term=%.6g",
                estimator_name(args),
                # The log line carries the level AND whether it collapsed, so a run
                # that is GRPO under a gspo label is readable from its own log.
                ("sequence-degenerate" if degenerate_kl else "sequence") if is_gspo(args) else "token",
                len(response_lengths),
                int(w.numel()),
                len(set(episode_ids)),
                float(w.mean()),
                float(w.min()),
                float(w.max()),
                float(pg_loss.mean()) if pg_loss.numel() else 0.0,
                float(per_token.mean()) if per_token.numel() else 0.0,
            )

    return loss, metrics
