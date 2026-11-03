"""Official OPSD objective as a slime custom loss (--loss-type custom_loss).

Extracted from the official OPSD port (correctness oracle):
    repos/opsd-npu/src/opsd_trainer.py  ->  OPSDTrainer.compute_loss
                                           + OPSDTrainer.generalized_jsd_loss

Semantics preserved exactly:
  * student forward over [student prompt | completion]; teacher forward over
    [privileged prompt | completion]; both evaluate the SAME completion tokens
    (shift-by-one: logits[P-1 : P-1+R] predict tokens[P : P+R]).
  * generalized JSD with beta=0 -> KL(teacher || student) via
    F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).
  * temperature applied to logits of BOTH sides (official runs the loss at the
    rollout temperature, 1.1).
  * pointwise clipping of the per-token divergence (jsd_token_clip=0.05) applied
    BEFORE masking/reduction.
  * reduction "batchmean": masked token sum / masked token count. slime's
    --calculate-per-token-loss turns `sum_of_sample_mean` into the masked token
    sum and Megatron divides by the masked token count, i.e. the same
    sum/count form, evaluated across the whole global batch.
  * full vocabulary (top_k_loss=0): V=151936 for Qwen3; no sampled-token KL.

Teacher policy: fixed teacher = initial policy. Each training rank lazily loads
one frozen HF copy of the initial checkpoint (`--hf-checkpoint`, the same weights
the Megatron actor starts from), runs it under eval()/no_grad() and never
updates it. This mirrors the official port, where teacher and student share the
same HF implementation and the teacher is the base model with LoRA disabled.
"""

import logging
import os
from argparse import Namespace
from typing import Callable

import torch
import torch.nn.functional as F

from opsd_core.alignment import assert_completion_alignment  # noqa: F401  (re-export)
from opsd_core.batch import build_right_padded_batch  # noqa: F401  (re-export)
from opsd_core.teacher import (  # noqa: F401  (re-export: imported by frameworks + task lines)
    _get_teacher,
    get_teacher,
    slice_response_logits,
    teacher_forward_completion_logits,
)

logger = logging.getLogger(__name__)

# Defaults mirror the official run command line exactly.
DEFAULT_BETA = 0.0
DEFAULT_TOKEN_CLIP = 0.05
DEFAULT_TEMPERATURE = 1.1


# ---------------------------------------------------------------------------
# Official generalized JSD (verbatim port of generalized_jsd_loss, beta=0 path)
# ---------------------------------------------------------------------------


def generalized_jsd_per_token(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float = DEFAULT_BETA,
    temperature: float = DEFAULT_TEMPERATURE,
    token_clip: float | None = DEFAULT_TOKEN_CLIP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token generalized JSD between full-vocab student/teacher logits.

    Shapes: [T, V] each. Returns ``(per_token_clipped, per_token_raw)``, both [T].
    Ported from OPSDTrainer.generalized_jsd_loss with reduction stripped.
    """
    # Apply temperature scaling to logits before computing probabilities.
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    # Sum over the vocabulary, then pointwise-clip each token's divergence
    # (official: clipping BEFORE masking/reduction).
    jsd = jsd.sum(dim=-1)
    per_token_raw = jsd
    per_token_clipped = jsd.clamp(max=token_clip) if token_clip is not None else jsd
    return per_token_clipped, per_token_raw


# ---------------------------------------------------------------------------
# Frozen initial-policy teacher  ->  moved to opsd_core/teacher.py
# ---------------------------------------------------------------------------
# `_get_teacher`, `teacher_forward_completion_logits` and `slice_response_logits`
# are imported at the top of this module from opsd_core.teacher and re-exported
# here, so the vendored framework and the task lines that import them from
# `math_opsd.opsd_loss` keep resolving (REFACTOR_PLAN.md §1.3).


# ---------------------------------------------------------------------------
# slime custom loss entry point
# ---------------------------------------------------------------------------


def opsd_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """slime custom loss: official OPSD objective on the current microbatch."""
    args._current_max_seq_lens = batch.get("max_seq_lens", [0])[0] if batch.get("max_seq_lens") else 0
    response_lengths = batch["response_lengths"]
    teacher_tokens = batch["teacher_tokens"]
    teacher_prompt_lengths = batch["teacher_prompt_lengths"]
    unconcat_tokens = batch["unconcat_tokens"]

    temperature = float(os.environ.get("OPSD_LOSS_TEMPERATURE", DEFAULT_TEMPERATURE))
    beta = float(os.environ.get("OPSD_BETA", DEFAULT_BETA))
    token_clip_env = os.environ.get("OPSD_JSD_TOKEN_CLIP")
    token_clip = float(token_clip_env) if token_clip_env is not None else DEFAULT_TOKEN_CLIP

    # Student: full-vocab logits over response positions (carries gradient).
    student_logits = slice_response_logits(args, logits, unconcat_tokens, batch["total_lengths"], response_lengths)

    # Alignment contract: teacher completion span must equal the student
    # completion span, token by token (same positions are being evaluated).
    for tokens, plen, rlen, ucon in zip(
        teacher_tokens, teacher_prompt_lengths, response_lengths, unconcat_tokens, strict=True
    ):
        assert list(tokens[plen:].cpu()) == list(ucon[-rlen:].cpu()), (
            "OPSD alignment violated: teacher completion tokens differ from student completion tokens"
        )

    # Teacher: frozen initial policy, full-vocab logits over the SAME positions.
    import time as _time

    _t0 = _time.time()
    teacher_logits = teacher_forward_completion_logits(
        args, teacher_tokens, teacher_prompt_lengths, response_lengths
    )
    _t_teacher = _time.time() - _t0

    _t1 = _time.time()
    per_token, per_token_raw = generalized_jsd_per_token(
        student_logits,
        teacher_logits,
        beta=beta,
        temperature=temperature,
        token_clip=token_clip,
    )

    # success-only / failure-only ablation (OPSD_OUTCOME_ACTIVE): zero the
    # per-token loss of inactive samples; reduction and batch composition
    # unchanged (directive Phase 5).
    active = os.environ.get("OPSD_OUTCOME_ACTIVE", "both")
    if active in ("success", "failure") and batch.get("outcome_rewards") is not None:
        keep = []
        for i, rlen in enumerate(response_lengths):
            r = batch["outcome_rewards"][i]
            keep += [1.0 if (r == 1 if active == "success" else r == 0) else 0.0] * rlen
        mask = torch.tensor(keep, device=per_token.device, dtype=per_token.dtype)
        per_token = per_token * mask
        logger.info("[outcome-active] %s: active fraction=%.3f", active, float(mask.mean()))
    logger.info(
        "[opsd-kl] teacher_forward=%.2fs loss=%.2fs mean_KL(q||p)=%.5f clipped_frac=%.4f (microbatch=%d samples, %d tokens)",
        _t_teacher,
        _time.time() - _t1,
        float(per_token_raw.mean()) if per_token_raw.numel() else 0.0,
        float((per_token_raw > token_clip).float().mean()) if per_token_raw.numel() else 0.0,
        len(response_lengths),
        int(per_token.numel()),
    )
    logger.info(
        "[opsd-timer] teacher_forward=%.2fs student_loss=%.2fs (microbatch=%d samples, %d response tokens)",
        _t_teacher,
        _time.time() - _t1,
        len(response_lengths),
        int(per_token.numel()),
    )

    # Gate-1 diagnostic dump (env-gated, no training use): per-token unclipped
    # divergence for offline mean/median/p95 analysis and metric explanation.
    dump_path = os.environ.get("OPSD_KL_DUMP")
    if dump_path and per_token_raw.numel():
        import json as _json

        with torch.no_grad():
            vals = [float(v) for v in per_token_raw.detach().cpu()]
            # first-response-token diagnostics per sample: top-5 of both sides
            # distinguish structural (wrong tokens) from numerical (shifted
            # probability) divergence
            firsts = []
            off = 0
            for i, rlen in enumerate(response_lengths):
                t_logits = teacher_logits[off]
                s_logits = student_logits[off]
                # alignment probe: argmax at the first 64 response positions vs
                # the ACTUAL shared completion tokens (pinpoints any logits
                # stream offset by k positions)
                head = min(64, rlen)
                s_am = student_logits[off : off + head].argmax(dim=-1).tolist()
                t_am = teacher_logits[off : off + head].argmax(dim=-1).tolist()
                _completion = unconcat_tokens[i][-rlen:]
                actual = _completion[:head].tolist()
                off += rlen
                t_top = torch.topk(torch.softmax(t_logits / temperature, dim=-1), 5)
                s_top = torch.topk(torch.softmax(s_logits / temperature, dim=-1), 5)
                firsts.append(
                    {
                        "response_len": rlen,
                        "kl_first": vals[off - rlen],
                        "kl_last": vals[off - 1],
                        "teacher_top5": [
                            [int(t), float(p)] for t, p in zip(t_top.indices.tolist(), t_top.values.tolist())
                        ],
                        "student_top5": [
                            [int(t), float(p)] for t, p in zip(s_top.indices.tolist(), s_top.values.tolist())
                        ],
                        # alignment probe: argmax at the first 64 response
                        # positions vs the ACTUAL shared completion tokens
                        "student_argmax_head": s_am,
                        "teacher_argmax_head": t_am,
                        "actual_tokens_head": actual,
                    }
                )
        with open(dump_path, "a") as fh:
            fh.write(
                _json.dumps(
                    {
                        "n_tokens": len(vals),
                        "mean": sum(vals) / len(vals),
                        "sorted_p50": sorted(vals)[len(vals) // 2],
                        "sorted_p95": sorted(vals)[int(0.95 * len(vals)) - 1],
                        "sorted_max": max(vals),
                        "clipped_frac_gt_0.05": sum(v > 0.05 for v in vals) / len(vals),
                        "token_kls": vals,
                        "first_token_diag": firsts,
                    }
                )
                + "\n"
            )

    # Mask via slime's loss_masks; with --calculate-per-token-loss this is the
    # masked token sum (Megatron then divides by the masked token count).
    loss = sum_of_sample_mean(per_token)
    if per_token.numel() == 0:
        loss = loss + 0 * logits.sum()

    with torch.no_grad():
        mean_unclipped = per_token_raw.mean() if per_token_raw.numel() > 0 else torch.zeros((), device=logits.device)
        if per_token_raw.numel() > 0:
            clipped_fraction = (per_token_raw > token_clip).float().mean()
        else:
            clipped_fraction = torch.zeros((), device=logits.device)

    metrics = {
        "loss": loss.clone().detach(),
        "opsd/mean_unclipped_kl": mean_unclipped,
        "opsd/clipped_fraction": clipped_fraction,
        "opsd/num_response_tokens": torch.tensor(float(per_token.numel()), device=logits.device),
    }
    return loss, metrics
