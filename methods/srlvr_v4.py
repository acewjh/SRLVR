"""SRLVR-v4: trajectory-level privileged teacher, per-turn distillation.

Why v4
------
v2/v3 build one Full-HSI teacher view *per turn* (the future after that turn)
and score every actor token against it.  That is exact, but the teacher forward
is paid once per turn and every turn's view has to be rendered separately.

v4 asks the same scientific question at trajectory granularity: after the
episode is finished, render **one** privileged view containing the complete
executed interaction trace and the terminal outcome (the same ``compact``
rendering available to v2/v3), run the frozen teacher over
``[privileged prompt | full continuation]`` **once**, and reuse the captured
hidden states for every turn's tokens.

The teacher is the frozen initial policy (``--hf-checkpoint``), the same
frozen-teacher contract as OPSD.  It is not the current actor: the teacher's
conditioning must not move while the student is trained.  The student is the
trainable LoRA policy on the *unprivileged* causal prompt.

What each turn sees
-------------------
The teacher prompt carries the privileged block (complete interaction trace +
outcome).  Its continuation is the RAW action-token stream
(``y_0 | y_1 | ... | y_{T-1}``, newline-separated), not the student's per-turn
prompts: those are re-rendered from the parsed record with past thinking
removed, so they are not token-level extensions of one another.  A turn's
teacher hidden state is therefore conditioned on the privileged block **and**
the preceding action tokens.  The student sees only its own causal prompt.  The
loss is the per-token generalized JSD (``beta=0`` -> ``KL(teacher || student)``)
over the turn's actor tokens, exactly the OPSD divergence, with OPSD's pointwise
token clip.

Efficiency
----------
* One teacher forward per trajectory, not per turn or per token.
* The teacher forward is hidden-state only: the lm-head is tapped, so no
  ``[T, V]`` logits tensor is ever allocated for the teacher sequence.
* Only the turn's positions are projected, then reduced with the student logits
  in bounded token chunks.

Reduction
---------
``--calculate-per-token-loss`` stays enabled for v4 (the launcher keeps it), so
``sum_of_sample_mean`` is slime's masked token sum and Megatron divides by the
global actor-token count.  That is the same global token-mean reduction OPSD
uses.

Failure policy
--------------
Everything that can silently mis-pair teacher and student tokens raises:
missing view, span/target disagreement, a teacher sequence over the context
limit, a vocab mismatch, or a topology the loss cannot reduce over exactly
(Tensor Parallel > 1 is supported only through the selective hidden-state path,
which performs the full-vocabulary reduction over the TP vocabulary shard).
No truncation and no silent fallback.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from methods.opsd_loss import generalized_jsd_per_token
from opsd_core.teacher import slice_response_logits

logger = logging.getLogger(__name__)

#: Frozen HF teacher (one per training process), and its hidden-state cache.
_TEACHER: dict[str, object] = {"model": None, "path": None}
_TEACHER_CACHE: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
_STATS = {"hits": 0, "misses": 0, "forwards": 0, "forward_seconds": 0.0}
_LOG_EVERY = 50


@dataclass(frozen=True)
class Settings:
    variant: str
    beta: float = 0.0
    token_clip: float = 0.05
    temperature: float = 1.0
    compact_future: bool = True
    teacher_cache: int = 8
    teacher_attn: str = "sdpa"
    token_chunk: int = 512
    max_context: int = 32768

    @classmethod
    def from_env(cls, args=None) -> "Settings":
        if os.environ.get("SRLVR_VARIANT", "") != "v4":
            raise ValueError("SRLVR-v4 requires SRLVR_VARIANT=v4")
        default_temp = getattr(args, "rollout_temperature", 1.0) if args is not None else 1.0
        values = dict(
            variant="v4",
            beta=float(os.environ.get("SRLVR_V4_BETA") or "0.0"),
            token_clip=float(os.environ.get("SRLVR_V4_TOKEN_CLIP") or "0.05"),
            temperature=float(os.environ.get("SRLVR_V4_TEMPERATURE") or str(default_temp)),
            compact_future=os.environ.get("SRLVR_V4_COMPACT_FUTURE", "1").lower()
            not in ("0", "false", "no"),
            teacher_cache=int(os.environ.get("SRLVR_V4_TEACHER_CACHE") or "8"),
            teacher_attn=os.environ.get("SRLVR_V4_TEACHER_ATTN") or "sdpa",
            token_chunk=int(os.environ.get("SRLVR_V4_TOKEN_CHUNK") or "512"),
            max_context=int(os.environ.get("SEQ_LENGTH") or "32768"),
        )
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.variant != "v4":
            raise ValueError("SRLVR_V4 settings require variant='v4'")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("SRLVR_V4_BETA must be in [0, 1]")
        if not (math.isfinite(self.token_clip) and self.token_clip > 0):
            raise ValueError("SRLVR_V4_TOKEN_CLIP must be positive")
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise ValueError("SRLVR_V4_TEMPERATURE must be positive")
        if self.teacher_cache < 1 or self.token_chunk < 1 or self.max_context < 2:
            raise ValueError("invalid SRLVR-v4 cache/chunk/context setting")
        if self.teacher_attn not in ("sdpa", "eager", "flash_attention_2"):
            raise ValueError("SRLVR_V4_TEACHER_ATTN must be sdpa, eager, or flash_attention_2")


def validate_runtime(args, cfg: Settings | None = None) -> Settings:
    """Fail closed before any rollout is paid for."""
    cfg = cfg or Settings.from_env(args)

    if getattr(args, "n_samples_per_prompt", 1) != 1:
        raise ValueError("SRLVR-v4 requires G=1 (one trajectory per sampled task)")
    if args.rollout_top_p != 1 or args.rollout_top_k != -1 or args.rollout_temperature <= 0:
        raise ValueError("SRLVR-v4 requires positive-temperature full-softmax sampling")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0").lower() not in ("0", "false"):
        raise ValueError("SRLVR-v4 needs actual temperature-scaled rollout log-probs")
    if any(
        getattr(args, k, 0)
        for k in ("attention_dropout", "hidden_dropout", "entropy_coef")
    ) or float(os.environ.get("OPSD_LORA_DROPOUT", "0")):
        raise ValueError("SRLVR-v4 requires zero dropout and zero entropy bonus")
    if not getattr(args, "calculate_per_token_loss", False):
        raise ValueError(
            "SRLVR-v4 uses the global actor-token mean; "
            "--calculate-per-token-loss must stay enabled"
        )
    for name in ("pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"SRLVR-v4 requires {name}=1")
    tp = int(getattr(args, "tensor_model_parallel_size", 1))
    if tp < 1:
        raise ValueError("SRLVR-v4 requires tensor_model_parallel_size >= 1")
    if getattr(args, "qkv_format", "thd") != "bshd":
        raise ValueError("SRLVR-v4 requires --qkv-format bshd")
    selective = os.environ.get("OPSD_SRLVR_SELECTIVE_LOGITS", "0").lower() in ("1", "true")
    if selective and getattr(args, "micro_batch_size", 1) != 1:
        raise ValueError("SRLVR-v4 selective logits require micro_batch_size=1")
    if tp > 1 and not selective:
        # TP shards the vocabulary, so the full-vocabulary JSD has to be an exact
        # sharded reduction. That reduction lives on the hidden-state path
        # (``_sharded_jsd_and_logp``); the stock ``[T, V]`` path would hand the
        # loss a rank-local vocabulary slice it would silently compare against a
        # global teacher.
        raise ValueError(
            "SRLVR-v4 with tensor_model_parallel_size>1 requires "
            "OPSD_SRLVR_SELECTIVE_LOGITS=1"
        )
    if tp > 1:
        vocab = int(getattr(args, "padded_vocab_size", 0) or getattr(args, "vocab_size", 0) or 0)
        if vocab and vocab % tp:
            # SGLang discards padded vocabulary rows before its softmax. When the
            # vocabulary does not divide across TP, one rank would carry padding
            # inside the sharded log-partition and the two sides would disagree.
            raise ValueError(
                f"SRLVR-v4 sharded full-vocabulary reduction needs vocab {vocab} "
                f"divisible by tensor_model_parallel_size {tp}"
            )
    if getattr(args, "micro_batch_size", 1) < 1:
        raise ValueError("SRLVR-v4 requires micro_batch_size >= 1")
    if getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("SRLVR-v4 requires static micro-batches")
    if not getattr(args, "hf_checkpoint", None):
        raise ValueError("SRLVR-v4 needs --hf-checkpoint for the frozen teacher")
    if getattr(args, "use_kl_loss", False) and getattr(args, "kl_loss_type", "low_var_kl") != "low_var_kl":
        raise ValueError("SRLVR-v4 initial-model KL requires low_var_kl")
    return cfg


# ---------------------------------------------------------------------------
# Frozen teacher
# ---------------------------------------------------------------------------


def _teacher_model(args):
    path = str(args.hf_checkpoint)
    if _TEACHER["model"] is not None and _TEACHER["path"] == path:
        return _TEACHER["model"]

    from transformers import AutoModelForCausalLM

    cfg = Settings.from_env(args)
    started = time.perf_counter()
    logger.info("[srlvr-v4] loading frozen teacher from %s (attn=%s)", path, cfg.teacher_attn)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.teacher_attn,
    )
    model.to("npu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _TEACHER.update(model=model, path=path)
    _TEACHER_CACHE.clear()
    logger.info(
        "[srlvr-v4] frozen teacher ready in %.1fs (vocab=%d device=%s)",
        time.perf_counter() - started,
        int(getattr(model.config, "vocab_size", 0)),
        next(model.parameters()).device,
    )
    return model


def _ids_key(ids: Iterable[int]) -> bytes:
    values = [int(v) for v in ids]
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(values).to_bytes(8, "little"))
    # 4-byte little-endian is enough for token ids and avoids a numpy dependency.
    digest.update(b"".join(int(v).to_bytes(4, "little", signed=False) for v in values))
    return digest.digest()


@torch.no_grad()
def teacher_hidden_states(args, ids: list[int], cfg: Settings | None = None) -> torch.Tensor:
    """``[len(ids), H]`` post-final-norm hidden states of the frozen teacher.

    The lm-head is temporarily tapped, so the teacher's transformer runs exactly
    once and the full-vocabulary logits are never materialised.  Results are
    cached by token-id content: every turn of one trajectory shares the same
    teacher sequence, so the cache turns T losses into one forward.
    """
    cfg = cfg or Settings.from_env(args)
    if len(ids) > cfg.max_context:
        raise ValueError(
            f"SRLVR-v4 teacher sequence is {len(ids)} tokens > context {cfg.max_context}; "
            "no context was truncated"
        )
    key = _ids_key(ids)
    cached = _TEACHER_CACHE.get(key)
    if cached is not None:
        _TEACHER_CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return cached

    model = _teacher_model(args)
    device = next(model.parameters()).device
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)
    captured: dict[str, torch.Tensor] = {}
    head = model.lm_head
    original_forward = head.forward

    def _tap(x, *unused, **kwargs):
        captured["hidden"] = x
        return x

    head.forward = _tap
    was_training = model.training
    started = time.perf_counter()
    try:
        model.eval()
        model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        head.forward = original_forward
        model.train(was_training)
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("SRLVR-v4 teacher lm-head tap captured no hidden states")
    if hidden.dim() == 3 and hidden.size(0) == 1:
        hidden = hidden[0]
    elif hidden.dim() != 2:
        raise RuntimeError(f"unexpected teacher hidden shape {tuple(hidden.shape)}")
    hidden = hidden.detach()

    _TEACHER_CACHE[key] = hidden
    while len(_TEACHER_CACHE) > cfg.teacher_cache:
        _TEACHER_CACHE.popitem(last=False)
    _STATS["misses"] += 1
    _STATS["forwards"] += 1
    _STATS["forward_seconds"] += time.perf_counter() - started
    return hidden


@torch.no_grad()
def teacher_logits(args, hidden: torch.Tensor) -> torch.Tensor:
    """Project a hidden-state slice to full-vocabulary fp32 logits."""
    model = _teacher_model(args)
    head = model.lm_head
    weight = head.weight
    bias = getattr(head, "bias", None)
    logits = F.linear(hidden.to(weight.dtype), weight, bias)
    return logits.float()


def _compact_student_source(args, batch, hidden):
    """Response hidden states, targets and head for the selective-logits path.

    ``srlvr_compact.hidden_forward`` returns the decoder's hidden states and
    defers the lm-head.  We select only the actor-response positions, so the
    prompt/padding rows never produce ``[S, V]`` logits.  The per-turn rows have
    an all-ones response mask, which makes ``response_index`` contiguous; a gap
    would mean the batch layout changed and is refused rather than mis-scattered.
    """
    from methods import srlvr_compact

    head = getattr(args, "_srlvr_head", None)
    if head is None:
        raise RuntimeError("SRLVR-v4 selective path has no captured output head")
    positions, targets, response_index, response = srlvr_compact._response_selection(
        batch, hidden
    )
    expected = torch.arange(response, device=response_index.device)
    if not torch.equal(response_index, expected):
        raise RuntimeError(
            "SRLVR-v4 compact path needs a contiguous actor response; "
            "per-turn rows have full masks, so a gap means the batch layout changed"
        )
    flat = hidden.reshape(-1, hidden.size(-1))
    return flat.index_select(0, positions), targets, response, head


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _student_projection(selected: torch.Tensor, head) -> torch.Tensor:
    """Project response hidden states against this rank's vocabulary shard.

    With TP > 1 the output layer is vocabulary-parallel: ``head.weight`` is the
    rank's contiguous shard, so the returned logits are the local slice. The
    hidden input is replicated, therefore its gradient is the *sum* over shards;
    ``tp_utils.grad_sum`` performs that one collective in backward.
    """
    from methods import tp_utils

    x = selected.to(head.weight.dtype)
    if head.tp_size > 1:
        x = tp_utils.grad_sum(x, head.tp_group)
    return F.linear(x, head.weight, head.bias).float()


def _sharded_jsd_and_logp(
    student: torch.Tensor,
    teacher: torch.Tensor,
    target_ids: torch.Tensor,
    head,
    cfg: "Settings",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact full-vocabulary KL(teacher || student) from a sharded student.

    ``teacher`` is the global full-vocabulary logits; ``student`` is this rank's
    [C, V/TP] slice and ``target_ids`` are the sampled token ids. Per token the
    vocabulary is split across ranks, so the local max / sum-of-exp / divergence
    are merged with collectives, and the chosen target log-probability is summed
    across the shards that hold it. Only local slices ever exist; the ``[C, V]``
    tensor is never materialised.

    Returns ``(clipped, raw, student_logp)`` — all [C], all replicated.
    """
    from methods import tp_utils

    group = head.tp_group
    temperature = float(cfg.temperature)
    student = student / temperature
    teacher = teacher / temperature
    if group is None or not tp_utils.dist.is_initialized():
        # Defensive: the single-rank case is handled by the dense path in the
        # caller, but never crash on a head that carries no group.
        dense_clipped, dense_raw = generalized_jsd_per_token(
            student, teacher, beta=cfg.beta, temperature=1.0, token_clip=cfg.token_clip
        )
        dense_logp = F.log_softmax(student, dim=-1).gather(
            1, target_ids.long().unsqueeze(1)
        ).squeeze(1)
        return dense_clipped, dense_raw, dense_logp

    # The teacher is complete on every rank; only the student side is sharded, so
    # the teacher slice for this rank's vocabulary columns is what pairs with it.
    cols = slice(head.vocab_start, head.vocab_start + student.size(-1))
    teacher_log_probs = F.log_softmax(teacher, dim=-1)[:, cols]
    # Global log-partition from local statistics: the max is a detached numerical
    # shift, the sum-of-exp is all-reduced with autograd so each shard's
    # contribution reaches the hidden-state gradient. No [C, V] tensor is built.
    local_max = student.max(dim=-1).values
    global_max = tp_utils.tp_max_detached(local_max, group)
    local_sum = torch.exp(student - global_max.unsqueeze(-1)).sum(dim=-1)
    # SHARED quantity: every rank's local divergence below depends on this same
    # global log-partition, so its backward must accumulate all ranks' partials.
    # ``tp_sum`` (pass-through backward) yields an exact value with a wrong
    # gradient here -- see tp_utils.SumGradFromTensorParallel.
    global_sum = tp_utils.tp_sum_shared(local_sum, group)
    student_log_probs = student - (global_max + torch.log(global_sum)).unsqueeze(-1)

    if cfg.beta == 0:
        local = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
    elif cfg.beta == 1:
        local = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
    else:
        beta_t = torch.tensor(cfg.beta, dtype=student.dtype, device=student.device)
        # Mixture membership is global (both p and q are global probabilities);
        # the mixture value at these columns only needs the two local slices.
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        kl_student = F.kl_div(
            mixture_log_probs, student_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
        local = beta_t * kl_teacher + (1 - beta_t) * kl_student

    jsd = tp_utils.tp_sum(local, group, differentiable=True)

    # Chosen-token log-probability: gather from the shard that owns the token.
    # ``target_logits`` is unused: the sampled ids travel with the chunk.
    targets = target_ids.to(student_log_probs.device).long()
    local_idx = targets - head.vocab_start
    owned = (local_idx >= 0) & (local_idx < student_log_probs.size(-1)) & (
        targets < head.vocab_size
    )
    picked = student_log_probs.gather(
        1, local_idx.clamp(0, student_log_probs.size(-1) - 1).unsqueeze(1)
    ).squeeze(1)
    student_logp = tp_utils.tp_sum(
        torch.where(owned, picked, torch.zeros_like(picked)), group, differentiable=True
    )

    per_token_raw = jsd
    per_token_clipped = jsd.clamp(max=cfg.token_clip) if cfg.token_clip is not None else jsd
    return per_token_clipped, per_token_raw, student_logp


def _row_view(batch, index: int):
    views = batch.get("hpi_views")
    prompts = batch.get("hpi_prompt_lengths")
    targets = batch.get("hpi_target_lengths")
    spans = batch.get("hpi_user_lengths")
    if not views or index >= len(views) or not views[index]:
        raise RuntimeError(f"SRLVR-v4 row {index} has no privileged teacher view")
    teacher_ids = views[index][0]
    if torch.is_tensor(teacher_ids):
        teacher_ids = teacher_ids.tolist()
    teacher_ids = [int(v) for v in teacher_ids]
    if not prompts or not prompts[index] or not targets or not targets[index]:
        raise RuntimeError(f"SRLVR-v4 row {index} has an incomplete teacher view")
    prompt_len = int(prompts[index][0])
    target_len = int(targets[index][0])
    span = spans[index] if spans is not None else None
    if isinstance(span, (list, tuple)) and span and isinstance(span[0], (list, tuple)):
        span = span[0]
    if not span or len(span) != 2:
        raise RuntimeError(f"SRLVR-v4 row {index} has no (span_start, span_len)")
    return teacher_ids, prompt_len, target_len, int(span[0]), int(span[1])


def _assert_alignment(
    index: int,
    teacher_ids: list[int],
    prompt_len: int,
    target_len: int,
    span_start: int,
    span_len: int,
    response_length: int,
    student_tokens,
) -> None:
    if span_len != response_length:
        raise RuntimeError(
            f"SRLVR-v4 row {index}: span length {span_len} != response length {response_length}"
        )
    if prompt_len < 1 or span_start < 0 or span_start + span_len > target_len:
        raise RuntimeError(
            f"SRLVR-v4 row {index}: invalid teacher span "
            f"(prompt={prompt_len}, start={span_start}, len={span_len}, target={target_len})"
        )
    if len(teacher_ids) != prompt_len + target_len:
        raise RuntimeError(
            f"SRLVR-v4 row {index}: teacher sequence {len(teacher_ids)} != "
            f"prompt {prompt_len} + target {target_len}"
        )
    teacher_target = teacher_ids[prompt_len + span_start : prompt_len + span_start + span_len]
    student_target = [int(v) for v in student_tokens]
    if teacher_target != student_target:
        raise RuntimeError(
            f"SRLVR-v4 row {index}: teacher completion tokens differ from the student "
            "response; refusing to mis-pair the distillation target"
        )


# ---------------------------------------------------------------------------
# slime custom loss
# ---------------------------------------------------------------------------


def actor_loss(args, batch, logits, sum_of_sample_mean):
    """Per-token privileged distillation; clears the transient output head.

    ``srlvr_compact`` publishes its ``HeadRef`` on ``args`` so the loss can
    project response positions.  That object holds the TP ``ProcessGroup``, which
    is not picklable; Megatron's checkpoint validation pickles the args namespace,
    so leaving it on ``args`` crashes the first save (measured 2026-10-27).  The
    next forward rebuilds it, so dropping the reference after each loss is safe.
    """
    try:
        return _actor_loss_impl(args, batch, logits, sum_of_sample_mean)
    finally:
        if getattr(args, "_srlvr_head", None) is not None:
            args._srlvr_head = None
        if getattr(args, "_srlvr_full_loss_masks", None) is not None:
            args._srlvr_full_loss_masks = None


def _actor_loss_impl(args, batch, logits, sum_of_sample_mean):
    """Per-token privileged distillation over each turn's actor tokens.

    Two student paths, one objective:

    * selective logits (the launcher default): ``logits`` is the decoder's
      hidden states; only the response positions are projected, in bounded
      token chunks, so no ``[S, V]`` prompt logits are ever formed.
    * stock forward: ``logits`` is the full ``[T, V]`` output and the response
      slice is used. Kept as a debugging/legacy path.
    """
    cfg = validate_runtime(args)
    from tasks.alfworld.loss import reference_kl

    response_lengths = [int(v) for v in batch["response_lengths"]]
    hidden_mode = bool(getattr(args, "_srlvr_hidden", False))
    vocab = int(getattr(args, "vocab_size", 0) or 0)
    head = None
    sharded = False

    if hidden_mode:
        _, _, _, first_span, _ = _row_view(batch, 0)
        first_mask = batch["loss_masks"][0] if batch.get("loss_masks") else None
        first_active = int(first_mask.sum()) > 0 if first_mask is not None else True
        if first_span < 0 or not first_active:
            # Overlong-teacher fallback: the row is kept for batch composition
            # but its loss is zero, so no response projection is needed.
            if len(response_lengths) != 1:
                raise RuntimeError("SRLVR-v4 compact path expects one row per microbatch")
            student_lengths = [response_lengths[0]]
            student_targets = [torch.empty(0, dtype=torch.long)]

            def student_chunk(start: int, stop: int, offset: int = 0) -> torch.Tensor:
                del start, stop, offset
                raise RuntimeError("skipped SRLVR-v4 row has no student projection")
        else:
            selected, targets, response, head = _compact_student_source(args, batch, logits)
            if len(response_lengths) != 1 or response != response_lengths[0]:
                raise RuntimeError("SRLVR-v4 compact path expects one row per microbatch")
            student_lengths = [response]
            student_targets = [targets]
            sharded = head.tp_size > 1
            vocab = int(head.vocab_size)

            def student_chunk(start: int, stop: int, offset: int = 0) -> torch.Tensor:
                del offset
                return _student_projection(selected[start:stop], head)
    else:
        # bshd slicing needs the microbatch's padded row length, exactly as OPSD.
        args._current_max_seq_lens = (
            batch.get("max_seq_lens", [0])[0] if batch.get("max_seq_lens") else 0
        )
        student_logits = slice_response_logits(
            args,
            logits,
            batch["unconcat_tokens"],
            batch["total_lengths"],
            response_lengths,
        )
        if vocab and student_logits.size(-1) != vocab:
            student_logits = student_logits[..., :vocab]
        student_lengths = response_lengths
        student_targets = [
            batch["unconcat_tokens"][i][-r:].detach().cpu()
            for i, r in enumerate(response_lengths)
        ]

        def student_chunk(start: int, stop: int, offset: int = 0) -> torch.Tensor:
            return student_logits[offset + start : offset + stop].float()

    references = batch.get("ref_log_probs")
    use_reference = bool(getattr(args, "use_kl_loss", False))
    if use_reference and references is None:
        raise RuntimeError("SRLVR-v4 initial-model KL is enabled but ref_log_probs are missing")

    clipped_parts, raw_parts, reference_parts = [], [], []
    offset = 0
    rows = 0
    tokens = 0
    forward_before = _STATS["forwards"]
    started = time.perf_counter()
    for index, response_length in enumerate(student_lengths):
        teacher_ids, prompt_len, target_len, span_start, span_len = _row_view(batch, index)
        row_mask = batch["loss_masks"][index] if batch.get("loss_masks") else None
        row_active = int(row_mask.sum()) > 0 if row_mask is not None else True
        if span_start < 0 or not row_active:
            # Overlong teacher view OR a zero-mask DP padding row: keep the row's
            # token count for the reducer, contribute exactly zero loss, and never
            # load/project its teacher or student sequence. The generic convert
            # padding copies a valid hpi view; the zero mask is what marks it.
            zero = torch.zeros(response_length, device=logits.device, dtype=torch.float32)
            clipped_parts.append(zero)
            raw_parts.append(zero)
            if use_reference:
                reference_parts.append(zero)
            offset += response_length
            tokens += response_length
            rows += 1
            continue
        targets_row = student_targets[index]
        student_tokens = [int(v) for v in targets_row.detach().cpu().tolist()]
        _assert_alignment(
            index,
            teacher_ids,
            prompt_len,
            target_len,
            span_start,
            span_len,
            response_length,
            student_tokens,
        )

        hidden = teacher_hidden_states(args, teacher_ids, cfg)
        teacher_start = prompt_len + span_start - 1
        teacher_stop = teacher_start + response_length
        if teacher_start < 0 or teacher_stop > hidden.size(0):
            raise RuntimeError(
                f"SRLVR-v4 row {index}: teacher hidden slice "
                f"[{teacher_start}, {teacher_stop}) outside [0, {hidden.size(0)})"
            )
        teacher_hidden = hidden[teacher_start:teacher_stop]

        for chunk_start in range(0, response_length, cfg.token_chunk):
            chunk_stop = min(chunk_start + cfg.token_chunk, response_length)
            student = student_chunk(chunk_start, chunk_stop, offset)
            teacher = teacher_logits(args, teacher_hidden[chunk_start:chunk_stop])
            target_chunk = targets_row[chunk_start:chunk_stop].to(student.device)
            if sharded:
                # Student logits are a vocabulary shard; the teacher is global.
                if teacher.size(-1) != student.size(-1) * head.tp_size:
                    raise RuntimeError(
                        f"SRLVR-v4 vocab mismatch under TP{head.tp_size}: teacher "
                        f"{teacher.size(-1)} student shard {student.size(-1)}"
                    )
                clipped, raw, student_logp = _sharded_jsd_and_logp(
                    student, teacher, target_chunk, head, cfg
                )
            else:
                if teacher.size(-1) != student.size(-1):
                    raise RuntimeError(
                        f"SRLVR-v4 vocab mismatch: teacher {teacher.size(-1)} "
                        f"student {student.size(-1)}"
                    )
                clipped, raw = generalized_jsd_per_token(
                    student,
                    teacher,
                    beta=cfg.beta,
                    temperature=cfg.temperature,
                    token_clip=cfg.token_clip,
                )
            clipped_parts.append(clipped)
            raw_parts.append(raw)
            if use_reference:
                if not sharded:
                    student_logp = F.log_softmax(
                        student / cfg.temperature, dim=-1
                    ).gather(1, target_chunk.long().unsqueeze(1)).squeeze(1)
                reference_parts.append(
                    reference_kl(
                        args,
                        student_logp,
                        references[index][chunk_start:chunk_stop].to(student.device),
                    )
                )
        offset += response_length
        tokens += response_length
        rows += 1

    if not hidden_mode and offset != sum(student_lengths):
        raise RuntimeError(
            f"SRLVR-v4 consumed {offset} student logits of {sum(student_lengths)}"
        )
    if not clipped_parts:
        loss = logits.sum() * 0
        zero = torch.zeros((), device=logits.device)
        return loss, {
            "srlvr_v4/mean_unclipped_kl": zero,
            "srlvr_v4/clipped_fraction": zero,
            "srlvr_v4/reference_kl": zero,
            "srlvr_v4/num_tokens": zero,
        }

    clipped = torch.cat(clipped_parts)
    raw = torch.cat(raw_parts)
    terms = clipped
    reference_mean = torch.zeros((), device=clipped.device)
    if reference_parts:
        reference = torch.cat(reference_parts)
        terms = terms + float(getattr(args, "kl_loss_coef", 0.0)) * reference
        reference_mean = reference.detach().mean()
    loss = sum_of_sample_mean(terms)

    forwarded = _STATS["forwards"] - forward_before
    elapsed = time.perf_counter() - started
    _STATS["calls"] = _STATS.get("calls", 0) + 1
    if rows and (_STATS["calls"] <= 3 or _STATS["calls"] % _LOG_EVERY == 0):
        logger.info(
            "[srlvr-v4] rows=%d tokens=%d mean_KL=%.5f clipped=%.4f ref_KL=%.5f "
            "teacher_forwards=%d elapsed=%.2fs cache hits/misses=%d/%d hidden=%s",
            rows,
            tokens,
            float(raw.detach().mean()),
            float((raw.detach() > cfg.token_clip).float().mean()),
            float(reference_mean),
            forwarded,
            elapsed,
            _STATS["hits"],
            _STATS["misses"],
            hidden_mode,
        )
    return loss, {
        "srlvr_v4/mean_unclipped_kl": raw.detach().mean(),
        "srlvr_v4/clipped_fraction": (raw.detach() > cfg.token_clip).float().mean(),
        "srlvr_v4/reference_kl": reference_mean.detach(),
        "srlvr_v4/num_tokens": torch.tensor(float(tokens), device=clipped.device),
    }
