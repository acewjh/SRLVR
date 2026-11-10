"""SRLVR-v2/v3: causal student versus same-snapshot Full HSI teacher.

The two variants differ only in feature(). Vocabulary centering is conditional
on the realized hindsight view, which depends on this trajectory's future and
outcome. It does not establish a zero-mean correction or an unbiased control
variate. C/D is empirical second-moment calibration, using prior batches only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from methods.srlvr import capture_full_gradient, distribution_summary, history_update, verify_dp_capture

logger = logging.getLogger(__name__)
LOG2 = math.log(2.0)
REDUCTION = "trajectory_token_mean_v1"


# Public alias retained for existing method callers.
from opsd_core.srlvr_settings import Settings, advantage_mode, batch_baseline_and_scale


@torch.no_grad()
def feature(lp, lq, variant):
    """Full-vocabulary f; absent mass in both views has a=1/2."""
    delta = lp - lq
    delta = torch.where(torch.isneginf(lp) & torch.isneginf(lq), torch.zeros_like(delta), delta)
    a = torch.sigmoid(delta)
    if variant == "v2":
        return a
    if variant != "v3":
        raise ValueError("SRLVR variant must be v2 or v3")
    # Only the logarithm argument is bounded to avoid 0*log(0). Endpoints
    # contribute exactly zero, without clipping the feature or probabilities.
    tiny = torch.finfo(a.dtype).tiny
    other = 1 - a
    entropy = -(a * a.clamp_min(tiny).log() + other * other.clamp_min(tiny).log())
    return 1 - entropy / LOG2


@torch.no_grad()
def centered_features(lp, lq, chosen, variant):
    """Exact p expectation, chosen f and e. No sampling/candidate approximation."""
    f = feature(lp, lq, variant)
    p = lp.exp()
    # Ascend supports FP32 vocabulary reductions, but no device FP64. CPU
    # acceptance checks use FP64; neither path retains trajectory logits.
    dtype = torch.float64 if lp.device.type == "cpu" else torch.float32
    mu = (p * f).sum(-1, dtype=dtype) / p.sum(-1, dtype=dtype)
    picked = f.gather(-1, chosen.long().unsqueeze(-1)).squeeze(-1)
    return mu, picked, picked.to(mu.dtype) - mu


def future_corrections(turn_features, gamma):
    """One reverse actor-token scan; B_t reads the suffix after turn t."""
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("gamma must satisfy 0 < gamma < 1")
    result = [0.] * len(turn_features)
    suffix = 0.
    for t in range(len(turn_features) - 1, -1, -1):
        result[t] = suffix
        for value in reversed(turn_features[t]):
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise ValueError("nonfinite or out-of-range centered feature")
            suffix = (1 - gamma) * value + gamma * suffix
    return result


def prepare_history(state, cfg, rollout_id):
    """Separate mean-score statistics from legacy sum-score checkpoint history.

    Old records contain R, squared score norm and the then-frozen b, but not
    the per-trajectory token count. They cannot be converted exactly. Reset
    method statistics only; keep checkpoint position and optimizer accounting.
    """
    mode = advantage_mode()
    if state is None:
        state = dict(b=.5, s=1., history=[], last_rollout=rollout_id - 1,
                     actor_optimizer_steps=rollout_id, advantage_mode=mode, batch_fallbacks=0,
                     ablations=cfg.ablations())
    if state.get("ablations", Settings(cfg.variant, cfg.gamma).ablations()) != cfg.ablations():
        raise RuntimeError("SRLVR resume changed ablation settings")
    if state.get("last_rollout", -1) != rollout_id - 1:
        raise RuntimeError("SRLVR history and generation snapshot are not consecutive")
    if state.get("variant", cfg.variant) != cfg.variant or state.get("gamma", cfg.gamma) != cfg.gamma:
        raise RuntimeError("SRLVR resume changed variant or frozen gamma")
    # The advantage mode is run identity: a history run resumed as `batch` (or the
    # reverse) would silently change the estimator mid-run.
    if state.get("advantage_mode", mode) != mode:
        raise RuntimeError(
            f"SRLVR resume changed advantage mode: {state['advantage_mode']!r} -> {mode!r}"
        )
    reduction = state.get("reduction")
    if reduction == REDUCTION:
        return state
    if reduction not in (None, "trajectory_token_sum_v1"):
        raise RuntimeError(f"unknown SRLVR history reduction: {reduction!r}")
    if state.get("history") or rollout_id:
        logger.warning("[srlvr-history-reset] rollout=%s reduction=%s -> %s; "
                       "reset b/s/history/lambda; preserve model, optimizer and step count",
                       rollout_id, reduction or "legacy_sum", REDUCTION)
    return dict(b=.5, s=1., history=[], lambda_raw=0., lambda_next=0.,
                last_rollout=rollout_id - 1, variant=cfg.variant, gamma=cfg.gamma,
                actor_optimizer_steps=state.get("actor_optimizer_steps", rollout_id),
                reduction=REDUCTION, advantage_mode=mode,
                ablations=cfg.ablations(),
                batch_fallbacks=state.get("batch_fallbacks", 0))


def trajectory_token_counts(metadata, masks, expected_positions):
    """Count full-trajectory active response tokens before any DP/micro split."""
    counts = {position: 0 for position in expected_positions}
    for meta, mask in zip(metadata, masks, strict=True):
        if meta.get("padding"):
            continue
        position = meta["position"]
        if position not in counts:
            raise RuntimeError("SRLVR trajectory position is outside the real batch")
        if any(keep not in (0, 1) for keep in mask):
            raise ValueError("SRLVR actor loss mask must be binary")
        counts[position] += int(sum(mask))
    if any(length <= 0 for length in counts.values()):
        raise ValueError("SRLVR trajectory has no effective actor tokens")
    return counts


def frozen_history(state, cfg, rollout_id):
    if state.get("last_rollout", -1) != rollout_id - 1:
        raise RuntimeError("SRLVR history and generation snapshot are not consecutive")
    if state.get("variant", cfg.variant) != cfg.variant or state.get("gamma", cfg.gamma) != cfg.gamma:
        raise RuntimeError("SRLVR resume changed variant or frozen gamma")
    b, s = float(state["b"]), float(state["s"])
    if not 0 < b < 1 or not math.isfinite(s) or s <= 0:
        raise RuntimeError("invalid historical baseline/scale")
    ratio = float(state.get("lambda_raw", 0.))
    if not math.isfinite(ratio):
        raise RuntimeError("invalid historical empirical coefficient")
    limit = cfg.rho * min(b, 1 - b)
    return b, s, max(-limit, min(limit, ratio))


def frozen_lambda_for_batch(state, cfg, rollout_id, b_effective):
    """Coefficient frozen at entry, clipped ONCE against the baseline in use.

    ``frozen_history`` returns the published-history clip. Applying that first
    and then re-clipping at a different effective baseline can only narrow the
    coefficient (history b=.01 -> limit .009 even when the batch b=.5 would
    allow .45). Read the un-clipped C/D ratio instead. ``history`` mode and the
    degenerate-batch fallback pass the historical b as ``b_effective``, so they
    reproduce the published value exactly.

    Returns ``(lambda, raw, limit)``; the calibration order is untouched: the
    ratio in ``state`` was frozen by the previous rollout and the current batch's
    C/D only feeds ``lambda_next``.
    """
    frozen_history(state, cfg, rollout_id)  # consecutiveness/variant/gamma checks
    raw = float(state.get("lambda_raw", 0.))
    if not math.isfinite(raw):
        raise RuntimeError("invalid historical empirical coefficient")
    if not (math.isfinite(b_effective) and 0 < b_effective < 1):
        raise RuntimeError("invalid effective baseline for the lambda bound")
    limit = cfg.rho * min(b_effective, 1 - b_effective)
    return max(-limit, min(limit, raw)), raw, limit


def next_history(state, records, cfg, c_hat, d_hat, rollout_id):
    if not math.isfinite(c_hat) or not math.isfinite(d_hat) or d_hat < 0:
        raise RuntimeError("invalid empirical second-moment calibration")
    result = history_update(state, records, cfg)
    ratio = c_hat / d_hat if d_hat > 0 else 0.
    if not math.isfinite(ratio):
        raise RuntimeError("nonfinite empirical C/D")
    result.update(lambda_raw=ratio, last_rollout=rollout_id, variant=cfg.variant,
                  gamma=cfg.gamma, actor_optimizer_steps=state.get("actor_optimizer_steps", 0) + 1,
                  reduction=REDUCTION, advantage_mode=advantage_mode(),
                  ablations=cfg.ablations(),
                  batch_fallbacks=int(state.get("batch_fallbacks", 0)))
    result["lambda_next"] = frozen_history(result, cfg, rollout_id + 1)[2]
    return result


def batch_rewards(trajectories, expected):
    """One terminal reward per REAL trajectory, ordered by task position.

    ``trajectories`` maps a task position to that episode's turn rows (built by
    ``fill_future`` after the DP gather). Rows repeat the episode reward, so
    reading ``srlvr_rows`` directly would count a multi-turn episode many times.
    Every real position must appear exactly once, and padded rows never do.
    """
    expected_positions = set(range(int(expected)))
    positions = set(trajectories)
    if positions != expected_positions:
        raise RuntimeError(
            "SRLVR batch statistics need every real trajectory exactly once: "
            f"missing={sorted(expected_positions - positions)} "
            f"extra={sorted(positions - expected_positions)}"
        )
    return [float(trajectories[position][0]["meta"]["terminal_reward"])
            for position in range(int(expected))]


def effective_advantage_stats(mode, rewards, b_history, s_history, cfg):
    """(b, s, source) actually used for this batch's advantages.

    ``history`` keeps the published calibration-window values. ``batch``
    recomputes the same two statistics over every real trajectory of this
    rollout; a degenerate batch falls back to history and reports why.
    """
    if mode == "history":
        return float(b_history), float(s_history), "history"
    return batch_baseline_and_scale(rewards, b_history, s_history,
                                    cfg.delta_s, cfg.eps_b0)


def clip_lambda(lam, b, cfg):
    """Clip the effective coefficient against the baseline actually in use."""
    limit = cfg.rho * min(b, 1 - b)
    return max(-limit, min(limit, lam))


def probe_advantage(reward, correction, b, s, which):
    """Calibration probe advantage: Z removes the baseline, Y isolates B."""
    if which == "Z":
        return -((reward - b) / s)
    if which == "Y":
        return correction / s
    raise ValueError(f"unknown calibration probe kind: {which!r}")


def score_norm(zz, reward, b, s):
    """W_i = ||S_i||^2 from ||Z_i||^2, with Z_i=(R_i-b) S_i/s."""
    if reward == b:
        raise RuntimeError("cannot recover the score norm when the reward equals the baseline")
    return zz * (s / (reward - b)) ** 2


def advance_fallbacks(prior, source, mode):
    """Cumulative count of batch statistic fallbacks (``history`` adds none)."""
    return int(prior) + int(mode == "batch" and source != "batch")


def validate_batch_mode(recorded, mode):
    """Fail closed when the mode changed between sampling and training."""
    if recorded != mode:
        raise RuntimeError(
            f"SRLVR advantage mode changed after sampling: sampled {recorded!r}, "
            f"training {mode!r}"
        )


def validate_runtime(args, cfg=None):
    cfg = cfg or Settings.from_env()
    for name in ("pipeline_model_parallel_size", "context_parallel_size", "expert_model_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"SRLVR exact scoring/calibration requires {name}=1")
    tp = int(getattr(args, "tensor_model_parallel_size", 1))
    if tp < 1:
        raise ValueError("SRLVR requires tensor_model_parallel_size >= 1")
    actor_gpus = int(getattr(args, "actor_num_nodes", 1)) * int(getattr(args, "actor_num_gpus_per_node", 1))
    if actor_gpus % tp:
        raise ValueError(f"SRLVR topology: actor GPUs {actor_gpus} not divisible by TP {tp}")
    if actor_gpus // tp < 1:
        raise ValueError("SRLVR topology: data-parallel size must be >= 1")
    if args.n_samples_per_prompt != 1 or not 2 <= cfg.calibration < args.rollout_batch_size:
        raise ValueError("SRLVR requires G=1 and 2 <= probes < real trajectories")
    if args.rollout_top_p != 1 or args.rollout_top_k != -1 or args.rollout_temperature <= 0:
        raise ValueError("current SRLVR sampler requires positive-temperature full softmax; sampling settings were not changed")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0").lower() not in ("0", "false"):
        raise ValueError("SRLVR needs actual temperature-scaled rollout logprobs")
    if any(getattr(args, k, 0) for k in ("attention_dropout", "hidden_dropout", "entropy_coef")) or float(os.environ.get("OPSD_LORA_DROPOUT", "0")):
        raise ValueError("SRLVR requires the existing zero-dropout, zero-entropy configuration")
    if getattr(args, "calculate_per_token_loss", False):
        raise ValueError("SRLVR compensates framework row normalization; token normalization must be disabled")
    if args.micro_batch_size < 1 or getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("SRLVR requires static microbatches")
    if getattr(args, "update_weights_interval", 1) != 1 or getattr(args, "keep_old_actor", False):
        raise ValueError("SRLVR scoring must use the current rollout snapshot")
    return cfg


def attach_rows(rows, episode, group_id, rollout_id, start_rollout_id=0):
    cfg = Settings.from_env()
    # The batch is filed under `rollout_id`, but the weights that actually
    # generated it are `rollout_id - stale_lag` when rollout is overlapped with
    # the actor update. The first rollout of an async run is generated before
    # any update, so it is on-policy even when the lag is armed.
    start = 0 if start_rollout_id is None else int(start_rollout_id)
    lag = cfg.stale_lag if rollout_id > start else 0
    policy_version = rollout_id - lag
    if policy_version < 0:
        raise RuntimeError("SRLVR stale lag exceeds the rollout counter")
    actor_end = 0
    for row, turn in zip(rows, episode.turns, strict=True):
        if row.response_length == 0:
            raise RuntimeError("SRLVR cannot silently drop an empty actor turn")
        view = row.metadata.get("hsi_teacher_view")
        if not view:
            raise RuntimeError("SRLVR row is missing its Full HSI teacher view")
        actor_end += sum(row.loss_mask)
        row.metadata["srlvr"] = dict(position=group_id, turn=turn.turn_idx,
            trajectory_id=f"{rollout_id}:{turn.episode_id}", policy_version=policy_version,
            calibration=group_id < cfg.calibration, terminal_reward=episode.reward,
            turn_end=actor_end, teacher_view=view, B=0., mu=[], f_chosen=[], e=[])


def convert(args, samples):
    cfg = validate_runtime(args)
    while samples and isinstance(samples[0], list):
        samples = [s for group in samples for s in group]
    if not samples or {s.metadata["srlvr"]["position"] for s in samples} != set(range(args.rollout_batch_size)):
        raise RuntimeError("SRLVR rollout lost real trajectories")
    real_rows = len(samples)
    counts = trajectory_token_counts([s.metadata["srlvr"] for s in samples],
                                     [s.loss_mask for s in samples], range(args.rollout_batch_size))
    for sample in samples:
        if len(sample.loss_mask) != sample.response_length:
            raise ValueError("SRLVR actor loss mask and response length disagree")
        sample.metadata["srlvr"]["actor_token_count"] = counts[sample.metadata["srlvr"]["position"]]
    dp = args.actor_num_gpus_per_node * args.actor_num_nodes
    # Static microbatches must consume every real row, including the final tail.
    multiple = dp * args.micro_batch_size
    source = min(samples, key=lambda s: len(s.tokens))
    for _ in range((-real_rows) % multiple):
        dummy = deepcopy(source)
        dummy.loss_mask = [0] * dummy.response_length
        dummy.metadata["srlvr"]["padding"] = True
        samples.append(dummy)
    return dict(tokens=[s.tokens for s in samples], response_lengths=[s.response_length for s in samples],
        loss_masks=[s.loss_mask for s in samples], rollout_log_probs=[s.rollout_log_probs for s in samples],
        rewards=[float(s.reward) for s in samples], raw_reward=[float(s.reward) for s in samples],
        truncated=[int(s.status.name == "TRUNCATED") for s in samples], sample_indices=list(range(len(samples))),
        turn_ids=[s.turn_ids for s in samples], opsd_advantages=[[0.] * s.response_length for s in samples],
        srlvr_rows=[s.metadata["srlvr"] for s in samples],
        srlvr_batch=dict(N=args.rollout_batch_size, rows=len(samples), real_rows=real_rows,
                         mode=advantage_mode(),
                         implementation="hsi", reduction=REDUCTION, config=asdict(cfg)),
        dynamic_global_batch_size=len(samples))


@torch.no_grad()
def _local_global_log_probs(hidden, weight, bias, temperature, start, vocab_size, group):
    """Full-vocabulary log p(v) for this rank's contiguous vocab shard.

    Local max/sum are combined with TP collectives; only [C, V/TP] logits exist.
    """
    from methods import tp_utils
    # Shard bounds are host integers: don't synchronize the NPU with
    # bool(valid.all()) on every position chunk. Skip padded projection rows.
    valid_rows = max(0, min(weight.size(0), int(vocab_size) - int(start)))
    if valid_rows:
        local_bias = bias[:valid_rows] if bias is not None else None
        logits = F.linear(hidden.to(weight.dtype), weight[:valid_rows], local_bias).float()
    else:
        # Keep a nonempty all-masked shard so max/gather are defined even if
        # this TP rank owns only padding. Other ranks supply the normalization.
        logits = torch.full((hidden.size(0), weight.size(0)), float("-inf"),
                            dtype=torch.float32, device=hidden.device)
    if temperature != 1.0:
        logits = logits.div(temperature)
    if group is None or torch.distributed.get_world_size(group) == 1:
        # A complete vocabulary needs no explicit max/exp/sum reduction chain.
        return torch.log_softmax(logits, dim=-1)
    local_max = logits.max(dim=-1).values
    global_max = tp_utils.tp_max_detached(local_max, group)
    local_sum = torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1)
    global_sum = tp_utils.tp_sum(local_sum, group, differentiable=False)
    return logits - (global_max + torch.log(global_sum)).unsqueeze(-1)


@torch.no_grad()
def _centered_features_sharded(lp, lq, targets, variant, start, vocab_size, group):
    """Exact full-vocab expectation/chosen/error/student-logp over shards."""
    from methods import tp_utils
    f = feature(lp, lq, variant)
    p = lp.exp()
    rows = lp.size(0)
    dtype = torch.float32
    local_pf = (p * f).sum(dim=-1, dtype=dtype)
    local_p = p.sum(dim=-1, dtype=dtype)
    local = targets - start
    in_range = (local >= 0) & (local < lp.size(-1)) & (targets < vocab_size)
    idx = local.clamp(0, max(lp.size(-1) - 1, 0)).unsqueeze(1)
    zeros = torch.zeros(rows, device=lp.device, dtype=dtype)
    picked_local = torch.where(in_range, f.gather(1, idx).squeeze(1), zeros)
    student_local = torch.where(in_range, lp.gather(1, idx).squeeze(1), zeros)
    # One small collective per chunk instead of four serialized reductions.
    total_pf, total_p, picked, student = tp_utils.tp_sum(
        torch.stack((local_pf, local_p, picked_local, student_local)),
        group, differentiable=False).unbind(0)
    mu = total_pf / total_p
    return mu, picked, (picked.to(mu.dtype) - mu), student


@torch.no_grad()
def score_row(actor, tokens, mask, view, cfg, temperature, vocab_size, context_limit=None):
    # Reuse HSI's hidden-state tap: no full trajectory logits are ever built.
    from methods.hsi2_jsd import _find_lm_head, _hidden_states, _lm_head_weight
    model = actor[0] if isinstance(actor, (list, tuple)) else actor
    while hasattr(model, "module"):
        model = model.module
    head = _find_lm_head(model)
    if head is None:
        raise RuntimeError("SRLVR cannot locate the same-snapshot lm head")
    from methods import tp_utils
    weight = _lm_head_weight(model, head)
    tp_size, tp_rank, tp_group = tp_utils.tp_info()
    vstart = tp_utils.vocab_start(weight.size(0), tp_rank, tp_size)
    ids, plen, n = view
    n = int(n)
    context_limit = int(context_limit or os.environ.get("SEQ_LENGTH", "32768"))
    if len(ids) > context_limit or len(tokens) > context_limit:
        raise ValueError(
            f"SRLVR scoring sequence exceeds context limit {context_limit}: "
            f"student={len(tokens)} teacher={len(ids)}; no context was truncated"
        )
    if n != len(mask) or n <= 0 or len(ids) != plen + n or list(ids[plen:]) != tokens[-n:]:
        raise RuntimeError("HSI target tokens do not align with the causal rollout target")
    start = len(tokens) - n - 1
    if start < 0 or plen < 1:
        raise RuntimeError("HSI response has no causal prefix")
    hn = _hidden_states(model, head, tokens)[start:start + n].clone()
    hq = _hidden_states(model, head, ids)[plen - 1:plen + n - 1].clone()
    if len(hn) != n or len(hq) != n:
        raise RuntimeError("HSI hidden-state positions are misaligned")
    # SGLang's logits processor discards padded vocabulary entries before
    # softmax. Trim padded rows on each local shard before projection, retaining
    # the model-native global support and original shard start for token IDs.
    bias = getattr(head, "bias", None)
    # The mask is already on the host. Avoid a device nonzero and a host
    # synchronization for indices on every output-head chunk.
    host_indices = [i for i, keep in enumerate(mask) if keep]
    active = torch.tensor(host_indices, dtype=torch.long, device=hn.device)
    target = torch.tensor(tokens[-n:], dtype=torch.long, device=hn.device)
    result = {key: [0.] * n for key in ("mu", "f_chosen", "e", "student_logp")}
    feature_chunks = []
    for offset in range(0, len(active), cfg.position_chunk):
        indices = active[offset:offset + cfg.position_chunk]
        # Project at most C positions. Each rank projects only its vocabulary
        # shard, then exact max/sum/expectation reductions run over TP.
        lp = _local_global_log_probs(
            hn[indices], weight, bias, temperature, vstart, vocab_size, tp_group
        )
        lq = _local_global_log_probs(
            hq[indices], weight, bias, temperature, vstart, vocab_size, tp_group
        )
        mu, chosen, error, student = _centered_features_sharded(
            lp, lq, target[indices], cfg.variant, vstart, vocab_size, tp_group
        )
        # Retain only four scalars per position, never full-vocabulary logits.
        # One row-level transfer lets the device queue successive chunks.
        feature_chunks.append(torch.stack((mu, chosen, error, student)))
        del lp, lq, mu, chosen, error, student
    if feature_chunks:
        host_values = torch.cat(feature_chunks, dim=1).cpu().tolist()
        for key, values in zip(result, host_values, strict=True):
            for index, value in zip(host_indices, values, strict=True):
                if not math.isfinite(value):
                    raise RuntimeError("nonfinite SRLVR full-vocabulary reduction")
                result[key][index] = value
    return result


def fill_future(rows, gamma, trajectory_uniform_b=False):
    trajectories = {}
    for row in rows:
        trajectories.setdefault(row["meta"]["position"], []).append(row)
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda r: r["meta"]["turn"])
        turns = [r["meta"]["turn"] for r in trajectory]
        if turns != list(range(len(turns))):
            raise RuntimeError("duplicated or missing turns after DP gather")
        errors = [[e for e, mask in zip(r["meta"]["e"], r["mask"], strict=True) if mask]
                  for r in trajectory]
        corrections = future_corrections(errors, gamma)
        if trajectory_uniform_b:
            lengths = [sum(r["mask"]) for r in trajectory]
            total = sum(lengths)
            if total <= 0:
                raise ValueError("SRLVR trajectory has no effective actor tokens")
            mean = math.fsum(length * value for length, value in
                             zip(lengths, corrections, strict=True)) / total
            # B is the single source read by the actor AND the Y/C/D probes.
            # Include the final turn even though its original suffix B is zero.
            corrections = [mean] * len(trajectory)
        for row, value in zip(trajectory, corrections, strict=True):
            row["meta"]["B"] = value
    return trajectories


def actor_loss(args, batch, logits, sum_of_sample_mean):
    """PPO token surrogate; drop transient TP head before checkpoint pickling.

    ``srlvr_compact`` stores a HeadRef (holding the TP ProcessGroup) on ``args``
    for the next forward.  Megatron's checkpoint validation pickles the args
    namespace, so leaving it there makes the first save fail with
    ``cannot pickle 'ReloadableProcessGroup'``.  The next forward rebuilds it;
    dropping the reference after each loss is safe (same fix as SRLVR-v4).
    """
    try:
        return _actor_loss_impl(args, batch, logits, sum_of_sample_mean)
    finally:
        if getattr(args, "_srlvr_head", None) is not None:
            args._srlvr_head = None
        if getattr(args, "_srlvr_full_loss_masks", None) is not None:
            args._srlvr_full_loss_masks = None


def _actor_loss_impl(args, batch, logits, sum_of_sample_mean):
    from tasks.alfworld.loss import log_probs_and_entropy, reference_kl
    from methods import srlvr_compact
    if srlvr_compact.enabled(args):
        if not getattr(args, "_srlvr_hidden", False):
            raise RuntimeError("selective SRLVR loss received projected logits, not hidden states")
        # Same response tokens, temperature, full-vocabulary softmax and masks;
        # only the prompt/padding rows of the lm_head output are not built.
        forward = srlvr_compact.log_probs_from_hidden(args, batch, logits)
    else:
        vocab = getattr(args, "vocab_size", None)
        if vocab:
            logits = logits[..., :vocab]
        _, forward = log_probs_and_entropy(logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
            with_entropy=False, max_seq_lens=batch.get("max_seq_lens"))
    loss = logits.sum() * 0
    reference_kl_sum = loss.detach().clone()
    if batch.get("srlvr_rows") is None:
        raise RuntimeError("SRLVR loss requires full-trajectory actor token counts")
    for row_index, (lp, old, advantage, mask, meta) in enumerate(zip(
            forward["log_probs"], batch["rollout_log_probs"], batch["opsd_advantages"],
            batch["loss_masks"], batch["srlvr_rows"], strict=True)):
        length = meta["actor_token_count"]
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("SRLVR loss requires a positive integer trajectory token count")
        a = advantage.detach().float()
        if getattr(args, "srlvr_calibration", False):
            loss = loss - (lp * a * mask).sum() / length
        else:
            ratio = (lp.float() - old.detach().float()).exp()
            clipped = ratio.clamp(1 - args.eps_clip, 1 + args.eps_clip_high)
            loss = loss - (torch.minimum(ratio * a, clipped * a) * mask).sum() / length
            refs = batch.get("ref_log_probs")
            kl = reference_kl(args, lp, refs[row_index] if refs is not None else None)
            reference_kl_sum = reference_kl_sum + (kl.detach() * mask).sum() / length
            loss = loss + getattr(args, "kl_loss_coef", 0.0) * (kl * mask).sum() / length
    # The framework applies DP/rows then Megatron applies 1/num_microbatches
    # and DP averaging. rows/N cancels the row denominator exactly once;
    # the per-trajectory token denominator is applied only above.
    scaled_loss = loss * args.srlvr_loss_scale
    # The framework averages metrics over rows too. Carry the same rows/N
    # correction so the final logs report the trajectory-mean objective.
    return scaled_loss, {"srlvr_loss_sum": scaled_loss.detach(),
                         "srlvr_reference_kl_sum": reference_kl_sum * args.srlvr_loss_scale}


def probe_accum_on_cpu():
    """Whether the calibration gradient vectors are staged in host memory."""
    value = os.environ.get("SRLVR_PROBE_ACCUM", "device").lower()
    if value not in ("device", "cpu"):
        raise ValueError("SRLVR_PROBE_ACCUM must be 'device' or 'cpu'")
    return value == "cpu"


@torch.no_grad()
def _accumulate_probe_gradient(model, entries, total, *, reconstruct_dp=True, profile=None):
    """Accumulate TP-local gradient slices without a full-vector host transfer.

    ``SRLVR_PROBE_ACCUM=cpu`` stages the two probe vectors in host memory
    instead of keeping a second copy of the fp32 gradient on each NPU. The
    slices and the dot products use fp32 arithmetic; only the device that
    holds the vectors changes. CPU staging avoids keeping both full vectors
    beside the model, activations and ``main_grad``. The current 64 GB cards
    may permit device staging, but that needs its own peak-memory measurement.

    ``reconstruct_dp=False`` is the isolated-DP probe path: DP finalization was
    skipped, so ``param.main_grad`` already holds this rank's *complete* raw
    gradient for its own trajectory and no DP all-gather may run here.
    """
    if reconstruct_dp:
        from methods.srlvr import _reconstruct_dp_gradients
        started = time.perf_counter()
        _reconstruct_dp_gradients(model)
        if profile is not None:
            profile["dp_sync_s"] = profile.get("dp_sync_s", 0.) + time.perf_counter() - started
    host = probe_accum_on_cpu()
    started = time.perf_counter()
    if total is None:
        values = []
        for param, _ in entries:
            grad = param.main_grad.detach().float().reshape(-1)
            values.append(grad.cpu().clone() if host else grad.clone())
        total = values
    else:
        for accumulator, (param, _) in zip(total, entries, strict=True):
            grad = param.main_grad.detach().float().reshape(-1)
            accumulator.add_(grad.cpu() if host else grad)
    if profile is not None:
        profile["copy_s"] = profile.get("copy_s", 0.) + time.perf_counter() - started
    return total


def _probe_terms(z, y):
    """One fused pass producing per-parameter zz/zy/yy (fp32 accumulation)."""
    if y is None:
        zz = torch.stack([value.pow(2).sum() for value in z])
        zero = torch.zeros_like(zz)
        return torch.stack((zz, zero, zero.clone()))
    if len(z) != len(y):
        raise RuntimeError("calibration probe Z/Y vectors are misaligned")
    zz, zy, yy = [], [], []
    for left, right in zip(z, y, strict=True):
        zz.append(left.pow(2).sum())
        zy.append((left * right).sum())
        yy.append(right.pow(2).sum())
    return torch.stack((torch.stack(zz), torch.stack(zy), torch.stack(yy)))


@torch.no_grad()
def _probe_moments(z, y, scales, tp_size, tp_group, profile=None):
    """Count each logical parameter once; send only scalar terms to the host."""
    started = time.perf_counter()
    terms = _probe_terms(z, y)
    # With SRLVR_PROBE_ACCUM=cpu the vectors (and these three per-parameter
    # sums) are host tensors; the TP reduction still runs on the accelerator.
    if terms.device != scales.device:
        terms = terms.to(scales.device)
    terms = terms * scales.unsqueeze(0)
    if tp_size > 1:
        torch.distributed.all_reduce(terms, group=tp_group)
    values = tuple(math.fsum(row) for row in terms.cpu().tolist())
    if profile is not None:
        profile["moments_s"] = profile.get("moments_s", 0.) + time.perf_counter() - started
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("nonfinite calibration gradient moments")
    return values


@torch.no_grad()
def _finalize_probe_tp_gradients(model):
    """Run only the tensor-parallel part of ``finalize_model_grads``.

    The isolated-DP probe deliberately skips the data-parallel reduce-scatter:
    each DP group owns one calibration trajectory and needs its raw local
    gradient. Tensor-parallel reductions are still required (for example the
    q/k layernorm grads and replicated adapters) and are reproduced exactly by
    the same helper the normal finalizer calls.
    """
    from megatron.core import mpu
    from megatron.core.distributed.finalize_model_grads import (
        _allreduce_non_tensor_model_parallel_grads, reset_model_temporary_tensors)
    from megatron.core.utils import get_model_config
    config = get_model_config(model[0])
    _allreduce_non_tensor_model_parallel_grads(
        model, config, mpu.get_tensor_model_parallel_group())
    reset_model_temporary_tensors(config, model)


PROBE_MODE_ENV = "SRLVR_PROBE_MODE"
PROBE_BENCH_ENV = "SRLVR_PROBE_BENCH"
PROBE_MODES = ("serial", "parallel", "auto")


@contextmanager
def probe_numerics():
    """Optional deterministic probe kernels, preserving the caller's settings.

    Default 0 leaves the original kernel policy unchanged. Explicit 1 is a
    diagnostic override; installed NPU attention kernels can still vary between
    repeats. Never change attention/parameter precision to hide that variation.
    """
    enabled = os.environ.get("SRLVR_PROBE_DETERMINISTIC", "0")
    if enabled not in ("0", "1"):
        raise ValueError("SRLVR_PROBE_DETERMINISTIC must be 0 or 1")
    if enabled == "0":
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    npu = getattr(torch, "npu", None)
    synchronize = (npu.synchronize if npu is not None
                   and npu.is_initialized() else lambda: None)
    synchronize()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        try:
            synchronize()
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def probe_mode():
    """Probe execution mode. It never changes estimator identity or checkpoints."""
    value = os.environ.get(PROBE_MODE_ENV, "auto").strip().lower()
    if value not in PROBE_MODES:
        raise ValueError(f"{PROBE_MODE_ENV} must be one of {PROBE_MODES}")
    return value


def probe_bench_enabled():
    return os.environ.get(PROBE_BENCH_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def probe_parallel_blockers(args):
    """Why the isolated-DP raw-gradient probe is unsafe for this runtime."""
    blockers = []
    if not getattr(args, "use_distributed_optimizer", False):
        blockers.append("use_distributed_optimizer is required for full local grad buffers")
    if getattr(args, "overlap_grad_reduce", False):
        blockers.append("overlap_grad_reduce would issue DP collectives inside backward")
    if int(getattr(args, "num_distributed_optimizer_instances", 1)) != 1:
        blockers.append("num_distributed_optimizer_instances must be 1")
    if getattr(args, "use_megatron_fsdp", False):
        blockers.append("megatron FSDP buffers are not supported by the raw probe path")
    return blockers


def resolve_probe_mode(args):
    """Return the effective mode; explicit parallel fails closed if unsafe."""
    requested = probe_mode()
    if requested == "serial":
        return "serial"
    blockers = probe_parallel_blockers(args)
    if blockers:
        if requested == "parallel":
            raise RuntimeError(
                "SRLVR probe parallel mode is unsafe here: " + "; ".join(blockers))
        logger.info("[srlvr-probe] serial fallback: %s", "; ".join(blockers))
        return "serial"
    return "parallel"


def assign_probe_positions(calibration, dp_rank, dp_size):
    """Round-robin calibration positions per DP group.

    Empty when ``dp_size > calibration`` (those ranks stay idle and only join
    the final scalar-record gather). Never assigns one position to two groups.
    """
    if calibration < 1 or dp_size < 1 or not 0 <= dp_rank < dp_size:
        raise ValueError(f"invalid calibration/DP assignment: n={calibration} "
                         f"rank={dp_rank} dp={dp_size}")
    return list(range(dp_rank, calibration, dp_size))


PROBE_GRAD_COPY_ENV = "SRLVR_PROBE_GRAD_COPY"
PROBE_ASSIGN_ENV = "SRLVR_PROBE_ASSIGN"


def probe_grad_copy_mode():
    value = os.environ.get(PROBE_GRAD_COPY_ENV, "per_trajectory").strip().lower()
    if value not in ("per_chunk", "per_trajectory"):
        raise ValueError("SRLVR_PROBE_GRAD_COPY must be per_chunk or per_trajectory")
    return value


def _probe_position_work(trajectory):
    """Expected probe work for one position: Z plus Y sequence tokens."""
    z = sum(len(r["tokens"]) for r in trajectory if any(r["mask"]))
    y = sum(len(r["tokens"]) for r in trajectory
            if any(r["mask"]) and r["meta"]["B"] != 0.)
    return z + y


def assign_balanced_probe_positions(trajectories, calibration, dp_rank, dp_size):
    """Pick the 2+2 pairing whose heavier DP group is cheapest.

    Only the execution assignment changes: every position is still probed
    exactly once, with the same Z/Y row sets and the same C/D aggregation.
    """
    if dp_size != 2 or calibration != 4:
        return assign_probe_positions(calibration, dp_rank, dp_size)
    work = [_probe_position_work(trajectories[i]) for i in range(calibration)]
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    def cost(pairing):
        left = sum(work[i] for i in pairing[0])
        right = sum(work[i] for i in pairing[1])
        return (max(left, right), pairing)
    return list(min(pairings, key=cost)[dp_rank])


def _gather_probe_records(records, dp_group, dp_size, profile=None):
    """Gather the small per-position scalar records once per rollout."""
    import torch.distributed as dist
    started = time.perf_counter()
    gathered = [None] * int(dp_size)
    dist.all_gather_object(gathered, records, group=dp_group)
    if profile is not None:
        profile["records_gather_s"] = profile.get("records_gather_s", 0.) + time.perf_counter() - started
    return [record for part in gathered for record in part]


def _finalize_probe_records(records, cfg, eff_b, eff_s):
    """Position-order C/D/W records; identical on every rank after the gather."""
    ordered = sorted(records, key=lambda record: record["position"])
    positions = [record["position"] for record in ordered]
    if positions != list(range(cfg.calibration)):
        raise RuntimeError(
            "SRLVR calibration records must cover every probe position exactly once: "
            f"{positions}")
    c_hat = sum(record["zy"] for record in ordered) / cfg.calibration
    d_hat = sum(record["yy"] for record in ordered) / cfg.calibration
    history = [[record["reward"],
                score_norm(record["zz"], record["reward"], eff_b, eff_s),
                eff_b] for record in ordered]
    return dict(c_hat=c_hat, d_hat=d_hat, records=history, positions=positions)


def _pad_probe_rows(rows, micro_batch_size):
    """Zero-mask fill rows only; never drops a real turn."""
    padded = list(rows)
    while len(padded) % micro_batch_size:
        padding = deepcopy(padded[0])
        padding["mask"] = [0] * len(padding["old"])
        padded.append(padding)
    return padded


def _select_probe_rows(probe_rows, which):
    """Turns carrying gradient for this probe; zero-Y falls through to an empty list."""
    if which not in ("Z", "Y"):
        raise ValueError(f"unknown calibration probe kind: {which!r}")
    return [row for row in probe_rows if any(row["mask"])
            and (which != "Y" or row["meta"]["B"] != 0.)]


def _probe_chunk_size(args):
    """Chunk length respecting static microbatch boundaries."""
    chunk = max(1, int(os.environ.get("SRLVR_PROBE_CHUNK", "5")))
    return max(args.micro_batch_size,
               chunk // args.micro_batch_size * args.micro_batch_size)


def _probe_trajectory_vectors(actor, rollout_id, position, trajectory, reward, which, entries,
                              eff_b, eff_s, device, *, isolated_dp, profile=None):
    """Raw Z or Y gradient for one calibration trajectory.

    The objective, masks, temperature, chunk accumulation order and padding are
    unchanged from the serial reference. Only the DP reduction differs:
    ``isolated_dp`` keeps the rank-local raw gradient and applies the TP-only
    part of finalization.
    """
    from megatron.core import mpu
    from megatron.core.utils import get_model_config
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.model import train

    from methods import srlvr_compact

    args = actor.args
    probe_rows = _pad_probe_rows(list(trajectory), args.micro_batch_size)
    selected = _select_probe_rows(probe_rows, which)
    if not selected:
        logger.info("[srlvr-probe-local] which=%s position=%s dp_rank=%s turns=0/%s "
                    "accum=%s isolated=%s", which, position, mpu.get_data_parallel_rank(),
                    len(probe_rows), "cpu" if probe_accum_on_cpu() else "device", isolated_dp)
        return None
    selected = _pad_probe_rows(selected, args.micro_batch_size)
    lengths = [len(row["tokens"]) for row in selected]
    responses = [len(row["old"]) for row in selected]
    multiple = args.data_pad_size_multiplier
    maxlen = (max(lengths) + multiple - 1) // multiple * multiple
    # micro_batch_size == 1 means every micro holds exactly one turn; pad each
    # turn to its own aligned length instead of the trajectory max.
    per_row = srlvr_compact.padded_lengths(args, lengths)
    max_seq_lens = per_row if per_row is not None else [maxlen] * len(selected)
    chunk = _probe_chunk_size(args)
    dp = mpu.get_data_parallel_world_size()
    config = get_model_config(actor.model[0])
    before = getattr(args, "srlvr_actual_optimizer_steps", 0)
    total = None
    chunks = 0
    for start in range(0, len(selected), chunk):
        rows = selected[start:start + chunk]
        idx = slice(start, start + len(rows))
        batch = dict(
            tokens=[torch.tensor(row["tokens"], device=device) for row in rows],
            loss_masks=[torch.tensor(row["mask"], device=device) for row in rows],
            rollout_log_probs=[torch.tensor(row["old"], device=device) for row in rows],
            total_lengths=lengths[idx], response_lengths=responses[idx],
            max_seq_lens=max_seq_lens[idx],
            srlvr_rows=[row["meta"] for row in rows],
            dynamic_global_batch_size=dp * len(rows),
            # Both probes use the SAME effective (b, s) as the actor advantage.
            opsd_advantages=[torch.full((n,), probe_advantage(reward, row["meta"]["B"],
                                                             eff_b, eff_s, which), device=device)
                             for row, n in zip(rows, responses[idx], strict=True)])
        iterator, micro = get_data_iterator(args, actor.model, batch)
        if len(micro) != 1 or micro[0] * args.micro_batch_size != len(rows):
            raise RuntimeError("raw score probe must accumulate the complete chunk")
        captured = []
        # Device-side accumulation is only valid for isolated-DP probes: the
        # serial path runs DP finalization per chunk, which would re-reduce an
        # already-accumulated buffer. Keep serial byte-for-byte unchanged.
        per_trajectory = isolated_dp and probe_grad_copy_mode() == "per_trajectory"

        def capture(model):
            nonlocal total
            if per_trajectory:
                # Gradients stay in main_grad; the trajectory is copied once
                # after the last chunk (see below).
                captured.append(True)
                return
            if isolated_dp:
                if config.finalize_model_grads_func is not None:
                    raise RuntimeError(
                        "isolated probe ran with DP finalization enabled; gradients would mix")
                sync_started = time.perf_counter()
                _finalize_probe_tp_gradients(model)
                if profile is not None:
                    profile["tp_sync_s"] = profile.get("tp_sync_s", 0.) + time.perf_counter() - sync_started
            if isolated_dp or probe_bench_enabled():
                total = _accumulate_probe_gradient(
                    model, entries, total, reconstruct_dp=not isolated_dp,
                    profile=profile)
            else:
                # Keep the published serial call shape (also used by tests that
                # substitute this helper) byte-for-byte compatible.
                total = _accumulate_probe_gradient(model, entries, total)
            captured.append(True)

        args.srlvr_loss_scale = len(rows)
        args.srlvr_calibration = True
        args.srlvr_probe_isolated_dp = isolated_dp
        args.srlvr_probe_skip_grad_zero = per_trajectory and chunks > 0
        args.srlvr_probe_keep_grad = per_trajectory
        args.srlvr_gradient_probe = capture
        callbacks = {key: getattr(config, key, None) for key in (
            "no_sync_func", "grad_sync_func", "param_sync_func",
            "finalize_model_grads_func")}
        srlvr_compact.memory_report(
            logger, f"probe {which} position={position} rows={start}:{start+len(rows)} before train")
        train_started = time.perf_counter()
        try:
            with probe_numerics():
                train(rollout_id, actor.model, actor.optimizer,
                      actor.opt_param_scheduler, iterator, micro)
        finally:
            if profile is not None:
                profile["train_s"] = profile.get("train_s", 0.) + time.perf_counter() - train_started
            args.srlvr_gradient_probe = None
            args.srlvr_calibration = False
            args.srlvr_probe_isolated_dp = False
            args.srlvr_probe_skip_grad_zero = False
            args.srlvr_probe_keep_grad = False
            for key, value in callbacks.items():
                setattr(config, key, value)
        if getattr(args, "srlvr_actual_optimizer_steps", 0) != before:
            raise RuntimeError("raw calibration probe stepped the actor optimizer")
        if len(captured) != 1:
            raise RuntimeError("probe chunk must have exactly one backward and no optimizer step")
        chunks += 1
        del captured, batch, iterator
        if os.environ.get("SRLVR_PROBE_EMPTY_CACHE", "0") == "1":
            empty_started = time.perf_counter()
            srlvr_compact.empty_cache()
            if profile is not None:
                profile["empty_cache_s"] = profile.get("empty_cache_s", 0.) + time.perf_counter() - empty_started
    if per_trajectory:
        if isolated_dp:
            sync_started = time.perf_counter()
            _finalize_probe_tp_gradients(actor.model)
            if profile is not None:
                profile["tp_sync_s"] = profile.get("tp_sync_s", 0.) + time.perf_counter() - sync_started
        total = _accumulate_probe_gradient(
            actor.model, entries, None, reconstruct_dp=not isolated_dp, profile=profile)
        for model_chunk in actor.model:
            model_chunk.zero_grad_buffer()
        actor.optimizer.zero_grad()
    if total is None:
        raise RuntimeError("probe had no turns")
    if profile is not None:
        profile["chunks"] = profile.get("chunks", 0) + chunks
        profile["turns"] = profile.get("turns", 0) + len(selected)
    logger.info("[srlvr-probe-local] which=%s position=%s dp_rank=%s turns=%s/%s "
                "chunks=%s accum=%s isolated=%s", which, position,
                mpu.get_data_parallel_rank(), len(selected), len(probe_rows), chunks,
                "cpu" if probe_accum_on_cpu() else "device", isolated_dp)
    return total


def _probe_vector_error(reference, value):
    """Diagnostic full-vector error, using bounded CPU float64 reductions."""
    if reference is None or value is None:
        if reference is None and value is None:
            return dict(relative_l2=0., max_abs=0., different=0)
        raise RuntimeError("probe reference and current zero-gradient status differ")
    if len(reference) != len(value):
        raise RuntimeError("probe gradient parameter counts differ")
    squares, reference_squares = [], []
    max_abs, different = 0., 0
    for old, new in zip(reference, value, strict=True):
        if old.numel() != new.numel():
            raise RuntimeError("probe gradient parameter shapes differ")
        for start in range(0, old.numel(), 1 << 20):
            a = old.reshape(-1)[start:start + (1 << 20)].cpu().double()
            b = new.reshape(-1)[start:start + (1 << 20)].cpu().double()
            delta = b - a
            squares.append(float(delta.square().sum()))
            reference_squares.append(float(a.square().sum()))
            max_abs = max(max_abs, float(delta.abs().max()))
            different += int(torch.count_nonzero(delta))
    return dict(relative_l2=math.sqrt(math.fsum(squares) /
                                    max(math.fsum(reference_squares), 1e-300)),
                max_abs=max_abs, different=different)


def _run_calibration_probes(actor, rollout_id, cfg, trajectories, eff_b, eff_s, *, isolated_dp,
                            entries, scales, tp_size, tp_group, device, profile):
    """Compute every position's C/D/W under either execution mode.

    ``isolated_dp=False`` reproduces the published serial reference exactly
    (every rank probes every position, DP gradients reconstructed per chunk).
    ``isolated_dp=True`` assigns positions round-robin to DP groups, keeps the
    rank-local raw gradient, and gathers only the four small scalar records.
    """
    from megatron.core import mpu
    from methods import srlvr_compact
    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    if not isolated_dp:
        positions = list(range(cfg.calibration))
    elif os.environ.get(PROBE_ASSIGN_ENV, "balanced").strip().lower() == "balanced":
        positions = assign_balanced_probe_positions(
            trajectories, cfg.calibration, dp_rank, dp_size)
    elif os.environ.get(PROBE_ASSIGN_ENV).strip().lower() == "roundrobin":
        positions = assign_probe_positions(cfg.calibration, dp_rank, dp_size)
    else:
        raise ValueError("SRLVR_PROBE_ASSIGN must be balanced or roundrobin")
    profile["positions"] = positions
    records = []
    for position in positions:
        trajectory = trajectories[position]
        reward = trajectory[0]["meta"]["terminal_reward"]
        z = _probe_trajectory_vectors(
            actor, rollout_id, position, trajectory, reward, "Z", entries, eff_b, eff_s,
            device, isolated_dp=isolated_dp, profile=profile)
        y = _probe_trajectory_vectors(
            actor, rollout_id, position, trajectory, reward, "Y", entries, eff_b, eff_s,
            device, isolated_dp=isolated_dp, profile=profile)
        moments_started = time.perf_counter()
        zz, zy, yy = _probe_moments(z, y, scales, tp_size, tp_group)
        profile["moments_s"] = profile.get("moments_s", 0.) + (
            time.perf_counter() - moments_started)
        records.append(dict(position=position, reward=reward,
                            zz=float(zz), zy=float(zy), yy=float(yy)))
        if os.environ.get("SRLVR_PROBE_VECTOR_COMPARE", "0") == "1":
            references = getattr(actor.args, "_srlvr_probe_references", None)
            if references is None:
                references = actor.args._srlvr_probe_references = {}
            if not isolated_dp and position % dp_size == dp_rank:
                # Retain only this DP group's assigned position. These are
                # diagnostic CPU tensors, never checkpointed or used by Adam.
                references[position] = (z, y)
            elif isolated_dp and position in references:
                compare_started = time.perf_counter()
                zr, yr = references.pop(position)
                error = dict(position=position, dp_rank=dp_rank,
                             tp_rank=mpu.get_tensor_model_parallel_rank(),
                             Z=_probe_vector_error(zr, z),
                             Y=_probe_vector_error(yr, y))
                logger.info("[srlvr-probe-vector-error] %s", json.dumps(error))
                profile["vector_compare_s"] = profile.get("vector_compare_s", 0.) + (
                    time.perf_counter() - compare_started)
                del zr, yr
        del z, y
        srlvr_compact.memory_report(logger, f"probe position={position} done")
        logger.info("[srlvr-mem] probe position=%s %s", position, srlvr_compact.peak_memory_text())
        empty_started = time.perf_counter()
        srlvr_compact.empty_cache()
        profile["empty_cache_s"] = profile.get("empty_cache_s", 0.) + time.perf_counter() - empty_started
    if isolated_dp:
        records = _gather_probe_records(
            records, mpu.get_data_parallel_group_gloo(), dp_size, profile=profile)
    return _finalize_probe_records(records, cfg, eff_b, eff_s)


def _probe_metric_values(stats, state, cfg, eff_b, eff_s, rollout_id):
    """Flatten one probe set to the scalar quantities the calibration feeds."""
    nxt = next_history(deepcopy(state), stats["records"], cfg,
                       stats["c_hat"], stats["d_hat"], rollout_id)
    values = dict(C_hat=stats["c_hat"], D_hat=stats["d_hat"], b_next=nxt["b"],
                  s_next=nxt["s"], lambda_raw_next=nxt["lambda_raw"],
                  lambda_next=nxt["lambda_next"])
    for position, record in enumerate(stats["records"]):
        values[f"W_{position}"] = record[1]
    return values


def _probe_bench_compare(serial, parallel, cfg, state, eff_b, eff_s, rollout_id):
    """Same-input numeric comparison of both probe paths (opt-in benchmark).

    Strict comparison is the default. A diagnostic caller can explicitly choose
    ``SRLVR_PROBE_BENCH_STRICT=0`` to record a mismatch without aborting. A
    measured same-path noise floor is useful for diagnosis, but is not proof
    that a cross-path error was caused by nondeterministic kernels.
    """
    left = _probe_metric_values(serial, state, cfg, eff_b, eff_s, rollout_id)
    right = _probe_metric_values(parallel, state, cfg, eff_b, eff_s, rollout_id)
    tolerance = float(os.environ.get("SRLVR_PROBE_BENCH_TOL", "5e-5"))
    report = dict(tolerance=tolerance)
    failures = []
    def check(name, serial_value, parallel_value):
        error = abs(serial_value - parallel_value)
        scale = max(abs(serial_value), abs(parallel_value), 1e-12)
        report[name] = dict(serial=serial_value, parallel=parallel_value,
                            abs_error=error, rel_error=error / scale)
        if error > tolerance * scale:
            failures.append(f"{name}: {serial_value:.12g} vs {parallel_value:.12g} "
                            f"(rel {error/scale:.3e})")
    for name in left:
        check(name, left[name], right[name])
    report["passed"] = not failures
    report["failures"] = failures
    report["strict"] = os.environ.get("SRLVR_PROBE_BENCH_STRICT", "1") == "1"
    logger.info("[srlvr-probe-bench] %s", json.dumps(report, allow_nan=False))
    if failures and report["strict"]:
        raise RuntimeError("calibration probe difference exceeds the requested comparison tolerance: "
                           + "; ".join(failures))
    return report


def _probe_control_compare(runs, cfg, state, eff_b, eff_s, rollout_id):
    """Cross-path difference relative to the same-path run-to-run noise floor.

    Order is serial, parallel, serial, parallel on the same frozen batch. NPU
    backward kernels are not run-to-run deterministic, so exact fp32 agreement
    cannot be required. The meaningful test is that the serial↔parallel spread
    is commensurate with the serial↔serial and parallel↔parallel spreads.
    """
    serial_runs = [run for run in runs if not run["isolated"]]
    parallel_runs = [run for run in runs if run["isolated"]]
    serial_values = [_probe_metric_values(run["stats"], state, cfg, eff_b, eff_s, rollout_id)
                     for run in serial_runs]
    parallel_values = [_probe_metric_values(run["stats"], state, cfg, eff_b, eff_s, rollout_id)
                       for run in parallel_runs]
    floor = float(os.environ.get("SRLVR_PROBE_BENCH_FLOOR", "1e-7"))
    factor = float(os.environ.get("SRLVR_PROBE_BENCH_NOISE_FACTOR", "3"))
    report = dict(order=[run["label"] for run in runs], floor=floor, noise_factor=factor)
    failures = []
    for name in serial_values[0]:
        serials = [value[name] for value in serial_values]
        parallels = [value[name] for value in parallel_values]
        serial_spread = max(abs(a - b) for a in serials for b in serials)
        parallel_spread = max(abs(a - b) for a in parallels for b in parallels)
        cross = max(abs(a - b) for a in serials for b in parallels)
        noise = max(serial_spread, parallel_spread, floor)
        report[name] = dict(serial=serials, parallel=parallels,
                            serial_spread=serial_spread, parallel_spread=parallel_spread,
                            cross=cross, noise=noise, cross_over_noise=cross / noise)
        if cross > factor * noise:
            failures.append(f"{name}: cross={cross:.3e} > {factor}x noise={noise:.3e}")
    serial_wall = [run["wall"] for run in serial_runs]
    parallel_wall = [run["wall"] for run in parallel_runs]
    report["timing"] = dict(
        serial_wall=serial_wall, parallel_wall=parallel_wall,
        speedup_mean=sum(serial_wall) / max(sum(parallel_wall), 1e-9),
        speedup_warm=serial_wall[-1] / max(parallel_wall[-1], 1e-9))
    report["passed"] = not failures
    report["failures"] = failures
    report["strict"] = os.environ.get("SRLVR_PROBE_BENCH_STRICT", "0") == "1"
    logger.info("[srlvr-probe-control] %s", json.dumps(report, allow_nan=False))
    if failures and report["strict"]:
        raise RuntimeError("calibration probe parallel path exceeds same-path noise: "
                           + "; ".join(failures))
    return report


def resolve_policy_gap(rows, rollout_id, max_stale):
    """Return ``(behavior_version, gap)`` for one complete batch.

    A batch must come from exactly one generation snapshot; overlapping rollout
    may put that snapshot one weight version behind the training step. Anything
    deeper than ``max_stale`` is rejected because the actor loss only truncates
    the importance ratio at ``eps_clip``, which is not a valid correction once
    the policies have diverged further.
    """
    versions = {int(r["meta"]["policy_version"]) for r in rows}
    if len(versions) != 1:
        raise RuntimeError(f"SRLVR mixed rollout snapshots: {sorted(versions)}")
    behavior_version = versions.pop()
    gap = int(rollout_id) - behavior_version
    if gap < 0:
        raise RuntimeError(
            f"SRLVR batch for rollout {rollout_id} claims future policy {behavior_version}")
    if gap > int(max_stale):
        raise RuntimeError(
            f"SRLVR stale policy gap {gap} exceeds max_stale={max_stale}")
    return behavior_version, gap


def train_batch(actor, rollout_id, data):
    import torch.distributed as dist
    from megatron.core import mpu
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.model import train

    args = actor.args
    ref_seconds = 0.
    cfg = validate_runtime(args)
    mode = advantage_mode()
    if data["srlvr_batch"]["config"] != asdict(cfg):
        raise RuntimeError("SRLVR configuration changed after sampling")
    validate_batch_mode(data["srlvr_batch"].get("mode"), mode)
    state = prepare_history(getattr(args, "srlvr_state", None), cfg, rollout_id)
    args.srlvr_state = state
    from methods import srlvr_compact
    row_padding_tokens = sum(data.get("max_seq_lens") or data["total_lengths"])
    legacy_padding_tokens = len(data["total_lengths"]) * (
        (max(data["total_lengths"]) + srlvr_compact.pad_multiple(args) - 1)
        // srlvr_compact.pad_multiple(args) * srlvr_compact.pad_multiple(args))
    logger.info("[srlvr-row-padding] rollout=%s padded_tokens=%s batch_max_tokens=%s",
                rollout_id, row_padding_tokens, legacy_padding_tokens)
    srlvr_compact.reset_peak_memory()
    srlvr_compact.memory_report(logger, f"train_batch entry rollout={rollout_id}")
    if getattr(args, "use_kl_loss", False):
        ref_started = time.perf_counter()
        iterator, micro = get_data_iterator(args, actor.model, data)
        actor._switch_model("ref")
        try:
            data.update(actor.compute_log_prob(iterator, micro, store_prefix="ref_"))
        finally:
            actor._switch_model("actor")
        srlvr_compact.memory_report(logger, f"after ref_log_probs rollout={rollout_id}")
        srlvr_compact.empty_cache()
        srlvr_compact.memory_report(logger, f"after ref empty_cache rollout={rollout_id}")
        ref_seconds = time.perf_counter() - ref_started
    # Validate the frozen history and take b/s. The coefficient itself is read
    # un-clipped so it can be bounded ONCE at this batch's effective baseline.
    b, s, _ = frozen_history(state, cfg, rollout_id)
    started = time.perf_counter()
    local = []
    # Every rank scores its own rows, before any backward/update. Gather only
    # scalar features and the causal rows required by the existing raw probes.
    # Weight publication merges LoRA before serving. Reuse that exact merge for
    # both scoring views, then restore the trainable adapters before raw probes.
    from methods.lora_megatron import merged_lora_sync
    with merged_lora_sync(actor.model):
        for i, meta in enumerate(data["srlvr_rows"]):
            if meta.get("padding"):
                continue
            tokens = data["tokens"][i].cpu().tolist()
            mask = data["loss_masks"][i].cpu().tolist()
            scored = score_row(actor.model, tokens, mask, meta["teacher_view"], cfg,
                               args.rollout_temperature, args.vocab_size,
                               context_limit=getattr(args, "seq_length", None))
            meta.update(scored)
            compact = {k: v for k, v in meta.items() if k != "teacher_view"}
            local.append(dict(meta=compact, tokens=tokens, mask=mask,
                              old=data["rollout_log_probs"][i].cpu().tolist()))
    srlvr_compact.memory_report(logger, f"after scoring rollout={rollout_id}")
    srlvr_compact.empty_cache()
    srlvr_compact.memory_report(logger, f"after scoring empty_cache rollout={rollout_id}")
    gathered = [None] * mpu.get_data_parallel_world_size()
    dist.all_gather_object(gathered, local, group=mpu.get_data_parallel_group_gloo())
    rows = sorted([r for part in gathered for r in part], key=lambda r: (r["meta"]["position"], r["meta"]["turn"]))
    behavior_version, policy_gap = resolve_policy_gap(rows, rollout_id, cfg.max_stale)
    logger.info("[srlvr-stale] rollout=%s behavior_policy=%s gap=%s max_stale=%s mode=%s",
                rollout_id, behavior_version, policy_gap, cfg.max_stale,
                "truncated_is" if policy_gap else "on_policy")
    trajectories = fill_future(rows, cfg.gamma, cfg.trajectory_uniform_b)
    if set(trajectories) != set(range(data["srlvr_batch"]["N"])):
        raise RuntimeError("SRLVR DP split lost real trajectories")
    counts = trajectory_token_counts([r["meta"] for r in rows], [r["mask"] for r in rows],
                                     range(data["srlvr_batch"]["N"]))
    for row in rows:
        row["meta"]["actor_token_count"] = counts[row["meta"]["position"]]
    lookup = {(r["meta"]["position"], r["meta"]["turn"]): r["meta"] for r in rows}
    for meta in data["srlvr_rows"]:
        if not meta.get("padding"):
            meta.update(lookup[(meta["position"], meta["turn"])])
    score_seconds = time.perf_counter() - started
    deltas = [lp - old for r in rows for lp, old, keep in zip(r["meta"]["student_logp"], r["old"], r["mask"], strict=True) if keep]
    logger.info("[srlvr-hsi-views] variant=%s rollout=%s p=causal q=%s temperature=%s support=%s bias=%s mae=%s",
                cfg.variant, rollout_id, "outcome_only" if cfg.teacher_outcome_only else "Full", args.rollout_temperature, args.vocab_size,
                sum(deltas) / max(1, len(deltas)), sum(abs(v) for v in deltas) / max(1, len(deltas)))
    if policy_gap:
        # The actor loss turns these log-ratios into the truncated importance
        # ratio exp(lp - old); log how far off-policy the batch actually is so a
        # diverging actor/rollout pair is visible before the clip hides it.
        low = math.log(max(1 - getattr(args, "eps_clip", 0.2), 1e-12))
        high = math.log(1 + getattr(args, "eps_clip_high", 0.2))
        clipped = sum(1 for value in deltas if value < low or value > high)
        logger.info("[srlvr-stale-ratio] rollout=%s gap=%s tokens=%s mean_ratio=%s clipped_frac=%s",
                    rollout_id, policy_gap, len(deltas),
                    math.exp(sum(deltas) / max(1, len(deltas))),
                    clipped / max(1, len(deltas)))

    # Effective baseline/scale for EVERY consumer of this batch: the calibration
    # probes, the recovered score norm, and the actor advantage. `history` keeps
    # the published calibration-window values; `batch` uses all real trajectories
    # (each reward once, never per turn/token/padded row). The frozen C/D ratio is
    # read UN-clipped and bounded exactly once at `eff_b`; the historical boundary
    # must not pre-narrow it when the effective baseline is more centred.
    rewards_now = batch_rewards(trajectories, data["srlvr_batch"]["N"])
    eff_b, eff_s, source = effective_advantage_stats(mode, rewards_now, b, s, cfg)
    lam, lambda_raw, lambda_limit = frozen_lambda_for_batch(state, cfg, rollout_id, eff_b)
    fallbacks = advance_fallbacks(state.get("batch_fallbacks", 0), source, mode)
    logger.info(
        "[srlvr-adv] mode=%s source=%s b_eff=%.6f s_eff=%.6f "
        "(history b=%.6f s=%.6f) lambda=%.6f lambda_raw=%.6f lambda_limit=%.6f "
        "trajectories=%d fallbacks=%d",
        mode, source, eff_b, eff_s, b, s, lam, lambda_raw, lambda_limit,
        len(rewards_now), fallbacks,
    )
    if not math.isfinite(eff_b) or not 0 < eff_b < 1 or not math.isfinite(eff_s) or eff_s <= 0:
        raise RuntimeError("invalid effective SRLVR baseline/scale")

    if not getattr(args, "srlvr_dp_verified", False):
        verify_dp_capture(actor.model, actor.optimizer)
        args.srlvr_dp_verified = True
    device = data["tokens"][0].device
    from methods.srlvr import _calibration_entries
    entries, tp_size, _, tp_group = _calibration_entries(actor.model)
    scales = torch.tensor([weight for _, weight in entries], device=device, dtype=torch.float32)
    probe_mode_used = resolve_probe_mode(args)
    probe_profiles = {"mode": probe_mode_used,
                      "deterministic": os.environ.get("SRLVR_PROBE_DETERMINISTIC", "0"),
                      "call": "parallel" if probe_mode_used == "parallel" else "serial"}
    bench_setting = os.environ.get(PROBE_BENCH_ENV, "0").strip().lower()
    bench_control = bench_setting in ("control", "noise", "spsp")
    bench_enabled = probe_bench_enabled()
    if (bench_enabled or bench_control) and not probe_parallel_blockers(args):
        order = ([("serial1", False), ("parallel1", True), ("serial2", False),
                  ("parallel2", True)] if bench_control
                 else [("serial", False), ("parallel", True)])
        runs = []
        for label, isolated in order:
            run_profile = {}
            started = time.perf_counter()
            run_stats = _run_calibration_probes(
                actor, rollout_id, cfg, trajectories, eff_b, eff_s, isolated_dp=isolated,
                entries=entries, scales=scales, tp_size=tp_size, tp_group=tp_group,
                device=device, profile=run_profile)
            elapsed = time.perf_counter() - started
            runs.append(dict(label=label, isolated=isolated, stats=run_stats,
                             wall=elapsed - run_profile.get("vector_compare_s", 0.),
                             profile=run_profile))
            logger.info("[srlvr-probe-run] %s", json.dumps(dict(
                label=label, isolated=isolated, wall_s=runs[-1]["wall"],
                deterministic=os.environ.get("SRLVR_PROBE_DETERMINISTIC", "0"),
                values=_probe_metric_values(run_stats, state, cfg, eff_b, eff_s, rollout_id),
                profile=run_profile), allow_nan=False))
        serial_walls = [run["wall"] for run in runs if not run["isolated"]]
        parallel_walls = [run["wall"] for run in runs if run["isolated"]]
        logger.info("[srlvr-probe-speed] serial=%s parallel=%s speedup_mean=%.3fx "
                    "speedup_warm=%.3fx", [round(value, 3) for value in serial_walls],
                    [round(value, 3) for value in parallel_walls],
                    sum(serial_walls) / max(sum(parallel_walls), 1e-9),
                    serial_walls[-1] / max(parallel_walls[-1], 1e-9))
        probe_profiles.update(
            runs={run["label"]: dict(wall_s=run["wall"], **run["profile"]) for run in runs})
        probe_profiles["bench"] = (
            _probe_control_compare(runs, cfg, state, eff_b, eff_s, rollout_id)
            if bench_control else
            _probe_bench_compare(runs[0]["stats"], runs[1]["stats"], cfg, state,
                                 eff_b, eff_s, rollout_id))
        # Use the LAST run of the selected mode so the chosen value is not the
        # cold-start sample when both modes executed twice.
        chosen = [run for run in runs
                  if run["isolated"] == (probe_mode_used == "parallel")][-1]
        stats, probe_seconds = chosen["stats"], chosen["wall"]
        if hasattr(args, "_srlvr_probe_references"):
            del args._srlvr_probe_references
    else:
        call_profile = {}
        call_started = time.perf_counter()
        stats = _run_calibration_probes(
            actor, rollout_id, cfg, trajectories, eff_b, eff_s,
            isolated_dp=probe_mode_used == "parallel",
            entries=entries, scales=scales, tp_size=tp_size, tp_group=tp_group,
            device=device, profile=call_profile)
        probe_seconds = time.perf_counter() - call_started
        probe_profiles[probe_mode_used] = dict(wall_s=probe_seconds, **call_profile)
    records, c_hat, d_hat = stats["records"], stats["c_hat"], stats["d_hat"]
    logger.info("[srlvr-probe-profile] %s",
                json.dumps(dict(probe_profiles, chosen=probe_mode_used, wall_s=probe_seconds),
                           allow_nan=False))
    srlvr_compact.memory_report(logger, f"probes done rollout={rollout_id}")
    srlvr_compact.empty_cache()
    srlvr_compact.memory_report(logger, f"after probes empty_cache rollout={rollout_id}")
    pending = next_history(state, records, cfg, c_hat, d_hat, rollout_id)
    pending["batch_fallbacks"] = fallbacks
    # Current real rows INCLUDING probes all use the coefficient frozen at entry
    # (re-clipped above against the effective baseline/scale of this batch).
    # This override is actor-only: every Z/Y probe and history update above ran
    # unchanged, including calibration of the next historical lambda.
    actor_lam = 0. if cfg.actor_lambda_zero else lam
    for index, meta in enumerate(data["srlvr_rows"]):
        advantage = 0. if meta.get("padding") else (meta["terminal_reward"] - eff_b - actor_lam * meta["B"]) / eff_s
        data["opsd_advantages"][index] = torch.full_like(data["opsd_advantages"][index], advantage)
    args.srlvr_loss_scale = data["srlvr_batch"]["rows"] / data["srlvr_batch"]["N"]
    iterator, micro = get_data_iterator(args, actor.model, data)
    if len(micro) != 1 or micro[0] * args.micro_batch_size != len(data["tokens"]):
        raise RuntimeError("actor must accumulate every local row before one optimizer step")
    steps_before = getattr(args, "srlvr_actual_optimizer_steps", 0)
    actor_started = time.perf_counter()
    from megatron.core.utils import get_model_config
    config = get_model_config(actor.model[0])
    callbacks = {key: getattr(config, key, None) for key in ("no_sync_func", "grad_sync_func", "param_sync_func")}
    srlvr_compact.memory_report(logger, f"actor update before train rollout={rollout_id}")
    try:
        train(rollout_id, actor.model, actor.optimizer, actor.opt_param_scheduler, iterator, micro)
    finally:
        for key, value in callbacks.items():
            setattr(config, key, value)
    srlvr_compact.memory_report(logger, f"actor update done rollout={rollout_id}")
    logger.info("[srlvr-mem] actor update peak rollout=%s %s",
                rollout_id, srlvr_compact.peak_memory_text())
    if getattr(args, "srlvr_actual_optimizer_steps", 0) != steps_before + 1:
        raise RuntimeError("SRLVR rollout did not execute exactly one successful actor update")
    # Commit statistics computed at theta_old only after the sole actor update.
    args.srlvr_state = pending
    actor.weights_backuper.backup("actor")
    if mpu.get_data_parallel_rank() == 0:
        outcomes = [trajectory[0]["meta"]["terminal_reward"] for trajectory in trajectories.values()]
        report = dict(algorithm=f"srlvr_{cfg.variant}", rollout=rollout_id,
            optimizer_steps=1, cumulative_optimizer_steps=pending["actor_optimizer_steps"],
            real_rollouts=len(outcomes), actor_tokens=sum(sum(r["mask"]) for r in rows),
            success_rate=sum(outcomes)/len(outcomes), b=b, s=s, lambda_value=actor_lam,
            lambda_calibrated=lam, lambda_raw_frozen=lambda_raw, lambda_limit=lambda_limit,
            ablations=cfg.ablations(),
            lambda_next=pending["lambda_next"], C_hat=c_hat, D_hat=d_hat,
            # Which (b, s) actually produced the advantages, and how they were
            # obtained. `b`/`s` above remain the historical calibration values.
            advantage_mode=mode, advantage_source=source,
            b_effective=eff_b, s_effective=eff_s,
            batch_rewards=len(rewards_now),
            batch_fallback_this_rollout=(mode == "batch" and source != "batch"),
            batch_fallbacks=pending["batch_fallbacks"],
            calibration="empirical_second_moment", probes=cfg.calibration,
            reduction=REDUCTION, trajectory_actor_tokens=counts,
            B=distribution_summary([r["meta"]["B"] for r in rows]),
            e=distribution_summary([e for r in rows for e, keep in zip(r["meta"]["e"], r["mask"], strict=True) if keep]),
            reference_seconds=ref_seconds, score_seconds=score_seconds, calibration_seconds=probe_seconds,
            calibration_mode=probe_mode_used, probe_profile=probe_profiles,
            padded_tokens=row_padding_tokens, batch_max_padding_tokens=legacy_padding_tokens,
            actor_seconds=time.perf_counter()-actor_started, config=asdict(cfg))
        with (Path(args.save).parent / "srlvr_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(report, allow_nan=False) + "\n")
        logger.info("[srlvr-hsi] %s", json.dumps(report, allow_nan=False))
