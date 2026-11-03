"""Single-rollout, full-future turn correction (SRLVR).

Task execution, sampler transport, packing, Megatron backward and Adam remain
shared. Only this module defines the new estimator. Initial distributed support
is TP=PP=CP=EP=1, DP arbitrary; reject other layouts rather than count shards
incorrectly. The actor parameter space is the existing trainable LoRA space.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from opsd_core.srlvr_settings import advantage_mode, batch_baseline_and_scale

logger = logging.getLogger(__name__)


def enabled():
    return os.environ.get("OPSD_SRLVR", "0") == "1"


@dataclass(frozen=True)
class Settings:
    mode: str = "empirical"
    calibration: int = 4
    c: float = 10.0
    rho: float = 0.9
    fixed: float | None = None
    delta: float = 0.05
    gz: float | None = None
    gy: float | None = None
    bound_basis: str = ""
    window: int = 128
    eps_b0: float = 0.01
    delta_s: float = 0.1
    ratio_clip_low: float = 0.8
    ratio_clip_high: float = 1.2

    @classmethod
    def from_env(cls):
        values = {}
        for key, typ in (("mode", str), ("calibration", int), ("c", float),
                         ("rho", float), ("fixed", float), ("delta", float),
                         ("gz", float), ("gy", float), ("bound_basis", str),
                         ("window", int), ("eps_b0", float), ("delta_s", float),
                         ("ratio_clip_low", float), ("ratio_clip_high", float)):
            raw = os.environ.get("SRLVR_" + key.upper())
            if raw is not None and raw != "":
                values[key] = typ(raw)
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if self.mode not in ("zero", "fixed", "empirical", "certified"):
            raise ValueError("SRLVR_MODE must be zero, fixed, empirical or certified")
        if not (self.calibration >= 2 and self.window >= 1 and self.c > 0
                and 0 < self.rho < 1 and 0 < self.delta < 1
                and 0 < self.eps_b0 < 0.5 and self.delta_s > 0
                and 0 < self.ratio_clip_low <= 1 <= self.ratio_clip_high):
            raise ValueError("invalid SRLVR calibration/feature/history settings")
        if self.mode == "fixed" and (self.fixed is None or not math.isfinite(self.fixed)):
            raise ValueError("fixed mode requires explicit finite SRLVR_FIXED")
        if self.mode == "certified" and (
            self.gz is None or self.gy is None or not self.bound_basis.strip()
            or not math.isfinite(self.gz) or not math.isfinite(self.gy)
            or self.gz < 0 or self.gy < 0
        ):
            raise ValueError("certified requires valid population SRLVR_GZ, SRLVR_GY and SRLVR_BOUND_BASIS; sample maxima are not bounds")


def centered_features(probabilities, chosen, c):
    """Exact support expectation; zeros contribute zero, with f(0)=-1."""
    if c <= 0:
        raise ValueError("c must be positive")
    p = probabilities.float() if probabilities.dtype != torch.float64 else probabilities
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid sampler probabilities")
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=2e-6, rtol=2e-6):
        raise ValueError("sampler probabilities must be normalized")
    f = (p.log() / c).clamp(-1, 0)
    mu = (p * f).sum(-1)
    picked = f.gather(-1, chosen.long().unsqueeze(-1)).squeeze(-1)
    return mu, picked, picked - mu


def future_corrections(turn_features):
    """Reference O(TL) recurrence, using every strictly later actor token."""
    result = []
    for t in range(len(turn_features)):
        value = 0.0
        for suffix in turn_features[t + 1:]:
            for e in suffix:
                if not math.isfinite(e) or not -1.000001 <= e <= 1.000001:
                    raise ValueError("invalid centered sampler feature")
                value += 0.5 * (1 - abs(value)) * float(e)
        result.append(value)
    return result


def choose_lambda(cfg, b, n, c_hat, d_hat, q_hat):
    cfg.validate()
    limit = cfg.rho * min(b, 1 - b)
    if cfg.mode == "zero":
        return 0.0, {"zero_reason": "zero_mode"}
    if cfg.mode == "fixed":
        if abs(cfg.fixed) > limit:
            raise ValueError(f"SRLVR_FIXED={cfg.fixed} exceeds frozen bound {limit}")
        return cfg.fixed, {"zero_reason": "fixed_zero" if cfg.fixed == 0 else ""}
    if cfg.mode == "empirical":
        # Independent C trajectories estimate the variance-minimizing C/D.
        # This is a plug-in estimate, not a population-bound certificate.
        if not all(math.isfinite(x) for x in (c_hat, d_hat)) or d_hat < 0:
            raise ValueError("nonfinite/invalid empirical calibration statistics")
        value = max(-limit, min(limit, c_hat / d_hat)) if d_hat > 0 else 0.0
        return value, {"zero_reason": "D_zero" if d_hat == 0 else "C_zero" if value == 0 else ""}
    k = n // 2
    if n < 2:
        raise ValueError("certified needs at least two calibration trajectories")
    ell = math.log(8 / cfg.delta)
    rc = cfg.gz * cfg.gy * math.sqrt(2 * ell / n)
    rd = cfg.gy ** 2 * math.sqrt(ell / (2 * n))
    rq = cfg.gz ** 2 * math.sqrt(2 * ell / k)
    lc, ud, lq = max(abs(c_hat) - rc, 0), d_hat + rd, max(q_hat - rq, 0)
    info = dict(LC=lc, UD=ud, LQ=lq, rC=rc, rD=rd, rQ=rq, zero_reason="")
    if ud == 0 or lq == 0:
        info["zero_reason"] = "UD_zero" if ud == 0 else "LQ_zero"
        return 0.0, info
    value = min(lc / ud, limit, min(b, 1 - b) * math.sqrt(lq / ud))
    if value == 0:
        info["zero_reason"] = "LC_zero"
    return math.copysign(value, c_hat), info


def calibration_statistics(zs, ys):
    """Full vector reference, also used for the CPU numerical acceptance test."""
    if len(zs) != len(ys) or len(zs) < 2:
        raise ValueError("need at least two paired calibration gradients")
    dot = lambda x, y: torch.dot(x.double().reshape(-1), y.double().reshape(-1)).item()
    return (sum(dot(z, y) for z, y in zip(zs, ys)) / len(zs),
            sum(dot(y, y) for y in ys) / len(ys),
            sum(dot(zs[2*r], zs[2*r+1]) for r in range(len(zs)//2)) / (len(zs)//2))


def history_update(state, records, cfg):
    history = (state.get("history", []) + records)[-cfg.window:]
    total = sum(r[1] for r in history)
    b = state.get("b", 0.5)
    if total > 0:
        b = min(1 - cfg.eps_b0, max(cfg.eps_b0, sum(r[0]*r[1] for r in history) / total))
    s = math.sqrt(cfg.delta_s**2 + sum((r[0]-r[2])**2 for r in history) / len(history)) if history else state.get("s", 1.0)
    return dict(b=b, s=s, history=history)


def distribution_summary(values):
    if not values:
        return dict(count=0, mean=0.0, min=0.0, max=0.0, std=0.0)
    x = torch.tensor(values, dtype=torch.float64)
    return dict(count=len(values), mean=x.mean().item(), min=x.min().item(),
                max=x.max().item(), std=x.std(unbiased=False).item(),
                q10=x.quantile(.1).item(), q50=x.quantile(.5).item(), q90=x.quantile(.9).item())


def validate_runtime(args):
    cfg = Settings.from_env()
    # TP-aware calibration gathers sharded gradients in capture_full_gradient;
    # pipeline/context/expert parallelism remains unsupported for this path.
    for name in ("pipeline_model_parallel_size", "context_parallel_size",
                 "expert_model_parallel_size"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"SRLVR exact calibration currently requires {name}=1")
    if args.n_samples_per_prompt != 1 or args.rollout_batch_size <= cfg.calibration:
        raise ValueError("SRLVR requires G=1 and N=n+m with n>=2, m>=1")
    if (args.rollout_top_p != 1 or args.rollout_top_k != -1
            or args.rollout_temperature <= 0 or os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "0") not in ("0", "false", "False")):
        raise ValueError("strict SRLVR supports positive-temperature full-softmax sampler only")
    if any(getattr(args, k, 0) for k in ("attention_dropout", "hidden_dropout")) or float(os.environ.get("OPSD_LORA_DROPOUT", "0")) != 0:
        raise ValueError("SRLVR requires sampler/actor dropout alignment (dropout=0)")
    if getattr(args, "use_dynamic_batch_size", False) or args.micro_batch_size != 1:
        raise ValueError("SRLVR initial path requires micro_batch_size=1 and static microbatches")
    if getattr(args, "entropy_coef", 0):
        raise ValueError("SRLVR requires zero entropy coefficient")
    if os.environ.get("ALFWORLD_TASK_SAMPLING", "frontier") not in ("iid", "frontier"):
        raise ValueError("SRLVR requires IID sampling within the frozen batch distribution")
    return cfg


def attach_rows(rows, episode, group_id, rollout_id):
    """Metadata only: contexts/actor masks come from the shared episode path."""
    cfg = Settings.from_env()
    features = [t.sampler_e for t in episode.turns]
    corrections = future_corrections(features)
    for row, turn, correction in zip(rows, episode.turns, corrections, strict=True):
        if row.response_length == 0:
            raise RuntimeError("SRLVR cannot silently drop an empty actor turn")
        row.metadata["srlvr"] = dict(
            task_sample_id=f"{rollout_id}:{group_id}", trajectory_id=f"{rollout_id}:{turn.episode_id}",
            position=group_id, calibration=group_id < cfg.calibration,
            policy_version=rollout_id, sampler_version="ascend-fp32-full-softmax-v1",
            turn=turn.turn_idx, turn_end=sum(len(t.generated_token_ids) for t in episode.turns[:turn.turn_idx+1]),
            B=correction, mu=turn.sampler_mu, f_chosen=turn.sampler_f,
            e=turn.sampler_e, terminal_reward=episode.reward,
            termination_reason="success" if episode.reward else "truncated" if episode.trajectory.truncated else "failure",
        )


def convert(args, samples):
    """Preserve all rows; pad masks, never top up real environment attempts."""
    from copy import deepcopy
    from slime.utils.types import Sample
    cfg = validate_runtime(args)
    while samples and isinstance(samples[0], list):
        samples = [s for group in samples for s in group]
    if not samples:
        raise RuntimeError("empty SRLVR rollout")
    positions = {s.metadata["srlvr"]["position"] for s in samples}
    if positions != set(range(args.rollout_batch_size)):
        raise RuntimeError("task samples missing from single-rollout batch")
    dp = args.actor_num_gpus_per_node * args.actor_num_nodes
    real_rows = len(samples)
    for _ in range((-len(samples)) % dp):
        dummy = deepcopy(samples[0])
        dummy.loss_mask = [0] * dummy.response_length
        dummy.metadata["srlvr"] = {**dummy.metadata["srlvr"], "padding": True}
        samples.append(dummy)
    result = {
        "tokens": [s.tokens for s in samples],
        "response_lengths": [s.response_length for s in samples],
        "loss_masks": [s.loss_mask for s in samples],
        "rollout_log_probs": [s.rollout_log_probs for s in samples],
        "rewards": [float(s.reward) for s in samples],
        "raw_reward": [float(s.reward) for s in samples],
        "truncated": [int(s.status == Sample.Status.TRUNCATED) for s in samples],
        "sample_indices": list(range(len(samples))),
        "turn_ids": [s.turn_ids for s in samples],
        "opsd_advantages": [[0.] * s.response_length for s in samples],
        "srlvr_rows": [s.metadata["srlvr"] for s in samples],
        "srlvr_batch": dict(N=args.rollout_batch_size, rows=len(samples), real_rows=real_rows,
                            config=asdict(cfg)),
        "dynamic_global_batch_size": len(samples),
    }
    return result


def actor_loss(args, batch, logits, sum_of_sample_mean):
    """PPO token surrogate; raw score probes; global token sum / task count."""
    from tasks.alfworld.loss import log_probs_and_entropy, reference_kl
    _, forward = log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=False, max_seq_lens=batch.get("max_seq_lens"))
    logps = forward["log_probs"]
    loss = logits.sum() * 0
    reference_kl_sum = loss.detach().clone()
    for row_index, (lp, advantage, mask, old) in enumerate(zip(logps, batch["opsd_advantages"], batch["loss_masks"], batch["rollout_log_probs"], strict=True)):
        active = mask.bool()
        calibration = getattr(args, "srlvr_calibration", False)
        if active.any():
            delta = (lp.detach().float() - old.float())[active]
            bias, mae = delta.mean().abs().item(), delta.abs().mean().item()
            args.srlvr_alignment_bias_max = max(getattr(args, "srlvr_alignment_bias_max", 0.), bias)
            args.srlvr_alignment_mae_max = max(getattr(args, "srlvr_alignment_mae_max", 0.), mae)
            tolerance = float(os.environ.get("SRLVR_ALIGNMENT_TOL", "0.05"))
            if bias > tolerance or mae > tolerance:
                import torch.distributed as dist
                report = dict(bias=bias, mae=mae, max_error=delta.abs().max().item(),
                              temperature=args.rollout_temperature,
                              top_p=args.rollout_top_p, top_k=args.rollout_top_k,
                              tokens=batch["unconcat_tokens"][row_index].detach().cpu().tolist(),
                              response_length=int(batch["response_lengths"][row_index]),
                              sampler_logp=old.detach().float().cpu().tolist(),
                              actor_logp=lp.detach().float().cpu().tolist(),
                              loss_mask=mask.detach().cpu().tolist())
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                path = Path(args.save).parent / f"alignment_failure_rank{rank}.json"
                try:
                    path.write_text(json.dumps(report, allow_nan=False))
                except OSError:
                    logger.debug("could not persist SRLVR alignment diagnostic at %s", path)
                logger.warning("[srlvr-alignment] bias=%.6f mae=%.6f; raw calibration=%s, token PPO actor surrogate", bias, mae, calibration)
        if calibration:
            loss = loss - (lp * advantage.detach() * mask).sum()
        else:
            # Ratio carries the gradient exactly once. Do not multiply by lp:
            # that would add an unintended logp-dependent gradient factor.
            ratio = torch.exp(lp.float() - old.detach().float())
            low = float(os.environ.get("SRLVR_RATIO_CLIP_LOW", "0.8"))
            high = float(os.environ.get("SRLVR_RATIO_CLIP_HIGH", "1.2"))
            a = advantage.detach().float()
            surrogate = torch.minimum(ratio * a, ratio.clamp(low, high) * a)
            loss = loss - (surrogate * mask).sum()
            refs = batch.get("ref_log_probs")
            kl = reference_kl(args, lp, refs[row_index] if refs is not None else None)
            reference_kl_sum = reference_kl_sum + (kl.detach() * mask).sum()
            loss = loss + getattr(args, "kl_loss_coef", 0.0) * (kl * mask).sum()
            if active.any():
                args.srlvr_ratio_mean = getattr(args, "srlvr_ratio_mean", 0.) + ratio.detach()[active].sum().item()
                args.srlvr_ratio_count = getattr(args, "srlvr_ratio_count", 0) + active.sum().item()
                clipped = (ratio * a > ratio.clamp(low, high) * a) & active
                args.srlvr_clip_count = getattr(args, "srlvr_clip_count", 0) + clipped.sum().item()
    scale = args.srlvr_loss_scale
    return loss * scale, {"srlvr_raw_loss": loss.detach(), "srlvr_reference_kl_sum": reference_kl_sum}


# One DP all-gather stages at most this many output bytes. A whole-bucket
# gather needs a second copy of the bucket as collective staging (7.5 GiB for
# Qwen3-4B), which does not fit beside the model on a 32 GB card.
_DP_GATHER_CHUNK_BYTES = 64 * 1024 * 1024


def _param_is_tp_sharded(param, tp_size):
    """Whether ``param``'s gradient is split across TP ranks."""
    mode = getattr(param, "_srlvr_tp_mode", "")
    if tp_size <= 1 or mode == "replica":
        return False
    if mode == "shard":
        return True
    return bool(getattr(param, "tensor_model_parallel", False))


def _reconstruct_dp_gradients(model, chunk_bytes=_DP_GATHER_CHUNK_BYTES):
    """Rebuild each DP rank's full DP-reduced gradient buckets in place.

    Each DP rank intentionally probes the SAME trajectory. Reduce-scatter leaves
    only one valid shard of each bucket on a rank; gather those disjoint shards
    exactly once. Do not add DP replicas or read stale main_grad regions.

    The gather is chunked so the collective staging window stays bounded (a
    whole-bucket all_gather used to OOM the first update on 32 GB cards). The
    reconstructed bytes are identical to the unchunked gather.
    """
    import torch.distributed as dist
    from megatron.core import mpu
    group = mpu.get_data_parallel_group()
    dp, rank = mpu.get_data_parallel_world_size(), mpu.get_data_parallel_rank()
    if dp <= 1:
        return
    budget = max(1, int(chunk_bytes))
    for chunk in model:
        for buffer in chunk.buffers + chunk.expert_parallel_buffers:
            for bucket in buffer.buckets:
                if not chunk.ddp_config.use_distributed_optimizer:
                    continue
                grad_data = bucket.grad_data
                if grad_data is None or grad_data.numel() == 0:
                    continue
                shards = grad_data.chunk(dp)
                shard_len = shards[0].numel() if shards else 0
                if len(shards) != dp or any(s.numel() != shard_len for s in shards):
                    raise RuntimeError(
                        f"DP gradient bucket of {grad_data.numel()} elements does not "
                        f"split into {dp} equal shards")
                step = max(1, budget // max(1, dp * grad_data.element_size()))
                local = shards[rank]
                for start in range(0, shard_len, step):
                    end = min(shard_len, start + step)
                    pieces = [torch.empty_like(local[start:end]) for _ in range(dp)]
                    dist.all_gather(pieces, local[start:end].contiguous(), group=group)
                    for src, piece in enumerate(pieces):
                        grad_data[src * shard_len + start:src * shard_len + end].copy_(piece)


def _calibration_entries(model):
    """Trainable gradient slices and their TP counting weight.

    After :func:`_reconstruct_dp_gradients` every rank holds the complete DP
    gradient for its TP slice. Replicated parameters are counted on TP rank 0
    only; sharded parameters are counted on every TP rank. One TP all-reduce of
    the per-parameter scalar terms therefore counts each parameter exactly once.
    """
    from methods import tp_utils
    tp_size, tp_rank, tp_group = tp_utils.tp_info()
    entries = []
    for chunk in model:
        for param in chunk.parameters():
            if not param.requires_grad:
                continue
            sharded = _param_is_tp_sharded(param, tp_size)
            entries.append((param, 1.0 if (sharded or tp_rank == 0) else 0.0))
    return entries, tp_size, tp_rank, tp_group


def _scalar_terms(values, reference=None):
    """Per-parameter scalar reductions, still on the device."""
    if reference is None:
        return torch.stack([value.pow(2).sum() for value in values])
    return torch.stack([(value * other).sum() for value, other in zip(values, reference)])


def _scaled_total(terms, scales, tp_group, tp_size):
    """Sum per-parameter terms across TP without moving full gradients."""
    total = terms * scales
    if tp_size > 1:
        import torch.distributed as dist
        dist.all_reduce(total, group=tp_group)
    # Per-parameter fp32 sums are accurate; accumulate the ~400 of them in
    # float64 on the host rather than losing precision in one fp32 reduction.
    return math.fsum(total.detach().cpu().tolist())


def capture_calibration_probe(model, kind, entries, scales, tp_size, tp_group, reference=None):
    """One calibration forward/backward, reduced to scalars on device.

    ``kind == "Z"`` returns the cloned TP-local gradient slices (needed for
    z_i·y_i, z_i·z_j and the certified norm check). ``kind == "Y"`` returns
    y·y and z·y and releases y immediately. Unlike ``capture_full_gradient``
    this never all-gathers TP shards or copies the full LoRA gradient to CPU.
    """
    _reconstruct_dp_gradients(model)
    values = [param.main_grad.detach().float().reshape(-1) for param, _ in entries]
    if kind == "Z":
        norm_sq = _scaled_total(_scalar_terms(values), scales, tp_group, tp_size)
        return dict(values=[value.clone() for value in values], norm_sq=norm_sq)
    if reference is None:
        raise RuntimeError("Y calibration probe requires the stored Z gradient")
    yy = _scaled_total(_scalar_terms(values), scales, tp_group, tp_size)
    zy = _scaled_total(_scalar_terms(values, reference), scales, tp_group, tp_size)
    return dict(yy=yy, zy=zy)


def capture_full_gradient(model):
    """Full-vector gradient, kept for the CPU tests.

    Runtime probes do not call this: the DP acceptance check verifies the
    reconstructed shards in place (:func:`verify_dp_capture`), and calibration
    reduces the same gradients to scalars on the device
    (:func:`capture_calibration_probe`). Both avoid materialising the full
    fp32 vector beside the model.
    """
    _reconstruct_dp_gradients(model)
    import torch.distributed as dist
    from methods import tp_utils

    tp_size, tp_rank, tp_group = tp_utils.tp_info()

    def _tp_full_grad(param, grad):
        mode = getattr(param, "_srlvr_tp_mode", "")
        if tp_size <= 1 or mode == "replica":
            return grad
        if mode == "shard":
            dim = int(getattr(param, "_srlvr_tp_dim", 0))
        elif getattr(param, "tensor_model_parallel", False):
            dim = int(getattr(param, "partition_dim", 0))
        else:
            return grad
        pieces = [torch.empty_like(grad) for _ in range(tp_size)]
        dist.all_gather(pieces, grad.contiguous(), group=tp_group)
        if grad.dim() == 1:
            return torch.cat(pieces, dim=0)
        return torch.cat(pieces, dim=dim)

    values = []
    for chunk in model:
        for p in chunk.parameters():
            if not p.requires_grad:
                continue
            g = p.main_grad.detach().float()
            g = _tp_full_grad(p, g)
            values.append(g.cpu().reshape(-1).clone())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite calibration gradient")
    return result


def _max_abs_error(tensor, expected, chunk_elements):
    """Absolute error over ``tensor`` with a bounded fp32 staging window."""
    flat = tensor.reshape(-1)
    worst = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        delta = (flat[start:start + chunk_elements].float() - expected).abs().max()
        worst = max(worst, float(delta.item()))
    return worst


def verify_dp_capture(model, optimizer):
    """First-batch acceptance: known full gradient through actual DP shards.

    Every DP rank injects a distinct constant, reduce-scatter averages it, and
    the reconstruction must yield the known mean. The check reads each
    trainable parameter's slice in bounded chunks after the in-place
    reconstruction instead of materialising the full fp32 vector: the union
    over DP and TP ranks covers exactly the elements the old full-vector
    comparison covered, without the 7.5 GiB gather staging or the 16 GiB host
    vector that used to OOM the first update.
    """
    from megatron.core import mpu
    from methods import tp_utils
    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    for chunk in model:
        chunk.zero_grad_buffer()
        for param in chunk.parameters():
            if param.requires_grad:
                param.main_grad.fill_(1 + dp_rank)
        chunk.finish_grad_sync()
    _reconstruct_dp_gradients(model)
    expected = (1 + dp_size) / 2
    chunk_elements = max(1, _DP_GATHER_CHUNK_BYTES // 4)
    tp_size = tp_utils.tp_info()[0]
    error, parameters = 0.0, 0
    for chunk in model:
        for param in chunk.parameters():
            if not param.requires_grad:
                continue
            numel = param.main_grad.numel()
            parameters += numel * tp_size if _param_is_tp_sharded(param, tp_size) else numel
            error = max(error, _max_abs_error(param.main_grad, expected, chunk_elements))
    if error > 1e-6:
        raise RuntimeError(f"DP gradient shard reconstruction failed: max_error={error}")
    for chunk in model:
        chunk.zero_grad_buffer()
    optimizer.zero_grad()
    logger.info("[srlvr-dp-check] parameters=%d expected=%s max_error=%s chunk_bytes=%d",
                parameters, expected, error, _DP_GATHER_CHUNK_BYTES)


def train_batch(actor, rollout_id, data):
    """Shared Megatron backward for probes and one final shared optimizer step."""
    import torch.distributed as dist
    from megatron.core import mpu
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.model import train

    args = actor.args
    cfg = validate_runtime(args)
    if getattr(args, "use_kl_loss", False):
        iterator, micro = get_data_iterator(args, actor.model, data)
        actor._switch_model("ref")
        try:
            data.update(actor.compute_log_prob(iterator, micro, store_prefix="ref_"))
        finally:
            actor._switch_model("actor")
    if not getattr(args, "srlvr_dp_verified", False):
        verify_dp_capture(actor.model, actor.optimizer)
        args.srlvr_dp_verified = True
    if args.calculate_per_token_loss:
        raise RuntimeError("SRLVR needs row normalization compensation, not token averaging")
    t0 = time.perf_counter()
    state = getattr(args, "srlvr_state", dict(b=0.5, s=1., history=[]))
    if state.get("last_rollout", rollout_id-1) != rollout_id-1:
        raise RuntimeError("SRLVR historical state/policy version is not consecutive")
    b, s = state["b"], state["s"]
    # Only zero/fixed lambda is chosen before observing C.
    frozen_fixed = choose_lambda(cfg, b, cfg.calibration, 0., 0., 0.)[0] if cfg.mode in ("zero", "fixed") else None
    for name in ("alignment_bias_max", "alignment_mae_max", "ratio_mean", "ratio_count", "clip_count"):
        setattr(args, "srlvr_" + name, 0.)
    local = []
    for i, meta in enumerate(data["srlvr_rows"]):
        if not meta.get("padding"):
            local.append(dict(meta=meta, tokens=data["tokens"][i].cpu().tolist(),
                              old=data["rollout_log_probs"][i].cpu().tolist(),
                              mask=data["loss_masks"][i].cpu().tolist()))
    gathered = [None] * mpu.get_data_parallel_world_size()
    dist.all_gather_object(gathered, local, group=mpu.get_data_parallel_group_gloo())
    rows = sorted([r for rank_rows in gathered for r in rank_rows],
                  key=lambda r: (r["meta"]["position"], r["meta"]["turn"]))
    comm_seconds = time.perf_counter() - t0
    if any(r["meta"]["policy_version"] != rollout_id for r in rows):
        raise RuntimeError("mixed policy versions")
    if {r["meta"]["position"] for r in rows} != set(range(data["srlvr_batch"]["N"])):
        raise RuntimeError("lost task samples during DP split")
    by_position = [[r for r in rows if r["meta"]["position"] == pos] for pos in range(cfg.calibration)]
    entries, tp_size, tp_rank, tp_group = _calibration_entries(actor.model)
    device = data["tokens"][0].device
    scales = torch.tensor([weight for _, weight in entries], device=device, dtype=torch.float32)
    zs, ys, history = [], [], []
    probe_t0 = time.perf_counter()
    for trajectory in by_position:
        reward = trajectory[0]["meta"]["terminal_reward"]
        def probe(which, reference=None):
            lengths = [len(r["tokens"]) for r in trajectory]
            responses = [len(r["old"]) for r in trajectory]
            dp = mpu.get_data_parallel_world_size()
            pad = args.data_pad_size_multiplier
            maxlen = (max(lengths) + pad - 1) // pad * pad
            batch = dict(tokens=[torch.tensor(r["tokens"], device=device) for r in trajectory],
                         loss_masks=[torch.tensor(r["mask"], device=device) for r in trajectory],
                         rollout_log_probs=[torch.tensor(r["old"], device=device) for r in trajectory],
                         total_lengths=lengths, response_lengths=responses,
                         max_seq_lens=[maxlen]*len(lengths),
                         dynamic_global_batch_size=dp*len(lengths),
                         opsd_advantages=[torch.full((n,), -((reward-b)/s if which == "Z" else r["meta"]["B"]/s), device=device)
                                          for r, n in zip(trajectory, responses)])
            iterator, micro = get_data_iterator(args, actor.model, batch)
            captured = []
            args.srlvr_loss_scale = len(trajectory)
            args.srlvr_calibration = True
            args.srlvr_gradient_probe = lambda model: captured.append(
                capture_calibration_probe(model, which, entries, scales, tp_size, tp_group, reference))
            started = time.perf_counter()
            try:
                train(rollout_id, actor.model, actor.optimizer, actor.opt_param_scheduler, iterator, micro)
            finally:
                args.srlvr_gradient_probe = None
                args.srlvr_calibration = False
            if len(captured) != 1:
                raise RuntimeError("calibration must perform one full trajectory backward, no optimizer step")
            logger.info("[srlvr-probe] rollout=%s traj=%d which=%s seconds=%.1f",
                        rollout_id, len(zs), which, time.perf_counter()-started)
            return captured[0]
        z = probe("Z")
        y = probe("Y", reference=z["values"])
        zs.append(z); ys.append(y)
        score_norm = z["norm_sq"] * (s/(reward-b))**2
        history.append([reward, score_norm, b])
    c_hat = math.fsum(y["zy"] for y in ys) / len(ys)
    d_hat = math.fsum(y["yy"] for y in ys) / len(ys)
    pairs = [_scaled_total(_scalar_terms(zs[2*r+1]["values"], reference=zs[2*r]["values"]),
                           scales, tp_group, tp_size)
             for r in range(len(zs)//2)]
    q_hat = math.fsum(pairs) / len(pairs) if pairs else 0.0
    logger.info("[srlvr-calibration] rollout=%s C=%s D=%s Q=%s", rollout_id, c_hat, d_hat, q_hat)
    if cfg.mode == "certified":
        # Observed violations disprove supplied bounds, but passing cannot prove them.
        if any(math.sqrt(z["norm_sq"]) > cfg.gz for z in zs) or any(math.sqrt(y["yy"]) > cfg.gy for y in ys):
            raise RuntimeError("supplied population bounds violated by calibration gradients")
    # SRLVR_ADVANTAGE_MODE: where (b, s) come from. `history` is the published
    # method; `batch` recomputes the SAME statistics on this batch -- the
    # REINFORCE++-style alternative, since one trajectory per task position
    # leaves no group to normalise against. lambda's clip and the coefficient
    # sign/bound checks must then use the effective values.
    eff_b, eff_s = b, s
    if advantage_mode() == "batch":
        rewards_now = [float(m["terminal_reward"]) for m in data["srlvr_rows"]
                       if not m.get("padding")]
        eff_b, eff_s, how = batch_baseline_and_scale(rewards_now, b, s, cfg.delta_s)
        logger.info(
            "[srlvr-adv] mode=batch source=%s b=%.4f s=%.4f (history b=%.4f s=%.4f) trajectories=%d",
            how, eff_b, eff_s, b, s, len(rewards_now),
        )
    lam, cert = choose_lambda(cfg, eff_b, cfg.calibration, c_hat, d_hat, q_hat)
    if frozen_fixed is not None and lam != frozen_fixed:
        raise RuntimeError("fixed lambda changed after observing rewards")
    calibration_seconds = time.perf_counter() - probe_t0
    del zs, ys
    weights, bs, coefficients = [], [], []
    for idx, meta in enumerate(data["srlvr_rows"]):
        correction = 0. if meta["calibration"] else lam * meta["B"]
        a = (meta["terminal_reward"] - eff_b - correction) / eff_s
        w = 1-correction/(meta["terminal_reward"]-eff_b)
        if not (a*(2*meta["terminal_reward"]-1) > 0 and 1-cfg.rho-1e-6 <= w <= 1+cfg.rho+1e-6):
            raise RuntimeError("SRLVR coefficient sign/bound violated")
        data["opsd_advantages"][idx] = torch.full_like(data["opsd_advantages"][idx], a)
        if not meta.get("padding"):
            weights.append(1 - correction/(meta["terminal_reward"]-b))
            bs.append(meta["B"]); coefficients.append(a)
    args.srlvr_loss_scale = data["srlvr_batch"]["rows"] / data["srlvr_batch"]["N"]
    args.srlvr_calibration = False
    iterator, micro = get_data_iterator(args, actor.model, data)
    if len(micro) != 1:
        raise RuntimeError("SRLVR cannot update the policy within a task batch")
    actor_t0 = time.perf_counter()
    train(rollout_id, actor.model, actor.optimizer, actor.opt_param_scheduler, iterator, micro)
    actor_seconds = time.perf_counter() - actor_t0
    # Only after this batch's sole actor update; saved inside normal checkpoint args.
    args.srlvr_state = history_update(state, history, cfg)
    args.srlvr_state["last_rollout"] = rollout_id
    actor.weights_backuper.backup("actor")
    if mpu.get_data_parallel_rank() == 0:
        outcomes = {r["meta"]["position"]: r["meta"]["terminal_reward"] for r in rows}
        all_b = [r["meta"]["B"] for r in rows]
        all_e = [e for r in rows for e in r["meta"]["e"]]
        all_f = [f for r in rows for f in r["meta"]["f_chosen"]]
        all_w = [1-(0 if r["meta"]["calibration"] else lam*r["meta"]["B"])/(r["meta"]["terminal_reward"]-b) for r in rows]
        actor_tokens = sum(len(r["old"]) for r in rows)
        report = dict(algorithm="srlvr", task="alfworld", seed=args.seed, rollout=rollout_id,
                      tasks=len(outcomes), real_rollouts=len(outcomes), actor_tokens=actor_tokens,
                      successes=sum(outcomes.values()), failures=len(outcomes)-sum(outcomes.values()),
                      success_rate=sum(outcomes.values())/len(outcomes), optimizer_steps=1,
                      b=b, s=s, C_hat=c_hat, D_hat=d_hat, Q_hat=q_hat, lambda_value=lam,
                      B=distribution_summary(all_b), w=distribution_summary(all_w),
                      e=distribution_summary(all_e), all_centered_features_zero=all(e == 0 for e in all_e),
                      chosen_feature_saturation_fraction=sum(f == -1 for f in all_f)/max(len(all_f),1),
                      nonzero_correction_fraction=sum(w != 1 for w in all_w)/len(all_w),
                      calibration_seconds=calibration_seconds, actor_seconds=actor_seconds,
                      metadata_communication_seconds=comm_seconds,
                      actor_phase_wall_seconds=time.perf_counter()-t0,
                      train_device_seconds=2*(time.perf_counter()-t0),
                      alignment_bias_max=getattr(args, "srlvr_alignment_bias_max", 0.),
                      alignment_mae_max=getattr(args, "srlvr_alignment_mae_max", 0.),
                      importance_ratio_mean=getattr(args, "srlvr_ratio_mean", 0.) / max(getattr(args, "srlvr_ratio_count", 1), 1),
                      ppo_clip_fraction=getattr(args, "srlvr_clip_count", 0.) / max(getattr(args, "srlvr_ratio_count", 1), 1),
                      config=asdict(cfg), optimizer=dict(name=args.optimizer, lr=args.lr,
                      weight_decay=args.weight_decay, clip_grad=args.clip_grad,
                      parameter_space="all trainable LoRA parameters"), **cert)
        report["per_turn"] = {str(t): distribution_summary([r["meta"]["B"] for r in rows if r["meta"]["turn"] == t])
                              for t in sorted({r["meta"]["turn"] for r in rows})}
        path = Path(args.save).parent / "srlvr_metrics.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(report, allow_nan=False)+"\n")
        logger.info("[srlvr] %s", json.dumps(report, allow_nan=False))


def restore_history(args):
    """Use existing Megatron common checkpoint args; no independent state file."""
    from megatron.core import dist_checkpointing
    tracker = Path(args.load) / "latest_checkpointed_iteration.txt"
    iteration = int(tracker.read_text().strip())
    common = dist_checkpointing.load_common_state_dict(str(Path(args.load) / f"iter_{iteration:07d}"))
    saved_args = common["args"]
    if not hasattr(saved_args, "srlvr_state"):
        raise RuntimeError("SRLVR resume checkpoint is missing historical normalization state")
    args.srlvr_state = saved_args.srlvr_state
    if os.environ.get("SRLVR_VARIANT") in ("v2", "v3"):
        from methods.srlvr_hsi import Settings as HSISettings, prepare_history
        args.srlvr_state = prepare_history(args.srlvr_state, HSISettings.from_env(), iteration + 1)
