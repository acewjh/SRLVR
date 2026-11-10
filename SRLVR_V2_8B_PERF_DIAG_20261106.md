# SRLVR-v2 / Qwen3-8B 性能诊断材料（只读整理）

- run ID: `alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837`
- 日志: `cua/repos/opsd-slime/logs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837.log`
- 数据目录: `cua/repos/opsd-slime/runs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837/`（`rollout_*.pt` 为每个 rollout 的原始 turn rows）
- git commit: `6bf752bc9f9220a8368e5b762c1d7cfe3e8ef5e9`（2026-11-04，message `20261104`）
- 未提交修改（与本材料相关）: `methods/srlvr_hsi.py`、`opsd_core/srlvr_settings.py`、`tasks/alfworld/config.py`、`tasks/alfworld/pool.py`、`tasks/alfworld/rollout.py`、`tasks/alfworld/scripts/_common_v2.sh`、`tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh`；另有两个新文档 `SRLVR_REVISION_NOTE.md`（追加 §2.5）、`SRLVR_SCORE_REVIEW_20261105.md`、本文件。
- 说明：本材料未修改代码、未重启训练、未启动新实验；所有计时取自日志原文，缺少的直接标“未记录”。

---

## 1. 关键源码

### 1.1 `score_row` 与调用点

`srlvr_hsi.py:406-472`

```python
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

```

调用点（`train_batch`，逐 row 调用；row 来自本 rank 的 `data["srlvr_rows"]`）：

`srlvr_hsi.py:1182-1195`

```python
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
```


### 1.2 `_hidden_states`

`hsi2_jsd.py:258-283`

```python
def _hidden_states(model, head, ids) -> "object":
    """Run one forward with the head bypassed; return ``[T, H]`` fp32 hidden."""
    import torch

    device = next(model.parameters()).device
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        with _HiddenTap(head) as tap, torch.no_grad():
            model(input_ids=tensor, position_ids=None, attention_mask=None, labels=None)
        hidden = tap.hidden
        if hidden is None:
            raise RuntimeError("HSI v2: output-layer tap captured nothing")
        if hidden.dim() == 3:
            # mcore is sequence-first [S, B, H]; accept batch-first too rather
            # than silently scoring the wrong positions.
            if hidden.shape[1] == 1:
                hidden = hidden[:, 0, :]
            elif hidden.shape[0] == 1:
                hidden = hidden[0]
        return hidden.float()
    finally:
        if was_training:
            model.train()
        del tensor
```


### 1.3 `_probe_moments` / `_probe_terms`

`srlvr_hsi.py:612-646`

```python
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

```


### 1.4 calibration 的 DP-parallel 调度、梯度提取/复制、C/D 计算

#### 1.4.1 位置分配与行筛选/分块

`srlvr_hsi.py:744-806`

```python
def assign_probe_positions(calibration, dp_rank, dp_size):
    """Round-robin calibration positions per DP group.

    Empty when ``dp_size > calibration`` (those ranks stay idle and only join
    the final scalar-record gather). Never assigns one position to two groups.
    """
    if calibration < 1 or dp_size < 1 or not 0 <= dp_rank < dp_size:
        raise ValueError(f"invalid calibration/DP assignment: n={calibration} "
                         f"rank={dp_rank} dp={dp_size}")
    return list(range(dp_rank, calibration, dp_size))


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

```


#### 1.4.2 单条轨迹的 Z/Y 前向、反向与梯度复制

`srlvr_hsi.py:808-930`

```python
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

        def capture(model):
            nonlocal total
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

```


梯度提取/复制（`isolated_dp=True` 时 `reconstruct_dp=False`）：

`srlvr_hsi.py:574-605`

```python
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
```


TP 梯度收尾（每个 chunk 一次；不做 DP reduce-scatter）：

`srlvr_hsi.py:648-671`

```python
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

```


#### 1.4.3 探针总调度与 C/D 汇总

`srlvr_hsi.py:958-1019`

```python
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
    positions = (assign_probe_positions(cfg.calibration, dp_rank, dp_size)
                 if isolated_dp else list(range(cfg.calibration)))
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

```


`srlvr_hsi.py:756-781`

```python
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

```


#### 1.4.4 参与“完整梯度”的参数集合与 DP 重建（后者在 parallel 模式不执行）

`srlvr.py:370-387`

```python
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
```


`srlvr.py:328-367`

```python
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
```


#### 1.4.5 模式选择（parallel 由 `SRLVR_PROBE_MODE=auto` 解析得到）

`srlvr_hsi.py:1308-1317`

```python
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
```


### 1.5 `score_row` 输出的使用位置

- `e`：进入 `fill_future()` 计算未来校正 `B`：

`srlvr_hsi.py:483-484`

```python
        errors = [[e for e, mask in zip(r["meta"]["e"], r["mask"], strict=True) if mask]
                  for r in trajectory]
```


- `student_logp`：**只用于诊断日志** `[srlvr-hsi-views]`，与 rollout logprob 求 bias/mae：

`srlvr_hsi.py:1216-1219`

```python
    deltas = [lp - old for r in rows for lp, old, keep in zip(r["meta"]["student_logp"], r["old"], r["mask"], strict=True) if keep]
    logger.info("[srlvr-hsi-views] variant=%s rollout=%s p=causal q=%s temperature=%s support=%s bias=%s mae=%s",
                cfg.variant, rollout_id, "outcome_only" if cfg.teacher_outcome_only else "Full", args.rollout_temperature, args.vocab_size,
                sum(deltas) / max(1, len(deltas)), sum(abs(v) for v in deltas) / max(1, len(deltas)))
```


- PPO 的 old logprob **不是** `student_logp`，而是 `batch["rollout_log_probs"]`（rollout 采样时记录）：

`srlvr_hsi.py:538-545`

```python
        raise RuntimeError("SRLVR loss requires full-trajectory actor token counts")
    for row_index, (lp, old, advantage, mask, meta) in enumerate(zip(
            forward["log_probs"], batch["rollout_log_probs"], batch["opsd_advantages"],
            batch["loss_masks"], batch["srlvr_rows"], strict=True)):
        length = meta["actor_token_count"]
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("SRLVR loss requires a positive integer trajectory token count")
        a = advantage.detach().float()
```


该字段在 `convert()` 中由样本携带：

`srlvr_hsi.py:340-340`

```python
        loss_masks=[s.loss_mask for s in samples], rollout_log_probs=[s.rollout_log_probs for s in samples],
```


- `mu`、`f_chosen`：`score_row` 写入 `meta`，但 **v2/v3 路径没有消费者**；仅 legacy `methods/srlvr.py` 的 `attach_rows`/`train_batch` 使用它们（该入口不在本次 v2/v3 修改范围）。

---

## 2. calibration 原始 profile（同一 8B run，3 个已完成 batch）

选择依据：`[srlvr-hsi].calibration_seconds` 分别取较小 / 中间 / 较大；三个 rollout 都在同一进程、同一几何（TP4/DP2，2×TP4 引擎）下完成，且对应的 `rollout_*.pt` 未被后续重启覆盖。（rollout 11 的 calibration 更慢，888.0 s，但它的 `rollout_11.pt` 已被 11-06 00:38 的重启 attempt 覆盖，其 profile 属于另一个 batch，故不选入本节的原始记录。）

| batch | 选择 | calibration_seconds | 行号来源 |
|---|---|---:|---|
| rollout 4 | 较快 | 411.8 s | `[srlvr-hsi]` @ 2026-11-05 15:55:53 |
| rollout 1 | 一般 | 528.6 s | `[srlvr-hsi]` @ 2026-11-05 13:35:39 |
| rollout 8 | 较慢 | 789.0 s | `[srlvr-hsi]` @ 2026-11-05 19:17:08 |

### 2.1 原始记录
#### rollout 4（较快）

```text
[hsi-roll]     2026-11-05 15:36:44  rollout=4 engines=2 sampled_in_C=0.000000 full_k=0.000000 scored_turns=716/716 episodes=32 topup_rounds=0 topup_episodes=0 generated_tokens=745214 rollout_seconds=1432.4
[row-padding]  2026-11-05 15:36:56  rollout=4 padded_tokens=910848 batch_max_tokens=2396160
[hsi-views]    2026-11-05 15:43:45  variant=v2 rollout=4 p=causal q=Full temperature=1.0 support=151936 bias=-21.561011918133428 mae=21.561066485999117
[probe-local]  2026-11-05 15:45:08  which=Z position=1 dp_rank=1 turns=18/18 chunks=4 accum=cpu isolated=True
[probe-local]  2026-11-05 15:45:26  which=Z position=0 dp_rank=0 turns=21/21 chunks=5 accum=cpu isolated=True [repeated 2x across cluster]
[probe-local]  2026-11-05 15:45:34  which=Z position=0 dp_rank=0 turns=21/21 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 15:46:25  which=Y position=1 dp_rank=1 turns=17/18 chunks=4 accum=cpu isolated=True
[probe-local]  2026-11-05 15:46:30  which=Y position=1 dp_rank=1 turns=17/18 chunks=4 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 15:47:06  which=Y position=0 dp_rank=0 turns=20/21 chunks=4 accum=cpu isolated=True
[probe-local]  2026-11-05 15:48:25  which=Z position=3 dp_rank=1 turns=17/17 chunks=4 accum=cpu isolated=True
[probe-local]  2026-11-05 15:48:35  which=Z position=3 dp_rank=1 turns=17/17 chunks=4 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 15:48:40  which=Z position=2 dp_rank=0 turns=6/6 chunks=2 accum=cpu isolated=True [repeated 4x across cluster]
[probe-local]  2026-11-05 15:48:55  which=Y position=2 dp_rank=0 turns=5/6 chunks=1 accum=cpu isolated=True
[probe-local]  2026-11-05 15:49:03  which=Y position=2 dp_rank=0 turns=5/6 chunks=1 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 15:49:41  which=Y position=3 dp_rank=1 turns=16/17 chunks=4 accum=cpu isolated=True
[probe-local]  2026-11-05 15:49:46  which=Y position=3 dp_rank=1 turns=16/17 chunks=4 accum=cpu isolated=True [repeated 3x across cluster]
[probe-profile] 2026-11-05 15:50:37  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 411.78851692564785, "positions": [0, 2], "tp_sync_s": 0.13518432527780533, "copy_s": 127.68649684637785, "train_s": 248.28929291851819, "chunks": 12, "turns": 52, "moments_s": 129.85350653715432, "empty_cache_s": 0.044836532324552536, "records_gather_s": 29.0816877130419}, "chosen": "parallel", "wall_s": 411.78851692564785}
[probe-profile] 2026-11-05 15:50:38  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 412.59333985857666, "positions": [1, 3], "tp_sync_s": 0.1790719609707594, "copy_s": 209.05066379345953, "train_s": 303.14688481390476, "chunks": 16, "turns": 68, "moments_s": 104.00945312343538, "empty_cache_s": 0.04024859331548214, "records_gather_s": 0.0476196613162756}, "chosen": "parallel", "wall_s": 412.59333985857666} [repeated 7x across cluster]
[srlvr-hsi]    2026-11-05 15:55:53  {"algorithm": "srlvr_v2", "rollout": 4, "optimizer_steps": 1, "cumulative_optimizer_steps": 5, "real_rollouts": 32, "actor_tokens": 745214, "success_rate": 0.84375, "b": 0.99, "s": 0.3401659751136201, "lambda_value": 0.140625, "lambda_calibrated": 0.140625, "lambda_raw_frozen": 1.6808574288739513, "lambda_limit": 0.140625, "ablations": {"actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}, "lambda_next": 0.009000000000000008, "C_hat": 0.7044384894694299, "D_hat": 0.42421651831117346, "advantage_mode": "batch", "advantage_source": "batch", "b_effective": 0.84375, "s_effective": 0.37661112237957073, "batch_rewards": 32, "batch_fallback_this_rollout": false, "batch_fallbacks": 0, "calibration": "empirical_second_moment", "probes": 4, "reduction": "trajectory_token_mean_v1", "trajectory_actor_tokens": {"0": 16086, "1": 12338, "2": 2624, "3": 24498, "4": 61262, "5": 6505, "6": 30128, "7": 12195, "8": 7145, "9": 9250, "10": 5485, "11": 10919, "12": 42162, "13": 23741, "14": 1903, "15": 70757, "16": 89287, "17": 10505, "18": 4599, "19": 25061, "20": 7942, "21": 35375, "22": 8659, "23": 57996, "24": 7475, "25": 53697, "26": 17831, "27": 60759, "28": 13818, "29": 12092, "30": 2127, "31": 993}, "B": {"count": 716, "mean": -0.06871103120815047, "min": -0.24383767301818088, "max": 0.02833776579649884, "std": 0.05031099112113316, "q10": -0.1408493920156275, "q50": -0.06219388424939677, "q90": -0.00824617426502536}, "e": {"count": 745214, "mean": -0.04512412913182367, "min": -0.9998306632041931, "max": 0.4999794363975525, "std": 0.2734146587399089, "q10": -0.4305705189704895, "q50": -0.020423173904418945, "q90": 0.30269878506660475}, "reference_seconds": 86.01214095205069, "score_seconds": 322.7211734801531, "calibration_seconds": 411.78851692564785, "calibration_mode": "parallel", "probe_profile": {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 411.78851692564785, "positions": [0, 2], "tp_sync_s": 0.13518432527780533, "copy_s": 127.68649684637785, "train_s": 248.28929291851819, "chunks": 12, "turns": 52, "moments_s": 129.85350653715432, "empty_cache_s": 0.044836532324552536, "records_gather_s": 29.0816877130419}}, "padded_tokens": 910848, "batch_max_padding_tokens": 2396160, "actor_seconds": 315.7753605339676, "config": {"variant": "v2", "gamma": 0.99, "calibration": 4, "rho": 0.9, "window": 128, "eps_b0": 0.01, "delta_s": 0.1, "position_chunk": 64, "actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}}
```

#### rollout 1（一般）

```text
[hsi-roll]     2026-11-05 13:13:27  rollout=1 engines=2 sampled_in_C=0.000000 full_k=0.000000 scored_turns=788/788 episodes=32 topup_rounds=0 topup_episodes=0 generated_tokens=847088 rollout_seconds=1748.3
[row-padding]  2026-11-05 13:13:42  rollout=1 padded_tokens=1047040 batch_max_tokens=2838528
[hsi-views]    2026-11-05 13:21:09  variant=v2 rollout=1 p=causal q=Full temperature=1.0 support=151936 bias=-21.426943236980964 mae=21.426956997385904
[probe-local]  2026-11-05 13:21:53  which=Z position=1 dp_rank=1 turns=15/15 chunks=3 accum=cpu isolated=True
[probe-local]  2026-11-05 13:21:55  which=Z position=1 dp_rank=1 turns=15/15 chunks=3 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 13:22:39  which=Y position=1 dp_rank=1 turns=14/15 chunks=3 accum=cpu isolated=True
[probe-local]  2026-11-05 13:22:40  which=Y position=1 dp_rank=1 turns=14/15 chunks=3 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 13:23:43  which=Z position=0 dp_rank=0 turns=49/49 chunks=10 accum=cpu isolated=True
[probe-local]  2026-11-05 13:23:44  which=Z position=0 dp_rank=0 turns=49/49 chunks=10 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 13:24:38  which=Z position=3 dp_rank=1 turns=22/22 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 13:24:40  which=Z position=3 dp_rank=1 turns=22/22 chunks=5 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 13:26:21  which=Y position=3 dp_rank=1 turns=21/22 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 13:27:28  which=Y position=0 dp_rank=0 turns=48/49 chunks=10 accum=cpu isolated=True
[probe-local]  2026-11-05 13:27:36  which=Y position=0 dp_rank=0 turns=48/49 chunks=10 accum=cpu isolated=True [repeated 2x across cluster]
[probe-local]  2026-11-05 13:28:46  which=Z position=2 dp_rank=0 turns=4/4 chunks=1 accum=cpu isolated=True
[probe-local]  2026-11-05 13:29:00  which=Y position=2 dp_rank=0 turns=3/4 chunks=1 accum=cpu isolated=True
[probe-local]  2026-11-05 13:28:47  which=Z position=2 dp_rank=0 turns=4/4 chunks=1 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 13:29:03  which=Y position=2 dp_rank=0 turns=3/4 chunks=1 accum=cpu isolated=True [repeated 3x across cluster]
[probe-profile] 2026-11-05 13:29:58  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 528.6065278165042, "positions": [0, 2], "tp_sync_s": 0.2460758350789547, "copy_s": 316.6670331582427, "train_s": 414.84825840219855, "chunks": 22, "turns": 104, "moments_s": 109.19018807820976, "empty_cache_s": 0.045247338712215424, "records_gather_s": 0.004268310964107513}, "chosen": "parallel", "wall_s": 528.6065278165042}
[probe-profile] 2026-11-05 13:29:59  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 529.5904630944133, "positions": [1, 3], "tp_sync_s": 0.1771643217653036, "copy_s": 167.6691993549466, "train_s": 261.141368849203, "chunks": 16, "turns": 72, "moments_s": 108.16121510975063, "empty_cache_s": 0.04882653057575226, "records_gather_s": 154.92519348673522}, "chosen": "parallel", "wall_s": 529.5904630944133} [repeated 7x across cluster]
[srlvr-hsi]    2026-11-05 13:35:39  {"algorithm": "srlvr_v2", "rollout": 1, "optimizer_steps": 1, "cumulative_optimizer_steps": 2, "real_rollouts": 32, "actor_tokens": 847088, "success_rate": 0.84375, "b": 0.99, "s": 0.4487830767754061, "lambda_value": 0.140625, "lambda_calibrated": 0.140625, "lambda_raw_frozen": 1.9823108409456685, "lambda_limit": 0.140625, "ablations": {"actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}, "lambda_next": 0.009000000000000008, "C_hat": 0.46823258954786273, "D_hat": 0.30518397625748217, "advantage_mode": "batch", "advantage_source": "batch", "b_effective": 0.84375, "s_effective": 0.37661112237957073, "batch_rewards": 32, "batch_fallback_this_rollout": false, "batch_fallbacks": 0, "calibration": "empirical_second_moment", "probes": 4, "reduction": "trajectory_token_mean_v1", "trajectory_actor_tokens": {"0": 26440, "1": 9191, "2": 1954, "3": 21665, "4": 31136, "5": 7686, "6": 1440, "7": 59165, "8": 6722, "9": 79164, "10": 2357, "11": 2679, "12": 20128, "13": 43760, "14": 2830, "15": 61142, "16": 14470, "17": 44312, "18": 1375, "19": 65716, "20": 7401, "21": 82282, "22": 3115, "23": 19878, "24": 24904, "25": 36186, "26": 1502, "27": 76225, "28": 8807, "29": 15885, "30": 12792, "31": 54779}, "B": {"count": 788, "mean": -0.06704467528356203, "min": -0.2397266905894475, "max": 0.028947702824567732, "std": 0.05428090224673097, "q10": -0.1440676355989219, "q50": -0.0561760255367664, "q90": -0.005785260833077045}, "e": {"count": 847088, "mean": -0.04568661351513369, "min": -0.9997575879096985, "max": 0.49998974800109863, "std": 0.25015957528188154, "q10": -0.3845629394054413, "q50": -0.023777663707733154, "q90": 0.25945317149162295}, "reference_seconds": 95.1570462435484, "score_seconds": 352.0535941198468, "calibration_seconds": 528.6065278165042, "calibration_mode": "parallel", "probe_profile": {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 528.6065278165042, "positions": [0, 2], "tp_sync_s": 0.2460758350789547, "copy_s": 316.6670331582427, "train_s": 414.84825840219855, "chunks": 22, "turns": 104, "moments_s": 109.19018807820976, "empty_cache_s": 0.045247338712215424, "records_gather_s": 0.004268310964107513}}, "padded_tokens": 1047040, "batch_max_padding_tokens": 2838528, "actor_seconds": 340.8384981043637, "config": {"variant": "v2", "gamma": 0.99, "calibration": 4, "rho": 0.9, "window": 128, "eps_b0": 0.01, "delta_s": 0.1, "position_chunk": 64, "actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}}
```

#### rollout 8（较慢）

```text
[hsi-roll]     2026-11-05 18:52:04  rollout=8 engines=2 sampled_in_C=0.000000 full_k=0.000000 scored_turns=701/701 episodes=32 topup_rounds=0 topup_episodes=0 generated_tokens=623005 rollout_seconds=1538.7
[row-padding]  2026-11-05 18:52:16  rollout=8 padded_tokens=856064 batch_max_tokens=1982464
[hsi-views]    2026-11-05 18:58:52  variant=v2 rollout=8 p=causal q=Full temperature=1.0 support=151936 bias=-21.504186398890674 mae=21.50425556179141
[probe-local]  2026-11-05 18:59:27  which=Z position=0 dp_rank=0 turns=8/8 chunks=2 accum=cpu isolated=True
[probe-local]  2026-11-05 19:00:12  which=Y position=0 dp_rank=0 turns=7/8 chunks=2 accum=cpu isolated=True
[probe-local]  2026-11-05 19:02:20  which=Z position=1 dp_rank=1 turns=50/50 chunks=10 accum=cpu isolated=True
[probe-local]  2026-11-05 19:02:26  which=Z position=1 dp_rank=1 turns=50/50 chunks=10 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 19:03:11  which=Z position=2 dp_rank=0 turns=24/24 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 19:03:18  which=Z position=2 dp_rank=0 turns=24/24 chunks=5 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 19:05:05  which=Y position=2 dp_rank=0 turns=23/24 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 19:06:07  which=Y position=1 dp_rank=1 turns=49/50 chunks=10 accum=cpu isolated=True
[probe-local]  2026-11-05 19:06:14  which=Y position=1 dp_rank=1 turns=49/50 chunks=10 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 19:08:59  which=Z position=3 dp_rank=1 turns=26/26 chunks=6 accum=cpu isolated=True
[probe-local]  2026-11-05 19:09:06  which=Z position=3 dp_rank=1 turns=26/26 chunks=6 accum=cpu isolated=True [repeated 3x across cluster]
[probe-local]  2026-11-05 19:10:55  which=Y position=3 dp_rank=1 turns=25/26 chunks=5 accum=cpu isolated=True
[probe-local]  2026-11-05 19:11:04  which=Y position=3 dp_rank=1 turns=25/26 chunks=5 accum=cpu isolated=True [repeated 3x across cluster]
[probe-profile] 2026-11-05 19:12:01  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 788.3750987462699, "positions": [0, 2], "tp_sync_s": 0.175762502476573, "copy_s": 215.31646987050772, "train_s": 313.88990879431367, "chunks": 14, "turns": 62, "moments_s": 116.5222393553704, "empty_cache_s": 0.028177721425890923, "records_gather_s": 352.90035411156714}, "chosen": "parallel", "wall_s": 788.3750987462699}
[probe-profile] 2026-11-05 19:12:02  {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 789.365770149976, "positions": [1, 3], "tp_sync_s": 0.39799390360713005, "copy_s": 531.8676611837, "train_s": 687.1343970242888, "chunks": 31, "turns": 150, "moments_s": 96.53073810599744, "empty_cache_s": 0.06176735460758209, "records_gather_s": 0.0491776205599308}, "chosen": "parallel", "wall_s": 789.365770149976} [repeated 7x across cluster]
[srlvr-hsi]    2026-11-05 19:17:08  {"algorithm": "srlvr_v2", "rollout": 8, "optimizer_steps": 1, "cumulative_optimizer_steps": 9, "real_rollouts": 32, "actor_tokens": 623005, "success_rate": 0.875, "b": 0.99, "s": 0.3733557703451227, "lambda_value": 0.1125, "lambda_calibrated": 0.1125, "lambda_raw_frozen": 1.1496740995280936, "lambda_limit": 0.1125, "ablations": {"actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}, "lambda_next": 0.009000000000000008, "C_hat": 0.36261064860869396, "D_hat": 0.3361551328040662, "advantage_mode": "batch", "advantage_source": "batch", "b_effective": 0.875, "s_effective": 0.34550687402713137, "batch_rewards": 32, "batch_fallback_this_rollout": false, "batch_fallbacks": 0, "calibration": "empirical_second_moment", "probes": 4, "reduction": "trajectory_token_mean_v1", "trajectory_actor_tokens": {"0": 3776, "1": 48889, "2": 11939, "3": 24744, "4": 15338, "5": 21267, "6": 2199, "7": 12314, "8": 6849, "9": 4239, "10": 12376, "11": 28463, "12": 10417, "13": 5399, "14": 3806, "15": 26267, "16": 2969, "17": 7482, "18": 33159, "19": 10295, "20": 5795, "21": 12130, "22": 12651, "23": 16224, "24": 38664, "25": 3101, "26": 44034, "27": 28991, "28": 80374, "29": 2721, "30": 76642, "31": 9491}, "B": {"count": 701, "mean": -0.059645369079988526, "min": -0.22786317221960947, "max": 0.016912071964671215, "std": 0.04532931486988633, "q10": -0.12508426030914346, "q50": -0.0482526191006571, "q90": -0.008277577223172117}, "e": {"count": 623005, "mean": -0.05198121898139901, "min": -0.9990490674972534, "max": 0.4999721646308899, "std": 0.2761600284656807, "q10": -0.44536303281784057, "q50": -0.02920880913734436, "q90": 0.3057736039161682}, "reference_seconds": 86.14383109658957, "score_seconds": 309.857742305845, "calibration_seconds": 788.9611805081367, "calibration_mode": "parallel", "probe_profile": {"mode": "parallel", "deterministic": "0", "call": "parallel", "parallel": {"wall_s": 788.9611805081367, "positions": [0, 2], "tp_sync_s": 0.16265542432665825, "copy_s": 153.48810384981334, "train_s": 295.4607340004295, "chunks": 14, "turns": 62, "moments_s": 136.2477231360972, "empty_cache_s": 0.027866361662745476, "records_gather_s": 353.759791303426}}, "padded_tokens": 856064, "batch_max_padding_tokens": 1982464, "actor_seconds": 306.9654386844486, "config": {"variant": "v2", "gamma": 0.99, "calibration": 4, "rho": 0.9, "window": 128, "eps_b0": 0.01, "delta_s": 0.1, "position_chunk": 64, "actor_lambda_zero": false, "trajectory_uniform_b": false, "teacher_outcome_only": false}}
```

> 注：以 `[repeated Nx across cluster]` 结尾的行是日志聚合器对同一事件的重复输出（来自其它 rank），内容相同。

### 2.2 外层墙钟边界（来自日志时间戳）

| batch | rollout 结束 | row-padding | score 视图完成 | probe 完成 | step JSON |
|---|---|---|---|---|---|
| 4 | 2026-11-05 15:36:44 | 2026-11-05 15:36:56 | 2026-11-05 15:43:45 | 2026-11-05 15:50:37 | 2026-11-05 15:55:53 |
| 1 | 2026-11-05 13:13:27 | 2026-11-05 13:13:42 | 2026-11-05 13:21:09 | 2026-11-05 13:29:58 | 2026-11-05 13:35:39 |
| 8 | 2026-11-05 18:52:04 | 2026-11-05 18:52:16 | 2026-11-05 18:58:52 | 2026-11-05 19:12:01 | 2026-11-05 19:17:08 |

### 2.3 各 rank 的 probe 汇总与计时边界

| batch | rank/位置 | wall | train | copy | moments | chunks | turns | tp_sync | records_gather | empty_cache |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | `[0, 2]` | 411.8 | 248.3 | 127.7 | 129.9 | 12 | 52 | 0.14 | 29.082 | 0.045 |
| 4 | `[1, 3]` | 412.6 | 303.1 | 209.1 | 104.0 | 16 | 68 | 0.18 | 0.048 | 0.040 |
| 1 | `[0, 2]` | 528.6 | 414.8 | 316.7 | 109.2 | 22 | 104 | 0.25 | 0.004 | 0.045 |
| 1 | `[1, 3]` | 529.6 | 261.1 | 167.7 | 108.2 | 16 | 72 | 0.18 | 154.925 | 0.049 |
| 8 | `[0, 2]` | 788.4 | 313.9 | 215.3 | 116.5 | 14 | 62 | 0.18 | 352.900 | 0.028 |
| 8 | `[1, 3]` | 789.4 | 687.1 | 531.9 | 96.5 | 31 | 150 | 0.40 | 0.049 | 0.062 |

计时边界（按源码定位）：

- `wall_s`：`train_batch` 中包住整次 `_run_calibration_probes` 的 `perf_counter` 差值（`probe_seconds = time.perf_counter() - call_started`），**包含**本 rank 所有位置、最后由 `_gather_probe_records` 的 `all_gather_object` 造成的等待（快 rank 上尤其明显），也包含每个位置后的 `empty_cache`。与 `[srlvr-hsi].calibration_seconds` 是同一个值（r8: 789.4 vs 788.96）。
- `train_s`：`_probe_trajectory_vectors` 中 `train_started = perf_counter()` 到 `train(...)` 返回；**包含** backward 及其 finalize 钩子里执行的梯度复制（`copy_s` 是它的子集）。实测可验证：r8 `[1,3]` 的 train 687.1 + moments 96.5 ≈ wall 789.4，而 copy 531.9 已含在 train 内。
- `copy_s`：`_accumulate_probe_gradient` 内 `perf_counter` 差值，覆盖 `_reconstruct_dp_gradients`（parallel 模式跳过）与逐参数的 `param.main_grad.detach().float().reshape(-1)` → `.cpu()` → `add_`/`clone`。
- `moments_s`：`_probe_moments` 的计时（`_probe_terms` 的 z²/zy/y² 归约、乘 `scales`、TP `all_reduce`、`.cpu().tolist()`、`math.fsum`），外加 `_run_calibration_probes` 中包住它的 `moments_started` 区间。
- `chunks`：`train()` 被调用的次数（Z 与 Y 分开计），每 chunk 最多 `SRLVR_PROBE_CHUNK`（默认 5）条 turn row；profile 里的 `chunks` 是本次两个位置、Z+Y 的总和。
- `turns`：进入 Z/Y chunk 的 turn row 数（同样 Z+Y 求和）。Y 会剔除 `B==0` 的行（每条轨迹的最后一轮）。
- `tp_sync_s`：`_finalize_probe_tp_gradients` 的耗时（TP all-reduce 非张量梯度 + reset temporary tensors），每 chunk 一次。
- `records_gather_s`：`dist.all_gather_object(records, group=dp_group_gloo)` 的耗时；数值大的一方是**在等另一个 DP rank**。
- `empty_cache_s`：每个 position 结束后 `srlvr_compact.empty_cache()` 的耗时（无条件执行，见 `_run_calibration_probes` 末尾）。

### 2.4 每条 probe 的工作量（来自 `[srlvr-probe-local]`）

| batch | position | which | dp_rank | turns | chunks |
|---|---:|---|---:|---:|---:|
| 4 | 1 | Z | 1 | 18/18 | 4 |
| 4 | 0 | Z | 0 | 21/21 | 5 |
| 4 | 0 | Z | 0 | 21/21 | 5 |
| 4 | 1 | Y | 1 | 17/18 | 4 |
| 4 | 1 | Y | 1 | 17/18 | 4 |
| 4 | 0 | Y | 0 | 20/21 | 4 |
| 4 | 3 | Z | 1 | 17/17 | 4 |
| 4 | 3 | Z | 1 | 17/17 | 4 |
| 4 | 2 | Z | 0 | 6/6 | 2 |
| 4 | 2 | Y | 0 | 5/6 | 1 |
| 4 | 2 | Y | 0 | 5/6 | 1 |
| 4 | 3 | Y | 1 | 16/17 | 4 |
| 4 | 3 | Y | 1 | 16/17 | 4 |
| 1 | 1 | Z | 1 | 15/15 | 3 |
| 1 | 1 | Z | 1 | 15/15 | 3 |
| 1 | 1 | Y | 1 | 14/15 | 3 |
| 1 | 1 | Y | 1 | 14/15 | 3 |
| 1 | 0 | Z | 0 | 49/49 | 10 |
| 1 | 0 | Z | 0 | 49/49 | 10 |
| 1 | 3 | Z | 1 | 22/22 | 5 |
| 1 | 3 | Z | 1 | 22/22 | 5 |
| 1 | 3 | Y | 1 | 21/22 | 5 |
| 1 | 0 | Y | 0 | 48/49 | 10 |
| 1 | 0 | Y | 0 | 48/49 | 10 |
| 1 | 2 | Z | 0 | 4/4 | 1 |
| 1 | 2 | Y | 0 | 3/4 | 1 |
| 1 | 2 | Z | 0 | 4/4 | 1 |
| 1 | 2 | Y | 0 | 3/4 | 1 |
| 8 | 0 | Z | 0 | 8/8 | 2 |
| 8 | 0 | Y | 0 | 7/8 | 2 |
| 8 | 1 | Z | 1 | 50/50 | 10 |
| 8 | 1 | Z | 1 | 50/50 | 10 |
| 8 | 2 | Z | 0 | 24/24 | 5 |
| 8 | 2 | Z | 0 | 24/24 | 5 |
| 8 | 2 | Y | 0 | 23/24 | 5 |
| 8 | 1 | Y | 1 | 49/50 | 10 |
| 8 | 1 | Y | 1 | 49/50 | 10 |
| 8 | 3 | Z | 1 | 26/26 | 6 |
| 8 | 3 | Z | 1 | 26/26 | 6 |
| 8 | 3 | Y | 1 | 25/26 | 5 |
| 8 | 3 | Y | 1 | 25/26 | 5 |

每条 probe 的有效 actor token 数（日志未记录，按 `.pt` 的 `loss_mask` 逐 position 求和；保存的 `.pt` 中 `B` 仍是训练前的占位 0，因此 Y 的集合按“排除每条轨迹的最后一轮”重建，该重建与日志里 `[srlvr-probe-local]` 的 Y turns = Z turns − 1 完全一致）：

| batch | position | Z turns | Z actor tokens | Y turns | Y actor tokens |
|---|---:|---:|---:|---:|---:|
| 4 | 0 | 21 | 16,086 | 20 | 15,879 |
| 4 | 1 | 18 | 12,338 | 17 | 10,591 |
| 4 | 2 | 6 | 2,624 | 5 | 2,475 |
| 4 | 3 | 17 | 24,498 | 16 | 22,989 |
| 1 | 0 | 49 | 26,440 | 48 | 25,443 |
| 1 | 1 | 15 | 9,191 | 14 | 9,034 |
| 1 | 2 | 4 | 1,954 | 3 | 1,001 |
| 1 | 3 | 22 | 21,665 | 21 | 19,976 |
| 8 | 0 | 8 | 3,776 | 7 | 3,625 |
| 8 | 1 | 50 | 48,889 | 49 | 47,091 |
| 8 | 2 | 24 | 11,939 | 23 | 11,717 |
| 8 | 3 | 26 | 24,744 | 25 | 24,473 |

由此可得（同一批次的 Z/Y 行集合不同）：

- **Z/Y 前向-反向次数**：每个 chunk 一次 `train()`（1 次 forward + 1 次 backward），因此forward 次数 = backward 次数 = chunks；Z 与 Y **不共享前向**（两次独立 `train()` 调用，行集合也不同）。
- **是否复制完整梯度到 CPU**：是（`SRLVR_PROBE_ACCUM=cpu`）。每个 chunk 的 backward finalize 钩子执行一次整参数集的 `grad.float().cpu()` 累积；复制次数 = chunks（Z+Y 合计），精度 FP32。复制对象是每 rank 的 TP 分片（另外参与计数的 replicated 参数每 rank 各有一份）。单 chunk 字节数未记录。
- **C/D 计算时机**：单条轨迹的 Z、Y 全部 chunk 累积完成后，`_probe_moments(z,y,...)` 计算 zz/zy/yy（每参数标量、乘 `scales`、TP all-reduce、转 CPU）；四个 position 的 records 再经 `_gather_probe_records`（DP `all_gather_object`）后由 `_finalize_probe_records` 求 `C_hat=mean(zy)`、`D_hat=mean(yy)`。
- **跨卡同步/传输点**：
  1. 每 chunk：`_finalize_probe_tp_gradients`（TP all-reduce，`tp_sync_s`）；
  2. 每 position：`_probe_moments` 内 TP `all_reduce` + 标量 D2H（`.cpu().tolist()`）；
  3. 每 chunk：整参数梯度 D2H（fp32，`copy_s`）；
  4. 每 position 结束：`empty_cache()`；
  5. 全部 position 结束：DP `all_gather_object`（gloo，`records_gather_s`，含等待慢 rank）。

### 2.5 DP 负载不均衡（同一批次的直接证据）

`assign_probe_positions` 按 **position 序号**轮转（`range(dp_rank, calibration, dp_size)`），但不同 position 的 turn 数差异极大，导致两个 DP rank 的 chunk 数不同，wall 取两者较大者：

| batch | rank0 positions | rank0 chunks | rank1 positions | rank1 chunks | wall |
|---|---|---:|---|---:|---:|
| 4 | `[0, 2]` | 12 | `[1, 3]` | 16 | 412.6 |
| 1 | `[0, 2]` | 22 | `[1, 3]` | 16 | 529.6 |
| 8 | `[0, 2]` | 14 | `[1, 3]` | 31 | 789.4 |

---

## 3. score 的真实工作量（同一 3 个 batch）

来源：`runs/<run>/rollout_{id}.pt` 的 `samples`（每条 turn row 的 `tokens`、`loss_mask`、`metadata.srlvr.teacher_view=(ids, plen, n)`）。`actor_tokens` 三项与 `[srlvr-hsi].actor_tokens` 完全一致，说明 `.pt` 的 row 集合就是本 batch 被 score 的行集合。

| batch | 真实轨迹数 | turn rows | rank0 rows | rank1 rows | student 输入 token | teacher 输入 token | 有效 actor token | score position chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 32 | 716 | 358 | 358 | 1,630,165 | 2,228,053 | 745,214 | 12,004 |
| 1 | 32 | 788 | 394 | 394 | 1,885,119 | 2,649,591 | 847,088 | 13,634 |
| 8 | 32 | 701 | 351 | 350 | 1,536,678 | 2,077,427 | 623,005 | 10,079 |

口径说明：

- “真实轨迹数”= `real_rollouts` = 32（G=1 × 32 个采样位置），不是 turn row 数。
- rank 行数：日志**未直接记录**每 rank 行数。此处按 `balance_data=False`（args dump 已确认）时`_split_train_data_by_dp` 的 round-robin 规则 `range(i, N, dp_size)`，对 `.pt` 的 samples 顺序推导；若上游顺序与 `.pt` 不同，这两个数需要以运行期计数为准（未记录）。
- student/teacher 输入 token：逐 row 的 `len(tokens)` / `len(ids)` 求和，**非 padding**。
- **score 阶段没有 padding**：`score_row` 直接把 `tokens` / `teacher_view[0]` 原序列送进 `_hidden_states`，不经过 `get_data_iterator` 的 padding；因此 padding 后 token 数 = 非 padding 数（本节两列相同，不单列）。
  日志里的 `padded_tokens`（如 r1=1,047,040）属于 **actor 训练 batch** 的 padding 统计，不是 score 的输入。
- score position chunks：对每条 row 按 `cfg.position_chunk=64` 对**有效位置**分块，即 `sum(ceil(active_positions/64))`；每个 chunk 内对 `lp`、`lq` 各做一次全词表 log-prob 归约。

### 3.1 代表性 row（rollout 4，取 LCP 中位数附近的一行）

| 字段 | 值 |
|---|---:|
| position / turn | 6 / 4 |
| student 序列长度 `len(tokens)` | 3029 |
| teacher 序列长度 `len(ids)` | 4794 |
| `plen`（teacher 前缀长度） | 2887 |
| `n`（被评分 token 数） | 1907 |
| student 前缀长度 `len(tokens)-n` | 1122 |
| 按 token id 的最长公共前缀 LCP | 1116 |
| student 评分位置（0-based） | 1121 .. 3027 |
| teacher 评分位置（0-based） | 2886 .. 4792 |
| `ids[plen:] == tokens[-n:]` | True |
| 公共前缀 token 完全一致 | True |

说明：teacher 与 student 在 LCP 处开始分叉（hindsight 插入点），此前缀之后 teacher 还有 1771 个额外前缀 token，student 则有 6 个自己的 token。

`rollout 4` 全部 716 行的 LCP 分布（按 token id，实测）：

- min=400, p25=813, median=1115, p75=1557, max=3112
- `LCP == student 前缀长度` 的行数：**0 / 716**（teacher 在 student 前缀结束前就与 student 分叉）。

position id / attention mask：`_hidden_states` 调用 `model(input_ids=tensor, position_ids=None, attention_mask=None)`。两条序列都从 0 开始连续编号，因此在 LCP 范围内 position id 相同（0..LCP-1），attention 都是默认 causal mask；LCP 之后两序列的 position id 继续各自递增，但对应位置内容不同。

---

## 4. 生效的计算路径与显存依据

### 4.1 并行与批（来自 `/proc/<pid>/cmdline` 与日志 args dump）

| 项 | 值 | 来源 |
|---|---|---|
| TP / PP / CP / EP | 4 / 1 / 1 / 1 | argv |
| DP（actor） | 2（8 卡 / TP4） | `[v2-efficiency] actor=8 train_tp=4` |
| rollout 并行 | 8 卡，`ROLLOUT_GPUS_PER_ENGINE=4`，2 个 TP4 引擎，`COLOCATE=1` | 启动 banner + argv |
| micro-batch | 1 | args dump |
| global batch / rollout batch | 32 / 32 | args dump |
| sequence length | 32768 | args dump |
| 更新范围 | 全量 RL（`OPSD_LORA=0`） | 启动 env |
| 参数精度 | `bf16=True, fp16=False` | args dump |
| 梯度精度 | fp32 `main_grad`（Megatron 默认） | 代码 `param.main_grad.detach().float()` |
| 激活重计算 | `full` / `uniform` / `1` 层 | args dump |
| optimizer | `adam` + `--optimizer-cpu-offload --optimizer-offload-fraction 1 --use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d` | argv |
| distributed optimizer | True | args dump |
| `balance_data` | False（round-robin DP 切分） | args dump |
| selective logits | **未启用**（`OPSD_SRLVR_SELECTIVE_LOGITS` 未设置；`srlvr_compact.enabled()` 返回 False） | 进程 env / 代码 |
| score position chunk | 64（`SRLVR_POSITION_CHUNK` / `OPSD_HSI2_POS_CHUNK` 均未设置，取默认） | 进程 env / 代码 |
| probe chunk | 5（`SRLVR_PROBE_CHUNK` 未设置） | 进程 env / 代码 |
| probe 累加位置 | CPU（`SRLVR_PROBE_ACCUM=cpu`） | 进程 env |
| probe 模式 | `auto` → `parallel`（profile `mode=parallel`） | 进程 env + profile |
| vocab 分片 | 151936 / TP4 = 37984 行/rank（无额外 vocab chunk） | argv + `_local_global_log_probs` |

### 4.2 已有峰值显存记录（本 run 日志原文）

```text
[srlvr-mem] probe position=0 peak_alloc=13.15GiB peak_reserved=18.29GiB
[srlvr-mem] actor update peak rollout=23 peak_alloc=15.07GiB peak_reserved=19.96GiB
[Rank 0] Memory-Usage after update_weights: {'gpu': '0', 'total_GB': 29.49, 'free_GB': 4.65, 'used_GB': 24.84, 'allocated_GB': 11.44, 'reserved_GB': 18.1, 'host_total_GB': 1511.6, 'host_available_GB': 1180.35, 'host_used_GB': 331.25, 'host_free_GB': 405.98}
[srlvr-mem] train_batch entry rollout=0 alloc=11.46GiB reserved=11.46GiB free=15.64GiB
```

### 4.3 TP2 是否尝试过

- **SRLVR-v2 / Qwen3-8B 的 TP2：未验证**。本仓库 `logs/` 下所有 8B 日志都不含 `tensor-model-parallel-size 2`（`grep -l` 无匹配），没有 SRLVR 8B 的 TP2 运行记录，也没有对应的 OOM 记录，因此“每卡约 2B 参数”不能作为 TP4 必须性的实测证明。
- 存在两次 **非 SRLVR-8B-TP2** 的 OOM 记录，仅作参考，不能直接外推：
  - `logs/qwen3-8b_smoke1.log`：NPU OOM，原文 `RuntimeError: NPU out of memory. Tried to allocate 194.00 MiB (NPU 1; 29.49 GiB total capacity; 26.73 GiB already allocated; 26.73 GiB current active; 364.00 MiB free; 26.88 GiB reserved in total by PyTorch)`；该日志没有 argv/并行度记录，无法确认 TP。
  - `logs/alfworld-qwen3-8b-grpo-s42x200-20261015_205919.log`：Ray **host-memory** OOM `ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.`，不是 NPU 显存 OOM。

---

## 5. Qwen3-4B SRLVR 的实际算法流程与时间构成（对照 GRPO）

### 5.1 数据来源与缺口

- 指定 run：`runs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_232412`（batch 臂，`rollout_0..49`，`iter_0000024` 落盘）。dashboard 事件文件名显示它跑在 `ma-job-ca427658-...-worker-1` 上。
- **该 run 的训练日志不在本机 `logs/` 中**：`logs/` 下该臂只有 10-30 的 5 个更早 attempt（020227 / 022359 / 040506 / 054022 / 124056，均早于 23:12 启动）。因此：
  - 每步周期：由该 run 的 `rollout_*.pt` **mtime** 实测（rollout 完成即写文件）；
  - 该 run 的 rollout / score / probe / actor 阶段拆分：**未记录**；
  - 阶段量级改用同臂兄弟 run `logs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_054022.log`（actor=2 / train_tp=2 / rollout=6，几何不同，仅作量级参照）。
- 算法配置可由 launcher 与落盘数据确认：run 名含 `batch`（advantage_mode=batch）；`run_alfworld_v2_srlvr_v2_4b.sh` 固定 G=1、ROLLOUT_BS=32、`SRLVR_CALIBRATION=4`、`SRLVR_GAMMA=0.99`、全量 RL；`.pt` 中 32 个 position、calibration 集合 = `[0,1,2,3]`、`policy_version=rollout_id`。

### 5.2 这一臂实际执行的算法流程（按代码顺序）

1. **rollout**（`tasks/alfworld/rollout.py::generate_rollout`）：G=1，32 个采样位置各生成 1 条真实环境轨迹；每条 turn row 由 `attach_rows` 写入 `position/turn/trajectory_id/policy_version/calibration/terminal_reward/teacher_view`。（rollout 0 实测 833 行、32 个 position、calibration=[0,1,2,3]。）
2. **convert**（`methods.srlvr_hsi.convert`）：turn rows → 训练样本；校验 position 集合 = `range(32)` 与 actor token 计数。
3. **train_batch**（`methods/srlvr_hsi.train_batch`）：
   - `ref_log_probs`：参考模型 logprob（KL 低方差项）；
   - **score_row 逐 row**：student/teacher 各一次 `_hidden_states`（整序列 no_grad 前向），对有效位置按 `position_chunk=64` 分块做 full-vocab log-prob，产出 `mu / f_chosen / e / student_logp`；
   - `fill_future`：用 `e` 计算未来校正 `B`（γ=0.99），末轮 B=0；
   - **calibration probes**：4 个 position 按 DP 轮转（4B TP2/DP4 时每 rank 1 个）；每个 position 对 Z（全部有 mask 的 turn）和 Y（除末轮）各跑 chunk=5 的 forward+backward，梯度以 FP32 累积到 CPU；
   - **C/D**：`_probe_moments` 得到每参数 zz/zy/yy → TP all-reduce → `_finalize_probe_records` 求 `C_hat/D_hat`；
   - **batch advantage**：`b` = 本批成功率、`s` = 本批奖励残差二阶矩、λ = 上一批冻结的 clip(C/D)；
   - **loss**：token-level PPO clip（0.8–1.2，length-normalized）+ λ·B 的未来校正 + 0.001·KL(ref)；
   - optimizer step；每 25 个 rollout 保存完整训练状态。

### 5.3 实测时间（同 4B、cap 4096、32 episodes/step）

| 指标 | 4B SRLVR（batch，指定 run） | 4B GRPO（`...-s42-20261004_125000`） |
|---|---:|---:|
| 统计范围 | 49 个相邻周期（50 rollouts） | 114 个相邻周期（115 rollouts） |
| 每步周期中位 | **29.2 min** | **30.1 min** |
| 周期 p25–p75 | 26.6–32.2 min | 25.7–34.7 min |
| 长间隔（>120 min，已从分位统计剔除） | 0 个 | 1 个（max 9429 min） |
| rollout 墙钟中位 | 未记录（日志缺失） | **23.7 min** |
| step − rollout（中位） | 未记录 | ≈6.3 min（推导） |
| 每步生成 token（中位） | 617,685（50 个 `rollout_*.pt` 实测） | 678,745 |
| 每步 turn rows（中位） | 831 | 未记录 |

对照 run 的配置：GRPO `G=8 bs=4 GBS=32 hist=8 cap=4096`；SRLVR `G=1 bs=32 GBS=32 hist=0 cap=4096`（`hist=0` 在 `_v2_defaults.sh` 里表示保留完整历史，`hist=8` 表示截断到 8 轮）。

### 5.4 同臂 4B 的阶段量级（几何不同，仅作参照）

`..._054022.log`（actor=2 / train_tp=2 / rollout=6；该 run 的 cycle 约 60 min，比指定 run 慢，说明两者不是同一放置）：

| 阶段 | 中位 | 范围 |
|---|---:|---:|
| score | 702 s | 627–884 |
| ref logprob | 202 s | 182–262 |
| calibration probes | 761 s | 595–1124 |
| actor step | 697 s | 630–864 |

### 5.5 对“4B 下 SRLVR 与 GRPO 时间相当”的分析

**实测结论**：指定 4B SRLVR run 的每步周期中位 **29.2 min**，4B GRPO 参照 run 为 **30.1 min**，两者基本相等（差 0.9 min）。

可以确认的事实：

1. 每步都是 32 条 episode、cap 4096；SRLVR 每步生成 token（617,685）比 GRPO（678,745）少约 9%。
2. GRPO 参照 run 的 rollout 占其周期的 **79%**（23.7/30.1 min），actor 侧只占约 6–7 min。
3. SRLVR 的 actor 侧（score + probes + ref + actor）在 4B 上是可观的（同臂兄弟 run 中位：2362 s，但那是 DP1 几何；指定 run 的几何未记录）。

**因此，“时间相当”不是 SRLVR 的额外开销消失了，而是两项成本在 4B 上大致抵消**；下列机制与实测一致，但**未做逐项归因**（指定 run 的阶段拆分缺失）：

- SRLVR 的 rollout 负载略小：每步生成 token 少 9%（32 个不同任务、G=1），rollout 是两边周期的主项；
- SRLVR 多出 score + calibration + ref 的 actor 侧开销。指定 run 没有阶段拆分，**无法验证这部分到底抵消了多少**；兄弟 run（DP1）的 score 中位 702 s / actor 697 s 与 8B（DP2）的 319 s / 319 s 几何不同，不能按模型规模直接换算，因此这里不给“4B 应为 8B 的 x 倍”之类的推算；
- GRPO 参照 run 使用 `hist=8`（截断历史）而 SRLVR 用 `hist=0`（完整历史），两者对 prompt/prefill 长度的影响相反，使每步总 token 量落在同一量级。

**反例/边界**：同一套 SRLVR 流程在 8B 上抵消关系被打破——rollout 27.6 min 之外还要加约 21.6 min 的 actor 侧（score 5.3 + probes 10.2 + ref 1.5 + actor 5.3），单步 ≈ 49 min，而用户给出的 8B GRPO 对照是通过 async 把 rollout 与训练重叠（本仓库的 async 需要非共卡，见 §4）。所以“4B 相当”是 4B 的算力/放置与任务长度共同决定的，不应外推到 8B。

## 6. 数据来源

- 日志：`cua/repos/opsd-slime/logs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837.log`（本次 attempt 与 11-05 首次 attempt 追加在同一文件）
- 每 rollout turn rows：`cua/repos/opsd-slime/runs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837/rollout_{1,4,8}.pt`
- checkpoint 目录：`cua/repos/opsd-slime/runs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837/`（本材料生成时 `iter_0000009` 已按用户指示删除，`latest_checkpointed_iteration.txt` 仍为 9；下一个 checkpoint 为 `iter_24`）
- 4B：`runs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_232412/rollout_*.pt`（周期/行数/token）、`logs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_054022.log`（阶段参照）；GRPO 对照 `logs/alfworld-qwen3-4b-grpo-s42-20261004_125000.log`
- 代码：`methods/srlvr_hsi.py`、`methods/srlvr.py`、`methods/hsi2_jsd.py`、`methods/srlvr_compact.py`、`frameworks/slime-ascend/slime/ray/rollout.py`、`frameworks/slime-ascend/slime/utils/data.py`、`frameworks/slime-ascend/slime/backends/megatron_utils/actor.py`

---

## 7. score 与 rollout 的概率差异核查（2026-11-06）

### 7.1 结论（先给结果）

- 差异**不在 TP 归一化**：用 4 进程 gloo 直接调用 `_local_global_log_probs` / `_centered_features_sharded`，`sum_v p = 1.0`，与 `torch.log_softmax` 的最大误差 `9.5e-7`，非 owner rank 对 `picked/student` 贡献为 0。
- 差异**不在 token 对齐**：`ids[plen:] == tokens[-n:]` 的校验在代码里，且 `.pt` 中 716/716 行成立。
- 差异**不在权重版本**：同一份代码下，**4B（tied embedding）的 `[srlvr-hsi-views]` bias ≈ −0.0013、mae ≈ 0.0148**；**8B（`UNTIE_EMBEDDINGS=1`）的 bias ≈ −21.43…−21.61、mae ≈ 同值**。
- 根因定位到**概率计算的 lm-head 权重选择**：`_lm_head_weight()` 优先返回 `model.shared_embedding_or_output_weight()`；Megatron 的该函数在 `pre_process=True`（PP=1 时恒为真）分支**直接返回输入 embedding 权重**，不检查 `share_embeddings_and_output_weights`。4B 是 tied，返回的 embedding 就是真正的 lm head；8B 用 `--untie-embeddings-and-output-weights`，embedding ≠ output layer，于是 score 的 logits 用错了矩阵。
- 影响范围**不止日志**：`mu/picked/e/student_logp` 都来自同一组 `lp/lq`，所以 8B 的 `e → fill_future → B`（未来校正项）也是用错误分布算的；actor 更新本身走 Megatron 正常 forward（正确），probes 走 `train()`（正确）。

### 7.2 相关源码

词表分片 log-prob：

`srlvr_hsi.py:352-380`

```python
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


```


分片特征与 chosen 汇总：

`srlvr_hsi.py:382-404`

```python
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


```


lm-head 权重选择（问题所在）：

`hsi2_jsd.py:218-228`

```python
def _lm_head_weight(model, head):
    """The ``[V, H]`` matrix the head applies, tied or not."""
    getter = getattr(model, "shared_embedding_or_output_weight", None)
    if callable(getter):
        weight = getter()
        if weight is not None:
            return weight
    weight = getattr(head, "weight", None)
    if weight is None:
        raise RuntimeError("HSI v2: the actor's output layer exposes no weight matrix")
    return weight
```


Megatron 侧 `shared_embedding_or_output_weight`（`pre_process` 分支优先）：

`gpt_model.py:702-721`

```python
    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None
```


reference 评分后恢复 actor、再进入 score：

`srlvr_hsi.py:1156-1192`

```python
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
```


### 7.3 10 个有效 token（8B `rollout_9.pt`，position 0 / turn 0 / policy_version=9）

| # | token 位置 | token id | rollout logp | score logp | 正常 actor 前向 logp |
|---:|---:|---:|---:|---|---|
| 0 | 0 | 151667 | -0.000000 | 无法复现 | 无法复现 |
| 1 | 1 | 198 | -0.000075 | 无法复现 | 无法复现 |
| 2 | 2 | 32313 | -0.000002 | 无法复现 | 无法复现 |
| 3 | 3 | 11 | 0.000000 | 无法复现 | 无法复现 |
| 4 | 4 | 1077 | -0.023293 | 无法复现 | 无法复现 |
| 5 | 5 | 594 | -0.000075 | 无法复现 | 无法复现 |
| 6 | 6 | 1490 | -0.038042 | 无法复现 | 无法复现 |
| 7 | 7 | 13 | 0.000000 | 无法复现 | 无法复现 |
| 8 | 8 | 576 | -0.008615 | 无法复现 | 无法复现 |
| 9 | 9 | 3383 | -0.000096 | 无法复现 | 无法复现 |

**为什么两列无法复现**：`student_logp` 在训练时计算后没有落盘；且该 batch 的“更新前权重”（policy_version=9）从未保存——本 run 的 checkpoint 间隔是 25，只有 `iter_0000009`（已按用户指示删除）与 `iter_0000024`（=step 24 之后，比 rollout 9 晚 15 次更新）。用当前 checkpoint 重放旧样本会把权重差异混入，不能作为判据。

可用的间接判据：rollout logp 中位数 0.000（30.6% 恰为 0，54.6% 绝对值 <1e-3），说明同一权重下被采样 token 的概率接近 1；同分布的 score logp 不可能平均低 21.5 nats。这与“score 用了错误分布”一致。

### 7.4 TP 词表归约的实测校验

用 4 进程 gloo（V=16, TP=4, 每片 4 个 token，真实调用 `_local_global_log_probs` 与 `_centered_features_sharded`）：

```text
sum_v p            = [1.0, 1.0, 1.0]         # 每个位置的全词表概率和
max |lp - ref|     = 9.5e-07                 # 与 torch.log_softmax(全词表) 的最大误差
picked             = [0.5, 0.5, 0.5]         # v2 的 f = p/(p+q)，p=q 时为 0.5
student            = [-4.493629, -12.955717, -10.624264]   # 与 ref logp 完全一致
ref logp           = [-4.493629, -12.955717, -10.624263]
rank1: owns targets [1]; contributes 0 to the rest
rank2: owns targets [];  contributes 0 to the rest
rank3: owns targets [2]; contributes 0 to the rest
```

owner 判定为 `target - start ∈ [0, 每片大小)`，非 owner 的 `picked_local/student_local` 被 `torch.where(in_range, ..., 0)` 置零；归一化项 `local_p` 只累加本片真实概率，`tp_sum` 汇总后等于全词表归一化常数。**该链路没有发现错误。**

### 7.5 为什么 4B 不受影响

| | 4B | 8B |
|---|---|---|
| `UNTIE_EMBEDDINGS` | 未设置（Qwen3-4B `tie_word_embeddings: true`） | `_8b_backbone.sh:42` 设为 1（argv 有 `--untie-embeddings-and-output-weights`） |
| `shared_embedding_or_output_weight()` 返回 | 输入 embedding（= 真正的 lm head） | 输入 embedding（**≠** output layer） |
| `[srlvr-hsi-views]` bias | **−0.0013 / −0.0012**（mae 0.0148 / 0.0145） | **−21.61 / −21.43**（mae 同量级） |
| 结论 | score 与 rollout 一致 | score 用错矩阵 |

4B 的数字取自 `logs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_054022.log`（及 history 同批）r0/r1；8B 取自本 run r0/r1。两边的 `support=151936` 相同。

### 7.6 需要的确认（不改训练即可做）

在任意一次 score 调用时打印一行：`weight is model.embedding.word_embeddings.weight`，以及 `model.share_embeddings_and_output_weights`。若前者为 True 且后者为 False，即确认本结论。修复方向是优先使用 `head.weight`（untied 时）或在 `pre_process` 分支检查 `share_embeddings_and_output_weights`；按本次要求未改代码。

---

## 8. 指定 4B run 的实测阶段耗时（第 2 项）

数据只取该 run 自己的 artifact：`dashboard/events-ma-job-ca427658-...-worker-1-...jsonl`（每步 train metrics + 时间戳）与 `rollout_*.pt` 的 mtime（rollout 完成时刻）。dashboard **只记录 train metrics**，没有 score/calibration/reference/actor 的计时字段。

| batch | rollout（min） | score | calibration | reference | actor | 完整周期（min） |
|---:|---:|---|---|---|---|---:|
| 1 | 18.5 | 未记录 | 未记录 | 未记录 | 未记录 | 31.2 |
| 2 | 20.4 | 未记录 | 未记录 | 未记录 | 未记录 | 32.7 |
| 3 | 18.3 | 未记录 | 未记录 | 未记录 | 未记录 | 29.2 |

口径：`rollout = rollout_k.pt mtime − step_{k-1} 结束时间`；`完整周期 = step_k 结束 − step_{k-1} 结束`；`其余阶段 = 周期 − rollout`（batch 1–3 分别为 12.7 / 12.3 / 10.9 min，但无法拆成 score/calibration/reference/actor）。

该 run 的 dashboard config 记录：`rollout_batch_size=32`、`n_samples_per_prompt=1`、`global_batch_size=32`、`lr=1e-6`、`seq_length=32768`、`start_rollout_id=0`。**实际 TP/DP、是否共卡/异步、probe 累加位置与 chunk、是否执行 student/teacher 双前向：该 run 的日志缺失，均未记录**；不能用其它 attempt 或兄弟 run 替代。

