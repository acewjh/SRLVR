"""Default local training telemetry. Standard library only; no accelerator imports."""
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import math
import os
from pathlib import Path
import socket
import time
import uuid

log = logging.getLogger(__name__)
_current = contextvars.ContextVar("opsd_dashboard_batch", default=None)
_writers = {}
_warned = set()


def enabled():
    return os.environ.get("OPSD_DASHBOARD", "1").lower() not in {"0", "false", "off", "no"}


def data_dir(args):
    override = os.environ.get("OPSD_DASHBOARD_DIR")
    path = override or getattr(args, "save", None)
    if not path:
        return None
    root = Path(path).expanduser().resolve()
    # ALFWorld launchers use --save=<run>/checkpoints; math/code may save
    # directly under <run>. Explicit dashboard overrides retain their meaning.
    if not override and root.name in {"checkpoints", "checkpoint"}:
        root = root.parent
    return root / "dashboard"


def clean(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except (ValueError, RuntimeError):
            return None
    return str(value)


def _warn(exc):
    key = (type(exc).__name__, str(exc))
    if key not in _warned:
        _warned.add(key)
        log.warning("Dashboard data could not be saved: %s", exc)


def emit(args, event, trace=None):
    """One append-only file per process; no shared SQLite/WAL on the shared FS."""
    if not enabled() or data_dir(args) is None:
        return
    try:
        root = data_dir(args)
        root.mkdir(parents=True, exist_ok=True)
        key = (str(root), os.getpid())
        if key not in _writers:
            writer_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
            _writers[key] = root / f"events-{writer_id}.jsonl"
            metadata = {
                "kind": "config", "time": time.time(), "schema": 1,
                "run": root.parent.name,
                # Only explicit non-secret experiment fields, never dump credentials.
                "config": {k: getattr(args, k, None) for k in (
                    "hf_checkpoint", "seed", "rollout_function_path", "custom_rm_path",
                    "custom_loss_function_path", "rollout_batch_size", "n_samples_per_prompt",
                    "global_batch_size", "lr", "seq_length", "save", "start_rollout_id")},
                "environment": {k: os.environ.get(k) for k in (
                    "ARM", "OPSD_METHOD", "MODEL_SLUG", "MODEL_TASK", "SRLVR_VARIANT")},
            }
            _append(_writers[key], metadata)
        event = dict(schema=1, time=time.time(), **event)
        if trace is not None:
            trace_path = root / "trajectories" / (event["id"] + ".json")
            trace_path.parent.mkdir(exist_ok=True)
            temp = trace_path.with_suffix(".tmp")
            temp.write_text(json.dumps(clean(trace), ensure_ascii=False, allow_nan=False), encoding="utf-8")
            temp.replace(trace_path)
            event["trace"] = str(trace_path.relative_to(root))
        _append(_writers[key], event)
    except (OSError, TypeError, ValueError) as exc:
        _warn(exc)


def _append(path, event):
    # A process is the sole writer of its file; no buffered line survives a crash.
    payload = (json.dumps(clean(event), ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def record_metrics(args, metrics, step_key):
    emit(args, {"kind": "metrics", "step_key": step_key,
                "step": metrics.get(step_key), "metrics": metrics})


def _status(sample):
    value = getattr(sample, "status", None)
    return getattr(value, "value", value) or "unknown"


def sample_record(sample, task, seconds=None, error=None):
    md = getattr(sample, "metadata", None) or {}
    tokens = getattr(sample, "tokens", None)
    output = getattr(sample, "response_length", None) if tokens is not None else None
    prompt = len(tokens) - output if tokens is not None and output is not None else None
    status = "error" if error else _status(sample)
    success = md.get("dashboard_success")
    if task == "code":
        success = (md.get("code_grpo_rm") or {}).get("exact_pass")
    elif task == "math" and success is None:
        # OPSD's placeholder reward=0 is not a verifier result.
        if md.get("outcome_audit") is not None or md.get("dashboard_graded"):
            success = getattr(sample, "reward", None) == 1
    return {
        "task": task, "task_id": str(getattr(sample, "index", None)),
        "status": status, "completed": status == "completed" if status != "unknown" else None,
        "success": success, "reward": getattr(sample, "reward", None),
        "duration_seconds": seconds, "turns": 1 if tokens is not None else None,
        "prompt_tokens": prompt, "generated_tokens": output,
        # Single-turn exact sequence length; multi-turn handled at episode level.
        "trajectory_tokens": len(tokens) if tokens is not None else None,
        "prompt_tokens_processed": prompt, "valid_actions": None, "actions": None,
        "error": error,
    }


def _attempt(record, trace=None):
    batch = _current.get()
    if batch is None:
        return None
    record = dict(kind="attempt", id=uuid.uuid4().hex, batch_id=batch["id"],
                  step=batch["step"], phase=batch["phase"], **record)
    batch["count"] += 1
    emit(batch["args"], record, trace)
    return record["id"]


def observe_group(task):
    """Capture original samples before HSI chunks replace them with training rows."""
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapped(args, state, group, *a, **kw):
            if not enabled() or _current.get() is None:
                return await fn(args, state, group, *a, **kw)
            start = time.perf_counter()
            error = None
            result = None
            try:
                result = await fn(args, state, group, *a, **kw)
                return result
            except BaseException as exc:
                error = "cancelled" if isinstance(exc, asyncio.CancelledError) else type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                ids = []
                # HSI/OPSD consume only group[0]; GRPO consumes all samples.
                originals = group if fn.__module__ in {"methods.grpo_rollout", "tasks.code.rollout"} else group[:1]
                for sample in originals:
                    md = getattr(sample, "metadata", None) or {}
                    began = md.pop("_dashboard_started", start)
                    record = sample_record(sample, task, time.perf_counter() - began, error)
                    trace = {"prompt": getattr(sample, "prompt", None), "response": getattr(sample, "response", None),
                             "tokens": getattr(sample, "tokens", None), "verdict": md.get("code_grpo_rm") or md.get("outcome_audit")}
                    ident = _attempt(record, trace)
                    ids.append(ident)
                for row in result or []:
                    row.metadata = row.metadata or {}
                    row.metadata["dashboard_attempt_ids"] = ids
        return wrapped
    return decorate


def generation_started(sample):
    if enabled() and _current.get() is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["_dashboard_started"] = time.perf_counter()


def episode_record(episode):
    t = episode.trajectory
    turns = t.turns
    generated = sum(len(x.generated_token_ids) for x in turns)
    initial = len(turns[0].prompt_token_ids) if turns else 0
    # All intermediate observations/template changes are in trace. No fabricated
    # unique observation token count: prompts are re-rendered, not prefix extensions.
    return dict(task="alfworld", task_id=t.task_id, task_type=t.task_type, split=t.split,
                status="truncated" if t.truncated else "completed",
                completed=bool(turns and turns[-1].done), success=t.success,
                reward=t.terminal_reward, turns=len(turns), duration_seconds=episode.wall_seconds,
                generation_seconds=episode.generation_seconds, environment_seconds=episode.environment_seconds,
                generated_tokens=generated, prompt_tokens=initial,
                trajectory_tokens=initial + generated,
                prompt_tokens_processed=sum(len(x.prompt_token_ids) for x in turns),
                valid_actions=sum(bool(x.action_valid) for x in turns), actions=len(turns))


def observe_episode(fn):
    @functools.wraps(fn)
    async def wrapped(args, tokenizer, game, *a, **kw):
        if not enabled() or _current.get() is None:
            return await fn(args, tokenizer, game, *a, **kw)
        start = time.perf_counter()
        try:
            result = await fn(args, tokenizer, game, *a, **kw)
        except BaseException as exc:
            _attempt(dict(task="alfworld", task_id=game.game_id, status="error",
                          completed=False, success=None, duration_seconds=time.perf_counter()-start,
                          error=type(exc).__name__ + ": " + str(exc)))
            raise
        _attempt(episode_record(result), result.trajectory.to_dict())
        return result
    return wrapped


def _flatten(samples):
    for item in samples:
        if isinstance(item, list):
            yield from _flatten(item)
        else:
            yield item


def capture_rollout(original, fn, args, rollout_id, *a, evaluation=False, **kw):
    """Called by compat hook at the common pre-trim rollout boundary."""
    if not enabled() or data_dir(args) is None:
        return original(fn, args, rollout_id, *a, evaluation=evaluation, **kw)
    batch = dict(id=uuid.uuid4().hex, args=args, step=rollout_id,
                 phase="eval" if evaluation else "train", count=0)
    token = _current.set(batch)
    start = time.perf_counter()
    ok = False
    selected = set()
    try:
        output = original(fn, args, rollout_id, *a, evaluation=evaluation, **kw)
        # Generic eval / non-instrumented rollout fallback: exact token counts,
        # unknown per-sample wall time; never divide batch latency by sample count.
        if batch["count"] == 0:
            datasets = output.data if evaluation else {"train": {"samples": output.samples}}
            for dataset, info in datasets.items():
                for sample in _flatten(info.get("samples", [])):
                    record = sample_record(sample, "unknown")
                    record["dataset"] = dataset
                    md = getattr(sample, "metadata", None) or {}
                    code = md.get("code_grpo_rm")
                    if code:
                        record.update(task="code", success=code.get("exact_pass"))
                    elif "math" in str(getattr(args, "custom_rm_path", "")):
                        record.update(task="math", success=getattr(sample, "reward", None) == 1)
                    _attempt(record, {"prompt": sample.prompt, "response": sample.response, "tokens": sample.tokens})
        elif not evaluation:
            for sample in _flatten(output.samples):
                selected.update((getattr(sample, "metadata", None) or {}).get("dashboard_attempt_ids", []))
        if output.metrics:
            record_metrics(args, dict(output.metrics, **{batch["phase"] + "/step": rollout_id}), batch["phase"] + "/step")
        ok = True
        return output
    finally:
        emit(args, {"kind": "batch", "id": batch["id"], "step": rollout_id,
                    "phase": batch["phase"], "ok": ok, "attempts": batch["count"],
                    "duration_seconds": time.perf_counter() - start,
                    "selected_attempt_ids": sorted(x for x in selected if x)})
        _current.reset(token)
