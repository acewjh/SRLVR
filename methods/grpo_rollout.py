"""G-aware rollout for STANDARD GRPO on the math task.

Why this exists: `math_rollout.py` is G=1 only (`sample = group[0]`) because the OPSD
family always uses one sample per prompt with a privileged teacher attached. GRPO needs
G > 1 samples per prompt and needs NO teacher, so the OPSD rollout cannot serve it.

What it does differently from `math_rollout.py`:
  * generates ALL G samples of a group (standard GRPO group sampling);
  * attaches NO teacher payload (GRPO uses the plain verifier reward);
  * leaves `sample.reward` to slime's `--custom-rm-path` (math_opsd.rm_math_verify),
    which is the same repaired official grader used everywhere else.

Everything else is deliberately identical to the OPSD harness so the GRPO arm is
comparable to the Reward-CV / HPI arms: the same blind student prompt (thinking OFF),
the same sampling parameters (T=1.1 / top_p 0.95 / top_k 20), the same 1024-token cap.
"""

import asyncio
import logging
import os
from argparse import Namespace

import torch

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import get, post
from slime.utils.types import Sample

from tasks.math.prompts import build_student_prompt_ids

__all__ = ["generate_rollout"]

from opsd_core.telemetry import observe_group, generation_started

logger = logging.getLogger(__name__)


async def _generate_one(
    args: Namespace,
    state: GenerateState,
    sample: Sample,
    student_prompt_ids: list[int],
    sampling_params: dict,
) -> None:
    """One sample of a GRPO group: student rollout on SGLang (token-in, tokens-out)."""
    generation_started(sample)
    payload = {
        "input_ids": student_prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    engine_url = os.environ.get("OPSD_ENGINE_URL") or (
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    )
    # Bounded attempts: slime's client uses httpx.Timeout(None), so an unbounded
    # request turns a stalled engine into a silent whole-run hang.
    timeout_s = float(os.environ.get("OPSD_ROLLOUT_TIMEOUT_S", "900"))
    attempts = max(1, int(os.environ.get("OPSD_ROLLOUT_ATTEMPTS", "3")))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            output = await asyncio.wait_for(post(f"{engine_url}/generate", payload), timeout=timeout_s)
            break
        except (asyncio.TimeoutError, TimeoutError) as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning(
                "[grpo-rollout] /generate exceeded %.0fs (attempt %d/%d) engine=%s -- retrying",
                timeout_s, attempt + 1, attempts, engine_url,
            )
    else:
        raise RuntimeError(
            f"/generate did not return after {attempts} attempts x {timeout_s}s ({engine_url})"
        ) from last_exc

    meta = output["meta_info"]
    if "output_token_logprobs" in meta:
        response_tokens = [item[1] for item in meta["output_token_logprobs"]]
        response_log_probs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        response_tokens, response_log_probs = [], []

    sample.tokens = list(student_prompt_ids) + list(response_tokens)
    sample.response_length = len(response_tokens)
    sample.response = output["text"]
    sample.rollout_log_probs = list(response_log_probs)
    sample.reward = 0.0  # replaced by --custom-rm-path
    if sample.response_length == 0:
        sample.status = Sample.Status.TRUNCATED
    elif meta.get("finish_reason", {}).get("type") == "length":
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED


@observe_group("math")
async def _generate_group(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict,
) -> list[Sample]:
    """Standard GRPO group: ALL G samples for the same prompt."""
    sample = group[0]
    problem = sample.prompt
    student_prompt_ids = build_student_prompt_ids(state.tokenizer, problem)

    if state.aborted:
        for s in group:
            s.status = Sample.Status.ABORTED
        return group

    async with state.semaphore:
        if state.aborted:
            for s in group:
                s.status = Sample.Status.ABORTED
            return group
        await asyncio.gather(
            *[_generate_one(args, state, s, student_prompt_ids, sampling_params) for s in group]
        )

    # [OPSD 2026-09-21] Apply the reward model HERE.
    #
    # slime only calls `async_rm` / `batched_async_rm` from
    # slime/rollout/sglang_rollout.py -- i.e. from the STOCK rollout path. A custom
    # rollout function is responsible for its own rewards, and this one assumed
    # otherwise: `_generate_one` sets `sample.reward = 0.0  # replaced by
    # --custom-rm-path`, but nothing ever replaced it. Every sample therefore scored
    # 0.0, so every GRPO group had zero reward variance, every advantage was 0, and
    # the run produced `train/loss = 0.0` and `train/grad_norm = 0.0` for 12+ steps:
    # an expensive way to copy the base model.
    #
    # Using `async_rm` (per sample, not the batched variant) is deliberate: it is the
    # entry point that honours `args.custom_rm_path`, and our `rm_math_verify` takes a
    # SINGLE sample, whereas `batched_async_rm` would hand it the whole list.
    from slime.rollout.rm_hub import async_rm

    for s in group:
        s.loss_mask = [1] * s.response_length
        s.reward = float(await async_rm(args, s))
        s.metadata = s.metadata or {}
        s.metadata["dashboard_graded"] = True
    return group


def generate_rollout(args: Namespace, rollout_id: int, data_source, evaluation: bool = False):
    """Custom rollout entry point (see slime.ray.rollout.call_rollout_fn)."""
    from slime.utils.async_utils import run

    assert not evaluation, "grpo rollout does not implement eval generation"
    return run(_generate_rollout_async(args, rollout_id, data_source))


async def _generate_rollout_async(args: Namespace, rollout_id: int, data_source):
    # [OPSD 2026-09-21] `slime.rollout.sglang_rollout` does NOT export an `_abort`
    # helper -- the original import here raised
    #   ImportError: cannot import name '_abort' from 'slime.rollout.sglang_rollout'
    # and killed the GRPO smoke gate at its first rollout. The bounded abort helper
    # lives in our own math_rollout; import that instead.
    from tasks.math.math_rollout import _abort

    state = GenerateState(args)
    sampling_params = dict(state.sampling_params)
    target_data_size = args.rollout_batch_size  # number of PROMPTS

    data: list[list[Sample]] = []
    ready: list[list[Sample]] = []
    pendings: set[asyncio.Task] = set()

    while len(data) < target_data_size:
        while len(ready) + len(pendings) + len(data) < target_data_size:
            if hasattr(data_source, "get_samples"):
                samples = data_source.get_samples(args.over_sampling_batch_size)
            else:
                samples = data_source(args.over_sampling_batch_size)
            for group in samples:
                pendings.add(asyncio.create_task(_generate_group(args, state, group, sampling_params)))
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            ready.append(task.result())
        while ready and len(data) < target_data_size:
            data.append(ready.pop(0))

    assert len(data) == args.rollout_batch_size
    await _abort(args, state)
    state.reset()

    data = sorted(data, key=lambda group: group[0].index)
    n_samples = sum(len(g) for g in data)
    metrics = {
        "rollout/avg_response_len": sum(s.response_length for g in data for s in g) / max(n_samples, 1),
        "rollout/n_truncated": sum(
            1 for g in data for s in g if s.status == Sample.Status.TRUNCATED
        ),
        "rollout/group_size": n_samples / max(len(data), 1),
    }
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
