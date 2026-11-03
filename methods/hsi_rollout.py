"""Math HSI rollout -- same-policy hindsight credit over a chunked single generation.

This is the math analogue of the ALFWorld v2 HSI arm (``tasks.alfworld/rollout.py``
with ``ALFWORLD_V2_HSI_V2=1``): the rollout emits **one training row per chunk** of
the completion, exactly as the ALFWorld arm emits one row per turn. That lets the
whole validated downstream stack be reused verbatim:

  * ``methods.hsi2_jsd.postprocess``                     -- SIGN x exact-JSD credit
  * ``tasks.alfworld.hsi_provider.custom_model_provider`` -- the theta_old handle
  * ``tasks.alfworld.loss.alfworld_v2_loss``              -- PPO clip 0.2, one epoch
  * ``tasks.alfworld.train_data.convert_samples_to_train_data`` -- transport + weighting

For chunk ``c = [start, end)`` of the completion::

    sample.tokens      = student_prompt_ids + completion_ids[:end]
    response_length    = end - start
    rollout_log_probs  = the sampler's own logp for completion_ids[start:end]
    hsi_teacher_view   = (view_ids, prompt_len, target_len)
        view_ids   = chat(user: hindsight block) ++ completion_ids[:end]
        prompt_len = len(chat_prefix) + start
        target_len = end - start
    hsi_candidates     = [(ids, probs)] per target position, from the engine's top-K

The hindsight block is::

    [HINDSIGHT PRIVILEGED INFORMATION]
    <outcome>...</outcome>            <- section 3 verifier outcome descriptor
    FUTURE REASONING AND ANSWER:
    <decoded completion_ids[end:] | (the trajectory ended here)>
    [/HINDSIGHT PRIVILEGED INFORMATION]

    Problem:
    <problem>

Section 5's leak rules are enforced structurally: the future is ``completion_ids[end:]``
(a strict suffix of the target span, so ``y_t`` is never repeated), no ground-truth
solution or reference answer ever enters the block, and only the verifier's own verdict
reaches the model (never a bare reward number). ``assert_no_leak`` additionally refuses
any overlap between the future span and the target span.

Sampler contract (section 4): the engine is asked for exactly ``HSI_CANDIDATE_TOP_K``
candidates per position from a temperature-1 / full-support policy (``--rollout-top-p
1.0 --rollout-top-k -1``), so ``C_t`` is the head of the true softmax and ``rho_t`` --
``sampling_probabilities`` with ``top_p = 1.0`` -- is ``pi_old`` renormalised inside it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from argparse import Namespace

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from opsd_core.chunking import split_chunks
from tasks.math.prompts import build_student_prompt_ids

__all__ = ["generate_rollout", "build_hsi_view", "render_hindsight_block"]

from opsd_core.telemetry import observe_group, generation_started

logger = logging.getLogger(__name__)

_FIRST_SAMPLE_PRINTED = False

#: The candidate budget is a property of the candidate transport
#: (``tasks.alfworld.hsi_shared``); read the same env the engine request uses so
#: the two can never disagree.
_DEFAULT_TOP_K = 128

#: How many contiguous chunks the completion is split into. Each chunk becomes one
#: training row and gets its own hindsight view (future = the suffix after it).
#: Chunk 0 carries the longest future; the LAST chunk has no continuation left and
#: is therefore scored on the outcome descriptor alone -- exactly as the ALFWorld
#: arm scores its final turn.
_DEFAULT_CHUNKS = 4

_NO_FUTURE = "(the trajectory ended here)"

_BLOCK_TEMPLATE = (
    "[HINDSIGHT PRIVILEGED INFORMATION]\n"
    "{outcome}\n"
    "FUTURE REASONING AND ANSWER:\n"
    "{future}\n"
    "[/HINDSIGHT PRIVILEGED INFORMATION]\n"
    "\n"
    "Problem:\n{problem}"
)


def _candidate_top_k() -> int:
    return int(os.environ.get("ALFWORLD_V2_HSI_CANDIDATE_K") or _DEFAULT_TOP_K)


def _estimator() -> str:
    """``jsd`` (the 2026-10-04 V2 spec) or ``hsi_v2`` (retired 2026-10-15)."""
    return os.environ.get("OPSD_HSI_ESTIMATOR", "jsd").strip().lower() or "jsd"


def _wants_candidates() -> bool:
    """Whether the engine must also return a per-position top-K candidate table.

    The JSD estimator's WHERE is a full-vocab divergence between two views of the
    same policy, so it has no candidate set at all -- asking the engine for 128
    extra log-probs per token would be pure transport cost. The retired
    candidate-centred estimator still needs them.
    """
    return _estimator() != "jsd"


def _chunk_count() -> int:
    return max(1, int(os.environ.get("OPSD_HSI_CHUNKS") or _DEFAULT_CHUNKS))


# ---------------------------------------------------------------------------
# View construction (sections 3 and 5)
# ---------------------------------------------------------------------------


def render_hindsight_block(problem: str, future_text: str, outcome_block: str) -> str:
    """The privileged block text: outcome descriptor + realised future + problem.

    ``outcome_block`` is the ``<outcome>...</outcome>`` text the shared renderer
    produces; the verifier's real semantics are carried by that string, never as a
    bare scalar.
    """
    body = future_text if future_text and future_text.strip() else _NO_FUTURE
    return _BLOCK_TEMPLATE.format(outcome=outcome_block, future=body, problem=problem)


def _chat_user_ids(tokenizer, user_content: str) -> list[int]:
    """Chat-templated USER turn ending at the assistant prompt (thinking off).

    ``enable_thinking=False`` matches the student rollout, which builds its prompt
    the same way, so the replay view is faithful to the policy's own convention.
    """
    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_hsi_view(
    tokenizer,
    problem: str,
    completion_ids: list[int],
    chunk: tuple[int, int],
    outcome_block: str,
) -> tuple[list[int], int, int]:
    """``(view_ids, prompt_len, target_len)`` for one chunk.

    ``view_ids = chat(user: block) ++ completion_ids[:end]`` and the scored span is
    ``view_ids[prompt_len : prompt_len + target_len]``. The prefix
    ``completion_ids[:start]`` rides along as RAW sampled tokens (the model's own
    autoregressive state), never restated as user text.
    """
    start, end = chunk
    future_text = tokenizer.decode(completion_ids[end:], skip_special_tokens=True)
    user_ids = _chat_user_ids(tokenizer, render_hindsight_block(problem, future_text, outcome_block))
    prefix_ids = [int(t) for t in completion_ids[:end]]
    view_ids = list(user_ids) + prefix_ids
    prompt_len = len(user_ids) + start
    target_len = end - start
    return view_ids, prompt_len, target_len


def assert_no_leak(chunk: tuple[int, int], rlen: int) -> None:
    """Fail closed if the scored span and its future are not disjoint.

    The future is ``completion_ids[end:]`` by construction, so this can only fire if
    a future edit moves the boundary. It is the one invariant section 5 names
    explicitly ("当前 y_t 本身不得重复放进 future").
    """
    start, end = chunk
    if not (0 <= start < end <= rlen):
        raise AssertionError(f"HSI chunk [{start},{end}) is not inside [0,{rlen})")
    if end > rlen:
        raise AssertionError("HSI future span starts before the target span ends -- leak")


# ---------------------------------------------------------------------------
# Engine call (token-in / tokens-out, plus the sampler's own top-K)
# ---------------------------------------------------------------------------


async def _generate_student(
    args: Namespace,
    sample: Sample,
    student_prompt_ids: list[int],
    sampling_params: dict,
) -> dict:
    """One bounded ``/generate`` call returning tokens, logprobs and top-K candidates."""
    params = dict(sampling_params)
    payload = {
        "input_ids": student_prompt_ids,
        "sampling_params": params,
        "return_logprob": True,
    }
    if _wants_candidates():
        # The candidate set is read off the SAMPLER's own distribution, not a
        # re-forward. sglang takes top_logprobs_num as a TOP-LEVEL field of
        # /generate (not a SamplingParams key); mirror tasks.alfworld.episode.
        payload["top_logprobs_num"] = _candidate_top_k()
    engine_url = os.environ.get("OPSD_ENGINE_URL") or (
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    )
    # slime's `post` runs with httpx.Timeout(None); an unanswered /generate would hang
    # the rollout forever. Bound every attempt and retry (same guard as math_rollout).
    timeout_s = float(os.environ.get("OPSD_ROLLOUT_TIMEOUT_S", "900"))
    attempts = max(1, int(os.environ.get("OPSD_ROLLOUT_ATTEMPTS", "3")))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            output = await asyncio.wait_for(
                post(f"{engine_url}/generate", payload), timeout=timeout_s
            )
            break
        except (asyncio.TimeoutError, TimeoutError) as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning(
                "[hsi-math] /generate exceeded %.0fs (attempt %d/%d) engine=%s prompt_len=%d -- retrying",
                timeout_s, attempt + 1, attempts, engine_url, len(student_prompt_ids),
            )
    else:
        raise RuntimeError(
            f"/generate did not return after {attempts} attempts x {timeout_s}s ({engine_url}); "
            "refusing to hang the rollout silently"
        ) from last_exc

    meta = output.get("meta_info", {}) or {}
    entries = meta.get("output_token_logprobs") or []
    top = meta.get("output_top_logprobs") or []
    candidate_ids: list[list[int]] = []
    candidate_log_probs: list[list[float]] = []
    for position in range(len(entries)):
        pairs = top[position] if position < len(top) else None
        if not pairs:
            candidate_ids.append([])
            candidate_log_probs.append([])
            continue
        candidate_ids.append([int(item[1]) for item in pairs])
        candidate_log_probs.append([float(item[0]) for item in pairs])
    return {
        "token_ids": [int(item[1]) for item in entries],
        "log_probs": [float(item[0]) for item in entries],
        "candidate_ids": candidate_ids,
        "candidate_log_probs": candidate_log_probs,
        "text": output.get("text", ""),
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
    }


# ---------------------------------------------------------------------------
# Group -> chunk rows
# ---------------------------------------------------------------------------


def _candidate_table(candidate_ids, candidate_log_probs, token_ids, start, end):
    """Per-position ``(ids, probs)`` for the target span ``[start, end)``.

    ``probs`` is ``sampling_probabilities`` of the engine's top-K at that position --
    ``rho_t`` = pi_old renormalised inside ``C_t``. A position the engine returned no
    candidates for yields ``([], [])`` and is MASKED by the scorer (section 8).
    """
    from tasks.alfworld.hsi_shared import sampling_probabilities

    table: list[tuple[list[int], list[float]]] = []
    for position in range(start, end):
        ids = candidate_ids[position] if position < len(candidate_ids) else []
        logprobs = candidate_log_probs[position] if position < len(candidate_log_probs) else []
        if not ids or len(ids) != len(logprobs):
            table.append(([], []))
            continue
        probs = sampling_probabilities(logprobs)
        if not probs or len(probs) != len(ids):
            table.append(([], []))
            continue
        table.append((list(ids), list(probs)))
    return table


def _chunk_rows(
    tokenizer,
    problem: str,
    completion_ids: list[int],
    log_probs: list[float],
    candidate_ids,
    candidate_log_probs,
    student_prompt_ids: list[int],
    outcome_block: str,
    reward: float,
    status,
    *,
    group_index: int,
    index_start: int,
    stride: int,
) -> list[Sample]:
    """One ``Sample`` per chunk of the completion (the math analogue of per-turn rows)."""
    rlen = len(completion_ids)
    rows: list[Sample] = []
    for chunk_index, (start, end) in enumerate(split_chunks(rlen, _chunk_count())):
        if end <= start:
            continue
        assert_no_leak((start, end), rlen)
        sample = Sample(index=index_start + len(rows), prompt="")
        sample.group_index = group_index
        sample.tokens = list(student_prompt_ids) + list(completion_ids[:end])
        sample.response_length = end - start
        sample.loss_mask = [1] * sample.response_length
        sample.rollout_log_probs = [float(v) for v in log_probs[start:end]]
        if len(sample.rollout_log_probs) != sample.response_length:
            raise ValueError(
                f"hsi-math chunk {chunk_index}: {len(sample.rollout_log_probs)} log-probs "
                f"for {sample.response_length} tokens"
            )
        sample.reward = float(reward)
        sample.status = status
        # Reward-free by contract: the scorer overwrites this with the detached
        # hindsight advantage; the placeholder only has to be the right shape.
        sample.opsd_advantages = [0.0] * sample.response_length
        sample.fit_turn_weights = [1.0] * sample.response_length
        # turn_ids is the one per-token integer channel slime whitelists for the DP
        # split and forwards into the batch. All chunks of one prompt share an episode
        # id, so the trajectory-balanced reduction treats the prompt as one trajectory.
        sample.turn_ids = [group_index * stride + chunk_index] * sample.response_length
        view_ids, prompt_len, target_len = build_hsi_view(
            tokenizer, problem, completion_ids, (start, end), outcome_block
        )
        sample.metadata["hsi_teacher_view"] = (view_ids, prompt_len, target_len)
        sample.metadata["hsi_estimator"] = _estimator()
        if _wants_candidates():
            sample.metadata["hsi_candidates"] = _candidate_table(
                candidate_ids, candidate_log_probs, completion_ids, start, end
            )
        sample.metadata["outcome_reward"] = int(reward)
        rows.append(sample)
    return rows


@observe_group("math")
async def _generate_group(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict,
    group_index: int,
    index_start: int,
) -> list[Sample]:
    global _FIRST_SAMPLE_PRINTED

    sample = group[0]
    problem = sample.prompt
    solution = sample.label
    if solution is None:
        raise ValueError("Sample.label is empty: pass --label-key solution for the verifier")

    student_prompt_ids = build_student_prompt_ids(state.tokenizer, problem)

    if state.aborted:
        sample.status = Sample.Status.ABORTED
        return [sample]

    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return [sample]
        out = await _generate_student(args, sample, student_prompt_ids, sampling_params)

    sample.tokens = list(student_prompt_ids) + list(out["token_ids"])
    sample.response_length = len(out["token_ids"])
    sample.response = out["text"]
    sample.rollout_log_probs = list(out["log_probs"])

    # The verifier (official grader, reused verbatim) runs on the FULL completion and
    # yields the binary outcome. Section 3: a bool renders the binary verdict text
    # through the shared renderer; the block never sees a raw reward number.
    from tasks.math.reward import outcome_reward

    R, audit = outcome_reward(sample.response, solution)
    sample.reward = float(R)
    sample.metadata["outcome_audit"] = audit
    if _wants_candidates():
        from tasks.alfworld.hsi_shared import render_outcome_descriptor
    else:
        from methods.hsi2_jsd import render_outcome_descriptor
    outcome_block = render_outcome_descriptor(bool(R))

    if sample.response_length == 0:
        sample.status = Sample.Status.TRUNCATED
        logger.info("[hsi-math] empty completion (problem=%r) -- no rows", problem[:60])
        return []

    if out.get("finish_reason") == "length":
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED

    from tasks.alfworld.config import TURN_ID_STRIDE

    rows = _chunk_rows(
        state.tokenizer,
        problem,
        out["token_ids"],
        out["log_probs"],
        out["candidate_ids"],
        out["candidate_log_probs"],
        student_prompt_ids,
        outcome_block,
        R,
        sample.status,
        group_index=group_index,
        index_start=index_start,
        stride=TURN_ID_STRIDE,
    )

    if not _FIRST_SAMPLE_PRINTED and rows:
        _FIRST_SAMPLE_PRINTED = True
        first = rows[0]
        logger.info(
            "[hsi-math] estimator=%s problem_len=%d response_len=%d chunks=%d R=%d | chunk0 "
            "prompt_len=%d target_len=%d block_head=%r",
            _estimator(),
            len(student_prompt_ids),
            sample.response_length,
            len(rows),
            int(R),
            first.metadata["hsi_teacher_view"][1],
            first.metadata["hsi_teacher_view"][2],
            render_hindsight_block(problem, sample.response[-200:], outcome_block)[:160],
        )
    return rows


def generate_rollout(args: Namespace, rollout_id: int, data_source, evaluation: bool = False):
    """Custom rollout entry point (see ``slime.ray.rollout.call_rollout_fn``)."""
    from slime.utils.async_utils import run

    assert not evaluation, "math HSI rollout does not implement eval generation"
    return run(_generate_rollout_async(args, rollout_id, data_source))


async def _generate_rollout_async(args: Namespace, rollout_id: int, data_source):
    from tasks.math.math_rollout import _abort

    state = GenerateState(args)
    sampling_params = dict(state.sampling_params)
    # Full-support sampler: the candidate set must be the head of the true softmax.
    # Assert rather than silently patch, so a misconfigured launch fails loudly.
    top_p = float(getattr(args, "rollout_top_p", 1.0))
    top_k = int(getattr(args, "rollout_top_k", -1))
    if top_p != 1.0 or top_k != -1:
        raise RuntimeError(
            "math HSI requires a full-softmax rollout sampler: got "
            f"top_p={top_p}, top_k={top_k}; expected 1.0 / -1"
        )
    sampling_params["top_p"] = 1.0
    sampling_params["top_k"] = -1
    sampling_params["min_p"] = 0.0
    sampling_params["presence_penalty"] = 0.0
    sampling_params["frequency_penalty"] = 0.0
    sampling_params["repetition_penalty"] = 1.0

    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    ready: list[list[Sample]] = []
    pendings: dict[asyncio.Task, int] = {}
    index_start = 0
    next_group = 0

    while len(data) < target_data_size:
        while len(ready) + len(pendings) + len(data) < target_data_size:
            if hasattr(data_source, "get_samples"):
                samples = data_source.get_samples(args.over_sampling_batch_size)
            else:
                samples = data_source(args.over_sampling_batch_size)
            for group in samples:
                gid = next_group
                next_group += 1
                task = asyncio.create_task(
                    _generate_group(args, state, group, sampling_params, gid, index_start)
                )
                # Reserve index space generously; rows are re-indexed below.
                index_start += max(1, _chunk_count())
                pendings[task] = gid
        done, _ = await asyncio.wait(set(pendings), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pendings.pop(task, None)
            ready.append(task.result())
        while ready and len(data) < target_data_size:
            data.append(ready.pop(0))

    assert len(data) == target_data_size
    await _abort(args, state)
    state.reset()

    # An aborted / empty completion yields a group with no rows; keep those groups in
    # their submitted position rather than indexing into them.
    data = sorted(data, key=lambda group: group[0].group_index if group else -1)
    # Re-index globally so every row carries a unique index regardless of chunk count.
    flat = [s for group in data for s in group]
    for position, s in enumerate(flat):
        s.index = position

    total_completion_tokens = sum(s.response_length for s in flat)
    metrics = {
        "rollout/avg_completion_tokens": total_completion_tokens / max(len(data), 1),
        "rollout/n_rows": float(len(flat)),
        "rollout/n_truncated": sum(
            1 for s in flat if s.status == Sample.Status.TRUNCATED
        ),
    }
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
