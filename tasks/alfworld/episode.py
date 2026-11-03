"""Run one ALFWorld episode and emit ONE TRAINING RECORD PER TURN.

Why per-turn rows (the v2 structural change)
--------------------------------------------
v1 packed a whole episode into a single Megatron row
(``prompt_0 | y_0 | prompt_1 | y_1 | ...``). Two consequences:

* the row length grew with the turn budget, and the LoRA lm_head -- frozen, but
  still materialising a full-vocab fp32 tensor over the whole row -- is what set
  the horizon. That is why v1 could only afford 5 turns, which in turn is why base
  success was near zero and GRPO had almost no signal;
* the O(T^2) segment layout meant most of the row was prompt re-rendering rather
  than decisions.

Emitting one row per turn decouples horizon from train row length: every row is a
bounded prompt plus at most ``MAX_RESPONSE_TOKENS`` generated tokens, regardless of
how long the episode runs. Horizon can go to 20 (``config.MAX_TURNS``) without
touching the memory ceiling.

History contains every previous (observation, action) pair and past
``<think>`` is never rendered -- see ``prompts.py``.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

from tasks.alfworld.v1.env_adapter import DEFAULT_MAX_STEPS, GameRef
from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord

from . import config
from .action_harness import HARNESS_VERSION, parse_action
from .env_actors import acquire_env
from .prompts import TaskView, build_turn_prompt_ids

__all__ = ["TurnSample", "run_episode"]

from opsd_core.telemetry import observe_episode

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 900


@dataclass
class TurnSample:
    """One training row: a single decision turn of an episode."""

    episode_id: int
    group_id: int
    task_id: str
    turn_idx: int

    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    #: The TaskView this turn was generated from, kept so the privileged views can
    #: be rebuilt byte-identically after the episode ends.
    view: TaskView
    #: Exact behaviour-policy log-prob of every generated token, straight from the
    #: engine (no replay, no re-forward). This is the PPO "old" log-prob.
    generated_log_probs: list[float] = field(default_factory=list)
    #: Sampling candidates per generated position: the top-k ids the engine returned
    #: and their log-probabilities under the sampler's own (temperature-scaled)
    #: distribution. Aligned 1:1 with ``generated_token_ids``.
    candidate_ids: list[list[int]] = field(default_factory=list)
    candidate_log_probs: list[list[float]] = field(default_factory=list)

    trajectory_success: bool = False
    sampler_mu: list[float] = field(default_factory=list)
    sampler_f: list[float] = field(default_factory=list)
    sampler_e: list[float] = field(default_factory=list)
    harness_source: str = "structured"


@dataclass
class EpisodeResult:
    trajectory: Trajectory
    turns: list[TurnSample] = field(default_factory=list)
    generated_tokens: int = 0
    generation_seconds: float = 0.0
    environment_seconds: float = 0.0
    wall_seconds: float = 0.0
    feature_seconds: float = 0.0

    @property
    def reward(self) -> float:
        return 1.0 if self.trajectory.success else 0.0


async def _generate(args, prompt_ids: list[int], sampling_params: dict, engine_url: str | None = None) -> dict:
    """One token-in / token-out engine call, bounded, generation stops at </action>."""
    from slime.utils.http_utils import post

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        # The candidate set is read off the SAMPLER's own distribution, not a
        # re-forward: sglang reports the top-k log-probs it actually normalised over
        # (temperature-scaled), so C_t is exactly what the sampler could have drawn.
        # Vanilla GRPO/GSPO need the sampled token's log-prob, not a top-k table.
        # Keep candidates available for the HPI scorers that actually use them.
        "top_logprobs_num": config.get_int(
            "ALFWORLD_V2_HSI_CANDIDATE_K",
            20 if os.environ.get("ALFWORLD_V2_HPI", "0") == "1" else 0,
        ),
    }
    engine_url = engine_url or os.environ.get("OPSD_ENGINE_URL") or (
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    )
    timeout_s = float(os.environ.get("OPSD_ROLLOUT_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S)))
    attempts = max(1, int(os.environ.get("OPSD_ROLLOUT_ATTEMPTS", "3")))

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            output = await asyncio.wait_for(post(f"{engine_url}/generate", payload), timeout=timeout_s)
            break
        except (asyncio.TimeoutError, TimeoutError) as exc:  # noqa: PERF203
            last_exc = exc
            logger.warning("[v2] /generate exceeded %.0fs (attempt %d/%d)", timeout_s, attempt + 1, attempts)
    else:
        raise RuntimeError(f"/generate did not return after {attempts} attempts x {timeout_s}s") from last_exc

    meta = output.get("meta_info", {}) or {}
    entries = meta.get("output_token_logprobs") or []
    if os.environ.get("OPSD_SRLVR", "0") == "1":
        keys = ("srlvr_logp",) if os.environ.get("SRLVR_VARIANT") in ("v2", "v3") else (
            "srlvr_mu", "srlvr_f", "srlvr_e", "srlvr_logp")
        for key in keys:
            if len(meta.get(key, [])) != len(entries):
                raise RuntimeError(f"SRLVR sampler metadata {key} missing/misaligned: {len(meta.get(key, []))} vs {len(entries)}")
        # Actual FP32 multinomial law, supplied by the same sampling operation.
        entries = [[lp, entry[1]] for lp, entry in zip(meta["srlvr_logp"], entries, strict=True)]
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
        "sampler_mu": meta.get("srlvr_mu", []),
        "sampler_f": meta.get("srlvr_f", []),
        "sampler_e": meta.get("srlvr_e", []),
        "feature_seconds": sum(meta.get("srlvr_feature_seconds", [])),
    }


def _strip_trailing_specials(token_ids: list[int], log_probs: list[float], tokenizer, *extra) -> None:
    """Drop trailing special tokens so the stored ids are pure assistant content.

    ``extra`` are further per-position lists (candidate ids / log-probs) that must stay
    aligned with ``token_ids`` -- trimming one and not the others would silently
    mis-align every candidate after the first dropped token.
    """
    stop_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop_ids.add(eos)
    kept = len(token_ids)
    while kept and token_ids[kept - 1] in stop_ids:
        kept -= 1
    del token_ids[kept:]
    del log_probs[kept:]
    for sequence in extra:
        del sequence[kept:]


def _history_pairs(trajectory: Trajectory, upto_turn: int) -> list[tuple[str, str]]:
    """All previous (observation, action) pairs before turn ``upto_turn``."""
    pairs: list[tuple[str, str]] = []
    for index in range(upto_turn):
        turn = trajectory.turns[index]
        pairs.append((turn.observation, turn.action_text))
    return pairs


@observe_episode
async def run_episode(
    args,
    tokenizer,
    game: GameRef,
    sampling_params: dict,
    *,
    episode_id: int,
    group_id: int,
    engine_url: str | None = None,
) -> EpisodeResult:
    """Run one episode; environment work is hosted by a persistent env actor.

    TextWorld is parser-serialised inside an interpreter, so each episode leases
    its own actor process (``tasks/alfworld/env_actors.py``) instead of sharing
    one global lock. ``ALFWORLD_ENV_ACTORS=0`` restores the in-process path.
    """
    max_turns = config.get_int("ALFWORLD_V2_MAX_TURNS", config.MAX_TURNS)

    episode_start = time.perf_counter()
    env_start = episode_start
    env = await acquire_env(game, max(max_turns, DEFAULT_MAX_STEPS))
    try:
        result = await env.reset()
        task_description = env.task_description

        trajectory = Trajectory(
            task_id=game.game_id,
            task_type=game.task_type,
            split=game.split,
            task_description=task_description,
            seed=episode_id,
            success=False,
            terminal_reward=0.0,
            num_turns=0,
            truncated=False,
            invalid_action_count=0,
        )
        episode = EpisodeResult(trajectory=trajectory)
        episode.environment_seconds = time.perf_counter() - env_start
        current = result
        rejection: str | None = None

        for turn_idx in range(max_turns):
            observation_text = rejection if rejection is not None else current.observation
            view = TaskView(
                task_description=task_description,
                current_observation=observation_text,
                admissible_actions=list(current.admissible_actions),
                history=_history_pairs(trajectory, turn_idx),
            )
            rejection = None
            prompt_ids = build_turn_prompt_ids(tokenizer, view)
            limit = int(getattr(args, "seq_length", 32768))
            if len(prompt_ids) + sampling_params["max_new_tokens"] > limit:
                raise ValueError(
                    f"Complete history needs {len(prompt_ids)} prompt tokens plus "
                    f"{sampling_params['max_new_tokens']} response tokens; SEQ_LENGTH={limit}. "
                    "Increase SEQ_LENGTH; history is never silently truncated."
                )

            generation_start = time.perf_counter()
            output = await _generate(args, prompt_ids, sampling_params, engine_url)
            episode.generation_seconds += time.perf_counter() - generation_start
            episode.feature_seconds += output.get("feature_seconds", 0.)
            y_ids = list(output["token_ids"])
            y_log_probs = list(output["log_probs"])
            y_candidate_ids = [list(row) for row in output.get("candidate_ids") or []]
            y_candidate_log_probs = [list(row) for row in output.get("candidate_log_probs") or []]
            if os.environ.get("OPSD_SRLVR", "0") != "1":
                _strip_trailing_specials(
                    y_ids, y_log_probs, tokenizer, y_candidate_ids, y_candidate_log_probs
                )
            y_text = output["text"]
            episode.generated_tokens += len(y_ids)

            truncated_turn = output["finish_reason"] == "length"
            harness = parse_action(y_text, current.admissible_actions, truncated=truncated_turn)
            parsed = harness.parsed

            episode.turns.append(
                TurnSample(
                    episode_id=episode_id,
                    group_id=group_id,
                    task_id=game.game_id,
                    turn_idx=turn_idx,
                    prompt_token_ids=list(prompt_ids),
                    generated_token_ids=list(y_ids),
                    generated_text=y_text,
                    view=view,
                    generated_log_probs=list(y_log_probs),
                    candidate_ids=y_candidate_ids,
                    candidate_log_probs=y_candidate_log_probs,
                    sampler_mu=output.get("sampler_mu", []),
                    sampler_f=output.get("sampler_f", []),
                    sampler_e=output.get("sampler_e", []),
                    harness_source=harness.source,
                )
            )

            if parsed.is_valid:
                env_start = time.perf_counter()
                step = await env.step(parsed.action)
                episode.environment_seconds += time.perf_counter() - env_start
                next_observation = step.observation
                done, won = bool(step.done), bool(step.won)
            else:
                trajectory.invalid_action_count += 1
                step = None
                attempted = parsed.raw_action or "(no action)"
                next_observation = (
                    f"Your previous action {attempted!r} was rejected: it is not one of the "
                    "valid commands. You are still in the same situation. Choose a command "
                    "using the exact object names in observations.\n\n"
                    f"{current.observation}"
                )
                done, won = False, False
                rejection = next_observation if True else None

            trajectory.turns.append(
                TurnRecord(
                    turn_idx=turn_idx,
                    observation=observation_text,
                    admissible_actions=list(current.admissible_actions),
                    student_prompt="",
                    prompt_token_ids=list(prompt_ids),
                    generated_token_ids=list(y_ids),
                    generated_text=y_text,
                    think_text=parsed.think_text,
                    action_text=parsed.action or "",
                    action_valid=parsed.is_valid,
                    parse_status=parsed.status.value,
                    next_observation=next_observation,
                    env_reward=1.0 if (done and won) else 0.0,
                    done=done,
                    # The state the policy stood in, not the rejection wrapper.
                    # An invalid action does not advance the environment, so all
                    # rejected attempts from one state share this anchor.
                    anchor_observation=current.observation,
                )
            )
            trajectory.num_turns = turn_idx + 1

            if done:
                trajectory.success = won
                break
            if turn_idx + 1 >= max_turns:
                trajectory.truncated = True
                break

            if step is not None:
                current = step

        trajectory.terminal_reward = 1.0 if trajectory.success else 0.0
        for turn_sample in episode.turns:
            turn_sample.trajectory_success = trajectory.success
    finally:
        await env.aclose()

    episode.wall_seconds = time.perf_counter() - episode_start
    logger.info(
        "[alfworld-episode] id=%d task=%s success=%d turns=%d tokens=%d "
        "generation_seconds=%.2f environment_seconds=%.2f wall_seconds=%.2f "
        "harness=%s valid_actions=%d bare_commands=%d",
        episode_id, game.game_id, int(trajectory.success), trajectory.num_turns,
        episode.generated_tokens, episode.generation_seconds,
        episode.environment_seconds, episode.wall_seconds,
        HARNESS_VERSION, sum(t.action_valid for t in trajectory.turns),
        sum(t.harness_source == "bare_command" for t in episode.turns),
    )
    return episode
