"""v2 rollout: one training row per turn, episode-level GRPO advantage.

Data flow (the order in section 17 of the spec)
-----------------------------------------------
    rollout -> environment rewards -> GRPO group advantages -> identify mixed
    groups -> build Full/Control views only where they can matter -> (teacher
    scoring happens later, in the training process, OUTSIDE the loss) -> convert

What each emitted ``Sample`` carries
------------------------------------
* ``tokens`` / ``response_length`` / ``loss_mask`` -- one turn, mask all ones
  (the prompt is the sample's prompt region).
* ``opsd_advantages`` -- the EPISODE's GRPO advantage, constant across the turn's
  tokens. Consumed via ``--use-opsd``, which slime passes through verbatim and
  which bypasses its positional group normalisation. That is what keeps the
  semantics "GRPO over episodes" even though the rows are turns.
* ``turn_ids`` -- ``episode_id * TURN_ID_STRIDE + turn_idx`` per token. ``turn_ids``
  is the only per-token integer channel slime already forwards to the training
  batch, so it is how the loss and the scorer recover the episode structure.
* ``fit_turn_weights`` -- the per-token HPI weight, filled by the scorer. Left at
  1.0 here (vanilla GRPO). The field is whitelisted and reaches the batch but is
  unused by slime unless FIT credit is enabled.
* ``hpi_views`` / ``hpi_prompt_lengths`` / ``hpi_target_lengths`` /
  ``hpi_user_lengths`` -- the Full/Control pair for this turn, present ONLY when
  the group is mixed and the turn has a future to condition on.
"""

import asyncio
import logging
import os
import random
from argparse import Namespace

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from tasks.alfworld.v1.env_adapter import GameRef

from . import config
from .action_harness import HARNESS_VERSION
from .advantages import (
    GigpoGroupAdvantages,
    compute_gigpo_advantages,
    compute_group_advantages,
)
from .episode import run_episode
from .privilege import has_future, render_hpi_block, render_trajectory_block

__all__ = ["generate_rollout", "build_group_samples"]

logger = logging.getLogger(__name__)

#: Turns whose privileged block tripped the leak guard and were therefore left
#: unscored (``ĉ = 0``). Surfaced as a rollout metric so a silently-degrading
#: guard cannot go unnoticed.
_LEAK_GUARD_SKIPS = 0

#: Upper bound on the same-task top-up rounds (see ``_generate_rollout_async``).
#: 8 rounds x ``rollout_batch_size`` extra episodes is well past the point where
#: a still-short rollout means something is genuinely wrong, and it keeps a
#: pathological policy from looping forever inside one rollout.
_MAX_TOPUP_ROUNDS = 8
#: Seed offset between top-up rounds, so round N re-samples its episodes instead
#: of replaying round 0 verbatim.
_TOPUP_SEED_STRIDE = 1_000_003


def _gigpo_metrics(groups) -> dict[str, float]:
    """Aggregate per-group GiGPO diagnostics over this rollout.

    Ratios are re-derived from the additive counts (turns, singleton turns,
    anchor groups), never averaged from per-group ratios: four tasks with very
    different turn counts would otherwise weight a short group as much as a
    long one.
    """
    objects = [adv for _episodes, adv in groups if isinstance(adv, GigpoGroupAdvantages)]
    if not objects:
        return {}
    anchor_groups = sum(obj.diagnostics["anchor_groups"] for obj in objects)
    turns = sum(obj.diagnostics["turns"] for obj in objects)
    singleton_turns = sum(obj.diagnostics["singleton_turns"] for obj in objects)
    nonzero_turns = sum(obj.diagnostics["nonzero_step_fraction"] * obj.diagnostics["turns"] for obj in objects)
    step_abs = sum(obj.diagnostics["step_adv_mean_abs"] * obj.diagnostics["turns"] for obj in objects)
    episode_count = sum(obj.num_episodes for obj in objects)
    episode_abs = sum(
        obj.diagnostics["episode_adv_mean_abs"] * obj.num_episodes for obj in objects
    )
    return {
        "rollout/gigpo_task_groups": float(len(objects)),
        "rollout/gigpo_anchor_groups": anchor_groups,
        "rollout/gigpo_turns": turns,
        "rollout/gigpo_singleton_fraction": singleton_turns / max(turns, 1.0),
        "rollout/gigpo_mean_group_size": turns / max(anchor_groups, 1.0),
        "rollout/gigpo_max_group_size": max(
            obj.diagnostics["max_group_size"] for obj in objects
        ),
        "rollout/gigpo_step_nonzero_fraction": nonzero_turns / max(turns, 1.0),
        "rollout/gigpo_step_adv_mean_abs": step_abs / max(turns, 1.0),
        "rollout/gigpo_episode_adv_mean_abs": episode_abs / max(episode_count, 1),
        "rollout/gigpo_step_w": objects[0].step_w,
        "rollout/gigpo_gamma": objects[0].gamma,
        "rollout/gigpo_mode_mean_std_norm": float(objects[0].mode == "mean_std_norm"),
    }


def _turn_rows(groups) -> int:
    """How many turn-rows (i.e. training rows) the finished groups would produce.

    One turn becomes exactly one ``Sample`` row in ``build_group_samples``, so this
    is the same number slime's trim sees after it flattens the returned batches.
    ``tests/test_rollout_turn_floor.py`` pins that 1:1 mapping; if it ever breaks,
    ``_generate_rollout_async``'s second floor check re-runs the gate against the
    rows that were actually emitted.
    """
    return sum(len(episode.turns) for episodes, _adv in groups for episode in episodes)


async def _top_up_round(args, state, games, seeds, engines, groups, group_games, group_ids, topup_round) -> None:
    """Append ONE round of extra same-task episodes (mutates the group lists).

    Fresh seeds and fresh group ids, but the same ``games`` -- so no task changes
    and nothing is selected on reward. Shared by both floor checks below.
    """
    base = len(games) * topup_round
    extra = await asyncio.gather(
        *[
            _run_group(
                args,
                state,
                games[i],
                seeds[i] + topup_round * _TOPUP_SEED_STRIDE,
                base + i,
                engines,
            )
            for i in range(len(games))
        ]
    )
    groups.extend(extra)
    group_games.extend(games)
    group_ids.extend(range(base, base + len(games)))


async def _top_up_to_floor(
    args, state, games, seeds, engines, groups, group_games, group_ids, min_rows
) -> int:
    """Append same-task episodes to ``groups`` until it clears ``min_rows`` turn-rows.

    A rollout must hand the trainer at least one optimizer step's worth of rows,
    because slime's trim (``slime/ray/rollout.py::_get_rollout_data``) raises
    "Not enough samples" when it cannot form one -- and that kills the whole run.
    For HSI (G=1) one rollout is only ``rollout_batch_size`` episodes, so a step
    whose episodes all terminate early can fall under the floor: measured
    2026-10-04 15:01, rollout 10 produced 28 rows against a floor of 32 and took
    the run down with it (with ``save_interval=25`` the restart-from-cp0 that
    follows loses every rollout since the last checkpoint).

    Top up with ADDITIONAL episodes of the SAME scheduled tasks -- fresh seeds,
    fresh group ids -- so no task changes, no row is dropped and nothing is
    selected on reward. The GRPO arm has G=8 and clears the floor by a wide
    margin, so in practice this only ever fires for HSI.

    ``groups`` / ``group_games`` / ``group_ids`` are extended in place; returns
    the number of top-up rounds performed.
    """
    topup_round = 0
    while min_rows > 0 and _turn_rows(groups) < min_rows and topup_round < _MAX_TOPUP_ROUNDS:
        topup_round += 1
        await _top_up_round(
            args, state, games, seeds, engines, groups, group_games, group_ids, topup_round
        )
    return topup_round


def _hsi_mode() -> bool:
    """True for the Ours arm: single rollout + outcome-anchored hindsight credit."""
    return os.environ.get("ALFWORLD_V2_HSI", "0") == "1"


def _gigpo_mode() -> bool:
    """True for GiGPO: GRPO episode advantage + anchor-state step advantage."""
    return config.gigpo_enabled()


def validate_gigpo_stack() -> None:
    """Fail BEFORE the rollout if GiGPO is combined with another estimator.

    The check used to sit after ``asyncio.gather`` of all G episodes, so a
    leaked ``ALFWORLD_V2_HSI`` cost one whole rollout before dying -- and
    ``ALFWORLD_V2_ADVANTAGE_MODE=batch`` was only caught at convert time. Both
    are launch-configuration errors, so they are checked at rollout entry.
    """
    if not _gigpo_mode():
        return
    if os.environ.get("OPSD_SRLVR", "0") == "1" or _hsi_mode():
        raise RuntimeError(
            "GiGPO (ALFWORLD_V2_GIGPO=1) must run alone; unset OPSD_SRLVR and "
            "ALFWORLD_V2_HSI for this arm"
        )
    if os.environ.get("ALFWORLD_V2_ADVANTAGE_MODE", "group").lower() == "batch":
        raise RuntimeError(
            "GiGPO provides per-turn advantages; ALFWORLD_V2_ADVANTAGE_MODE=batch "
            "would overwrite them with one batch-level scalar"
        )


def _candidate_mode() -> bool:
    """True for Candidate-Centered HSI (one frozen teacher, sparse lm-head on C_t)."""
    return _hsi_mode() and os.environ.get("ALFWORLD_V2_HSI_CANDIDATE", "0") == "1"


def _hsi_v2_mode() -> bool:
    """True for the 2026-10-04 HSI protocol: SAME-policy hindsight, top-128, no anchor.

    The structural difference from `_candidate_mode` that shows up here is the
    future gate. The retired estimator required a tail TRANSITION and therefore
    never scored the last turn; the V2 protocol's future is defined as everything
    experienced after position ``t``, which always includes the terminal outcome
    context. So every turn gets a view, and a turn whose view cannot be built is
    MASKED (loss_mask 0) rather than silently left at an invented advantage.
    """
    return _hsi_mode() and os.environ.get("ALFWORLD_V2_HSI_V2", "0") == "1"


#: Candidate-set coverage counters. Reset at the start of each rollout so the
#: reported rates describe THIS step, not the run.
_CANDIDATE_COVERAGE = {"positions": 0, "sampled_in_set": 0, "full_k": 0}


class OutcomeAnchor:
    """Per-trajectory outcome anchor ``2R - 1`` (no group statistics).

    Stands in for ``GroupAdvantages`` so the rest of the rollout is unchanged; the
    scorer adds the trajectory-internal hindsight term on top of it.
    """

    def __init__(self, advantages: list[float], rewards: list[float]):
        self.advantages = advantages
        self.rewards = rewards
        self.mean_reward = sum(rewards) / max(len(rewards), 1)
        self.std_reward = 0.0
        # `mixed` is meaningless here and deliberately always True so no caller can
        # resurrect the old "skip unless the group is mixed" gate.
        self.mixed = True
        self.num_episodes = len(rewards)


def _sampling_params(state: GenerateState, *, seed: int, max_response: int) -> dict:
    """Per-turn sampler. Stops the moment the action closes."""
    params = dict(state.sampling_params)
    params["max_new_tokens"] = max_response
    params["sampling_seed"] = seed
    # `no_stop_trim` keeps "</action>" in the emitted text, so the parser sees a
    # closed block instead of a turn that looks truncated.
    params["stop"] = [config.STOP_STRING]
    params["no_stop_trim"] = True
    return params


def _engine_url(args: Namespace) -> str:
    return os.environ.get("OPSD_ENGINE_URL") or (
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    )


def _engine_urls(args: Namespace) -> list[str]:
    """Every rollout engine URL, so episodes can be spread over ALL of them.

    The training rollout path has always posted every episode to
    ``OPSD_ENGINE_URL`` -- which is ``engines[0]`` -- leaving the other
    ``--rollout-num-gpus`` engines idle while the first one serialises the whole
    batch. ``OPSD_ENGINE_URLS`` (comma-separated, exported by the rollout manager)
    is the full list; the router is the fallback when it is absent.
    """
    raw = os.environ.get("OPSD_ENGINE_URLS", "")
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    return urls or [_engine_url(args)]


async def _run_group(
    args, state, game: GameRef, seed_base: int, group_id: int, engines: list[str]
) -> tuple[list, object]:
    """Run all G episodes of one task; return (episodes, advantages)."""
    g = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    max_response = config.get_int("ALFWORLD_V2_MAX_RESPONSE", config.MAX_RESPONSE_TOKENS)

    async def one(index: int):
        params = _sampling_params(state, seed=seed_base + index, max_response=max_response)
        return await run_episode(
            args,
            state.tokenizer,
            game,
            params,
            episode_id=group_id * 1000 + index,
            group_id=group_id,
            engine_url=engines[(group_id * max(g, 1) + index) % len(engines)],
        )

    async with state.semaphore:
        episodes = await asyncio.gather(*[one(i) for i in range(g)])

    rewards = [episode.reward for episode in episodes]
    if _gigpo_mode():
        # Standalone objective: GiGPO's step term is defined over the SAME
        # group's trajectories, and OPSD/SRLVR already own this batch's
        # advantage channel. Refuse the combination rather than silently
        # letting one estimator overwrite the other.
        validate_gigpo_stack()
        return list(episodes), compute_gigpo_advantages(
            [episode.trajectory for episode in episodes],
            step_w=config.gigpo_step_weight(),
            gamma=config.gigpo_gamma(),
            mode=config.gigpo_mode(),
        )
    if os.environ.get("OPSD_SRLVR", "0") == "1":
        return list(episodes), OutcomeAnchor([0.] * len(rewards), rewards)
    if _hsi_mode():
        # HSI is SINGLE-ROLLOUT: the outcome anchor is 2R-1 per trajectory, so a
        # group of one still carries signal. Group-relative advantage is not used
        # and, crucially, its zero on a uniform group is not inherited -- that was
        # the conceptual error in the previous version (w * L_GRPO with L_GRPO == 0).
        return list(episodes), OutcomeAnchor([2.0 * r - 1.0 for r in rewards], rewards)
    return list(episodes), compute_group_advantages(rewards)


def _srlvr_v4_mode() -> bool:
    """True for SRLVR-v4: one trajectory-level privileged teacher view."""
    return (
        os.environ.get("OPSD_SRLVR", "0") == "1"
        and os.environ.get("SRLVR_VARIANT", "") == "v4"
    )


def _turn_action_tokens(tokenizer, turn):
    """Executed-action token ids and their ``[start, end)`` span.

    v4's hindsight continuation deliberately excludes the chain of thought: the
    privileged block already carries the complete executed trace and outcome,
    while the continuation only needs the action tokens the student is trained
    on.  ``SRLVR_V4_HINDSIGHT_NO_THINK=0`` restores the historical full
    generation stream for an A/B.

    A turn with no reliable ``<action>`` span (no action emitted, or an action
    the harness could not recover) returns an EMPTY span; it does NOT fall back
    to the full generation.  That fallback inverted this function's own
    contract: ``NO_THINK=1`` exists to keep thinking out of the target, yet the
    fallback trained the whole chain of thought exactly on the turns that had
    failed.  Consequences measured 2026-10-28 on the two-node full-RL v4 arm:
    19 of 32 rows became full-generation fallbacks with a 470-token target (the
    5 genuinely admissible rows had a 5-token target), the overlong-drop count
    grew 150 -> 400 rows in four batches, and action-valid collapsed
    0.30 -> 0.11 with success at 0.  It is a self-reinforcing loop: longer
    failure reasoning -> longer teacher view -> dropped rows -> the surviving
    gradient comes from the remaining long failure reasoning.
    """
    generated = list(turn.generated_token_ids)
    if os.environ.get("SRLVR_V4_HINDSIGHT_NO_THINK", "1").lower() in ("0", "false", "no"):
        return generated, (0, len(generated))
    if not generated:
        return generated, (0, 0)
    from .privilege import action_token_span

    span = action_token_span(tokenizer, turn.generated_text, generated)
    if span is None or span[1] <= span[0]:
        return generated, (0, 0)
    return generated, span


def _teacher_continuation(tokenizer, turns) -> tuple[list[int], list[tuple[int, int]]]:
    """Executed-action token stream and each turn's span inside it.

    The student's per-turn prompts are NOT token-level extensions of one
    another: ``prompts.build_turn_messages`` re-renders history from the parsed
    action/observation record and deliberately omits past ``<think>``.  So
    ``prompt_{t+1}`` does not start with ``prompt_t + y_t`` and concatenating
    them would drop or duplicate tokens.

    The teacher therefore reads a synthetic continuation made of each turn's
    executed ``<action>`` content, separated by a newline.  Observations,
    outcomes and the full interaction trace are carried by the privileged block
    that precedes this stream.  Thinking tokens are excluded by default (see
    ``SRLVR_V4_HINDSIGHT_NO_THINK``): they dominate the overlong teacher views
    and are not the action the hindsight is meant to teach.
    """
    separator = tokenizer.encode("\n", add_special_tokens=False)
    continuation: list[int] = []
    spans: list[tuple[int, int]] = []
    for index, turn in enumerate(turns):
        generated, (action_start, action_end) = _turn_action_tokens(tokenizer, turn)
        action = generated[action_start:action_end]
        if not action:
            # No recoverable action on this turn: it contributes no continuation
            # token and no distillation target. The row is kept (batch
            # composition unchanged) with an empty span, which the loss treats
            # as zero contribution -- see _turn_action_tokens for why copying the
            # full generation here instead collapses the policy.
            spans.append((0, 0))
            continue
        if continuation and separator:
            continuation.extend(separator)
        start = len(continuation)
        continuation.extend(action)
        spans.append((start, len(action)))
    return continuation, spans


def _v4_rows(tokenizer, episode, group_id: int, rollout_id: int) -> list[Sample]:
    """One row per turn, all sharing ONE trajectory-level teacher view.

    The teacher view is ``[privileged prompt | executed-action token stream]``.
    Each row records the span of its own action tokens inside that stream; the
    loss slices the teacher hidden states at ``prompt_len + span_start - 1``.
    The student row also carries the action tokens only: the thinking chain is
    generated during rollout but is neither copied into hindsight nor trained.
    """
    from .hsi_shared import render_outcome_descriptor

    turns = episode.turns
    if not turns:
        return []
    continuation, spans = _teacher_continuation(tokenizer, turns)
    compact = os.environ.get("SRLVR_V4_COMPACT_FUTURE", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    # Hindsight leakage guard: the privileged block shows the WHOLE episode, so
    # the teacher can in principle recite turn t+1 while the student is being
    # scored on turn t. Withholding a random few transitions keeps the trace
    # informative (outcome, most of the path) while removing the option of pure
    # recitation. Dropped indices leave gaps, so the trace is visibly partial.
    # Reproducible per (episode, rollout): the same episode always loses the same
    # turns, so a resumed run and an A/B differ only by the flag.
    transitions = list(episode.trajectory.iter_tail_transitions(0))
    drop_turns = int(os.environ.get("SRLVR_V4_HINDSIGHT_DROP_TURNS", "0") or 0)
    dropped_indices: list[int] = []
    if drop_turns > 0:
        if not compact:
            raise ValueError(
                "SRLVR_V4_HINDSIGHT_DROP_TURNS requires SRLVR_V4_COMPACT_FUTURE=1 "
                "(the non-compact renderer cannot drop transitions)"
            )
        if len(transitions) > 1:
            keep_n = max(1, len(transitions) - min(drop_turns, len(transitions) - 1))
            rng = random.Random(
                f"{episode.trajectory.task_id}|{turns[0].episode_id}|{rollout_id}"
            )
            keep = sorted(rng.sample(range(len(transitions)), keep_n))
            dropped_indices = [
                transitions[i][0] for i in range(len(transitions)) if i not in set(keep)
            ]
            transitions = [transitions[i] for i in keep]
    block = render_trajectory_block(
        episode.trajectory,
        outcome_text=render_outcome_descriptor(episode.trajectory.success),
        compact_future=compact,
        transitions=transitions if dropped_indices else None,
    )
    teacher_prompt = _prompt_ids_for(tokenizer, turns[0].view, block)
    teacher_ids = list(teacher_prompt) + continuation
    context_limit = int(os.environ.get("SEQ_LENGTH", "32768"))
    overlong = len(teacher_ids) > context_limit
    if overlong:
        mode = os.environ.get("SRLVR_V4_OVERLONG", "skip").lower()
        if mode not in ("skip", "error"):
            raise ValueError("SRLVR_V4_OVERLONG must be 'skip' or 'error'")
        if mode == "error":
            raise ValueError(
                f"SRLVR-v4 teacher view is {len(teacher_ids)} tokens "
                f"(prompt {len(teacher_prompt)} + continuation {len(continuation)}); "
                f"context limit is {context_limit}. No context was truncated."
            )
        # Stability fallback: keep the trajectory's row count (the batch
        # composition and checkpoint cadence stay intact) but zero its loss.
        # The student forward is still paid for, so this is never silent: the
        # warning names the episode and the exact overrun.
        logger.warning(
            "[srlvr-v4] overlong teacher view episode=%s tokens=%d (prompt=%d "
            "continuation=%d limit=%d); rows emitted with zero loss",
            turns[0].episode_id,
            len(teacher_ids),
            len(teacher_prompt),
            len(continuation),
            context_limit,
        )

    compression = None
    if compact:
        from .privilege import _render_future

        raw = _render_future(episode.trajectory, 0, 0)
        compact_text = block.split("Complete interaction trace:\n", 1)[1].rsplit(
            "\n</privileged_hindsight>", 1
        )[0]
        compression = dict(
            before_tokens=len(tokenizer.encode(raw, add_special_tokens=False)),
            after_tokens=len(tokenizer.encode(compact_text, add_special_tokens=False)),
            transitions=sum(1 for _ in episode.trajectory.iter_tail_transitions(0)),
            folded_transitions=raw.count("[Future transition ")
            - compact_text.count("[Future transition "),
            delta_events=compact_text.count(
                "Resulting environment observation: line delta from turn "
            ),
        )

    episode_id = turns[0].episode_id
    trajectory_payload = episode.trajectory.to_dict()
    rows: list[Sample] = []
    for index, turn in enumerate(turns):
        generated, (action_start, action_end) = _turn_action_tokens(tokenizer, turn)
        action = generated[action_start:action_end]
        if not action:
            # The turn produced no recoverable action, so it has no distillation
            # target and no teacher continuation token. Emitting a row here is
            # what used to happen with the full generation as the target; see
            # _turn_action_tokens. The row count already varies with the episode
            # length, so omitting the row keeps the batch well-formed.
            continue
        action_log_probs = list(turn.generated_log_probs)[action_start:action_end]
        span_start, span_len = spans[index]
        sample = Sample(index=len(rows), prompt="")
        sample.group_index = group_id
        sample.tokens = list(turn.prompt_token_ids) + action
        sample.response_length = len(action)
        sample.loss_mask = [1] * sample.response_length
        sample.rollout_log_probs = action_log_probs
        if len(sample.rollout_log_probs) != sample.response_length:
            raise ValueError(
                f"turn {index}: {len(sample.rollout_log_probs)} log-probs for "
                f"{sample.response_length} action tokens"
            )
        sample.reward = episode.reward
        sample.status = (
            Sample.Status.TRUNCATED if episode.trajectory.truncated else Sample.Status.COMPLETED
        )
        sample.opsd_advantages = [0.0] * sample.response_length
        sample.turn_ids = [
            episode_id * config.TURN_ID_STRIDE + turn.turn_idx
        ] * sample.response_length
        sample.fit_turn_weights = [1.0] * sample.response_length
        if overlong:
            # Zero the loss but keep the row: the batch still contains the same
            # trajectory count and the optimizer step is not rescaled. The loss
            # sees span_start=-1 and skips the teacher forward for this row.
            sample.loss_mask = [0] * sample.response_length
            sample.metadata["hsi_teacher_view"] = ([0], 1, 0)
            sample.metadata["srlvr_v4"] = dict(
                span_start=-1,
                span_len=0,
                trajectory_id=str(episode_id),
                turn=turn.turn_idx,
                policy_version=int(rollout_id),
                teacher_prompt_len=len(teacher_prompt),
                continuation_len=len(continuation),
                compact_future=bool(compact),
                terminal_reward=float(episode.reward),
                skipped_overlong=True,
                hindsight_dropped=list(dropped_indices),
            )
        # The trajectory payload contains every turn's token arrays. Attaching it
        # to every row makes serialization quadratic in the turn count (the exact
        # reason the per-turn v2 path keeps one copy per episode); v4 needs it only
        # for the rollout dump, so the first EMITTED row carries it (turn 0 may
        # have been skipped for having no action).
        if not rows:
            sample.metadata["trajectory"] = trajectory_payload
        sample.metadata["action_harness"] = dict(
            version=HARNESS_VERSION,
            source=getattr(turn, "harness_source", "structured"),
            action=episode.trajectory.turns[turn.turn_idx].action_text,
            valid=episode.trajectory.turns[turn.turn_idx].action_valid,
        )
        if not overlong:
            sample.metadata["hsi_teacher_view"] = (
                list(teacher_ids),
                len(teacher_prompt),
                len(continuation),
            )
            sample.metadata["srlvr_v4"] = dict(
                span_start=span_start,
                span_len=span_len,
                trajectory_id=str(episode_id),
                turn=turn.turn_idx,
                policy_version=int(rollout_id),
                teacher_prompt_len=len(teacher_prompt),
                continuation_len=len(continuation),
                compact_future=bool(compact),
                terminal_reward=float(episode.reward),
                hindsight_dropped=list(dropped_indices),
            )
        if compression is not None:
            sample.metadata["future_compression"] = dict(compression)
        rows.append(sample)
    return rows


def build_group_samples(
    tokenizer,
    game: GameRef,
    episodes: list,
    advantages,
    group_id: int,
    *,
    with_hpi: bool,
    future_gap: int,
    sibling_pool=None,
    rollout_id: int = 0,
) -> list[Sample]:
    """Turn a finished group into per-turn ``Sample`` rows."""
    global _LEAK_GUARD_SKIPS
    rows: list[Sample] = []
    gigpo = isinstance(advantages, GigpoGroupAdvantages)
    for episode_index, episode in enumerate(episodes):
        advantage = advantages.advantages[episode_index]
        per_turn = None
        if gigpo:
            per_turn = advantages.combined_advantages[episode_index]
            if len(per_turn) != len(episode.turns):
                # The step term is indexed by turn position; a length mismatch
                # would shift a later turn's credit onto an earlier decision.
                raise ValueError(
                    f"GiGPO episode {episode_index}: {len(per_turn)} per-turn "
                    f"advantages for {len(episode.turns)} turns"
                )
        episode_id = episode.turns[0].episode_id if episode.turns else group_id * 1000 + episode_index
        if _srlvr_v4_mode():
            # One teacher view per trajectory, shared by every turn row.
            rows.extend(_v4_rows(tokenizer, episode, group_id, rollout_id))
            continue
        # The finished trajectory is read-only. Copying all its token arrays for
        # EVERY turn makes serialization quadratic in the episode's turn count.
        trajectory_payload = episode.trajectory.to_dict()
        for turn_position, turn in enumerate(episode.turns):
            sample = Sample(index=len(rows), prompt="")
            sample.group_index = group_id
            sample.tokens = list(turn.prompt_token_ids) + list(turn.generated_token_ids)
            sample.response_length = len(turn.generated_token_ids)
            turn_advantage = per_turn[turn_position] if per_turn is not None else advantage
            sample.loss_mask = [1] * sample.response_length
            # The behaviour policy's OWN log-probabilities, read off the sampler.
            # Never a placeholder and never a replay: with --use-rollout-logprobs the
            # PPO ratio is exp(new - these), so a placeholder would make ratio == 1
            # and silently disable the trust region (see loss.resolve_old_log_probs).
            sample.rollout_log_probs = list(turn.generated_log_probs)
            if len(sample.rollout_log_probs) != sample.response_length:
                # The engine must report one log-prob per generated token; a
                # mis-alignment here would silently mis-pair old and new log-probs
                # in the ratio. Fail loudly rather than trim into a wrong pairing.
                raise ValueError(
                    f"turn {turn.turn_idx}: {len(sample.rollout_log_probs)} log-probs for "
                    f"{sample.response_length} generated tokens"
                )
            sample.reward = episode.reward
            sample.status = (
                Sample.Status.TRUNCATED if episode.trajectory.truncated else Sample.Status.COMPLETED
            )
            sample.opsd_advantages = [turn_advantage] * sample.response_length
            sample.turn_ids = [
                episode_id * config.TURN_ID_STRIDE + turn.turn_idx
            ] * sample.response_length
            # Filled by the scorer for the HPI arm; 1.0 keeps the GRPO arm exact.
            sample.fit_turn_weights = [1.0] * sample.response_length
            # The full trajectory rides along so a saved rollout dump is a frozen,
            # replayable record: the diagnostic rebuilds every privileged condition
            # from it without re-rolling out (and without re-sampling the policy).
            sample.metadata["trajectory"] = trajectory_payload
            sample.metadata["action_harness"] = dict(
                version=HARNESS_VERSION, source=getattr(turn, "harness_source", "structured"),
                action=episode.trajectory.turns[turn.turn_idx].action_text,
                valid=episode.trajectory.turns[turn.turn_idx].action_valid,
            )

            if (os.environ.get("OPSD_SRLVR", "0") == "1"
                    and os.environ.get("SRLVR_VARIANT") in ("v2", "v3")):
                # Built after the full rollout, using the existing Full HSI view.
                # The causal student tokens and mask above are unchanged.
                from .hsi_shared import render_outcome_descriptor
                from opsd_core.srlvr_settings import Settings
                outcome_only = Settings.from_env().teacher_outcome_only
                compact = os.environ.get("SRLVR_COMPACT_FUTURE", "0") == "1"
                block = render_hpi_block(
                    episode.trajectory, turn.turn_idx,
                    with_future=(not outcome_only and has_future(episode.trajectory, turn.turn_idx, future_gap)),
                    future_gap=future_gap,
                    outcome_text=render_outcome_descriptor(episode.trajectory.success),
                    compact_future=compact,
                )
                if compact and not outcome_only:
                    from .privilege import _render_future
                    future = block.split("Future interaction trace:\n", 1)[1].rsplit("\n</privileged_hindsight>", 1)[0]
                    raw = _render_future(episode.trajectory, turn.turn_idx, future_gap)
                    sample.metadata["future_compression"] = dict(
                        before_tokens=len(tokenizer.encode(raw, add_special_tokens=False)),
                        after_tokens=len(tokenizer.encode(future if raw else "", add_special_tokens=False)),
                        transitions=sum(1 for _ in episode.trajectory.iter_tail_transitions(turn.turn_idx + future_gap)),
                        folded_transitions=raw.count("[Future transition ") - future.count("[Future transition "),
                        delta_events=future.count("Resulting environment observation: line delta from turn "),
                    )
                attach_candidate_view(sample, tokenizer, turn, block)

            if with_hpi and _hsi_v2_mode():
                # HSI V2: one same-policy hindsight view per turn. The outcome
                # descriptor is rendered from the verifier's own verdict and is the
                # part of the future that is ALWAYS present, which is why the last
                # turn is scored here rather than skipped.
                from .hsi_shared import render_outcome_descriptor

                try:
                    block = render_hpi_block(
                        episode.trajectory,
                        turn.turn_idx,
                        with_future=has_future(episode.trajectory, turn.turn_idx, future_gap),
                        future_gap=future_gap,
                        outcome_text=render_outcome_descriptor(episode.trajectory.success),
                    )
                except AssertionError as exc:
                    _LEAK_GUARD_SKIPS += 1
                    logger.warning(
                        "[hsi-v2] leak guard dropped turn %d of episode %s: %s",
                        turn.turn_idx, episode_id, exc,
                    )
                    # Section 8: mask the token, do not invent an advantage for it.
                    # Zeroing the mask here (rather than in the scorer) keeps the
                    # convert-time trajectory weighting consistent with the loss.
                    sample.loss_mask = [0] * sample.response_length
                else:
                    attach_candidate_view(sample, tokenizer, turn, block)
            elif with_hpi and has_future(episode.trajectory, turn.turn_idx, future_gap):
                # Candidate-Centered HSI: exactly ONE frozen teacher view, the
                # TRUE-future one, and the sampler's own candidate table.
                #
                # `render_hpi_block` is fail-closed by design: it raises when the trace
                # body would leak reasoning. That guard is correct, but it must not take
                # down a multi-hour run over ONE bad turn. A turn that trips it gets no
                # view; the scorer then leaves its credit at 0, which is exactly the
                # `A_t = s` fallback the method defines for an unscorable turn. The
                # invariant (leaked reasoning never reaches the teacher) is unchanged;
                # only the blast radius is.
                try:
                    block = render_hpi_block(
                        episode.trajectory, turn.turn_idx, with_future=True, future_gap=future_gap
                    )
                except AssertionError as exc:
                    _LEAK_GUARD_SKIPS += 1
                    logger.warning(
                        "[hsi-cand] leak guard skipped turn %d of episode %s: %s",
                        turn.turn_idx, episode_id, exc,
                    )
                else:
                    attach_candidate_view(sample, tokenizer, turn, block)
            rows.append(sample)
    return rows


def attach_candidate_view(sample: Sample, tokenizer, turn, block: str) -> None:
    """Attach the single future-conditioned teacher view + this turn's candidates.

    ``C_t`` is the engine's own top-k list (no re-forward, no replay); ``p_t`` is the
    sampler's distribution restricted to it by the top-k/top-p pipeline, normalised to
    sum to 1. Positions the engine returned no candidates for are dropped from the
    credit by leaving their candidate list empty (the scorer then contributes 0 there).
    """
    from .hsi_shared import sampling_probabilities

    prompt = _prompt_ids_for(tokenizer, turn.view, block)
    target = list(turn.generated_token_ids)
    if (os.environ.get("OPSD_SRLVR") == "1"
            and os.environ.get("SRLVR_VARIANT") in ("v2", "v3")):
        context_limit = int(os.environ.get("SEQ_LENGTH", "32768"))
        if len(prompt) + len(target) > context_limit:
            raise ValueError(
                f"Full HSI teacher view needs {len(prompt)} prompt tokens plus "
                f"{len(target)} target tokens; context limit is {context_limit}. "
                "The causal prefix, observation, action constraints, and eligible future are preserved."
            )
    sample.metadata["hsi_teacher_view"] = (list(prompt) + target, len(prompt), len(target))

    table: list[tuple[list[int], list[float]]] = []
    for position, token_id in enumerate(target):
        ids = turn.candidate_ids[position] if position < len(turn.candidate_ids) else []
        logprobs = (
            turn.candidate_log_probs[position]
            if position < len(turn.candidate_log_probs)
            else []
        )
        if not ids or len(ids) != len(logprobs):
            table.append(([], []))
            continue
        probs = sampling_probabilities(logprobs)
        ids = ids[: len(probs)]
        if len(probs) != len(ids) or not probs:
            table.append(([], []))
            continue
        _CANDIDATE_COVERAGE["positions"] += 1
        _CANDIDATE_COVERAGE["sampled_in_set"] += int(int(token_id) in ids)
        # Completeness: the sampler applies top_k BEFORE top_p, so its support is a
        # subset of the top-k we were handed -- as long as the engine actually
        # returned a full k-list. A short list is the one way C_t could understate
        # the support, and it is reported rather than hidden.
        _CANDIDATE_COVERAGE["full_k"] += int(len(logprobs) >= config.HSI_CANDIDATE_K)
        table.append((ids, probs))
    sample.metadata["hsi_candidates"] = table


def _prompt_ids_for(tokenizer, view, block: str) -> list[int]:
    """Shared student template plus privilege, followed by the original target.

    Teacher forcing retains every earlier token within this turn, including
    already generated thinking; no history absent from the student is added.
    """
    from .prompts import build_turn_messages, render_chat

    messages = build_turn_messages(view)
    messages[-1] = dict(messages[-1], content=messages[-1]["content"] + "\n\n" + block)
    text = render_chat(tokenizer, messages, add_generation_prompt=True)
    return list(tokenizer.encode(text, add_special_tokens=False))

def generate_rollout(args: Namespace, rollout_id: int, data_source, evaluation: bool = False):
    from slime.utils.async_utils import run

    if evaluation:
        return run(_with_token_cache(_eval_async(args, rollout_id), rollout_id))
    return run(_with_token_cache(_generate_rollout_async(args, rollout_id), rollout_id))


async def _with_token_cache(coro, rollout_id):
    from opsd_core.token_cache import token_cache_scope

    # Scope inside the event-loop task; spawned episode tasks inherit it.
    with token_cache_scope(
        max_entries=int(os.environ.get("ALFWORLD_TOKEN_CACHE_ENTRIES", "128")),
        max_tokens=int(os.environ.get("ALFWORLD_TOKEN_CACHE_TOKENS", "262144")),
    ) as cache:
        try:
            return await coro
        finally:
            logger.info("[prompt-cache] rollout=%s hits=%d misses=%d saved_encode_tokens=%d",
                        rollout_id, cache.hits, cache.misses, cache.saved_tokens)


async def _generate_rollout_async(args: Namespace, rollout_id: int):
    from .schedule import task_batch

    validate_gigpo_stack()
    state = GenerateState(args)
    schedule_path = os.environ.get("ALFWORLD_V2_SCHEDULE", "")
    if not schedule_path:
        raise ValueError("ALFWORLD_V2_SCHEDULE must point at the frozen schedule jsonl")
    games, seeds = task_batch(schedule_path, rollout_id, int(args.rollout_batch_size), args.seed, args)
    tasks = len(games)

    # Ledger: the rollout phase is one of the three per-cycle times the task book
    # asks for, and nothing upstream prints it with wandb disabled.
    import time as _time

    _rollout_t0 = _time.perf_counter()

    mode = os.environ.get("ALFWORLD_V2_HPI", "0") == "1"
    future_gap = config.get_int("ALFWORLD_V2_FUTURE_GAP", config.HPI_FUTURE_GAP)
    # Per-rollout delta, not the cumulative counter, so the metric reads "how many
    # turns did the guard drop THIS step".
    leak_guard_before = _LEAK_GUARD_SKIPS
    for key in _CANDIDATE_COVERAGE:
        _CANDIDATE_COVERAGE[key] = 0

    engines = _engine_urls(args)
    group_games: list = list(games)
    group_ids: list[int] = list(range(len(games)))
    groups: list = list(
        await asyncio.gather(
            *[_run_group(args, state, game, seeds[i], i, engines) for i, game in enumerate(games)]
        )
    )

    # -- turn-row floor -------------------------------------------------------
    # A rollout must hand the trainer at least one optimizer step's worth of
    # rows, because slime's trim (slime/ray/rollout.py::_get_rollout_data) raises
    # "Not enough samples" when it cannot form one -- and that kills the whole
    # run. For HSI (G=1) one rollout is only ``rollout_batch_size`` episodes, so a
    # step whose episodes all terminate early can fall under the floor: measured
    # 2026-10-04 15:01, rollout 10 produced 28 rows against a floor of 32 and took
    # the run down with it (with save_interval=25, every rollout since the last
    # checkpoint is lost to the restart-from-cp0 that follows).
    #
    # Top up with ADDITIONAL episodes of the SAME scheduled tasks -- fresh seeds,
    # fresh group ids -- until the floor is met. Same tasks, no row dropped, no
    # selection on reward. The GRPO arm has G=8 and clears the floor by a wide
    # margin, so in practice this only ever fires for HSI.
    # `global_batch_size` is the trainer's own one-step floor (32 for both arms).
    # `ALFWORLD_V2_MIN_TURN_ROWS` is only an escape hatch: set it to raise the
    # floor, leave it unset and the trainer's value is used.
    min_rows = 0  # One real attempt per sample; pad rows rather than resampling.
    topup_round = await _top_up_to_floor(
        args, state, games, seeds, engines, groups, group_games, group_ids, min_rows
    )

    def _emit():
        """Build the trainer rows for every group currently in ``groups``."""
        built: list[list[Sample]] = []
        counts = {"groups": 0, "mixed_groups": 0, "turns": 0, "scored_turns": 0, "successes": 0, "episodes": 0}
        # The dev-curve success rate is an EPISODE-level statistic over the scheduled
        # task set (one episode per task), so the top-up episodes are excluded from it
        # -- counting them would silently re-weight whichever tasks needed topping up.
        primary_eps = 0
        primary_ok = 0
        sibling_pool = [e.trajectory for _eps, _adv in groups for e in _eps if e.trajectory.turns]
        for pos, (episodes, advantages) in enumerate(groups):
            group_id = group_ids[pos]
            rows = build_group_samples(
                state.tokenizer,
                group_games[pos],
                episodes,
                advantages,
                group_id,
                with_hpi=mode,
                future_gap=future_gap,
                sibling_pool=sibling_pool,
                rollout_id=rollout_id,
            )
            if os.environ.get("OPSD_SRLVR", "0") == "1":
                variant = os.environ.get("SRLVR_VARIANT", "")
                if len(episodes) != 1:
                    raise RuntimeError("SRLVR requires exactly one environment attempt per task sample")
                if variant == "v4":
                    # `_v4_rows` already attached the trajectory-level teacher
                    # view and the per-turn span; v2/v3's per-turn metadata is
                    # not used by the v4 loss.
                    pass
                else:
                    if variant in ("v2", "v3"):
                        from methods.srlvr_hsi import attach_rows
                    else:
                        from methods.srlvr import attach_rows
                    attach_rows(rows, episodes[0], group_id, rollout_id)
            if not rows:
                continue
            built.append(rows)
            counts["groups"] += 1
            counts["mixed_groups"] += int(getattr(advantages, "mixed", False))
            counts["turns"] += len(rows)
            counts["scored_turns"] += sum(
                1 for r in rows if ("hsi_teacher_view" in r.metadata or "hpi_views" in r.metadata)
            )
            counts["episodes"] += len(episodes)
            counts["successes"] += sum(1 for e in episodes if e.trajectory.success)
            if pos < len(games):
                primary_eps += len(episodes)
                primary_ok += sum(1 for e in episodes if e.trajectory.success)
        return built, counts, primary_eps, primary_ok

    data, stats, primary_episodes, primary_successes = _emit()
    compression = [r.metadata["future_compression"] for group in data for r in group
                   if "future_compression" in r.metadata]
    if compression:
        totals = {key: sum(item[key] for item in compression) for key in compression[0]}
        logger.info("[future-compression] rollout=%s %s", rollout_id, totals)
    if os.environ.get("ALFWORLD_TASK_SAMPLING", "frontier") == "frontier":
        from .curriculum import current_curriculum
        current_curriculum(args).observe(rollout_id, [
            (game, [episode.reward for episode in episodes])
            for game, (episodes, _) in zip(games, groups, strict=True)
        ])

    # -- floor re-check, in slime's own units ---------------------------------
    # `_top_up_to_floor` above counts turns, because that is cheap before any row
    # exists. slime's trim counts the flattened ``Sample`` rows it is handed. One
    # turn emits exactly one row, so those numbers agree and this loop never runs
    # -- but a `Not enough samples N` abort kills the run and costs every rollout
    # since the last checkpoint, so the floor is re-checked here against the rows
    # that were actually emitted and topped up again if the two ever diverge.
    while (
        min_rows > 0
        and sum(len(rows) for rows in data) < min_rows
        and topup_round < _MAX_TOPUP_ROUNDS
    ):
        topup_round += 1
        logger.warning(
            "[hsi-roll] emitted %d rows < floor %d after top-up round %d; "
            "turn count and row count disagree -- topping up again",
            sum(len(rows) for rows in data),
            min_rows,
            topup_round - 1,
        )
        await _top_up_round(
            args, state, games, seeds, engines, groups, group_games, group_ids, topup_round
        )
        data, stats, primary_episodes, primary_successes = _emit()

    await _abort(args, state)
    state.reset()

    episodes = max(primary_episodes, 1)
    positions = max(_CANDIDATE_COVERAGE["positions"], 1)
    generated_tokens = sum(int(s.response_length) for rows in data for s in rows)
    all_episodes = [episode for eps, _ in groups for episode in eps]
    all_turns = [turn for episode in all_episodes for turn in episode.trajectory.turns]
    valid_actions = sum(turn.action_valid for turn in all_turns)
    bare_commands = sum(getattr(turn, "harness_source", "structured") == "bare_command"
                        for episode in all_episodes for turn in episode.turns)
    metrics = {
        # Sums include overlapping episodes; they are work totals, not phase wall time.
        "rollout/generation_seconds_sum": sum(e.generation_seconds for e in all_episodes),
        "rollout/environment_seconds_sum": sum(e.environment_seconds for e in all_episodes),
        "rollout/episode_wall_seconds_max": max((e.wall_seconds for e in all_episodes), default=0.),
        "rollout/action_valid_rate": valid_actions / max(len(all_turns), 1),
        "rollout/harness_bare_commands": float(bare_commands),
        "rollout/success_rate": primary_successes / episodes,
        # Top-up accounting: `topup_rounds > 0` means this step would otherwise
        # have crashed the run for want of turn-rows, and `topup_episodes` is what
        # it cost. Surfaced so the floor can never degrade silently.
        "rollout/hsi_topup_rounds": float(topup_round),
        "rollout/hsi_topup_episodes": float(max(stats["episodes"] - primary_episodes, 0)),
        "rollout/groups": float(stats["groups"]),
        "rollout/mixed_groups": float(stats["mixed_groups"]),
        "rollout/mixed_group_rate": stats["mixed_groups"] / max(stats["groups"], 1),
        "rollout/turns": float(stats["turns"]),
        "rollout/hpi_scored_turns": float(stats["scored_turns"]),
        "rollout/leak_guard_skips": float(_LEAK_GUARD_SKIPS - leak_guard_before),
        # Live coverage of the candidate set: sampled token inside C_t (must be 1.0)
        # and the engine returned a full k-list. Both are properties the method
        # depends on, so they are reported every step rather than asserted once.
        "rollout/hsi_candidate_positions": float(_CANDIDATE_COVERAGE["positions"]),
        "rollout/hsi_sampled_in_candidates": _CANDIDATE_COVERAGE["sampled_in_set"] / positions,
        "rollout/hsi_full_k_rate": _CANDIDATE_COVERAGE["full_k"] / positions,
        "rollout/hsi_engines": float(len(engines)),
    }
    gigpo_metrics = _gigpo_metrics(groups)
    metrics.update(gigpo_metrics)
    if gigpo_metrics:
        # Anchor coverage is the health signal of this method: a rollout whose
        # anchors are almost all singletons has degenerated to plain GRPO
        # (step term identically 0), so it is reported every step.
        logger.info(
            "[gigpo] rollout=%d task_groups=%d anchors=%.0f turns=%.0f "
            "singleton=%.3f mean_group=%.2f max_group=%.0f step_nonzero=%.3f "
            "step_abs=%.4f episode_abs=%.4f w=%.3f gamma=%.3f mode=%s",
            rollout_id,
            int(gigpo_metrics["rollout/gigpo_task_groups"]),
            gigpo_metrics["rollout/gigpo_anchor_groups"],
            gigpo_metrics["rollout/gigpo_turns"],
            gigpo_metrics["rollout/gigpo_singleton_fraction"],
            gigpo_metrics["rollout/gigpo_mean_group_size"],
            gigpo_metrics["rollout/gigpo_max_group_size"],
            gigpo_metrics["rollout/gigpo_step_nonzero_fraction"],
            gigpo_metrics["rollout/gigpo_step_adv_mean_abs"],
            gigpo_metrics["rollout/gigpo_episode_adv_mean_abs"],
            gigpo_metrics["rollout/gigpo_step_w"],
            gigpo_metrics["rollout/gigpo_gamma"],
            "mean_std_norm" if gigpo_metrics["rollout/gigpo_mode_mean_std_norm"] else "mean_norm",
        )
    # Distinct prefix from the scorer's `[hsi-cand]` lines: the ledger splits the
    # cycle into phases by timestamp, and two different modules writing the same
    # tag made every phase boundary ambiguous (measured: cycle 0's "teacher span"
    # came out as 1045 s, spanning two cycles, because this line matched first).
    logger.info(
        "[hsi-roll] rollout=%d engines=%d sampled_in_C=%.6f full_k=%.6f scored_turns=%d/%d "
        "episodes=%d topup_rounds=%d topup_episodes=%d generated_tokens=%d rollout_seconds=%.1f",
        rollout_id,
        len(engines),
        metrics["rollout/hsi_sampled_in_candidates"],
        metrics["rollout/hsi_full_k_rate"],
        stats["scored_turns"],
        stats["turns"],
        primary_episodes,
        topup_round,
        max(stats["episodes"] - primary_episodes, 0),
        generated_tokens,
        _time.perf_counter() - _rollout_t0,
    )
    from opsd_core.experiment_metrics import record
    record(args, rollout_id, "rollout",
           action_harness=HARNESS_VERSION, action_valid_rate=valid_actions / max(len(all_turns), 1),
           valid_actions=valid_actions, environment_turns=len(all_turns), harness_bare_commands=bare_commands,
           success_rate=primary_successes/episodes, successes=primary_successes,
           failures=primary_episodes-primary_successes, tasks=len(games),
           real_rollouts=stats["episodes"], actor_tokens=generated_tokens,
           rollout_seconds=_time.perf_counter()-_rollout_t0,
           inference_allocated_device_seconds=len(engines)*(_time.perf_counter()-_rollout_t0),
           feature_reduction_host_seconds=sum(getattr(e, "feature_seconds", 0.) for eps,_ in groups for e in eps))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


async def _abort(args, state) -> None:
    from tasks.math.math_rollout import _abort as _impl

    await _impl(args, state)


# ---------------------------------------------------------------------------
# Evaluation: one row per episode is enough for success-rate reporting
# ---------------------------------------------------------------------------


async def _eval_async(args: Namespace, rollout_id: int):
    from slime.rollout.base_types import RolloutFnEvalOutput

    from .eval_data import load_eval_games

    state = GenerateState(args)
    # Spread episodes over EVERY exported engine, exactly as the training path
    # (_run_group) already does.  Omitting ``engine_url`` here made
    # episode._generate fall back to ``OPSD_ENGINE_URL``, which the rollout
    # manager pins to engines[0]: measured 2026-10-07 on the 1.7B HSI alfworld
    # eval, engines[0] served 3549 requests / 2.246e6 tokens while the other six
    # reported ``num_running_reqs 0`` and no request counter at all -- 6 of the
    # 7 engines (and 6 of the node's 8 NPUs) idled for the whole evaluation.
    # Engine choice cannot change a result: every episode and every turn carries
    # its own sampling seed, and an episode keeps one engine for all its turns,
    # so the prefix cache is still reused within the episode.
    engines = _engine_urls(args)
    results: dict[str, dict] = {}
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        key = getattr(dataset_cfg, "name", None) or str(dataset_cfg.path)
        games = load_eval_games(dataset_cfg.path)
        semaphore = asyncio.Semaphore(max(1, int(os.environ.get("ALFWORLD_EVAL_CONCURRENCY", "8"))))
        max_response = config.get_int("ALFWORLD_V2_MAX_RESPONSE", config.MAX_RESPONSE_TOKENS)

        async def one(index: int, game: GameRef):
            async with semaphore:
                params = _sampling_params(state, seed=10000 + index, max_response=max_response)
                return await run_episode(
                    args,
                    state.tokenizer,
                    game,
                    params,
                    episode_id=index,
                    group_id=index,
                    engine_url=engines[index % len(engines)],
                )

        episodes = await asyncio.gather(*[one(i, g) for i, g in enumerate(games)])
        samples = []
        for index, episode in enumerate(episodes):
            sample = Sample(index=index, prompt=episode.trajectory.task_description)
            sample.reward = episode.reward
            sample.response = ""
            sample.status = (
                Sample.Status.TRUNCATED if episode.trajectory.truncated else Sample.Status.COMPLETED
            )
            sample.metadata["trajectory"] = episode.trajectory.to_dict()
            sample.metadata["action_harness"] = dict(version=HARNESS_VERSION)
            samples.append(sample)
        results[key] = {
            "rewards": [float(s.reward) for s in samples],
            "samples": samples,
            "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in samples],
        }
        logger.info(
            "[v2-eval] %s: n=%d success_rate=%.4f",
            key,
            len(samples),
            sum(float(s.reward) for s in samples) / max(len(samples), 1),
        )
    return RolloutFnEvalOutput(data=results)
