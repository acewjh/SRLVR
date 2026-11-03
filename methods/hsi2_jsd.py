"""HSI v2 (JSD) -- running-baseline SIGN x exact-normalized-JSD WHERE.

AUTHORITY: ``TASK_MATH_HSI_V2_SPEC.md`` (project root, 2026-10-04 20:4x). This
module implements that spec literally and **supersedes** the candidate-centered
estimator (the retired ``tasks.alfworld/hsi_v2.py``, archived 2026-10-15 under
``_retired/methods/``). The small model-access helpers below are re-implemented
here rather than imported from it -- a rename over there must not be able to break
this run. The one thing it does share is the actor handle
(``tasks.alfworld.hsi_shared.actor_model``), which the provider publishes for
every task.

The estimator
-------------

Per generated token position ``t`` of a trajectory ``i`` with verifier reward
``R_i`` (the SAME scalar the matched GRPO arm consumes)::

    p_t   = pi_{theta_old}(. | h_t)                    normal prefix view
    q_t   = pi_{theta_old}(. | h_t, F_t, O)            hindsight view
    m_t   = (p_t + q_t) / 2
    JSD_t = 0.5 KL(p_t||m_t) + 0.5 KL(q_t||m_t)
    w_t   = JSD_t / log 2                              in [0, 1]

    b_k   = reward_sum / reward_count                  ONLY prior trajectories
    A_i,t = (R_i - b_k) * w_i,t                        detached

Both views are produced by the **same theta_old** -- the actor's own live weights
at the instant this runs, which before ``optimizer.step()`` ARE the weights that
generated the rollout. Features per the spec's exclusion list:

* **no frozen theta_0 teacher, no external teacher, no matched future, no
  reference solution, no old-actor replay.** ``F_t`` is the trajectory's own
  realized continuation (a strict suffix of the scored span, so ``y_t`` can
  never be repeated into it -- enforced structurally in ``hsi_rollout``).
* **no candidate set, no ``g(y) - sum rho g``, no ``2R-1``, no trajectory
  credit normalisation, no whitening, no credit clipping.** SIGN is the running
  baseline, WHERE is the JSD, and nothing else touches ``A``.
* **no ``[T, V]`` materialisation.** The lm-head is applied only to a
  ``JSD_POSITION_CHUNK``-wide slice of positions at a time and the full-vocab
  tensor is released before the next slice (spec section 8).
* **the outcome descriptor never enters SIGN.** It only appears as text in the
  hindsight context; the advantage multiplies a reward number and a JSD.

Freezing order (spec section 6): ``postprocess`` runs from Megatron's actor at
``actor.py`` -- after ``compute_advantages_and_returns`` and **before**
``train()`` -- so every ``w`` and ``A`` in a rollout is produced with one
theta_old, before the first ``optimizer.step()`` of that rollout.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

__all__ = [
    "JSD_POSITION_CHUNK",
    "estimator",
    "render_outcome_descriptor",
    "jsd_token_weights",
    "paired_jsd_weights",
    "baseline_state",
    "postprocess",
    "reset_counters",
    "stats",
]

logger = logging.getLogger(__name__)

#: Positions projected through the lm-head per slice. The spec pins 64 as the
#: default and allows exactly the fallback chain 64 -> 32 -> 16 as a pure
#: execution knob (section 8). Never an estimator parameter.
JSD_POSITION_CHUNK = int(os.environ.get("OPSD_HSI2_POS_CHUNK") or 64)

_FALLBACK_CHUNKS = (64, 32, 16)

_LOG2 = math.log(2.0)

#: Estimator selector. Every live arm exports ``jsd`` (this module); the historic
#: ``hsi_v2`` spelling selected the candidate-centred estimator, retired 2026-10-15
#: with the ALFWorld arm that carried it. The value is kept only so a stale launcher
#: still fails closed (the guard below returns without scoring) instead of silently
#: running JSD under the old label.
_ESTIMATOR_ENV = "OPSD_HSI_ESTIMATOR"


def estimator() -> str:
    """Which estimator this process runs: ``jsd`` (this module) or the retired
    ``hsi_v2``."""
    return os.environ.get(_ESTIMATOR_ENV, "jsd").strip().lower() or "jsd"


# ---------------------------------------------------------------------------
# Section 2/3 -- the unified outcome descriptor renderer
# ---------------------------------------------------------------------------


def render_outcome_descriptor(verifier_result, task_metadata=None) -> str:
    """Short natural-language outcome text for the hindsight context.

    Same framework/contract as ``tasks.alfworld.hsi_shared.render_outcome_descriptor``
    (the task book builds one renderer shared by Math / Code / ALFWorld): the
    verifier is not assumed to be a bool, semantics the verifier really produced
    are preserved, and a bare reward number is never emitted as if it explained
    itself. Math's binary verifier therefore renders the spec's wording -- it
    says what the verifier *judged*, not what a number was.
    """
    body = outcome_body(verifier_result)
    return f"<outcome>{body}</outcome>"


def outcome_body(verifier_result) -> str:
    """The prose inside ``<outcome>...</outcome>`` (no tags)."""
    graded = _graded_score(verifier_result)
    if graded is not None:
        if graded >= 1.0:
            return "The verifier judged the completed solution as correct."
        if graded <= 0.0:
            return "The verifier judged the completed solution as incorrect."
        return (
            f"The verifier judged the completed solution as partially correct "
            f"(score {graded:.2f} out of 1.00)."
        )

    passed = bool(verifier_result)
    if isinstance(verifier_result, dict):
        for key in ("success", "passed", "ok", "won", "correct"):
            if key in verifier_result:
                passed = bool(verifier_result[key])
                break
    if passed:
        return "The verifier judged the completed solution as correct."
    return "The verifier judged the completed solution as incorrect."


def _graded_score(verifier_result):
    """A numeric score in [0, 1] the verifier actually reported, else ``None``."""
    if isinstance(verifier_result, dict):
        for key in ("score", "fraction"):
            value = verifier_result.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _clamp_unit(float(value))
        metrics = verifier_result.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("score"), (int, float)):
            return _clamp_unit(float(metrics["score"]))
        return None
    if isinstance(verifier_result, (int, float)) and not isinstance(verifier_result, bool):
        return _clamp_unit(float(verifier_result))
    return None


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Model access (mirrors the retired tasks.alfworld.hsi_v2 -- re-implemented, not imported)
# ---------------------------------------------------------------------------


def _actor_model():
    from tasks.alfworld.hsi_shared import actor_model

    return actor_model()


def _accelerator_device():
    """The trainer's device index, or the host when there is no accelerator.

    Under torch_npu ``torch.cuda`` *is* the NPU interface, so this is the trainer
    device in production and plain CPU under the contract tests.
    """
    import torch

    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return "cpu"


def _unwrap(model):
    """Strip DDP / wrapper layers down to the module that owns the lm-head."""
    current, depth = model, 0
    while current is not None and depth < 8:
        if _find_lm_head(current) is not None:
            return current
        nxt = getattr(current, "module", None)
        if nxt is None or nxt is current:
            break
        current = nxt
        depth += 1
    return current


def _find_lm_head(model):
    """The output layer of a Megatron actor chunk, or ``None``.

    Qwen3-4B ties its embeddings, so mcore builds the output layer with
    ``skip_weight_param_allocation=True`` and ``output_layer.weight`` is ``None``
    -- requiring a weight here is what made an earlier run abort on a healthy
    model. Accept ``output_layer`` / ``lm_head`` by name as a backstop.
    """
    for holder in (model, getattr(model, "language_model", None)):
        if holder is None:
            continue
        head = getattr(holder, "output_layer", None)
        if head is not None and hasattr(head, "forward"):
            return head
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in ("output_layer", "lm_head") and hasattr(module, "forward"):
            return module
    return None


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


class _HiddenTap:
    """Bypass the lm-head and keep its input (the post-final-norm hidden states).

    Megatron calls ``logits, _ = self.output_layer(hidden)``; the tap returns the
    same pair with the projection skipped, so the transformer still runs exactly
    once and no ``[T, V]`` tensor is ever formed.
    """

    def __init__(self, head):
        self.head = head
        self.hidden = None

    def __enter__(self):
        self._original = self.head.forward

        def _tap(x, *args, **kwargs):
            self.hidden = x
            return (x, None)

        self.head.forward = _tap
        return self

    def __exit__(self, *exc):
        self.head.forward = self._original
        return False


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


# ---------------------------------------------------------------------------
# Section 5/8 -- exact normalized JSD, position-chunked
# ---------------------------------------------------------------------------


def _jsd_from_logits(logits_p, logits_q):
    """``[C]`` normalized JSD (base-2) for aligned position slices.

    Everything after the projection is fp32 log-space (spec section 8). The two
    ``[C, V]`` tensors below are the only full-vocab materialisations anywhere in
    this file and they are released as soon as the ``[C]`` reduction is done.

    ``log m`` is formed with ``logaddexp(lp, lq) - log 2`` rather than by
    exponentiating and re-taking a log. That is not just cheaper (it skips the
    ``p``/``q``/``m`` intermediates): the naive form is a **NaN hazard**. Any
    token whose probability underflows to exactly 0 makes ``log m`` ``-inf``, and
    ``0 * (-inf - lp)`` is NaN -- so a single vanishingly unlikely token would
    poison the whole position's weight. In log space the difference
    ``lp - log_m`` stays finite for every token and the ``exp(lp)`` factor drives
    the contribution to 0 on its own.
    """
    import torch

    lp = torch.log_softmax(logits_p, dim=-1)
    lq = torch.log_softmax(logits_q, dim=-1)
    del logits_p, logits_q

    log_m = torch.logaddexp(lp, lq) - _LOG2
    kl_p = (lp.exp() * (lp - log_m)).sum(dim=-1)
    kl_q = (lq.exp() * (lq - log_m)).sum(dim=-1)
    del lp, lq, log_m
    return (0.5 * (kl_p + kl_q)) / _LOG2


def paired_jsd_weights(actor, normal_ids, normal_start, hindsight_ids, hindsight_start, n_positions):
    """Per-position ``w_t = JSD(p_t, q_t)/log 2`` for one training row.

    ``normal_start`` / ``hindsight_start`` are the *hidden-state* index at which
    the first scored position begins, i.e. ``target_token_index - 1`` in each
    view's own coordinates (hidden state ``h`` predicts token ``h+1``).

    Returns a list of ``n_positions`` floats, or ``None`` when the row cannot be
    scored at all (no actor head / empty span), so the caller can MASK it instead
    of inventing a weight.
    """
    import torch

    if n_positions <= 0:
        return None
    model = _unwrap(actor)
    head = _find_lm_head(model)
    if head is None:
        raise RuntimeError(
            "HSI v2 could not locate the actor's output layer; the paired "
            "same-policy scoring pass cannot be issued (is PP > 1?)"
        )
    weight = _lm_head_weight(model, head)
    head_dtype = getattr(weight, "dtype", torch.float32)

    hidden_n = _hidden_states(model, head, normal_ids)
    hidden_h = _hidden_states(model, head, hindsight_ids)

    if hidden_n.shape[0] < normal_start + n_positions:
        raise RuntimeError(
            f"HSI v2: normal view has {hidden_n.shape[0]} hidden states, needs "
            f"{normal_start + n_positions}"
        )
    if hidden_h.shape[0] < hindsight_start + n_positions:
        raise RuntimeError(
            f"HSI v2: hindsight view has {hidden_h.shape[0]} hidden states, needs "
            f"{hindsight_start + n_positions}"
        )

    weights: list[float] = []
    with torch.no_grad():
        for start in range(0, n_positions, JSD_POSITION_CHUNK):
            stop = min(start + JSD_POSITION_CHUNK, n_positions)
            hn = hidden_n[normal_start + start : normal_start + stop]
            hh = hidden_h[hindsight_start + start : hindsight_start + stop]
            logits_p = (hn.to(head_dtype) @ weight.t()).float()
            logits_q = (hh.to(head_dtype) @ weight.t()).float()
            jsd = _jsd_from_logits(logits_p, logits_q)
            weights.extend(float(v) for v in jsd.tolist())
    del hidden_n, hidden_h
    return weights


def jsd_token_weights(actor, normal_ids, normal_start, hindsight_ids, hindsight_start, n_positions):
    """``paired_jsd_weights`` with the spec's execution-only chunk fallback.

    The spec allows exactly one deviation when memory is tight: shrink the
    position chunk along ``64 -> 32 -> 16``. Anything else (top-k JSD, a smaller
    model, a different G or sampler) would change the estimator and is refused.
    """
    import torch

    last: Exception | None = None
    for chunk in _FALLBACK_CHUNKS:
        if chunk > JSD_POSITION_CHUNK:
            continue
        globals()["JSD_POSITION_CHUNK"] = chunk
        try:
            return paired_jsd_weights(
                actor, normal_ids, normal_start, hindsight_ids, hindsight_start, n_positions
            )
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203
            last = exc
            torch.cuda.empty_cache()
            logger.warning("[hsi2] position chunk %d OOMed; falling back", chunk)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last = exc
            torch.cuda.empty_cache()
            logger.warning("[hsi2] position chunk %d OOMed (%s); falling back", chunk, exc)
    raise RuntimeError("HSI v2: every position chunk size OOMed; refusing to shrink further") from last


# ---------------------------------------------------------------------------
# Section 4 -- historical running baseline, DP-global, checkpoint-coupled
# ---------------------------------------------------------------------------

_BASELINE_FILENAME = "hsi2_baseline.json"
_baseline_singleton: "_BaselineLedger | None" = None


class _BaselineLedger:
    """``b_k`` = mean reward over trajectories seen in **earlier** rollouts.

    The current rollout's rewards never enter their own baseline (spec section 4).
    State lives in one JSON file in the run directory, rewritten atomically after
    every rollout, so a resume restores it with the Megatron checkpoint: the
    ledger is keyed by ``rollout_id``, which makes a replayed rollout (HF resume
    replays the rollouts after the last saved iteration) an exact no-op instead of
    a double count.
    """

    def __init__(self, path: str | None):
        self.path = path
        self.records: dict[int, dict] = {}
        self._loaded = False

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        self.records = {}
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path) as handle:
                    raw = json.load(handle)
                self.records = {int(k): v for k, v in raw.items()}
            except (OSError, ValueError):
                logger.warning("[hsi2] baseline ledger %s unreadable; restarting the baseline", self.path)
                self.records = {}
        self._loaded = True

    def _flush(self) -> None:
        if not self.path or not _is_writer():
            return
        tmp = f"{self.path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as handle:
                json.dump({str(k): v for k, v in sorted(self.records.items())}, handle)
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("[hsi2] could not persist baseline ledger: %s", exc)

    # -- arithmetic --------------------------------------------------------
    def _prior(self, rollout_id: int) -> tuple[float, int]:
        """``(reward_sum, reward_count)`` accumulated strictly before this rollout."""
        best = None
        for key, record in self.records.items():
            if key < rollout_id and (best is None or key > best):
                best = key
        if best is None:
            return 0.0, 0
        record = self.records[best]
        return float(record["cum_sum"]), int(record["cum_count"])

    def baseline(self, rollout_id: int) -> float:
        """``b_k`` for this rollout -- recorded on replay, derived otherwise."""
        record = self.records.get(rollout_id)
        if record is not None:
            return float(record["b_k"])
        total, count = self._prior(rollout_id)
        return total / count if count > 0 else 0.0

    def commit(self, rollout_id: int, cur_sum: float, cur_count: int) -> None:
        """Fold this rollout's rewards in -- unless it is a replay of a known one."""
        if rollout_id in self.records:
            # HF resume replays rollouts after the last saved iteration. The
            # ledger already holds their contribution; adding it again would
            # silently bias every later baseline.
            return
        total, count = self._prior(rollout_id)
        self.records[rollout_id] = {
            "b_k": total / count if count > 0 else 0.0,
            "cur_sum": float(cur_sum),
            "cur_count": int(cur_count),
            "cum_sum": total + float(cur_sum),
            "cum_count": count + int(cur_count),
        }
        self._flush()


def _is_writer() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:  # pragma: no cover - non-distributed tooling / tests
        return True
    return True


def _baseline_path(args) -> str | None:
    override = os.environ.get("OPSD_HSI2_STATE_DIR")
    base = override or getattr(args, "save", None)
    if not base:
        return None
    return os.path.join(base, _BASELINE_FILENAME)


def baseline_state(args) -> _BaselineLedger:
    """The process-wide ledger, loaded once."""
    global _baseline_singleton
    if _baseline_singleton is None or _baseline_singleton.path != _baseline_path(args):
        ledger = _BaselineLedger(_baseline_path(args))
        ledger.load()
        _baseline_singleton = ledger
    elif not _baseline_singleton._loaded:
        _baseline_singleton.load()
    return _baseline_singleton


def reset_baseline_for_tests() -> None:
    global _baseline_singleton
    _baseline_singleton = None


# ---------------------------------------------------------------------------
# DP collectives
# ---------------------------------------------------------------------------


def _dp_group_and_size():
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu

        if dist.is_available() and dist.is_initialized():
            return mpu.get_data_parallel_group(), int(mpu.get_data_parallel_world_size())
    except Exception:  # pragma: no cover - unit tests / offline tooling
        return None, 1
    return None, 1


def _gather_pairs(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """All ``(trajectory_id, reward)`` pairs of this rollout, on every rank.

    Trajectories are NOT guaranteed to sit on one DP rank: the rollout manager
    partitions rows, and a 4-chunk trajectory can straddle ranks. A local
    ``sum``/``count`` all-reduce would then weight a straddling trajectory by how
    many ranks it happened to touch, so the pairs are gathered and de-duplicated
    by trajectory id instead. The payload is 32 pairs.
    """
    import torch.distributed as dist

    group, size = _dp_group_and_size()
    if group is None or size <= 1:
        return pairs
    gathered: list[list[tuple[int, float]]] = [None] * size  # type: ignore[list-item]
    dist.all_gather_object(gathered, pairs, group=group)
    merged: list[tuple[int, float]] = []
    for chunk in gathered:
        if chunk:
            merged.extend(chunk)
    return merged


# ---------------------------------------------------------------------------
# Counters / stats
# ---------------------------------------------------------------------------

_SCORER_CALLS = 0
_SCORED_ROWS = 0
_SCORED_POSITIONS = 0
_MASKED_POSITIONS = 0
_SKIPPED_ROWS = 0
_STATS: dict[str, float] = {}


def reset_counters() -> None:
    global _SCORER_CALLS, _SCORED_ROWS, _SCORED_POSITIONS, _MASKED_POSITIONS, _SKIPPED_ROWS
    _SCORER_CALLS = 0
    _SCORED_ROWS = 0
    _SCORED_POSITIONS = 0
    _MASKED_POSITIONS = 0
    _SKIPPED_ROWS = 0
    _STATS.clear()


def stats() -> dict[str, float]:
    return {
        "scorer_calls": float(_SCORER_CALLS),
        "scored_rows": float(_SCORED_ROWS),
        "scored_positions": float(_SCORED_POSITIONS),
        "masked_positions": float(_MASKED_POSITIONS),
        "skipped_rows": float(_SKIPPED_ROWS),
        **_STATS,
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))]


# ---------------------------------------------------------------------------
# slime's --rollout-data-postprocess-path entry point
# ---------------------------------------------------------------------------


def postprocess(args, rollout_id: int, rollout_data: dict) -> None:
    """Fill ``opsd_advantages`` with ``(R_i - b_k) * w_i,t`` (spec sections 4-6).

    Called by Megatron's actor after ``compute_advantages_and_returns`` and before
    ``train()``, so the model this forwards is exactly the theta_old that produced
    the batch and every ``w``/``A`` is frozen before the first optimizer step.
    """
    global _SCORER_CALLS, _SCORED_ROWS, _SCORED_POSITIONS, _MASKED_POSITIONS, _SKIPPED_ROWS

    if estimator() != "jsd":
        return

    actor = _actor_model()
    if actor is None:
        raise RuntimeError(
            "HSI v2 requires the actor handle; launch with "
            "--custom-model-provider-path tasks.alfworld.hsi_provider.custom_model_provider"
        )

    import torch

    t_start = time.perf_counter()

    rewards = rollout_data.get("rewards") or []
    tokens = rollout_data.get("tokens") or []
    response_lengths = rollout_data.get("response_lengths") or []
    turn_ids = rollout_data.get("turn_ids") or []
    views = rollout_data.get("hpi_views") or []
    prompt_lengths = rollout_data.get("hpi_prompt_lengths") or []
    target_lengths = rollout_data.get("hpi_target_lengths") or []
    loss_masks = rollout_data.get("loss_masks")

    from tasks.alfworld.config import TURN_ID_STRIDE

    # ---- section 4: the SIGN, from trajectories strictly before this rollout --
    ledger = baseline_state(args)
    pairs: list[tuple[int, float]] = []
    for index in range(len(response_lengths)):
        if index >= len(turn_ids) or index >= len(rewards):
            continue
        if loss_masks is not None and not bool(torch.as_tensor(loss_masks[index]).any()):
            continue  # DP padding is not a real trajectory/reward observation.
        first = turn_ids[index][0]
        first = int(first.item()) if hasattr(first, "item") else int(first)
        pairs.append((first // TURN_ID_STRIDE, float(rewards[index])))
    merged = _gather_pairs(pairs)
    unique: dict[int, float] = {}
    for trajectory, reward in merged:
        unique[trajectory] = reward
    cur_sum = float(sum(unique.values()))
    cur_count = len(unique)

    b_k = ledger.baseline(rollout_id)
    logger.info(
        "[hsi2] rollout=%d baseline b_k=%.6f (prior_reward_sum=%.3f prior_count=%d) "
        "this_rollout: trajectories=%d mean_reward=%.4f",
        rollout_id,
        b_k,
        *ledger._prior(rollout_id),
        cur_count,
        (cur_sum / cur_count) if cur_count else 0.0,
    )

    row_results: list = []
    score_seconds = 0.0
    jsd_values: list[float] = []
    adv_values: list[float] = []

    for index in range(len(response_lengths)):
        rlen = int(response_lengths[index])
        if loss_masks is not None and not bool(torch.as_tensor(loss_masks[index]).any()):
            row_results.append(None)
            continue
        if rlen <= 0:
            row_results.append(None)
            continue

        view = views[index] if index < len(views) else None
        plen_row = prompt_lengths[index] if index < len(prompt_lengths) else None
        tlen_row = target_lengths[index] if index < len(target_lengths) else None
        has_view = bool(view) and bool(view[0]) and bool(plen_row) and bool(tlen_row)
        if not has_view or int(tlen_row[0]) != rlen:
            _SKIPPED_ROWS += 1
            _MASKED_POSITIONS += rlen
            row_results.append(None)
            continue

        row_tokens = tokens[index]
        row_tokens = row_tokens.tolist() if hasattr(row_tokens, "tolist") else list(row_tokens)
        total = len(row_tokens)
        if total - rlen < 1:
            # A forward needs at least one context token before the scored span.
            _SKIPPED_ROWS += 1
            _MASKED_POSITIONS += rlen
            row_results.append(None)
            continue

        _SCORER_CALLS += 1
        t_score = time.perf_counter()
        try:
            weights = jsd_token_weights(
                actor,
                row_tokens,
                total - rlen - 1,
                list(view[0]),
                int(plen_row[0]) - 1,
                rlen,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning("[hsi2] row %d OOMed; masking it", index)
            weights = None
        score_seconds += time.perf_counter() - t_score

        if weights is None or len(weights) != rlen:
            _SKIPPED_ROWS += 1
            _MASKED_POSITIONS += rlen
            row_results.append(None)
            continue

        sign = float(rewards[index]) - b_k
        row = [0.0] * rlen
        keep = [1] * rlen
        scored = 0
        for position in range(rlen):
            weight = weights[position]
            if not math.isfinite(weight):
                keep[position] = 0
                continue
            row[position] = sign * weight
            jsd_values.append(weight)
            adv_values.append(row[position])
            scored += 1
        _SCORED_ROWS += 1
        _SCORED_POSITIONS += scored
        _MASKED_POSITIONS += rlen - scored
        row_results.append((row, keep))

    import torch

    device = _accelerator_device()
    advantages = []
    for index, entry in enumerate(row_results):
        rlen = int(response_lengths[index]) if index < len(response_lengths) else 0
        if entry is None:
            advantages.append(torch.zeros(rlen, dtype=torch.float32, device=device))
            if loss_masks is not None and index < len(loss_masks):
                loss_masks[index] = loss_masks[index] * 0
            continue
        row, keep = entry
        # 1-D TENSORS, not Python lists: slime's `log_rollout_data` branches on
        # `isinstance(val[0], torch.Tensor)` and falls through to `sum(val)/len(val)`
        # otherwise, which is an int + list TypeError. Every other consumer
        # (`torch.as_tensor` in the loss) accepts either, so tensors are the shape
        # that satisfies all of them.
        advantages.append(torch.tensor(row, dtype=torch.float32, device=device))
        if 0 in keep and loss_masks is not None and index < len(loss_masks):
            original = loss_masks[index]
            gate = torch.tensor(keep[: len(original)], dtype=original.dtype, device=original.device)
            loss_masks[index] = original * gate

    rollout_data["opsd_advantages"] = advantages
    if "advantages" in rollout_data:
        rollout_data["advantages"] = [a.clone() for a in advantages]

    # ---- section 4: the baseline absorbs this rollout only AFTER all A exist --
    ledger.commit(rollout_id, cur_sum, cur_count)

    ordered = sorted(jsd_values)
    _STATS["b_k"] = b_k
    _STATS["reward_mean"] = (cur_sum / cur_count) if cur_count else 0.0
    _STATS["score_seconds"] = score_seconds
    _STATS["total_seconds"] = time.perf_counter() - t_start
    _STATS["n_trajectories"] = float(cur_count)
    if ordered:
        _STATS["jsd_mean"] = sum(ordered) / len(ordered)
        _STATS["jsd_p50"] = _percentile(ordered, 0.50)
        _STATS["jsd_p95"] = _percentile(ordered, 0.95)
        _STATS["jsd_max"] = ordered[-1]
    if adv_values:
        adv_sorted = sorted(adv_values)
        _STATS["adv_mean"] = sum(adv_sorted) / len(adv_sorted)
        _STATS["adv_p01"] = _percentile(adv_sorted, 0.01)
        _STATS["adv_p50"] = _percentile(adv_sorted, 0.50)
        _STATS["adv_p99"] = _percentile(adv_sorted, 0.99)
        _STATS["adv_pos_frac"] = sum(1 for a in adv_sorted if a > 0) / len(adv_sorted)
        _STATS["adv_neg_frac"] = sum(1 for a in adv_sorted if a < 0) / len(adv_sorted)
        _STATS["adv_near_zero_frac"] = sum(1 for a in adv_sorted if abs(a) < 1e-6) / len(adv_sorted)
        _STATS["adv_min"] = adv_sorted[0]
        _STATS["adv_max"] = adv_sorted[-1]
    logger.info(
        "[hsi2] rollout=%d rows=%d scored_rows=%d positions=%d masked=%d | "
        "b_k=%.6f reward_mean=%.4f | jsd mean=%.5f p50=%.5f p95=%.5f max=%.5f | "
        "adv mean=%.5f p01=%.5f p99=%.5f pos=%.3f neg=%.3f ~0=%.3f | score_s=%.2f total_s=%.2f",
        rollout_id,
        len(response_lengths),
        _SCORED_ROWS,
        _SCORED_POSITIONS,
        _MASKED_POSITIONS,
        b_k,
        _STATS["reward_mean"],
        _STATS.get("jsd_mean", 0.0),
        _STATS.get("jsd_p50", 0.0),
        _STATS.get("jsd_p95", 0.0),
        _STATS.get("jsd_max", 0.0),
        _STATS.get("adv_mean", 0.0),
        _STATS.get("adv_p01", 0.0),
        _STATS.get("adv_p99", 0.0),
        _STATS.get("adv_pos_frac", 0.0),
        _STATS.get("adv_neg_frac", 0.0),
        _STATS.get("adv_near_zero_frac", 0.0),
        score_seconds,
        _STATS["total_seconds"],
    )
