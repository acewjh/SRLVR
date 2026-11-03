"""Shared HSI primitives that outlived the ALFWorld arms that introduced them.

Several symbols used to live in the three ALFWorld arms retired on 2026-10-15
(``hsi_v2.py``, ``candidate_center.py``, ``candidate_credit.py``, ``scorer.py``)
but are NOT specific to those arms -- the live HSI path
(``math_opsd/hsi2_jsd.py`` + ``tasks/alfworld/loss.py`` via
``tasks/alfworld/{rollout,train_data,hsi_provider}.py``) still imports them.
They are collected here so archiving the arms does not break the live method.

Three groups:

* **outcome descriptor** -- :func:`render_outcome_descriptor`: the ALFWorld-flavoured
  verifier-prose renderer used by the live alfworld rollout
  (``tasks/alfworld/rollout.py``) and the math/code rollout
  (``math_opsd/hsi_rollout.py``, candidate branch). Moved verbatim from
  ``hsi_v2.py``.
* **the actor handle** -- :func:`register_actor_model` / :func:`actor_model`: the
  trained-model handle that ``tasks/alfworld/hsi_provider.py`` publishes and every
  task's HSI postprocess reads, because slime calls the postprocess hook with no
  model argument. Moved verbatim from ``hsi_v2.py``.
* **candidate transport** -- :func:`sampling_probabilities` and the
  :func:`encode_candidates` / :func:`decode_candidates` codec: the rollout engine's
  per-position candidate table, normalised and packed into the already-whitelisted
  ``hpi_user_lengths`` int channel. Moved verbatim from ``candidate_center.py``.
  The live alfworld rollout/convert still build and pack this table (the JSD
  estimator does not consume it, but the table is produced on every HSI turn).

Only what the live tree actually references was moved; the arms' estimators
(themselves, their postprocess hooks, the credit maths, the frozen teacher) stay in
``_retired/methods/``.
"""

from __future__ import annotations

import logging
import math
import os

__all__ = [
    "render_outcome_descriptor",
    "register_actor_model",
    "actor_model",
    "sampling_probabilities",
    "encode_candidates",
    "decode_candidates",
    "PROB_SCALE",
    "CANDIDATE_K",
    "TOP_P",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The actor handle (from hsi_v2.py)
# ---------------------------------------------------------------------------
#: Set once by :mod:`tasks.alfworld.hsi_provider` when the actor is built. The
#: postprocess hook receives ``(args, rollout_id, rollout_data)`` and no model, so
#: this is how the scorer reaches theta_old. The optimizer updates the very same
#: Parameter objects in place, so the reference stays current across steps.
_ACTOR: dict[str, object] = {"model": None}


def register_actor_model(model) -> None:
    """Record the actor so the postprocess hook can score with theta_old."""
    _ACTOR["model"] = model


def actor_model():
    """The registered actor, or ``None`` when nothing registered one."""
    return _ACTOR["model"]


# ---------------------------------------------------------------------------
# The verifier / outcome descriptor (from hsi_v2.py)
# ---------------------------------------------------------------------------


def render_outcome_descriptor(verifier_result, task_metadata=None) -> str:
    """Short natural-language outcome text for the hindsight context.

    The task book is explicit that the verifier is not assumed to be a bool, and
    that the descriptor must (a) keep whatever semantics the verifier really has,
    (b) never fabricate information the verifier did not produce, and (c) never
    reduce to a bare reward scalar. Three shapes are recognised, in order:

    * ``dict``/object with a numeric ``score`` (or ``metrics["score"]``) in
      ``[0, 1]`` -- graded / partial credit, reported as such;
    * ``dict`` with a ``passed``/``success`` flag plus an optional ``detail``;
    * anything truthy/falsy -- the binary terminal verdict ALFWorld emits today.

    The return value is the ``<outcome>...</outcome>`` block itself, so callers
    can drop it straight into the privileged block.
    """
    score = _graded_score(verifier_result)
    if score is not None:
        if score >= 1.0:
            body = "The verifier score was 1.00 out of 1.00, indicating full task completion."
        elif score <= 0.0:
            body = "The verifier score was 0.00 out of 1.00, indicating no task completion."
        else:
            body = (
                f"The verifier score was {score:.2f} out of 1.00, indicating partial task "
                "completion."
            )
        return f"<outcome>{body}</outcome>"

    passed = _verdict_bool(verifier_result)
    if passed:
        return "<outcome>The verifier judged the completed trajectory as successful.</outcome>"
    return "<outcome>The verifier judged the completed trajectory as unsuccessful.</outcome>"


def _graded_score(verifier_result):
    """A numeric score in [0, 1] the verifier actually reported, else ``None``."""
    if isinstance(verifier_result, dict):
        for key in ("score", "reward", "fraction"):
            value = verifier_result.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _clamp_unit(float(value))
        metrics = verifier_result.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get("score")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _clamp_unit(float(value))
        return None
    if isinstance(verifier_result, (int, float)) and not isinstance(verifier_result, bool):
        return _clamp_unit(float(verifier_result))
    return None


def _verdict_bool(verifier_result) -> bool:
    if isinstance(verifier_result, dict):
        for key in ("success", "passed", "ok", "won"):
            if key in verifier_result:
                return bool(verifier_result[key])
        return False
    return bool(verifier_result)


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# The sampling distribution on the candidate set (from candidate_center.py)
# ---------------------------------------------------------------------------

#: ``--rollout-top-k``. The engine's returned top-k list is the candidate set; the
#: sampling pipeline never draws outside it, so ``y_t in C_t`` holds by construction.
CANDIDATE_K = int(os.environ.get("ALFWORLD_V2_HSI_CANDIDATE_K") or 20)

#: ``--rollout-top-p``. Applied to the candidate list to reproduce the nucleus the
#: sampler actually normalised over.
TOP_P = float(os.environ.get("ALFWORLD_V2_HSI_TOP_P") or 0.95)

#: Fixed-point scale for the probability payload. ``q = round(p * SCALE)``, clamped
#: to >= 1 so no candidate is erased, then re-normalised on decode -- the
#: re-normalisation makes the decoded vector sum to 1 exactly.
PROB_SCALE = 10**9


def _sampler_logprob_temperature() -> float:
    """Temperature to divide the engine's log-probs by, or 0 when they are already
    the sampling distribution.

    The engine returns ``log_softmax(logits / T)`` by DEFAULT. If it is instead run
    with ``SGLANG_RETURN_ORIGINAL_LOGPROB=1`` (which the candidate launcher sets so
    the PPO old log-prob shares the TRAINER's raw-logit convention) the values are
    pre-temperature, and they must be divided by T here before softmax. Configured
    by ``ALFWORLD_V2_HSI_SAMPLER_TEMP``; 0 (the default) leaves the values untouched,
    so every pre-existing caller is bit-for-bit unchanged.
    """
    raw = os.environ.get("ALFWORLD_V2_HSI_SAMPLER_TEMP", "")
    try:
        return float(raw) if raw.strip() else 0.0
    except ValueError:
        return 0.0


def sampling_probabilities(
    top_logprobs: list[float],
    *,
    top_p: float = TOP_P,
    top_k: int = CANDIDATE_K,
    temperature: float | None = None,
) -> list[float]:
    """``p_t`` over ``C_t`` from the engine's returned top-k log-probabilities.

    By default ``top_logprobs`` are the sampler's own values, already temperature-scaled
    (sglang computes them as ``log_softmax(logits / T)``), sorted descending -- so the
    distribution is ``softmax`` of the values as given. When the engine is run with
    ``SGLANG_RETURN_ORIGINAL_LOGPROB=1`` they are pre-temperature instead, and
    ``temperature`` (or ``ALFWORLD_V2_HSI_SAMPLER_TEMP``) divides them back into the
    sampling convention before the softmax. ``top_k``/``top_p`` then reproduce the
    sampler's nucleus and the result re-normalises to sum to 1 (gate #1), with the
    support equal to the candidate set.
    """
    if not top_logprobs:
        return []
    if temperature is None:
        temperature = _sampler_logprob_temperature()
    k = max(1, min(int(top_k), len(top_logprobs)))
    lp = [float(v) for v in top_logprobs[:k]]
    if temperature and temperature > 0.0:
        # Division by a positive scale preserves the descending order, so the top-k
        # set the sampler used is unchanged.
        lp = [v / temperature for v in lp]
    top = max(lp)
    weights = [math.exp(v - top) for v in lp]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        # Degenerate/NaN log-probs: fall back to the uniform draw over what the
        # engine returned rather than inventing a distribution.
        return [1.0 / k] * k
    probs = [w / total for w in weights]

    if 0.0 < float(top_p) < 1.0:
        cumulative = 0.0
        keep = len(probs)
        for index, value in enumerate(probs):
            cumulative += value
            if cumulative >= float(top_p):
                keep = index + 1
                break
        probs = probs[:keep]

    renorm = sum(probs)
    if renorm <= 0.0:
        return [1.0 / len(probs)] * len(probs)
    return [v / renorm for v in probs]


# ---------------------------------------------------------------------------
# Integer payload codec (the only channel slime forwards is an int list)
# ---------------------------------------------------------------------------


def encode_candidates(per_position: list[tuple[list[int], list[float]]]) -> list[int]:
    """Flat integer payload for one row's candidate table.

    Layout::

        [rlen,  K_0, id_0..K_0-1, q_0..q_0-1,  K_1, id..., q..., ...]

    ``K_j == 0`` marks a position with no candidate set (only reachable for rows the
    scorer is not asked to score). ``q = max(1, round(p * PROB_SCALE))``.
    """
    payload: list[int] = [len(per_position)]
    for ids, probs in per_position:
        if not ids:
            payload.append(0)
            continue
        payload.append(len(ids))
        payload.extend(int(i) for i in ids)
        payload.extend(max(1, int(round(float(p) * PROB_SCALE))) for p in probs)
    return payload


def decode_candidates(payload) -> list[tuple[list[int], list[float]]]:
    """Inverse of :func:`encode_candidates`; probabilities re-normalised to sum 1."""
    if payload is None:
        return []
    values = [int(v) for v in payload]
    if not values:
        return []
    rlen = values[0]
    cursor = 1
    out: list[tuple[list[int], list[float]]] = []
    for _ in range(rlen):
        if cursor >= len(values):
            break
        k = values[cursor]
        cursor += 1
        if k <= 0:
            out.append(([], []))
            continue
        ids = values[cursor : cursor + k]
        cursor += k
        raw = values[cursor : cursor + k]
        cursor += k
        total = float(sum(raw))
        probs = [v / total for v in raw] if total > 0 else [1.0 / k] * k
        out.append((ids, probs))
    return out
