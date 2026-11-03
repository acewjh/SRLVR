"""v2 configuration -- every knob in one place, env-overridable.

All methods use complete observation/action history and an </action> stop.
Thinking style, response cap, and context are selected by the shared launcher;
the 1.7B campaign defaults to native thinking without a separate thinking-token budget, 4096 response
tokens, and a 32768-token context. Rows
remain per-turn; all rows in a rollout are accumulated into one optimizer
update. Historical curriculum sampling is implemented in curriculum.py.

Reused from v1 unchanged (imported, never modified): ``paths``, ``env_adapter``,
``action_parser``, ``trajectory``, ``reward``, ``eval``.
"""

import os

__all__ = [
    "HISTORY_LENGTH",
    "MAX_TURNS",
    "MAX_RESPONSE_TOKENS",
    "HPI_FUTURE_GAP",
    "STOP_STRING",
    "GIGPO_DEFAULT_GAMMA",
    "GIGPO_DEFAULT_MODE",
    "GIGPO_DEFAULT_STEP_WEIGHT",
    "get_int",
    "get_float",
    "candidate_centered",
    "gigpo_enabled",
    "gigpo_gamma",
    "gigpo_mode",
    "gigpo_step_weight",
    "HSI_CANDIDATE_K",
    "CAND_K",
    "TEACHER_TEMP",
]


def get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def candidate_centered() -> bool:
    """True for the Candidate-Centered HSI arm (``ALFWORLD_V2_CC=1``).

    This is the only structural switch of the new method: one rollout + one
    future-conditioned frozen-teacher forward + one policy train step, with the
    hindsight credit centred over the rollout's own sampling candidates. When off,
    every legacy code path (matched teacher, trajectory standardisation) is
    bit-for-bit unchanged.
    """
    return os.environ.get("ALFWORLD_V2_CC", "0") == "1"


#: Number of sampling candidates ``C_t`` requested from the rollout engine per
#: generated token. Must be >= ``--rollout-top-k`` (20) so the truncated top-k
#: sampling set is fully contained; kept equal to it by default so the storage is
#: exactly the sampler's candidate set, O(T*K).
#:
#: ``HSI_CANDIDATE_K`` is the name the candidate-centered rollout codes read;
#: ``CAND_K`` is kept as an alias for the same value so both spellings agree.
HSI_CANDIDATE_K = int(os.environ.get("ALFWORLD_V2_HSI_CANDIDATE_K") or 20)
CAND_K = HSI_CANDIDATE_K

#: Teacher score temperature ``T_teacher`` in ``a_t(v) = z_F,t(v) / T_teacher``.
#: The frozen teacher is the initial policy evaluated at its native scale, so 1.0;
#: exposed so the sensitivity can be measured without a code change.
TEACHER_TEMP = get_float("ALFWORLD_V2_TEACHER_TEMP", 1.0)

#: Zero denotes complete history; retained for diagnostic callers.
HISTORY_LENGTH = 0  # All previous turns, without a sliding window.

# ---- GiGPO (Group-in-Group Policy Optimization) ------------------------------
# GiGPO adds an anchor-state step-level advantage to GRPO's episode advantage:
# turns from the same task group that stand in the same environment observation
# form a group, and their discounted returns-to-go are standardised inside it.
# Defaults follow the official implementation (verl-agent, arXiv:2505.10978):
# gamma=0.95, step_advantage_w=1.0, mean+std normalisation.
GIGPO_DEFAULT_GAMMA = 0.95
GIGPO_DEFAULT_STEP_WEIGHT = 1.0
GIGPO_DEFAULT_MODE = "mean_std_norm"


def gigpo_enabled() -> bool:
    """True for the GiGPO arm (``ALFWORLD_V2_GIGPO=1``)."""
    return os.environ.get("ALFWORLD_V2_GIGPO", "0") == "1"


def gigpo_gamma() -> float:
    """Discount factor of the per-turn return-to-go (``ALFWORLD_V2_GIGPO_GAMMA``)."""
    return get_float("ALFWORLD_V2_GIGPO_GAMMA", GIGPO_DEFAULT_GAMMA)


def gigpo_step_weight() -> float:
    """Weight ``w`` in ``A_{i,t} = A^E_i + w * A^S_{i,t}``."""
    return get_float("ALFWORLD_V2_GIGPO_STEP_W", GIGPO_DEFAULT_STEP_WEIGHT)


def gigpo_mode() -> str:
    """Step normalisation: ``mean_std_norm`` (default) or ``mean_norm``.

    Mirrors the official ``algorithm.gigpo.mode`` switch. ``mean_std_norm``
    standardises the anchor group (mean and sample std); ``mean_norm`` only
    subtracts the anchor mean.
    """
    return os.environ.get("ALFWORLD_V2_GIGPO_MODE", GIGPO_DEFAULT_MODE).strip().lower()


#: Turn budget. No longer bounded by the lm_head, only by the task. 50 because at
#: 20 the `truncated_ratio` was 0.87-0.99 for every model size -- episodes were
#: ending by running out of turns, not by failing the task, which both suppressed
#: the success rate and starved GRPO of reward variance.
MAX_TURNS = int(os.environ.get("ALFWORLD_V2_MAX_TURNS") or 50)

#: Per-turn generation cap. Generation stops at ``</action>`` well before this in
#: the common case; the cap only bounds a runaway turn.
MAX_RESPONSE_TOKENS = int(os.environ.get("ALFWORLD_V2_MAX_RESPONSE") or 4096)

#: Transition gap. ``transition i = (action_i, observation_{i+1})``; scoring turn
#: ``t`` with gap 1 means the Full view may NOT contain transition ``t`` (i.e. the
#: immediate result of ``a_t``), so the future trace starts at ``t+1``. The
#: terminal outcome is present and identical in both views.
HPI_FUTURE_GAP = 1

#: Stop generation here. ``no_stop_trim`` is forced on so the tag survives into
#: the output text and the parser sees a complete ``</action>``.
STOP_STRING = "</action>"

#: Episode id is packed into ``turn_ids`` as ``episode_id * TURN_ID_STRIDE + turn``.
#: ``turn_ids`` is the only per-token *integer* channel slime already whitelists
#: for the DP split and forwards into the training batch, so it is the one place a
#: per-turn identity can ride without touching the framework. See loss.py/scorer.py.
TURN_ID_STRIDE = 10_000


def get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)
