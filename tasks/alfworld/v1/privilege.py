"""Hindsight Privileged Information (HPI) for ALFWorld turns.

What the teacher sees
---------------------
At turn ``t`` the student conditions on ``h_t`` = task + causal action history +
current observation + current admissible actions, and generates
``y_t = think_t + action_t``.

The privileged teacher conditions on ``h_t`` **plus** a hindsight block ``P_t``
and is then teacher-forced on the SAME ``y_t`` token ids. The teacher never
samples: it only re-scores the student's own tokens.

``P_t`` contains, by mode:

===============  =========================================================
mode             contents
===============  =========================================================
``none``         no block at all (this is the control / "blind" view)
``outcome_only`` the terminal episode outcome R
``future_outcome`` R + the delayed future interaction trace
===============  =========================================================

Index convention (exact -- off-by-one bugs here are silent and fatal)
---------------------------------------------------------------------
* Turns are 0-based, ``t = 0 .. T-1``. Turn ``t`` produced ``think_t + action_t``.
* Transition ``i`` is the pair ``(action_i, observation_{i+1})``: the
  environment's response to the action generated at turn ``i``.
* ``hpi_future_gap = g`` means the privileged trace contains transitions with

      ``i >= t + g``

  * ``g = 2`` (**default**) -> ``t+2, t+3, ...``. Both the current action's own
    result (transition ``t``, whose observation is ``o_{t+1}``) and the
    immediately adjacent transition ``t+1`` are withheld. This is the
    conservative setting required because ``o_{t+1}`` frequently narrates the
    current action verbatim ("You pick up the mug ...").
  * ``g = 1`` -> ``t+1, t+2, ...``: only the current action's own result withheld.
  * ``g = 0`` -> includes transition ``t``, i.e. the current action's result.
    LEAKY; debug/ablation only.

* The terminal outcome R is **always** included, for every mode except ``none``
  and regardless of gap.

What is never included (enforced, not documented-and-hoped)
-----------------------------------------------------------
* Current turn's response -- neither ``think_t`` nor ``action_t``.
* Any future ``<think>`` block or future raw model text. The builder only ever
  reads ``TurnRecord.action_text`` / ``TurnRecord.observation`` /
  ``TurnRecord.next_observation``; ``generated_text`` and ``think_text`` are
  structurally out of reach, and the rendered result is additionally scanned for
  reasoning markers.
"""

import os
from dataclasses import dataclass

from .trajectory import FORBIDDEN_HISTORY_MARKERS, Trajectory

__all__ = [
    "DEFAULT_FUTURE_GAP",
    "MODE_FUTURE_OUTCOME",
    "MODE_NONE",
    "MODE_OUTCOME_ONLY",
    "MODES",
    "PrivilegeConfig",
    "build_hpi_privilege",
    "assert_no_current_response_leak",
    "assert_privilege_isolated",
    "build_privileged_user_message",
    "render_privilege_block",
]

MODE_NONE = "none"
MODE_OUTCOME_ONLY = "outcome_only"
MODE_FUTURE_OUTCOME = "future_outcome"

MODES = (MODE_NONE, MODE_OUTCOME_ONLY, MODE_FUTURE_OUTCOME)

#: Conservative default: withhold the current action's result AND the adjacent
#: transition. See the module docstring for the exact semantics.
DEFAULT_FUTURE_GAP = 2

_PRIVILEGE_OPEN = "<privileged_hindsight>"
_PRIVILEGE_CLOSE = "</privileged_hindsight>"

_DISCLAIMER = (
    "This information is available only during training. It was observed AFTER "
    "the action being evaluated and is not part of the conversation."
)
_NO_FUTURE_PLACEHOLDER = "(no further interaction was observed)"


def default_future_gap() -> int:
    return int(os.environ.get("ALFWORLD_HPI_FUTURE_GAP", DEFAULT_FUTURE_GAP))


@dataclass(frozen=True)
class PrivilegeConfig:
    """Which hindsight the teacher is allowed to see."""

    mode: str = MODE_NONE
    future_gap: int = DEFAULT_FUTURE_GAP
    #: Cap on how many future steps are rendered (None = the whole tail).
    #: Bounds teacher prompt growth on long episodes; the terminal outcome is
    #: always rendered regardless of the cap.
    max_future_steps: int | None = 12

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown privilege mode {self.mode!r}; expected one of {MODES}")
        if self.future_gap < 0:
            raise ValueError(f"future_gap must be >= 0, got {self.future_gap}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_outcome_block(trajectory: Trajectory) -> list[str]:
    return ["Final episode outcome:", "SUCCESS" if trajectory.success else "FAILURE"]


def _render_future_block(trajectory: Trajectory, turn_idx: int, config: PrivilegeConfig) -> list[str]:
    """The delayed future interaction trace for turn ``turn_idx``.

    Window is transitions ``i >= turn_idx + future_gap`` (see module docstring).
    """
    start = turn_idx + config.future_gap
    steps = list(trajectory.iter_tail_transitions(start))
    if config.max_future_steps is not None:
        # Keep the EARLIEST steps of the window: they are the ones closest to the
        # evaluated action and carry the most credit-assignment signal; the
        # terminal outcome is rendered separately and is never truncated away.
        steps = steps[: config.max_future_steps]

    lines = ["Delayed future interaction trace:"]
    if not steps:
        lines.append(_NO_FUTURE_PLACEHOLDER)
        return lines

    for order, (idx, action, _next_obs) in enumerate(steps, start=1):
        lines.append(f"[Future step {order}] (turn {idx})")
        lines.append(f"Observation: {trajectory.observation(idx).strip()}")
        lines.append(f"Executed action: {action}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    return lines


def _render_final_observation(trajectory: Trajectory, config: PrivilegeConfig) -> list[str]:
    """The environment's terminal observation, when one exists.

    This is environment text, not model text. It is included because the last
    observation is where the environment usually states that the task was
    completed, and it lives *after* the future window's last rendered step.
    """
    if config.mode != MODE_FUTURE_OUTCOME:
        return []
    for turn in reversed(trajectory.turns):
        if turn.next_observation:
            return ["Final environment observation:", turn.next_observation.strip()]
    return []


def render_privilege_block(trajectory: Trajectory, turn_idx: int, config: PrivilegeConfig) -> str:
    """Render ``P_t`` (no surrounding whitespace). Empty string for mode ``none``."""
    if config.mode == MODE_NONE:
        return ""

    body: list[str] = [_DISCLAIMER, ""]
    body.extend(_render_outcome_block(trajectory))
    if config.mode == MODE_FUTURE_OUTCOME:
        body.append("")
        body.extend(_render_future_block(trajectory, turn_idx, config))
        final_obs = _render_final_observation(trajectory, config)
        if final_obs:
            body.append("")
            body.extend(final_obs)

    text = _PRIVILEGE_OPEN + "\n" + "\n".join(body) + "\n" + _PRIVILEGE_CLOSE
    _assert_no_reasoning(text)
    return text


def build_hpi_privilege(
    trajectory: Trajectory,
    turn_idx: int,
    mode: str,
    future_gap: int = DEFAULT_FUTURE_GAP,
    *,
    max_future_steps: int | None = 12,
) -> str:
    """Unified privilege builder -- the single entry point used by tests + rollout.

    Returns ``""`` for ``mode="none"``.
    """
    config = PrivilegeConfig(mode=mode, future_gap=future_gap, max_future_steps=max_future_steps)
    if not 0 <= turn_idx < len(trajectory.turns):
        raise IndexError(f"turn_idx {turn_idx} out of range for {len(trajectory.turns)} turns")
    text = render_privilege_block(trajectory, turn_idx, config)
    if text and future_gap >= 1:
        # gap == 0 is the deliberately leaky ablation: it renders transition t,
        # which restates the current action by construction. The self-check is
        # skipped there but `assert_privilege_isolated` still flags it, so a run
        # can never reach a gate believing gap=0 is clean.
        assert_no_current_response_leak(text, trajectory.turns[turn_idx])
    return text


# ---------------------------------------------------------------------------
# Leakage assertions
# ---------------------------------------------------------------------------


def _assert_no_reasoning(text: str) -> None:
    lowered = text.lower()
    for marker in FORBIDDEN_HISTORY_MARKERS:
        if marker in lowered:
            raise AssertionError(
                f"privileged hindsight contains forbidden reasoning marker {marker!r}"
            )


def assert_no_current_response_leak(privilege_text: str, turn) -> None:
    """The privilege for turn ``t`` must not restate turn ``t``'s own response.

    Two distinct checks:

    * the generated response (and, when present, its reasoning) must not appear
      anywhere in the privilege text;
    * the current action must not be presented as an executed future action --
      a future step narrating ``a_t`` would hand the teacher the answer.
    """
    if not privilege_text:
        return

    generated = (turn.generated_text or "").strip()
    if generated and generated in privilege_text:
        raise AssertionError(
            f"privilege for turn {turn.turn_idx} contains the turn's full generated response"
        )

    think = (turn.think_text or "").strip()
    if len(think) >= 8 and think in privilege_text:
        raise AssertionError(
            f"privilege for turn {turn.turn_idx} contains the turn's own reasoning block"
        )

    # The current ACTION is checked by INDEX, not by string.
    #
    # A string check ("does the current action text appear as a future action?")
    # is a false-positive machine in ALFWorld: agents legitimately repeat commands
    # ("look", "go to desk 1"), so a later turn performing the same action would
    # trip it even though nothing about the evaluated turn leaked. That is what
    # killed the first real HPI training run, on a turn whose action the agent
    # later repeated.
    #
    # What actually matters is that the evaluated turn is not rendered as one of
    # the future steps. The window is fixed by ``future_gap`` (see
    # ``assert_privilege_isolated``), so this checks the rendered label directly.
    marker = f"(turn {turn.turn_idx})"
    if marker in privilege_text:
        raise AssertionError(
            f"privilege for turn {turn.turn_idx} renders the evaluated turn itself "
            f"as a future step ({marker!r})"
        )


def assert_privilege_isolated(
    trajectory: Trajectory,
    turn_idx: int,
    privilege_text: str,
    future_gap: int,
) -> None:
    """Full information-isolation gate for one turn.

    Checks (a) no reasoning markers, (b) no current-response leak, and (c) the
    gap contract -- no rendered future step may sit closer than ``future_gap``
    transitions from the evaluated turn.
    """
    _assert_no_reasoning(privilege_text)
    assert_no_current_response_leak(privilege_text, trajectory.turns[turn_idx])

    for idx, _action, _obs in trajectory.iter_tail_transitions(turn_idx + future_gap):
        if idx < turn_idx + future_gap:
            raise AssertionError("future trace contains a transition inside the gap window")
    # And nothing strictly inside the gap may appear at all.
    for idx in range(turn_idx, min(turn_idx + future_gap, len(trajectory.turns))):
        turn = trajectory.turns[idx]
        if turn.action_text and f"(turn {idx})" in privilege_text:
            raise AssertionError(
                f"future trace exposes turn {idx}, which is inside the gap window "
                f"(future_gap={future_gap}, evaluated turn={turn_idx})"
            )


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_privileged_user_message(base_user_message: str, privilege_text: str) -> str:
    """Append ``P_t`` to the student's user turn.

    The block goes at the END (closest to the generation slot) because a causal
    LM can only condition on tokens that precede the ones being scored. Placing
    it after the turn would make it invisible to the teacher's forward pass.

    A ``user`` message is used rather than a ``tool`` message: user turns are
    legal at any position, whereas a tool result must directly follow an
    assistant tool call, which the scaffold does not produce.
    """
    if not privilege_text:
        return base_user_message
    return base_user_message + "\n\n" + privilege_text


def build_privileged_prompt_ids(tokenizer, view, privilege_text: str) -> list[int]:
    """Token ids of the privileged user turn + assistant generation prompt.

    ``view`` is a ``prompts.TaskView``. Returns ids for the teacher's prefix; the
    caller appends the student's raw generated token ids (never re-tokenised).
    """
    from .prompts import build_system_message, build_turn_user_message, render_chat

    content = build_privileged_user_message(build_turn_user_message(view), privilege_text)
    messages = [build_system_message(), {"role": "user", "content": content}]
    text = render_chat(tokenizer, messages, add_generation_prompt=True)
    return list(tokenizer.encode(text, add_special_tokens=False))
