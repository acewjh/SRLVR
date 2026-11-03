"""Prompt construction for the Qwen3-1.7B ALFWorld agent.

Two rules drive the whole file:

1. **Causal history only.** The transcript a turn sees is
   ``task, o_0, a_0, o_1, a_1, ... o_t`` plus the current admissible actions.
   Past ``<think>`` blocks are never rendered -- they exist for diagnostics, not
   as context. See ``trajectory.FORBIDDEN_HISTORY_MARKERS``.

2. **One live observation.** Observations are cumulative: ``o_t`` already
   contains the room description plus everything that changed. Replaying every
   historic observation verbatim would blow the context window, so the transcript
   keeps the *latest* observation and the *full* action history (which is short).
   This matches the standard ALFWorld ReAct recipe and is what makes
   ``--alfworld-max-turns`` usable at 30+ turns.

The system prompt is identical across all four experiment arms; the only
per-arm difference is the privileged block appended by ``privilege.py``.
"""

from dataclasses import dataclass

__all__ = [
    "SYSTEM_PROMPT",
    "ADMISSIBLE_ACTIONS_HEADER",
    "TaskView",
    "build_history_entries",
    "build_system_message",
    "build_turn_user_message",
    "build_turn_messages",
    "render_chat",
]

SYSTEM_PROMPT = """You are an expert agent operating in the ALFWorld text environment.

You are given a task, the history of your actions and observations so far, and a list of admissible actions.

At each step you must:
1. Think about the current situation inside <think>...</think>.
2. Then choose exactly ONE action from the admissible actions and put it inside <action>...</action>.

Rules:
- The action MUST be copied verbatim from the admissible action list. Do not
  rephrase it, do not add words, and do not invent a similar-sounding command.
  If you want to inspect something, use an action that is already in the list
  (e.g. prefer `examine ...` only when `examine ...` is listed).
- Emit exactly one <action>...</action> block per turn.
- Do not invent actions that are not in the list.
- Reason about the observations and history as needed, then commit to an action."""

ADMISSIBLE_ACTIONS_HEADER = "Admissible actions:"

#: Emitted when the environment exposes no admissible commands (degenerate state).
_NO_ACTIONS_PLACEHOLDER = "(none available)"

#: How many past actions to keep in the transcript. The full list lives on the
#: Trajectory record for diagnostics; the prompt only needs enough to disambiguate
#: the current situation, and capping it bounds prompt growth on long episodes
#: (which the O(T^2) per-turn-segment layout makes load-bearing).
DEFAULT_ACTION_HISTORY = 8


@dataclass(frozen=True)
class TaskView:
    """Everything the student is allowed to condition on at one turn."""

    task_description: str
    current_observation: str
    admissible_actions: list[str]
    action_history: list[str]


def build_system_message() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def build_history_entries(trajectory, upto_turn: int, limit: int | None = None) -> list[str]:
    """Transcript lines for turn ``upto_turn`` -- actions only, never reasoning.

    Returned as plain strings so the no-reasoning assertion can run on the exact
    text that will be rendered.
    """
    turns = trajectory.turns
    if not 0 <= upto_turn < len(turns):
        raise IndexError(f"upto_turn {upto_turn} out of range for {len(turns)} turns")

    if limit is None:
        from . import config

        limit = config.action_history_limit()

    actions = [t.action_text for t in turns[:upto_turn]]
    recent = actions[-limit:] if limit else []

    entries: list[str] = [f"Task: {trajectory.task_description}"]
    if recent:
        skipped = len(actions) - len(recent)
        if skipped > 0:
            entries.append(f"({skipped} earlier actions omitted)")
        entries.append("Previous actions:")
        entries.extend(f"  {idx + skipped + 1}. {action}" for idx, action in enumerate(recent))
    return entries


def _render_admissible(actions: list[str]) -> list[str]:
    if not actions:
        return [ADMISSIBLE_ACTIONS_HEADER, f"  {_NO_ACTIONS_PLACEHOLDER}"]
    return [ADMISSIBLE_ACTIONS_HEADER, *(f"  - {action}" for action in actions)]


def build_turn_user_message(view: TaskView) -> str:
    """The student-visible user turn for one step.

    Byte-identical across all four experiment arms -- the privileged block is
    appended by ``privilege.build_privileged_user_message`` on top of this.
    """
    lines: list[str] = []
    lines.append(f"Task: {view.task_description}")
    lines.append("")
    if view.action_history:
        lines.append("Previous actions:")
        lines.extend(f"  {idx + 1}. {action}" for idx, action in enumerate(view.action_history))
        lines.append("")
    lines.append("Current observation:")
    lines.append(view.current_observation.strip())
    lines.append("")
    lines.extend(_render_admissible(view.admissible_actions))
    lines.append("")
    lines.append("Choose exactly one admissible action.")
    return "\n".join(lines)


def build_turn_messages(view: TaskView) -> list[dict[str, str]]:
    return [build_system_message(), {"role": "user", "content": build_turn_user_message(view)}]


def render_chat(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
    """Chat-template ``messages`` with thinking explicitly enabled.

    ``enable_thinking=True`` is the Qwen3 contract used by every arm. It is
    passed explicitly rather than relying on the template default so a template
    revision cannot silently flip the whole harness to non-thinking.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def build_turn_prompt_ids(tokenizer, view: TaskView) -> list[int]:
    """Token ids for one student turn (system + user + assistant generation prompt)."""
    text = render_chat(tokenizer, build_turn_messages(view), add_generation_prompt=True)
    return list(tokenizer.encode(text, add_special_tokens=False))
