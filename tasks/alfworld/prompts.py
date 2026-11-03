"""Action-only ALFWorld prompts shared by rollout and hindsight scoring."""
import os
from dataclasses import dataclass
from opsd_core.token_cache import encode_prompt

SYSTEM_PROMPT = """You are an agent operating in the ALFWorld text environment.
Follow the task using the observations and the complete interaction history.
Output exactly one environment command inside <action>...</action>, then stop.
Do not output reasoning or commentary.
Commands include: look; inventory; go to RECEP; open RECEP; close RECEP;
take OBJECT from RECEP; put OBJECT in/on RECEP; clean OBJECT with RECEP;
heat OBJECT with RECEP; cool OBJECT with RECEP; use OBJECT.
Use the exact object and receptacle names and numbers from observations."""

@dataclass(frozen=True)
class TaskView:
    task_description: str
    current_observation: str
    admissible_actions: list[str]  # Rendered only when ALFWORLD_SHOW_ADMISSIBLE=1.
    history: list[tuple[str, str]]

def show_admissible() -> bool:
    """Whether the current state's admissible commands are rendered in the prompt.

    Default ON (2026-10-27): on the train split the base policy emits 43-56%
    inadmissible commands without the list, which left G=1 SRLVR with almost no
    complete trajectories. Rendering the list raised rollout-0 success on
    ``pick_and_place_simple`` from 0/32 to 19-20/32; the extra ~384 prompt tokens
    per turn are cheap under prefix caching. Set ``ALFWORLD_SHOW_ADMISSIBLE=0``
    to reproduce the historical (list-hidden) protocol.
    """
    return os.environ.get("ALFWORLD_SHOW_ADMISSIBLE", "1").lower() not in ("0", "false", "no")

def build_turn_user_message(view: TaskView) -> str:
    lines = [f"Task: {view.task_description.strip()}", ""]
    if view.history:
        lines.append("Complete history (oldest first):")
        for observation, action in view.history:
            lines.extend([f"Observation: {observation.strip()}", f"Action: {action}"])
        lines.append("")
    lines.extend(["Current observation:", view.current_observation.strip(), ""])
    if show_admissible() and view.admissible_actions:
        lines.append("Admissible actions in the current state (choose exactly one):")
        lines.extend(f"- {action}" for action in view.admissible_actions)
        lines.append("")
    lines.extend(["Output one <action>...</action> command."])
    return '\n'.join(lines)

def build_system_prompt() -> str:
    system = SYSTEM_PROMPT
    style = os.environ.get("ALFWORLD_THINKING_STYLE", "brief")
    if style in ("brief", "legacy"):
        system = system.replace(
            "Output exactly one environment command inside <action>...</action>, then stop.\nDo not output reasoning or commentary.",
            "Reason about the current situation inside <think>...</think>.\n"
            "Then output exactly one environment command inside <action>...</action>, then stop.")
        if style == "brief":
            raw_budget = os.environ.get("ALFWORLD_THINKING_TOKENS", "")
            if raw_budget:
                budget = int(raw_budget)
                if budget < 1:
                    raise ValueError("ALFWORLD_THINKING_TOKENS must be positive")
                system += f"\n\nUse up to {budget} tokens inside <think> to plan the next action. "
            system += (
                "\n\n"
                "Consider the observations, relevant history, and alternative actions as needed. "
                "Then close </think> and immediately emit one <action> command. Use only the "
                "command forms shown above and exact object names and numbers in the observation. "
                "Use `look` or `inventory` without adding an object argument; do not invent "
                "commands such as `look at OBJECT`. Emit no text after </action>.")
    elif style != "off":
        raise ValueError(f"unknown ALFWORLD_THINKING_STYLE: {style!r}")
    if show_admissible():
        system += ("\n\nThe observation lists every command that is valid in the current state. "
                   "Emit exactly one of those commands verbatim; never emit a command that is "
                   "not on the list.")
    return system

def build_turn_messages(view: TaskView) -> list[dict[str, str]]:
    return [{"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_turn_user_message(view)}]

def render_chat(tokenizer, messages, *, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=os.environ.get("ALFWORLD_THINKING_STYLE", "brief") != "off"
    )

def build_turn_prompt_ids(tokenizer, view: TaskView) -> list[int]:
    return encode_prompt(tokenizer, render_chat(tokenizer, build_turn_messages(view)))
