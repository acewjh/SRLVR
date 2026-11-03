"""Deterministic command-format recovery shared by V2 training and evaluation.

The sampled text/tokens stay untouched. Only the environment-facing action is
normalized or repaired; the admissible set validates every decision.

``ALFWORLD_ACTION_HARNESS`` selects the harness:

* ``command-normalization-v1`` -- historical form recovery only.
* ``command-normalization-v2-form`` (default) -- v1 plus explicit form/synonym
  repairs (``examine`` -> ``look``, ``move X to Y`` -> ``put X in/on Y``, ...).
* ``command-normalization-v2`` -- the above plus navigate-before-act:
  ``open fridge 1`` becomes ``go to fridge 1`` when that navigation is in the
  current admissible set. Opt-in because it executes an action the model did
  not sample.

The repair never invents an object or a different receptacle and never mines a
command out of multi-command/prose output. Ambiguous candidates stay invalid.
"""
import os
import re
from dataclasses import dataclass

from .action_repair import repair_action
from .v1.action_parser import ParseStatus, ParsedAction, normalize_action, parse_turn_output

_MODES = {
    "v1": "v1",
    "command-normalization-v1": "v1",
    "v2-form": "v2-form",
    "command-normalization-v2-form": "v2-form",
    "form": "v2-form",
    "v2": "v2",
    "command-normalization-v2": "v2",
    "repair": "v2",
}
_VERSIONS = {
    "v1": "command-normalization-v1",
    "v2-form": "command-normalization-v2-form",
    "v2": "command-normalization-v2",
}


def resolve_harness_mode(value: str | None = None) -> str:
    """Resolve an env/argument value to ``v1``, ``v2-form`` or ``v2``."""
    raw = (value if value is not None else os.environ.get(
        "ALFWORLD_ACTION_HARNESS", "command-normalization-v2-form")).strip().lower()
    if not raw:
        return "v2-form"
    try:
        return _MODES[raw]
    except KeyError:
        raise ValueError(
            "ALFWORLD_ACTION_HARNESS must be one of "
            "command-normalization-v1|command-normalization-v2-form|"
            f"command-normalization-v2 (got {raw!r})"
        ) from None


_DEFAULT_MODE = resolve_harness_mode()
REPAIR_MODE = {"v1": None, "v2-form": "form", "v2": "full"}[_DEFAULT_MODE]
HARNESS_VERSION = _VERSIONS[_DEFAULT_MODE]

_END_MARKER = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")
_COMMAND = re.compile(
    r"(?:look|inventory|help|go to .+|open .+|close .+|take .+ from .+|"
    r"put .+ in/on .+|clean .+ with .+|heat .+ with .+|cool .+ with .+|use .+)"
)


@dataclass(frozen=True)
class HarnessAction:
    parsed: ParsedAction
    source: str


def _repair(
    parsed: ParsedAction,
    admissible_actions: list[str],
    mode: str,
) -> HarnessAction | None:
    if mode == "v1":
        return None
    repaired = repair_action(
        parsed.action,
        admissible_actions,
        allow_precondition=(mode == "v2"),
    )
    if repaired.status == "repaired":
        return HarnessAction(
            ParsedAction(
                ParseStatus.OK,
                repaired.action,
                parsed.raw_action,
                parsed.think_text,
                parsed.has_think,
            ),
            f"repaired:{repaired.rule}",
        )
    if repaired.status == "ambiguous":
        return HarnessAction(parsed, "ambiguous")
    return None


def parse_action(
    text: str,
    admissible_actions: list[str],
    *,
    truncated: bool = False,
    repair_mode: str | None = None,
) -> HarnessAction:
    """Recover, repair and validate the environment-facing action.

    ``repair_mode`` is a test/operator override; ``None`` uses the value
    resolved from ``ALFWORLD_ACTION_HARNESS`` at import time.
    """
    mode = _DEFAULT_MODE if repair_mode is None else resolve_harness_mode(repair_mode)
    parsed = parse_turn_output(text, admissible_actions, truncated=truncated)

    if parsed.status is ParseStatus.NOT_ADMISSIBLE:
        return _repair(parsed, admissible_actions, mode) or HarnessAction(parsed, "structured")
    if parsed.status is not ParseStatus.NO_ACTION:
        return HarnessAction(parsed, "structured")

    # Never interpret a partial command or choose an action without environment
    # validation. Existing structured-output parsing retains its original rules.
    if truncated or not admissible_actions:
        return HarnessAction(parsed, "unrecovered")
    body = text.strip()
    while (match := _END_MARKER.search(body)) is not None:
        body = body[:match.start()].rstrip()
    # A single bare command only: no prose extraction, multiple commands,
    # malformed tags, code fences, or object-name guessing.
    if any(char in body for char in ("\n", "\r", ";", "<", ">", "`")):
        return HarnessAction(parsed, "unrecovered")
    action = normalize_action(body)
    if not _COMMAND.fullmatch(action):
        return HarnessAction(parsed, "unrecovered")
    admissible = {normalize_action(value) for value in admissible_actions}
    if action in admissible:
        return HarnessAction(
            ParsedAction(ParseStatus.OK, action, body, parsed.think_text, parsed.has_think),
            "bare_command",
        )
    repaired = _repair(
        ParsedAction(
            ParseStatus.NOT_ADMISSIBLE, action, body, parsed.think_text, parsed.has_think
        ),
        admissible_actions,
        mode,
    )
    if repaired is not None:
        return HarnessAction(repaired.parsed, f"{repaired.source}:bare")
    return HarnessAction(
        ParsedAction(
            ParseStatus.NOT_ADMISSIBLE, action, body, parsed.think_text, parsed.has_think
        ),
        "bare_command",
    )
