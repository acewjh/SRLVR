"""Conservative, explicit repair of an ALFWorld action before ``env.step``.

The repair is deliberately narrower than fuzzy matching:

* it may translate a command *form* (``examine``/``look at`` -> ``look``,
  ``move X to Y`` -> ``put X in/on Y``, ...),
* it may navigate to the receptacle the model itself named
  (``open fridge 1`` -> ``go to fridge 1``) when that navigation is in the
  current admissible set,
* it never chooses an object or a different receptacle, and
* it refuses multi-command or prose output.

Every returned action is an exact member of the environment's admissible set
(after normalisation). Ambiguity is reported, never resolved by guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .v1.action_parser import normalize_action

_WHITESPACE = re.compile(r"\s+")
_ARTICLES = re.compile(r"\bthe\b\s*")
# Any of these means the model produced multiple commands or markup; such text
# must never be mined for a command.
_UNSAFE = re.compile(r"[;\n\r<>`]")

_LOOK = re.compile(
    r"^(?:examine|inspect|check|view|survey|peer at|search|scan)\b|^look\b"
)
_INVENTORY = re.compile(r"^(?:check|show|view) (?:the )?inventory$|^inventory$")
_GO = re.compile(r"^(?:go|navigate|walk|head) (?:to )?(.+)$")
_MOVE_TO = re.compile(r"^(?:move|place|carry|bring) (.+?) to (.+)$")
_MOVE_IN = re.compile(r"^(?:move|place|put) (.+?) (?:in|on|into|onto) (.+)$")
_GRAB = re.compile(r"^(?:grab|pick up|pick|collect|get) (.+?) from (.+)$")
_WASH = re.compile(r"^(?:wash|rinse) (.+?) with (.+)$")
_MICRO = re.compile(r"^(?:microwave|heat) (.+)$")
_COOL = re.compile(r"^(?:chill|refrigerate|cool) (.+)$")

# (pattern, receptacle group) for the navigate-before-acting repair.
_PRECONDITION = (
    (re.compile(r"^open (.+)$"), 1),
    (re.compile(r"^close (.+)$"), 1),
    (re.compile(r"^take (.+?) from (.+)$"), 2),
    (re.compile(r"^put (.+?) (?:in|on) (.+)$"), 2),
    (re.compile(r"^clean (.+?) with (.+)$"), 2),
    (re.compile(r"^(?:heat|cool) (.+?) with (.+)$"), 2),
)


@dataclass(frozen=True)
class RepairResult:
    """One repair decision.

    ``status`` is ``"repaired"``, ``"ambiguous"`` or ``"unrepaired"``.
    ``candidates`` is populated for ambiguity so the caller can log what was
    rejected instead of silently choosing one.
    """

    status: str
    action: str | None = None
    rule: str | None = None
    candidates: tuple[str, ...] = ()


def _canon(text: str) -> str:
    value = normalize_action(text)
    value = _ARTICLES.sub("", value)
    value = value.replace(" onto ", " on ").replace(" into ", " in ")
    return _WHITESPACE.sub(" ", value).strip()


def _pick(candidates: list[tuple[str, str]]) -> RepairResult | None:
    """Return a repair only when all surviving candidates name one action."""
    targets = {action for _, action in candidates}
    if len(targets) == 1:
        target = next(iter(targets))
        rule = next(rule for rule, action in candidates if action == target)
        return RepairResult("repaired", target, rule)
    if len(targets) > 1:
        return RepairResult("ambiguous", None, None, tuple(sorted(targets)))
    return None


def _form_candidates(command: str, admissible: set[str]) -> list[tuple[str, str]]:
    """Command-form translations. No object or receptacle is invented."""
    out: list[tuple[str, str]] = []
    if _INVENTORY.match(command):
        out.append(("inventory", "inventory"))
    elif _LOOK.match(command):
        # Covers examine/inspect/check/look at|in|on|under, and look with an
        # object argument. The bare `look` action is the only legal ALFWorld
        # look command.
        out.append(("look_family", "look"))
    match = _GO.match(command)
    if match and not command.startswith("go to "):
        out.append(("go_to", f"go to {_canon(match.group(1))}"))
    match = re.match(r"^move to (.+)$", command)
    if match:
        out.append(("move_to_go", f"go to {_canon(match.group(1))}"))
    match = _MOVE_TO.match(command)
    if match:
        obj, receptacle = _canon(match.group(1)), _canon(match.group(2))
        out.extend([
            ("move_to_put_in", f"put {obj} in {receptacle}"),
            ("move_to_put_on", f"put {obj} on {receptacle}"),
        ])
    match = _MOVE_IN.match(command)
    if match:
        obj, receptacle = _canon(match.group(1)), _canon(match.group(2))
        out.extend([
            ("move_in_put_in", f"put {obj} in {receptacle}"),
            ("move_in_put_on", f"put {obj} on {receptacle}"),
        ])
    match = _GRAB.match(command)
    if match:
        out.append(("grab_to_take",
                    f"take {_canon(match.group(1))} from {_canon(match.group(2))}"))
    match = _WASH.match(command)
    if match:
        out.append(("wash_to_clean",
                    f"clean {_canon(match.group(1))} with {_canon(match.group(2))}"))
    for pattern, rule, verb in ((_MICRO, "heat_unique_with", "heat"),
                                (_COOL, "cool_unique_with", "cool")):
        match = pattern.match(command)
        if match:
            obj = _canon(match.group(1))
            hits = [a for a in admissible if a.startswith(f"{verb} {obj} with ")]
            if len(hits) == 1:
                out.append((rule, hits[0]))
    return [(rule, action) for rule, action in out if action != command]


def _precondition_candidates(command: str) -> list[tuple[str, str]]:
    for pattern, group in _PRECONDITION:
        match = pattern.match(command)
        if match:
            return [("navigate_precondition", f"go to {_canon(match.group(group))}")]
    return []


def repair_action(
    raw_action: str | None,
    admissible_actions: list[str] | None,
    *,
    allow_precondition: bool = True,
) -> RepairResult:
    """Return the safe repair of ``raw_action`` or an explicit non-repair."""
    if not raw_action or not admissible_actions or _UNSAFE.search(raw_action):
        return RepairResult("unrepaired")
    admissible = {normalize_action(value) for value in admissible_actions}
    if not admissible:
        return RepairResult("unrepaired")
    command = _canon(raw_action)
    if not command:
        return RepairResult("unrepaired")
    if command in admissible:
        return RepairResult("repaired", command, "format")

    picked = _pick([
        (rule, action)
        for rule, action in _form_candidates(command, admissible)
        if normalize_action(action) in admissible
    ])
    if picked is not None:
        return picked

    if allow_precondition:
        picked = _pick([
            (rule, action)
            for rule, action in _precondition_candidates(command)
            if normalize_action(action) in admissible
        ])
        if picked is not None:
            return picked

    return RepairResult("unrepaired")
