"""Robust ``<think>``/``<action>`` parsing for the Qwen3 ALFWorld agent.

The parser is deliberately total: it never raises on model output, and every
failure mode is reported as an explicit :class:`ParseStatus`. A malformed turn
costs one rollout step, never a crashed rollout.

Admissibility is checked by *normalised exact match* against the environment's
admissible-command list. Fuzzy/prefix matching is intentionally not used: the
environment only accepts a command verbatim, so a "close enough" match would
silently disagree with what the env actually executed.
"""

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ParseStatus",
    "ParsedAction",
    "extract_think",
    "extract_action",
    "normalize_action",
    "parse_turn_output",
]

#: Matches the LAST ``<action>...</action>`` block. Models occasionally emit an
#: aborted/empty tag before the real one, so we prefer the last non-empty match.
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)

#: Same idea for the reasoning block.
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)

#: Fallback: a bare ``Action:`` line, accepted only when no tag is present.
_ACTION_LINE_RE = re.compile(r"^\s*action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

_WHITESPACE_RE = re.compile(r"\s+")

#: Characters stripped from both ends of a candidate command.
_STRIP_CHARS = " \t\r\n`\"'*."


class ParseStatus(str, Enum):
    """Outcome of parsing one model turn. Never raises; always one of these."""

    OK = "ok"
    #: No ``<action>`` tag and no ``Action:`` line anywhere in the response.
    NO_ACTION = "no_action"
    #: An ``<action>`` tag exists but its body is empty/whitespace.
    EMPTY_ACTION = "empty_action"
    #: A syntactically valid command that is not in the admissible set.
    NOT_ADMISSIBLE = "not_admissible"
    #: The response hit the token cap mid-action, so the tag never closed.
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class ParsedAction:
    """Result of parsing one model turn.

    ``action`` is the normalised command when one was recovered (even when it was
    not admissible) and ``None`` otherwise. ``raw_action`` preserves the model's
    literal text for diagnostics.
    """

    status: ParseStatus
    action: str | None
    raw_action: str | None
    think_text: str
    has_think: bool

    @property
    def is_valid(self) -> bool:
        return self.status is ParseStatus.OK


def normalize_action(text: str) -> str:
    """Normalise a candidate command for comparison with the admissible set.

    Lowercases, collapses internal whitespace and trims surrounding punctuation.
    ALFWorld's admissible commands are already lowercase single-spaced strings,
    so this only absorbs the formatting noise models introduce.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text.strip().strip(_STRIP_CHARS)).lower()


def extract_think(text: str) -> tuple[str, bool]:
    """Return ``(think_text, has_think)``; missing block yields ``("", False)``."""
    matches = [m for m in _THINK_RE.findall(text or "") if m.strip()]
    if not matches:
        return "", False
    return matches[-1].strip(), True


def extract_action(text: str, *, truncated: bool = False) -> tuple[str | None, bool, bool]:
    """Return ``(raw_action, tag_present, closed)``.

    ``tag_present`` distinguishes "the model emitted an empty ``<action></action>``"
    from "the model emitted no action tag at all" -- two different failure modes
    that deserve different statuses and different diagnostics.

    ``closed`` reports whether a *closed* action tag was found. When ``truncated``
    and the tag never closed, an unclosed ``<action>`` opening is still inspected
    so a turn cut off mid-command reports TRUNCATED rather than NO_ACTION.
    """
    body = text or ""

    closed_matches = _ACTION_RE.findall(body)
    non_empty = [m for m in closed_matches if m.strip()]
    if non_empty:
        return non_empty[-1].strip(), True, True
    if closed_matches:
        # A tag exists but every occurrence is empty/whitespace.
        return None, True, True

    if truncated:
        idx = body.lower().rfind("<action>")
        if idx != -1:
            tail = body[idx + len("<action>") :].strip()
            return (tail or None), True, False

    line = _ACTION_LINE_RE.search(body)
    if line:
        return line.group(1).strip(), True, True
    return None, False, False


def parse_turn_output(
    text: str,
    admissible_actions: list[str],
    *,
    truncated: bool = False,
) -> ParsedAction:
    """Parse one model turn into a :class:`ParsedAction`.

    ``admissible_actions`` is the environment's command list for the *current*
    observation. An empty list is treated as "no constraint information", so a
    syntactically valid command is reported OK rather than NOT_ADMISSIBLE --
    otherwise a degenerate env state would make every turn look invalid.
    """
    think_text, has_think = extract_think(text)
    raw_action, tag_present, closed = extract_action(text, truncated=truncated)
    action = normalize_action(raw_action) if raw_action else None

    if not tag_present:
        status = ParseStatus.TRUNCATED if truncated else ParseStatus.NO_ACTION
        return ParsedAction(status, None, None, think_text, has_think)

    if not action:
        # Tag present but empty -> EMPTY_ACTION; a missing tag is NO_ACTION above.
        status = ParseStatus.TRUNCATED if (truncated and not closed) else ParseStatus.EMPTY_ACTION
        return ParsedAction(status, None, raw_action, think_text, has_think)

    if truncated and not closed:
        # The tag never closed: the command we recovered is a prefix, not a command.
        return ParsedAction(ParseStatus.TRUNCATED, action, raw_action, think_text, has_think)

    admissible = {normalize_action(a) for a in (admissible_actions or [])}
    if admissible and action not in admissible:
        return ParsedAction(ParseStatus.NOT_ADMISSIBLE, action, raw_action, think_text, has_think)

    return ParsedAction(ParseStatus.OK, action, raw_action, think_text, has_think)
