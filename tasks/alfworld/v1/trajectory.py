"""Structured episode records for ALFWorld.

Two contracts matter here and both are enforced, not assumed:

1. **Round-trip fidelity.** ``Trajectory.from_dict(to_dict())`` is lossless, so a
   completed episode can be scored for Future-HPI *after* the rollout finishes
   (the trajectory is serialised to JSONL at episode end and is the only input
   the privilege builder gets in the offline path).

2. **Causal history excludes past reasoning.** ``history_messages`` emits
   task + past actions + past observations only. ``think_text`` is retained on
   the record for diagnostics but never enters any prompt. The invariant is
   asserted, not merely documented -- see ``assert_no_reasoning_in_history``.

Index convention (fixed, used identically by ``privilege.py`` and the tests)
----------------------------------------------------------------------------
* Turns are 0-based: ``t = 0 .. T-1``. Turn ``t`` is the model generation that
  produced ``think_t + action_t``.
* Transition ``i`` is the pair ``(action_i, observation_{i+1})`` -- the
  environment's response to the action generated at turn ``i``.
* While generating ``y_t`` the student has seen ``o_0 .. o_t`` and
  ``a_0 .. a_{t-1}``; the outcome of ``a_t`` (which is ``o_{t+1}``) is NOT yet
  visible.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "FORBIDDEN_HISTORY_MARKERS",
    "TurnRecord",
    "Trajectory",
    "assert_no_reasoning_in_history",
]

#: Any of these appearing in a *history* or *privileged* rendering means past or
#: future chain-of-thought leaked into the context. ``<|im_start|>``/``<|im_end|>``
#: are included because a re-tokenisation seam bug would surface as literal
#: template control tokens inside message content.
FORBIDDEN_HISTORY_MARKERS = ("<think>", "</think>", "reasoning_content", "<|im_start|>", "<|im_end|>")


@dataclass
class TurnRecord:
    """One agent turn: what the model saw, what it emitted, what happened next."""

    turn_idx: int
    observation: str
    admissible_actions: list[str]

    #: Chat-templated prompt fed to the engine for THIS turn (text form, for logs).
    student_prompt: str
    #: Exact token ids sent to the engine for this turn. Load-bearing: this is the
    #: teacher's control-view prefix, so it is never re-derived from text.
    prompt_token_ids: list[int]

    #: Exact token ids the model generated. Never decode/re-encode these.
    generated_token_ids: list[int]
    generated_text: str
    think_text: str
    action_text: str

    action_valid: bool
    parse_status: str
    #: ``None`` for the final turn (the episode ended before another observation).
    next_observation: str | None
    env_reward: float
    done: bool
    #: The raw environment state this turn acted on, WITHOUT the parser's
    #: invalid-action rejection wrapper. GiGPO's anchor key uses this: two turns
    #: that attempted different rejected commands are still standing in the same
    #: state, so they must share an anchor group. Empty for records written before
    #: this field existed (callers fall back to ``observation``).
    anchor_observation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TurnRecord":
        return cls(**payload)


@dataclass
class Trajectory:
    """A complete episode plus the metadata the HPI builder needs."""

    task_id: str
    task_type: str
    split: str
    task_description: str
    seed: int
    success: bool
    terminal_reward: float
    num_turns: int
    truncated: bool
    invalid_action_count: int
    turns: list[TurnRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived accessors -- keep the index convention in ONE place
    # ------------------------------------------------------------------

    @property
    def num_transitions(self) -> int:
        """Number of environment transitions == number of turns that were stepped.

        A turn whose action never reached the environment (parse failure) still
        contributes a transition in the records, because the environment was
        stepped with the fallback/no-op. Turns produced after ``done`` do not
        exist -- the loop stops.
        """
        return len(self.turns)

    def observation(self, idx: int) -> str:
        """``o_idx`` -- the observation the student saw at the start of turn ``idx``."""
        return self.turns[idx].observation

    def transition(self, idx: int) -> tuple[str, str]:
        """``(action_idx, observation_{idx+1})`` -- the transition at index ``idx``.

        Raises ``IndexError`` when ``idx`` is not a valid transition, which is a
        programming error in the privilege builder rather than a data condition.
        """
        turn = self.turns[idx]
        if turn.next_observation is None:
            raise IndexError(
                f"trajectory {self.task_id}: turn {idx} has no resulting observation "
                "(it is the terminal turn); transition index must be < num_transitions - 1"
            )
        return turn.action_text, turn.next_observation

    def iter_tail_transitions(self, start: int) -> Iterator[tuple[int, str, str]]:
        """Transitions ``i >= start`` as ``(i, action_i, observation_{i+1})``.

        Skips a trailing turn that has no resulting observation (episode ended on
        the previous step), so callers never have to special-case the tail.
        """
        for idx in range(max(start, 0), len(self.turns)):
            turn = self.turns[idx]
            if turn.next_observation is None:
                continue
            yield idx, turn.action_text, turn.next_observation

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["turns"] = [t.to_dict() for t in self.turns]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Trajectory":
        payload = dict(payload)
        turns = [TurnRecord.from_dict(t) for t in payload.pop("turns", [])]
        return cls(turns=turns, **payload)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def append_jsonl(self, path: str | Path) -> None:
        """Append this episode as one JSONL line (used for debug artifacts)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(self.to_jsonl_line() + "\n")

    # ------------------------------------------------------------------
    # Prompt-facing views
    # ------------------------------------------------------------------

    def history_messages(self, upto_turn: int) -> list[dict[str, str]]:
        """Causal history for turn ``upto_turn`` -- reasoning excluded.

        Emits ``task``, then ``a_0, o_1, a_1, o_2, ...`` up to and including
        ``o_{upto_turn}``, then the admissible actions for that observation. The
        exact wording lives in ``prompts.py``; this method only fixes WHAT is
        included.
        """
        from .prompts import build_history_entries

        entries = build_history_entries(self, upto_turn)
        assert_no_reasoning_in_history(entries)
        return entries

    def summary(self) -> dict:
        """Small dict for rollout metrics/logging."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "split": self.split,
            "success": self.success,
            "num_turns": self.num_turns,
            "truncated": self.truncated,
            "invalid_action_count": self.invalid_action_count,
        }


def assert_no_reasoning_in_history(entries) -> None:
    """Fail loudly if past model reasoning leaked into a context rendering.

    Accepts either a list of ``{"role", "content"}`` messages or a list of strings.
    """
    for entry in entries:
        content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
        lowered = content.lower()
        for marker in FORBIDDEN_HISTORY_MARKERS:
            if marker in lowered:
                raise AssertionError(
                    f"reasoning leaked into history/privileged context: found {marker!r}"
                )
