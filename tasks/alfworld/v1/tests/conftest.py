"""Shared fixtures for the ALFWorld HPI CPU test suite.

Every test in this directory must run without NPUs, without SGLang and without
the network. Tests that genuinely need ALFWorld's Python packages go through
``ensure_alfworld_importable`` and skip when the overlay is absent, so the suite
still gives a clean PASS/FAIL signal on a machine where setup has not run.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord  # noqa: E402


class FakeTokenizer:
    """Byte-level stand-in for the Qwen3 tokenizer.

    Prompt-construction tests only need determinism and a stable ``encode`` →
    ``decode`` relationship; using a character codec keeps the assertions exact
    (including whitespace) and keeps the suite independent of a 1.7B checkpoint.
    """

    eos_token_id = 151645
    all_special_ids = [151643, 151644, 151645, 151646]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 + 10 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((i - 10) % 1000 + 32) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def make_turn(
    turn_idx: int,
    observation: str,
    action: str,
    next_observation: str | None,
    *,
    think: str = "",
    admissible=None,
    done: bool = False,
    valid: bool = True,
) -> TurnRecord:
    return TurnRecord(
        turn_idx=turn_idx,
        observation=observation,
        admissible_actions=list(admissible or [action, "look"]),
        student_prompt="",
        prompt_token_ids=[100 + turn_idx],
        generated_token_ids=[200 + turn_idx, 201 + turn_idx],
        generated_text=f"<think>{think}</think>\n<action>{action}</action>",
        think_text=think,
        action_text=action,
        action_valid=valid,
        parse_status="ok" if valid else "not_admissible",
        next_observation=next_observation,
        env_reward=1.0 if (done and valid) else 0.0,
        done=done,
    )


def make_trajectory(
    num_turns: int = 4,
    *,
    success: bool = True,
    task_type: str = "pick_and_place_simple",
    truncated: bool = False,
    split: str = "train",
    invalid_actions: int = 0,
) -> Trajectory:
    """A synthetic episode whose observations/actions are easy to assert on.

    ``observation i`` is ``"obs{i}"``, ``action i`` is ``"act{i}"`` and the action's
    result is ``"obs{i+1}"``.

    Every turn gets a resulting observation, including the terminal one. That is
    what the rollout actually produces: a turn's record is written *after* the
    environment step, and the final step of a completed episode does return an
    observation (the one that shows the task was completed). So there are exactly
    ``num_turns`` transitions, indexed ``0 .. num_turns-1``.
    """
    turns = []
    for index in range(num_turns):
        terminal = index == num_turns - 1
        turns.append(
            make_turn(
                index,
                f"obs{index}",
                f"act{index}",
                f"obs{index + 1}",
                think=f"reasoning {index}",
                done=terminal,
            )
        )
    return Trajectory(
        task_id="valid_seen/game-1/trial_1",
        task_type=task_type,
        split=split,
        task_description="put a mug on the table",
        seed=0,
        success=success,
        terminal_reward=1.0 if success else 0.0,
        num_turns=num_turns,
        truncated=truncated,
        invalid_action_count=invalid_actions,
        turns=turns,
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "alfworld: requires the ALFWorld overlay packages")
