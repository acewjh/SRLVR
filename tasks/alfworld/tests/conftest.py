"""Fixtures for the v2 contract tests. CPU-only, no NPU, no ALFWorld needed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord  # noqa: E402


class FakeTokenizer:
    """Deterministic stand-in.

    ``__call__`` with ``return_offsets_mapping`` reproduces the same char-per-token
    mapping ``encode`` implies, which is what ``action_token_span`` verifies against.
    """

    eos_token_id = 151645
    all_special_ids = [151643, 151644, 151645, 151646]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 + 10 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((i - 10) % 1000 + 32) for i in ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids = [ord(c) % 1000 + 10 for c in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def make_trajectory(num_turns=4, *, success=True, task_type="pick_and_place_simple"):
    """Observations ``obs{i}``, actions ``act{i}``, every turn yields ``obs{i+1}``."""
    turns = []
    for index in range(num_turns):
        terminal = index == num_turns - 1
        turns.append(
            TurnRecord(
                turn_idx=index,
                observation=f"obs{index}",
                admissible_actions=[f"act{index}", "look"],
                student_prompt="",
                prompt_token_ids=[100 + index],
                generated_token_ids=[200 + index, 201 + index],
                generated_text=f"<think>r{index}</think>\n<action>act{index}</action>",
                think_text=f"r{index}",
                action_text=f"act{index}",
                action_valid=True,
                parse_status="ok",
                next_observation=f"obs{index + 1}",
                env_reward=1.0 if terminal and success else 0.0,
                done=terminal,
            )
        )
    return Trajectory(
        task_id="valid_seen/game-1/trial_1",
        task_type=task_type,
        split="train",
        task_description="put a mug on the table",
        seed=0,
        success=success,
        terminal_reward=1.0 if success else 0.0,
        num_turns=num_turns,
        truncated=False,
        invalid_action_count=0,
        turns=turns,
    )
