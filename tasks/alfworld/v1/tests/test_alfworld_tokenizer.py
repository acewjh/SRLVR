"""Prompt rendering against the REAL Qwen3-1.7B tokenizer.

The rest of the suite uses a byte-level fake so it stays fast and checkpoint-free.
That fake cannot catch a chat-template behaviour change, and two real template
behaviours are load-bearing here:

1. ``enable_thinking=True`` must be honoured (thinking is the fixed protocol for
   every arm -- a silent fall back to non-thinking would change all four).
2. Qwen3's template STRIPS past ``<think>`` blocks from assistant history. That is
   desirable (it enforces the no-past-CoT rule) but it also means turn ``t+1``'s
   prompt is not a token-level extension of turn ``t``'s -- the reason the rollout
   uses a per-turn segment layout rather than one chained sequence.

Skipped when the checkpoint is absent.
"""

import os

import pytest

CHECKPOINT = os.environ.get(
    "ALFWORLD_TOKENIZER",
    "/workspace/cua/checkpoints/Qwen3-1.7B",
)


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(CHECKPOINT):
        pytest.skip(f"Qwen3 tokenizer not found at {CHECKPOINT}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CHECKPOINT)


def _view():
    from tasks.alfworld.v1.prompts import TaskView

    return TaskView(
        task_description="put a cool tomato in microwave",
        current_observation="You arrive at countertop 1. On the countertop 1, you see a tomato 1.",
        admissible_actions=["cool tomato 1 with fridge 1", "take tomato 1 from countertop 1"],
        action_history=["go to countertop 1"],
    )


def test_thinking_mode_is_enabled(tokenizer):
    """The assistant generation prompt must let the model emit its own reasoning.

    With ``enable_thinking=True`` Qwen3 does NOT prefill an empty think block;
    the model writes ``<think>...</think>`` itself, which is what the parser and
    the "past CoT excluded from history" rule assume.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert text.endswith("<|im_start|>assistant\n"), repr(text[-60:])
    assert "<think>" not in text, "thinking mode must not prefill a think block"


def test_prompt_carries_task_observation_and_actions(tokenizer):
    from tasks.alfworld.v1.prompts import build_turn_prompt_ids

    ids = build_turn_prompt_ids(tokenizer, _view())
    text = tokenizer.decode(ids)
    assert "put a cool tomato in microwave" in text
    assert "You arrive at countertop 1" in text
    assert "cool tomato 1 with fridge 1" in text
    assert "go to countertop 1" in text


def test_user_turn_contains_no_reasoning(tokenizer):
    """Reasoning must be absent from the user turn, not merely from the transcript.

    The SYSTEM prompt legitimately contains ``<think>...</think>`` -- it is the
    format instruction telling the model where to put its reasoning. That is why
    the rendered-prompt check has to be scoped to the user turn: a blanket
    substring scan over the whole prompt would fail on the instruction itself and
    train the reader to ignore it.
    """
    from tasks.alfworld.v1.prompts import build_turn_user_message

    message = build_turn_user_message(_view())
    assert "<think>" not in message
    assert "</think>" not in message
    assert "reasoning_content" not in message

    # and the only <think> in the full rendering is the one in the instruction
    from tasks.alfworld.v1.prompts import build_turn_prompt_ids

    full = tokenizer.decode(build_turn_prompt_ids(tokenizer, _view()))
    user_start = full.index("<|im_start|>user")
    assert "<think>" not in full[user_start:], "reasoning leaked into the user turn"


def test_prompt_ids_are_deterministic(tokenizer):
    from tasks.alfworld.v1.prompts import build_turn_prompt_ids

    assert build_turn_prompt_ids(tokenizer, _view()) == build_turn_prompt_ids(tokenizer, _view())


def test_privileged_prompt_differs_only_by_the_block(tokenizer):
    """The teacher's full view must extend the control view, not replace it."""
    from tasks.alfworld.v1.privilege import MODE_FUTURE_OUTCOME, build_hpi_privilege, build_privileged_prompt_ids
    from tasks.alfworld.v1.prompts import build_turn_prompt_ids

    trajectory = _trajectory()
    view = _view()
    control = build_turn_prompt_ids(tokenizer, view)
    privilege = build_hpi_privilege(trajectory, 0, MODE_FUTURE_OUTCOME, future_gap=1)
    full = build_privileged_prompt_ids(tokenizer, view, privilege)

    assert full != control
    assert len(full) > len(control)
    # the privileged rendering must still contain the student's task text verbatim
    control_text = tokenizer.decode(control)
    full_text = tokenizer.decode(full)
    assert "put a cool tomato in microwave" in full_text
    assert "<privileged_hindsight>" in full_text

    # the privilege block sits BEFORE the generation slot, so the teacher can
    # condition on it (a causal LM cannot attend to tokens placed after the target)
    assert full_text.index("<privileged_hindsight>") < full_text.rindex("<|im_start|>assistant")


def test_qwen3_chat_template_strips_past_reasoning_from_history(tokenizer):
    """Documents the template behaviour the sequence layout is designed around.

    If a future transformers release stops stripping, this test fails and the
    per-turn segment layout can be revisited -- it will not silently start
    training on contexts that differ from what the engine saw.
    """
    messages = [
        {"role": "user", "content": "Task: X"},
        {"role": "assistant", "content": "<think>\nsecret plan\n</think>\n<action>look</action>"},
        {"role": "user", "content": "Observation: a mug."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert "secret plan" not in text, (
        "Qwen3's template no longer strips past reasoning; the history would now "
        "contain chains of thought the harness intends to exclude"
    )
    assert "look" in text, "the past ACTION must survive into the history"


def test_rollout_log_probs_align_with_a_real_generation(tokenizer):
    """Encode -> decode -> re-encode must be identity for the tags we rely on."""
    for text in (
        "<think>\nreason\n</think>\n<action>go to desk 1</action>",
        "<action>cool tomato 1 with fridge 1</action>",
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.encode(tokenizer.decode(ids), add_special_tokens=False) == ids


def _trajectory():
    from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord

    def turn(index, action, next_obs, done=False):
        return TurnRecord(
            turn_idx=index,
            observation=f"obs{index}",
            admissible_actions=[action],
            student_prompt="",
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_text=f"<think>r{index}</think><action>{action}</action>",
            think_text=f"r{index}",
            action_text=action,
            action_valid=True,
            parse_status="ok",
            next_observation=next_obs,
            env_reward=1.0 if done else 0.0,
            done=done,
        )

    return Trajectory(
        task_id="t",
        task_type="pick_and_place_simple",
        split="train",
        task_description="put a cool tomato in microwave",
        seed=0,
        success=True,
        terminal_reward=1.0,
        num_turns=3,
        truncated=False,
        invalid_action_count=0,
        turns=[turn(0, "a0", "obs1"), turn(1, "a1", "obs2"), turn(2, "a2", "obs3", done=True)],
    )
