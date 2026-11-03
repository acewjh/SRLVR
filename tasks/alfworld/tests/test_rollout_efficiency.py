"""Keep rollout optimizations from changing PPO's sampled-token evidence."""

import asyncio
from argparse import Namespace

import pytest

from slime.utils import http_utils
from tasks.alfworld.episode import _generate
from tasks.alfworld.prompts import SYSTEM_PROMPT, TaskView, build_turn_messages, render_chat


@pytest.mark.parametrize("hpi,expected_k", [("0", 0), ("1", 20)])
def test_candidate_tables_optional_but_sampled_logprobs_preserved(monkeypatch, hpi, expected_k):
    monkeypatch.setenv("ALFWORLD_V2_HPI", hpi)
    monkeypatch.delenv("ALFWORLD_V2_HSI_CANDIDATE_K", raising=False)
    calls = []

    async def post(url, payload):
        calls.append(payload)
        return {"text": "<action>look</action>", "meta_info": {
            "output_token_logprobs": [[-0.25, 123, None], [-0.5, 456, None]],
            "finish_reason": {"type": "stop"},
        }}

    monkeypatch.setattr(http_utils, "post", post)
    out = asyncio.run(_generate(Namespace(), [1, 2], {"max_new_tokens": 4096}, "http://engine"))
    assert calls[0]["return_logprob"] is True
    assert calls[0]["top_logprobs_num"] == expected_k
    assert out["token_ids"] == [123, 456]
    assert out["log_probs"] == [-0.25, -0.5]


def test_action_only_template_hides_action_list_even_with_legacy_env(monkeypatch):
    view = TaskView("put the mug away", "You see a mug.", ["look"], [])
    monkeypatch.setenv("ALFWORLD_THINKING_STYLE", "brief")
    monkeypatch.delenv("ALFWORLD_THINKING_TOKENS", raising=False)
    brief = build_turn_messages(view)
    monkeypatch.setenv("ALFWORLD_THINKING_STYLE", "legacy")
    original = build_turn_messages(view)
    assert brief[1] == original[1]
    assert brief[0]["content"] != SYSTEM_PROMPT
    assert "tokens inside <think>" not in brief[0]["content"]
    assert "one short sentence" not in brief[0]["content"]
    assert "do not invent" in brief[0]["content"]
    assert "<think>" in brief[0]["content"]
    assert "<think>" in original[0]["content"]
    assert "Admissible actions:" not in brief[1]["content"]
    assert "  - look" not in brief[1]["content"]

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return "action-only template"

    assert render_chat(Tokenizer(), brief) == "action-only template"


def test_hindsight_template_matches_thinking_enabled_student(monkeypatch):
    from tasks.alfworld.privilege import _prompt_ids

    monkeypatch.setenv("ALFWORLD_THINKING_STYLE", "brief")
    monkeypatch.setenv("ALFWORLD_THINKING_TOKENS", "512")
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "rendered"

        def encode(self, text, add_special_tokens=False):
            assert text == "rendered"
            return [1, 2, 3]

    assert _prompt_ids(Tokenizer(), "same user view") == [1, 2, 3]
    assert captured["kwargs"]["enable_thinking"] is True
    assert "up to 512 tokens" in captured["messages"][0]["content"]
    assert captured["messages"][0]["content"] == build_turn_messages(
        TaskView("task", "observation", [], [])
    )[0]["content"]


def test_complete_trajectory_serialized_once_per_episode(tokenizer):
    from conftest import make_trajectory
    from tasks.alfworld.advantages import compute_group_advantages
    from tasks.alfworld.episode import EpisodeResult, TurnSample
    from tasks.alfworld.rollout import build_group_samples

    trajectory = make_trajectory(num_turns=3)
    original = trajectory.to_dict()
    calls = []

    def to_dict():
        calls.append(1)
        return original

    trajectory.to_dict = to_dict
    episode = EpisodeResult(trajectory=trajectory, turns=[
        TurnSample(0, 0, "game", i, [10], [20], "action", None, [-0.5])
        for i in range(3)
    ])
    rows = build_group_samples(tokenizer, None, [episode],
                               compute_group_advantages([1.0]), 0,
                               with_hpi=False, future_gap=1)
    assert len(rows) == 3
    assert len(calls) == 1
    assert all(row.metadata["trajectory"] == original for row in rows)
    assert rows[0].metadata["trajectory"] is rows[2].metadata["trajectory"]
    assert [row.tokens for row in rows] == [[10, 20]] * 3
    assert [row.rollout_log_probs for row in rows] == [[-0.5]] * 3
