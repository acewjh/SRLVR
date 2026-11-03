import asyncio
from types import SimpleNamespace

import pytest

from tasks.alfworld.action_harness import HARNESS_VERSION, parse_action
from tasks.alfworld.v1.action_parser import ParseStatus


@pytest.mark.parametrize("text,action", [
    ("look<|im_end|>", "look"),
    (" Go To Shelf 1. <|endoftext|> ", "go to shelf 1"),
    ("take apple 1 from shelf 1", "take apple 1 from shelf 1"),
    ("<action>look</action>", "look"),
    ("Action: look", "look"),
])
def test_complete_command_is_validated_exactly(text, action):
    result = parse_action(text, [action])
    assert result.parsed.is_valid
    assert result.parsed.action == action


@pytest.mark.parametrize("text", [
    "look; inventory", "look\ninventory", "I will look around.",
    "<action>look", "look</action>", "```look```", "<think>look</think>",
    "look<|im_end|>inventory", "", "open", "go to",
])
def test_does_not_extract_choose_or_complete_an_action(text):
    assert not parse_action(text, ["look", "inventory"]).parsed.is_valid


def test_no_object_guessing_or_prefix_matching():
    for command in ("open RECEP", "open drawer", "go to shelf 2"):
        result = parse_action(command, ["open drawer 1", "go to shelf 1", "look"])
        assert result.parsed.status is ParseStatus.NOT_ADMISSIBLE
        assert result.parsed.action == command.lower()


def test_bare_recovery_requires_complete_output_and_environment_validation():
    assert not parse_action("look", ["look"], truncated=True).parsed.is_valid
    assert not parse_action("look", []).parsed.is_valid
    assert parse_action("<action>look</action>", []).parsed.is_valid
    assert not parse_action("<action></action>", ["look"]).parsed.is_valid


def test_recovered_actions_execute_without_rewriting_training_tokens(monkeypatch, tokenizer):
    from tasks.alfworld import episode as E, rollout as R

    calls, environments = [], []
    class Env:
        task_description = "take the apple"

        def __init__(self, *_args):
            self.actions = []
            environments.append(self)

        async def reset(self):
            return SimpleNamespace(observation="You see shelf 1.", admissible_actions=["go to shelf 1"])

        async def step(self, action):
            self.actions.append(action)
            if len(self.actions) == 1:
                assert action == "go to shelf 1"
                return SimpleNamespace(observation="On shelf 1 you see apple 1.",
                                       admissible_actions=["take apple 1 from shelf 1"], done=False, won=False)
            assert action == "take apple 1 from shelf 1"
            return SimpleNamespace(observation="You win.", admissible_actions=[], done=True, won=True)

        async def aclose(self):
            pass

    outputs = [dict(text="go to shelf 1<|im_end|>", token_ids=[101, 102, 151645],
                    log_probs=[-.3, -.4, -.5], finish_reason="stop"),
               dict(text="take apple 1 from shelf 1<|im_end|>", token_ids=[201, 202, 151645],
                    log_probs=[-.6, -.7, -.8], finish_reason="stop")]

    async def generate(_args, prompt, _params, _url):
        calls.append(prompt)
        return outputs[len(calls) - 1]

    async def acquire_env(*args):
        return Env(*args)

    monkeypatch.setattr(E, "acquire_env", acquire_env)
    monkeypatch.setattr(E, "_generate", generate)
    monkeypatch.setenv("OPSD_SRLVR", "1")  # Retain sampled EOS exactly as this method does.
    monkeypatch.setenv("ALFWORLD_V2_MAX_TURNS", "2")
    game = SimpleNamespace(game_id="train/game/trial", task_type="pick_and_place_simple", split="train")
    result = asyncio.run(E.run_episode(SimpleNamespace(seq_length=32768), tokenizer, game,
                                      {"max_new_tokens":128}, episode_id=0, group_id=0))
    assert result.trajectory.success and result.trajectory.invalid_action_count == 0
    assert environments[0].actions == ["go to shelf 1", "take apple 1 from shelf 1"]
    rows = R.build_group_samples(tokenizer, game, [result], SimpleNamespace(advantages=[1.]),
                                 0, with_hpi=False, future_gap=1)
    for i, (sample, raw) in enumerate(zip(rows, outputs)):
        assert result.turns[i].generated_text == raw["text"]
        assert sample.tokens == calls[i] + raw["token_ids"]
        assert sample.rollout_log_probs == raw["log_probs"]
        assert sample.response_length == 3 and sample.loss_mask == [1, 1, 1]
        assert sample.metadata["action_harness"] == dict(
            version=HARNESS_VERSION, source="bare_command", action=environments[0].actions[i], valid=True)
    assert E._history_pairs(result.trajectory, 2)[0][1] == "go to shelf 1"
