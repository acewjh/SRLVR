"""End-to-end rollout layout test with a scripted engine and a scripted env.

No NPU, no SGLang, no ALFWorld packages: ``_generate_tokens`` and ``AlfWorldEnv``
are replaced by deterministic stand-ins. That makes the *whole* per-turn layout
path testable -- sequence assembly, loss mask, turn ids, HPI views and the
train-data conversion -- which is where an off-by-one would otherwise only show up
as a silently wrong training run.
"""

import asyncio

import pytest

from tasks.alfworld.v1 import rollout as rollout_module
from tasks.alfworld.v1.privilege import MODE_FUTURE_OUTCOME, MODE_OUTCOME_ONLY
from tasks.alfworld.v1.env_adapter import GameRef
from tasks.alfworld.v1.rollout import build_episode_hpi_views, run_episode
from tasks.alfworld.v1.train_data import convert_samples_to_train_data, validate_layout

from conftest import FakeTokenizer
from slime.utils.types import Sample


class ScriptedEnv:
    """A deterministic two-room environment: task completes after ``solve_at`` steps."""

    script = ["go to desk 1", "take mug 1 from desk 1", "look"]
    task_description = "put a mug on the table"

    def __init__(self, game, max_steps):
        self.game = game
        self.max_steps = max_steps
        self.steps = 0

    @classmethod
    def reset(cls):
        raise NotImplementedError

    def _result(self, observation, done, won):
        class R:
            pass

        result = R()
        result.observation = observation
        result.admissible_actions = list(cls_admissible := type(self).script)
        result.done = done
        result.won = won
        result.score = 1.0 if won else 0.0
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


class FakeEnv:
    """Scripted ALFWorld stand-in.

    Observations are ``obs{t}``; the episode succeeds once the scripted action
    ``take mug 1 from desk 1`` has been executed, else it truncates.
    """

    script = ["go to desk 1", "take mug 1 from desk 1", "look"]

    def __init__(self, game, max_steps):
        self.game = game
        self.max_steps = max_steps
        self.task_description = "put a mug on the table"
        self.steps = 0

    def reset(self):
        return self._result(0, done=False, won=False)

    def step(self, action):
        self.steps += 1
        won = action == "take mug 1 from desk 1"
        done = won
        return self._result(self.steps, done=done, won=won)

    def _result(self, index, done, won):
        class R:
            pass

        result = R()
        result.observation = f"obs{index}"
        result.admissible_actions = list(self.script)
        result.done = done
        result.won = won
        result.score = 1.0 if won else 0.0
        return result

    def close(self):
        pass


class ScriptedModel:
    """Emits the next scripted action, so an episode always completes."""

    def __init__(self):
        self.turn = 0

    def next_tokens(self, tokenizer):
        action = FakeEnv.script[min(self.turn, len(FakeEnv.script) - 1)]
        self.turn += 1
        text = f"<think>\nstep {self.turn}\n</think>\n<action>{action}</action>"
        ids = tokenizer.encode(text)
        return {
            "token_ids": ids,
            "log_probs": [-0.5] * len(ids),
            "text": text,
            "finish_reason": "stop",
        }


class FakeEngineState:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _run(args, tokenizer, mode, monkeypatch, max_turns=6, game=None):
    model = ScriptedModel()
    monkeypatch.setattr(rollout_module, "AlfWorldEnv", FakeEnv)

    async def fake_generate(_args, _prompt_ids, _sampling_params):
        return model.next_tokens(tokenizer)

    monkeypatch.setattr(rollout_module, "_generate_tokens", fake_generate)

    game = game or GameRef(
        game_id="train/game-1/trial_1", path="/nonexistent/game.tw-pddl",
        task_type="pick_and_place_simple", split="train",
    )
    monkeypatch.setenv("ALFWORLD_HPI_MODE", mode)
    monkeypatch.setenv("ALFWORLD_MAX_TURNS", str(max_turns))
    state = FakeEngineState(tokenizer)
    return asyncio.run(run_episode(args, tokenizer, game, {}, seed=0))


class Args:
    pass


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setenv("ALFWORLD_ACTION_HISTORY", "4")
    return Args()


def _sample_from(trajectory, layout, tokenizer, mode, gap=2):
    sample = Sample(index=0, prompt="")
    sample.tokens = list(layout.tokens)
    sample.response_length = layout.response_length
    sample.loss_mask = list(layout.loss_mask)
    sample.turn_ids = list(layout.turn_ids)
    sample.rollout_log_probs = list(layout.rollout_log_probs)
    sample.reward = trajectory.terminal_reward
    sample.status = Sample.Status.COMPLETED
    sample.metadata["turn_spans"] = [list(s) for s in layout.turn_spans]
    sample.metadata["num_turns"] = trajectory.num_turns
    sample.metadata["invalid_action_count"] = trajectory.invalid_action_count
    if mode != "none":
        views, plens, tlens = build_episode_hpi_views(tokenizer, trajectory, layout, mode, gap)
        sample.metadata["hpi_views"] = views
        sample.metadata["hpi_prompt_lengths"] = plens
        sample.metadata["hpi_target_lengths"] = tlens
    return sample


# ---------------------------------------------------------------------------


def test_episode_completes_and_reports_success(args, tokenizer, monkeypatch):
    trajectory, layout, metrics = _run(args, tokenizer, "none", monkeypatch)
    assert trajectory.success is True
    assert trajectory.terminal_reward == 1.0
    assert trajectory.num_turns == 2, "the scripted episode solves on its second action"
    assert metrics["success"] == 1


def test_layout_loss_mask_is_one_only_on_generated_tokens(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, "none", monkeypatch)
    assert len(layout.loss_mask) == layout.response_length
    assert set(layout.loss_mask) <= {0, 1}
    assert sum(layout.loss_mask) == sum(
        len(t.generated_token_ids) for t in trajectory.turns
    )
    for mask, turn in zip(layout.loss_mask, layout.turn_ids, strict=True):
        assert (turn >= 0) == bool(mask)


def test_each_turn_span_holds_exactly_that_turns_tokens(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, "none", monkeypatch)
    response = layout.response_tokens
    for turn, (start, end) in zip(trajectory.turns, layout.turn_spans, strict=True):
        assert response[start:end] == list(turn.generated_token_ids)
        assert layout.turn_ids[start] == turn.turn_idx


def test_stored_prompt_ids_are_the_prefix_of_the_layout(args, tokenizer, monkeypatch):
    """The control view reuses prompt_t, so it must match the layout exactly."""
    trajectory, layout, _ = _run(args, tokenizer, "none", monkeypatch)
    for turn_index, (prompt_ids, y_ids, _view) in enumerate(layout.turns):
        start, end = layout.turn_spans[turn_index]
        # prompt_t immediately precedes y_t inside the episode layout
        offset = layout.prefix_len + start
        assert layout.tokens[offset - len(prompt_ids) : offset] == prompt_ids
        assert layout.tokens[offset:end + layout.prefix_len] == y_ids


def test_rollout_log_probs_align_with_generated_tokens(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, "none", monkeypatch)
    for (start, end) in layout.turn_spans:
        assert layout.rollout_log_probs[start:end] == [-0.5] * (end - start)
    for position, mask in enumerate(layout.loss_mask):
        if mask == 0:
            assert layout.rollout_log_probs[position] == 0.0


def test_invalid_action_does_not_consume_an_environment_step(args, tokenizer, monkeypatch):
    """A malformed turn must not advance the env, so max_turns bounds real steps."""
    model = ScriptedModel()
    monkeypatch.setattr(rollout_module, "AlfWorldEnv", FakeEnv)

    calls = {"n": 0}

    async def fake_generate(_args, _prompt_ids, _sampling_params):
        calls["n"] += 1
        text = "<think>confused</think>\n<action>not a real command</action>"
        ids = tokenizer.encode(text)
        return {"token_ids": ids, "log_probs": [-1.0] * len(ids), "text": text, "finish_reason": "stop"}

    monkeypatch.setattr(rollout_module, "_generate_tokens", fake_generate)
    monkeypatch.setenv("ALFWORLD_HPI_MODE", "none")
    game = GameRef("g", "/nonexistent", "pick_and_place_simple", "train")
    trajectory, layout, _ = asyncio.run(
        run_episode(args, tokenizer, game, {}, seed=0)
    )
    assert trajectory.num_turns == calls["n"]
    assert trajectory.invalid_action_count == trajectory.num_turns
    assert trajectory.success is False
    # The env is not stepped, but the rejection IS fed back: the next turn's
    # observation must contain the notice and still carry the original room state,
    # otherwise the model cannot tell that its action was refused.
    next_obs = trajectory.turns[0].next_observation
    assert "was rejected" in next_obs
    assert "obs0" in next_obs, "the original observation must survive in the notice"


def test_truncation_is_recorded_when_the_turn_budget_runs_out(args, tokenizer, monkeypatch):
    async def never_done(*_a, **_k):
        text = "<think>keep looking</think>\n<action>look</action>"
        ids = tokenizer.encode(text)
        return {"token_ids": ids, "log_probs": [-0.1] * len(ids), "text": text, "finish_reason": "stop"}

    monkeypatch.setattr(rollout_module, "AlfWorldEnv", FakeEnv)
    monkeypatch.setattr(rollout_module, "_generate_tokens", never_done)
    monkeypatch.setenv("ALFWORLD_HPI_MODE", "none")
    monkeypatch.setenv("ALFWORLD_MAX_TURNS", "3")
    game = GameRef("g", "/nonexistent", "pick_and_place_simple", "train")
    trajectory, layout, _ = asyncio.run(
        run_episode(args, tokenizer, game, {}, seed=0)
    )
    assert trajectory.truncated is True
    assert trajectory.num_turns == 3
    assert trajectory.terminal_reward == 0.0


# ---------------------------------------------------------------------------
# HPI wiring
# ---------------------------------------------------------------------------


def test_no_views_are_built_for_mode_none(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, "none", monkeypatch)
    views, plens, tlens = build_episode_hpi_views(tokenizer, trajectory, layout, "none", 2)
    assert (views, plens, tlens) == ([], [], [])


@pytest.mark.parametrize("mode", [MODE_OUTCOME_ONLY, MODE_FUTURE_OUTCOME])
def test_views_come_in_full_control_pairs_with_identical_targets(mode, args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, mode, monkeypatch)
    views, plens, tlens = build_episode_hpi_views(tokenizer, trajectory, layout, mode, gap=2)
    assert len(views) == 2 * trajectory.num_turns

    response = layout.response_tokens
    for turn_index, (start, end) in enumerate(layout.turn_spans):
        full, control = views[2 * turn_index], views[2 * turn_index + 1]
        expected = response[start:end]
        # the teacher-forced target is the student's own tokens in BOTH views
        assert list(full[plens[2 * turn_index] :]) == expected
        assert list(control[plens[2 * turn_index + 1] :]) == expected
        # the control view is the student's literal prompt, so it is shorter
        # by exactly the privileged block
        assert control[: plens[2 * turn_index + 1]] == layout.turns[turn_index][0]
        assert plens[2 * turn_index] > plens[2 * turn_index + 1]


def test_future_views_carry_more_context_than_outcome_only(args, tokenizer, monkeypatch):
    t_out, l_out, _ = _run(args, tokenizer, MODE_OUTCOME_ONLY, monkeypatch)
    t_fut, l_fut, _ = _run(args, tokenizer, MODE_FUTURE_OUTCOME, monkeypatch)
    outcome_views, outcome_plens, _ = build_episode_hpi_views(
        tokenizer, t_out, l_out, MODE_OUTCOME_ONLY, 2
    )
    future_views, future_plens, _ = build_episode_hpi_views(
        tokenizer, t_fut, l_fut, MODE_FUTURE_OUTCOME, 2
    )
    # same episode, same number of turns -> same control prompts
    assert outcome_plens[1::2] == future_plens[1::2]
    # the future privilege block is strictly longer than the outcome-only one
    for index in range(0, len(outcome_plens), 2):
        assert future_plens[index] > outcome_plens[index]


def test_views_and_layout_survive_train_data_conversion(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, MODE_FUTURE_OUTCOME, monkeypatch)
    sample = _sample_from(trajectory, layout, tokenizer, MODE_FUTURE_OUTCOME, gap=2)
    validate_layout(sample)

    class ConvertArgs:
        global_batch_size = 1

    train_data = convert_samples_to_train_data(ConvertArgs(), [sample])
    assert len(train_data["tokens"][0]) == len(sample.tokens)
    assert len(train_data["loss_masks"][0]) == sample.response_length
    assert len(train_data["turn_ids"][0]) == sample.response_length
    assert len(train_data["hpi_views"][0]) == 2 * trajectory.num_turns


def test_conversion_rejects_a_batch_with_a_missing_view_payload(args, tokenizer, monkeypatch):
    trajectory, layout, _ = _run(args, tokenizer, MODE_FUTURE_OUTCOME, monkeypatch)
    good = _sample_from(trajectory, layout, tokenizer, MODE_FUTURE_OUTCOME, gap=2)
    bad = _sample_from(trajectory, layout, tokenizer, "none", gap=2)

    class ConvertArgs:
        global_batch_size = 2

    with pytest.raises(ValueError, match="hpi_views present on only"):
        convert_samples_to_train_data(ConvertArgs(), [good, bad])
