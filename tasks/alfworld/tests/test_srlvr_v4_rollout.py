"""SRLVR-v4 rollout/convert contracts: one teacher view, per-turn spans.

The teacher view is built once per trajectory and shared by every turn row; each
row carries the span of its generated tokens inside the shared continuation.
These tests pin the reconstruction (no duplicated or lost history tokens), the
span offsets, and the ``hpi_*`` transport that carries the view to the loss.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.alfworld import rollout
from tasks.alfworld.episode import EpisodeResult, TurnSample
from tasks.alfworld.privilege import render_trajectory_block
from tasks.alfworld.prompts import TaskView
from tasks.alfworld.train_data import convert_samples_to_train_data
from tasks.alfworld.v1.trajectory import Trajectory, TurnRecord

PROMPTS = [
    [10, 11],
    [10, 11, 200, 201, 30],
    [10, 11, 200, 201, 30, 210, 211, 31],
]
# The generated ids are exactly what the fake tokenizer produces for the turn's
# ``<action>...</action>`` text, so ``action_token_span`` really maps the tag
# CONTENT back onto them. The previous fixture used short arbitrary ids that
# never matched the re-encoding, so those tests only ever exercised the
# full-generation fallback this suite now pins as REMOVED.
GENERATED_TEXT = [f"<action>act{index}</action>" for index in range(3)]
GENERATED_IDS = [[ord(c) % 1000 + 10 for c in text] for text in GENERATED_TEXT]
# What the row actually trains on: the action CONTENT, tags excluded.
GENERATED = [[ord(c) % 1000 + 10 for c in f"act{index}"] for index in range(3)]
_NL = [ord("\n") % 1000 + 10]
# FakeTokenizer.encode("\n") == [20]; the teacher stream is the action tokens
# with one newline between turns (observations live in the privilege block).
CONTINUATION = GENERATED[0] + _NL + GENERATED[1] + _NL + GENERATED[2]
SPANS = [(0, len(GENERATED[0])),
         (len(GENERATED[0]) + 1, len(GENERATED[1])),
         (len(GENERATED[0]) + len(GENERATED[1]) + 2, len(GENERATED[2]))]
EPISODE_ID = 123


def _episode(tokenizer=None) -> EpisodeResult:
    records, turns = [], []
    for index, (prompt, generated) in enumerate(zip(PROMPTS, GENERATED, strict=True)):
        text = f"<action>act{index}</action>"
        ids = tokenizer.encode(text) if tokenizer is not None else list(GENERATED_IDS[index])
        records.append(
            TurnRecord(
                turn_idx=index,
                observation=f"obs{index}",
                admissible_actions=[f"act{index}", "look"],
                student_prompt="",
                prompt_token_ids=list(prompt),
                generated_token_ids=list(ids),
                generated_text=text,
                think_text="",
                action_text=f"act{index}",
                action_valid=True,
                parse_status="ok",
                next_observation=f"obs{index + 1}",
                env_reward=1.0 if index == len(PROMPTS) - 1 else 0.0,
                done=index == len(PROMPTS) - 1,
            )
        )
        view = TaskView("put a mug on the table", f"obs{index}", [f"act{index}", "look"], [])
        turns.append(
            TurnSample(
                episode_id=EPISODE_ID,
                group_id=0,
                task_id="valid_seen/game-1/trial_1",
                turn_idx=index,
                prompt_token_ids=list(prompt),
                generated_token_ids=list(ids),
                generated_text=text,
                view=view,
                generated_log_probs=[-0.1] * len(ids),
            )
        )
    trajectory = Trajectory(
        task_id="valid_seen/game-1/trial_1",
        task_type="pick_and_place_simple",
        split="train",
        task_description="put a mug on the table",
        seed=0,
        success=True,
        terminal_reward=1.0,
        num_turns=len(records),
        truncated=False,
        invalid_action_count=0,
        turns=records,
    )
    return EpisodeResult(trajectory, turns)


@pytest.fixture(autouse=True)
def _v4_env(monkeypatch):
    monkeypatch.setenv("OPSD_SRLVR", "1")
    monkeypatch.setenv("SRLVR_VARIANT", "v4")
    monkeypatch.setenv("SRLVR_V4_COMPACT_FUTURE", "0")
    # The teacher prompt build is exercised separately; here it is a fixed prefix.
    monkeypatch.setattr(rollout, "_prompt_ids_for", lambda tokenizer, view, block: [901, 902, 903, 904])


def _rows(tokenizer, *, rollout_id=7):
    episode = _episode(tokenizer)
    return rollout.build_group_samples(
        tokenizer,
        None,
        [episode],
        rollout.OutcomeAnchor([0.0], [1.0]),
        0,
        with_hpi=False,
        future_gap=1,
        rollout_id=rollout_id,
    )


def test_rows_share_one_teacher_view_and_carry_exact_spans(tokenizer):
    rows = _rows(tokenizer)
    assert len(rows) == len(PROMPTS)

    for index, (row, prompt, generated) in enumerate(
        zip(rows, PROMPTS, GENERATED, strict=True)
    ):
        assert row.tokens == prompt + generated
        assert row.response_length == len(generated)
        assert row.loss_mask == [1] * len(generated)
        assert row.rollout_log_probs == [-0.1] * len(generated)
        assert row.opsd_advantages == [0.0] * len(generated)
        assert row.turn_ids == [EPISODE_ID * rollout.config.TURN_ID_STRIDE + index] * len(generated)

        teacher, prompt_len, target_len = row.metadata["hsi_teacher_view"]
        assert prompt_len == 4
        assert target_len == len(CONTINUATION)
        assert teacher[:prompt_len] == [901, 902, 903, 904]
        assert teacher[prompt_len:] == CONTINUATION

        meta = row.metadata["srlvr_v4"]
        span_start, span_len = meta["span_start"], meta["span_len"]
        assert (span_start, span_len) == SPANS[index]
        assert CONTINUATION[span_start : span_start + span_len] == generated
        assert meta["trajectory_id"] == str(EPISODE_ID)
        assert meta["policy_version"] == 7
        assert meta["continuation_len"] == len(CONTINUATION)


def test_empty_actor_turn_is_skipped_not_trained_on_its_generation(monkeypatch, tokenizer):
    """A turn with no recoverable action contributes NO row and no target.

    Training the full generation instead (the removed fallback) made the failed,
    longest turns the dominant training signal and collapsed the policy; see
    rollout._turn_action_tokens.
    """
    episode = _episode(tokenizer)
    episode.turns[1].generated_token_ids = []
    episode.turns[1].generated_text = ""
    monkeypatch.setattr(rollout, "_prompt_ids_for", lambda *a, **k: [901, 902, 903, 904])
    rows = rollout.build_group_samples(
        tokenizer,
        None,
        [episode],
        rollout.OutcomeAnchor([0.0], [1.0]),
        0,
        with_hpi=False,
        future_gap=1,
    )
    # Two rows survive (turns 0 and 2); the empty turn is gone, not faked.
    assert len(rows) == 2
    for row in rows:
        assert row.response_length > 0
        assert sum(row.loss_mask) == row.response_length
        span_start, span_len = row.metadata["srlvr_v4"]["span_start"], row.metadata["srlvr_v4"]["span_len"]
        teacher, prompt_len, _ = row.metadata["hsi_teacher_view"]
        assert teacher[prompt_len + span_start: prompt_len + span_start + span_len] == list(row.tokens[-row.response_length:])


def test_convert_packs_teacher_views_and_spans(tokenizer):
    rows = _rows(tokenizer)
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    data = convert_samples_to_train_data(args, rows)

    assert len(data["hpi_views"]) == 3
    for index, (view, prompt_len, target_len, span) in enumerate(
        zip(
            data["hpi_views"],
            data["hpi_prompt_lengths"],
            data["hpi_target_lengths"],
            data["hpi_user_lengths"],
            strict=True,
        )
    ):
        assert view == [[901, 902, 903, 904] + CONTINUATION]
        assert prompt_len == [4]
        assert target_len == [len(CONTINUATION)]
        assert span == list(SPANS[index])
    assert "srlvr_batch" not in data
    assert data["dynamic_global_batch_size"] == 3


def test_compact_and_full_keep_identical_student_rows(tokenizer, monkeypatch):
    full = _rows(tokenizer)
    monkeypatch.setenv("SRLVR_V4_COMPACT_FUTURE", "1")
    compact = _rows(tokenizer)
    assert len(full) == len(compact)
    for left, right in zip(full, compact, strict=True):
        for key in ("tokens", "loss_mask", "rollout_log_probs", "turn_ids", "opsd_advantages"):
            assert getattr(left, key) == getattr(right, key)
        assert left.metadata["srlvr_v4"]["span_start"] == right.metadata["srlvr_v4"]["span_start"]
        assert right.metadata["future_compression"]["transitions"] == 3


def test_overlong_teacher_view_keeps_rows_with_zero_loss(monkeypatch, tokenizer):
    # Fake teacher prompt is 4 tokens; continuation is 8; limit 8 => overlong.
    monkeypatch.setenv("SEQ_LENGTH", "8")
    rows = _rows(tokenizer)
    assert len(rows) == len(PROMPTS)
    for row in rows:
        assert row.loss_mask == [0] * row.response_length
        assert row.metadata["srlvr_v4"]["skipped_overlong"] is True
        assert row.metadata["hsi_teacher_view"] == ([0], 1, 0)


def test_trajectory_block_contains_the_whole_episode_and_outcome():
    trajectory = _episode().trajectory
    block = render_trajectory_block(
        trajectory, outcome_text="<outcome>SUCCESS</outcome>", compact_future=False
    )
    assert "<outcome>SUCCESS</outcome>" in block
    assert "Complete interaction trace:" in block
    for index in range(3):
        assert f"[Future transition {index}]" in block
        assert f"Executed action: act{index}" in block
        assert f"Resulting environment observation: obs{index + 1}" in block
    assert "<think>" not in block
    assert "Future interaction trace:" not in block  # per-turn template, not the v4 one

    compact = render_trajectory_block(
        trajectory, outcome_text="<outcome>SUCCESS</outcome>", compact_future=True
    )
    assert "act0" in compact and "act1" in compact and "act2" in compact
    assert "<outcome>SUCCESS</outcome>" in compact


def _action_episode(tokenizer):
    """Episode whose generated ids are the real tokenization of think+action."""
    records, turns, actions = [], [], []
    for index in range(3):
        text = f"<think>reason{index}</think>\n<action>act{index}</action>"
        generated = list(tokenizer.encode(text))
        action = list(tokenizer.encode(f"act{index}"))
        actions.append(action)
        records.append(
            TurnRecord(
                turn_idx=index,
                observation=f"obs{index}",
                admissible_actions=[f"act{index}", "look"],
                student_prompt="",
                prompt_token_ids=list(PROMPTS[index]),
                generated_token_ids=list(generated),
                generated_text=text,
                think_text=f"reason{index}",
                action_text=f"act{index}",
                action_valid=True,
                parse_status="ok",
                next_observation=f"obs{index + 1}",
                env_reward=1.0 if index == 2 else 0.0,
                done=index == 2,
            )
        )
        view = TaskView("put a mug on the table", f"obs{index}", [f"act{index}", "look"], [])
        turns.append(
            TurnSample(
                episode_id=EPISODE_ID,
                group_id=0,
                task_id="valid_seen/game-1/trial_1",
                turn_idx=index,
                prompt_token_ids=list(PROMPTS[index]),
                generated_token_ids=list(generated),
                generated_text=text,
                view=view,
                generated_log_probs=[-0.1] * len(generated),
            )
        )
    trajectory = Trajectory(
        task_id="valid_seen/game-1/trial_1",
        task_type="pick_and_place_simple",
        split="train",
        task_description="put a mug on the table",
        seed=0,
        success=True,
        terminal_reward=1.0,
        num_turns=len(records),
        truncated=False,
        invalid_action_count=0,
        turns=records,
    )
    return EpisodeResult(trajectory, turns), actions


def test_hindsight_drops_thinking_and_trains_the_action_span(tokenizer, monkeypatch):
    monkeypatch.setenv("SEQ_LENGTH", "32768")
    monkeypatch.setenv("SRLVR_V4_COMPACT_FUTURE", "0")
    monkeypatch.setenv("SRLVR_V4_HINDSIGHT_NO_THINK", "1")
    monkeypatch.setattr(rollout, "_prompt_ids_for", lambda *a: [901, 902])
    episode, actions = _action_episode(tokenizer)
    rows = rollout.build_group_samples(
        tokenizer,
        None,
        [episode],
        rollout.OutcomeAnchor([0.0], [1.0]),
        0,
        with_hpi=False,
        future_gap=1,
        rollout_id=3,
    )
    newline = list(tokenizer.encode("\n"))
    expected = []
    for index, action in enumerate(actions):
        if index:
            expected.extend(newline)
        expected.extend(action)

    assert len(rows) == 3
    for index, row in enumerate(rows):
        # Student row: action tokens only, no think prefix.
        assert row.tokens == PROMPTS[index] + actions[index]
        assert row.response_length == len(actions[index])
        assert row.loss_mask == [1] * len(actions[index])
        assert row.rollout_log_probs == [-0.1] * len(actions[index])
        # Teacher continuation: executed actions only, newline separated.
        teacher, prompt_len, target_len = row.metadata["hsi_teacher_view"]
        assert prompt_len == 2
        assert target_len == len(expected)
        assert teacher[:prompt_len] == [901, 902]
        assert teacher[prompt_len:] == expected
        meta = row.metadata["srlvr_v4"]
        start, length = meta["span_start"], meta["span_len"]
        assert length == len(actions[index])
        assert teacher[prompt_len + start : prompt_len + start + length] == actions[index]


def test_hindsight_drop_removes_exactly_k_turns_and_is_reproducible(monkeypatch, tokenizer):
    """SRLVR_V4_HINDSIGHT_DROP_TURNS withholds k transitions from the privileged trace.

    The privileged block shows the whole episode, so the teacher could in
    principle recite turn t+1 while the student is scored on turn t. Dropping a
    random few transitions removes that option while keeping the trace useful.
    The choice must be stable for a given (task, episode, rollout) so a resumed
    run reproduces the same views.
    """
    monkeypatch.setenv("SRLVR_V4_COMPACT_FUTURE", "1")
    monkeypatch.setenv("SRLVR_V4_HINDSIGHT_DROP_TURNS", "2")
    first = _rows(tokenizer)
    dropped = first[0].metadata["srlvr_v4"]["hindsight_dropped"]
    assert len(dropped) == 2, dropped

    # Same inputs -> same withheld turns (seeded on task|episode|rollout).
    again = _rows(tokenizer)
    assert again[0].metadata["srlvr_v4"]["hindsight_dropped"] == dropped

    # The trace really lost those turns and kept the rest.
    block = render_trajectory_block(
        _episode(tokenizer).trajectory, compact_future=True,
    )
    full_n = block.count("[Future transition ")
    kept_block = render_trajectory_block(
        _episode(tokenizer).trajectory, compact_future=True,
        transitions=[
            t for t in _episode(tokenizer).trajectory.iter_tail_transitions(0)
            if t[0] not in set(dropped)
        ],
    )
    assert kept_block.count("[Future transition ") == full_n - 2
    for index in dropped:
        assert f"[Future transition {index}]" not in kept_block


def test_hindsight_drop_refuses_the_non_compact_renderer(monkeypatch, tokenizer):
    """Dropping needs the transition-list renderer; the legacy path must fail closed."""
    monkeypatch.setenv("SRLVR_V4_COMPACT_FUTURE", "0")
    monkeypatch.setenv("SRLVR_V4_HINDSIGHT_DROP_TURNS", "1")
    with pytest.raises(ValueError, match="requires SRLVR_V4_COMPACT_FUTURE=1"):
        rollout._v4_rows(tokenizer, _episode(tokenizer), 0, 0)


def test_batch_advantage_mode_centres_over_the_whole_batch(monkeypatch):
    """REINFORCE++ uses ONE baseline over all trajectories, not one per group.

    With 4 task slots x G=8, GRPO zeroes any group whose 8 samples agree (all-fail
    is the common case early on). Pooling the 32 trajectories means such a task is
    still judged against the batch instead of contributing nothing.
    """
    from slime.utils.types import Sample
    from tasks.alfworld import config, train_data

    def _sample(index, episode, reward, turns=2):
        s = Sample(index=index, prompt="")
        s.tokens = [1] * (turns + 1)
        s.response_length = turns
        s.loss_mask = [1] * turns
        s.reward = reward
        s.status = Sample.Status.COMPLETED
        s.opsd_advantages = [0.0] * turns
        s.turn_ids = [episode * config.TURN_ID_STRIDE] * turns
        s.fit_turn_weights = [1.0] * turns
        return s

    # Two tasks: group A all failures (GRPO advantage 0), group B 1 success + 1 failure.
    samples = [
        _sample(0, 1, 0.0), _sample(1, 2, 0.0),          # task A, both fail
        _sample(2, 3, 1.0), _sample(3, 4, 0.0),          # task B, mixed
    ]
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1)

    monkeypatch.setenv("ALFWORLD_V2_ADVANTAGE_MODE", "batch")
    data = train_data.convert_samples_to_train_data(args, samples)
    advs = [row[0] for row in data["opsd_advantages"]]
    # Rewards {0,0,1,0}: mean .25, population std sqrt(.1875)=.4330
    assert advs[0] == pytest.approx((0.0 - 0.25) / 0.4330127, rel=1e-4)
    assert advs[2] == pytest.approx((1.0 - 0.25) / 0.4330127, rel=1e-4)
    # A failing task is NOT zeroed any more -- that is the whole point.
    assert advs[0] != 0.0
    # The batch is centred.
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_group_advantage_mode_is_unchanged(monkeypatch):
    """The default must stay byte-identical: whatever the rollout wrote is kept."""
    from slime.utils.types import Sample
    from tasks.alfworld import config, train_data

    s = Sample(index=0, prompt="")
    s.tokens = [1, 2]
    s.response_length = 1
    s.loss_mask = [1]
    s.reward = 1.0
    s.status = Sample.Status.COMPLETED
    s.opsd_advantages = [0.7]
    s.turn_ids = [5 * config.TURN_ID_STRIDE]
    s.fit_turn_weights = [1.0]
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1)
    monkeypatch.delenv("ALFWORLD_V2_ADVANTAGE_MODE", raising=False)
    data = train_data.convert_samples_to_train_data(args, [s])
    assert data["opsd_advantages"] == [[0.7]]
