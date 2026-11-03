"""The rollout must always hand the trainer at least one optimizer step's rows.

A rollout that produces fewer turn-rows than ``global_batch_size`` makes slime's
trim raise "Not enough samples", which kills the run; with ``save_interval=25``
the restart-from-cp0 that follows loses every rollout since the last checkpoint.
These tests pin the same-task top-up that prevents it. CPU-only: ``_run_group``
is stubbed, so no engine, tokenizer or NPU is involved.
"""

import asyncio
import types

import pytest

from tasks.alfworld import rollout as R


class FakeEpisode:
    def __init__(self, num_turns: int):
        # ``_turn_rows`` only reads ``.turns``; the rest of the sample builder is
        # never reached because ``_top_up_to_floor`` does not build samples.
        self.turns = [object()] * num_turns
        self.trajectory = types.SimpleNamespace(success=False, turns=self.turns)


def _make_run_group(turns_per_episode: int, calls: list):
    """Stub ``_run_group`` returning ``len(games)`` episodes of a fixed size."""

    async def _run_group(args, state, game, seed_base, group_id, engines):
        calls.append({"seed_base": seed_base, "group_id": group_id})
        return [FakeEpisode(turns_per_episode)], object()

    return _run_group


def _drive(monkeypatch, *, turns_per_episode, per_round, min_rows, games=4):
    """Run the floor once with a stubbed ``_run_group`` and return the outcome."""
    calls: list = []
    monkeypatch.setattr(R, "_run_group", _make_run_group(per_round, calls))
    seeds = list(range(1000, 1000 + games))
    group_ids = list(range(games))
    group_games = ["game"] * games
    # ``_generate_rollout_async`` seeds ``groups`` from the first round; reproduce
    # that shape so only the top-up path is under test.
    groups = [([FakeEpisode(turns_per_episode)], object()) for _ in range(games)]

    topup = asyncio.run(
        R._top_up_to_floor(
            args=None,
            state=None,
            games=["game"] * games,
            seeds=seeds,
            engines=["engine"],
            groups=groups,
            group_games=group_games,
            group_ids=group_ids,
            min_rows=min_rows,
        )
    )
    return types.SimpleNamespace(
        topup=topup,
        groups=groups,
        group_games=group_games,
        group_ids=group_ids,
        calls=calls,
        seeds=seeds,
    )


def test_turn_rows_sums_every_group():
    # Arrange
    groups = [([FakeEpisode(3), FakeEpisode(2)], None), ([FakeEpisode(4)], None)]

    # Act
    total = R._turn_rows(groups)

    # Assert
    assert total == 9


def test_no_top_up_when_the_floor_is_already_met(monkeypatch):
    # Arrange: 4 episodes x 10 turns = 40 rows, floor 32.
    # Act
    out = _drive(monkeypatch, turns_per_episode=10, per_round=10, min_rows=32)

    # Assert
    assert out.topup == 0
    assert len(out.groups) == 4
    assert out.calls == []


def test_zero_floor_disables_the_top_up(monkeypatch):
    # Arrange: 1 row per episode, floor 0 means "off".
    # Act
    out = _drive(monkeypatch, turns_per_episode=1, per_round=1, min_rows=0)

    # Assert
    assert out.topup == 0
    assert out.calls == []


def test_top_up_adds_same_task_episodes_until_the_floor_is_met(monkeypatch):
    # Arrange: 4 episodes x 4 turns = 16 rows against a floor of 32 -> one round
    # of 4 more episodes (16 more rows) clears it.
    # Act
    out = _drive(monkeypatch, turns_per_episode=4, per_round=4, min_rows=32)

    # Assert
    assert out.topup == 1
    assert len(out.groups) == 8
    # Strictly more rows, still the same four tasks, ids unique across rounds.
    assert R._turn_rows(out.groups) >= 32
    assert out.group_games == ["game"] * 8
    assert sorted(out.group_ids) == list(range(8))
    # Four extra episodes, one per scheduled task.
    assert len(out.calls) == 4


def test_top_up_uses_fresh_seeds_for_each_round(monkeypatch):
    # Arrange: 4 episodes x 4 turns -> 16 rows vs floor 32 needs exactly 1 round.
    # Act
    out = _drive(monkeypatch, turns_per_episode=4, per_round=4, min_rows=32)

    # Assert: the top-up round must not replay round 0's seeds.
    round_seeds = [call["seed_base"] for call in out.calls]
    assert all(seed >= min(out.seeds) + R._TOPUP_SEED_STRIDE for seed in round_seeds)


def test_top_up_is_capped_when_the_policy_stays_short(monkeypatch):
    # Arrange: episodes never grow, so the floor is unreachable. 4 games.
    # Act
    out = _drive(monkeypatch, turns_per_episode=1, per_round=1, min_rows=10_000)

    # Assert: bounded effort, then hand back whatever there is (the caller's
    # downstream guard still owns the "genuinely broken" case).
    assert out.topup == R._MAX_TOPUP_ROUNDS
    assert len(out.groups) == 4 * (1 + R._MAX_TOPUP_ROUNDS)
    # Group ids stay unique and in range.
    assert sorted(out.group_ids) == list(range(len(out.groups)))


def test_top_up_seed_stride_separates_rounds(monkeypatch):
    # Arrange: two rounds of top-up needed to clear a floor of 48.
    # Act
    out = _drive(monkeypatch, turns_per_episode=4, per_round=4, min_rows=48)

    # Assert
    assert out.topup == 2
    seeds_by_round = {}
    for call in out.calls:
        seeds_by_round.setdefault(call["group_id"] // 4, set()).add(call["seed_base"])
    assert len(seeds_by_round[1]) == 4 and len(seeds_by_round[2]) == 4
    assert seeds_by_round[1].isdisjoint(seeds_by_round[2])


# ---------------------------------------------------------------------------
# The floor's two checks must agree on what a "row" is
# ---------------------------------------------------------------------------


class FakeTurn:
    """The subset of a turn that ``build_group_samples`` reads (non-HPI path)."""

    def __init__(self, turn_idx: int, episode_id: int, response_length: int = 3):
        self.turn_idx = turn_idx
        self.episode_id = episode_id
        self.prompt_token_ids = [11, 12, 13]
        self.generated_token_ids = list(range(100 + turn_idx * response_length, 100 + (turn_idx + 1) * response_length))
        self.generated_log_probs = [-0.5] * response_length


class BuildableEpisode:
    """An episode rich enough for the real ``build_group_samples``."""

    def __init__(self, num_turns: int, episode_id: int = 7):
        self.turns = [FakeTurn(t, episode_id) for t in range(num_turns)]
        self.reward = 1.0
        self.trajectory = types.SimpleNamespace(
            success=True, truncated=False, turns=self.turns, to_dict=lambda: {"turns": num_turns}
        )


class FakeAdvantages:
    def __init__(self, n: int):
        self.advantages = [0.5] * n
        self.mixed = True


def _rows_for(episodes):
    return R.build_group_samples(
        None,
        "game",
        episodes,
        FakeAdvantages(len(episodes)),
        group_id=0,
        with_hpi=False,
        future_gap=1,
        sibling_pool=[],
    )


def test_one_turn_becomes_exactly_one_training_row():
    # Arrange: the pre-build gate counts turns, slime's trim counts the flattened
    # Sample rows it is handed. They are only interchangeable while this holds.
    episodes = [BuildableEpisode(num_turns=5, episode_id=1)]

    # Act
    rows = _rows_for(episodes)

    # Assert
    assert len(rows) == sum(len(e.turns) for e in episodes) == 5


def test_turn_rows_matches_the_emitted_row_count_across_groups():
    # Arrange: the same equality, at the granularity the floor gate uses.
    groups = [
        ([BuildableEpisode(num_turns=2, episode_id=0)], None),
        ([BuildableEpisode(num_turns=4, episode_id=1)], None),
    ]

    # Act: the gate's count vs. what slime would flatten out of the returned data.
    gate_count = R._turn_rows(groups)
    emitted = sum(len(_rows_for(episodes)) for episodes, _adv in groups)

    # Assert
    assert gate_count == emitted == 6


def test_top_up_round_is_the_shared_mechanism_for_both_floor_checks(monkeypatch):
    # Arrange: the second (emitted-row) check reuses the same round helper as the
    # first, so one round must behave identically from either entry point.
    calls: list = []
    monkeypatch.setattr(R, "_run_group", _make_run_group(4, calls))
    games = ["game"] * 4
    seeds = list(range(1000, 1004))
    groups: list = []
    group_games: list = []
    group_ids: list = []

    # Act
    asyncio.run(
        R._top_up_round(
            args=None,
            state=None,
            games=games,
            seeds=seeds,
            engines=["engine"],
            groups=groups,
            group_games=group_games,
            group_ids=group_ids,
            topup_round=1,
        )
    )

    # Assert: same four tasks, fresh seeds, ids that do not collide with round 0.
    assert len(groups) == 4
    assert group_games == games
    assert group_ids == [4, 5, 6, 7]
    assert all(call["seed_base"] >= min(seeds) + R._TOPUP_SEED_STRIDE for call in calls)


def test_second_floor_check_stops_asking_once_the_emitted_rows_clear_it(monkeypatch):
    # Arrange: driving _top_up_round twice then measuring in emitted-row units is
    # exactly what the second check does; it must terminate on the real count.
    calls: list = []

    async def _run_group(args, state, game, seed_base, group_id, engines):
        calls.append({"seed_base": seed_base, "group_id": group_id})
        return [BuildableEpisode(4, episode_id=group_id)], FakeAdvantages(1)

    monkeypatch.setattr(R, "_run_group", _run_group)
    games = ["game"] * 4
    seeds = list(range(1000, 1004))
    groups = [([BuildableEpisode(1, episode_id=i)], None) for i in range(4)]
    group_games = list(games)
    group_ids = list(range(4))

    # Act
    rounds = 0
    while sum(len(_rows_for(eps)) for eps, _adv in groups) < 32 and rounds < R._MAX_TOPUP_ROUNDS:
        rounds += 1
        asyncio.run(
            R._top_up_round(
                args=None,
                state=None,
                games=games,
                seeds=seeds,
                engines=["engine"],
                groups=groups,
                group_games=group_games,
                group_ids=group_ids,
                topup_round=rounds,
            )
        )

    # Assert: 4 one-turn rows + rounds x 4 four-turn rows reaches 32 at round 2.
    assert rounds == 2
    assert sum(len(_rows_for(eps)) for eps, _adv in groups) == 4 + 2 * 16
