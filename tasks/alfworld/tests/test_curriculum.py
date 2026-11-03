from types import SimpleNamespace

import pytest

from tasks.alfworld.curriculum import Curriculum, FAMILIES, success_probability, task_weight


def pool():
    return [SimpleNamespace(game_id=f"{family}/{i}", task_type=family)
            for family in FAMILIES for i in range(3)]


def test_group_weights_and_single_rollout_reward_second_moment():
    assert task_weight(.5, "grpo", 8) == pytest.approx(1 - 2 * .5 ** 8)
    assert task_weight(.05, "grpo", 8) < task_weight(.5, "grpo", 8)
    assert task_weight(.9, "srlvr", 1, .2) == pytest.approx(.9 * .8 ** 2 + .1 * .2 ** 2)
    assert task_weight(.01, "srlvr", 1) > 0
    with pytest.raises(ValueError):
        task_weight(.5, "grpo", 1)


def test_grpo_rotates_family_slots_and_keeps_four_tasks(tmp_path):
    sampler = Curriculum(tmp_path)
    batches = [sampler.sample(pool(), k, 4, 42, "grpo", 8)[0] for k in range(3)]
    assert [g.task_type for batch in batches for g in batch] == list(FAMILIES) * 2


def test_single_rollout_mixes_four_families_with_iid_positions(tmp_path):
    sampler = Curriculum(tmp_path)
    for k in range(len(FAMILIES)):
        games, seeds = sampler.sample(pool(), k, 32, 42, "srlvr", 1)
        start = (k * 4) % len(FAMILIES)
        expected = [FAMILIES[(start + (i % 4)) % len(FAMILIES)] for i in range(32)]
        assert [g.task_type for g in games] == expected
        counts = {family: expected.count(family) for family in set(expected)}
        assert len(counts) == 4
        assert set(counts.values()) == {8}
        # Calibration positions 0-3 cover exactly the same four-family mixture
        # as the update positions 4-31.
        assert {g.task_type for g in games[:4]} == set(expected[:4])
        assert len(seeds) == 32
        assert sampler.sample(pool(), k, 32, 42, "srlvr", 1) == (games, seeds)


def test_discount_history_replay_and_checkpoint_rewind(tmp_path):
    sampler = Curriculum(tmp_path)
    game = pool()[0]
    sampler.observe(0, [(game, [1., 0., 1., 1.])])
    assert sampler.prior(0) == {"tasks": {}, "families": {}}
    assert success_probability(sampler.prior(1)["tasks"][game.game_id]) == pytest.approx(4 / 6)
    expected = sampler.sample(pool(), 1, 4, 42, "grpo", 8)
    sampler.observe(1, [(game, [0.])])
    assert sampler.prior(2)["tasks"][game.game_id] == pytest.approx([2.7, 4.6])
    sampler.observe(2, [(game, [1.])])
    assert Curriculum(tmp_path).sample(pool(), 1, 4, 42, "grpo", 8) == expected
    sampler.observe(1, [(game, [0.])])
    assert not (sampler.directory / "0000002.json").exists()
    assert sampler.prior(2)["tasks"][game.game_id] == pytest.approx([2.7, 4.6])
