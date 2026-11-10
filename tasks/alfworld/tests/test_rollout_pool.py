import asyncio

import pytest

from tasks.alfworld.pool import BatchIncomplete, EpisodePool, PoolContractError, collect_ordered


def test_pool_requires_complete_batch_and_drains_in_position_order():
    pool = EpisodePool(policy_version=7, expected=3)
    pool.add(2, "slow")
    pool.add(0, "first")
    with pytest.raises(BatchIncomplete):
        pool.drain()
    pool.add(1, "middle")
    assert pool.complete
    assert pool.drain() == ["first", "middle", "slow"]
    assert not pool.complete


def test_pool_rejects_duplicate_out_of_range_and_mixed_policy():
    pool = EpisodePool(policy_version=4, expected=2)
    pool.add(0, "a")
    with pytest.raises(PoolContractError):
        pool.add(0, "duplicate")
    with pytest.raises(PoolContractError):
        pool.add(2, "out-of-range")
    with pytest.raises(PoolContractError):
        pool.add(1, "new-policy", policy_version=5)


def test_pool_can_declare_two_batch_capacity_but_requires_one_batch_to_drain():
    pool = EpisodePool(policy_version=9, expected=2, capacity=4)
    pool.add(3, "future")
    pool.add(0, "first")
    with pytest.raises(BatchIncomplete):
        pool.drain()
    pool.add(1, "second")
    assert pool.drain() == ["first", "second"]


def test_collect_ordered_preserves_schedule_and_bounds_inflight():
    active = 0
    peak = 0

    async def make(position):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep((3 - position) * 0.001)
        active -= 1
        return position

    factories = [lambda position=position: make(position) for position in range(6)]
    assert asyncio.run(collect_ordered(factories, policy_version=11, max_inflight=2)) == list(range(6))
    assert peak <= 2
