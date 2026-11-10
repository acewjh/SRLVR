"""Small policy-versioned pool used by the custom ALFWorld rollout.

The Slime ``RolloutDataSourceWithBuffer`` is a prompt/sample buffer owned by
the rollout manager.  The ALFWorld function receives a data source but creates
episodes itself, so that buffer cannot provide continual collection here.  This
module keeps the same useful contract locally: episodes may finish in any
order, but a complete batch is drained in scheduled order before scoring.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


class BatchIncomplete(RuntimeError):
    """Raised when a caller tries to train before a pool is complete."""


class PoolContractError(RuntimeError):
    """Raised when a result cannot belong to this policy-versioned batch."""


@dataclass(frozen=True)
class _Entry(Generic[T]):
    policy_version: int
    value: T


class EpisodePool(Generic[T]):
    """Collect exactly one batch for one frozen policy version."""

    def __init__(self, policy_version: int, expected: int, capacity: int | None = None):
        if expected < 1:
            raise ValueError("expected must be positive")
        self.policy_version = int(policy_version)
        self.expected = int(expected)
        self.capacity = self.expected if capacity is None else int(capacity)
        if self.capacity < self.expected:
            raise ValueError("capacity must be at least expected")
        self._entries: dict[int, _Entry[T]] = {}

    def add(self, position: int, value: T, *, policy_version: int | None = None) -> None:
        position = int(position)
        version = self.policy_version if policy_version is None else int(policy_version)
        if version != self.policy_version:
            raise PoolContractError(
                f"policy version mismatch: {version} != {self.policy_version}"
            )
        if position < 0 or position >= self.capacity:
            raise PoolContractError(f"position {position} outside [0, {self.capacity})")
        if position in self._entries:
            raise PoolContractError(f"duplicate batch position {position}")
        self._entries[position] = _Entry(version, value)

    @property
    def complete(self) -> bool:
        return all(index in self._entries for index in range(self.expected))

    def drain(self) -> list[T]:
        if not self.complete:
            missing = sorted(set(range(self.expected)) - self._entries.keys())
            raise BatchIncomplete(f"batch has {len(self._entries)}/{self.expected}; missing {missing}")
        values = [self._entries[index].value for index in range(self.expected)]
        for index in range(self.expected):
            del self._entries[index]
        return values


async def collect_ordered(
    factories: list[Callable[[], Awaitable[T]]],
    *,
    policy_version: int,
    max_inflight: int | None = None,
    pool_capacity: int | None = None,
) -> list[T]:
    """Run factories concurrently and return results in factory order.

    Completion order never determines task position.  A failed factory cancels
    outstanding work so a partial batch cannot accidentally reach training.
    """
    if not factories:
        return []
    limit = len(factories) if max_inflight is None else int(max_inflight)
    if limit < 1:
        raise ValueError("max_inflight must be positive")
    limit = min(limit, len(factories))
    pool: EpisodePool[T] = EpisodePool(policy_version, len(factories), pool_capacity)
    pending: dict[asyncio.Task[T], int] = {}
    next_position = 0

    def start_one(position: int) -> None:
        pending[asyncio.create_task(factories[position]())] = position

    while next_position < limit:
        start_one(next_position)
        next_position += 1
    try:
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                position = pending.pop(task)
                pool.add(position, task.result(), policy_version=policy_version)
                if next_position < len(factories):
                    start_one(next_position)
                    next_position += 1
    except BaseException:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise
    return pool.drain()
