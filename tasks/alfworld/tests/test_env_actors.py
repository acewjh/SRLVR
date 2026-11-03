"""Minimal CPU tests for the persistent ALFWorld env-actor pool.

1. Operation semantics must match the in-process ``AlfWorldEnv`` byte for byte.
2. 32 episodes must run on 32 independent actor processes at the same time
   without parser races, deadlocks, or errors.

No NPU / SGLang is touched: the actors only speak TextWorld over IPC.
"""

import asyncio
import json
import os
import time

import pytest

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCHEDULE = os.path.join(R, "tasks", "alfworld", "configs", "train_schedule_seed42.jsonl")


def _game():
    from tasks.alfworld.v1.env_adapter import GameRef

    with open(SCHEDULE, encoding="utf-8") as handle:
        task = json.loads(handle.readline())["tasks"][0]
    if not os.path.exists(task["path"]):
        pytest.skip("ALFWorld game data not present")
    return GameRef(
        game_id=task["game_id"],
        path=task["path"],
        task_type=task.get("task_type", ""),
        split=task.get("split", "train"),
    )


def _enable(workers: int) -> None:
    from tasks.alfworld import env_actors

    env_actors.shutdown_pool()
    os.environ["ALFWORLD_ENV_ACTORS"] = "1"
    os.environ["ALFWORLD_ENV_WORKERS"] = str(workers)


def test_task_defaults_enable_actors_and_admissible_list(monkeypatch):
    """Both 2026-10-27 improvements are on by default, no env vars required."""
    from tasks.alfworld import env_actors, prompts

    monkeypatch.delenv("ALFWORLD_ENV_ACTORS", raising=False)
    monkeypatch.delenv("ALFWORLD_SHOW_ADMISSIBLE", raising=False)
    assert env_actors.env_actors_enabled()
    assert prompts.show_admissible()

    view = prompts.TaskView("put it away", "room", ["go to shelf 1", "look"], [])
    assert "go to shelf 1" in prompts.build_turn_user_message(view)
    monkeypatch.setenv("ALFWORLD_SHOW_ADMISSIBLE", "0")
    assert "go to shelf 1" not in prompts.build_turn_user_message(view)


def test_actor_semantics_match_in_process():
    from tasks.alfworld import env_actors
    from tasks.alfworld.v1.env_adapter import AlfWorldEnv

    game = _game()
    actions = ["look", "inventory", "go to shelf 1", "look", "go to shelf 2"]
    _enable(workers=1)

    reference = AlfWorldEnv(game, 50)
    try:
        first = reference.reset()
        expected = [(first.observation, first.admissible_actions, first.done, first.won)]
        for action in actions:
            step = reference.step(action)
            expected.append((step.observation, step.admissible_actions, step.done, step.won))
    finally:
        reference.close()

    async def run():
        env = await env_actors.acquire_env(game, 50)
        try:
            reset = await env.reset()
            assert env.task_description == reference.task_description
            got = [(reset.observation, reset.admissible_actions, reset.done, reset.won)]
            for action in actions:
                step = await env.step(action)
                got.append((step.observation, step.admissible_actions, step.done, step.won))
            return got
        finally:
            await env.aclose()

    assert asyncio.run(run()) == expected
    env_actors.shutdown_pool()


def test_32_episodes_run_concurrently():
    from tasks.alfworld import env_actors

    game = _game()
    _enable(workers=32)

    async def one(_index: int) -> int:
        env = await env_actors.acquire_env(game, 50)
        try:
            await env.reset()
            steps = 0
            for action in ("look", "inventory", "look"):
                await env.step(action)
                steps += 1
            return steps
        finally:
            await env.aclose()

    async def run():
        started = time.perf_counter()
        results = await asyncio.gather(*[one(i) for i in range(32)])
        return results, time.perf_counter() - started

    results, elapsed = asyncio.run(run())
    assert results == [3] * 32
    # 32 sequential in-process resets alone would take ~32 * 8 s; the point of the
    # pool is that they overlap. Generous bound to keep the test non-flaky.
    print(f"[env-actors] 32 concurrent episodes: {elapsed:.1f}s")
    assert elapsed < 120
    env_actors.shutdown_pool()
