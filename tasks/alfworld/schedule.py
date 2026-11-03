"""Frozen task schedule -- both nodes must see the identical tasks per step.

The two arms are a matched pair, so they must agree on *which tasks* are trained at
each rollout, not just on the aggregate distribution. A schedule generated once and
read by both nodes removes any dependence on each node's shuffle state, DataSource
ordering or restart history.

Layout: one JSON object per rollout step::

    {"step": 0, "tasks": [{"game_id": ..., "game_path": ..., "task_type": ...,
                           "split": ...}, ...],
     "seeds": [<int per task>]}

``seeds`` are the per-task sampling seeds; the rollout derives each episode's seed
from (step, task, episode index) so a restart at the same step reproduces it.
"""

import json
import os
import random
from pathlib import Path

from tasks.alfworld.v1.env_adapter import GameRef, load_game_index

__all__ = ["build_schedule", "load_schedule", "schedule_length", "task_for_step"]

DEFAULT_SEED = 42


def build_schedule(
    out_path: str,
    *,
    index_path: str,
    num_steps: int,
    tasks_per_step: int,
    seed: int = DEFAULT_SEED,
) -> int:
    """Write a deterministic schedule; returns the number of steps written."""
    index = load_game_index(index_path)
    pool: list[GameRef] = list(index.get("train", []))
    if not pool:
        raise ValueError(f"no train games in {index_path}")
    rng = random.Random(seed)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for step in range(num_steps):
            # Sample WITHOUT replacement inside a step so a step never trains the
            # same task twice (which would make its group degenerate by design).
            chosen = rng.sample(pool, min(tasks_per_step, len(pool)))
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "tasks": [game.to_dict() for game in chosen],
                        "seeds": [rng.randrange(0, 2**31 - 1) for _ in chosen],
                    }
                )
                + "\n"
            )
    return num_steps


def load_schedule(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def schedule_length(path: str) -> int:
    return len(load_schedule(path))


def task_for_step(path: str, step: int) -> tuple[list[GameRef], list[int]]:
    """``(games, seeds)`` for one rollout step, wrapping around past the end.

    Wrapping keeps a long run alive if ``--num-rollout`` exceeds the schedule length,
    and is deterministic either way, so both nodes still agree.
    """
    schedule = load_schedule(path)
    if not schedule:
        raise ValueError(f"empty schedule at {path}")
    entry = schedule[step % len(schedule)]
    games = [GameRef(**task) for task in entry["tasks"]]
    return games, list(entry["seeds"])


def task_batch(path: str, step: int, count: int, seed: int, args=None):
    """Common task sampler. The existing frozen-schedule mode stays unchanged."""
    mode = os.environ.get("ALFWORLD_TASK_SAMPLING", "frontier")
    if mode == "schedule":
        games, seeds = task_for_step(path, step)
        return games[:count], seeds[:count]
    if mode not in ("iid", "frontier"):
        raise ValueError("ALFWORLD_TASK_SAMPLING must be schedule, iid or frontier")
    index = os.environ.get("ALFWORLD_GAME_INDEX", str(Path(__file__).parent / "v1/configs/game_index.json"))
    pool = load_game_index(index)["train"]
    if mode == "frontier":
        from .curriculum import current_curriculum
        return current_curriculum(args).sample(
            pool, step, count, seed,
            os.environ.get("ALFWORLD_CURRICULUM_METHOD", "grpo"),
            args.n_samples_per_prompt,
        )
    rng = random.Random(seed + 1_000_003 * step)
    games = [rng.choice(pool) for _ in range(count)]
    return games, [rng.randrange(2**31-1) for _ in games]
