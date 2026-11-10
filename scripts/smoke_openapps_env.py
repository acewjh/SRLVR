#!/usr/bin/env python3
"""Smoke: drive a real OpenApps task through the browsergym environment.

Proves the environment half of the GUI-GRPO integration before any model is
involved: the OpenApps app server is up, the task registers with browsergym,
reset yields a screenshot + goal, and step returns a reward.

Usage:
    python scripts/smoke_openapps_env.py --base-url http://localhost:5055 \
        [--task add_meeting_with_dennis]
"""

from __future__ import annotations

import argparse
import sys

from open_apps.agent.env_args import DeviceEnvArgs
from open_apps.tasks import list_task_keys, load_task
from open_apps.tasks.add_tasks_to_browsergym import register_tasks_with_browsergym


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--task", default="add_meeting_with_dennis")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    keys = list_task_keys()
    assert args.task in keys, f"{args.task} not in {len(keys)} tasks"
    task = load_task(args.task)
    register_tasks_with_browsergym([task])
    print(f"[smoke] registered task {task.task_id!r}, goal={task.goal!r}")

    viewport = {"width": 1920, "height": 1080}
    env_args = DeviceEnvArgs(
        task_name=task.task_id,
        task_seed=42,
        max_steps=30,
        headless=True,
        task_kwargs={
            "base_url": args.base_url,
            "screen_resolution": (viewport["width"], viewport["height"]),
        },
    )
    from pathlib import Path
    import tempfile

    from browsergym.core.action.highlevel import HighLevelActionSet

    # Pixel-level computer-use actions (mouse_click(x, y), keyboard_type, ...):
    # the policy sees only the screenshot, exactly like the OSWorld arm.
    action_set = HighLevelActionSet(subsets=["coord"], strict=False, multiaction=False)
    exp_dir = Path(tempfile.mkdtemp(prefix="openapps_smoke_"))
    env = env_args.make_env(action_mapping=action_set.to_python_code, exp_dir=exp_dir)
    obs, info = env.reset()
    shot = obs["screenshot"]
    print(f"[smoke] reset ok: screenshot={getattr(shot, 'shape', None) or shot.size} goal={info.get('goal')!r}")

    for i in range(args.steps):
        obs, reward, terminated, truncated, info = env.step("noop()")
        print(
            f"[smoke] step {i}: reward={reward} terminated={terminated} "
            f"truncated={truncated} shot={obs['screenshot'].size}"
        )
        if terminated or truncated:
            break
    env.close()
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
