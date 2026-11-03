"""Lightweight algorithm x task x seed loop over the shared launcher."""
import argparse
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import time


def commands(algorithms, tasks, seeds, rounds, batch_size, stamp, mode):
    root = Path(__file__).resolve().parents[1]
    for algorithm, task, seed in itertools.product(algorithms, tasks, seeds):
        if task != "alfworld" or algorithm not in ("grpo", "gspo", "srlvr"):
            raise ValueError("verified common entry supports ALFWorld with grpo/gspo/srlvr")
        group = 1 if algorithm == "srlvr" else 8
        if batch_size <= 0 or batch_size % group:
            raise ValueError("real trajectory batch size must be positive and divisible by group size")
        tasks_per_round = batch_size // group
        variant = f"{mode}-" if algorithm == "srlvr" else ""
        run = f"{task}-qwen3-1p7b-{algorithm}-{variant}s{seed}-r{rounds}-n{tasks_per_round}-{stamp}"
        env = dict(ARM=algorithm, SEED=str(seed), NUM_ROLLOUT=str(rounds),
                   ROLLOUT_BS=str(tasks_per_round), RUN_NAME=run, SRLVR_MODE=mode,
                   ACTOR_GPUS="2", ROLLOUT_GPUS="6", SGLANG_PREFIX_CACHE="1",
                   ALFWORLD_THINKING_STYLE="off", MAX_RESPONSE="128", HISTORY_LENGTH="0", LR="1e-6")
        yield env, ["bash", str(root / "tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh")]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithms", nargs="+", default=["grpo", "gspo", "srlvr"])
    p.add_argument("--tasks", nargs="+", default=["alfworld"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32, help="real trajectories per rollout (GRPO tasks=batch/8, SRLVR tasks=batch)")
    p.add_argument("--srlvr-mode", choices=["zero", "fixed", "empirical", "certified"], default="empirical")
    p.add_argument("--execute", action="store_true", help="otherwise print commands only")
    p.add_argument("--parse-only", action="store_true", help="actual parsers, no training")
    a = p.parse_args()
    for env, argv in commands(a.algorithms, a.tasks, a.seeds, a.rounds, a.batch_size, time.strftime("%Y%m%d_%H%M%S"), a.srlvr_mode):
        print(shlex.join(["env", *[f"{k}={v}" for k, v in env.items()], *argv]), flush=True)
        if a.execute or a.parse_only:
            if a.parse_only:
                env["DRY_RUN_PARSE"] = "1"
            subprocess.run(argv, env={**os.environ, **env}, check=True)
