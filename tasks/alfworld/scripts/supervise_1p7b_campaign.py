#!/usr/bin/env python3
"""Persistent, one-arm supervisor for the two-node Qwen3-1.7B campaign."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[3]
STATE_ROOT = PROJECT.parents[1] / "logs" / "alfworld_1p7b_two_nodes" / "v3_fast"


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def metrics(log):
    if not log.exists():
        return {}
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 2_000_000))
        text = handle.read().decode(errors="replace")
    out = {"log_bytes": log.stat().st_size, "log_modified_at": log.stat().st_mtime}
    steps = re.findall(r"step (\d+): (\{[^\n]+\})", text)
    if steps:
        step, line = steps[-1]
        out["last_optimizer_step"] = int(step)
        match = re.search(r"'train/grad_norm': ([^,}]+)", line)
        if match:
            out["last_grad_norm"] = float(match[1])
        out["last_step_signature"] = f"{step}:{line}"
    sampled = re.findall(r"rollout\.py:\d+ - perf (\d+):", text)
    trained = re.findall(r"train_metric_utils\.py:\d+ - perf (\d+):", text)
    if sampled:
        out["last_sampled_rollout"] = int(sampled[-1])
    if trained:
        out["last_trained_rollout"] = int(trained[-1])
    return out


def validate_checkpoint(run_dir):
    tracker = run_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    try:
        iteration = int(tracker.read_text().strip())
    except (OSError, ValueError):
        return None
    command = ('PROJ="$1"; source "$PROJ/train_scripts/lib/npu_env.sh"; '
               'opsd_enter_env; opsd_export_npu_env "$PROJ"; '
               'python "$PROJ/tasks/alfworld/scripts/check_campaign_checkpoint.py" "$2" "$3"')
    checked = subprocess.run(
        ["bash", "-c", command, "validate", str(PROJECT), str(run_dir), str(iteration)],
        capture_output=True, text=True, timeout=120,
    )
    if checked.returncode:
        return None
    details = json.loads(checked.stdout.splitlines()[-1])
    details["load_path"] = str(run_dir)
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("grpo", "gspo"), required=True)
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    arm = args.arm
    worker = 0 if arm == "grpo" else 1
    if not os.uname().nodename.endswith(f"worker-{worker}"):
        raise RuntimeError(f"refusing {arm} on unexpected node {os.uname().nodename}")
    state = (STATE_ROOT if arm == "grpo" else STATE_ROOT.parent / "v4_gspo_clip3e4_lr1e6") / arm
    state.mkdir(parents=True, exist_ok=True)
    if (state / "STOP").exists() or (state / "COMPLETE").exists():
        return
    lock = open("/tmp/alfworld_1p7b_campaign_supervisor.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if args.ensure:
        lock.close()
        with (state / "daemon.log").open("ab", buffering=0) as out:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--arm", arm],
                stdout=out, stderr=out, stdin=subprocess.DEVNULL,
                start_new_session=True, close_fds=True,
            )
        return

    def log(message):
        with (state / "supervisor.log").open("a") as out:
            out.write(f"{time.strftime('%F %T')} {message}\n")

    run = f"alfworld_v2_{arm}_1p7b_200_6r2a_brief64_cache1_serial1"
    if arm == "gspo":
        run += "_clip3e4_4e4_lr1e6"
    run_dir = PROJECT / "runs" / run
    train_log = PROJECT / "logs" / f"{run}.log"
    snapshot = state / "launch_snapshot"
    snapshot.mkdir(exist_ok=True)
    wrapper_name = f"run_alfworld_v2_{arm}_1p7b.sh"
    for name in (wrapper_name, "_common_v2.sh"):
        target = snapshot / name
        if not target.exists():
            shutil.copy2(PROJECT / "tasks/alfworld/scripts" / name, target)
    wrapper = snapshot / wrapper_name
    process = None
    attempt = 0
    chunk = 128
    last_progress = time.time()
    previous_signature = None
    next_launch = 0
    attempt_start = 0
    log_offset = train_log.stat().st_size if train_log.exists() else 0
    log(f"supervisor started pid={os.getpid()} host={os.uname().nodename}")

    while True:
        try:
            now = time.time()
            details = metrics(train_log)
            signature = (details.get("last_step_signature"), details.get("last_sampled_rollout"))
            if signature != previous_signature:
                previous_signature, last_progress = signature, now
            status = {
                "arm": arm, "run": run, "host": os.uname().nodename,
                "supervisor_pid": os.getpid(), "heartbeat_at": time.strftime("%F %T"),
                "heartbeat_epoch": now, "attempt": attempt, "log_probs_chunk_size": chunk,
                **{k: v for k, v in details.items() if k != "last_step_signature"},
            }
            checkpoint = validate_checkpoint(run_dir)
            tracker = run_dir / "latest_checkpointed_iteration.txt"
            if tracker.exists():
                status["checkpoint_iteration"] = tracker.read_text().strip()
            if (state / "STOP").exists():
                if process and process.poll() is None:
                    os.killpg(process.pid, 15)
                status["phase"] = "stopped"
                atomic_json(state / "status.json", status)
                return
            if process and process.poll() is None:
                status.update(phase="running", launcher_pid=process.pid)
                stale_log = now - max(details.get("log_modified_at", attempt_start), attempt_start) > 1800
                stalled = now - last_progress > 10800
                grad = details.get("last_grad_norm", 0.0)
                if stale_log or stalled or not math.isfinite(grad):
                    log(f"stalled/nonfinite training; terminating launcher pid={process.pid}")
                    os.killpg(process.pid, 15)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, 9)
                    next_launch = now + 120
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            if process:
                log(f"launcher exited rc={process.returncode}; checking checkpoint before recovery")
                with train_log.open("rb") as handle:
                    handle.seek(max(log_offset, train_log.stat().st_size - 300_000))
                    errors = handle.read().decode(errors="replace")
                if "out of memory" in errors.lower():
                    chunk = max(16, chunk // 2)
                    log(f"OOM recovery: execution chunk={chunk}")
                process = None
                next_launch = max(next_launch, now + min(600, 30 * attempt))
            if checkpoint and checkpoint["iteration"] >= 199:
                (state / "COMPLETE").write_text(time.strftime("%F %T") + "\n")
                status["phase"] = "complete"
                atomic_json(state / "status.json", status)
                log("200 rollouts complete; final checkpoint validated")
                return
            if now < next_launch:
                status["phase"] = "restart_backoff"
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            alive = []
            for cmd in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    argv = cmd.read_bytes().split(b"\0")
                    driver = (any(b"slime-ascend/train.py" in x for x in argv)
                              and str(run_dir).encode() in argv)
                    shell = str(wrapper).encode() in argv or str(snapshot / "_common_v2.sh").encode() in argv
                    if driver or shell:
                        alive.append(int(cmd.parent.name))
                except (OSError, ValueError):
                    pass
            if alive:
                status.update(phase="adopted", trainer_pids=alive)
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            env = dict(os.environ)
            for key in ("RESUME_FROM", "LOAD_DEBUG_ROLLOUT_DATA", "DRY_RUN_PARSE",
                        "SAVE_DIR", "LOG", "PYTHONPATH"):
                env.pop(key, None)
            env.update(
                RUN_NAME=run, NUM_ROLLOUT="200", SAVE_INTERVAL="50", SAVE_TRAIN_STATE="1",
                MAX_TURNS="50", MAX_RESPONSE="128", HISTORY_LENGTH="0", SEQ_LENGTH="32768",
                SEED="42", ALFWORLD_V2_PROTOCOL="1",
                MODEL=str(PROJECT.parents[1] / "checkpoints/Qwen3-1.7B"),
                ROLLOUT_GPUS="6", ACTOR_GPUS="2",
                EPS_CLIP_HIGH="0.2", SGLANG_PREFIX_CACHE="1",
                ALFWORLD_THINKING_STYLE="brief",
                ALFWORLD_V2_HSI_CANDIDATE_K="0",
            )
            if arm == "gspo":
                env.update(EPS_CLIP="0.0003", EPS_CLIP_HIGH="0.0004", LR="1e-6")
            if checkpoint:
                env["RESUME_FROM"] = checkpoint["load_path"]
            cleanup = ('PROJ="$1"; source "$PROJ/train_scripts/lib/npu_env.sh"; '
                       'opsd_enter_env; opsd_export_npu_env "$PROJ"; ray stop --force')
            cleared = subprocess.run(
                ["bash", "-c", cleanup, "cleanup", str(PROJECT)],
                env=env, capture_output=True, text=True, timeout=120,
            )
            if cleared.returncode:
                log(f"Ray cleanup failed rc={cleared.returncode}: {cleared.stderr[-400:]}")
                next_launch = now + 120
                continue
            attempt += 1
            last_progress = attempt_start = time.time()
            log_offset = train_log.stat().st_size if train_log.exists() else 0
            with (state / "launcher.log").open("ab", buffering=0) as out:
                process = subprocess.Popen(
                    ["bash", str(wrapper)], cwd=PROJECT, env=env,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                    start_new_session=True, close_fds=True,
                )
            log(f"launch attempt={attempt} pid={process.pid} checkpoint={checkpoint}")
            status.update(phase="starting", launcher_pid=process.pid, attempt=attempt)
            atomic_json(state / "status.json", status)
        except Exception:
            log(traceback.format_exc())
        time.sleep(30)


if __name__ == "__main__":
    main()
