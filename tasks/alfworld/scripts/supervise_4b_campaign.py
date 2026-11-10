#!/usr/bin/env python3
"""One independent, persistent GRPO/GSPO supervisor per ModelArts worker."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[3]
STATE_ROOT = PROJECT.parents[1] / "logs" / "alfworld_4b_two_nodes" / "v2"
ENV_SCRIPT = PROJECT / "train_scripts/lib/npu_env.sh"


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
    result = {}
    steps = re.findall(r"step (\d+): (\{[^\n]+\})", text)
    if steps:
        step, line = steps[-1]
        result["last_optimizer_step"] = int(step)
        match = re.search(r"'train/grad_norm': ([^,}]+)", line)
        if match:
            result["last_grad_norm"] = float(match[1])
        result["nonzero_updates_in_log_tail"] = sum(
            float(m.group(1)) > 0 for _, values in steps
            if (m := re.search(r"'train/grad_norm': ([\d.eE+-]+)", values))
        )
        result["last_step_signature"] = f"{step}:{line}"
    rollouts = re.findall(r"rollout\.py:\d+ - perf (\d+):", text)
    if rollouts:
        result["last_sampled_rollout"] = int(rollouts[-1])
    trained = re.findall(r"train_metric_utils\.py:\d+ - perf (\d+):", text)
    if trained:
        result["last_trained_rollout"] = int(trained[-1])
    result["log_modified_at"] = log.stat().st_mtime
    result["log_bytes"] = log.stat().st_size
    return result


def checkpoint(run_dir, state, write_log):
    tracker = run_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    try:
        advertised = int(tracker.read_text().strip())
    except (ValueError, OSError):
        return None
    choices = sorted((int(d.name[5:]) for d in run_dir.glob("iter_*")
                      if d.is_dir() and d.name[5:].isdigit()), reverse=True)
    # The tracker is committed by Megatron only after all ranks finish saving.
    # A newer directory may still be incomplete after an interrupted write.
    for iteration in (i for i in choices if i <= advertised):
        command = ('PROJ="$1"; source "$PROJ/train_scripts/lib/npu_env.sh"; '
                   'opsd_enter_env; opsd_export_npu_env "$PROJ"; '
                   'python "$PROJ/tasks/alfworld/scripts/check_campaign_checkpoint.py" "$2" "$3"')
        checked = subprocess.run(["bash", "-c", command, "checkpoint", str(PROJECT),
                                  str(run_dir), str(iteration)], capture_output=True, text=True, timeout=120)
        if checked.returncode:
            write_log(f"checkpoint {iteration} rejected: {checked.stderr[-800:]}")
            continue
        details = json.loads(checked.stdout.splitlines()[-1])
        if iteration == advertised:
            details["load_path"] = str(run_dir)
        else:
            mirror = state / "resume"
            mirror.mkdir(exist_ok=True)
            link = mirror / f"iter_{iteration:07d}"
            if not link.exists():
                link.symlink_to(run_dir / link.name, target_is_directory=True)
            rollout = mirror / "rollout"
            if not rollout.exists() and (run_dir / "rollout").exists():
                rollout.symlink_to(run_dir / "rollout", target_is_directory=True)
            (mirror / "latest_checkpointed_iteration.txt").write_text(str(iteration))
            details["load_path"] = str(mirror)
        return details
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["grpo", "gspo"], required=True)
    parser.add_argument("--ensure", action="store_true")
    options = parser.parse_args()
    arm = options.arm
    worker = 0 if arm == "grpo" else 1
    if not os.uname().nodename.endswith(f"worker-{worker}"):
        raise RuntimeError(f"refusing {arm} on unexpected node {os.uname().nodename}")
    state = STATE_ROOT / arm
    state.mkdir(parents=True, exist_ok=True)
    if (state / "STOP").exists() or (state / "COMPLETE").exists():
        return
    lock = open("/tmp/alfworld_4b_campaign_supervisor.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if options.ensure:
        lock.close()
        with (state / "daemon.log").open("ab", buffering=0) as output:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--arm", arm],
                             stdout=output, stderr=output, stdin=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
        return

    def write_log(message):
        with (state / "supervisor.log").open("a") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")

    run = f"alfworld_v2_{arm}_4b_200"
    run_dir = PROJECT / "runs" / run
    log = PROJECT / "logs" / f"{run}.log"
    # bash reads a script incrementally. Keep the validated launcher immutable
    # while another server edits launchers in the shared working tree.
    snapshot = state / "launch_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in [f"run_alfworld_v2_{arm}_4b_200.sh", "_4b_backbone.sh", "_common_v2.sh"]:
        destination = snapshot / name
        if not destination.exists():
            shutil.copy2(PROJECT / "tasks/alfworld/scripts" / name, destination)
    launcher = snapshot / f"run_alfworld_v2_{arm}_4b_200.sh"
    process = None
    attempt = 0
    chunk = 128
    last_progress = time.time()
    previous_signature = None
    next_launch = 0
    attempt_start = 0
    log_offset = log.stat().st_size if log.exists() else 0
    write_log(f"supervisor started pid={os.getpid()} host={os.uname().nodename}")
    while True:
        try:
            now = time.time()
            details = metrics(log)
            signature = (details.get("last_step_signature"), details.get("last_sampled_rollout"))
            if signature != previous_signature:
                previous_signature, last_progress = signature, now
            status = {"arm": arm, "run": run, "host": os.uname().nodename,
                      "supervisor_pid": os.getpid(), "heartbeat_at": time.strftime('%F %T'),
                      "heartbeat_epoch": now, "attempt": attempt, "log_probs_chunk_size": chunk,
                      **{k: v for k, v in details.items() if k != "last_step_signature"}}
            tracker = run_dir / "latest_checkpointed_iteration.txt"
            if tracker.exists():
                status["checkpoint_iteration"] = tracker.read_text().strip()
            if (state / "STOP").exists():
                if process and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                status["phase"] = "stopped"
                atomic_json(state / "status.json", status)
                return
            if process and process.poll() is None:
                status.update(phase="running", launcher_pid=process.pid)
                stale_log = now - max(details.get("log_modified_at", attempt_start), attempt_start) > 1800
                stalled = now - last_progress > 10800
                grad = details.get("last_grad_norm", 0.0)
                if stale_log or stalled or not math.isfinite(grad):
                    write_log(f"stalled/nonfinite training; terminating launcher pid={process.pid}")
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    next_launch = now + 120
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            if process:
                code = process.poll()
                write_log(f"launcher exited rc={code}; inspect checkpoint before recovery")
                with log.open("rb") as handle:
                    handle.seek(max(log_offset, log.stat().st_size - 300_000))
                    errors = handle.read().decode(errors="replace")
                if "out of memory" in errors.lower():
                    chunk = max(16, chunk // 2)
                    write_log(f"OOM recovery: execution chunk={chunk}; protocol unchanged")
                process = None
                next_launch = max(next_launch, now + min(600, 30 * attempt))
            cp = checkpoint(run_dir, state, write_log)
            if cp:
                status["checkpoint"] = cp
            if cp and cp["iteration"] >= 199:
                (state / "COMPLETE").write_text(time.strftime('%F %T') + "\n")
                status["phase"] = "complete"
                atomic_json(state / "status.json", status)
                write_log("200 rollouts complete; final checkpoint validated; standing down")
                return
            if now < next_launch:
                status["phase"] = "restart_backoff"
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            # A restarted supervisor adopts a living launcher rather than
            # starting another Ray job. The normal launcher lock is a second guard.
            alive = []
            for cmd in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    argv = cmd.read_bytes().split(b'\0')
                    driver = any(b'slime-ascend/train.py' in a for a in argv) and str(run_dir).encode() in argv
                    shell = str(launcher).encode() in argv or str(snapshot / "_common_v2.sh").encode() in argv
                    if driver or shell:
                        alive.append(int(cmd.parent.name))
                except (OSError, ValueError):
                    pass
            if alive:
                status.update(phase="adopted", trainer_pids=alive)
                if now - last_progress > 10800:
                    write_log(f"adopted trainers stalled for 3h; terminate PIDs {alive}")
                    for pid in alive:
                        os.kill(pid, signal.SIGTERM)
                    last_progress = now
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            env = dict(os.environ)
            for key in ["RESUME_FROM", "LOAD_DEBUG_ROLLOUT_DATA", "DRY_RUN_PARSE", "SAVE_DIR", "LOG", "PYTHONPATH"]:
                env.pop(key, None)
            env.update(RUN_NAME=run, NUM_ROLLOUT="200", SAVE_INTERVAL="50", SAVE_TRAIN_STATE="1",
                       MAX_RESPONSE="128", LOG_PROBS_CHUNK_SIZE=str(chunk),
                       MODEL_PROVIDER_PATH="tasks.alfworld.memory_provider.custom_model_provider",
                       RECOMPUTE_NUM_LAYERS="36", RECOMPUTE_LOSS="1", RESUME_WEIGHTS_ONLY="0",
                       G="8", ROLLOUT_BS="4", GBS_OVERRIDE="32", HISTORY_LENGTH="0", MAX_TURNS="50",
                       SEQ_LENGTH="32768", SEED="42", TP="2", EPS_CLIP_HIGH="0.2", LR="1e-6",
                       MODEL=str(PROJECT.parents[1] / "checkpoints/Qwen3-4B"))
            if cp:
                env["RESUME_FROM"] = cp["load_path"]
                env["RESUME_WEIGHTS_ONLY"] = "0" if cp["full_train_state"] else "1"
            # Ray actors live outside the launcher's process group. A killed
            # driver can leave them reserving every accelerator indefinitely.
            cleanup = ('PROJ="$1"; source "$PROJ/train_scripts/lib/npu_env.sh"; '
                       'opsd_enter_env; opsd_export_npu_env "$PROJ"; ray stop --force')
            cleared = subprocess.run(["bash", "-c", cleanup, "cleanup", str(PROJECT)],
                                     env=env, capture_output=True, text=True, timeout=120)
            if cleared.returncode:
                write_log(f"Ray cleanup failed rc={cleared.returncode}; retrying before launch")
                next_launch = now + 120
                continue
            attempt += 1
            last_progress = attempt_start = time.time()
            log_offset = log.stat().st_size if log.exists() else 0
            with (state / "launcher.log").open("ab", buffering=0) as output:
                process = subprocess.Popen(["bash", str(launcher)], cwd=PROJECT, env=env,
                                           stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                           start_new_session=True, close_fds=True)
            write_log(f"launch attempt={attempt} pid={process.pid} checkpoint={cp}")
            status.update(phase="starting", launcher_pid=process.pid, attempt=attempt)
            atomic_json(state / "status.json", status)
        except Exception:
            write_log(traceback.format_exc())
        time.sleep(30)


if __name__ == "__main__":
    main()
