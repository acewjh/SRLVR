"""Supervise one default-protocol run on its assigned worker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise_1p7b_campaign import atomic_json, metrics

PROJECT = Path(__file__).resolve().parents[3]
ENTRY = PROJECT / "tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh"
BENCHMARK = PROJECT / "tasks/alfworld/scripts/eval_alfworld_ckpt.sh"
ROLLOUTS = 25
FINAL_ITERATION = ROLLOUTS - 1
THINKING_STYLE = "brief"
CONTEXT_TOKENS = 32768
RESPONSE_TOKENS = 4096


def check_checkpoint(root, arm):
    import torch
    from check_campaign_checkpoint import validate
    save = root / "checkpoints"
    iteration = int((save / "latest_checkpointed_iteration.txt").read_text())
    result = validate(save, iteration)
    if not result["full_train_state"]:
        raise ValueError("checkpoint lacks full training state")
    common = torch.load(save / f"iter_{iteration:07d}/common.pt", map_location="cpu", weights_only=False)
    import pickle
    with (save / f"iter_{iteration:07d}/.metadata").open("rb") as handle:
        metadata = pickle.load(handle)
    keys = list(common) + list(metadata.state_dict_metadata)
    for required in ("rng_state", "opt_param_scheduler"):
        if not any(required in k for k in keys):
            raise ValueError(f"missing {required}")
    if arm == "srlvr" and getattr(common["args"], "srlvr_state", {}).get("last_rollout") != iteration:
        raise ValueError("missing or inconsistent SRLVR history")
    curriculum = save / "curriculum" / f"{iteration:07d}.json"
    state = json.loads(curriculum.read_text())
    if not state["families"] or not state["tasks"]:
        raise ValueError("empty curriculum state")
    return {**result, "scheduler_rng_method_curriculum_validated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("srlvr", "grpo"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = PROJECT / "runs" / args.run_name
    if args.check:
        print(json.dumps(check_checkpoint(root, args.arm)))
        return
    worker = 0 if args.arm == "srlvr" else 1
    if not os.uname().nodename.endswith(f"worker-{worker}"):
        raise RuntimeError("method assigned to a different worker")
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (state / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    def log(message):
        with (state / "supervisor.log").open("a") as handle:
            handle.write(f"{time.strftime('%F %T')} {message}\n")
    log(f"started host={os.uname().nodename} pid={os.getpid()} arm={args.arm} root={root}")
    process = None
    benchmark_process = None
    attempt = 0
    checkpoint = None
    checked_tracker = None
    log_path = root / "train.log"
    benchmark_state = state / "benchmark_started.json"
    benchmark_log = root / "benchmark" / "evaluation.log"

    def checkpoint_is_final():
        return bool(checkpoint and checkpoint["iteration"] >= FINAL_ITERATION)

    def benchmark_succeeded():
        if not benchmark_log.exists():
            return False
        with benchmark_log.open("rb") as handle:
            handle.seek(max(0, benchmark_log.stat().st_size - 2_000_000))
            tail = handle.read().decode(errors="replace")
        return (f"[eval] VERIFIED: loaded iteration {FINAL_ITERATION}" in tail
                and "[eval] launcher exited rc=0" in tail)

    def finish_benchmark(returncode):
        good = returncode == 0 and benchmark_succeeded()
        status.update(
            phase="complete" if good else "benchmark_failed",
            benchmark_returncode=returncode,
            benchmark_checkpoint_iteration=FINAL_ITERATION,
            benchmark_checkpoint_verified=benchmark_succeeded(),
            benchmark_log=str(benchmark_log),
        )
        atomic_json(state / "status.json", status)
        if good:
            (state / "COMPLETE").touch()
            log("100 rollout batches complete; final checkpoint verified and benchmark completed")
        else:
            log(f"benchmark failed rc={returncode}; evidence preserved at {benchmark_log}")

    def launch_benchmark():
        nonlocal benchmark_process
        benchmark_dir = root / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_name = (f"alfworld-qwen3-1p7b-{args.arm}-think-ctx32k-r{ROLLOUTS}-"
                          f"cp{FINAL_ITERATION}-{stamp}")
        env = dict(os.environ)
        prefixes = ("ARM", "SRLVR_", "ROLLOUT_", "ACTOR_", "ALFWORLD_", "SGLANG_",
                    "OPSD_", "EPS_", "KL_", "RUN_", "SAVE_", "MAX_", "HISTORY_",
                    "FUTURE_", "SEQ_", "CLIP_")
        explicit = {"MODEL", "LR", "G", "GBS_OVERRIDE", "NUM_ROLLOUT", "SEED", "SEQ_LENGTH",
                    "TP", "PP", "RESUME_FROM", "RESUME_WEIGHTS_ONLY", "EVAL_ONLY", "EVAL_LOAD",
                    "EVAL_LOAD_FROM", "DRY_RUN_PARSE", "LOAD_DEBUG_ROLLOUT_DATA", "DEBUG_ROLLOUT_DIR",
                    "LOG", "PYTHONPATH", "SAVE_DIR", "METHOD_SLUG", "MODEL_SLUG", "BACKBONE",
                    "MAX_RESPONSE", "MAX_TURNS", "N_SAMPLES_PER_EVAL_PROMPT", "EVAL_CONFIG"}
        for key in list(env):
            if key.startswith(prefixes) or key in explicit:
                env.pop(key)
        env.update(
            ARM="eval", RUN_NAME=benchmark_name, METHOD_SLUG=args.arm,
            MODEL_SLUG="qwen3-1p7b", BACKBONE="1p7b",
            MODEL="/workspace/cua/checkpoints/Qwen3-1.7B",
            ALFWORLD_V2_PROTOCOL="1", ALFWORLD_ACTION_HARNESS="command-normalization-v2-form",
            ALFWORLD_THINKING_STYLE=THINKING_STYLE,
            ALFWORLD_TASK_SAMPLING="frontier", ALFWORLD_EVAL_CONCURRENCY="8",
            SGLANG_PREFIX_CACHE="1", SGLANG_DTYPE="bfloat16", ACTOR_GPUS="8", ROLLOUT_GPUS="8", TP="2", PP="1",
            COLOCATE="1", ROLLOUT_GPUS_PER_ENGINE="1",
            SGLANG_MEM_FRACTION_STATIC="0.5", SGLANG_MAX_TOTAL_TOKENS="65536",
            SGLANG_MAX_RUNNING_REQUESTS="8", SGLANG_SERVER_CONCURRENCY="8",
            NUM_ROLLOUT="0", MAX_TURNS="50", MAX_RESPONSE=str(RESPONSE_TOKENS),
            SEQ_LENGTH=str(CONTEXT_TOKENS), OPSD_LORA=os.environ.get("OPSD_LORA", "0"), OPSD_LORA_R="128",
            OPSD_LORA_ALPHA="128", OPSD_SRLVR="0", OPSD_EXPERIMENT_METRICS="0",
            SRLVR_VARIANT="", SAVE_TRAIN_STATE="0", RESUME_WEIGHTS_ONLY="1",
            N_SAMPLES_PER_EVAL_PROMPT="1", EVAL_CONFIG=str(PROJECT / "tasks/alfworld/v1/configs/eval_alfworld.json"),
            SAVE_DIR=str(benchmark_dir / "checkpoints"),
            DEBUG_ROLLOUT_DIR=str(benchmark_dir / "rollouts"),
            LOG=str(benchmark_log),
        )
        out = benchmark_log.open("ab", buffering=0)
        benchmark_process = subprocess.Popen(
            ["bash", str(BENCHMARK), str(root / "checkpoints"), str(FINAL_ITERATION)],
            cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
            start_new_session=True, close_fds=True,
        )
        out.close()
        atomic_json(benchmark_state, {
            "pid": benchmark_process.pid, "run_name": benchmark_name,
            "checkpoint_iteration": FINAL_ITERATION, "log": str(benchmark_log),
            "started_at": time.time(),
        })
        status.update(phase="benchmark", benchmark_pid=benchmark_process.pid,
                      benchmark_run_name=benchmark_name, benchmark_log=str(benchmark_log))
        atomic_json(state / "status.json", status)
        log(f"final checkpoint verified; benchmark started pid={benchmark_process.pid} run={benchmark_name}")
    while True:
        details = metrics(log_path)
        status = dict(arm=args.arm, run_dir=str(root), host=os.uname().nodename,
                      supervisor_pid=os.getpid(), heartbeat_epoch=time.time(), attempt=attempt,
                      **{k:v for k,v in details.items() if k != "last_step_signature"})
        tracker = root / "checkpoints/latest_checkpointed_iteration.txt"
        value = tracker.read_text().strip() if tracker.exists() else None
        if value is not None and value != checked_tracker:
            command = ('PROJ="$1"; source "$PROJ/train_scripts/lib/npu_env.sh"; '
                       'opsd_enter_env; opsd_export_npu_env "$PROJ"; '
                       'python "$2" --arm "$3" --run-name "$4" --state-dir "$5" --check')
            checked = subprocess.run(["bash", "-c", command, "check", str(PROJECT),
                                      str(Path(__file__).resolve()), args.arm, args.run_name, str(state)],
                                     capture_output=True, text=True, timeout=120)
            with (state / "checkpoint_validation.log").open("a") as handle:
                handle.write(checked.stdout + checked.stderr)
            if checked.returncode == 0:
                checkpoint = json.loads(checked.stdout.splitlines()[-1])
                checked_tracker = value
                log(f"validated checkpoint {checkpoint}")
        status["checkpoint"] = checkpoint
        if (state / "STOP").exists():
            status["phase"] = "supervision_stopped"
            atomic_json(state / "status.json", status)
            log("STOP requested; training process remains independently managed")
            return
        if checkpoint_is_final() and process is None:
            # If the supervisor itself restarted during evaluation, adopt its PID
            # and infer completion only from the benchmark's verified load record.
            if benchmark_state.exists():
                started = json.loads(benchmark_state.read_text())
                pid = int(started["pid"])
                if benchmark_process is not None:
                    rc = benchmark_process.poll()
                    if rc is None:
                        status.update(phase="benchmark", benchmark_pid=pid,
                                      benchmark_run_name=started["run_name"],
                                      benchmark_log=str(benchmark_log))
                        atomic_json(state / "status.json", status)
                        time.sleep(30)
                        continue
                    finish_benchmark(rc)
                    return
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                if alive:
                    status.update(phase="benchmark", benchmark_pid=pid,
                                  benchmark_run_name=started["run_name"],
                                  benchmark_log=str(benchmark_log))
                    atomic_json(state / "status.json", status)
                    time.sleep(30)
                    continue
                finish_benchmark(0 if benchmark_succeeded() else 1)
                return
            launch_benchmark()
            time.sleep(5)
            continue
        if process is not None and process.poll() is None:
            status.update(phase="running", launcher_pid=process.pid)
            age = time.time() - details.get("log_modified_at", time.time())
            status["alerts"] = ["training log stale for over 45 minutes; inspect process/device activity"] if age > 2700 else []
            atomic_json(state / "status.json", status)
            time.sleep(30)
            continue
        if process is not None:
            log(f"launcher exited rc={process.returncode} pid={process.pid}")
            (state / f"attempt_{attempt:02d}_exit.json").write_text(json.dumps({"rc":process.returncode, "time":time.time(), "checkpoint":checkpoint}))
            if log_path.exists():
                with log_path.open("rb") as handle:
                    handle.seek(max(0, log_path.stat().st_size - 200000))
                    (state / f"attempt_{attempt:02d}_failure_tail.log").write_bytes(handle.read())
            if checkpoint_is_final():
                status["training_returncode"] = process.returncode
                process = None
                if not benchmark_state.exists():
                    launch_benchmark()
                continue
            process = None
        if root.exists():
            # Adopt an existing launcher/driver after restarting this supervisor.
            alive = []
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    argv = path.read_bytes().split(b"\0")
                    if any(b"slime-ascend/train.py" in x for x in argv) and str(root / "checkpoints").encode() in argv:
                        alive.append(int(path.parent.name))
                except (OSError, ValueError):
                    pass
            if alive:
                status.update(phase="adopted", trainer_pids=alive)
                atomic_json(state / "status.json", status)
                time.sleep(30)
                continue
            if not checkpoint and attempt < 3:
                # Before the first save interval there is no resume point. Retire
                # the failed attempt (rename, never delete) and start fresh so a
                # startup/early-rollout crash does not end the run.
                retired = root.with_name(f"{root.name}-retry{attempt:02d}-{int(time.time())}")
                try:
                    root.rename(retired)
                    log(f"no checkpoint yet; retired failed run to {retired.name}; retrying fresh")
                except OSError as exc:
                    status.update(phase="needs_repair", reason=f"no checkpoint and could not retire failed run: {exc}")
                    atomic_json(state / "status.json", status)
                    log(status["reason"])
                    return
            elif not checkpoint or attempt >= 3:
                status.update(phase="needs_repair", reason="no complete checkpoint or restart limit reached; evidence preserved")
                atomic_json(state / "status.json", status)
                log(status["reason"])
                return
            if checkpoint_is_final():
                if not benchmark_state.exists():
                    launch_benchmark()
                continue
        env = dict(os.environ)
        # Never inherit a previous experiment's knobs or checkpoint.
        prefixes = ("ARM", "SRLVR_", "ROLLOUT_", "ACTOR_", "ALFWORLD_", "SGLANG_", "OPSD_", "EPS_", "KL_", "RUN_", "SAVE_", "MAX_", "HISTORY_")
        explicit = {"MODEL", "LR", "G", "GBS_OVERRIDE", "NUM_ROLLOUT", "SEED", "SEQ_LENGTH", "TP", "PP", "RESUME_FROM", "RESUME_WEIGHTS_ONLY", "EVAL_ONLY", "EVAL_LOAD", "EVAL_LOAD_FROM", "DRY_RUN_PARSE", "LOAD_DEBUG_ROLLOUT_DATA", "DEBUG_ROLLOUT_DIR", "LOG", "PYTHONPATH"}
        for key in list(env):
            if key.startswith(prefixes) or key in explicit:
                env.pop(key)
        env.update(
            ARM=args.arm, RUN_NAME=args.run_name,
            NUM_ROLLOUT=str(ROLLOUTS), SAVE_INTERVAL="1",
            SEQ_LENGTH=str(CONTEXT_TOKENS), MAX_RESPONSE=str(RESPONSE_TOKENS),
            ALFWORLD_THINKING_STYLE=THINKING_STYLE,
            ALFWORLD_ACTION_HARNESS="command-normalization-v2-form",
            OPSD_LORA_R="128", OPSD_LORA_ALPHA="128", LR="1e-6",
            KL_COEF="0.001", MAX_TURNS="50", SEED="42",
            SAVE_TRAIN_STATE="1", SGLANG_PREFIX_CACHE="1", ACTOR_GPUS="8", ROLLOUT_GPUS="8", TP="2", PP="1",
            COLOCATE="1", ROLLOUT_GPUS_PER_ENGINE="1",
            SGLANG_MEM_FRACTION_STATIC="0.5", SGLANG_MAX_TOTAL_TOKENS="65536",
            SGLANG_MAX_RUNNING_REQUESTS="8", SGLANG_SERVER_CONCURRENCY="8",
            LOG_PROBS_CHUNK_SIZE=os.environ.get("ALFWORLD_LOG_PROBS_CHUNK_SIZE", "128"),
            RECOMPUTE_LOSS=os.environ.get("ALFWORLD_RECOMPUTE_LOSS", "1"),
            RECOMPUTE_NUM_LAYERS=os.environ.get(
                "ALFWORLD_RECOMPUTE_NUM_LAYERS", "1"),
            MODEL_PROVIDER_PATH=os.environ.get(
                "ALFWORLD_MODEL_PROVIDER_PATH",
                "tasks.alfworld.memory_provider.custom_model_provider" if args.arm == "srlvr"
                else "methods.lora_megatron.custom_model_provider"),
        )
        if root.exists():
            env["RESUME_FROM"] = str(root / "checkpoints")
            (root / "NON_CONTIGUOUS").touch()
            log(f"resume from validated full checkpoint {checkpoint['iteration']} (same {ROLLOUTS}-batch target)")
        attempt += 1
        with (state / f"launch_attempt_{attempt:02d}.log").open("ab", buffering=0) as out:
            process = subprocess.Popen(["bash", str(ENTRY)], cwd=PROJECT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                       start_new_session=True, close_fds=True)
        (state / "launcher.pid").write_text(str(process.pid) + "\n")
        log(f"launch attempt={attempt} pid={process.pid}")
        time.sleep(5)


if __name__ == "__main__":
    main()
