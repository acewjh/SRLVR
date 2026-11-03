"""Read-only training monitor; write progress/alerts without changing trainers."""

import argparse
import ast
import datetime
import fcntl
import json
import math
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_metrics(path):
    text = ANSI.sub("", path.read_text(errors="replace"))
    cycles, updates, alerts = {}, [], []
    for line in text.splitlines():
        match = re.search(r"perf (\d+): (\{.*\})", line)
        update = re.search(r"model.py:\d+ - step (\d+): (\{.*\})", line)
        if not (match or update):
            continue
        match = match or update
        try:
            data = ast.literal_eval(match[2])
        except (ValueError, SyntaxError):
            alerts.append("metric dictionary could not be parsed; inspect possible nonfinite values")
            continue
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in data.values()):
            alerts.append("nonfinite training/rollout metric")
        if update:
            updates.append(data)
        else:
            cycles.setdefault(int(match[1]), {}).update(data)
    return cycles, updates, sorted(set(alerts))


def summarize(method, stamp):
    folder = ROOT / f"runs/alfworld-qwen3-8b-{method}/s42x200-a2r6-brief64-k0-cache1/{stamp}"
    state_path = ROOT / "logs/alfworld-fast-campaign/state.json"
    if state_path.exists():
        active = json.loads(state_path.read_text()).get("full_run_roots", {}).get(method)
        if active:
            folder = ROOT / active
    log = folder / "train.log"
    current, updates, alerts = read_metrics(log)
    reference = ROOT / f"logs/alfworld-qwen3-8b-{method}-s42x200-20261020_002240.log"
    old = read_metrics(reference)[0] if reference.exists() else {}
    matched = sorted(i for i in current if i in old and "perf/step_time" in current[i]
                     and "perf/step_time" in old[i])
    comparisons = {}
    for key in ("perf/rollout_time", "perf/train_time", "perf/step_time",
                "rollout/success_rate", "rollout/response_len/mean"):
        if matched:
            before = statistics.mean(old[i][key] for i in matched)
            after = statistics.mean(current[i][key] for i in matched)
            comparisons[key] = {"old": before, "new": after,
                                "relative_change": after / before - 1 if before else None}
    # A screen for sustained change on the SAME scheduled batches, not a claim of
    # held-out regression. Different batches have very different task difficulty.
    window = matched[-5:]
    if len(window) == 5:
        delta = statistics.mean(current[i]["rollout/success_rate"] - old[i]["rollout/success_rate"]
                                for i in window)
        if delta <= -0.10:
            alerts.append("last five matched batches: success down at least 10 percentage points")
        ratio = sum(current[i]["rollout/response_len/mean"] for i in window) / sum(
            old[i]["rollout/response_len/mean"] for i in window)
        if ratio > 1.5:
            alerts.append("last five matched batches: response length exceeds old run by 50%")
    tracker = folder / "checkpoints/latest_checkpointed_iteration.txt"
    age = time.time() - log.stat().st_mtime
    if age > 3600:
        alerts.append("no train-log update for one hour; inspect node/process activity")
    gradients = [d.get("train/grad_norm", 0) for d in updates]
    sampling_ids = sorted(i for i in current if i in old and "rollout/success_rate" in current[i]
                          and "rollout/success_rate" in old[i])
    success_comparison = None
    if sampling_ids:
        before = sum(old[i]["rollout/success_rate"] for i in sampling_ids) / len(sampling_ids)
        after = sum(current[i]["rollout/success_rate"] for i in sampling_ids) / len(sampling_ids)
        success_comparison = {"rollouts": sampling_ids, "old": before, "new": after,
                              "percentage_point_change": (after - before) * 100}
    return {"run_dir": str(folder), "latest_sampled_rollout": max(current, default=None),
            "latest_trained_rollout": max((i for i, d in current.items() if "perf/train_time" in d), default=None),
            "checkpoint_iteration": tracker.read_text().strip() if tracker.exists() else None,
            "matched_rollouts": matched, "matched_comparison": comparisons,
            "sampled_success_comparison": success_comparison,
            "recent_sampling": [{"rollout": i, **{k: v for k, v in current[i].items()
                if k in ("rollout/success_rate", "rollout/response_len/mean", "rollout/truncated_ratio",
                         "rollout/mixed_group_rate", "perf/rollout_time", "perf/train_time", "perf/step_time")}}
                for i in sorted(current)[-5:]],
            "logged_optimizer_updates": len(updates),
            "nonzero_gradient_updates": sum(g > 0 for g in gradients),
            "last_gradient_norm": gradients[-1] if gradients else None,
            "log_age_seconds": age, "alerts": alerts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stamp")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    output = ROOT / "logs/alfworld-fast-campaign" / args.stamp
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        report = {"updated_at": datetime.datetime.now().astimezone().isoformat(),
                  "comparison_note": "matched training batches, not held-out evaluation", "methods": {}}
        for method in ("grpo", "gspo"):
            try:
                report["methods"][method] = summarize(method, args.stamp)
            except Exception as exc:
                report["methods"][method] = {"alerts": [f"monitor read failed: {type(exc).__name__}: {exc}"]}
        audit = output / "action_quality_audit.json"
        if audit.exists():
            report["action_quality_audit"] = json.loads(audit.read_text())
        temporary = output / "health.tmp.json"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "health.json")
        with (output / "health_history.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        if not args.watch:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
