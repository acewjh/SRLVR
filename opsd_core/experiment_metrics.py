"""Small shared JSONL result records for the existing experiment entry."""
import json
import os
from pathlib import Path
import time


def record(args, rollout, phase, **values):
    if os.environ.get("OPSD_EXPERIMENT_METRICS") != "1":
        return
    root = Path(args.save).parent
    item = dict(algorithm=os.environ["ARM"], mode=os.environ.get("SRLVR_MODE") if os.environ["ARM"]=="srlvr" else None, task="alfworld", seed=args.seed,
                rollout=rollout, phase=phase, unix_time=time.time(), **values)
    with (root / "common_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(item, allow_nan=False)+"\n")


def resolved(args):
    if os.environ.get("OPSD_EXPERIMENT_METRICS") == "1":
        root = Path(args.save).parent
        root.mkdir(parents=True, exist_ok=True)
        effective = vars(args).copy()
        if os.environ.get("OPSD_SRLVR") == "1" and os.environ.get("SRLVR_VARIANT") in ("v2", "v3"):
            from dataclasses import asdict
            from opsd_core.srlvr_settings import Settings
            effective["srlvr_hsi"] = asdict(Settings.from_env())
        (root / "effective_args.json").write_text(json.dumps(effective, default=str, indent=2))


def finish(args, elapsed):
    if os.environ.get("OPSD_EXPERIMENT_METRICS") != "1":
        return
    root = Path(args.save).parent
    records = [json.loads(line) for line in (root/"common_metrics.jsonl").read_text().splitlines()]
    rollouts = [r for r in records if r["phase"] == "rollout"]
    counts = {k: sum(r.get(k,0) for r in rollouts)
              for k in ("tasks","real_rollouts","actor_tokens","successes","failures")}
    result = dict(algorithm=os.environ["ARM"],mode=os.environ.get("SRLVR_MODE") if os.environ["ARM"]=="srlvr" else None,task="alfworld",seed=args.seed,**counts,
                  success_rate=counts["successes"]/max(counts["real_rollouts"],1),
                  rollout_rounds=len(rollouts),
                  training_steps=sum(r.get("optimizer_steps",0) for r in records if r["phase"]=="optimizer"),
                  total_seconds=elapsed,
                  effective_config=str(root/"effective_args.json"),
                  train_allocated_device_seconds=sum(r.get("train_allocated_device_seconds",0) for r in records if r["phase"]=="train_batch"),
                  inference_allocated_device_seconds=sum(r.get("inference_allocated_device_seconds",0) for r in records if r["phase"]=="train_batch"))
    (root/"results.json").write_text(json.dumps(result,indent=2))
