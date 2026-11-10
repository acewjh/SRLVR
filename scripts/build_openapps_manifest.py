#!/usr/bin/env python3
"""Build the slime prompt manifest for OpenApps GUI tasks.

One row per task, shaped like the OSWorld manifest the GUI rollout already
consumes:

    instruction (str)                  the task goal, used by --input-key
    metadata (struct)                  copied verbatim onto Sample.metadata
        task_id / task_key             OpenApps config key (YAML name)
        instruction                    duplicate, for the rollout's fallback
        task_payload                   JSON string sent to the gateway reset;
                                       {"instruction": ..., "task_key": ...}
        domain                         "openapps"
        supported_eval / verified_eval  every OpenApps task grades its own state

Usage:
    python scripts/build_openapps_manifest.py --output data/openapps_manifest.parquet [--limit N]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/openapps_manifest.parquet")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N tasks")
    ap.add_argument("--include-abstention", action="store_true")
    args = ap.parse_args()

    from open_apps.tasks import list_task_keys, load_task

    keys = list_task_keys()
    rows = []
    for key in keys:
        task = load_task(key)
        cls = type(task).__name__
        if not args.include_abstention and ("Abstention" in cls or cls == "AnswerTask"):
            continue
        goal = task.goal
        rows.append(
            {
                "instruction": goal,
                "metadata": {
                    "task_id": key,
                    "task_key": key,
                    "instruction": goal,
                    "task_payload": json.dumps({"instruction": goal, "task_key": key}),
                    "domain": "openapps",
                    "task_class": cls,
                    "supported_eval": True,
                    "verified_eval": True,
                },
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df)} tasks to {args.output}")
    print(df.head(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
