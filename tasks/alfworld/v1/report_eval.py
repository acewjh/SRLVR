"""Aggregate ALFWorld eval dumps into the success-rate table.

Reads the ``--save-debug-rollout-data`` torch dumps written during an eval-only
run (``--num-rollout 0 --eval-interval 1``) and emits the paper metric: success
rate overall, per split and per task category, plus the mean-turns /
invalid-action / truncation diagnostics.

This mirrors the repo's existing eval flow (``math_opsd/aggregate_dev_eval.py``):
eval runs under ``train.py`` and the table is produced offline, so a scoring bug
can be fixed and the same dumps re-read without re-running on NPUs.

Usage::

    python tasks/alfworld/v1/report_eval.py --dump runs/alfworld/base_eval/dev_eval/*.pt
    python tasks/alfworld/v1/report_eval.py --dump <dump.pt> --json out.json --out reports/alfworld_eval.md
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tasks.alfworld.v1.eval import EvalSample, aggregate, format_report  # noqa: E402

__all__ = ["load_eval_samples", "main"]


def _as_dict(sample) -> dict:
    if isinstance(sample, dict):
        return sample
    return {
        "reward": getattr(sample, "reward", None),
        "metadata": getattr(sample, "metadata", None) or {},
    }


def load_eval_samples(paths: list[str]) -> list[EvalSample]:
    """Extract one :class:`EvalSample` per dumped episode."""
    import torch

    rows: list[EvalSample] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        for raw in samples:
            sample = _as_dict(raw)
            metadata = sample.get("metadata") or {}
            trajectory = metadata.get("trajectory")
            if not trajectory:
                raise ValueError(
                    f"{path}: sample carries no metadata['trajectory']; was it produced by "
                    "the ALFWorld rollout?"
                )
            reward = sample.get("reward")
            reward = float(reward.item() if hasattr(reward, "item") else (reward or 0.0))
            rows.append(
                EvalSample(
                    task_id=trajectory["task_id"],
                    task_type=trajectory["task_type"],
                    split=trajectory["split"],
                    success=bool(trajectory["success"]),
                    num_turns=int(trajectory["num_turns"]),
                    truncated=bool(trajectory["truncated"]),
                    invalid_action_count=int(trajectory["invalid_action_count"]),
                    generated_tokens=int(metadata.get("generated_tokens", 0) or 0),
                )
            )
            # reward and success must agree -- the reward IS terminal success
            if bool(trajectory["success"]) != (reward >= 0.5):
                raise AssertionError(
                    f"{path}: reward {reward} disagrees with trajectory success "
                    f"{trajectory['success']} for {trajectory['task_id']}"
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate ALFWorld eval dumps.")
    parser.add_argument("--dump", nargs="+", required=True, help="dump file(s) or glob(s)")
    parser.add_argument("--json", default=None, help="optional path for the raw metrics JSON")
    parser.add_argument("--out", default=None, help="optional path for the markdown report")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.dump:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or [pattern])
    if not paths:
        print("[report_eval] no dump files matched", file=sys.stderr)
        return 1

    samples = load_eval_samples(paths)
    if not samples:
        print("[report_eval] dumps contained no episodes", file=sys.stderr)
        return 1

    report = aggregate(samples)
    payload = report.as_dict()
    rendered = format_report(report)
    print(rendered)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(f"\n### {', '.join(Path(p).name for p in paths)}\n```\n{rendered}\n```\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
