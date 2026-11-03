"""Check short-run quality and speed before starting a 200-rollout campaign."""

import ast
import json
import re
import sys
from pathlib import Path


def check(log_path, method):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(log_path).read_text(errors="replace"))
    rollout, train = {}, {}
    for line in text.splitlines():
        match = re.search(r"perf (\d+): (\{.*\})", line)
        if match:
            data = ast.literal_eval(match[2])
            if "perf/rollout_time" in data:
                rollout[int(match[1])] = data
            if "perf/train_time" in data:
                train[int(match[1])] = data
    errors = []
    if not {0, 1}.issubset(rollout.keys() & train.keys()):
        errors.append("two complete sampling/training cycles required")
    gradients = [float(x) for x in re.findall(r"'train/grad_norm': ([0-9.eE+-]+)", text)]
    if not any(x > 0 for x in gradients):
        errors.append("no nonzero optimizer gradient")
    summaries = []
    for i in sorted(rollout.keys() & train.keys()):
        r, t = rollout[i], train[i]
        summaries.append({"rollout": i, "success_rate": r["rollout/success_rate"],
                          "mean_response_tokens": r["rollout/response_len/mean"],
                          "rollout_seconds": r["perf/rollout_time"],
                          "train_seconds": t["perf/train_time"],
                          "step_seconds": t["perf/step_time"]})
    if summaries:
        # The old base-policy rollout 0 was 27 min end-to-end and 84-88% success.
        # This is a small train-task screen, not a substitute for held-out eval.
        first = summaries[0]
        if first["success_rate"] < 0.75:
            errors.append("rollout 0 success below 75%; review prompt before long run")
        # Compare the same two scheduled task batches, rather than later, harder
        # batches or the literal 64-token prompt target. Reference measurements
        # are from the preserved 20261020_002240 runs, rollout IDs 0 and 1.
        reference = {
            "grpo": {"mean_response_tokens": (776.2608 + 614.9507) / 2,
                     "mean_step_seconds": (1581.2291 + 1330.0297) / 2},
            "gspo": {"mean_response_tokens": (781.2988 + 631.0516) / 2,
                     "mean_step_seconds": (1584.0816 + 1369.5576) / 2},
        }[method]
        mean_tokens = sum(x["mean_response_tokens"] for x in summaries) / len(summaries)
        mean_step = sum(x["step_seconds"] for x in summaries) / len(summaries)
        if mean_tokens > reference["mean_response_tokens"] * 0.75:
            errors.append("mean response reduction below 25% on matched task batches")
        if mean_step > reference["mean_step_seconds"] * 0.95:
            errors.append("complete-cycle time reduction below 5% on matched task batches")
        if sum(x["success_rate"] for x in summaries) / len(summaries) < 0.65:
            errors.append("mean success below 65%; review prompt before long run")
    return {"passed": not errors, "errors": errors, "cycles": summaries,
            "nonzero_gradient_steps": sum(x > 0 for x in gradients)}


if __name__ == "__main__":
    method = sys.argv[2] if len(sys.argv) > 2 else next(
        m for m in ("grpo", "gspo") if f"alfworld-qwen3-8b-{m}/" in sys.argv[1]
    )
    result = check(sys.argv[1], method)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
