"""Check the NPU smoke-gate conditions against a run log.

Gates A-D are only meaningful if they actually PASS, and a training log is a lot
of text to eyeball. This turns each gate's pass conditions into an explicit check
so "Gate C PASS" cannot mean "the script exited 0" -- a run can exit 0 having
trained on a degenerate batch.

Deliberately conservative: a metric that never appears is a FAIL, not a skip,
because a missing metric is exactly the failure mode this is meant to catch.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

__all__ = ["check_log", "GateReport"]

#: Substrings that mean the harness's own correctness assertions fired.
_FATAL_MARKERS = (
    "alignment violated",
    "AssertionError",
    "loss_mask/turn_ids disagree",
    "teacher-forced target tokens differ",
    "reasoning leaked into history",
    "restates the current action",
    "Traceback (most recent call last)",
)

_NUMBER = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|-inf)"


def _metric_values(text: str, key: str) -> list[float]:
    """All numeric values logged for ``key`` across the run.

    Matches both ``key: 1.23`` (logger output) and ``'key': 1.23`` (dict repr).
    """
    pattern = re.compile(re.escape(key) + r"['\"]?\s*[:=]\s*" + _NUMBER)
    values = []
    for match in pattern.finditer(text):
        raw = match.group(1)
        if raw in ("nan", "inf", "-inf"):
            values.append(float(raw))
        else:
            values.append(float(raw))
    return values


class GateReport:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))

    @property
    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.checks if not ok]

    def render(self) -> str:
        lines = [f"ALFWorld gate report -- {self.log_path}", "=" * 64]
        for name, ok, detail in self.checks:
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        lines.append("-" * 64)
        lines.append("GATES = " + ("PASS" if not self.failed else "FAIL"))
        return "\n".join(lines)


def _finite(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(v) for v in values)


def check_log(log_path: str, *, gate: str) -> GateReport:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    report = GateReport(log_path)

    # ---- universal: the harness's own assertions must not have fired ----
    for marker in _FATAL_MARKERS:
        report.check(f"no {marker!r} in log", marker not in text)

    # ---- the run must have produced training output at all ----
    losses = _metric_values(text, "train/loss")
    grad_norms = _metric_values(text, "train/grad_norm")
    report.check("train/loss present and finite", _finite(losses),
                 f"{len(losses)} value(s)")
    report.check("train/grad_norm present and finite", _finite(grad_norms),
                 f"{len(grad_norms)} value(s)")

    # A zero grad norm means the advantage was identically zero -- the exact
    # failure mode that made an earlier GRPO arm copy the base model for 12 steps.
    report.check("grad_norm is not identically zero",
                 any(abs(v) > 0.0 for v in grad_norms) if grad_norms else False,
                 "a frozen policy means every group had zero reward variance")

    success = _metric_values(text, "rollout/success_rate")
    report.check("rollout/success_rate logged", bool(success), f"{len(success)} value(s)")

    turns = _metric_values(text, "rollout/mean_turns")
    report.check("rollout/mean_turns > 0", any(v > 0 for v in turns) if turns else False,
                 f"mean over run: {sum(turns) / len(turns):.2f}" if turns else "missing")

    if gate in ("D", "C"):
        # Reward variance inside a group is what makes GRPO's advantage non-zero.
        kl = _metric_values(text, "ppo_kl")
        report.check("ppo_kl logged (policy actually moved)", bool(kl), f"{len(kl)} value(s)")

    if gate == "D":
        scored = _metric_values(text, "hpi/scored_turns")
        skipped = _metric_values(text, "hpi/skipped_turns")
        report.check("hpi/scored_turns logged", bool(scored))
        report.check("hpi/scored_turns > 0", any(v > 0 for v in scored) if scored else False,
                     "zero means no turn produced a hindsight score and w == 1 everywhere")
        # A high skip count is not fatal (unscored turns fall back to w=1), but if
        # EVERY turn is skipped the arm silently degenerates to plain GRPO.
        report.check("not every turn was skipped",
                     bool(scored) and bool(skipped) and any(s > 0 for s in scored),
                     f"scored={scored[-1] if scored else 'n/a'} skipped={skipped[-1] if skipped else 'n/a'}")

        mean_w = _metric_values(text, "hpi/mean_weight")
        report.check("hpi/mean_weight near 1 (normalised weight)",
                     any(abs(v - 1.0) < 0.5 for v in mean_w) if mean_w else False,
                     f"{mean_w[-1]:.4f}" if mean_w else "missing")

        p75 = _metric_values(text, "hpi/abs_e_p75")
        if p75:
            report.check("hpi/abs_e_p75 > 0 (hindsight moved the teacher)",
                         p75[-1] > 0.0,
                         f"{p75[-1]:.6g} -> suggested ALFWORLD_HPI_SCALE")

    if gate in ("A", "B"):
        report.check("episodes completed", "rollout/mean_turns" in text or "alfworld-eval" in text)
        report.check("no engine timeout retries",
                     "[alfworld] /generate exceeded" not in text)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check ALFWorld NPU gate conditions.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--gate", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = check_log(args.log, gate=args.gate)
    print(report.render())
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "log": report.log_path,
                    "gate": args.gate,
                    "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in report.checks],
                    "gates": "PASS" if not report.failed else "FAIL",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
