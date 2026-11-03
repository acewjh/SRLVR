"""ALFWorld evaluation aggregation.

Primary paper metric: **success rate**. Reported overall, per split
(seen/unseen) and per task category. Secondary diagnostics (mean turns,
invalid-action rate, truncation rate) are reported because they explain a
success-rate change, not because they are results.

Task categories are the six official ALFWorld task types; the short labels below
are the conventional ones (Pick / Look / Clean / Heat / Cool / Pick2).
"""

from collections import defaultdict
from dataclasses import dataclass, field

__all__ = [
    "TASK_TYPE_LABELS",
    "CATEGORY_ORDER",
    "EvalSample",
    "EvalReport",
    "aggregate",
    "format_report",
]

#: ALFWorld task_type string -> conventional short label.
TASK_TYPE_LABELS = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}

CATEGORY_ORDER = ("Pick", "Look", "Clean", "Heat", "Cool", "Pick2")


def category_of(task_type: str) -> str:
    """Short label for an ALFWorld task_type; unknown types pass through as-is."""
    return TASK_TYPE_LABELS.get(task_type, task_type)


@dataclass
class EvalSample:
    """One finished evaluation episode."""

    task_id: str
    task_type: str
    split: str
    success: bool
    num_turns: int
    truncated: bool
    invalid_action_count: int
    generated_tokens: int = 0


@dataclass
class SplitStats:
    n: int = 0
    successes: int = 0
    turns: int = 0
    truncated: int = 0
    invalid_actions: int = 0
    generated_tokens: int = 0
    by_category: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def add(self, sample: EvalSample) -> None:
        self.n += 1
        self.successes += int(sample.success)
        self.turns += sample.num_turns
        self.truncated += int(sample.truncated)
        self.invalid_actions += sample.invalid_action_count
        self.generated_tokens += sample.generated_tokens
        bucket = self.by_category[category_of(sample.task_type)]
        bucket[0] += int(sample.success)
        bucket[1] += 1

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "success_rate": _ratio(self.successes, self.n),
            "mean_turns": _ratio(self.turns, self.n),
            "truncation_rate": _ratio(self.truncated, self.n),
            "invalid_action_rate": _ratio(self.invalid_actions, self.turns),
            "mean_generated_tokens": _ratio(self.generated_tokens, self.n),
            "by_category": {
                cat: {"n": total, "success_rate": _ratio(hits, total)}
                for cat, (hits, total) in sorted(self.by_category.items())
            },
        }


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass
class EvalReport:
    overall: SplitStats = field(default_factory=SplitStats)
    by_split: dict[str, SplitStats] = field(default_factory=lambda: defaultdict(SplitStats))

    def add(self, sample: EvalSample) -> None:
        self.overall.add(sample)
        self.by_split[sample.split].add(sample)

    def as_dict(self) -> dict:
        payload = {"overall": self.overall.as_dict()}
        for split in ("valid_seen", "valid_unseen"):
            if split in self.by_split:
                payload[split] = self.by_split[split].as_dict()
        payload["seen_success_rate"] = payload.get("valid_seen", {}).get("success_rate", 0.0)
        payload["unseen_success_rate"] = payload.get("valid_unseen", {}).get("success_rate", 0.0)
        return payload


def aggregate(samples: list[EvalSample]) -> EvalReport:
    report = EvalReport()
    for sample in samples:
        report.add(sample)
    return report


def format_report(report: EvalReport) -> str:
    """Human-readable summary for RESULTS.md / logs."""
    payload = report.as_dict()
    overall = payload["overall"]
    lines = [
        f"ALFWorld eval: n={overall['n']}  success_rate={overall['success_rate']:.4f}",
        f"  seen   success_rate={payload['seen_success_rate']:.4f}",
        f"  unseen success_rate={payload['unseen_success_rate']:.4f}",
        f"  mean_turns={overall['mean_turns']:.2f}  truncation_rate={overall['truncation_rate']:.4f}"
        f"  invalid_action_rate={overall['invalid_action_rate']:.4f}",
        "  per-category success:",
    ]
    for category in CATEGORY_ORDER:
        stats = overall["by_category"].get(category)
        if stats:
            lines.append(f"    {category:<6} n={stats['n']:<4} success_rate={stats['success_rate']:.4f}")
    return "\n".join(lines)
