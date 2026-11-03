"""Build slime-readable prompt datasets from the ALFWorld game files.

The prompt text is intentionally empty: the ALFWorld task instruction is only
known *after* ``env.reset()`` reads it out of the opening observation, so the
rollout builds every prompt itself. The dataset carries only the game identity in
``metadata``, which is what the rollout needs to instantiate the right episode.

Run this once per split (see ``scripts/setup_alfworld.sh``), then point
``--prompt-data`` at the produced JSONL with ``--input-key prompt
--metadata-key metadata``. ``--apply-chat-template`` must stay OFF: this harness
tokenises each turn explicitly so the teacher can reuse the student's exact
prompt ids.

The game index is cached (``--index``) because discovering 3.5k train games walks
the whole tree; the rollout must never pay that cost.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .env_adapter import discover_games, load_game_index, write_game_index
from .eval import CATEGORY_ORDER, category_of

__all__ = ["build_split_dataset", "write_index", "main", "SAMPLE_COUNT_MAX"]

#: An example dataset, sized for smoke tests and quick iteration.
SAMPLE_COUNT_MAX = 512


def _dataset_root() -> str:
    return str(Path(__file__).resolve().parent / "configs" / "data")


def build_split_dataset(
    split: str,
    out_path: str,
    *,
    limit: int | None = None,
    games=None,
) -> int:
    """Write one split's prompt JSONL. Returns the number of records."""
    games = games if games is not None else discover_games(split, limit=limit)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for game in games:
            handle.write(
                json.dumps(
                    {
                        # Rendering is deferred to the rollout (see module docstring).
                        "prompt": "",
                        "metadata": {
                            "game_id": game.game_id,
                            "game_path": game.path,
                            "task_type": game.task_type,
                            "split": game.split,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(games)


def write_index(out_path: str, *, splits=("train", "valid_seen", "valid_unseen")) -> dict:
    return write_game_index(splits, out_path)


def write_eval_config(out_path: str, data_dir: str, *, temperature: float = 0.6) -> None:
    """Write the ``--eval-config`` consumed by the Base eval arm.

    Sampling is the FROZEN eval protocol (T=0.6 / top_p 0.95 / top_k 20) -- this
    is deliberately NOT the training sampler (T=1.0 / top_p=1.0 / top_k=-1, see
    CAMPAIGN_QUEUE_1p7b.md section 0); it is applied identically to every arm so
    the comparison stays fair. Do not "align" it to the training sampler without
    re-freezing the protocol on both lanes.

    ``max_response_len`` must stay >= 4096: at 1024 roughly 22% of ALFWorld turns
    are cut mid-action, which turns the score into a truncation artifact rather
    than a measurement (CAMPAIGN_QUEUE_1p7b.md section 19.2). The eval driver's
    preflight refuses any config below that, so keep the two in step.
    """
    payload = {
        "eval": {
            "defaults": {},
            "datasets": [
                {
                    "name": split,
                    "path": str(Path(data_dir) / f"{split}.jsonl"),
                    "input_key": "prompt",
                    "metadata_key": "metadata",
                    "n_samples_per_eval_prompt": 1,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_response_len": 4096,
                }
                for split in ("valid_seen", "valid_unseen")
            ],
        }
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_smoke_eval_config(out_path: str, index_path: str, *, per_category: int = 1) -> dict:
    """A tiny eval config covering every task category, for the NPU smoke gates.

    Gates A/B need a handful of episodes that exercise all six task types, not the
    full 274-game split. One game per category per split keeps the smoke bounded
    while still catching a category-specific parser or env failure.
    """
    index = load_game_index(index_path)
    datasets = []
    picked: dict[str, int] = {}
    for split in ("valid_seen", "valid_unseen"):
        games = index.get(split, [])
        selected, seen = [], {}
        for game in games:
            label = category_of(game.task_type)
            if seen.get(label, 0) >= per_category:
                continue
            seen[label] = seen.get(label, 0) + 1
            selected.append(game)
        picked[split] = len(selected)
        path = Path(out_path).parent / "data" / f"smoke_{split}.jsonl"
        build_split_dataset(split, str(path), games=selected)
        datasets.append(
            {
                "name": f"smoke_{split}",
                "path": str(path),
                "input_key": "prompt",
                "metadata_key": "metadata",
                "n_samples_per_eval_prompt": 1,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_response_len": 1024,
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"eval": {"defaults": {}, "datasets": datasets}}, indent=2), encoding="utf-8")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ALFWorld prompt datasets for slime.")
    parser.add_argument("--out-dir", default=_dataset_root())
    parser.add_argument("--index", default=str(Path(_dataset_root()) / "game_index.json"))
    parser.add_argument("--train-limit", type=int, default=None, help="cap train games (default: all)")
    parser.add_argument(
        "--eval-limit", type=int, default=None, help="cap eval games per split (default: all)"
    )
    parser.add_argument("--splits", default="train,valid_seen,valid_unseen")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_index = write_index(args.index)
    print(f"[prepare] game index -> {args.index}: {counts_index}")

    from .env_adapter import load_game_index

    index = load_game_index(args.index)
    counts = {}
    for split in args.splits.split(","):
        split = split.strip()
        if split not in index:
            print(f"[prepare] skipping unknown split {split!r}", file=sys.stderr)
            continue
        limit = args.train_limit if split == "train" else args.eval_limit
        path = out_dir / f"{split}.jsonl"
        counts[split] = build_split_dataset(split, str(path), limit=limit, games=index[split][:limit] if limit else index[split])
        print(f"[prepare] {split}: {counts[split]} games -> {path}")

    by_category: dict[str, int] = {}
    for split in ("valid_seen", "valid_unseen"):
        for game in index.get(split, []):
            label = category_of(game.task_type)
            by_category[label] = by_category.get(label, 0) + 1
    print("[prepare] eval games per category:", {k: by_category.get(k, 0) for k in CATEGORY_ORDER})

    config_dir = Path(__file__).resolve().parent / "configs"
    eval_config = config_dir / "eval_alfworld.json"
    write_eval_config(str(eval_config), str(out_dir))
    print(f"[prepare] eval config -> {eval_config}")

    smoke_config = config_dir / "eval_alfworld_smoke.json"
    picked = write_smoke_eval_config(str(smoke_config), args.index)
    print(f"[prepare] smoke eval config -> {smoke_config} {picked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
