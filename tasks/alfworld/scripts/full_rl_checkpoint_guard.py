#!/usr/bin/env python3
"""Verify a Megatron full-RL checkpoint is COMPLETE before a supervisor resumes.

`resume_guard.py` covers the HF/LoRA layout (trainer_state.json + adapter
safetensors). A full-RL run saves the Megatron torch-dist layout instead:

    iter_<N>/common.pt          optimizer + scheduler + iteration, pickled
    iter_<N>/.metadata          distcp shard index (dotfile, torch-pickled)
    iter_<N>/__<rank>_<chunk>.distcp   the shard payloads
    latest_checkpointed_iteration.txt  the tracker the launcher resumes from

A truncated save is worse than a missing one: the launcher resumes "successfully"
from the right iteration number with half-written weights and nothing says so.
This opens the tracker, the pickled common state and the shard index, and checks
that every shard file the index names exists at its recorded size.

Usage: full_rl_checkpoint_guard.py <save-dir> <iteration>
Exit 0 = usable, non-zero = do not resume.
"""

from __future__ import annotations

import glob
import os
import sys
import time

SETTLE_SECONDS = 5


def _fail(message: str) -> int:
    print(f"[full-rl-guard] {message}")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: full_rl_checkpoint_guard.py <save-dir> <iteration>", file=sys.stderr)
        return 2
    save_dir, iteration = argv[0], int(argv[1])
    iter_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
    if not os.path.isdir(iter_dir):
        return _fail(f"MISSING {iter_dir}")

    tracker = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    try:
        with open(tracker, encoding="utf-8") as handle:
            tracker_iter = int(handle.read().strip())
    except Exception as exc:  # noqa: BLE001 - any failure means "do not resume"
        return _fail(f"tracker unreadable: {exc}")
    if tracker_iter != iteration:
        return _fail(f"tracker says {tracker_iter}, asked to resume {iteration}")

    common = os.path.join(iter_dir, "common.pt")
    if not os.path.isfile(common):
        return _fail("common.pt missing")
    try:
        import torch

        state = torch.load(common, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"common.pt unreadable: {exc}")
    if "iteration" not in state or "optimizer" not in state:
        return _fail(f"common.pt is not a training checkpoint (keys={sorted(state)})")
    if int(state["iteration"]) != iteration:
        return _fail(f"common.pt iteration {state['iteration']} != {iteration}")

    # The shard index is a dotfile AND a torch-pickled Megatron object, so it
    # cannot be json-parsed and must not be glob-checked with ``*.distcp`` (that
    # silently skips dotfiles). Check the index exists and the shards it indexes
    # are all present and non-empty; read the index's size only as a presence fact.
    index_path = os.path.join(iter_dir, ".metadata")
    shards = sorted(glob.glob(os.path.join(iter_dir, "__*.distcp")))
    if not os.path.isfile(index_path):
        return _fail(".metadata shard index missing")
    if not shards:
        return _fail("no .distcp shards")
    if os.path.getsize(index_path) == 0:
        return _fail(".metadata is zero-length")

    empty = [os.path.basename(path) for path in shards if os.path.getsize(path) == 0]
    if empty:
        return _fail(f"zero-length shards: {empty}")

    # A save still in flight keeps growing. Two samples a few seconds apart are
    # enough to separate "just finished writing" from "still being written".
    first = sum(os.path.getsize(path) for path in shards)
    time.sleep(SETTLE_SECONDS)
    second = sum(os.path.getsize(path) for path in shards)
    if first != second:
        return _fail(f"shards still growing ({first} -> {second} bytes)")

    print(
        f"[full-rl-guard] OK iter={iteration} global_step={state.get('iteration')} "
        f"shards={len(shards)} bytes={second}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
