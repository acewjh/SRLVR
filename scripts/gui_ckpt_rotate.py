#!/usr/bin/env python3
"""Keep the newest checkpoint's optimizer state; older ones stay benchmark-only.

Policy mirrors ``scripts/hf_ckpt_rotate.py`` for the OpenApps GUI runs:

  * the newest ``iter_N`` (the one ``latest_checkpointed_iteration.txt`` points
    at) keeps its full resume state: weights + the ``optim_sidecar_rank*.pt``
    Adam moments written by ``methods/optim_sidecar.py``;
  * every older ``iter_M`` keeps only its weights (``*.distcp`` / ``common.pt``),
    which is all ``--load iter_M --debug-rollout-only`` needs for a benchmark;
  * never touch a directory whose mtime is inside ``--min-age`` seconds (a save
    may still be in flight), and never touch ``iter_N`` with N >= latest;
  * every removal is appended to ``<run>/ckpt_rotate.log``; ``--dry-run`` prints
    the same list without deleting.

This is the GUI analogue of the ALFWorld rotation: the newest checkpoint is the
resume point, older ones are kept only in the form the benchmark consumes.
It deletes optimizer *sidecars*, not model weights -- ``--load`` still works
for every retained iteration.

Usage:
  python scripts/gui_ckpt_rotate.py --run-dir runs/<run> --dry-run
  python scripts/gui_ckpt_rotate.py --run-dir runs/<run> --once
  python scripts/gui_ckpt_rotate.py --run-dir runs/<run> --watch --interval 600
"""
from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

SIDECAR_GLOB = "optim_sidecar_rank*.pt"
TRACKER = "latest_checkpointed_iteration.txt"
LOG_NAME = "ckpt_rotate.log"


def latest_iter(run_dir: Path) -> int:
    tracker = run_dir / TRACKER
    if not tracker.is_file():
        return -1
    digits = "".join(ch for ch in tracker.read_text().strip() if ch.isdigit())
    return int(digits) if digits else -1


def iter_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("iter_*"):
        if not path.is_dir():
            continue
        digits = "".join(ch for ch in path.name.split("_", 1)[1] if ch.isdigit())
        if digits:
            found.append((int(digits), path))
    return sorted(found)


def log(run_dir: Path, message: str, echo: bool = True) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    if echo:
        print(line, flush=True)
    with (run_dir / LOG_NAME).open("a") as handle:
        handle.write(line + "\n")


def rotate_once(run_dir: Path, min_age: float, dry_run: bool) -> int:
    latest = latest_iter(run_dir)
    if latest < 0:
        return 0
    removed = 0
    now = time.time()
    for iteration, path in iter_dirs(run_dir):
        if iteration >= latest:
            continue
        if now - path.stat().st_mtime < min_age:
            log(run_dir, f"skip {path.name}: modified {now - path.stat().st_mtime:.0f}s ago")
            continue
        sidecars = sorted(path.glob(SIDECAR_GLOB))
        if not sidecars:
            continue
        total = sum(f.stat().st_size for f in sidecars)
        if dry_run:
            log(run_dir, f"DRY-RUN would remove {len(sidecars)} sidecar(s) "
                         f"({total / 2**30:.1f} GiB) from {path.name}")
            removed += len(sidecars)
            continue
        for sidecar in sidecars:
            sidecar.unlink()
            removed += 1
        log(run_dir, f"trimmed {path.name}: removed {len(sidecars)} sidecar(s) "
                     f"({total / 2**30:.1f} GiB); weights kept for benchmark")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--min-age", type=float, default=600.0,
                        help="seconds a checkpoint dir must be idle before trimming")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    lock_path = run_dir / ".ckpt_rotate.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another rotator holds {lock_path}")

        while True:
            n = rotate_once(run_dir, args.min_age, args.dry_run)
            if args.once or args.dry_run:
                print(f"[gui-rotate] processed {n} sidecar file(s)", flush=True)
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
