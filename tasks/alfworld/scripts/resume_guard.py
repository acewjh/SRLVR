"""Verify a checkpoint is COMPLETE before a supervisor resumes from it.

A truncated checkpoint is worse than a missing one: the launcher's resume path
loads it without complaint, the run continues from the right iteration number with
the wrong weights, and nothing in the log says so. This opens the two files that
have to be readable for a resume to be meaningful:

  ``iter_<N>/trainer_state.json``          -- reports ``global_step``
  ``iter_<N>/adapter_model.safetensors``   -- the LoRA weights themselves

(the second name is accepted as ``adapter/adapter_model.safetensors`` and as the
tokenizer-adjacent layout, because the writer has moved it before).

Usage: resume_guard.py <run-dir> <iteration>
Exit 0 = usable, non-zero = do not resume.
"""

from __future__ import annotations

import glob
import json
import os
import sys


def _find_adapter(iter_dir: str) -> str | None:
    for pattern in (
        "adapter_model.safetensors",
        "adapter/adapter_model.safetensors",
        "*adapter*.safetensors",
    ):
        hits = sorted(glob.glob(os.path.join(iter_dir, pattern)))
        if hits:
            return hits[0]
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: resume_guard.py <run-dir> <iteration>", file=sys.stderr)
        return 2
    run_dir, iteration = argv[0], argv[1]
    iter_dir = os.path.join(run_dir, f"iter_{int(iteration):07d}")
    if not os.path.isdir(iter_dir):
        print(f"[resume-guard] MISSING {iter_dir}")
        return 1

    state_path = os.path.join(iter_dir, "trainer_state.json")
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - any failure means "do not resume"
        print(f"[resume-guard] trainer_state.json unreadable: {exc}")
        return 1
    global_step = state.get("global_step")
    if not isinstance(global_step, int):
        print(f"[resume-guard] trainer_state.json has no integer global_step: {global_step!r}")
        return 1

    adapter = _find_adapter(iter_dir)
    if adapter is None:
        print(f"[resume-guard] no adapter safetensors under {iter_dir}")
        return 1
    try:
        from safetensors import safe_open

        with safe_open(adapter, framework="pt") as handle:
            names = list(handle.keys())
    except Exception as exc:  # noqa: BLE001
        print(f"[resume-guard] adapter unreadable: {exc}")
        return 1
    if not names:
        print(f"[resume-guard] adapter {adapter} has no tensors")
        return 1

    print(
        f"[resume-guard] OK iter={iteration} global_step={global_step} "
        f"adapter={os.path.basename(adapter)} tensors={len(names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
