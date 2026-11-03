#!/bin/bash
# Qwen3-4B entry point.
#
# Merged 2026-09-28 from the Server-B replica `cua/repos/opsd-slime-cua-rl`. All
# of the launcher logic now lives in run_math_opsd_slime_npu.sh; the size-specific
# values live in configs/models/qwen3-4b.sh. This file only picks the card.
#
# NOTE: the replica existed because this filesystem does not invalidate
# __pycache__ on mtime, so two conda envs sharing one tree can silently reuse each
# other's bytecode. The merge does not remove that hazard -- clear __pycache__ for
# any file you edit before running from a second env.
set -uo pipefail
export MODEL="${MODEL:-qwen3-4b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_math_opsd_slime_npu.sh" "$@"
