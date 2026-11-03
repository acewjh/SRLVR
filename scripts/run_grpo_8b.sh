#!/bin/bash
# Qwen3-8B GRPO control arm.
#
# All logic lives in scripts/run_grpo.sh; the size-specific values live in
# configs/models/qwen3-8b.sh (PP2 x DP2 -- same geometry as the 8B Ours arm, so a
# size's two arms differ only in the method).
set -uo pipefail
export MODEL="${MODEL:-qwen3-8b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_grpo.sh" "$@"
