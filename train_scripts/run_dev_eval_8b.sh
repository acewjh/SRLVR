#!/bin/bash
# Qwen3-8B dev-eval entry point. All logic lives in run_dev_eval.sh; the
# size-specific values live in configs/models/qwen3-8b.sh.
#
# Note the actor budget: the 8B card is PP2, so this eval drives 2 actor NPUs and
# 6 rollout engines (see scripts/eval_gpu_defaults.sh, EVAL_ACTOR_NPUS).
set -uo pipefail
export MODEL="${MODEL:-qwen3-8b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dev_eval.sh" "$@"
