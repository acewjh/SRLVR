#!/bin/bash
# v2 Arm E -- GSPO on the per-turn layout.
#
# Byte-for-byte the GRPO arm (run_alfworld_v2_grpo.sh) except that
# _common_v2.sh passes `--advantage-estimator gspo`, which
# tasks/alfworld/loss.py turns into slime's sequence-level KL. Same rollout,
# same rows, same trajectory-mean reduction, same eps_clip, no HPI.
#
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_gspo.sh
HERE="$(cd "$(dirname "$0")" && pwd)"

export ARM=gspo
# Run identity comes from the repo's single naming rule (scripts/run_names.py),
# never from a hand-written string; `gspo` is a method in that vocabulary. RUN_NAME
# stays overridable so an operator can pin a name deliberately.
if [ -z "${RUN_NAME:-}" ]; then
  RUN_NAME=$(python "$HERE/../../scripts/run_names.py" --allocate \
               --task alfworld \
               --model "${RUN_NAME_MODEL:-qwen3-1p7b}" \
               --method gspo) || exit 1
fi
export RUN_NAME

exec bash "$HERE/_common_v2.sh"
