#!/bin/bash
# Create a checkpoint staging dir so slime's --load resolves a SPECIFIC training
# iteration (slime/Megatron read latest_checkpointed_iteration.txt from --load).
# The original run dirs are never modified; the iter dir is symlinked.
#
# Usage: bash scripts/make_eval_staging.sh <RUN_DIR> <ITER> <STAGING_DIR>
#   e.g. bash scripts/make_eval_staging.sh runs/outcome_v2_slime 24 runs/eval_staging/v2_25

set -euo pipefail

RUN_DIR="${1:?usage: make_eval_staging.sh <RUN_DIR> <ITER> <STAGING_DIR>}"
ITER="${2:?usage: make_eval_staging.sh <RUN_DIR> <ITER> <STAGING_DIR>}"
STAGING="${3:?usage: make_eval_staging.sh <RUN_DIR> <ITER> <STAGING_DIR>}"

SRC_ITER=$(printf "%s/iter_%07d" "$RUN_DIR" "$ITER")
[ -d "$SRC_ITER" ] || { echo "missing checkpoint: $SRC_ITER"; exit 1; }

mkdir -p "$STAGING"
rm -rf "$STAGING/iter_$ITER" 2>/dev/null || true
ITER_NAME=$(printf "iter_%07d" "$ITER")
ln -sfn "$(readlink -f "$SRC_ITER")" "$STAGING/$ITER_NAME"
printf "%07d\n" "$ITER" > "$STAGING/latest_checkpointed_iteration.txt"
echo "staging ready: $STAGING -> $ITER_NAME of $RUN_DIR"
