#!/usr/bin/env bash
# Minimal verification run for one V2 arm, into a `_`-prefixed exempt run dir.
#
# The V2 protocol changes three things that only a real trainer can prove out:
#   1. the sampler is T=1.0 / top_p=1.0 / top_k=-1 (a negative int flag, so this
#      is also the check that slime's argparse accepts it),
#   2. the HSI scorer forwards the LIVE actor from inside the postprocess hook and
#      bypasses its output layer (the sparse-projection path),
#   3. the section-6 old-policy contract assertion fires on the first loss call.
# None of those is reachable from a CPU unit test, so this runs 2 cycles and looks
# at the log for the specific lines each one emits.
#
# Usage: smoke_v2.sh <hsi_v2|grpo_v2> [steps]
set -uo pipefail

ARM_CASE="${1:?usage: smoke_v2.sh <hsi_v2|grpo_v2> [steps]}"
STEPS="${2:-2}"
R=/workspace/cua/repos/opsd-slime
STAMP=$(date '+%Y%m%d_%H%M%S')
RUN="_smoke_v2_${ARM_CASE}_${STAMP}"
LOG="$R/logs/$RUN.log"
STATE="$R/runs/$RUN/_smoke_state"
mkdir -p "$STATE"

cd "$R" || exit 1
exec 9>&- 2>/dev/null || true

# A one-card-each split keeps the smoke cheap while still exercising both roles.
# Absolute path for the launcher: a `bash -c` child inherits the parent's cwd,
# but relying on that made the Node1 smoke die with "No such file or directory"
# while the identical Node0 run worked -- not worth re-deriving why.
setsid nohup env \
  RUN_NAME="$RUN" NUM_ROLLOUT="$STEPS" SAVE_INTERVAL=1 \
  bash -c 'exec 9>&-; exec bash "$1" "$2"' _ "$R/tasks/alfworld/scripts/launch_c64.sh" "$ARM_CASE" \
  >> "$STATE/launch.out" 2>&1 < /dev/null &
echo "$! $RUN" > "$STATE/smoke.pid"

echo "RUN=$RUN"
echo "LOG=$LOG"
echo "STATE=$STATE"
