#!/usr/bin/env bash
# Stop a GRPO run the moment it starts destroying its own output, and keep the
# corpse on disk for diagnosis.
#
# WHY. The 2026-10-01 runs collapsed at ~rollout 28 and then burned 21 more
# rollouts producing bit-identical checkpoints at grad_norm exactly 0 -- the
# policy had already destroyed its `</think>` structure, so no action parsed, no
# episode succeeded, every GRPO group had zero reward variance, and there was no
# gradient left to recover with. Nothing about that is visible in `success_rate`
# until it is far too late to matter.
#
# Deliberately NOT trigger-happy: an occasional all-zero-reward group is normal
# (that is what `--use-opsd` gives you on an unsolved task) and must not stop a
# run. Only PERSISTENT, whole-batch collapse does.
#
# Usage: bash collapse_guard.sh <run-name> [<log>]
set -uo pipefail

NAME="${1:?usage: collapse_guard.sh <run-name> [log]}"
R=/workspace/cua/repos/opsd-slime
LOG="${2:-$R/logs/$NAME.log}"
DIR="$R/runs/$NAME"
STATE="/tmp/afw_guard_$NAME"
mkdir -p "$STATE"
GLOG="$STATE/guard.log"

# Thresholds, with the reasoning for each:
#   LEN_FLOOR       a turn's mean response falling under ~120 tokens for 3
#                   consecutive rollouts. Healthy runs sit at 300-800; the
#                   collapse drove it to 16-30.
#   ZERO_STD_STREAK all 4 groups at zero reward variance for 4 consecutive
#                   rollouts. Single rollouts hit this by chance; four in a row
#                   at 4/4 is the frozen state.
#   DEAD_STREAK     success_rate == 0 for 4 consecutive rollouts.
LEN_FLOOR="${LEN_FLOOR:-120}"
LEN_STREAK="${LEN_STREAK:-3}"
ZERO_STD_STREAK="${ZERO_STD_STREAK:-4}"
DEAD_STREAK="${DEAD_STREAK:-4}"
# Rollouts to observe before any verdict. The spec's smoke gate covers the first
# few; the guard is for the slow burn that follows.
MIN_ROLLOUTS="${MIN_ROLLOUTS:-6}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$GLOG"; }

# Tails of the per-rollout metric series. One value per rollout, oldest last.
#
# Every value is stripped to digits first. The first version of this script used
# `grep -c ... || echo 0` bare, which yields the two lines "0\n0" when there is
# no match; the numeric test then errors, execution FALLS THROUGH to the
# detection block, and empty input makes the awk predicates vacuously true -- so
# the guard would have killed a healthy run at rollout 0. Never let a guard's
# failure mode be "stop the thing it is guarding".
series() {
  grep -oE "'$1': [0-9.]+" "$LOG" 2>/dev/null | awk '{print $2}' | tail -"$2" | tr -dc '0-9.\n'
}
count_rollouts() {
  local c
  c=$(grep -c 'perf/rollout_time' "$LOG" 2>/dev/null) || true
  c=$(printf '%s' "$c" | head -1 | tr -dc '0-9')
  printf '%s' "${c:-0}"
}

say "guard watching $NAME (log=$LOG)"
last_rollouts=0
while true; do
  sleep 120

  # One launcher owns the run; without it the guard has nothing to guard.
  if ! ps -eo cmd | grep -qE "[t]rain[.]py.*--save $DIR\b"; then
    say "trainer for $NAME is gone; guard exiting"
    exit 0
  fi

  n=$(count_rollouts)
  [ "$n" -le "$last_rollouts" ] && continue
  last_rollouts=$n
  [ "$n" -lt "$MIN_ROLLOUTS" ] && continue

  reason=""
  lens=$(series 'rollout/response_len/mean' "$LEN_STREAK")
  zeros=$(series 'rollout/zero_std/count_0.0' "$ZERO_STD_STREAK")
  dead=$(series 'rollout/success_rate' "$DEAD_STREAK")

  # A verdict needs a full window of REAL numbers, and every one of them must
  # point the same way. Two ways this went wrong before, both costing real runs:
  #   1. awk exits 0 when its predicate never fires, so `! awk ...` reads as
  #      "collapse" on EMPTY input;
  #   2. `printf '%s\n' "$multiline"` appends a trailing blank line, and awk then
  #      compares "" against a number as a STRING (always false) -- so a window of
  #      1066/912/841 was declared "below 120" and a healthy run was killed.
  # The helper below makes the two conditions explicit instead of relying on
  # negation of a predicate: at least `need` numeric lines, and ALL of them on the
  # collapsing side. Anything else is "not enough evidence", never "collapse".
  window_verdict() { # $1=series $2=need $3=test(lt|gt) $4=threshold
    printf '%s\n' "$1" | awk -v need="$2" -v how="$3" -v thr="$4" '
      /^[0-9]+(\.[0-9]+)?$/ { n++; if (how == "lt") { if ($1+0 <  thr) bad++ }
                                     else             { if ($1+0 >  thr) bad++ } }
      END { exit !(n >= need && bad == n) }'
  }

  no_numbers() { [ "$(printf '%s\n' "$1" | grep -cE '^[0-9]+(\.[0-9]+)?$')" -lt "$2" ]; }

  if no_numbers "$lens" "$LEN_STREAK"; then
    say "rollout $n: length window incomplete ($(printf '%s ' $lens)) -- not enough evidence"
  elif window_verdict "$lens" "$LEN_STREAK" lt "$LEN_FLOOR"; then
    reason="response_len/mean below $LEN_FLOOR for $LEN_STREAK rollouts: $(printf '%s ' $lens)"
  elif no_numbers "$zeros" "$ZERO_STD_STREAK"; then
    say "rollout $n: zero_std window incomplete -- not enough evidence"
  elif window_verdict "$zeros" "$ZERO_STD_STREAK" lt 4; then
    reason="all 4 groups at zero reward variance for $ZERO_STD_STREAK rollouts"
  elif no_numbers "$dead" "$DEAD_STREAK"; then
    say "rollout $n: success window incomplete -- not enough evidence"
  elif window_verdict "$dead" "$DEAD_STREAK" gt 0; then
    reason="success_rate 0 for $DEAD_STREAK consecutive rollouts"
  fi
  # The last word, and the reason this guard may not kill on a hunch: the lengths
  # above are a proxy. Only stop when the proxy AND the outcome agree.
  if [ -n "$reason" ] && [ "$(printf '%s\n' "$dead" | tail -1)" != "0" ]; then
    say "suppressing verdict ($reason): latest success_rate is not 0, so this is a proxy artefact"
    reason=""
  fi

  if [ -n "$reason" ]; then
    say "COLLAPSE DETECTED at rollout $n: $reason"
    say "stopping $NAME; dumps and checkpoints are left in place"
    echo "$(date '+%F %T') rollout=$n $reason" > "$STATE/COLLAPSED"
    # Kill only this run's trainer, by PID, so the other node is untouched.
    for pid in $(ps -eo pid,cmd | grep -E "[t]rain[.]py.*--save $DIR\b" | awk '{print $1}'); do
      kill "$pid" 2>/dev/null
    done
    exit 1
  fi

  say "rollout $n ok: len=$(echo "$lens" | tail -1) zero_std=$(echo "$zeros" | tail -1) succ=$(echo "$dead" | tail -1)"
done
