#!/usr/bin/env bash
# Start the 2026-10-04 two-node campaign: HSI on Node0, matched GRPO on Node1.
#
# Both arms get their own 8 cards; there is NO cross-node distributed training.
# Run names are allocated by the repo's single naming rule and written to disk
# before anything starts, so the two nodes cannot disagree about which directory
# is theirs and a re-run cannot silently adopt a name from an earlier attempt.
#
# Idempotent in the useful sense: if both supervisors are already running it says
# so and exits without starting a second one (the supervisors themselves are
# flock-guarded, so this is belt and braces).
#
# Usage: launch_v2_campaign.sh [num_rollout]
set -uo pipefail

STEPS="${1:-100}"
R=/workspace/cua/repos/opsd-slime
BG=/workspace/logs/bg_candidate_hsi
mkdir -p "$BG"

NODE0_RUN_FILE="$BG/NODE0_V2_RUN_NAME.txt"
NODE1_RUN_FILE="$BG/NODE1_V2_RUN_NAME.txt"

WORKER1="$(echo "${VC_WORKER_HOSTS:-}" | cut -d, -f2)"
[ -n "$WORKER1" ] || { echo "FATAL: VC_WORKER_HOSTS does not name a second worker" >&2; exit 2; }

SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -p 2222 "$WORKER1")

alloc() { # $1 = method slug
  python "$R/scripts/run_names.py" --allocate --task alfworld --model qwen3-4b \
    --method "$1" --variant s42
}

# Reuse an existing name only when its supervisor is still alive; otherwise mint a
# fresh one. Adopting a name whose run is dead would resume/overwrite history.
node0_run=""
if [ -s "$NODE0_RUN_FILE" ]; then
  candidate=$(cat "$NODE0_RUN_FILE")
  if pgrep -f "supervise_v2[.]sh $candidate" >/dev/null 2>&1; then node0_run="$candidate"; fi
fi
[ -n "$node0_run" ] || node0_run=$(alloc hsi) || exit 1
echo "$node0_run" > "$NODE0_RUN_FILE"

node1_run=""
if [ -s "$NODE1_RUN_FILE" ]; then
  candidate=$(cat "$NODE1_RUN_FILE")
  if "${SSH[@]}" "pgrep -f 'supervise_v2[.]sh $candidate' >/dev/null" 2>/dev/null; then node1_run="$candidate"; fi
fi
[ -n "$node1_run" ] || node1_run=$(alloc grpo) || exit 1
echo "$node1_run" > "$NODE1_RUN_FILE"

echo "[campaign] node0 HSI  -> $node0_run"
echo "[campaign] node1 GRPO -> $node1_run"

# --- Node0 (this node) --------------------------------------------------------
if pgrep -f "supervise_v2[.]sh $node0_run" >/dev/null 2>&1; then
  echo "[campaign] node0 supervisor already running for $node0_run"
else
  cd "$R" || exit 1
  setsid nohup bash "$R/tasks/alfworld/scripts/supervise_v2.sh" \
    "$node0_run" hsi_v2 hsi "$STEPS" \
    >> "$BG/node0_v2_supervisor.out" 2>&1 < /dev/null &
  echo "[campaign] node0 supervisor started (pid $!)"
fi

# --- Node1 (worker-1, port 2222 -- NOT 22) ------------------------------------
if "${SSH[@]}" "pgrep -f 'supervise_v2[.]sh $node1_run' >/dev/null" 2>/dev/null; then
  echo "[campaign] node1 supervisor already running for $node1_run"
else
  "${SSH[@]}" "cd $R && setsid nohup bash $R/tasks/alfworld/scripts/supervise_v2.sh $node1_run grpo_v2 grpo $STEPS >> $BG/node1_v2_supervisor.out 2>&1 < /dev/null & sleep 2; pgrep -f 'supervise_v2[.]sh $node1_run' | head -1"
  echo "[campaign] node1 supervisor start requested"
fi

cat <<EOF

Started. Watch with:
  tail -f $R/logs/$node0_run.log              # node0 HSI
  tail -f $BG/node0_v2_supervisor.out
  ${SSH[*]} "tail -f $R/logs/$node1_run.log"  # node1 GRPO

Checkpoints (0-based rollout ids, save_interval 25): iter_24 / 49 / 74 / 99
Evaluation is Plan B: it starts automatically once each arm reaches iter_99.
EOF
