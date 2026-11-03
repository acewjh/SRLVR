#!/bin/bash
# ============================================================
# One-shot CPU verification gate for the ALFWorld HPI harness.
#
# Runs entirely on CPU: no NPU, no Ray, no SGLang, no network. It is safe to run
# while another experiment owns the accelerators, which is the whole point -- the
# correctness contracts can be verified before the NPU gates are even reachable.
#
# Exits 0 only if every check passes. Prints a PASS/FAIL summary at the end.
#
# Usage: bash tasks/alfworld/v1/scripts/smoke_alfworld_cpu.sh
# ============================================================
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
OVERLAY=$BASE/envs/alfworld_overlay
OV_PY="$OVERLAY/bin/python"

cd "$PROJ" || exit 1

FAILED=0
declare -a RESULTS

record() {  # record <name> <rc>
  if [ "$2" -eq 0 ]; then RESULTS+=("PASS  $1"); else RESULTS+=("FAIL  $1"); FAILED=1; fi
}

run() {  # run <name> <cmd...>
  local name="$1"; shift
  echo "----------------------------------------------------------------"
  echo ">>> $name"
  echo "----------------------------------------------------------------"
  "$@"
  local rc=$?
  record "$name" "$rc"
  return $rc
}

# ---------------------------------------------------------------------------
# 0. stale bytecode
#
# This filesystem does not reliably invalidate __pycache__ on mtime, so an edited
# module can keep running its previous version. Clearing is cheap insurance and
# has already masked one real bug here.
# ---------------------------------------------------------------------------
find "$PROJ/tasks.alfworld.v1" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

if [ ! -x "$OV_PY" ]; then
  echo "FATAL: overlay venv missing at $OVERLAY -- run setup_alfworld.sh first" >&2
  exit 1
fi
export PYTHONPATH="$PROJ/frameworks/slime-ascend:$PROJ"
export ALFWORLD_HPI_DATA="${ALFWORLD_HPI_DATA:-$BASE/data/alfworld/json_2.1.1}"
# The base (cua-rl) site-packages, read off the overlay interpreter's own
# sys.path -- the overlay venv is built with --system-site-packages, so cua-rl's
# directory is already there and this needs no guessing about paths or versions.
export CUA_RL_SITE_PACKAGES="$($OV_PY -c 'import sys; print(next((p for p in sys.path if "cua-rl" in p), ""))' 2>/dev/null)"
[ -n "$CUA_RL_SITE_PACKAGES" ] || echo "[smoke] WARN: could not locate the cua-rl site-packages; the isolation check will skip" >&2

run "harness imports" "$OV_PY" -c "
import tasks.alfworld.v1.rollout, tasks.alfworld.v1.loss, tasks.alfworld.v1.train_data, tasks.alfworld.v1.reward, tasks.alfworld.v1.report_eval
print('tasks.alfworld.v1 imports OK')
" || true

run "ALFWorld packages import" "$OV_PY" -c "
from tasks.alfworld.v1.paths import ensure_alfworld_importable
import textworld, alfworld
print('textworld', textworld.__version__)
print('alfworld OK')
" || true

run "dataset discovery" "$OV_PY" -c "
from tasks.alfworld.v1.env_adapter import discover_games
for split in ('train', 'valid_seen', 'valid_unseen'):
    games = discover_games(split)
    assert games, f'{split} produced no games'
    print(f'{split}: {len(games)} games')
" || true

run "environment reset + step" "$OV_PY" -c "
from tasks.alfworld.v1.env_adapter import AlfWorldEnv, discover_games
game = discover_games('valid_seen', limit=1)[0]
with AlfWorldEnv(game, max_steps=6) as env:
    first = env.reset()
    assert first.observation and first.admissible_actions and env.task_description
    second = env.step(first.admissible_actions[0])
    assert second.won in (True, False)
    assert not (second.won and not second.done), 'won=True with done=False is the batch-unwrap bug'
print('reset/step OK; task:', env.task_description)
" || true

run "privilege + leakage contracts" "$OV_PY" -m pytest tasks/alfworld/v1/tests/test_alfworld_privilege.py -q --tb=short -p no:cacheprovider || true

run "prompt + parser + past-CoT stripping" "$OV_PY" -m pytest tasks/alfworld/v1/tests/test_alfworld_prompts.py -q --tb=short -p no:cacheprovider || true

run "layout + loss mask + token alignment" "$OV_PY" -m pytest tasks/alfworld/v1/tests/test_alfworld_layout.py -q --tb=short -p no:cacheprovider || true

run "rollout end-to-end (scripted engine)" "$OV_PY" -m pytest tasks/alfworld/v1/tests/test_alfworld_rollout.py -q --tb=short -p no:cacheprovider || true

run "env + eval aggregation" "$OV_PY" -m pytest tasks/alfworld/v1/tests/test_alfworld_env.py -q --tb=short -p no:cacheprovider || true

run "loss weight degenerates to GRPO when alpha=0" "$OV_PY" -c "
import torch
from tasks.alfworld.v1.loss import compute_hindsight_weights
turn_ids = [[0,0,0,-1,-1,1,1]]
e = [[torch.tensor([3.0, 0.5, 0.1]), torch.tensor([0.2, 0.4])]]
w, stats = compute_hindsight_weights(e, turn_ids, [7], alpha=0.0, scale=1.0)
assert torch.allclose(w, torch.ones(7)), w
print('alpha=0 -> w == 1 exactly; stats:', stats)
" || true

echo
echo "================================================================"
echo "ALFWorld CPU smoke summary"
echo "================================================================"
for line in "${RESULTS[@]}"; do echo "  $line"; done
echo "----------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "SMOKE_ALFWORLD_CPU = PASS"
else
  echo "SMOKE_ALFWORLD_CPU = FAIL"
fi
exit $FAILED
