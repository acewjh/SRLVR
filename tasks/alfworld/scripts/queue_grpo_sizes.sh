#!/usr/bin/env bash
# One queue per node: 3-size GRPO campaign (1.7B / 4B / 8B) + downstream ALFWorld eval.
#
#   bash queue_grpo_sizes.sh 0     # worker-0
#   bash queue_grpo_sizes.sh 1     # worker-1
#
# WHY THIS EXISTS, AND WHAT IT PINS
# ---------------------------------
# Every ALFWorld v2 GRPO number collected before 2026-10-01 is void: the custom
# loss returned a per-trajectory MEAN where slime's contract requires a token SUM
# (cp_utils.py:120), so the framework divided a second time and the gradient came
# out ~400x too small -- 60x below `--clip-grad`, i.e. clipping never fired and the
# policy barely moved. This queue re-runs the baselines with that fixed
# (OPSD_V2_LOSS_NORM=sum) and nothing else changed, so the new 4B arm is directly
# comparable to the already-collected cp25/cp50 evals: one variable moved.
#
# GBS IS NOT TOUCHED, ON PURPOSE. A first reading of the harness suggested
# GBS=32 meant "32 turn-rows per optimizer step" (~2 trajectories) and should be
# raised. It does not: `latest_checkpointed_iteration` equals `--num-rollout` in
# every historical run (afw64b_grpo 50->49, math-qwen3-8b-grpo 100->99, the 3-step
# A/B 3->2), so ONE ROLLOUT = ONE OPTIMIZER STEP and the whole rollout (~544 rows
# ~= 32 trajectories) is already that step's batch. GBS is only the gradient-
# accumulation granularity, and under `--calculate-per-token-loss` the normaliser
# is N_global, so it is not semantically load-bearing. Raising it would shrink HBM
# headroom and change nothing else.
#
# IDEMPOTENT: a cell whose marker exists is skipped; a killed training run resumes
# from its own latest checkpoint with the same run name (allocated once, kept in
# $STATE, so a restart never forks a second run directory).
set -uo pipefail

ROLE="${1:?usage: queue_grpo_sizes.sh <0|1>}"
R=/workspace/cua/repos/opsd-slime
CKPT=/workspace/cua/checkpoints
cd "$R" || exit 1

# Fresh state dir for the fixed-harness campaign. The old one holds TRAIN_OK
# markers for the collapsed runs, and reusing it would make the queue skip
# training entirely. Nothing is deleted -- the previous markers stay on disk.
STATE="/tmp/afw_grpo_fixed_role$ROLE"
mkdir -p "$STATE"
QLOG="$STATE/queue.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

NUM_ROLLOUT=50
SAVE_INTERVAL=25
EVAL_ITERS=(24 49) # 0-indexed, the same two points the old campaign used

slug()  { case "$1" in 1p7b) echo qwen3-1p7b ;; 4b) echo qwen3-4b ;; 8b) echo qwen3-8b ;; esac; }
hfdir() { case "$1" in 1p7b) echo "$CKPT/Qwen3-1.7B" ;; 4b) echo "$CKPT/Qwen3-4B" ;; 8b) echo "$CKPT/Qwen3-8B" ;; esac; }
launcher() { case "$1" in 1p7b) echo run_alfworld_v2_grpo.sh ;; 4b) echo run_alfworld_v2_grpo_4b.sh ;; 8b) echo run_alfworld_v2_grpo_8b.sh ;; esac; }

# A name is allocated once and remembered. Allocating per attempt would create a
# fresh run dir on every retry, and the resume would silently start from base
# weights -- the exact failure the eval-side mirror trick exists to prevent.
name_for() { # $1=key $2..=allocate args
  local f="$STATE/$1.name"
  if [ ! -s "$f" ]; then
    local n
    n=$(python "$R/scripts/run_names.py" --allocate "${@:2}") || return 1
    echo "$n" > "$f"
  fi
  cat "$f"
}

train_cell() { # $1=size
  local size="$1" name dir rc
  name=$(name_for "train_$size" --task alfworld --model "$(slug "$size")" --method grpo --variant c4096) || return 1
  dir="$R/runs/$name"
  if [ -f "$STATE/TRAIN_OK_$size" ]; then say "train $size already done ($name)"; return 0; fi

  # Resume from our own checkpoint if one exists; otherwise a stale SAVE_DIR makes
  # _common_v2.sh FATAL by design (it refuses to overwrite a run it did not start).
  local resume_args=()
  if [ -s "$dir/latest_checkpointed_iteration.txt" ]; then
    resume_args=(RESUME_FROM="$dir")
    say "train $size resuming from $dir (iter $(cat "$dir/latest_checkpointed_iteration.txt"))"
  fi

  say "TRAIN start size=$size run=$name"
  env "${resume_args[@]+"${resume_args[@]}"}" \
      RUN_NAME="$name" NUM_ROLLOUT="$NUM_ROLLOUT" SAVE_INTERVAL="$SAVE_INTERVAL" \
      MAX_RESPONSE=4096 ROLLOUT_BS=4 GBS_OVERRIDE=32 \
      MAX_TURNS=50 HISTORY_LENGTH=8 SEQ_LENGTH=32768 OPSD_V2_LOSS_NORM=traj \
      bash "$R/tasks/alfworld/scripts/$(launcher "$size")" >> "$STATE/train_$size.out" 2>&1
  rc=$?
  say "TRAIN exit rc=$rc size=$size"

  # rc!=0 is not automatically a retry: a completed run also exits through here in
  # some paths. The tracker is the authority on whether the arm finished.
  local last=""
  [ -s "$dir/latest_checkpointed_iteration.txt" ] && last=$(cat "$dir/latest_checkpointed_iteration.txt")
  if [ "$last" = "$((NUM_ROLLOUT - 1))" ] || [ "$last" = "$NUM_ROLLOUT" ]; then
    touch "$STATE/TRAIN_OK_$size"
    say "TRAIN_OK size=$size at iter $last"
  else
    say "TRAIN INCOMPLETE size=$size at iter '${last:-none}' (rc=$rc) -- cell will retry"
    return 1
  fi
}

eval_cell() { # $1=size $2=src-run-name $3=iter
  # Same split as base_eval, for the same reason. Here it only "worked" because
  # every caller happened to have a `size` in scope, so the wrong-scope read was
  # invisible -- exactly the kind of latent bug that survives a happy path.
  local size="$1" src="$2" it="$3"
  local tag="${size}_it${it}"
  local name rc
  [ -f "$STATE/EVAL_OK_$tag" ] && return 0
  name=$(name_for "$tag" --task alfworld --model "$(slug "$size")" --method grpo --variant "cp$it") || return 1
  if [ -e "$R/runs/$name/latest_checkpointed_iteration.txt" ]; then
    # The eval launcher refuses to reuse a run dir, and a leftover tracker means an
    # earlier attempt got that far. Re-running is not safe to assume, so mark and
    # move on rather than fork a second name for the same checkpoint.
    touch "$STATE/EVAL_OK_$tag"; say "eval $tag already has a dir ($name)"; return 0
  fi
  say "EVAL start $tag src=$src"
  BACKBONE="$size" MODEL_SLUG="$(slug "$size")" METHOD_SLUG=grpo RUN_NAME="$name" \
    bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$R/runs/$src" "$it" \
    >> "$STATE/eval_$tag.out" 2>&1
  rc=$?
  # exit 4 == the launcher could not prove it loaded iteration N. That is a VOID
  # number, not a low one; only exit 0 counts.
  if [ "$rc" -eq 0 ]; then touch "$STATE/EVAL_OK_$tag"; say "EVAL_OK $tag"; else say "EVAL FAIL rc=$rc $tag"; fi
  return "$rc"
}

base_eval() { # $1=size
  # `local a="$1" b="x$a"` DOES NOT WORK under `set -u`: bash expands every word
  # of the command before `local` runs, so `$a` here still refers to the CALLER's
  # scope. Inside arm() that silently picked up the enclosing `size` and looked
  # fine; from the top level there is no `size` at all and the script aborted on
  # `unbound variable` -- after the whole 8B arm, right before its Base eval.
  local size="$1"
  local tag="base_$size"
  local name rc
  [ -f "$STATE/EVAL_OK_$tag" ] && return 0
  name=$(name_for "$tag" --task alfworld --model "$(slug "$size")" --method base) || return 1
  say "BASE EVAL start $size"
  BACKBONE="$size" MODEL_SLUG="$(slug "$size")" RUN_NAME="$name" \
    bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" --base "$(hfdir "$size")" \
    >> "$STATE/base_$size.out" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then touch "$STATE/EVAL_OK_$tag"; say "BASE EVAL OK $size"; else say "BASE EVAL FAIL rc=$rc $size"; fi
  return "$rc"
}

arm() { # $1=size : train then both checkpoint evals
  local size="$1"
  train_cell "$size" || { say "arm $size aborted at train"; return 1; }
  local name; name=$(cat "$STATE/train_$size.name")
  for it in "${EVAL_ITERS[@]}"; do
    [ -d "$R/runs/$name/iter_$(printf '%07d' "$it")" ] || { say "arm $size: no iter $it, skipping"; continue; }
    eval_cell "$size" "$name" "$it" || true
  done
  say "arm $size complete"
}

# 8B is the long pole (~2x a 4B rollout), so it gets a node to itself and the
# three shorter cells share the other one. Both nodes touch only their own run
# dirs, and the shared FS means either can be restarted independently.
if [ "$ROLE" = "0" ]; then
  arm 8b
  base_eval 8b
  # 1.7B is the fastest arm (~11 min/rollout); whoever finishes first takes it.
  # It is NOT a new ablation -- it is the third size of the established list.
  arm 1p7b
  base_eval 1p7b
else
  arm 4b
  base_eval 4b
fi
say "QUEUE role=$ROLE COMPLETE"
touch "$STATE/QUEUE_COMPLETE"
