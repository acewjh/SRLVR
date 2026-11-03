#!/usr/bin/env bash
# Evaluate ONE specific checkpoint iteration of an ALFWorld v2 arm.
#
# Why a mirror dir: `_common_v2.sh` resolves the checkpoint through
# `<RESUME_FROM>/latest_checkpointed_iteration.txt`, so pointing RESUME_FROM at a
# live run dir can only ever evaluate the LATEST iteration. To pin an arbitrary
# one, build a dir that symlinks `iter_00000NN` and writes N as the tracker --
# the same mechanism as eval_node/afw_eval_supervisor.sh::build_mirror.
#
# Why RESUME_FROM and NOT EVAL_LOAD_FROM for a LoRA checkpoint: only RESUME_FROM
# makes `_common_v2.sh` emit `--no-load-optim --no-load-rng`. Both arms save with
# `--no-save-optim`, so the file carries no optimizer state and loading it without
# those flags dies on `KeyError: 'optimizer'`. For a plain HF dir (Base) the
# reverse holds: RESUME_FROM would FATAL on the missing tracker, so Base must use
# EVAL_LOAD_FROM.
#
# Usage:
#   bash eval_alfworld_ckpt.sh <run-dir> <iter>          # a LoRA checkpoint
#   bash eval_alfworld_ckpt.sh --base <hf-dir>           # Base (plain HF dir)
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
cd "$PROJ" || exit 1

# A queued eval may be moved to another host while its queue is waiting.
# Consult the shared claim before touching local Ray or allocating accelerators.
if [ -n "${RUN_NAME:-}" ] && [ -f "$PROJ/runs/_eval_claims/$RUN_NAME.json" ]; then
  python3 - "$PROJ/runs/_eval_claims/$RUN_NAME.json" <<'CLAIM'
import json, socket, sys
claim = json.load(open(sys.argv[1]))
if claim['host'] != socket.gethostname():
    print('[eval] DELEGATED: this evaluation was moved to ' + claim['host'] +
          '; local result: ' + claim['report'], flush=True)
    sys.exit(75)
CLAIM
  claim_rc=$?
  [ "$claim_rc" = 0 ] || exit "$claim_rc"
fi

# Never fight the trainer for the cards: a training step owns all 8. Checked
# before anything is created so a refused eval leaves no half-built state.
if pgrep -f '[s]lime-ascend/train.py' > /dev/null; then
  echo "FATAL: a trainer is running; eval needs all 8 NPUs. Refusing to start." >&2
  exit 1
fi

if [ "${1:-}" = "--base" ]; then
  HF_DIR="${2:?usage: eval_alfworld_ckpt.sh --base <hf-dir>}"
  # Same rule as the checkpoint branch: the `base` method slug exists for exactly
  # this case, so a Base yardstick is named like everything else.
  RUN_NAME="${RUN_NAME:-$(python "$PROJ/scripts/run_names.py" --allocate \
               --task alfworld --model "${MODEL_SLUG:-qwen3-4b}" --method base)}"
  [ -d "$HF_DIR" ] || { echo "FATAL: no such HF dir: $HF_DIR" >&2; exit 1; }
  unset RESUME_FROM
  export EVAL_LOAD_FROM="$HF_DIR"
  EXPECT_ITER=""
else
  RUN_DIR="${1:?usage: eval_alfworld_ckpt.sh <run-dir> <iter>}"
  ITER="${2:?usage: eval_alfworld_ckpt.sh <run-dir> <iter>}"
  IR=$(printf 'iter_%07d' "$ITER")
  if [ ! -d "$RUN_DIR/$IR" ] && [ -d "$RUN_DIR/checkpoints/$IR" ]; then
    RUN_DIR="$RUN_DIR/checkpoints"
  fi
  RUN_DIR=$(realpath "$RUN_DIR") || exit 1
  if [ ! -d "$RUN_DIR/$IR" ]; then
    echo "FATAL: $RUN_DIR/$IR does not exist" >&2
    exit 1
  fi
  # Refuse a checkpoint that is still being written.
  s1=$(du -sb "$RUN_DIR/$IR" 2>/dev/null | cut -f1)
  sleep 20
  s2=$(du -sb "$RUN_DIR/$IR" 2>/dev/null | cut -f1)
  if [ -n "${s1:-}" ] && [ "$s1" != "$s2" ]; then
    echo "FATAL: $RUN_DIR/$IR is still growing -- refusing to eval a moving target" >&2
    exit 1
  fi
  # Run identity follows the repo's single naming rule (scripts/run_names.py), like
  # every other run: <task>-<model>-<method>-<variant>-<stamp>. An eval is still a
  # run, and `<srcdir>_eval_cpN` -- the previous default -- is the off-rule shape
  # the audit lists for the whole alfworld family. The checkpoint iteration goes in
  # the variant because it is what separates this eval from the other checkpoints.
  #
  # METHOD_SLUG is required rather than guessed: deriving it from the directory name
  # would silently mislabel an eval whenever the source dir is itself off-rule, and
  # a mislabelled eval is exactly the failure this repo keeps re-learning.
  if [ -z "${RUN_NAME:-}" ]; then
    if [ -z "${METHOD_SLUG:-}" ]; then
      echo "FATAL: set METHOD_SLUG (e.g. grpo|hsi|gspo|opsd) or pass RUN_NAME explicitly." >&2
      echo "       The rule is <task>-<model>-<method>-<variant>-<stamp>; refusing to guess." >&2
      exit 2
    fi
    RUN_NAME=$(python "$PROJ/scripts/run_names.py" --allocate \
                 --task alfworld --model "${MODEL_SLUG:-qwen3-4b}" \
                 --method "$METHOD_SLUG" --variant "cp${ITER}") || exit 1
  fi
  MIRROR="$PROJ/runs/_mirror_${RUN_NAME}_cp${ITER}"
  if [ -e "$MIRROR/$IR" ] && [ "$(realpath "$MIRROR/$IR")" != "$RUN_DIR/$IR" ]; then
    echo "FATAL: $MIRROR/$IR points to a different checkpoint" >&2
    exit 1
  fi
  if [ ! -f "$MIRROR/latest_checkpointed_iteration.txt" ] || [ ! -e "$MIRROR/$IR" ]; then
    TMP="$MIRROR.tmp.$$"
    mkdir -p "$TMP" || exit 1
    ln -s "$RUN_DIR/$IR" "$TMP/$IR" || exit 1
    printf '%s' "$ITER" > "$TMP/latest_checkpointed_iteration.txt" || exit 1
    mv "$TMP" "$MIRROR" || exit 1
  fi
  export RESUME_FROM="$MIRROR"
  unset EVAL_LOAD EVAL_LOAD_FROM
  EXPECT_ITER="$ITER"
fi

CHECKPOINT_GROUP="${CHECKPOINT_GROUP:-alfworld-${MODEL_SLUG:-qwen3-4b}-${METHOD_SLUG:-base}}"
CHECKPOINT_RUN_NAME="${RUN_NAME#${CHECKPOINT_GROUP}-}"
SAVE_DIR="${SAVE_DIR:-$PROJ/runs/$CHECKPOINT_GROUP/$CHECKPOINT_RUN_NAME}"
LOG="${LOG:-$PROJ/logs/$CHECKPOINT_GROUP/$RUN_NAME.log}"
mkdir -p "$(dirname "$SAVE_DIR")" "$(dirname "$LOG")"

if [ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "FATAL: $SAVE_DIR already holds a run; pick another RUN_NAME" >&2
  exit 1
fi

echo "[eval] run_name=$RUN_NAME resume_from=${RESUME_FROM:-<base>} load=${EVAL_LOAD_FROM:-<resume>} backbone=${BACKBONE}" | tee -a "$LOG"

# The backbone must be the one the checkpoint was TRAINED with. Getting it wrong
# does not error -- _common_v2.sh would build a different module tree, fail to
# match the checkpoint, and score something that is not the trained model. So the
# size is an explicit input, resolved through the same backbone cards the
# launchers use, and an unknown value is fatal rather than defaulted.
case "${BACKBONE:-4b}" in
  1p7b|1.7b) EVAL_SH=run_alfworld_v2_eval.sh ;;
  4b)        EVAL_SH=run_alfworld_v2_eval_4b.sh ;;
  8b)        EVAL_SH=run_alfworld_v2_eval_8b.sh ;;
  *) echo "[eval] FATAL: unknown BACKBONE='$BACKBONE' (want 1p7b|4b|8b)" >&2; exit 2 ;;
esac

RUN_NAME="$RUN_NAME" SAVE_DIR="$SAVE_DIR" LOG="$LOG" \
  bash "$PROJ/tasks/alfworld/scripts/$EVAL_SH" >> "$LOG" 2>&1
rc=$?
echo "[eval] launcher exited rc=$rc" | tee -a "$LOG"

# The load line is the only thing that distinguishes "evaluated the checkpoint"
# from "silently evaluated Base" -- a failure mode this repo has already hit.
if [ -n "$EXPECT_ITER" ]; then
  if grep -q "successfully loaded checkpoint.*iteration $EXPECT_ITER" "$LOG"; then
    echo "[eval] VERIFIED: loaded iteration $EXPECT_ITER from $RESUME_FROM"
  else
    echo "[eval] FATAL: no 'successfully loaded checkpoint ... iteration $EXPECT_ITER' in $LOG" >&2
    echo "[eval] the eval did NOT load the requested checkpoint -- treat its numbers as VOID" >&2
    exit 4
  fi
fi
exit "$rc"
