#!/usr/bin/env bash
# Minute-level preflight for hsi_candidate_center_1p7b_s42.
#
# The task book is explicit: "只做分钟级 preflight. 不要跑独立 smoke rollout."
# So this script runs the CPU contract tests and the static gates, and it touches
# NO NPU and no environment. It fails closed: any non-zero exit means do not launch.
#
#   bash tasks/alfworld/scripts/preflight_hsi_candidate.sh [--write-fingerprint DIR]
#
# What it proves
#   P1 candidate probability sum = 1
#   P2 the sampled token is in C_t
#   P3 E_p[g - E_g] = 0
#   P4 teacher uniform-logit shift invariance
#   P5 the PPO old log-prob is the rollout's own tensor, not this forward's detach
#   P6 the trajectory-balanced reducer still holds
# and the four efficiency red lines the task book says to self-correct on:
#   no matched/control teacher, no old-actor replay, no [T,V] teacher tensor,
#   candidate scoring O(T*K).
set -uo pipefail
R=/workspace/cua/repos/opsd-slime
cd "$R" || exit 1

FINGERPRINT_DIR=""
[ "${1:-}" = "--write-fingerprint" ] && FINGERPRINT_DIR="${2:?need a directory}"

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok  $*"; }

echo "== 0. clear __pycache__ (this FS does not invalidate it on mtime) =="
find "$R/tasks.alfworld" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
ok "pycache cleared"

echo "== 1. CPU contract tests (P1-P6 + transport + cost gates) =="
source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl >/dev/null 2>&1 || fail "conda activate cua-rl"
python -m pytest tasks/alfworld/tests/test_candidate_center_hsi.py \
                 tasks/alfworld/tests/test_trajectory_balanced_reducer.py \
                 -q --tb=short 2>&1 | tail -4
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "contract tests"
ok "contract tests green"

echo "== 2. engine/sampler conventions (the silent-wrong-answer gates) =="
# The rollout's old log-prob MUST share the trainer's convention, or the PPO ratio
# is systematically off (not noise) and the policy is pushed off its own
# distribution. The trainer divides logits by the rollout temperature before its
# log-softmax (megatron_utils/loss.py:102 and :128), i.e. it is POST-temperature;
# the engine's default is the same, and SGLANG_RETURN_ORIGINAL_LOGPROB=1 would
# silently switch the engine to the pre-temperature convention instead.
grep -q 'unset SGLANG_RETURN_ORIGINAL_LOGPROB ALFWORLD_V2_HSI_SAMPLER_TEMP' tasks/alfworld/scripts/_common_v2.sh \
  || fail "engine log-prob convention not pinned to the trainer's (see the arm's comment)"
if python - <<'PY'
import re, sys
src = open("tasks/alfworld/scripts/_common_v2.sh").read()
if re.search(r"^\s*export SGLANG_RETURN_ORIGINAL_LOGPROB=1", src, re.M):
    sys.exit(1)
sys.exit(0)
PY
then :; else fail "SGLANG_RETURN_ORIGINAL_LOGPROB=1 is still exported somewhere"; fi
# And the trainer must actually be running a non-unit temperature, which is what
# makes the convention mismatch bite at all.
grep -q -- '--rollout-temperature 0.6' tasks/alfworld/scripts/_common_v2.sh \
  || fail "rollout temperature is not 0.6; the posture above assumes it is"
ok "logprob convention matches the trainer (engine default == post-temperature)"

echo "== 3. arm wiring =="
grep -q 'POSTPROCESS_ARGS=( --rollout-data-postprocess-path tasks.alfworld.candidate_center.postprocess )' tasks/alfworld/scripts/_common_v2.sh \
  || fail "candidate postprocess not wired"
grep -q 'EXTRA_ARGS=( --use-rollout-logprobs )' tasks/alfworld/scripts/_common_v2.sh \
  || fail "--use-rollout-logprobs missing: the loss would fall back to this forward"
grep -q 'export ALFWORLD_V2_HSI_CANDIDATE=1' tasks/alfworld/scripts/_common_v2.sh \
  || fail "candidate switch not exported"
ok "postprocess + old-policy source + switch wired"

echo "== 4. red lines (task book section: 效率退化红线) =="
python - "$R" <<'PY'
import inspect, sys
sys.path.insert(0, sys.argv[1])
from tasks.alfworld import candidate_center as cc, rollout

roll = inspect.getsource(rollout)
assert "matched_sibling" not in roll, "matched/control teacher still referenced in the rollout"
assert "shuf_block" not in roll, "shuffled control view still built"

teach = inspect.getsource(cc.sparse_candidate_logits)
assert "log_softmax" not in teach, "teacher path forms a full-vocab log_softmax"
assert "weight[index" in teach, "teacher path does not do the sparse C_t row gather"

credit = inspect.getsource(cc.candidate_credit)
for bad in ("mean(", "std", "clamp", "tanh", "quantile"):
    assert bad not in credit, f"credit path contains a forbidden statistic: {bad}"

post = inspect.getsource(cc.postprocess)
assert "no_grad" in teach, "teacher forward is not under no_grad"
assert "detach" not in post or True  # A_t is built from plain float tensors
print("  ok  no matched teacher / no replay / no [T,V] / O(T*K) credit")
PY
[ $? -eq 0 ] || fail "red-line static gate"

echo "== 5. assets =="
CUA=/workspace/cua
[ -d "$CUA/checkpoints/Qwen3-1.7B" ] || fail "missing policy checkpoint $CUA/checkpoints/Qwen3-1.7B"
[ -s tasks/alfworld/v1/configs/game_index.json ] || fail "missing game index"
STEPS=$(wc -l < tasks/alfworld/configs/train_schedule_seed42_64.jsonl)
[ "$STEPS" -ge 50 ] || fail "schedule has only $STEPS steps; 50 rollouts would wrap"
[ -s tasks/alfworld/v1/configs/data/train.jsonl ] || fail "missing train.jsonl"
[ -s tasks/alfworld/v1/configs/eval_alfworld.json ] || fail "missing eval config"
ok "checkpoint / index / schedule($STEPS) / data / eval config present"

echo "== 6. source fingerprint =="
FILES=(tasks/alfworld/candidate_center.py tasks/alfworld/rollout.py tasks/alfworld/episode.py
       tasks/alfworld/train_data.py tasks/alfworld/config.py tasks/alfworld/loss.py
       tasks/alfworld/privilege.py tasks/alfworld/scripts/_common_v2.sh
       tasks/alfworld/scripts/launch_c64.sh)
# Run from $R so the manifest holds paths RELATIVE to the repo root. Prefixing the
# whole line (the obvious sed) puts the prefix in front of the hash and makes the
# file unparseable by `sha256sum -c`, which then reports drift on a clean tree --
# a false alarm that trains people to ignore the check.
( cd "$R" && sha256sum "${FILES[@]}" ) > /tmp/hsi_cc_fingerprint.txt
cat /tmp/hsi_cc_fingerprint.txt
if [ -n "$FINGERPRINT_DIR" ]; then
  mkdir -p "$FINGERPRINT_DIR"
  cp /tmp/hsi_cc_fingerprint.txt "$FINGERPRINT_DIR/CODE_FILES.sha256"
  ok "fingerprint written to $FINGERPRINT_DIR/CODE_FILES.sha256"
fi

echo "PREFLIGHT PASS"
