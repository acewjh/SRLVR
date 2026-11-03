#!/usr/bin/env bash
# Two-node 64G campaign: one algorithm per node, 8 cards each.
#
#   bash launch_c64.sh grpo_v2     # matched GRPO (G=8)
#   bash launch_c64.sh hsi_v3      # our method (canonical `hsi`, exact JSD)
#
# The other historic arms (hsi → hpi_future, hsi_v2, hsi_candidate) were retired
# 2026-10-15 and their cases removed; the registry refuses them with rc=2.
#
# Config rationale (see reports/grpo_degradation_analysis.md + ALFWORLD_FIX_HANDOFF.md):
#
#   MAX_RESPONSE=4096  The 32G runs used 1024 and that is the whole story of why they
#                      degraded: training drove resp_mean (458 -> ~1010) into the cap,
#                      so `</action>` was never emitted, the policy stopped producing
#                      admissible actions, reward went to 0 and so did the gradient.
#                      The base model already truncated 92.8% of episodes at 1024.
#                      4096 is 4x the previous room and was chosen as the smallest
#                      power of two with real headroom; 64G makes it affordable.
#                      ACCEPTANCE GATE before any algorithm comparison: truncated_ratio
#                      stably <0.3 AND action_valid >80%. Check it in the first ~3 steps.
#
#   ROLLOUT_BS=4       UNCHANGED from the previous campaign, deliberately. Raising it
#                      to 8 was tried and reverted: with G=8 the GRPO arm's GBS becomes
#                      ROLLOUT_BS*G=64 while _common_v2.sh hard-pins the HSI arm's GBS
#                      to 32, so the two arms would no longer take equally sized
#                      optimizer steps -- and train_data.py trims trailing turns to a
#                      multiple of GBS, so the mismatch also costs data. Matching GBS
#                      across arms is a load-bearing invariant of this harness.
#                      Changing the cap ALONE also keeps the diagnosis clean: the cap
#                      is the one thing the analysis indicts, so it is the one thing
#                      that changes.
#
#   everything else is held identical to the previous campaign so the two arms stay
#   matched and the comparison is against the same harness.
set -uo pipefail

ARM_NAME="${1:?usage: launch_c64.sh <grpo|hsi>}"
R=/workspace/cua/repos/opsd-slime
cd "$R" || exit 1
# Never inherit a supervisor's flock fd into the long-lived trainer.
exec 9>&- 2>/dev/null || true

# Every one of these is overridable from the environment: extending an arm from 50
# to 100 steps is a resume with a larger NUM_ROLLOUT (slime's --num-rollout is the
# run's TOTAL, so it must exceed the checkpoint's iteration), and that must not
# require editing this file.
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
# 5, not 25: the 2026-10-03 campaign reports cp10/cp25/cp50 (plus cp0 = the base
# model the run loads), so the save grid must land on 10 as well as 25 and 50.
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# 50, not 20. This file's own default silently OVERRODE _common_v2.sh's (it is
# exported after, and `_common_v2.sh` only supplies a default), so a launch that
# looked correct in the banner of the inner script ran with the old turn budget.
# At 20 the truncated_ratio was 0.87-0.99 for every model size: episodes ended by
# exhausting turns rather than by failing the task.
export MAX_TURNS="${MAX_TURNS:-50}"

case "$ARM_NAME" in
  grpo)
    export ARM=grpo
    METHOD_SLUG=grpo
    LAUNCHER=tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh
    ;;
  hsi_v3)
    # 2026-10-04 V3 protocol, Node0. HSI redefined by BG_TASK_PROMPT_V3:
    #   SIGN  = lagged historical reward baseline, A_seq = R_i - b_k  (b_k from
    #           strictly earlier rollouts; committed only after this batch's A exist)
    #   WHERE = exact normalized JSD between the normal-prefix view p and the
    #           hindsight view q, both from the SAME theta_old; A_i,t = (R_i-b_k)*w
    # The estimator is the shared math_opsd/hsi2_jsd.py, aliased at
    # tasks/alfworld/hsi2_jsd.py; the arm branch in _common_v2.sh is what selects
    # it over the retired V2/candidate-centered ones.
    export ARM=hsi_v3
    export ALFWORLD_V2_PROTOCOL=1
    METHOD_SLUG=hsi
    RUN_VARIANT=s42
    LAUNCHER=tasks/alfworld/scripts/run_alfworld_v2_hsi_4b.sh
    export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
    ;;
  grpo_v2)
    # 2026-10-04 protocol switch, Node1. Matched GRPO: identical model, seed,
    # frozen schedule, verifier, sampler (T=1/top_p=1/top_k=-1), LoRA, lr,
    # clip-grad, batch protocol, trajectory reducer and eval protocol. No
    # future/outcome context is added on this arm.
    export ARM=grpo
    export ALFWORLD_V2_PROTOCOL=1
    METHOD_SLUG=grpo
    RUN_VARIANT=s42
    LAUNCHER=tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh
    export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
    export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
    ;;
  grpo_matched)
    # 2026-10-03 campaign, Node1: the matched GRPO baseline (G=8).
    export ARM=grpo
    METHOD_SLUG=grpo_matched
    LAUNCHER=tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh
    ;;
  *) echo "FATAL: unknown arm '$ARM_NAME' (want grpo|grpo_v2|grpo_matched|hsi_v3); hsi/hsi_v2/hsi_candidate were retired 2026-10-15" >&2; exit 2 ;;
esac

# Run identity comes from the repo's single naming rule (CLAUDE.md / scripts/run_names.py),
# never from a hand-written string -- the earlier `afw64b_*` names in this file are why
# the audit lists the alfworld family as off-rule. The cap deviates from the model card's
# recipe, so it is carried as the `c4096` variant; the stamp is the launch time.
# RUN_NAME stays overridable so an operator can pin a name deliberately.
if [ -z "${RUN_NAME:-}" ]; then
  RUN_NAME=$(python "$R/scripts/run_names.py" --allocate \
               --task alfworld --model "${MODEL_SLUG:-qwen3-4b}" --method "$METHOD_SLUG" \
               --variant "${RUN_VARIANT:-c${MAX_RESPONSE}}") || exit 1
fi
export RUN_NAME

echo "[c64] arm=$ARM_NAME run=$RUN_NAME cap=$MAX_RESPONSE bs=$ROLLOUT_BS steps=$NUM_ROLLOUT"
exec bash "$R/$LAUNCHER"
