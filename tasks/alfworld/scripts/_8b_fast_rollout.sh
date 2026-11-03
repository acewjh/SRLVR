# Sourced by the 8B baseline launchers. No change to rewards or PPO estimators.
# Action-only output uses the shared 128-token cap.
# 32 GB-card deviation from the shared 8+8 colocated default: an 8B actor plus
# colocated engines does not fit. Keep the measured 2 actor + 6 rollout split.
export COLOCATE="${COLOCATE:-0}"
export ACTOR_GPUS="${ACTOR_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
export SGLANG_PREFIX_CACHE="${SGLANG_PREFIX_CACHE:-1}"
export ALFWORLD_V2_HSI_CANDIDATE_K="${ALFWORLD_V2_HSI_CANDIDATE_K:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_defaults.sh"
export KL_COEF="${KL_COEF:-0.001}"
