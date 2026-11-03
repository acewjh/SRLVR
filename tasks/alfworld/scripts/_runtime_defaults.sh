# Shared ALFWorld defaults for every method; explicit overrides are retained.
export OPSD_LORA="${OPSD_LORA:-0}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}"
export ALFWORLD_THINKING_STYLE="${ALFWORLD_THINKING_STYLE:-brief}"
# ALFWORLD_THINKING_TOKENS is deliberately unset by default: no thinking budget.
