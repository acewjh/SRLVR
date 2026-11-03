# Qwen3-1.7B model card for the opsd-slime math harness (Server A main table).
#
# This file, plus its qwen3-4b / qwen3-8b siblings, is the ONLY size-specific
# input to the math launchers (train_scripts/run_math_opsd_slime_npu.sh,
# train_scripts/run_dev_eval.sh, scripts/run_grpo.sh). Everything else is shared
# code. Declarative assignments only -- no side effects, no commands.
#
# 1.7B is the frozen paper baseline: do not retune these values, they are what
# the main table in reports/STATUS.md was produced with.

MODEL_NAME="Qwen3-1.7B"
MODEL_DIR="$CUA_CHECKPOINTS/Qwen3-1.7B"

# Run identity is NOT here. A model card describes the model; the run directory
# is named by the rule in scripts/run_names.py
# (<task>-<model>-<method>[-<variant>]-<stamp>), allocated by the launcher. Keeping
# `math_opsd_slime` / `grpo_standard_bshd` in this file is exactly what let four
# different naming schemes coexist.
MODEL_TASK="math"

# Parallelism. 1.7B fits whole on one card, so the main table runs TP1 x PP1 x DP4
# (4 actor NPUs + 4 rollout NPUs).
MODEL_TP=1
MODEL_PP=1

# Frozen 1.7B value. The 4B and 8B cards had to drop to 1 (memory); the objective
# is unaffected either way -- GBS stays 32 and --calculate-per-token-loss divides
# by N_global, not by the microbatch.
MODEL_MBS_DEFAULT=2

# Allocator: stock torch-npu behaviour. The anti-fragmentation expandable
# allocator (expandable_segments) is a 4B/8B card setting -- those sit close
# enough to the HBM ceiling that fragmentation alone can tip them over.
MODEL_ALLOC_CONF=""

# Rollout engine sizing. The [chunk, V] logits spike scales with
# chunked_prefill_size x vocab; 151936-wide vocab is shared by all three sizes,
# so the chunk budget is only squeezed on the larger cards.
MODEL_SGLANG_MEM_FRACTION="0.82"
MODEL_SGLANG_CHUNKED_PREFILL="8192"

# Architecture, read off cua/checkpoints/Qwen3-1.7B/config.json (not guessed):
# 28 layers, hidden 2048, ffn 6144, 16 heads / 8 kv groups, head_dim 128,
# rope theta 1e6, tied embeddings (so no --untie-embeddings-and-output-weights).
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --padded-vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
