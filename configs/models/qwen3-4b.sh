# Qwen3-4B model card for the opsd-slime math harness (Server B 4B campaign).
#
# See qwen3-1p7b.sh for the contract: this file is the only size-specific input
# to the shared math launchers. Declarative assignments only.

MODEL_NAME="Qwen3-4B"
MODEL_DIR="$CUA_CHECKPOINTS/Qwen3-4B"

# Run identity comes from scripts/run_names.py, not from here (see qwen3-1p7b.sh).
MODEL_TASK="math"

# Parallelism: 4B fits whole on one card -> TP1 x PP1 x DP4, the same geometry as
# the 1.7B arm (keeps the two campaigns comparable).
MODEL_TP=1
MODEL_PP=1

# Deviation D1 of reports/QWEN3_4B_MANIFEST.md: MBS=2 OOMed at step 0 on 4B
# ("Tried to allocate 48.00 MiB; 7.54 MiB free"). MBS=1 is a pure memory knob --
# GBS stays 32, and under --calculate-per-token-loss the normaliser is N_global,
# so micro-batch size is not semantically load-bearing. Still env-overridable.
MODEL_MBS_DEFAULT=1

# Deviation D5: allocator anti-fragmentation at the 4B memory ceiling.
MODEL_ALLOC_CONF="expandable_segments:True"

MODEL_SGLANG_MEM_FRACTION="0.82"
MODEL_SGLANG_CHUNKED_PREFILL="8192"

# Architecture off cua/checkpoints/Qwen3-4B/config.json: 36 layers, hidden 2560,
# ffn 9728, 32 heads / 8 kv groups, head_dim 128, rope 1e6, tied embeddings
# (same as 1.7B -> no --untie flag).
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
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
