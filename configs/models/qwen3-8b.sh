# Qwen3-8B model card for the opsd-slime math harness.
#
# See qwen3-1p7b.sh for the contract: this file is the only size-specific input
# to the shared math launchers. Declarative assignments only.
#
# WHY THIS CARD IS NOT "4B WITH BIGGER NUMBERS"
# ---------------------------------------------
# The 1.7B and 4B cards both run TP1 x PP1 x DP4, i.e. every actor NPU holds a
# whole copy of the model. 8B cannot: this node's cards are 32 GB
# (`npu-smi` HBM 32768 MB/card), and this harness needs TWO copies of the policy
# in actor memory at once --
#     * the Megatron student (base weights + LoRA adapters), and
#     * the frozen initial-policy teacher, which methods/opsd_loss.py
#       `_get_teacher()` loads as a separate HF model on the same NPU.
# 2 x 16 GB of weights is already 32 GB before a single activation, so TP1/PP1
# is arithmetically impossible here, not merely tight.
#
# PP (not TP) is the lever that shrinks the student:
#   * PP splits *layers* between ranks, so no weight matrix is ever sharded
#     along a tensor axis. That matters because methods/lora_megatron.py
#     derives its LoRA section offsets from the full head/ffn geometry
#     (`_sections_for`); under TP those offsets would have to be divided by the
#     TP size, and the adapter checkpoint sharding map is registered as
#     replicated. PP leaves all of that untouched.
#   * The custom loss (and therefore the teacher forward) runs on the LAST
#     pipeline stage only, so the 16 GB HF teacher is resident on the last-stage
#     card(s) instead of on all four.
#
# With PP2 x DP2 on 4 actor NPUs the worst card is a last stage holding
# 18 layers + lm_head + teacher ~= 22 GB of weights, which leaves room for the
# teacher's transient full-vocab logits. MODEL_PP is the single knob to raise if
# that still does not fit (PP4 x DP1 puts <=10 layers per card).

MODEL_NAME="Qwen3-8B"
MODEL_DIR="$CUA_CHECKPOINTS/Qwen3-8B"

# Run identity comes from scripts/run_names.py, not from here (see qwen3-1p7b.sh).
MODEL_TASK="math"

# Pipeline 4, tensor 1, data-parallel 1 over --actor-num-gpus-per-node 4
# (4 actor NPUs + 4 rollout NPUs = the 8 cards). GBS 32 must be a multiple of
# MBS x DP = 1 x 1 = 1 -> holds (32 microbatches, 8 per pipeline stage).
#
# PP2 x DP2 was tried first and OOMed at the end of step 0 with 26.73 of
# 29.49 GiB live on the last stage. The last stage carries, in addition to its
# share of the student, two things PP2 does not shrink:
#   * the 15.26 GiB HF teacher (resident on every last-stage rank), and
#   * a full copy of that stage's base linear weights, which
#     methods/lora_megatron.py::merged_lora_sync clones for the duration of
#     every update_weights() so the bf16 merge can be undone without drift.
# At PP2 those two alone are 7.63 + 15.26 GiB, before any activation. At PP4 the
# student's share drops to 9 layers (3.8 GiB), which is what buys the headroom.
#
# Measured on the 2026-09-30 real-length gate (runs/math-qwen3-8b-rewardcv-vnext-gate1024-20260930_134249:
# 1 rollout, 32 prompts, max_response 1024, avg completion 995.5 tokens, 0 OOM,
# grad_norm 0.00631), as reported by the harness's own allocator telemetry:
#     [Rank 3] total 29.49 GB, used 26.21 GB, free 3.28 GB   <- last stage, binding
#     [Rank 0] total 29.49 GB, used 14.71 GB, free 14.78 GB
# So the working geometry is PP4 and the last stage is the card to watch. It is
# also what pins the two constraints below: a 1.7B/4B-style TP1 x PP1 card has no
# room for the teacher at all, and longer sequences would need the response cap
# raised rather than more parallelism.
MODEL_TP=1
MODEL_PP=4

# Same reasoning as 4B's D1: at 8B the per-card budget is the binding constraint,
# so the default is the smallest microbatch. Env-overridable (MBS=...).
MODEL_MBS_DEFAULT=1

MODEL_ALLOC_CONF="expandable_segments:True"

# Rollout engines are TP1 (one per NPU) and therefore hold the full 16 GB of
# weights, unlike the vendored slime-ascend 8B example whose engines are TP2 with
# mem-fraction 0.4. 0.82 x 32 GB = 27 GB, i.e. ~11 GB of KV on top of the weights.
MODEL_SGLANG_MEM_FRACTION="0.82"
# The per-chunk logits spike is chunked_prefill_size x 151936 (vocab) x 2 B on top
# of the weights; halved versus the 1.7B/4B cards because those cards carry 8 GB
# of weights where this one carries 16 GB.
MODEL_SGLANG_CHUNKED_PREFILL="4096"

# Architecture off cua/checkpoints/Qwen3-8B/config.json: 36 layers, hidden 4096,
# ffn 12288, 32 heads / 8 kv groups, head_dim 128, rope 1e6 -- and, unlike 1.7B
# and 4B, **untied** embeddings (config.json tie_word_embeddings=false), hence
# --untie-embeddings-and-output-weights. That also means the last pipeline stage
# carries a real 151936 x 4096 lm_head on top of the decoder layers.
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
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
   --untie-embeddings-and-output-weights
)
