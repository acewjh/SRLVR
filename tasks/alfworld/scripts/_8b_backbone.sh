#!/bin/bash
# Qwen3-8B backbone override for the v2 ALFWorld arms.
#
# Sourced (not exec'd) by run_alfworld_v2_*_8b.sh, so every 8B arm provably shares
# one geometry -- the same "matched by construction" argument _4b_backbone.sh
# provides for 4B.
#
# WHY THIS IS NOT THE MATH HARNESS's 8B CARD
# ------------------------------------------
# configs/models/qwen3-8b.sh pins MODEL_PP=4 and its comment explains the whole
# reason: the MATH harness holds TWO copies of the policy in actor memory at once
# -- the Megatron student and `math_opsd/opsd_loss.py::_get_teacher()`'s separate
# 16 GB HF teacher -- on 32 GB cards. 8B GRPO has no teacher (the privileged
# signal is the outcome-anchored advantage, computed in the rollout, not a second
# forward). On the 64 GB worker, PP1 retains four actor replicas. Set PP=2 on
# 32 GB cards to split the actor across two ranks while retaining two replicas.
#
# The v2 harness holds the full 16 GB bf16 base weights + LoRA adapters per
# actor rank at TP1 x PP1 x DP4; PP2 halves the layer weights and uses DP2.
# Both layouts keep the full head/FFN geometry intact for the LoRA provider.
# Adam moments are only for LoRA parameters (`--only-train-params-name-list` in
# _common_v2.sh; the 4B arms already run this way).
#
# The card is still worth watching: the 4B arm at cap=4096 reported a peak of
# 26.44/29.49 GB on 32 GB nodes (TRANSFER_4B_STATUS.md). Use PP2 there and verify
# the first 8B optimizer step; the PP1 default is for 64 GB cards.
#
# Values are the official Qwen3-8B config.json (num_hidden_layers=36,
# hidden_size=4096, intermediate_size=12288, num_attention_heads=32,
# num_key_value_heads=8, head_dim=128, vocab 151936, rope_theta 1e6,
# tie_word_embeddings=false).
BASE="${CUA_BASE:-/workspace/cua}"
export MODEL="${MODEL:-$BASE/checkpoints/Qwen3-8B}"
export NUM_LAYERS=36
export HIDDEN_SIZE=4096
export FFN_HIDDEN_SIZE=12288
export NUM_ATTENTION_HEADS=32
export NUM_QUERY_GROUPS=8
# Untied embeddings, unlike 1.7B and 4B. _common_v2.sh's MODEL_ARGS have no
# --untie flag, so the last pipeline/rank would otherwise build a tied lm_head and
# silently mismatch the checkpoint.
export UNTIE_EMBEDDINGS=1
# Match the ALFWorld 4B default: TP2 training and TP2 inference. On the default
# eight-card rollout allocation this creates four inference engines.
export TP="${TP:-2}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
