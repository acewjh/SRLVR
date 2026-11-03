#!/bin/bash
# Qwen3-4B backbone override for the v2 ALFWorld arms.
#
# Sourced (not exec'd) by run_alfworld_v2_*_4b.sh, so BOTH 4B arms provably share
# one geometry -- the same "matched by construction" argument that _common_v2.sh
# provides for the 1.7B arms. Only the backbone changes; the method is frozen.
#
# Values are the official Qwen3-4B config.json (num_hidden_layers=36,
# hidden_size=2560, intermediate_size=9728, num_attention_heads=32,
# num_key_value_heads=8, head_dim=128, vocab 151936, rope_theta 1e6) and agree
# with frameworks/slime-ascend/scripts/models/qwen3-4B.sh.
BASE=/workspace/cua
export MODEL="${MODEL:-$BASE/checkpoints/Qwen3-4B}"
export NUM_LAYERS=36
export HIDDEN_SIZE=2560
export FFN_HIDDEN_SIZE=9728
export NUM_ATTENTION_HEADS=32
export NUM_QUERY_GROUPS=8

# Tensor parallelism on the 32 GB cards this node runs (910B4, 29.49 GiB usable).
#
# TP=1 full RL does NOT fit: measured 2026-10-29, the actor init died before the
# first rollout with
#   NPU out of memory. Tried to allocate 14.99 GiB
#   (29.49 GiB capacity; 14.98 GiB already allocated; 13.42 GiB free).
# TP=2 shards the 4B model across the two actor cards and completed a 2-step
# GRPO smoke end to end (rc=0, two optimizer steps, iter_0000001): actor
# ~18.7/29.49 GB with ~10.8 GB free, rollouts on the other six cards.
# See reports/grpo4b_smoke_20261029_215819/SMOKE_REPORT.md.
#
# An arm that has its own validated reason to differ (e.g. the frozen
# repeat-stop arm needs TP=1 for its teacher/chunking layout) must set TP
# explicitly AFTER sourcing this file; `${TP:-2}` keeps that override intact.
export TP="${TP:-2}"

# 8+8 colocate is the target 4B layout. The earlier graph-enabled smoke failed
# during initial weight sync with torch_npu's captures_underway assertion; the
# 2026-10-30 on-card trial passed that sync, the 32-trajectory rollout and the
# optimizer step. NPU graph is therefore NOT overridden here: it is the shared
# default in _v2_defaults.sh (on for every size, explicit =0 still wins). Only
# the bounded memory settings below remain 4B-specific.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.30}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-32768}"
