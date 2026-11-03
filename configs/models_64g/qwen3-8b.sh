# Qwen3-8B on a 64 GB node -- the geometry override the HSI arm needs.
#
# WHY THIS FILE EXISTS
# --------------------
# configs/models/qwen3-8b.sh pins MODEL_PP=4 and its own header explains both
# reasons: that node's cards are 32 GB, and the MATH harness holds TWO copies of
# the policy in actor memory at once -- the Megatron student and
# `methods/opsd_loss.py::_get_teacher()`'s separate 16 GB HF teacher.
#
# Neither condition holds for the runs this card is for:
#   * the node is a 64 GB node (npu-smi: 65536 MB/card -- pod 8ab2ddd1,
#     ma-job-ca427658-...-worker-0), and
#   * the HSI arm has NO teacher. Its privileged signal is built from the actor's
#     own theta_old (methods/hsi2_jsd.py), so `train/v2/teacher_forward_calls`
#     is 0.0 for the whole run -- measured over 100 rollouts on the 4B math HSI
#     run, and over 100 more on the 4B ALFWorld HSI run.
#
# tasks.alfworld/scripts/_8b_backbone.sh makes exactly this argument for the
# ALFWorld arms and lands on TP1 x PP1 x DP4; BG_TASK_PROMPT_V4.md section 3 pins
# the same geometry for the 8B cells and says in as many words not to carry the
# math card's PP4 over ("那是 32G 的约束"). There is a correctness reason on top
# of the memory one:
#   * the HSI scorer runs inside the actor on EVERY rank and forwards the actor
#     module directly (methods/hsi2_jsd.py::_hidden_states -> _find_lm_head),
#     which needs the output layer resident locally. A PP>1 rank owns a slice of
#     the layers and no lm_head, so the tap would capture nothing.
#   * the only validated HSI geometry is PP1: every 4B math/ALFWorld HSI run used
#     the 4B card, which is TP1 x PP1 x DP4.
#
# What the actor rank holds at TP1 x PP1 x DP4 on 64 GB, per the same accounting
# _8b_backbone.sh does: ~16 GB bf16 base weights + LoRA adapters, Adam moments for
# the LoRA parameters only (`--only-train-params-name-list`), and the transient bf16
# base-weight clone `methods/lora_megatron.py::merged_lora_sync` takes for the
# duration of each `update_weights()`. That last one is a full copy at PP1, so the
# first steps of the first 8B cell are the place to watch `npu-smi`; the expected
# failure mode is a loud OOM, not a silent wrong number.
#
# Everything else comes from the real card -- this file must never grow a second
# copy of the model. Select it with CUA_MODELS_DIR pointing at this directory;
# MODEL_SLUG stays `qwen3-8b`, so the run name and the model vocabulary are
# unchanged and the geometry deviation is carried by the run's `variant` (pp1).
source "$CUA_PROJ/configs/models/qwen3-8b.sh"

MODEL_PP=1
