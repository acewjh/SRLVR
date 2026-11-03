"""LoRA provider with gradient-safe full activation recomputation on 32 GB NPUs."""

import logging

import torch

logger = logging.getLogger(__name__)


def require_embedding_output_grad(module, inputs, output):
    # Reentrant checkpoints only propagate parameter gradients when an input
    # requires grad. Frozen token embeddings otherwise disconnect every fully
    # checkpointed layer, even though its LoRA parameters are trainable.
    if module.training and torch.is_grad_enabled():
        output.requires_grad_(True)
    return output


def custom_model_provider(pre_process=True, post_process=True, vp_stage=None):
    from methods.lora_megatron import custom_model_provider as lora_provider, lora_enabled

    model = lora_provider(pre_process, post_process, vp_stage)
    if pre_process and lora_enabled():
        model.embedding.register_forward_hook(require_embedding_output_grad)
        logger.info("[alfworld-memory] embedding output gradients enabled for LoRA recomputation")
    return model
