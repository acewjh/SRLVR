"""Model provider for the HSI arm: the stock LoRA provider plus an actor handle.

The HSI scorer has to forward theta_old from inside ``--rollout-data-postprocess-path``
(``methods.hsi2_jsd.postprocess`` on every task), and slime calls that hook with
``(args, rollout_id, rollout_data)`` -- there is no model argument, and the hook has no
supported way to ask the trainer for one. Rather than edit shared source, this provider
wraps the project's existing one and records the object it just built.

Why that is the *same* theta_old and not a stale copy: the registered object is the
Megatron GPT model itself, and the optimizer updates its ``Parameter`` objects in
place. The postprocess hook runs after ``compute_advantages_and_returns`` and
before ``train()``, so at the moment it forwards, the parameters still hold the
weights the rollout was generated with and ``optimizer.step()`` has not run.

This is a pure addition -- ``methods.lora_megatron.custom_model_provider`` is
called unchanged, so the model is byte-identical to every other arm's.
"""

import logging

from methods.lora_megatron import custom_model_provider as _stock_provider

from . import hsi_shared

logger = logging.getLogger(__name__)

__all__ = ["custom_model_provider"]


def custom_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    """Build the actor exactly as the stock provider does, then publish it."""
    import inspect

    has_vp = "vp_stage" in inspect.signature(_stock_provider).parameters
    model = (
        _stock_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if has_vp
        else _stock_provider(pre_process=pre_process, post_process=post_process)
    )
    if post_process:
        hsi_shared.register_actor_model(model)
        logger.info("[hsi] actor handle registered for same-policy hindsight scoring")
    return model
