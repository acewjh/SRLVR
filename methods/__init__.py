"""L1 method layer: one module per training objective (or its plumbing).

    opsd_loss.py        official OPSD objective (frozen teacher, beta=0 KL)
    hsi2_jsd.py         HSI -- exact normalized JSD hindsight credit
    hsi_rollout.py      the math/code HSI rollout (chunked hindsight view)
    hsi2_train_data.py  HSI sample -> slime train-row conversion
    grpo_rollout.py     the GRPO control rollout for math
    lora_megatron.py    LoRA model provider (also imported by the vendored
                        framework via the ``math_opsd`` shim)
    optim_sidecar.py    optimizer-state sidecar (same)

Adding a method = add one file here plus a ``MethodSpec`` in
``opsd_core/registry.py``. ``_retired/`` holds the objectives the paper no
longer reports.

Dependency direction: a method may import ``opsd_core``.
Three HSI modules also import ``tasks.alfworld`` for shared constants/state
(``TURN_ID_STRIDE``, ``hsi_shared``); that inversion is documented and
pre-existing -- see §9.4.
"""
