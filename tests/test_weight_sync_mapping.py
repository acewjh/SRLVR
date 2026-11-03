"""Verify slime's megatron->HF weight-sync mapping (update_weights path) against
the HF checkpoint, tensor by tensor.

Builds the megatron model exactly like the training actor does (TP1), loads HF
weights via the same code path as slime's _load_hf_weights (mbridge), then runs
every parameter through slime's convert_to_hf (what update_weights sends to
sglang) and compares each converted tensor with the tensor of the same name in
the HF checkpoint. Any mismatch is a real weight-sync corruption.

Run:  torchrun --nproc_per_node=1 tests/test_weight_sync_mapping.py
"""

import json
import os
import sys

import torch
import torch_npu  # noqa: F401

PROJ = "/workspace/cua/repos/opsd-slime"
MODEL = "/workspace/cua/checkpoints/Qwen3-1.7B"
sys.path.insert(0, f"{PROJ}/frameworks/slime-ascend")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-LM")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-Bridge/src")
sys.path.insert(0, PROJ)


def main() -> None:
    import mindspeed.megatron_adaptor  # noqa: F401

    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.training import get_model
    from slime.backends.megatron_utils.arguments import set_default_megatron_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf

    model_args = [
        "--swiglu", "--num-layers", "28", "--hidden-size", "2048", "--ffn-hidden-size", "6144",
        "--num-attention-heads", "16", "--group-query-attention", "--num-query-groups", "8",
        "--use-rotary-position-embeddings", "--disable-bias-linear", "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6", "--rotary-base", "1000000", "--vocab-size", "151936",
        "--padded-vocab-size", "151936", "--kv-channels", "128", "--qk-layernorm",
        "--micro-batch-size", "1", "--global-batch-size", "1", "--train-iters", "1",
        "--seq-length", "512", "--transformer-impl", "local", "--no-gradient-accumulation-fusion",
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0", "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
    ]
    sys.argv = ["probe", *model_args]
    args = parse_args()
    args.hf_checkpoint = MODEL
    args.megatron_to_hf_mode = "raw"
    args.model_name = "qwen3"
    args = set_default_megatron_args(args)
    validate_args(args)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hccl")
    torch.npu.set_device(0)
    init(args)

    from megatron.core import mpu

    mpu.set_tensor_model_parallel_world_size(1)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # load HF weights exactly like the training actor does
    # (megatron_bridge_utils.patch_megatron_model makes tied-embedding configs loadable)
    from slime.backends.megatron_utils.checkpoint import _load_checkpoint_hf
    from slime.utils import megatron_bridge_utils

    with megatron_bridge_utils.patch_megatron_model(model):
        _load_hf_model = _load_checkpoint_hf
        _load_hf_model(model, None, args, MODEL)
    torch.distributed.barrier()

    # HF reference tensors
    from safetensors.torch import load_file
    import glob

    hf_tensors = {}
    for shard in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        hf_tensors.update(load_file(shard))

    # model_name like the actor: type(hf_config).__name__.lower()
    from transformers import AutoConfig

    model_name = type(AutoConfig.from_pretrained(MODEL)).__name__.lower().replace("config", "")

    mismatches = []
    checked = set()
    with torch.no_grad():
        for name, param in model[0].named_parameters():
            if "_extra_state" in name:
                continue
            param_gpu = param.detach().to("npu")
            try:
                base = name
                while base.startswith("module."):
                    base = base[len("module."):]
                converted = convert_to_hf(args, model_name, "module.module." + base, param_gpu, None)
            except Exception as e:
                mismatches.append((name, f"CONVERT-ERROR {e}"))
                continue
            for hf_name, tensor in converted:
                t = tensor.detach().float().cpu()
                ref = hf_tensors.get(hf_name)
                checked.add(hf_name)
                if ref is None:
                    mismatches.append((name, f"{hf_name}: NOT IN HF CKPT (shape {tuple(t.shape)})"))
                    continue
                ref = ref.float()
                if ref.shape != t.shape:
                    mismatches.append((name, f"{hf_name}: SHAPE {tuple(t.shape)} vs ckpt {tuple(ref.shape)}"))
                    continue
                diff = (t - ref).abs().max().item()
                if diff > 1e-3:
                    # diagnose: which HF tensor does the megatron value actually match?
                    import re as _re
                    diag = ""
                    if "layernorm" in name:
                        layer = _re.search(r"layers\.(\d+)", name).group(1)
                        cands = {
                            f"same-layer input_layernorm": hf_tensors.get(f"model.layers.{layer}.input_layernorm.weight"),
                            f"same-layer post_attention": hf_tensors.get(f"model.layers.{layer}.post_attention_layernorm.weight"),
                            f"next-layer input": hf_tensors.get(f"model.layers.{int(layer)+1}.input_layernorm.weight") if int(layer)+1 < 28 else None,
                            f"next-layer post_attention": hf_tensors.get(f"model.layers.{int(layer)+1}.post_attention_layernorm.weight") if int(layer)+1 < 28 else None,
                        }
                        for tag, c in cands.items():
                            if c is None:
                                continue
                            d = (t - c.float()).abs().max().item()
                            diag += f" [{tag}: {d:.4f}]"
                    mismatches.append((name, f"{hf_name}: MAXDIFF {diff:.4f}{diag}"))

    print(f"checked {len(checked)} hf tensors; megatron params with issues: {len(mismatches)}")
    for name, msg in mismatches[:20]:
        print(f"  MISMATCH {name}: {msg}")
    # reverse check: hf tensors never sent
    missing = [k for k in hf_tensors if k not in checked]
    print(f"hf tensors never sent by updater ({len(missing)}): {missing[:8]}")
    if not mismatches and not missing:
        print("WEIGHT-SYNC MAPPING: EXACT MATCH")


if __name__ == "__main__":
    main()
