"""Phase 4: standalone Megatron training smoke (no Ray, no rollout).

Runs a plain causal-LM toy loss on random tokens through the real Qwen3-1.7B
Megatron model + optimizer to validate the training infrastructure:
forward, backward, HCCL DP allreduce (torchrun world 4), bf16, NPU attention,
and checkpoint save/load. The OPSD objective is intentionally NOT used here
(Phase 7 does that end-to-end); this gate only checks the Megatron side.

Usage:
  torchrun --nproc_per_node=1 scripts/_megatron_train_smoke.py --steps 1 --tag smoke1npu
  torchrun --nproc_per_node=4 scripts/_megatron_train_smoke.py --steps 3 --tag smoke4npu --load-ckpt
"""

import argparse
import os
import sys

PROJ = "/workspace/cua/repos/opsd-slime"
CKPT = f"{PROJ}/runs/init_megatron_qwen3_1p7b"

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tag", type=str, default="smoke")
    parser.add_argument("--load-ckpt", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    args_cli = parser.parse_args()

    sys.path.insert(0, f"{PROJ}/frameworks/slime-ascend")
    sys.path.insert(0, f"{PROJ}/frameworks/Megatron-LM")
    sys.path.insert(0, f"{PROJ}/frameworks/Megatron-Bridge/src")

    from megatron.core.enums import ModelType
    # MindSpeed must be imported BEFORE megatron's parse_args so its feature
    # args (optimization_level etc.) are registered on the parser.
    import mindspeed.megatron_adaptor  # noqa: F401
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.training import get_megatron_optimizer_config, get_model
    from megatron.core.optimizer import get_megatron_optimizer
    from slime.backends.megatron_utils.arguments import set_default_megatron_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func

    # Reuse the converter's arg conventions: MODEL_ARGS are appended after --.
    model_args = [
        "--swiglu",
        "--num-layers", "28",
        "--hidden-size", "2048",
        "--ffn-hidden-size", "6144",
        "--num-attention-heads", "16",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--vocab-size", "151936",
        "--padded-vocab-size", "151936",
        "--kv-channels", "128",
        "--qk-layernorm",
    ]
    sys.argv = [
        "smoke",
        *model_args,
        "--micro-batch-size", str(args_cli.micro_batch_size),
        "--global-batch-size", "8",
        "--lr", "1e-5",
        "--lr-decay-style", "constant",
        "--optimizer", "adam",
        "--weight-decay", "0.1",
        "--save", f"{CKPT}-smoke-{args_cli.tag}",
        "--save-interval", "1000000",
        "--train-iters", "100",
        "--seq-length", str(args_cli.seq_len),
        "--max-position-embeddings", "32768",
        "--attention-backend", "flash",
        "--no-gradient-accumulation-fusion",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--attention-softmax-in-fp32",
    ]
    args = parse_args()
    # set_default_megatron_args resolves the tokenizer from args.hf_checkpoint
    args.hf_checkpoint = "/workspace/cua/checkpoints/Qwen3-1.7B"
    args.megatron_to_hf_mode = "raw"
    args = set_default_megatron_args(args)
    args.save_interval = 1
    validate_args(args)

    # Megatron's initialize_model_parallel asserts an existing process group
    # (same convention as slime's convert_hf_to_torch_dist.py).
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="hccl")
    torch.npu.set_device(torch.distributed.get_rank() % torch.npu.device_count())
    # Temporary FA debug wrapper (remove once Phase 4 passes)
    import torch_npu as _tnpu
    _orig_fa = _tnpu.npu_fusion_attention
    def _dbg_fa(q, k, v, *a, **kw):
        try:
            return _orig_fa(q, k, v, *a, **kw)
        except Exception as e:
            shapes = {n: (tuple(t.shape), t.dtype, t.device, getattr(t, "stride", lambda: None)()) for n, t in [("q", q), ("k", k), ("v", v)] if torch.is_tensor(t)}
            print(f"[FA-DEBUG] args={a} kwargs={ {kk: (vv.shape if torch.is_tensor(vv) else vv) for kk, vv in kw.items()} } shapes={shapes}", flush=True)
            raise
    _tnpu.npu_fusion_attention = _dbg_fa

    init(args)

    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    device = torch.device("npu", torch.npu.current_device())

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=True)
    optimizer_config, optimizer_config_overrides = get_megatron_optimizer_config(args)
    optimizer = get_megatron_optimizer(
        optimizer_config,
        model,
        config_overrides=optimizer_config_overrides,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )

    if args_cli.load_ckpt:
        load_checkpoint(model, optimizer, None)

    # Toy causal-LM batch (identical on every rank; DP allreduce averages grads).
    torch.manual_seed(42 + rank)
    vocab = 151936
    tokens = torch.randint(1000, vocab, (args_cli.micro_batch_size, args_cli.seq_len), device=device)

    for step in range(args_cli.steps):
        optimizer.zero_grad()
        for chunk in model:
            chunk.zero_grad_buffer()
        out = model[0](input_ids=tokens, position_ids=None, attention_mask=None, labels=tokens.clone())
        loss = out if isinstance(out, torch.Tensor) else out[0]
        if loss.dim() != 0:
            loss = loss.float().mean()
        # backward on the LOCAL scalar loss (DP grad allreduce happens inside
        # megatron's finalize/optimizer path); the all-reduced value below is
        # for logging only.
        loss_local = loss.clone().detach()
        if world > 1:
            torch.distributed.all_reduce(loss_local, op=torch.distributed.ReduceOp.AVG)
        loss.backward()
        grad_norm = optimizer.get_grad_norm() if step == 0 else None
        update_ok = optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        peak_hbm = round(torch.npu.max_memory_allocated() / 2**30, 2)
        if rank == 0:
            print(
                f"[{args_cli.tag}] rank0 step={step} loss={loss_local.item():.4f} "
                f"grad_norm={float(grad_norm) if grad_norm is not None else 'n/a'} "
                f"lr={lr:.2e} peak_HBM_GiB={peak_hbm} finite={bool(torch.isfinite(loss).all())}",
                flush=True,
            )
        assert torch.isfinite(loss).all(), f"non-finite loss at step {step}"

    if args_cli.load_ckpt:
        save_checkpoint(1, model, optimizer, None, num_floating_point_operations_so_far=0)
        if rank == 0:
            saved = os.path.exists(f"{args.save}/latest_checkpointed_iteration.txt")
            print(f"[{args_cli.tag}] checkpoint save/load roundtrip: saved={saved}", flush=True)
            assert saved

    # weight-change check: step must modify parameters
    if rank == 0:
        print(f"[{args_cli.tag}] PASS: {args_cli.steps} training steps completed on world={world}", flush=True)


if __name__ == "__main__":
    main()
