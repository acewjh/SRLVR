"""Validate this project's Megatron DCP checkpoint before unattended recovery."""

import json
import pickle
import sys
from pathlib import Path

import torch


def validate(run_dir, iteration):
    folder = Path(run_dir) / f"iter_{int(iteration):07d}"
    common = torch.load(folder / "common.pt", map_location="cpu", weights_only=False)
    if int(common["iteration"]) != int(iteration):
        raise ValueError("common.pt iteration does not match checkpoint directory")
    with (folder / ".metadata").open("rb") as handle:
        metadata = pickle.load(handle)
    if not metadata.storage_data or not any("lora_" in k for k in metadata.state_dict_metadata):
        raise ValueError("checkpoint contains no LoRA tensors or shard storage")
    for storage in metadata.storage_data.values():
        shard = folder / storage.relative_path
        if shard.stat().st_size < storage.offset + storage.length:
            raise ValueError(f"truncated checkpoint shard: {shard}")
    args = common["args"]
    full_state = not args.no_save_optim and not args.no_save_rng
    if full_state and "optimizer" not in common and not any(
        "optimizer" in k for k in metadata.state_dict_metadata
    ):
        raise ValueError("checkpoint claims optimizer state but contains none")
    return {"iteration": int(iteration), "full_train_state": full_state,
            "tensor_entries": len(metadata.state_dict_metadata), "validated": True}


if __name__ == "__main__":
    try:
        print(json.dumps(validate(sys.argv[1], int(sys.argv[2]))))
    except Exception as exc:
        print(f"checkpoint validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
