"""Minimal HCCL 2-card allreduce check (invoked by scripts/check_env.sh via torchrun)."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> None:
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)

    x = torch.ones(1024, device=f"npu:{rank}") * (rank + 1)
    dist.all_reduce(x)
    expected = float(sum(range(1, world + 1)))
    assert torch.allclose(x, torch.full_like(x, expected)), f"rank {rank}: {x[0].item()} != {expected}"
    if rank == 0:
        print(f"hccl allreduce ok: world={world}, value={x[0].item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
