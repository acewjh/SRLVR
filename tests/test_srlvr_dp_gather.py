"""Bounded DP gradient reconstruction and the first-batch acceptance probe.

The Qwen3-4B gradient bucket is ~7.5 GiB. A whole-bucket ``all_gather`` stages a
second copy of it inside the collective, which OOMed the first training update
on a 29.49 GiB card. These tests pin the two properties of the fix:

* the reconstruction gathers disjoint DP shards in bounded chunks and produces
  the same bytes as an unchunked gather;
* the acceptance probe verifies every trainable slice without materialising the
  full gradient.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from methods import srlvr


class _Bucket:
    def __init__(self, grad_data):
        self.grad_data = grad_data


class _Buffer:
    def __init__(self, buckets):
        self.buckets = buckets


class _ModelChunk:
    def __init__(self, buckets, params=()):
        self.buffers = [_Buffer(buckets)]
        self.expert_parallel_buffers = []
        self.ddp_config = SimpleNamespace(use_distributed_optimizer=True)
        self._params = list(params)

    def parameters(self):
        return iter(self._params)


class _Param:
    def __init__(self, main_grad, requires_grad=True):
        self.main_grad = main_grad
        self.requires_grad = requires_grad
        self._srlvr_tp_mode = ""


def _install_dp(monkeypatch, rank_data, rank):
    """Install a fake DP group whose ranks own ``rank_data[i]``.

    The gathered offset is read back from the source view's storage offset, so
    the fake collective also checks that each rank contributed its own shard.
    """
    dp = len(rank_data)
    shard_len = rank_data[0].numel()
    calls = []

    def all_gather(output_list, input_tensor, group=None, async_op=False):
        offset = input_tensor.storage_offset() - rank * shard_len
        size = input_tensor.numel()
        calls.append(sum(out.numel() for out in output_list) * input_tensor.element_size())
        for i, out in enumerate(output_list):
            out.copy_(rank_data[i][offset:offset + size])

    mpu = SimpleNamespace(
        get_data_parallel_group=lambda: None,
        get_data_parallel_world_size=lambda: dp,
        get_data_parallel_rank=lambda: rank,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    return calls


def test_reconstruct_dp_gradients_chunks_and_orders_shards(monkeypatch):
    # Arrange
    dp, shard_len = 4, 4096
    grad_data = torch.zeros(dp * shard_len, dtype=torch.float32)
    rank_data = [torch.arange(shard_len, dtype=torch.float32) + (r + 1) * 1e5
                 for r in range(dp)]
    rank = 2
    grad_data.chunk(dp)[rank].copy_(rank_data[rank])
    calls = _install_dp(monkeypatch, rank_data, rank)
    model = [_ModelChunk([_Bucket(grad_data)])]
    chunk_bytes = 4 * 4 * 512  # four ranks x 512 fp32 elements per collective

    # Act
    srlvr._reconstruct_dp_gradients(model, chunk_bytes=chunk_bytes)

    # Assert
    assert torch.equal(grad_data, torch.cat(rank_data))
    assert len(calls) == shard_len // 512
    assert max(calls) <= chunk_bytes


def test_reconstruct_dp_gradients_is_noop_without_dp(monkeypatch):
    grad_data = torch.zeros(16)
    model = [_ModelChunk([_Bucket(grad_data)])]
    mpu = SimpleNamespace(
        get_data_parallel_group=lambda: None,
        get_data_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    monkeypatch.setattr(torch.distributed, "all_gather",
                        lambda *a, **k: pytest.fail("no collective for dp=1"))

    srlvr._reconstruct_dp_gradients(model)

    assert torch.equal(grad_data, torch.zeros(16))


def test_reconstruct_dp_gradients_rejects_uneven_shards(monkeypatch):
    grad_data = torch.zeros(10)  # 10 is not divisible by dp=4
    model = [_ModelChunk([_Bucket(grad_data)])]
    _install_dp(monkeypatch, [torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2)], 0)

    with pytest.raises(RuntimeError, match="split into 4 equal shards"):
        srlvr._reconstruct_dp_gradients(model)


def _acceptance_fixture(monkeypatch, dp=4, numel=1024, rank=1):
    expected = (1 + dp) / 2
    grad_data = torch.zeros(numel, dtype=torch.bfloat16)
    per_param = numel // 4
    params = [_Param(grad_data[i * per_param:(i + 1) * per_param]) for i in range(4)]
    chunk = _ModelChunk([_Bucket(grad_data)], params)
    chunk.zero_grad_buffer = lambda: grad_data.zero_()

    def finish_grad_sync():
        # A real reduce-scatter leaves only this rank's shard populated.
        grad_data.chunk(dp)[rank].fill_(expected)

    chunk.finish_grad_sync = finish_grad_sync
    rank_data = [torch.full((numel // dp,), expected, dtype=torch.bfloat16)
                 for _ in range(dp)]
    _install_dp(monkeypatch, rank_data, rank)
    monkeypatch.setattr(srlvr, "_DP_GATHER_CHUNK_BYTES", 64)
    monkeypatch.setattr("methods.tp_utils.tp_info", lambda: (1, 0, None))
    optimizer = SimpleNamespace(zero_grad=lambda: None)
    return [chunk], optimizer, rank_data


def test_verify_dp_capture_accepts_bounded_reconstruction(monkeypatch, caplog):
    model, optimizer, _ = _acceptance_fixture(monkeypatch)

    with caplog.at_level("INFO", logger="methods.srlvr"):
        srlvr.verify_dp_capture(model, optimizer)

    assert "[srlvr-dp-check]" in caplog.text
    assert "max_error=0.0" in caplog.text
    assert "parameters=1024" in caplog.text


def test_verify_dp_capture_rejects_corrupted_shard(monkeypatch):
    model, optimizer, rank_data = _acceptance_fixture(monkeypatch)
    rank_data[3] = torch.zeros_like(rank_data[3])

    with pytest.raises(RuntimeError, match="DP gradient shard reconstruction failed"):
        srlvr.verify_dp_capture(model, optimizer)
