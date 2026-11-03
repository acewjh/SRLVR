import copy

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tasks.alfworld.memory_provider import require_embedding_output_grad


def test_full_recompute_preserves_frozen_embedding_lora_gradients():
    torch.manual_seed(42)
    embedding = nn.Embedding(17, 8)
    embedding.weight.requires_grad_(False)
    embedding.register_forward_hook(require_embedding_output_grad)
    layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)])
    reference = copy.deepcopy(layers)
    tokens = torch.tensor([[1, 2, 3]])
    x = embedding(tokens)
    for layer in layers:
        x = checkpoint(layer, x, use_reentrant=True)
    x.square().mean().backward()
    y = embedding(tokens)
    for layer in reference:
        y = layer(y)
    y.square().mean().backward()
    torch.testing.assert_close(x, y)
    for actual, expected in zip(layers.parameters(), reference.parameters()):
        assert actual.grad is not None and actual.grad.norm() > 0
        torch.testing.assert_close(actual.grad, expected.grad)
    assert embedding.weight.grad is None
    with torch.no_grad():
        assert not embedding(tokens).requires_grad
