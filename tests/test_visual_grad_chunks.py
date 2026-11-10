"""Bounded gradient communication retains the exact elementwise reduction."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('op', ['sum', 'average'])
def test_production_gradient_chunks_are_views_and_keep_reduction(dtype, op):
    path = Path(__file__).resolve().parents[1] / 'frameworks/Megatron-LM/megatron/core/distributed/finalize_model_grads.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_allreduce_contiguous_grads_in_chunks')
    grads = [torch.arange(17, dtype=dtype), torch.empty(0, dtype=dtype), torch.arange(11, dtype=dtype).reshape(1, 11)]
    originals = [g.clone() for g in grads]
    calls = []

    def reduce(view, *, op, group):
        assert group == 'tp'
        assert view.numel() * view.element_size() <= 16
        assert any(view.untyped_storage().data_ptr() == g.untyped_storage().data_ptr() for g in grads)
        calls.append(op)
        # Synthetic peer contributes 3*x; SUM gives4*x, AVG gives2*x.
        view.mul_(4 if op == 'sum' else 2)

    namespace = {'torch': SimpleNamespace(distributed=SimpleNamespace(all_reduce=reduce))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    namespace[node.name](grads, op, 'tp', chunk_bytes=16)
    assert calls and all(value == op for value in calls)
    for got, before in zip(grads, originals):
        torch.testing.assert_close(got, before * (4 if op == 'sum' else 2), rtol=0, atol=0)
