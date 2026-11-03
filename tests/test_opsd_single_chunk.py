"""Guard the single-chunk short-circuit in teacher_forward_completion_logits.

torch.cat on a one-element list still allocates a full second copy of that
tensor. That copy is what OOMed the 4B run (594 MB = ~977 response tokens x
151936 vocab x 4 B, with 82 MiB free), so the single-chunk case returns the
tensor itself. The short-circuit must be a pure memory no-op: same values,
same shape, and no cat call.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJ = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJ)

from methods import opsd_loss  # noqa: E402

# The implementation moved to opsd_core/teacher.py (REFACTOR_PLAN.md §3);
# `methods.opsd_loss` re-exports it. The teacher seam must be patched on the
# module that DEFINES `teacher_forward_completion_logits` and resolves
# `get_teacher` from its own globals — patching the shim is a silent no-op.
from opsd_core import teacher as _teacher_impl  # noqa: E402

VOCAB = 8
PROMPT_LEN = 3
RESPONSE_LEN = 4
_SEEN_BATCHES: list[int] = []


def _stub_logits(sample_ids: torch.Tensor, total_len: int) -> torch.Tensor:
    """Deterministic [B, T, V] logits; the value encodes (sample id, position, vocab).

    Keyed on the sample's own id rather than its row within the chunk, so the
    expected output is independent of how the batch happened to be chunked.
    """
    b = sample_ids.to(torch.float32).view(-1, 1, 1) * 1000.0
    t = torch.arange(total_len, dtype=torch.float32).view(1, total_len, 1) * 10.0
    v = torch.arange(VOCAB, dtype=torch.float32).view(1, 1, -1)
    return b + t + v


def _expected(n_prompts: int) -> torch.Tensor:
    """The rows the teacher should return: positions [P-1, P-1+R) of each sample."""
    ids = torch.arange(n_prompts, dtype=torch.long)
    full = _stub_logits(ids, PROMPT_LEN + RESPONSE_LEN)
    return torch.cat([full[i, PROMPT_LEN - 1 : PROMPT_LEN - 1 + RESPONSE_LEN, :] for i in range(n_prompts)])


class _StubModel(torch.nn.Module):
    """Stands in for the frozen teacher: shape-aware logits, no real weights."""

    def __init__(self) -> None:
        super().__init__()
        # a real parameter, because the caller reads next(model.parameters()).device
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):  # noqa: ARG002
        batch, total_len = input_ids.shape
        _SEEN_BATCHES.append(batch)
        # each sample carries its own id in every token, so it survives chunking
        return SimpleNamespace(logits=_stub_logits(input_ids[:, 0], total_len))


@pytest.fixture(autouse=True)
def teacher(monkeypatch):
    _SEEN_BATCHES.clear()
    monkeypatch.setattr(_teacher_impl, "get_teacher", lambda args: _StubModel())  # noqa: ARG005
    monkeypatch.setenv("OPSD_TEACHER_CHUNK", "2")
    return None


def _call(n_prompts: int) -> torch.Tensor:
    tokens = [
        torch.full((PROMPT_LEN + RESPONSE_LEN,), i, dtype=torch.long) for i in range(n_prompts)
    ]
    return opsd_loss.teacher_forward_completion_logits(
        args=None,
        teacher_tokens=tokens,
        teacher_prompt_lengths=[PROMPT_LEN] * n_prompts,
        response_lengths=[RESPONSE_LEN] * n_prompts,
    )


def test_single_chunk_returns_values_without_calling_cat(monkeypatch):
    # Arrange: build the expected value FIRST -- patching torch.cat is global (it
    # is the same module object), so it would break _expected()'s own cat too.
    expected = _expected(1)

    # any torch.cat call fails loudly, so only the short-circuit can pass
    def _no_cat(*a, **k):  # noqa: ARG001
        raise AssertionError("torch.cat must not be called for a single chunk")

    monkeypatch.setattr(opsd_loss.torch, "cat", _no_cat)

    # Act
    out = _call(n_prompts=1)

    # Assert
    assert out.shape == (RESPONSE_LEN, VOCAB)
    assert torch.equal(out, expected)


def test_single_chunk_matches_the_cat_path_numerically(monkeypatch):
    # Arrange: capture what the concatenating path would have produced.
    captured = {}
    real_cat = torch.cat

    def _spy(tensors, *a, **k):
        captured["out"] = real_cat(tensors, *a, **k)
        return captured["out"]

    fast = _call(n_prompts=1)
    monkeypatch.setattr(opsd_loss.torch, "cat", _spy)

    # Act
    batched = _call(n_prompts=2)

    # Assert: the N=1 result is the same tensor the cat path yields for its chunk.
    assert "out" in captured
    assert batched.shape == (2 * RESPONSE_LEN, VOCAB)
    assert torch.equal(fast, captured["out"][:RESPONSE_LEN])
    assert not torch.equal(fast, batched)  # sanity: the slice really is a prefix


def test_multi_chunk_still_concatenates_in_order(monkeypatch):
    # Arrange: chunk size 1 over two prompts forces two cat-able chunks.
    monkeypatch.setenv("OPSD_TEACHER_CHUNK", "1")

    # Act
    out = _call(n_prompts=2)

    # Assert
    assert out.shape == (2 * RESPONSE_LEN, VOCAB)
    assert torch.equal(out, _expected(2))


def test_single_chunk_output_is_fp32():
    # Arrange / Act
    out = _call(n_prompts=1)

    # Assert: the downstream loss consumes fp32 logits.
    assert out.dtype == torch.float32


def test_chunk_size_env_is_honoured(monkeypatch):
    # Arrange: 4 prompts at chunk_size 2 => 2 forwards of 2 => 2 chunks => cat path.
    monkeypatch.setenv("OPSD_TEACHER_CHUNK", "2")

    # Act
    out = _call(n_prompts=4)

    # Assert
    assert _SEEN_BATCHES == [2, 2]
    assert out.shape == (4 * RESPONSE_LEN, VOCAB)
    assert torch.equal(out, _expected(4))


def test_single_chunk_calls_cat_exactly_zero_times(monkeypatch):
    # Arrange: expected first, because the patch below is global to the torch module.
    expected = _expected(1)

    # count, rather than forbid, so the assertion states the invariant
    calls = []
    real_cat = torch.cat

    def _counting(*a, **k):
        calls.append(1)
        return real_cat(*a, **k)

    monkeypatch.setattr(opsd_loss.torch, "cat", _counting)

    # Act
    out = _call(n_prompts=1)

    # Assert: the whole point of the fix -- no second copy is allocated.
    assert calls == []
    assert torch.equal(out, expected)
