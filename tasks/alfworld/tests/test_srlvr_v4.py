"""SRLVR-v4 loss contracts: alignment, span slicing, KL, cache.

CPU-only.  The frozen teacher is monkeypatched; no HF model, NPU, or Megatron
process is needed.  The tests pin exactly the failure modes that would silently
train on the wrong target: a span that does not match the student response, a
teacher hidden slice starting at the wrong position, and the reduction units.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from methods import srlvr_v4 as v4
from methods.opsd_loss import generalized_jsd_per_token


def _args(**overrides):
    values = dict(
        n_samples_per_prompt=1,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        entropy_coef=0.0,
        calculate_per_token_loss=True,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=1,
        qkv_format="bshd",
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        hf_checkpoint="/tmp/fake-teacher",
        use_kl_loss=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.001,
        vocab_size=8,
        loss_type="custom_loss",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(response_tokens, teacher_ids, prompt_len, target_len, span_start, span_len,
           ref_log_probs=None, student_prompt=(99,)):
    tokens = list(student_prompt) + list(response_tokens)
    response_length = len(response_tokens)
    batch = dict(
        tokens=[torch.tensor(tokens)],
        unconcat_tokens=[torch.tensor(tokens)],
        total_lengths=[len(tokens)],
        response_lengths=[response_length],
        max_seq_lens=[len(tokens)],
        loss_masks=[torch.ones(response_length)],
        hpi_views=[[list(teacher_ids)]],
        hpi_prompt_lengths=[[int(prompt_len)]],
        hpi_target_lengths=[[int(target_len)]],
        hpi_user_lengths=[[int(span_start), int(span_len)]],
    )
    if ref_log_probs is not None:
        batch["ref_log_probs"] = [torch.tensor(ref_log_probs, dtype=torch.float32)]
    return batch


@pytest.fixture(autouse=True)
def _v4_env(monkeypatch):
    monkeypatch.setenv("SRLVR_VARIANT", "v4")
    monkeypatch.setenv("SRLVR_V4_BETA", "0.0")
    monkeypatch.setenv("SRLVR_V4_TOKEN_CLIP", "0.05")
    monkeypatch.setenv("SRLVR_V4_TEMPERATURE", "1.0")
    monkeypatch.setenv("SRLVR_V4_TOKEN_CHUNK", "2")
    v4._TEACHER_CACHE.clear()
    v4._STATS.update(hits=0, misses=0, forwards=0, forward_seconds=0.0, calls=0)


def _fixed_teacher(hidden, teacher_logits):
    """Install a deterministic teacher and return the captured hidden slices."""
    slices = []

    def fake_hidden(args, ids, cfg=None):
        assert list(ids) == [11, 12, 21, 22, 23, 24]
        return hidden

    def fake_logits(args, slice_):
        slices.append(slice_.clone())
        return teacher_logits[: slice_.size(0)]

    return fake_hidden, fake_logits, slices


def test_loss_matches_manual_kl_and_slices_the_turn_span(monkeypatch):
    torch.manual_seed(3)
    vocab = 8
    hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    teacher = torch.randn(2, vocab)
    fake_hidden, fake_logits, slices = _fixed_teacher(hidden, teacher)
    monkeypatch.setattr(v4, "teacher_hidden_states", fake_hidden)
    monkeypatch.setattr(v4, "teacher_logits", fake_logits)

    args = _args(vocab_size=vocab)
    batch = _batch(
        response_tokens=[22, 23],
        teacher_ids=[11, 12, 21, 22, 23, 24],
        prompt_len=2,
        target_len=4,
        span_start=1,
        span_len=2,
    )
    logits = torch.randn(1, 3, vocab, requires_grad=True)
    reduction = lambda x: x.sum()  # noqa: E731 — sum_of_token with --calculate-per-token-loss

    loss, metrics = v4.actor_loss(args, batch, logits, reduction)

    # Teacher hidden positions: P + span_start - 1 .. + R = 2,3 (NOT 1,2).
    assert len(slices) == 1
    assert torch.equal(slices[0], hidden[2:4])

    student = logits[0, 0:2].float()
    expected, raw = generalized_jsd_per_token(student, teacher, beta=0.0, temperature=1.0, token_clip=0.05)
    assert torch.allclose(loss, expected.sum())
    assert torch.allclose(metrics["srlvr_v4/mean_unclipped_kl"], raw.mean())

    # Gradient reaches the student logits; the teacher is frozen.
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert not teacher.requires_grad


def test_alignment_mismatch_fails_closed(monkeypatch):
    hidden = torch.zeros(6, 4)
    fake_hidden, fake_logits, _ = _fixed_teacher(hidden, torch.zeros(2, 8))
    monkeypatch.setattr(v4, "teacher_hidden_states", fake_hidden)
    monkeypatch.setattr(v4, "teacher_logits", fake_logits)
    batch = _batch(
        response_tokens=[22, 99],  # second token is not the teacher target
        teacher_ids=[11, 12, 21, 22, 23, 24],
        prompt_len=2,
        target_len=4,
        span_start=1,
        span_len=2,
    )
    with pytest.raises(RuntimeError, match="teacher completion tokens differ"):
        v4.actor_loss(_args(), batch, torch.randn(1, 3, 8), lambda x: x.sum())


def test_span_length_mismatch_fails_closed(monkeypatch):
    fake_hidden, fake_logits, _ = _fixed_teacher(torch.zeros(6, 4), torch.zeros(1, 8))
    monkeypatch.setattr(v4, "teacher_hidden_states", fake_hidden)
    monkeypatch.setattr(v4, "teacher_logits", fake_logits)
    batch = _batch([22, 23], [11, 12, 21, 22, 23, 24], 2, 4, 1, 1)
    with pytest.raises(RuntimeError, match="span length"):
        v4.actor_loss(_args(), batch, torch.randn(1, 3, 8), lambda x: x.sum())


def test_reference_kl_is_added_in_actor_units(monkeypatch):
    torch.manual_seed(11)
    vocab = 32  # response token ids 22/23 must be inside the student vocabulary
    hidden = torch.zeros(6, 4)
    teacher = torch.randn(2, vocab)
    fake_hidden, fake_logits, _ = _fixed_teacher(hidden, teacher)
    monkeypatch.setattr(v4, "teacher_hidden_states", fake_hidden)
    monkeypatch.setattr(v4, "teacher_logits", fake_logits)

    args = _args(vocab_size=vocab, use_kl_loss=True, kl_loss_coef=0.01)
    ref = [-0.2, -0.4]
    batch = _batch([22, 23], [11, 12, 21, 22, 23, 24], 2, 4, 1, 2, ref_log_probs=ref)
    logits = torch.randn(1, 3, vocab, requires_grad=True)
    loss, _ = v4.actor_loss(args, batch, logits, lambda x: x.sum())

    student = logits[0, 0:2].float()
    kl, _ = generalized_jsd_per_token(student, teacher, beta=0.0, temperature=1.0, token_clip=0.05)
    from tasks.alfworld.loss import reference_kl

    student_logp = torch.log_softmax(student, dim=-1).gather(
        1, torch.tensor([[22], [23]]).long()
    ).squeeze(1)
    expected = kl.sum() + 0.01 * reference_kl(args, student_logp, torch.tensor(ref)).sum()
    assert torch.allclose(loss, expected)


def test_teacher_hidden_is_cached_by_token_content(monkeypatch):
    class ToyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50, 4)
            self.lm_head = torch.nn.Linear(4, 9, bias=False)
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            hidden = self.embedding(input_ids)
            self.lm_head(hidden)
            return None

    toy = ToyTeacher()
    monkeypatch.setattr(v4, "_teacher_model", lambda args: toy)
    v4._TEACHER_CACHE.clear()
    v4._STATS.update(hits=0, misses=0, forwards=0, forward_seconds=0.0, calls=0)
    cfg = v4.Settings(variant="v4", teacher_cache=2)
    args = _args()

    first = v4.teacher_hidden_states(args, [1, 2, 3], cfg)
    second = v4.teacher_hidden_states(args, [1, 2, 3], cfg)
    assert toy.calls == 1
    assert first is second
    assert v4._STATS["hits"] == 1 and v4._STATS["misses"] == 1

    # A different sequence evicts the oldest entry when the cache is full.
    v4.teacher_hidden_states(args, [4, 5, 6], cfg)
    v4.teacher_hidden_states(args, [7, 8, 9], cfg)
    assert toy.calls == 3
    assert len(v4._TEACHER_CACHE) == 2


def test_hidden_only_student_path_matches_the_dense_projection(monkeypatch):
    """Selective logits: only response positions are projected; no prompt logits."""
    import torch.nn.functional as F

    from methods import srlvr_compact

    torch.manual_seed(7)
    vocab, width = 16, 4
    hidden = torch.randn(4, 1, width, requires_grad=True)
    weight = torch.randn(vocab, width)
    head = srlvr_compact.HeadRef(weight, None, 1, 0, 0, vocab)
    tokens = torch.tensor([[1, 2, 3, 4]])
    row_mask = torch.tensor([1.0, 1.0])
    full_mask = F.pad(row_mask, (1, 1))  # prompt=2 -> (prompt-1, 1)
    batch = dict(
        tokens=tokens,
        unconcat_tokens=[tokens[0]],
        total_lengths=[4],
        response_lengths=[2],
        max_seq_lens=[4],
        loss_masks=[row_mask],
        full_loss_masks=full_mask.unsqueeze(0),
        hpi_views=[[[11, 12, 3, 4, 5, 6]]],
        hpi_prompt_lengths=[[2]],
        hpi_target_lengths=[[4]],
        hpi_user_lengths=[[0, 2]],
    )
    args = _args(vocab_size=vocab)
    args._srlvr_hidden = True
    args._srlvr_head = head
    teacher_hidden = torch.zeros(6, width)
    teacher = torch.randn(2, vocab)
    monkeypatch.setattr(v4, "teacher_hidden_states", lambda *a, **k: teacher_hidden)
    monkeypatch.setattr(v4, "teacher_logits", lambda *a, **k: teacher)

    loss, _ = v4.actor_loss(args, batch, hidden, lambda x: x.sum())

    selected = hidden.reshape(-1, width).index_select(0, torch.tensor([1, 2]))
    student = F.linear(selected, weight).float()
    expected, _ = generalized_jsd_per_token(
        student, teacher, beta=0.0, temperature=1.0, token_clip=0.05
    )
    assert torch.allclose(loss, expected.sum())
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_overlong_row_is_zero_loss_and_never_loads_teacher(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an overlong row must not load the teacher")

    monkeypatch.setattr(v4, "teacher_hidden_states", boom)
    monkeypatch.setattr(v4, "teacher_logits", boom)
    batch = _batch([22, 23], [0], 1, 0, -1, 0)
    batch["loss_masks"] = [torch.zeros(2)]
    batch["hpi_target_lengths"] = [[0]]
    loss, metrics = v4.actor_loss(
        _args(),
        batch,
        torch.randn(1, 3, 8),
        lambda x: (x * torch.tensor([0.0, 0.0])).sum(),
    )
    assert torch.allclose(loss, torch.tensor(0.0))
    assert float(metrics["srlvr_v4/num_tokens"]) == 2.0


def test_zero_mask_padding_row_with_a_valid_view_is_zero_loss(monkeypatch):
    """The generic convert padding copies a real hpi view; its zero mask skips it.

    This is the exact 2026-10-27 crash: a DP padding row had a valid span, so
    the compact path tried to select a response from an all-zero loss mask.
    """
    def boom(*args, **kwargs):
        raise AssertionError("a padding row must not load or project")

    monkeypatch.setattr(v4, "teacher_hidden_states", boom)
    monkeypatch.setattr(v4, "teacher_logits", boom)
    batch = _batch([22, 23], [11, 12, 3, 4, 5, 6], 2, 4, 0, 2)
    batch["loss_masks"] = [torch.zeros(2)]
    reduction = lambda x: (x * torch.zeros(2)).sum()  # noqa: E731

    for hidden_mode in (False, True):
        args = _args()
        if hidden_mode:
            args._srlvr_hidden = True
            args._srlvr_head = object()  # never dereferenced on the skip path
        loss, metrics = v4.actor_loss(args, batch, torch.randn(1, 3, 8), reduction)
        assert torch.allclose(loss, torch.tensor(0.0))
        assert float(metrics["srlvr_v4/num_tokens"]) == 2.0


def test_settings_and_runtime_fail_closed(monkeypatch):
    settings = v4.Settings.from_env(_args())
    assert settings.compact_future is True
    assert settings.beta == 0.0 and settings.token_clip == 0.05

    monkeypatch.setenv("SRLVR_V4_BETA", "1.5")
    with pytest.raises(ValueError, match="BETA"):
        v4.Settings.from_env(_args())

    monkeypatch.setenv("SRLVR_V4_BETA", "0.0")
    # TP>1 is accepted only together with the selective (hidden-state) path; the
    # stock path would compare a rank-local vocabulary slice to a global teacher.
    with pytest.raises(ValueError, match="OPSD_SRLVR_SELECTIVE_LOGITS"):
        v4.validate_runtime(_args(tensor_model_parallel_size=2))
    monkeypatch.setenv("OPSD_SRLVR_SELECTIVE_LOGITS", "1")
    v4.validate_runtime(_args(tensor_model_parallel_size=2))
    v4.validate_runtime(_args(tensor_model_parallel_size=4, vocab_size=8))
    monkeypatch.delenv("OPSD_SRLVR_SELECTIVE_LOGITS")

    with pytest.raises(ValueError, match="calculate-per-token-loss"):
        v4.validate_runtime(_args(calculate_per_token_loss=False))


def _tp_sharded_worker(rank, rendezvous, vocab_size, temperature, beta=0.0):
    """The sharded KL must equal the dense full-vocabulary KL in VALUE *and* GRADIENT.

    The gradient half is the load-bearing assertion. The first version of this
    test only compared values, and a pass-through backward on the shared global
    log-partition matched dense to 4e-16 in value while the hidden-state gradient
    was off by 49%: a real 2-node full-RL run then collapsed from 0.34 to 0.00
    action-valid rate with a healthy-looking ``grad_norm``. Values alone certify
    nothing about the training signal.
    """
    import torch.distributed as dist
    import torch.nn.functional as F

    from methods import srlvr_compact

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(11)
        rows, width = 5, 4
        hidden = torch.randn(rows, width)
        teacher_hidden = torch.randn(rows, width)
        weight = torch.randn(vocab_size, width)
        targets = torch.tensor([0, vocab_size - 1, vocab_size // 2, 3, 7])
        cfg = v4.Settings(variant="v4", temperature=temperature, beta=beta)

        # Dense reference, including its gradient.
        dense_hidden = hidden.clone().requires_grad_(True)
        dense_student = F.linear(dense_hidden, weight).float()
        dense_teacher = F.linear(teacher_hidden, weight).float()
        dense_clipped, dense_raw = generalized_jsd_per_token(
            dense_student, dense_teacher, beta=beta, temperature=temperature,
            token_clip=cfg.token_clip,
        )
        dense_logp = F.log_softmax(dense_student / temperature, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        dense_raw.sum().backward()

        shard = weight[rank * (vocab_size // 2) : (rank + 1) * (vocab_size // 2)]
        head = srlvr_compact.HeadRef(
            shard, None, 2, rank, rank * (vocab_size // 2),
            dist.group.WORLD, vocab_size,
        )
        sharded_hidden = hidden.clone().requires_grad_(True)
        student = v4._student_projection(sharded_hidden, head)
        clipped, raw, logp = v4._sharded_jsd_and_logp(
            student, dense_teacher, targets, head, cfg
        )

        assert torch.allclose(clipped.double(), dense_clipped.double(), atol=1e-5)
        assert torch.allclose(raw.double(), dense_raw.double(), atol=1e-5)
        assert torch.allclose(logp.double(), dense_logp.double(), atol=1e-5)

        raw.sum().backward()
        assert sharded_hidden.grad is not None and torch.isfinite(sharded_hidden.grad).all()
        # The hidden state is replicated across TP; _student_projection's grad_sum
        # performs the required SUM, so each rank must end up with the full dense
        # gradient, not one shard's contribution.
        err = (sharded_hidden.grad - dense_hidden.grad).abs().max().item()
        scale = dense_hidden.grad.abs().max().item() + 1e-12
        assert err / scale < 1e-5, f"gradient mismatch: {err:.3e} (rel {err / scale:.3e})"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_tp2_sharded_jsd_matches_the_dense_full_vocabulary(tmp_path, beta):
    """Value AND gradient must match dense for KL (beta=0), JSD (0.5) and reverse KL (1).

    beta=0.5 is the arm that actually runs on the cluster now (generalized JSD),
    so the mixture branch is not allowed to be the untested one.
    """
    torch.multiprocessing.spawn(
        _tp_sharded_worker,
        args=(f"file://{tmp_path}/gloo", 16, 1.0, beta),
        nprocs=2,
        join=True,
    )
