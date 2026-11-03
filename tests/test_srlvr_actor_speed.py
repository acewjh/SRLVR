"""Correctness gates for row-batched scoring and response-only TP gradients."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from methods import hsi2_jsd, srlvr_compact, srlvr_hsi, tp_utils


@pytest.mark.parametrize("variant", ["v2", "v3"])
@pytest.mark.parametrize("mask", [[1, 0, 1, 1, 0], [0, 0, 0, 0, 0]])
def test_row_scoring_preserves_dense_features(monkeypatch, variant, mask):
    torch.manual_seed(19)
    model = torch.nn.Linear(4, 9, bias=False)
    hidden = torch.randn(8, 4)
    teacher = torch.randn(9, 4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    ids = [0, 1, 2, 3, *tokens[-5:]]
    monkeypatch.setattr(hsi2_jsd, "_find_lm_head", lambda m: m)
    monkeypatch.setattr(hsi2_jsd, "_lm_head_weight", lambda m, h: h.weight)
    monkeypatch.setattr(hsi2_jsd, "_hidden_states", lambda m, h, x: hidden if x == tokens else teacher)
    monkeypatch.setattr(tp_utils, "tp_info", lambda: (1, 0, None))
    got = srlvr_hsi.score_row(model, tokens, mask, (ids, 4, 5),
                            SimpleNamespace(variant=variant, position_chunk=2), .7, 9)
    lp = (hidden[2:7] @ model.weight.t() / .7).log_softmax(-1)
    lq = (teacher[3:8] @ model.weight.t() / .7).log_softmax(-1)
    targets = torch.tensor(tokens[-5:])
    expected = (*srlvr_hsi.centered_features(lp, lq, targets, variant),
                lp[torch.arange(5), targets])
    for key, values in zip(got, expected):
        assert torch.allclose(torch.tensor(got[key]).double(), values.detach().double() * torch.tensor(mask), atol=1e-6)


def _projection_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(21)
        h0 = torch.randn(8, 1, 4)
        w0 = torch.randn(14, 4)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # Three response positions, one masked out; padded vocab on rank 1.
        batch = dict(tokens=tokens, total_lengths=[8], response_lengths=[3],
                     loss_masks=[torch.tensor([1, 0, 1])],
                     full_loss_masks=torch.tensor([[0, 0, 0, 0, 1, 0, 1, 0]]))
        hidden = h0.clone().requires_grad_()
        weight = w0[rank * 7:(rank + 1) * 7].clone().requires_grad_()
        head = srlvr_compact.HeadRef(weight, None, 2, rank, rank * 7, dist.group.WORLD, 11)
        args = SimpleNamespace(rollout_temperature=.7)
        got = srlvr_compact.log_probs_from_hidden(args, batch, hidden, head)["log_probs"][0]
        coeff = torch.tensor([.3, -.2, .8])
        (got * coeff).sum().backward()
        dense_h = h0.clone().requires_grad_()
        dense_w = w0.clone().requires_grad_()
        lp = (dense_h[4:7, 0] @ dense_w[:11].t() / .7).log_softmax(-1)
        expected = lp[torch.arange(3), tokens[0, -3:]] * batch["loss_masks"][0]
        (expected * coeff).sum().backward()
        assert torch.allclose(got, expected, atol=2e-6)
        assert torch.allclose(hidden.grad, dense_h.grad, atol=2e-6), (hidden.grad, dense_h.grad)
        assert torch.allclose(weight.grad, dense_w.grad[rank * 7:(rank + 1) * 7], atol=2e-6)
    finally:
        dist.destroy_process_group()


def test_response_only_tp2_loss_and_gradients(tmp_path, monkeypatch):
    monkeypatch.setenv("OPSD_SRLVR_SELECTIVE_CHUNK", "1")
    monkeypatch.setenv("OPSD_SRLVR_SELECTIVE_VOCAB_CHUNK", "3")
    torch.multiprocessing.spawn(_projection_worker,
        args=(f"file://{tmp_path}/gloo",), nprocs=2, join=True)
