"""Section-10 correctness assertions for the HSI v2 (JSD) estimator.

The V2 spec forbids a separate smoke campaign: these minute-level checks are the
gate, and the run starts from cp0 as soon as they pass. Everything here is
CPU-only (no NPU, no engine, no Megatron): the estimator's arithmetic, the
baseline's ordering guarantees, and the structural contract that the scoring
happens before the first optimizer step.
"""

import json
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods import hsi2_jsd  # noqa: E402

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Section 5 / 8 -- exact normalized JSD
# ---------------------------------------------------------------------------


def test_jsd_is_zero_when_the_two_views_agree():
    """p == q => JSD == 0. The estimator's most basic sanity property."""
    logits = torch.randn(5, 97) * 3.0
    w = hsi2_jsd._jsd_from_logits(logits.clone(), logits.clone())
    assert w.shape == (5,)
    assert torch.allclose(w, torch.zeros(5), atol=1e-6)


def test_jsd_is_finite_nonnegative_and_capped_by_one():
    """JSD is finite, >= 0, and (normalized by log 2) <= 1."""
    torch.manual_seed(0)
    for _ in range(20):
        a = torch.randn(7, 211) * torch.rand(1) * 5
        b = torch.randn(7, 211) * torch.rand(1) * 5
        w = hsi2_jsd._jsd_from_logits(a, b)
        assert torch.isfinite(w).all()
        assert (w >= -1e-7).all(), w.min()
        assert (w <= 1.0 + 1e-6).all(), w.max()


def test_jsd_is_near_one_for_disjoint_support():
    """Two one-hot distributions on different tokens => JSD -> log 2 => w -> 1."""
    a = torch.full((1, 64), -1e4)
    b = torch.full((1, 64), -1e4)
    a[0, 3] = 0.0
    b[0, 40] = 0.0
    w = hsi2_jsd._jsd_from_logits(a, b)
    assert abs(float(w[0]) - 1.0) < 1e-3, float(w[0])


def test_jsd_is_invariant_under_swapping_the_two_views():
    """swap(p, q) leaves JSD unchanged -- it is a divergence, not a direction."""
    torch.manual_seed(1)
    a = torch.randn(11, 333) * 2.0
    b = torch.randn(11, 333) * 2.0
    assert torch.allclose(
        hsi2_jsd._jsd_from_logits(a, b), hsi2_jsd._jsd_from_logits(b, a), atol=1e-6
    )


def test_jsd_matches_the_entropy_identity():
    """Cross-check against JSD = H(m) - (H(p) + H(q))/2 -- different algebra."""
    torch.manual_seed(2)
    a = torch.randn(4, 128)
    b = torch.randn(4, 128)

    def entropy(log_p):
        return -(log_p.exp() * log_p).sum(-1)

    lp = torch.log_softmax(a, dim=-1)
    lq = torch.log_softmax(b, dim=-1)
    m = 0.5 * (lp.exp() + lq.exp())
    expected = (entropy(m.log()) - 0.5 * entropy(lp) - 0.5 * entropy(lq)) / LOG2
    assert torch.allclose(hsi2_jsd._jsd_from_logits(a, b), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 4 -- historical running baseline
# ---------------------------------------------------------------------------


def _ledger(tmp_path, name="b.json"):
    ledger = hsi2_jsd._BaselineLedger(str(tmp_path / name))
    ledger.load()
    return ledger


def test_current_reward_never_enters_its_own_baseline(tmp_path):
    """Spec section 4: b_k uses ONLY trajectories that finished earlier."""
    ledger = _ledger(tmp_path)
    assert ledger.baseline(0) == 0.0  # nothing seen yet -> b_0 = 0
    ledger.commit(0, cur_sum=8.0, cur_count=8)
    assert ledger.baseline(1) == pytest.approx(1.0)  # 8/8, from rollout 0 only
    ledger.commit(1, cur_sum=0.0, cur_count=8)
    assert ledger.baseline(2) == pytest.approx(0.5)  # 8/16
    # Rollout 2's own (all-fail) rewards must not move its own baseline.
    assert ledger.baseline(2) == pytest.approx(8.0 / 16)


def test_baseline_is_zero_before_any_trajectory(tmp_path):
    ledger = _ledger(tmp_path)
    assert ledger.baseline(7) == 0.0


def test_replayed_rollout_is_not_counted_twice(tmp_path):
    """HF resume replays rollouts; a replay must be an exact no-op."""
    ledger = _ledger(tmp_path)
    ledger.commit(0, cur_sum=4.0, cur_count=8)
    first = dict(ledger.records[0])
    ledger.commit(0, cur_sum=4.0, cur_count=8)  # replay
    assert ledger.records[0] == first
    assert ledger.baseline(1) == pytest.approx(0.5)
    assert ledger.records[0]["cum_count"] == 8


def test_baseline_state_survives_a_reload(tmp_path):
    """The ledger is checkpoint-coupled: it must round-trip through disk."""
    ledger = _ledger(tmp_path)
    ledger.commit(0, cur_sum=6.0, cur_count=8)
    ledger.commit(1, cur_sum=2.0, cur_count=8)

    reloaded = _ledger(tmp_path)
    assert reloaded.baseline(2) == pytest.approx(8.0 / 16)
    assert reloaded.baseline(1) == pytest.approx(6.0 / 8)
    with open(tmp_path / "b.json") as handle:
        raw = json.load(handle)
    assert set(raw) == {"0", "1"}


def test_baseline_ignores_later_rollouts_when_replaying(tmp_path):
    """A replay of rollout 0 must not see rollout 1's contribution."""
    ledger = _ledger(tmp_path)
    ledger.commit(1, cur_sum=8.0, cur_count=8)  # crash-resume with a gap
    assert ledger.baseline(0) == 0.0


# ---------------------------------------------------------------------------
# Section 2/3 -- the outcome descriptor
# ---------------------------------------------------------------------------


def test_outcome_descriptor_renders_the_binary_verdict():
    assert hsi2_jsd.render_outcome_descriptor(1) == (
        "<outcome>The verifier judged the completed solution as correct.</outcome>"
    )
    assert hsi2_jsd.render_outcome_descriptor(0) == (
        "<outcome>The verifier judged the completed solution as incorrect.</outcome>"
    )


def test_outcome_descriptor_keeps_real_graded_semantics():
    text = hsi2_jsd.render_outcome_descriptor({"score": 0.7})
    assert "partially correct" in text and "0.70" in text
    assert hsi2_jsd.render_outcome_descriptor({"score": 1.0}).endswith("as correct.</outcome>")


def test_outcome_descriptor_never_emits_a_bare_number():
    """A raw reward scalar must never be the whole descriptor."""
    text = hsi2_jsd.render_outcome_descriptor({"passed": False})
    assert "verifier judged" in text
    assert text.strip() != "<outcome>0</outcome>"


# ---------------------------------------------------------------------------
# Section 7/9 -- paired scoring plumbing (fake actor, CPU)
# ---------------------------------------------------------------------------


class _FakeHead(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden) * 0.05)

    def forward(self, x):
        return x @ self.weight.t(), None


class _FakeModel(nn.Module):
    """Minimal stand-in for the Megatron actor: ids in, hidden states out."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.output_layer = _FakeHead(vocab, hidden)
        self.training = False

    def forward(self, input_ids, position_ids=None, attention_mask=None, labels=None):
        ids = input_ids[0] if input_ids.dim() == 2 else input_ids
        hidden = self.embed(ids)
        return self.output_layer(hidden)  # the tap intercepts this call


def test_paired_scorer_returns_one_weight_per_scored_position():
    vocab, hidden = 64, 16
    model = _FakeModel(vocab, hidden)
    normal = list(range(1, 40))
    hindsight = list(range(5, 50))

    weights = hsi2_jsd.paired_jsd_weights(
        model,
        normal_ids=normal,
        normal_start=len(normal) - 9 - 1,
        hindsight_ids=hindsight,
        hindsight_start=len(hindsight) - 9 - 1,
        n_positions=9,
    )
    assert weights is not None and len(weights) == 9
    assert all(math.isfinite(w) and -1e-6 <= w <= 1.0 + 1e-6 for w in weights)


def test_paired_scorer_is_chunk_size_invariant():
    """The position chunk is an execution knob; it must not change the value."""
    vocab, hidden = 64, 16
    model = _FakeModel(vocab, hidden)
    normal = list(range(1, 30))
    hindsight = list(range(3, 33))
    args = dict(
        normal_ids=normal,
        normal_start=len(normal) - 11 - 1,
        hindsight_ids=hindsight,
        hindsight_start=len(hindsight) - 11 - 1,
        n_positions=11,
    )
    original = hsi2_jsd.JSD_POSITION_CHUNK
    try:
        hsi2_jsd.JSD_POSITION_CHUNK = 64
        whole = hsi2_jsd.paired_jsd_weights(model, **args)
        hsi2_jsd.JSD_POSITION_CHUNK = 3
        chunked = hsi2_jsd.paired_jsd_weights(model, **args)
    finally:
        hsi2_jsd.JSD_POSITION_CHUNK = original
    assert whole == pytest.approx(chunked, abs=1e-6)


def test_paired_scorer_rejects_a_span_that_runs_past_the_sequence():
    model = _FakeModel(64, 16)
    with pytest.raises(RuntimeError, match="hidden states"):
        hsi2_jsd.paired_jsd_weights(
            model, list(range(1, 10)), 0, list(range(1, 10)), 0, 500
        )


# ---------------------------------------------------------------------------
# Section 6 -- the freeze ordering, enforced structurally
# ---------------------------------------------------------------------------


def test_postprocess_runs_before_the_first_optimizer_step():
    """The hook must sit before train() in Megatron's actor, not after it.

    This is the spec's section-6 guarantee -- every w/A is produced with one
    theta_old -- and it is a property of the call site, so it is checked at the
    call site rather than asserted at runtime.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "frameworks",
        "slime-ascend",
        "slime",
        "backends",
        "megatron_utils",
        "actor.py",
    )
    with open(os.path.abspath(path)) as handle:
        source = handle.read()
    hook = source.index("self.rollout_data_postprocess(self.args, rollout_id, rollout_data)")
    train_call = source.index("with timer(\"actor_train\"):")
    assert hook < train_call


def test_estimator_default_is_jsd(monkeypatch):
    monkeypatch.delenv("OPSD_HSI_ESTIMATOR", raising=False)
    assert hsi2_jsd.estimator() == "jsd"


# ---------------------------------------------------------------------------
# postprocess end-to-end (fake actor, CPU, no dist)
# ---------------------------------------------------------------------------


def _rollout_row(group_index, rlen, vocab=64):
    """One shard row: 26-token normal view + a hindsight view with its own ids."""
    normal = list(range(1, 27))
    hindsight = list(range(31, 57))
    plen = len(hindsight) - rlen
    return {
        "tokens": torch.tensor(normal),
        "response_lengths": rlen,
        "rewards": 1.0,
        "turn_ids": torch.tensor([group_index * 10_000] * rlen),
        "view": [hindsight],
        "plen": [plen],
        "tlen": [rlen],
    }


def test_postprocess_emits_advantage_tensors(tmp_path, monkeypatch):
    """slime's log_rollout_data branches on `isinstance(val[0], torch.Tensor)`.

    Emitting Python lists there is a hard `TypeError: int + list` crash at the
    first rollout, so the emitted type is a contract, not a detail.
    """
    from types import SimpleNamespace

    pytest.importorskip("slime", reason="postprocess needs the slime runtime package")
    monkeypatch.delenv("OPSD_HSI_ESTIMATOR", raising=False)
    hsi2_jsd.reset_baseline_for_tests()
    model = _FakeModel(64, 16)
    monkeypatch.setattr(hsi2_jsd, "_actor_model", lambda: model)

    rows = [_rollout_row(0, 6), _rollout_row(1, 5), _rollout_row(0, 6)]
    rollout_data = {
        "tokens": [r["tokens"] for r in rows],
        "response_lengths": [r["response_lengths"] for r in rows],
        "rewards": [r["rewards"] for r in rows],
        "turn_ids": [r["turn_ids"] for r in rows],
        "hpi_views": [r["view"] for r in rows],
        "hpi_prompt_lengths": [r["plen"] for r in rows],
        "hpi_target_lengths": [r["tlen"] for r in rows],
        "loss_masks": [torch.ones(6, dtype=torch.int32), torch.ones(5, dtype=torch.int32),
                       torch.zeros(6, dtype=torch.int32)],
    }
    hsi2_jsd.postprocess(SimpleNamespace(save=str(tmp_path)), 0, rollout_data)

    advantages = rollout_data["opsd_advantages"]
    assert all(isinstance(a, torch.Tensor) for a in advantages)
    assert [a.shape[0] for a in advantages] == [6, 5, 6]
    assert advantages[2].abs().sum() == 0

    # b_0 = 0 and R = 1, so A = 1.0 * w exactly.
    assert advantages[0].min() > 0.0
    assert (advantages[0] <= 1.0 + 1e-6).all()
    # The baseline absorbed exactly this rollout's two trajectories, once.
    assert hsi2_jsd.baseline_state(SimpleNamespace(save=str(tmp_path))).records[0]["cum_count"] == 2


def test_postprocess_masks_a_row_with_no_hindsight_view(tmp_path, monkeypatch):
    """A row the leak guard dropped must be MASKED, not trained towards A = 0."""
    from types import SimpleNamespace

    pytest.importorskip("slime", reason="postprocess needs the slime runtime package")
    monkeypatch.delenv("OPSD_HSI_ESTIMATOR", raising=False)
    hsi2_jsd.reset_baseline_for_tests()
    model = _FakeModel(64, 16)
    monkeypatch.setattr(hsi2_jsd, "_actor_model", lambda: model)

    row = _rollout_row(0, 4)
    rollout_data = {
        "tokens": [row["tokens"]],
        "response_lengths": [row["response_lengths"]],
        "rewards": [row["rewards"]],
        "turn_ids": [row["turn_ids"]],
        "hpi_views": [[]],  # no view -> unscorable
        "hpi_prompt_lengths": [[]],
        "hpi_target_lengths": [[]],
        "loss_masks": [torch.ones(4, dtype=torch.int32)],
    }
    hsi2_jsd.postprocess(SimpleNamespace(save=str(tmp_path)), 0, rollout_data)
    assert torch.equal(rollout_data["opsd_advantages"][0], torch.zeros(4))
    assert int(rollout_data["loss_masks"][0].sum()) == 0
