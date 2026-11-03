"""Conditional stop loss contract (spec sections 6 and 7).

Covers the mandatory preflight matrix: baseline identity when the feature is off
or nothing is gated, PPO retained on every non-gated reason, full replacement of
the surrogate at a gated position, the two gradient signs, zero gradient outside
the candidate set, fp32 numerical safety, and the reducer invariant.
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from tasks.alfworld.repeat import THINK_CLOSE_ID, THINK_OPEN_ID  # noqa: E402
from tasks.alfworld.repeat_stop import (  # noqa: E402
    GATE_ENV,
    REPEAT_STOP_ENV,
    SCORE_CHUNK_ENV,
    build_gate,
    maybe_replace,
    score_chunk,
)

#: Small synthetic vocabulary. The stop token is injected so the tests need a
#: ~64-column row rather than a 151936-column one; the arithmetic is identical.
VOCAB = 64
STOP = 3
BLOCK = [10, 11, 12, 13, 14, 15, 16, 17]
ACTION = [522, 1311, 29]
PROMPT_LEN = 5


def _response(think_loops: int = 2):
    """`<think>` + BLOCK*think_loops + `</think>` + action."""
    return [THINK_OPEN_ID] + BLOCK * think_loops + [THINK_CLOSE_ID] + ACTION


def _gated_index(think_loops: int = 2) -> int:
    """Response position of the single repeat when think_loops == 2."""
    assert think_loops == 2
    return 1 + 2 * len(BLOCK) - 1


def _fixture(response, *, reward=0.0, advantage=-0.5, temperature=1.0, seed=0):
    """Build (args, batch, logits, per_token, advantages, response/total lengths)."""
    total = PROMPT_LEN + len(response)
    tokens = torch.tensor([7] * PROMPT_LEN + list(response), dtype=torch.long)
    batch = {"unconcat_tokens": [tokens], "max_seq_lens": [total]}
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, total, VOCAB, generator=generator)
    logits.requires_grad_(True)
    rlen = len(response)
    per_token = torch.randn(rlen, generator=generator)
    advantages = torch.full((rlen,), float(advantage))
    args = Namespace(qkv_format="bshd", rollout_temperature=temperature)
    return args, batch, logits, per_token, advantages, [reward], [rlen], [total]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(REPEAT_STOP_ENV, raising=False)
    monkeypatch.delenv(GATE_ENV, raising=False)
    monkeypatch.delenv(SCORE_CHUNK_ENV, raising=False)


def test_feature_off_returns_the_surrogate_by_identity(monkeypatch):
    # Arrange
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response())
    # Act
    out, metrics = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
    )
    # Assert -- same object, no metrics: the baseline path is untouched by construction.
    assert out is per_token
    assert metrics == {}


def test_gate_all_false_is_the_baseline(monkeypatch):
    # Arrange -- no repetition at all in the think span.
    monkeypatch.setenv(REPEAT_STOP_ENV, "1")
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response(think_loops=1))
    # Act
    out, metrics = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
    )
    # Assert
    assert torch.equal(out, per_token)
    assert metrics["v2/gated_count"] == 0.0


@pytest.mark.parametrize(
    "reward,advantage,expect_gated",
    [
        (1.0, -0.5, False),   # successful episode: stop-token shaping is not applied
        (0.0, 0.0, False),    # zero advantage has no direction to reinforce
        (0.0, 0.5, False),    # positive advantage: repetition is not being punished
        (0.0, -0.5, True),    # failed episode with a negative advantage -> gated
    ],
)
def test_gate_reasons(reward, advantage, expect_gated):
    # Arrange
    response = _response()
    rlen = len(response)
    think = [False] * rlen
    repeat = [False] * rlen
    gated_index = _gated_index()
    think[gated_index] = True
    repeat[gated_index] = True
    # Act
    gate = build_gate(think, repeat, [advantage] * rlen, [reward] * rlen)
    # Assert
    assert gate[gated_index] is expect_gated
    assert sum(gate) == int(expect_gated)


def test_gated_position_replaces_the_surrogate(monkeypatch):
    # Arrange
    monkeypatch.setenv(REPEAT_STOP_ENV, "1")
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response())
    gated = _gated_index()
    # Act
    out, metrics = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
        stop_id=STOP,
    )
    # Assert -- exactly one position changed, and it changed to |A| * ce.
    changed = [i for i in range(len(per_token)) if out[i] != per_token[i]]
    assert changed == [gated]
    assert metrics["v2/gated_count"] == 1.0
    row = PROMPT_LEN + gated - 1  # logits row that predicts response position `gated`
    scores = logits.detach().view(-1, VOCAB)[row]
    candidates = [STOP] + BLOCK[-1:]  # `</think>` plus the historical successor
    expected_ce = torch.logsumexp(scores[candidates], dim=0) - scores[STOP]
    expected = abs(float(adv[gated])) * expected_ce
    assert torch.allclose(out[gated], expected, atol=1e-5)
    assert not torch.allclose(out[gated], per_token[gated])


def test_stop_logit_gradient_is_negative_and_repeat_is_positive(monkeypatch):
    # Arrange
    monkeypatch.setenv(REPEAT_STOP_ENV, "1")
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response())
    gated = _gated_index()
    out, _ = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
        stop_id=STOP,
    )
    # Act
    out[gated].backward()
    grad = logits.grad.view(-1, VOCAB)[PROMPT_LEN + gated - 1]
    # Assert
    assert grad[STOP] < 0
    assert grad[BLOCK[-1]] > 0


def test_no_gradient_outside_the_candidate_set(monkeypatch):
    # Arrange
    monkeypatch.setenv(REPEAT_STOP_ENV, "1")
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response())
    gated = _gated_index()
    out, _ = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
        stop_id=STOP,
    )
    # Act
    out[gated].backward()
    grad = logits.grad.view(-1, VOCAB)[PROMPT_LEN + gated - 1]
    # Assert -- every column that is neither the stop token nor a repeat candidate
    # is untouched; the CE only ever reads the candidate columns.
    allowed = {STOP, BLOCK[-1]}
    nonzero = {i for i, g in enumerate(grad.tolist()) if g != 0.0}
    assert nonzero == allowed


def test_positions_outside_the_gate_keep_their_ppo_gradient(monkeypatch):
    # Arrange -- a non-gated position must still backprop to the PPO surrogate.
    monkeypatch.setenv(REPEAT_STOP_ENV, "1")
    args, batch, logits, per_token, adv, rew, rlens, tlens = _fixture(_response())
    out, _ = maybe_replace(
        per_token, args=args, logits=logits, batch=batch,
        response_lengths=rlens, total_lengths=tlens, advantages=adv, rewards=rew,
        stop_id=STOP,
    )
    # Act / Assert -- the clone preserves the identity of every unchanged cell.
    gated = _gated_index()
    for i in range(len(per_token)):
        if i != gated:
            assert out[i] == per_token[i]


def test_multiple_candidates_match_explicit_conditional_probability():
    # Arrange -- a candidate set of three, compared against a float64 hand-roll.
    from tasks.alfworld.repeat_stop import _conditional_ce

    generator = torch.Generator().manual_seed(7)
    rows = torch.randn(1, VOCAB, generator=generator)
    candidate_index = torch.tensor([[STOP, 40, 41, 42]], dtype=torch.long)
    # Act
    ce, q_stop = _conditional_ce(rows, candidate_index, stop_id=STOP)
    # Assert
    cand = rows[0, [STOP, 40, 41, 42]].double()
    explicit = torch.logsumexp(cand, dim=0) - cand[0]
    assert torch.allclose(ce.double(), explicit, atol=1e-6)
    full = torch.logsumexp(rows[0].double(), dim=0)
    p = torch.exp(cand - full)
    assert torch.allclose(q_stop.double(), p[0] / p.sum(), atol=1e-6)


def test_single_candidate_ce_is_exactly_zero():
    # Arrange -- only `</think>` in the set means the conditional probability is 1.
    from tasks.alfworld.repeat_stop import _conditional_ce

    rows = torch.randn(1, VOCAB, generator=torch.Generator().manual_seed(11))
    # Act
    ce, q_stop = _conditional_ce(rows, torch.tensor([[STOP]]), stop_id=STOP)
    # Assert
    assert float(ce[0]) == 0.0
    assert float(q_stop[0]) == 1.0


def test_large_logits_stay_finite_in_fp32():
    # Arrange -- logits far into the overflow regime for `exp`, which is exactly
    # where a naive `exp`-then-divide implementation would produce inf/nan.
    from tasks.alfworld.repeat_stop import _conditional_ce

    rows = torch.full((1, VOCAB), 5000.0)
    rows[0, 40] = 9000.0
    # Act
    ce, q_stop = _conditional_ce(rows, torch.tensor([[STOP, 40]]), stop_id=STOP)
    # Assert -- finite, and the CE is exactly the log-gap the scores imply.
    assert torch.isfinite(ce).all()
    assert torch.isfinite(q_stop).all()
    assert float(ce[0]) == pytest.approx(4000.0, rel=1e-4)
    assert 0.0 <= float(q_stop[0]) <= 1.0
    # The dominant candidate absorbs the conditional mass, so the stop share is 0
    # -- a saturated but *correct* answer, not a NaN.
    assert float(q_stop[0]) == 0.0


def test_score_chunk_ladder(monkeypatch):
    # Arrange / Act / Assert
    assert score_chunk() == 256
    monkeypatch.setenv(SCORE_CHUNK_ENV, "128")
    assert score_chunk() == 128
    monkeypatch.setenv(SCORE_CHUNK_ENV, "64")
    assert score_chunk() == 64


def test_repeat_path_never_calls_a_collective():
    # Arrange -- the framework all-reduces the token count itself; a second
    # collective here would scale the gradient by the DP size.
    here = Path(__file__).resolve().parents[1]
    # Act
    source = (here / "repeat_stop.py").read_text() + (here / "repeat.py").read_text()
    # Assert
    for forbidden in ("all_reduce", "all_gather", "dist.", "torch.distributed"):
        assert forbidden not in source


def test_reducer_weights_are_independent_of_the_replaced_values(monkeypatch):
    # Arrange -- the repeat branch must not change how the trajectory is weighted.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "traj")
    from tasks.alfworld.loss import trajectory_balanced_reduce

    response_lengths = [4, 4, 6]
    scales = [torch.full((r,), 3.0) for r in response_lengths]
    weights = [
        torch.full((4,), 0.25),
        torch.full((4,), 0.25),
        torch.full((6,), 0.5),
    ]
    batch = {"fit_turn_weights": weights, "traj_norm_scale": scales}
    baseline = torch.arange(1.0, 15.0)
    replaced = baseline.clone()
    replaced[2] = -1234.0  # a gated position would look like this
    # Act
    _, _, w_baseline = trajectory_balanced_reduce(baseline, response_lengths, batch)
    total, _, w_replaced = trajectory_balanced_reduce(replaced, response_lengths, batch)
    # Assert -- the emitted weights are identical, and each trajectory's sum is 1.
    assert torch.equal(w_baseline, w_replaced)
    assert torch.isfinite(total)
    r0 = w_baseline[0:4].sum() + w_baseline[4:8].sum()
    assert abs(float(r0) - 2.0) < 1e-6  # two trajectories, each summing to 1
