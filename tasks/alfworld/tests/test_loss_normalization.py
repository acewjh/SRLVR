"""The v2 loss must return a SUM over tokens, not a mean.

Under `--calculate-per-token-loss` slime hands `sum_of_token` in as the
`sum_of_sample_mean` argument (`cp_utils.py:120`), so a custom loss is contracted
to return a sum which the framework divides -- once, later, in
`finalize_model_grads` -- by its own all-reduced `num_tokens`. Returning a mean
instead makes the framework divide a second time: the gradient comes out smaller
by the microbatch's token count, and -- worse than a constant offset -- that
factor tracks response length, coupling the effective learning rate to how long
the policy's responses happen to be.

`OPSD_V2_LOSS_NORM=legacy` (default) keeps the shipped behaviour so historical
runs stay reproducible; `=sum` multiplies by exactly this rank's token count.
These tests pin both, pin the exact factor, and pin the one thing that would
re-introduce the bug in a new shape: the multiplier must NOT run its own
collective, because the framework all-reduces `num_tokens` itself.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from tasks.alfworld.loss import (  # noqa: E402
    apply_framework_normalization,
    norm_mode,
)


def _masks(lengths):
    return [torch.ones(n, dtype=torch.float32) for n in lengths]


def test_default_mode_is_legacy_and_leaves_the_loss_alone(monkeypatch):
    # Arrange
    monkeypatch.delenv("OPSD_V2_LOSS_NORM", raising=False)
    loss = torch.tensor(-0.375)

    # Act
    out, mult, n_tokens = apply_framework_normalization(loss, _masks([10, 20]))

    # Assert
    assert norm_mode() == "legacy"
    assert mult == 1.0
    assert n_tokens == 0.0
    assert torch.equal(out, loss)


def test_sum_mode_scales_by_exactly_this_ranks_token_count(monkeypatch):
    # Arrange: 10 + 20 tokens, local. The framework will divide by the same
    # quantity once it has summed this over microbatches and the dp_cp group.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "sum")
    loss = torch.tensor(-0.375)
    lengths = [10, 20]

    # Act
    out, mult, n_tokens = apply_framework_normalization(loss, _masks(lengths))

    # Assert
    assert n_tokens == float(sum(lengths))
    assert mult == float(sum(lengths))
    assert torch.allclose(out, loss * float(sum(lengths)))


def test_the_multiplier_does_not_run_its_own_collective(monkeypatch):
    # Arrange: the framework accumulates `num_tokens` across microbatches and
    # all-reduces it over dp_cp itself (finalize_model_grads.py:492). A second
    # all-reduce here would multiply the divisor by the DP size -- a silent
    # DP-dependent scale error, which is exactly how a scale bug hides.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "sum")

    def _explode(*args, **kwargs):
        raise AssertionError("apply_framework_normalization must not all-reduce")

    monkeypatch.setattr(torch.distributed, "all_reduce", _explode)
    loss = torch.tensor(1.0)

    # Act
    out, _, n_tokens = apply_framework_normalization(loss, _masks([7, 3]))

    # Assert
    assert n_tokens == 10.0
    assert torch.allclose(out, torch.tensor(10.0))


def test_the_scaling_is_linear_in_the_token_count(monkeypatch):
    # Arrange: doubling every response must double the returned loss, because that
    # is exactly the factor the framework will divide back out. If this ever stops
    # holding, the effective learning rate has become length-dependent again.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "sum")
    loss = torch.tensor(1.0)

    # Act
    small, _, _ = apply_framework_normalization(loss, _masks([5, 5]))
    big, _, _ = apply_framework_normalization(loss, _masks([10, 10]))

    # Assert
    assert torch.allclose(big, small * 2.0)


def test_empty_masks_do_not_scale(monkeypatch):
    # Arrange: clamp_min(..., 1) means an all-zero mask still counts as 1 token, so
    # n_tokens can never reach 0 and the guard is about degenerate input only.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "sum")
    loss = torch.tensor(2.0)

    # Act
    out, mult, n_tokens = apply_framework_normalization(loss, [torch.zeros(0)])

    # Assert
    assert n_tokens == 1.0
    assert torch.allclose(out, loss)


def test_unknown_mode_falls_back_to_legacy(monkeypatch):
    # Arrange: a typo must not silently change the gradient scale.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "summ")
    loss = torch.tensor(3.0)

    # Act
    out, mult, _ = apply_framework_normalization(loss, _masks([7]))

    # Assert
    assert mult == 1.0
    assert torch.equal(out, loss)


def test_a_retired_mode_name_is_not_silently_accepted(monkeypatch):
    # Arrange: `global` was the first, WRONG attempt at this fix -- it all-reduced
    # over DP and stopped one factor short, so the arm it ran on is not comparable
    # to a `legacy` arm either. Anyone who still has it in a launcher must get the
    # legacy path (and therefore a visibly unfixed run), never a half-fix.
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "global")
    loss = torch.tensor(3.0)

    # Act
    out, mult, _ = apply_framework_normalization(loss, _masks([7, 11]))

    # Assert: byte-for-byte the legacy path, so the run is visibly unfixed.
    monkeypatch.delenv("OPSD_V2_LOSS_NORM")
    legacy, legacy_mult, _ = apply_framework_normalization(loss, _masks([7, 11]))
    assert mult == legacy_mult == 1.0
    assert torch.equal(out, legacy)


def test_the_fix_is_what_makes_the_two_arms_comparable(monkeypatch):
    # Arrange: same per-token losses, different response lengths. The SUMMED and the
    # MEAN-reduced quantities must differ by exactly the token count -- that gap is
    # the bug, and it is why two arms whose lengths drift apart (one rambling, one
    # silent) stop being comparable.
    loss = torch.tensor(-0.25)
    lengths = [100, 200, 300]

    # Act
    legacy, _, _ = apply_framework_normalization(loss, _masks(lengths))
    monkeypatch.setenv("OPSD_V2_LOSS_NORM", "sum")
    fixed, _, n_tokens = apply_framework_normalization(loss, _masks(lengths))

    # Assert
    assert n_tokens == float(sum(lengths))
    assert torch.allclose(fixed, legacy * n_tokens)
    assert not torch.isclose(fixed, legacy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
