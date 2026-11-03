"""Contract tests for the L0 layer (REFACTOR_PLAN.md §3, §8).

These lock in the two things the refactor must not break:

1. **Identity** — every historical import path still resolves to the SAME
   function object as the opsd_core implementation. If a shim silently
   re-implements instead of re-exporting, this fails.
2. **Numerics** — the extracted helpers are bit-for-bit equal to the pre-refactor
   inline code. The weight test reconstructs the old formula verbatim and
   compares with ``torch.equal``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import opsd_core
from opsd_core import alignment, batch, chunking, teacher, weights, writer

CORE = pathlib.Path(opsd_core.__file__).resolve().parent
# L0 must not import the layers above it. `methods` and `tasks` are the
# post-2026-10-15 names (REFACTOR_PLAN.md §9); the four old package names are
# still listed because they survive as alias shims and an import of one would
# resolve back into `tasks/` / `methods/`.
FORBIDDEN_ROOTS = {
    "methods",
    "tasks",
    "math_opsd",
    "alfworld_hpi",
    "alfworld_hpi_v2",
    "code_grpo",
    "frameworks",
    "slime",
}


# --------------------------------------------------------------------------- #
# 1. layering: opsd_core imports nothing upward
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_opsd_core_has_no_upward_imports():
    offences: list[str] = []
    for path in sorted(CORE.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                if mod.split(".")[0] in FORBIDDEN_ROOTS:
                    offences.append(f"{path.name}:{node.lineno} -> {mod}")
    assert not offences, f"opsd_core must not depend on the layers above it: {offences}"


# --------------------------------------------------------------------------- #
# 2. identity of the re-export shims
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_teacher_primitives_are_reexported_not_recopied():
    pytest.importorskip("torch")
    # `methods.opsd_loss` is the live surface the vendored framework and the task
    # lines import from -- the historical `math_opsd.opsd_loss` alias shim was
    # retired 2026-10-15 (REFACTOR_PLAN.md §9). It must still hand back the very
    # same function objects as `opsd_core.teacher`, not a second copy.
    import methods.opsd_loss as opsd_loss

    assert opsd_loss._get_teacher is teacher.get_teacher
    assert opsd_loss.teacher_forward_completion_logits is teacher.teacher_forward_completion_logits
    assert opsd_loss.slice_response_logits is teacher.slice_response_logits


@pytest.mark.unit
def test_hpi_primitives_are_reexported_not_recopied():
    pytest.importorskip("torch")
    # The `math_opsd.hpi` bundle was retired with the HPI family (2026-10-14); the
    # primitives themselves survive in `opsd_core` and the live callers bind them
    # from there directly. The alias surface that DOES survive is the ALFWorld HSI
    # entry point, and it must stay a THIN alias of the shared estimator -- a copy
    # here is exactly how the math and ALFWorld campaigns would drift apart.
    import methods.hsi2_jsd as jsd
    import methods.hsi_rollout as hsi_rollout
    import tasks.alfworld.hsi2_jsd as afw_jsd

    assert afw_jsd.postprocess is jsd.postprocess
    assert afw_jsd.jsd_token_weights is jsd.jsd_token_weights
    assert afw_jsd.paired_jsd_weights is jsd.paired_jsd_weights
    assert afw_jsd.baseline_state is jsd.baseline_state
    assert afw_jsd.estimator is jsd.estimator
    # and the primitives underneath are opsd_core's, not re-derived per task
    assert hsi_rollout.split_chunks is chunking.split_chunks


@pytest.mark.unit
def test_batch_and_writer_shims():
    # `math_opsd.opsd_batch` was retired with the rest of the alias shims
    # (2026-10-15); `methods.opsd_loss` is where the live callers bind these from,
    # and it must re-export rather than re-implement.
    import methods.opsd_loss as opsd_loss

    assert opsd_loss.build_right_padded_batch is batch.build_right_padded_batch
    assert opsd_loss.assert_completion_alignment is alignment.assert_completion_alignment

    # The writer election has no shim left either: `opsd_core.writer` is the single
    # definition site (its PP-aware contract is pinned in test_writer_election.py).
    assert writer.is_loss_writer.__module__ == "opsd_core.writer"


# --------------------------------------------------------------------------- #
# 3. behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rlen", "k", "expected"),
    [
        (10, 4, [(0, 2), (2, 5), (5, 7), (7, 10)]),
        (3, 4, [(0, 0), (0, 1), (1, 2), (2, 3)]),
        (0, 4, [(0, 0)] * 4),
        (4, 4, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ],
)
def test_split_chunks_partition(rlen, k, expected):
    # Arrange / Act
    chunks = chunking.split_chunks(rlen, k)

    # Assert
    assert chunks == expected
    assert len(chunks) == k


@pytest.mark.unit
def test_normalize_weights_is_bit_identical_to_the_old_inline_formula():
    """The shared normaliser must reproduce the retired HPI weight line exactly.

    The HPI family was retired 2026-10-14, but this four-line tail — mean-1,
    clamp, stopgrad — is what every surviving weighting method relies on, so the
    bit-identity is still the contract. The raw-weight expression is rebuilt here
    verbatim as the oracle.
    """
    # Arrange
    torch.manual_seed(0)
    chunks = chunking.split_chunks(37, 4)
    abs_e = [torch.randn(end - start).abs() for start, end in chunks]
    alpha, scale = 0.5, 1.0

    raw = torch.ones(37, dtype=torch.float32)
    for c, (start, end) in enumerate(chunks):
        if end <= start:
            continue
        raw[start:end] = 1.0 + alpha * torch.tanh(abs_e[c].float() / scale)
    legacy = (raw / raw.mean()).clamp(0.70, 1.30).detach()

    # Act
    new = weights.normalize_weights(raw)

    # Assert
    assert torch.equal(legacy, new)


@pytest.mark.unit
def test_alpha_zero_degrades_to_uniform_weight():
    # Arrange
    chunks = chunking.split_chunks(37, 4)
    abs_e = [torch.randn(end - start).abs() for start, end in chunks]

    # Act
    w = weights.normalize_weights(torch.ones(37))

    # Assert — documented degeneration: alpha == 0 => w == 1 exactly
    assert w.numel() == 37
    assert bool((w == 1.0).all())


@pytest.mark.unit
def test_chunking_helper_matches_abs_e_shape():
    # Arrange / Act / Assert — guards the callers that zip chunks with abs_e
    chunks = chunking.split_chunks(11, 4)
    assert sum(end - start for start, end in chunks) == 11
