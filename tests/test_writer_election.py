"""Writer election for the Reward-CV state files.

Regression guard for a silent PP-only defect found on the 8B run (2026-09-30):
`_is_writer()` used `dist.get_rank() == 0`, but the state machine runs inside the
custom loss, which Megatron evaluates on the LAST PIPELINE STAGE only. At PP=1
those coincide; at PP=4 they do not, and the rank-0 process simply never reaches
the write -- leaving an empty trajectory JSONL and no cv_vnext_state.json, with
healthy-looking training (grad_norm 0.0069, no OOM) hiding it.

The observable contract asserted here: **exactly one rank is the writer**, and it
is a rank that actually runs the loss (last pipeline stage).

Run:  python -m pytest tests/test_writer_election.py -q
"""

import pathlib
import sys

import pytest

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from opsd_core import writer as writer_election  # noqa: E402

# The implementation lives in opsd_core/writer.py; the
# old `math_opsd.writer_election` shim was retired 2026-10-15. The two I/O seams
# must be patched where `is_loss_writer` actually resolves them — on the module
# that DEFINES it.
from opsd_core import writer as _writer_impl  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_seams(monkeypatch):
    """Tests replace the two I/O seams; nothing else about the module changes."""
    yield


def _fake_position(monkeypatch, position):
    monkeypatch.setattr(_writer_impl, "_pipeline_position", lambda: position)


def _fake_rank(monkeypatch, rank):
    monkeypatch.setattr(_writer_impl, "_global_rank", lambda: rank)


def test_no_distributed_context_is_a_single_writer(monkeypatch) -> None:
    """Offline tooling and unit tests must still write (no cohort to elect from)."""
    _fake_position(monkeypatch, None)
    _fake_rank(monkeypatch, None)
    assert writer_election.is_loss_writer() is True


def test_pp1_matches_the_historical_rank0_rule(monkeypatch) -> None:
    """PP1 is the configuration every 1.7B/4B run used; behaviour must not move."""
    for tp, dp, expected in [(0, 0, True), (0, 1, False), (1, 0, False)]:
        _fake_position(monkeypatch, (0, 1, tp, dp))
        assert writer_election.is_loss_writer() is expected, (tp, dp)


@pytest.mark.parametrize(
    "pp_size,tp_size,dp_size",
    [(1, 1, 1), (1, 1, 4), (4, 1, 1), (2, 1, 2), (2, 2, 1), (4, 2, 2)],
)
def test_exactly_one_writer_per_topology(monkeypatch, pp_size, tp_size, dp_size) -> None:
    """The point of the election: one writer, on every geometry the harness runs.

    Ranks are enumerated in megatron's layout, rank = tp + tp_size*(pp + pp_size*dp).
    """
    writers = []
    for dp in range(dp_size):
        for pp in range(pp_size):
            for tp in range(tp_size):
                rank = tp + tp_size * (pp + pp_size * dp)
                _fake_position(monkeypatch, (pp, pp_size, tp, dp))
                _fake_rank(monkeypatch, rank)
                if writer_election.is_loss_writer():
                    writers.append((pp, tp, dp))

    assert len(writers) == 1, f"tp{tp_size} pp{pp_size} dp{dp_size}: writers={writers}"
    pp, tp, dp = writers[0]
    assert pp == pp_size - 1, (
        f"tp{tp_size} pp{pp_size} dp{dp_size}: elected pp_rank={pp}, but only the "
        "last pipeline stage runs the custom loss, so its write is never reached"
    )
    assert (tp, dp) == (0, 0)


def test_without_megatron_it_falls_back_to_global_rank0(monkeypatch) -> None:
    """A dist-only context (no megatron parallel state) keeps the old rule."""
    _fake_position(monkeypatch, None)
    _fake_rank(monkeypatch, 0)
    assert writer_election.is_loss_writer() is True
    _fake_rank(monkeypatch, 3)
    assert writer_election.is_loss_writer() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
