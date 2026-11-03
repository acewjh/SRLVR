"""Every key the v2 convert emits must survive slime's per-key metric reduction.

THE OUTAGE THIS TESTS FOR. `megatron_utils/data.py::log_rollout_data` runs
`val = sum(val) / len(val)` on every train_data key that is not in its skip list.
A key whose value is a list-of-lists -- which is exactly what a per-token array
looks like -- raises

    TypeError: unsupported operand type(s) for +: 'int' and 'list'

and it raises at the END of the first rollout, after the whole rollout has been
produced and trained on. On 2026-10-02 that killed the 8B and 4B arms and cost
~9 hours of wall clock before anyone noticed, because the failure lands in the
logging path and the trainer exits rc=0.

This test replays that reduction against the real skip list parsed out of the
framework source, so adding a new per-token channel to the convert without
registering it fails here instead of nine hours into a run.
"""

import os
import re
import sys
from pathlib import Path

import pytest

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from tasks.alfworld import config  # noqa: E402
from tasks.alfworld.train_data import convert_samples_to_train_data  # noqa: E402

DATA_PY = Path(PROJ) / "frameworks/slime-ascend/slime/backends/megatron_utils/data.py"


class _Sample:
    def __init__(self, episode, turn, rlen, advantage):
        self.index = turn
        self.tokens = [1] * (10 + rlen)
        self.response_length = rlen
        self.loss_mask = [1] * rlen
        self.turn_ids = [episode * config.TURN_ID_STRIDE + turn] * rlen
        self.opsd_advantages = [advantage] * rlen
        self.fit_turn_weights = [1.0] * rlen
        self.rollout_log_probs = [0.0] * rlen
        self.reward = 1.0 if advantage > 0 else 0.0
        self.status = None
        self.metadata = {}


class _Args:
    global_batch_size = 4


def _skip_list() -> set[str]:
    """The keys `log_rollout_data` refuses to reduce, parsed from the source.

    Parsed rather than imported because the list is a local literal inside a large
    function; parsing keeps this test honest even if the list is moved.
    """
    source = DATA_PY.read_text()
    # `if key in [ ... ]:` followed by `continue`, inside log_rollout_data.
    blocks = re.findall(r"if key in \[(.*?)\]:\s*\n\s*continue", source, re.DOTALL)
    for block in blocks:
        if "fit_turn_weights" in block:
            return set(re.findall(r'"([a-z0-9_]+)"', block))
    raise AssertionError("could not find the skip list in data.py; the test needs updating")


def _samples():
    return [
        _Sample(0, 0, 40, +0.8),
        _Sample(0, 1, 60, +0.8),
        _Sample(1, 0, 120, -0.2),
        _Sample(1, 1, 120, -0.2),
    ]


def test_replaying_the_metric_reduction_does_not_raise():
    # Arrange: exactly what log_rollout_data does, over the keys v2 owns -- this
    # is the line that took the campaign down.
    train_data = convert_samples_to_train_data(_Args(), _samples())
    skip = _skip_list()

    # Act / Assert
    for key in V2_FLOAT_PER_TOKEN_CHANNELS:
        if key not in train_data or key in skip:
            continue
        sum(train_data[key]) / len(train_data[key])  # must not raise TypeError


#: The per-token float channels THIS harness introduces. Scoped deliberately:
#: `opsd_advantages` and `rollout_log_probs` are also list-of-lists but reach
#: `log_rollout_data` as tensors through a path the v2 harness does not control,
#: so requiring them in the skip list would be asserting something untrue.
#: What v2 must guarantee is that its OWN channels are registered.
V2_FLOAT_PER_TOKEN_CHANNELS = ("fit_turn_weights", "traj_norm_scale")


def test_v2_per_token_channels_are_registered_as_skipped():
    # Arrange
    train_data = convert_samples_to_train_data(_Args(), _samples())
    skip = _skip_list()

    # Act
    emitted = {k for k in V2_FLOAT_PER_TOKEN_CHANNELS if k in train_data}
    unregistered = sorted(emitted - skip)

    # Assert
    assert not unregistered, (
        f"{unregistered} are per-token (list-of-lists) float channels the v2 convert "
        "emits but which are NOT in data.py's skip list; log_rollout_data will do "
        "sum(list_of_lists) and raise TypeError at the end of rollout 0"
    )


def test_an_unregistered_float_channel_really_would_crash():
    # Arrange: prove the hazard is still the hazard, so the test above cannot rot
    # into a tautology if the framework's reduction is ever rewritten.
    train_data = convert_samples_to_train_data(_Args(), _samples())

    # Act / Assert
    with pytest.raises(TypeError):
        sum(train_data["traj_norm_scale"]) / len(train_data["traj_norm_scale"])


def test_the_skip_list_actually_contains_the_key_that_crashed():
    # Arrange / Act
    skip = _skip_list()

    # Assert: pin the specific key, so a future edit to data.py that drops it
    # fails here rather than in a multi-hour run.
    assert "traj_norm_scale" in skip
    assert "fit_turn_weights" in skip


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
