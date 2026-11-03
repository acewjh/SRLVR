"""Elect the ONE rank that writes the shared per-run state files.

Moved verbatim from ``math_opsd/writer_election.py`` (same contract; that module
now re-exports from here).

The per-run state machines (lagged baseline ``b``, calibrated coefficient
``lambda``, EMA accumulators) keep statistics that do not live in the distributed
checkpoint. They persist them into the run directory, and they do it from inside
the custom loss.

That location is what makes the election non-obvious. Megatron computes the loss
on the **last pipeline stage only** (``core/pipeline_parallel/schedules.py``,
``forward_step`` calls ``loss_func`` under ``is_pipeline_last_stage()``), so the
state machine exists on the last-stage ranks and nowhere else. Electing
``dist.get_rank() == 0`` is correct only while PP == 1, where the last stage and
global rank 0 happen to be the same process.

Under PP > 1 they are not, and the failure is silent: the rank-0 process never
reaches ``_log_step``, so it never writes, and the run produces an empty
trajectory JSONL and no ``*_state.json``. Observed on the 8B PP4 run (2026-09-30)
-- the training itself was healthy, which is exactly why it went unnoticed until
the lambda evidence was looked for. The missing state file also breaks resume.

So the writer is: the last pipeline stage, tensor-parallel rank 0, data-parallel
rank 0 of its DP group -- one rank, and a rank that actually runs the loss.

Importable without megatron and safe to call outside a distributed run, so the
election can be unit-tested on CPU (tests/test_writer_election.py).
"""

from __future__ import annotations


def _pipeline_position() -> tuple[int, int, int, int] | None:
    """``(pp_rank, pp_size, tp_rank, dp_rank)``, or None if not in a megatron run."""
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return None
        return (
            int(mpu.get_pipeline_model_parallel_rank()),
            int(mpu.get_pipeline_model_parallel_world_size()),
            int(mpu.get_tensor_model_parallel_rank()),
            int(mpu.get_data_parallel_rank()),
        )
    except Exception:  # noqa: BLE001 - not a megatron context; fall back below
        return None


def _global_rank() -> int | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - single-process context
        return None
    return None


def is_loss_writer() -> bool:
    """True on exactly one rank: the one whose custom loss reaches this code.

    Outside a megatron context this degrades to the historical rule (global rank
    0, or True when there is no distributed run at all), which is what the
    single-process tests and offline tooling rely on.
    """
    position = _pipeline_position()
    if position is not None:
        pp_rank, pp_size, tp_rank, dp_rank = position
        return pp_rank == pp_size - 1 and tp_rank == 0 and dp_rank == 0

    rank = _global_rank()
    return True if rank is None else rank == 0
