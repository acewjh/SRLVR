"""Calibration-probe acceleration: planning, exact moments, and Gloo wiring.

The CPU test runs real autograd on a toy model and compares the per-position
C/D/W/history produced through the new helpers with the full-vector double
reference. The distributed test uses real Gloo TP/DP groups to check that
replicated versus sharded parameters are counted exactly once, that records
from different DP groups are never mixed, and that idle/uneven assignments do
not deadlock.
"""
from __future__ import annotations

import json
import math
from datetime import timedelta
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from methods import srlvr_hsi as hsi
from methods.srlvr_hsi import Settings, next_history


@pytest.mark.parametrize("raises", [False, True])
def test_probe_numerics_restores_determinism_and_warning_mode(monkeypatch, raises):
    monkeypatch.setenv("SRLVR_PROBE_DETERMINISTIC", "1")
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with pytest.raises(RuntimeError) if raises else nullcontext():
            with hsi.probe_numerics():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                if raises:
                    raise RuntimeError("probe failure")
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)


def test_probe_numerics_opt_out_and_invalid_value(monkeypatch):
    monkeypatch.setenv("SRLVR_PROBE_DETERMINISTIC", "0")
    before = torch.are_deterministic_algorithms_enabled()
    with hsi.probe_numerics():
        assert torch.are_deterministic_algorithms_enabled() == before
    monkeypatch.setenv("SRLVR_PROBE_DETERMINISTIC", "yes")
    with pytest.raises(ValueError, match="SRLVR_PROBE_DETERMINISTIC"):
        with hsi.probe_numerics():
            pass


def test_full_probe_vector_error_keeps_small_component_differences():
    reference = [torch.tensor([1., 2., 0.]), torch.tensor([3.])]
    value = [torch.tensor([1., 2., .01]), torch.tensor([3.])]
    report = hsi._probe_vector_error(reference, value)
    assert report["different"] == 1
    assert report["max_abs"] == pytest.approx(.01)
    assert report["relative_l2"] == pytest.approx(.01 / math.sqrt(14))
    assert hsi._probe_vector_error(None, None)["relative_l2"] == 0.
    with pytest.raises(RuntimeError, match="zero-gradient"):
        hsi._probe_vector_error(None, value)


# ---------------------------------------------------------------------------
# planner / row helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("calibration,dp_size,expected", [
    (4, 4, [[0], [1], [2], [3]]),
    (4, 2, [[0, 2], [1, 3]]),
    (4, 1, [[0, 1, 2, 3]]),
    (3, 4, [[0], [1], [2], []]),
    (5, 3, [[0, 3], [1, 4], [2]]),
])
def test_assign_probe_positions_covers_each_position_once(calibration, dp_size, expected):
    got = [hsi.assign_probe_positions(calibration, rank, dp_size) for rank in range(dp_size)]
    assert got == expected
    flat = [position for part in got for position in part]
    assert sorted(flat) == list(range(calibration))


def test_assign_probe_positions_rejects_invalid_topology():
    with pytest.raises(ValueError, match="invalid calibration/DP"):
        hsi.assign_probe_positions(0, 0, 1)
    with pytest.raises(ValueError, match="invalid calibration/DP"):
        hsi.assign_probe_positions(4, 4, 4)


def test_pad_and_select_never_drop_real_turns_and_keep_zero_y():
    rows = [dict(tokens=[1, 2], mask=[1, 1], old=[0., 0.], meta=dict(B=.3)),
            dict(tokens=[3, 4], mask=[0, 0], old=[0., 0.], meta=dict(B=.1)),
            dict(tokens=[5, 6], mask=[1, 1], old=[0., 0.], meta=dict(B=0.))]
    padded = hsi._pad_probe_rows(rows, 2)
    assert len(padded) == 4 and len(rows) == 3
    assert padded[3]["mask"] == [0, 0]
    assert [row["tokens"] for row in hsi._select_probe_rows(rows, "Z")] == [[1, 2], [5, 6]]
    assert [row["tokens"] for row in hsi._select_probe_rows(rows, "Y")] == [[1, 2]]
    zero_y = [dict(rows[0], meta=dict(B=0.))]
    assert hsi._select_probe_rows(zero_y, "Y") == []


@pytest.mark.parametrize("mbs,chunk,expected", [(1, 5, 5), (2, 5, 4), (2, 1, 2), (4, 7, 4)])
def test_probe_chunk_size_respects_microbatch(monkeypatch, mbs, chunk, expected):
    monkeypatch.setenv("SRLVR_PROBE_CHUNK", str(chunk))
    assert hsi._probe_chunk_size(SimpleNamespace(micro_batch_size=mbs)) == expected


def test_finalize_probe_records_is_order_independent_and_fail_closed():
    cfg = Settings(variant="v2", gamma=.99, calibration=2)
    records = [dict(position=1, reward=0., zz=4., zy=.2, yy=.1),
               dict(position=0, reward=1., zz=9., zy=.4, yy=.2)]
    stats = hsi._finalize_probe_records(records, cfg, eff_b=.5, eff_s=2.)
    assert stats["positions"] == [0, 1]
    assert stats["c_hat"] == pytest.approx(.3)
    assert stats["d_hat"] == pytest.approx(.15)
    assert stats["records"][0][1] == pytest.approx(9. * (2. / .5) ** 2)
    with pytest.raises(RuntimeError, match="exactly once"):
        hsi._finalize_probe_records(records + records[:1], cfg, .5, 2.)
    with pytest.raises(RuntimeError, match="exactly once"):
        hsi._finalize_probe_records(records[:1], cfg, .5, 2.)


def _synthetic_run(label, isolated, c_hat, d_hat, w0, w1, wall):
    return dict(label=label, isolated=isolated, wall=wall,
                stats=dict(c_hat=c_hat, d_hat=d_hat,
                           records=[[1., w0, .5], [0., w1, .5]]))


def test_control_compare_uses_same_path_noise_floor(monkeypatch):
    cfg = Settings(variant="v2", gamma=.99, calibration=2)
    state = dict(b=.5, s=1., history=[], lambda_raw=.2)
    # Parallel sits inside the serial/parallel noise spread.
    noisy = [
        _synthetic_run("serial1", False, .1000, .2000, 3.0, 5.0, 10.),
        _synthetic_run("parallel1", True, .1002, .2003, 3.006, 5.01, 4.),
        _synthetic_run("serial2", False, .0998, .1997, 2.994, 4.99, 10.),
        _synthetic_run("parallel2", True, .1001, .1999, 3.001, 5.005, 4.),
    ]
    report = hsi._probe_control_compare(noisy, cfg, state, .5, 1., 3)
    assert report["passed"], report["failures"]
    assert report["timing"]["speedup_mean"] == pytest.approx(2.5)
    # A systematic offset larger than the same-path spread fails.
    biased = [
        _synthetic_run("serial1", False, .10, .20, 3.0, 5.0, 10.),
        _synthetic_run("parallel1", True, .13, .20, 3.0, 5.0, 4.),
        _synthetic_run("serial2", False, .10, .20, 3.0, 5.0, 10.),
        _synthetic_run("parallel2", True, .13, .20, 3.0, 5.0, 4.),
    ]
    report = hsi._probe_control_compare(biased, cfg, state, .5, 1., 3)
    assert not report["passed"]
    assert any(failure.startswith("C_hat") for failure in report["failures"])


def test_probe_mode_dispatcher_fails_closed(monkeypatch):
    safe = SimpleNamespace(use_distributed_optimizer=True, overlap_grad_reduce=False,
                           num_distributed_optimizer_instances=1)
    unsafe = SimpleNamespace(use_distributed_optimizer=True, overlap_grad_reduce=True,
                             num_distributed_optimizer_instances=1)
    monkeypatch.setenv("SRLVR_PROBE_MODE", "auto")
    assert hsi.resolve_probe_mode(safe) == "parallel"
    assert hsi.resolve_probe_mode(unsafe) == "serial"
    monkeypatch.setenv("SRLVR_PROBE_MODE", "parallel")
    with pytest.raises(RuntimeError, match="overlap_grad_reduce"):
        hsi.resolve_probe_mode(unsafe)
    monkeypatch.setenv("SRLVR_PROBE_MODE", "serial")
    assert hsi.resolve_probe_mode(unsafe) == "serial"
    monkeypatch.setenv("SRLVR_PROBE_MODE", "bogus")
    with pytest.raises(ValueError, match="SRLVR_PROBE_MODE"):
        hsi.resolve_probe_mode(safe)


# ---------------------------------------------------------------------------
# full-vector CPU equivalence with real autograd
# ---------------------------------------------------------------------------

DIM = 13


class FakeParam:
    def __init__(self):
        self.main_grad = None


def _chunks(items, size):
    return [items[start:start + size] for start in range(0, len(items), size)]


def _probe_vector(seed, position, which, turns, chunk_size=2):
    """Actual autograd gradient accumulated through the production helper."""
    generator = torch.Generator().manual_seed(seed * 1000 + position * 10 + (which == "Y"))
    weight = torch.randn(DIM, generator=generator, dtype=torch.float32, requires_grad=True)
    fake = FakeParam()
    entries = [(fake, 1.0)]
    total = None
    for chunk in _chunks(turns, chunk_size):
        loss = torch.zeros((), dtype=torch.float32)
        for x, advantage, mask, length in chunk:
            if not mask:
                continue
            loss = loss - advantage * (weight * x).sum() / length
        weight.grad = None
        loss.backward()
        fake.main_grad = weight.grad.detach().clone()
        total = hsi._accumulate_probe_gradient(
            None, entries, total, reconstruct_dp=False,
            profile={} if which == "Z" else None)
    return total[0]


def _double_vector(turns):
    value = torch.zeros(DIM, dtype=torch.float64)
    for x, advantage, mask, length in turns:
        if mask:
            value += float(advantage) * x.double() / length
    return value


def _toy_positions():
    generator = torch.Generator().manual_seed(7)
    positions = []
    for position, (turns, reward) in enumerate([
        (7, 1.), (3, 0.), (5, 1.), (4, 0.),
    ]):
        rows = []
        for turn in range(turns):
            x = torch.randn(DIM, generator=generator, dtype=torch.float64)
            b = 0. if position == 3 else float(torch.randn((), generator=generator))
            rows.append(dict(x=x, B=b, reward=reward, turn=turn, mask=turn != 1,
                             length=float(turns)))
        positions.append(dict(rows=rows, reward=reward))
    return positions


def test_parallel_records_match_full_vector_reference(monkeypatch):
    monkeypatch.setenv("SRLVR_PROBE_ACCUM", "cpu")
    cfg = Settings(variant="v2", gamma=.99, calibration=4)
    eff_b, eff_s = .5, .7
    positions = _toy_positions()

    def trajectory_turns(position, which):
        rows = positions[position]["rows"]
        selected = [row for row in rows if row["mask"] and (which != "Y" or row["B"] != 0.)]
        return [(row["x"], hsi.probe_advantage(positions[position]["reward"], row["B"],
                                               eff_b, eff_s, which), row["mask"], row["length"])
                for row in selected]

    reference_z = [_double_vector(trajectory_turns(p, "Z")) for p in range(4)]
    reference_y = [_double_vector(trajectory_turns(p, "Y")) for p in range(4)]
    for position in range(4):
        if positions[position]["rows"][0]["B"] == 0.:
            reference_y[position] = torch.zeros(DIM, dtype=torch.float64)

    expected_c = sum(torch.dot(z, y).item() for z, y in zip(reference_z, reference_y)) / 4
    expected_d = sum(torch.dot(y, y).item() for y in reference_y) / 4
    expected_w = [torch.dot(z, z).item() * (eff_s / (positions[p]["reward"] - eff_b)) ** 2
                  for p, z in enumerate(reference_z)]

    for dp_size in (1, 2, 4, 6):
        records = []
        for rank in range(dp_size):
            for position in hsi.assign_probe_positions(4, rank, dp_size):
                z = _probe_vector(11, position, "Z", trajectory_turns(position, "Z"))
                y_turns = trajectory_turns(position, "Y")
                y = _probe_vector(11, position, "Y", y_turns) if y_turns else None
                zz, zy, yy = hsi._probe_moments([z], [y] if y is not None else None,
                                                torch.tensor([1.]), 1, None)
                records.append(dict(position=position, reward=positions[position]["reward"],
                                    zz=zz, zy=zy, yy=yy))
        stats = hsi._finalize_probe_records(records, cfg, eff_b, eff_s)
        assert stats["c_hat"] == pytest.approx(expected_c, rel=2e-5, abs=2e-7), dp_size
        assert stats["d_hat"] == pytest.approx(expected_d, rel=2e-5, abs=2e-7), dp_size
        for position, got in enumerate(stats["records"]):
            assert got[1] == pytest.approx(expected_w[position], rel=2e-5), dp_size

        # History / next lambda must be identical to the full reference records.
        reference_records = [[positions[p]["reward"], expected_w[p], eff_b] for p in range(4)]
        state = dict(b=.5, s=1., history=[], lambda_raw=.2)
        expected_next = next_history(state, reference_records, cfg, expected_c, expected_d, 3)
        got_next = next_history(state, stats["records"], cfg, stats["c_hat"], stats["d_hat"], 3)
        assert got_next["b"] == pytest.approx(expected_next["b"], rel=2e-5)
        assert got_next["s"] == pytest.approx(expected_next["s"], rel=2e-5)
        assert got_next["lambda_next"] == pytest.approx(expected_next["lambda_next"], rel=2e-5)


def test_zero_y_degrades_to_zero_moments(monkeypatch):
    monkeypatch.setenv("SRLVR_PROBE_ACCUM", "device")
    turns = [(torch.randn(DIM, dtype=torch.float64), .4, True, 3.)]
    z = _probe_vector(3, 0, "Z", turns)
    zz, zy, yy = hsi._probe_moments([z], None, torch.tensor([1.], dtype=torch.float32), 1, None)
    assert zz > 0 and zy == 0 and yy == 0


# ---------------------------------------------------------------------------
# Gloo: real TP scaling, DP record gathering, idle and uneven ranks
# ---------------------------------------------------------------------------

def _logical_vectors(position, which):
    """Full logical parameter vectors (replica + concatenated TP shard)."""
    offset = 1. if which == "Z" else 2.
    replica = torch.arange(6, dtype=torch.float64) + 1 + 10 * position + offset
    shard0 = torch.arange(5, dtype=torch.float64) + 2 + 7 * position + offset + 100.
    shard1 = torch.arange(5, dtype=torch.float64) + 2 + 7 * position + offset + 200.
    return replica, torch.cat([shard0, shard1])


def _rank_vectors(position, which, tp_rank):
    """This TP rank's replica copy and its own shard half."""
    replica, shard = _logical_vectors(position, which)
    return replica.float(), shard[tp_rank * 5:(tp_rank + 1) * 5].float()


def _expected_moments(position, zero_y=False):
    zr, zs = _logical_vectors(position, "Z")
    if zero_y:
        return float((zr ** 2).sum() + (zs ** 2).sum()), 0., 0.
    yr, ys = _logical_vectors(position, "Y")
    return (float((zr ** 2).sum() + (zs ** 2).sum()),
            float((zr * yr).sum() + (zs * ys).sum()),
            float((yr ** 2).sum() + (ys ** 2).sum()))


def _gloo_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=60))
    try:
        tp_rank, dp_rank = rank % 2, rank // 2
        # All ranks must issue the same new_group sequence; only membership and
        # the locally selected handle differ.
        tp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        all_group = dist.new_group([0, 1, 2, 3])
        tp_group = tp_groups[rank // 2]
        dp_group = dp_groups[rank % 2]
        scales = torch.tensor([1.0 if tp_rank == 0 else 0.0, 1.0])

        # 1) Real TP scaling: replica counted once, both shard halves summed.
        for position in hsi.assign_probe_positions(3, dp_rank, 2):
            z = _rank_vectors(position, "Z", tp_rank)
            y = None if position == 2 else _rank_vectors(position, "Y", tp_rank)
            got = hsi._probe_moments(z, y, scales, 2, tp_group)
            assert got == pytest.approx(_expected_moments(position, zero_y=position == 2),
                                        rel=2e-6), (rank, position, got)

        # 2) DP record gather keeps positions disjoint (never mixes trajectories).
        records = []
        for position in hsi.assign_probe_positions(3, dp_rank, 2):
            z = _rank_vectors(position, "Z", tp_rank)
            y = None if position == 2 else _rank_vectors(position, "Y", tp_rank)
            zz, zy, yy = hsi._probe_moments(z, y, scales, 2, tp_group)
            records.append(dict(position=position, zz=zz, zy=zy, yy=yy,
                                reward=float(position % 2)))
        stats = hsi._finalize_probe_records(
            hsi._gather_probe_records(records, dp_group, 2), Settings(variant="v2", gamma=.99, calibration=3), .5, 1.)
        expected = [_expected_moments(position, zero_y=position == 2) for position in range(3)]
        assert stats["positions"] == [0, 1, 2]
        assert stats["c_hat"] == pytest.approx(sum(e[1] for e in expected) / 3, rel=2e-6)
        assert stats["d_hat"] == pytest.approx(sum(e[2] for e in expected) / 3, rel=2e-6)
        assert stats["records"][2][1] > 0  # W from Z survives a zero-Y position

        # 3) DP > calibration and uneven assignment: three ranks carry work, one
        #    is idle, yet every rank joins the same record gather.
        own = []
        for position in hsi.assign_probe_positions(3, rank, 4):
            zr, zs = _logical_vectors(position, "Z")
            yr, ys = (None, None) if position == 1 else _logical_vectors(position, "Y")
            z = [zr.float(), zs.float()]
            y = None if yr is None else [yr.float(), ys.float()]
            zz, zy, yy = hsi._probe_moments(z, y, torch.tensor([1., 1.]), 1, None)
            own.append(dict(position=position, zz=zz, zy=zy, yy=yy,
                            reward=float(position % 2)))
        stats_all = hsi._finalize_probe_records(
            hsi._gather_probe_records(own, all_group, 4), Settings(variant="v2", gamma=.99, calibration=3), .5, 1.)
        assert stats_all["positions"] == [0, 1, 2]
        assert stats_all["d_hat"] == pytest.approx(
            sum(_expected_moments(position, zero_y=position == 1)[2]
                for position in range(3)) / 3, rel=2e-6)
    finally:
        dist.destroy_process_group()


def test_gloo_tp_scaling_and_dp_record_gather(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker, args=(f"file://{tmp_path}/gloo",),
                                nprocs=4, join=True)
