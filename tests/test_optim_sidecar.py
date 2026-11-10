"""CPU-only gates for the LoRA optimizer-state sidecar (A0).

The claim that matters is not "the file round-trips" but "a resumed arm is the
same experiment as a continuous one". G2 below is that claim, stated as an
equality between two two-step trajectories.

Run: python -m pytest tests/test_optim_sidecar.py -q
"""

import os
import ast
import sys
from pathlib import Path

import pytest
import torch

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from methods import optim_sidecar as sc  # noqa: E402


def test_training_initialization_propagates_missing_sidecar_error(monkeypatch):
    source = ast.parse((PROJ / 'frameworks/slime-ascend/slime/backends/megatron_utils/model.py').read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef)
              and n.name == 'initialize_model_and_optimizer')
    from types import SimpleNamespace
    namespace = dict(Namespace=object, DDP=object, MegatronOptimizer=object,
                    OptimizerParamScheduler=object, RolloutBatch=dict,
                    torch=SimpleNamespace(version=SimpleNamespace(hip=False)),
                    setup_model_and_optimizer=lambda args, role: ([SimpleNamespace()], object(), object()),
                    clear_memory=lambda: None, load_checkpoint=lambda *a, **kw: (4,None))
    def missing(*args):
        raise sc.SidecarError('no sidecar')
    monkeypatch.setattr(sc, 'restore_if_enabled', missing)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<model-initialization>', 'exec'), namespace)
    with pytest.raises(sc.SidecarError, match='no sidecar'):
        namespace['initialize_model_and_optimizer'](SimpleNamespace(load='saved-run'))


class FakeMegatronOptimizer:
    """Mirrors the only shape `optim_sidecar` depends on.

    Real Megatron wraps the torch optimizer as `.optimizer` and exposes
    `param_groups` on itself; the DistributedOptimizer additionally replaces the
    group params with flattened fp32 clones, which is reproduced here by
    registering flat clones so the positional keying is exercised honestly.
    """

    def __init__(self, params, flatten=True, lr=1e-3):
        self._params = list(params)
        flat = [p.detach().clone().float().view(-1).requires_grad_(True)
                if flatten else p for p in self._params]
        self.param_groups = [{"params": flat, "lr": lr}]
        self.optimizer = torch.optim.AdamW(self.param_groups, lr=lr)

    @property
    def live_params(self):
        """The tensors Adam actually updates (flat fp32 clones, as in Megatron)."""
        return self.param_groups[0]["params"]


def _step(opt, grad_value=1.0):
    """One optimizer step. Gradients go on the optimizer's OWN params -- setting
    them on the originals would leave Adam untouched and silently empty."""
    opt.optimizer.zero_grad()
    for p in opt.live_params:
        p.grad = torch.full_like(p, grad_value)
    opt.optimizer.step()


def test_g1_round_trip_restores_adam_moments_exactly(tmp_path):
    torch.manual_seed(0)
    src = FakeMegatronOptimizer([torch.randn(6, 4), torch.randn(3)])
    for _ in range(3):
        _step(src)

    payload = sc.capture(src)
    dst = FakeMegatronOptimizer([torch.randn(6, 4), torch.randn(3)])
    assert sc.restore(dst, payload) == len(dst.param_groups[0]["params"])

    src_state = src.optimizer.state
    dst_state = dst.optimizer.state
    assert len(src_state) == len(dst_state)
    for sp, dp in zip(src.param_groups[0]["params"], dst.param_groups[0]["params"]):
        torch.testing.assert_close(src_state[sp]["exp_avg"], dst_state[dp]["exp_avg"])
        torch.testing.assert_close(src_state[sp]["exp_avg_sq"], dst_state[dp]["exp_avg_sq"])
        assert sc._step_to_state(src_state[sp]["step"]) == sc._step_to_state(dst_state[dp]["step"])


def test_g2_resumed_two_step_trajectory_equals_continuous(tmp_path):
    """A1's core requirement: a resume with the sidecar is not a new experiment."""
    torch.manual_seed(1)
    init = [torch.randn(5, 3), torch.randn(4)]

    # Continuous: two steps.
    cont_params = [p.detach().clone().requires_grad_(True) for p in init]
    cont = FakeMegatronOptimizer(cont_params)
    _step(cont)
    _step(cont)

    # Resumed: one step + sidecar, then a FRESH optimizer standing on the
    # checkpointed (post-step-1) weights -- which is exactly what megatron
    # restores from `iter_N/` -- and one more step.
    run_params = [p.detach().clone().requires_grad_(True) for p in init]
    run = FakeMegatronOptimizer(run_params)
    _step(run)
    sc.dump(run, tmp_path, iteration=24)
    checkpointed = [p.detach().clone() for p in run.live_params]

    resumed = FakeMegatronOptimizer([w.clone() for w in checkpointed])
    (tmp_path / sc.TRACKER_NAME).write_text("24")
    assert sc.restore_from_dir(resumed, tmp_path) == len(resumed.param_groups[0]["params"])
    _step(resumed)

    for a, b in zip(cont.live_params, resumed.live_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_g3_layout_change_is_refused_not_silently_transposed(tmp_path):
    """The failure this design exists to prevent: a permuted optimizer layout."""
    torch.manual_seed(2)
    src = FakeMegatronOptimizer([torch.randn(6, 4), torch.randn(3)])
    _step(src)
    payload = sc.capture(src)

    different = FakeMegatronOptimizer([torch.randn(3), torch.randn(6, 4)])
    with pytest.raises(sc.SidecarError, match="shapes changed"):
        sc.restore(different, payload)


def test_g4_group_count_change_is_refused():
    src = FakeMegatronOptimizer([torch.randn(4)])
    _step(src)
    payload = sc.capture(src)

    two_groups = FakeMegatronOptimizer([torch.randn(4)])
    two_groups.param_groups.append({"params": [], "lr": 1e-3})
    with pytest.raises(sc.SidecarError, match="group sizes changed"):
        sc.restore(two_groups, payload)


def test_g5_non_finite_moments_are_rejected():
    opt = FakeMegatronOptimizer([torch.randn(4)])
    _step(opt)
    param = opt.param_groups[0]["params"][0]
    opt.optimizer.state[param]["exp_avg"] = torch.full_like(param, float("nan"))
    with pytest.raises(sc.SidecarError, match="non-finite"):
        sc.capture(opt)

    opt2 = FakeMegatronOptimizer([torch.randn(4)])
    _step(opt2)
    payload = sc.capture(opt2)
    payload["groups"][0][0]["exp_avg"] = torch.full((4,), float("inf"))
    with pytest.raises(sc.SidecarError, match="non-finite"):
        sc.restore(FakeMegatronOptimizer([torch.randn(4)]), payload)


def test_g6_unstepped_params_keep_the_layout_aligned():
    opt = FakeMegatronOptimizer([torch.randn(4), torch.randn(2)])
    payload = sc.capture(opt)  # nothing stepped yet
    assert payload["groups"][0] == [None, None]
    dst = FakeMegatronOptimizer([torch.randn(4), torch.randn(2)])
    assert sc.restore(dst, payload) == 0
    assert dst.optimizer.state == {}


def test_g7_step_type_round_trips():
    for as_tensor in (False, True):
        opt = FakeMegatronOptimizer([torch.randn(3)])
        _step(opt)
        p = opt.param_groups[0]["params"][0]
        opt.optimizer.state[p]["step"] = (
            torch.tensor(7.0) if as_tensor else 7
        )
        payload = sc.capture(opt)
        dst = FakeMegatronOptimizer([torch.randn(3)])
        sc.restore(dst, payload)
        dp = dst.param_groups[0]["params"][0]
        assert isinstance(dst.optimizer.state[dp]["step"], torch.Tensor) is as_tensor
        assert sc._step_to_state(dst.optimizer.state[dp]["step"])[1] == 7


def test_g8_gate_off_is_inert(tmp_path):
    opt = FakeMegatronOptimizer([torch.randn(3)])
    _step(opt)
    os.environ.pop(sc.ENV_GATE, None)
    sc.dump_if_enabled(opt, tmp_path, iteration=24)
    assert not sc.sidecar_path(tmp_path, 24).exists()
    assert sc.restore_if_enabled(opt, tmp_path) == 0


def test_g9_gate_on_missing_sidecar_fails_loudly(tmp_path):
    """Silently restarting Adam is the bug; a missing sidecar must raise."""
    os.environ[sc.ENV_GATE] = "1"
    try:
        (tmp_path / sc.TRACKER_NAME).write_text("24")
        opt = FakeMegatronOptimizer([torch.randn(3)])
        with pytest.raises(sc.SidecarError, match="no sidecar"):
            sc.restore_if_enabled(opt, tmp_path)
    finally:
        os.environ.pop(sc.ENV_GATE, None)


def test_g10_dump_is_atomic_and_survives_a_repeat(tmp_path):
    os.environ[sc.ENV_GATE] = "1"
    try:
        opt = FakeMegatronOptimizer([torch.randn(3)])
        _step(opt)
        sc.dump_if_enabled(opt, tmp_path, iteration=24)
        sc.dump_if_enabled(opt, tmp_path, iteration=24)
        path = sc.sidecar_path(tmp_path, 24)
        assert path.is_file()
        assert not path.with_suffix(path.suffix + ".tmp").exists()
    finally:
        os.environ.pop(sc.ENV_GATE, None)


def test_g11_group_level_step_counter_round_trips():
    """TE FusedAdam keeps `step` on the param group, not in `state`. Dropping it
    silently breaks Adam's bias correction on the first resumed step -- the exact
    ~1-lr divergence the NPU gate caught."""
    src = FakeMegatronOptimizer([torch.randn(4)])
    _step(src)
    src.param_groups[0]["step"] = 7

    payload = sc.capture(src)
    assert payload["group_steps"] == [{"step_is_tensor": False, "step_value": 7}]

    dst = FakeMegatronOptimizer([torch.randn(4)])
    dst.param_groups[0]["step"] = 0
    sc.restore(dst, payload)
    assert dst.param_groups[0]["step"] == 7


def test_g12_counter_is_created_on_a_fresh_optimizer():
    """A freshly built FusedAdam has no group 'step' at all (it appears on the
    first step), so the sidecar has to create it -- that is the resume case."""
    src = FakeMegatronOptimizer([torch.randn(4)])
    _step(src)
    src.param_groups[0]["step"] = 3
    payload = sc.capture(src)

    dst = FakeMegatronOptimizer([torch.randn(4)])
    dst.param_groups[0].pop("step", None)
    sc.restore(dst, payload)
    assert dst.param_groups[0]["step"] == 3
