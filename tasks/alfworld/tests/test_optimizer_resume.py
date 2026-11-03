import pytest
import torch
from types import SimpleNamespace
import copy
import os


def test_fresh_adam_checkpoint_template_then_trained_step(monkeypatch):
    from megatron.core.optimizer import distrib_optimizer as module

    monkeypatch.setattr(module, "HAVE_APEX_OR_TE", False)
    wrapped = object.__new__(module.DistributedOptimizer)
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    wrapped.optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    wrapped.grad_scaler = None
    assert wrapped.state_dict()["optimizer"]["param_groups"][0]["step"] == 0
    for _ in range(2):
        wrapped.optimizer.zero_grad()
        parameter.square().sum().backward()
        wrapped.optimizer.step()
    assert wrapped.state_dict()["optimizer"]["param_groups"][0]["step"] == 2


def test_checkpoint_template_still_rejects_inconsistent_adam_steps(monkeypatch):
    from megatron.core.optimizer import distrib_optimizer as module

    monkeypatch.setattr(module, "HAVE_APEX_OR_TE", False)
    wrapped = object.__new__(module.DistributedOptimizer)
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    wrapped.optimizer = torch.optim.AdamW(parameters)
    wrapped.grad_scaler = None
    for step, parameter in enumerate(parameters, start=1):
        wrapped.optimizer.state[parameter]["step"] = torch.tensor(float(step))
    with pytest.raises(AssertionError):
        wrapped.state_dict()


def test_model_space_restore_keeps_adam_scalar_step():
    from megatron.core.optimizer import distrib_optimizer as module

    wrapped = object.__new__(module.DistributedOptimizer)
    parameter = torch.nn.Parameter(torch.ones(2))
    wrapped.optimizer = torch.optim.AdamW([parameter])
    wrapped.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
    wrapped.model_param_group_index_map = {parameter: (0, 0)}
    state = wrapped.optimizer.state[parameter]
    state.update(step=torch.tensor(7.0), exp_avg=torch.zeros(2), exp_avg_sq=torch.zeros(2))
    tensors = {"param": torch.full((2,), 3.0), "exp_avg": torch.full((2,), 4.0),
               "exp_avg_sq": torch.full((2,), 5.0)}
    with torch.no_grad():
        wrapped._set_main_param_and_optimizer_states(parameter, tensors)
    assert state["step"].item() == 7
    torch.testing.assert_close(parameter, tensors["param"])
    torch.testing.assert_close(state["exp_avg"], tensors["exp_avg"])
    torch.testing.assert_close(state["exp_avg_sq"], tensors["exp_avg_sq"])


@pytest.mark.parametrize("loader", ["megatron", "mindspeed"])
@pytest.mark.parametrize("device", ["cpu", "npu:0"] if os.environ.get("ALFWORLD_TEST_NPU") == "1" else ["cpu"])
def test_resumed_adam_counters_and_updates_match_uninterrupted(loader, device, monkeypatch):
    if device.startswith("npu"):
        import torch_npu  # noqa: F401 - register the actual NPU backend
    from megatron.core.optimizer import distrib_optimizer as module

    monkeypatch.setattr(module, "HAVE_APEX_OR_TE", False)
    wrapped = object.__new__(module.DistributedOptimizer)
    params = [torch.nn.Parameter(torch.tensor([1.0, 2.0], device=device)),
              torch.nn.Parameter(torch.tensor([3.0, 4.0], device=device))]
    group = {"params": params, "wd_mult": 1.0, "lr_mult": 1.0,
             "is_expert_parallel": False, "is_decoupled_lr": False}
    wrapped.optimizer = torch.optim.AdamW([group], lr=1e-3)
    wrapped.grad_scaler = None
    wrapped.config = SimpleNamespace(fp16=False)
    wrapped.ddp_config = SimpleNamespace(use_megatron_fsdp=False, use_custom_fsdp=False)
    for _ in range(2):
        wrapped.optimizer.zero_grad()
        sum(p.square().sum() for p in params).backward()
        wrapped.optimizer.step()
    common = wrapped.state_dict()
    reference_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    reference = torch.optim.AdamW(reference_params, lr=1e-3)
    reference.load_state_dict(copy.deepcopy(wrapped.optimizer.state_dict()))
    for state in wrapped.optimizer.state.values():
        state["step"].fill_(100)
    if loader == "mindspeed":
        from mindspeed.core.megatron_basic import megatron_basic

        monkeypatch.setattr(megatron_basic, "HAVE_APEX_OR_TE", False)
        megatron_basic.dist_optim_load_state_dict(wrapped, common)
    else:
        wrapped.load_state_dict(common)
    counters = [wrapped.optimizer.state[p]["step"] for p in params]
    assert all(s.item() == 2 for s in counters)
    assert len({s.data_ptr() for s in counters}) == len(params)
    for optimizer, parameters in [(wrapped.optimizer, params), (reference, reference_params)]:
        optimizer.zero_grad()
        sum(p.square().sum() for p in parameters).backward()
        optimizer.step()
    assert all(s.item() == 3 for s in counters)
    for actual, expected in zip(params, reference_params):
        torch.testing.assert_close(actual, expected)
