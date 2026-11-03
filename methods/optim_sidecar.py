"""LoRA optimizer-state sidecar: make a resumed arm equal a continuous one.

Why this exists
---------------
The LoRA arms are launched with `--no-save-optim` because the
`DistributedOptimizer`'s `sharded_state_dict` cannot be matched against the model
tree and the periodic save crashes with `Duplicate ShardedObject keys`
(see `train_scripts/run_math_opsd_slime_npu.sh` and
`reports/reward_cv_100step_results.md`). The consequence is that a resumed arm
restarts Adam from zero at the resume boundary, so two arms with different
resume counts are not the same experiment. Measured on the real optimizer
(`tests/probe_optim_names.py`), the optimizer's parameters are:

  * **fp32 flattened clones** (`main_param = param.detach().clone().float()`),
    so they share no storage with the model parameters;
  * **carrying no back-pointer** (`param.model_param` is unset on the
    distributed path — 0/28 hit rate);
  * **in a different order** than `named_parameters()` (20/28 positions differ),
    while the multiset of element counts is identical.

So no name can be recovered from the optimizer object, and a positional zip
against `named_parameters()` would silently transpose LoRA A/B moments between
layers. What *is* stable is the optimizer's **own construction order**: the same
launcher, same model, same `--only-train-params-name-list`, same world size and
same Megatron bucketing produce the same `param_groups` layout on every start.
The sidecar therefore keys on `(group_index, param_index)` and **validates the
layout on load** (group sizes + per-parameter element counts) so that any drift
in that construction is a loud failure rather than a silent corruption.

Scope: the optimizer state only. `--no-save-rng` is still in force, so RNG state
is not restored; the data order is re-seeded absolutely from `seed + epoch`
(RUN_NOTES §4). Both arms inherit those identically.

Enabled only when `OPSD_OPTIM_SIDECAR=1`; otherwise this module is inert and
every other arm is bit-for-bit unaffected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

import torch

LOGGER = logging.getLogger("opsd.optim_sidecar")

SIDECAR_TEMPLATE = "optim_sidecar_rank{rank:05d}.pt"
ENV_GATE = "OPSD_OPTIM_SIDECAR"
SCHEMA_VERSION = 1
CKPT_DIR_TEMPLATE = "iter_{iteration:07d}"
TRACKER_NAME = "latest_checkpointed_iteration.txt"


class SidecarError(RuntimeError):
    """Raised on any sidecar/layout mismatch. Never swallowed."""


def is_enabled() -> bool:
    """True when the caller asked for sidecar persistence."""
    return os.environ.get(ENV_GATE, "0") == "1"


def current_rank() -> int:
    """This process's global rank, or 0 outside a distributed run.

    The optimizer state is DP-sharded -- `--use-distributed-optimizer` is forced
    on by `set_default_megatron_args` -- so each rank holds a *different* slice
    of Adam's moments and the sidecar is necessarily one file per rank.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - fall through to the env/default
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)


def checkpoint_dir(save_dir: str | os.PathLike[str], iteration: int) -> Path:
    return Path(save_dir) / CKPT_DIR_TEMPLATE.format(iteration=iteration)


def sidecar_path(
    save_dir: str | os.PathLike[str], iteration: int, rank: int | None = None
) -> Path:
    rank = current_rank() if rank is None else rank
    return checkpoint_dir(save_dir, iteration) / SIDECAR_TEMPLATE.format(rank=rank)


def _inner_optimizer(optimizer: Any) -> Any:
    """The torch optimizer holding `state`. Megatron wraps it as `.optimizer`."""
    return getattr(optimizer, "optimizer", optimizer)


def _group_params(optimizer: Any) -> list[list[Any]]:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        raise SidecarError("optimizer exposes no param_groups")
    return [list(g["params"]) for g in groups]


def _step_to_state(step: Any) -> tuple[bool, int]:
    """Normalise Adam's `step` to (is_tensor, int_value).

    torch's Adam keeps `step` as a plain int unless the capturable/fused kernels
    are used, where it is a 0-dim tensor, and the NPU fork can do either. Storing
    both the flag and the value preserves whichever the live optimizer expects.
    """
    if isinstance(step, torch.Tensor):
        return True, int(step.reshape(-1)[0].item())
    return False, int(step)


def _state_from_step(is_tensor: bool, value: int, device: torch.device) -> Any:
    if is_tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=device)
    return value


def _group_step(group: Mapping[str, Any]) -> dict | None:
    """Read the per-group step counter, if this optimizer keeps one there.

    Megatron builds **TransformerEngine FusedAdam** for `--optimizer adam`, and
    that optimizer keeps `step` on the *param group* rather than in `state` --
    measured on the real LoRA optimizer, whose state entries hold only
    `exp_avg`/`exp_avg_sq`. `step` is not cosmetic: Adam's bias correction is
    `1 - beta**step`, so dropping it makes the first resumed step roughly
    `(1-b1^1)/(1-b1^(k+1))` times too large. That is exactly the ~1 lr-unit
    divergence the NPU verification caught before this was added.
    """
    if "step" not in group:
        return None
    is_tensor, value = _step_to_state(group["step"])
    return {"step_is_tensor": is_tensor, "step_value": value}


def capture(optimizer: Any) -> dict:
    """Snapshot every Adam moment, keyed by (group, position).

    Parameters that have not been stepped yet hold no state and are recorded as
    None so that the layout stays aligned with `param_groups`.
    """
    inner = _inner_optimizer(optimizer)
    state = getattr(inner, "state", {})
    groups = _group_params(optimizer)

    out_groups: list[list[dict | None]] = []
    for params in groups:
        out_params: list[dict | None] = []
        for param in params:
            entry = state.get(param)
            if entry is None:
                out_params.append(None)
                continue
            exp_avg = entry["exp_avg"].detach().to(device="cpu", dtype=torch.float32)
            exp_avg_sq = entry["exp_avg_sq"].detach().to(device="cpu", dtype=torch.float32)
            if not (torch.isfinite(exp_avg).all() and torch.isfinite(exp_avg_sq).all()):
                raise SidecarError(
                    "refusing to write a sidecar with non-finite Adam moments "
                    f"(group sizes={[len(p) for p in groups]})"
                )
            is_tensor, step_value = _step_to_state(entry.get("step", 0))
            out_params.append(
                {
                    "numel": int(param.numel()),
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                    "step_is_tensor": is_tensor,
                    "step_value": step_value,
                }
            )
        out_groups.append(out_params)

    return {
        "schema": SCHEMA_VERSION,
        "group_sizes": [len(p) for p in groups],
        "numels": [[int(p.numel()) for p in params] for params in groups],
        "group_steps": [_group_step(g) for g in optimizer.param_groups],
        "groups": out_groups,
    }


def validate(payload: dict, optimizer: Any) -> None:
    """Fail loudly if the live optimizer layout does not match the payload."""
    schema = payload.get("schema")
    if schema != SCHEMA_VERSION:
        raise SidecarError(f"sidecar schema {schema!r} != expected {SCHEMA_VERSION}")
    groups = _group_params(optimizer)
    live_sizes = [len(p) for p in groups]
    if list(payload.get("group_sizes", [])) != live_sizes:
        raise SidecarError(
            f"optimizer group sizes changed: sidecar {payload.get('group_sizes')} "
            f"vs live {live_sizes} -- refusing to restore Adam state positionally"
        )
    live_numels = [[int(p.numel()) for p in params] for params in groups]
    if payload.get("numels") != live_numels:
        for gi, (want, got) in enumerate(zip(payload.get("numels", []), live_numels)):
            if want != got:
                raise SidecarError(
                    f"optimizer parameter shapes changed in group {gi}: "
                    f"sidecar {want} vs live {got} -- refusing to restore"
                )
        raise SidecarError("optimizer parameter shapes changed -- refusing to restore")


def restore(optimizer: Any, payload: dict) -> int:
    """Load Adam moments into the live optimizer. Returns params restored."""
    validate(payload, optimizer)
    inner = _inner_optimizer(optimizer)
    groups = _group_params(optimizer)

    for group, entry in zip(optimizer.param_groups, payload.get("group_steps") or []):
        if entry is None:
            continue
        # A freshly built optimizer does not have the key at all -- FusedAdam
        # only creates it when it takes its first step -- so this both restores
        # and, on a resume, re-creates the counter. A sidecar written by an
        # optimizer that had no per-group counter carries None and is skipped,
        # leaving a torch-native AdamW's own `state['step']` untouched.
        params = group.get("params") or []
        device = params[0].device if params else torch.device("cpu")
        group["step"] = _state_from_step(
            bool(entry["step_is_tensor"]), int(entry["step_value"]), device
        )

    restored = 0
    for params, entries in zip(groups, payload["groups"]):
        for param, entry in zip(params, entries):
            if entry is None:
                continue
            exp_avg = entry["exp_avg"].to(device=param.device, dtype=torch.float32)
            exp_avg_sq = entry["exp_avg_sq"].to(device=param.device, dtype=torch.float32)
            if not (torch.isfinite(exp_avg).all() and torch.isfinite(exp_avg_sq).all()):
                raise SidecarError("sidecar contains non-finite Adam moments")
            inner.state[param] = {
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "step": _state_from_step(
                    bool(entry["step_is_tensor"]), int(entry["step_value"]), param.device
                ),
            }
            restored += 1
    return restored


def dump(optimizer: Any, save_dir: str | os.PathLike[str], iteration: int) -> Path:
    """Write the sidecar next to the model checkpoint. Atomic; never partial."""
    payload = capture(optimizer)
    target = sidecar_path(save_dir, iteration)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def _latest_iteration(load_dir: str | os.PathLike[str]) -> int:
    path = Path(load_dir) / TRACKER_NAME
    if not path.is_file():
        raise SidecarError(f"no {TRACKER_NAME} under {load_dir}")
    return int(path.read_text().strip())


def restore_from_dir(optimizer: Any, load_dir: str | os.PathLike[str]) -> int:
    """Restore the sidecar belonging to `load_dir`'s latest checkpoint."""
    iteration = _latest_iteration(load_dir)
    path = sidecar_path(load_dir, iteration)
    if not path.is_file():
        raise SidecarError(
            f"{ENV_GATE}=1 but no sidecar at {path} -- cannot make this resume "
            "equivalent to a continuous run"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return restore(optimizer, payload)


def dump_if_enabled(optimizer: Any, save_dir: str | os.PathLike[str], iteration: int) -> None:
    """Hook called after Megatron writes `iter_N/`. Never breaks the main save."""
    if not is_enabled() or optimizer is None:
        return
    try:
        path = dump(optimizer, save_dir, iteration)
    except Exception as exc:  # noqa: BLE001 - the model checkpoint is already
        # written and is the primary artifact; damage to it would be worse than
        # a missing sidecar, so this is reported and the run continues. A resume
        # will then fail loudly in restore_from_dir rather than silently
        # restart Adam.
        LOGGER.warning("[optim-sidecar] dump FAILED for iter %s: %s", iteration, exc)
        return
    LOGGER.info("[optim-sidecar] wrote %s", path)


def restore_if_enabled(optimizer: Any, load_dir: str | os.PathLike[str] | None) -> int:
    """Hook called after `load_checkpoint`.

    A *fresh* start also reaches here, with `--load` pointing at the HF
    checkpoint; that directory has no tracker, so there is nothing to restore and
    returning 0 is correct. A directory that IS a checkpoint (tracker present)
    without a sidecar is the dangerous case and raises -- silently restarting
    Adam is precisely the confound this module exists to remove.
    """
    if not is_enabled() or optimizer is None or not load_dir:
        return 0
    if not (Path(load_dir) / TRACKER_NAME).is_file():
        LOGGER.info("[optim-sidecar] %s is not a megatron checkpoint -- fresh start", load_dir)
        return 0
    restored = restore_from_dir(optimizer, load_dir)
    LOGGER.info("[optim-sidecar] restored Adam state for %d tensors from %s", restored, load_dir)
    return restored
