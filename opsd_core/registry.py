"""L3 entry point: the ONE place that maps (task, method) to what to run.

REFACTOR_PLAN.md §4/§6 (R13). Before this module the same mapping was maintained
in three places that could drift apart:

  * ``train_scripts/run_math_opsd_slime_npu.sh``  — a chain of ``if [ "$OPSD_METHOD" = ... ]``
  * ``alfworld_hpi_v2/scripts/_common_v2.sh``     — a ``case "$ARM"``
  * ``scripts/run_names.py:METHOD_SLUG``          — the run-name vocabulary

``METHODS`` below is the single source. The launcher reads it; ``run_names.py``
cross-checks against ``canonical_slugs()`` (tests/test_registry.py).

Stdlib-only on purpose: the launcher calls this with a bare ``python -c`` before
any conda env with torch is needed.

Usage
-----
    python -m opsd_core.registry --task math --method hsi
prints ``KEY=value`` lines the launcher sources directly.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace as _replace
from typing import Mapping

# --------------------------------------------------------------------------- #
# Sampler conventions
# --------------------------------------------------------------------------- #

# OPSD / GRPO arms sample from a nucleus (top_p=0.95, top_k=20); the HSI arms must
# sample the TRUE temperature-1 policy (top_p=1 / top_k=-1, i.e. slime's own
# defaults) because the credit estimator reads the head of that distribution.
# The launcher *omits* the flags rather than passing them: `--rollout-top-k -1`
# (a negative number as a separate token) corrupts slime's arg parser and kills
# the run at import (2026-09-23, root-caused to an empty dataclass field name).
SAMPLER_TOP_P_095 = "top_p_0.95"
SAMPLER_FULL_SOFTMAX = "full_softmax"


@dataclass(frozen=True)
class TaskSpec:
    """Everything that is true of a task regardless of method."""

    name: str
    run_namespace: str
    train_data: str
    eval_config: str
    # per-turn / per-response generation cap
    max_response_train: int
    max_response_eval: int
    sglang_context_length: int
    eval_k: int
    # the reward source used by the eval/eval-only path (`--custom-rm-path`)
    rm: str | None
    thinking: bool


@dataclass(frozen=True)
class MethodSpec:
    """Everything that is true of a method for one task."""

    name: str
    slug: str
    loss: str | None  # None => slime's stock policy loss (GRPO/GSPO)
    rollout: str
    convert: str | None
    postprocess: str | None
    provider: str | None
    rm: str | None
    sampler: str
    lora: bool
    # Optimizer-step horizon and checkpoint cadence. ``None`` means "the
    # launcher's own default applies" — the registry must only carry values that
    # DEVIATE from it. (The math launcher defaults to num_rollout=25,
    # save_interval=5; emitting 25/25 unconditionally silently cut the checkpoint
    # cadence 5x.)
    num_rollout: int | None = None
    save_interval: int | None = None
    # Generations per prompt. The HSI arms collapse to 1: their advantage is a
    # same-policy hindsight contrast, not a group statistic.
    n_samples_per_prompt: int | None = None
    # Pinned optimizer-step size in ROWS, when the arm must hold it constant
    # rather than let it follow rollout_bs x G (see _common_v2.sh's rationale).
    gbs_override: int | None = None
    # True => do NOT pass --rollout-top-p/-k (slime's own 1.0/-1 defaults apply).
    # See SAMPLER_FULL_SOFTMAX for why the flags are omitted rather than set.
    omit_sampler_flags: bool = False
    extra_args: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()
    # extra_env is emitted overridably (`: "${X:=v}"`) by default. The ALFWorld
    # launcher instead assigns its arm-dictated env UNCONDITIONALLY (`export X=v`),
    # so an operator cannot preset G or ADVANTAGE_ESTIMATOR against the arm. This
    # flag preserves each launcher's own semantics rather than picking one.
    force_env: bool = False
    supports_eval: bool = True


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

TASKS: Mapping[str, TaskSpec] = {
    "math": TaskSpec(
        name="math",
        run_namespace="math",
        train_data="data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet",
        eval_config="configs/eval_aime24_dev.json",
        max_response_train=1024,
        max_response_eval=38912,
        sglang_context_length=32768,
        eval_k=12,
        rm="tasks.math.rm_math_verify.rm_math_verify",
        thinking=False,
    ),
    "alfworld": TaskSpec(
        name="alfworld",
        run_namespace="alfworld",
        train_data="tasks/alfworld/configs/train_schedule_seed42.jsonl",
        eval_config="tasks/alfworld/v1/configs/eval_alfworld.json",
        max_response_train=128,
        max_response_eval=128,
        sglang_context_length=32768,
        eval_k=1,
        rm="tasks.alfworld.v1.reward.reward_from_trajectory",
        thinking=False,
    ),
    "code": TaskSpec(
        name="code",
        run_namespace="code",
        train_data="tasks/code/data/train_v2.jsonl",
        eval_config="tasks/code/eval/lcb_eval_config.json",
        max_response_train=4096,  # run_code_{grpo,hsi}.sh: MAX_RESPONSE=4096
        max_response_eval=8192,
        sglang_context_length=16384,
        eval_k=1,
        rm="tasks.code.rm_code.rm_code",
        thinking=False,
    ),
}

# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #

_STOCK_LORA_PROVIDER = "methods.lora_megatron.custom_model_provider"
_HSI_PROVIDER = "tasks.alfworld.hsi_provider.custom_model_provider"
_HSI_LOSS = "tasks.alfworld.loss.alfworld_v2_loss"

# The HSI runtime env, identical across tasks (the estimator is shared:
# math_opsd/hsi2_jsd.py, aliased by alfworld_hpi_v2/hsi2_jsd.py). Every live HSI
# arm -- the only credit method left after the 2026-10-15 retirement -- is the
# exact-JSD estimator, so the selector is pinned here rather than kept in a
# separate "common" block.
_HSI_ENV: tuple[tuple[str, str], ...] = (
    ("OPSD_HSI_CHUNKS", "4"),
    ("OPSD_ENABLED", "1"),
    ("OPSD_V2_LOSS_NORM", "traj"),
    ("ROLLOUT_TEMPERATURE", "1.0"),
    ("OPSD_CLIP_GRAD", "1.0"),
    ("ALFWORLD_V2_HPI", "1"),
    ("ALFWORLD_V2_HSI", "1"),
    ("ALFWORLD_V2_HSI_TOP_P", "1.0"),
    ("OPSD_HSI_ESTIMATOR", "jsd"),
)


def _spec(task: str, method: str, **kw) -> MethodSpec:
    return MethodSpec(name=method, slug=kw.pop("slug", method), **kw)


METHODS: Mapping[str, Mapping[str, MethodSpec]] = {
    "math": {
        # Official OPSD objective: frozen initial-policy teacher, beta=0 KL.
        "opsd": _spec(
            "math",
            "opsd",
            loss="methods.opsd_loss.opsd_loss_function",
            rollout="tasks.math.math_rollout.generate_rollout",
            convert="tasks.math.math_data.convert_samples_to_train_data",
            postprocess=None,
            provider=None,
            rm=None,
            sampler=SAMPLER_TOP_P_095,
            lora=False,
            # Uniform checkpoint cadence across arms (user, 2026-10-14). The
            # launcher's own fallback is 5; pinning 25 here is deliberate.
            save_interval=25,
        ),
        # Our method: same-policy hindsight JSD credit over a chunked generation.
        "hsi": _spec(
            "math",
            "hsi",
            loss=_HSI_LOSS,
            rollout="methods.hsi_rollout.generate_rollout",
            convert="methods.hsi2_train_data.convert_samples_to_train_data",
            postprocess="methods.hsi2_jsd.postprocess",
            provider=_HSI_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            lora=True,
            num_rollout=100,
            save_interval=25,  # ckpt at rollout 24/49/74/99 (cp25/50/75/100)
            omit_sampler_flags=True,
            extra_args=("--use-opsd", "--use-rollout-logprobs"),
            extra_env=_HSI_ENV + (("HSI_STEP_PROMPTS", "32"), ("ROLLOUT_BS", "36"), ("OVERSAMPLE_BS", "36")),
        ),
        # Matched control: stock GRPO policy loss, G=8.
        "grpo": _spec(
            "math",
            "grpo",
            loss=None,
            rollout="methods.grpo_rollout.generate_rollout",
            convert=None,
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm="tasks.math.rm_math_verify.rm_math_verify",
            sampler=SAMPLER_TOP_P_095,
            lora=True,
            extra_env=(("ROLLOUT_BS", "4"), ("N_SAMPLES_PER_PROMPT", "8")),
        ),
    },
    "alfworld": {
        "srlvr": _spec(
            "alfworld", "srlvr",
            loss="methods.srlvr.actor_loss",
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="methods.srlvr.convert", postprocess=None,
            provider=_STOCK_LORA_PROVIDER, rm=None,
            sampler=SAMPLER_FULL_SOFTMAX, lora=True,
            n_samples_per_prompt=1, omit_sampler_flags=True,
            extra_args=("--use-rollout-logprobs", "--disable-rollout-trim-samples"),
            extra_env=(("ALFWORLD_V2_HPI", "0"), ("OPSD_SRLVR", "1")),
        ),
        "srlvr_v2": _spec(
            "alfworld", "srlvr_v2",
            loss="methods.srlvr_hsi.actor_loss",
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="methods.srlvr_hsi.convert", postprocess=None,
            provider=_STOCK_LORA_PROVIDER, rm=None,
            sampler=SAMPLER_FULL_SOFTMAX, lora=True,
            n_samples_per_prompt=1, omit_sampler_flags=True,
            num_rollout=25,
            extra_args=("--use-rollout-logprobs", "--disable-rollout-trim-samples"),
            extra_env=(("ALFWORLD_V2_HPI", "0"), ("OPSD_SRLVR", "1"), ("SRLVR_VARIANT", "v2")),
        ),
        "srlvr_v3": _spec(
            "alfworld", "srlvr_v3",
            loss="methods.srlvr_hsi.actor_loss",
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="methods.srlvr_hsi.convert", postprocess=None,
            provider=_STOCK_LORA_PROVIDER, rm=None,
            sampler=SAMPLER_FULL_SOFTMAX, lora=True,
            n_samples_per_prompt=1, omit_sampler_flags=True,
            num_rollout=25,
            extra_args=("--use-rollout-logprobs", "--disable-rollout-trim-samples"),
            extra_env=(("ALFWORLD_V2_HPI", "0"), ("OPSD_SRLVR", "1"), ("SRLVR_VARIANT", "v3")),
        ),
        # --- matched baseline pair (G=8) ------------------------------------
        "grpo": _spec(
            "alfworld",
            "grpo",
            loss=_HSI_LOSS,
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="tasks.alfworld.train_data.convert_samples_to_train_data",
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            omit_sampler_flags=True,
            lora=True,
            n_samples_per_prompt=8,
            extra_env=(("ALFWORLD_V2_HPI", "0"),),
        ),
        # GiGPO (arXiv:2505.10978): GRPO's episode advantage PLUS an anchor-state
        # step advantage. Same rollout/convert/loss as GRPO -- the step term rides
        # in the existing per-token `opsd_advantages` channel, computed once per
        # task group inside the rollout where the episode structure is visible.
        # G=8 so every anchor group is formed from the same eight trajectories the
        # matched GRPO control uses; only the advantage estimator differs.
        "gigpo": _spec(
            "alfworld",
            "gigpo",
            loss=_HSI_LOSS,
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="tasks.alfworld.train_data.convert_samples_to_train_data",
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            omit_sampler_flags=True,
            lora=True,
            n_samples_per_prompt=8,
            extra_env=(
                ("ALFWORLD_V2_HPI", "0"),
                ("ALFWORLD_V2_GIGPO", "1"),
            ),
        ),
        # The registry selects the sequence-level surrogate. Wrappers own its
        # clipping/optimizer configuration, independently of the GRPO defaults.
        "gspo": _spec(
            "alfworld",
            "gspo",
            loss=_HSI_LOSS,
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="tasks.alfworld.train_data.convert_samples_to_train_data",
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            omit_sampler_flags=True,
            lora=True,
            n_samples_per_prompt=8,
            extra_env=(("ALFWORLD_V2_HPI", "0"), ("ADVANTAGE_ESTIMATOR", "gspo")),
        ),
        # --- HSI: G=1, GBS pinned at 32 turn-rows ---------------------------
        # G collapses to 1 because the advantage is a same-policy hindsight
        # contrast, not a group statistic; GBS is held at 32 so the optimizer step
        # size stays comparable to the GRPO control.
        #
        # This is the same objective as math/ and code/ `hsi` (math_opsd/hsi2_jsd.py,
        # aliased at alfworld_hpi_v2/hsi2_jsd.py): exact normalized JSD between the
        # normal and hindsight views of the SAME theta_old. The launcher's ARM `hsi_v3`
        # aliases here (see ALIASES).
        "hsi": _spec(
            "alfworld",
            "hsi",
            loss=_HSI_LOSS,
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="tasks.alfworld.train_data.convert_samples_to_train_data",
            postprocess="tasks.alfworld.hsi2_jsd.postprocess",
            provider=_HSI_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            omit_sampler_flags=True,
            lora=True,
            n_samples_per_prompt=1,
            gbs_override=32,
            extra_args=("--use-rollout-logprobs",),
            # ALFWorld's arm-dictated env is ONLY these four. Everything else
            # (OPSD_ENABLED, OPSD_V2_LOSS_NORM, LoRA, NPU) is set globally by
            # _common_v2.sh outside the switch.
            extra_env=(
                ("ALFWORLD_V2_HPI", "1"),
                ("ALFWORLD_V2_HSI", "1"),
                ("ALFWORLD_V2_HSI_V2", "1"),
                ("OPSD_HSI_ESTIMATOR", "jsd"),
            ),
        ),
        # SRLVR-v4: ONE trajectory-level privileged teacher view (complete compact
        # trace + outcome) distilled into every turn's causal prompt. The teacher
        # is the frozen initial policy (OPSD's teacher contract), the student is
        # the LoRA actor; unlike v2/v3 there is no calibration probe and no
        # per-turn teacher forward. G=1, one optimizer update per rollout.
        "srlvr_v4": _spec(
            "alfworld",
            "srlvr_v4",
            loss="methods.srlvr_v4.actor_loss",
            rollout="tasks.alfworld.rollout.generate_rollout",
            convert="tasks.alfworld.train_data.convert_samples_to_train_data",
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm=None,
            sampler=SAMPLER_FULL_SOFTMAX,
            omit_sampler_flags=True,
            lora=True,
            n_samples_per_prompt=1,
            num_rollout=100,
            save_interval=25,
            extra_args=("--use-rollout-logprobs", "--disable-rollout-trim-samples"),
            extra_env=(
                ("ALFWORLD_V2_HPI", "0"),
                ("OPSD_SRLVR", "1"),
                ("SRLVR_VARIANT", "v4"),
                ("SRLVR_V4_COMPACT_FUTURE", "1"),
            ),
        ),
    },
    "code": {
        "grpo": _spec(
            "code",
            "grpo",
            loss=None,
            rollout="tasks.code.rollout.generate_rollout",
            convert=None,
            postprocess=None,
            provider=_STOCK_LORA_PROVIDER,
            rm="tasks.code.rm_code.rm_code",
            sampler=SAMPLER_TOP_P_095,
            lora=True,
            n_samples_per_prompt=8,
            num_rollout=100,
        ),
        "hsi": _spec(
            "code",
            "hsi",
            loss=_HSI_LOSS,
            rollout="tasks.code.hsi_rollout.generate_rollout",
            convert="methods.hsi2_train_data.convert_samples_to_train_data",
            postprocess="methods.hsi2_jsd.postprocess",
            provider=_HSI_PROVIDER,
            rm="tasks.code.rm_code.rm_code",
            sampler=SAMPLER_FULL_SOFTMAX,
            lora=True,
            omit_sampler_flags=True,
            num_rollout=100,
            n_samples_per_prompt=1,
            # GLOB_BS = HSI_STEP_PROMPTS x OPSD_HSI_CHUNKS, computed by the launcher
            # (8 unique problems per step for code, not math's 32).
            extra_args=("--use-opsd", "--use-rollout-logprobs"),
            extra_env=_HSI_ENV + (("HSI_STEP_PROMPTS", "8"),),
        ),
    },
}

# The ALFWorld launcher assigns its arm-dictated env UNCONDITIONALLY
# (`export ALFWORLD_V2_HPI=1`, `ADVANTAGE_ESTIMATOR="gspo"`), unlike math/code
# which default it overridably. Preserve each launcher's own semantics instead of
# picking one for both — `_emit_shell` reads `force_env`.
METHODS = {
    _task: {
        _m: (_replace(_s, force_env=True) if _task == "alfworld" else _s)
        for _m, _s in _table.items()
    }
    for _task, _table in METHODS.items()
}

# Launcher-side historical spellings that resolve to a canonical method.
# `hsi_v3` is what alfworld_hpi_v2/scripts/*.sh pass as ARM; it IS our method.
ALIASES: Mapping[tuple[str, str], tuple[str, str]] = {
    ("alfworld", "hsi_v3"): ("alfworld", "hsi"),
}

# Methods retired 2026-10-14 (REFACTOR_PLAN.md §7) and 2026-10-15 (ALFWorld credit
# arms; see _retired/README.md Batch 2/3). Kept here so that a stale launcher
# invocation fails LOUDLY with the retirement note rather than silently falling
# through to the default method.
RETIRED_METHODS: Mapping[str, str] = {
    "reward_cv": "Reward-CV control-variate objective",
    "reward_cv_calibrated": "Calibrated Reward-CV (online lambda)",
    "reward_cv_hpi": "HPI v1",
    "reward_cv_hpi_v2": "HPI v2",
    "reward_cv_hpi_v15": "HPI v1.5",
    "reward_cv_vnext": "Reward-CV vNext",
    "reward_only_vnext": "Reward-CV vNext (reward-only control)",
    "outcome_posterior_cv": "Outcome-Posterior Control Variate",
    "outcome_pi": "Outcome-PI",
    "outcome_v3": "Outcome-PI v3",
    # ALFWorld credit arms retired 2026-10-15 (the paper reports HP-RLVR = HSI vs
    # GRPO only). `hpi_future` is the two-teacher Full/Control scorer; `hsi_v2` is
    # the same-policy top-128 candidate projection; `hsi_candidate` is the
    # frozen-theta0 candidate centre. Implementations live under
    # `_retired/methods/`; the modules they imported are archived there too.
    "hpi_future": "HPI future (ALFWorld two-teacher Full/Control scorer)",
    "hsi_v2": "HSI v2 (same-policy top-128 candidate projection)",
    "hsi_candidate": "HSI candidate-centred (frozen-theta0 centre)",
}


class UnknownSelection(KeyError):
    """Raised when the (task, method) pair is not a known, live combination."""


def resolve(task: str, method: str) -> tuple[TaskSpec, MethodSpec]:
    """Resolve a (task, method) pair into the full run configuration.

    Fails closed: an unknown method, a retired method, or a method that does not
    exist for the requested task all raise, naming the valid options.
    """
    # Normalise a launcher's historical spelling BEFORE the lookups, so an alias
    # never shadows a real name and a retired name is still caught by its own
    # spelling.
    task, method = ALIASES.get((task, method), (task, method))

    if task not in TASKS:
        raise UnknownSelection(f"unknown task {task!r}; known: {sorted(TASKS)}")
    if method in RETIRED_METHODS:
        raise UnknownSelection(
            f"method {method!r} was retired ({RETIRED_METHODS[method]}); "
            f"see REFACTOR_PLAN.md §7 and _retired/README.md"
        )
    table = METHODS[task]
    if method not in table:
        raise UnknownSelection(
            f"method {method!r} is not defined for task {task!r}; known: {sorted(table)}"
        )
    return TASKS[task], table[method]


def known_methods(task: str) -> list[str]:
    if task not in TASKS:
        raise UnknownSelection(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return sorted(METHODS[task])


def canonical_slugs() -> dict[str, str]:
    """Canonical run-name method slug for every live (task, method).

    ``scripts/run_names.py`` keeps its own legacy vocabulary for parsing OLD run
    directories; this is what it must agree with for NEW ones.
    """
    return {f"{t}/{m}": spec.slug for t, table in METHODS.items() for m, spec in table.items()}


# --------------------------------------------------------------------------- #
# CLI: the launcher sources this
# --------------------------------------------------------------------------- #


def _emit(task: str, method: str) -> int:
    try:
        task_spec, spec = resolve(task, method)
    except UnknownSelection as exc:
        print(f"opsd_registry: {exc}", file=sys.stderr)
        return 2

    out = {
        "OPSD_TASK": task_spec.name,
        "OPSD_METHOD": spec.name,
        "OPSD_METHOD_SLUG": spec.slug,
        "OPSD_TRAIN_DATA": task_spec.train_data,
        "OPSD_EVAL_CONFIG": task_spec.eval_config,
        "OPSD_MAX_RESPONSE": str(task_spec.max_response_train),
        "OPSD_SGLANG_CONTEXT_LENGTH": str(task_spec.sglang_context_length),
        "OPSD_SAMPLER": spec.sampler,
        "OPSD_LORA": "1" if spec.lora else "0",
        "OPSD_THINKING": "1" if task_spec.thinking else "0",
        "CUSTOM_LOSS_FN": spec.loss or "",
        "ROLLOUT_FN": spec.rollout,
        "CONVERT_FN": spec.convert or "",
        "POSTPROCESS_FN": spec.postprocess or "",
        "MODEL_PROVIDER": spec.provider or "",
        "CUSTOM_RM_FN": spec.rm or "",
    }
    for key, value in spec.extra_env:
        out[f"ENV_{key}"] = value
    for idx, arg in enumerate(spec.extra_args):
        out[f"EXTRA_ARG_{idx}"] = arg
    for key, value in sorted(out.items()):
        print(f"{key}={value}")
    return 0


# Values are emitted INSIDE "${VAR:=word}". Quote characters inside that word are
# literal (bash keeps them), so we do not quote and instead refuse anything that
# is not plainly shell-safe. Every value here is a dotted module path, a flag, or
# a number, so the whitelist never bites in practice — but a poisoned value must
# fail closed rather than be evaluated.
_SAFE_VALUE = re.compile(r"\A[A-Za-z0-9_./:@=+-]*\Z")


def _safe(value: str) -> str:
    if not _SAFE_VALUE.match(value):
        raise ValueError(f"refusing to emit shell fragment for unsafe value: {value!r}")
    return value


def _emit_shell(task: str, method: str) -> int:
    """Emit a bash fragment the launcher evals.

    Every assignment uses ``${VAR:=default}`` so an explicitly exported variable
    still wins — the launcher's historical override semantics are preserved.
    """
    try:
        task_spec, spec = resolve(task, method)
    except UnknownSelection as exc:
        print(f"opsd_registry: {exc}", file=sys.stderr)
        return 2

    def overridable(key: str, value: str, *, export: bool = False) -> None:
        line = f'${{{key}:={_safe(value)}}}'
        print(f'export {key}="{line}"' if export else f': "{line}"')

    overridable("OPSD_TASK", task_spec.name, export=True)
    overridable("OPSD_METHOD", spec.name, export=True)
    overridable("OPSD_METHOD_SLUG", spec.slug, export=True)
    # Only emitted when the method deviates from the launcher's own defaults.
    if spec.num_rollout is not None:
        overridable("NUM_ROLLOUT", str(spec.num_rollout))
    if spec.save_interval is not None:
        overridable("SAVE_INTERVAL", str(spec.save_interval))
    # Generation count and step size are DICTATED by the arm, not defaulted: the
    # original switches assign them unconditionally, so a preset must not win.
    if spec.n_samples_per_prompt is not None:
        print(f"G={spec.n_samples_per_prompt}")
    if spec.gbs_override is not None:
        print(f"GBS_OVERRIDE={spec.gbs_override}")
    overridable("MAX_RESPONSE", str(task_spec.max_response_train), export=True)
    overridable("SGLANG_CONTEXT_LENGTH", str(task_spec.sglang_context_length), export=True)
    overridable("DATA", task_spec.train_data, export=True)

    # A None slot means "leave it unset" — the launcher then either omits the
    # flag (stock loss) or applies its own default provider.
    for shell_var, value in (
        ("CUSTOM_LOSS_FN", spec.loss),
        ("ROLLOUT_FN", spec.rollout),
        ("CONVERT_FN", spec.convert),
        ("POSTPROCESS_FN", spec.postprocess),
        ("MODEL_PROVIDER", spec.provider),
        ("CUSTOM_RM_FN", spec.rm),
    ):
        if value:
            overridable(shell_var, value)

    print(f': "${{OPSD_STOCK_LOSS:={"1" if not spec.loss else "0"}}}"')
    print(f': "${{OPSD_OMIT_SAMPLER_FLAGS:={"1" if spec.omit_sampler_flags else "0"}}}"')
    print(f': "${{OPSD_LORA:={"1" if spec.lora else "0"}}}"')

    for key, value in spec.extra_env:
        if spec.force_env:
            print(f"export {key}={_safe(value)}")
        else:
            overridable(key, value, export=True)

    # Array literal rather than a space-joined string: the launcher needs a real
    # bash array, and each element is whitelisted so emitting it unquoted is safe.
    # These are intrinsic to the method, so they are not env-overridable.
    print(f"EXTRA_ARGS=({' '.join(_safe(a) for a in spec.extra_args)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opsd_core.registry")
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", help="required unless --list is given")
    parser.add_argument("--list", action="store_true", help="list the live methods for the task")
    parser.add_argument(
        "--shell", action="store_true", help="emit a bash fragment for the launcher to eval"
    )
    args = parser.parse_args(argv)

    if args.task not in TASKS:
        print(f"opsd_registry: unknown task {args.task!r}; known: {sorted(TASKS)}", file=sys.stderr)
        return 2

    if args.list:
        for name in known_methods(args.task):
            print(name)
        return 0

    if not args.method:
        parser.error("--method is required unless --list is given")
    return _emit_shell(args.task, args.method) if args.shell else _emit(args.task, args.method)


if __name__ == "__main__":
    raise SystemExit(main())
