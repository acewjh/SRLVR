"""Model-card contract tests for the size-independent math launchers.

The contract (2026-09-30): running 1.7B / 4B / 8B is a matter of picking
`MODEL=<slug>`, and the ONLY size-specific file is configs/models/<slug>.sh. These
tests hold that line from both sides:

  * every card resolves through train_scripts/lib/model_env.sh and is consistent
    with the checkpoint it names (architecture args read off the real config.json,
    not guessed -- a wrong --num-layers fails deep inside Megatron, ~20 minutes in);
  * the shared launchers contain no size-specific value, so a new size cannot
    smuggle one in and silently reuse another size's geometry;
  * the per-size shims select the right card.

Run:  python -m pytest tests/test_model_configs.py -q
      (or: python tests/test_model_configs.py)
"""

import json
import pathlib
import shlex
import subprocess
import sys

import pytest

PROJ = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = PROJ / "configs" / "models"
CHECKPOINTS = PROJ.parent / "checkpoints"

CARDS = sorted(p.stem for p in MODELS_DIR.glob("*.sh"))

# flag -> checkpoint config.json key, for the values that must not drift
ARCH_FLAGS = {
    "--num-layers": "num_hidden_layers",
    "--hidden-size": "hidden_size",
    "--ffn-hidden-size": "intermediate_size",
    "--num-attention-heads": "num_attention_heads",
    "--num-query-groups": "num_key_value_heads",
    "--vocab-size": "vocab_size",
    "--kv-channels": "head_dim",
}

REQUIRED_KEYS = (
    "MODEL_NAME",
    "MODEL_DIR",
    "MODEL_SLUG",
    "MODEL_TASK",
    "MODEL_TP",
    "MODEL_PP",
    "MODEL_MBS_DEFAULT",
)


def resolve(card: str) -> dict[str, str]:
    """Resolve one card exactly the way a launcher does, and return the result."""
    script = f"""
set -uo pipefail
export CUA_PROJ={shlex.quote(str(PROJ))}
export MODEL={shlex.quote(card)}
source {shlex.quote(str(PROJ / 'train_scripts' / 'lib' / 'model_env.sh'))} || exit 9
for k in {' '.join(REQUIRED_KEYS)} MODEL_SGLANG_MEM_FRACTION MODEL_SGLANG_CHUNKED_PREFILL; do
  printf '%s\\t%s\\n' "$k" "${{!k}}"
done
printf 'MODEL_ARGS\\t%s\\n' "${{MODEL_ARGS[*]}}"
"""
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, (
        f"MODEL={card} did not resolve through train_scripts/lib/model_env.sh\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return dict(
        line.split("\t", 1) for line in proc.stdout.strip().splitlines() if "\t" in line
    )


@pytest.mark.parametrize("card", CARDS)
def test_card_resolves_with_required_keys(card: str) -> None:
    resolved = resolve(card)
    for key in REQUIRED_KEYS:
        assert resolved.get(key), f"{card}.sh resolved an empty {key}"
    assert resolved["MODEL_SLUG"] == card, (
        f"{card}.sh resolved MODEL_SLUG={resolved['MODEL_SLUG']}"
    )
    for key in ("MODEL_TP", "MODEL_PP", "MODEL_MBS_DEFAULT"):
        assert int(resolved[key]) >= 1, f"{card}.sh has {key}={resolved[key]}"
    # The card must name the checkpoint it claims to describe.
    assert pathlib.Path(resolved["MODEL_DIR"]).name == resolved["MODEL_NAME"]


@pytest.mark.parametrize("card", CARDS)
def test_arch_args_match_checkpoint_config(card: str) -> None:
    """MODEL_ARGS must agree with the checkpoint's own config.json.

    This is the failure mode that costs real time: a wrong --num-layers or
    --hidden-size is accepted by the argument parser and only explodes when
    mbridge tries to map the HF weights onto the Megatron model.
    """
    resolved = resolve(card)
    cfg_path = pathlib.Path(resolved["MODEL_DIR"]) / "config.json"
    if not cfg_path.is_file():
        pytest.skip(f"{cfg_path} not present on this node")

    cfg = json.loads(cfg_path.read_text())
    args = resolved["MODEL_ARGS"].split()
    for flag, key in ARCH_FLAGS.items():
        assert flag in args, f"{card}.sh MODEL_ARGS is missing {flag}"
        got = int(args[args.index(flag) + 1])
        assert got == cfg[key], (
            f"{card}.sh MODEL_ARGS {flag}={got} but {cfg_path} says {key}={cfg[key]}"
        )

    # 8B is the only size in this family with untied embeddings, and MBridge needs
    # to be told explicitly -- getting it wrong silently pairs a tied lm_head with
    # an untied checkpoint.
    untied_flag = "--untie-embeddings-and-output-weights"
    tied = cfg.get("tie_word_embeddings", True)
    assert (untied_flag in args) is (not tied), (
        f"{card}.sh: config.json tie_word_embeddings={tied} but MODEL_ARGS "
        f"{'has' if untied_flag in args else 'lacks'} {untied_flag}"
    )


@pytest.mark.parametrize("card", CARDS)
def test_pp_tp_leaves_a_valid_data_parallel_group(card: str) -> None:
    """The default 4-actor topology must satisfy GBS % (MBS x DP) == 0.

    Both launchers enforce this at runtime; checking it here turns an 8B card that
    can never start into a test failure instead of a run that dies at step 0.
    """
    resolved = resolve(card)
    tp, pp = int(resolved["MODEL_TP"]), int(resolved["MODEL_PP"])
    mbs = int(resolved["MODEL_MBS_DEFAULT"])
    actor_npus = 4  # the launchers' default NUM_TRAIN_GPUS on this 8-NPU node
    assert actor_npus % (tp * pp) == 0, (
        f"{card}.sh: TP{tp} x PP{pp} does not divide the default {actor_npus} actors"
    )
    dp = actor_npus // (tp * pp)
    assert 32 % (mbs * dp) == 0, (
        f"{card}.sh: GBS=32 is not a multiple of MBS={mbs} x DP={dp}"
    )


@pytest.mark.parametrize(
    "launcher",
    [
        "train_scripts/run_math_opsd_slime_npu.sh",
        "train_scripts/run_dev_eval.sh",
        "scripts/run_grpo.sh",
    ],
)
def test_shared_launcher_has_no_size_specific_values(launcher: str) -> None:
    """The point of the refactor: shared code must not carry one size's geometry."""
    text = (PROJ / launcher).read_text()
    for flag in ARCH_FLAGS:
        assert flag not in text, (
            f"{launcher} hardcodes {flag}; it belongs in configs/models/<slug>.sh"
        )
    for literal in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"):
        assert literal not in text, (
            f"{launcher} hardcodes the checkpoint {literal}; it should come from "
            "MODEL_DIR in the model card"
        )
    assert "model_env.sh" in text, f"{launcher} does not resolve a model card"
    # ...and must abort on a bad slug instead of carrying on with unset values.
    assert 'model_env.sh" || exit 1' in text, (
        f"{launcher} sources the model card without a failure guard; a typo in "
        "MODEL would then run with an unexpanded configuration"
    )


@pytest.mark.parametrize(
    "shim,expected",
    [
        ("train_scripts/run_math_opsd_slime_npu_4b.sh", "qwen3-4b"),
        ("train_scripts/run_math_opsd_slime_npu_8b.sh", "qwen3-8b"),
        ("train_scripts/run_dev_eval_4b.sh", "qwen3-4b"),
        ("train_scripts/run_dev_eval_8b.sh", "qwen3-8b"),
        ("scripts/run_grpo_4b.sh", "qwen3-4b"),
        ("scripts/run_grpo_8b.sh", "qwen3-8b"),
    ],
)
def test_shim_selects_its_card(shim: str, expected: str) -> None:
    text = (PROJ / shim).read_text()
    assert f'MODEL="${{MODEL:-{expected}}}"' in text, (
        f"{shim} does not default MODEL to {expected}"
    )
    assert "exec bash" in text, f"{shim} should exec the shared launcher"


def _source_with_status(model: str) -> str:
    """Shell snippet that sources a card and reports the source's own exit status."""
    return f"""
export CUA_PROJ={shlex.quote(str(PROJ))}
export MODEL={shlex.quote(model)}
source {shlex.quote(str(PROJ / 'train_scripts' / 'lib' / 'model_env.sh'))}
echo "SOURCE_RC=$?"
"""


def test_unknown_model_slug_fails_fast() -> None:
    """An unknown slug must make the `source` itself fail.

    `return 1` inside a sourced file stops that file and sets the source's exit
    status; it does NOT stop the caller, which is why every launcher writes
    `source ... || exit 1`. The test mirrors that usage, so what is asserted here
    is the contract the launchers actually rely on.
    """
    proc = subprocess.run(
        ["bash", "-c", _source_with_status("definitely-not-a-model")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.stdout.strip() != "SOURCE_RC=0", (
        "an unknown MODEL slug resolved successfully:\n" + proc.stdout
    )
    assert "FATAL" in proc.stderr


def test_bad_card_knob_is_rejected(tmp_path: pathlib.Path) -> None:
    """A non-positive TP/PP/MBS must be rejected, not silently used."""
    (tmp_path / "broken.sh").write_text(
        'MODEL_NAME="Broken"\nMODEL_DIR="/nonexistent"\nMODEL_TP=0\nMODEL_ARGS=(--swiglu)\n'
    )
    script = f"""
export CUA_MODELS_DIR={shlex.quote(str(tmp_path))}
export CUA_PROJ={shlex.quote(str(PROJ))}
export MODEL=broken
source {shlex.quote(str(PROJ / 'train_scripts' / 'lib' / 'model_env.sh'))}
echo "SOURCE_RC=$?"
"""
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.stdout.strip() != "SOURCE_RC=0", (
        "a card with MODEL_TP=0 and a nonexistent MODEL_DIR resolved successfully:\n"
        + proc.stdout
    )
    assert "FATAL" in proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
