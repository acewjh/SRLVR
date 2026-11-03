"""Replay a launcher's exact argv through slime's parser, without any NPU.

WHY THIS EXISTS. Twice on this project a launch died several minutes in, in
``train.py``'s ``parse_args()``, with a traceback pointing deep into
Megatron-Bridge's dataclass codegen (``SyntaxError: invalid syntax`` on a
generated ``__init__``). The real cause was a launcher flag that parses into an
EMPTY-STRING key in the args namespace -- a bare negative value such as
``--rollout-top-k -1`` is the usual way in -- which MindSpeed's
``transformer_config_init_subclass`` then copies into
``TransformerConfig.__annotations__``, where dataclasses renders it as a field
literally named ``""``.

The failure is invisible from the flag list and expensive from the log, so the
cheap gate is to run the parser over the SAME argv in seconds and no NPU:

    DRY_RUN_PARSE=1 bash tasks/alfworld/scripts/run_alfworld_v2_hsi_4b.sh

Exit code 0 means the argv is sound. A non-zero exit prints the offending argv.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if not argv:
        print("argv_preflight: no arguments given", file=sys.stderr)
        return 2
    # `parse_args` reads sys.argv[1:]; give it the vector under test verbatim.
    sys.argv = ["train.py", *argv]
    try:
        from slime.utils.arguments import parse_args

        args = parse_args()
    except SyntaxError as exc:  # the documented dataclass-codegen signature
        print(
            "argv_preflight: FAILED with a SyntaxError while parsing -- this is the "
            f"empty-arg-key failure mode, not a bad value. {exc}",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # noqa: BLE001 - report, never mask
        print(f"argv_preflight: FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 4

    # The sampler triple is the thing this campaign changes; print what slime
    # actually resolved rather than what the launcher asked for.
    print(
        "argv_preflight: OK\n"
        f"  rollout_temperature = {getattr(args, 'rollout_temperature', None)}\n"
        f"  rollout_top_p       = {getattr(args, 'rollout_top_p', None)}\n"
        f"  rollout_top_k       = {getattr(args, 'rollout_top_k', None)}\n"
        f"  n_samples_per_prompt= {getattr(args, 'n_samples_per_prompt', None)}\n"
        f"  save_interval       = {getattr(args, 'save_interval', None)}\n"
        f"  sglang_disable_cuda_graph = {getattr(args, 'sglang_disable_cuda_graph', None)}\n"
        f"  sglang_cuda_graph_max_bs = {getattr(args, 'sglang_cuda_graph_max_bs', None)}\n"
        f"  num_rollout         = {getattr(args, 'num_rollout', None)}\n"
        f"  use_rollout_logprobs= {getattr(args, 'use_rollout_logprobs', None)}\n"
        f"  advantage_estimator= {getattr(args, 'advantage_estimator', None)}\n"
        f"  eps_clip           = {getattr(args, 'eps_clip', None)}\n"
        f"  eps_clip_high      = {getattr(args, 'eps_clip_high', None)}\n"
        f"  lr                 = {getattr(args, 'lr', None)}\n"
        f"  global_batch_size   = {getattr(args, 'global_batch_size', None)}\n"
        f"  custom_model_provider_path = {getattr(args, 'custom_model_provider_path', None)}\n"
        f"  rollout_data_postprocess_path = {getattr(args, 'rollout_data_postprocess_path', None)}"
    )
    import os
    if os.environ.get("OPSD_SRLVR") == "1" and os.environ.get("SRLVR_VARIANT") in ("v2", "v3"):
        from methods.srlvr_hsi import validate_runtime
        cfg = validate_runtime(args)
        print(f"  SRLVR variant={cfg.variant} gamma={cfg.gamma} probes={cfg.calibration} "
              f"reference={args.use_kl_loss} ref_load={args.ref_load} "
              f"kl={args.kl_loss_type}:{args.kl_loss_coef} "
              f"token_normalizer={args.calculate_per_token_loss} "
              f"context_train={args.seq_length} context_inference={args.sglang_context_length} "
              f"thinking={os.environ.get('ALFWORLD_THINKING_STYLE')} "
              f"thinking_tokens={os.environ.get('ALFWORLD_THINKING_TOKENS')} "
              f"response_cap={args.rollout_max_response_len} "
              f"future_compact={os.environ.get('SRLVR_COMPACT_FUTURE', '0')}")
        if os.environ.get("SRLVR_PREFLIGHT_OUTPUT"):
            import json
            from pathlib import Path
            Path(os.environ["SRLVR_PREFLIGHT_OUTPUT"]).write_text(json.dumps(vars(args), default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
