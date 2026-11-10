#!/usr/bin/env python3
"""Lightweight verification for the OpenApps install + Qwen2.5-VL-3B checkpoint.

Two independent checks, each failing loudly with a non-zero exit code:

  1. OpenApps: import the package, compose its Hydra config, instantiate the
     launcher, and instantiate a real built-in task from ``all_tasks.yaml``.
  2. Qwen2.5-VL-3B-Instruct: load ``config.json`` + tokenizer through
     transformers and verify every tensor named by
     ``model.safetensors.index.json`` is present in the shard the index says it
     lives in (headers only -- the 7.5 GB of weights are never materialised).

Usage:
    python scripts/verify_openapps_qwen25vl.py [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL_DIR = Path(
    "/workspace/cua/checkpoints/Qwen2.5-VL-3B-Instruct"
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def verify_openapps() -> None:
    print("== OpenApps ==")
    import open_apps

    print(f"  package: {open_apps.__file__}")
    _check(callable(getattr(open_apps, "hello", None)), "open_apps imported")

    from open_apps import config_dir

    cfg_dir = Path(config_dir())
    _check((cfg_dir / "config.yaml").is_file(), f"hydra config present: {cfg_dir}")

    # Compose the real config and construct the launcher (no servers started).
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from open_apps.launcher import OpenAppsLauncher

    with tempfile.TemporaryDirectory(prefix="openapps_verify_") as tmp:
        logs_dir = Path(tmp) / "logs"
        with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"logs_dir={logs_dir}",
                    "use_wandb=False",
                    "+web_app_port=5099",
                ],
            )
        OmegaConf.resolve(cfg)
        launcher = OpenAppsLauncher(cfg)
        _check(launcher.web_app_url.startswith("http://"), "OpenAppsLauncher instantiated")
        _check(launcher.config.logs_dir == str(logs_dir.resolve()), "logs_dir resolved")

    # Instantiate a real built-in task -- this exercises the task registry and
    # the task classes, not just module imports.
    from open_apps.tasks import list_task_keys, load_task

    keys = list_task_keys()
    _check(len(keys) > 0, f"task registry has {len(keys)} tasks")
    task = load_task(keys[0])
    _check(task is not None and task.__class__.__name__, f"instantiated task {keys[0]!r} -> {task.__class__.__name__}")

    # The FastAPI surface the browser agents talk to must import too.
    from open_apps.apps.start_page.main import app

    _check(app is not None, "start-page FastAPI app imported")


def verify_model(model_dir: Path) -> None:
    print(f"== Qwen2.5-VL-3B-Instruct: {model_dir} ==")
    _check(model_dir.is_dir(), "model directory exists")

    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        _check((model_dir / name).is_file(), f"{name} present")

    # ---- config ----
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_dir))
    arch = (config.architectures or [None])[0]
    _check(arch == "Qwen2_5_VLForConditionalGeneration", f"architecture {arch}")
    _check(config.model_type == "qwen2_5_vl", f"model_type {config.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    probe = "OpenApps: click the To Do app."
    ids = tokenizer(probe)["input_ids"]
    _check(len(ids) > 0, f"tokenizer encodes probe -> {len(ids)} ids")
    _check(tokenizer.decode(ids) is not None, "tokenizer decodes")

    # ---- weights: index vs. safetensors headers ----
    index_path = model_dir / "model.safetensors.index.json"
    _check(index_path.is_file(), "model.safetensors.index.json present")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    expected_total = int(index.get("metadata", {}).get("total_size", 0))
    _check(len(weight_map) > 0, f"index names {len(weight_map)} tensors")

    from safetensors import safe_open

    dtype_bytes = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    }
    shard_files = sorted(set(weight_map.values()))
    print(f"  shards: {shard_files}")
    actual_total = 0
    keys_in_shards: dict[str, set[str]] = {}
    unknown_dtypes: set[str] = set()
    for shard in shard_files:
        shard_path = model_dir / shard
        _check(shard_path.is_file(), f"shard {shard} present")
        with safe_open(str(shard_path), framework="pt") as f:
            keys_in_shards[shard] = set(f.keys())
            for key in f.keys():
                sl = f.get_slice(key)
                n = 1
                for dim in sl.get_shape():
                    n *= dim
                dtype = str(sl.get_dtype())
                if dtype not in dtype_bytes:
                    unknown_dtypes.add(dtype)
                    continue
                actual_total += n * dtype_bytes[dtype]
    _check(not unknown_dtypes, f"all tensor dtypes known (unknown={sorted(unknown_dtypes)})")

    missing = [k for k, shard in weight_map.items() if k not in keys_in_shards.get(shard, set())]
    _check(not missing, f"all indexed tensors present (missing={len(missing)})")
    extra = [
        k
        for shard, keys in keys_in_shards.items()
        for k in keys
        if weight_map.get(k) != shard
    ]
    _check(not extra, f"no unindexed tensors (extra={len(extra)})")
    if expected_total:
        _check(
            actual_total == expected_total,
            f"byte total matches index ({actual_total} == {expected_total})",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--skip-openapps", action="store_true", help="only verify the model checkpoint"
    )
    parser.add_argument(
        "--skip-model", action="store_true", help="only verify the OpenApps install"
    )
    args = parser.parse_args()

    if not args.skip_openapps:
        verify_openapps()
    if not args.skip_model:
        verify_model(args.model_dir)
    print("\nVERIFY_OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - report the failure, exit non-zero
        print(f"\nVERIFY_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
