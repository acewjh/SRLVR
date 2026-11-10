#!/usr/bin/env python3
"""Convert older Megatron checkpoints to HF and (optionally) reclaim the space.

Policy (see FIX_START / run docs):
  * never touch the newest ``iter_N`` that ``latest_checkpointed_iteration.txt``
    points at -- that is the resume point;
  * convert an older ``iter_M`` to ``hf_iter_M`` (bf16 safetensors) only after
    the trainer has fully landed a newer checkpoint;
  * verify the HF output structurally and with a real ``from_pretrained`` load
    before writing ``hf_iter_M/VERIFIED``;
  * only then, with ``--delete``, remove the Megatron ``iter_M`` directory.

Usage:
  # one pass (cron/watchdog friendly); converts pending old checkpoints
  python scripts/hf_ckpt_rotate.py --run-dir runs/<run> \
      --origin-hf-dir cua/checkpoints/Qwen3-8B --vocab-size 151936 --once --delete
  # detached watcher
  python scripts/hf_ckpt_rotate.py ... --watch --interval 600 --delete
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONVERTER = REPO / "frameworks/slime-ascend/tools/convert_torch_dist_to_hf.py"


# ---------------------------------------------------------------- conversion --
def _install_import_shims() -> None:
    """Make the in-tree converter runnable on the torch 2.1 training env."""
    import torch.distributed.checkpoint as dist_cp

    if not hasattr(dist_cp.state_dict_loader, "_load_state_dict"):
        dist_cp.state_dict_loader._load_state_dict = dist_cp.load_state_dict

    # The converter imports slime's megatron_to_hf package, which pulls a
    # hardware-specific sglang module that cannot import under this torch build.
    # Nothing on the conversion path uses it, so provide a minimal stub.
    try:
        import sglang  # noqa: F401
        return
    except Exception:
        for name in [k for k in sys.modules if k == "sglang" or k.startswith("sglang.")]:
            del sys.modules[name]

    import types

    root = types.ModuleType("sglang"); root.__path__ = []
    srt = types.ModuleType("sglang.srt"); srt.__path__ = []
    utils = types.ModuleType("sglang.srt.utils")
    class MultiprocessingSerializer:  # noqa: D401
        @staticmethod
        def serialize(obj):
            return obj
        @staticmethod
        def deserialize(data):
            return data
    utils.MultiprocessingSerializer = MultiprocessingSerializer
    patch = types.ModuleType("sglang.srt.utils.patch_torch")
    patch.monkey_patch_torch_reductions = lambda: None
    weight_sync = types.ModuleType("sglang.srt.weight_sync"); weight_sync.__path__ = []
    bucket = types.ModuleType("sglang.srt.weight_sync.tensor_bucket")
    class FlattenedTensorBucket:  # noqa: D401
        pass
    bucket.FlattenedTensorBucket = FlattenedTensorBucket
    sys.modules.update({
        "sglang": root, "sglang.srt": srt, "sglang.srt.utils": utils,
        "sglang.srt.utils.patch_torch": patch,
        "sglang.srt.weight_sync": weight_sync,
        "sglang.srt.weight_sync.tensor_bucket": bucket,
    })


def _patched_converter(tmp_dir: Path) -> Path:
    """Write a copy of the converter with the torch 2.1 checkpoint-API fixes."""
    src = CONVERTER.read_text()
    old_read = '''    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata'''
    new_read = '''    @override
    def read_metadata(self):
        path = os.path.join(self.path, ".metadata")
        with open(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "planner_data", None) is None:
            metadata.planner_data = {}
        return metadata'''
    if old_read in src:
        src = src.replace(old_read, new_read, 1)
    src = src.replace("dist_cp.state_dict_loader._load_state_dict(",
                      "dist_cp.load_state_dict(")
    out = tmp_dir / "convert_torch_dist_to_hf_torch21.py"
    out.write_text(src)
    return out


def convert(iter_dir: Path, out_dir: Path, origin_hf: Path, vocab_size: int) -> None:
    """Run the conversion in-process (invoked as a subprocess by the watcher)."""
    _install_import_shims()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hfconv_") as tmp:
        tool = _patched_converter(Path(tmp))
        sys.argv = [str(tool), "--input-dir", str(iter_dir), "--output-dir", str(out_dir),
                    "--origin-hf-dir", str(origin_hf), "--vocab-size", str(vocab_size),
                    "--force"]
        import runpy
        runpy.run_path(str(tool), run_name="__main__")


# --------------------------------------------------------------- verification --
def verify_hf(out_dir: Path, origin_hf: Path) -> dict:
    import torch
    import safetensors.torch
    from transformers import AutoConfig, AutoModelForCausalLM

    index = out_dir / "model.safetensors.index.json"
    if not index.is_file():
        # single-shard checkpoints have no index
        shards = sorted(out_dir.glob("model*.safetensors"))
        if len(shards) != 1:
            raise RuntimeError(f"{out_dir}: no index and {len(shards)} shards")
        weight_map = {"": shards[0].name}
    else:
        weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    total = 0
    for name in shards:
        path = out_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing/empty shard {path}")
        total += path.stat().st_size
    cfg = AutoConfig.from_pretrained(origin_hf, trust_remote_code=True)
    expected = cfg.num_hidden_layers * cfg.hidden_size * cfg.hidden_size * cfg.vocab_size  # loose lower bound
    if total < 0.5 * cfg.vocab_size * cfg.hidden_size * 2:
        raise RuntimeError(f"{out_dir}: shards too small ({total} bytes)")
    model = AutoModelForCausalLM.from_pretrained(
        str(out_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    params = sum(p.numel() for p in model.parameters())
    emb = model.model.embed_tokens.weight
    finite = bool(torch.isfinite(emb.float()).all())
    del model
    if not finite or params <= 0:
        raise RuntimeError(f"{out_dir}: invalid HF weights (finite={finite}, params={params})")
    return dict(shards=shards, bytes=total, params=params, finite=finite)


# -------------------------------------------------------------------- watcher --
def latest_iter(run_dir: Path) -> int:
    tracker = run_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return -1
    digits = "".join(ch for ch in tracker.read_text().strip() if ch.isdigit())
    return int(digits) if digits else -1


def pending_iters(run_dir: Path, latest: int) -> list[int]:
    found = []
    for path in run_dir.glob("iter_*"):
        if not path.is_dir():
            continue
        digits = "".join(ch for ch in path.name.split("_")[1] if ch.isdigit())
        if not digits:
            continue
        it = int(digits)
        if it < latest:
            found.append(it)
    return sorted(found)


def log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "hf_rotate.log").open("a") as fh:
        fh.write(line + "\n")


def rotate_once(run_dir: Path, origin_hf: Path, vocab_size: int, delete: bool) -> None:
    latest = latest_iter(run_dir)
    if latest < 0:
        return
    for it in pending_iters(run_dir, latest):
        src = run_dir / f"iter_{it:07d}"
        out = run_dir / f"hf_iter_{it}"
        marker = out / "VERIFIED"
        if not src.is_dir():
            continue
        if marker.is_file():
            if delete:
                shutil.rmtree(src)
                log(run_dir, f"deleted {src.name} (HF verified earlier)")
            continue
        if out.exists():
            shutil.rmtree(out)
        log(run_dir, f"converting {src.name} -> {out.name}")
        started = time.time()
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()),
                              "--convert", str(src), str(out),
                              "--origin-hf-dir", str(origin_hf),
                              "--vocab-size", str(vocab_size)])
        if rc != 0:
            log(run_dir, f"conversion FAILED rc={rc} for {src.name}; keeping Megatron checkpoint")
            continue
        try:
            info = verify_hf(out, origin_hf)
        except Exception as exc:  # noqa: BLE001
            log(run_dir, f"verification FAILED for {out.name}: {exc}; keeping Megatron checkpoint")
            continue
        marker.write_text(json.dumps(dict(source=src.name, verified_at=time.time(),
                                          elapsed_s=round(time.time() - started, 1), **info)) + "\n")
        log(run_dir, f"verified {out.name}: {info['params']/1e9:.2f}B params, "
                     f"{info['bytes']/2**30:.1f} GiB, {time.time()-started:.0f}s")
        if delete:
            freed = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
            shutil.rmtree(src)
            log(run_dir, f"deleted {src.name} (freed {freed/2**30:.1f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=False)
    parser.add_argument("--origin-hf-dir", required=False)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--delete", action="store_true", help="remove the Megatron dir after HF verification")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--convert", nargs=2, metavar=("INPUT_DIR", "OUTPUT_DIR"))
    args = parser.parse_args()

    if args.convert:
        convert(Path(args.convert[0]), Path(args.convert[1]),
                Path(args.origin_hf_dir), args.vocab_size)
        return 0

    run_dir = Path(args.run_dir).resolve()
    origin_hf = Path(args.origin_hf_dir).resolve()
    lock_path = run_dir / ".hf_rotate.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(run_dir, "another rotation holds the lock; exiting")
            return 0
        if args.once or not args.watch:
            rotate_once(run_dir, origin_hf, args.vocab_size, args.delete)
            return 0
        log(run_dir, f"watcher start interval={args.interval}s delete={args.delete}")
        while True:
            try:
                rotate_once(run_dir, origin_hf, args.vocab_size, args.delete)
            except Exception as exc:  # noqa: BLE001
                log(run_dir, f"rotation error: {exc}")
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
