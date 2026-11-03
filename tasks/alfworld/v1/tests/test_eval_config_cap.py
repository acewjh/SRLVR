"""Guards on the generated ``--eval-config`` (cap + frozen sampler).

Two failure modes this file exists to catch, both of which are *silent*:

1. **Truncation cap.** ``write_eval_config`` hardcoded ``max_response_len=1024``.
   At that cap ~22% of ALFWorld turns are cut mid-action, so the reported score
   measures the truncation rather than the policy (CAMPAIGN_QUEUE_1p7b.md §19.2).
   The 2026-10-06 fix (1024 -> 4096) edited the *generated JSON*; a later re-run
   of ``data_prepare.py`` would have quietly reverted it. Pinning the generator
   is what makes the fix durable.
2. **Sampler drift.** The eval sampler is a frozen protocol (T=0.6 / top_p=0.95 /
   top_k=20) that every arm shares. Values that differ per arm turn a fair
   comparison into an apples-to-oranges one, and nothing in the pipeline errors.
"""

import json

import pytest

from tasks.alfworld.v1.data_prepare import write_eval_config

# The eval driver refuses any config below this (eval_pending_hsi.sh:alfworld_cap_check).
MIN_EVAL_CAP = 4096

# Frozen per CAMPAIGN_QUEUE_1p7b.md section 0 -- deliberately != the training
# sampler (T=1.0 / top_p=1.0 / top_k=-1).
FROZEN_EVAL_SAMPLER = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


@pytest.fixture()
def generated_eval_config(tmp_path):
    out = tmp_path / "eval_alfworld.json"
    write_eval_config(str(out), str(tmp_path / "data"))
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_dataset_meets_the_minimum_response_cap(generated_eval_config):
    datasets = generated_eval_config["eval"]["datasets"]

    assert datasets, "eval config must describe at least one dataset"
    for ds in datasets:
        assert ds["max_response_len"] >= MIN_EVAL_CAP, (
            f"{ds['name']}: max_response_len={ds['max_response_len']} truncates "
            f"turns below the {MIN_EVAL_CAP} floor"
        )


def test_eval_sampler_matches_the_frozen_protocol(generated_eval_config):
    for ds in generated_eval_config["eval"]["datasets"]:
        for key, want in FROZEN_EVAL_SAMPLER.items():
            assert ds[key] == want, f"{ds['name']}: {key}={ds[key]!r}, expected {want!r}"


def test_dataset_entries_are_wellformed(generated_eval_config):
    names = [ds["name"] for ds in generated_eval_config["eval"]["datasets"]]

    assert names == ["valid_seen", "valid_unseen"]
    for ds in generated_eval_config["eval"]["datasets"]:
        assert ds["input_key"] == "prompt"
        assert ds["metadata_key"] == "metadata"
        assert ds["path"].endswith(f"data/{ds['name']}.jsonl")
