"""Generation client for the SGLang rollout smoke (see run_sglang_smoke.sh).

Uses the exact student prompt construction of the OPSD harness (tasks.math.prompts)
and the official rollout sampling params, and verifies: token ids returned, text
decodes, EOS respected, no NaN/inf, throughput. Writes reports/sglang_smoke.md.
"""

import json
import os
import time
import urllib.request

from transformers import AutoTokenizer

from tasks.math.prompts import build_student_prompt_ids
from tasks.math.math_data import convert_samples_to_train_data  # noqa: F401  (import sanity)
from methods.opsd_loss import generalized_jsd_per_token  # noqa: F401  (import sanity)

BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
DATA = f"{BASE}/data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet"
PORT = os.environ.get("PORT", "37881")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "8"))

import pyarrow.parquet as pq


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    table = pq.read_table(DATA, columns=["problem"])
    problems = table.column("problem").to_pylist()[:NUM_PROMPTS]
    assert len(problems) == NUM_PROMPTS

    prompts = [build_student_prompt_ids(tokenizer, p) for p in problems]
    prompt_lens = [len(p) for p in prompts]

    payload_prompts = [
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 1.1,
                "top_p": 0.95,
                "top_k": 20,
                "max_new_tokens": MAX_TOKENS,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": True,
        }
        for ids in prompts
    ]

    url = f"http://127.0.0.1:{PORT}/generate"
    t0 = time.time()
    outputs = []
    for payload in payload_prompts:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            outputs.append(json.loads(resp.read()))
    elapsed = time.time() - t0

    report = []
    report.append("# SGLang rollout smoke (Phase 3)")
    report.append("")
    report.append(f"- model: `{MODEL}`")
    report.append(f"- engine: TP1, text-only, `--device npu --attention-backend ascend`, NPU 7")
    report.append(f"- sampling: temperature=1.1, top_p=0.95, top_k=20, max_new_tokens={MAX_TOKENS} (official rollout params)")
    report.append(f"- prompts: {NUM_PROMPTS} (student prompt = chat template, thinking OFF)")
    report.append("")

    ok = True
    total_new_tokens = 0
    eos_id = tokenizer.eos_token_id
    for i, out in enumerate(outputs):
        meta = out["meta_info"]
        token_logprobs = meta.get("output_token_logprobs", [])
        new_ids = [t[1] for t in token_logprobs]
        new_tokens = len(new_ids)
        total_new_tokens += new_tokens
        text = out["text"]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        text_matches = (decoded == text) or (decoded.strip() == text.strip())
        hit_eos = any(tid == eos_id for tid in new_ids)
        finite = all(
            lp[0] == lp[0] and abs(lp[0]) != float("inf") for lp in token_logprobs
        )
        finished = meta.get("finish_reason", {}).get("type")
        entry_ok = new_tokens > 0 and text_matches and finite
        ok = ok and entry_ok
        report.append(
            f"| {i} | {prompt_lens[i]} | {new_tokens} | {finished} | eos={hit_eos} | decode_match={text_matches} | finite={finite} |"
        )
        if i == 0:
            report.append("")
            report.append(f"sample completion (prompt 0, first 300 chars): `{text[:300]}`")
            report.append("")

    report.insert(6, "| prompt | prompt_len | new_tokens | finish | eos | decode==text | finite_logprobs |")
    report.insert(7, "|---|---|---|---|---|---|---|")

    total_time = elapsed
    tok_per_s = total_new_tokens / total_time if total_time > 0 else 0.0
    report.append("")
    report.append(f"- total new tokens: {total_new_tokens}")
    report.append(f"- wall-clock (sequential {NUM_PROMPTS} requests): {total_time:.2f}s -> {tok_per_s:.1f} tok/s")
    report.append(f"- checks: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    report.append("- CUDA fallback: N/A (torch_npu-only image; sglang launched with --device npu)")
    report.append("- peak HBM: see npu-smi output in logs/sglang_smoke_client.log")

    out_path = f"{BASE}/repos/opsd-slime/reports/sglang_smoke.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
