"""Throughput-sweep load client: sends REAL eval-shaped requests to one or more
SGLang engines with N concurrent connections and measures aggregate decode
throughput from the engine's own decode log within a fixed window.

Requests mirror the formal eval exactly (chat-templated input_ids, formal
sampling params, max_new_tokens=38912) EXCEPT return_logprob=False.

Usage:
  python tests/sweep_client.py --ports 37995 --concurrency 16 --window 600
"""

import argparse
import asyncio
import json
import time

import aiohttp

BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
PROMPTS_FILE = f"{BASE}/repos/opsd-slime/data/eval/aime24_dev30.jsonl"


async def one(session, url, ids, conc_sem, window_end, stats):
    async with conc_sem:
        if time.time() > window_end:
            return
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": -1,
                "max_new_tokens": 38912,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": False,
        }
        try:
            async with session.post(f"{url}/generate", json=payload) as resp:
                await resp.read()
                stats["completed"] += 1
        except Exception:
            stats["errors"] += 1


async def run(urls, conc, window_s, prompts_ids):
    conc_sem = asyncio.Semaphore(conc)
    window_end = time.time() + window_s
    stats = {"completed": 0, "errors": 0}
    conn = aiohttp.TCPConnector(limit=conc * len(urls))
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        tasks = []
        i = 0
        while True:  # keep the load saturated for the whole window
            url = urls[i % len(urls)]
            ids = prompts_ids[i % len(prompts_ids)]
            tasks.append(asyncio.create_task(one(session, url, ids, conc_sem, window_end, stats)))
            i += 1
            if i % (conc * 4) == 0:
                await asyncio.sleep(0.05)
            if time.time() > window_end + 60 and i > conc * 2:
                break
            if i > 400:
                break
        await asyncio.gather(*tasks, return_exceptions=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", type=str, default="37995")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--window", type=int, default=600)
    args = ap.parse_args()

    import pyarrow.parquet as pq  # noqa: F401  (unused but keeps parity)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = [json.loads(l)["problem"] for l in open(PROMPTS_FILE) if l.strip()][:8]
    prompts = [p + "\n\nPlease reason step by step, and put your final answer within \\boxed{}." for p in prompts]
    prompts_ids = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True)
        for p in prompts
    ]
    urls = [f"http://127.0.0.1:{p}" for p in args.ports.split(",")]
    print(f"[sweep] {len(urls)} engine(s), conc={args.concurrency}, window={args.window}s", flush=True)

    t0 = time.time()
    stats = asyncio.run(run(urls, args.concurrency, args.window, prompts_ids))
    print(f"[sweep] done in {time.time()-t0:.0f}s: {stats}", flush=True)
    print(
        "[sweep] read aggregate tok/s + running batch from the engine log "
        "(Decode batch lines within the window)",
        flush=True,
    )


if __name__ == "__main__":
    main()
