"""No-logprob fast-path gate (directive: eval-only return_logprob=False).

Runs against a STANDALONE smoke engine (separate NPU/port/log dir, base weights)
while the formal eval keeps running untouched. The return_logprob flag is a
weight-independent reporting switch (statically verified: it gates ONLY the
logprob reporting/calculation paths, never the sampler), so gate equivalence on
base weights transfers to the LoRA checkpoint.

Gates:
  A. deterministic exact: temperature=0, max_new_tokens=256 -> token IDs must
     match exactly between return_logprob=True and False.
  B. formal sampling: temperature=1.0/top_p=0.95/top_k=-1 + per-request
     sampling_seed, max_new_tokens=1024, sequential (conc=1) -> token IDs must
     match exactly under the same seed.
  T. throughput: max_new_tokens=2048, concurrency 1/2/4, logprob True vs False ->
     wall-time, tok/s, peak HBM (npu-smi poll on the smoke NPU).

Usage: python tests/nologprob_gate_smoke.py --port 37990
"""

import argparse
import json
import subprocess
import time
import urllib.request

BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"

PROMPTS = [
    "Solve: find the sum of all positive divisors of 720. Reason step by step.",
    "Prove that the square root of 2 is irrational. Give a complete argument.",
    "Compute the determinant of a 3x3 matrix with entries 1..9 and explain.",
    "Find all triples (a,b,c) of positive integers with a!+b!+c! = 100. Explain.",
]


def post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen(base_url, prompt, max_new, logprob, seed=None, temperature=1.0, top_p=0.95, top_k=-1):
    sp = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    if seed is not None:
        sp["sampling_seed"] = seed
    payload = {
        "text": prompt,
        "sampling_params": sp,
        "return_logprob": logprob,
    }
    t0 = time.time()
    d = post(f"{base_url}/generate", payload)
    dt = time.time() - t0
    meta = d.get("meta_info", {})
    ids = meta.get("output_ids") or []
    if not ids:
        # fall back: some forks put output_ids at top level
        ids = d.get("output_ids", [])
    ntok = meta.get("completion_tokens", len(ids))
    return {
        "ids": ids,
        "text": d.get("text", ""),
        "finish": meta.get("finish_reason", {}).get("type")
        if isinstance(meta.get("finish_reason"), dict)
        else meta.get("finish_reason"),
        "ntok": ntok,
        "dt": dt,
        "tok_s": ntok / dt if dt > 0 else 0.0,
    }


def npu_peak_hbm(npu_id, window_s, interval=2.0):
    """Poll npu-smi for peak HBM on one NPU during a window."""
    peak = 0
    t_end = time.time() + window_s
    while time.time() < t_end:
        try:
            out = subprocess.run(
                ["npu-smi", "info", "-t", "usage", "-i", str(npu_id)],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.splitlines():
                if "HBM" in line or "Memory" in line:
                    for tok in line.replace("|", " ").split():
                        if tok.isdigit() and 1000 < int(tok) <= 32768:
                            peak = max(peak, int(tok))
        except Exception:
            pass
        time.sleep(interval)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=37990)
    ap.add_argument("--npu", type=int, default=0)
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--max-new", type=int, default=2048)
    args = ap.parse_args()
    base_url = f"http://127.0.0.1:{args.port}"

    if not args.skip_engine:
        print("[smoke] engine expected at", base_url)

    results = {}

    # ---------- Gate A: deterministic exact (temperature=0) ----------
    print("\n=== Gate A: deterministic exact (T=0, 256 tokens) ===", flush=True)
    ok_a = True
    for pi, prompt in enumerate(PROMPTS[:2]):
        a = gen(base_url, prompt, 256, logprob=True, temperature=0.0)
        b = gen(base_url, prompt, 256, logprob=False, temperature=0.0)
        match = a["ids"] == b["ids"] and a["text"] == b["text"] and a["finish"] == b["finish"]
        ok_a &= match
        print(
            f"  prompt{pi}: ids_match={a['ids'] == b['ids']} len={len(a['ids'])}/{len(b['ids'])} "
            f"finish={a['finish']}/{b['finish']} text_match={a['text'] == b['text']}",
            flush=True,
        )
    results["gate_a"] = ok_a

    # ---------- Gate B: formal sampling + request-level seed ----------
    print("\n=== Gate B: formal sampling (T=1.0/top_p0.95, seed, 1024 tokens, conc=1) ===", flush=True)
    ok_b = True
    for pi, prompt in enumerate(PROMPTS[:2]):
        a = gen(base_url, prompt, 1024, logprob=True, seed=1000 + pi)
        b = gen(base_url, prompt, 1024, logprob=False, seed=1000 + pi)
        match = a["ids"] == b["ids"] and a["finish"] == b["finish"]
        ok_b &= match
        print(
            f"  prompt{pi} seed{1000+pi}: ids_match={a['ids'] == b['ids']} "
            f"len={len(a['ids'])}/{len(b['ids'])} finish={a['finish']}/{b['finish']}",
            flush=True,
        )
    results["gate_b"] = ok_b

    # ---------- Throughput T: conc 1/2/4 x logprob T/F ----------
    print("\n=== Throughput (max_new=%d) ===" % args.max_new, flush=True)
    prompts = PROMPTS * 4  # 16 requests max per arm
    for conc in (1, 2, 4):
        import concurrent.futures as cf

        for logprob in (True, False):
            # warmup 1 request
            gen(base_url, prompts[0], 128, logprob=logprob)
            peak_hbm = None
            import threading

            poll_stop = threading.Event()
            poll_res = {}

            def poll():
                poll_res["hbm"] = npu_peak_hbm(args.npu, 10**9, 2.0)

            th = threading.Thread(target=poll, daemon=True)
            t0 = time.time()
            th.start()
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [
                    ex.submit(gen, base_url, p, args.max_new, logprob=logprob)
                    for p in prompts[: max(conc, 4)]
                ]
                outs = [f.result() for f in futs]
                poll_stop.set()
                dt = time.time() - t0
            th.join(timeout=5)
            ntok = sum(o["ntok"] for o in outs)
            results[f"conc{conc}_lp{int(logprob)}"] = {
                "agg_tok_s": ntok / dt,
                "n_req": len(outs),
                "dt": dt,
                "per_stream": (ntok / dt) / conc,
            }
            print(
                f"  conc={conc} logprob={logprob}: n={len(outs)} agg={ntok/dt:.0f} tok/s "
                f"({dt:.0f}s) per-stream={ntok/dt/conc:.0f}",
                flush=True,
            )

    print("\n=== SUMMARY ===")
    print(
        json.dumps(
            {k: v for k, v in results.items() if not isinstance(v, bool)},
            indent=1,
        )
    )
    print(f"Gate A: {'PASS' if results['gate_a'] else 'FAIL'}")
    print(f"Gate B: {'PASS' if results['gate_b'] else 'FAIL'}")


if __name__ == "__main__":
    main()
