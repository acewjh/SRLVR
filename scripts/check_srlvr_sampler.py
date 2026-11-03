"""Check a live SRLVR SGLang engine: >300-token transport and real cache reuse."""
import argparse
import json
import urllib.request


def check(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = dict(input_ids=[9707, 13]*160, return_logprob=True, top_logprobs_num=0,
                   sampling_params=dict(temperature=1., top_p=1., top_k=-1,
                                        sampling_seed=5321, max_new_tokens=320,
                                        min_new_tokens=320))
    outputs = []
    for _ in range(2):
        request = urllib.request.Request(url.rstrip("/")+"/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
        with opener.open(request, timeout=120) as response:
            outputs.append(json.load(response)["meta_info"])
    keys = ("srlvr_mu", "srlvr_f", "srlvr_e", "srlvr_logp")
    for meta in outputs:
        n = len(meta["output_token_logprobs"])
        assert n == 320, n
        for key in keys:
            assert len(meta[key]) == n, (key, len(meta[key]), n)
        for lp,mu,f,e in zip(*(meta[k] for k in ("srlvr_logp","srlvr_mu","srlvr_f","srlvr_e"))):
            assert abs(f-max(-1,min(0,lp/10))) < 1e-6
            assert abs(e-(f-mu)) < 1e-6
    # Existing engines use global stochastic RNG unless deterministic inference
    # is enabled. Compare the identical initial context's conditional statistic,
    # not different randomly drawn suffixes.
    assert abs(outputs[0]["srlvr_mu"][0]-outputs[1]["srlvr_mu"][0]) < 2e-4
    assert outputs[1]["cached_tokens"] >= 128, outputs[1]["cached_tokens"]
    return dict(passed=True, generated_tokens_per_request=320,
                cached_tokens=[m["cached_tokens"] for m in outputs],
                exact_metadata_count=True, initial_context_mu_matches=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check(args.url)
    print(json.dumps(result))
    if args.output:
        with open(args.output,"w") as handle:
            json.dump(result,handle,indent=2)
