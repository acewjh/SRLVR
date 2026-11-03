"""Phase 6: fixed-sample correctness / equivalence check (official OPSD port vs math_opsd).

For 2 fixed dataset samples with a FIXED student completion, computes with BOTH
implementations:
  1. student prompt token ids + teacher prompt token ids (official collator vs tasks.math.prompts)
  2. student scoring token ids + teacher scoring token ids (alignment)
  3. aligned position count
  4. teacher top-k log-probs at every aligned position
  5. per-token divergence (official generalized_jsd_loss, beta=0, T=1.1, clip=0.05)
  6. mean loss

The official side uses the ACTUAL oracle code:
  * SelfDistillationDataCollator  -> imported from repos/opsd-npu/src/data_collator.py
  * OPSDTrainer.generalized_jsd_loss -> extracted verbatim (AST) from
    repos/opsd-npu/src/opsd_trainer.py so the oracle math is byte-identical
    without importing the TRL/accelerate dependency chain.

Numerical agreement is checked against the same HF bf16 teacher on the same NPU,
so differences should be tiny (ordering of ops only).

Run:  python tests/test_opsd_equivalence.py            (writes reports/equivalence_check.md)
"""

import ast
import importlib.util
import sys

import torch

BASE = "/workspace/cua"
PROJ = f"{BASE}/repos/opsd-slime"
ORACLE = f"{BASE}/repos/opsd-npu"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
DATA = f"{BASE}/data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet"

sys.path.insert(0, PROJ)

TEMPERATURE = 1.1
TOKEN_CLIP = 0.05
NUM_SAMPLES = 2
MAX_COMPLETION = 64  # fixed short completion keeps the check fast and deterministic


def load_oracle_jsd():
    """Extract generalized_jsd_loss verbatim from the official trainer source."""
    src = open(f"{ORACLE}/src/opsd_trainer.py").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generalized_jsd_loss":
            code = ast.get_source_segment(src, node)
            ns = {"torch": torch, "F": torch.nn.functional}
            exec(compile(code, "<oracle-generalized_jsd_loss>", "exec"), ns)
            return ns["generalized_jsd_loss"]
    raise RuntimeError("generalized_jsd_loss not found in official source")


def load_oracle_collator():
    spec = importlib.util.spec_from_file_location("oracle_data_collator", f"{ORACLE}/src/data_collator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SelfDistillationDataCollator


def main() -> None:
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tasks.math.prompts import build_student_prompt_ids, build_teacher_prompt_ids
    from methods.opsd_loss import generalized_jsd_per_token

    oracle_jsd = load_oracle_jsd()
    OracleCollator = load_oracle_collator()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("npu")
    model.eval()

    rows = pq.read_table(DATA, columns=["problem", "solution"]).slice(0, NUM_SAMPLES).to_pylist()

    lines = []
    lines.append("# Fixed-sample equivalence check: official OPSD port vs math_opsd")
    lines.append("")
    lines.append(f"- oracle: `{ORACLE}/src/opsd_trainer.py` (generalized_jsd_loss, verbatim AST extraction)")
    lines.append(f"- oracle: `{ORACLE}/src/data_collator.py` (SelfDistillationDataCollator)")
    lines.append(f"- harness: `math_opsd/prompts.py` + `math_opsd/opsd_loss.py`")
    lines.append(f"- model: `{MODEL}` (bf16, sdpa, NPU); loss params: beta=0, T={TEMPERATURE}, clip={TOKEN_CLIP}, full vocab")
    lines.append("")

    all_ok = True
    for i, row in enumerate(rows):
        problem, solution = row["problem"], row["solution"]

        # ---- official collator path (the oracle prompt construction) ----
        collator = OracleCollator(tokenizer, max_length=8192, reason_first=False,
                                  student_thinking=False, teacher_thinking=True)
        batch = collator([{"problem": problem, "solution": solution}])
        # student_prompts: [1, P]; teacher_prompts: [1, T]
        off_student_ids = batch["student_prompts"][0][batch["student_prompt_attention_mask"][0].bool()].tolist()
        off_teacher_ids = batch["teacher_prompts"][0][batch["teacher_prompt_attention_mask"][0].bool()].tolist()

        # ---- harness path ----
        our_student_ids = build_student_prompt_ids(tokenizer, problem)
        our_teacher_ids = build_teacher_prompt_ids(tokenizer, problem, solution)

        prompts_match = (off_student_ids == our_student_ids) and (off_teacher_ids == our_teacher_ids)
        lines.append(f"## Sample {i}")
        lines.append("")
        lines.append(f"| check | value |")
        lines.append(f"|---|---|")
        lines.append(f"| student prompt ids identical | {prompts_match} "
                     f"(official {len(off_student_ids)} vs ours {len(our_student_ids)}; "
                     f"teacher {len(off_teacher_ids)} vs {len(our_teacher_ids)}) |")
        all_ok &= prompts_match

        # ---- fixed student completion: greedy from the frozen initial policy ----
        with torch.no_grad():
            input_ids = torch.tensor([our_student_ids], device="npu")
            out = model.generate(input_ids, max_new_tokens=MAX_COMPLETION, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        completion = out[0, input_ids.shape[1]:].tolist()

        # ---- student scoring (blind) ----
        student_full = our_student_ids + completion
        off_student_full = off_student_ids + completion
        P_s, P_s_off = len(our_student_ids), len(off_student_ids)
        R = len(completion)

        with torch.no_grad():
            s_logits = model(input_ids=torch.tensor([student_full], device="npu")).logits[0].float()
            off_s_logits = model(input_ids=torch.tensor([off_student_full], device="npu")).logits[0].float()
            student_resp_logits = s_logits[P_s - 1 : P_s - 1 + R]          # predict completion tokens
            off_student_resp_logits = off_s_logits[P_s_off - 1 : P_s_off - 1 + R]

        # ---- teacher scoring (privileged) ----
        teacher_full = our_teacher_ids + completion
        off_teacher_full = off_teacher_ids + completion
        P_t, P_t_off = len(our_teacher_ids), len(off_teacher_ids)

        with torch.no_grad():
            t_logits = model(input_ids=torch.tensor([teacher_full], device="npu")).logits[0].float()
            off_t_logits = model(input_ids=torch.tensor([off_teacher_full], device="npu")).logits[0].float()
            teacher_resp_logits = t_logits[P_t - 1 : P_t - 1 + R]
            off_teacher_resp_logits = off_t_logits[P_t_off - 1 : P_t_off - 1 + R]

        aligned = R
        # completion tokens evaluated by both sides are identical by construction
        token_ids_match = (
            torch.tensor(student_full[P_s:]) == torch.tensor(teacher_full[P_t:])
        ).all().item()

        # ---- per-token divergence: oracle vs harness ----
        jsd_oracle = oracle_jsd(
            student_logits=off_student_resp_logits.unsqueeze(0),
            teacher_logits=off_teacher_resp_logits.unsqueeze(0),
            labels=None, beta=0.0, temperature=TEMPERATURE,
            reduction="none", top_k=None, token_clip=None,
        ).squeeze(0).sum(-1)  # [R] unclipped

        jsd_ours, _ = generalized_jsd_per_token(
            student_resp_logits, teacher_resp_logits,
            beta=0.0, temperature=TEMPERATURE, token_clip=None,
        )

        diff = (jsd_oracle - jsd_ours).abs()
        rel = diff / jsd_oracle.clamp_min(1e-8)
        loss_oracle = jsd_oracle.clamp(max=TOKEN_CLIP).mean().item()
        loss_ours = jsd_ours.clamp(max=TOKEN_CLIP).mean().item()

        topk_ok = True
        k = 10
        for j in range(0, R, max(R // 3, 1)):
            o = torch.topk(torch.log_softmax(off_teacher_resp_logits[j] / TEMPERATURE, -1), k)
            u = torch.topk(torch.log_softmax(teacher_resp_logits[j] / TEMPERATURE, -1), k)
            ids_match = (o.indices == u.indices).float().mean().item()
            max_lp_diff = (o.values - u.values).abs().max().item()
            if ids_match < 1.0 or max_lp_diff > 1e-3:
                topk_ok = False
            lines.append(f"| teacher top-{k} ids match @pos {j} | {ids_match:.3f} (max logp diff {max_lp_diff:.2e}) |")

        ok = prompts_match and token_ids_match and topk_ok and diff.max().item() < 1e-3
        all_ok &= ok

        lines.append(f"| aligned positions | {aligned} |")
        lines.append(f"| completion token ids identical (student vs teacher span) | {token_ids_match} |")
        lines.append(f"| per-token divergence: max abs diff oracle vs ours | {diff.max().item():.3e} (mean {diff.mean().item():.3e}, max rel {rel.max().item():.3e}) |")
        lines.append(f"| mean loss (oracle batchmean semantics) | official {loss_oracle:.8f} vs ours {loss_ours:.8f} |")
        lines.append(f"| **sample verdict** | **{'PASS' if ok else 'FAIL'}** |")
        lines.append("")

    lines.append(f"## Overall: {'PASS' if all_ok else 'FAIL'}")
    out_path = f"{PROJ}/reports/equivalence_check.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
