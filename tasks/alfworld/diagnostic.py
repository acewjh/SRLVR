"""Frozen Full-vs-Control diagnostic -- no training, no re-sampling, no optimizer.

Question this answers
---------------------
The v2 pilot measured `e_t ~ 0` on the action span and nobody could say whether that
means (a) the delayed future trace genuinely carries little information for the
current decision, or (b) the measurement is conditioned away. This script separates
the two by scoring FOUR conditions against the SAME frozen trajectory, the same
model, the same reward and the same target tokens.

Conditions (identical scaffold, identical outcome, identical target; only the
privileged body changes):

======  ======================================================================
C       CONTROL -- no future trace
T       TRUE_GAP1 -- the real future, transitions ``i >= t+1``
S       SHUFFLED_GAP1 -- a MATCHED sibling episode's future, ``i >= t+1``
P       TRUE_GAP0 -- the real future INCLUDING transition ``t`` (deliberate
        leak, positive control: proves whether the teacher can use a future
        block at all). NEVER used for training.
======  ======================================================================

Spans, because the action tokens are already conditioned on the turn's own
``<think>``, which may wash out the effect:

* ``think``  -- the tokens inside ``<think>...</think>``
* ``action`` -- the tokens inside ``<action>...</action>``
* ``full``   -- every generated token (think + action)

The target token ids are byte-identical in every condition, so the differences are
purely the teacher's conditioning.

Output: one JSONL line per (turn, condition) plus a summary, so the numbers can be
re-analysed without re-running. Nothing here touches weights.

Usage::

    python -m tasks.alfworld.diagnostic --dump runs/<run>/rollout_0.pt \
        --out /tmp/hpi_diag.jsonl --conditions C,T,S,P
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

__all__ = ["build_conditions", "content_span", "main", "score_conditions"]


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def content_span(tokenizer, text: str, ids: list[int], tag: str) -> tuple[int, int] | None:
    """Token span of a tag's CONTENT (``think`` / ``action``), verified by re-encoding."""
    if not text or not ids:
        return None
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    open_idx = text.find(open_tag)
    close_idx = text.rfind(close_tag)
    if open_idx == -1 or close_idx <= open_idx:
        return None
    start_char = open_idx + len(open_tag)
    end_char = close_idx
    if end_char <= start_char:
        return None
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if list(enc["input_ids"]) != list(ids):
        return None
    offsets = list(enc["offset_mapping"])
    span = [i for i, (s, e) in enumerate(offsets) if s < end_char and e > start_char]
    return (span[0], span[-1] + 1) if span else None


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def _matched_sibling(trajectory, pool, *, max_tail: int):
    """A sibling whose category, outcome and remaining horizon match.

    Matching matters: an unmatched shuffled control would confound "the teacher
    reacted to a different future" with "the teacher reacted to a SUCCESS instead
    of a FAILURE", which is exactly the outcome effect the scaffold is supposed to
    hold constant.
    """
    best = None
    for candidate in pool:
        if candidate is trajectory:
            continue
        if candidate.task_type != trajectory.task_type:
            continue
        if candidate.success != trajectory.success:
            continue
        if not candidate.turns:
            continue
        remaining = len(candidate.turns)
        key = abs(remaining - max_tail)
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1] if best else None


def build_conditions(tokenizer, trajectory, turn_idx: int, generated_ids, generated_text,
                     *, history_length: int, sibling=None):
    """The four prompt prefixes for one turn. Returns ``{name: (prompt_ids, spans)}``."""
    from .privilege import _prompt_ids, render_hpi_block
    from .privilege import view_for_turn

    view = view_for_turn(trajectory, turn_idx, history_length)
    base_user = None

    def prefix_for(block: str) -> list[int]:
        nonlocal base_user
        if base_user is None:
            from .prompts import build_turn_user_message

            base_user = build_turn_user_message(view)
        return _prompt_ids(tokenizer, base_user + ("\n\n" + block if block else ""))

    spans = {
        "think": content_span(tokenizer, generated_text, generated_ids, "think"),
        "action": content_span(tokenizer, generated_text, generated_ids, "action"),
        "full": (0, len(generated_ids)),
    }

    out = {}
    out["C"] = (prefix_for(render_hpi_block(trajectory, turn_idx, with_future=False, future_gap=1)), spans)
    out["T"] = (prefix_for(render_hpi_block(trajectory, turn_idx, with_future=True, future_gap=1)), spans)
    if sibling is not None:
        out["S"] = (
            prefix_for(
                render_hpi_block(trajectory, turn_idx, with_future=True, future_gap=1, trace_source=sibling)
            ),
            spans,
        )
    # gap=0: deliberately includes transition t (the evaluated action's own result).
    out["P"] = (prefix_for(render_hpi_block(trajectory, turn_idx, with_future=True, future_gap=0)), spans)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_conditions(teacher, tokenizer, prefix_ids, target_ids, spans, chunk=8):
    """Teacher-force one target under one condition; return per-span mean logprob."""
    import torch

    from opsd_core.scoring import score_view_logprobs

    plen = len(prefix_ids)
    logprobs = score_view_logprobs(teacher, [(list(prefix_ids) + list(target_ids), plen, len(target_ids))], chunk)
    lp = logprobs[0]
    out = {}
    for name, span in spans.items():
        if span is None:
            out[name] = None
            continue
        start, end = span
        if end > lp.numel():
            out[name] = None
            continue
        out[name] = float(lp[start:end].mean().item())
    return out


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {}
    ordered = sorted(values)
    q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": max(values),
        "mean_abs": statistics.mean(abs(v) for v in values),
        "p90_abs": sorted(abs(v) for v in values)[min(len(values) - 1, int(0.9 * len(values)))],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen Full-vs-Control HPI diagnostic.")
    parser.add_argument("--dump", required=True, help="frozen rollout dump (has metadata['trajectory'])")
    parser.add_argument("--out", required=True, help="per-turn JSONL output")
    parser.add_argument("--conditions", default="C,T,S,P")
    parser.add_argument("--limit", type=int, default=0, help="cap scored turns (0 = all)")
    parser.add_argument("--checkpoint", default=os.environ.get("ALFWORLD_V2_MODEL"))
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from methods.opsd_loss import _get_teacher

    from tasks.alfworld.v1.trajectory import Trajectory
    from . import config
    from .privilege import has_future

    ckpt = args.checkpoint or "/workspace/cua/checkpoints/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = payload["samples"] if isinstance(payload, dict) else payload

    def meta(x):
        return (x.get("metadata") or {}) if isinstance(x, dict) else (x.metadata or {})

    trajectories = {}
    for x in samples:
        md = meta(x)
        td = md.get("trajectory")
        if td:
            trajectories[td["task_id"] + "|" + str(td["seed"])] = Trajectory.from_dict(td)
    pool = list(trajectories.values())
    print(f"[diag] trajectories={len(pool)}", file=sys.stderr)

    class _Args:
        pass

    teacher_args = _Args()
    teacher_args.hf_checkpoint = ckpt
    teacher = _get_teacher(teacher_args)

    history_length = config.get_int("ALFWORLD_V2_HISTORY", config.HISTORY_LENGTH)
    wanted = set(args.conditions.split(","))
    rows = []
    scored = 0

    for x in samples:
        md = meta(x)
        td = md.get("trajectory")
        if not td:
            continue
        trajectory = Trajectory.from_dict(td)
        first_turn = x.get("turn_ids") if isinstance(x, dict) else x.turn_ids
        if first_turn is None or len(first_turn) == 0:
            continue
        tid = int(first_turn[0].item()) if hasattr(first_turn[0], "item") else int(first_turn[0])
        turn_idx = tid % config.TURN_ID_STRIDE
        if not has_future(trajectory, turn_idx, 1):
            continue  # no future -> T and S are identical to C by construction

        response_length = int(x.get("response_length") or 0)
        if response_length <= 0:
            continue
        generated_ids = list(x["tokens"])[-response_length:]
        turn = trajectory.turns[turn_idx]
        generated_text = turn.generated_text

        sibling = _matched_sibling(trajectory, pool, max_tail=len(trajectory.turns) - turn_idx)
        conditions = build_conditions(
            tokenizer, trajectory, turn_idx, generated_ids, generated_text,
            history_length=history_length, sibling=sibling,
        )

        record = {"task_id": td["task_id"], "turn_idx": turn_idx, "success": trajectory.success,
                  "num_turns": trajectory.num_turns, "response_length": response_length}
        base = None
        for name in ("C", "T", "S", "P"):
            if name not in wanted or name not in conditions:
                continue
            prefix_ids, spans = conditions[name]
            scores = score_conditions(teacher, tokenizer, prefix_ids, generated_ids, spans)
            if name == "C":
                base = scores
            record[f"lp_{name}"] = scores
            record[f"prompt_len_{name}"] = len(prefix_ids)
            if base is not None and name != "C":
                for span in ("think", "action", "full"):
                    if scores.get(span) is not None and base.get(span) is not None:
                        record[f"e_{name}_{span}"] = scores[span] - base[span]
        # e -> r -> log r invariant, per span and condition
        for name in ("T", "S", "P"):
            for span in ("think", "action", "full"):
                key = f"e_{name}_{span}"
                if key in record:
                    e = record[key]
                    r = math.exp(e)
                    record[f"r_{name}_{span}"] = r
                    record[f"logr_minus_e_{name}_{span}"] = math.log(r) - e
        rows.append(record)
        scored += 1
        if scored % 10 == 0:
            print(f"[diag] scored {scored} turns", file=sys.stderr)
        if args.limit and scored >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in rows:
            fh.write(json.dumps(record) + "\n")

    # ---- report -------------------------------------------------------------
    inv = [abs(rec[k]) for rec in rows for k in rec if k.startswith("logr_minus_e_")]
    print(f"[diag] turns scored = {len(rows)}")
    print(f"[diag] INVARIANT max|log(r)-e| = {max(inv) if inv else 0.0:.3e}  "
          f"({'PASS' if not inv or max(inv) < 1e-6 else 'FAIL'})")
    for name in ("T", "S", "P"):
        for span in ("think", "action", "full"):
            key = f"e_{name}_{span}"
            vals = [rec[key] for rec in rows if key in rec]
            st = _stats(vals)
            if st:
                print(f"[diag] e_{name}_{span:6s} n={st['n']:4d} mean={st['mean']:+.5f} "
                      f"std={st['std']:.5f} min={st['min']:+.4f} p50={st['p50']:+.5f} "
                      f"p90={st['p90']:+.4f} max={st['max']:+.4f} mean|e|={st['mean_abs']:.5f}")
    # paired true-vs-shuffled
    for span in ("think", "action", "full"):
        pairs = [(rec[f"e_T_{span}"], rec[f"e_S_{span}"]) for rec in rows
                 if f"e_T_{span}" in rec and f"e_S_{span}" in rec]
        if pairs:
            mt = statistics.mean(abs(a) for a, _ in pairs)
            ms = statistics.mean(abs(b) for _, b in pairs)
            print(f"[diag] paired |e_T|={mt:.5f} vs |e_S|={ms:.5f} on {span}: ratio={mt / max(ms, 1e-12):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
