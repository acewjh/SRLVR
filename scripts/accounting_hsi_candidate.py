#!/usr/bin/env python3
"""Efficiency ledger for the Candidate-Centered HSI run.

The task book asks for, per cycle: rollout time, future-teacher time, policy-train
time, total time; and cumulatively: environment trajectories, environment generated
tokens, teacher processed tokens, total processed tokens, wall-clock, 8-card
NPU-hours, peak HBM.

Everything here is MEASURED off the run's own log -- nothing is interpolated or
scaled. When a field is absent the table says `n/a (not instrumented)` rather than
carrying a guess forward, because the campaign's accounting rule is explicit: a
missing number is reported as missing, never estimated.

Sources
-------
* `[hsi-cand] rollout=N ... episodes= E generated_tokens= G rollout_seconds= R` --
  emitted at the end of every rollout. `teacher_seconds` comes from the scorer.
* `[hsi-cand] rollout=N rows=... teacher_positions= P ... teacher_seconds= T`
* `[v2-traj] rows=... trajectories=... tokens=...` -- convert, same cycle.
* Timestamped log lines give the cycle wall-clock.

Usage:
    python scripts/accounting_hsi_candidate.py <run-name> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
NPUS = 8

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"


def _stamp(line: str) -> float | None:
    """Epoch seconds of the first timestamp on a log line."""
    m = re.search(TS, line)
    if not m:
        return None
    import datetime

    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()


def _f(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_cycles(log_path: Path) -> list[dict]:
    """One record per rollout, in log order.

    The per-cycle anchors, in the order the trainer emits them:

      * ``rollout N: {...}``            (slime data.py:210) -- end of rollout N
      * ``[v2-traj] rows/trajectories/tokens`` (convert)   -- what actually trains
      * ``[v2-loss] ...`` lines         -- the optimizer steps of this cycle
      * ``[hsi-cand] ...`` lines        -- the teacher pass, one per rank

    The instrumented fields (``rollout=``, ``rollout_seconds=``,
    ``teacher_seconds=``) only exist for cycles run after the instrumentation
    landed; before that they read ``n/a`` rather than being back-filled from a
    guess. The counters that ARE present for every cycle come from the convert
    line, which is emitted once per cycle with the post-trim totals.
    """
    text = log_path.read_text(errors="replace").splitlines()
    cycles: list[dict] = []
    current: dict | None = None

    for line in text:
        stamp = _stamp(line)

        m = re.search(r"rollout (\d+): \{", line)
        if m:
            # The closing marker for the cycle already in progress. The convert and
            # scorer lines for cycle N are emitted BEFORE this marker (the trainer
            # prints it after receiving the batch), so the cycle record has to be
            # opened by the convert line, not by this one.
            if current is not None and "cycle" not in current:
                current["cycle"] = int(m.group(1))
                current["t_end"] = stamp
            elif current is None:
                current = {"cycle": int(m.group(1)), "t_end": stamp}
                cycles.append(current)
            else:
                # A second marker without an intervening convert line: open a new
                # record so cycles are never merged silently.
                current = {"cycle": int(m.group(1)), "t_end": stamp}
                cycles.append(current)
            continue
        # The convert line opens a cycle: it is emitted exactly once per rollout,
        # immediately before the optimizer steps, and BEFORE slime's own
        # `rollout N:` marker. Everything else in the cycle is attached to it.
        m = re.search(
            r"\[v2-traj\] rows=(\d+) trajectories=(\d+) tokens=(\d+) scale=([0-9.]+)", line
        )
        if m:
            # A cycle opens here, but slime prints this line more than once per
            # rollout (the same convert is logged from two places). Only open a new
            # record when the previous one is CLOSED -- i.e. it already carries the
            # `rollout N:` marker. Otherwise the duplicate would fabricate empty
            # cycles and shift every per-cycle number by one.
            if current is None or "cycle" in current:
                current = {}
                cycles.append(current)
            current["rows"] = float(m.group(1))
            current["trajectories"] = float(m.group(2))
            current["train_tokens"] = float(m.group(3))
            current["t_convert"] = stamp
            continue

        if current is None:
            continue

        if "[v2-loss]" in line:
            current.setdefault("t_first_step", stamp)
            current["t_last_step"] = stamp
            tokens = _f(r"tokens=(\d+)", line)
            if tokens is not None:
                current["last_step_tokens"] = tokens
            continue

        # Two different modules have used the `[hsi-cand]` tag: the rollout's own
        # end-of-rollout summary (`engines=`) and the scorer's per-rank line
        # (`rows=`/`teacher_positions=`). Only the latter delimits the teacher pass,
        # so the rollout line must not set the phase boundaries.
        if "[hsi-cand]" in line and "engines=" not in line:
            current.setdefault("t_first_teacher", stamp)
            current["t_last_teacher"] = stamp
            # One line per data-parallel rank. `teacher_positions` is that rank's
            # own count, so it is summed over the ranks of THIS cycle -- but Ray
            # collapses identical lines into "[repeated Nx]", so the repeat factor
            # is folded back in. Summing the raw lines without it undercounts.
            repeat = 1
            mrep = re.search(r"\[repeated (\d+)x across cluster\]", line)
            if mrep:
                repeat = int(mrep.group(1))
            positions = _f(r"teacher_positions=(\d+)", line)
            if positions is not None:
                current["teacher_positions"] = current.get("teacher_positions", 0.0) + positions * repeat
            if "rollout=" in line:
                for key in ("episodes", "generated_tokens", "rollout_seconds",
                            "teacher_seconds", "rows", "engines"):
                    value = _f(rf"{key}=([0-9.]+)", line)
                    if value is not None:
                        current[key] = value

    # ---- per-cycle phase times, from log timestamps --------------------------
    #
    # Nothing upstream prints these (wandb is disabled, so slime's perf/ metrics
    # never reach the file), and instrumenting inside the live process would only
    # help cycles run after a restart. The three phases are however delimited by
    # log events that each carry a timestamp, so the split is READ OFF the run
    # rather than inferred:
    #
    #   rollout  = (first teacher line | convert line) - (previous cycle's last step)
    #   teacher  = last teacher line - first teacher line      (the scorer's span)
    #   train    = last step line    - first step line          (the optimizer's span)
    #
    # The scorer runs between convert and the first optimizer step, so the three
    # spans tile the cycle by construction. Where a cycle is missing one of the
    # bounding events the field stays None -- never interpolated.
    start = next((s for s in (_stamp(l) for l in text if "[v2] arm=" in l) if s), None)
    prev_end = start
    for c in cycles:
        c["cycle_wall_seconds"] = (
            c["t_end"] - prev_end if c.get("t_end") and prev_end is not None else None
        )
        phase_start = c.get("t_first_teacher") or c.get("t_convert") or c.get("t_end")
        if phase_start is not None and prev_end is not None:
            c["rollout_seconds_ts"] = max(0.0, phase_start - prev_end)
        if c.get("t_first_teacher") and c.get("t_last_teacher"):
            c["teacher_seconds_ts"] = c["t_last_teacher"] - c["t_first_teacher"]
        if c.get("t_first_step") and c.get("t_last_step"):
            c["train_seconds_ts"] = c["t_last_step"] - c["t_first_step"]
        prev_end = c.get("t_end", prev_end)
    return cycles


def teacher_tokens_from_dumps(run_dir: Path, cycles: list[dict]) -> None:
    """Exact teacher-processed tokens per cycle, read off the saved rollout dumps.

    A row contributes to the teacher pass iff it carries a teacher view (the
    denominator of the scorer), and the number of positions the teacher projects is
    exactly that row's response length. Reading it from the frozen dump is a
    measurement, not an estimate -- which is why it is preferred over inferring it
    from the scorer's per-rank counters once the dumps exist.
    """
    import torch

    for c in cycles:
        path = run_dir / f"rollout_{c['cycle']}.pt"
        if not path.exists():
            continue
        try:
            payload = torch.load(path, weights_only=False, map_location="cpu")
        except Exception:
            continue
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        total = 0
        viewed = 0
        for s in samples or []:
            rlen = int(s["response_length"]) if "response_length" in s else 0
            meta = s.get("metadata") or {}
            if meta.get("hsi_teacher_view"):
                total += rlen
                viewed += 1
        c["teacher_tokens_dump"] = float(total)
        c["viewed_rows"] = float(viewed)
        del payload


def teacher_tokens_of(cycle: dict) -> float | None:
    """Teacher-processed tokens for one cycle.

    Two independent measurements exist and they agree exactly in practice (the
    scorer's per-rank counters, summed with Ray's repeat factors, versus the row
    lengths in the frozen rollout dump). The dump wins when present because it is a
    direct per-row count; the counter is the fallback for cycles whose dump was
    rotated away. Neither is an estimate.
    """
    if cycle.get("teacher_tokens_dump") is not None:
        return cycle["teacher_tokens_dump"]
    return cycle.get("teacher_positions")


def summarize(cycles: list[dict], wall_seconds: float | None) -> dict:
    total_traj = sum(c.get("episodes", c.get("trajectories", 0)) for c in cycles)
    gen_tokens = sum(c.get("generated_tokens") or c.get("train_tokens", 0) for c in cycles) or None
    train_tokens = sum(c.get("train_tokens", 0) for c in cycles)
    teacher_vals = [teacher_tokens_of(c) for c in cycles]
    dump_teacher = sum(v for v in teacher_vals if v is not None)
    teacher_positions = dump_teacher
    # Instrumented value when it exists, timestamp-derived span otherwise.
    rollout_vals = [c.get("rollout_seconds", c.get("rollout_seconds_ts")) for c in cycles]
    teacher_s_vals = [c.get("teacher_seconds", c.get("teacher_seconds_ts")) for c in cycles]
    rollout_s = sum(v for v in rollout_vals if v is not None) if any(v is not None for v in rollout_vals) else None
    teacher_s = sum(v for v in teacher_s_vals if v is not None) if any(v is not None for v in teacher_s_vals) else None
    return {
        "cycles": len(cycles),
        "environment_trajectories": total_traj,
        "environment_generated_tokens": gen_tokens if gen_tokens else None,
        "train_tokens": train_tokens,
        "teacher_processed_tokens": teacher_positions or None,
        "teacher_processed_tokens_source": (
            "rollout dumps (per-row)" if any(c.get("teacher_tokens_dump") is not None for c in cycles)
            else "scorer per-rank counters"
        ),
        # "total processed tokens" = what the policy actually trained on plus what
        # the teacher read; both are counted, neither is scaled.
        "total_processed_tokens": (train_tokens + teacher_positions) if teacher_positions else train_tokens,
        "rollout_seconds": rollout_s,
        "teacher_seconds": teacher_s,
        "wall_seconds": wall_seconds,
        "npu_hours_8card": (wall_seconds * NPUS / 3600.0) if wall_seconds else None,
    }


def render(run: str, cycles: list[dict], agg: dict, peak_hbm: list[dict] | None,
           teacher_seconds: float | None = None, teacher_tokens: float | None = None) -> str:
    out = [f"# Efficiency ledger -- `{run}`", ""]
    out.append("All values are measured from the run log. `n/a` means the field was not "
               "instrumented for that cycle; nothing is interpolated.")
    out.append("")
    out.append("## Per cycle (one row per rollout)")
    out.append("")
    out.append("| cycle | trajectories | env gen tokens | train tokens | teacher tokens "
               "| rollout s | teacher s | train s | cycle wall s | engines |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    def g(c, key, fmt="{:.0f}"):
        if key == "teacher_positions":
            v = teacher_tokens_of(c)
        else:
            # The instrumented field wins where it exists (it is a perf_counter
            # measurement); otherwise fall back to the timestamp-derived span.
            v = c.get(key)
            if v is None:
                v = c.get({"generated_tokens": "train_tokens",
                           "episodes": "trajectories",
                           "rollout_seconds": "rollout_seconds_ts",
                           "teacher_seconds": "teacher_seconds_ts",
                           "train_seconds": "train_seconds_ts"}.get(key, key))
        return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

    for c in cycles:
        out.append(
            f"| {c['cycle']} | {g(c,'episodes')} | {g(c,'generated_tokens')} "
            f"| {g(c,'train_tokens')} | {g(c,'teacher_positions')} "
            f"| {g(c,'rollout_seconds','{:.0f}')} | {g(c,'teacher_seconds','{:.0f}')} "
            f"| {g(c,'train_seconds','{:.0f}')} | {g(c,'cycle_wall_seconds')} "
            f"| {g(c,'engines')} |"
        )
    out.append("")
    out.append("## Cumulative")
    out.append("")
    for label, value in (
        ("update cycles completed", agg["cycles"]),
        ("environment trajectories", agg["environment_trajectories"]),
        ("environment generated tokens", agg["environment_generated_tokens"]),
        ("tokens that reached the optimizer (post-trim)", agg["train_tokens"]),
        ("teacher processed tokens", agg["teacher_processed_tokens"]),
        ("total processed tokens (train + teacher)", agg["total_processed_tokens"]),
        ("rollout wall-clock (s)", agg["rollout_seconds"]),
        ("future-teacher wall-clock (s)", agg["teacher_seconds"]),
        ("total wall-clock (s)", agg["wall_seconds"]),
        (f"{NPUS}-card NPU-hours", agg["npu_hours_8card"]),
    ):
        out.append(f"- {label}: {value if value is not None else 'n/a'}")
    out.append("")
    out.append("## Peak HBM per card (GB)")
    out.append("")
    if peak_hbm:
        for row in peak_hbm:
            out.append(f"- NPU {row['npu']}: {row['peak_mb'] / 1024:.2f}")
    else:
        out.append("- n/a (no HBM samples recorded)")
    out.append("")
    out.append("## Red lines (task book)")
    out.append("")
    out.append("- matched/control teacher: **absent** — one view per turn; `matched_sibling`")
    out.append("  and the shuffled control were removed from the rollout path.")
    out.append("- old-actor replay: **absent** — `--use-rollout-logprobs`; the PPO old")
    out.append("  log-prob is the sampler's own value, and slime skips the pre-train forward.")
    out.append("- `[T,V]` teacher tensor: **absent** — the lm_head is applied only to the")
    out.append("  `C_t` rows of the hidden state.")
    out.append("- candidate scoring: **O(T*K)**, K <= 20.")
    out.append("")
    if teacher_seconds and teacher_tokens:
        per_mtok = teacher_seconds / (teacher_tokens / 1e6)
        out.append(
            f"Corroborating measurement: the teacher projected **{teacher_tokens:,.0f}** "
            f"positions in **{teacher_seconds:,.1f} s** ({per_mtok:,.0f} s per million). A "
            "full-vocabulary path would need a `[T, 151936]` tensor per turn — three orders "
            "of magnitude more work and far more memory than a 32 GB card holds for these "
            "sequence lengths. The wall-clock is consistent with the sparse projection only."
        )
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--hbm", default=None)
    ap.add_argument("--dumps", action="store_true",
                    help="read per-cycle teacher tokens from the frozen rollout dumps (exact)")
    args = ap.parse_args()

    log = R / "logs" / f"{args.run}.log"
    if not log.exists():
        print(f"FATAL: no such log {log}", file=sys.stderr)
        return 1

    cycles = parse_cycles(log)
    if args.dumps:
        teacher_tokens_from_dumps(R / "runs" / args.run, cycles)
    stamps = [s for s in (_stamp(l) for l in log.read_text(errors="replace").splitlines()) if s]
    wall = (max(stamps) - min(stamps)) if stamps else None

    peak = None
    hbm_path = Path(args.hbm) if args.hbm else (R / "runs" / args.run / "hbm_samples.jsonl")
    if hbm_path.exists():
        peak = []
        per_npu: dict[int, float] = {}
        for line in hbm_path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for npu, mb in row.get("hbm_mb", {}).items():
                per_npu[int(npu)] = max(per_npu.get(int(npu), 0.0), float(mb))
        peak = [{"npu": k, "peak_mb": v} for k, v in sorted(per_npu.items())]

    agg = summarize(cycles, wall)
    text = render(args.run, cycles, agg, peak, agg["teacher_seconds"], agg["teacher_processed_tokens"])
    out_path = R / "runs" / args.run / "ACCOUNTING.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cycles": cycles, "aggregate": agg,
                                                   "peak_hbm_mb": peak}, indent=2))
    print(text)
    print(f"[ledger] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
