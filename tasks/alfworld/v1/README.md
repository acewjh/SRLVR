# ALFWorld + Hindsight Privileged Information (Qwen3-1.7B)

Multi-turn ALFWorld agent RL on the existing slime + SGLang + Megatron stack, with
an HPI teacher that re-weights the GRPO token loss using hindsight the student did
not have.

This package lives at `$PROJ/tasks/alfworld/v1/` (importable as `alfworld_hpi.*`) rather
than the suggested `$PROJ/examples/tasks/alfworld/v1/`: `$PROJ/frameworks/slime-ascend`
precedes `$PROJ` on `PYTHONPATH`, so `import examples` already resolves to the
vendored slime examples tree and a sibling `examples` package would be shadowed. A
top-level package mirrors `math_opsd`.

**No file outside this directory was modified.** The harness plugs into slime
through its existing hooks only.

---

## 1. Quick start (CPU only, safe while other experiments run)

```bash
bash tasks/alfworld/v1/scripts/setup_alfworld.sh        # overlay env + dataset + datasets
bash tasks/alfworld/v1/scripts/smoke_alfworld_cpu.sh    # the correctness gate
```

`setup_alfworld.sh` is idempotent and installs nothing into the `cua-rl` base
environment: ALFWorld's packages go into `$CUA/envs/alfworld_overlay`, a venv
created with `--system-site-packages`, and the training interpreter reaches it by
*appending* that venv's site-packages to `sys.path`. Appending means the overlay
can only supply modules `cua-rl` lacks. Verified overlap is limited to
`pip`/`setuptools`/`pkg_resources` — a test asserts this stays true.

The smoke gate runs entirely on CPU (no NPU, no Ray, no SGLang, no network) and
exits non-zero unless every contract holds. **Run it before any NPU work.**

---

## 2. What HPI does

At turn `t` the student sees `h_t` = task + past actions + past observations +
current admissible actions, and generates `y_t = think_t + action_t`.

The **frozen initial policy** (the same weights the actor starts from) re-scores
that *same* `y_t` twice, teacher-forced:

| view | context |
|---|---|
| full | `h_t` + privileged block `P_t` |
| control | `h_t` (the student's own prompt ids, verbatim) |

```
e_t = log q_teacher(y_t | h_t, P_t) - log q_teacher(y_t | h_t)
w_t = clamp( raw_w / mean(raw_w), 0.70, 1.30 ),  raw_w_t = 1 + alpha * tanh(|e_t| / scale)
L   = sum_of_sample_mean( w_t * pg_loss_t )          # pg_loss_t is slime's stock GRPO term
```

`|e_t|` only measures *how much the hindsight moved the plausibility of an action
that was actually taken*. The direction of learning stays entirely with the policy
gradient. `alpha = 0` gives `w == 1` exactly, so the file reduces to stock GRPO.

The weight formula is inherited unchanged from `math_opsd/hpi_v15.py` (per-trajectory
normalisation, `[0.70, 1.30]` clamp, stopgrad). What changes for ALFWorld is the
*substrate*: there is no ground-truth solution channel, so the chunk becomes a turn
and the base objective is GRPO rather than Reward-CV.

### The four arms

| arm | privilege | loss |
|---|---|---|
| `base_eval` | — | none (eval only) |
| `grpo` | — | `policy_loss` (stock) |
| `hpi_outcome_only` | terminal outcome | HPI-weighted GRPO |
| `hpi_future` | outcome + delayed future trace | HPI-weighted GRPO |

**Outcome-only vs Future-HPI differ in exactly one place: whether the privileged
block contains the future interaction trace.** Same model, prompt, sampling, split,
G, batch, optimizer, max turns and eval — all enforced by a single shared
`scripts/_common.sh` rather than four copies of the flags.

---

## 3. Index convention (exact)

Off-by-one errors here are silent and fatal, so the convention is fixed in one
place and asserted in tests.

* Turns are 0-based: `t = 0 .. T-1`. Turn `t` produced `think_t + action_t`.
* Transition `i` is the pair `(action_i, observation_{i+1})` — the environment's
  response to the action generated at turn `i`.
* `ALFWORLD_HPI_FUTURE_GAP = g` ⇒ the privileged trace contains transitions
  `i >= t + g`.

| gap | withheld | included |
|---|---|---|
| `2` (default) | transitions `t` and `t+1` | `t+2 …` |
| `1` | transition `t` (the action's own result) | `t+1 …` |
| `0` | nothing — **leaky**, ablation only | `t …` |

The terminal outcome is **always** included, for every gap.

Gap 2 is the conservative default because `o_{t+1}` frequently narrates the current
action verbatim ("You pick up the mug …"), which would hand the teacher the answer.

### Information isolation (enforced, not documented-and-hoped)

* The builder only ever reads `TurnRecord.action_text` / `.observation` /
  `.next_observation`. `generated_text` and `think_text` are structurally out of
  reach, and the rendered text is additionally scanned for `<think>`/`</think>`/
  `reasoning_content`.
* No current-turn response: neither `think_t` nor `action_t` may appear in `P_t`.
* `build_hpi_privilege` self-checks for `gap >= 1`; `assert_privilege_isolated` is
  the explicit gate (used by tests) and flags `gap = 0` under the production gap.

---

## 4. Sequence layout (why it is not one chained conversation)

The next turn's context keeps the task, past **actions** and past **observations**
— never past `<think>`. Qwen3's chat template independently strips `<think>` blocks
from assistant history (a test pins this behaviour). So `h_{t+1}` is **not** a
token-level extension of `h_t + y_t`, and chaining turns into one causal sequence
would train on contexts the engine never saw.

Each episode is therefore laid out as per-turn segments:

```
tokens    = prompt_0 | y_0 | prompt_1 | y_1 | prompt_2 | y_2 | ...
loss_mask =   0 ...  |  1  |   0 ...  |  1  |   0 ...  |  1  | ...
turn_ids  =  -1 ...  |  0  |  -1 ...  |  1  |  -1 ...  |  2  | ...
```

Every `y_t` is immediately preceded by exactly `prompt_t`, so a causal forward
conditions each generated token on precisely the context the engine used.
`loss_mask` is 0 on every prompt segment, so the segment-boundary discontinuities
carry no gradient.

Alignment guarantees:

* `y_t` ids are stored exactly as returned by the engine — never decoded and
  re-encoded.
* `prompt_t` ids are stored exactly as sent to the engine, so the teacher's control
  view is byte-identical to the student's context for free.
* `loss_mask == 1` iff the token was generated by the model; `turn_ids >= 0` iff
  `loss_mask == 1`.
* The teacher-forced target span equals the student's tokens token-for-token,
  asserted in `train_data.validate_layout` **and** again inside the loss.

**Cost:** O(T²) growth in context length, bounded by `ALFWORLD_MAX_TURNS` and
`ALFWORLD_ACTION_HISTORY`. The teacher uses the same layout, so arms stay
comparable. See §7 for the memory implication.

---

## 5. Hooks used (slime unmodified)

| flag | value |
|---|---|
| `--rollout-function-path` | `alfworld_hpi.rollout.generate_rollout` |
| `--eval-function-path` | same callable (`evaluation=True`) |
| `--custom-convert-samples-to-train-data-path` | `alfworld_hpi.train_data.convert_samples_to_train_data` |
| `--custom-loss-function-path` | `alfworld_hpi.loss.alfworld_hpi_loss` (HPI arms only) |

`--custom-generate-function-path` was rejected: it is only reachable from the
*stock* sglang rollout, which would then own reward application and data conversion
and fight with this harness's convert/loss.

Per-turn views ride on the `hpi_views` / `hpi_prompt_lengths` / `hpi_target_lengths`
keys that `_split_train_data_by_dp` already whitelists (added for the math HPI
work), and the turn→span mapping rides on `turn_ids`. **No new train-data key was
needed, so the vendored slime tree is untouched.**

---

## 6. Configuration

Configuration is by environment variable, matching this repo's convention for
custom-loss knobs (`OPSD_HPI_*`, `OPSD_CV_*`). An `args` attribute of the same name
wins if a future slime registers one.

| variable | default | meaning |
|---|---|---|
| `ALFWORLD_HPI_MODE` | `none` | `none` \| `outcome_only` \| `future_outcome` |
| `ALFWORLD_HPI_FUTURE_GAP` | `2` | see §3 |
| `ALFWORLD_HPI_ALPHA` | `0.5` | weight strength |
| `ALFWORLD_HPI_SCALE` | — | **required** for HPI arms; calibrate (below) |
| `ALFWORLD_MAX_TURNS` | `15` | agent turns per episode |
| `ALFWORLD_ACTION_HISTORY` | `8` | recent actions rendered into the transcript |
| `ALFWORLD_OVERLAY_SITE` | derived | overlay site-packages |
| `ALFWORLD_HPI_DATA` | `$CUA/data/alfworld/json_2.1.1` | dataset root |

### Calibrating `ALFWORLD_HPI_SCALE`

Convention inherited from `math_opsd`: `scale = p75 of |e|` (the math task
calibrated to `0.045126`). Run Gate D once with a placeholder scale, then read the
logged percentile:

```bash
bash tasks/alfworld/v1/scripts/smoke_alfworld_train_hpi_1p7b.sh
grep -o "hpi/abs_e_p75[^,}]*" logs/alfworld_smoke_train_hpi_future_1p7b.log | tail -1
# then export ALFWORLD_HPI_SCALE=<that value> for the real run
```

The launcher refuses to start an HPI arm without a scale rather than silently
picking the strength of the intervention.

---

## 7. Known risks / must-measure-before-long-runs

1. **Loss-forward memory.** The per-turn-segment layout is O(T²) in context, and
   Megatron materialises full-vocab logits (`[rows, 151936]`). At `max_turns=15`
   with an 8-action history a 1.7B episode lands around 10–15k rows. This is the
   first thing to measure in Gate C (`npu-smi` peak HBM); `ALFWORLD_MAX_TURNS` and
   `ALFWORLD_ACTION_HISTORY` are the knobs to shrink if it does not fit.
2. **`ROLLOUT_BS` is a placeholder.** `G=8` is fixed by the recipe and must not be
   swept, but the number of tasks per step was *not* guessed — the default (2, so
   GBS 16) is deliberately small and should be set from the Gate B throughput
   numbers.
3. **Environment setup cost.** Each episode builds its own textworld env
   (~2 s make + ~3 s reset). At G=8 that is ~40 s of CPU per task, overlapped across
   the group but not free. Fine at smoke scale; worth measuring at full scale.
4. **Eval path is untested on hardware.** The `evaluation=True` branch and
   `report_eval.py` are exercised on CPU with synthetic dumps, but have not run
   against a live engine.

---

## 8. Layout

```
tasks/alfworld/v1/
  paths.py          overlay/sys.path isolation + data root resolution
  env_adapter.py    textworld.gym registration, reset/step, task extraction
  prompts.py        system prompt + per-turn transcript (no past CoT)
  action_parser.py  total <think>/<action> parser (never raises)
  trajectory.py     TurnRecord/Trajectory, JSONL round-trip
  privilege.py      build_hpi_privilege + leakage assertions
  rollout.py        slime rollout + eval, per-turn segment layout
  reward.py         terminal success only
  train_data.py     convert + layout validation
  loss.py           HPI-weighted GRPO
  eval.py           success-rate aggregation by split and category
  report_eval.py    aggregate saved eval dumps
  check_gates.py    PASS/FAIL for the NPU smoke gates
  data_prepare.py   game index, prompt datasets, eval configs
  configs/          generated datasets + eval configs (checked in for convenience)
  scripts/          setup, CPU smoke, NPU smoke gates, the four run entrypoints
  tests/            107 CPU tests (+2 conditional skips)
```

## 9. Tests

```bash
bash tasks/alfworld/v1/scripts/smoke_alfworld_cpu.sh     # one-shot PASS/FAIL
```

Covers: import isolation, dataset discovery, `env.reset`/`env.step`, seen/unseen
samples, prompt construction, the action parser, trajectory serialisation, past- and
future-CoT stripping, `future_gap` indexing (including an explicit off-by-one
sweep), current-action leakage, outcome-only and future-outcome privilege, loss-mask
construction, the rollout `Sample` contract, teacher/student token alignment, eval
aggregation, and a scripted end-to-end rollout.
