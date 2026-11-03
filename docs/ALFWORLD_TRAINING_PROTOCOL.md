# ALFWorld default training protocol — 2026-10-23

The 1.7B training campaign enables brief native Qwen3 thinking in rollout and
hindsight templates, asks for at most 24 thinking tokens, allows at most 1024
generated tokens per turn, and stops at </action> with the closing tag retained.
Prompts keep complete observation/action history and omit the per-state
admissible-action list. That list comes from ALFWorld, not SGLang; it remains
available to environment validation and logs. A static command-syntax guide
describes the output format.

The shared V2 training/evaluation episode runner defaults to
`command-normalization-v2`. After existing structured parsing it can (a) recover
one complete bare command, (b) translate an explicit command form
(`examine`/`look at` -> `look`, `move X to Y` -> `put X in/on Y`, `grab` ->
`take`, `wash` -> `clean`, `go X` -> `go to X`), and (c) navigate to the
receptacle the model itself named (`open fridge 1` -> `go to fridge 1`) when
that navigation is in the current admissible set. It never chooses an object or
a different receptacle and never mines a command from multi-command/prose text;
ambiguous candidates stay invalid. Every executed action is an exact member of
the current admissible set.

The repair changes only the environment-facing action: raw generated text,
token IDs, log-probs, sampler features, and actor loss masks are not rewritten.
The executed action enters the observation/action history. The precondition
step is an expert correction, not an on-policy sample: the model's sampled
tokens may say `take ...` while the environment executes `go to ...`. Use
`command-normalization-v2-form` for form-only repair, or
`command-normalization-v1` for the historical harness. Per-row metadata records
the harness version and parse source; rollout metrics report action validity
and bare-command recovery. Base and trained-model evaluation use this same
episode runner and harness.

All real turn rows are accumulated before one optimizer update per rollout.
Zero-mask padding aligns rows to data parallelism. There is no turn trimming,
real-episode top-up, or reward-based rejection/resampling. GRPO keeps 4 tasks x
8 trajectories. SRLVR keeps 32 task samples x 1 trajectory: the first 8 positions
are calibration, the other 24 receive the calibrated future correction.

Shared V2 entry defaults: 100 rollout batches with full checkpoints every 25,
G=8, 4 tasks per rollout, GBS=32, 50 turns, cap=4096, complete history,
LoRA r256/alpha256 when enabled, full RL by default, constant LR=1e-6,
Adam(.9,.999), weight_decay=0, grad clip=1, symmetric token PPO clip=.2, and
frozen initial-model KL=.001 with low_var_kl. Prompts show admissible actions,
use the `command-normalization-v2-form` harness, brief thinking and frontier
curriculum with seed 42. GRPO retains trajectory-balanced token-mean reduction;
SRLVR retains token-sum/task-count reduction. KL is reduced in each method's
actor-objective units. SRLVR calibration probes exclude KL, which enters only
the final update. GSPO retains its sequence-ratio clip configuration.

Serving and training windows agree: SEQ_LENGTH=32768 by default. Complete history
is never silently truncated. The 32K limit is within Qwen3-1.7B's native
40960-token position capacity; the 1.7B thinking output cap is 1024 tokens. An input
exceeding the window raises an explicit
error; increase SEQ_LENGTH when needed. Evaluation retains its existing
stochastic decoding; this revision does not switch evaluation to greedy.

## Historical curriculum

tasks/alfworld/curriculum.py samples training games only. Before each batch,
task estimates are frozen using earlier rollouts with Beta(1,1) smoothing:
p=(wins+1)/(trials+2). Old wins/trials receive a .9 discount per rollout.
Within each family, a 10% uniform mixture preserves exploration. No current
outcome influences its own task selection; all sampled outcomes are retained.

GRPO/GSPO rotate six families across task positions. Within the selected family,
weights are 1-p**G-(1-p)**G with G=8. Four positions per rollout cover each family
twice every three rollouts.

SRLVR/HSI rotate six families across whole batches. All positions in a batch
are independent draws with replacement from one frozen family distribution,
so SRLVR calibration and update positions remain identically distributed.
Within-family weights use p*(1-b_f)**2+(1-p)*b_f**2, with b_f the historical
smoothed family success rate. This approximates reward-advantage second moment
around a family baseline; it does not change the method's actual advantage or
baseline. At b_f=.5 this is uniform. As a family becomes easy it emphasizes
less-solved games; for difficult families it emphasizes games with some success.
This is a sampling heuristic whose performance benefit remains to be measured.

State is stored in curriculum/000000k.json under the checkpoint save directory.
Snapshot k records outcomes from rollout k. Sampling rollout k reads only
snapshots strictly before k. Replaying replaces its snapshot and removes
abandoned future snapshots, preventing double-counted rewards after rewind.
Preserve curriculum/ alongside checkpoint iterations when moving a run.

## Entry and verification

The common 1.7B entry is tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh.
The following commands start the 1.7B training run. The campaign supervisor
starts the existing checkpoint benchmark immediately after it validates rollout
100's final checkpoint; benchmark prompts use the same thinking mode, context,
and command-normalization harness as training.

```bash
ARM=grpo bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=gigpo bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=srlvr bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
```

`ARM=gigpo` is the group-in-group arm: the GRPO recipe above with one extra
per-turn anchor-state advantage term. See `docs/GIGPO.md`; its matched control
is the `ARM=grpo` cell with the same seed, schedule, sampler and checkpoint
cadence.

Qwen3-4B ALFWorld arms default to **TP=2** on the 32 GB cards via
`_4b_backbone.sh`. The final GRPO layout is 8 actor + 8 colocated rollout engines;
it disables NPU graph and bounds the static pool at 0.30/32768 after the first
graph-enabled sync failed with a `captures_underway` assertion. The previous
2 actor + 6 rollout TP2 smoke remains the full-RL optimizer-step evidence; the
8+8 rollout was stopped after it proved too slow to serve as a bounded smoke.
The frozen repeat-stop arm keeps its explicit TP=1 override; `TP=<n>` still
overrides everything.

Use fresh RUN_NAME/RUN_ROOT values. The former fullbatch launcher is only a
legacy alias of the normal GRPO entry; there is no separate fullbatch flag.
The batch_experiments.py script prints commands by default; --parse-only
checks actual parsers, while --execute starts training.

The ignored local frameworks already contain the needed patches. Migration
requires compat/patches/srlvr-framework.patch and
compat/patches/alfworld-reference-kl.patch. The new reference patch clears LoRA B
tensors while loading the initial model so resume cannot contaminate the KL
target with trained adapters.

CPU verification: 174 SRLVR/HSI/ALFWorld tests passed, plus 21 registry tests.
Actual GRPO and SRLVR argument parsing passed. No training or evaluation was
started for this revision. Old checkpoint cleanup is recorded in reports/.
