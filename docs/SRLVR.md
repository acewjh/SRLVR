Current defaults (2026-10-23) are defined in
[ALFWORLD_TRAINING_PROTOCOL.md](ALFWORLD_TRAINING_PROTOCOL.md). They supersede
the historical settings and verification records below: brief thinking, 1024 output
tokens, complete history, no per-state action list, one optimizer update per
rollout, LR=1e-6, and initial-model KL=.001 low_var_kl. Training and inference
context capacity is now 32768 tokens; the 1.7B response cap is 1024. GRPO uses 4x8 trajectories;
SRLVR uses 32x1 with 8 calibration positions. Both now use historical curricula.
SRLVR's curriculum keeps calibration/update draws IID within a rotating family;
KL enters the final actor update and is excluded from calibration probes.
Migration also requires compat/patches/alfworld-reference-kl.patch.

SRLVR uses the existing ALFWorld task, environment, observation/chat template,
action parser, terminal verifier, Sample, SGLang servers, Megatron model,
optimizer, synchronization and checkpoint entry. The objective is implemented in
methods/srlvr.py and selected by ARM=srlvr in the existing registry.

The shared model entry is tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh.
Its defaults are Qwen3-1.7B, seed 42, 100 rollout rounds, 8 iid task samples per
round, 2 actor + 6 inference devices, TP/PP/CP/EP=1, prefix cache, brief native
thinking (soft budget 64), response cap 4096, history 8, maximum 50 turns.
Both algorithms use the same task samples. Existing legacy launchers retain
their frozen-schedule task sampling and original baseline behavior.

Each new common-entry experiment explicitly uses BF16 inference, matching BF16
training. Qwen3-1.7B's model config otherwise makes SGLang auto-select FP16.

GRPO samples eight episodes per task, uses the existing group advantage and
token ratio clipping, and preserves the existing optimizer minibatches.
GSPO uses the same episodes/advantages, but its sequence is one assistant turn:
exp(mean(logp_new-logp_old)), with actor gradients through that mean.
Both retain the repository's trajectory-balanced token-mean reduction.
SRLVR samples one episode per task, keeps every actor token including generated
stop/EOS, computes B from strictly subsequent turns, calibrates before updating,
then minimizes -sum_i,t A_it sum_j logp_itj / N with one optimizer step.

The new entry doubles tasks per round from 4 to 8. Counts are explicit:
SRLVR: 8 tasks / 8 real episodes per round; GRPO/GSPO: 8 / 64.
One hundred rounds therefore means 800 task samples and respectively 800/6400
real environment trajectories, unless a baseline explicitly reports top-up.
Calibration uses the first n positions before seeing rewards, with n=2 by default;
the remaining six positions form U. Calibration episodes also receive exactly
one baseline actor contribution.

From the repository directory, the three small real commands are:

```bash
ARM=grpo NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 SAVE_INTERVAL=1 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=gspo NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 SAVE_INTERVAL=1 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=srlvr SRLVR_MODE=zero NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 SAVE_INTERVAL=1 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
python3 scripts/batch_experiments.py --algorithms grpo gspo srlvr --tasks alfworld --seeds 42 43 --rounds 100
```

The matrix command prints commands only. --parse-only invokes the real parser
for every combination without training; --execute runs combinations sequentially
on the current node. Output directories include algorithm, seed, round/task
budget and timestamp. For a fresh 100-round run, omit the small-run overrides
NUM_ROLLOUT=1 and MAX_TURNS=2. Use distinct RUN_NAME / RUN_ROOT for reruns.

Necessary method settings are public environment parameters:

| Parameter | Default | Meaning |
|---|---|---|
| SRLVR_MODE | zero | zero / fixed / certified |
| SRLVR_CALIBRATION | 2 | n, must be >=2 and < ROLLOUT_BS |
| SRLVR_C | 10 | clipped log-probability feature scale |
| SRLVR_RHO | .9 | coefficient bound margin |
| SRLVR_FIXED | absent | required explicit lambda in fixed mode |
| SRLVR_DELTA | .05 | certified confidence probability |
| SRLVR_GZ / SRLVR_GY | absent | population bounds for the frozen policy |
| SRLVR_BOUND_BASIS | absent | supplied mathematical justification for bounds |
| SRLVR_WINDOW | 128 | number of historical calibration trajectories |
| SRLVR_EPS_B0 | .01 | reference clipping margin |
| SRLVR_DELTA_S | .1 | reward scale regularizer |
| SRLVR_ALIGNMENT_TOL | .05 | maximum absolute turn-mean logp drift at theta_old |

Fixed mode enforces abs(lambda)<=rho*min(b,1-b) at every batch. Certified mode
fails before launching if its population-bound inputs/basis are missing. Merely
passing observed norm checks cannot validate a population bound. No numerical
population bounds have been established for this Qwen/LoRA experiment. Zero
mode measures C/D/Q and uses the same C/U split, history, loss and optimizer;
it is the single-rollout control and does not test nonzero future correction.

Sampler statistics use the FP32 probability tensor that is actually sampled,
then the existing customized_info transport. No top-k candidate table, full
vocabulary transfer, replay forward or additional environment attempt is used.
The initial strict implementation supports stochastic full-softmax, TP1
inference, no grammar/custom logit processor; unsupported configurations raise.
The mathematical feature helper separately tests temperature, zero probability
and masked support. Actual grammar/top-p/top-k actor distributions are not
implemented and must not be advertised as supported.

Calibration uses two differentiable full-trajectory backwards per C trajectory,
not per-turn full parameter gradients. All actor DP ranks probe the same
trajectory; post-finalize reduce-scatter shards are all-gathered once before
reading every trainable LoRA parameter gradient. This initial verified topology
requires TP=PP=CP=EP=1. Other parameter sharding topologies fail explicitly.
Probe backwards do not prepare/clip/step Adam, advance its scheduler, or leave
gradients in its buffers.

Probe *execution* is independent of the estimator. The default `SRLVR_PROBE_MODE=auto` assigns calibration positions round-robin to data-parallel groups and
keeps each rank's raw local gradient (`finalize_model_grads` is skipped for the
probe; only the tensor-parallel reductions are applied), then all-gathers the
small per-position C/D/W records once. It requires the distributed optimizer and
`overlap_grad_reduce=0`; otherwise `auto` falls back to `serial` and an explicit
`parallel` fails closed. `serial` reproduces the original all-ranks-probe-all-
positions path with per-chunk DP reconstruct. `SRLVR_PROBE_BENCH=control` runs
serial, parallel, serial, parallel on the same frozen batch and reports the
serial↔parallel spread against same-path variation as a diagnostic, not a proof
of equivalence. `SRLVR_PROBE_BENCH=1` is the direct numeric comparison and aborts
on mismatch unless `SRLVR_PROBE_BENCH_STRICT=0` is explicitly supplied.
`SRLVR_PROBE_DETERMINISTIC=1` (opt-in; default 0) scopes deterministic kernels to probe forwards and
backwards and restores the caller's setting afterward; it does not itself
guarantee that every installed accelerator kernel is deterministic.
The production path retains original BF16 attention. An FP32-attention trial
changed a real trajectory's score norm by about 10%, so it was discarded.
A full 4B same-batch control measured calibration speedup 3.47--3.67x;
maximum observed serial/parallel differences across two repeats were C 0.31%,
D 0.060%, W 0.214%. These are empirical roundoff differences, not a guarantee
for every batch. See `reports/srlvr_v2_calibration_accel_20261030_185900/REPORT.md`.
`SRLVR_PROBE_VECTOR_COMPARE=1` adds full Z/Y vector diagnostics to the benchmark
and is unsuitable for production timing because it adds host transfers.
The mode is not part of
the checkpoint/config identity, so existing state still resumes. The final actor backward uses all samples, a dynamic
row batch plus zero-mask padding, and compensates Megatron's row averaging to
produce global sum/N. Baselines keep their original token normalization.
For framework migration, apply `compat/patches/srlvr-probe-parallel.patch`
after `srlvr-hsi-framework.patch`; the former preserves the probe-only
DP-finalization gate in the otherwise untracked framework checkout.

The existing weight updater increments its version and flushes inference KV
cache before loading new weights. Request metadata is accumulated for generated
tokens only; cached prompts are not actor outputs. Earlier turn contexts remain
the original bounded-history prompts. b/s uses historical calibration records
at their original snapshot and is stored in existing Megatron common checkpoint
args (srlvr_state), including the last rollout. Resume refuses missing history.

Outputs: effective_config.json, train.log, rollouts/rollout_*.pt,
checkpoints/, common_metrics.jsonl and (SRLVR) srlvr_metrics.jsonl.
The common JSONL records terminal success counts, tasks, real attempts, actor
tokens, rollout/actor/sync/checkpoint/batch wall-clock phases. SRLVR adds b/s,
C/D/Q, confidence limits and zero reasons, lambda, B/w summaries, per-turn B,
alignment diagnostics, calibration and actor timing. Allocated device seconds
are phase wall time times 2 actor / 6 inference devices; they are resource-time,
not measured active kernel time. Feature reduction host timing includes scalar
transfer; it is additional computation even though no replay is needed.

The paper's raw estimator is before clipping, Adam preconditioning and decay.
Existing Adam (weight_decay=0, clip_grad=1) is retained; actual parameter steps
must not be presented as the raw policy gradient or inherit all its guarantees.
The two frameworks directories are git-ignored in this repository. Their
necessary local integration changes are also delivered as
compat/patches/srlvr-framework.patch for migration/review.

Verification on 2026-10-20:

- 99 mathematical, registry/interface, baseline-loss/mask, rollout and batch
  expansion tests passed. One filesystem-wide old run-directory naming audit
  was excluded after identifying its pre-existing off-rule directories.
- All three real argument parsers passed: G=8/8/1, full-softmax 1/1/-1,
  prefix cache, brief thinking, 2 actor + 6 inference devices.
- A two-device SRLVR smoke completed 4 task samples / 4 episodes / 8 full turn
  rows / 2020 actor tokens, C=2 and U=2. One actor optimizer update was recorded.
  Actual DP reduce-scatter plus full-gradient reconstruction over 69,730,304
  parameters matched a known synthetic gradient exactly (maximum error 0).
- Smoke C_hat=89675.23582260935, D_hat=40000.459071079276,
  Q_hat=-17822.710716087688. lambda=0 by explicit zero mode. B ranged
  [-0.3395811897294149, 0.018109716393117172]; final-turn B was 0.
  e was nonzero, no chosen feature saturated at -1. These are diagnostics,
  not evidence for a nonzero correction's benefit.
- Largest smoke turn-mean logp bias was .0289539 nats; largest mean absolute
  discrepancy was .0424141. Both are reported finite-precision differences
  between the BF16 inference/training implementations, not bitwise equality.
- Calibration backward/vector statistics took 17.93 s; final actor train took
  1.73 s. The rollout phase took 22.52 s, including feature reduction/transfer
  host time of 1.20 s. Whole batch including checkpoint/sync took 61.98 s.
  These timings use a 2-turn smoke, not a final 50-turn training benchmark.
- Complete checkpoint (model, optimizer, scheduler, RNG) saved at iteration 0.
  Its common args contain b_next=.01, s_next=.5099019513592785, both historical
  W records at their generating snapshot and last_rollout=0; no gradient-probe
  callback remained. Resume history is read from this same checkpoint.
- Live 320-token requests validated accumulation across forced 300-token
  packets, exact scalar/token counts, f/e identities and actual prefix reuse:
  cached tokens changed from 0 to 256 while initial-context mu matched.
  Engines use global stochastic RNG by default; per-request sampling_seed is
  effective only when deterministic inference is enabled in existing SGLang.
  No deterministic/greedy setting was changed for this experiment.
- GRPO two-device smoke completed 8 tasks / 64 episodes / 128 turn rows /
  28476 actor tokens, two original minibatch optimizer updates, a full
  checkpoint and cache-flushing weight synchronization; exit code 0.

The requested 100-round launch directories on the two independent workers are:

```text
worker-0: runs/alfworld-qwen3-1p7b-srlvr-zero-s42-r100-n8-20261020_2231
worker-1: runs/alfworld-qwen3-1p7b-grpo-s42-r100-n8-20261020_2230
```

These run directories distinguish zero from the full certified method.
The first full 50-turn SRLVR batch collected 8 episodes (3 success / 5 failure,
171631 actor tokens) and completed calibration, but the strict actor/sampler
alignment gate then stopped it at bias=.0778502375 / MAE=.0785977170 nats.
No actor optimizer step ran and no checkpoint was committed for that batch.
This is a blocking condition for the 100-round SRLVR run; it is not evidence
that strict on-policy training completed. The saved rollout is used for an
offline diagnostic replay without new environment attempts.
That replay reproduced the same violation at task sample position 5, turn 4:
711 total input tokens, 15 actor tokens; selected token 151668 had sampler
logp=-3.6537616253 versus actor logp=-2.5835657120. The persisted actor input
exactly matches the collected turn. Temperature=1, top_p=1, top_k=-1 and BF16
were verified. The remaining backend numerical/policy mismatch has not been
resolved, so strict 100-round SRLVR training is blocked. The alignment threshold,
thinking protocol and prefix cache were not changed to bypass the violation.
The installed SGLang on-policy/deterministic option rejects Ascend attention
as an unsupported backend, and is not a verified fix for this machine.

See reports/srlvr_validation_20261020/ for the mathematical/CPU test log,
live cache checks, smoke metrics, checkpoint-history restore evidence, full
batch counts and the replay's token-level alignment_failure_rank0.json.
The stopped 8B GRPO/GSPO supervisors, their training trees and the legacy
background watchdog/supervisor were stopped; their two active run roots
(approximately 19 GiB + .28 GiB) were removed. Pretrained models and task
datasets were retained.
