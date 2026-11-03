# Training data dashboard

The dashboard is based on the private `agentic_train_scale` dashboard layout: global overview, current-run KPIs, grouped metric charts, batch summaries, trajectory/curriculum information, logs, run comparison, and history. It reads the local telemetry emitted by opsd-slime and does not require W&B.

Start it from the repository root:

```bash
PYTHONPATH=. python3 -m opsd_dashboard.server --root runs
```

Open `http://127.0.0.1:8765/`. The page supports run selection, comparison, light/dark mode, chart/table toggles, and manual refresh. Automatic refresh defaults to five minutes and can be disabled with the checkbox. Metrics are read from append-only `<run>/dashboard/events-*.jsonl` files and tolerate incomplete final lines.

The local adapters expose completion rate, success rate, action validity, rollout duration, turns, trajectory/generated tokens, training scalars, and the reference-style rollout/trajectory metric groups. Missing values stay empty rather than being estimated. Historical runs only contain fields that were persisted at the time.

## Shared runs and metric semantics

A run and its `checkpoints/` directory are one experiment. The reader merges
`<run>/dashboard` and the legacy `<run>/checkpoints/dashboard` location without
moving files. New training processes put telemetry at `<run>/dashboard` even
when `--save` points at `<run>/checkpoints`. Active processes can keep writing
their previous location. Independent nested experiments remain selectable;
model shard, rollout and trajectory directories are excluded from discovery.

Metrics are combined from native telemetry, `train.log`, `common_metrics.jsonl`
and `srlvr_metrics.jsonl`. The first two take precedence over ledger fallbacks;
within each source tier the latest record wins. Each metric has one sorted point
per step. Actor curves keep the original train step; episode and rollout curves
keep rollout IDs. Training and evaluation metrics use separate namespaces.

- `rollout/response_len/*` and `rollout/total_lengths` are framework **training-row**
  lengths. In ALFWorld a row may represent one turn, not a whole episode.
- `episodes/generated_tokens/*` counts generated tokens once per episode.
  `episodes/trajectory_tokens/*` is initial prompt plus generated tokens, excluding
  repeated history prompts. `batch/generated_tokens` is a batch total.
- A legacy common ledger can provide a mean from an explicit total and episode
  count, but cannot supply missing per-episode duration, min/max, or a unique
  trajectory length. No artificial episode is constructed from a batch total.
- Actor charts retain persisted values without attempting to reconstruct missing
  denominators. Despite their names, SRLVR v2/v3 `train/srlvr_loss_sum` and
  `train/srlvr_reference_kl_sum` are token sums **divided by training-row count**
  (including padding) by the framework logger. The former includes weighted
  reference KL, not just PPO surrogate loss. The actual v2/v3 objective divides
  the token sum by trajectory count, using `rows/N` to compensate row reduction.
  GRPO's default `OPSD_V2_LOSS_NORM=traj` objective averages tokens within each
  trajectory, then averages trajectories; SRLVR v4 uses a global token mean.
  GRPO `train/v2/*` and v4 mean/clip diagnostics are already microbatch means
  before the framework divides their accumulated values by global token count;
  they are shown as logged diagnostics, not true means or percentages. GRPO's
  `train/loss` carries the objective reduction. Historical values are not rescaled.
- `train/grad_norm` is the norm of the already normalized training gradient,
  returned by the optimizer **before clipping**. A value above the configured
  clipping threshold is expected and does not imply clipping failed. Different
  objective reductions make norms across methods unsuitable for direct comparison.
- SRLVR `reference_seconds`, `score_seconds`, `calibration_seconds` and
  `actor_seconds` are shown separately. The common `actor_phase_seconds` spans
  the actor phase including its preparation. These intervals are not added to
  fabricate a new total. `batch_wall_seconds` is the recorded end-to-end total.
  Runs without a particular timer leave that chart empty.
- Remote process liveness is unknown unless an explicit exit record exists;
  this server's PID namespace is not used to judge another server's experiment.

Refresh defaults to **five minutes**, with a manual refresh button. Log/JSONL
readers cache parsed records and read only appended complete lines; partial final
lines are retried on the next refresh. No `.pt` checkpoint files are loaded.
