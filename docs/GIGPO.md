# GiGPO on ALFWorld

GiGPO (*Group-in-Group Policy Optimization for LLM Agent Training*,
arXiv:2505.10978) adds a step-level advantage to GRPO's episode-level one. It is
implemented here as a **pure advantage estimator**: the rollout, per-turn rows,
custom loss (`tasks.alfworld.loss.alfworld_v2_loss`), trajectory-balanced
reduction, LoRA/full-RL geometry and evaluator are the ones the matched GRPO arm
already uses. Only `Sample.opsd_advantages` changes.

## Method

For task group `g` (the G=8 episodes of one ALFWorld game) and turn `t` of
trajectory `i`:

```
A_{i,t} = A^E_i + w * A^S_{i,t}
A^E_i   = (R_i - mean_g(R)) / (std_g(R) + eps)            # existing GRPO
A^S_i,t = (G_{i,t} - mean_a(G)) / (std_a(G) + eps)        # anchor-state group
G_{i,t} = sum_{k>=t} gamma^(k-t) * r_{i,k}                # return-to-go
```

* **Anchor state group** `a`: all turns of task group `g` whose raw environment
  state is identical, `(task_id, anchor_observation)` exactly -- the same
  `anchor` semantics as the reference implementation (history and thinking are
  excluded). `anchor_observation` is the environment state, NOT the parser's
  invalid-action rejection message: two rejected attempts from one room share an
  anchor even when they tried different commands, because the state did not
  change. Records written before this field existed fall back to `observation`.
  The task id is part of the key, so two games that happen to share a room
  description never pool returns.
* A group of one, or a group whose returns are all equal, contributes exactly
  `0`. This is the important degenerate case: an anchor set that is mostly
  singletons reduces GiGPO to plain GRPO rather than injecting noise.
* `mixed` describes the EPISODE term (reward variance). The step term can carry
  signal even when every episode succeeds, because with `gamma < 1` two
  successful trajectories of different lengths have different return-to-go at a
  shared anchor. `has_signal` therefore checks the combined advantage; do not
  gate GiGPO on `mixed`.
* ALFWorld pays a sparse terminal reward, so `G_{i,t}` is
  `gamma^(T-1-t)` for a success and `0` otherwise. The recurrence is written
  generally (`tasks/alfworld/advantages.py::_step_returns`) so dense reward would
  work unchanged.
* `A^E` is byte-for-byte `compute_group_advantages`, i.e. the GRPO control's
  episode estimator. The official repo's default `compute_mean_std_cross_steps`
  (baseline mean over step rows, which weights long trajectories more) is
  deliberately **not** used, so the A/B against GRPO differs only by the step
  term.

## Knobs

| env | default | meaning |
|---|---:|---|
| `ALFWORLD_V2_GIGPO` | `0` | select the arm (the registry sets `1`) |
| `ALFWORLD_V2_GIGPO_STEP_W` | `1.0` | `w` in the combined advantage |
| `ALFWORLD_V2_GIGPO_GAMMA` | `0.95` | discount of the step return-to-go |
| `ALFWORLD_V2_GIGPO_MODE` | `mean_std_norm` | `mean_std_norm` or `mean_norm` |

`mean_std_norm` is the reference default (standardise the anchor group);
`mean_norm` only subtracts the anchor mean.

## Running

```bash
# matched control (already exists): ARM=grpo bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
RUN_NAME=... NUM_ROLLOUT=50 bash tasks/alfworld/scripts/run_alfworld_v2_gigpo_1p7b.sh
```

The entry is `run_alfworld_v2_1p7b.sh` with `ARM=gigpo`; the registry resolves
the loss/rollout/convert/G=8 and force-exports `ALFWORLD_V2_GIGPO=1`. A non-GiGPO
arm clears a leaked value, so a supervisor shell cannot silently switch methods.

## Observability

Every rollout logs one `[gigpo]` line and these metrics:

```
rollout/gigpo_task_groups            groups in this rollout (4 at ROLLOUT_BS=4)
rollout/gigpo_anchor_groups          distinct anchor states
rollout/gigpo_turns                  turn rows
rollout/gigpo_singleton_fraction     fraction of turns with no anchor peer
rollout/gigpo_mean_group_size        turns / anchor groups
rollout/gigpo_max_group_size         largest anchor group
rollout/gigpo_step_nonzero_fraction  fraction of turns with a nonzero step term
rollout/gigpo_step_adv_mean_abs      mean |A^S|
rollout/gigpo_episode_adv_mean_abs   mean |A^E|
```

**Health check:** if `singleton_fraction` is near 1 and
`step_nonzero_fraction` near 0, the step term is dead and the run is plain GRPO.
On ALFWorld the first turn is always shared by all G trajectories, and identical
action prefixes create deeper shared anchors, so the expected singleton fraction
is well below 1.

## Caveats

* The step term is computed inside `_run_group`, where the episode structure is
  still visible. Do not move it into the loss: after the DP split no rank sees a
  whole task group.
* GiGPO and OPSD/SRLVR/HSI are mutually exclusive; the rollout raises rather
  than letting one estimator overwrite another's advantage channel.
* `ALFWORLD_V2_ADVANTAGE_MODE=batch` (REINFORCE++) is refused under GiGPO: it
  would replace the per-turn advantages with a single batch scalar.
* The step term rides the shared trajectory-balanced reducer: within one
  trajectory, turn rows are weighted by their token share (`w_it = N_it / N_i`).
  The anchor-group centring is unweighted over turns, so a long turn carries
  more gradient than a short one -- the same token-vs-step weighting the
  reference implementation has, and the reason the metric to watch is
  `step_nonzero_fraction`, not an exact zero-sum of the reduced gradient.
* `anchor_observation` is additive on `TurnRecord` (`""` = fall back to
  `observation`), so old rollout dumps remain replayable.
