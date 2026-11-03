# SRLVR-v2 / SRLVR-v3

`methods/srlvr_hsi.py` is the shared implementation. `SRLVR_VARIANT=v2|v3`
selects only `f`: v2 is `sigmoid(log p-log q)` and v3 is one minus its normalized
binary entropy. `mu=sum_v p(v)f(v)` is a full-vocabulary student expectation;
v3's mixture integral is normalized JSD, while its student expectation generally
is not. Features, centering, and future corrections are detached.

The causal student tokens come directly from the rollout. After each complete
trajectory, `tasks/alfworld/rollout.py` uses the existing `render_hpi_block`,
`render_outcome_descriptor`, and `hsi_teacher_view` attachment to build the Full
teacher view, including the final turn's outcome descriptor. No Control view is
used. HSI's hidden-state tap and prediction-position alignment are reused.
Projection handles at most `SRLVR_POSITION_CHUNK` positions (default existing HSI
chunk size, 64) at once and immediately reduces the complete vocabulary to
scalars. The scorer uses the same merged LoRA publication weights, positive
temperature, FP32 softmax/normalization and model-native vocabulary as the sampler.
Non-default penalties, grammar or support constraints are rejected explicitly,
not silently replaced by a different distribution. The current full-softmax
sampling configuration is unchanged. Recorded chosen-token replay errors expose
remaining numerical differences between serving and trainer kernels.

A reverse actor-token scan computes
`S[j]=(1-gamma)e[j]+gamma*S[j+1]` and reads `B[t]` strictly after each turn.
Actor masks exclude environment/padding positions; the last turn is zero.
There is no extra clipping or length/trajectory normalization.

Historical `b,s` use the existing score-norm weighting/window rules. Initial
lambda is zero. Every real trajectory, including each calibration probe, uses
the lambda frozen from previous batches. Raw score-function probes calculate
only C and D at the generation snapshot. The current C/D ratio is retained for
the next batch and bounded with that next batch's frozen baseline. Probes do not
step the optimizer or scheduler. The actor uses the same tokenwise clipped PPO
sum divided once by the real trajectory count in both variants; static
microbatch and DP normalization are compensated. The feature itself does not
require an initial-reference model; the shared ALFWorld v2 launcher retains the
existing initial-checkpoint `low_var_kl` term and reference path in the actor
objective.

Because the Full teacher input contains the realized trajectory's future and
outcome, this is vocabulary centering conditional on that view. It does not
prove a zero-mean future correction, an unbiased control variate, or a martingale
property. C/D is empirical second-moment calibration; no variance-optimality,
certified confidence bound or Q estimator is asserted or implemented.

## Entrypoints and launch

The legacy `srlvr` registry entry, `methods/srlvr.py`, legacy defaults and existing
HSI estimator are preserved. New registry entries are `srlvr_v2` / `srlvr_v3`.
Both new wrappers inherit the server's current 1.7B model, LoRA, learning rate,
harness, prompt, sampling, parallel layout, saving and evaluation defaults. The
pair wrappers default to 25 rollout/update cycles. The full-r100 launcher sets
100 cycles and save interval 25; framework final-save behavior writes
`iter_0000099` after update 100. The wrapper verifies the configured successful
update count and final checkpoint, then invokes the existing checkpoint
benchmark automatically with the same model, LoRA and interaction settings.

Gamma is **0.99** for the paired full/compact runs. The dedicated
`SRLVR-v2-thinking-full-r100` entry sets it explicitly:

```bash
SRLVR_GAMMA=0.99 \
  python3 tasks/alfworld/scripts/launch_srlvr_hsi_pair.py
```

The pair launcher uses `VC_WORKER_HOSTS` and SSH port 2222 as documented in
`NODE_CONNECTIVITY.md`, verifies both devices are available, and starts v2 on
worker-0 and v3 on worker-1 through detached SSH commands. Both nodes source one
common experiment environment; run names and output paths differ. It performs
one launch, with no polling, auto-retry, tuning, or existing-job termination.
`--check` checks identities and device availability without starting a run.
Individual entries are `run_alfworld_srlvr_v2_1p7b.sh` and
`run_alfworld_srlvr_v3_1p7b.sh`; both chain the benchmark by default.

## Memory and verification

Initial batches remain unchanged: 32 real trajectories, configured probe count,
actor microbatch 1, inference prefill chunk 8192, and scoring position chunk 64.
No OOM has been observed for the new runs because training has not yet started.
Only an observed offending batch parameter may be halved, identically for both
runs. `SGLANG_CHUNKED_PREFILL_SIZE` exposes the existing prefill batch knob without
changing its default. An actor microbatch of 1 cannot be halved legally; report
that case rather than shortening context, responses, or turn limits.

`tasks/alfworld/tests/test_srlvr_hsi.py` checks the formula/JSD identity, exact
student expectation, endpoints, detached outputs, Full-view isolation, alignment,
chunk limits, actor masks, reverse suffix scan, history timing, both shared
variants, actual framework normalization algebra and 25 updates including raw
probe exclusion. The 25-update CPU test uses mocked optimizer execution; hardware
DP reconstruction is also checked by the existing runtime acceptance hook on
first training batch. These checks do not substitute for an actual completed run.

The ignored local framework changes are reviewable/reproducible in
`compat/patches/srlvr-hsi-framework.patch`, applied after the existing
`srlvr-framework.patch`. They dispatch new batches to the shared method, keep
sampler feature computation out of rollout for v2/v3, reject unsupported sampler
constraints, and count successful actor optimizer steps.
