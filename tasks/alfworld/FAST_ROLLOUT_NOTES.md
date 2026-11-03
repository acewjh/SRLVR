# Qwen3-8B ALFWorld rollout efficiency experiment

2026-10-20. GRPO and GSPO use the same settings. Dynamic sampling and a rollout
pool remain disabled. Each optimizer rollout still contains the four scheduled
tasks with eight independent episodes each; all episodes finish before training.

## Motivation and changes

Recent completed cycles spent about 85% of time sampling/waiting. GRPO's latest
ten cycles averaged 35.9 minutes waiting and 6.5 minutes training; GSPO averaged
30.0 and 5.2 minutes. A cycle generated approximately 0.6–1.4 million tokens.

- Default execution split: actor DP2, six single-card SGLang engines. TP1/PP1,
  micro-batch 1, global batch 32, Qwen3-8B, LoRA, LR and losses stay the same.
- Enable radix prefix caching; the existing weight-update path pauses engines
  and flushes caches before replacing policy weights.
- Baselines request zero top-k log-prob candidates. Sampled-token log-probs
  remain enabled and aligned with generated tokens. HPI retains its default 20.
- A shared prompt asks for 64 thinking tokens in one to three short sentences,
  describing only the immediate subgoal/decision. This is a soft instruction,
  not a forced decoder cutoff. Native Qwen thinking stays enabled, output cap
  remains 4096, max turns 50, history 8, and sampling T=1/top_p=1/top_k=-1.
- Per-episode logs record generation, environment, and wall time.
- Live profiling (`py-spy dump --nonblocking`) found the CPU still deep-copying
  complete trajectories for each turn after all 32 episodes finished. Serialize
  each finished trajectory once and share its read-only payload across turn rows.
  All stored trajectory content, token IDs, rewards, and log-probs are retained.

## Related implementations consulted

1. Qwen3: `/think`, `/no_think` and `enable_thinking=False` control thinking mode.
   Disabling thinking is not the same as brief thinking, so the initial experiment
   keeps thinking enabled.
   https://github.com/QwenLM/Qwen3#readme
   https://huggingface.co/Qwen/Qwen3-8B#advanced-usage-switching-between-thinking-and-non-thinking-modes-via-user-input
2. TALE (Token-Budget-Aware LLM Reasoning): explicit token budgets in prompts
   shorten reasoning; budget selection affects accuracy. This motivates the soft
   prompt budget, without adopting a new reward or extra budget-estimation call.
   https://arxiv.org/abs/2412.18547
   https://github.com/GeniusHTX/TALE
3. s1 uses budget forcing to control reasoning length. Local SGLang implements
   `Qwen3ThinkingBudgetLogitProcessor`, which forces newline/end-think logits.
   That route would require matching forced-token behavior in PPO probabilities
   and masks; it is not enabled for these native baselines.
   https://github.com/simplescaling/s1#vllm-with-budget-forcing
   frameworks/sglang/python/sglang/srt/sampling/custom_logit_processor.py

## Validation and run layout

`launch_alfworld_8b_fast_campaign.sh` first runs two fresh-base rollouts with the
same schedule/seeds and validates the full optimizer/RNG checkpoint. Its screen
requires nonzero gradients, rollout-0 success >=75%, mean success >=65%, at least
25% fewer mean response tokens and 5% shorter complete cycles compared with
rollouts 0 and 1 of the preserved 20261020_002240 runs. The literal 64-token prompt
target is not a hard cutoff. These thresholds only decide whether to
start the long run; this small training screen cannot establish held-out accuracy.
If it passes, a separate experiment starts from the base model for 200 rollouts,
saving at 50/100/150/200. The final checkpoint is evaluated on both full held-out
splits with the same brief-thinking prompt and full 4096-token response allowance.
The GRPO node subsequently evaluates the unchanged base model under this same
prompt, providing a matching control for both methods.

Run root:
`runs/alfworld-qwen3-8b-{method}/{parameter-group}/{stamp}/`
with `checkpoints/`, `rollouts/`, `train.log`, `supervisor.log`, `config.env`, and
`eval/` underneath. Preflight runs are separate and identified by `x2` and the
`-preflight` suffix. Old 4:4 runs are retained as diagnostics and are superseded.

## Initial live measurements (not held-out evaluation)

Preflight stamp: `20261020_125554`. The initially loaded workers still use the
old per-turn trajectory serialization; newly started full runs pick up its fix.

| Rollout 0, same scheduled tasks | Old GRPO | New GRPO | Old GSPO | New GSPO |
| --- | ---: | ---: | ---: | ---: |
| Episode success | 28/32 | 30/32 | 27/32 | 28/32 |
| Mean response tokens | 776.3 | 540.7 | 781.3 | 570.5 |
| Sampling seconds | 1292.2 | 826.9 | 1292.1 | 1065.4 |
| Complete cycle seconds | 1581.2 | 1216.2 | 1584.1 | 1529.3 |

This screen checks that the changes are worth testing longer; it does not show
that either trained policy outperforms the base model on the full benchmark.

Final two-rollout screen: GRPO complete-cycle means 1348.3 vs 1455.6 seconds
(7.4% shorter); GSPO 1404.2 vs 1476.8 seconds (4.9% shorter). Mean response
reductions were 30.0% and 31.4%. Combined episode successes: GRPO 50/64 vs
47/64, GSPO 49/64 vs 49/64. Both iteration-1 full-state checkpoints validated
212 tensor entries. GSPO missed the 5% speed gate narrowly: the failed gate is
retained alongside an explicit execution decision to test the serialization fix
in fresh full-run workers. Neither arm resumes the two-rollout checkpoint.

Preflight launchers returned rc=2 after their checkpoints were saved: a concurrent
shared-tree edit moved `_common_v2.sh` while Bash retained its old read offset,
producing `ard-port: command not found` / unexpected `fi` on return from Python.
The current complete script passes `bash -n`; full-run shells started after that
edit. Both training phases and validated checkpoints remain usable evidence.

## GSPO clean restart, 2026-10-20

User requested deletion of local 8B GSPO logs/checkpoints and a fresh-base run.
Old GSPO run folders, rollout dumps, checkpoints, and dedicated logs were removed
(22.37 GiB). GRPO continues separately. The new GSPO defaults are EPS_CLIP=0.0003,
EPS_CLIP_HIGH=0.0004, LR=1e-6; the run group includes these values. Both the direct
wrapper and fast launcher honor the defaults, and config.env records the actual
values. Two actor plus six rollout NPUs, brief native thinking, prefix cache,
G=8, task batch=4, 200 rollouts and 50-rollout checkpoints remain selected.
The supervisor evaluates iteration 199 after the complete fresh-base run.
Cleanup paths and new run identity: logs/alfworld-fast-campaign/20261020_172614/restart_manifest.json.

GSPO clean restart verified at 2026-10-20T17:36:28.861664+08:00: /workspace/cua/repos/opsd-slime/runs/alfworld-qwen3-8b-gspo/s42x200-a2r6-brief64-k0-cache1-clip0.0003_0.0004-lr1e-6/20261020_172614; clip=0.0003/0.0004, lr=1e-6, base load, 2/6, 200 rollouts, checkpoints every 50. Removed old GSPO outputs and logs: 22.37 GiB. GRPO remains running at runs/alfworld-qwen3-8b-grpo/s42x200-a2r6-brief64-k0-cache1/20261020_125554.
