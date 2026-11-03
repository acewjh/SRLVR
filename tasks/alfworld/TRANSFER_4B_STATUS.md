# TRANSFER_4B 执行状态（按 TRANSFER_4B_PROMPT.md 执行）

## 2026-10-30 配置重构与 8+8 smoke 结论

`_v2_defaults.sh` 现在承载 1.7B/4B/8B 共用的 ALFWorld 协议默认：
`actor=8 rollout=8 colocate=1, NUM_ROLLOUT=100, SAVE_INTERVAL=25, G=8,
ROLLOUT_BS=4, GBS=32, turns=50, cap=4096, complete history, full RL,
LR=1e-6, KL=0.001, eps_clip=0.2, grad_clip=1.0, LoRA r/alpha=256,
show_admissible=1, command-normalization-v2-form, thinking=brief, frontier,
seed=42, OPSD_EXPERIMENT_METRICS=1`。4B 的硬件覆盖在 `_4b_backbone.sh` 固化为
`TP=2`、`SGLANG_ENABLE_NPU_GRAPH=0`、`SGLANG_MEM_FRACTION_STATIC=0.30`、
`SGLANG_MAX_TOTAL_TOKENS=32768`；目标布局仍是 8+8 colocate。8B 继续由其
32 GB 覆盖使用 cap=128、200 rollouts、cp50、TP1/PP1、2+6、non-colocate。

`colocfit2-r1-20261030_103156` 的 8+8 smoke 通过了初始 weight sync，但完整
ALFWorld rollout 过重：单个 episode 生成约 3,400--4,100 秒，运行约 76 分钟仍
没有 `step 0` 或 checkpoint。该进程组已停止，日志和 run 目录保留；它不构成
optimizer-step 通过证据。此前的 2+6 TP2 2-step smoke 仍是 TP2 的 full-RL
显存证据，不能替代本次 8+8 colocate 的端到端 step 验证。

最终 100 步 run `alfworld-qwen3-4b-grpo-s42-r100-20261030_120525` 已按上述
布局启动。日志显示 12:14:55--12:14:56 初始 `update_weights` 成功，12:15:07
开始 rollout；截至记录时仍在第一个 rollout，尚无 `step 0` 或 checkpoint。
12:24 因本服务器让给其他任务而主动停止，run、日志和状态目录均保留，便于在
其他服务器按同一配置恢复；本次没有生成 checkpoint。

## 2026-10-30 更新：4B ALFWorld 默认 TP=2

在 8×910B4-32GB 节点（PAAS_POD_ID d523eb43）实测：
`run_alfworld_v2_grpo_4b.sh` 用 TP=1 full RL 在 `MegatronTrainRayActor.init`
OOM（申请 14.99 GiB，29.49 GiB 卡已用 14.98、仅余 13.42），未跑到 rollout。
仅把 TP 改为 2（2 actor + 6 rollout，其余默认）后完整跑通：2/2 rollouts、
2/2 optimizer steps、rc=0，`iter_0000001` 53G 落盘，step grad_norm 0.145/0.290，
actor 峰值 ~18.7/29.49 GB。详见
`reports/grpo4b_smoke_20261029_215819/SMOKE_REPORT.md`。

因此 `tasks/alfworld/scripts/_4b_backbone.sh` 现在默认 `export TP="${TP:-2}"`；
所有 ALFWorld 4B wrapper（含 eval）继承该默认。例外：
`run_alfworld_v2_repeat_stop_4b.sh` 保留其显式 `TP=1`（冻结实验配置，teacher/
chunking 布局依赖）；调用方仍可用 `TP=<n>` 覆盖。测试：
`tasks/alfworld/tests/test_4b_tp_default.py`。

机器：`PAAS_POD_ID=5acbcf28-1179-477f-a014-3de133a3cb16`（Server A / OPSD baseline 节点，
8×910B4，29.49 GB/卡）。`/workspace` 为 A/B 共享盘 ⇒ 本文件在另一台机上可见。
暂停时刻：**2026-09-25 22:50 CST**。

## 1. 已完成

| 步骤 | 结果 |
|---|---|
| CPU 契约测试 | `tasks/alfworld/tests/` **34 passed**（与 prompt §2 期望一致） |
| frozen schedule | 复用既有 `configs/train_schedule_seed42.jsonl`：40 step × 4 task = 160 slot / 157 unique（与 prompt §7 的 `build_schedule(40,4,seed=42)` 同参） |
| **HSI 1-step smoke** | **PASS**（详见 §3） |
| GRPO 1-step smoke | **判据全过、随后被回收**（见 §6）：640 行训练步跑通，`grad_norm=0.00212` 非零；但 `truncated=53.6%`（见 §6b，4B 的 cap 问题） |

### 改了什么（方法未动）

1. `scripts/_common_v2.sh`：`MODEL` 与 5 个几何参数改为 env 可覆盖，**默认值原样保留 1.7B**
   （28/2048/6144/16/8）⇒ 1.7B 臂行为逐字节不变。
2. 新增 `scripts/_4b_backbone.sh`：4B 几何（36/2560/9728/32/8，与官方
   `config.json` 及 `frameworks/slime-ascend/scripts/models/qwen3-4B.sh` 一致）。
3. 新增 `run_alfworld_v2_hpi_future_4b.sh` / `run_alfworld_v2_grpo_4b.sh`：
   两臂共用 `_4b_backbone.sh` ⇒ 构造上保证同几何。
4. `_common_v2.sh` 增加 `EVAL_ONLY=1` 分支（对齐 v1 的 `base_eval`）：
   `--num-rollout 0` + `--eval-interval 1` + `--eval-config`，GPU 切法 7 rollout/1 actor，
   并在 LR guard **之前**置 `NUM_ROLLOUT=0`（否则回落 0 触发 `lr_decay_steps > 0` 断言）。
5. 新增 `run_alfworld_v2_eval_4b.sh`：`EVAL_LOAD_FROM=<ckpt>` 评 LoRA checkpoint
   （provider 由 `_common_v2.sh` 常设），不设则评 Base。

## 2. 4B 显存实测（关键新数据）

`turns=20 / cap=512` 在 4B 上**跑得下**：HSI 臂峰值 **26.44 / 29.49 GB**（89.7%）。
无需按 prompt §6.1 降参。这是 per-turn 布局的直接收益。

## 3. HSI 1-step smoke 判据（prompt §7）

日志 `logs/alfworld_v2_hsi_smoke_4b.log`（4 episode，用时 ≈12 min，其中 init ≈5 min）：

```
[v2-hpi] scored=3 skipped=5 groups=4 teacher_views=14
[v2-loss] rows=1 tokens=512 trajectories=1 w(mean=1.000000 …)
step 0: {'train/loss': -0.0018190, 'train/v2/pg_loss_mean': -0.0018190,
         'train/v2/teacher_forward_calls': 0.0, 'train/grad_norm': 0.0017615}
rollout 0: {'rollout/raw_reward': 0.875, 'rollout/truncated': 0.125,
            'rollout/opsd_advantages': 0.7515, 'rollout/response_lengths': 412.3}
```

| 判据 | 结果 |
|---|---|
| rc=0，无 OOM / NaN / Inf | PASS |
| `scored > 0` | PASS（3） |
| `pg_loss_mean` 非零 | PASS（−0.00182） |
| 非零 grad | PASS（0.00176） |
| `teacher_forward_calls = 0` | PASS |

- teacher 是 4B 自身（`_get_teacher(args.hf_checkpoint)`），每 rank 常驻，已正常加载。
- 至少一个 rank 的 `c_hat` 非零（min=−0.6841 / max=+0.6841）⇒ HSI 项真的触发了，
  不是 w≡1 的退化路径。
- `weight_mean / n_trajectories` 的数值是 slime「metrics 除以 batch token 数」的已知
  缩放伪影（prompt §5.11），不是 bug。

## 4. 暂停时刻的剩余工作（未开始）

```bash
cd /workspace/cua/repos/opsd-slime
P=$PWD

# A) HSI（G=1）cp25 —— 约 2–4 h
SCHEDULE=$P/tasks/alfworld/configs/train_schedule_seed42.jsonl \
NUM_ROLLOUT=25 ROLLOUT_BS=4 SAVE_INTERVAL=25 MAX_TURNS=20 MAX_RESPONSE=512 \
RUN_NAME=alfworld_v2_hpi_4b SAVE_DIR=$P/runs/alfworld_v2_hpi_4b LOG=$P/logs/alfworld_v2_hpi_4b.log \
setsid nohup bash tasks/alfworld/scripts/run_alfworld_v2_hpi_future_4b.sh > /tmp/hsi4b.log 2>&1 < /dev/null &

# B) GRPO（G=8）cp25 —— 约 12–16 h，必须等 A 结束（本机只有 8 卡，
#    两臂各要 actor4+rollout4，无法真并行 ⇒ 串行；schedule/配置仍完全同构）
RUN_NAME=alfworld_v2_grpo_4b SAVE_DIR=$P/runs/alfworld_v2_grpo_4b LOG=$P/logs/alfworld_v2_grpo_4b.log bash .../run_alfworld_v2_grpo_4b.sh

# C) eval（cp25 后，三者同一口径）
EVAL_LOAD_FROM= CUDA_VISIBLE… RUN_NAME=alfworld_v2_base_eval_4b bash .../run_alfworld_v2_eval_4b.sh   # Base
EVAL_LOAD_FROM=$P/runs/alfworld_v2_hpi_4b  RUN_NAME=alfworld_v2_hpi_4b_eval  bash ...
EVAL_LOAD_FROM=$P/runs/alfworld_v2_grpo_4b RUN_NAME=alfworld_v2_grpo_4b_eval bash ...
python tasks/alfworld/v1/report_eval.py --dump "$P/runs/alfworld_v2_*_4b_eval/*.pt"
```

（prompt §7 写的是 `tasks/alfworld/report_eval.py`，**v2 无此文件**，用 v1 的
`tasks/alfworld/v1/report_eval.py`，它读的是 `--save-debug-rollout-data` 的 `*.pt`。）

## 5. 两条必须记住的约束

1. **一次一个 launcher**：本机 8 卡，两臂必须串行，不能真"同时启动"（prompt §7 的
   "同时启动"是针对双节点各 8 卡的原始设定）。Ray GCS 冲突是历史事故（prompt §5.7）。
2. **recompute × LoRA 的静默零梯度**：兄弟仓 `opsd-slime-cua-rl/train_scripts/
   run_math_opsd_slime_npu_4b.sh` 实测 `--recompute-num-layers 36`（全层）会因
   Megatron reentrant CheckpointFunction 把 LoRA 参数从图上摘掉 ⇒ grad 恒 0、100 步权重不变。
   `_common_v2.sh` 用的是 `--recompute-num-layers 1`，本次 4B smoke 实测
   `grad_norm=0.00176 ≠ 0`，**未触发**；但正式 cp25 仍应在 step 2–3 复核 grad_norm 非零。

## 6. GRPO smoke：训练步跑通了，随后被杀

日志在 **23:04:07 `Timer actor_train start` → 23:04:12 `step 0`** 写下了训练步结果，
之后（23:04:55 / 引擎侧 23:05:12）**没有** `[v2] grpo exited rc=` 行就断了
⇒ step 跑完，启动器是在 save/teardown 阶段被杀的，不是 OOM 也不是崩溃
（`Traceback` 只有 SGLang `openai_harmony` 缺失那 4 处无害告警）。

```
rollout 0: {'rollout/raw_reward': 0.4639, 'rollout/truncated': 0.5361,
            'rollout/response_lengths': 419.7, 'rollout/advantages': -0.205}
step 0:    {'train/loss': 0.0027901, 'train/grad_norm': 0.0021230,
            'train/v2/teacher_forward_calls': 0.0}
```

⇒ **640 行（32 episode × 20 轮）的训练步在 4B 上跑得下**，`grad_norm` 非零。
（先前"未验证训练步"的判断是错的 —— 那次读日志读早了。）

**仍成立的部分**：`setsid nohup … &` 活不过会话退出，死亡时刻 ≈ 会话退出时刻。
正式长跑**必须配会话外的 supervisor**（可抄 `scripts/serverA_supervisor.sh`：
flock 单实例 + 心跳文件 + `COMPLETE` stand-down；注意其锁 fd 会被子进程继承，
续跑要带 `exec 9>&-` 之类的手脚）。

## 6b. 4B 的 cap 问题（比调度更值得先决策）

同 schedule、同 `cap=512`，两臂截断率差 4 倍：

| 臂 | episode | `rollout/truncated` | `raw_reward` |
|---|---|---|---|
| HSI (G=1) | 4 | **12.5%** | 0.875 |
| GRPO (G=8) | 32 | **53.6%** | 0.464 |

## 6c. 真相：cap 直接决定"有多少 turn 作废"（cp25 因此用 cap=1024）

**判别方法（可复用）**：读 `runs/<run>/rollout_0.pt` 的 `samples[*]`，
把 `tokens[-response_length:]` 用 Qwen3-4B tokenizer decode，
用 `<action>(.*?)</action>` 正则判"有没有合法动作"，再和 `response_length` 比对。

| run | n | 无 `<action>` | 顶到上限 | 两者是否同一集合 |
|---|---|---|---|---|
| cap=512 | 416 | **236 (56.7%)** | 236 | **True**（差集 0） |
| cap=1024 | 256 | **45 (17.6%)** | 45 | **True**（差集 0） |

**完全同一集合，零误差** ⇒ 模型只要给够 token 就必然吐出合法 `<action>`，
作废 100% 由长度切断造成。按 cap=1024 的长度分布外推：

| cap | 作废 turn 比例 |
|---|---|
| 512 | **56.7%** |
| 728 | **34.4%** |
| 1024 | **17.6%** |

`rollout/truncated` 这个指标**不能用来看这件事**：它只是引擎 finish_reason 的透传，
`action_parser.py:157` 是 `ParsedAction(ParseStatus.TRUNCATED, action, ...)`
——**提取到合法动作照样标 TRUNCATED**。实测 83% 被标 truncated 的 turn 带合法动作，
且它与"有无动作"几乎不相关（truncated 组 111/134 有动作，completed 组 100/122 有动作）。
我（本会话）第一轮读长度直方图读对了，第二轮改用 `status` 分桶反而把自己带偏，
得出过「cap 无关」的错误结论 —— **以后判 turn 是否可用，只认 `<action>` 正则，不认 status**。

## 6d. cp25 最终配置

两臂都跑 **cap=1024**（25 步、GBS=32、turns=20、同一份 frozen schedule）：

| 节点 | RUN_NAME | 启动 |
|---|---|---|
| worker-0 | `alfworld_v2_grpo_4b_c1024` | 2026-09-26 00:21 |
| worker-1 | `alfworld_v2_hpi_4b_c1024` | 2026-09-26 00:26 |

`runs/alfworld_v2_{hpi,grpo}_4b`（无后缀）是 cap=512 那次**已废弃**的尝试，
只留 rollout dump、无 checkpoint，**未删除**，留作对照证据。
`runs/alfworld_v2_hpi_4b_c1024_oom_at_step9` 是 cap=1024 首次尝试的 OOM 现场（改名保留）。

### 4B cap=1024 在 HSI 臂 OOM —— 与修法

死因（step 9）：`megatron_utils/loss.py:84 logits.div(...)`，要 1.09 GiB、只剩 1.016 GiB，
**差 ~0.09 GiB（2%）**。最长行长实测 1842 token → 单份 fp32 词表张量 1.04 GiB，`.div()` 再一份。

根因是 **HSI 臂常驻的 4B teacher（~6 GB）**：同 cap 下 GRPO 峰值 20.65 GB vs HSI ~26.5 GB。

修法 = `scorer.py` 加 `_park_teacher` / `_teacher_on_device`：postprocess 评完分把 teacher
`to("cpu")`，下次评分前移回（`score_view_logprobs` 每调用从参数读 device，自动跟随）。
**纯显存放置，teacher 冻结 ⇒ 数值不变**；34 项契约测试重跑全过。实测修复后 HSI 重跑无 OOM。

### 真实 schedule 上的两项关键实测（2026-09-26 02:10）

| 量 | GRPO cp25 | HSI cp25 |
|---|---|---|
| 无合法 `<action>` 的行 | **20.9%**（347/1664） | **13.3%**（34/256） |
| max_row（token） | 1816 | 1854 |
| **零梯度优化步占比** | **62%**（45 步中 28 步） | **0%**（8 步） |

- cap=1024 的修复在真实 schedule 上同样有效（diag schedule 的 52%→18% 规律成立）。
  `rollout/truncated` 报的 93% 是假象（该指标是 finish_reason 透传，见 §6c）。
- GRPO 的零梯度机制 = 真实 schedule 上 `success_rate` 仅 0.09–0.25 ⇒ 同 task 的 8 条轨迹
  常全失败 ⇒ `mixed_group_rate=0` ⇒ advantage 0。**这是 GRPO 在 ALFWorld 上的固有性质**
  （1.7B 版笔记记的同一现象），改它等于改方法。HSI 因 `s=2R−1` 对失败轨迹给 −1，
  天然规避 —— 0% vs 62% 正是该方法设计主张的直接证据。

## 6e. 评 LoRA checkpoint 的第二个坑：`KeyError: 'optimizer'`

`--load <ckpt>` 走 `EVAL_LOAD_FROM` 时**不会**触发 `_common_v2.sh` 里给 `RESUME_FROM` 准备的
`--no-load-optim`。而训练臂用 `--no-save-optim --no-save-rng` 存盘，checkpoint 里没有
optimizer 状态，Megatron 却硬读：

```
megatron/training/checkpointing.py:1720  optimizer.load_state_dict(state_dict['optimizer'])
KeyError: 'optimizer'   →  eval 启动 ~6 min 后 rc=1
```

**修法 = 评 checkpoint 时传 `RESUME_FROM=<ckpt dir>` 而不是 `EVAL_LOAD_FROM`**：
`--load` 会回落到 `RESUME_FROM`，同时 `RESUME_ARGS=(--no-load-optim --no-load-rng)` 被填上。
两者都指向同一个 checkpoint，语义等价，但只有 `RESUME_FROM` 那条路会带 no-load 标志。

⚠️ **不要把这个 eval 挂在 `supervise_4b.sh` 下面**：supervisor 重启时会**显式**
`RESUME_FROM="$RESUME_FROM_V"`，而 eval 的 SAVE_DIR 永远没有 checkpoint ⇒ `RESUME_FROM=""`
⇒ `--load` 回落到 `$MODEL`（Base）⇒ **会静默产生一个标着 HSI 的 Base 评测结果**。
这正是 §5.12 那个坑的静默变体，比崩溃更难发现。评 checkpoint 时**直接 setsid nohup 起**。

### checkpoint 时机（协议固有风险）

`slime/utils/misc.py:should_run_periodic_action` 只在 `rollout_id == num_rollout-1`
（**最后一个 rollout**）返回 True ⇒ **全程只有末次落盘，没有中间 checkpoint**。
`--num-rollout 25` 是 25 个 **rollout**，每个 rollout 切出 ~15 个优化步（per-turn 行 ÷ GBS），
所以总步数是几百步。中途挂掉 = 全部重来。

## 7. 其他

- 未删除任何 `runs/` 产物。仅在跑 pytest 前按本仓惯例清过 `__pycache__`
  （该 FS mtime 不失效，不清会跑到旧字节码）。
- Server A 的 paper baseline 队列已完成（`logs/serverA_queue.COMPLETE` 在位），
  supervisor 处于 standing-down，**不会**自动起队列与本任务抢卡。
