# 交给 eval 节点 agent 的任务书：ALFWorld Future-HPI 评测监督

> 这份文档自包含。你的角色是**评测节点**，训练节点在另一台机器上。
> **不要起任何训练**；你只做评测与汇报。

---

## 0. 你的目标

对训练节点产出的 ALFWorld 检查点做 `valid_seen` / `valid_unseen` 评测，并把
**每臂每 split 的 success rate + n** 汇报出来。训练端会往共享盘的
`runs/alfworld_hsi_cp25/` 与 `runs/alfworld_v2_grpo_cp25/` 里写新 checkpoint
（`iter_0000049` / `iter_0000074` / `iter_0000099`）。

---

## 1. 环境（共享盘，无需重装）

```bash
BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
conda activate cua-rl          # 训练/评测固定用这个 env
export PYTHONPATH="$PROJ/frameworks/slime-ascend:$PROJ"
export ALFWORLD_HPI_DATA=$BASE/data/alfworld/json_2.1.1
cd $PROJ
```

数据、代码、checkpoint 都在共享盘上，两台机看到的是同一份。

---

## 2. 🚨 一条会直接让评测崩掉的坑（必须照做）

**评一个 LoRA checkpoint 时，必须用 `RESUME_FROM=`，绝对不能用 `EVAL_LOAD=`。**

原因：`_common_v2.sh` 只在 `RESUME_FROM` 分支里加 `--no-load-optim --no-load-rng`。
两臂训练都用 `--no-save-optim` 存的，checkpoint 里根本没有 optimizer 状态；用
`EVAL_LOAD` 会走 Megatron 的默认加载路径，直接崩：

```
File ".../megatron/training/checkpointing.py", line 1720, in load_checkpoint
    optimizer.load_state_dict(state_dict['optimizer'])
KeyError: 'optimizer'
```

（这个坑真踩过，浪费了 ~1 小时节点时间。）

例外：**评 Base 用 `EVAL_LOAD`**，因为 Base 是纯 HF 目录，没有 Megatron
checkpoint 可加载，`RESUME_FROM` 反而会因为找不到
`latest_checkpointed_iteration.txt` 而退出。

---

## 3. 一条评测的完整命令

```bash
# ---- 评一个 LoRA checkpoint（cp49 / cp74 / cp99 都照这个改 RUN） ----
cd $PROJ
export RESUME_FROM=$PROJ/runs/alfworld_hsi_cp25        # ← 换臂就换这里
export RUN_NAME=alfworld_eval_hsi_cp49
export SAVE_DIR=$PROJ/runs/alfworld_eval_hsi_cp49
export LOG=$PROJ/logs/alfworld_eval_hsi_cp49.log
unset EVAL_LOAD EVAL_LOAD_FROM
setsid nohup bash $PROJ/tasks/alfworld/scripts/run_alfworld_v2_eval.sh \
    > /tmp/launch_$RUN_NAME.log 2>&1 < /dev/null &

# ---- 评 Base（纯 HF 目录，只用这一次） ----
export EVAL_LOAD=$BASE/checkpoints/Qwen3-1.7B
export RUN_NAME=alfworld_eval_base_1p7b
unset RESUME_FROM
```

两臂的 run 目录：
- Ours / HSI：`runs/alfworld_hsi_cp25`
- GRPO：`runs/alfworld_v2_grpo_cp25`

LoRA provider 已经在 `_common_v2.sh` 里写死，不用你加参数。

---

## 4. 硬性纪律

1. **一个节点同时只能有一个 launcher。** 评测脚本自带 flock（按 `RUN_NAME` 上锁），
   但别靠它兜底——起之前先 `pgrep -f '[s]lime-ascend/train.py'` 确认为空。
2. **`wandb` 全关**（脚本已设 `WANDB_MODE=disabled`）。
3. **不要删任何东西。** `runs/` 下的产物（adapter / `trainer_state` / rollout dump）
   是论文可复现证据；要删先列清单等用户确认。
4. **不要碰训练节点**，也不要在训练节点上起评测。
5. 一次评测（两 split 共 274 局）约 **2h15m**；`valid_seen`(140) 先出，约 65–70 min。

---

## 5. 怎么判断跑完 / 怎么汇报

评测脚本会把结果以 `[v2-eval]` 行写进 `$LOG`：

```bash
grep -aE '\[v2-eval\]' $LOG | grep -aoE '\[v2-eval\] [a-z_]+: n=[0-9]+ success_rate=[0-9.]+'
# 期望两行：
# [v2-eval] valid_seen:   n=140 success_rate=0.xxxx
# [v2-eval] valid_unseen: n=134 success_rate=0.xxxx
```

健康判据：`n` 必须**正好是 140 和 134**。对不上说明有局没跑完，别报。

汇报格式（一行一臂，务必带 n）：

```
<arm> cp<NN>  valid_seen 0.xxxx (n=140)   valid_unseen 0.xxxx (n=134)
```

ignore 这些 SGLang 噪音，不是故障：`Invalid HTTP request`、`OpenAIServingResponses`、
`error_injection`。

---

## 6. 两件必须随数字一起说的事

1. **分辨率**：n=274（140+134），单臂 1 SE ≈ ±1.6 pp。**跨臂差 <3 pp 不可分辨**，
   别把噪声当结论，也别用 p 值下断言。
2. **HSI 臂的 hindsight 项实测 `e_t ≡ 0`**：privileged 块在 assistant turn 之前，
   影响全被 `<think>` 吸收，到 `<action>` 时两个视图的 logprob 已经精确饱和成同一个值。
   所以 **HSI 那一行只能按 `G=1` outcome-anchored REINFORCE 引用，不能说成
   Future-HPI 的验证**。完整证据在
   `cua/repos/paper_snapshots/04_tasks/alfworld/v1/04_alfworld_hpi.md` 的 §1。

另外：旧 harness（`tasks/alfworld/v1/` v1）记过 "Base valid_seen = 16.7%"，**那个数已作废**，
与 v2 口径不可比；本任务所有数只能用 v2 口径。

---

## 7. 已完成的基线（不用重跑，直接引用）

| arm | valid_seen | valid_unseen | 合计 |
|---|---:|---:|---:|
| Base Qwen3-1.7B | 6.43% (140) | 7.46% (134) | 6.93% (274) |
| HSI cp25 | 7.86% (140) | 6.72% (134) | 7.30% (274) |
| GRPO cp25 | 5.00% (140) | 8.21% (134) | 6.57% (274) |

三个数全在一个噪声带内，且最优臂跨 split 翻转（seen 上 HSI、unseen 上 GRPO）。

---

## 8. 如果评测中途死掉

判死：`pgrep -f '[s]lime-ascend/train.py'` 找不到进程，且 `$LOG` 里没有两行
`[v2-eval]`，且日志超过 30 分钟没写。

看真正的报错（滤掉上面那三类 SGLang 噪音），然后**原命令重跑即可**——评测是幂等的。
若想接着上一个臂跑，改 `RESUME_FROM` 到下一个 checkpoint 的 run 目录。
