# 交给另一台机器的 agent：把 ALFWorld True-vs-Shuffled HSI 移植到 Qwen3-4B

> 这份文档是**自包含**的任务书。目标机器：单节点 8×NPU（32 GB/卡）。
> 源实现已经在 Qwen3-1.7B 上跑通，绝大多数东西与 4B 相同 —— **不要重新设计方法**，
> 只做移植 + 适配显存。

---

## 0. 一句话任务

把 `tasks/alfworld/`（ALFWorld 多轮 agent RL，GRPO + **True-vs-Matched-Shuffled
future 的 hindsight credit**）移植到 **Qwen3-4B**，先跑 **1-step correctness smoke**，
再跑 **GRPO(G=8)** 与 **HSI(G=1)** 两个 matched arm 到 cp25，然后 eval
`valid_seen` / `valid_unseen`。

---

## 1. 要拷什么

源机器上的这些目录**整份拷过去**（共享盘的话直接可见）：

```
<BASE>/repos/opsd-slime/tasks/alfworld/v1/          # v1：被 v2 import 复用的稳定模块
<BASE>/repos/opsd-slime/tasks/alfworld/       # 主实现
<BASE>/data/alfworld/json_2.1.1/               # 游戏数据（36 MB zip 解压后）
```

`alfworld_hpi_v2` 里的 `scripts/`、`configs/`、`tests/` 都要。
**不要修改 `tasks/alfworld/v1/`（v1）任何文件** —— v2 只 import 它。

模型：`<BASE>/checkpoints/Qwen3-4B`（HF 格式目录）。

---

## 2. 环境隔离（照做，不要装进 base env）

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate cua-rl
python -m venv --system-site-packages <BASE>/envs/alfworld_overlay
OV=<BASE>/envs/alfworld_overlay
$OV/bin/pip install --no-cache-dir fast-downward-textworld
$OV/bin/pip install --no-cache-dir alfworld==0.4.2
# textworld 要特殊处理，见下
```

**三个 setup 坑（都已解决，照抄即可）：**

1. **textworld 的 sdist 构建期 `thirdparty/setup.sh` 会 curl Inform7**（emshort.com，
   通常不可达）。ALFWorld 的 tw-pddl 游戏是**预编译**的，运行时不需要 Inform7。
   → 解包 sdist，把 `setup.py` 里 `check_call(['./setup.sh'], ...)` 那行换成 `print(...)`，
   再 `pip install --no-build-isolation .`。
   （`tasks/alfworld/v1/scripts/setup_alfworld.sh` 里有完整可抄的版本。）

2. **官方 `json_2.1.3_tw-pddl.zip` 没有 `traj_data.json`** ⇒
   `AlfredTWEnv.collect_game_files()` 会**静默返回 0 个游戏**（不报错）。
   → 本项目不用 AlfredTWEnv，直接 `textworld.gym.register_games(game.tw-pddl)`，
   已在 `tasks/alfworld/v1/env_adapter.py` 里做好。

3. **`import alfworld` 会把 `ALFWORLD_DATA` 覆写成 `~/.cache/alfworld`（空目录）**。
   `tasks/alfworld/v1/paths.py` 在 import 期就把数据根**钉住**了；也可以直接用
   `ALFWORLD_HPI_DATA=<path>` 覆盖。

**隔离方式**：overlay 的 site-packages 是 **append** 到 `sys.path`（不是 prepend），
所以只能补缺、不能遮蔽 cua-rl 的包。实测与 cua-rl 的顶层包名交集只有
pip/setuptools。训练进程跑在 cua-rl，ALFWorld 的包从 overlay 来。

验收：

```bash
cd <BASE>/repos/opsd-slime
export PYTHONPATH="$PWD/frameworks/slime-ascend:$PWD"
export ALFWORLD_HPI_DATA=<BASE>/data/alfworld/json_2.1.1
$OV/bin/python -m pytest tasks/alfworld/tests/ -q      # 期望 34 passed
```

（测试用 CPU + fake tokenizer，不需要 NPU。）

---

## 3. 方法（**已冻结，不要改**）

### 3.1 两个 arm，唯一差异是 method

| | worker A | worker B |
|---|---|---|
| method | **HSI** | **GRPO** |
| G（每 task rollout 数） | **1** | **8** |
| 脚本 | `run_alfworld_v2_hpi_future.sh` | `run_alfworld_v2_grpo.sh` |
| 环境变量 | `ALFWORLD_V2_HSI=1`（脚本已设） | — |

两臂**其他一切都相同**：模型、LoRA(r64/α128)、LR(5e-6 constant)、Adam、
clip-grad 0.1、thinking=ON、`history_length=2`、`max_turns=20`、`response_cap=512`、
`FUTURE_GAP=1`、采样(T=0.6/top_p=0.95/top_k=20)、**同一份 frozen schedule**、
`GBS=32`、checkpoint 时机、eval 协议。

### 3.2 HSI 的数学（照抄，不要加 shaping）

对 trajectory `i` 的 turn `t`：

```
e_it = mean_{k ∈ action 内容 token} [ log q_true(a_ik) − log q_shuffle(a_ik) ]     # MEAN 不是 SUM
s_i  = 2·R_i − 1                                                                    # SUCCESS +1 / FAILURE −1
c_it = s_i · e_it
ĉ_it = (c_it − mean_t c_it) / (std_t c_it + 1e-6)        # <2 个有效 turn 或 std 退化 ⇒ ĉ = 0
A_it = s_i + ĉ_it                                        # 送进 clipped PPO/GRPO surrogate
```

- **A 是 advantage，不是 loss 权重。** 绝不要写成 `w_t * L_GRPO` —— 那样在
  group 内奖励全同（GRPO advantage=0）时会把 hindsight 项整体乘零，这正是上一版的概念错误。
- trajectory 内 `mean_t(A_it) ≈ 2R_i − 1`：**verifier 定方向，hindsight 只做轨迹内再分配**。
- **失败 rollout 一样是监督**（`s_i=−1`，把"真实失败未来更支持的动作"压下去）。
  所以**不要**再要求 mixed-reward group，也不要做任何 zero-advantage skip。

### 3.3 两个视图

同一个 scaffold（同 task、同 h_t、同当前观测、同 admissible actions、
**同 terminal outcome R**、**同 student 生成的 target y_t**），只有 privileged body 不同：

```
<privileged_hindsight>
This information is available only during training. ...
Final episode outcome:
SUCCESS | FAILURE
Future interaction trace:
[Future transition i]
Executed action: <action_i>
Resulting environment observation: <o_{i+1}>
...
</privileged_hindsight>
```

- **TRUE** = 该轨迹自己的真实 future，`i >= t + FUTURE_GAP`
- **SHUFFLED** = 另一条轨迹的 future（**matched**：同 task category + 同 terminal
  outcome + 最近剩余 horizon；无匹配则该 turn 不建视图，`ĉ=0`，不要 crash）
- `FUTURE_GAP=1` ⇒ **transition `t`（即 `a_t → o_{t+1}`）必须被排除**。
  注意：future trace 里**只能**有 future action 与 future **resulting** observation，
  **绝不能**有 future `<think>`/reasoning，也**不能**把 `o_t`（action 之前的观测）当结果渲染
  —— 对 `i = t+1` 来说那正是 `o_{t+1}`，就是泄漏。
- shuffled 的候选来自**同一个 rollout batch 的其他轨迹**（teacher replay control），
  **不是**额外的环境 rollout ⇒ 环境采样仍是 **1 rollout / task**。

---

## 4. 工程结构（已实现，照用）

```
--rollout-function-path                      alfworld_hpi_v2.rollout.generate_rollout
--custom-convert-samples-to-train-data-path  alfworld_hpi_v2.train_data.convert_samples_to_train_data
--custom-loss-function-path                  alfworld_hpi_v2.loss.alfworld_v2_loss
--rollout-data-postprocess-path              alfworld_hpi_v2.scorer.postprocess   # 仅 HSI 臂
--use-opsd                                   # 让 slime 原样透传 opsd_advantages
```

- **一 episode 一行 → 一轮一行**（per-turn）。这是解决 lm_head 显存的关键：
  行长不再随轮数增长，`max_turns=20` 才可能。**不要改回整局一行。**
- **轨迹级 advantage** 通过 `Sample.opsd_advantages` + `--use-opsd` 注入（slime 会
  绕过它自己的 group 归一化）。**不要**依赖 slime 的 `--advantage-estimator grpo`
  做分组 —— 它是按位置切组的，per-turn 行会被按"轮"而非"episode"归一化。
- **轨迹身份**：`turn_ids = episode_id * 10000 + turn_idx`（唯一可用的 per-token 整数白名单通道）。
- **teacher 打分在 loss 之外**：`--rollout-data-postprocess-path` 在 convert 之后、
  训练步之前、训练进程内运行；teacher 常驻、batched。
  **`loss.py` 不 import teacher**，`teacher_forward_calls` 恒 0。
- **action span** 由 target token ids decode + offset 映射恢复，
  **必须校验重编码 ids 与原 ids 完全一致**才采用，否则该 turn 记为 unscored。
- **HPI 权重不再走 `fit_turn_weights`**（那是上一版），现在直接改 `opsd_advantages`。

---

## 5. 已经踩过的坑（**逐条避免**）

| # | 症状 | 根因 / 对策 |
|---|---|---|
| 1 | `assert self.lr_decay_steps > 0`，actor init 就死 | slime 用 `train_iters = num_rollout*rollout_bs*G//GBS` 回落 `lr_decay_iters`；**HSI 的 G=1 会让它变 0**。对策：`TRAIN_ITERS` 与 `GLOB_BS` **必须在 ARM switch 之后算**（switch 才把 G 设成 1），并在 `<=0` 时传 `--lr-decay-iters 1`。`_common_v2.sh` 已修。 |
| 2 | `tatsu.FailedToken` 环境构造崩，报错位置**随机漂移** | textworld 的 `_PARSER` 是 **module 级 tatsu 单例、非线程安全**；rollout 用 `asyncio.to_thread` 并发建 env 会互相踩。`env_adapter.py` 已加 module 级 RLock 串行化 construct/reset/step/close。**单线程 CPU probe 永远测不出这个。** |
| 3 | 引擎 400 | 这个 sglang 版本字段是 `sampling_seed`，不是 `seed`。 |
| 4 | 动作被拒后成功率恒 0 | 拒绝后必须把原因**回灌进下一轮 observation**（保留原观测），否则模型看到逐字节相同的输入、无从自查。 |
| 5 | 44% 的 turn 卡在 `<think>` 里 | 加"简明思考"指令 + `max_new_tokens` 要够（`response_cap` 太小会大量截断，实测 cap=384 时 38% 截断）。 |
| 6 | 注释插进 `\` 续行的参数列表 → `rc=127` | bash 续行里**不能放注释行**。改完 launcher 务必 `bash -n`。 |
| 7 | Ray GCS 莫名消失、`Failed to connect to GCS` | **同一个节点同时跑了两个 launcher**，互相 `ray stop`。**一次一个 launcher**，重启前先 `pkill` + `ray stop --force` 并确认干净。 |
| 8 | 信息隔离断言误报 | 用**子串**判断"privilege 里是否含被扣住的内容"在 ALFWorld 上不可用 —— 环境会重复文本（`"Nothing happens."` 每次失败都出现）。必须**结构化解析**（按 `[Future transition i]` 标签逐条核对）。 |
| 9 | 又一种误报：多行观测被截断 | 拒绝反馈消息内含换行，按行解析会只取第一行。解析时要把**续行折回**，并在 `</privileged_hindsight>` 处终止。 |
| 10 | `turn_ids` 传进来是 torch tensor | 迭代 tensor 得到 0-dim **tensor 标量**，拿它当 dict key **不按值去重** —— 5 轮的 episode 会被读成 "1613 turns"。用之前先 `int()`。 |
| 11 | slime 的 metrics dict 数值不可信 | slime 会把返回 metrics 的**每一项除以 batch token 总数**（`n_response_tokens` 会变成 1.0）。要读绝对量就自己 `logger.info`。 |
| 12 | `hf_checkpoint` 的 LoRA 坑 | 训练臂是 LoRA，**评自己的 checkpoint 时必须走** `--custom-model-provider-path math_opsd.lora_megatron.custom_model_provider`，否则会得到一个与 Base 无法区分的假结果。评纯 HF dir（如 Base）则不需要。 |

---

## 6. 4B 的额外注意（**与 1.7B 最大的不同**）

1. **显存更紧。** 1.7B 用的是 `actor=4 / rollout=4` 的 8 卡切法、`micro-batch-size 1`、
   `turns=20 / cap=512`。4B 参数量 ~2.4×，**同样的行长会更吃紧**。
   先跑 smoke，若 OOM 按此顺序降：`max_turns` → `response_cap` → 增 `--recompute` 力度。
   **不要**先加 LoRA 权重以外的参数、也不要先动 TP（见下）。
2. **TP 暂时不要动。** `math_opsd.lora_megatron.custom_model_provider` **不是 TP-aware**：
   `--tensor-model-parallel-size 2` 会直接 `4096 and 2048 cannot broadcast`
   （hidden 维被切但 provider 仍按原值算）。单节点内要保持 TP=1。
3. **激活重算**已默认开（`--recompute-granularity full --recompute-method block
   --recompute-num-layers 1`），4B 上更要保留。
4. **不要让 4B 的 thinking 失控**：`response_cap` 是每轮上限，4B 可能想得更久；
   若截断率明显上升，先调 `max_turns`/prompt 措辞，而不是无限加 cap（cap 直接决定显存）。

---

## 7. 执行顺序

```bash
BASE=<...>; PROJ=$BASE/repos/opsd-slime
cd $PROJ
export PYTHONPATH="$PROJ/frameworks/slime-ascend:$PROJ"
export ALFWORLD_HPI_DATA=$BASE/data/alfworld/json_2.1.1

# 0) CPU 契约测试（不需要 NPU）
<overlay>/bin/python -m pytest tasks/alfworld/tests/ -q          # 期望 34 passed

# 1) 生成 frozen schedule（两臂必须用同一份）
<overlay>/bin/python -c "
from alfworld_hpi_v2.schedule import build_schedule
build_schedule('tasks/alfworld/configs/train_schedule_seed42.jsonl',
               index_path='tasks/alfworld/v1/configs/game_index.json',
               num_steps=40, tasks_per_step=4, seed=42)"

# 2) 1-step smoke（HSI 臂；用能命中 mixed/success 的 task 更易看清）
SCHEDULE=$PWD/tasks/alfworld/configs/diag_schedule.jsonl \
NUM_ROLLOUT=1 ROLLOUT_BS=4 SAVE_INTERVAL=1000 MAX_TURNS=20 MAX_RESPONSE=512 \
RUN_NAME=alfworld_hsi_smoke SAVE_DIR=$PWD/runs/alfworld_hsi_smoke LOG=$PWD/logs/alfworld_hsi_smoke.log \
setsid nohup bash tasks/alfworld/scripts/run_alfworld_v2_hpi_future.sh > /tmp/hsi_smoke.log 2>&1 < /dev/null &
```

**smoke PASS 判据（必须全满足）：**

- `rc=0`，无 OOM、无 NaN/Inf
- `[v2-hpi] scored=... skipped=...` 出现且 `scored > 0`
  （**失败轨迹也必须被 score** —— 这正是本方法的意义）
- `[v2-loss]` 里 `pg_loss_mean` 非零，且**存在非零 grad**
- log 内 `teacher_forward_calls = 0`（自定义 loss 里绝不能有 teacher 前向）

smoke PASS 后：

```bash
# worker A: HSI（G=1）
NUM_ROLLOUT=25 ... bash tasks/alfworld/scripts/run_alfworld_v2_hpi_future.sh
# worker B: GRPO（G=8）
NUM_ROLLOUT=25 ... bash tasks/alfworld/scripts/run_alfworld_v2_grpo.sh
```

两臂用**同一份 schedule**、同时启动。cp25 后立即 eval：

```bash
bash tasks/alfworld/scripts/run_alfworld_base_eval_1p7b.sh   # 改成 4B 的 load path
python tasks/alfworld/report_eval.py --dump 'runs/<...>/*.pt'
```

---

## 8. 不要做的事

不要：改方法（G、gap、think、prompt、alpha/lambda、任何 shaping、任何 sweep）、
把 advantage 改回 loss 权重、恢复 mixed-group gating、动 TP、
评 LoRA checkpoint 时忘了 provider、同时跑两个 launcher、
删除任何 `runs/` 产物（要删先列清单等确认）、开 wandb。

**只做移植 + 显存适配**；方法本身已经冻结并且有 34 项 CPU 契约测试守着。

---

## 9. 参考数字（1.7B 上实测，供对照）

- per-turn 布局让 `max_turns=20` 可行：1 个 rollout = 4 task × G × ~16 turns
- 1.7B HSI 的 scorer 单步：`scored=44 skipped=84 teacher_views=126`（DP=4 每 rank）
- 1.7B `e_t`（旧 empty-control 版）≈ 0，换 True-vs-Shuffled 后 `r_t ∈ [0.71, 1.004]`、
  `w ∈ [0.79, 1.11]`、`pg_loss_mean = -1.62` —— 说明换对照后信号明显变强
- 1.7B 单 rollout wall ≈ 22–26 min（4 task × 8 episode × 20 turn）
