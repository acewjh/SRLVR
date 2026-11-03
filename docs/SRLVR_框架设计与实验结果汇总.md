# SRLVR：框架设计与实验结果汇总

> 更新：2026-11-03
> 代码基线：`545e7bb`（2026-11-02）+ 8B 适配
> 覆盖规模：Qwen3-1.7B / 4B / 8B，任务为 ALFWorld（多轮具身决策）
> 数据：逐轮指标见 [`data/`](data/)；设计细节见 [`SRLVR.md`](SRLVR.md)、[`SRLVR_V2_V3.md`](SRLVR_V2_V3.md)

---

## 一、方法要解决的问题

多轮 agent 任务里，奖励只在轨迹末尾出现一次。GRPO 的做法是对同一个任务采样 G 条轨迹，用组内均值做基线、组内标准差做尺度。这在数学任务上可行，但在 agent 任务上有两个代价：

1. **采样预算被重复任务吃掉**。固定每步 32 条轨迹的预算下，G=8 意味着每步只见 4 个不同任务。
2. **组内方差会塌缩**。当模型在某任务上已经稳定成功，组内全对 → 标准差为 0 → advantage 恒为 0 → **该优化步没有梯度**。

SRLVR 的目标是**用 G=1 做单轨迹学习**：每步 32 条轨迹就是 32 个不同任务，不需要组内对比。代价是必须另外构造基线和尺度。

---

## 二、框架设计

### 2.1 估计量

优势的基本形式是

```
A_it = (R_i − b − λ·B_it) / s
```

| 符号 | 含义 | 来源 |
|---|---|---|
| `R_i` | 轨迹 i 的终端奖励 | 环境裁决 |
| `b` | 基线 | 历史滑动统计，或当前 batch（见 2.2） |
| `s` | 尺度 | 同上 |
| `B_it` | 后见修正项 | 特权教师视角，见 2.3 |
| `λ` | 修正强度 | 标定探针给出，见 2.4 |

`λ=0` 时退化成最朴素的 `(R_i − b) / s`——即 REINFORCE 加一个滑动基线。这一档不需要探针，是最便宜也最稳的配置。

### 2.2 `b, s` 的两种来源：`SRLVR_ADVANTAGE_MODE`

| 模式 | 含义 |
|---|---|
| `history`（默认） | 跨批次滑动统计，论文里的做法 |
| `batch` | 用当前这批 32 条奖励就地重算 |

`batch` 是 REINFORCE++ 式的替代：G=1 时每个任务位置只有一条轨迹、没有组可归一化，于是退而用**整批**做归一化。实现见 `opsd_core/srlvr_settings.py::batch_baseline_and_scale`，并在 `methods/srlvr.py` 里把 λ 的裁剪和系数符号/边界检查一并切到有效值上。

### 2.3 后见修正项 `B`：特权教师视角

`methods/srlvr_hsi.py` 是 v2/v3 的共享实现，两者只差一个 `f`：

* **v2**：`f = sigmoid(log p − log q)`
* **v3**：`f = 1 − 归一化二元熵`

对每条**已完成**的轨迹，用 `render_hpi_block` / `render_outcome_descriptor` 构造 Full 教师视角——**它包含该轨迹真实的未来和结局**。`mu = Σ_v p(v) f(v)` 是全词表上的学生期望。然后一次反向扫描：

```
S[j] = (1−γ)·e[j] + γ·S[j+1]
B[t] = S 在第 t 轮之后的值        # 严格取后续轮次，最后一轮为 0
```

**边界说明（重要）**：因为教师输入含有该轨迹的未来与结局，这是**以该视角为条件的词表中心化**。它**没有**证明未来修正项零均值、无偏控制变量或鞅性质。C/D 只是经验二阶矩标定，**不主张**方差最优性或置信界。

### 2.4 标定探针与 λ

每轮取前 n 个位置（`SRLVR_CALIBRATION`）作为探针，在**看到奖励之前**计算 C 和 D，其比值用于下一批的 λ，并用下一批冻结的基线做界。探针**不走优化器和调度器**。每条真实轨迹（含探针）都用**从先前批次冻结的 λ**，避免用当批信息自我确认。

### 2.5 工程上的关键约束

| 约束 | 原因 |
|---|---|
| 词表分片归约全部在 `@torch.no_grad()` 下 | v2/v3 的梯度结构上不经过 TP 分片路径，只产出标量系数 |
| `actor_loss` 是纯 PPO token surrogate | 不做分片归约 |
| `micro_batch_size == 1`，拒绝 `use_dynamic_batch_size` | 探针需要对同一轨迹做额外前/反向 |
| 投影每次最多处理 `SRLVR_POSITION_CHUNK`（默认 64）个位置 | 立即把全词表约简成标量，控制显存 |
| 打分器与采样器用同一份权重、正温度、FP32 softmax、模型原生词表 | 非默认惩罚/语法/支持集约束会被显式拒绝而非静默替换 |

### 2.6 消融开关

`opsd_core/srlvr_settings.py` 暴露三个：

* `SRLVR_ACTOR_LAMBDA_ZERO` — actor 强制用 λ=0（探针仍跑，但结果只进日志）
* `SRLVR_TRAJECTORY_UNIFORM_B` — B 在轨迹内取均匀值
* `SRLVR_TEACHER_OUTCOME_ONLY` — 教师视角只给结局，不给完整未来

---

## 三、实验结果

### 3.1 全部 14 个臂

轮数按 events 的 `step` 字段去重后统计。完整数据见 [`data/summary_all_arms.csv`](data/summary_all_arms.csv)。

| 臂 | 轮数 | 起点 | 均值 | 标准差 | **趋势** | 零梯度轮 |
|---|---|---|---|---|---|---|
| 1.7B GiGPO | 25 | 0.000 | 0.247 | 0.174 | **+0.120** | 0% |
| 1.7B GRPO-rpp | 50 | 0.000 | 0.233 | 0.169 | **+0.083** | 0% |
| 1.7B SRLVR v3 | 50 | 0.188 | 0.272 | 0.119 | **+0.061** | 0% |
| 1.7B SRLVR v2 | 50 | 0.156 | 0.257 | 0.118 | **+0.049** | 0% |
| 1.7B SRLVR v3 compact | 50 | 0.250 | 0.258 | 0.107 | **+0.037** | 0% |
| 1.7B SRLVR v2 compact | 50 | 0.188 | 0.249 | 0.103 | **+0.021** | 0% |
| 4B GRPO | 23 | 0.906 | 0.868 | 0.130 | +0.030 | **22%** |
| 4B SRLVR outcome-only | 15 | 0.688 | 0.800 | 0.066 | +0.027 | 0% |
| 4B SRLVR history-uniform-b | 11 | 0.719 | 0.790 | 0.072 | +0.016 | 0% |
| 4B SRLVR v2 history | 57 | 0.781 | 0.784 | 0.074 | −0.012 | 0% |
| 4B SRLVR v2 batch | 50 | 0.750 | 0.784 | 0.083 | −0.044 | 0% |
| 4B GRPO-rpp | 41 | 0.844 | 0.814 | 0.174 | −0.055 | 5% |
| **8B SRLVR v2 (λ=0,batch)** | 21 | 0.844 | 0.838 | 0.052 | **+0.001** | **0%** |
| **8B GRPO** | 20 | 0.969 | 0.914 | 0.110 | **−0.009** | **35%** |

「趋势」= 后半程均值 − 前半程均值。「零梯度轮」= success_rate 恰为 1.0 的轮占比（GRPO 在这些轮组内无方差，advantage 恒为 0）。

### 3.2 结论一：任务在 4B 以上饱和

| 规模 | 臂数 | 趋势范围 | 中位趋势 | 起点范围 |
|---|---|---|---|---|
| **1.7B** | 6 | **+0.021 ~ +0.120** | **+0.055** | 0.000 ~ 0.250 |
| 4B | 6 | −0.055 ~ +0.030 | −0.002 | 0.688 ~ 0.906 |
| 8B | 2 | −0.009 ~ +0.001 | −0.004 | 0.844 ~ 0.969 |

**1.7B 的六个臂趋势全部为正，没有一个例外；4B 和 8B 在零附近正负各半。**

起点越高，训练收益越小，到 4B 已归零。8B 起点 0.844、剩余空间 0.156、单轮标准差 0.052——**ALFWorld 不再是 4B 以上模型的有效评测集**。

### 3.3 结论二：SRLVR 对零梯度步免疫，这是唯一确立的结构性优势

| 臂 | 零梯度轮占比 |
|---|---|
| 8B GRPO | **35%** |
| 4B GRPO | **22%** |
| 4B GRPO-rpp | 5% |
| 1.7B GRPO-rpp | 0% |
| **全部 SRLVR 臂（G=1×32）** | **0%** |

8B GRPO 名义上的 20 个优化步，实际只有 13 步带梯度。**随规模单调恶化**——成功率越高，组内全对越频繁。

SRLVR 的 G=1×32 结构天然不受影响：32 个**不同**任务不会同时全对。这是这组实验里 SRLVR 相对 GRPO 唯一确立的优势，而且它**在 success_rate 曲线上完全看不出来**。

### 3.4 一个不能当作结论的差异

8B 上 GRPO 均值 0.914 vs SRLVR 0.838，高 0.076。**这几乎可以肯定是采样口径差异而非算法优势**：

| | 每轮不同任务数 | 每任务重复 | 每轮轨迹 |
|---|---|---|---|
| SRLVR（G=1） | **32** | 1 | 32 |
| GRPO（bs=4 × G=8） | **4** | 8 | 32 |

两臂预算严格匹配（每步都是 32 条真实轨迹），但 SRLVR 的单轮 success_rate 建立在 32 个任务上，GRPO 只有 4 个。三条证据：

1. GRPO 标准差是 SRLVR 的两倍（0.110 vs 0.052），正是「样本量少 8 倍」的预期
2. 同样的差距在 4B 复现（0.868 vs 0.784，差 0.084），不是 8B 的产物
3. GRPO 频繁打满 1.0（8B 35%、4B 22%），而所有 SRLVR 臂都是 0%

**要比算法优劣，必须在固定 held-out 测试集上离线评测 checkpoint**，绕开训练时的采样差异。

---

## 四、最优配置

### 4.1 推荐配置（8B 全参数，4 训练 + 4 推理，单节点 8×32GB）

```bash
# 方法
ARM=srlvr_v2
SRLVR_ADVANTAGE_MODE=batch        # G=1 无组可归一化，用整批
SRLVR_ACTOR_LAMBDA_ZERO=1         # 最稳的一档，见 4.2
SRLVR_MODE=empirical
SRLVR_CALIBRATION=2               # λ=0 时探针只进日志，2 是代码下限
SRLVR_GAMMA=0.99
SRLVR_PROBE_ACCUM=cpu             # 探针向量放主机内存
SRLVR_COMPACT_FUTURE=0

# 预算（与 GRPO 的 4×8 匹配）
ROLLOUT_BS=32  GBS_OVERRIDE=32    # G=1，batch 就是轨迹数
MAX_RESPONSE=4096  SEQ_LENGTH=32768
MAX_TURNS=50  HISTORY_LENGTH=0  SEED=42

# 并行与显存
ACTOR_GPUS=4  ROLLOUT_GPUS=4      # 非共置：两侧各占 4 卡
TP=4  PP=1
OPSD_LORA=0                       # 全参数
OPTIMIZER_CPU_OFFLOAD=1
SGLANG_MEM_FRACTION_STATIC=0.85

# 吞吐（缺了这两个并发会退化到 1）
ALFWORLD_ENV_ACTORS=1  ALFWORLD_ENV_WORKERS=32
```

### 4.2 为什么推荐 λ=0

`methods/srlvr_hsi.py` 里：

```python
actor_lam = 0. if cfg.actor_lambda_zero else lam
```

λ=0 时标定出的 `lambda_calibrated` 不进 actor，后见修正项完全不参与梯度。在当前数据上，带 λ 的臂没有显示出优于 λ=0 的证据，而标定探针占训练段 16%（实测 938 秒/轮）。**在任务饱和的情况下，先把最简单的一档跑稳更有价值。**

### 4.3 8B 的显存账

```
权重 bf16 16.4 GB + 梯度 bf16 16.4 GB = 32.8 GB 在卡上，按 TP 切分
AdamW fp32 m+v 65.6 GB → CPU offload

TP=1   32.8 GB/卡                      → OOM
TP=2   两块扁平 DDP buffer 各 15.26 GiB  → 30.5 > 29.49，OOM
TP=4   7.63 GiB/块                     → 可行
```

Megatron 的 DDP 会分配 **param buffer 和 grad buffer 两块扁平连续内存**，这是 TP=2 装不下的直接原因——不是激活。

TP=4 的等价性已验证：分片词表打分 vs dense，2/4/8 分片 `dloss=3.73e-08`、hidden 梯度相对误差 ~7e-07。结构上 v2/v3 的梯度不经过分片路径（见 2.5）。

---

## 五、效率

### 5.1 一个优化步到底有多大

| 层级 | 值 |
|---|---|
| 每轮真实轨迹 | 32 |
| `GBS_OVERRIDE` 请求 | 32 |
| **slime 实际 `global_batch_size`** | **728 ~ 916 行（动态）** |
| `micro_batch_size` | 1 |
| 每轮优化步 | 1 |

v2 协议**按 turn 训练**：32 条轨迹展开成约 900 个 turn，每个 turn 一行。**一个优化步吃 91 万个带梯度的 token（完整前向 257 万）**。

> 注意 `actor_tokens` 只统计 loss mask 内的 token，`padded_tokens` 是完整序列长度，两者 2.8 倍的差额主要是 prompt（必须前向），**不是 padding 浪费**。

### 5.2 单轮耗时拆解（8B，实测）

| 段 | 耗时 | 占比 |
|---|---|---|
| **采样** | ~54 min | 54% |
| 参考 logprobs | 220 s | 4% |
| 后见打分（SRLVR 特有） | 832 s | 14% |
| **标定探针（4 个）** | **938 s** | **16%** |
| actor 前向+反向 | 772 s | 13% |

整体：**SRLVR 80 分钟/轮（21 轮 28 小时），GRPO 51 分钟/轮（20 轮 17 小时）**。GRPO 快 36%，因为没有后见打分和标定探针（合计占 SRLVR 训练段 64%）。

### 5.3 采样段被超时轨迹锁死

统计 8B SRLVR 臂全部 416 条 episode：

| | 条数 | turns 中位数 | 生成耗时中位数 |
|---|---|---|---|
| 成功 | 347 (83.4%) | 13 | 279 s |
| 失败 | 69 (16.6%) | **全部恰好 50**（超时） | 1979 s |

**每一条失败都是撞上 `MAX_TURNS=50`，没有一条是中途判负。** 失败轨迹耗时是成功的 7 倍。

> 总生成 77.9 小时中，**37.1 小时（47.7%）花在这 16.6% 的轨迹上。**

成功轨迹的 turns 分位数：`p50=13  p75=23  p90=35  p95=40  p99=47`

| MAX_TURNS | 损失的成功轨迹 | 省下的生成时间 |
|---|---|---|
| 40 | 4.6% | 9.5% |
| **35** | **8.4%** | **14.3%** |
| 30 | 14.7% | 19.1% |

并发侧已无空间：32 条轨迹 / 32 个 env worker / 4 引擎 × 8 并发已完全匹配，并发倍数实测 8.7，KV 池峰值使用率仅 0.33，环境交互只占生成的 1.1%。**瓶颈是单条最慢轨迹，不是吞吐。**

### 5.4 异步 / continuous rollout：在这个任务上不能用

slime 支持 `--partial-rollout` / `--over-sampling-batch-size`，但实测否决：

```
按生成耗时排序：
  最快的 2/3 (n=277)  成功率 = 1.000
  最慢的 1/3 (n=139)  成功率 = 0.504
```

「多生成一些，谁先跑完用谁」在 ALFWorld 上会**系统性地只选中成功轨迹** → 奖励方差塌缩 → advantage 恒为 0 → **训练静默变成 no-op，而 `grad_norm` 看起来完全正常**。

（这与 3.3 的 GRPO 零梯度是同一机制，只是触发方式不同。）

第二个障碍：`--partial-rollout` 回收未完成样本的设计意图是**续写 token**；ALFWorld 的未完成 episode 需要环境状态跨轮存活，当前没有代码处理。

---

## 六、两个会静默失败的坑

这两个都**不报错**，指标看起来健康，但什么也没学到。

### 6.1 `MAX_RESPONSE` 过小

8B 上继承了 1.7B 的 `MAX_RESPONSE=128`，实测：

```
truncated_ratio   = 1.0     每条响应都撞上限
action_valid_rate = 0.0     动作解析不出来
success_rate      = 0.0     reward 恒为 0
```

reward 全 0 → advantage 恒 0 → **没有梯度信号，但 `grad_norm` 正常**。改成 4096 后 `0.0 → 0.844`。

**建议**：首个 rollout 后若 `truncated_ratio > 0.9` 直接失败退出。

### 6.2 并发退化

缺少 `ALFWORLD_ENV_ACTORS=1 ALFWORLD_ENV_WORKERS=32` 时，环境侧一次只喂一条轨迹，SGLang 的并发全程闲置（`#running-req` 在 0/1 之间震荡）。实测 cap-probe：`generation_seconds_sum` 17891 s vs `episode_wall_seconds_max` 2277 s，32 条几乎不重叠。加上之后单轮从 72 分钟降到 ~54 分钟。

**自检**：日志里 `#running-req` 的最大值应 > 1。

---

## 七、统计数据时的一个陷阱

**轮数要按 `step` 字段去重，不能按日志行数。** 同一轮的指标在 `dashboard/events-*.jsonl` 里会重复出现，而**不同 run 的重复倍数不一致**（2 倍、3 倍都有）。按行计数或按固定倍数抽样都会得到错误轮数。

```python
s = {}
for line in open(events_jsonl):
    d = json.loads(line)
    m = d.get("metrics") or {}
    if "rollout/success_rate" in m and d.get("step") is not None:
        s[int(d["step"])] = m["rollout/success_rate"]   # 后写覆盖先写
series = [s[k] for k in sorted(s)]
```

训练日志里同样的问题表现为 `perf N:` 行：GRPO 每轮打印两次，用
`grep -oE "perf [0-9]+:" | sort -u | wc -l`，不要用 `grep -c`。

---

## 八、下一步

### 这组实验回答了什么

* ALFWorld 在 4B 以上已饱和，1.7B 仍有学习信号
* SRLVR 的 G=1×32 结构对零梯度步免疫，GRPO 的该问题随规模单调恶化
* 8B 全参数链路可跑通并稳定（TP=4、4/4 分卡、零 OOM）

### 没回答什么

* **SRLVR 与 GRPO 的算法优劣**。两臂采样口径不同，当前数据不支撑任何结论。

### 要继续，按优先级

1. **固定 held-out 测试集离线评测**——唯一能绕开采样口径差异、消除 ±0.05 单轮噪声的办法。注意 8B SRLVR 臂目前**没有 checkpoint**（`SAVE_INTERVAL=25` 而只跑 21 轮），需补跑。
2. **换更难的任务**。OSWorld 的难度与 4B/8B 更匹配。
3. **把零梯度步占比作为一级指标**跟踪——它比 success_rate 更能区分算法在饱和任务上的表现。
4. 下个 campaign 用 `MAX_TURNS=35`（省 14% 生成时间，损失 8.4% 成功轨迹）；λ=0 臂用 `SRLVR_CALIBRATION=2`（省 8%，不影响梯度）。

---

## 附：原始数据

| 文件 | 内容 |
|---|---|
| [`data/qwen3_8b_srlvr_vs_grpo_rollouts.csv`](data/qwen3_8b_srlvr_vs_grpo_rollouts.csv) | 8B 两臂逐轮 11 项指标 |
| [`data/qwen3_8b_srlvr_estimator.csv`](data/qwen3_8b_srlvr_estimator.csv) | SRLVR 估计量内部状态与分段耗时 |
| [`data/colleague_4b_1p7b_success_rate.csv`](data/colleague_4b_1p7b_success_rate.csv) | 4B / 1.7B 参照组逐轮 success_rate |
| [`data/summary_all_arms.csv`](data/summary_all_arms.csv) | 14 个臂的汇总统计 |

字段说明见 [`data/README.md`](data/README.md)。
