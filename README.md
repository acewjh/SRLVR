# SRLVR

**S**ingle-**R**ollout **L**earning from **V**erifiable **R**ewards — 面向多轮 agent 任务的单轨迹策略优化。

在 ALFWorld（多轮具身决策）上，与 GRPO / GSPO / GiGPO 在**匹配的采样预算**下对比，覆盖 Qwen3-1.7B / 4B / 8B 三个规模。

---

## 动机

多轮 agent 任务的奖励只在轨迹末尾出现一次。GRPO 的做法是对同一任务采样 G 条轨迹，用组内均值做基线、组内标准差做尺度。这在数学任务上可行，但在 agent 任务上有两个代价：

1. **采样预算被重复任务吃掉**。固定每步 32 条轨迹时，G=8 意味着每步只见 4 个不同任务。
2. **组内方差会塌缩**。模型在某任务上稳定成功后，组内全对 → 标准差为 0 → advantage 恒为 0 → **该优化步没有梯度**。实测这一比例随模型规模单调恶化：

   | | 零梯度步占比 |
   |---|---|
   | GRPO @ 8B | **35%** |
   | GRPO @ 4B | **22%** |
   | GRPO @ 1.7B | 0% |
   | **SRLVR（G=1×32，全部规模）** | **0%** |

SRLVR 用 **G=1** 做单轨迹学习：每步 32 条轨迹就是 32 个**不同**任务，不需要组内对比，因此天然不受零梯度步影响。代价是必须另外构造基线与尺度。

## 方法

优势的基本形式：

```
A_it = (R_i − b − λ·B_it) / s
```

* `b, s` — 基线与尺度。`SRLVR_ADVANTAGE_MODE=history` 用跨批次滑动统计；`batch` 用当前这批 32 条奖励就地重算（G=1 时没有组可归一化，退而用整批）。
* `B_it` — 后见修正项。对已完成的轨迹构造**包含其真实未来与结局**的特权教师视角，取全词表上的学生期望 `mu = Σ_v p(v) f(v)`，再反向扫描 `S[j] = (1−γ)e[j] + γS[j+1]`，`B[t]` 严格取第 t 轮之后的值。
* `λ` — 修正强度，由标定探针在**看到奖励之前**算出 C/D，用于**下一批**；每条真实轨迹都用从先前批次冻结的 λ。
* `λ=0` 时退化为 `(R_i − b)/s`，不需要探针，是最便宜也最稳的一档。

变体只差一个 `f`：**v2** 用 `sigmoid(log p − log q)`，**v3** 用 `1 − 归一化二元熵`。

> **边界声明**：教师输入含该轨迹的未来与结局，所以这是*以该视角为条件的*词表中心化。本实现**不主张**未来修正项零均值、无偏控制变量或鞅性质；C/D 只是经验二阶矩标定，**不主张**方差最优性或置信界。

详见 [`docs/SRLVR_框架设计与实验结果汇总.md`](docs/SRLVR_框架设计与实验结果汇总.md)。

## 结果

14 个臂的完整数据在 [`docs/data/`](docs/data/)。两条主要结论：

**1. ALFWorld 在 4B 以上已饱和。** 趋势 = 后半程均值 − 前半程均值：

| 规模 | 臂数 | 趋势范围 | 中位 | 起点范围 |
|---|---|---|---|---|
| **1.7B** | 6 | **+0.021 ~ +0.120** | **+0.055** | 0.000 ~ 0.250 |
| 4B | 6 | −0.055 ~ +0.030 | −0.002 | 0.688 ~ 0.906 |
| 8B | 2 | −0.009 ~ +0.001 | −0.004 | 0.844 ~ 0.969 |

1.7B 的六个臂趋势全部为正，无一例外；4B 和 8B 在零附近。起点越高，训练收益越小。

**2. SRLVR 对零梯度步免疫**（见上表），这是当前唯一确立的结构性优势，且**在 success_rate 曲线上完全看不出来**。

> **一个不能当作结论的差异**：8B 上 GRPO 均值 0.914 vs SRLVR 0.838。这几乎可以肯定是采样口径差异——SRLVR 每轮统计 32 个不同任务，GRPO 只有 4 个。GRPO 标准差是 SRLVR 的两倍（0.110 vs 0.052），同样差距在 4B 复现。**要比算法优劣，必须在固定 held-out 测试集上离线评测 checkpoint。**

## 快速开始

```bash
# 1.7B，三种算法各跑一个最小回合
ARM=grpo     NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=gspo     NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
ARM=srlvr_v2 NUM_ROLLOUT=1 ROLLOUT_BS=8 MAX_TURNS=2 bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
```

8B 全参数（4 训练 + 4 推理，单节点 8×32GB）的推荐配置见汇总文档第四节。要点：

```bash
ARM=srlvr_v2 SRLVR_ADVANTAGE_MODE=batch SRLVR_ACTOR_LAMBDA_ZERO=1
ROLLOUT_BS=32 GBS_OVERRIDE=32 MAX_RESPONSE=4096
ACTOR_GPUS=4 ROLLOUT_GPUS=4 TP=4 OPSD_LORA=0 OPTIMIZER_CPU_OFFLOAD=1
ALFWORLD_ENV_ACTORS=1 ALFWORLD_ENV_WORKERS=32   # 缺这两个并发会退化到 1
```

## 两个会静默失败的坑

都**不报错**，`grad_norm` 看起来正常，但什么也没学到：

1. **`MAX_RESPONSE` 过小** → `truncated_ratio=1.0` → 动作解析不出来 → reward 恒 0 → advantage 恒 0。8B 上继承 1.7B 的 128 会直接导致这个；改成 4096 后 success_rate 从 0.0 变 0.844。**自检**：首轮后 `truncated_ratio` 应 < 0.9。
2. **并发退化** → 缺 `ALFWORLD_ENV_ACTORS` / `ALFWORLD_ENV_WORKERS` 时环境一次只喂一条轨迹，推理引擎全程闲置。**自检**：日志里 `#running-req` 最大值应 > 1。

同类机制也否决了异步 rollout：ALFWorld 上先跑完的轨迹几乎全是成功的（最快 2/3 成功率 1.000，最慢 1/3 仅 0.504），"谁先跑完用谁"会让奖励方差塌缩成 no-op。

## 目录

| 路径 | 内容 |
|---|---|
| `methods/` | 算法实现。`srlvr_hsi.py` 是 v2/v3 共享估计量，`srlvr.py` 是训练循环与 DP 梯度重建 |
| `opsd_core/` | 任务/方法注册表、标定设置、打分、遥测。不依赖 `methods/` 与 `tasks/`（分层由 `tests/test_opsd_core.py` 的 AST 检查强制） |
| `opsd_dashboard/` | 本地训练看板，读 `<run>/dashboard/events-*.jsonl`，无需 W&B |
| `tasks/alfworld/` | ALFWorld 任务、rollout、动作解析、启动脚本 |
| `docs/` | 设计文档与实验结果；`docs/data/` 是逐轮原始指标 |
| `tests/` | 单元测试，含分层检查与 DP 梯度聚合测试 |

补充材料：

| 文档 | 内容 |
|---|---|
| [`SRLVR_SCORE_REVIEW_20261105.md`](SRLVR_SCORE_REVIEW_20261105.md) | score 阶段耗时归因：8B 慢 1.8× 完全由并行几何与模型宽度解释，非实现退化 |
| [`SRLVR_V2_8B_PERF_DIAG_20261106.md`](SRLVR_V2_8B_PERF_DIAG_20261106.md) | 8B 单 run 的逐段性能诊断材料 |
| [`docs/SRLVR_GAMMA_SWEEP.md`](docs/SRLVR_GAMMA_SWEEP.md) | γ 消融指南。注意 B 不归一化，γ 同时改变回看覆盖与幅度 |

## 依赖

训练栈为 slime（Ray）+ Megatron-LM + SGLang，在 Ascend NPU 上另需 MindSpeed。这些**不包含在本仓库**，环境搭建见 [`docs/ENV_SETUP.md`](docs/ENV_SETUP.md)。

## 说明

* 本仓库是内部研究代码的导出，**不含**第三方 vendored 组件、预编译二进制与评测数据集；路径默认值已改为通用形式，按实际环境用 `CUA_BASE` / `CUA_PROJ` 覆盖。
* 实验在 Ascend 910B4（32GB）单节点 8 卡上完成。
