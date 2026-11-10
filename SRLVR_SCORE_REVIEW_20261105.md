# SRLVR "score" 阶段诊断：8B 的难点，以及 4B 为什么 actor 侧快、评测好

> 目的：给方法/科研 review 提供可核对的事实与待决问题。所有数字都来自本机日志，路径见 §7。
> 结论先行：**8B 的 score 慢不是实现退化；4B 的"快"只在 actor 侧，rollout 反而慢 3 倍。**

---

## 0. TL;DR

| 问题 | 结论 |
|---|---|
| 8B score 多慢 | 每步 **319 s**（259–416 s），占 48 min 单步的 ~11% |
| 4B score 多快 | 每步 **174 s**（155–192 s），比 8B 快 **1.8×** |
| 为什么 | 4B 可 TP2 ⇒ **DP4**（每 rank 8 行）；8B 必须 TP4 ⇒ **DP2**（每 rank 16 行）。行数 2× × 每卡 FLOPs 1.28× ≈ 2.56×，实测 1.8×（lm_head 侧 8B 反而更省） |
| 4B 整体更快吗 | **不是**。同一 4B 日志 rollout 4895–5245 s；8B 1566–1983 s（4B 用 TP1 引擎，慢 3×） |
| 4B 评测好吗 | cp24 复评 valid_seen **0.829 / 0.879**、valid_unseen **0.866 / 0.881**（两次 attempt）。但 base 对照口径不同，见 §5 |
| 现在最该动谁 | 不是 score（5.3 min），是 **calibration probes（10.2 min，4B 时代 13–14 min）** 和 rollout（27.6 min） |

---

## 1. "score" 到底是哪一步

代码：`methods/srlvr_hsi.py::score_row`（在 `train_batch` 里、actor 更新之前，对每条 turn row 执行）。

每个 row 做两件事：

1. **两次整序列前向**（`_hidden_states`，`no_grad`，绕过 lm_head，只取 hidden）：
   - student 序列 `tokens`（causal rollout 的原始 token）
   - teacher view 序列 `ids`（在第 t 轮之后插入未来上下文后的序列）
2. 对 mask 内的每个 actor token，按 `position_chunk=64` 分块做**精确 FP32 full-softmax**：
   - 每个 TP rank 只投影自己的词表分片（`vocab/TP`），跨 TP 做 exact max/sum 归约；
   - 同时算 student/teacher 两侧的 log-prob 与 centered features（`mu, f_chosen, e, student_logp`）；
   - 只保留每位置 4 个标量，不落完整 logits。

成本量级：

```
cost ≈ rows × seq_len × layers × (hidden² / TP)        # 两次前向
     + active_tokens × (hidden × vocab / TP)           # lm_head 投影
```

关键点：**teacher view 是不同 token 序列**，student 的 KV / hidden 不能直接复用；所以每次打分天然是"两遍模型"。

---

## 2. 实测（同一套字段，`[srlvr-hsi]` JSON 的 `*_seconds`）

### 8B（2026-11-05，TP4/DP2，2×TP4 引擎，cap 4096）

| rollout | tokens | score | ref | calib | actor |
|---|---:|---:|---:|---:|---:|
| 0 | 666,894 | 266 s | 76 | 672 | 291 |
| 1 | 847,088 | 352 | 95 | 529 | 341 |
| 2 | 630,678 | 300 | 81 | 424 | 299 |
| 3 | 701,255 | 326 | 92 | 703 | 326 |
| 4 | 745,214 | 323 | 86 | 412 | 316 |
| 5 | 799,242 | 402 | 111 | 635 | 394 |
| 6 | 565,095 | 259 | 76 | 562 | 262 |
| 7 | 573,964 | 267 | 72 | 602 | 272 |
| 8 | 623,005 | 310 | 86 | 789 | 307 |
| 9 | 631,111 | 298 | 81 | 657 | 300 |
| 10 | 630,723 | 310 | 84 | 458 | 319 |
| 11 | 849,633 | 416 | 114 | 888 | 401 |
| **均值** | | **319** | **88** | **611** | **319** |

rollout 墙钟 1432–1983 s（均值 1659 s）；单步 ≈ 49.5 min。

### 4B（2026-10-30，TP2/DP4，**TP1 引擎**，cap 4096）

| rollout | tokens | score | ref | calib | actor |
|---|---:|---:|---:|---:|---:|
| 0 | 562,371 | 155 s | 51 | 776 | 168 |
| 1 | 762,292 | 192 | 54 | 869 | 195 |

rollout 墙钟 **4894.8 / 5245.1 s**（均值 ~85 min）。

> 注意：4B 的 calib 比 8B **更慢**（777–869 s vs 412–888 s）。4B 那两次跑的是旧串行 probe；8B 现在走 DP-parallel probe（`mode=parallel`）。所以"4B probe 快"是记忆错位，**4B 快的是 score/ref/actor 这三个 actor 侧阶段**。

---

## 3. 为什么 4B 的 score 快 1.8×

| 因子 | 8B | 4B |
|---|---|---|
| 每卡 hidden-pass FLOPs ∝ hidden²/TP | 4096²/4 = 4.19 M | 2560²/2 = 3.28 M |
| 每 rank 行数 = 32/DP | **16** | **8** |
| lm_head 每 token ∝ vocab/TP | 151936/4 = 38 K | 151936/2 = 76 K |
| 预测倍数（行数 × 每卡 FLOPs） | **2.56×** | 1× |
| 实测 score 倍数 | **1.83×** | 1× |

差的 0.7× 由 lm_head 侧抵消（8B 的词表分片小一半）。结论：**1.8× 完全由并行几何 + 模型宽度解释，没有实现层面的退化**。DP 数由 `TP×PP×DP=8` 和"每卡 ≈2B 参数"共同锁死，8B/32GB 上 DP 只能是 2。

---

## 4. 8B score 的实际难点

1. **两遍整序列前向**：teacher view 与 student 不是同一 token 序列，KV 不能复用。这是方法定义带来的固定成本。
2. **padding 税**：每步 padded tokens 0.82–1.23 M，而真实生成只有 0.57–0.85 M → **多 27–46%**。
3. **逐行串行**：不同 row 长度不同，score 按 row 循环；DP2 下每个 rank 串行处理 16 行，无法用 batch 填满设备。
4. **小 matmul + TP 归约**：`position_chunk=64`，每块都要跨 TP 做 exact max/sum；通信占比随 chunk 变小而上升。
5. **FP32 全精度约束**：方法要求 full-softmax 精确概率（不是近似 top-k），不能换 fp16/近似。
6. **与 rollout 无法重叠**：共卡模式下 8 张卡在 rollout 阶段被 SGLang 占满，score 只能串行等在后面（这也是 async 在 8 卡上无收益的原因之一）。

---

## 5. 4B 的"评测效果好"——证据与警告

**证据**（`_mirror_alfworld-qwen3-4b-srlvr_v2-cp24-*` 的两次独立复评）：

| attempt | valid_seen (n=140) | valid_unseen (n=134) |
|---|---:|---:|
| 20261103_172909 | **0.8786** | **0.8806** |
| 20261103_181605 | 0.8286 | 0.8657 |

**警告 1（口径不可比）**：base 对照 `alfworld_eval_base.log` 是 `turns=20 cap=512 hist=2`，得 seen 0.1786 / unseen 0.0970；上面的 cp24 是 `turns=50 cap=4096 hist=0`。两者差 5×，远超 protocol 伪影，但**绝对值不能直接引用**。要给论文用，需在同一 `turns=50 cap=4096` 下重跑 base。

**警告 2（cp49 无可靠数字）**：cp49 评测日志以 `rc=134`（外部 SIGTERM 指纹）结束，没有 success_rate。**目前只有 cp24 的可信评测数**。

**因果**：4B 的评测好是"方法 + 训练预算"的结果，不是 score 快带来的。4B 的 actor 侧每步 ~500 s，但 rollout ~5000 s，100 步约 6 天；它的优势不是省算力，而是：
- 4B 在该任务下方法方差更小（假设，未验证）；
- 4B 的 100 步预算里能完成的 optimizer 更新次数与 8B 相同（都是 100 次），但每次更新看到的 token 更少 / 策略更稳定（假设，未验证）。

这两条需要科研侧判断，目前没有实验能区分。

---

## 6. 可选降本方案（按已验证程度排序）

| 方案 | 预期收益 | 风险/代价 |
|---|---|---|
| `SRLVR_CALIBRATION: 4 → 2` | **−5 min/step**（calib 611→~300 s） | 改估计量；C/D 方差上升 |
| calibration 每 k 步重算（中间沿用冻结 λ） | −(k−1)/k × 611 s | 校准滞后；需定 k |
| score 只打 U 轨迹 / 抽样 turn | 线性下降 | 改变优势估计的覆盖 |
| 长度分桶 + 多行批量前向 | 10–30% | 显存；padding 无法完全消除 |
| teacher view 前缀 KV 复用 | 理论 ~40–50% 的 score | 需要自定义 attention mask/缓存，工程量最大 |
| 16 卡（TP4×DP4） | score + probe 各 ~2× | 硬件 |
| top-k 近似 softmax | 大幅 | **方法定义不允许**（当前要求 exact FP32） |

---

## 7. 数据来源（可复核）

- 8B：`logs/alfworld-qwen3-8b-srlvr-v2-32g-tp4-colocate-r4096-r100-20261105_114837.log`
  （`[srlvr-hsi]` JSON 的 `score_seconds / reference_seconds / calibration_seconds / actor_seconds`）
- 4B：`logs/alfworld-qwen3-4b-srlvr_v2-batch-s42-r100-n32-20261030_124056.log`
- 4B 评测：`logs/alfworld-qwen3-4b-srlvr_v2/alfworld-qwen3-4b-srlvr_v2-cp24-20261103_172909.log`
  与 `..._181605.log`（`[v2-eval] valid_*`）
- base 4B：`logs/alfworld_eval_base.log`
- probe profile：8B 日志中的 `[srlvr-probe-profile]`（wall/train/copy/moments/chunks）

## 8. 给科研 AI 的问题清单

1. score 是否必须覆盖**每一轮**？能否只对 U 轨迹或最后 k 轮打分，把 C 行的 score 复用上一批？
2. teacher view 在插入点之后的部分才是新增的——**插入点之前的前缀与 student 完全同 token**，能否用这一事实复用 student 的 KV / hidden，只对后缀重算？（需要 exact attention mask，不能破坏 causality）
3. exact FP32 full-softmax 是**方法定义**还是**实现选择**？如果是后者，top-k + 残差界能省多少？
4. 8B 上 calibration 比 score 更贵（611 s vs 319 s）。C/D 估计是否必须每步更新？能否按历史窗口（k 步）摊销？
5. 4B cp24 的 0.88 与 base 0.18（不同口径）之间，有多少来自方法、多少来自 cap/turns 协议？需要在同一协议下补 base。
6. 8B 目前没有可信评测。是否需要先停掉训练、用现成的 cp9 做一次同协议 eval，来确认 8B 是否值得继续跑 80 小时？
