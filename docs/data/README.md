# 实验结果数据（Qwen3-8B SRLVR vs GRPO，2026-11-01）

原始日志和 checkpoint 按仓库约定不入库（`.gitignore` 含 `runs/`、`*.log`、`reports/`）。
这里只放可直接重新分析的逐轮指标。

## 文件

| 文件 | 内容 |
|---|---|
| `qwen3_8b_srlvr_vs_grpo_rollouts.csv` | 8B 两臂的逐轮全量指标（41 行 = SRLVR 21 + GRPO 20） |
| `qwen3_8b_srlvr_estimator.csv` | 8B SRLVR 臂每轮的估计量内部状态与分段耗时（21 行） |
| `colleague_4b_1p7b_success_rate.csv` | 同事 4B / 1.7B 各臂的逐轮 success_rate（参照组） |
| `summary_all_arms.csv` | 全部臂的汇总统计，文档正文表格由此生成 |

## 字段

**`qwen3_8b_srlvr_vs_grpo_rollouts.csv`**
`arm, rollout, success_rate, action_valid_rate, truncated_ratio, turns,`
`generation_seconds_sum, environment_seconds_sum, episode_wall_seconds_max,`
`mixed_group_rate, tokens_per_gpu_per_sec, grad_norm`

**`qwen3_8b_srlvr_estimator.csv`**
`rollout, success_rate, lambda_value, lambda_calibrated, b_effective, s_effective,`
`advantage_source, batch_fallbacks, probes, actor_tokens, padded_tokens,`
`reference_seconds, score_seconds, calibration_seconds, actor_seconds, C_hat, D_hat`

**`summary_all_arms.csv`**
`arm, n, first, mean, stdev, first_half_mean, second_half_mean, trend, min, max, frac_at_1.0`
`trend` = 后半程均值 − 前半程均值。`frac_at_1.0` = success_rate 恰为 1.0 的轮占比
（GRPO 在这些轮组内无方差，advantage 恒为 0）。

## 一个必须注意的坑

**轮数要按 `step` 字段去重，不能按日志行数。** 同一轮的指标在
`dashboard/events-*.jsonl` 里会重复出现，而**不同 run 的重复倍数不一致**
（2 倍、3 倍都有，取决于 DP 副本数和写入路径）。按行计数或按固定倍数抽样
都会得到错误的轮数——本文档初版就因此把 4B batch 臂记成 71 轮（实为 50）。

正确做法：

```python
s = {}
for line in open(events_jsonl):
    d = json.loads(line)
    m = d.get("metrics") or {}
    if "rollout/success_rate" in m and d.get("step") is not None:
        s[int(d["step"])] = m["rollout/success_rate"]   # 后写覆盖先写
series = [s[k] for k in sorted(s)]
```

同样的问题在训练日志里表现为 `perf N:` 行：GRPO 每轮打印两次，
`grep -c` 会得到 2 倍轮数。用 `grep -oE "perf [0-9]+:" | sort -u | wc -l`。

## 原始产物位置（gy-rl-4）

```
/cache/hp-rlvr/srlvr8b-lambda0-batch-02.final.log   6.2 MB，21 轮
/cache/hp-rlvr/grpo8b-01.final.log                  20 轮
/cache/hp-rlvr/opsd-8b/runs/srlvr8b-lambda0-batch-02/   dashboard events + rollout_*.pt
/cache/hp-rlvr/opsd-8b/runs/grpo8b-01/                  含 iter_0000009 / iter_0000019 checkpoint
```

SRLVR 臂没有 checkpoint（`SAVE_INTERVAL=25` 而只跑了 21 轮）。
