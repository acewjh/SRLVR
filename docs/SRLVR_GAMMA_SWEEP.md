# SRLVR-v2 8B：gamma 消融任务指南（4 个完整 run）

> 给执行人：单台 8×910B 节点，**串行**跑 4 个 γ 臂，每个 100 rollouts。
> **只改 γ**；方法、loss、归一化、并行、数据、评测协议一律与当前 8B SRLVR 配置相同。

## 0. 背景（一句话）

B_t 是"未来 token 居中特征"的几何加权和 `(1-γ)γ^(k-1)`，**不归一化**（保持已发表口径）。
γ 决定回看多远：

| γ | 平均回看 | 95% 质量 |
|---:|---:|---:|
| 0.99 | 99 tokens | 前 298 |
| 0.999 | 999 | 前 2994 |
| 0.9999 | 9999 | 前 29955 |
| 0.99999 | 99999 | 前 299k（上下文内近似均匀） |

注意：**不归一化时 γ 越大，B 的总质量 `1-γ^N` 越小**，所以各臂同时改变了"覆盖"和"幅度"；
这是已发表定义本身的性质，解读结果时要记住（λ 由 C/D 校准，未触界时可部分抵消幅度；
触界时 `λ·B` 会随 B 一起变小）。

## 1. 前置检查

```bash
CUA_ROOT=/workspace/cua
cd "$CUA_ROOT/repos/opsd-slime"
git pull --ff-only            # 需要 >= 382c3b0
conda activate cua-rl
python -c "import torch, torch_npu, transformers; print(torch.__version__, torch_npu.__version__, transformers.__version__)"
# 期望: 2.1.0 / 2.1.0.post17 / 4.51.3
npu-smi info | tail -20            # 8 卡空闲
df -h /workspace | tail -1 # ≥ 700G
free -g | head -2                  # available ≥ 1200G
```

确认基模：`$CUA_ROOT/checkpoints/Qwen3-8B`。

## 2. 运行（一个臂跑完再开下一个）

公共参数（四臂完全一致，只差 `SRLVR_GAMMA` 和 `RUN_NAME`）：

```bash
export ACTOR_GPUS_32G=8 ROLLOUT_GPUS_32G=8 ROLLOUT_GPUS_PER_ENGINE_32G=4 COLOCATE_32G=1
export NUM_ROLLOUT=100 SAVE_INTERVAL=25 MAX_RESPONSE=4096
export SGLANG_ENABLE_NPU_GRAPH=0   # 四臂固定
```

```bash
# 臂 1  γ=0.99
SRLVR_GAMMA=0.99 RUN_NAME=alfworld-qwen3-8b-srlvr-v2-g99-32g-tp4-colocate-r4096-r100-<stamp> \
  bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh

# 臂 2  γ=0.999
SRLVR_GAMMA=0.999 RUN_NAME=alfworld-qwen3-8b-srlvr-v2-g999-32g-tp4-colocate-r4096-r100-<stamp> \
  bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh

# 臂 3  γ=0.9999
SRLVR_GAMMA=0.9999 RUN_NAME=alfworld-qwen3-8b-srlvr-v2-g9999-32g-tp4-colocate-r4096-r100-<stamp> \
  bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh

# 臂 4  γ=0.99999
SRLVR_GAMMA=0.99999 RUN_NAME=alfworld-qwen3-8b-srlvr-v2-g99999-32g-tp4-colocate-r4096-r100-<stamp> \
  bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh
```

`<stamp>` 用启动时间（`$(date +%Y%m%d_%H%M%S)`）。

**预计时长**：实测 31–43 min/step（均值 ~38），**60–70 h/臂**；4 臂 + 评测约 **11–12 天**。

## 3. 启动后 3 分钟自检

```bash
grep -a "srlvr-hsi-config" logs/<run>.log | head -1   # variant=v2 gamma=<本臂值> ... rollouts=100
grep -a "srlvr-hsi-views"  logs/<run>.log | head -1   # bias ≈ -0.001（不是 -21）
grep -a "probe-profile"    logs/<run>.log | head      # copy_s ≈ 40
```

`bias` 若为 −20 量级：输出头选择退化，**立即停止**并报告。

## 4. 里程碑与监控

| 检查点 | 应出现 |
|---|---|
| `runs/<run>/iter_0000024/` | 第 25 步后（约 16 h） |
| `iter_0000049` | 约 32 h |
| `iter_0000074` | 约 48 h |
| `iter_0000099` + `latest_checkpointed_iteration.txt`=99 | 结束（60–70 h） |

```bash
grep -a "\[srlvr-hsi\] {" logs/<run>.log | tail -1   # b / s / λ / C / D / 各阶段耗时
grep -a "hsi-roll" logs/<run>.log | tail -1          # rollouts / tokens / rollout_seconds
```

结果表里必须记录：`b`、`lambda_value`、`C_hat`、`D_hat`（λ 长期贴 `±0.9·min(b,1-b)`
是已知现象，注明即可，不是故障）。

## 5. 断点续训（唯一允许的恢复方式）

```bash
RESUME_FROM="$PWD/runs/<run>" SAVE_DIR="$PWD/runs/<run>" LOG="$PWD/logs/<run>.log" \
SRLVR_GAMMA=<同一个γ> \
ACTOR_GPUS_32G=8 ROLLOUT_GPUS_32G=8 ROLLOUT_GPUS_PER_ENGINE_32G=4 COLOCATE_32G=1 \
NUM_ROLLOUT=100 SAVE_INTERVAL=25 RUN_NAME=<run> \
  bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh
```

硬约束：
- `NUM_ROLLOUT` 必须与首次启动一致（checkpoint 里存了 `total_num_iters`）；
- **`SRLVR_GAMMA` 是 run identity**，续训改值会 `RuntimeError`；
- variant / advantage mode 同理；
- 从 tracker 指向的最新完整 checkpoint 续，不要手改 tracker。

## 6. 磁盘管理（每个臂结束时做）

```bash
# HF 轮转：新 checkpoint 落地后把旧的转 HF 并回收（每个省 ~91G）
setsid nohup env PYTHONPATH="frameworks/Megatron-LM:frameworks/slime-ascend:." \
  /opt/conda/envs/cua-rl/bin/python scripts/hf_ckpt_rotate.py \
  --run-dir "runs/<run>" --origin-hf-dir "$CUA_ROOT/checkpoints/Qwen3-8B" \
  --vocab-size 151936 --watch --interval 600 --delete \
  > "/tmp/hf_rotate_<run>.out" 2>&1 &

du -sh runs/<run>   # 终态 ≈ iter_99(107G) + hf_iter_24/49/74(各15.3G) ≈ 153G
```

4 臂 ≈ 610G。磁盘不足时**先列清单报告**，不要自行删除任何 run / checkpoint。

## 7. 评测（每个臂结束后）

```bash
# cp99
EVAL_LOAD="$PWD/runs/<run>" RUN_NAME=<run>-eval-cp99 \
COLOCATE=1 ACTOR_GPUS=8 ROLLOUT_GPUS=8 \
  bash tasks/alfworld/scripts/run_alfworld_v2_eval_8b.sh

# base（只跑一次，四臂共用）
EVAL_LOAD="$CUA_ROOT/checkpoints/Qwen3-8B" RUN_NAME=alfworld-qwen3-8b-base-eval \
COLOCATE=1 ACTOR_GPUS=8 ROLLOUT_GPUS=8 \
  bash tasks/alfworld/scripts/run_alfworld_v2_eval_8b.sh
```

首次跑 eval 注意是否 OOM；8 卡不足时退回 wrapper 默认 2+6 布局并记录。

## 8. 交付物

维护 `runs/GAMMA_SWEEP_RESULTS.md`：

| run | γ | cp99 valid_seen | cp99 valid_unseen | λ 行为/异常 |
|---|---:|---:|---:|---|
| `...-g99-...` | 0.99 | | | |
| `...-g999-...` | 0.999 | | | |
| `...-g9999-...` | 0.9999 | | | |
| `...-g99999-...` | 0.99999 | | | |
| base | — | | | |

每个 run 的 `FIX_START.txt`、日志、`iter_*`、`hf_iter_*` 都是复现证据，**不删**。

## 9. 禁止事项

- 不要改 `SRLVR_GAMMA` 后复用同一个 run 名；每个臂独立目录/日志；
- 不要并行跑两个臂（只有 8 卡）；
- 不要在 run 进行中修改 `run_alfworld_v2_srlvr_v2_8b_32g.sh`；
- 不要删除任何 `runs/`、`logs/`、`iter_*`、`hf_iter_*`（清理必须列清单并由负责人确认）。
