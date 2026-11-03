# SRLVR / GRPO 主线效率修订（2026-10-27）

本次修改源码与启动入口，现有训练未重启。以下代码改进的端到端收益要在下一次启动后实测；79 tok/s 是用户提供的既有 CDF baseline。

## 瓶颈与选择

当前同名 run 的日志含多个 attempt，不能把所有记录混成一次实验。19:47 重启后的 rollout 5：

| 方法 | 真实轨迹 | turn rows | 生成 token | rollout 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| SRLVR v2 | 32 | 1572 | 775688 | 1926.0 s |
| SRLVR v3 | 32 | 1600 | 823025 | 2017.3 s |

证据：`runs/alfworld-qwen3-1p7b-srlvr_v{2,3}-s42-r25-oom-colocate-tp2-admlist-envact-20261027_111859/train.log`，分别 20:26:53、20:27:43 的 `[hsi-roll]`。v2 对应 `common_metrics.jsonl` 还记录成功 1/32、动作有效率 0.6832。大量轨迹接近 50 轮；约 500 token/turn 的实际输出量也是大头。日志尾部只剩一个请求时，单引擎约 23 tok/s，不能与 3–4 并发时的聚合吞吐直接比较。

训练阶段同样需要关注：v2 的较早 attempt 有 `actor_phase_seconds=3072.1`、`rollout_seconds=2720.8`，证明只加速 decode 不足以解决整批训练耗时；该记录不是重启后最新批次的阶段对照。新增 rollout metrics 记录 generation/environment 秒数之和与最长 episode 墙钟；和包含并发重叠，不能当作串行阶段时间。

现有代码已经有：环境进程池、HTTP 连接池、episode 固定引擎映射、radix KV cache；权重更新执行 pause → flush cache → 发布新版本。保留这些机制。新增缓存只复用精确分词结果，收益重点是 GRPO 同题多轨迹的重复 prompt，不能替代 decode 优化。

## 已改主线

| 位置 | 改动 |
| --- | --- |
| SGLang `layers/sampler.py` | 严格路径去掉 softmax 后的再次归一化；普通 sampled-token logprob 复用已 gather 的标量，避免整张 `p.log()`。请求 top/token-id logprobs 时保留原 API；v1 特征仍做全词表统计。CDF 均匀数使用区间中点，避免 hash=0 时抽到前导零质量 token。 |
| `methods/srlvr_hsi.py` | 词表 padding 在投影前裁掉；去掉每块 `bool(valid.all())` 设备同步；TP1 使用原生 log_softmax；每块四次 TP 求和合并为一次；四个特征向量合包传回 CPU，索引只传一次。 |
| `opsd_core/token_cache.py`、ALFWorld prompts/rollout | 精确 rendered-text + tokenizer 身份的 LRU，默认 128 条、262144 token、1048576 字符；仅一个 rollout 调用内存活，结束即清空；返回列表副本。模板/thinking/历史变化体现在文本键中，完全不缓存模型输出或环境状态。`[prompt-cache]` 记录命中、未命中、跳过编码的 token 数。 |
| `opsd_core/srlvr_settings.py`、experiment_metrics | v2/v3 Settings 原样下沉至无 torch/method 依赖的低层模块；`methods.srlvr_hsi.Settings` 保留导出。消除 L0→L1 导入。 |
| Megatron `training/checkpointing.py` | 保存 args 的独立快照，排除 `_srlvr_head`、`_srlvr_full_loss_masks`、`_srlvr_hidden` 三个前向临时字段。HeadRef 携带 TP ProcessGroup，是日志中的不可 pickle 对象来源。保留模型、优化器、scheduler、RNG 和方法历史；不修改运行中的 args。既有 actor_loss 清理引用的修复保留。 |
| `train_scripts/lib/npu_env.sh`、`_common_v2.sh`、kernel shim | `OPSD_QWEN35_GDN=1` 时剔除继承的本仓 shim 路径，不再前置；ALFWorld 入口检查真实 GDN 所需 kernel 的导入路径。stub 自身在该模式直接报明确导入错误。密集 Qwen3 保留现有兼容路径。 |
| `_common_v2.sh`、argv_preflight | `SGLANG_ENABLE_NPU_GRAPH=1` 显式启用 graph，`SGLANG_CUDA_GRAPH_MAX_BS=8` 为初始 capture 上限；默认关闭（**2026-10-30 起改为所有尺寸默认开启**，显式 `SGLANG_ENABLE_NPU_GRAPH=0` 仍可关闭）。preflight 打印最终 graph 参数。 |
| `.gitattributes` | shell / Python / env 文件固定 LF checkout。节点上的既有部署规范化仍需保留；没有批量重写历史文件。 |

取消重复归一化与 CDF 端点处理会产生浮点/RNG 边界差异，续跑不保证逐 token 复现旧实现。完整 softmax 支持与 unsupported 变换拒绝规则保留。

## 全量 RL 与推理 TP

两者现在可以独立配置。之前仅设置 `OPSD_LORA=0` 不够：provider 仍插入 LoRA 并冻结基模，启动器仍只训练 `lora_*`。本次修复：

- `methods/lora_megatron.py` 在关闭 LoRA 时直接返回基模；`memory_provider` 只为 LoRA 添加冻结 embedding 所需的梯度 hook。
- `_common_v2.sh` 仅在 LoRA 模式附加参数白名单；日志同时打印训练 TP、推理 TP、LoRA 开关。
- `run_alfworld_v2_1p7b.sh` 保留全量设置到自动评测，默认 run 名附加 `full`。

LoRA 推理发布采用合并权重，推理引擎执行的是合并后的稠密模型。因此全量 RL 的主要区别在训练更新范围、反向/优化器开销和权重合并成本；推理 TP=2 本身并不要求切成全量 RL。8 张推理卡下 TP1 是 8 个实例，TP2 是 4 个实例；固定 32 条轨迹时初始并发约为每实例 4 与 8，需要比较整批墙钟，而不只看单序列速度。

设备空闲后，使用同一基模、任务预算、响应上限与种子，按下面三组比较。此处仅给出命令，没有启动新训练：

```bash
# v2；v3 替换 ARM；GRPO 替换 ARM=grpo，组大小由现有入口决定。
ARM=srlvr_v2 OPSD_LORA=1 ROLLOUT_GPUS_PER_ENGINE=1 \
  SRLVR_AUTO_BENCHMARK=0 NUM_ROLLOUT=5 SAVE_INTERVAL=5 \
  bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh

ARM=srlvr_v2 OPSD_LORA=1 ROLLOUT_GPUS_PER_ENGINE=2 \
  SRLVR_AUTO_BENCHMARK=0 NUM_ROLLOUT=5 SAVE_INTERVAL=5 \
  bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh

ARM=srlvr_v2 OPSD_LORA=0 ROLLOUT_GPUS_PER_ENGINE=2 \
  SRLVR_AUTO_BENCHMARK=0 NUM_ROLLOUT=5 SAVE_INTERVAL=5 \
  bash tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
```

训练 TP 保持现有 `TP=2`。完整 RL 请从基模新开实验，不能把旧 LoRA checkpoint 当完整模型续训。保持三组 CPU-offload 设置一致并在日志记录；全量显存需求更高，是否 offload 由实际余量决定。比较 `rollout_seconds`、`actor_phase_seconds`、`sync_communication_seconds`、`batch_wall_seconds`、生成 token、峰值显存和实际 trainable 参数。不要同时改变 graph；选定拓扑后再单独追加 `SGLANG_ENABLE_NPU_GRAPH=1 SGLANG_CUDA_GRAPH_MAX_BS=8` 比较共卡 onload/offload 周期。

缓存关闭对照：`ALFWORLD_TOKEN_CACHE_ENTRIES=0`。响应 cap 128/256 可作为后续协议实验；本次维持现有 4096，不把减少生成预算当作同协议提速。

## 源码交付与验证

`frameworks/` 被本仓 `.gitignore` 排除。两处框架源码已修改，同时保留可提交的增量补丁 `compat/patches/training-efficiency-20261027.patch`；它以 §2.2 当前 CDF 源码为前置。在同版本新部署目录使用：

```bash
patch -p1 --dry-run < compat/patches/training-efficiency-20261027.patch
patch -p1 < compat/patches/training-efficiency-20261027.patch
```

已经应用时可用 `patch -p1 --dry-run --reverse < ...` 检查。此节点已应用，不需重复应用，也不自动在启动器里改写共享框架。

已完成的聚焦验证：

- CPU：`tests/test_training_efficiency.py tests/test_srlvr.py tests/test_import_graph.py`，42 passed。使用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ALFWORLD_ENV_ACTOR_PROCESS=1 PYTHONPATH=.:frameworks/slime-ascend python -m pytest -q ...`；env 标志仅避免测试加载 ALFWorld eager runtime。包含双进程 Gloo TP2 与稠密评分对照（部分/完全 padding shard）、采样返回路径/CDF 边界、缓存批次生命周期、GDN stub 拒绝、分层检查。checkpoint 检查执行真实 generate_state_dict 定义与 CPU Adam/scheduler 第 5 步保存和恢复，第 6 步更新一致；模型/优化器分片接口使用 CPU stand-in。
- 本机 cua-rl：`tests/test_hsi_rollout.py tasks/alfworld/tests/test_srlvr_hsi.py tasks/alfworld/tests/test_rollout_efficiency.py`，33 passed。命令环境 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:compat:frameworks/slime-ascend:frameworks/sglang/python:frameworks/Megatron-LM /opt/conda/envs/cua-rl/bin/python -m pytest -q ...`；均使用 CPU 张量。
- full-RL 新开关只做 provider 定向 smoke：替换昂贵的 Megatron 构造器，确认关闭 LoRA 返回所有参数可训练的基模；以及改动 shell 的 `bash -n`。按用户要求停止扩大测试。

真实 NPU 吞吐、colocate+TMS graph 稳定性、真实 GDN kernel 执行、全量 RL 显存/收敛和 HCCL/DCP 完整保存恢复，留给上述主线实验验收；没有以 CPU 回归代替这些结果。
