# SRLVR 方法修订与训推实验配置

本文记录可用于论文方法与实验部分的算法定义、训推协议和诊断结果。SRLVR-v2/v3 的长度归约以 2026-10-29 修订（§2.4）为准；历史诊断单独标明对应设置。

**ALFWorld 统一默认（所有方法）**：全量 RL（`OPSD_LORA=0`）＋推理 TP1
（`ROLLOUT_GPUS_PER_ENGINE=1`）＋NPU graph（`SGLANG_ENABLE_NPU_GRAPH=1`）。
默认不设置 thinking token 预算，不添加 24／512-token 限制提示；响应总上限仍由
`MAX_RESPONSE` 控制（1.7B 入口默认 4096）。显式环境变量可覆盖默认；以下试跑中的
LoRA／thinking512 是历史对照配置。

## 1. 当前方法与优化目标

SRLVR 保留每个任务采样位置只生成一条真实环境轨迹、turn-level 优势系数和严格后续轨迹构造的未来校正。当前使用 **token-level PPO clipped surrogate**，替代早期 turn-mean ratio 乘 logprob 的形式；比率通过自身对当前 logprob 求导，不再额外乘一次 logprob。

每批真实轨迹数为 `N=32`，校准集合 `C={0,...,3}`，其余 28 个位置为 `U`。采样有放回，因此 32 个位置不保证对应 32 个不同任务 ID。对轨迹 i、turn t、actor token j：

```text
A_it = (R_i - b - 1{i in U} * lambda * B_it) / s
lambda = clip(C_hat / D_hat, -rho * min(b, 1-b), rho * min(b, 1-b))
r_itj = exp(logp_new_itj - stopgrad(logp_rollout_itj))
L_i = sum_t,j mask_itj  # 仅该轨迹的有效 actor response tokens

L_PPO = -(1/N) * sum_i (1/L_i) * sum_t,j mask_itj *
        min(r_itj * stopgrad(A_it),
            clip(r_itj, 0.8, 1.2) * stopgrad(A_it))

d_itj = stopgrad(logp_initial_itj) - logp_new_itj
k_itj = clamp(exp(d_itj) - 1 - d_itj, -10, 10)
L_KL  = (1/N) * sum_i (1/L_i) * sum_t,j mask_itj * k_itj
L_actor = L_PPO + 0.001 * L_KL

g_it = grad_theta[sum_j mask_itj * logp_itj]
S_i = sum_t g_it / L_i
Z_i = (R_i - b) * S_i / s
Y_i = sum_t B_it * g_it / (s * L_i)
C_hat = mean_{i in C} dot(Z_i, Y_i)
D_hat = mean_{i in C} squared_norm(Y_i)

W_i = squared_norm(S_i)
b_next = clip(sum_history(R_i * W_i) / sum_history(W_i), eps_b0, 1-eps_b0)
s_next = sqrt(delta_s^2 + mean_history[(R_i - b_at_collection_i)^2])
```

`D_hat=0` 时 `lambda=0`。`C_hat/D_hat` 是独立校准样本上的 plug-in 估计，默认模式为 `empirical`，不等同于 certified population bound。C 中轨迹参与不带未来校正的 baseline actor 更新；未来校正只用于 U。

`B_it` 使用 turn t 之后所有 actor token 的中心化采样特征构造，末轮 `B=0`，不包含当前 turn 的 token。采样特征由实际 FP32 full-softmax 采样分布计算，不另采环境轨迹、不传输完整词表，也不通过 replay forward 重算该特征。

`b,s` 仅依赖先前批次的校准历史，在本批唯一一次 actor 更新后推进。`b` 使用历史**逐轨迹 token 平均 score 梯度**的平方范数 W 加权奖励均值；W 总和为零时保留原 b。`s` 使用各样本采集时冻结的 b 计算历史奖励残差二阶矩，并加尺度正则；其更新规则不变。默认参数：

| 参数 | 值 | 含义 |
|---|---:|---|
| 校准轨迹数 | 4 | C 覆盖四族各 1 条；U 为其余 28 个位置 |
| `rho` | 0.9 | 限制未来校正系数，保持奖励系数符号 |
| 特征尺度 `c` | 10 | clipped log-probability 特征尺度 |
| 历史窗口 | 128 | 历史校准轨迹数 |
| `eps_b0` | 0.01 | baseline 裁剪边界 |
| `delta_s` | 0.1 | reward scale 正则 |
| 初始 `b,s` | 0.5，1.0 | 无历史时的初值 |

C/D 校准使用逐轨迹 token 平均的 score-function 梯度，不使用 PPO ratio 或初始模型 KL；probe backward 不执行 optimizer/scheduler step。KL 进入最终 actor 更新，并使用与 PPO 相同的 L_i 分母。参考策略固定为初始基模；加载参考模型时清零 LoRA B 增量，防止续训 adapter 污染参考目标。

SRLVR-v2/v3 与当前 GRPO 均采用轨迹内 token mean、轨迹间 mean 的长度归约；各自 KL 随 actor objective 使用同一归约。L_i 不包含历史输入、环境文本、零 mask 或 padding，不用 turn 长度或整批平均长度替代。旧 token-sum 口径是历史实现，不能与本修订当作同一优化目标；legacy `methods/srlvr.py` 的独立旧方法入口不在本次 v2/v3 修改范围内。相同轨迹预算仍不代表相同 token 数或计算成本。

## 2. 模型、硬件与训练预算

| 项目 | GRPO | SRLVR |
|---|---|---|
| 初始模型 | Qwen3-1.7B，从基模初始化 | 相同 |
| 更新范围 | 全量 RL（默认 `OPSD_LORA=0`） | 相同 |
| 可选 LoRA | 显式 `OPSD_LORA=1`；1.7B 入口 r=256、alpha=256、dropout=0 | 相同 |
| 每批任务采样位置 | 4 | 32 |
| 每任务位置真实轨迹数 | 8 | 1 |
| 每批真实环境轨迹数 | 32 | 32 |
| 每批 optimizer 更新数 | 1 | 1；校准 backward 不计为更新 |
| 默认训练批数 | 50 | 50 |
| 总真实轨迹预算 | 1600 | 1600，其中 200 条兼作校准样本 |
| optimizer 更新预算 | 50 | 50 |
| 保存间隔 | 每 25 批保存完整训练状态 | 相同 |
| 随机种子 | 42 | 42 |

两方法各使用一个独立节点，每节点 8 张 Ascend 910B1；当前 1.7B v2 入口默认训练与推理共卡，训练 TP=2、actor DP=4；所有 ALFWorld 方法的 SGLang 每个推理实例默认 1 卡（TP>1 已放开，采样器只告警，见 §2.1）。训练栈为 Slime-Ascend、Megatron-LM、MindSpeed 和 SGLang-Ascend。训练和推理均为 BF16，prefix cache 默认开启（跨 turn 复用历史 prefill），发布新策略权重时清理推理缓存。

### 2.1 当前默认显存与训推共卡配置

公共入口 `tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh` 默认设置
`ACTOR_GPUS=8`、`ROLLOUT_GPUS=8`、`TP=2`、`COLOCATE=1`；所有方法默认
`ROLLOUT_GPUS_PER_ENGINE=1`（显式设为 4 等即可用 TP>1 引擎，见下）。`--colocate` 使训练 actor
与 SGLang 复用同一组 8 张卡，框架在训练/推理阶段自动切换 offload。训练命令启用：

```text
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1
```

优化器状态 CPU offload（`--optimizer-cpu-offload --optimizer-offload-fraction 1
--use-precision-aware-optimizer --overlap-cpu-optimizer-d2h-h2d`）由
`OPTIMIZER_CPU_OFFLOAD` 控制，SRLVR 1.7B 臂默认关闭：本栈上该组合是静默 no-op，
见下。

`--sequence-parallel` 是显式开关（`OOM_SEQUENCE_PARALLEL=1`）且对 LoRA 臂强制关闭：
列并行 base 层在 SP 下 all-gather 序列分片，而 LoRA delta 按本地分片计算，残差相加
报 `The size of tensor a (S) must match the size of tensor b (S/TP)`。

CPU-offload Adam 需关闭 `fused`：Ascend 的 torch 没有 `aten::_fused_adamw_` 的 CPU
kernel，Megatron 默认 `fused=True` 会在首个 `optimizer.step()` 报
`Could not run 'aten::_fused_adamw_' ... 'CPU' backend`；已在 Megatron
`optimizer/__init__.py` 改为 `fused=False`。

**冻结 embedding + reentrant 全量重计算会静默断开 LoRA 梯度**（2026-10-27
实测）：`train/grad_norm` 恒 0，且 step 1 的 `srlvr_reference_kl_sum` 仍恰好为 0，
说明权重从未更新，但 loss 非零。reentrant checkpoint 只有在前序输入
requires_grad 时才回传参数梯度，冻结的 token embedding 会让所有被 checkpoint 的
层断开。修法：`MODEL_PROVIDER_PATH=tasks.alfworld.memory_provider.custom_model_provider`
（给 embedding 输出加 `requires_grad` hook）；同一套 loss/重计算/CPU offload 配置下
用它 `grad_norm≈1700`，不用则恒 0。CPU offload 本身可用（fastcal-colocate 开启后
仍正常），1.7B 臂是否开启按显存预算决定；验收以 step 1 的
`reference_kl_sum > 0` 为准。

SGLang 使用 `mem_fraction_static=0.42`、`max_total_tokens=65536`、
`max_running_requests=8`，开启 radix cache（跨 turn 复用历史 prefill）与 NPU graph（capture batch≤8）。

**KV 预留池的两个旋钮必须一起调（2026-10-28 实测修订）**：池大小由
`capacity = min(profiled_tokens, max_total_tokens)` 决定
（`model_runner_kv_cache_mixin.py::_resolve_token_capacity`），因此
`mem_fraction_static` 只决定 profile 出来的上限，真正封顶的常常是
`max_total_tokens`。该池是从引擎静态显存里切出、**每次权重同步后要重新映射**的块；
全量 RL 的训练器常驻时，`update_weights` 之后每卡只剩 10.2–12 GB，池一旦接近这个数就会
映射失败（`torch_memory_saver.cpp: aclrtMallocPhysical failed`）。

单卡实测（Qwen3-1.7B，910B）：

| `mem_fraction_static` | `max_total_tokens` | 实际池 | KV(K+V) | 分配后可用 |
|---:|---:|---:|---:|---:|
| 0.5（旧默认） | 98304（旧默认） | 98304 | 10.52 GB | 15.09 GB |
| 0.5 | 不设 | 103552 | 11.08 GB | 14.53 GB |
| 0.42 | 不设 | 81792 | 8.76 GB | 16.85 GB |
| **0.42（新默认）** | **65536（新默认）** | **65536** | **7.02 GB** | **18.59 GB** |

旧组合（0.5 + 98304）的 10.52 GB 池落在 10.2–12 GB 的噪声带里，安全边际不足 1 GB，
两条 SRLVR 臂都因此中招（v3 在第一次权重同步、v2 在第二次）。新组合留出约 3.2 GB 余量。
`max_total_tokens` 仍是 KV **预留池**、不是上下文预算：`SEQ_LENGTH` 保持 32768、
响应上限保持 4096，且实测池利用率峰值 0.31、`queue-req` 全程为 0。
注意仅调 `mem_fraction_static` 到 0.42 并不够（池只降到 8.76 GB），必须同时把
`max_total_tokens` 降到 65536。

实测关闭 radix cache 时 ALFWorld 每轮都要重 prefill 完整历史（缓存命中 0%，单 episode
生成约 1280 s）；开启后 prefill 命中率 86.6%，单 episode 约 495 s，因此默认开启。
响应上限 4096 token；这些数值可通过 `SGLANG_MEM_FRACTION_STATIC`、
`SGLANG_MAX_TOTAL_TOKENS`、`SGLANG_MAX_RUNNING_REQUESTS`、`MAX_RESPONSE` 覆盖。

SRLVR 严格采样器（`frameworks/sglang/.../layers/sampler.py`）的随机 full-softmax 与
实际 logprob 共享同一份 FP32 概率张量。`tp_sync_group` world size ≠ 1 以前会直接
报错，2026-10-27 起改为**只打一次 warning**（每引擎首次采样时），以便全量微调等场景
使用多卡引擎；默认仍为每卡一个 TP1 引擎。使用 TP>1 时应确认引擎日志中的 warning，
并验证采样分布/对数概率正常（该路径假设 logits 已 all-gather，不是词表分片）。
top_p/top_k/min_p、grammar、custom logit processor 以及 v2/v3 的
logit_bias/vocab_mask/penalty 仍然 fail-closed，因为它们会改变采样分布本身。

### 2.2 采样实现与 decode 延迟（2026-10-27）

**主线核心改动**：

- 采样：CDF 替代全词表 Gumbel；去掉重复归一化，普通返回只计算采样 token 的 logprob。
- 特征评分：投影前裁掉词表 padding，减少设备同步，四次 TP 求和合并为一次，特征合包回传。
- 缓存：保留环境池、连接池及跨 turn KV cache；新增每批有界的精确 prompt 分词缓存，不复用旧策略轨迹。
- 全量 RL：`OPSD_LORA=0` 贯通 provider、训练参数选择和评测；推理 TP 独立由 `ROLLOUT_GPUS_PER_ENGINE` 控制。
- 工程修复：GDN 禁用 stub shim；消除指标 L0→L1 依赖；checkpoint 排除持有进程组的临时 HeadRef。
- NPU graph：默认 `SGLANG_ENABLE_NPU_GRAPH=1`，可显式设为 0 关闭。

**本次双节点试跑（2026-10-27 21:36）**：v2/v3 各占一个 8 卡节点，从同一基模开始；
全量 RL、训练 TP=2、推理 TP=2（4 引擎×8 并发）、32 条轨迹、4 个校准位置、四族各 8 条；
响应上限仍为 4096，graph 关闭。早期实测：

| 并发口径 | v2 | v3 |
|---|---:|---:|
| 旧 TP1：每引擎 4 请求，中位 tok/s | 82.59 | 81.03 |
| 新 TP2：每引擎 8 请求，中位 tok/s | 125.91 | 125.53 |
| 折算满并发节点吞吐：旧 → 新 tok/s | 661 → 504 | 648 → 502 |

相同生成量下，TP2 预计耗时为旧配置的 **1.29–1.31 倍**，暂未显示提速。
这是 rollout 0 未结束时的吞吐估计；四族混采与基模重启会改变输出量，完整 rollout 墙钟不能直接混比。
运行及样本记录：`reports/fullrl_tp2_20261027_213627/{manifest,early_throughput}.json`。

**恢复 LoRA（2026-10-27 22:38）**：r=256、alpha=256、dropout=0，保持
`alpha/r=1`、LR=1e-6 与原有全部 attention/MLP targets；可训练参数约 2.79 亿，
比原 r128 翻倍。依据 [LoRA](https://arxiv.org/abs/2106.09685) 与
[PEFT 参数定义](https://huggingface.co/docs/peft/en/package_reference/lora) 扩大低秩容量，
不同时提高更新缩放。Thinking 改为 512-token 提示预算，去掉“一句话”等短思考限制；
严格采样不强插 `</think>`，响应总上限仍为 4096。

训练端：校准分块梯度留在设备累积，TP 按逻辑参数去重，仅回传 C/D/范数标量；
跳过 Y 探针 B=0 行，取消每小块 allocator 清空。micro-batch=1 的 reference、probe、
actor 改为逐 turn 长度 padding，避免每条都计算整批最长 padding；KL、4 条校准与
唯一 actor 更新保留。指标新增 reference 阶段时间与实际 padding token 数。

双节点均恢复推理 TP1、四族混采。v2 先试开 NPU graph（capture batch≤8），
同为每引擎 4 并发，吞吐中位数 **339.4 tok/s**，同期 v3 eager 为 **81.4 tok/s**，
约 **4.2×**；40 秒全节点计数窗口中 v2 为 1451 tok/s（活跃轨迹 20→15），
v3 为 646 tok/s（27→28）。随后 v3 也开启 graph，已实测 4 并发约 336 tok/s。512 是提示预算，未强制截断思考。
新运行与计数证据：`reports/lora256_think512_20261027_223845/`。
**首个完整 v2 rollout 实测：652.6 秒（10.9 分钟），922160 token、32 条轨迹、
1293 个 turn，无补采样，已低于 30 分钟。**这是生成阶段墙钟；训练端节省另按阶段计时。
已完成 graph→训练的显存切换；逐 turn padding 在两个 DP 分片实测从
174 万／224 万计算 token 降至约 89 万，reference／校准／actor 的阶段耗时继续记录。

**全量 RL＋graph 短测（2026-10-27 23:12）**：512-token 思考提示预算、32 轨迹、训练 TP=2；
推理 TP1（8 引擎×4 并发）中位 348.8 tok/s/引擎，TP2（4 引擎×8 并发）为 541.7。
折算满并发节点吞吐为 **2790 vs 2167 tok/s**，单序列 **87.2 vs 67.7 tok/s**，优先推理 TP1。
45 秒实际计数约 **1973 vs 2012 tok/s**，活跃轨迹分别 23→24 与 27→27，不能据此判作同速。
全量 TP1 与 LoRA256＋graph（约339 tok/s/引擎）接近；此次未测完整反向/优化器成本。
两组短测已停止，待选参数后启动正式训练测评。证据：`reports/fullrl_graph_tp_compare_20261027_231204/result.json`。

**正式双节点启动（2026-10-28 03:30，进行中）**：v2/v3 各占一个 8 卡节点，从基模新开，
全量 RL、训练 TP2/DP4、推理 8×TP1、NPU graph、thinking 开且无独立 token 预算、
响应上限 4096、完整历史、四族混采各 8、32 条轨迹 + 4 校准、50 批、每 25 批保存。
编排与全部证据：`reports/srlvr_v23_formal_20261028_015123/STATUS.md`。

- 首个完整 rollout：v2 816–845 s / 1443–1507 turns / 0.89–0.93 M generated tokens，
  v3 997–1020 s / ~0.94 M；均 32 条真实轨迹、`topup_rounds=0`、`npu graph: True`。
- 首个 actor 更新：v2 `grad_norm=14069.6`、v3 `14225.1`；第二步
  `srlvr_reference_kl_sum=1.37`（v2）/`1.24`（v3）> 0，确认权重真实更新。
- 单批：rollout ~740–1020 s + actor 阶段 ~737–892 s（含 ref logprob、4 个校准
  backward、1 次更新），权重同步 ~2 s。
- **共卡 OOM 与修复**：引擎 KV 池 98304 token（K/V 各 5.26 GB）在第一次权重同步后
  恢复时放不下（`torch_memory_saver.cpp: aclrtMallocPhysical failed`；v3 首轮必中，
  v2 撑到 rollout 1 后同样命中），整机以 `onload_kv` 的 `ConnectionError` 退出。
  池降至 65536 token（K/V 各 3.51 GB）后两臂连续多批无 OOM。该值不是上下文预算
  （上下文仍 32768，实测 `token usage` 峰值 0.10–0.12），未缩短上下文或减少轨迹。

严格 SRLVR 采样器原先使用全词表 Gumbel：每生成一个 token 都要对
`[批量, 151936]` 做 murmur 哈希、转 float64 做两次 log/neg + argmax，并执行一次
`tolist()` 的 device→host 同步。在 Ascend 上这是 decode 的主要成本——实测每序列
仅约 11 tok/s（每引擎 4 条并发、聚合 46–49 tok/s），rollout 墙钟由最慢 episode 的
生成时间决定。

新实现（`frameworks/sglang/.../layers/sampler.py::categorical_with_seed_cdf`）：

- 每行只算一个 murmur 均匀数（由 `seed + position` 决定），`cdf = probs.cumsum(-1)`
  （fp32），再用 `searchsorted` 取第一个 `cdf ≥ u` 的词表下标。采样分布与
  `categorical(probs)` 完全等价，仍按 `(seed, position)` 确定。
- 每 token 的采样成本从「O(V) 次哈希 + 多次 float64 逐元素运算」降为
  「一次 O(V) fp32 cumsum + 一次查找」。
- `srlvr_logp`（以及 v1 的 `srlvr_mu/f/e`）保留为 device tensor：scheduler 在
  每个输出包（约 300 token 或请求结束）一次性 `stack().cpu().tolist()`，
  去掉每 token 一次的 host 同步。
- **默认开启**（`SRLVR_SAMPLING` 缺省为 `cdf`）；`SRLVR_SAMPLING=gumbel` 回退到旧
  实现。注意 RNG→类别映射与旧实现不同：同一 seed 抽到的 token 序列会变化（分布
  相同），与旧 run 的逐步轨迹不逐位可比。

共卡显存让渡依赖 **Ascend 版** `torch-memory-saver`（`sgl-kernel-npu` 的
`contrib/torch_memory_saver`，本仓库源码在
`compat/vendor/sgl-kernel-npu-memory-saver/`，构建产物在 `compat/wheels/`）。
**不要**安装 PyPI 的 `torch-memory-saver==0.0.10`：那是 CUDA 版，SGLang 引擎
初始化会报 `could not detect CUDA runtime. Tried torch.version.cuda and
libcudart.so.{13,12}`。

```bash
# 编译
cd compat/vendor/sgl-kernel-npu-memory-saver/contrib/torch_memory_saver/python
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.1 \
  /opt/conda/envs/cua-rl/bin/python setup.py bdist_wheel \
  --dist-dir <repo>/compat/wheels

# 两个节点分别安装（Python 环境是节点本地的）
/opt/conda/envs/cua-rl/bin/pip install --no-deps --force-reinstall \
  <repo>/compat/wheels/torch_memory_saver-0.0.8-cp310-cp310-linux_aarch64.whl
```

共卡时 `_common_v2.sh` 自动设置（手工启动必须一致）：

```bash
export TMS_HOOK_MODE=torch                 # NPU 走 torch allocator + MemPool
export ASCEND_COREDUMP_SIGNAL=none         # CANN 信号处理与 TMS 冲突
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
unset PYTORCH_NPU_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF   # 与 TMS pause 不兼容
```

- 漏掉 `HCCL_*_PORT_RANGE`：训练 actor 与同卡引擎争用默认端口 16666，训练端以
  `hcclCommInitRootInfoConfig ... error code is 7`（`HCCL_E_UNAVAIL`）退出。
- 保留 `expandable_segments:True`：TMS `pause()` 静默不释放，最终 OOM。
- 漏装包：`ModuleNotFoundError: torch_memory_saver`；装错版本：上面的 CUDA 探测错误。

安装后应先用 `DRY_RUN_PARSE=1` 检查最终参数，确认 `colocate=True`、
`optimizer_cpu_offload=True` 和 `enable_memory_saver=True`。

所有真实 turn rows 均保留，按 DP 数补零 mask 行，整批梯度累积后只更新一次。不再需要单独的 fullbatch 配置，也不通过丢弃尾部 turn 或增加真实环境尝试来对齐 batch。padding 不进入奖励与方法历史统计。

### 2.3 SRLVR-v2/v3 actor 计算优化（2026-10-28）

本节记录训练端实现变更，供论文的实现细节与计算成本说明使用。适用对象为
`methods/srlvr_hsi.py` 的 v2/v3 路径；实际启动配置为 Qwen3-1.7B 全量 RL、
单节点 8 卡、TP2/DP4、PP=CP=1、`bshd`、micro-batch=1、BF16、全层激活重计算。
Flash Attention、逐 turn 长度 padding 和关闭 optimizer CPU offload 在此次修改前
已经生效，不计为本次新增优化。

**修改前的阶段耗时与口径。** 2026-10-28 调查时读取正式运行
`alfworld-qwen3-1p7b-srlvr_v{2,3}-full-s42-r50-n32-20261028_015123` 的
`common_metrics.jsonl` 与 `srlvr_metrics.jsonl`，取 rollout 13–17 五批。
每批两个 TP rank 的重复内部记录先取均值，再跨批平均；不把 rank 记录相加。

| actor 内部阶段 | v2 均值（秒） | v3 均值（秒） | 占各自内部四阶段总和 |
|---|---:|---:|---:|
| reference log-prob、参考权重切换与阶段清理 | 68.85 | 69.22 | 9.4% / 9.3% |
| causal/hindsight 打分、特征汇总与未来校正 | 245.75 | 251.62 | 33.4% / 33.7% |
| 四条校准轨迹的 Z/Y 梯度探针与统计 | 215.97 | 218.05 | 29.4% / 29.2% |
| 正式训练更新、权重备份及报告准备 | 204.72 | 207.59 | 27.8% / 27.8% |
| 外层 `actor_phase_seconds` | 737.89 | 748.91 | 含其他准备及等待 |

外层 actor 计时覆盖 `async_train` 的等待完成；checkpoint 写入和后续权重发布
另计。内部 `actor_seconds` 包含模型前向、反向、激活重计算和 optimizer，不能
解释成 optimizer 单独耗时。上述为修改前已有墙钟记录，尚无算子级拆分；主要
开销集中在打分与校准，两者合计约占内部耗时的 63%。它们也不是相对 GRPO 的
受控额外开销测量。

**变更一：hindsight 特征每行一次回传。** `score_row` 保留每个 turn 的 causal
与 hindsight 两次 hidden-state forward，以及原有全词表分片计算。有效位置索引
直接由已有 host mask 构造；每个位置块只在设备上保存 `mu/f_chosen/e/student_logp`
四个标量，完成一行后合并为一次 `.cpu().tolist()`，替代每块回传特征和索引。
`position_chunk` 默认仍为 64。若一行有 M 个有效位置，则特征包回传次数由
`ceil(M/64)` 降为一次（M=0 时不回传）；这一计数不包括 TP collectives 或其他
设备同步。新增留存空间为 O(4M)，完整词表 logits 仍逐块释放，不缓存整行 logits。
特征输出位置、零 mask、有限值检查与后续未来校正定义保留。

**变更二：有效 response 位置的输出投影（默认已回退为关闭）。**
`run_alfworld_v2_1p7b.sh` 曾对 v2/v3 默认设置
`OPSD_SRLVR_SELECTIVE_LOGITS=1`、`OPSD_SRLVR_SELECTIVE_CHUNK=128`，
复用 `srlvr_compact.py` 已有的 hidden-only forward 与分块输出头路径。
decoder 仍读取完整历史；只跳过 loss 不使用的 prompt、padding 和零 mask 位置
的 lm-head 投影。有效位置仍对完整模型词表归一化，按 TP 分片计算选中 token 的
log-prob；没有 top-k 近似或词表截断。输出头按位置及词表分块，训练时使用
checkpoint 重计算控制中间张量显存；词表块默认 8192，可用
`OPSD_SRLVR_SELECTIVE_VOCAB_CHUNK` 覆盖。额外分块与重计算是否改善墙钟需实测。
当前默认 `OPSD_SRLVR_SELECTIVE_LOGITS=0`，使用原输出头路径；显式设为 `1`
仍可启用 response-only 路径。该路径要求
micro-batch=1；本次验证范围为 PP=CP=1 的 `bshd` 配置。

**TP 梯度修正与同步消除。** 对上述 selected-token log-prob 路径，loss 在
各 TP rank 上复制。全局 log-partition 和选中 logit 的 SUM 归约均采用反向
pass-through；复制的 hidden 输入通过 `grad_sum` 在反向合并各词表分片贡献。
此前工作区版本对 log-partition 使用 SUM backward，会额外乘上 TP 数，仅前向
数值相同不足以证明正确。现已修正，并对照完整输出头检查 hidden 与输出权重
梯度。该说明针对复制的 selected-token loss；不能将同一规则直接套到按词表
分片累加的分布损失。另将词表 padding 判断从设备布尔张量转成 host 整数边界
判断，去掉每个词表块的 `bool(tensor)` 隐式同步。

这些变更在实数运算下保留相同有效 token 的全词表 log-prob 与梯度；浮点分块及
归约顺序可能产生舍入差异，不保证逐位相同。不改采样分布、上下文、轨迹数、
四条校准轨迹、Z/Y 探针定义、每批一次 optimizer 更新或损失归约，也未实施
校准轨迹在 DP 副本间重新分工。

**验证与性能证据状态。** `tests/test_srlvr_actor_speed.py` 的 CPU/Gloo 检查
为 **5 passed（24.94 s）**：覆盖 v2/v3 特征、非连续 mask、全零 mask，以及
带词表 padding 的 TP2 response log-prob、hidden 梯度和输出权重梯度，对照 dense
数学结果。另通过 shell 语法与改动空白检查。这些是小规模正确性检查，尚未完成
Ascend/BF16 整批梯度或修改后 actor 墙钟验证；未做性能 A/B。

单次 actor 回放入口为 `reports/actor_only_20261028_131625/replay.sh`，复用保存的
rollout-0 与基模权重，32 条轨迹、TP2/DP4、四条校准，不重新生成环境轨迹。
预定报告 ref/score/calibration/update 及外层 actor 时间，加载与 checkpoint 另计。
回放不含 SGLang 引擎驻留，且使用初始方法历史，不能直接当作正式共卡训练的
同条件对照。此次检查时两个可连接节点的 16 张卡均被训练占用，启动检查退出，
**没有得到修改后的 actor 时间，也没有可报告的加速比**。记录与检查日志见
`reports/actor_only_20261028_131625/{RESULT.md,correctness.log,device_check.log}`。
已加载旧模块的活跃进程未重启；启动器新默认在后续启动时生效。

**后续观测与默认回退（2026-10-28）。** 操作员报告另一服务器的 v2 未启用
response-only，而稍后启动的 v3 启用了该路径；reference 约 30→83 s、正式更新
约 110→240 s。这是不同方法臂、不同批数据的运行观测，不是受控 A/B 加速比。
当前双重分块（位置 128、词表 8192）引入大量小矩阵计算和 TP collectives，
训练时还增加输出头 checkpoint 重计算，因此省显存不保证提速。按操作员要求，
v2/v3 启动器默认回退到 `OPSD_SRLVR_SELECTIVE_LOGITS=0`，保留显式开启入口；
hindsight 每行一次特征回传的修改保留。未重启活跃训练，也不改变已启动进程
继承的环境变量。上述两臂的效率比较须标明实际执行路径及源码版本。

可用于论文实现段落的表述：hindsight 打分按位置分块计算，并将每行的低维特征
合并传回主机，减少细粒度设备同步。有效响应位置的输出头投影作为可选显存路径
保留，其数值与梯度契约通过小规模 TP2 检查；由于观测到训练变慢，未作为默认
提速方案。具体实验是否启用须以启动环境和投影日志为准。

### 2.4 v2/v3 逐轨迹 token 平均与历史状态迁移（2026-10-29）

本次是优化目标及校准统计口径修正，区别于 §2.3 的计算实现优化。旧运行的 loss、
梯度范数、C/D 与历史 baseline 均不能当作新口径结果直接混用。

- **轨迹长度与传递**：在 `srlvr_hsi.convert` 中、DP 切分前，按采样位置汇总所有
  turn 的二值 loss mask，生成每条轨迹的 `actor_token_count=L_i`，写入已有
  `srlvr_rows` 元数据。零 mask 填充行不计入长度或真实轨迹数；有效 actor token
  总数为零的真实轨迹明确报错。该元数据随原 DP partition 和 microbatch iterator
  到达 actor loss。打分后的全局行汇总再次计算 L_i，校准与正式更新使用同一值。
- **actor 与探针**：`_actor_loss_impl` 对每个 turn 的 PPO、KL 或原始 probe loss
  总和只除一次所属轨迹 L_i。`rows/N` 只补偿框架原有行平均，保留跨轨迹 mean；
  不再另除 turn 长度、chunk 长度或整批 token 数。Y 探针跳过 B=0 的 turn、探针
  为微批对齐补零 mask 行时，分母仍是筛选前整条真实轨迹的 L_i。探针的完整梯度
  仍由各 chunk 累积后计算内积；相对旧 sum 梯度，Z/Y 分别乘 1/L_i，C/D 的每条
  校准轨迹贡献分别乘 1/L_i²，再对校准轨迹取均值。
  日志中的历史字段名 `train/srlvr_loss_sum`、`train/srlvr_reference_kl_sum` 保留，
  但也补偿框架行平均，新口径分别报告逐轨迹平均的完整 actor loss 与未乘 beta 的
  KL，不能再与旧 row-mean token-sum 数字直接比较。
- **baseline 权重**：探针返回的 `zz=||Z_i||²` 已包含 1/L_i²，因此历史记录使用
  `W_i=zz*(s/(R_i-b))²=||S_i||²`，不再次除 L_i²。历史仍记录 `[R_i,W_i,b_i]`，
  b 按 W 加权并沿用原边界裁剪；s 沿用原残差二阶矩公式；lambda 沿用新 C/D 比值
  及原裁剪规则。B 的定义不变，当前批始终使用先前历史冻结的 b/s/lambda，新的
  统计在本批唯一一次成功 actor 更新后提交。
- **checkpoint 兼容**：新方法状态带 `reduction=trajectory_token_mean_v1`。
  `restore_history` 与训练入口均检查版本。无标记的旧 sum 历史（或显式
  `trajectory_token_sum_v1`）没有每样本 L_i，不能精确换算，故重置方法统计为
  `b=.5,s=1,history=[],lambda_raw=0,lambda_next=0`，保留 `last_rollout`、累计
  optimizer 更新计数和模型/优化器状态。重置会写诊断日志；新标记状态恢复后
  原样保留，不每批重置。未知版本拒绝加载。继续旧 checkpoint 只是保留参数和
  optimizer 后切换新目标，不代表此前批次已经使用 token mean；论文须注明切换点。

验证入口为 `tasks/alfworld/tests/test_srlvr_normalization.py` 及更新后的
`tasks/alfworld/tests/test_srlvr_hsi.py`：使用不同长度轨迹、长 prompt、非连续 mask、
零 mask padding、实际框架 loss 缩放和 DP/微批元数据传递，对照直接构造的
逐轨迹平均 loss/梯度；另验证不同探针 chunk 与 microbatch、Z/Y、C/D 的 1/L_i²
权重、W 无二次归约、b/s/lambda 时序以及旧/新 checkpoint 状态。测试为 CPU
数学与接线验证；本次未重启活跃训练，未声称完成新的 NPU 训练或性能测量。
复现所需的微批字段接线已同步到 `compat/patches/srlvr-framework.patch`。

### 2.5 Qwen3-8B / 8×32GB 节点配置（2026-11-05）

单节点 8 张 Ascend 910B4（32 GB）跑 Qwen3-8B SRLVR-v2，全量 RL（`OPSD_LORA=0`）。
入口 `tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh`，当前默认：

| 项目 | 值 |
|---|---|
| 放置 | `ACTOR_GPUS=8`、`ROLLOUT_GPUS=8`、`COLOCATE=1`、`PP=1` |
| 训练并行 | `TP=4` → **actor DP=2** |
| 推理并行 | `ROLLOUT_GPUS_PER_ENGINE=4` → 2 个 TP4 引擎（占满 8 卡）|
| 批组织 | `G=1`、`ROLLOUT_BS=32`、`GBS=32`（直接传 `--rollout-batch-size/--global-batch-size 32`，不经 `_v2_defaults.sh` 的 `ROLLOUT_BS*G` 推导）|
| 预算 | `MAX_TURNS=50`、`MAX_RESPONSE=4096`、`SEQ_LENGTH=32768` |
| 方法 | `SRLVR_CALIBRATION=4`、`SRLVR_GAMMA=0.99`、`ALFWORLD_V2_ADVANTAGE_MODE=batch` |
| 持续采样 | `ALFWORLD_V2_CONTINUAL=1`、`MAX_INFLIGHT=32`、`ALFWORLD_V2_POOL_CAPACITY=64` |
| SGLang | `mem_fraction_static=0.42`、`max_total_tokens=65536`、`chunked_prefill=8192`、`server_concurrency=8`、`max_running_requests=8`、`cuda_graph_max_bs=8`、prefix cache 与 NPU graph 开 |
| 优化器 | `adam`、CPU offload（fraction 1、precision-aware）、`lr=1e-6`、`clip_grad=1.0`、`eps_clip=0.2/0.2`、KL `low_var_kl:0.001`、full uniform recompute 1 layer |

启动（同步 100 rollout）：

```bash
ACTOR_GPUS_32G=8 ROLLOUT_GPUS_32G=8 ROLLOUT_GPUS_PER_ENGINE_32G=4 COLOCATE_32G=1 \
MAX_RESPONSE=4096 NUM_ROLLOUT=100 SAVE_INTERVAL=10 SAVE_TRAIN_STATE=1 \
ALFWORLD_V2_CONTINUAL=1 ALFWORLD_V2_POOL_CAPACITY=64 \
RUN_NAME=<task>-<model>-<method>-<variant>-<stamp> \
bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh
```

**单步实测拆解**（`colocate-r4096-20261105_103559`，rollout 0，cap 4096）：

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| 启动（Ray/SGLang 加载/首轮权重同步） | 7.3 min | 一次性 |
| rollout | **1566.5 s（26.1 min）** | 32 eps / 651 turns / 689,065 tokens，成功率 81.25%，440 tok/s |
| ref logprob | 82.8 s | |
| calibration probes | **907 s（15.1 min）** | 4 个位置，Z/Y 各一遍 |
| actor step + initial ref KL | 311 s | 1 次 optimizer step，grad_norm 0.365 |
| save | 19.9 s | `iter_0000000` ≈16 GiB |
| **稳态合计** | **≈48 min/step** | 与 100-rollout 实测 46–49 min 一致 |

显存：probe 峰值 `peak_reserved=20.23 GiB`；actor update 后 `alloc 11.51 / used 25.33 / free 4.16 GiB`（29.49 GiB 可用）。

**probe 为什么比 4B 慢**：4B 用 TP2/DP4，每个 DP rank 1 个位置；8B 在 32 GB 卡上必须
`TP×PP=4`（每卡约 2B 参数：bf16 4 GiB + fp32 main_grad 8 GiB ≈ alloc 11.5 GiB），
8 卡只剩 `DP=2`，每 rank 2 个位置。单次 profile（positions [1,3]）：
`wall 625.8 s = train 517.3 + copy 314.9 + moments 104.8`（重叠），`chunks=25`、
`turns=118`。换 TP2 每卡需 ≈24 GiB 权重+梯度，加激活必然超 32 GB，故 **DP=4 在 8B/32 GB
上不可达**；要 1 probe/rank 只能 16 卡或 64 GB 卡。合批两个 position 也不可行：
梯度可加，合批只能得到两条轨迹的和，probe 需要逐 position 的 zz/zy/yy。

**async 放置约束**：Slime `train_async.py` 要求 `COLOCATE=0`，8 卡只能 4+4，rollout 吞吐
约减半（~279 tok/s 实测），且 actor 退化为 TP4/DP1 → 每 rank 4 个 probe 位置、probe 翻倍。
因此单机 8 卡的 async 单步 ≈ `max(41, 43)` 分钟，与共卡同步的 48 分钟基本持平；
`max(rollout, actor) ≈ 26 min` 只在 16 卡（8+8）成立。

**stale-policy 校正（为 async 准备的实现）**：`opsd_core/srlvr_settings.py` 增加
`stale_lag/max_stale`（`SRLVR_STALE_LAG`/`SRLVR_MAX_STALE`，限制 `0≤lag≤max≤1`）；
`methods/srlvr_hsi.attach_rows` 按 `rollout_id - lag` 打生成版本（首个 async rollout 强制
lag=0），`resolve_policy_gap` 拒绝混合快照/未来版本/超过 max 的 gap，并输出
`[srlvr-stale]`、`[srlvr-stale-ratio]`（mean ratio、clipped_frac）。梯度校正沿用 loss 中
已有的 `exp(logp_new - rollout_logprob)` PPO clip（async 下必须 `--use-rollout-logprobs`）。
`_common_v2.sh` 的闸：async+SRLVR 要求 `SRLVR_STALE_POLICY_CORRECTION=1`、variant ∈
{v2,v3}、`COLOCATE=0`，且**暂不允许 `RESUME_FROM`**（rollout manager 看不到
`start_rollout_id`）。测试：`tasks/alfworld/tests/test_srlvr_hsi.py` 新增 7 条 stale 用例。

## 3. 交互协议与优化器

| 参数 | 当前默认设置 |
|---|---|
| 任务 | ALFWorld text-only，六任务族 |
| 奖励 | 仅终局二值奖励，成功 1、失败 0，无中间奖励塑形 |
| 最大交互轮数 | 50 |
| Thinking | 默认启用，不设独立 thinking token 预算；v4 hindsight 只保留 action span |
| 单轮输出 | 最多 4096 token，输出一个 `<action>...</action>` |
| 停止条件 | `</action>`，保留闭合标签 |
| 当前可执行动作列表 | 默认展示（`ALFWORLD_SHOW_ADMISSIBLE=1`，2026-10-27）；保留环境合法性检查和诊断信息 |
| 系统提示 | 包含静态命令语法说明；默认追加当前状态的 admissible actions（`ALFWORLD_SHOW_ADMISSIBLE=0` 可回退） |
| 历史 | 完整 observation/action 历史，无最近 8 轮滑窗 |
| 训练／推理上下文窗口 | 默认 32768 token |
| 训练解码 | temperature=1、top_p=1、top_k=-1，完整 softmax |
| 学习率 | 1e-6，constant，无 warmup/decay |
| 优化器 | Adam，beta1=0.9、beta2=0.999，weight_decay=0 |
| 梯度裁剪 | 1.0 |
| PPO clip | 上下对称 0.2，即 ratio 区间 [0.8,1.2] |
| 初始模型 KL | 系数 0.001，`low_var_kl` |
| 熵奖励系数 | 0 |
| micro batch | 1 个 turn row；整批累积后更新 |
| 重计算 | full activation recomputation |

每条训练记录仍对应一个 turn，但输入包含该轮之前的完整 observation/action 历史。上下文窗口由 8192 提高到 32768，训练序列长度、训练端位置容量与 SGLang context length 同步；单轮输出上限为 4096 token。Qwen3-1.7B 的原生位置上限为 40960 token。完整输入加响应预算超过窗口时明确报错，不静默裁剪历史。GRPO 的 old logprob 默认来自更新前的 trainer forward；SRLVR 的 old logprob 来自实际 rollout sampler。

## 4. 历史成功率课程与四族混采

课程仅使用训练池。当前 game index 包含 3553 个训练游戏，课程不使用验证集结果更新采样权重。每批采样前冻结历史统计，Beta(1,1) 平滑成功率为：

```text
p_i = (wins_i + 1) / (trials_i + 2)
```

历史 wins/trials 每批乘 0.9，再加入该批全部观测结果。族内采样概率由归一化权重与 10% 均匀探索混合。保留全部采样结果，不筛除全成功／全失败组，不重采样直到得到正例。

### GRPO：混合奖励组概率

```text
w_i = 1 - p_i^8 - (1-p_i)^8
```

六任务族按任务位置轮转。每批 4 个任务位置，连续 3 批覆盖每族 2 次；每个位置在所属族内按上述权重有放回抽样，随后生成 8 条轨迹。该权重针对 GRPO 全同奖励组的优势为零这一性质。

### SRLVR：历史奖励信号二阶矩代理

令 `b_f` 为任务族 f 的历史平滑成功率，族内权重为：

```text
w_i = p_i * (1-b_f)^2 + (1-p_i) * b_f^2
```

这是围绕历史族 baseline 的奖励优势二阶矩代理，不改变 SRLVR 实际使用的 b、s、A 或 B。`b_f=0.5` 时权重均匀；任务族较易时偏向未掌握任务，较难时偏向已有一定成功机会的任务。它不要求 G>1，也不增加每题环境尝试数。

单轨迹方法默认采用与 GRPO 相同的四族混采：每批与 GRPO 的四任务槽共享同一个四族窗口（`start = (rollout_id * 4) % 6`），32 个位置按 `start + position % 4` 循环，每族 8 个位置。C（calibration）与 U（update）位置因此看到相同的四族混合分布，而不是整批只有一个族；六个族仍跨 rollout 轮转。两方法训练池、轨迹预算和六族覆盖相同，具体任务序列及课程权重不同。

### 2026-10-27 v4 默认变更：hindsight 去思维链

- v4 的 teacher continuation 现在只拼接每轮 `<action>` 内容的 token（`SRLVR_V4_HINDSIGHT_NO_THINK=1`，默认），不再把整段 raw generation（含 `<think>`）放进 teacher view；privileged trace、observation 和 outcome 不变。
- 学生 row 也只保留 action token（`prompt | action`），loss 对这些 action token 做 per-token distillation；rollout 仍按 `ALFWORLD_THINKING_STYLE` 生成 thinking，但 thinking 不进入 hindsight，也不进入 v4 loss。
- 目的：v4 的 overlong teacher view 主要由 continuation 的 thinking token 造成；去掉后 teacher view 显著变短，`overlong teacher view ... rows emitted with zero loss` 的比例应下降。`SRLVR_V4_HINDSIGHT_NO_THINK=0` 可回退旧行为做 A/B。
- 该变更是方法口径变更，与旧 v4 checkpoint/结果不可直接混用；评测和对比时应记录该 flag 以及四族混采默认。

课程状态按批保存。回退到某批时只读取该批之前的统计，重放不重复累计奖励。该单轨迹课程是新的采样启发式，其提点效果需要独立评测验证。

## 5. 评测与论文报告口径

公共 ALFWorld eval-only 入口默认 `COLOCATE=1`，actor 与推理共用节点全部设备；
标准 8 卡节点默认 8 个 TP1 推理引擎（`ROLLOUT_GPUS_PER_ENGINE=1`）。
设备数由 `NUM_GPUS_PER_NODE` 推导。显式 `COLOCATE=0` 保留独立 actor／推理分配，
`ACTOR_GPUS`、`ROLLOUT_GPUS` 和推理 TP 仍可覆盖；训练默认与评测交互／采样口径不变。

- 官方划分：`valid_seen` 140 题、`valid_unseen` 134 题；仓库另有 dev64 评测配置。dev64 与训练池的隔离应以具体实验数据清单核实，不能仅凭文件名认定为 held-out。
- 每题评测 1 条轨迹，报告 episode success rate；不是多次采样 pass@k。训练模型与基模使用一致的新版交互协议：关闭 thinking、完整历史、128 token cap、50 turns，不展示动作列表。
- 当前 ALFWorld v2 evaluator 从 `GenerateState` 获取采样参数，没有按数据集覆盖解码参数。沿用公共默认入口时，实际为 T=1、top_p=1、top_k=-1；评测 JSON 中旧的 T=0.6、top_p=0.95、top_k=20、4096 cap 不能直接当作生效设置。该版本未将评测改为 mentor 的 T=0 贪心协议。
- 主要效果指标为独立评测成功率，可辅以分任务族成功率、非法动作率和达到步数上限的比例。训练采样成功率只作为优化诊断；课程在改变题目难度，不应直接视作泛化学习曲线。
- 效率比较同时报告真实轨迹数、生成 actor token 数、optimizer 更新数、校准开销和 wall-clock／设备时间。SRLVR 包含额外校准 backward，32 条轨迹预算相同不代表计算预算完全相同。
- 本记录未包含新版协议的独立评测成绩，不将历史诊断样本的训练成功率作为新版性能结论。

## 6. 历史诊断结果与修订依据

### 6.1 训推 logprob 偏差

早期同一冻结策略的采样端与训练端出现局部 logprob 偏差：一个 15-token turn 中，某 token 的两端 logprob 为 -3.65376 和 -2.58357，该 turn 的平均 bias 为 0.07785。另一旧预跑的 10-token turn bias=0.199637、MAE=0.229550。

这些观测支持显式记录行为策略概率并使用重要性比率，同时保留对齐诊断。`alignment_bias_max`／`alignment_mae_max` 是各 turn 误差统计的最大值，不是整批平均；局部告警不能单独证明系统性分布错位，也不能断言仅为数值噪声。当前 ratio 使用 token-level clipped surrogate，校准仍使用原始 score。

### 6.2 旧 GRPO 的 batch 组织问题

旧配置使用 8 题×8轨迹，GBS=64 指 turn rows。数据按 task/episode/turn 排序，DP 按行交错分配，iterator 按行取 minibatch；因此单次更新常只接触少数轨迹的一段。旧 rollout dump 的离线统计为：

| rollout | 保留 turn rows | 保留轨迹数（预算 64） | optimizer minibatches | 全零优势批 | 单一非零符号批 | 同时含正负优势批 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2368 | 62 | 37 | 16 | 13 | 8 |
| 24 | 1984 | 62 | 31 | 14 | 6 | 11 |
| 49 | 2432 | 63 | 38 | 18 | 13 | 7 |

该旧配置的 2385 条 optimizer 日志中，50.27% 的 gradient norm 为零；已有 Adam 动量仍可能在零当前梯度下改变参数。尾部裁剪还可能破坏完整轨迹和组中心，例如一批的 `traj_normalized_adv_sum=-0.935`。在转换时对完整 rollout 计算轨迹权重，不能保证后续多个独立 minibatch Adam 更新仍等价于完整组目标。

这些结果解释了改为保留全部轨迹、整批一次更新的依据，不单独证明旧评测不提升的全部原因。历史 `5e-5` 步长试验不属于当前默认配置，也不等价于对旧 minibatch 更新预算的换算。

诊断证据：旧 GRPO run `alfworld-qwen3-1p7b-grpo-s42-r100-n8-20261020_2230` 中的 `rollouts/rollout_{0,24,49}.pt`、`train.log`；训推对齐证据来自旧 SRLVR alignment 记录。这些标识仅用于追溯诊断数据。

## 7. 实现与复现入口

| 内容 | 代码 |
|---|---|
| 公共模型与实验入口 | `tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh` |
| 共用训练参数 | `tasks/alfworld/scripts/_common_v2.sh` |
| SRLVR 系数、校准与 actor loss | `methods/srlvr.py::{choose_lambda,actor_loss,train_batch}` |
| SRLVR-v2/v3 轨迹 token 平均、C/D/W 与历史版本 | `methods/srlvr_hsi.py::{convert,train_batch,actor_loss,prepare_history,next_history}` |
| GRPO 组优势与 rollout | `tasks/alfworld/rollout.py` |
| 完整历史与 action-only 提示 | `tasks/alfworld/{episode,prompts}.py` |
| 历史课程 | `tasks/alfworld/curriculum.py` |
| 轨迹归约、初始模型 KL | `tasks/alfworld/{train_data,loss}.py` |
| LoRA provider | `methods.lora_megatron.custom_model_provider` |

具体实验的有效参数以 `effective_args.json` 和 `effective_config.json` 为准。恢复实验需同时保留 optimizer／scheduler／RNG、SRLVR 的历史归一化状态和课程统计。框架目录不受主仓库 Git 跟踪，迁移时还需包含 `compat/patches/srlvr-framework.patch` 与 `compat/patches/alfworld-reference-kl.patch`。
