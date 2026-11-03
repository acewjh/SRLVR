# 运行注意事项（opsd-slime / Server A）

一句话：**开工先跑 canary；一次只允许一个 launcher；卡住先 `py-spy` 再动手。**
每条都是实际踩过并复现过的，不是预防性猜测：§0–§6、§8 来自 2026-09-18~09-23，
**§7（ALFWorld v2 harness）来自 2026-09-30~10-03**，**§9（code RLVR）来自 2026-10-01~03**，
标注了对应数据。§1/§2/§5/§8 也各补了 10-01~03 的新条目。

## 0. 开工前（每次都做）

```bash
conda activate cua-rl
# grader canary —— 依赖会静默消失，缺了会让整条流水线 fail-closed 中止
python -c "from math_opsd.reward import grade_answer; print(grade_answer('540','\\textbf{(540)}'))"
#   必须 True 且 stderr 出现 [grader] official math_verify grader ACTIVE
#   缺失时补装： pip install --no-deps latex2sympy2_extended

npu-smi info                       # 谁在占卡
pgrep -af 'slime-ascend/train[.]py'
cat /tmp/opsd_watchdog/active.env  # 有没有 training 在跑
```

## 1. 打分器（最容易"静默出错"的地方）

- **绝不要因为非主线程就把超时置 0**。`math_verify` 用 `signal.alarm`（只在主线程可用），
  而训练 rollout 跑在 RolloutManager 的**工作线程**里。以前就是在这里把超时关掉的
  ⇒ 一条病态答案让 sympy `solve` 无限展开，线程占满 GIL、零日志，**看起来和 SGLang 死锁一模一样**
  （2026-09-18/19/20 三次"静默 wedge"真因就是它）。离线打分一律走 `math_opsd/grader_pool.py`。
- **传给 `math_verify` 的 `timeout_seconds` 必须是 `int`**：
  `verify(x, y, timeout_seconds=5)` → `True`，而 `timeout_seconds=5.0` → **`False`**。
  `signal.alarm` 不收 float，异常被吞成 `False` ⇒ **float 超时会静默把所有答案判错**。
- `timeout_seconds <= 0 / None` 会**关闭超时**并打印 `Timeout is disabled`；别用它来"省事"。
- **同一个 `int` 陷阱在 LiveCodeBench 官方 runner 上重现**（2026-10-02）：LCB 的
  `testing_util.run_test` 用 `signal.alarm(timeout)`，给它 `6.0` 时每个评测进程抛
  `TypeError: 'float' object cannot be interpreted as an integer`，**worker 静默死掉、结果里直接少一批题**
  （Python 版本还能从字面量贴出来的 traceback 里看到）。凡是要过 `signal.alarm` 的超时一律 `int`。
- **`reliability_guard()` 会破坏它自己所在的那个解释器**（把 `os.environ` 等换成受限对象），
  所以 LCB 的 `run_test` **一个进程只能评一次**。想手工复核若干条判定，必须一条一个子进程；
  在同一进程里连着调第二次会 `TypeError: 'NoneType' object is not callable`（2026-10-02 抽查时踩到）。
- **训练用的 reward 必须由与评测同一份代码产生**——这是本仓库最贵的一条教训（2026-10-02）：
  训练侧曾自己写 stdio verifier + 自己的抽取器，评测侧用 LCB 官方 `grade_stdio`/`grade_call_based` +
  官方 `extract_code`，两者对"最终提交是哪一块"的理解不同 ⇒ 训练在优化一个**和评测不一样的量**。
  改成都走官方 runner 后，评测里的 `extraction_disagreement` 从 43/47 变成 **0**。
  推论：**抽取器只能有一套语义**。官方规则是"取最后一个 fenced block"，训练侧就绝不许
  再往前搜索"更正确的那个 block"——那等于教模型"最终提交错了也没关系"，而评测不是这么判的。
- **在沙箱里跑官方 grader 前，把 BLAS/OpenMP 线程钉成 1**：本机 192 核，未钉线程时
  numpy/OpenBLAS 会预留巨量**虚拟**地址空间，撞上我们给子进程设的 `RLIMIT_AS` 后导入失败，
  报的却是极具误导性的 `Error importing numpy: you should not try to import numpy from its
  source directory`。同一条命令，不钉线程 → ImportError；钉了 →
  正常（`OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_MAX_THREADS=1`）。

## 2. 启动参数

- **不要显式传 `--rollout-top-k -1`**（负数作为独立 token 会让 parser 产生空字符串 key，
  MindSpeed 把它注入 `TransformerConfig.__annotations__`，dataclasses 生成非法 `__init__`，
  进程死在 import）。要全 softmax 就直接用 slime 默认值 `top_p=1.0 / top_k=-1`。
- 只训练 LoRA：`--only-train-params-name-list '.*\.lora_.*' --no-save-optim --no-save-rng`。
- 改完 `.py` 记得清 `__pycache__`：本 FS 的 mtime 不失效，Python 会加载旧字节码。
- **`--use-kl-loss` 必须同时给 `--ref-load <初始策略>`**。只给前者会在 `parse_args` 阶段
  `slime_validate_args` 里 `os.path.exists(None)` 崩掉（`TypeError: stat: path should be string...`），
  连模型都没建。它是 **actor loss 上的 KL**（`loss.py:908`），不是把 KL 塞进 reward——
  reward-KL 是另一个开关 `--kl-coef`，两者别混。参考模型 = 初始策略时 `--ref-load` 给 base HF 目录即可。
- **eval-only（`--num-rollout 0`）必须补 `--lr-decay-iters 1`**：slime 由
  `train_iters = num_rollout * rollout_bs * G // GBS` 推出 lr 衰减步数，为 0 时 Megatron 在
  actor init 阶段 `assert self.lr_decay_steps > 0` 直接中止（`_common_v2.sh` 里早有同款处理）。
- **`--clip-grad` 记的是 pre-clip norm**：Megatron 的 `clip_grad_by_total_norm_fp32` 先算 `total_norm`
  再按 `min(1, max_norm/total_norm)` 缩放，`optimizer.step()` 返回的是**缩放前**那个值
  ⇒ `train/grad_norm` 是 preclip，postclip = `min(preclip, max_norm)`。
  判断"裁剪是否在正常保护"要看**触发率**：2026-10-01 的 code run 用 `--clip-grad 0.1`
  而 grad_norm 长期 ~0.17 ⇒ **99/100 步都在砍梯度**，那不是保护而是持续压制；
  改成 1.0 后同一配方 0/100 步触发。**别把"每步都触发"当成 matched setting 留着。**

## 3. 自定义 loss

- `batch["loss_masks"]` 在 loss 时刻是**变长 list**（每样本一个 1-D 张量），**不要 `torch.stack`**
  （NPU 上报 `aclnnStack ... dim 0 ... [768] vs [952]`）。只做 `sum()` 和逐样本切片。
- "正确性守卫"的容差要取**量的绝对尺度**，不要拿 per-batch 统计量做相对阈值：
  曾用 `-1e-3 * max(|d|)` 守 `D = ||C||² >= 0`，而整块 D 都趋近 0 时等于拿舍入噪声比噪声，
  `D=-7e-10` 把一条跑到 14 步的健康 run 判死。
- CV 滞后状态（b/λ/N/D）会在 resume 时从 `<run_dir>/cv_vnext_state.json` 恢复，
  但**仅当**同目录存在 `latest_checkpointed_iteration.txt`（全新 run 仍用规定初值）。

## 4. 续跑 ≠ 等价重跑

- checkpoint 是 `--no-save-optim` 写的 ⇒ **Adam 动量不落盘，resume 后归零**，续跑段轨迹与
  连续训练不同。这是 harness 缺陷（LoRA 的 DistributedOptimizer `sharded_state_dict` 无法与模型树
  匹配，存盘会崩），不是配置选择。要真正等价得把优化器状态单独落盘。
- 数据序由 `random.seed(seed + epoch)` **绝对重播种** ⇒ resume 会**重放同一页样本**。

## 5. 编排：一次只能有一个 launcher

- **重启训练前先停 `wedge_watchdog.sh`**，等训练日志写出新内容（age < 2 min）**再启动 watchdog**。
  否则 watchdog 判定 "active.env 存在 + 日志 stale ≥20 min + train.py 存活" 会**再拉一个 launcher**，
  两个 `run_math_opsd_slime_npu.sh` 各自 `ray stop --force` 互相拆台 ⇒ 两个 train.py 抢同一批卡、
  引擎全停，**表现为又一次"静默卡死"**（2026-09-23 实际发生）。
- 判"在不在跑"用 `ps -eo pid,ppid,cmd | grep ...`，**不要用 `pgrep -fc`**：它会自我匹配，
  计数虚高（实测把 1 个 launcher 数成 4 个）。必须用 `pgrep` 时给模式加锚点，如 `'run_x[.]sh'`。
- campaign 的 `flock` fd 会被子进程继承（bash 无 CLOEXEC）⇒ 长任务子进程用 `( exec 8>&-; ... )`。
- **"watchdog 在跑"不等于"watchdog 有效"**：它会卡在 `do_wait` 等自己拉起的 resume 子进程，
  此后再也不 poll（实测卡了 2.5 天）。判断要看 `logs/wedge_watchdog.log` 有没有近期活动。
- **训练路径只导出 `engines[0]` ⇒ 整步 rollout 被钉在一个引擎上**（2026-10-01 实测）：
  `RolloutManager.generate` 只写 `OPSD_ENGINE_URL = engines[0]`，而 eval 路径会写全部
  `OPSD_ENGINE_URLS`；`sglang_rollout._pick_engine_url` 只有在后者存在时才 round-robin。
  症状：一步 64 条序列**全打到同一个 pid**（日志里单个引擎 `#running-req: 64`，另外 3 个引擎
  一行 Decode 都没有），rollout 吞吐 ~213 tok/s；补上多引擎导出后 4 个引擎各 16 条、~1300 tok/s（**约 4×**）。
  凡是自定义 rollout 克隆 `math_opsd/grpo_rollout.py` 的都吃这个亏（它走的就是单 URL）。
  注意 router（smg）的 h2c 探测与 sglang HTTP/1.1 server 不兼容，必须绕过 router 直连引擎。

## 6. 卡住了怎么办（按顺序）

1. **先取证，别猜**：`py-spy dump --pid <pid>`（`/opt/conda/envs/PyTorch-2.9.0/bin/py-spy`）。
   注意 `train.py` 通常只是在 `ray.get(...)`，真正干活的在 **RolloutManager / MegatronTrainRayActor**，
   要 dump 那一个。
2. **不要默认是 SGLang**。空转引擎的 `sglang::schedul` 本来就会 100%+ CPU 自旋，是红鲱鱼。
3. 确认是真死锁后才恢复：
   `kill <train.py PID>` → `pkill -9 -f 'sglang[:]:sched'`/`'[:]:detok'`/`'[.]launch_serve'`
   → `ray stop --force` → 确认 `npu-smi | grep -c 'No running processes found'` 为 8
   → `bash scripts/resume_training.sh <stage>`。

## 7. ALFWorld v2 harness（2026-09-30 ~ 10-03 实测，2 节点 × 8 卡 64G）

改任何损失/采样配置前先读这一节；下面每条都有实测数据支撑。

- **归一化契约**（先读这条，改任何 loss 前都要知道）：`--calculate-per-token-loss` 下 slime 传给
  自定义 loss 的 `sum_of_sample_mean` **实际是 `sum_of_token`**（`cp_utils.py:120`），即**应返回 token
  求和**；唯一的归一化在 `finalize_model_grads.py:492`（`all_reduce(num_tokens, dp_cp_group)` 后取倒数）。
  所以自定义 loss 里**不要再自己做 collective**——框架会跨 microbatch/DP 求和，重复 all_reduce 会按
  DP 倍数放大。
- **真正的优化器 bug：一行 = 一个 turn，每行等权 ⇒ 轨迹长度泄漏进优化器权重**（2026-10-02 定因）。
  每条轨迹被拆成"每回合一行"、每行都携带该 episode 的 GRPO advantage，于是**回合多的轨迹声音更大**。
  实测一个正常 8B rollout：**36 行正 vs 572 行负（15.9:1）**，梯度 94% 在说"压低这些 token"，而最省力
  的服从方式就是**少说/乱说** ⇒ `</think>` 结构被毁 ⇒ 动作解析不出 ⇒ reward 恒 0 ⇒ 组内零方差
  ⇒ `grad_norm ≡ 0` 冻死（`trajectory_mean_reduce` 本该纠偏，但在 `MBS=1` 下每行各自成"一集"，惰性）。
- **修法**：`OPSD_V2_LOSS_NORM=traj`（**现为默认**，`legacy` 保留给历史复现）。三件事缺一不可：
  ① `train_data.trajectory_weighting()` **在 DP 切分之前**算 `w_it = N_it/N_i`（Σ_t = 1）与 step 常数
  `scale = N_global/(T·G)`——切分后没有任何 rank 能看到整条轨迹；② `loss.trajectory_balanced_reduce()`
  按 `w` 加权**每行的 token 均值**（*不要*加权逐 token loss，那等于把长度从后门放回来）；③ 白名单三处都要加：
  `ray/rollout.py`(DP 切分)、`model.py`(padding)、**以及 `megatron_utils/data.py` 的指标跳过表**。
- **白名单第三处漏了会炸得很贵**：`log_rollout_data` 对**不在跳过表里**的每个 key 做 `sum(val)/len(val)`，
  list-of-lists 直接 `TypeError: unsupported operand type(s) for +: 'int' and 'list'`；它发生在
  **rollout 0 的末尾**（整个 rollout 已经付过钱），且 `train.py` 是 **rc=0 退出**、看不出崩了 ——
  实测两条臂 15:22/15:38 死掉、节点空转到次日 00:01，**白丢 9 小时**。
  回归测试：`tests/test_rollout_metric_contract.py`（重放那行归约 + 拿真实跳过表比对）。
- **旧策略来源退化 ⇒ PPO 裁剪从未生效**。v2 用 `log_probs.detach()`（当前前向）当 old，
  而 slime 契约是 `batch["log_probs"]`（旧 actor 前向）⇒ `ppo_kl ≡ 0`、ratio ≡ 1、
  `pg_clipfrac` 恒 0（两个 run 实测全程 0）。**任何依赖 ratio 的方法（GSPO）在此修好前都是空操作。**
- **GSPO 已实现且测过，但今天跑仍等于 GRPO**，原因同上。未修旧策略来源前不要起 GSPO run，
  也不要把它的数当方法差异。启用后看 `v2/gspo_degenerate_kl`。
- **`MAX_RESPONSE` 必须 ≥ 模型自然思考长度**。cap=1024 时 Base 就有 92.8% episode 被截断 ⇒
  `</think>`/`</action>` 生成不出 ⇒ reward 恒 0 ⇒ 梯度 0。cap=4096 下两臂 `action_valid` 才回到
  ~98%（判据：`action_valid > 80%`，**不要**用 `truncated_ratio`，它量的是 episode 撞回合上限）。
- **`--num-rollout` 是本次 run 的总步数，不是增量**。resume 时它必须 > `latest_checkpointed_iteration.txt`，
  否则 init + 下发权重走完后**rc=0 干净退出且一步 rollout 都不跑**（唯一特征是日志里 `perf N:` 为 0）。
- **v2 的 `rollout_log_probs` 是 `[0.0]*rlen` 占位符**，`teacher_log_probs`/`privileged_log_probs` 为 None
  ⇒ **没有可用的 logprob 诊断**（算不了 surprisal/KL）。要测就得开 `--use-rollout-logprobs`。
- **判存活用 `ps -eo pid,ppid,cmd | grep '[t]rain[.]py'`**；`pgrep -fc` 会自我匹配（见 §5）。
  两节点日志都在共享盘上：**直接读文件，不要 ssh**（本环境 ssh 返回通道会挂住）。
- **评测**：`alfworld_hpi_v2/scripts/eval_alfworld_ckpt.sh <run-dir> <iter>` 需设 `METHOD_SLUG`
  （脚本拒绝猜，猜错会把数静默挂到错误方法名下）；它用 mirror 目录钉住 iteration，并强制 grep
  `successfully loaded checkpoint ... at iteration N`，缺失即 exit 4 判该点作废。
- **run 目录命名**走 `scripts/run_names.py`（`--allocate`；`--check` 审计；`--plan/--apply` 处理遗留）。
  **正在被写的目录绝不改名**——共享盘上别的 pod 的活目录同理。
- 诊断脚本在 `reports/analysis_scripts/`：`grpo_curve.py`（训练曲线）、`act_valid.py`（合法动作率）、
  `length_by_outcome.py`（按结果分层的长度/絮语度）、`afw_eval_rates.py`（评测复算）、`rew_check.py`（reward 一致性）。

### 7.1 三尺寸 GRPO campaign（2026-10-01 起，两节点）

- **当前配置（2026-10-03 起）**：`GBS=32 / lr=5e-6 / MAX_RESPONSE=4096 / ROLLOUT_BS=4 / G=8 /
  MAX_TURNS=50 / HISTORY_LENGTH=8 / SEQ_LENGTH=8192 / NUM_ROLLOUT=50 / SAVE_INTERVAL=25`，
  训练必须带 `OPSD_V2_LOSS_NORM=traj`（`_common_v2.sh` 已设为默认）。
- **`HISTORY_LENGTH` 曾经是静默空操作**：`_common_v2.sh` 一直导出 `ALFWORLD_V2_HISTORY`，但 rollout 读的是
  `config.py` 的硬编码常量（只有 diagnostic 读 env）⇒ `HISTORY_LENGTH=8` 只改横幅。现在 config 真读 env 了。
  同类：**`--seq-length` 从不传时框架默认 4096**，而 H=8 的 prompt p99.9 已达 3517、加 cap 4096 约 7.6k，
  实测已有 5008 token 的行在跑（没报错但超出配置）⇒ 必须显式 `SEQ_LENGTH=8192`。
- **协议改动的收益（只改协议、不动优化器）**：Base 8B `0.1286/0.0970 → 0.2071/0.2090`，
  4B `0.1387 → 0.1643`（20 回合/H=2 → 50 回合/H=8）。回合上限与 2 步历史对绝对分的压制比优化器 bug 更大。
  **代价**：rollout ≈50 min/个 ⇒ 50 rollout ≈42 h/臂，每次 cp 评测 ≈7 h（Base 评测实测 7h11m）。
- **启动**：`bash alfworld_hpi_v2/scripts/queue_grpo_sizes.sh <0|1>`，role0=8B→1.7B，role1=4B。
  幂等：name 与标记在 `/tmp/afw_grpo_fixed_role<N>/`；`TRAIN_OK_*`/`EVAL_OK_*` 兼作断点；
  训练中途死掉重跑同一命令即可续（只占一个 launcher）。
  **换 state 目录而不是删标记**：直接复用旧 state 目录会让 `TRAIN_OK_*` 让队列跳过训练。
- **8B 已实测可行**：PP1/TP1/DP4、`mem 37.4/61 GB`、21.7 min/rollout。**不要**照抄 math 卡的 PP4
  （那是为了给小卡上放 teacher）；`_8b_backbone.sh` 设 `UNTIE_EMBEDDINGS=1`，漏了会静默建错模块树。
- **`truncated_ratio` 不是 token 截断**：cap=4096 下 8B 报 0.985 而 `response_len/max` 仅 2734，
  它量的是撞 `--max-turns 20`。判据看 `action_valid`，别去"修" cap。
- **判死**：最新 run 日志 30 min 无更新且无 `train.py` ⇒ 重启该节点队列。
- **`pkill -f` 的自匹配有个新变体**：`[x]` 括号技巧只保护**模式**，不保护**命令行里的其他字面量**。
  `pkill -f '[c]ollapse_guard'` 仍会杀掉自己——因为同一行里出现了 `.../scripts/collapse_guard.sh`
  这个真实路径。实测把自己杀了（exit 144）。做法：把脚本名拼出来（`G='collapse''_guard'`），
  命令行里不要出现目标字符串本身。
- **写守护脚本时先想"它会不会误杀自己守护的东西"**：第一版 collapse guard 用
  `n=$(grep -c ... || echo 0)`，无匹配时得到 `0\n0`，数值比较报错后**穿透**到检测分支，而空输入让
  awk 判定恒真 ⇒ 会在 rollout 0 就杀掉健康的 run。修法：每个指标先 `tr -dc '0-9.'` 清洗，并要求
  整个窗口都是合法数字才算"证据充分"。
- **队列脚本自身也会死**，且死在最贵的位置（跑完一整条臂、正要评 Base 时）。真因是 `set -u` 的
  another 变体：`local a="$1" b="x$a"` **不成立**——bash 在执行 `local` 前就展开了整条命令的词，
  `$a` 读的是**调用者**作用域。在函数内被调用时外层恰好有同名变量，看起来能跑；从顶层调用就
  `unbound variable` 中止。**每个变量单独一行 `local`**，别写自引用。
- **改脚本救不了已在跑的实例**：bash 在**定义时**就解析函数体并驻留内存，所以对 `queue_*.sh` 的
  修复只对**之后启动**的进程生效；在跑的那个仍执行旧代码。且编辑会移动文件偏移，正在执行的
  bash 可能从错位处继续解析。**改完必须重启对端队列**（幂等，标记会让它跳过已完成的 cell）。
  实测：修了 node0 的 bug 后，node1 的旧实例照旧死在 `base_eval`（跑完 1.7B 全部训练+评测之后，最贵的位置）。
- **重启沿用同一 run 名时，日志是追加的** ⇒ 任何"扫日志找报错"的脚本必须带**行号或时间戳偏移**
  （`START=$(wc -l < log); tail -n +$START`），否则会把历史故障当成现况（实测把 15:22 的旧崩溃报成
  00:16 的新错误）。同理，SGLang 的 `Can not initialize OpenAIServingResponses ... Traceback`
  是启动噪音，必须从错误判定里排除。
- **杀掉一个正在跑的 eval 会让脚本写出假的 `EVAL_OK`**：`eval_alfworld_ckpt.sh` 只要 grep 到
  `successfully loaded checkpoint` 就 VERIFIED，即使评测只跑了 18 分钟被中途 kill，返回码仍是 0。
  用 kill 中止评测后，记得**清掉对应的 `EVAL_OK_*` 标记与那个半成品 run 目录**，否则后续会静默跳过该评测。
- **两节点负载要按实测速率配平**（2026-10-03）：8B ≈54 min/rollout、4B/1.7B ≈42 min。原排法
  role0=8B+1.7B(≈80h)、role1=4B(≈35h) 会让 node1 空转 45h。要改分配**不能在跑的时候重启队列**
  （会再起一个训练撞车）；正确做法是等短的那条跑完、队列自然退出后再改，并用**往对方 state 目录写
  `TRAIN_OK_<size>` 标记**的方式让在跑的队列自己跳过该 cell。
- 诊断与证据：`reports/scaling_chain_fix.md`、`reports/grpo_collapse_postfix_2026-10-01.md`、
  `reports/fixed_harness_status.md`。

## 8. 评测

- **必须吃满 8 卡**，并核对进程真实参数
  `pgrep -af 'slime-ascend/train.py' | grep -oE '--rollout-num-gpus [0-9]+'`。
- 每格硬校验 dump 样本数 `== 30 * K`，不符即中止（K 没被 config 生效过）。
- 只取 `effective == original` 的那份 `results_*.json`；同目录可能并存多份（含残片）。
- **不要用训练 loop 跑评测**（2026-10-02）。`--num-rollout N` 是"rollout + 一次 policy 前反向"，
  一条长评测样本（prompt + 8192 回答）的 fp32 logits 就要 ~4.9 GiB，actor 直接 OOM；
  而 eval 路径 policy 根本不参与前向。eval-only 用
  `--num-rollout 0 --eval-interval 1 --eval-config <json>`，并配
  `--custom-generate-function-path` 让 prompt 仍由我们的 builder 构造（`--apply-chat-template` 不开，
  `sample.prompt` 才是原始题面）。§2 的 `--lr-decay-iters 1` 别忘了。
- **加载 LoRA checkpoint 必须 `--no-load-optim --no-load-rng`**：LoRA 用 `--no-save-optim` 存，
  checkpoint 里没有 optimizer 段，不带这两个参数会在 actor init 时 `KeyError: 'optimizer'`。
- **改 checkpoint 目录里的 iteration 用 mirror，不要动原 run**：`scripts/make_eval_staging.sh
  <run_dir> <iter> <staging>` 把 `iter_N` 软链成 staging 的 `latest_checkpointed_iteration.txt`，
  并从 load 路径**推导** `EVAL_LORA`（有 `latest_checkpointed_iteration.txt` ⇒ 1，有 `config.json` ⇒ 0）。
  推错/漏推的后果是**拿 base 权重冒充 checkpoint 出数**，而且数字看着完全合理。

## 9. Code RLVR（单轮代码生成 + 可执行测试奖励，2026-10-01~03）

代码在 `code_grpo/`（插件式，复用 math GRPO 的 loss/advantage/LoRA/SGLang 旋钮）。
这一节按"花了多少代价才搞明白"排序，前两条是**一次跑废一整轮 100 步**的教训。

- **训练任务的接口形态必须和评测一致，否则你优化的是另一个目标。** 第一版训练集是
  **100% stdin/stdout** 题，且每道题都追加同一句"read from stdin / write to stdout"。
  跑了 100 步 rc=0、训练 reward 0.129→0.455 看着很好，但 LiveCodeBench
  **functional 段（444 题）从 Base 0.2252 崩到 0.1171**；诊断显示模型开始把答案写成
  "先给 `class Solution`，再给一段 `sol = Solution(); print(...)` 的 driver"，
  而官方 `extract_code` 取**最后一个** fenced block ⇒ 抽到 driver ⇒ 判错（cp100 有 182/444 是这种，
  Base 只有 67/444）。改成 60/40 异构数据 + task-aware prompt 后，functional **0.2635**（还高于 Base）。
  **判据**：这类崩不会在训练指标上露头，只有按题型拆开看评测才发现。
- **自定义载荷必须放进行内的 `metadata` 对象里**。slime 的 `Dataset` 用 `data.get(metadata_key)`
  取元数据，启动器漏传 `--metadata-key metadata`（或把字段写在同一层）会拿到空 `{}`，
  RM 拿不到 `tests` ⇒ **全 0 reward、分片逐字节相同**，而整条 run 仍然"成功"跑完。
  现在 `rm_code` 在缺 `tests` 时**直接 raise**，早死 12 分钟好过白跑几小时。
  eval-only 场景用 `CODE_GRPO_ALLOW_MISSING_TESTS=1` 显式放行（LCB 的私有测试有好几个 GB，
  不塞进 prompt 文件）。
- **Qwen3-1.7B 在这类题上必须 thinking OFF。** thinking ON 时模型把整个预算耗在 `<think>` 里：
  cap=4096 下 75% 截断、95% 抽不出代码；cap=2048 时 64 条里只有 5 条出现 `</think>`。
  OFF 后同配方 reward 0.11→0.13、mixed_group 0.5、截断 0。（`CODE_GRPO_THINKING=1` 可复现原测量。）
- **参考解自证是 verifier 最划算的体检**：拿数据集自带的 reference solution 过一遍训练 verifier，
  只保留能拿满分的题。stdio 2500 候选里 2132 过（85.3%），TACO call_based 1045/1049 过。
  抽取/比较/编码只要有一样错了，这条通过率会直接塌下来——比任何人工抽查都灵敏。
  注意 **TACO 的 reference solution 是无围栏裸代码**，官方 extractor 读不了 ⇒ 只有**参考解校验**这条
  路径需要"裸代码也接受"的 fallback；**模型输出永远只走官方规则**。
- **TACO call_based 的返回值被包成单元素列表**，而 LCB 的 `grade_call_based` 直接
  `method(*args) == json.loads(output)`。不拆包的话 5/5 参考解全 `WRONG_ANSWER 0.00`，
  拆包后 5/5 `PASS 1.00`。
- **奖励用通过率 fraction 而不是 0/1**：1.7B 在竞赛题上大量只得部分分，二值奖励会把 G=8 的组
  几乎全摁成 0 ⇒ advantage 全 0 ⇒ 白跑。`exact_pass` 单独记录即可。
- **异构数据把单步时间缩短了约 4 倍**（同一 GBS/G/步数）：v1（纯 stdio）8h28m/100 步，
  v2（60/40）2h15m —— call_based 的回答明显更短。排期时别按 v1 的速度估。
- **别把"训练指标变好"当成方法有效**：v2 训练侧 reward 反而是平的（0.195→0.174），
  但评测 +0.0284 且 functional 崩盘消失。**以官方评测为准，训练曲线只用来判断健不健康。**
