# ALFWORLD_HPI_STATUS

Session: 2026-09-24 11:05 → 14:35 CST. Pod `5acbcf28-1179-477f-a014-3de133a3cb16`
(host `notebook-d03f9031-…`), the OPSD baseline node.
Harness: `$PROJ/alfworld_hpi/` (new package, 5.5k lines, **zero pre-existing files
modified**). Full design notes in `alfworld_hpi/README.md`.

---

## A. 原有实验

**全程未被干扰。** 本次会话是纯 CPU/文件系统工作：没有启动任何 SGLang / Megatron /
torch_npu 进程，没有 kill 任何进程，没有重启 Ray / HCCL / 机器，没有清理 /tmp 或共享内存，
没有 pip install 到 `cua-rl`，没有抢占 NPU。

只读审计（会话开始 11:05:40）：

```
PAAS_POD_ID = 5acbcf28-1179-477f-a014-3de133a3cb16   (OPSD baseline worker)
npu-smi     = 8 x 910B4, all 8 occupied
              NPU0-7: raySGLangEngine / sglangscheduler
              HBM 3.2-28.0 GB per die
运行中实验  = PID 2012269  train_scripts/run_dev_eval.sh \
                runs/eval_staging/cvvnext_clean_cp100_99  cvvnext_clean_cp100_aime25
              PID 2018953  frameworks/slime-ascend/train.py (eval-only, --num-rollout 0)
              (另有 PID 747451 prompt_b_orchestrator watchdog)
```

会话结束时（14:29）：

```
PID 2478394  run_dev_eval.sh → runs/eval_staging/cvonly_clean_cp100_99  cvonly_clean_cp100_aime25
ps -ef | grep -c train.py = 3      (eval 进程仍在跑)
npu-smi: 全部 8 张仍有 sglangscheduler / raySGLangEngine 占用
```

即 **原实验未自然结束，NPU 从未 idle**，所以按约束 **未进入 NPU smoke 阶段**。
中间 `cvvnext_clean_cp100_aime25` 那一轮在 12:31 自行结束时，eval queue 立即接续了
`cvonly_clean_cp100_aime25` —— 这是编排层自己的行为，与本任务无关。

写操作足迹（`find . -newermt "5 hours ago"` 排除 alfworld_hpi / runs / logs）：

```
./.pytest_cache/v/cache/nodeids     <- 本会话 pytest 产生的缓存
./scripts/serverA_stage.sh          <- 编排层（非本会话）
./scripts/serverA_stages.list       <- 编排层
./reports/dev_eval_table.md         <- 正在跑的 eval 写入
```

`git status` 中 17 个 ` M` 文件的 mtime 全部早于本会话开始。
**本任务新增文件全部在 `alfworld_hpi/` 内。**

---

## B. Environment

| 项 | 值 |
|---|---|
| ALFWorld | `alfworld==0.4.2` (PyPI wheel) |
| TextWorld | `textworld==1.7.0` (源码构建，见下) |
| jericho | `jericho==3.3.1` (C++ 扩展，本机 g++ 10.3.1 编译通过) |
| PDDL 后端 | `fast-downward-textworld==20.6.4` |
| 数据 | `json_2.1.3_tw-pddl.zip` (release 0.4.2, 36 MB, wget) |
| data path | `$CUA/data/alfworld/json_2.1.1/` |
| train | **3553** games |
| valid_seen | **140** games |
| valid_unseen | **134** games |
| eval 类别覆盖 | Pick 59 / Look 31 / Clean 58 / Heat 39 / Cool 46 / Pick2 41 |
| 依赖隔离 | overlay venv `$CUA/envs/alfworld_overlay`（`--system-site-packages`，从 cua-rl 派生） |
| CPU smoke | **PASS**（`SMOKE_ALFWORLD_CPU = PASS`，10/10 项） |

### 三个非显然的坑（都已解决并写进 setup 脚本）

1. **官方 tw-pddl 数据没有 `traj_data.json`。**
   `AlfredTWEnv.collect_game_files()` 要求每个 `game.tw-pddl` 旁边有 `traj_data.json`，
   否则**静默返回 0 个游戏**（不报错）。本机实测 `Overall we have 0 games in split=train`。
   → 不走 AlfredTWEnv，改为直接用 `textworld.gym.register_games` 注册游戏文件，
   复用 ALFWorld 自己的 `AlfredDemangler` / `AlfredInfos` wrapper，行为与参考实现一致。
   顺带好处：可以按 game 寻址（训练/评测需要指定具体任务）。

2. **textworld 的 sdist 在构建期 curl Inform7 编译器**（`thirdparty/setup.sh`，
   emshort.com，本机不可达）。ALFWorld 的 tw-pddl 游戏是**预编译**的，Inform7 只用于
   编译新游戏，运行时不需要。→ setup 脚本对 setup.py 打补丁跳过 `setup.sh`
   （补丁前会断言目标字符串仍在，不会盲改）。

3. **`ALFWORLD_DATA` 被 alfworld 包在 import 时覆写**成 `~/.cache/alfworld`（空目录）。
   任何"晚一点再读环境变量"的写法都会静默把所有游戏路径指向空缓存。
   → `paths.py` 在模块导入时（任何 alfworld import 之前）就把数据根**钉住**，
   并提供 `ALFWORLD_HPI_DATA` 作为本 harness 自己的覆盖变量。

### 依赖隔离方式（为什么不改 cua-rl）

overlay 的 site-packages 是 **append** 到 `sys.path` 的，不是 prepend。后面的条目只能
**补充**解释器找不到的模块，永远不能**替换**已经解析到的。实测 overlay 与 cua-rl 的
顶层包名交集只有 `pip / setuptools / pkg_resources / _distutils_hack`，全部不在训练路径上；
`test_overlay_shares_no_module_names_with_the_base_environment` 会在这条断言被破坏时失败。
训练核心 100% 留在 cua-rl + slime。**没有启用 RPC fallback**（不需要）。

---

## C. Implementation

### 文件列表（全部新增，无修改）

| 文件 | 作用 |
|---|---|
| `alfworld_hpi/paths.py` | overlay/sys.path 隔离、数据根解析 |
| `alfworld_hpi/env_adapter.py` | textworld.gym 注册、reset/step、task 提取 |
| `alfworld_hpi/prompts.py` | system prompt + 每轮 transcript（不含过去 CoT） |
| `alfworld_hpi/action_parser.py` | 全函数化的 `<think>/<action>` 解析器 |
| `alfworld_hpi/trajectory.py` | `TurnRecord` / `Trajectory`，JSONL 无损往返 |
| `alfworld_hpi/privilege.py` | `build_hpi_privilege` + 泄漏断言 |
| `alfworld_hpi/rollout.py` | slime rollout + eval，per-turn segment layout |
| `alfworld_hpi/reward.py` | 仅 terminal success |
| `alfworld_hpi/train_data.py` | convert + layout 校验 |
| `alfworld_hpi/loss.py` | HPI 加权的 GRPO |
| `alfworld_hpi/eval.py` | 按 split / category 聚合 |
| `alfworld_hpi/report_eval.py` | 聚合 eval dump → 表格 |
| `alfworld_hpi/check_gates.py` | NPU smoke gate 的 PASS/FAIL 判定 |
| `alfworld_hpi/data_prepare.py` | game index、prompt 数据集、eval config |
| `alfworld_hpi/config.py` | env-var 配置解析（唯一来源） |
| `scripts/` (10 个) | setup / CPU smoke / 3 个 NPU smoke / 4 个 run entrypoint + `_common.sh` |
| `tests/` (6 个文件) | 107 CPU tests（+2 条件 skip） |

### slime hook

| 用途 | flag |
|---|---|
| rollout | `--rollout-function-path alfworld_hpi.rollout.generate_rollout` |
| eval | 同一个 callable（`evaluation=True`） |
| 数据转换 | `--custom-convert-samples-to-train-data-path alfworld_hpi.train_data.convert_samples_to_train_data` |
| loss（仅 HPI 两臂） | `--custom-loss-function-path alfworld_hpi.loss.alfworld_hpi_loss` |

**没有升级 slime，没有修改 vendored slime 一个字节。**
`--custom-generate-function-path` 被否决：它只从 *stock* sglang rollout 可达，
那条路径还会接管 reward 与数据转换，会和本 harness 的 convert/loss 打架。

逐轮 view 走 `hpi_views` / `hpi_prompt_lengths` / `hpi_target_lengths` —— 这三个 key
`_split_train_data_by_dp` **已经在白名单里**（math HPI 时期加的）；turn→span 映射走
`turn_ids`。所以**不需要新增任何 train-data key**，这是不改 slime 的关键。

### trajectory representation

`Trajectory` / `TurnRecord` 字段严格按需求 #12；`from_dict(to_dict())` 无损
（有测试）。每轮额外保存 `prompt_token_ids`（发给引擎的**原样** id）与
`generated_token_ids`（引擎返回的**原样** id）—— 这两个是 token 对齐的载体。

### action parser

6 种显式 status：`OK / NO_ACTION / EMPTY_ACTION / NOT_ADMISSIBLE / TRUNCATED`。
永不抛异常（有 hostile-input 测试）。adm 判定用**归一化精确匹配**（小写 + 折叠空白 +
去首尾标点），不做模糊匹配 —— 环境只接受 verbatim 命令，模糊匹配会与"环境实际执行了什么"脱节。
`<action>` 取**最后一个非空**块（模型有时先吐一个空/废弃 tag）。

### loss mask

```
tokens    = prompt_0 | y_0 | prompt_1 | y_1 | prompt_2 | y_2 | ...
loss_mask =   0 ...  |  1  |   0 ...  |  1  |   0 ...  |  1  | ...
turn_ids  =  -1 ...  |  0  |  -1 ...  |  1  |  -1 ...  |  2  | ...
```

`loss_mask==1` **当且仅当**该 token 由模型生成。prompt / observation / admissible list /
privileged context 一律 0。`loss_mask == 1 ⟺ turn_ids >= 0` 在 `validate_layout`
与 loss 两处断言。

### future privilege builder

`build_hpi_privilege(trajectory, turn_idx, mode, future_gap)`，mode ∈
`{none, outcome_only, future_outcome}`，格式见需求 #9，实际渲染样例：

```
<privileged_hindsight>
This information is available only during training. It was observed AFTER the action
being evaluated and is not part of the conversation.

Final episode outcome:
FAILURE

Delayed future interaction trace:
[Future step 1] (turn 2)
Observation: obs2
Executed action: go to bed 1

[Future step 2] (turn 3)
Observation: obs3
Executed action: look

Final environment observation:
obs4
</privileged_hindsight>
```

privilege block 放在 user turn 的**末尾**（紧邻 generation slot）—— 因果 LM 只能
condition 在被评分 token **之前**的内容，放后面等于没放。用 `user` role 而非 `tool`，
因为 tool 消息必须紧跟 assistant 的 tool_call，而本 scaffold 不产生 tool_call。

### future_gap 精确语义

- 轮次 0-based，`t = 0..T-1`；轮 `t` 产出 `think_t + action_t`。
- **transition `i` 定义为 `(action_i, observation_{i+1})`** —— 即第 `i` 轮动作的环境结果。
- `future_gap = g` ⇒ privilege 只含 transition `i >= t + g`。

| gap | 被扣住 | 纳入 |
|---|---|---|
| `2`（默认） | transition `t`（本动作的直接结果 `o_{t+1}`）与 `t+1` | `t+2 …` |
| `1` | transition `t` | `t+1 …` |
| `0` | 无 —— **泄题**，仅消融 | `t …` |

terminal outcome **任何 gap 都保留**。有 `test_gap_zero_index_is_not_off_by_one`
显式扫描每个 turn 的窗口边界。

---

## D. Thinking

- chat template **`enable_thinking=True`**，显式传参，不依赖模板默认值
  （`prompts.render_chat`），四个 arm 完全一致。
- **实测 Qwen3 模板行为（重要）**：`enable_thinking=True` 时不做空 think 预填，
  generation prompt 结束于 `<|im_start|>assistant\n`，模型自己写 `<think>…</think>`。
  这与本 harness 的 parser 和"history 不含过去 CoT"的假设一致。
- **实测第二个模板行为**：Qwen3 模板会从 assistant history 中**剥掉 `<think>` 块**，
  只保留 `</think>` 之后的内容。这正是我们在 prompt 层面想要的（需求 #6），但它的
  直接后果是 **`h_{t+1}` 不是 `h_t + y_t` 的 token 级延拓**，因此不能用一条链式
  causal 序列训练（详见 §C 的 layout）。有测试钉住这个行为：若未来 transformers
  不再剥离，测试会失败而不是静默改变训练分布。

真实 turn 示例（Qwen3-1.7B tokenizer 实际渲染）：

```
<|im_start|>system
You are an expert agent operating in the ALFWorld text environment.
...
1. Think about the current situation inside <think>...</think>.
2. Then choose exactly ONE action from the admissible actions and put it inside <action>...</action>.
...
<|im_end|>
<|im_start|>user
Task: look at alarmclock under the desklamp.

Current observation:
You are in the middle of a room. Looking quickly around you, you see a bed 2, a bed 1,
a desk 1, a drawer 11, ..., and a sidetable 1.

Your task is to: look at alarmclock under the desklamp.

Admissible actions:
  - go to bed 1
  - go to desk 1
  ...
  - look

Choose exactly one admissible action.
<|im_end|>
<|im_start|>assistant
```

| 检查 | 结果 |
|---|---|
| past CoT 是否从 history 删除 | **PASS**（构造上只渲染 past ACTIONS；`assert_no_reasoning_in_history` 断言；tokenizer 层另有测试） |
| future CoT 是否从 privilege 删除 | **PASS**（builder 结构上读不到 `generated_text`/`think_text`；渲染结果另行扫描 `<think>`/`</think>`/`reasoning_content`） |

---

## E. Leakage checks

| 检查 | 结果 | 依据 |
|---|---|---|
| current response 不在 privilege 内 | **PASS** | `assert_no_current_response_leak`：完整 generated_text、think_text（≥8 字符）、以及 `Executed action: {a_t}` 三种形态都检查；8 轮×3 种 gap 全扫描测试 |
| immediate transition gap | **PASS** | gap=2 时 `(turn t)`/`(turn t+1)` 均不出现，`(turn t+2)` 起出现；`assert_privilege_isolated` 在 gap 窗口内命中即 raise |
| outcome-only construction | **PASS** | 只含 `Final episode outcome:`，不含任何 observation / action / trace；对任意 turn 渲染结果逐字节相同 |
| token target equality | **PASS** | 两层：`train_data.validate_layout`（convert 期）与 `loss.alfworld_hpi_loss`（每次 forward 前）都断言 `view[plen:] == response_tokens[start:end]`；有"故意位移一位必须被抓到"的测试 |
| privilege 中无未来 reasoning | **PASS** | `_assert_no_reasoning` + 逐轮扫描测试 |
| outcome-only vs future 只差 trace | **PASS** | 测试断言两臂 outcome 块逐字节相同、只有 future 臂含 trace；rollout 级测试断言两臂 control prompt **完全相同**、full prompt 严格更长 |
| alpha=0 退化为纯 GRPO | **PASS** | `w == 1` 精确成立（`torch.allclose`），不是"近似" |

---

## F. NPU smoke

**未执行。** 原实验全程占用 8 张 NPU，从未 idle，按约束 (§0) 不得启动任何
SGLang / Megatron / torch_npu workload。

已就绪、等 NPU 空闲即可按序执行的脚本：

| Gate | 脚本 | 判定 |
|---|---|---|
| A + B | `alfworld_hpi/scripts/smoke_alfworld_rollout_1p7b.sh` | 12 局（每 split 每类别 1 局），记录 success / turns / invalid / tokens / wall time |
| C | `alfworld_hpi/scripts/smoke_alfworld_train_grpo_1p7b.sh` | 1 task × G=8 × 1 step |
| D | `alfworld_hpi/scripts/smoke_alfworld_train_hpi_1p7b.sh` | 1 step，同时是 `ALFWORLD_HPI_SCALE` 的标定 run |

`alfworld_hpi/check_gates.py --log <log> --gate C|D` 给出逐条 PASS/FAIL。
已在合成日志上验证过它**会正确地失败**：zeros-grad 的退化 GRPO 日志 → FAIL，
alignment-violation 日志 → FAIL，正常日志 → PASS。

peak HBM 与 step wall time：**尚无数据**（未跑）。

---

## G. Ready-to-run commands

前置（一次性，已在本会话完成，重跑幂等）：

```bash
cd /workspace/cua/repos/opsd-slime
bash alfworld_hpi/scripts/setup_alfworld.sh
bash alfworld_hpi/scripts/smoke_alfworld_cpu.sh      # SMOKE_ALFWORLD_CPU = PASS
```

**Gate A/B（推理，先跑这个确认 SGLang + parser + env）**

```bash
bash alfworld_hpi/scripts/smoke_alfworld_rollout_1p7b.sh
python alfworld_hpi/check_gates.py --log logs/alfworld_smoke_rollout_1p7b.log --gate B
```

**Gate D 兼 scale 标定（必须早于两条 HPI 正式 run）**

```bash
ALFWORLD_HPI_SCALE=1.0 bash alfworld_hpi/scripts/smoke_alfworld_train_hpi_1p7b.sh
python alfworld_hpi/check_gates.py --log logs/alfworld_smoke_train_hpi_future_1p7b.log --gate D
grep -o "hpi/abs_e_p75[^,}]*" logs/alfworld_smoke_train_hpi_future_1p7b.log | tail -1
```

**A. Base eval**

```bash
bash alfworld_hpi/scripts/run_alfworld_base_eval_1p7b.sh
python alfworld_hpi/report_eval.py --dump 'runs/alfworld_base_eval_1p7b/*.pt' \
       --json reports/alfworld_base_eval.json
```

**B. GRPO**

```bash
bash alfworld_hpi/scripts/run_alfworld_grpo_1p7b.sh
```

**C. Outcome-only**（`ALFWORLD_HPI_SCALE` 换成 Gate D 标定值）

```bash
ALFWORLD_HPI_SCALE=<calibrated> bash alfworld_hpi/scripts/run_alfworld_hpi_outcome_only_1p7b.sh
```

**D. Future-HPI**（唯一与 C 不同的变量 = privileged block 里的 future trace）

```bash
ALFWORLD_HPI_SCALE=<calibrated> bash alfworld_hpi/scripts/run_alfworld_hpi_future_1p7b.sh
```

四者共用 `scripts/_common.sh`，因此在 model / prompt / thinking / sampling /
split / G / batch / optimizer / max turns / eval 上**由构造保证一致**，只有 METHOD
（及其蕴含的 loss/hook 选择）不同。没有 scale 时 HPI 启动器**拒绝启动**，不会静默
替用户决定干预强度。

---

## H. Blockers

1. **原实验仍在占用 8 张 NPU**（唯一真 blocker）。这不是工程问题：A–E 全部已完成，
   只等 NPU 自然空闲后按 Gate A → B → D(标定) → 正式 run 的顺序执行。
   不要为了本任务 kill 它。

2. **`ALFWORLD_HPI_SCALE` 尚未标定**（结构性依赖，非缺陷）：scale 是 HPI 权重的
   强度参数，必须用本模型/本任务实测的 `|e|` p75 定，不能用 math 任务的 `0.045126`
   顶替。Gate D 标定 run 会直接打印该值。启动器已强制此依赖。

3. **loss-forward 显存未实测**：per-turn segment layout 是 O(T²) 上下文，
   `max_turns=15` + 8 条 action history 时单局约 10–15k rows，Megatron 会物化
   `[rows, 151936]` logits。这是 Gate C 第一件要测的事（`npu-smi` peak HBM）。
   放不下就调小 `ALFWORLD_MAX_TURNS` / `ALFWORLD_ACTION_HISTORY`（两者都是 env var，
   不需要改代码）。**注意：GRPO 臂与 HPI 臂 layout 相同，所以这是四个 arm 共同的成本，
   不影响 arm 间可比性。**

4. **eval 路径未上机验证**：`evaluation=True` 分支与 `report_eval.py` 在 CPU 上用
   合成 dump 验证过，但没跑过真实引擎。Gate A/B 会覆盖它。

---

## READY_FOR_1P7B_ALFWORLD_FORMAL = NO

CPU 侧全部完成且已验证（107 tests + 10/10 smoke 项 PASS），但三个 NPU gate
（A/B、C、D）一次都没跑过，而它们正是"可信"的判据。按任务书的明文规定
（§19：只有 NPU 真正 idle 后才能继续；§20：仅当原任务自然结束**且**所有 gate PASS
才允许启动后续实验），现在开正式 run 是违规的。

**唯一下一步**：等当前 `cvonly_clean_cp100_aime25` eval 自然退出、`pgrep -f train.py`
为空、`npu-smi` 8 张均无进程并连续确认 idle 后，执行

```bash
cd /workspace/cua/repos/opsd-slime
bash alfworld_hpi/scripts/smoke_alfworld_rollout_1p7b.sh
python alfworld_hpi/check_gates.py --log logs/alfworld_smoke_rollout_1p7b.log --gate B
```

Gate B PASS 后再跑 Gate D 标定，然后才是 §G 的四个正式入口。

---

# 2026-09-25 上机实录（Gate A/B/C 已过，D 融入正式 run）

新服务器：ModelArts 双节点作业 `ma-job-ed05c1c3-…`，每节点 8×910B（29.49 GB 可用）。
跨节点通道 = `ssh -p 2222` + worker-1 内部 DNS 全限定名（`10.75.x.x` 是训练网络地址，
能 ping 但零监听端口，**不是** SSH 入口）。

## 修掉的 bug（每条都是实盘暴露）

1. **eval-only 缺 `--lr-decay-iters 1`** → `--num-rollout 0` 时 `lr_decay_iters` 回落 0，
   Megatron 在 actor init 撞 `assert self.lr_decay_steps > 0`。
2. **textworld 非线程安全（最关键）**：`textworld/envs/pddl/logic/_PARSER` 是 module 级
   tatsu 单例，带可变解析状态；rollout 用 `asyncio.to_thread` 并发建 env 会互相踩，
   抛 `tatsu.FailedToken` 且**报错位置随机漂移**（187:17 / 187:5 / 183:1 / 177:5 / 1:12）。
   实测：顺序加载 12/12 OK，12 线程并发 11~12/12 全挂。修法 = module 级 RLock 串行化
   construct/reset/step/close（`env_adapter._TEXTWORLD_LOCK`），已加回归测试。
   **单线程 CPU probe 永远测不出这个**。
3. **`SamplingParams` 字段名是 `sampling_seed` 不是 `seed`** → 引擎 400。
4. **动作被拒后无反馈** → 模型看到逐字节相同的 observation，无从自查，整局卡死；
   成功率恒 0。修法 = 把拒绝原因回灌进下一轮 observation（保留原始观测）。
5. **thinking 预算**：44% 的 turn 撞 cap 且仍在 `<think>` 内。加"简明思考"指令 +
   提高 cap，截断率降到 6%。
6. **注释插进 `\` 续行的参数列表**（我自己的 shell bug，犯了两次）→ rc=127。
   已加守卫检查 `comment-inside-continuation: NONE`。
7. **信息隔离断言误报**：用**字符串**判断"当前动作是否出现在 future trace"，
   而 ALFWorld 里重复同一动作很正常 → 把合法数据判成泄漏，杀掉了第一次 HPI 正式 run。
   修法 = 改按**索引**判断（privilege 是否渲染了 `(turn t)`）。

## 显存天花板（决定了全部超参）

峰值在 **LoRA 的 lm_head**（`tensor_parallel/layers.py:349 linear_with_frozen`）：
冻结但仍要物化 `[T, 151936]` fp32 输出，且 **bshd 会 pad 到 batch 内最长样本** ⇒
决定峰值的是 **max 而非 mean**，且 T = 整局的 packed 长度（Σ 每轮 prompt+gen）。

实测边界（TP=1 + 全量激活重算 + `--recompute-loss-function`）：

| turns / cap | response max | 结果 |
|---|---|---|
| 5 / 640 | 2775 | **fit**（Gate C v3 rc=0） |
| 5 / 768 | — | **OOM**（24.42/29.49） |
| 6 / 768 | 3073 | OOM |
| 8 / 384 | ~2500 | fit，但 **38% turn 被截断**（96 truncated vs 2 not_admissible） |

关键推论：**cap 必须 ≥ ~1024 模型才写得完 `<action>`**，而总量上限 ~2800 ⇒
**轮数被卡在 5 左右**。要放开必须让 `math_opsd.lora_megatron.custom_model_provider`
支持 TP（TP=2 直接 `4096 and 2048 cannot broadcast`，hidden 维被切但 provider 仍按 2048 算）。

**TP=8 的结论**：不对，但原因不是通信 —— 1.7B 的瓶颈是 vocab 侧显存，需要的是
**TP 切词表维**，且只在单节点内做；跨节点 TP 只会更慢。

## 当前状态

```
worker-0  Future-HPI  runs/alfworld_hpi_future_v3   8 卡
worker-1  GRPO        runs/alfworld_grpo_v4         8 卡
配置: G=8, rollout_bs=4 -> GBS 32, turns=5, cap=640, TP=1, 25 steps
```

## 未解决的问题（需要决策，不是工程问题）

**train split 上 `success_rate` 恒为 0**，导致每个 GRPO group 奖励全同 →
advantage 恒 0 → `train/loss` 与 `train/grad_norm` 全程为 0，**训练不产生任何学习**。

- base 在 valid_seen 上实测 16.7%（非 0，说明任务可解），但 train split + 5 轮下为 0
- 5 轮是显存逼出来的，偏离标准 ALFWorld recipe 很远，很可能是主因
- 非 0 的成功率是 GRPO 有信号的前提；这是**实验能否成立**的关键，不是调参能绕过的

可选方向（按性价比）：① 让 LoRA provider 支持 TP 以放开轮数；② 用 valid_seen 那类
更易的任务子集做训练集；③ 先跑 Outcome-only（不需要 future trace，可能更稳）。

---

# 2026-09-25 深夜续：HPI 链路打通，但暴露两个必须决策的问题

## 又修掉的 bug（4 个）

8. **loss-time 对齐门禁依赖了错的参照物**：它拿 `unconcat_tokens[-response_length:]`
   当"学生生成序列"，但那属于训练 batch，管线可以 pad/切。同一份数据 convert 期通过、
   loss 期报错。改成检查不依赖它的不变量（full 与 control 视图 target 必须逐 token 相同、
   且长度等于 turn_ids 推出的 span）。
9. **tensor 不能当 dict key**：`batch["turn_ids"]` 每项是 torch tensor，迭代产出 0-dim
   tensor 标量；拿它当 key **不按值去重**（每个 tensor 对象都是不同 key）。5 轮的 episode
   被读成 **"1613 turns"**，杀掉整轮 run。修法：先 `int()`。**同一个坑在
   `compute_hindsight_weights` 里也有。** 已加回归测试。
10. **teacher 在 backward 时占着 3.4 GB**：OOM 栈显示失败在 `custom_backward`，
    不在 teacher forward。修法 = 打分完把 teacher 挪到 CPU（backward 前释放），
    下一步再移回。**不要调 `empty_cache()`** —— 那会把块还给 driver，下个 microbatch
    再要回来，纯抖动。
11. **`--recompute-loss-function` 让昂贵 loss 跑两遍**：checkpoint 会在 backward 时
    重新执行 loss，而我们的 loss 含 teacher 打分 ⇒ 80 个 view 每步算两遍。
    修法 = 按 view 内容 memoize（teacher 冻结 ⇒ 结果确定，缓存语义安全；key 必须
    把 list 转 tuple 否则不可哈希）。
12. **wrapper 硬赋值 `RUN_NAME`** 覆盖了 env，而 lock 名由 RUN_NAME 算出 ⇒ 带
    `RUN_NAME=..._vN` 的重启全部被 flock 拒绝，我读不同名字的日志造成了"换了配置"的
    错觉。已改成 `${RUN_NAME:-...}`。

## 仍未解决：HPI 每步 ~30 分钟（比 GRPO 慢 15×）

py-spy 定位：时间花在 **HF teacher 的前向**（`modeling_qwen3.py`），不在 actor。
80 个 view × ~2.5k token 走 HF + sdpa（无 NPU graph），叠加每 microbatch 3.4 GB 上下机。

**架构性修法（推荐，下一轮做）**：把 teacher 打分从 loss **挪到 rollout 侧** ——
episode 结束后在 RolloutManager 里批量打分，只把 `e_t` 标量随 sample metadata 传下去。
这样训练步的开销与 GRPO 完全一致（参考实现 geo3k 就是这么做的：把 privileged
scoring 移出 per-turn loop）。当前实现放在 loss 里是**设计选择失误**。

## 仍未解决（科学层面，需用户决策）

两条臂 `train` split 上 `success_rate` 恒 0 ⇒ advantage 恒 0 ⇒ `loss`/`grad_norm` 恒 0。
GRPO 臂连跑 18 步全为 0，第 19 步因更长的 episode OOM。**训练不产生任何学习。**

- base 在 valid_seen 实测 16.7%（任务可解），train split + 5 轮下为 0
- 5 轮是显存逼出来的，标准 ALFWorld recipe 是 30–50 步

## 显存天花板（最终结论）

峰值 = **LoRA 的 lm_head**（冻结但仍物化 `[T,151936]` fp32），bshd pad 到 batch 内
**最长**样本 ⇒ max 决定峰值。实测：GRPO 在 response max 2775 撑了 18 步后 OOM，
HPI 在 2769 直接 OOM ⇒ **cap=640 对两臂都过了线**，最终定 **turns=5 / cap=512**。

`cap` 必须 ≥~1024 模型才写得完 `<action>`（cap=384 时 38% 截断），而总量上限 ~2800
⇒ **轮数被卡在 5**。放开必须让 `math_opsd.lora_megatron.custom_model_provider` 支持 TP
（TP=2 现在直接 `4096 and 2048 cannot broadcast`）。

---

# 2026-09-25 v2：per-turn 训练行 + Full/Control 密度比

新包 `alfworld_hpi_v2/`，**v1 文件一个字节未改**（w1 上的 v1 GRPO 仍在跑，它 import 的是旧字节）。
v1 只被 import 复用（paths / env_adapter / action_parser / trajectory / reward / eval）。

## v2 相对 v1 的三处结构性修正

1. **一 episode 一行 → 一轮一行**。v1 的行长随轮数增长，而 LoRA 的 lm_head
   （冻结但仍物化全词表 fp32）由行长决定 ⇒ 轮数被卡在 5。v2 每轮一行 ⇒
   **horizon 与训练行长解耦**，`MAX_TURNS=20` 实测可行（640 行 = 32 episode × 20 轮，无 OOM）。
2. **teacher 打分移出 loss**。v1 放在 loss 里，`--recompute-loss-function` 会让它跑两遍，
   且每 microbatch 要把 3.4 GB teacher 上下机 ⇒ **30–70 min/step**。v2 走
   `--rollout-data-postprocess-path`（`actor.py:506-507`：convert 之后、训练步之前、训练进程内），
   teacher 常驻、每 view 只打一次分。`loss.py` **不 import teacher**，`teacher_forward_calls` 恒 0。
3. **去掉手调权重**。v1 是 `1 + alpha*tanh(|e|/scale)` + 标定 scale + clamp。v2 用无 shaping 的
   `e_t = mean_{action tokens}(log q_full - log q_control)` → `r_t = exp(e_t)` →
   轨迹内 token 数加权自归一化 `w_it = r_it / mean_r_i`，使
   `Σ n_it·w_it / Σ n_it ≡ 1`。Full==Control 时 e=0,r=1,w=1，严格退化为 vanilla GRPO。

## 关键实现细节（都不需要改 slime）

- **轨迹级 GRPO advantage**：slime 的 GRPO 归一化是按**位置**切组的
  （`ray/rollout.py:717-728`），per-turn 行会让它按"轮"而非"episode"归一化。
  改用 `Sample.opsd_advantages` + `--use-opsd`/`OPSD_ENABLED=1` —— 唯一全通的旁路通道，
   slime 原样透传、绕过 estimator 与全部归一化。
- **轨迹身份**：`turn_ids`（唯一的 per-token 整数白名单通道）= `episode_id*10000 + turn_idx`。
- **HPI 权重**：`fit_turn_weights`（白名单 + 进 get_batch，FIT 关闭时无人使用）。
- **action span 不靠新通道**：scorer 只有 token ids，于是 decode → `<action>…</action>` →
  offset 映射 → **校验 ids 一致才用**。真实 Qwen3 tokenizer 实测精确还原动作文本。

## 实测（2026-09-25）

- `alfworld_hpi_v2/tests/` **19 passed**，覆盖规格 §20 的 13 条 + 额外 6 条
- v2 HPI 1-step：rc=0；v2 GRPO 1-step：rc=0。per-turn 布局生效（640 行/32 episode×20 轮）
- **HPI 路径未被触发**：4 个 group 的 reward 全相同 ⇒ `mixed=False` ⇒ 按规格 §15
  **跳过全部 teacher forward**（这正是零优势跳过的设计行为，但也意味着 1-step 没验证到打分链路）
- **坑**：slime 会把返回 metrics dict 的**每一项除以 batch token 总数**
  （`n_response_tokens` 被除成 1.0，`weight_mean/min/max/n_trajectories` 全变成同一个小数）。
  指标通道不可用于读绝对量 → 已在 loss 里改为直接 `logger.info`。

## 仍未完成

- §21 的 e/r/w 分布（需要至少一个 mixed group）、§23 的 5-step、§25 的 cp25、§22 的 v2 Base sanity
- v1_5turn_diagnostic（w1）跑到 step 20+，稀疏信号（step 5/8 等有非零 loss/grad）

---

# 2026-09-25 v2 诊断：真实 mixed group 上的 scorer 首次执行

## 修复：future-gap observation provenance（真泄漏）

v1/v2 早期版本渲染的是 `observation_before(action_i) + action_i`。对 gap=1，
第一条 future 是 i=t+1，而 `observation_before(action_{t+1})` **就是 `o_{t+1}`**，
即 `a_t` 的直接结果 —— 正是 gap=1 要扣住的东西。

改为按 transition 序列化：
```
[Future transition i]
Executed action: action_i
Resulting environment observation: o_{i+1}
```
并**删掉**了冗余的 "Final environment observation" 行（它既与最后一条 transition 重复，
又会在最后一轮打印 `o_{t+1}`，使该轮 Full ≠ Control）。

## 新增的结构化 provenance gate

`assert_transition_provenance` 解析 `[Future transition i]` 标签逐段核对，
不做子串匹配 —— **子串匹配在 ALFWorld 上不可用**，因为环境会重复文本
（`"Nothing happens."` 每次失败都出现），后一条观测合法地包含前一条。
gate 同时验证「窗口」与「provenance」：每条 transition 打印的必须是它的
**resulting observation**，而不是它之前的状态。后者正是抓 off-by-one 的那一半。

**这个 gate 连续误报了两次，都是 gate 自身的 bug**：(1) 子串匹配；
(2) 多行观测（拒绝反馈消息内含换行）被按行截断。均已修 + 加回归测试。
哨兵测试（`OBS_i`/`ACTION_i`）与「重复观测」回归测试都在。

## 真实 mixed group 上的 scorer 结果（§21/§6/§7/§8）

用 v1 中**真实产出过 success 的 task**（look_at_obj_in_light ×2、
pick_and_place_simple ×1）建诊断 schedule，`LR=0`（不动权重）跑 1 步：

```
rollout/success_rate : 0.34375
rollout/mixed_groups : 2 / 4
rollout/turns        : 513
rollout/hpi_scored_turns : 273
scorer: scored=44 skipped=84 groups=32 teacher_views=126  (per rank, DP=4)
```

| 量 | mean | std | min | p50 | p99 | max |
|---|---|---|---|---|---|---|
| `e_t` | -0.0003 | 0.0035 | **-0.0230** | 0.0000 | +0.0094 | +0.0094 |
| `r_t` | — | — | 0.9994 | 1 | 1 | 1 |
| `w_it` | 1.0000 | — | 0.9996 | 1.0000 | 1.0001 | 1.0001 |

**NaN = 0, Inf = 0**；`w` 的 token 加权均值 = 1 ✓；Full/Control 的 target ids、
action span、outcome block 逐字节相同（有测试）。

**结论：`e_t` 在 ALFWorld 上比 math 任务小约两个数量级**（math 的 `|e|` p90 = 0.425）。
冻结 teacher 对"延迟未来交互 trace"几乎不改变它对当前动作的密度评估。
这是**科学结论，不是 bug** —— 机制正确执行，信号本身很弱。

## 另一个确认

slime 会把返回 metrics 的每一项除以 batch token 总数（`n_response_tokens` 被除成 1.0）。
我自己的 `[v2-loss]` 日志显示 `w(mean=1.000000 min=1.000000 max=1.000000)`，
与 scorer 的 0.9996–1.0001 一致 —— 之前的 0.003 确为指标缩放假象。

---

# 2026-09-25 冻结诊断（frozen, no optimizer step）：Case D

规格要求：不训练、不更新权重、不重新 rollout，冻结 model/trajectory/reward/h_t/y_t，
只改变 privileged future。实现见 `alfworld_hpi_v2/diagnostic.py`，逐 turn 结果在
`/tmp/hpi_diag.jsonl`（481 行，可独立复核）。

## 条件与 span

| 条件 | 含义 |
|---|---|
| C | CONTROL，无 future trace |
| T | TRUE_GAP1，真实 future（`i >= t+1`）|
| S | SHUFFLED_GAP1，**匹配的兄弟 episode** 的 future（同类别、同 outcome、相近剩余长度）|
| P | TRUE_GAP0，**故意泄漏** `a_t → o_{t+1}`，正对照，绝不用于训练 |

span：`think`（`<think>` 内容）/ `action`（`<action>` 内容）/ `full`（全部生成 token）。
target token ids 在四个条件下**逐字节相同**。

## 结果（481 turns，32 条真实轨迹）

**invariant：`max|log(r) − e| = 1.110e-16` → PASS。**
（之前报的"e_min=-0.023 却 r_min=0.9994"是我把不同 rank/条件的日志行混在一起报了，不是数学错误。）

| span | mean\|e_T\| | mean\|e_S\| | **T/S ratio** | mean\|e_P\| |
|---|---|---|---|---|
| **think** | 0.4976 | 0.5098 | **0.98** | 0.6197 |
| action | 0.0422 | 0.0367 | 1.15 | 0.0315 |
| full | 0.3813 | 0.4024 | **0.95** | 0.4537 |

`e_T_think` mean = **−0.4975**（系统性偏负：塞入 future block 让 teacher 对学生 token 给更低概率）。

## 判读

1. **不是 Case C。** think span 的效应很大（0.50），所以**不是 action-only scoring 把信号 condition away 了**。
2. **是 Case D。** `T ≈ S`（think 0.98、full 0.95）—— teacher 对"另一条 episode 的未来"的反应与对"自己真实未来"的反应**一样强**。效应是**"这里有一段 future 文本"的通用存在效应**，不是关于本条轨迹未来的特异信息。
3. **gap0 正对照只强 ~25%**（0.6197 vs 0.4976）—— 连故意泄漏当前动作的直接结果，都没能让 teacher 产生显著更强的、特异的响应。

**结论：冻结初始策略（Qwen3-1.7B）没有编码"未来交互轨迹 → 当前决策"这层关系。**
Full-vs-Control 密度比在这种 teacher 上无法承载可用的 credit 信号 —— 这是**有对照支撑的科学结论，不是实现 bug**。

`READY_FOR_1P7B_ALFWORLD_CP25 = NO`，`READY_FOR_4B = NO`。

未做：cp25、4B、w mapping 改动、tanh/alpha/lambda、gap/horizon sweep、重新训练、easy subset。

---

# 2026-09-25 方法纠正：单 rollout HSI（advantage 而非 loss 权重）

## 概念性错误（用户指出，已修）

v2 前一版把 HPI 接成 **`w_t * L_GRPO`**。GRPO advantage 在组内奖励全同时
**恒为 0**，于是 `w * 0 == 0` —— hindsight 项被整体乘零。由此还派生出两个错误要求：
"必须出现 mixed-reward group"和"zero-advantage group 跳过 scorer"。
两者都建立在同一个误解上：**HSI 不依赖组内奖励方差**。

## 冻结后的方法

| | GRPO 臂 | HSI 臂 |
|---|---|---|
| G（每 task rollout 数）| 8 | **1** |
| advantage | 组内归一化 `(R−μ)/σ` | **`A_it = (2R_i − 1) + ĉ_it`** |
| loss | stock clipped PPO | 同上（**loss 侧不再乘任何权重**）|

```
e_it  = mean_{action 内容 token} [ log q_true(a) − log q_shuffle(a) ]     # MEAN
s_i   = 2R_i − 1                                                          # SUCCESS +1 / FAILURE −1
c_it  = s_i · e_it
ĉ_it  = (c_it − mean_t c_it) / (std_t c_it + 1e-6)      # <2 有效 turn 或 std 退化 ⇒ ĉ=0
A_it  = s_i + ĉ_it
```

- **失败 rollout 同样是监督**（`s_i=−1` ⇒ 压低"真实失败未来更支持"的动作）
- `mean_t(A_it) ≈ 2R_i − 1`：**verifier 定方向，hindsight 只做轨迹内再分配**（有断言 + 测试）
- 对照从 **empty-control 换成 matched-shuffled**：诊断显示 empty control 的 `e_t` 主要是
  "这里存在一段 future 文本"的通用效应（T/S ratio 0.98 ⇒ 不可分），换上同形状同长度的
  打乱 trace 后把它扣掉

## 实现改动（`alfworld_hpi_v2/`）

- `rollout.py`：HSI 臂 `G=1`、`OutcomeAnchor`（`2R−1`，不做组归一化）、
  **删除 mixed gating**、shuffled 候选来自**整个 batch 的轨迹池**（teacher replay control，
  不是额外环境 rollout）
- `scorer.py`：新增纯函数 `hindsight_credit`，postprocess 把 **`A_it` 写回 `opsd_advantages`**
  （不再写 `fit_turn_weights`）
- `loss.py`：**移除 loss 侧的权重乘法**（否则会二次施加，并重新引入乘零问题）
- `privilege.py`：新增 `matched_sibling`（同类别 → 同 outcome → 最近剩余 horizon；
  无匹配则不建视图、`ĉ=0`，不 crash）
- `_common_v2.sh`：新增 `eval` arm；`GLOB_BS` / `TRAIN_ITERS` **移到 ARM switch 之后**计算

## 测试

**34 passed**，含规格要求的合成用例：
R=0 产生非零 credit、R=1 产生非零 credit、两者符号相反、
`mean_t(A) == 2R−1` 断言、<2 turn 或退化 std ⇒ 0、G=1 下 `OutcomeAnchor` 两种 outcome 都非零、
**统一全失败组仍然建视图**（旧版会整组丢掉）。

## 1-step smoke（rc=0，无 OOM）

```
[v2-hpi] e_t   mean=0.0390 std=0.1234 p99=+0.4291 max=+0.4291     ← empty-control 版 ≈0
[v2-hpi] r_t   min=1 p50=1 p99=1.536 max=1.536  (finite=True)
[v2-hpi] c_hat min=-0.9904 p50=0.0000 max=+0.9904  (NaN/Inf = 0)
[v2-loss] pg_loss_mean = 1.0 / 2.73                               ← 非零梯度
```

## cp25（HSI 臂完成）

`runs/alfworld_hsi_cp25/iter_0000024`，**25 轮用时 ~1h15m（~3 min/轮）**；
GRPO 同 schedule 跑到 rollout 12 时已用 ~5h（~25 min/轮）—— **8× wall-clock 差**，
直接来自每 task 1 条 rollout vs 8 条。HSI 的 scorer **每轮都执行**
（`hpi_scored_turns` 52–76/轮），包括 `success_rate = 0.0` 的轮次。
