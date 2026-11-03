请在本台服务器实际启动 ALFWorld SRLVR-v2 与 SRLVR-v3 的正式训练和测评，一个任务独占一个节点，不再做拓扑测速。项目：
/workspace/cua/repos/opsd-slime

先读 AGENTS.md 与 SRLVR_REVISION_NOTE.md。旧 v2/v3 调试与未完成 runs 已清理，清单和压缩日志在 reports/_cleanup_v23_20261027_235709/；不要查找旧 run 续训。现有未提交源码是当前主线，保留另一台服务器的共享修改。此前 Claude bridge 连续报 402 余额不足，如仍不可用请直接执行，不要反复委派或停在计划。

节点分工：
- worker-0（本机）运行 SRLVR-v2。
- worker-1 运行 SRLVR-v3，SSH 端口 2222，地址：ma-job-2990cb8c-2431-4b04-b1a2-d20d13d7af64-worker-1.ma-job-2990cb8c-2431-4b04-b1a2-d20d13d7af64。
- 两个节点各 8 张 Ascend 910B，源码和 runs 共享，Python 环境是节点本地的。

正式配置两臂一致，仅 SRLVR_VARIANT 不同：
- Qwen3-1.7B，从 /workspace/cua/checkpoints/Qwen3-1.7B 新开训练。
- 全量 RL：OPSD_LORA=0；确认没有 LoRA 参数白名单、没有冻结基模。
- ACTOR_GPUS=8、ROLLOUT_GPUS=8、COLOCATE=1；训练 TP=2、PP/CP/EP=1；推理 ROLLOUT_GPUS_PER_ENGINE=1，即 8 个 TP1 引擎。
- SGLANG_ENABLE_NPU_GRAPH=1、SGLANG_CUDA_GRAPH_MAX_BS=8；prefix cache、环境池、prompt 分词缓存及主线训练优化保留。
- 默认启用 thinking，但不设独立 token 预算：清除 ALFWORLD_THINKING_TOKENS，提示词不得有 24/512-token 或“一句话”限制；MAX_RESPONSE=4096、SEQ_LENGTH=32768、MAX_TURNS=50、完整历史、展示 admissible actions。
- T=1、top_p=1、top_k=-1，严格 FP32 full-softmax CDF 采样；不强制插入思考结束标记，不增加采样尝试或丢弃轨迹。
- 每批 32 条真实轨迹、G=1、SRLVR_CALIBRATION=4；四族混采各 8 条，C/U 覆盖同一四族，保留当前课程与 Full hindsight 定义。
- 两臂均 NUM_ROLLOUT=50、SAVE_INTERVAL=5、SEED=42、LR=1e-6、KL_COEF=0.001，每批仅一次 actor optimizer 更新，保存完整模型/optimizer/scheduler/RNG/方法及课程状态。
- 其余沿用当前主线默认。全量训练比短测多 optimizer/梯度内存，按真实显存选用现有 optimizer CPU-offload；不要为省内存改回 LoRA、缩短上下文、减少轨迹或校准。

执行要求：
1. 先确认两节点无其他训练占用，检查当前源码与最终 argv，再用 tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh 分别设置 ARM=srlvr_v2/v3 启动。不要 source 旧 reports/common.env，它们会重新带入 LoRA 或 thinking512。不要重跑广泛测试。
2. 创建全新且合规的独立 run 目录，保留完整启动配置、host、PID、日志和 manifest。沿用已经验证的 Ascend TMS/colocate、HCCL 端口及 allocator 设置，wandb 关闭。
3. 真正观察到 rollout 生成、graph=True，随后完整 actor 更新且 grad_norm 非零。全量训练及更新后 graph 的恢复尚需实跑确认；遇到实际故障定位修复，不把采样测速当作完整训练验收。
4. 配置可靠的每节点单实例守护与最新完整 checkpoint 恢复，防止意外退出后空转或重复推进课程。不要影响其他节点任务。
5. 用现有 ALFWorld evaluator 评测初始模型与训练第25、50批完成后的模型，valid_seen/valid_unseen；训练/评测使用相同交互提示和解码口径，无独立 thinking 预算。确认真正加载对应 checkpoint，按任务族及总体记录成功率；可调度测评但不要同时挤占本节点训练。
6. 报告两节点 run 路径、实际生效参数、rollout/训练阶段耗时、完整保存恢复和测评状态。核心进展简洁补入 SRLVR_REVISION_NOTE.md。不要承诺新协议仍有10.9分钟：此前是 thinking512+LoRA 的一次实测，新全量无预算配置需重新记录。

请直接完成启动和必要问题修复，不要只给我命令或计划；正式长跑启动稳定后汇报，不必等50批全部结束才回复。
