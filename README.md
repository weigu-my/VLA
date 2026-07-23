# VLA 训练与评估实践

这是一个经过脱敏的个人项目档案，记录我围绕 OpenVLA、OpenVLA-OFT、π0 和 π0.5 完成的训练、仿真评估与实验诊断工作。核心目标是在资源受限环境下建立可复现的 VLA 实验流程，并分析离散动作 token 与连续 flow matching 策略的差异。

> 本仓库只应保存个人编写的文档、脚本和汇总结果。模型权重、数据集、公司私有源码、原始训练日志、凭据及内部地址均不应提交。

## 完成内容

- 在单卡 RTX 3090 上完成 OpenVLA-7B 4-bit QLoRA 微调，覆盖数据准备、训练监控、checkpoint 管理和 LIBERO 仿真评估。
- 定位 gradient checkpointing、DDP 和 checkpoint 保存流程中的工程问题，并将训练中保存 adapter 与离线合并权重解耦。
- 在统一的 LIBERO-Spatial 协议下评估 OpenVLA、OpenVLA-OFT、π0 和 π0.5，完成 50/500 episode 多组对照实验。
- 发现 `n_action_steps=50` 与 `replan_steps=5` 的协议差异会显著低估 action-chunk 策略，修正后 π0.5 成功率由 87.0% 提升至 96.6%。
- 完成 π0.5 去噪步数、推理延迟和动作平滑度实验。
- 使用 W&B API 导出并诊断一次 π0.5 全量微调运行，确认学习率调度、loss 收敛和远端 checkpoint 保存状态。

## 主要结果

| 模型 | 设置 | LIBERO-Spatial 成功率 |
|---|---|---:|
| 自训练 OpenVLA 15k，4-bit QLoRA | 500 episodes | 63.0% |
| 官方 OpenVLA finetuned | 500 episodes | 81.8% |
| π0 finetuned | 500 episodes，默认执行 50 步 chunk | 61.8% |
| π0 finetuned | 500 episodes，每 5 步重规划 | 69.4% |
| π0.5 finetuned | 500 episodes，默认执行 50 步 chunk | 87.0% |
| π0.5 finetuned | 500 episodes，每 5 步重规划 | **96.6%** |
| OpenVLA-OFT | 500 episodes | **98.4%** |

不同模型的训练数据和专用程度并不完全一致，因此该表主要用于工程复现和协议分析，不应直接解释为严格的架构排行榜。

## 文档索引

- [项目记忆与实验结论](docs/PROJECT_MEMORY.md)
- [公开发布检查清单](docs/PUBLISHING_CHECKLIST.md)
- [完整简历素材](openvla_resume.md)
- [OpenVLA 学习与复现计划](openvla_plan.md)
- [π0 / π0.5 学习与实验计划](pi0_plan.md)
- [LIBERO-Spatial 汇总数据](results/libero_spatial_summary.csv)
- [π0.5 去噪步数消融](results/pi05_denoising_ablation.csv)
- [π0.5 全量微调训练摘要](results/pi05_fullft_training_summary.json)

## 实验脚本

- `ablate_nsteps.sh`：π0.5 去噪步数消融。
- `bench_nsteps_latency.py`：测量不同去噪步数的 chunk 推理延迟。
- `bench_pi0_pi05.py`：对比 π0 与 π0.5 的推理速度和动作特征。
- `bench_smoothness.py`：在共同成功 episode 上计算动作变化率与 jerk。

这些脚本通过命令行参数或环境变量接收模型路径，不包含本机用户名和固定模型目录。

## 复现边界

公开仓库不包含模型、数据和完整 rollout。结果摘要保留了足够的实验设置与数值，但要完全复现仍需自行下载上游模型和 LIBERO 数据，并按各项目许可证配置运行环境。
