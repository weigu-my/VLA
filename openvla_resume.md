# OpenVLA 项目简历素材

## 项目名称

OpenVLA / OpenVLA-OFT 机器人策略微调与仿真评估

## 项目概述

基于 OpenVLA-7B 与 OpenVLA-OFT，在 LIBERO-Spatial 机器人操作任务集上完成模型微调、checkpoint 管理、仿真评估、rollout 可视化与实验结果分析。项目覆盖从环境搭建、训练问题定位、模型评估到官方 baseline 复现的完整流程。

## 技术栈

- 模型与算法：OpenVLA-7B、OpenVLA-OFT、LoRA、QLoRA、action chunk、continuous action head
- 训练与推理：PyTorch、Transformers、PEFT、bitsandbytes、torchrun、W&B
- 仿真与评估：LIBERO、MuJoCo、robosuite、RLDS、Hugging Face Hub
- 工程环境：Ubuntu、RTX 3090、Python venv、CUDA 12.1

## 关键工作

- 基于 OpenVLA-7B 构建 LIBERO-Spatial 机器人策略微调与评估 pipeline，完成数据集准备、单卡 LoRA / QLoRA 微调、checkpoint 保存、仿真评估与 rollout 视频管理。
- 使用 RTX 3090 单卡对 OpenVLA-7B 进行 4-bit LoRA 微调，解决 DDP 与 gradient checkpointing 冲突问题，使训练从 backward 报错推进到可稳定完成 20k step 级别训练。
- 分析训练过程中的 loss spike 与 GPU 利用率异常，定位 checkpoint 保存阶段重复合并 7B 模型导致 GPU 长时间 idle 的问题，并优化保存逻辑，改为训练中保存 LoRA adapter、离线合并完整模型。
- 搭建 LIBERO 仿真评估环境，配置 MuJoCo、robosuite、LIBERO、Hugging Face 模型缓存，完成官方 OpenVLA、个人微调 checkpoint、OpenVLA-OFT 官方模型的对比评估。
- 对 15k-step 自训练 OpenVLA checkpoint 进行 50-episode LIBERO-Spatial 评估，成功率达到 58.0%，相比早期 2.5k-step checkpoint 的 8.0% 有显著提升。
- 复现官方 OpenVLA finetuned checkpoint，在相同 50-episode 设置下达到 82.0% 成功率，并按任务维度分析模型短板。
- 为 OpenVLA-OFT 单独构建隔离 Python 环境，安装 custom `transformers-openvla-oft` fork，定位并修复因普通 Transformers 导致 OFT 成功率异常偏低的问题。
- 在正确 OFT 环境下完成官方 OpenVLA-OFT LIBERO-Spatial 500-episode 评估，达到 492/500 成功，成功率 98.4%，验证 OFT action chunk / continuous action head 相比原版 OpenVLA 的速度与成功率优势。
- 改造 OpenVLA 与 OpenVLA-OFT 评估脚本，使不同 run 的日志与 rollout 视频按 run_id 分目录保存，提升实验结果管理与可复现性。

## 实验结果

| 模型 / Checkpoint | 评估设置 | 成功次数 | 成功率 |
| --- | ---: | ---: | ---: |
| 自训练 OpenVLA 2.5k checkpoint | LIBERO-Spatial, 50 episodes | 4/50 | 8.0% |
| 自训练 OpenVLA 15k checkpoint | LIBERO-Spatial, 50 episodes | 29/50 | 58.0% |
| 官方 OpenVLA finetuned checkpoint | LIBERO-Spatial, 50 episodes | 41/50 | 82.0% |
| 官方 OpenVLA-OFT，错误环境 | LIBERO-Spatial, 50 episodes | 29/50 | 58.0% |
| 官方 OpenVLA-OFT，正确环境 | LIBERO-Spatial, 50 episodes | 50/50 | 100.0% |
| 官方 OpenVLA-OFT，正确环境 | LIBERO-Spatial, 500 episodes | 492/500 | 98.4% |

## 简历精简版

- 基于 OpenVLA-7B 构建 LIBERO-Spatial 机器人策略微调与评估 pipeline，完成 LoRA / QLoRA 单卡训练、checkpoint 保存优化、MuJoCo / robosuite 仿真评估与 rollout 可视化管理。
- 在 RTX 3090 上完成 OpenVLA-7B 4-bit LoRA 微调，解决 gradient checkpointing + DDP 冲突和 checkpoint 保存导致 GPU idle 的问题；15k-step checkpoint 在 LIBERO-Spatial 50-episode 评估中达到 58.0% 成功率。
- 复现并对比官方 OpenVLA 与 OpenVLA-OFT 策略，定位 OFT 环境依赖问题并构建独立 venv；官方 OpenVLA-OFT 在 500-episode LIBERO-Spatial 评估中达到 98.4% 成功率。

## 后续可补充

- 自训练 OpenVLA 20k checkpoint 的评估结果
- OpenVLA-OFT 自训练或微调结果
- Pi0 / Pi0.5 推理与微调实验
- 不同 batch size、学习率、量化策略对成功率的影响
- rollout 失败案例分析与任务维度误差归因
