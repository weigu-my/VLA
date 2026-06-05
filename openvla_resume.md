# OpenVLA 项目简历素材

## 项目名称

VLA 机器人策略对比研究：OpenVLA / OpenVLA-OFT / Pi0 / Pi0.5 微调与仿真评估

## 项目概述

基于 OpenVLA-7B、OpenVLA-OFT 与 Physical Intelligence 的 Pi0 / Pi0.5（flow matching VLA），在 LIBERO-Spatial 机器人操作任务集上完成模型微调、checkpoint 管理、仿真评估、rollout 可视化与实验结果分析。项目覆盖从环境搭建、训练问题定位、模型评估到官方 baseline 复现的完整流程，并横向对比了**离散动作 token（OpenVLA）与连续动作 flow matching（Pi0/Pi0.5）两类架构**在相同任务上的成功率与失败模式差异。

## 技术栈

- 模型与算法：OpenVLA-7B、OpenVLA-OFT、Pi0 / Pi0.5、LoRA、QLoRA、flow matching、action chunk、continuous action head、AdaRMS 条件化
- 训练与推理：PyTorch、Transformers、PEFT、bitsandbytes、torchrun、W&B、LeRobot
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
- 基于 LeRobot 搭建 Pi0 / Pi0.5（flow matching VLA）的 LIBERO 仿真评估链路，在 numpy 2 / Python 3.12 环境下解决 `hf-libero`、robosuite、mujoco 依赖与 cmake 构建、`~/.libero` 配置等环境问题，复用 LeRobot 原生 `lerobot-eval` + `libero` env 驱动评测。
- 在相同 50-episode（5 trials × 10 任务）设置下完成 OpenVLA、Pi0、Pi0.5 三方对比评估，成功率分别为 82% / 66% / 84%；定位官方 `lerobot/pi0_libero` 仓库缺失归一化统计量、为老格式 in-model norm buffer 的"基座占位"模型（输出恒为均值动作、夹爪不闭合、成功率 0%）的问题，改用自带 stats 的 `*_finetuned_v044` 微调版恢复正常评测。
- 定位评估协议差异（关键）：对比 openpi 官方 LIBERO 评测脚本，发现其 `replan_steps=5`（每 5 步闭环重规划），而 LeRobot 默认 `n_action_steps=50`（执行完整 50 步动作 chunk 才重规划，近似开环）。逐项核对两边评测 config（state 构成、图像 180° 翻转、resize_with_pad、num_steps_wait 均一致），确认 replan 频率为主因。修正为 replan=5 后，Pi0.5 500-ep 成功率从 87.0% 提升至 **96.6%**，逼近 openpi 官方宣传的 98.8%（残余差距来自渲染分辨率 640×480 vs 256、MEAN_STD vs 分位数归一化等二阶因素）。
- 按任务维度归因两类架构的失败模式：最难任务「碗叠在小碗上」（高窄易倾的精细接触抓取），在闭环协议下 Pi0.5 从 36% 提升到 70%（说明其具备能力，此前低分是开环协议所致），而 Pi0 始终为 16%（换协议无改善，为真实能力缺陷）；OpenVLA-OFT 在该任务达 96%。
- 验证 Pi0 → Pi0.5 架构改动（state 文本化、AdaRMS 时间步条件化、分位数归一化）的收益：相同数据与评估设置下，500-ep 成功率从 61.8% 提升至 87.0%（开环协议）、从 69.4% 提升至 96.6%（闭环协议），提升集中在精细定位与接触抓取任务。

## 实验结果

### OpenVLA / OpenVLA-OFT

| 模型 / Checkpoint | 评估设置 | 成功次数 | 成功率 |
| --- | ---: | ---: | ---: |
| 自训练 OpenVLA 2.5k checkpoint | LIBERO-Spatial, 50 episodes | 4/50 | 8.0% |
| 自训练 OpenVLA 15k checkpoint | LIBERO-Spatial, 50 episodes | 29/50 | 58.0% |
| 官方 OpenVLA finetuned checkpoint | LIBERO-Spatial, 50 episodes | 41/50 | 82.0% |
| 官方 OpenVLA-OFT，错误环境 | LIBERO-Spatial, 50 episodes | 29/50 | 58.0% |
| 官方 OpenVLA-OFT，正确环境 | LIBERO-Spatial, 50 episodes | 50/50 | 100.0% |
| 官方 OpenVLA-OFT，正确环境 | LIBERO-Spatial, 500 episodes | 492/500 | 98.4% |

### 四类架构横向对比（LIBERO-Spatial，500 episodes = 50 trials × 10 任务）

| 模型 | 动作表示 | 成功次数 | 成功率（500-ep） | 成功率（50-ep） |
| --- | --- | ---: | ---: | ---: |
| 自训练 OpenVLA 15k（4-bit QLoRA） | 离散 action token（自回归） | 315/500 | 63.0% | 58.0% |
| Pi0（微调版 v044） | 连续 / flow matching | 309/500 | 61.8% | 66.0% |
| 官方 OpenVLA finetuned | 离散 action token（自回归） | 409/500 | 81.8% | 82.0% |
| Pi0.5（微调版 v044, MEAN_STD） | 连续 / flow matching + AdaRMS | 435/500 | 87.0% | 84.0% |
| 官方 OpenVLA-OFT | 连续 L1 动作头 + action chunk（并行解码） | 492/500 | **98.4%** | 100.0% |

> 自训练 OpenVLA（15k）vs 官方 OpenVLA finetuned（同数据 libero_spatial_no_noops、同 LoRA rank 32、同动作表示）相差 63.0% vs 81.8%，差距来自 RTX 3090 单卡的两个资源妥协：(1) 4-bit QLoRA 量化损伤冻结骨干的表征精度（官方为 bf16）；(2) 训练欠充分——因 checkpoint 保存阶段重复合并 7B 模型导致 GPU idle，于 step 15000 手动中断（计划 20k），自身 2.5k→15k 仍处陡峭上升段（8%→58%）未收敛。即"数据/方法一致，差距主要是算力与训练步数"。

> 注：上表 Pi0/Pi0.5 使用 LeRobot 默认 `n_action_steps=50`（开环执行整段 chunk）。OpenVLA 为单步自回归（天然每步重规划），故上表在控制频率上对 Pi0/Pi0.5 不利；下表给出对齐 openpi 官方 `replan_steps=5` 闭环协议后的结果。

### 评估协议对齐后（replan=5，闭环重规划，LIBERO-Spatial 500-ep）

| 模型 | n_action_steps=50（开环）| n_action_steps=5（闭环，对齐 openpi）| 提升 |
| --- | ---: | ---: | ---: |
| Pi0（微调版 v044） | 61.8% | 69.4% | +7.6 |
| Pi0.5（微调版 v044, MEAN_STD） | 87.0% | **96.6%** | +9.6 |

> 结论：Pi0.5 在对齐官方协议后达 **96.6%**，逼近 openpi 官方宣传的 98.8%（残余差距来自渲染分辨率 640×480 vs 256、MEAN_STD vs 分位数归一化）。说明此前 87% 主要是评测协议（开环执行 50 步动作 chunk）造成的低估，而非模型能力差距。最难任务「on the ramekin」上 Pi0.5 随协议从 36%→70%（具备能力），Pi0 始终 16%（真实能力缺陷），OFT 96%——区分了"协议假象"与"真实架构差距"。四模型综合排名（闭环协议下）：OFT 98.4% > Pi0.5 96.6% > OpenVLA 81.8% > Pi0 69.4%。

## 简历精简版

- 基于 OpenVLA-7B 构建 LIBERO-Spatial 机器人策略微调与评估 pipeline，完成 LoRA / QLoRA 单卡训练、checkpoint 保存优化、MuJoCo / robosuite 仿真评估与 rollout 可视化管理。
- 在 RTX 3090 上完成 OpenVLA-7B 4-bit LoRA 微调，解决 gradient checkpointing + DDP 冲突和 checkpoint 保存导致 GPU idle 的问题；15k-step checkpoint 在 LIBERO-Spatial 50-episode 评估中达到 58.0% 成功率。
- 复现并对比官方 OpenVLA 与 OpenVLA-OFT 策略，定位 OFT 环境依赖问题并构建独立 venv；官方 OpenVLA-OFT 在 500-episode LIBERO-Spatial 评估中达到 98.4% 成功率。
- 基于 LeRobot 完成 Pi0 / Pi0.5（flow matching VLA）在 LIBERO-Spatial 的仿真评估，与 OpenVLA 横向对比（66% / 84% vs 82%），并从任务维度归因离散 token 与连续 flow matching 两类动作表示的失败模式差异。

## 后续可补充

- 自训练 OpenVLA 20k checkpoint 的评估结果
- OpenVLA-OFT 自训练或微调结果
- Pi0 / Pi0.5 自行微调（当前为官方/社区微调版评测）与去噪步数（1/5/10/50）消融
- 三方对比扩展到 500-episode 以降低小样本方差
- 不同 batch size、学习率、量化策略对成功率的影响
- rollout 失败案例分析与任务维度误差归因
