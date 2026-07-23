# VLA 项目记忆

更新日期：2026-07-23

## 1. 文档目的

本文档记录本项目中已经完成的工作、关键实验结果、问题定位过程和后续可继续的方向。内容已按公开 GitHub 的要求脱敏：不包含公司内部链接、私有数据集位置、账号凭据和远端机器地址。

结论分为三类：

- **已验证**：有本地结果文件或训练日志支持。
- **合理解释**：与配置和现象一致，但没有额外对照实验完全隔离变量。
- **未完成**：已经明确需求，但尚缺数据、环境或评估结果。

## 2. 项目范围

项目覆盖四类 VLA 策略：

| 策略 | 动作表示 | 主要实验目的 |
|---|---|---|
| OpenVLA | 离散 action token，自回归生成 | 单卡 QLoRA 微调与基础基线 |
| OpenVLA-OFT | 连续 action head 与 action chunk | 高性能官方基线和环境依赖排查 |
| π0 | flow matching 连续动作 chunk | 研究扩散式动作生成和闭环控制 |
| π0.5 | 改进的 flow matching VLA | 评估架构改进、重规划和去噪步数 |

主要评估环境为 LIBERO-Spatial。核心指标是任务成功率，辅助指标包括训练 loss、梯度范数、推理延迟、动作变化率和 jerk。

## 3. 工作时间线

### 3.1 OpenVLA 单卡微调

在 RTX 3090 24GB 上采用 4-bit QLoRA 微调 OpenVLA-7B：

- 使用 LoRA rank 32，只训练少量参数。
- 解决 DDP 与 gradient checkpointing 组合产生的 backward 问题。
- 定位 checkpoint 保存时重复合并 7B 权重导致 GPU 长时间空闲的问题。
- 将保存策略调整为训练中保存 LoRA adapter，完整模型在训练后离线合并。
- 计划训练 20k step，实际在 15k step 手动停止，因此该 checkpoint 仍可能欠训练。

训练效果随步数明显提升：2.5k checkpoint 的 50-episode 成功率为 8.0%，15k checkpoint 提升至 58.0%。15k checkpoint 在 500 episodes 上达到 63.0%。

### 3.2 OpenVLA-OFT 环境排查

首次评估 OFT 时成功率异常偏低。排查后确认普通 Transformers 与项目要求的定制 fork 不兼容。为 OFT 建立独立环境并安装正确依赖后：

- 50 episodes：50/50，100.0%。
- 500 episodes：492/500，98.4%。

这说明模型代码和依赖版本是 VLA 评估的一部分，不能只记录 checkpoint 名称。

### 3.3 π0 / π0.5 评估链路

基于 LeRobot、MuJoCo、robosuite 和 LIBERO 建立 π0 / π0.5 仿真评估流程，解决了 Python 3.12、NumPy 2、`hf-libero`、CMake 构建和无头渲染配置问题。

模型选择存在一个重要陷阱：部分名称看似 finetuned 的仓库没有匹配的归一化统计量，加载后模型可能只输出接近均值的动作并得到 0% 成功率。最终使用带完整 preprocessor、postprocessor 和统计量的 v0.4.4/community checkpoint。

### 3.4 评估协议差异

最重要的实验结论是：action chunk 的执行长度与重规划频率会直接改变成功率。

- LeRobot 默认 `n_action_steps=50`：一次预测 50 步并近似开环执行完整 chunk。
- 对齐 OpenPI 评估协议后使用 `n_action_steps=5`：每 5 步重新观测并规划。

500-episode 结果：

| 模型 | 执行 50 步 chunk | 每 5 步重规划 | 变化 |
|---|---:|---:|---:|
| π0 | 61.8% | 69.4% | +7.6 个百分点 |
| π0.5 | 87.0% | 96.6% | +9.6 个百分点 |

**已验证结论**：π0.5 先前的 87.0% 主要是评估协议造成的低估，而不是模型本身只有该水平。

在最难的精细接触任务上，π0.5 随闭环重规划由 36% 提升到 70%，说明模型具备相应能力但需要及时纠偏；π0 保持在 16%，更可能是真实能力差距。

### 3.5 去噪步数、速度和平滑度

固定 `n_action_steps=5`，对 π0.5 的 flow-matching 去噪步数进行消融：

| 去噪步数 | 50-episode 成功率 | 单次 chunk 延迟 |
|---:|---:|---:|
| 1 | 98% | 104.7 ms |
| 2 | 96% | 106.6 ms |
| 5 | 100% | 115.1 ms |
| 10 | 96% | 130.2 ms |
| 50 | 94% | 245.5 ms |

成功率没有呈现“去噪步数越多越好”的趋势。50 episodes 的样本量较小，因此不能把 94% 和 100% 解释为显著差异；可以确认的是延迟随去噪步数近似线性增长。在该任务上，将默认 10 步降到 1-5 步具有明显的实时性价值。

π0 与 π0.5 的辅助性能结果：

- 单次 50-step chunk 推理延迟约为 121 ms 和 140 ms。
- 有效动作吞吐约为 360-410 actions/s。
- 在相同任务、seed、初始状态且两模型都成功的 episode 上，π0.5 的平均一阶动作变化下降约 17%，jerk 下降约 33%。

## 4. π0.5 全量微调诊断

另一次 π0.5 全量微调在远端 GPU 环境完成，并通过 W&B 记录训练指标。公开摘要位于 `results/pi05_fullft_training_summary.json`。

核心配置：

| 配置 | 数值 |
|---|---:|
| 精度 | bfloat16 |
| 全局 batch size | 256 |
| 训练步数 | 50,000 |
| action horizon | 50 |
| action dimension | 32 |
| 学习率峰值 | 2e-5 |
| warmup | 2,500 steps |
| 衰减终值 | 2e-6 |
| checkpoint 间隔 | 5,000 steps |

训练曲线摘要：

- `action_loss/total_loss`：0.26713 → 0.001683，最小值 0.001649，出现在 step 48,600。
- 尾部 25 个采样点平均 loss 为 0.001712，较前 25 个采样点下降 91.18%。
- `grad_norm`：6.103 → 0.01759，后期保持在较低水平。
- `param_norm` 仅增长约 0.22%，没有参数范数失控迹象。
- 训练结束时成功保存 step 49,999 checkpoint。

**已验证结论**：优化过程稳定，训练 loss 已充分下降，未看到梯度爆炸。

**限制**：该运行没有记录独立 validation loss，也尚未在公开可复现的离线或仿真任务上评估。因此不能仅凭训练 loss 判断策略泛化能力和真实执行成功率。

## 5. 之前问答中的关键结论

### 5.1 W&B 登录是否需要重新登录

当命令显示凭据已经从 `~/.netrc` 加载且 API key 已配置时，说明认证正常，不需要 `wandb login --relogin`。只有更换账号、key 失效或权限变化时才需要重新登录。

不要将 `~/.netrc`、API key 或完整环境变量上传到 GitHub。

### 5.2 为什么学习率会变化

“学习率是超参数”并不意味着训练期间必须恒定。当前实验设置的超参数定义的是一条 schedule：

1. 从接近 0 开始 warmup，降低训练初期大梯度造成的不稳定。
2. 在约 2,500 step 达到峰值 `2e-5`。
3. 随训练逐渐衰减到 `2e-6`，让后期参数更新更细。

因此 W&B 中学习率变化是训练配置的预期行为。

### 5.3 常见训练指标含义

| 指标 | 含义 | 观察方式 |
|---|---|---|
| `action_loss` | 动作预测损失 | 主要优化目标，下降表示训练集拟合改善 |
| `total_loss` | 所有损失项的加权和 | 本次运行中与 action loss 相同 |
| `grad_norm` | 所有梯度的整体范数 | 长期异常增大可能表示训练不稳定 |
| `param_norm` | 模型参数整体范数 | 用于发现参数尺度漂移或发散 |
| `learning_rate` | 当前 step 实际使用的学习率 | 由 warmup/decay schedule 决定 |
| `step_time` | 每个训练 step 的耗时 | 首步包含编译初始化，不代表稳态速度 |
| `actions_mask_ratio` | 动作张量中参与/屏蔽位置的比例统计 | 用于确认 padding 和有效动作掩码是否稳定 |
| `final_mask_ratio` | 最终损失掩码覆盖比例 | 应与数据和序列结构保持一致 |
| `postfix_mask_ratio` | 后缀/action 区域的掩码比例 | 用于确认动作相关 token 区间构造是否稳定 |

### 5.4 W&B 能否下载 checkpoint

W&B run 会自动保存指标和少量运行文件，但不会自动上传本地 checkpoint。检查该全量微调运行时：

- run 文件中没有模型权重。
- `logged_artifacts` 和 `used_artifacts` 均为空。
- checkpoint 只存在于远端训练机器的 checkpoint 目录。

日志表明训练结束时保留了 10k、20k、30k、40k 和 49,999 等关键节点；45k checkpoint 按保留策略删除。若希望跨机器访问，需要显式上传 W&B Artifact、对象存储或使用 `rsync`，不能仅依赖 W&B 指标页面。

### 5.5 是否存在以前的 π0.5 LoRA 记录

本地搜索结果：

- 找到一个下载的 π0.5 LIBERO finetuned checkpoint 及其训练配置。
- 评估启动配置显示 `use_peft=False`，它不是本地 π0.5 LoRA adapter。
- 找到的本地 LoRA/QLoRA 训练记录属于 OpenVLA。
- 没有找到以前 π0.5 LoRA 的 W&B history、adapter checkpoint 或 loss 曲线。

因此不能把社区 π0.5 checkpoint 与“此前自行训练的 π0.5 LoRA”混为一谈，也不能在缺少原始 run 的情况下完成严格的 full fine-tune vs LoRA loss 对比。

## 6. 可复用的 W&B 检查方法

使用环境变量提供 run 路径，避免在脚本中硬编码账号：

```bash
export WANDB_RUN_PATH="entity/project/run_id"
python - <<'PY'
import os
import wandb

run = wandb.Api(timeout=60).run(os.environ["WANDB_RUN_PATH"])
print(run.name, run.state, run.url)
print("files:", [f.name for f in run.files()])
print("logged artifacts:", [a.qualified_name for a in run.logged_artifacts()])
PY
```

判断 checkpoint 能否从 W&B 拉取时，必须同时检查 run files 和 logged artifacts。日志中出现“Saving checkpoint”只证明训练机器写盘成功，不代表已经上传 W&B。

## 7. 本地成果索引

适合公开整理的自编脚本：

| 文件 | 用途 | 发布前处理 |
|---|---|---|
| `ablate_nsteps.sh` | 批量执行 π0.5 去噪步数消融 | 已使用环境变量接收路径和评估参数 |
| `bench_nsteps_latency.py` | 测量去噪步数与推理延迟 | 已参数化模型、step 和测量次数 |
| `bench_pi0_pi05.py` | π0/π0.5 速度与动作对比 | 已参数化模型、任务和 seed |
| `bench_smoothness.py` | 计算动作变化率和 jerk | 已修正成功判定并参数化 |
| `openvla_plan.md` | OpenVLA 学习路线 | 可公开 |
| `pi0_plan.md` | π0/π0.5 学习路线 | 已替换个人账号和绝对路径 |
| `openvla_resume.md` | 完整项目与简历素材 | 可按求职场景精简 |

原始模型、数据集、rollout 视频和 W&B 日志体积较大且可能包含内部路径，应保留在本地或私有存储，不进入公开仓库。

## 8. 经验与方法论

1. **先对齐评估协议再比较模型。** 图像预处理、等待步数、动作反归一化、chunk 长度和重规划频率都会改变成功率。
2. **模型 checkpoint 与 pre/postprocessor 必须成套。** 缺少归一化统计量时，即使权重能加载，输出也可能完全失效。
3. **小样本成功率只适合筛选。** 50 episodes 可快速排错；模型排名和细粒度差异应使用 500 episodes 或置信区间。
4. **训练 loss 不能替代 rollout。** loss 下降只说明优化有效，不能证明闭环控制成功。
5. **保存策略是训练系统的一部分。** 大模型训练中频繁合并完整权重会造成长时间停顿，应保存 adapter/训练状态并异步持久化。
6. **依赖版本需要写入实验记录。** OFT 的异常结果表明，错误依赖可能产生“能运行但结果错误”的静默失败。

## 9. 未完成工作

- 使用私有离线评估脚本评估最新 π0.5 全量微调 checkpoint，并输出与训练 step 对齐的指标。
- 找回以前 π0.5 LoRA 的准确 W&B run path 或 adapter 目录，再进行 full fine-tune vs LoRA 对比。
- 为最新全量微调增加 validation split、固定离线评估集和多个 checkpoint 的统一评测。
- 为 benchmark 脚本增加统一的 JSON/CSV 机器可读输出。
- 对去噪步数消融增加更多 seeds/episodes 和置信区间。
- 对 π0.5 的不同 checkpoint（如 10k/20k/30k/40k/49,999）做性能-训练步数曲线，避免只选最终 checkpoint。

## 10. 离职前保留建议

公开 GitHub 只保存本仓库中的脱敏文档、个人脚本和结果摘要。公司私有 fork、内部数据、远端路径、模型权重和原始日志应遵循公司制度处理，不应通过个人仓库带走。

若需要保留可复现能力，优先记录：上游开源 commit、Python/CUDA 版本、模型公开 ID、数据集公开版本、随机种子、评估 episode 数、重规划频率、去噪步数和汇总结果。这些信息比复制一个无法公开的大型工作目录更有价值。
