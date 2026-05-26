# Pi0 / Pi0.5 学习方案：从 OpenVLA 进阶到流匹配 VLA

## Context

基于 OpenVLA 的学习经验（已完成 QLoRA 微调 15K 步，action accuracy 75%），进阶学习 Physical Intelligence 的 Pi0/Pi0.5 模型。重点理解流匹配（Flow Matching）动作生成 vs OpenVLA 的离散 token 生成的差异。

**硬件环境：** RTX 3090 24GB / 64GB RAM / 627GB 磁盘（与 OpenVLA 相同）

**重要限制：** Pi0 微调在 JAX 官方实现中需要 40-48GB 显存（即使 LoRA），RTX 3090 无法直接跑。方案中采用 LeRobot（PyTorch）路径，尝试在 24GB 内完成微调。

---

## Pi0 vs OpenVLA 核心对比

| 方面 | OpenVLA | Pi0 |
|------|---------|-----|
| VLM 骨干 | Prismatic (LLaMA-2 7B + SigLIP + DinoV2) | PaliGemma (3B) |
| 总参数量 | 7B | ~3.3B |
| 动作生成方式 | **离散 token**（256 bins，自回归） | **流匹配**（连续动作，扩散式迭代去噪） |
| 动作头 | 复用 LLM 词表 | 独立 Action Expert（~300M 参数，双向注意力） |
| 架构模式 | 单一 Transformer | VLM Expert + Action Expert（类 MoE 跨注意力） |
| 推理方式 | 逐 token 生成（7 次前向） | 迭代去噪（多步 flow matching） |

**Pi0.5 相比 Pi0 的改进：**
- 知识隔离（Knowledge Insulation）防止多任务训练时遗忘
- 使用网页数据提升开放世界物体识别能力
- 在未见过的新环境中泛化能力显著提升

---

## 阶段一：环境搭建（预计 1 小时）

### 1.1 安装 LeRobot（PyTorch 路径，推荐）

LeRobot 是 HuggingFace 的机器人学习框架，已原生集成 Pi0/Pi0.5。

```bash
cd /home/wujie/VLA
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi0]"
```

### 1.2 克隆 OpenPI（JAX 官方仓库，用于参考架构）

```bash
cd /home/wujie/VLA
git clone https://github.com/Physical-Intelligence/openpi.git
```

> 注意：OpenPI 是 JAX 实现，主要用于阅读代码理解架构。实际训练推理走 LeRobot。

---

## 阶段二：理解架构差异（预计 4-6 小时）

### 核心概念：Flow Matching vs 离散 Token

**OpenVLA 的方式（你已经理解的）：**
```
连续动作 [0.03, -0.15, ...] → 量化为 256 bins → LLM 自回归生成 7 个 token
```

**Pi0 的方式（流匹配）：**
```
随机噪声 → 迭代去噪（多步） → 连续动作轨迹
VLM 提供条件信息（图像+语言），Action Expert 负责去噪生成
```

### 必读文件（LeRobot 路径）

| 优先级 | 文件 | 内容 |
|--------|------|------|
| 1 | `lerobot/common/policies/pi0/modeling_pi0.py` | Pi0 策略核心：flow matching 前向传播 |
| 2 | `lerobot/common/policies/pi0/configuration_pi0.py` | 模型配置：理解各组件维度 |
| 3 | `lerobot/common/policies/pi05/modeling_pi05.py` | Pi0.5 的改进之处 |
| 4 | OpenPI: `openpi/models/pi0.py` | JAX 版本参考对照 |

### 关键概念笔记

- **Flow Matching**：将动作生成建模为从噪声到目标的"流"，通过学习速度场（velocity field）来生成轨迹。比 DDPM 训练更稳定、推理更快。
- **Action Expert**：独立的 ~300M 参数 Transformer，与 VLM 通过跨注意力交互。好处是动作生成有专门的参数空间，不与语言能力竞争。
- **Action Chunking**：Pi0 生成的是未来若干步的动作序列（action chunk），而非 OpenVLA 的单步动作。

---

## 阶段三：模型下载与推理验证（预计 1-2 小时）

### 3.1 下载模型

```python
# Pi0 和 Pi0.5 基础模型均可从 HuggingFace 下载
# lerobot/pi0_base (~4B 参数, ~8GB bf16)
# lerobot/pi05_base
# lerobot/pi05_libero_finetuned  （已微调版本，可直接评估）
```

### 3.2 编写推理测试脚本

创建 `/home/wujie/VLA/test_pi0_inference.py`：

```python
"""Pi0 推理测试：验证模型加载和动作生成"""
import torch
from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy
from lerobot.common.policies.pi0.configuration_pi0 import Pi0Config

# 加载预训练模型
policy = Pi0Policy.from_pretrained("lerobot/pi0_base")
policy.eval()
policy = policy.to("cuda")

# 打印显存占用
mem = torch.cuda.memory_allocated() / 1024**3
print(f"GPU memory: {mem:.1f}GB")

# 测试推理（具体输入格式需参考 LeRobot 文档）
print("Pi0 inference test passed!")
```

验证点：模型成功加载、显存占用合理（预计 ~8-10GB bf16）

### 3.3 对比推理方式

写一个脚本同时跑 OpenVLA 和 Pi0 的推理，对比：
- 推理速度（Pi0 的 flow matching 多步去噪 vs OpenVLA 的 7 次自回归）
- 输出格式（Pi0 输出连续动作 chunk vs OpenVLA 输出单步离散动作）
- 显存占用

---

## 阶段四：数据准备（预计 30 分钟）

### 复用已有 LIBERO 数据集

你已经下载了 LIBERO Spatial 数据集。LeRobot 可能需要不同的数据格式，需要做转换：

```bash
# 检查 LeRobot 是否有现成的 LIBERO 数据集
# 通常 LeRobot 有自己的数据集注册表
python -c "from lerobot.common.datasets.factory import make_dataset; help(make_dataset)"
```

如果 LeRobot 自带 LIBERO 数据加载器，可以直接使用；否则需要写转换脚本将 RLDS 格式转为 LeRobot 格式。

---

## 阶段五：微调训练（预计探索 2-4 小时，训练视显存而定）

### 5.1 显存挑战

Pi0 微调的显存需求：

| 配置 | 估计显存 | RTX 3090 可行性 |
|------|----------|----------------|
| 全参数微调 | ~70GB | ❌ |
| LoRA (OpenPI/JAX) | ~44GB | ❌ |
| LoRA (LeRobot/PyTorch) + bf16 + gradient checkpointing + batch=1 | ~18-24GB | ⚠️ 需要尝试 |
| QLoRA (如果支持) | ~12-16GB | ✅ 但目前无官方支持 |

### 5.2 尝试 LeRobot LoRA 微调

```bash
cd /home/wujie/VLA/lerobot

python lerobot/scripts/train.py \
  --policy.type=pi0 \
  --dataset.repo_id=lerobot/libero_spatial \
  --training.batch_size=1 \
  --policy.use_gradient_checkpointing=true \
  --training.lr=2e-5 \
  --training.steps=10000 \
  --wandb.project="pi0-rtx3090" \
  --wandb.entity="weigu-tsinghua-university"
```

### 5.3 如果显存不足的备选方案

1. **只做推理和评估**，不做微调 — 用官方已微调的 `lerobot/pi05_libero_finetuned`
2. **手动添加 QLoRA 支持** — 参考 OpenVLA 的经验，给 Pi0 的 PaliGemma 骨干加 BitsAndBytesConfig 4-bit 量化
3. **租用云 GPU** — 用 AutoDL/RunPod 的 A100 跑训练，本地跑推理评估

---

## 阶段六：评估对比（预计 2-3 小时）

### 6.1 在 LIBERO Spatial 上评估

使用已安装的 LIBERO 仿真环境，分别评估：

| 模型 | Checkpoint |
|------|-----------|
| OpenVLA (你微调的 15K) | `/home/wujie/VLA/runs/...-15000_chkpt` |
| Pi0 (预训练) | `lerobot/pi0_base` |
| Pi0.5 (LIBERO 微调版) | `lerobot/pi05_libero_finetuned` |

对比指标：
- 任务成功率（Success Rate）
- 推理速度（FPS）
- 动作平滑度（L1 变化率）

### 6.2 定性分析

- 可视化 Pi0 生成的动作轨迹 vs OpenVLA 的离散动作
- 观察 flow matching 的迭代去噪过程（不同去噪步数的动作质量）
- 对比两种架构在不同任务上的表现差异

---

## 阶段七：深度代码阅读（与训练并行）

### Pi0 架构精读路线

```
1. PaliGemma VLM backbone
   └── 图像编码（SigLIP）+ 语言编码
       └── 与 OpenVLA 的 Prismatic 对比

2. Action Expert
   └── 独立 Transformer 参数
   └── 双向注意力（vs LLM 的因果注意力）
   └── 跨注意力连接 VLM

3. Flow Matching
   └── 噪声调度（noise schedule）
   └── 速度场预测（velocity field）
   └── 训练目标：条件流匹配损失
   └── 推理：ODE 求解（去噪步数可调）

4. Pi0-FAST 变体
   └── FAST tokenization（DCT 频域压缩）
   └── 训练速度 5x 提升的原因
```

---

## 阶段八：进阶探索（可选）

| 方向 | 内容 | 预计时间 |
|------|------|----------|
| 去噪步数消融 | 测试不同推理步数（1/5/10/50）对成功率的影响 | 半天 |
| Pi0 vs Pi0-FAST | 对比扩散 vs DCT token 化的推理速度和精度 | 1 天 |
| 手动加 QLoRA | 给 Pi0 的 PaliGemma 加 4-bit 量化微调支持 | 1-2 天 |
| 跨模型迁移 | OpenVLA 微调数据能否提升 Pi0 在相同任务上的表现 | 1 天 |
| 论文精读 | Pi0 论文 (arXiv:2410.24164) + Pi0.5 blog | 1 天 |

---

## 磁盘空间预算

| 项目 | 大小 | 累计 |
|------|------|------|
| 已有（OpenVLA 环境 + 数据 + checkpoints） | ~60GB | 60GB |
| LeRobot 代码 + 依赖 | ~3GB | 63GB |
| OpenPI 代码（参考） | ~1GB | 64GB |
| Pi0 base 模型 | ~8GB | 72GB |
| Pi0.5 base 模型 | ~8GB | 80GB |
| Pi0.5 LIBERO 微调版 | ~8GB | 88GB |
| 微调 checkpoints | ~10GB | 98GB |
| **总计** | | **~100GB** |

---

## 关键参考资源

| 资源 | 链接 |
|------|------|
| OpenPI 官方仓库 | https://github.com/Physical-Intelligence/openpi |
| LeRobot 仓库 | https://github.com/huggingface/lerobot |
| Pi0 HuggingFace 模型 | https://huggingface.co/lerobot/pi0_base |
| Pi0.5 HuggingFace 模型 | https://huggingface.co/lerobot/pi05_base |
| Pi0.5 LIBERO 微调版 | https://huggingface.co/lerobot/pi05_libero_finetuned |
| Pi0 论文 | https://arxiv.org/abs/2410.24164 |
| Pi0.5 博客 | https://www.pi.website/blog/pi05 |
| LeRobot Pi0 文档 | https://huggingface.co/docs/lerobot/pi0 |
| LeRobot Pi0.5 文档 | https://huggingface.co/docs/lerobot/en/pi05 |

---

## 验证清单

- [ ] 阶段三：Pi0 推理脚本输出连续动作轨迹，显存 < 12GB
- [ ] 阶段五：微调训练 loss 下降（如显存允许）
- [ ] 阶段六：LIBERO Spatial 评估成功率对比表（OpenVLA vs Pi0 vs Pi0.5）
- [ ] 阶段七：能清晰解释 Flow Matching 与离散 Token 化的优劣
