# OpenVLA 复现方案：从零开始学习 VLA 模型

## Context

目标是在单卡 RTX 3090 (24GB) 上复现 OpenVLA，快速掌握 VLA（Vision-Language-Action）模型的架构、训练流程和仿真评估方法。由于硬件限制，采用 QLoRA（4-bit 量化 + LoRA）策略，基于官方预训练权重进行微调，而非从头训练。

**硬件环境：** RTX 3090 24GB / 64GB RAM / 28 CPU cores / 627GB 磁盘

---

## 阶段一：环境搭建（预计 1-2 小时）

### 1.1 创建 Conda 环境

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 克隆仓库并安装依赖

```bash
cd "${VLA_ROOT:-$HOME/VLA}"
git clone https://github.com/openvla/openvla.git
cd openvla && pip install -e .

# QLoRA 必需
pip install bitsandbytes>=0.43.0 peft>=0.10.0

# Flash Attention 2（加速注意力计算，降低显存）
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# RLDS 数据集读取需要 TensorFlow（用 CPU 版本避免与 PyTorch CUDA 冲突）
pip install tensorflow-cpu tensorflow-datasets

# 训练监控
pip install wandb
```

### 1.3 克隆 OpenVLA-OFT（后续对比实验用）

```bash
cd "${VLA_ROOT:-$HOME/VLA}"
git clone https://github.com/moojink/openvla-oft.git
```

---

## 阶段二：理解代码架构（预计 4-6 小时）

### OpenVLA 三大核心组件

| 组件 | 作用 | 关键文件 |
|------|------|----------|
| 视觉编码器 | SigLIP + DinoV2 双骨干融合，提取图像特征 | `prismatic/models/backbones/vision/` |
| 投影器 | 将视觉特征映射到 LLM 输入空间 | `prismatic/models/vlms/prismatic.py` |
| 语言模型 | Llama 2 7B，预测 token 化的动作 | `prismatic/models/backbones/llm/` |

### 必读文件（按优先级排序）

1. **`prismatic/vla/action_tokenizer.py`** — 核心创新：连续 7-DoF 动作离散化为 256 个 bin，复用 Llama 词表中最少使用的 256 个 token
2. **`prismatic/models/vlms/prismatic.py`** — 前向传播流程：图像 → 双视觉编码器 → 特征拼接 → 投影 → 与文本 token 拼接 → LLM 推理
3. **`vla-scripts/finetune.py`** — 微调入口：重点关注 `FinetuneConfig` 数据类、`use_quantization` 标志、LoRA 配置、训练循环
4. **`prismatic/vla/datasets/rlds/oxe/configs.py`** — 数据集注册表，理解如何添加新数据集
5. **`prismatic/vla/datasets/rlds/oxe/transforms.py`** — 每个数据集的预处理变换

### 关键概念笔记

- **动作 Token 化**：每个动作维度独立量化到 0-255，边界由训练数据的 1%-99% 分位数决定（而非 min-max，避免异常值影响）
- **LoRA 微调**：仅训练 1.4% 参数，`lora_rank=32`，作用在所有线性层
- **OFT 改进**：用 MLP 回归头替代自回归 token 生成，推理速度提升 25-50x

---

## 阶段三：模型下载与推理验证（预计 1 小时）

### 3.1 编写推理测试脚本

创建 `${VLA_ROOT}/test_inference.py`：

```python
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# 4-bit 量化加载，显存占用 ~5-6GB
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# 测试推理
image = Image.new("RGB", (224, 224), color="white")
prompt = "In: What action should the robot take to pick up the red block?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(f"Predicted action (7-DoF): {action}")
# 预期输出：7 个浮点数 [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

验证点：模型成功加载、推理正常、显存占用合理（~6-8GB）

---

## 阶段四：数据准备（预计 30 分钟）

### 下载 LIBERO Spatial 数据集（~2.5GB）

推荐使用 LIBERO，因为评估流程完善、有对比基准。

```python
import os

from huggingface_hub import snapshot_download
snapshot_download(
    "openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir=f"{os.environ['VLA_ROOT']}/datasets/modified_libero_rlds",
    allow_patterns=["libero_spatial_no_noops/*"],
)
```

数据格式：RLDS（基于 TFRecord），每条轨迹包含 image observation、language instruction、7-DoF action。

---

## 阶段五：QLoRA 微调训练（预计 3-6 小时 / 20K steps）

### 5.1 启动训练

```bash
cd "${VLA_ROOT:-$HOME/VLA}/openvla"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "${VLA_ROOT:-$HOME/VLA}/datasets/modified_libero_rlds" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "${VLA_ROOT:-$HOME/VLA}/runs" \
  --use_quantization True \
  --use_lora True \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --max_steps 20000 \
  --save_steps 5000 \
  --image_aug True \
  --wandb_project "openvla-rtx3090"
```

### 显存预算

| 配置 | 估计显存 |
|------|----------|
| 4-bit QLoRA, batch=2 | ~16-18GB |
| 4-bit QLoRA, batch=1 | ~12-14GB |

### 5.2 如果显存不足的备选方案

1. 降低 `batch_size` 为 1，`grad_accumulation_steps` 改为 16
2. 在 `finetune.py` 中添加梯度检查点：
   ```python
   vla.enable_input_require_grads()
   vla.gradient_checkpointing_enable()
   ```
   （牺牲 ~30% 速度换取 ~40% 显存节省）

---

## 阶段六：仿真评估（预计 2-3 小时）

### 6.1 安装 LIBERO 仿真环境

```bash
cd "${VLA_ROOT:-$HOME/VLA}"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

cd openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt

# 无头服务器需要虚拟显示
sudo apt install xvfb
```

### 6.2 运行评估

```bash
xvfb-run -a python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${VLA_ROOT:-$HOME/VLA}/runs/checkpoint-20000" \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 10
```

评估指标：每个任务的成功率（Success Rate），期望微调后在 LIBERO Spatial 上达到 70%+ 成功率。

### 6.3 备选：简单离线评估

如果 LIBERO 仿真环境搭建遇到问题，可以先做离线动作预测评估：
- 在留出的测试集上计算预测动作与真实动作的 L1 误差
- 按动作维度分别报告误差

---

## 阶段七：进阶探索（可选）

| 方向 | 内容 | 预计时间 |
|------|------|----------|
| LoRA rank 消融 | rank=8/16/32/64，对比 loss 曲线 | 1 天 |
| OFT 回归头 | 对比自回归 vs L1 回归的推理速度和精度 | 1 天 |
| 扩散动作头 | OFT 中 `use_diffusion=True` | 1 天 |
| 自定义数据集 | 创建自己的 RLDS 数据集并注册 | 2 天 |
| 视觉编码器分析 | 可视化 SigLIP vs DinoV2 的注意力图 | 1 天 |

---

## 磁盘空间预算

| 项目 | 大小 | 累计 |
|------|------|------|
| Conda 环境 + 代码仓库 | ~5GB | 5GB |
| openvla-7b 模型缓存 | ~15GB | 20GB |
| LIBERO Spatial 数据集 | ~2.5GB | 22.5GB |
| 训练 checkpoint（LoRA 适配器） | ~0.5GB | 23GB |
| LIBERO 仿真环境 | ~2GB | 25GB |
| **总计** | | **~25GB** |

---

## 常见问题预案

1. **Flash Attention 编译失败**：去掉 `attn_implementation="flash_attention_2"` 参数，仅影响速度不影响功能
2. **RLDS 数据加载报错**：确保安装了 `tensorflow-cpu` 和 `tensorflow-datasets`
3. **第一次前向传播 OOM**：依次尝试 batch_size=1 → 梯度检查点 → 减小 `shuffle_buffer_size`
4. **加载 QLoRA checkpoint**：需先以 4-bit 加载基座模型，再叠加 LoRA 适配器；或用 `merge_and_unload()` 合并权重

---

## 验证方式

- [ ] 阶段三：推理脚本输出 7 维动作向量，显存 < 10GB
- [ ] 阶段五：训练 loss 持续下降，无 OOM
- [ ] 阶段六：LIBERO Spatial 评估成功率 > 50%（基线验证通过）
