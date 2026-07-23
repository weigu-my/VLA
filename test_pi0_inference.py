"""Pi0 推理测试：验证模型加载、显存占用和动作生成"""
import argparse
import time

import torch
from lerobot.policies.pi0 import PI0Policy


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model-path",
    default="lerobot/pi0_base",
    help="Pi0 checkpoint 路径或 Hugging Face 模型 ID",
)
parser.add_argument("--device", default="cuda")
parser.add_argument("--runs", type=int, default=5, help="测速重复次数")
args = parser.parse_args()

print("=" * 60)
print("Pi0 推理测试")
print("=" * 60)

# --- 1. 加载模型 ---
print("\n[1] 加载 Pi0 模型 (float32)...")
torch.cuda.reset_peak_memory_stats()

policy = PI0Policy.from_pretrained(args.model_path)
policy.to(device=args.device)
policy.eval()

mem_after_load = torch.cuda.max_memory_allocated() / 1024**3
print(f"    模型加载后峰值显存: {mem_after_load:.1f} GB")

# --- 2. 构造模拟输入 ---
print("\n[2] 构造模拟输入...")

# pi0_base 期望 3 个摄像头 (224x224) + 32维状态 + 语言指令
batch = {}
batch["observation.images.base_0_rgb"] = torch.randn(1, 3, 224, 224, device=args.device)
batch["observation.images.left_wrist_0_rgb"] = torch.randn(1, 3, 224, 224, device=args.device)
batch["observation.images.right_wrist_0_rgb"] = torch.randn(1, 3, 224, 224, device=args.device)
batch["observation.state"] = torch.randn(1, 32, device=args.device)

# 语言指令 tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
text = "pick up the red block and place it on the table"
tokens = tokenizer(text, padding="max_length", max_length=48, truncation=True, return_tensors="pt")
batch["observation.language.tokens"] = tokens["input_ids"].to(args.device)
batch["observation.language.attention_mask"] = tokens["attention_mask"].to(args.device).bool()

print(f"    图像: 3x [1, 3, 224, 224]")
print(f"    状态: [1, 32]")
print(f"    语言: \"{text}\"")

# --- 3. 推理测试 ---
print("\n[3] 运行推理...")
torch.cuda.reset_peak_memory_stats()

# 预热
with torch.no_grad():
    action = policy.select_action(batch)

mem_after_infer = torch.cuda.max_memory_allocated() / 1024**3
print(f"    推理后峰值显存: {mem_after_infer:.1f} GB")
print(f"    输出动作形状: {action.shape}")
print(f"    输出动作范围: [{action.min().item():.4f}, {action.max().item():.4f}]")
print(f"    输出动作均值: {action.mean().item():.4f}")

# 测速
policy.reset()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(args.runs):
    policy.reset()
    with torch.no_grad():
        _ = policy.select_action(batch)
    torch.cuda.synchronize()
t1 = time.time()
avg_time = (t1 - t0) / args.runs
print(f"\n    平均推理时间: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
print(f"    去噪步数: {policy.config.num_inference_steps}")
print(f"    动作 chunk 大小: {policy.config.chunk_size}")

# --- 4. 模型架构摘要 ---
print("\n[4] 模型架构摘要:")
total_params = sum(p.numel() for p in policy.parameters())
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"    总参数量: {total_params / 1e9:.2f}B")
print(f"    可训练参数: {trainable_params / 1e9:.2f}B")
print(f"    VLM 骨干: PaliGemma ({policy.config.paligemma_variant})")
print(f"    Action Expert: {policy.config.action_expert_variant}")
print(f"    数据类型: {policy.config.dtype}")

print("\n" + "=" * 60)
print("Pi0 推理测试完成!")
print("=" * 60)
