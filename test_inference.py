from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

print("Loading processor...")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# 4-bit 量化加载，显存占用 ~5-6GB
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print("Loading model (4-bit quantized)...")
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)

# 打印显存占用
mem_alloc = torch.cuda.memory_allocated() / 1024**3
mem_reserved = torch.cuda.memory_reserved() / 1024**3
print(f"GPU memory - allocated: {mem_alloc:.1f}GB, reserved: {mem_reserved:.1f}GB")

# 测试推理
print("Running inference...")
image = Image.new("RGB", (224, 224), color="white")
prompt = "In: What action should the robot take to pick up the red block?\nOut:"
inputs = processor(prompt, image)
# 对 4-bit 模型，手动将输入张量移到 GPU
inputs = {k: v.to("cuda:0", dtype=torch.bfloat16) if v.dtype in (torch.float32, torch.float16, torch.bfloat16) else v.to("cuda:0") for k, v in inputs.items()}
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(f"Predicted action (7-DoF): {action}")
# 预期输出：7 个浮点数 [dx, dy, dz, droll, dpitch, dyaw, gripper]

print("\nInference test PASSED!")
