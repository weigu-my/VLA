"""在 LIBERO-Spatial 中对比 Pi0 与 Pi0.5 的推理速度和动作平滑度。"""

import argparse
import time

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi0-path", required=True, help="Pi0 checkpoint 路径或 Hugging Face 模型 ID")
    parser.add_argument("--pi05-path", required=True, help="Pi0.5 checkpoint 路径或 Hugging Face 模型 ID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed-runs", type=int, default=10)
    parser.add_argument("--smooth-episodes", type=int, default=2)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def run_model(label, policy_class, path: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 55}\n{label}  ({path.rstrip('/').split('/')[-1]})\n{'=' * 55}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    policy = policy_class.from_pretrained(path).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[args.task_id])
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    vec = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_spatial"][0]
    chunk_size = policy.config.chunk_size
    num_inference_steps = policy.config.num_inference_steps

    def get_batch(raw):
        observation = preprocess_observation(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in raw.items()}
        )
        observation["task"] = list(vec.call("task_description"))
        return preprocessor(env_preprocessor(observation))

    raw, _ = vec.reset(seed=args.seed)
    batch = get_batch(raw)

    # 每次重置动作队列，确保 select_action 触发完整 chunk 预测。
    with torch.inference_mode():
        for _ in range(3):
            policy.reset()
            policy.select_action(batch)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.speed_runs):
            policy.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            policy.select_action(batch)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)

    latency = float(np.mean(samples))
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print("[推理速度]")
    print(f"  单次 chunk 推理延迟 : {latency * 1000:.1f} ms  (chunk={chunk_size}, 去噪={num_inference_steps})")
    print(f"  推理频率             : {1 / latency:.2f} Hz")
    print(f"  有效动作吞吐         : {chunk_size / latency:.1f} actions/s")
    print(f"  峰值显存             : {peak_vram:.1f} GB")

    # 在真实 rollout 中记录实际执行动作，使用固定闭环重规划频率。
    policy.config.n_action_steps = args.replan_steps
    executed = []
    with torch.inference_mode():
        for episode in range(args.smooth_episodes):
            raw, _ = vec.reset(seed=args.seed + episode)
            policy.reset()
            for _ in range(args.max_steps):
                action = postprocessor(policy.select_action(get_batch(raw)))
                action = torch.as_tensor(action).float().squeeze().cpu().numpy()
                executed.append(action)
                env_action = action.reshape(1, -1) if action.ndim == 1 else action
                raw, _, terminated, truncated, _ = vec.step(env_action)
                if bool(np.any(terminated)) or bool(np.any(truncated)):
                    break

    actions = np.array(executed)
    first_difference = np.abs(np.diff(actions, axis=0))
    second_difference = np.abs(np.diff(actions, n=2, axis=0))
    print(f"[动作平滑度]  (执行 {len(actions)} 步, replan={args.replan_steps})")
    print(f"  L1 变化率 mean|Δa|   : {first_difference.mean():.4f}")
    print(f"  jerk mean|Δ²a|       : {second_difference.mean():.4f}")
    print(f"  运动维(0-5) L1       : {first_difference[:, :6].mean():.4f}")
    print(f"  夹爪维(6) L1         : {first_difference[:, 6].mean():.4f}")

    vec.close()
    del policy
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    models = [
        ("Pi0", PI0Policy, args.pi0_path),
        ("Pi0.5", PI05Policy, args.pi05_path),
    ]
    for label, policy_class, path in models:
        run_model(label, policy_class, path, args)


if __name__ == "__main__":
    main()
