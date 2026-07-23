"""测量 Pi0.5 不同去噪步数下的单次 chunk 推理延迟。"""

import argparse
import time

import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Pi0.5 checkpoint 路径或 Hugging Face 模型 ID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 50])
    parser.add_argument("--warmup-runs", type=int, default=4)
    parser.add_argument("--measure-runs", type=int, default=15)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=[args.task_id])
    policy = PI05Policy.from_pretrained(args.model_path).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        args.model_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    vec = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_spatial"][0]
    raw, _ = vec.reset(seed=args.seed)
    observation = preprocess_observation(
        {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in raw.items()}
    )
    observation["task"] = list(vec.call("task_description"))
    batch = preprocessor(env_preprocessor(observation))
    vec.close()

    latencies: dict[int, float] = {}
    with torch.inference_mode():
        for num_steps in args.steps:
            policy.config.num_inference_steps = num_steps
            for _ in range(args.warmup_runs):
                policy.reset()
                policy.select_action(batch)

            torch.cuda.synchronize()
            samples = []
            for _ in range(args.measure_runs):
                policy.reset()
                torch.cuda.synchronize()
                start = time.perf_counter()
                policy.select_action(batch)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            latencies[num_steps] = float(np.mean(samples)) * 1000

    baseline = latencies.get(10)
    print(f"{'去噪步数':<10}{'延迟(ms)':>12}{'频率(Hz)':>12}{'相对10步':>12}")
    print("-" * 46)
    for num_steps in args.steps:
        latency = latencies[num_steps]
        relative = baseline / latency if baseline is not None else float("nan")
        print(f"{num_steps:<10}{latency:>11.1f}{1000 / latency:>12.2f}{relative:>11.2f}x")


if __name__ == "__main__":
    main()
