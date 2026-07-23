"""在相同任务和 seed 下，对两模型共同成功的 episode 计算动作平滑度。"""

import argparse

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
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000, 1006)))
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=280)
    return parser.parse_args()


def info_success(info) -> bool:
    if isinstance(info, dict) and "is_success" in info:
        return bool(np.any(info["is_success"]))
    return False


def collect(label, policy_class, path: str, args: argparse.Namespace):
    print(f"加载 {label} ...", flush=True)
    policy = policy_class.from_pretrained(path).to(args.device).eval()
    policy.config.n_action_steps = args.replan_steps
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_cfg = LiberoEnv(task="libero_spatial", task_ids=args.task_ids)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

    def get_batch(raw, vec):
        observation = preprocess_observation(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in raw.items()}
        )
        observation["task"] = list(vec.call("task_description"))
        return preprocessor(env_preprocessor(observation))

    results = {}
    with torch.inference_mode():
        for task_id in args.task_ids:
            vec = envs["libero_spatial"][task_id]
            for seed in args.seeds:
                raw, _ = vec.reset(seed=seed)
                policy.reset()
                actions = []
                success = False
                for _ in range(args.max_steps):
                    action = postprocessor(policy.select_action(get_batch(raw, vec)))
                    action = torch.as_tensor(action).float().squeeze().cpu().numpy()
                    actions.append(action)
                    env_action = action.reshape(1, -1) if action.ndim == 1 else action
                    raw, _, terminated, truncated, info = vec.step(env_action)
                    success = info_success(info) or bool(np.any(terminated))
                    if success or bool(np.any(terminated)) or bool(np.any(truncated)):
                        break
                results[(task_id, seed)] = (success, np.array(actions))
            vec.close()

    del policy
    torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    models = [
        ("Pi0", PI0Policy, args.pi0_path),
        ("Pi0.5", PI05Policy, args.pi05_path),
    ]
    results = {
        label: collect(label, policy_class, path, args)
        for label, policy_class, path in models
    }
    common_successes = [
        key
        for key in results["Pi0"]
        if results["Pi0"][key][0] and results["Pi0.5"][key][0]
    ]
    if not common_successes:
        raise RuntimeError("没有两模型共同成功的 episode，无法进行受控平滑度比较")

    print(f"\n两模型共同成功的 episode 数: {len(common_successes)} / {len(results['Pi0'])}")
    print(f"{'指标':<24}{'Pi0':>10}{'Pi0.5':>10}")
    print("-" * 44)

    def print_metric(label, function) -> None:
        pi0_value = np.mean([function(results["Pi0"][key][1]) for key in common_successes])
        pi05_value = np.mean([function(results["Pi0.5"][key][1]) for key in common_successes])
        print(f"{label:<24}{pi0_value:>10.4f}{pi05_value:>10.4f}")

    print_metric("L1变化率 mean|Δa|", lambda actions: np.abs(np.diff(actions, axis=0)).mean())
    print_metric("jerk mean|Δ²a|", lambda actions: np.abs(np.diff(actions, n=2, axis=0)).mean())
    print_metric("运动维(0-5) L1", lambda actions: np.abs(np.diff(actions, axis=0))[:, :6].mean())
    print_metric("夹爪维(6) L1", lambda actions: np.abs(np.diff(actions, axis=0))[:, 6].mean())
    print_metric("平均成功步数", len)


if __name__ == "__main__":
    main()
