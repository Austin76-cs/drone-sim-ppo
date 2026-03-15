"""Load trained model, run eval episodes, print metrics."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--normalize", type=str, default=None, help="Path to vec_normalize.pkl")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))

    def make_env():
        return DroneRaceEnv(config)

    vec_env = DummyVecEnv([make_env])
    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    crash_types: Counter[str] = Counter()
    all_gates: list[int] = []
    all_completions: list[float] = []
    all_rewards: list[float] = []
    successes = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_info: dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_info = infos[0]
            done = bool(dones[0])

        crash_type = ep_info.get("crash_type", "unknown")
        gates = ep_info.get("gates_cleared", 0)
        completion = ep_info.get("completion", 0.0)

        crash_types[crash_type] += 1
        all_gates.append(gates)
        all_completions.append(completion)
        all_rewards.append(ep_reward)
        if crash_type == "success":
            successes += 1

        print(f"  Episode {ep + 1:3d}: reward={ep_reward:7.2f}  gates={gates}  "
              f"completion={completion:.2f}  crash={crash_type}")

    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 60)
    print(f"{'Episodes':<25} {args.episodes:>10}")
    print(f"{'Success Rate':<25} {successes / args.episodes:>10.1%}")
    print(f"{'Mean Gates Cleared':<25} {np.mean(all_gates):>10.2f}")
    print(f"{'Mean Completion':<25} {np.mean(all_completions):>10.2f}")
    print(f"{'Mean Reward':<25} {np.mean(all_rewards):>10.2f}")
    print(f"{'Std Reward':<25} {np.std(all_rewards):>10.2f}")
    print("-" * 60)
    print("Crash Distribution:")
    for crash, count in crash_types.most_common():
        print(f"  {crash:<20} {count:>5}  ({count / args.episodes:.0%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
