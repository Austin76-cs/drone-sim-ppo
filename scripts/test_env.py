"""Smoke test: create env, run random actions, print shapes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from gymnasium.utils.env_checker import check_env

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv


def main() -> None:
    config = load_config(Path("configs/default.toml"))
    env = DroneRaceEnv(config)

    print("=== Gymnasium check_env ===")
    check_env(env, warn=True, skip_render_check=True)
    print("check_env passed!")

    print("\n=== Smoke test: 100 random steps ===")
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"action space: {env.action_space}")
    print(f"obs space: {env.observation_space}")
    print(f"initial info: {info}")

    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}: crash_type={info.get('crash_type')}")
            obs, info = env.reset()

    print(f"\nFinal obs: {obs}")
    print(f"Total reward over 100 steps: {total_reward:.3f}")
    print(f"Gates cleared: {info.get('gates_cleared', 0)}")
    print(f"Curriculum stage: {info.get('curriculum_stage', 0)}")
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
