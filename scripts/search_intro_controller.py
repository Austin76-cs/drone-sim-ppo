from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage


@dataclass
class ControllerParams:
    roll_sign: float
    pitch_sign: float
    kz_p: float
    kz_d: float
    ky_p: float
    ky_d: float
    kx_p: float
    kx_d: float
    kyaw_p: float
    pitch_limit_scale: float


def policy_action(env: DroneRaceEnv, params: ControllerParams) -> np.ndarray:
    assert env.state is not None
    obs = env._build_obs()
    cur_gate_body = obs[12:15].astype(np.float64)
    x_b, y_b, z_b = cur_gate_body
    state = env.state

    forward_ref = max(float(x_b), 0.2)
    yaw_err = math.atan2(float(y_b), forward_ref)

    vz_cmd = params.kz_p * z_b - params.kz_d * float(state.vel[2])
    roll_cmd = params.roll_sign * (params.ky_p * y_b - params.ky_d * float(state.omega[0]))
    pitch_target = min(max(x_b, 0.0), params.pitch_limit_scale)
    pitch_cmd = params.pitch_sign * (params.kx_p * pitch_target - params.kx_d * float(state.omega[1]))
    yaw_cmd = params.kyaw_p * yaw_err

    return np.clip(np.array([vz_cmd, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float32), -1.0, 1.0)


def evaluate(params: ControllerParams, episodes: int = 8) -> tuple[float, float, float, float]:
    config = load_config(Path("configs/default.toml"))
    config.device = "cpu"
    config.sim.backend_device = "cpu"
    env = DroneRaceEnv(config)
    env.stage_controller.force_stage(CurriculumStage.INTRO)

    rewards: list[float] = []
    gates: list[int] = []
    completions: list[float] = []
    successes = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=config.seed + ep * 100)
            done = False
            total_reward = 0.0
            info: dict[str, object] = {}
            while not done:
                action = policy_action(env, params)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            gates.append(int(info.get("gates_cleared", 0)))
            completions.append(float(info.get("completion", 0.0)))
            if str(info.get("crash_type", "")) == "success":
                successes += 1
    finally:
        env.close()

    return (
        successes / max(episodes, 1),
        float(np.mean(gates)),
        float(np.mean(completions)),
        float(np.mean(rewards)),
    )


def sample_params(rng: random.Random) -> ControllerParams:
    return ControllerParams(
        roll_sign=rng.choice([-1.0, 1.0]),
        pitch_sign=rng.choice([-1.0, 1.0]),
        kz_p=rng.uniform(0.4, 2.0),
        kz_d=rng.uniform(0.0, 0.8),
        ky_p=rng.uniform(0.3, 1.6),
        ky_d=rng.uniform(0.0, 0.8),
        kx_p=rng.uniform(0.1, 1.2),
        kx_d=rng.uniform(0.0, 0.8),
        kyaw_p=rng.uniform(0.2, 1.5),
        pitch_limit_scale=rng.uniform(0.5, 2.8),
    )


def main() -> None:
    rng = random.Random(7)
    best = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for i in range(80):
        params = sample_params(rng)
        score = evaluate(params, episodes=6)
        if score > best_score:
            best = params
            best_score = score
            print(f"iter={i:03d} best={best_score} params={best}")
    print(f"\nfinal best={best_score} params={best}")


if __name__ == "__main__":
    main()
