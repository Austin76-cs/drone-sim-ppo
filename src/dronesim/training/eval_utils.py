from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dronesim.config import RuntimeConfig
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage
from dronesim.training.torch_ppo import LoadedTorchPolicy, load_torch_policy


@dataclass(slots=True)
class EvalSummary:
    episodes: int
    success_rate: float
    mean_gates_cleared: float
    mean_completion: float
    mean_reward: float
    std_reward: float
    mean_episode_steps: float
    median_episode_steps: float
    crash_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_eval_env(
    config: RuntimeConfig,
    *,
    stage: int | None = None,
    sim_device: str = "cpu",
) -> DroneRaceEnv:
    eval_config = copy.deepcopy(config)
    eval_config.sim.backend_device = sim_device
    eval_config.device = "cpu"
    env = DroneRaceEnv(eval_config)
    if stage is not None:
        env.stage_controller.force_stage(CurriculumStage(stage))
    return env


def evaluate_torch_policy(
    policy: LoadedTorchPolicy,
    *,
    episodes: int,
    stage: int | None = None,
    deterministic: bool = True,
    sim_device: str = "cpu",
) -> EvalSummary:
    env = make_eval_env(policy.config, stage=stage, sim_device=sim_device)
    crash_types: Counter[str] = Counter()
    rewards: list[float] = []
    gates: list[int] = []
    completions: list[float] = []
    steps: list[int] = []
    successes = 0

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=env.config.seed + ep * 100)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info: dict[str, object] = {}

            while not done:
                action = policy.predict(obs, deterministic=deterministic)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1
                done = bool(terminated or truncated)

            crash_type = str(info.get("crash_type", "unknown"))
            gate_count = int(info.get("gates_cleared", 0))
            completion = float(info.get("completion", 0.0))

            crash_types[crash_type] += 1
            rewards.append(ep_reward)
            gates.append(gate_count)
            completions.append(completion)
            steps.append(ep_steps)
            if crash_type == "success":
                successes += 1
    finally:
        env.close()

    return EvalSummary(
        episodes=episodes,
        success_rate=float(successes / max(episodes, 1)),
        mean_gates_cleared=float(np.mean(gates)),
        mean_completion=float(np.mean(completions)),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_episode_steps=float(np.mean(steps)),
        median_episode_steps=float(np.median(steps)),
        crash_distribution=dict(crash_types),
    )


def evaluate_torch_checkpoint(
    model_path: str | Path,
    *,
    episodes: int,
    stage: int | None = None,
    deterministic: bool = True,
    policy_device: str = "cpu",
    sim_device: str = "cpu",
    config_path: str | None = None,
) -> tuple[LoadedTorchPolicy, EvalSummary]:
    policy = load_torch_policy(model_path, config_path=config_path, device=policy_device)
    summary = evaluate_torch_policy(
        policy,
        episodes=episodes,
        stage=stage,
        deterministic=deterministic,
        sim_device=sim_device,
    )
    return policy, summary
