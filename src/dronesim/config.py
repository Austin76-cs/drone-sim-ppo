from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SimConfig:
    sim_hz: int = 500
    policy_hz: int = 50
    episode_seconds: float = 20.0
    model_xml_path: str = "assets/mjcf/quadrotor.xml"
    gravity: float = 9.81
    mass_jitter: float = 0.1
    thrust_noise_std: float = 0.01


@dataclass(slots=True)
class DroneConfig:
    mass_kg: float = 1.0
    arm_length_m: float = 0.12
    max_total_thrust_n: float = 20.0
    max_vertical_velocity_m_s: float = 2.0
    max_tilt_rad: float = 0.45
    max_body_rate_rad_s: float = 6.0
    motor_time_constant_s: float = 0.03
    vertical_velocity_kp: float = 0.30
    attitude_kp: float = 4.0
    body_rate_kp: float = 0.45
    yaw_rate_kp: float = 0.30


@dataclass(slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    num_envs: int = 8
    total_timesteps: int = 2_000_000


@dataclass(slots=True)
class RewardConfig:
    gate_proximity: float = 0.35
    gate_passage_bonus: float = 5.0
    progress: float = 0.40
    velocity_alignment: float = 0.20
    forward_speed: float = 0.30
    time_penalty: float = -0.002
    collision_penalty: float = -5.0
    control_effort: float = -0.05
    alive_bonus: float = 0.05
    gate_miss_penalty: float = 0.0
    approach_angle: float = 0.0


@dataclass(slots=True)
class TaskConfig:
    curriculum: str = "gate_race"
    curriculum_window: int = 12
    curriculum_min_episodes: int = 6
    intro_threshold: float = 0.70
    slalom_threshold: float = 0.64
    sprint_threshold: float = 0.58
    competition_threshold: float = 0.50
    gate_radius_m: float = 0.45
    sprint_gate_radius_m: float = 0.75
    gate_depth_m: float = 0.15
    gate_pass_margin_m: float = 0.12
    base_gate_spacing_m: float = 2.5


@dataclass(slots=True)
class EvalConfig:
    n_episodes: int = 20
    deterministic: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    sim: SimConfig
    drone: DroneConfig
    ppo: PPOConfig
    reward: RewardConfig
    task: TaskConfig
    eval: EvalConfig
    seed: int = 0
    device: str = "cpu"


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    values = {}
    for field_name in cls.__dataclass_fields__.keys():
        if field_name in data:
            values[field_name] = data[field_name]
    return cls(**values)


def load_config(path: Path) -> RuntimeConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)

    sim = _from_dict(SimConfig, raw.get("sim", {}))
    drone = _from_dict(DroneConfig, raw.get("drone", {}))
    ppo = _from_dict(PPOConfig, raw.get("ppo", {}))
    reward = _from_dict(RewardConfig, raw.get("reward", {}))
    task = _from_dict(TaskConfig, raw.get("task", {}))
    eval_cfg = _from_dict(EvalConfig, raw.get("eval", {}))
    seed = raw.get("runtime", {}).get("seed", 0)
    device = raw.get("runtime", {}).get("device", "cpu")
    return RuntimeConfig(
        sim=sim,
        drone=drone,
        ppo=ppo,
        reward=reward,
        task=task,
        eval=eval_cfg,
        seed=seed,
        device=device,
    )
