from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DroneState:
    pos: NDArray[np.float64]       # (3,)
    vel: NDArray[np.float64]       # (3,)
    euler: NDArray[np.float64]     # (3,) roll, pitch, yaw
    omega: NDArray[np.float64]     # (3,) body rates
    motor: NDArray[np.float64]     # (4,) current motor commands


@dataclass(slots=True, frozen=True)
class GateSpec:
    center: NDArray[np.float64]    # (3,)
    normal: NDArray[np.float64]    # (3,)
    radius_m: float
    depth_m: float


@dataclass(slots=True)
class RewardInfo:
    total: float = 0.0
    gate_proximity: float = 0.0
    gate_passage: float = 0.0
    progress: float = 0.0
    velocity_alignment: float = 0.0
    lateral_velocity_penalty: float = 0.0
    attitude_stability: float = 0.0
    angular_rate_stability: float = 0.0
    time_penalty: float = 0.0
    collision_penalty: float = 0.0
    control_effort: float = 0.0
    alive_bonus: float = 0.0


@dataclass(slots=True)
class EpisodeMetrics:
    success: bool = False
    crash_type: str = "none"
    gates_cleared: int = 0
    total_gates: int = 0
    completion: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
