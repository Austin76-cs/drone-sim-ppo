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
    rot_matrix: NDArray[np.float64] = None  # (3,3) rotation matrix, set by sim


@dataclass(slots=True, frozen=True)
class GateSpec:
    center: NDArray[np.float64]    # (3,)
    normal: NDArray[np.float64]    # (3,)
    radius_m: float                # half-width of square gate (pass radius)
    depth_m: float
    width_m: float = 0.0          # full width of square gate (0 = use 2*radius_m)
    height_m: float = 0.0         # full height of square gate (0 = use 2*radius_m)


@dataclass(slots=True)
class RewardInfo:
    total: float = 0.0
    gate_proximity: float = 0.0
    gate_passage: float = 0.0
    progress: float = 0.0
    velocity_alignment: float = 0.0
    forward_speed: float = 0.0
    time_penalty: float = 0.0
    collision_penalty: float = 0.0
    control_effort: float = 0.0
    alive_bonus: float = 0.0
    gate_miss: float = 0.0
    approach_angle: float = 0.0
    gate_centering: float = 0.0


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
