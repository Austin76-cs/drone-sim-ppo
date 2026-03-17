from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from dronesim.config import RewardConfig
from dronesim.sim.env import euler_to_rotation_matrix
from dronesim.types import DroneState, GateSpec, RewardInfo


def _exp_score(error: float, scale: float) -> float:
    return math.exp(-error / max(scale, 1e-3))


def gate_relative_geometry(
    state: DroneState, gate: GateSpec
) -> tuple[float, float, float]:
    """Returns (forward_error, lateral_error, alignment_error)."""
    rel = gate.center - state.pos
    forward_error = float(np.dot(rel, gate.normal))
    lateral = rel - gate.normal * forward_error
    lateral_error = float(np.linalg.norm(lateral))

    forward_ref = max(float(rel[0]), 1e-4)
    yaw_error = float(np.arctan2(rel[1], forward_ref))
    pitch_error = float(np.arctan2(rel[2], forward_ref))
    alignment_error = math.sqrt(yaw_error * yaw_error + pitch_error * pitch_error)
    return forward_error, lateral_error, alignment_error


def gate_passed(
    state: DroneState,
    gate: GateSpec,
    gate_pass_margin_m: float,
    prev_forward_error: float | None = None,
) -> bool:
    forward_error, lateral_error, _ = gate_relative_geometry(state, gate)
    if forward_error > gate_pass_margin_m or lateral_error > gate.radius_m:
        return False
    if prev_forward_error is None:
        return True
    return prev_forward_error > gate_pass_margin_m


def body_frame_gate(state: DroneState, gate: GateSpec) -> NDArray[np.float64]:
    """Get gate position relative to drone in body frame."""
    rel_world = gate.center - state.pos
    rot = euler_to_rotation_matrix(state.euler)
    return rot.T @ rel_world


# --- Modular reward components ---

def gate_proximity_reward(state: DroneState, gate: GateSpec, scale: float) -> float:
    _, lateral_error, _ = gate_relative_geometry(state, gate)
    return _exp_score(lateral_error, scale)


def centering_factor(state: DroneState, gate: GateSpec) -> float:
    return gate_proximity_reward(state, gate, gate.radius_m)


def gate_passage_reward(state: DroneState, gate: GateSpec, margin: float) -> float:
    if gate_passed(state, gate, margin):
        return 1.0
    return 0.0


def progress_reward(
    prev_forward_error: float, curr_forward_error: float, gate_radius: float
) -> float:
    delta = max(0.0, prev_forward_error - curr_forward_error)
    return min(1.0, delta / max(0.5, gate_radius * 2.0))


def velocity_alignment_reward(
    state: DroneState, gate: GateSpec, scale: float
) -> float:
    """Reward for flying toward the gate — dot(velocity, direction_to_gate)."""
    rel = gate.center - state.pos
    dist = float(np.linalg.norm(rel))
    if dist < 1e-4:
        return 1.0
    direction = rel / dist
    forward_vel = float(np.dot(state.vel, direction))
    # Positive when flying toward gate, clipped to [0, scale]
    return np.clip(forward_vel / max(scale, 1e-3), 0.0, 1.0)


def lateral_velocity_penalty_reward(state: DroneState, gate: GateSpec, scale: float) -> float:
    lateral_vel = state.vel - gate.normal * float(np.dot(state.vel, gate.normal))
    return np.clip(float(np.linalg.norm(lateral_vel)) / max(scale, 1e-3), 0.0, 1.0)


def attitude_stability_reward(state: DroneState) -> float:
    tilt_mag = float(np.linalg.norm(state.euler[:2]))
    return np.clip(tilt_mag / 0.6, 0.0, 1.0)


def angular_rate_stability_reward(state: DroneState) -> float:
    rate_mag = float(np.linalg.norm(state.omega))
    return np.clip(rate_mag / 8.0, 0.0, 1.0)


def time_penalty_reward() -> float:
    return 1.0  # caller multiplies by weight (which is negative)


def collision_penalty_reward() -> float:
    return 1.0  # caller multiplies by weight


def control_effort_reward(action: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(action) / math.sqrt(max(len(action), 1)))


def alive_bonus_reward() -> float:
    return 1.0


def compute_total_reward(
    state: DroneState,
    action: NDArray[np.float64],
    gate: GateSpec,
    prev_forward_error: float,
    cfg: RewardConfig,
    terminated: bool,
    gate_pass_margin: float,
) -> RewardInfo:
    """Compute weighted sum of all reward components."""
    curr_forward_error, _, _ = gate_relative_geometry(state, gate)

    r_proximity = gate_proximity_reward(state, gate, gate.radius_m)
    r_passage = float(gate_passed(state, gate, gate_pass_margin, prev_forward_error))
    r_progress = progress_reward(prev_forward_error, curr_forward_error, gate.radius_m)
    r_align = velocity_alignment_reward(state, gate, 0.35)
    r_lat_vel = lateral_velocity_penalty_reward(state, gate, 1.2)
    r_attitude = attitude_stability_reward(state)
    r_rates = angular_rate_stability_reward(state)
    r_time = time_penalty_reward()
    r_collision = collision_penalty_reward() if terminated else 0.0
    r_effort = control_effort_reward(action)
    r_alive = alive_bonus_reward()

    total = (
        cfg.gate_proximity * r_proximity
        + cfg.gate_passage_bonus * r_passage
        + cfg.progress * r_progress
        + cfg.velocity_alignment * r_align
        + cfg.lateral_velocity_penalty * r_lat_vel
        + cfg.attitude_stability * r_attitude
        + cfg.angular_rate_stability * r_rates
        + cfg.time_penalty * r_time
        + cfg.collision_penalty * r_collision
        + cfg.control_effort * r_effort
        + cfg.alive_bonus * r_alive
    )

    return RewardInfo(
        total=total,
        gate_proximity=r_proximity,
        gate_passage=r_passage,
        progress=r_progress,
        velocity_alignment=r_align,
        lateral_velocity_penalty=r_lat_vel,
        attitude_stability=r_attitude,
        angular_rate_stability=r_rates,
        time_penalty=r_time,
        collision_penalty=r_collision,
        control_effort=r_effort,
        alive_bonus=r_alive,
    )
