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
    state: DroneState, gate: GateSpec, gate_pass_margin_m: float,
    prev_pos: NDArray[np.float64] | None = None,
) -> bool:
    forward_error, lateral_error, _ = gate_relative_geometry(state, gate)

    if prev_pos is not None:
        # Ray-plane crossing check: did the drone cross the gate plane
        # between prev_pos and current pos?
        prev_rel = gate.center - prev_pos
        prev_fwd = float(np.dot(prev_rel, gate.normal))
        # Crossed if prev was in front (positive) and now behind/at plane
        if prev_fwd > 0.0 and forward_error <= gate_pass_margin_m:
            # Interpolate to find crossing point and check lateral distance
            total = prev_fwd + abs(forward_error)
            if total > 1e-6:
                t = prev_fwd / total
                cross_pos = prev_pos + t * (state.pos - prev_pos)
                cross_lateral = float(np.linalg.norm(
                    (gate.center - cross_pos)
                    - gate.normal * float(np.dot(gate.center - cross_pos, gate.normal))
                ))
                return cross_lateral <= gate.radius_m
            return lateral_error <= gate.radius_m
        return False

    # Fallback: simple proximity check (no previous position available)
    return forward_error <= gate_pass_margin_m and lateral_error <= gate.radius_m


def body_frame_gate(state: DroneState, gate: GateSpec) -> NDArray[np.float64]:
    """Get gate position relative to drone in body frame."""
    rel_world = gate.center - state.pos
    rot = euler_to_rotation_matrix(state.euler)
    return rot.T @ rel_world


# --- Modular reward components ---

def gate_proximity_reward(state: DroneState, gate: GateSpec, scale: float) -> float:
    _, lateral_error, _ = gate_relative_geometry(state, gate)
    return _exp_score(lateral_error, scale)


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
    r_passage = gate_passage_reward(state, gate, gate_pass_margin)
    r_progress = progress_reward(prev_forward_error, curr_forward_error, gate.radius_m)
    r_align = velocity_alignment_reward(state, gate, 0.35)
    r_time = time_penalty_reward()
    r_collision = collision_penalty_reward() if terminated else 0.0
    r_effort = control_effort_reward(action)
    r_alive = alive_bonus_reward()

    total = (
        cfg.gate_proximity * r_proximity
        + cfg.gate_passage_bonus * r_passage
        + cfg.progress * r_progress
        + cfg.velocity_alignment * r_align
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
        time_penalty=r_time,
        collision_penalty=r_collision,
        control_effort=r_effort,
        alive_bonus=r_alive,
    )
