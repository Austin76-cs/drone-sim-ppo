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


_DRONE_RADIUS = 0.12  # account for drone body size (matches arm_length_m)


def _inside_square_gate(cross_pos: NDArray[np.float64], gate: GateSpec) -> bool:
    """Check if a crossing point is inside the square gate aperture.

    Shrinks the effective aperture by _DRONE_RADIUS on each side to account
    for the physical size of the drone — the center must clear the frame
    with margin so the whole drone actually fits through.
    """
    rel = cross_pos - gate.center
    # Remove the forward component (along gate normal)
    rel_plane = rel - gate.normal * float(np.dot(rel, gate.normal))

    half_w = (gate.width_m if gate.width_m > 0 else 2.0 * gate.radius_m) / 2.0
    half_h = (gate.height_m if gate.height_m > 0 else 2.0 * gate.radius_m) / 2.0

    # Shrink effective aperture by drone radius
    effective_half_w = half_w - _DRONE_RADIUS
    effective_half_h = half_h - _DRONE_RADIUS
    if effective_half_w <= 0 or effective_half_h <= 0:
        return False

    # Gate local axes: up is Z, lateral is perpendicular to normal in XY plane
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, gate.normal)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])
    else:
        lateral = lateral / lat_norm

    y_offset = abs(float(np.dot(rel_plane, lateral)))
    z_offset = abs(float(np.dot(rel_plane, up)))
    return y_offset <= effective_half_w and z_offset <= effective_half_h


def _crossing_offset(cross_pos: NDArray[np.float64], gate: GateSpec) -> float:
    """Return how far the crossing point is from gate center, normalized to [0, 1].

    0.0 = dead center, 1.0 = at the edge of the effective aperture.
    Values > 1.0 mean outside the gate.
    """
    rel = cross_pos - gate.center
    rel_plane = rel - gate.normal * float(np.dot(rel, gate.normal))

    half_w = (gate.width_m if gate.width_m > 0 else 2.0 * gate.radius_m) / 2.0
    half_h = (gate.height_m if gate.height_m > 0 else 2.0 * gate.radius_m) / 2.0
    effective_half_w = max(half_w - _DRONE_RADIUS, 1e-3)
    effective_half_h = max(half_h - _DRONE_RADIUS, 1e-3)

    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, gate.normal)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])
    else:
        lateral = lateral / lat_norm

    y_frac = abs(float(np.dot(rel_plane, lateral))) / effective_half_w
    z_frac = abs(float(np.dot(rel_plane, up))) / effective_half_h
    return max(y_frac, z_frac)


def gate_crossing_quality(
    state: DroneState, gate: GateSpec, gate_pass_margin_m: float,
    prev_pos: NDArray[np.float64] | None = None,
) -> tuple[bool, float]:
    """Like gate_passed but also returns crossing offset (0=center, 1=edge, >1=outside).

    Returns (passed, offset).  offset is only meaningful when a plane crossing occurred.
    """
    forward_error, _, _ = gate_relative_geometry(state, gate)

    if prev_pos is not None:
        prev_rel = gate.center - prev_pos
        prev_fwd = float(np.dot(prev_rel, gate.normal))
        if prev_fwd > 0.0 and forward_error <= gate_pass_margin_m:
            total = prev_fwd + abs(forward_error)
            if total > 1e-6:
                t = prev_fwd / total
                cross_pos = prev_pos + t * (state.pos - prev_pos)
            else:
                cross_pos = state.pos
            offset = _crossing_offset(cross_pos, gate)
            return _inside_square_gate(cross_pos, gate), offset
        return False, float("inf")

    offset = _crossing_offset(state.pos, gate)
    passed = forward_error <= gate_pass_margin_m and _inside_square_gate(state.pos, gate)
    return passed, offset


def gate_passed(
    state: DroneState, gate: GateSpec, gate_pass_margin_m: float,
    prev_pos: NDArray[np.float64] | None = None,
) -> bool:
    passed, _ = gate_crossing_quality(state, gate, gate_pass_margin_m, prev_pos)
    return passed


def body_frame_gate(state: DroneState, gate: GateSpec) -> NDArray[np.float64]:
    """Get gate position relative to drone in body frame."""
    rel_world = gate.center - state.pos
    if state.rot_matrix is not None:
        return state.rot_matrix.T @ rel_world
    rot = euler_to_rotation_matrix(state.euler)
    return rot.T @ rel_world


# --- Modular reward components ---

def gate_proximity_reward(state: DroneState, gate: GateSpec, scale: float) -> float:
    _, lateral_error, _ = gate_relative_geometry(state, gate)
    return _exp_score(lateral_error, scale)


def gate_missed(
    state: DroneState,
    gate: GateSpec,
    gate_pass_margin: float,
    prev_pos: NDArray[np.float64] | None = None,
) -> bool:
    """Returns True if the drone crossed the gate plane outside the square gate aperture."""
    if prev_pos is None:
        return False
    forward_error, _, _ = gate_relative_geometry(state, gate)
    prev_rel = gate.center - prev_pos
    prev_fwd = float(np.dot(prev_rel, gate.normal))
    if prev_fwd > 0.0 and forward_error <= gate_pass_margin:
        total = prev_fwd + abs(forward_error)
        if total > 1e-6:
            t = prev_fwd / total
            cross_pos = prev_pos + t * (state.pos - prev_pos)
            return not _inside_square_gate(cross_pos, gate)
        return not _inside_square_gate(state.pos, gate)
    return False


def gate_passage_reward(
    state: DroneState,
    gate: GateSpec,
    margin: float,
    prev_pos: NDArray[np.float64] | None = None,
) -> float:
    if gate_passed(state, gate, margin, prev_pos):
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


def approach_angle_reward(state: DroneState, gate: GateSpec) -> float:
    """Reward for velocity aligned with gate normal (perpendicular approach)."""
    speed = float(np.linalg.norm(state.vel))
    if speed < 1e-4:
        return 0.0
    vel_norm = state.vel / speed
    return float(np.clip(np.dot(vel_norm, gate.normal), 0.0, 1.0))


def forward_speed_reward(state: DroneState, max_speed: float = 5.0) -> float:
    """Reward for raw flight speed — incentivizes actually moving fast."""
    speed = float(np.linalg.norm(state.vel))
    return min(1.0, speed / max(max_speed, 1e-3))


def alive_bonus_reward() -> float:
    return 1.0


def gate_centering_reward(state: DroneState, gate: GateSpec) -> float:
    """Dense reward for being centered in the gate aperture when close to the gate plane.

    Activates within 2m in front of the gate and gives high reward for being
    well-centered on the aperture.  This bridges the gap between the proximity
    reward (coarse, always-on) and the passage bonus (sparse, only on crossing).
    """
    forward_error, lateral_error, _ = gate_relative_geometry(state, gate)

    # Only reward when approaching the gate (0-2m in front)
    if forward_error > 2.0 or forward_error < -0.5:
        return 0.0

    # Plane proximity factor: peaks when right at the gate plane
    plane_factor = math.exp(-abs(forward_error) / 0.5)

    # Centering factor: tight exponential — must be well-centered
    half_w = (gate.width_m if gate.width_m > 0 else 2.0 * gate.radius_m) / 2.0
    center_scale = max(0.05, half_w * 0.25)  # much tighter than proximity
    center_factor = math.exp(-lateral_error / center_scale)

    return plane_factor * center_factor


def action_diff_reward(
    action: NDArray[np.float64], prev_action: NDArray[np.float64]
) -> float:
    """Penalize large changes in action between steps (smoothness)."""
    return float(np.linalg.norm(action - prev_action) / max(math.sqrt(len(action)), 1.0))


def compute_total_reward(
    state: DroneState,
    action: NDArray[np.float64],
    gate: GateSpec,
    prev_forward_error: float,
    cfg: RewardConfig,
    terminated: bool,
    gate_pass_margin: float,
    prev_pos: NDArray[np.float64] | None = None,
    prev_action: NDArray[np.float64] | None = None,
    gate_contact: bool = False,
) -> RewardInfo:
    """Compute weighted sum of all reward components."""
    curr_forward_error, _, _ = gate_relative_geometry(state, gate)

    r_proximity = gate_proximity_reward(state, gate, gate.radius_m)
    r_passage = gate_passage_reward(state, gate, gate_pass_margin, prev_pos)
    r_miss = gate_missed(state, gate, gate_pass_margin, prev_pos)
    r_progress = progress_reward(prev_forward_error, curr_forward_error, gate.radius_m)
    r_align = velocity_alignment_reward(state, gate, 3.0)
    r_approach = approach_angle_reward(state, gate)
    r_speed = forward_speed_reward(state)
    r_time = time_penalty_reward()
    r_collision = collision_penalty_reward() if terminated else 0.0
    r_gate_contact = 1.0 if gate_contact else 0.0
    r_effort = control_effort_reward(action)
    r_alive = alive_bonus_reward()
    r_action_diff = action_diff_reward(action, prev_action) if prev_action is not None else 0.0
    r_centering = gate_centering_reward(state, gate)

    total = (
        cfg.gate_proximity * r_proximity
        + cfg.gate_passage_bonus * r_passage
        + cfg.gate_miss_penalty * float(r_miss)
        + cfg.progress * r_progress
        + cfg.velocity_alignment * r_align
        + cfg.approach_angle * r_approach
        + cfg.forward_speed * r_speed
        + cfg.time_penalty * r_time
        + cfg.collision_penalty * r_collision
        + cfg.collision_penalty * 0.3 * r_gate_contact  # soft: 30% of crash penalty
        + cfg.control_effort * r_effort
        + cfg.alive_bonus * r_alive
        + cfg.action_diff_penalty * r_action_diff
        + cfg.gate_centering * r_centering
    )

    return RewardInfo(
        total=total,
        gate_proximity=r_proximity,
        gate_passage=r_passage,
        progress=r_progress,
        velocity_alignment=r_align,
        approach_angle=r_approach,
        forward_speed=r_speed,
        time_penalty=r_time,
        collision_penalty=r_collision,
        control_effort=r_effort,
        alive_bonus=r_alive,
        gate_miss=float(r_miss),
        gate_centering=r_centering,
    )
