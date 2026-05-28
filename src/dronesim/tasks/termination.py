from __future__ import annotations

import numpy as np

from dronesim.tasks.rewards import gate_relative_geometry
from dronesim.types import DroneState, GateSpec


def compute_termination(
    state: DroneState,
    gate: GateSpec,
    max_distance_m: float,
) -> tuple[bool, str]:
    """Check termination conditions. Returns (terminated, crash_type).

    Soft collision: gate contact is NOT terminal — handled via penalty in reward.
    Only hard crashes (ground, flip, spin, out-of-bounds) terminate.
    """
    # Only kill on near-inversion (>115 degrees)
    tilt = float(np.linalg.norm(state.euler[:2]))
    if tilt > 2.0:
        return True, "flip"

    altitude = float(state.pos[2])
    if altitude < 0.10:
        return True, "ground"
    if altitude > 8.0:
        return True, "altitude"

    if float(np.linalg.norm(state.omega)) > 25.0:
        return True, "spin"

    forward_error, lateral_error, _ = gate_relative_geometry(state, gate)
    if forward_error > max_distance_m:
        return True, "behind_course"
    if lateral_error > max_distance_m:
        return True, "off_line"

    return False, "none"


def check_gate_collision(
    state: DroneState,
    gate: GateSpec,
    drone_radius: float = 0.15,
) -> bool:
    """Check if drone is colliding with gate frame (soft collision).

    Returns True if the drone is within drone_radius of the gate frame edges.
    The drone gets a penalty but does NOT terminate.
    """
    forward_error, _, _ = gate_relative_geometry(state, gate)

    # Only check collision when very close to gate plane
    if abs(forward_error) > gate.depth_m + drone_radius:
        return False

    # Check if drone is near the gate frame edges (outside aperture + margin)
    rel = state.pos - gate.center
    rel_plane = rel - gate.normal * float(np.dot(rel, gate.normal))

    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, gate.normal)
    lat_norm = float(np.linalg.norm(lateral))
    if lat_norm < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])
    else:
        lateral = lateral / lat_norm

    half_w = (gate.width_m if gate.width_m > 0 else 2.0 * gate.radius_m) / 2.0
    half_h = (gate.height_m if gate.height_m > 0 else 2.0 * gate.radius_m) / 2.0

    y_offset = abs(float(np.dot(rel_plane, lateral)))
    z_offset = abs(float(np.dot(rel_plane, up)))

    # Collision if drone is near gate frame (within drone_radius of the edges)
    # but outside the aperture
    near_frame_y = half_w - drone_radius < y_offset < half_w + drone_radius
    near_frame_z = half_h - drone_radius < z_offset < half_h + drone_radius
    inside_y = y_offset < half_w + drone_radius
    inside_z = z_offset < half_h + drone_radius

    return (near_frame_y and inside_z) or (near_frame_z and inside_y)
