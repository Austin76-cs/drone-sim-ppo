from __future__ import annotations

import numpy as np

from dronesim.tasks.rewards import gate_relative_geometry
from dronesim.types import DroneState, GateSpec


def compute_termination(
    state: DroneState,
    gate: GateSpec,
    max_distance_m: float,
) -> tuple[bool, str]:
    """Check termination conditions. Returns (terminated, crash_type)."""
    tilt = float(np.linalg.norm(state.euler[:2]))
    if tilt > 1.25:
        return True, "flip"

    altitude = float(state.pos[2])
    if altitude < 0.10:
        return True, "ground"
    if altitude > 4.5:
        return True, "altitude"

    if float(np.linalg.norm(state.omega)) > 18.0:
        return True, "spin"

    forward_error, lateral_error, _ = gate_relative_geometry(state, gate)
    if forward_error > max_distance_m:
        return True, "behind_course"
    if lateral_error > max_distance_m:
        return True, "off_line"

    return False, "none"
