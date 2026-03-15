from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mix_actions_to_rotors(action: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert [collective, roll, pitch, yaw] to [r1, r2, r3, r4] for X-layout quad."""
    action = np.clip(action, -1.0, 1.0)
    collective = (action[0] + 1.0) * 0.5
    roll = action[1] * 0.25
    pitch = action[2] * 0.25
    yaw = action[3] * 0.15

    rotors = np.array([
        collective + roll + pitch + yaw,
        collective - roll + pitch - yaw,
        collective - roll - pitch + yaw,
        collective + roll - pitch - yaw,
    ], dtype=np.float64)
    return np.clip(rotors, 0.0, 1.0)
