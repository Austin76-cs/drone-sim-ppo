from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dronesim.config import DroneConfig
from dronesim.sim.mixer import mix_actions_to_rotors
from dronesim.types import DroneState


def compute_rotor_commands(
    action: NDArray[np.float64],
    state: DroneState,
    cfg: DroneConfig,
    mass_scale: float,
    gravity: float,
) -> NDArray[np.float64]:
    """CTBR: Convert [thrust, roll_rate, pitch_rate, yaw_rate] to rotor commands.

    Action space matches champion drone racing (Swift, MonoRace):
      action[0]: normalized collective thrust [-1, 1] -> [0, 1]
      action[1]: desired roll rate  [-1, 1] -> [-max_rate, max_rate]
      action[2]: desired pitch rate [-1, 1] -> [-max_rate, max_rate]
      action[3]: desired yaw rate   [-1, 1] -> [-max_rate, max_rate]
    """
    action = np.clip(action, -1.0, 1.0)

    # Thrust: [-1, 1] -> [0, 1] normalized
    thrust_normalized = (float(action[0]) + 1.0) * 0.5

    # Desired body rates from policy
    max_rate = cfg.max_body_rate_rad_s
    desired_rates = np.array([
        float(action[1]) * max_rate,
        float(action[2]) * max_rate,
        float(action[3]) * max_rate,
    ], dtype=np.float64)

    # PD rate controller: compute torque commands from rate error
    rate_error = desired_rates - state.omega
    roll_cmd = cfg.body_rate_kp * rate_error[0] / max(max_rate, 1e-4)
    pitch_cmd = cfg.body_rate_kp * rate_error[1] / max(max_rate, 1e-4)
    yaw_cmd = cfg.yaw_rate_kp * rate_error[2] / max(max_rate, 1e-4)

    controller_action = np.array([
        thrust_normalized * 2.0 - 1.0,
        np.clip(roll_cmd, -1.0, 1.0),
        np.clip(pitch_cmd, -1.0, 1.0),
        np.clip(yaw_cmd, -1.0, 1.0),
    ], dtype=np.float64)
    return mix_actions_to_rotors(controller_action)
