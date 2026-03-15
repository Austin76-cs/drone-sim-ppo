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
    """Convert [vz, roll, pitch, yaw_rate] setpoint to normalized rotor commands."""
    action = np.clip(action, -1.0, 1.0)

    current_mass = cfg.mass_kg * mass_scale
    tilt_comp = max(0.55, float(np.cos(state.euler[0]) * np.cos(state.euler[1])))
    hover_ratio = (current_mass * gravity) / max(cfg.max_total_thrust_n, 1e-4)
    hover_ratio /= tilt_comp

    desired_vz = float(action[0]) * cfg.max_vertical_velocity_m_s
    desired_roll = float(action[1]) * cfg.max_tilt_rad
    desired_pitch = float(action[2]) * cfg.max_tilt_rad
    desired_yaw_rate = float(action[3]) * cfg.max_body_rate_rad_s

    vz_error = desired_vz - float(state.vel[2])
    collective_ratio = np.clip(hover_ratio + cfg.vertical_velocity_kp * vz_error, 0.0, 1.0)

    desired_roll_rate = cfg.attitude_kp * (desired_roll - float(state.euler[0]))
    desired_pitch_rate = cfg.attitude_kp * (desired_pitch - float(state.euler[1]))

    max_rate = cfg.max_body_rate_rad_s
    desired_rates = np.array([
        np.clip(desired_roll_rate, -max_rate, max_rate),
        np.clip(desired_pitch_rate, -max_rate, max_rate),
        desired_yaw_rate,
    ], dtype=np.float64)

    rate_error = desired_rates - state.omega
    roll_cmd = cfg.body_rate_kp * float(rate_error[0]) / max(max_rate, 1e-4)
    pitch_cmd = cfg.body_rate_kp * float(rate_error[1]) / max(max_rate, 1e-4)
    yaw_cmd = cfg.yaw_rate_kp * float(rate_error[2]) / max(max_rate, 1e-4)

    controller_action = np.array([
        collective_ratio * 2.0 - 1.0,
        np.clip(roll_cmd, -1.0, 1.0),
        np.clip(pitch_cmd, -1.0, 1.0),
        np.clip(yaw_cmd, -1.0, 1.0),
    ], dtype=np.float64)
    return mix_actions_to_rotors(controller_action)
