from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from dronesim.sim.env import euler_to_rotation_matrix
from dronesim.types import DroneState


def body_frame_velocity(state: DroneState) -> NDArray[np.float64]:
    rot = euler_to_rotation_matrix(state.euler)
    return rot.T @ state.vel


def guided_gate_action(
    state: DroneState,
    current_gate_body: NDArray[np.float64],
    next_gate_body: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    x_b, y_b, z_b = current_gate_body.astype(np.float64)
    lookahead = (
        next_gate_body.astype(np.float64)
        if next_gate_body is not None and float(np.linalg.norm(next_gate_body)) > 1e-6
        else current_gate_body.astype(np.float64)
    )
    vel_body = body_frame_velocity(state)
    omega = state.omega.astype(np.float64)

    forward_ref = max(float(x_b), 0.8)
    look_ref = max(float(lookahead[0]), 1.0)
    side_error = float(np.clip((y_b / forward_ref) * 0.75 + (lookahead[1] / look_ref) * 0.25, -1.0, 1.0))
    height_error = float(np.clip(z_b / forward_ref, -0.6, 0.6))
    yaw_error = float(math.atan2(y_b, max(float(x_b), 0.35)))

    vz_cmd = np.clip(0.95 * height_error - 0.40 * vel_body[2], -0.75, 0.75)
    roll_cmd = np.clip(-1.35 * side_error - 0.22 * vel_body[1] - 0.08 * omega[0], -0.65, 0.65)
    forward_bias = 0.42 + 0.12 * math.tanh((float(x_b) - 1.2) / 0.9)
    pitch_cmd = np.clip(
        forward_bias - 0.55 * abs(side_error) - 0.30 * abs(height_error) - 0.10 * omega[1],
        -0.25,
        0.60,
    )
    yaw_cmd = np.clip(-0.55 * yaw_error - 0.06 * omega[2], -0.35, 0.35)
    return np.array([vz_cmd, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float64)
