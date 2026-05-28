"""Build the 34-dim observation vector from MAVLink telemetry + gate info.

Observation layout (must match DroneRaceEnv._build_obs):
  [0:3]   pos (3)
  [3:6]   vel (3)
  [6:15]  rot_matrix flattened (9)
  [15:18] omega (3)
  [18:22] prev_action (4)
  [22:25] body_gate_0 (3)   — current gate, body frame
  [25:28] body_gate_1 (3)   — next gate
  [28:31] body_gate_2 (3)   — gate +2
  [31:34] body_gate_3 (3)   — gate +3
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dronesim.bridge.mavlink_client import TelemetryState
from dronesim.sim.env import euler_to_rotation_matrix

OBS_DIM = 34


class ObsNormalizer:
    """Apply VecNormalize observation stats without needing a VecEnv."""

    def __init__(self, obs_mean: NDArray, obs_var: NDArray, clip: float = 10.0,
                 epsilon: float = 1e-8) -> None:
        self.mean = np.asarray(obs_mean, dtype=np.float64)
        self.var = np.asarray(obs_var, dtype=np.float64)
        self.clip = clip
        self.epsilon = epsilon

    @classmethod
    def from_pkl(cls, path: str | Path) -> ObsNormalizer:
        """Load from a SB3 VecNormalize .pkl file."""
        with open(path, "rb") as f:
            vec_norm = pickle.load(f)
        return cls(
            obs_mean=vec_norm.obs_rms.mean,
            obs_var=vec_norm.obs_rms.var,
            clip=vec_norm.clip_obs,
            epsilon=vec_norm.epsilon,
        )

    def normalize(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        normed = (obs.astype(np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class ObservationBuilder:
    """Converts MAVLink telemetry + gate positions to the 34-dim obs vector."""

    def __init__(self) -> None:
        self.prev_action = np.zeros(4, dtype=np.float64)

    def reset(self) -> None:
        self.prev_action = np.zeros(4, dtype=np.float64)

    def build(
        self,
        telemetry: TelemetryState,
        gate_positions_world: list[NDArray[np.float64] | None],
        gate_index: int = 0,
    ) -> NDArray[np.float32]:
        """Build 34-dim observation from telemetry and known gate positions.

        Args:
            telemetry: Current MAVLink telemetry state.
            gate_positions_world: List of all gate center positions in world frame.
                                  None entries are treated as zeros (past end of course).
            gate_index: Index of the current target gate.

        Returns:
            (34,) float32 observation vector.
        """
        pos = telemetry.pos.copy()
        vel = telemetry.vel.copy()
        euler = np.array(
            [telemetry.roll, telemetry.pitch, telemetry.yaw], dtype=np.float64
        )
        omega = np.array(
            [telemetry.rollspeed, telemetry.pitchspeed, telemetry.yawspeed],
            dtype=np.float64,
        )
        rot_matrix = euler_to_rotation_matrix(euler)

        # Body-frame gate vectors for the next 4 gates
        body_gates: list[NDArray[np.float64]] = []
        for offset in range(4):
            idx = gate_index + offset
            if idx < len(gate_positions_world) and gate_positions_world[idx] is not None:
                rel_world = gate_positions_world[idx] - pos
                body_gates.append(rot_matrix.T @ rel_world)
            else:
                body_gates.append(np.zeros(3, dtype=np.float64))

        obs = np.concatenate([
            pos,                           # 3
            vel,                           # 3
            rot_matrix.flatten(),          # 9
            omega,                         # 3
            self.prev_action,              # 4
            body_gates[0],                 # 3
            body_gates[1],                 # 3
            body_gates[2],                 # 3
            body_gates[3],                 # 3
        ]).astype(np.float32)
        return obs

    def update_prev_action(self, action: NDArray[np.float64]) -> None:
        self.prev_action = action.copy()
