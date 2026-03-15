from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from dronesim.config import RuntimeConfig
from dronesim.types import DroneState


def quat_wxyz_to_euler(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert wxyz quaternion to roll-pitch-yaw Euler angles."""
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quat_wxyz(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert roll-pitch-yaw Euler angles to wxyz quaternion."""
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def euler_to_rotation_matrix(euler: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert roll-pitch-yaw to 3x3 rotation matrix."""
    roll, pitch, yaw = euler
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


class MuJoCoSim:
    """Thin MuJoCo wrapper. Handles physics only — no Gymnasium API."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.sim_dt = 1.0 / config.sim.sim_hz
        self.control_dt = 1.0 / config.sim.policy_hz
        self.decimation = max(1, config.sim.sim_hz // config.sim.policy_hz)

        xml_path = Path(config.sim.model_xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"missing MuJoCo model XML: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.sim_dt
        self.model.opt.gravity[2] = -float(config.sim.gravity)
        self.data = mujoco.MjData(self.model)

        self.ctrl_low = self.model.actuator_ctrlrange[:4, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:4, 1].copy()

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        if body_id < 0:
            raise RuntimeError("MuJoCo model must include body named 'drone'")
        self.drone_body_id = int(body_id)
        self.base_mass = float(self.model.body_mass[self.drone_body_id])
        self.base_inertia = self.model.body_inertia[self.drone_body_id, :].copy()

        self.motor_cmd = np.zeros(4, dtype=np.float64)
        self.mass_scale = 1.0

    def reset(
        self,
        qpos: NDArray[np.float64] | None = None,
        qvel: NDArray[np.float64] | None = None,
    ) -> DroneState:
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: len(qpos)] = qpos
        if qvel is not None:
            self.data.qvel[: len(qvel)] = qvel
        self.data.ctrl[:4] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.motor_cmd = np.zeros(4, dtype=np.float64)
        return self.read_state()

    def step(self, rotor_thrusts: NDArray[np.float64]) -> DroneState:
        """Apply motor smoothing, set actuator thrusts, step physics."""
        # Motor smoothing: prev + alpha * (target - prev)
        alpha = np.clip(self.control_dt / max(self.config.drone.motor_time_constant_s, 1e-6), 0.0, 1.0)
        self.motor_cmd = self.motor_cmd + alpha * (rotor_thrusts - self.motor_cmd)

        thrust_per_rotor = self.motor_cmd * (self.config.drone.max_total_thrust_n / 4.0)
        thrust_per_rotor = np.clip(thrust_per_rotor, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:4] = thrust_per_rotor

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        return self.read_state()

    def read_state(self) -> DroneState:
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:6].copy()
        pos = qpos[:3]
        euler = quat_wxyz_to_euler(qpos[3:7])
        vel = qvel[:3]
        omega = qvel[3:6]
        return DroneState(pos=pos, vel=vel, euler=euler, omega=omega, motor=self.motor_cmd.copy())

    def check_ground_contact(self) -> bool:
        return float(self.data.qpos[2]) < 0.10

    def apply_randomization(self, scale: float, rng: np.random.Generator) -> None:
        jitter = (rng.random() - 0.5) * 2.0
        self.mass_scale = 1.0 + self.config.sim.mass_jitter * scale * jitter
        self.model.body_mass[self.drone_body_id] = self.base_mass * self.mass_scale
        self.model.body_inertia[self.drone_body_id, :] = self.base_inertia * self.mass_scale
