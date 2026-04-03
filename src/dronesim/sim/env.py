from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import NDArray

from dronesim.config import RuntimeConfig
from dronesim.types import DroneState

if TYPE_CHECKING:
    from dronesim.types import GateSpec

_MAX_GATE_MOCAP = 10  # must match number of gate_N bodies in MJCF


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


def _normal_to_quat_wxyz(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quaternion (wxyz) rotating +X axis to align with `normal`."""
    n = np.asarray(normal, dtype=np.float64)
    n_len = np.linalg.norm(n)
    if n_len < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    n = n / n_len
    dot = float(np.clip(np.dot([1.0, 0.0, 0.0], n), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z
    axis = np.cross([1.0, 0.0, 0.0], n)
    axis /= np.linalg.norm(axis)
    half = math.acos(dot) / 2.0
    return np.array([math.cos(half), *(math.sin(half) * axis)])


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

        # --- Perception (Phase 2) ---
        # Find mocap indices for pre-allocated gate ring bodies.
        self._gate_mocap_ids: list[int] = []
        for i in range(_MAX_GATE_MOCAP):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"gate_{i}")
            if body_id >= 0:
                mocap_id = int(self.model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self._gate_mocap_ids.append(mocap_id)

        # Find drone camera ID (-1 if not in model).
        self._drone_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam"
        )

        self._renderer: mujoco.Renderer | None = None

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

    # ------------------------------------------------------------------
    # Phase 2: camera rendering + gate visual positioning
    # ------------------------------------------------------------------

    def set_gate_visuals(self, gates: list[GateSpec]) -> None:
        """Move pre-allocated gate mocap bodies to match current gate layout.

        Call after reset() so the camera can see the correct gate positions.
        Unused slots are hidden at (100, 0, 0).
        """
        hide = np.array([100.0, 0.0, 0.0])
        for i, mocap_id in enumerate(self._gate_mocap_ids):
            if i < len(gates):
                self.data.mocap_pos[mocap_id] = gates[i].center
                self.data.mocap_quat[mocap_id] = _normal_to_quat_wxyz(gates[i].normal)
            else:
                self.data.mocap_pos[mocap_id] = hide
        mujoco.mj_forward(self.model, self.data)

    def render_camera(self, width: int = 160, height: int = 120) -> NDArray[np.uint8]:
        """Render from the drone's forward camera. Returns (H, W, 3) uint8 RGB."""
        if self._drone_cam_id < 0:
            raise RuntimeError("drone_cam not found in MJCF model")
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=self._drone_cam_id)
        return self._renderer.render()

    def get_camera_intrinsics(
        self, width: int = 160, height: int = 120
    ) -> tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) for the drone camera at given resolution."""
        if self._drone_cam_id < 0:
            raise RuntimeError("drone_cam not found in MJCF model")
        fovy_rad = math.radians(float(self.model.cam_fovy[self._drone_cam_id]))
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy  # square pixels assumed
        return fx, fy, width / 2.0, height / 2.0

    def get_camera_extrinsics(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (cam_pos, cam_xmat) where cam_xmat columns are camera axes in world frame."""
        if self._drone_cam_id < 0:
            raise RuntimeError("drone_cam not found in MJCF model")
        return (
            self.data.cam_xpos[self._drone_cam_id].copy(),
            self.data.cam_xmat[self._drone_cam_id].reshape(3, 3).copy(),
        )

    def close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def apply_randomization(self, scale: float, rng: np.random.Generator) -> None:
        jitter = (rng.random() - 0.5) * 2.0
        self.mass_scale = 1.0 + self.config.sim.mass_jitter * scale * jitter
        self.model.body_mass[self.drone_body_id] = self.base_mass * self.mass_scale
        self.model.body_inertia[self.drone_body_id, :] = self.base_inertia * self.mass_scale
