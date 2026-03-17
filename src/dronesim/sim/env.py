from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from dronesim.config import RuntimeConfig
from dronesim.types import DroneState


def _import_mujoco_module(name: str) -> Any:
    module = importlib.import_module(name)
    return module


def _load_warp_backend() -> tuple[Any, Any, Any]:
    try:
        mujoco = _import_mujoco_module("mujoco")
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Warp requires the 'mujoco' package for model parsing and host-side viewer sync."
        ) from exc

    try:
        mjw = _import_mujoco_module("mujoco_warp")
        wp = _import_mujoco_module("warp")
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo Warp backend requires 'mujoco-warp' and its Warp runtime dependencies."
        ) from exc

    wp.init()
    return mujoco, mjw, wp


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
    """Thin MuJoCo Warp wrapper. Handles physics only — no Gymnasium API."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        if config.sim.backend.lower() != "warp":
            raise ValueError(f"unsupported sim backend: {config.sim.backend}")

        self._mujoco, self._mjw, self._wp = _load_warp_backend()
        self.sim_dt = 1.0 / config.sim.sim_hz
        self.control_dt = 1.0 / config.sim.policy_hz
        self.decimation = max(1, config.sim.sim_hz // config.sim.policy_hz)
        self._warp_device = config.sim.backend_device or config.device

        xml_path = Path(config.sim.model_xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"missing MuJoCo model XML: {xml_path}")
        self.model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.sim_dt
        self.model.opt.gravity[2] = -float(config.sim.gravity)
        self.data = self._mujoco.MjData(self.model)

        self.ctrl_low = self.model.actuator_ctrlrange[:4, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:4, 1].copy()

        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        if body_id < 0:
            raise RuntimeError("MuJoCo model must include body named 'drone'")
        self.drone_body_id = int(body_id)
        self.base_mass = float(self.model.body_mass[self.drone_body_id])
        self.base_inertia = self.model.body_inertia[self.drone_body_id, :].copy()

        self.motor_cmd = np.zeros(4, dtype=np.float64)
        self.mass_scale = 1.0
        self._ctrl_buffer = np.zeros((1, self.model.nu), dtype=np.float32)
        self._device_model = None
        self._device_data = None
        self._model_dirty = True
        self._rebuild_device_state()

    def _rebuild_device_state(self) -> None:
        with self._wp.ScopedDevice(self._warp_device):
            self._device_model = self._mjw.put_model(self.model)
            self._device_data = self._mjw.put_data(self.model, self.data, nworld=1)
        self._model_dirty = False

    def _upload_host_data(self) -> None:
        if self._model_dirty or self._device_model is None or self._device_data is None:
            self._rebuild_device_state()
            return

        with self._wp.ScopedDevice(self._warp_device):
            self._device_data = self._mjw.put_data(self.model, self.data, nworld=1)

    def _sync_host_data(self) -> None:
        if self._device_model is None or self._device_data is None:
            return

        with self._wp.ScopedDevice(self._warp_device):
            self._mjw.get_data_into(self.data, self.model, self._device_data)

    def reset(
        self,
        qpos: NDArray[np.float64] | None = None,
        qvel: NDArray[np.float64] | None = None,
    ) -> DroneState:
        self._mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: len(qpos)] = qpos
        if qvel is not None:
            self.data.qvel[: len(qvel)] = qvel
        self.data.ctrl[:4] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._upload_host_data()
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
        self._ctrl_buffer.fill(0.0)
        self._ctrl_buffer[0, :4] = thrust_per_rotor.astype(np.float32, copy=False)

        with self._wp.ScopedDevice(self._warp_device):
            self._device_data.ctrl.assign(self._ctrl_buffer)
            for _ in range(self.decimation):
                self._mjw.step(self._device_model, self._device_data)

        self._sync_host_data()

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
        self._model_dirty = True


class BatchedMuJoCoSim:
    """Batched MuJoCo Warp simulator for vectorized training."""

    def __init__(self, config: RuntimeConfig, num_worlds: int) -> None:
        self.config = config
        if config.sim.backend.lower() != "warp":
            raise ValueError(f"unsupported sim backend: {config.sim.backend}")

        self._mujoco, self._mjw, self._wp = _load_warp_backend()
        self.num_worlds = num_worlds
        self.sim_dt = 1.0 / config.sim.sim_hz
        self.control_dt = 1.0 / config.sim.policy_hz
        self.decimation = max(1, config.sim.sim_hz // config.sim.policy_hz)
        self._warp_device = config.sim.backend_device or config.device

        xml_path = Path(config.sim.model_xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"missing MuJoCo model XML: {xml_path}")

        self.model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.sim_dt
        self.model.opt.gravity[2] = -float(config.sim.gravity)
        self.ctrl_low = self.model.actuator_ctrlrange[:4, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:4, 1].copy()

        with self._wp.ScopedDevice(self._warp_device):
            self._device_model = self._mjw.put_model(self.model)
            self._device_data = self._mjw.put_data(
                self.model,
                self._mujoco.MjData(self.model),
                nworld=self.num_worlds,
            )

        self.motor_cmd = np.zeros((self.num_worlds, 4), dtype=np.float64)
        self.mass_scale = np.ones(self.num_worlds, dtype=np.float64)
        self._qpos = np.zeros((self.num_worlds, self.model.nq), dtype=np.float64)
        self._qvel = np.zeros((self.num_worlds, self.model.nv), dtype=np.float64)
        self._ctrl = np.zeros((self.num_worlds, self.model.nu), dtype=np.float32)
        self._time = np.zeros(self.num_worlds, dtype=np.float64)
        self._qpos_torch: torch.Tensor | None = None
        self._qvel_torch: torch.Tensor | None = None
        self._ctrl_torch: torch.Tensor | None = None
        self._time_torch: torch.Tensor | None = None
        self.motor_cmd_torch: torch.Tensor | None = None
        self.mass_scale_torch: torch.Tensor | None = None
        self.ctrl_low_torch: torch.Tensor | None = None
        self.ctrl_high_torch: torch.Tensor | None = None
        if self._warp_device.startswith("cuda"):
            self._qpos_torch = self._wp.to_torch(self._device_data.qpos)
            self._qvel_torch = self._wp.to_torch(self._device_data.qvel)
            self._ctrl_torch = self._wp.to_torch(self._device_data.ctrl)
            self._time_torch = self._wp.to_torch(self._device_data.time)
            self.motor_cmd_torch = torch.zeros(
                (self.num_worlds, 4), device=self._qpos_torch.device, dtype=torch.float64
            )
            self.mass_scale_torch = torch.ones(
                (self.num_worlds,), device=self._qpos_torch.device, dtype=torch.float64
            )
            self.ctrl_low_torch = torch.as_tensor(
                self.ctrl_low, device=self._qpos_torch.device, dtype=torch.float64
            )
            self.ctrl_high_torch = torch.as_tensor(
                self.ctrl_high, device=self._qpos_torch.device, dtype=torch.float64
            )
        self._sync_state()

    def _sync_state(self) -> None:
        if self._qpos_torch is not None and self._qvel_torch is not None:
            return
        with self._wp.ScopedDevice(self._warp_device):
            self._qpos = self._device_data.qpos.numpy()
            self._qvel = self._device_data.qvel.numpy()

    def reset_worlds(
        self,
        world_indices: NDArray[np.int32],
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
    ) -> None:
        if len(world_indices) == 0:
            return

        self.motor_cmd[world_indices] = 0.0
        self.mass_scale[world_indices] = 1.0
        if self.motor_cmd_torch is not None and self.mass_scale_torch is not None:
            idx = torch.as_tensor(world_indices, device=self.motor_cmd_torch.device, dtype=torch.long)
            self.motor_cmd_torch[idx] = 0.0
            self.mass_scale_torch[idx] = 1.0
        self._qpos[world_indices] = qpos
        self._qvel[world_indices] = qvel
        self._ctrl[world_indices] = 0.0
        self._time[world_indices] = 0.0
        reset_mask = np.zeros(self.num_worlds, dtype=bool)
        reset_mask[world_indices] = True

        with self._wp.ScopedDevice(self._warp_device):
            self._mjw.reset_data(
                self._device_model,
                self._device_data,
                reset=self._wp.array(reset_mask, dtype=self._wp.bool),
            )
            self._device_data.qpos.assign(self._qpos)
            self._device_data.qvel.assign(self._qvel)
            self._device_data.ctrl.assign(self._ctrl)
            self._device_data.time.assign(self._time)
            self._mjw.forward(self._device_model, self._device_data)

        self._sync_state()

    def step(self, rotor_thrusts: NDArray[np.float64]) -> None:
        if isinstance(rotor_thrusts, torch.Tensor):
            if self.motor_cmd_torch is None or self._ctrl_torch is None:
                raise TypeError("torch rotor thrusts require a CUDA-backed simulator")
            alpha = np.clip(
                self.control_dt / max(self.config.drone.motor_time_constant_s, 1e-6),
                0.0,
                1.0,
            )
            self.motor_cmd_torch.add_(alpha * (rotor_thrusts - self.motor_cmd_torch))

            thrust_per_rotor = self.motor_cmd_torch * (self.config.drone.max_total_thrust_n / 4.0)
            thrust_per_rotor = torch.clamp(thrust_per_rotor, self.ctrl_low_torch[:4], self.ctrl_high_torch[:4])
            self._ctrl_torch.zero_()
            self._ctrl_torch[:, :4] = thrust_per_rotor.to(dtype=self._ctrl_torch.dtype)

            with self._wp.ScopedDevice(self._warp_device):
                for _ in range(self.decimation):
                    self._mjw.step(self._device_model, self._device_data)
            return

        alpha = np.clip(
            self.control_dt / max(self.config.drone.motor_time_constant_s, 1e-6),
            0.0,
            1.0,
        )
        self.motor_cmd = self.motor_cmd + alpha * (rotor_thrusts - self.motor_cmd)

        thrust_per_rotor = self.motor_cmd * (self.config.drone.max_total_thrust_n / 4.0)
        thrust_per_rotor = np.clip(thrust_per_rotor, self.ctrl_low, self.ctrl_high)
        self._ctrl.fill(0.0)
        self._ctrl[:, :4] = thrust_per_rotor.astype(np.float32, copy=False)

        with self._wp.ScopedDevice(self._warp_device):
            self._device_data.ctrl.assign(self._ctrl)
            for _ in range(self.decimation):
                self._mjw.step(self._device_model, self._device_data)

        self._sync_state()

    @property
    def qpos(self) -> NDArray[np.float64]:
        return self._qpos

    @property
    def qvel(self) -> NDArray[np.float64]:
        return self._qvel

    @property
    def qpos_torch(self) -> torch.Tensor | None:
        return self._qpos_torch

    @property
    def qvel_torch(self) -> torch.Tensor | None:
        return self._qvel_torch
