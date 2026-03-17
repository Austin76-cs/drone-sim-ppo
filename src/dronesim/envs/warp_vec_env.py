from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from dronesim.config import RuntimeConfig
from dronesim.sim.attitude_controller import compute_rotor_commands
from dronesim.sim.env import BatchedMuJoCoSim, euler_to_quat_wxyz, quat_wxyz_to_euler
from dronesim.tasks.curriculum import EpisodeSummary, EpisodeTask, StageController
from dronesim.tasks.guidance import guided_gate_action
from dronesim.tasks.rewards import (
    body_frame_gate,
    compute_total_reward,
    gate_passed,
    gate_relative_geometry,
)
from dronesim.tasks.termination import compute_termination
from dronesim.types import DroneState, GateSpec, RewardInfo


class WarpVecDroneRaceEnv(VecEnv):
    """SB3 VecEnv backed by a single batched MuJoCo Warp simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RuntimeConfig,
        num_envs: int,
        stage_controller: StageController | None = None,
    ) -> None:
        self.config = config
        self.render_mode = None
        self.stage_controller = stage_controller or StageController(config.task)
        self.base_episode_steps = max(
            1, int(config.sim.episode_seconds * config.sim.policy_hz)
        )
        self.sim = BatchedMuJoCoSim(config, num_envs)

        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)

        self.rngs = [np.random.default_rng(config.seed + i) for i in range(num_envs)]
        self.current_tasks: list[EpisodeTask | None] = [None] * num_envs
        self.states: list[DroneState | None] = [None] * num_envs
        self.last_reward_info: list[RewardInfo | None] = [None] * num_envs
        self.gate_index = np.zeros(num_envs, dtype=np.int32)
        self.gates_cleared = np.zeros(num_envs, dtype=np.int32)
        self.step_idx = np.zeros(num_envs, dtype=np.int32)
        self.episode_steps = np.full(num_envs, self.base_episode_steps, dtype=np.int32)
        self.current_gate_forward_error = np.zeros(num_envs, dtype=np.float64)
        self.gate_start_distance = np.ones(num_envs, dtype=np.float64)
        self.gate_start_forward_error = np.ones(num_envs, dtype=np.float64)
        self.buf_obs = np.zeros((num_envs, 18), dtype=np.float32)
        self.buf_rews = np.zeros(num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(num_envs, dtype=bool)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._actions = np.zeros((num_envs, 4), dtype=np.float32)
        self._prev_actions = np.zeros((num_envs, 4), dtype=np.float64)
        self._use_torch_fast_path = (
            (config.sim.backend_device or config.device).startswith("cuda")
            and self.sim.qpos_torch is not None
            and self.sim.qvel_torch is not None
        )
        self._torch_device = self.sim.qpos_torch.device if self._use_torch_fast_path else None
        self._env_index_torch = (
            torch.arange(num_envs, device=self._torch_device, dtype=torch.long)
            if self._use_torch_fast_path
            else None
        )
        self._gate_centers_torch: torch.Tensor | None = None
        self._gate_normals_torch: torch.Tensor | None = None
        self._gate_radii_torch: torch.Tensor | None = None
        self._randomization_scale_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.float64)
            if self._use_torch_fast_path
            else None
        )
        self._max_distance_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.float64)
            if self._use_torch_fast_path
            else None
        )
        self._gate_index_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.long)
            if self._use_torch_fast_path
            else None
        )
        self._gates_cleared_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.long)
            if self._use_torch_fast_path
            else None
        )
        self._step_idx_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.long)
            if self._use_torch_fast_path
            else None
        )
        self._episode_steps_torch = (
            torch.full((num_envs,), self.base_episode_steps, device=self._torch_device, dtype=torch.long)
            if self._use_torch_fast_path
            else None
        )
        self._current_gate_forward_error_torch = (
            torch.zeros(num_envs, device=self._torch_device, dtype=torch.float64)
            if self._use_torch_fast_path
            else None
        )
        self._gate_start_distance_torch = (
            torch.ones(num_envs, device=self._torch_device, dtype=torch.float64)
            if self._use_torch_fast_path
            else None
        )
        self._prev_actions_torch = (
            torch.zeros((num_envs, 4), device=self._torch_device, dtype=torch.float64)
            if self._use_torch_fast_path
            else None
        )
        self._gate_capacity = 0

    def _current_gate(self, env_idx: int) -> GateSpec:
        task = self.current_tasks[env_idx]
        assert task is not None
        gate_idx = min(int(self.gate_index[env_idx]), len(task.gates) - 1)
        return task.gates[gate_idx]

    def _next_gate(self, env_idx: int) -> GateSpec | None:
        task = self.current_tasks[env_idx]
        assert task is not None
        next_idx = int(self.gate_index[env_idx]) + 1
        if next_idx < len(task.gates):
            return task.gates[next_idx]
        return None

    def _state_for(self, env_idx: int) -> DroneState:
        qpos = self.sim.qpos[env_idx, :7].copy()
        qvel = self.sim.qvel[env_idx, :6].copy()
        return DroneState(
            pos=qpos[:3],
            vel=qvel[:3],
            euler=quat_wxyz_to_euler(qpos[3:7]),
            omega=qvel[3:6],
            motor=self.sim.motor_cmd[env_idx].copy(),
        )

    def _ensure_gate_capacity(self, max_gates: int) -> None:
        if not self._use_torch_fast_path or max_gates <= self._gate_capacity:
            return
        device = self._torch_device
        assert device is not None
        centers = torch.zeros((self.num_envs, max_gates, 3), device=device, dtype=torch.float64)
        normals = torch.zeros((self.num_envs, max_gates, 3), device=device, dtype=torch.float64)
        radii = torch.zeros((self.num_envs, max_gates), device=device, dtype=torch.float64)
        if self._gate_centers_torch is not None:
            centers[:, : self._gate_capacity] = self._gate_centers_torch
            normals[:, : self._gate_capacity] = self._gate_normals_torch
            radii[:, : self._gate_capacity] = self._gate_radii_torch
        self._gate_centers_torch = centers
        self._gate_normals_torch = normals
        self._gate_radii_torch = radii
        self._gate_capacity = max_gates

    def _update_task_tensors(self, indices: NDArray[np.int32]) -> None:
        if not self._use_torch_fast_path or len(indices) == 0:
            return
        max_gates = max(len(self.current_tasks[int(env_idx)].gates) for env_idx in indices if self.current_tasks[int(env_idx)] is not None)
        self._ensure_gate_capacity(max_gates)
        assert self._gate_centers_torch is not None
        assert self._gate_normals_torch is not None
        assert self._gate_radii_torch is not None
        assert self._randomization_scale_torch is not None
        assert self._max_distance_torch is not None
        for env_idx in indices:
            task = self.current_tasks[int(env_idx)]
            assert task is not None
            gate_count = len(task.gates)
            self._gate_centers_torch[int(env_idx)].zero_()
            self._gate_normals_torch[int(env_idx)].zero_()
            self._gate_radii_torch[int(env_idx)].zero_()
            self._gate_centers_torch[int(env_idx), :gate_count] = torch.as_tensor(
                np.stack([gate.center for gate in task.gates]),
                device=self._torch_device,
                dtype=torch.float64,
            )
            self._gate_normals_torch[int(env_idx), :gate_count] = torch.as_tensor(
                np.stack([gate.normal for gate in task.gates]),
                device=self._torch_device,
                dtype=torch.float64,
            )
            self._gate_radii_torch[int(env_idx), :gate_count] = torch.as_tensor(
                [gate.radius_m for gate in task.gates],
                device=self._torch_device,
                dtype=torch.float64,
            )
            self._randomization_scale_torch[int(env_idx)] = task.randomization_scale
            self._max_distance_torch[int(env_idx)] = task.max_distance_m

    def _quat_wxyz_to_euler_torch(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=1)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack((roll, pitch, yaw), dim=1)

    def _gate_relative_geometry_torch(
        self,
        pos: torch.Tensor,
        gate_center: torch.Tensor,
        gate_normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = gate_center - pos
        forward_error = (rel * gate_normal).sum(dim=1)
        lateral = rel - gate_normal * forward_error.unsqueeze(1)
        lateral_error = torch.linalg.norm(lateral, dim=1)
        forward_ref = torch.clamp(rel[:, 0], min=1e-4)
        yaw_error = torch.atan2(rel[:, 1], forward_ref)
        pitch_error = torch.atan2(rel[:, 2], forward_ref)
        alignment_error = torch.sqrt(yaw_error.square() + pitch_error.square())
        return forward_error, lateral_error, alignment_error

    def _body_frame_gate_torch(
        self,
        pos: torch.Tensor,
        euler: torch.Tensor,
        gate_center: torch.Tensor,
    ) -> torch.Tensor:
        rel_world = gate_center - pos
        roll = euler[:, 0]
        pitch = euler[:, 1]
        yaw = euler[:, 2]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        rot = torch.stack((
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=1),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=1),
            torch.stack((-sp, cp * sr, cp * cr), dim=1),
        ), dim=1)
        return torch.bmm(rot.transpose(1, 2), rel_world.unsqueeze(2)).squeeze(2)

    def _compute_rotor_commands_torch(
        self,
        action: torch.Tensor,
        vel: torch.Tensor,
        euler: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config.drone
        assert self.sim.mass_scale_torch is not None
        action = torch.clamp(action, -1.0, 1.0)
        current_mass = cfg.mass_kg * self.sim.mass_scale_torch
        tilt_comp = torch.clamp(torch.cos(euler[:, 0]) * torch.cos(euler[:, 1]), min=0.55)
        hover_ratio = (current_mass * self.config.sim.gravity) / max(cfg.max_total_thrust_n, 1e-4)
        hover_ratio = hover_ratio / tilt_comp
        desired_vz = action[:, 0] * cfg.max_vertical_velocity_m_s
        desired_roll = action[:, 1] * cfg.max_tilt_rad
        desired_pitch = action[:, 2] * cfg.max_tilt_rad
        desired_yaw_rate = action[:, 3] * cfg.max_body_rate_rad_s
        vz_error = desired_vz - vel[:, 2]
        collective_ratio = torch.clamp(hover_ratio + cfg.vertical_velocity_kp * vz_error, 0.0, 1.0)
        max_rate = cfg.max_body_rate_rad_s
        desired_roll_rate = cfg.attitude_kp * (desired_roll - euler[:, 0])
        desired_pitch_rate = cfg.attitude_kp * (desired_pitch - euler[:, 1])
        desired_rates = torch.stack((
            torch.clamp(desired_roll_rate, -max_rate, max_rate),
            torch.clamp(desired_pitch_rate, -max_rate, max_rate),
            desired_yaw_rate,
        ), dim=1)
        rate_error = desired_rates - omega
        roll_cmd = cfg.body_rate_kp * rate_error[:, 0] / max(max_rate, 1e-4)
        pitch_cmd = cfg.body_rate_kp * rate_error[:, 1] / max(max_rate, 1e-4)
        yaw_cmd = cfg.yaw_rate_kp * rate_error[:, 2] / max(max_rate, 1e-4)
        collective = collective_ratio
        roll = torch.clamp(roll_cmd, -1.0, 1.0) * 0.25
        pitch = torch.clamp(pitch_cmd, -1.0, 1.0) * 0.25
        yaw = torch.clamp(yaw_cmd, -1.0, 1.0) * 0.15
        rotors = torch.stack((
            collective + roll + pitch + yaw,
            collective - roll + pitch - yaw,
            collective - roll - pitch + yaw,
            collective + roll - pitch - yaw,
        ), dim=1)
        return torch.clamp(rotors, 0.0, 1.0)

    def _build_obs(self, env_idx: int) -> NDArray[np.float32]:
        state = self.states[env_idx]
        assert state is not None
        cur_gate_body = body_frame_gate(state, self._current_gate(env_idx))
        next_gate = self._next_gate(env_idx)
        if next_gate is not None:
            next_gate_body = body_frame_gate(state, next_gate)
        else:
            next_gate_body = cur_gate_body.copy()

        return np.concatenate([
            state.pos,
            state.vel,
            state.euler,
            state.omega,
            cur_gate_body,
            next_gate_body,
        ]).astype(np.float32)

    def _current_gate_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._env_index_torch is not None
        assert self._gate_centers_torch is not None
        assert self._gate_normals_torch is not None
        assert self._gate_radii_torch is not None
        assert self._gate_index_torch is not None
        gate_index_t = self._gate_index_torch
        return (
            self._gate_centers_torch[self._env_index_torch, gate_index_t],
            self._gate_normals_torch[self._env_index_torch, gate_index_t],
            self._gate_radii_torch[self._env_index_torch, gate_index_t],
            gate_index_t,
        )

    def _build_obs_torch(self) -> torch.Tensor:
        assert self.sim.qpos_torch is not None
        assert self.sim.qvel_torch is not None
        cur_center, _, _, gate_index_t = self._current_gate_tensors()
        pos = self.sim.qpos_torch[:, :3].to(dtype=torch.float64)
        vel = self.sim.qvel_torch[:, :3].to(dtype=torch.float64)
        euler = self._quat_wxyz_to_euler_torch(self.sim.qpos_torch[:, 3:7].to(dtype=torch.float64))
        omega = self.sim.qvel_torch[:, 3:6].to(dtype=torch.float64)
        cur_gate_body = self._body_frame_gate_torch(pos, euler, cur_center)
        next_index_t = torch.clamp(gate_index_t + 1, max=max(0, self._gate_capacity - 1))
        assert self._env_index_torch is not None
        assert self._gate_centers_torch is not None
        next_gate_body = self._body_frame_gate_torch(
            pos,
            euler,
            self._gate_centers_torch[self._env_index_torch, next_index_t],
        )
        has_next = torch.as_tensor(
            [int(self.gate_index[i]) + 1 < len(self.current_tasks[i].gates) for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.bool,
        )
        next_gate_body = torch.where(has_next.unsqueeze(1), next_gate_body, cur_gate_body)
        return torch.cat((pos, vel, euler, omega, cur_gate_body, next_gate_body), dim=1)

    def _apply_guided_action_cpu(
        self,
        env_idx: int,
        action: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        guided_weight = float(np.clip(self.config.sim.guided_action_weight, 0.0, 1.0))
        if guided_weight <= 0.0:
            return action
        state = self.states[env_idx]
        assert state is not None
        current_gate_body = body_frame_gate(state, self._current_gate(env_idx))
        next_gate = self._next_gate(env_idx)
        next_gate_body = body_frame_gate(state, next_gate) if next_gate is not None else None
        base_action = guided_gate_action(state, current_gate_body, next_gate_body)
        residual_scale = float(max(self.config.sim.residual_action_scale, 0.0))
        return np.clip(guided_weight * base_action + residual_scale * action, -1.0, 1.0)

    def _guided_action_torch(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        euler: torch.Tensor,
        omega: torch.Tensor,
        current_gate_body: torch.Tensor,
        next_gate_body: torch.Tensor,
    ) -> torch.Tensor:
        x_b = current_gate_body[:, 0]
        y_b = current_gate_body[:, 1]
        z_b = current_gate_body[:, 2]
        next_norm = torch.linalg.norm(next_gate_body, dim=1)
        lookahead = torch.where(next_norm.unsqueeze(1) > 1e-6, next_gate_body, current_gate_body)
        look_x = lookahead[:, 0]
        look_y = lookahead[:, 1]

        roll = euler[:, 0]
        pitch = euler[:, 1]
        yaw = euler[:, 2]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        rot = torch.stack((
            torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=1),
            torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=1),
            torch.stack((-sp, cp * sr, cp * cr), dim=1),
        ), dim=1)
        vel_body = torch.bmm(rot.transpose(1, 2), vel.unsqueeze(2)).squeeze(2)

        forward_ref = torch.clamp(x_b, min=0.8)
        look_ref = torch.clamp(look_x, min=1.0)
        side_error = torch.clamp((y_b / forward_ref) * 0.75 + (look_y / look_ref) * 0.25, -1.0, 1.0)
        height_error = torch.clamp(z_b / forward_ref, -0.6, 0.6)
        yaw_error = torch.atan2(y_b, torch.clamp(x_b, min=0.35))

        vz_cmd = torch.clamp(0.95 * height_error - 0.40 * vel_body[:, 2], -0.75, 0.75)
        roll_cmd = torch.clamp(-1.35 * side_error - 0.22 * vel_body[:, 1] - 0.08 * omega[:, 0], -0.65, 0.65)
        forward_bias = 0.42 + 0.12 * torch.tanh((x_b - 1.2) / 0.9)
        pitch_cmd = torch.clamp(
            forward_bias - 0.55 * side_error.abs() - 0.30 * height_error.abs() - 0.10 * omega[:, 1],
            -0.25,
            0.60,
        )
        yaw_cmd = torch.clamp(-0.55 * yaw_error - 0.06 * omega[:, 2], -0.35, 0.35)
        return torch.stack((vz_cmd, roll_cmd, pitch_cmd, yaw_cmd), dim=1)

    def _set_gate_progress_reference(self, env_idx: int) -> None:
        gate = self._current_gate(env_idx)
        if self._use_torch_fast_path and self.sim.qpos_torch is not None:
            pos = self.sim.qpos_torch[env_idx, :3].to(dtype=torch.float64)
            center = torch.as_tensor(gate.center, device=pos.device, dtype=torch.float64)
            normal = torch.as_tensor(gate.normal, device=pos.device, dtype=torch.float64)
            rel = center - pos
            forward_error = float((rel * normal).sum().item())
            distance = float(torch.linalg.norm(rel).item())
        else:
            state = self.states[env_idx]
            assert state is not None
            forward_error, _, _ = gate_relative_geometry(state, gate)
            distance = float(np.linalg.norm(gate.center - state.pos))
        self.current_gate_forward_error[env_idx] = forward_error
        self.gate_start_forward_error[env_idx] = max(0.5, forward_error)
        self.gate_start_distance[env_idx] = max(gate.radius_m, distance)
        if self._use_torch_fast_path:
            assert self._current_gate_forward_error_torch is not None
            assert self._gate_start_distance_torch is not None
            self._current_gate_forward_error_torch[env_idx] = forward_error
            self._gate_start_distance_torch[env_idx] = max(gate.radius_m, distance)

    def _advance_gate(self, env_idx: int, prev_forward_error: float) -> bool:
        task = self.current_tasks[env_idx]
        state = self.states[env_idx]
        assert task is not None and state is not None
        if not gate_passed(
            state,
            self._current_gate(env_idx),
            task.gate_pass_margin_m,
            prev_forward_error,
        ):
            return False
        self.gates_cleared[env_idx] += 1
        if int(self.gate_index[env_idx]) + 1 < len(task.gates):
            self.gate_index[env_idx] += 1
            self._set_gate_progress_reference(env_idx)
            return False
        return True

    def _completion(self, env_idx: int) -> float:
        task = self.current_tasks[env_idx]
        assert task is not None
        total_gates = max(1, len(task.gates))
        fractional = 0.0
        if int(self.gates_cleared[env_idx]) < total_gates:
            if self._use_torch_fast_path and self.sim.qpos_torch is not None and self._gate_start_distance_torch is not None:
                pos = self.sim.qpos_torch[env_idx, :3].to(dtype=torch.float64)
                center = torch.as_tensor(self._current_gate(env_idx).center, device=pos.device, dtype=torch.float64)
                current_distance = float(torch.linalg.norm(center - pos).item())
                start_distance = float(self._gate_start_distance_torch[env_idx].item())
            else:
                state = self.states[env_idx]
                assert state is not None
                current_distance = float(np.linalg.norm(self._current_gate(env_idx).center - state.pos))
                start_distance = self.gate_start_distance[env_idx]
            fractional = max(
                0.0,
                min(
                    1.0,
                    1.0 - current_distance / max(start_distance, 1e-3),
                ),
            )
        return min(1.0, (float(self.gates_cleared[env_idx]) + fractional) / total_gates)

    def _reset_indices(self, indices: NDArray[np.int32]) -> None:
        if len(indices) == 0:
            return

        qpos = np.zeros((len(indices), 7), dtype=np.float64)
        qvel = np.zeros((len(indices), 6), dtype=np.float64)

        for offset, env_idx in enumerate(indices):
            seed = self._seeds[env_idx]
            if seed is not None:
                self.rngs[env_idx] = np.random.default_rng(seed)
            rng = self.rngs[env_idx]

            task = self.stage_controller.sample_task(
                rng=rng,
                base_episode_steps=self.base_episode_steps,
            )
            self.current_tasks[env_idx] = task
            self.gate_index[env_idx] = 0
            self.gates_cleared[env_idx] = 0
            self.step_idx[env_idx] = 0
            self.episode_steps[env_idx] = task.max_steps
            self.last_reward_info[env_idx] = None
            self._prev_actions[env_idx] = 0.0

            quat = euler_to_quat_wxyz(task.spawn_euler)
            qpos[offset, :3] = task.spawn_position
            qpos[offset, 3:7] = quat
            qvel[offset, :3] = task.spawn_velocity
            qvel[offset, 3:6] = task.spawn_omega

        self.sim.reset_worlds(indices, qpos, qvel)
        self._update_task_tensors(indices)
        if self._use_torch_fast_path:
            assert self._gate_index_torch is not None
            assert self._gates_cleared_torch is not None
            assert self._step_idx_torch is not None
            assert self._episode_steps_torch is not None
            idx = torch.as_tensor(indices, device=self._torch_device, dtype=torch.long)
            self._gate_index_torch[idx] = 0
            self._gates_cleared_torch[idx] = 0
            self._step_idx_torch[idx] = 0
            self._episode_steps_torch[idx] = torch.as_tensor(
                self.episode_steps[indices],
                device=self._torch_device,
                dtype=torch.long,
            )
            assert self._prev_actions_torch is not None
            self._prev_actions_torch[idx] = 0.0

        if self._use_torch_fast_path:
            obs = self._build_obs_torch().detach().to(dtype=torch.float32).cpu().numpy()
            for env_idx in indices:
                self.states[env_idx] = None
                self._set_gate_progress_reference(int(env_idx))
                self.buf_obs[int(env_idx)] = obs[int(env_idx)]
                self.reset_infos[int(env_idx)] = {
                    "curriculum_stage": int(self.stage_controller.stage),
                }
        else:
            for env_idx in indices:
                self.states[env_idx] = self._state_for(env_idx)
                self._set_gate_progress_reference(env_idx)
                self.buf_obs[env_idx] = self._build_obs(env_idx)
                self.reset_infos[env_idx] = {
                    "curriculum_stage": int(self.stage_controller.stage),
                }

    def reset(self) -> NDArray[np.float32]:
        indices = np.arange(self.num_envs, dtype=np.int32)
        self._reset_indices(indices)
        self._reset_seeds()
        self._reset_options()
        return self.buf_obs.copy()

    def reset_torch(self) -> torch.Tensor:
        if not self._use_torch_fast_path:
            return torch.as_tensor(self.reset(), dtype=torch.float32)
        indices = np.arange(self.num_envs, dtype=np.int32)
        self._reset_indices(indices)
        self._reset_seeds()
        self._reset_options()
        return self._build_obs_torch().detach().to(dtype=torch.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_torch(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        if not self._use_torch_fast_path:
            obs, rews, dones, infos = self.step(actions.detach().cpu().numpy())
            return (
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(rews, dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.bool),
                infos,
            )

        assert self._torch_device is not None
        assert self.sim.qpos_torch is not None
        assert self.sim.qvel_torch is not None
        assert self._randomization_scale_torch is not None
        assert self._max_distance_torch is not None

        pos = self.sim.qpos_torch[:, :3].to(dtype=torch.float64)
        vel = self.sim.qvel_torch[:, :3].to(dtype=torch.float64)
        euler = self._quat_wxyz_to_euler_torch(self.sim.qpos_torch[:, 3:7].to(dtype=torch.float64))
        omega = self.sim.qvel_torch[:, 3:6].to(dtype=torch.float64)
        cur_center, cur_normal, cur_radius, gate_index_t = self._current_gate_tensors()
        next_index_t = torch.clamp(gate_index_t + 1, max=max(0, self._gate_capacity - 1))
        assert self._env_index_torch is not None
        assert self._gate_centers_torch is not None
        current_gate_body_t = self._body_frame_gate_torch(pos, euler, cur_center)
        next_gate_body_t = self._body_frame_gate_torch(
            pos,
            euler,
            self._gate_centers_torch[self._env_index_torch, next_index_t],
        )
        has_next_t = torch.as_tensor(
            [int(self.gate_index[i]) + 1 < len(self.current_tasks[i].gates) for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.bool,
        )
        next_gate_body_t = torch.where(has_next_t.unsqueeze(1), next_gate_body_t, torch.zeros_like(next_gate_body_t))
        action_t = actions.to(device=self._torch_device, dtype=torch.float64)
        guided_weight = float(np.clip(self.config.sim.guided_action_weight, 0.0, 1.0))
        if guided_weight > 0.0:
            residual_scale = float(max(self.config.sim.residual_action_scale, 0.0))
            base_action_t = self._guided_action_torch(pos, vel, euler, omega, current_gate_body_t, next_gate_body_t)
            action_t = torch.clamp(guided_weight * base_action_t + residual_scale * action_t, -1.0, 1.0)
        smoothing = float(np.clip(self.config.sim.action_smoothing, 0.0, 1.0))
        assert self._prev_actions_torch is not None
        if smoothing < 1.0:
            action_t = self._prev_actions_torch + smoothing * (action_t - self._prev_actions_torch)
        self._prev_actions_torch.copy_(action_t)

        rotor_cmd = self._compute_rotor_commands_torch(action_t, vel, euler, omega)
        noise = torch.randn_like(rotor_cmd) * self.config.sim.thrust_noise_std
        rotor_cmd = torch.clamp(
            rotor_cmd + noise * self._randomization_scale_torch.unsqueeze(1),
            0.0,
            1.0,
        )
        gate_margin = torch.as_tensor(
            [self.current_tasks[i].gate_pass_margin_m for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.float64,
        )
        prev_forward_error_t = torch.as_tensor(
            self.current_gate_forward_error,
            device=self._torch_device,
            dtype=torch.float64,
        )
        self.sim.step(rotor_cmd)

        pos = self.sim.qpos_torch[:, :3].to(dtype=torch.float64)
        vel = self.sim.qvel_torch[:, :3].to(dtype=torch.float64)
        euler = self._quat_wxyz_to_euler_torch(self.sim.qpos_torch[:, 3:7].to(dtype=torch.float64))
        omega = self.sim.qvel_torch[:, 3:6].to(dtype=torch.float64)
        current_forward_error_t, lateral_error_t, _ = self._gate_relative_geometry_torch(
            pos, cur_center, cur_normal
        )
        self.current_gate_forward_error = current_forward_error_t.detach().cpu().numpy()

        terminated_t = torch.zeros(self.num_envs, device=self._torch_device, dtype=torch.bool)
        crash_code_t = torch.zeros(self.num_envs, device=self._torch_device, dtype=torch.int64)

        def apply_crash(mask: torch.Tensor, code: int) -> None:
            nonlocal terminated_t, crash_code_t
            fresh = mask & (~terminated_t)
            terminated_t = terminated_t | fresh
            crash_code_t = torch.where(
                fresh,
                torch.full_like(crash_code_t, code),
                crash_code_t,
            )

        apply_crash(torch.linalg.norm(euler[:, :2], dim=1) > 1.25, 1)
        apply_crash(pos[:, 2] < 0.10, 2)
        apply_crash(pos[:, 2] > 4.5, 3)
        apply_crash(torch.linalg.norm(omega, dim=1) > 18.0, 4)
        apply_crash(current_forward_error_t > self._max_distance_torch, 5)
        apply_crash(lateral_error_t > self._max_distance_torch, 6)

        passed_t = (
            (current_forward_error_t <= gate_margin)
            & (lateral_error_t <= cur_radius)
            & (prev_forward_error_t > gate_margin)
        )
        r_proximity_t = torch.exp(-lateral_error_t / torch.clamp(cur_radius, min=1e-3))
        r_passage_t = passed_t.to(dtype=torch.float64)
        denom_t = torch.maximum(
            torch.full_like(cur_radius, 0.5),
            cur_radius * 2.0,
        )
        r_progress_t = torch.clamp(prev_forward_error_t - current_forward_error_t, min=0.0) / torch.clamp(
            denom_t,
            min=1e-3,
        )
        r_progress_t = torch.clamp(r_progress_t, max=1.0)
        rel_t = cur_center - pos
        dist_t = torch.linalg.norm(rel_t, dim=1)
        direction_t = rel_t / torch.clamp(dist_t.unsqueeze(1), min=1e-4)
        forward_vel_t = (vel * direction_t).sum(dim=1)
        r_align_t = torch.clamp(forward_vel_t / 0.35, 0.0, 1.0)
        lateral_vel_t = vel - cur_normal * (vel * cur_normal).sum(dim=1, keepdim=True)
        r_lat_vel_t = torch.clamp(torch.linalg.norm(lateral_vel_t, dim=1) / 1.2, 0.0, 1.0)
        r_attitude_t = torch.clamp(torch.linalg.norm(euler[:, :2], dim=1) / 0.6, 0.0, 1.0)
        r_rates_t = torch.clamp(torch.linalg.norm(omega, dim=1) / 8.0, 0.0, 1.0)
        r_collision_t = terminated_t.to(dtype=torch.float64)
        r_effort_t = torch.linalg.norm(action_t, dim=1) / np.sqrt(max(action_t.shape[1], 1))

        cfg = self.config.reward
        reward_t = (
            cfg.gate_proximity * r_proximity_t
            + cfg.gate_passage_bonus * r_passage_t
            + cfg.progress * r_progress_t
            + cfg.velocity_alignment * r_align_t
            + cfg.lateral_velocity_penalty * r_lat_vel_t
            + cfg.attitude_stability * r_attitude_t
            + cfg.angular_rate_stability * r_rates_t
            + cfg.time_penalty
            + cfg.collision_penalty * r_collision_t
            + cfg.control_effort * r_effort_t
            + cfg.alive_bonus
        ).to(dtype=torch.float32)

        pre_reset_obs_t = self._build_obs_torch().detach().to(dtype=torch.float32)
        dones_t = torch.zeros(self.num_envs, device=self._torch_device, dtype=torch.bool)
        done_indices: list[int] = []
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        crash_map = {
            0: "none",
            1: "flip",
            2: "ground",
            3: "altitude",
            4: "spin",
            5: "behind_course",
            6: "off_line",
        }
        for env_idx in range(self.num_envs):
            task = self.current_tasks[env_idx]
            assert task is not None
            success = False
            if bool(passed_t[env_idx].item()):
                self.gates_cleared[env_idx] += 1
                if int(self.gate_index[env_idx]) + 1 < len(task.gates):
                    self.gate_index[env_idx] += 1
                    self._set_gate_progress_reference(env_idx)
                else:
                    success = True
            truncated = int(self.step_idx[env_idx]) + 1 >= int(self.episode_steps[env_idx])
            self.step_idx[env_idx] += 1
            crash_type = "success" if success else crash_map[int(crash_code_t[env_idx].item())]
            terminated = bool(terminated_t[env_idx].item())
            if truncated and not terminated and not success:
                crash_type = "timeout"
            done = success or terminated or truncated
            dones_t[env_idx] = done
            info: dict[str, Any] = {
                "curriculum_stage": int(self.stage_controller.stage),
                "gates_cleared": int(self.gates_cleared[env_idx]),
                "completion": self._completion(env_idx),
                "crash_type": crash_type,
                "reward_gate_proximity": float(r_proximity_t[env_idx].item()),
                "reward_gate_passage": float(r_passage_t[env_idx].item()),
                "reward_progress": float(r_progress_t[env_idx].item()),
                "reward_velocity_alignment": float(r_align_t[env_idx].item()),
                "reward_lateral_velocity_penalty": float(r_lat_vel_t[env_idx].item()),
                "reward_attitude_stability": float(r_attitude_t[env_idx].item()),
                "reward_angular_rate_stability": float(r_rates_t[env_idx].item()),
                "reward_control_effort": float(r_effort_t[env_idx].item()),
                "TimeLimit.truncated": truncated and not terminated,
            }
            if done:
                summary = EpisodeSummary(
                    stage=task.stage,
                    success=success,
                    terminated=terminated,
                    truncated=truncated and not success and not terminated,
                    crash_type=crash_type,
                    completion=info["completion"],
                    score=(
                        0.60 * info["completion"]
                        + 0.25 * float(success)
                        + 0.15 * min(1.0, int(self.step_idx[env_idx]) / max(1.0, float(self.episode_steps[env_idx])))
                    ),
                    gates_cleared=int(self.gates_cleared[env_idx]),
                    steps=int(self.step_idx[env_idx]),
                )
                self.stage_controller.record_episode(summary)
                info["terminal_observation"] = pre_reset_obs_t[env_idx].detach().cpu().numpy()
                done_indices.append(env_idx)
            infos[env_idx] = info

        if done_indices:
            self._reset_indices(np.asarray(done_indices, dtype=np.int32))

        obs_t = self._build_obs_torch().detach().to(dtype=torch.float32)
        return obs_t, reward_t, dones_t, infos

    def step_torch_train(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._use_torch_fast_path:
            obs, rews, dones, _ = self.step(actions.detach().cpu().numpy())
            return (
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(rews, dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.bool),
            )

        assert self._torch_device is not None
        assert self.sim.qpos_torch is not None
        assert self.sim.qvel_torch is not None
        assert self._randomization_scale_torch is not None
        assert self._max_distance_torch is not None
        assert self._gate_index_torch is not None
        assert self._gates_cleared_torch is not None
        assert self._step_idx_torch is not None
        assert self._episode_steps_torch is not None
        assert self._current_gate_forward_error_torch is not None
        assert self._gate_start_distance_torch is not None
        assert self._env_index_torch is not None

        pos = self.sim.qpos_torch[:, :3].to(dtype=torch.float64)
        vel = self.sim.qvel_torch[:, :3].to(dtype=torch.float64)
        euler = self._quat_wxyz_to_euler_torch(self.sim.qpos_torch[:, 3:7].to(dtype=torch.float64))
        omega = self.sim.qvel_torch[:, 3:6].to(dtype=torch.float64)
        cur_center, cur_normal, cur_radius, gate_index_t = self._current_gate_tensors()
        next_index_t = torch.clamp(gate_index_t + 1, max=max(0, self._gate_capacity - 1))
        assert self._gate_centers_torch is not None
        current_gate_body_t = self._body_frame_gate_torch(pos, euler, cur_center)
        next_gate_body_t = self._body_frame_gate_torch(
            pos,
            euler,
            self._gate_centers_torch[self._env_index_torch, next_index_t],
        )
        has_next_t = torch.as_tensor(
            [int(self.gate_index[i]) + 1 < len(self.current_tasks[i].gates) for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.bool,
        )
        next_gate_body_t = torch.where(has_next_t.unsqueeze(1), next_gate_body_t, torch.zeros_like(next_gate_body_t))
        action_t = actions.to(device=self._torch_device, dtype=torch.float64)
        guided_weight = float(np.clip(self.config.sim.guided_action_weight, 0.0, 1.0))
        if guided_weight > 0.0:
            residual_scale = float(max(self.config.sim.residual_action_scale, 0.0))
            base_action_t = self._guided_action_torch(pos, vel, euler, omega, current_gate_body_t, next_gate_body_t)
            action_t = torch.clamp(guided_weight * base_action_t + residual_scale * action_t, -1.0, 1.0)
        smoothing = float(np.clip(self.config.sim.action_smoothing, 0.0, 1.0))
        assert self._prev_actions_torch is not None
        if smoothing < 1.0:
            action_t = self._prev_actions_torch + smoothing * (action_t - self._prev_actions_torch)
        self._prev_actions_torch.copy_(action_t)
        rotor_cmd = self._compute_rotor_commands_torch(action_t, vel, euler, omega)
        noise = torch.randn_like(rotor_cmd) * self.config.sim.thrust_noise_std
        rotor_cmd = torch.clamp(
            rotor_cmd + noise * self._randomization_scale_torch.unsqueeze(1),
            0.0,
            1.0,
        )
        gate_margin = torch.as_tensor(
            [self.current_tasks[i].gate_pass_margin_m for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.float64,
        )
        prev_forward_error_t = self._current_gate_forward_error_torch.clone()
        self.sim.step(rotor_cmd)

        pos = self.sim.qpos_torch[:, :3].to(dtype=torch.float64)
        vel = self.sim.qvel_torch[:, :3].to(dtype=torch.float64)
        euler = self._quat_wxyz_to_euler_torch(self.sim.qpos_torch[:, 3:7].to(dtype=torch.float64))
        omega = self.sim.qvel_torch[:, 3:6].to(dtype=torch.float64)
        current_forward_error_t, lateral_error_t, _ = self._gate_relative_geometry_torch(pos, cur_center, cur_normal)
        self._current_gate_forward_error_torch.copy_(current_forward_error_t)

        terminated_t = torch.zeros(self.num_envs, device=self._torch_device, dtype=torch.bool)
        terminated_t |= torch.linalg.norm(euler[:, :2], dim=1) > 1.25
        terminated_t |= pos[:, 2] < 0.10
        terminated_t |= pos[:, 2] > 4.5
        terminated_t |= torch.linalg.norm(omega, dim=1) > 18.0
        terminated_t |= current_forward_error_t > self._max_distance_torch
        terminated_t |= lateral_error_t > self._max_distance_torch

        passed_t = (
            (current_forward_error_t <= gate_margin)
            & (lateral_error_t <= cur_radius)
            & (prev_forward_error_t > gate_margin)
        )

        cfg = self.config.reward
        r_proximity_t = torch.exp(-lateral_error_t / torch.clamp(cur_radius, min=1e-3))
        r_passage_t = passed_t.to(dtype=torch.float64)
        denom_t = torch.maximum(torch.full_like(cur_radius, 0.5), cur_radius * 2.0)
        r_progress_t = torch.clamp(prev_forward_error_t - current_forward_error_t, min=0.0) / torch.clamp(
            denom_t, min=1e-3
        )
        r_progress_t = torch.clamp(r_progress_t, max=1.0)
        rel_t = cur_center - pos
        dist_t = torch.linalg.norm(rel_t, dim=1)
        direction_t = rel_t / torch.clamp(dist_t.unsqueeze(1), min=1e-4)
        forward_vel_t = (vel * direction_t).sum(dim=1)
        r_align_t = torch.clamp(forward_vel_t / 0.35, 0.0, 1.0)
        lateral_vel_t = vel - cur_normal * (vel * cur_normal).sum(dim=1, keepdim=True)
        r_lat_vel_t = torch.clamp(torch.linalg.norm(lateral_vel_t, dim=1) / 1.2, 0.0, 1.0)
        r_attitude_t = torch.clamp(torch.linalg.norm(euler[:, :2], dim=1) / 0.6, 0.0, 1.0)
        r_rates_t = torch.clamp(torch.linalg.norm(omega, dim=1) / 8.0, 0.0, 1.0)
        r_collision_t = terminated_t.to(dtype=torch.float64)
        r_effort_t = torch.linalg.norm(action_t, dim=1) / np.sqrt(max(action_t.shape[1], 1))
        reward_t = (
            cfg.gate_proximity * r_proximity_t
            + cfg.gate_passage_bonus * r_passage_t
            + cfg.progress * r_progress_t
            + cfg.velocity_alignment * r_align_t
            + cfg.lateral_velocity_penalty * r_lat_vel_t
            + cfg.attitude_stability * r_attitude_t
            + cfg.angular_rate_stability * r_rates_t
            + cfg.time_penalty
            + cfg.collision_penalty * r_collision_t
            + cfg.control_effort * r_effort_t
            + cfg.alive_bonus
        ).to(dtype=torch.float32)

        advanced_mask = passed_t.clone()
        task_lengths = torch.as_tensor(
            [len(self.current_tasks[i].gates) for i in range(self.num_envs)],
            device=self._torch_device,
            dtype=torch.long,
        )
        success_t = passed_t & (gate_index_t + 1 >= task_lengths)
        advance_only_t = passed_t & (~success_t)
        self._gates_cleared_torch += passed_t.to(dtype=torch.long)
        self._gate_index_torch += advance_only_t.to(dtype=torch.long)
        self._step_idx_torch += 1
        truncated_t = self._step_idx_torch >= self._episode_steps_torch
        done_t = terminated_t | truncated_t | success_t

        if advance_only_t.any():
            new_center, new_normal, new_radius, _ = self._current_gate_tensors()
            rel = new_center - pos
            self._current_gate_forward_error_torch[advance_only_t] = (rel[advance_only_t] * new_normal[advance_only_t]).sum(dim=1)
            self._gate_start_distance_torch[advance_only_t] = torch.maximum(
                torch.linalg.norm(rel[advance_only_t], dim=1),
                new_radius[advance_only_t],
            )

        done_indices_t = torch.nonzero(done_t, as_tuple=False).flatten()
        if done_indices_t.numel() > 0:
            done_indices = done_indices_t.detach().cpu().numpy().astype(np.int32)
            self.gate_index = self._gate_index_torch.detach().cpu().numpy().astype(np.int32)
            self.gates_cleared = self._gates_cleared_torch.detach().cpu().numpy().astype(np.int32)
            self.step_idx = self._step_idx_torch.detach().cpu().numpy().astype(np.int32)
            self.current_gate_forward_error = self._current_gate_forward_error_torch.detach().cpu().numpy()
            self.gate_start_distance = self._gate_start_distance_torch.detach().cpu().numpy()
            for env_idx in done_indices:
                task = self.current_tasks[int(env_idx)]
                assert task is not None
                completion = self._completion(int(env_idx))
                summary = EpisodeSummary(
                    stage=task.stage,
                    success=bool(success_t[int(env_idx)].item()),
                    terminated=bool(terminated_t[int(env_idx)].item()),
                    truncated=bool(truncated_t[int(env_idx)].item() and not success_t[int(env_idx)].item() and not terminated_t[int(env_idx)].item()),
                    crash_type="success" if bool(success_t[int(env_idx)].item()) else "none",
                    completion=completion,
                    score=(
                        0.60 * completion
                        + 0.25 * float(bool(success_t[int(env_idx)].item()))
                        + 0.15 * min(1.0, int(self.step_idx[int(env_idx)]) / max(1.0, float(self.episode_steps[int(env_idx)])))
                    ),
                    gates_cleared=int(self.gates_cleared[int(env_idx)]),
                    steps=int(self.step_idx[int(env_idx)]),
                )
                self.stage_controller.record_episode(summary)
            self._reset_indices(done_indices)
            obs_t = self._build_obs_torch().detach().to(dtype=torch.float32)
        else:
            obs_t = self._build_obs_torch().detach().to(dtype=torch.float32)

        return obs_t, reward_t, done_t

    def _torch_step_wait(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        obs_t, rew_t, done_t, infos = self.step_torch(
            torch.as_tensor(self._actions, device=self._torch_device, dtype=torch.float32)
        )
        obs_np = obs_t.detach().cpu().numpy()
        rew_np = rew_t.detach().cpu().numpy()
        done_np = done_t.detach().cpu().numpy()
        self.buf_obs[:] = obs_np
        self.buf_rews[:] = rew_np
        self.buf_dones[:] = done_np
        self.buf_infos = infos
        return (
            self.buf_obs.copy(),
            self.buf_rews.copy(),
            self.buf_dones.copy(),
            [info.copy() for info in infos],
        )

    def step_wait(self) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        if self._use_torch_fast_path:
            return self._torch_step_wait()
        rotor_cmd = np.zeros((self.num_envs, 4), dtype=np.float64)
        clipped_actions = np.clip(self._actions.astype(np.float64), -1.0, 1.0)
        for env_idx in range(self.num_envs):
            clipped_actions[env_idx] = self._apply_guided_action_cpu(env_idx, clipped_actions[env_idx])
        smoothing = float(np.clip(self.config.sim.action_smoothing, 0.0, 1.0))
        if smoothing < 1.0:
            clipped_actions = self._prev_actions + smoothing * (clipped_actions - self._prev_actions)
        self._prev_actions[:] = clipped_actions

        for env_idx in range(self.num_envs):
            state = self.states[env_idx]
            task = self.current_tasks[env_idx]
            assert state is not None and task is not None
            rotor_cmd[env_idx] = compute_rotor_commands(
                clipped_actions[env_idx],
                state,
                self.config.drone,
                float(self.sim.mass_scale[env_idx]),
                self.config.sim.gravity,
            )
            noise = self.rngs[env_idx].normal(0.0, self.config.sim.thrust_noise_std, size=4)
            rotor_cmd[env_idx] = np.clip(
                rotor_cmd[env_idx] + noise * task.randomization_scale,
                0.0,
                1.0,
            )

        prev_forward_error = self.current_gate_forward_error.copy()
        self.sim.step(rotor_cmd)

        done_indices: list[int] = []
        for env_idx in range(self.num_envs):
            self.states[env_idx] = self._state_for(env_idx)
            state = self.states[env_idx]
            task = self.current_tasks[env_idx]
            assert state is not None and task is not None

            self.current_gate_forward_error[env_idx], _, _ = gate_relative_geometry(
                state, self._current_gate(env_idx)
            )

            terminated, crash_type = compute_termination(
                state,
                self._current_gate(env_idx),
                task.max_distance_m,
            )
            reward_info = compute_total_reward(
                state,
                clipped_actions[env_idx],
                self._current_gate(env_idx),
                float(prev_forward_error[env_idx]),
                self.config.reward,
                terminated,
                task.gate_pass_margin_m,
            )
            self.last_reward_info[env_idx] = reward_info

            success = self._advance_gate(env_idx, float(prev_forward_error[env_idx]))
            truncated = int(self.step_idx[env_idx]) + 1 >= int(self.episode_steps[env_idx])
            self.step_idx[env_idx] += 1

            if success:
                crash_type = "success"
            elif truncated and not terminated:
                crash_type = "timeout"

            done = success or terminated or truncated
            obs = self._build_obs(env_idx)
            self.buf_obs[env_idx] = obs
            self.buf_rews[env_idx] = reward_info.total
            self.buf_dones[env_idx] = done

            info: dict[str, Any] = {
                "curriculum_stage": int(self.stage_controller.stage),
                "gates_cleared": int(self.gates_cleared[env_idx]),
                "completion": self._completion(env_idx),
                "crash_type": crash_type,
                "reward_gate_proximity": reward_info.gate_proximity,
                "reward_gate_passage": reward_info.gate_passage,
                "reward_progress": reward_info.progress,
                "reward_velocity_alignment": reward_info.velocity_alignment,
                "reward_lateral_velocity_penalty": reward_info.lateral_velocity_penalty,
                "reward_attitude_stability": reward_info.attitude_stability,
                "reward_angular_rate_stability": reward_info.angular_rate_stability,
                "reward_control_effort": reward_info.control_effort,
                "TimeLimit.truncated": truncated and not terminated,
            }

            if done:
                summary = EpisodeSummary(
                    stage=task.stage,
                    success=success,
                    terminated=terminated,
                    truncated=truncated and not success and not terminated,
                    crash_type=crash_type,
                    completion=info["completion"],
                    score=(
                        0.60 * info["completion"]
                        + 0.25 * float(success)
                        + 0.15 * min(1.0, int(self.step_idx[env_idx]) / max(1.0, float(self.episode_steps[env_idx])))
                    ),
                    gates_cleared=int(self.gates_cleared[env_idx]),
                    steps=int(self.step_idx[env_idx]),
                )
                self.stage_controller.record_episode(summary)
                info["terminal_observation"] = obs.copy()
                done_indices.append(env_idx)

            self.buf_infos[env_idx] = info

        if done_indices:
            reset_indices = np.asarray(done_indices, dtype=np.int32)
            self._reset_indices(reset_indices)

        return (
            self.buf_obs.copy(),
            self.buf_rews.copy(),
            self.buf_dones.copy(),
            [info.copy() for info in self.buf_infos],
        )

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_indices = list(self._get_indices(indices))
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
            return [value for _ in target_indices]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_indices = list(self._get_indices(indices))
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
            return
        raise AttributeError(attr_name)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        if not hasattr(self, method_name):
            raise AttributeError(method_name)
        method = getattr(self, method_name)
        target_indices = list(self._get_indices(indices))
        return [method(*method_args, **method_kwargs) for _ in target_indices]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> list[None]:
        return [None] * self.num_envs
