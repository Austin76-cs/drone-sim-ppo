from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from dronesim.config import RuntimeConfig
from dronesim.sim.attitude_controller import compute_rotor_commands
from dronesim.sim.env import MuJoCoSim, euler_to_quat_wxyz
from dronesim.tasks.curriculum import (
    CurriculumStage,
    EpisodeSummary,
    EpisodeTask,
    StageController,
)
from dronesim.tasks.rewards import (
    body_frame_gate,
    compute_total_reward,
    gate_crossing_quality,
    gate_missed,
    gate_passed,
    gate_relative_geometry,
)
from dronesim.tasks.termination import check_gate_collision, compute_termination
from dronesim.types import DroneState, GateSpec, RewardInfo


class DroneRaceEnv(gym.Env):
    """Gymnasium env for drone gate racing with PPO (CTBR architecture).

    Observation (34,): [pos(3), vel(3), rot_matrix(9), omega(3), prev_action(4),
                        body_gate_cur(3), body_gate_next(3), body_gate_+2(3), body_gate_+3(3)]
    Gates beyond the course end are represented as zero vectors.
    Action (4,): [thrust, roll_rate, pitch_rate, yaw_rate] in [-1, 1] (CTBR)
    """

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 34  # 3+3+9+3+4+12

    def __init__(
        self,
        config: RuntimeConfig,
        stage_controller: StageController | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.sim = MuJoCoSim(config)
        self.stage_controller = stage_controller or StageController(config.task)
        self.rng = np.random.default_rng(config.seed)

        self.base_episode_steps = max(
            1, int(config.sim.episode_seconds * config.sim.policy_hz)
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Episode state
        self.current_task: EpisodeTask | None = None
        self.state: DroneState | None = None
        self.gate_index = 0
        self.gates_cleared = 0
        self.passed_gate_indices: set[int] = set()  # tracks actually passed gates
        self.step_idx = 0
        self.episode_steps = self.base_episode_steps
        self.current_gate_forward_error = 0.0
        self.gate_start_forward_error = 1.0
        self.last_reward_info: RewardInfo | None = None
        self.prev_pos: NDArray[np.float64] | None = None
        self.prev_action: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        self.miss_count = 0
        self.crossing_offsets: list[float] = []  # how centered each gate pass was

        # Stall detection: terminate if no progress toward gate for 5 seconds
        self._stall_window = int(5.0 * config.sim.policy_hz)
        self._stall_best_error = float("inf")
        self._stall_counter = 0

        # Action latency buffer
        latency = max(0, config.sim.action_latency_steps)
        self._action_buffer: list[NDArray[np.float64]] = [np.zeros(4, dtype=np.float64)] * (latency + 1)

    def _current_gate(self) -> GateSpec:
        assert self.current_task is not None
        idx = min(self.gate_index, len(self.current_task.gates) - 1)
        return self.current_task.gates[idx]

    def _next_gate(self) -> GateSpec | None:
        assert self.current_task is not None
        next_idx = self.gate_index + 1
        if next_idx < len(self.current_task.gates):
            return self.current_task.gates[next_idx]
        return None

    def _gate_at_offset(self, offset: int) -> NDArray[np.float64]:
        """Return gate body-frame vector at gate_index+offset, or zeros if past end."""
        assert self.current_task is not None and self.state is not None
        idx = self.gate_index + offset
        if idx < len(self.current_task.gates):
            return body_frame_gate(self.state, self.current_task.gates[idx])
        return np.zeros(3, dtype=np.float64)

    def _build_obs(self) -> NDArray[np.float32]:
        assert self.state is not None and self.current_task is not None
        sim_cfg = self.config.sim
        scale = self.current_task.randomization_scale

        # Apply observation noise scaled by difficulty
        pos = self.state.pos + self.rng.normal(0, sim_cfg.obs_noise_pos_std * scale, 3)
        vel = self.state.vel + self.rng.normal(0, sim_cfg.obs_noise_vel_std * scale, 3)
        omega = self.state.omega + self.rng.normal(0, sim_cfg.obs_noise_omega_std * scale, 3)

        obs = np.concatenate([
            pos,                                     # 3
            vel,                                     # 3
            self.state.rot_matrix.flatten(),          # 9
            omega,                                   # 3
            self.prev_action,                        # 4
            self._gate_at_offset(0),                 # 3
            self._gate_at_offset(1),                 # 3
            self._gate_at_offset(2),                 # 3
            self._gate_at_offset(3),                 # 3
        ]).astype(np.float32)
        return obs

    def _set_gate_progress_reference(self) -> None:
        assert self.state is not None
        forward_error, _, _ = gate_relative_geometry(self.state, self._current_gate())
        self.current_gate_forward_error = forward_error
        self.gate_start_forward_error = max(0.5, forward_error)

    def _reset_stall_tracker(self) -> None:
        self._stall_best_error = float("inf")
        self._stall_counter = 0

    def _advance_gate(self) -> bool:
        """Check gate passage. Returns True if all gates cleared (success)."""
        assert self.current_task is not None and self.state is not None
        passed, offset = gate_crossing_quality(
            self.state, self._current_gate(),
            self.current_task.gate_pass_margin_m, self.prev_pos,
        )
        if not passed:
            return False
        self.passed_gate_indices.add(self.gate_index)
        self.crossing_offsets.append(offset)
        self.gates_cleared += 1
        if self.gate_index + 1 < len(self.current_task.gates):
            self.gate_index += 1
            self._set_gate_progress_reference()
            self._reset_stall_tracker()
            return False
        return True

    def _completion(self) -> float:
        assert self.current_task is not None
        total_gates = max(1, len(self.current_task.gates))
        fractional = 0.0
        if self.gates_cleared < total_gates:
            fractional = max(0.0, min(1.0,
                (self.gate_start_forward_error - self.current_gate_forward_error)
                / max(self.gate_start_forward_error, 1e-3)
            ))
        return min(1.0, (self.gates_cleared + fractional) / total_gates)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_task = self.stage_controller.sample_task(
            rng=self.rng,
            base_episode_steps=self.base_episode_steps,
        )
        self.episode_steps = self.current_task.max_steps
        self.gate_index = 0
        self.gates_cleared = 0
        self.passed_gate_indices = set()
        self.step_idx = 0
        self.miss_count = 0
        self.crossing_offsets = []
        self.last_reward_info = None
        self.prev_action = np.zeros(4, dtype=np.float64)
        self._stall_best_error = float("inf")
        self._stall_counter = 0
        latency = max(0, self.config.sim.action_latency_steps)
        self._action_buffer = [np.zeros(4, dtype=np.float64)] * (latency + 1)

        # Build qpos/qvel for MuJoCo reset
        quat = euler_to_quat_wxyz(self.current_task.spawn_euler)
        qpos = np.zeros(7, dtype=np.float64)
        qpos[:3] = self.current_task.spawn_position
        qpos[3:7] = quat
        qvel = np.zeros(6, dtype=np.float64)
        qvel[:3] = self.current_task.spawn_velocity
        qvel[3:6] = self.current_task.spawn_omega

        self.sim.apply_randomization(self.current_task.randomization_scale, self.rng)
        self.state = self.sim.reset(qpos, qvel)
        self.prev_pos = self.state.pos.copy()
        self._set_gate_progress_reference()

        obs = self._build_obs()
        info: dict[str, Any] = {
            "curriculum_stage": int(self.stage_controller.stage),
        }
        return obs, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self.current_task is not None and self.state is not None

        action = np.clip(action.astype(np.float64), -1.0, 1.0)

        # Action latency: buffer the action and use the delayed one
        self._action_buffer.append(action.copy())
        delayed_action = self._action_buffer.pop(0)

        # CTBR controller -> rotor commands (uses delayed action)
        rotor_cmd = compute_rotor_commands(
            delayed_action, self.state, self.config.drone,
            self.sim.mass_scale, self.config.sim.gravity,
        )

        # Add thrust noise
        noise = self.rng.normal(0.0, self.config.sim.thrust_noise_std, size=4)
        rotor_cmd = np.clip(rotor_cmd + noise * self.current_task.randomization_scale, 0.0, 1.0)

        # Step physics
        prev_forward_error = self.current_gate_forward_error
        self.prev_pos = self.state.pos.copy()
        self.state = self.sim.step(rotor_cmd)

        # Update forward error
        self.current_gate_forward_error, _, _ = gate_relative_geometry(
            self.state, self._current_gate()
        )

        # Check termination (hard crashes only)
        terminated, crash_type = compute_termination(
            self.state, self._current_gate(), self.current_task.max_distance_m,
        )

        # Soft collision: penalty but no termination
        gate_contact = check_gate_collision(self.state, self._current_gate())

        # Compute reward (includes gate_miss_penalty if drone crossed gate plane off-center)
        reward_info = compute_total_reward(
            self.state, action, self._current_gate(),
            prev_forward_error, self.config.reward,
            terminated, self.current_task.gate_pass_margin_m,
            self.prev_pos,
            self.prev_action,
            gate_contact,
        )
        self.last_reward_info = reward_info
        self.prev_action = action.copy()

        # Check gate passage
        success = self._advance_gate()

        # Soft miss termination: allow misses before ending the episode.
        # SPRINT gets 4 misses (10 gates, tight turns) to provide more learning signal.
        # Other stages get 2 misses.
        # On a miss, advance the gate pointer so the drone always targets the next gate
        # rather than getting stuck chasing a gate it already flew past.
        if not success and reward_info.gate_miss > 0.5:
            self.miss_count += 1
            if self.gate_index + 1 < len(self.current_task.gates):
                self.gate_index += 1
                self._set_gate_progress_reference()
                self._reset_stall_tracker()
            if self.current_task.stage == CurriculumStage.SPRINT:
                # Scale miss limit with gate count: ~60% of gates allowed as misses
                miss_limit = max(3, int(len(self.current_task.gates) * 0.6))
            else:
                miss_limit = self.config.task.default_miss_limit
            if not terminated and self.miss_count >= miss_limit:
                terminated = True
                crash_type = "miss_limit"

        # Stall detection: terminate if no forward progress for too long
        if not terminated and not success:
            if self.current_gate_forward_error < self._stall_best_error - 0.05:
                self._stall_best_error = self.current_gate_forward_error
                self._stall_counter = 0
            else:
                self._stall_counter += 1
            if self._stall_counter >= self._stall_window:
                terminated = True
                crash_type = "stall"

        # Gate passage bonus handled in reward via gate_passage_reward
        truncated = self.step_idx + 1 >= self.episode_steps
        self.step_idx += 1

        if success:
            crash_type = "success"
        elif truncated and not terminated:
            crash_type = "timeout"

        done = success or terminated or truncated

        # Record episode summary on done
        if done:
            completion = self._completion()
            score = (
                0.40 * completion
                + 0.45 * float(success)
                + 0.15 * min(1.0, self.step_idx / max(1.0, float(self.episode_steps)))
            )
            summary = EpisodeSummary(
                stage=self.current_task.stage,
                success=success,
                terminated=terminated,
                truncated=truncated and not success and not terminated,
                crash_type=crash_type,
                completion=completion,
                score=score,
                gates_cleared=self.gates_cleared,
                steps=self.step_idx,
            )
            self.stage_controller.record_episode(summary)

        obs = self._build_obs()
        info: dict[str, Any] = {
            "curriculum_stage": int(self.stage_controller.stage),
            "gates_cleared": self.gates_cleared,
            "completion": self._completion(),
            "crash_type": crash_type,
            "reward_gate_proximity": reward_info.gate_proximity,
            "reward_gate_passage": reward_info.gate_passage,
            "reward_progress": reward_info.progress,
            "reward_velocity_alignment": reward_info.velocity_alignment,
            "reward_forward_speed": reward_info.forward_speed,
            "reward_control_effort": reward_info.control_effort,
            "reward_gate_centering": reward_info.gate_centering,
            "crossing_offsets": list(self.crossing_offsets),
            "avg_crossing_offset": (
                sum(self.crossing_offsets) / len(self.crossing_offsets)
                if self.crossing_offsets else float("nan")
            ),
        }

        return obs, reward_info.total, terminated or success, truncated, info
