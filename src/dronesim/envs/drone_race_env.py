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
    gate_passed,
    gate_relative_geometry,
)
from dronesim.tasks.termination import compute_termination
from dronesim.types import DroneState, GateSpec, RewardInfo


class DroneRaceEnv(gym.Env):
    """Gymnasium env for drone gate racing with PPO.

    Observation (18,): [pos(3), vel(3), euler(3), omega(3), body_gate_cur(3), body_gate_next(3)]
    Action (4,): [vz, roll, pitch, yaw_rate] in [-1, 1]
    """

    metadata = {"render_modes": ["human"]}

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
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Episode state
        self.current_task: EpisodeTask | None = None
        self.state: DroneState | None = None
        self.gate_index = 0
        self.gates_cleared = 0
        self.step_idx = 0
        self.episode_steps = self.base_episode_steps
        self.current_gate_forward_error = 0.0
        self.gate_start_forward_error = 1.0
        self.last_reward_info: RewardInfo | None = None
        self.prev_pos: NDArray[np.float64] | None = None

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

    def _build_obs(self) -> NDArray[np.float32]:
        assert self.state is not None
        cur_gate_body = body_frame_gate(self.state, self._current_gate())
        next_gate = self._next_gate()
        if next_gate is not None:
            next_gate_body = body_frame_gate(self.state, next_gate)
        else:
            next_gate_body = np.zeros(3, dtype=np.float64)

        obs = np.concatenate([
            self.state.pos,
            self.state.vel,
            self.state.euler,
            self.state.omega,
            cur_gate_body,
            next_gate_body,
        ]).astype(np.float32)
        return obs

    def _set_gate_progress_reference(self) -> None:
        assert self.state is not None
        forward_error, _, _ = gate_relative_geometry(self.state, self._current_gate())
        self.current_gate_forward_error = forward_error
        self.gate_start_forward_error = max(0.5, forward_error)

    def _advance_gate(self) -> bool:
        """Check gate passage. Returns True if all gates cleared (success)."""
        assert self.current_task is not None and self.state is not None
        if not gate_passed(self.state, self._current_gate(), self.current_task.gate_pass_margin_m, self.prev_pos):
            return False
        self.gates_cleared += 1
        if self.gate_index + 1 < len(self.current_task.gates):
            self.gate_index += 1
            self._set_gate_progress_reference()
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
        self.step_idx = 0
        self.last_reward_info = None

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

        # Attitude controller -> rotor commands
        rotor_cmd = compute_rotor_commands(
            action, self.state, self.config.drone,
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

        # Check termination
        terminated, crash_type = compute_termination(
            self.state, self._current_gate(), self.current_task.max_distance_m,
        )

        # Compute reward
        reward_info = compute_total_reward(
            self.state, action, self._current_gate(),
            prev_forward_error, self.config.reward,
            terminated, self.current_task.gate_pass_margin_m,
        )
        self.last_reward_info = reward_info

        # Check gate passage
        success = self._advance_gate()

        # Add gate passage bonus already handled in reward via gate_passage_reward
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
                0.60 * completion
                + 0.25 * float(success)
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
            "reward_control_effort": reward_info.control_effort,
        }

        return obs, reward_info.total, terminated or success, truncated, info
