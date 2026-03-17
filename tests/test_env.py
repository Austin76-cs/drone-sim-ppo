from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage, EpisodeTask
from dronesim.types import DroneState, GateSpec


class TestDroneRaceEnv(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path("configs/default.toml"))
        self.env = DroneRaceEnv(self.config)

    def test_reset_shape(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (18,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn("curriculum_stage", info)

    def test_step_shape(self):
        self.env.reset()
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, (18,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))

    def test_random_steps(self):
        self.env.reset()
        for _ in range(50):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertTrue(np.all(self.env.action_space.low == -1.0))
        self.assertTrue(np.all(self.env.action_space.high == 1.0))

    def test_completion_stays_below_full_for_missed_gate(self):
        gates = (
            GateSpec(center=np.array([2.5, 0.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]), radius_m=0.45, depth_m=0.15),
            GateSpec(center=np.array([5.0, 0.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]), radius_m=0.45, depth_m=0.15),
            GateSpec(center=np.array([7.5, 0.0, 1.0]), normal=np.array([1.0, 0.0, 0.0]), radius_m=0.45, depth_m=0.15),
        )
        self.env.current_task = EpisodeTask(
            stage=CurriculumStage.INTRO,
            spawn_position=np.zeros(3, dtype=np.float64),
            spawn_velocity=np.zeros(3, dtype=np.float64),
            spawn_euler=np.zeros(3, dtype=np.float64),
            spawn_omega=np.zeros(3, dtype=np.float64),
            gates=gates,
            max_steps=100,
            max_distance_m=5.0,
            randomization_scale=0.0,
            gate_pass_margin_m=0.12,
        )
        self.env.state = DroneState(
            pos=np.array([8.2, 1.0, 1.0], dtype=np.float64),
            vel=np.zeros(3, dtype=np.float64),
            euler=np.zeros(3, dtype=np.float64),
            omega=np.zeros(3, dtype=np.float64),
            motor=np.zeros(4, dtype=np.float64),
        )
        self.env.gates_cleared = 2
        self.env.gate_index = 2
        self.env.gate_start_distance = 2.0

        self.assertLess(self.env._completion(), 1.0)


if __name__ == "__main__":
    unittest.main()
