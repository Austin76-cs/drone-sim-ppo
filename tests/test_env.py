from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv


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


if __name__ == "__main__":
    unittest.main()
