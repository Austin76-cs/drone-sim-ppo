from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from dronesim.config import load_config
from dronesim.envs.warp_vec_env import WarpVecDroneRaceEnv


class TestWarpVecDroneRaceEnv(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config(Path("configs/default.toml")))
        self.config.device = "cpu"
        self.config.sim.backend_device = "cpu"
        self.env = WarpVecDroneRaceEnv(self.config, num_envs=2)

    def tearDown(self):
        self.env.close()

    def test_reset_shape(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (2, 18))
        self.assertEqual(obs.dtype, np.float32)

    def test_step_shape(self):
        self.env.reset()
        actions = np.zeros((2, 4), dtype=np.float32)
        obs, rewards, dones, infos = self.env.step(actions)
        self.assertEqual(obs.shape, (2, 18))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(dones.shape, (2,))
        self.assertEqual(len(infos), 2)

    def test_done_worlds_auto_reset(self):
        self.env.reset()
        self.env.episode_steps[:] = 1
        obs, rewards, dones, infos = self.env.step(np.zeros((2, 4), dtype=np.float32))
        self.assertTrue(np.all(dones))
        self.assertEqual(obs.shape, (2, 18))
        self.assertTrue(all("terminal_observation" in info for info in infos))


if __name__ == "__main__":
    unittest.main()
