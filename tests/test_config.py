from __future__ import annotations

import unittest
from pathlib import Path

from dronesim.config import load_config, RuntimeConfig


class TestConfig(unittest.TestCase):
    def test_load_default(self):
        cfg = load_config(Path("configs/default.toml"))
        self.assertIsInstance(cfg, RuntimeConfig)
        self.assertEqual(cfg.sim.sim_hz, 500)
        self.assertEqual(cfg.sim.policy_hz, 50)
        self.assertEqual(cfg.drone.mass_kg, 1.0)
        self.assertEqual(cfg.ppo.batch_size, 64)
        self.assertEqual(cfg.reward.gate_passage_bonus, 10.0)
        self.assertEqual(cfg.task.gate_radius_m, 0.40)

    def test_decimation(self):
        cfg = load_config(Path("configs/default.toml"))
        self.assertEqual(cfg.sim.sim_hz // cfg.sim.policy_hz, 10)


if __name__ == "__main__":
    unittest.main()
