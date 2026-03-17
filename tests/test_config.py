from __future__ import annotations

import unittest
from pathlib import Path

from dronesim.config import load_config, RuntimeConfig


class TestConfig(unittest.TestCase):
    def test_load_default(self):
        cfg = load_config(Path("configs/default.toml"))
        self.assertIsInstance(cfg, RuntimeConfig)
        self.assertEqual(cfg.sim.backend, "warp")
        self.assertEqual(cfg.sim.sim_hz, 250)
        self.assertEqual(cfg.sim.policy_hz, 50)
        self.assertEqual(cfg.drone.mass_kg, 1.0)
        self.assertEqual(cfg.ppo.batch_size, 64)
        self.assertEqual(cfg.ppo.num_envs, 16)
        self.assertEqual(cfg.reward.gate_passage_bonus, 10.0)
        self.assertEqual(cfg.task.gate_radius_m, 0.45)
        self.assertEqual(cfg.eval.n_episodes, 3)
        self.assertEqual(cfg.eval.eval_freq_timesteps, 100_000)
        self.assertEqual(cfg.eval.checkpoint_freq_timesteps, 100_000)

    def test_load_gpu_stable(self):
        cfg = load_config(Path("configs/gpu_stable.toml"))
        self.assertIsInstance(cfg, RuntimeConfig)
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.sim.backend_device, "cuda")
        self.assertEqual(cfg.sim.sim_hz, 250)
        self.assertEqual(cfg.ppo.n_steps, 256)
        self.assertEqual(cfg.ppo.batch_size, 1024)
        self.assertEqual(cfg.ppo.n_epochs, 4)
        self.assertEqual(cfg.ppo.num_envs, 128)
        self.assertEqual(cfg.reward.collision_penalty, -10.0)
        self.assertEqual(cfg.eval.n_episodes, 5)
        self.assertEqual(cfg.eval.eval_freq_timesteps, 50_000)

    def test_load_gpu_centering(self):
        cfg = load_config(Path("configs/gpu_centering.toml"))
        self.assertEqual(cfg.reward.gate_proximity, 0.35)
        self.assertEqual(cfg.reward.progress, 0.75)
        self.assertEqual(cfg.reward.velocity_alignment, 0.35)

    def test_load_gpu_low_tilt(self):
        cfg = load_config(Path("configs/gpu_low_tilt.toml"))
        self.assertEqual(cfg.drone.max_tilt_rad, 0.35)
        self.assertEqual(cfg.drone.max_body_rate_rad_s, 4.5)

    def test_load_gpu_stability_reward(self):
        cfg = load_config(Path("configs/gpu_stability_reward.toml"))
        self.assertEqual(cfg.reward.attitude_stability, -0.25)
        self.assertEqual(cfg.reward.angular_rate_stability, -0.20)

    def test_load_gpu_intro_focus(self):
        cfg = load_config(Path("configs/gpu_intro_focus.toml"))
        self.assertEqual(cfg.sim.action_smoothing, 0.35)
        self.assertEqual(cfg.task.gate_radius_m, 0.55)

    def test_load_gpu_intro_residual(self):
        cfg = load_config(Path("configs/gpu_intro_residual.toml"))
        self.assertEqual(cfg.sim.guided_action_weight, 1.0)
        self.assertEqual(cfg.sim.residual_action_scale, 0.35)
        self.assertEqual(cfg.reward.lateral_velocity_penalty, -0.30)
        self.assertEqual(cfg.task.gate_radius_m, 0.60)

    def test_load_gpu_intro_axisfix(self):
        cfg = load_config(Path("configs/gpu_intro_axisfix.toml"))
        self.assertEqual(cfg.sim.guided_action_weight, 0.0)
        self.assertEqual(cfg.drone.body_rate_kp, 0.80)
        self.assertEqual(cfg.reward.lateral_velocity_penalty, -0.25)

    def test_load_gpu_intro_finetune(self):
        cfg = load_config(Path("configs/gpu_intro_finetune.toml"))
        self.assertEqual(cfg.ppo.learning_rate, 1e-4)
        self.assertEqual(cfg.ppo.n_epochs, 6)
        self.assertEqual(cfg.reward.attitude_stability, -0.30)

    def test_load_gpu_intro_consolidate(self):
        cfg = load_config(Path("configs/gpu_intro_consolidate.toml"))
        self.assertEqual(cfg.ppo.learning_rate, 5e-5)
        self.assertEqual(cfg.ppo.n_epochs, 8)
        self.assertEqual(cfg.ppo.ent_coef, 0.0)

    def test_load_gpu_intro_terminal(self):
        cfg = load_config(Path("configs/gpu_intro_terminal.toml"))
        self.assertEqual(cfg.sim.action_smoothing, 0.45)
        self.assertEqual(cfg.reward.gate_proximity, 0.30)
        self.assertEqual(cfg.reward.lateral_velocity_penalty, -0.10)

    def test_decimation(self):
        cfg = load_config(Path("configs/default.toml"))
        self.assertEqual(cfg.sim.sim_hz // cfg.sim.policy_hz, 5)


if __name__ == "__main__":
    unittest.main()
