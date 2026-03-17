from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dronesim.training.torch_ppo import ActorCritic, RunningMeanStd, load_torch_policy, resolve_config_path
from scripts.train_torch_ppo import next_trigger_step


class TestTorchPPOUtils(unittest.TestCase):
    def test_next_trigger_step_advances_past_current_step(self):
        self.assertEqual(next_trigger_step(0, 50_000), 50_000)
        self.assertEqual(next_trigger_step(32_768, 50_000), 50_000)
        self.assertEqual(next_trigger_step(50_000, 50_000), 100_000)

    def test_resolve_config_path_uses_checkpoint_metadata(self):
        model_path = Path("checkpoints/test/model.pt")
        checkpoint = {"config_path": "configs/default.toml"}
        resolved = resolve_config_path(None, checkpoint, model_path)
        self.assertEqual(resolved, Path("configs/default.toml"))

    def test_load_torch_policy_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "policy.pt"
            agent = ActorCritic(obs_dim=18, act_dim=4, hidden_sizes=[256, 256])
            obs_rms = RunningMeanStd((18,), device=torch.device("cpu"))
            obs_rms.update(torch.ones((8, 18), dtype=torch.float32))
            torch.save(
                {
                    "model": agent.state_dict(),
                    "optimizer": {},
                    "obs_rms": obs_rms.state_dict(),
                    "global_step": 1234,
                    "update_idx": 7,
                    "config_path": "configs/default.toml",
                },
                checkpoint_path,
            )

            policy = load_torch_policy(checkpoint_path, device="cpu")
            action = policy.predict(np.zeros(18, dtype=np.float32), deterministic=True)

            self.assertEqual(action.shape, (1, 4))
            self.assertEqual(policy.global_step, 1234)
            self.assertEqual(policy.update_idx, 7)


if __name__ == "__main__":
    unittest.main()
