from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from dronesim.config import RuntimeConfig, load_config


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], device: torch.device, epsilon: float = 1e-4) -> None:
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(epsilon, device=device, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=torch.float32)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(x, -clip, clip)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_dim, hidden)))
            layers.append(nn.Tanh())
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = layer_init(nn.Linear(in_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(in_dim, 1), std=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self._features(obs))
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        if action is None:
            pre_tanh = dist.rsample()
            action = torch.tanh(pre_tanh)
        else:
            action = torch.clamp(action, -0.999999, 0.999999)
            pre_tanh = torch.atanh(action)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1e-6)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(self._features(obs)).squeeze(-1)
        return action, log_prob.sum(dim=-1), entropy, value


def resolve_config_path(config_arg: str | None, checkpoint: dict[str, Any], model_path: Path) -> Path:
    if config_arg:
        return Path(config_arg)
    checkpoint_config = checkpoint.get("config_path")
    if isinstance(checkpoint_config, str) and checkpoint_config:
        config_path = Path(checkpoint_config)
        if config_path.exists():
            return config_path
        candidate = (model_path.parent / config_path).resolve()
        if candidate.exists():
            return candidate
    return Path("configs/default.toml")


@dataclass(slots=True)
class LoadedTorchPolicy:
    agent: ActorCritic
    obs_rms: RunningMeanStd
    config: RuntimeConfig
    device: torch.device
    global_step: int
    update_idx: int

    @torch.no_grad()
    def predict(
        self,
        obs: np.ndarray | torch.Tensor,
        deterministic: bool = True,
    ) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        else:
            obs_t = obs.to(device=self.device, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        norm_obs = self.obs_rms.normalize(obs_t)
        if deterministic:
            action = torch.tanh(self.agent.actor_mean(self.agent._features(norm_obs)))
        else:
            action, _, _, _ = self.agent.get_action_and_value(norm_obs)
        return action.detach().cpu().numpy().astype(np.float32)


def load_torch_policy(
    model_path: str | Path,
    *,
    config_path: str | None = None,
    obs_dim: int = 18,
    act_dim: int = 4,
    device: str = "cpu",
) -> LoadedTorchPolicy:
    model_path = Path(model_path)
    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    resolved_config_path = resolve_config_path(config_path, checkpoint, model_path)
    config = load_config(resolved_config_path)
    agent = ActorCritic(obs_dim, act_dim, config.ppo.net_arch).to(torch_device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()

    obs_rms = RunningMeanStd((obs_dim,), device=torch_device)
    obs_rms.load_state_dict(checkpoint["obs_rms"])
    return LoadedTorchPolicy(
        agent=agent,
        obs_rms=obs_rms,
        config=config,
        device=torch_device,
        global_step=int(checkpoint.get("global_step", 0)),
        update_idx=int(checkpoint.get("update_idx", 0)),
    )
