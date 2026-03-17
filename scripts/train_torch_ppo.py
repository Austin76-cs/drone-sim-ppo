from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.envs.warp_vec_env import WarpVecDroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage, StageController
from dronesim.training.torch_ppo import ActorCritic, RunningMeanStd


@dataclass(slots=True)
class PPOStats:
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    entropy: float = 0.0
    pg_loss: float = 0.0
    value_loss: float = 0.0
    explained_variance: float = 0.0


@dataclass(slots=True)
class EvalStats:
    stage: int
    mean_reward: float
    std_reward: float
    success_rate: float
    mean_completion: float
    mean_gates_cleared: float
    mean_episode_steps: float


def next_trigger_step(current_step: int, frequency: int) -> int:
    return max(frequency, ((current_step // frequency) + 1) * frequency)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if torch.isclose(var_y, torch.tensor(0.0, device=y_true.device)):
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred) / var_y).item())


def save_checkpoint(
    path: Path,
    *,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_rms: RunningMeanStd,
    global_step: int,
    update_idx: int,
    config_path: str,
    best_eval_key: tuple[float, ...] | None = None,
) -> None:
    payload = {
        "model": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "global_step": global_step,
        "update_idx": update_idx,
        "config_path": config_path,
    }
    if best_eval_key is not None:
        payload["best_eval_key"] = torch.tensor(best_eval_key, dtype=torch.float32)
    torch.save(payload, path)


@torch.no_grad()
def evaluate_agent(
    *,
    agent: ActorCritic,
    obs_rms: RunningMeanStd,
    config,
    stage: CurriculumStage,
    episodes: int,
    device: torch.device,
) -> EvalStats:
    eval_config = copy.deepcopy(config)
    eval_config.sim.backend_device = "cpu"
    eval_config.device = "cpu"
    eval_stage_controller = StageController(eval_config.task)
    eval_stage_controller.force_stage(stage)
    env = DroneRaceEnv(eval_config, stage_controller=eval_stage_controller)

    rewards: list[float] = []
    completions: list[float] = []
    gates: list[int] = []
    steps: list[int] = []
    successes = 0

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=eval_config.seed + ep * 1000)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            info: dict[str, object] = {}

            while not done:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                norm_obs = obs_rms.normalize(obs_t)
                action = torch.tanh(agent.actor_mean(agent._features(norm_obs))).squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
                ep_reward += float(reward)
                ep_steps += 1
                done = bool(terminated or truncated)

            rewards.append(ep_reward)
            completions.append(float(info.get("completion", 0.0)))
            gates.append(int(info.get("gates_cleared", 0)))
            steps.append(ep_steps)
            if str(info.get("crash_type", "")) == "success":
                successes += 1
    finally:
        env.close()

    return EvalStats(
        stage=int(stage),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        success_rate=float(successes / max(episodes, 1)),
        mean_completion=float(np.mean(completions)),
        mean_gates_cleared=float(np.mean(gates)),
        mean_episode_steps=float(np.mean(steps)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-native PPO trainer for WarpVecDroneRaceEnv")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--run-name", type=str, default="torch_ppo")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sim-device", type=str, default="cuda")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sim-hz", type=int, default=None)
    parser.add_argument("--policy-hz", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--force-stage", type=int, default=None, choices=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for policy training but torch.cuda.is_available() is false")

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    config = load_config(Path(args.config))
    config.device = args.device
    config.sim.backend_device = args.sim_device
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if args.sim_hz is not None:
        config.sim.sim_hz = args.sim_hz
    if args.policy_hz is not None:
        config.sim.policy_hz = args.policy_hz
    if args.n_steps is not None:
        config.ppo.n_steps = args.n_steps
    if args.batch_size is not None:
        config.ppo.batch_size = args.batch_size
    if args.n_epochs is not None:
        config.ppo.n_epochs = args.n_epochs
    total_timesteps = args.total_timesteps or config.ppo.total_timesteps
    checkpoint_freq = args.checkpoint_freq or config.eval.checkpoint_freq_timesteps
    eval_freq = args.eval_freq or config.eval.eval_freq_timesteps

    stage_controller = StageController(config.task)
    if args.force_stage is not None:
        stage_controller.lock_to_stage(CurriculumStage(args.force_stage))
    env = WarpVecDroneRaceEnv(config, num_envs=config.ppo.num_envs, stage_controller=stage_controller)
    if not getattr(env, "_use_torch_fast_path", False):
        raise RuntimeError("Torch PPO trainer requires WarpVecDroneRaceEnv CUDA fast path")

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    agent = ActorCritic(obs_dim, act_dim, config.ppo.net_arch).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
    obs_rms = RunningMeanStd((obs_dim,), device=device)

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")

    global_step = 0
    start_step = 0
    start_time = time.perf_counter()
    update_idx = 0
    best_eval_key: tuple[float, ...] | None = None
    eval_history: dict[str, list[float]] = {
        "timesteps": [],
        "stage": [],
        "mean_reward": [],
        "std_reward": [],
        "success_rate": [],
        "mean_completion": [],
        "mean_gates_cleared": [],
        "mean_episode_steps": [],
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        obs_rms.load_state_dict(checkpoint["obs_rms"])
        global_step = int(checkpoint.get("global_step", 0))
        start_step = global_step
        update_idx = int(checkpoint.get("update_idx", 0))
        if "best_eval_key" in checkpoint:
            best_eval_key = tuple(float(v) for v in checkpoint["best_eval_key"].tolist())
        eval_history_path = run_dir / "evaluations.npz"
        if eval_history_path.exists():
            loaded_history = np.load(eval_history_path)
            for key in eval_history:
                if key in loaded_history:
                    eval_history[key] = loaded_history[key].astype(float).tolist()

    obs = env.reset_torch().to(device=device, dtype=torch.float32)
    obs_rms.update(obs)
    next_eval_step = next_trigger_step(global_step, eval_freq)
    next_checkpoint_step = next_trigger_step(global_step, checkpoint_freq)

    n_steps = config.ppo.n_steps
    num_envs = config.ppo.num_envs
    batch_size = n_steps * num_envs
    minibatch_size = config.ppo.batch_size
    if batch_size % minibatch_size != 0:
        raise ValueError(f"rollout batch {batch_size} must be divisible by minibatch size {minibatch_size}")

    obs_buf = torch.zeros((n_steps, num_envs, obs_dim), device=device, dtype=torch.float32)
    raw_obs_buf = torch.zeros((n_steps, num_envs, obs_dim), device=device, dtype=torch.float32)
    actions_buf = torch.zeros((n_steps, num_envs, act_dim), device=device, dtype=torch.float32)
    logprob_buf = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
    rewards_buf = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
    dones_buf = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)
    values_buf = torch.zeros((n_steps, num_envs), device=device, dtype=torch.float32)

    while global_step < total_timesteps:
        for step in range(n_steps):
            raw_obs_buf[step] = obs
            norm_obs = obs_rms.normalize(obs)
            obs_buf[step] = norm_obs

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(norm_obs)

            next_obs, reward, done = env.step_torch_train(action)
            next_obs = next_obs.to(device=device, dtype=torch.float32)
            reward = reward.to(device=device, dtype=torch.float32)
            done = done.to(device=device)

            actions_buf[step] = action
            logprob_buf[step] = log_prob
            rewards_buf[step] = reward
            dones_buf[step] = done.float()
            values_buf[step] = value

            obs = next_obs
            global_step += num_envs

        obs_rms.update(raw_obs_buf.reshape(-1, obs_dim))

        with torch.no_grad():
            next_value = agent.get_value(obs_rms.normalize(obs))

        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = torch.zeros(num_envs, device=device, dtype=torch.float32)
        next_done = dones_buf[-1]
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones_buf[step + 1]
                next_values = values_buf[step + 1]
            delta = rewards_buf[step] + config.ppo.gamma * next_values * next_non_terminal - values_buf[step]
            lastgaelam = (
                delta
                + config.ppo.gamma * config.ppo.gae_lambda * next_non_terminal * lastgaelam
            )
            advantages[step] = lastgaelam
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, act_dim))
        b_logprob = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        stats = PPOStats()
        clipfracs: list[float] = []
        for _ in range(config.ppo.n_epochs):
            batch_indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = batch_indices[start : start + minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_idx], b_actions[mb_idx]
                )
                logratio = new_logprob - b_logprob[mb_idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(float(((ratio - 1.0).abs() > config.ppo.clip_range).float().mean().item()))

                pg_loss1 = -b_advantages[mb_idx] * ratio
                pg_loss2 = -b_advantages[mb_idx] * torch.clamp(
                    ratio,
                    1.0 - config.ppo.clip_range,
                    1.0 + config.ppo.clip_range,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (new_value - b_returns[mb_idx]).square().mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + 0.5 * value_loss - config.ppo.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                stats.approx_kl = float(approx_kl.item())
                stats.entropy = float(entropy_loss.item())
                stats.pg_loss = float(pg_loss.item())
                stats.value_loss = float(value_loss.item())

        stats.clip_fraction = float(np.mean(clipfracs)) if clipfracs else 0.0
        stats.explained_variance = explained_variance(b_values, b_returns)
        update_idx += 1

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        fps = int((global_step - start_step) / elapsed)
        print("-" * 40)
        print(f"update={update_idx} timesteps={global_step} fps={fps}")
        print(
            f"stage={stage_controller.stage} mastery={stage_controller.mastery:.3f}"
        )
        print(
            f"kl={stats.approx_kl:.4f} clip_frac={stats.clip_fraction:.3f} "
            f"entropy={stats.entropy:.3f} pg_loss={stats.pg_loss:.4f} value_loss={stats.value_loss:.4f}"
        )

        writer.add_scalar("time/fps", fps, global_step)
        writer.add_scalar("time/iterations", update_idx, global_step)
        writer.add_scalar("curriculum/stage", int(stage_controller.stage), global_step)
        writer.add_scalar("curriculum/mastery", stage_controller.mastery, global_step)
        writer.add_scalar("train/approx_kl", stats.approx_kl, global_step)
        writer.add_scalar("train/clip_fraction", stats.clip_fraction, global_step)
        writer.add_scalar("train/entropy_loss", -stats.entropy, global_step)
        writer.add_scalar("train/policy_gradient_loss", stats.pg_loss, global_step)
        writer.add_scalar("train/value_loss", stats.value_loss, global_step)
        writer.add_scalar("train/explained_variance", stats.explained_variance, global_step)
        writer.add_scalar("train/std", float(agent.log_std.exp().mean().item()), global_step)

        if global_step >= next_eval_step:
            eval_stats = evaluate_agent(
                agent=agent,
                obs_rms=obs_rms,
                config=config,
                stage=stage_controller.stage,
                episodes=config.eval.n_episodes,
                device=device,
            )
            eval_key = (
                float(eval_stats.stage),
                eval_stats.success_rate,
                eval_stats.mean_completion,
                eval_stats.mean_gates_cleared,
                eval_stats.mean_reward,
            )
            eval_history["timesteps"].append(float(global_step))
            eval_history["stage"].append(float(eval_stats.stage))
            eval_history["mean_reward"].append(eval_stats.mean_reward)
            eval_history["std_reward"].append(eval_stats.std_reward)
            eval_history["success_rate"].append(eval_stats.success_rate)
            eval_history["mean_completion"].append(eval_stats.mean_completion)
            eval_history["mean_gates_cleared"].append(eval_stats.mean_gates_cleared)
            eval_history["mean_episode_steps"].append(eval_stats.mean_episode_steps)
            np.savez(run_dir / "evaluations.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in eval_history.items()})

            writer.add_scalar("eval/stage", eval_stats.stage, global_step)
            writer.add_scalar("eval/mean_reward", eval_stats.mean_reward, global_step)
            writer.add_scalar("eval/std_reward", eval_stats.std_reward, global_step)
            writer.add_scalar("eval/success_rate", eval_stats.success_rate, global_step)
            writer.add_scalar("eval/mean_completion", eval_stats.mean_completion, global_step)
            writer.add_scalar("eval/mean_gates_cleared", eval_stats.mean_gates_cleared, global_step)
            writer.add_scalar("eval/mean_episode_steps", eval_stats.mean_episode_steps, global_step)

            print(
                f"eval stage={eval_stats.stage} reward={eval_stats.mean_reward:.2f} "
                f"success={eval_stats.success_rate:.1%} completion={eval_stats.mean_completion:.2f} "
                f"gates={eval_stats.mean_gates_cleared:.2f}"
            )

            if best_eval_key is None or eval_key > best_eval_key:
                best_eval_key = eval_key
                best_path = run_dir / "best_model.pt"
                save_checkpoint(
                    best_path,
                    agent=agent,
                    optimizer=optimizer,
                    obs_rms=obs_rms,
                    global_step=global_step,
                    update_idx=update_idx,
                    config_path=args.config,
                    best_eval_key=best_eval_key,
                )
                print(f"saved best model to {best_path}")
            while next_eval_step <= global_step:
                next_eval_step += eval_freq

        if global_step >= next_checkpoint_step:
            checkpoint_path = run_dir / f"ppo_{global_step}_steps.pt"
            save_checkpoint(
                checkpoint_path,
                agent=agent,
                optimizer=optimizer,
                obs_rms=obs_rms,
                global_step=global_step,
                update_idx=update_idx,
                config_path=args.config,
                best_eval_key=best_eval_key,
            )
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += checkpoint_freq

    final_path = run_dir / "final_model.pt"
    save_checkpoint(
        final_path,
        agent=agent,
        optimizer=optimizer,
        obs_rms=obs_rms,
        global_step=global_step,
        update_idx=update_idx,
        config_path=args.config,
        best_eval_key=best_eval_key,
    )
    writer.close()
    env.close()
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
