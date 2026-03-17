from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from dronesim.config import load_config
from dronesim.envs.warp_vec_env import WarpVecDroneRaceEnv
from dronesim.tasks.curriculum import StageController
from dronesim.training.callbacks import CurriculumCallback, RewardBreakdownCallback


def tensorboard_log_dir(run_name: str) -> str | None:
    if importlib.util.find_spec("tensorboard") is None:
        print("TensorBoard not installed; continuing without TensorBoard logging.")
        return None
    return f"runs/{run_name}"


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Configured device '{device}' but CUDA is not available in this Python environment."
        )
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO drone racing agent")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip")
    parser.add_argument("--run-name", type=str, default="ppo_drone")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv, 1 env")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sim-device", type=str, default=None)
    parser.add_argument("--sim-hz", type=int, default=None)
    parser.add_argument("--policy-hz", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.sim_device is not None:
        config.sim.backend_device = args.sim_device
    if args.sim_hz is not None:
        config.sim.sim_hz = args.sim_hz
    if args.policy_hz is not None:
        config.sim.policy_hz = args.policy_hz
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if args.n_steps is not None:
        config.ppo.n_steps = args.n_steps
    if args.batch_size is not None:
        config.ppo.batch_size = args.batch_size
    if args.n_epochs is not None:
        config.ppo.n_epochs = args.n_epochs
    stage_controller = StageController(config.task)
    device = resolve_device(config.device)
    sim_device = config.sim.backend_device or config.device
    if device == "cpu" and sim_device.startswith("cuda") and args.sim_device is None:
        sim_device = "cpu"
        config.sim.backend_device = sim_device
        print(
            "Switching sim backend to CPU because CPU PPO plus CUDA physics "
            "forces a host-device sync every control step."
        )

    num_envs = 1 if args.debug else config.ppo.num_envs
    print(f"Using Warp sim on {sim_device} with PPO on {device} across {num_envs} worlds.")
    if device.startswith("cuda"):
        print("PPO with MlpPolicy is usually faster on CPU; keep sim.backend_device on CUDA if you want GPU physics.")
    vec_env = WarpVecDroneRaceEnv(config, num_envs=num_envs, stage_controller=stage_controller)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    tensorboard_log = tensorboard_log_dir(args.run_name)

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=tensorboard_log, device=device)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            policy_kwargs={"net_arch": config.ppo.net_arch},
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )

    checkpoint_dir = f"checkpoints/{args.run_name}"
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, config.eval.checkpoint_freq_timesteps // num_envs),
            save_path=checkpoint_dir,
            name_prefix="ppo_drone",
            save_vecnormalize=True,
        ),
        EvalCallback(
            VecNormalize(
                WarpVecDroneRaceEnv(config, num_envs=1, stage_controller=stage_controller),
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            ),
            best_model_save_path=checkpoint_dir,
            log_path=checkpoint_dir,
            eval_freq=max(1, config.eval.eval_freq_timesteps // num_envs),
            n_eval_episodes=config.eval.n_episodes,
            deterministic=True,
        ),
        RewardBreakdownCallback(),
        CurriculumCallback(stage_controller),
    ]

    total_timesteps = args.total_timesteps
    if total_timesteps is None:
        total_timesteps = 50_000 if args.debug else config.ppo.total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(f"{checkpoint_dir}/final_model")
    vec_env.save(f"{checkpoint_dir}/vec_normalize.pkl")
    print(f"Training complete. Model saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
