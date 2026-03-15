from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import StageController
from dronesim.training.callbacks import CurriculumCallback, RewardBreakdownCallback


def make_env(config, stage_controller, seed):
    def _init():
        import copy
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        env = DroneRaceEnv(cfg, stage_controller=stage_controller)
        return env
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO drone racing agent")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .zip")
    parser.add_argument("--run-name", type=str, default="ppo_drone")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv, 1 env")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    stage_controller = StageController(config.task)

    num_envs = 1 if args.debug else config.ppo.num_envs
    env_fns = [make_env(config, stage_controller, config.seed + i) for i in range(num_envs)]

    if args.debug:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    tensorboard_log = f"runs/{args.run_name}"

    if args.resume:
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=tensorboard_log)
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
            tensorboard_log=tensorboard_log,
            verbose=1,
        )

    checkpoint_dir = f"checkpoints/{args.run_name}"
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, 50_000 // num_envs),
            save_path=checkpoint_dir,
            name_prefix="ppo_drone",
        ),
        EvalCallback(
            VecNormalize(
                DummyVecEnv([make_env(config, stage_controller, config.seed + 1000)]),
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            ),
            best_model_save_path=checkpoint_dir,
            log_path=checkpoint_dir,
            eval_freq=max(1, 25_000 // num_envs),
            n_eval_episodes=10,
            deterministic=True,
        ),
        RewardBreakdownCallback(),
        CurriculumCallback(stage_controller),
    ]

    total_timesteps = 50_000 if args.debug else config.ppo.total_timesteps
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(f"{checkpoint_dir}/final_model")
    vec_env.save(f"{checkpoint_dir}/vec_normalize.pkl")
    print(f"Training complete. Model saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
