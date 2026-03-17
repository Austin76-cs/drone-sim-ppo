"""Evaluate SB3 or torch-native PPO checkpoints and print episode metrics."""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import RuntimeConfig, load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage
from dronesim.training.eval_utils import evaluate_torch_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip or .pt")
    parser.add_argument("--normalize", type=str, default=None, help="Path to SB3 vec_normalize.pkl")
    parser.add_argument("--config", type=str, default=None, help="Override runtime config path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Force curriculum stage: 0=INTRO, 1=OFFSET, 2=SLALOM",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch checkpoint inference device")
    parser.add_argument("--sim-device", type=str, default="cpu", help="Simulation device for torch checkpoint eval")
    return parser.parse_args()


def resolve_normalize_path(model_path: Path, normalize_arg: str | None) -> Path | None:
    if normalize_arg:
        return Path(normalize_arg)
    default_path = model_path.parent / "vec_normalize.pkl"
    if default_path.exists():
        return default_path
    candidates = sorted(model_path.parent.glob("*vecnormalize*.pkl"))
    if candidates:
        return candidates[-1]
    return None


def make_env(config: RuntimeConfig, stage: int | None) -> DroneRaceEnv:
    env = DroneRaceEnv(config)
    if stage is not None:
        env.stage_controller.force_stage(CurriculumStage(stage))
    return env


def evaluate_sb3(args: argparse.Namespace, config_path: Path) -> None:
    model_path = Path(args.model)
    config = copy.deepcopy(load_config(config_path))
    config.sim.backend_device = args.sim_device
    config.device = "cpu"
    vec_env = DummyVecEnv([lambda: make_env(config, args.stage)])
    normalize_path = resolve_normalize_path(model_path, args.normalize)
    if normalize_path is not None:
        vec_env = VecNormalize.load(str(normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {normalize_path}")
    elif args.normalize:
        print(f"VecNormalize file not found at {args.normalize}; continuing without normalization.")

    model = PPO.load(args.model, env=vec_env, device="cpu")

    crash_types: Counter[str] = Counter()
    rewards: list[float] = []
    gates: list[int] = []
    completions: list[float] = []
    steps: list[int] = []
    successes = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_info: dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            ep_info = infos[0]
            done = bool(dones[0])

        crash_type = ep_info.get("crash_type", "unknown")
        gate_count = int(ep_info.get("gates_cleared", 0))
        completion = float(ep_info.get("completion", 0.0))

        crash_types[crash_type] += 1
        rewards.append(ep_reward)
        gates.append(gate_count)
        completions.append(completion)
        steps.append(ep_steps)
        if crash_type == "success":
            successes += 1

        print(
            f"  Episode {ep + 1:3d}: reward={ep_reward:8.2f}  steps={ep_steps:4d}  "
            f"gates={gate_count}  completion={completion:.2f}  crash={crash_type}"
        )

    print_summary(args.episodes, successes, gates, completions, rewards, steps, crash_types)


def evaluate_torch(args: argparse.Namespace) -> None:
    policy, summary = evaluate_torch_checkpoint(
        args.model,
        episodes=args.episodes,
        stage=args.stage,
        deterministic=args.deterministic,
        policy_device=args.device,
        sim_device=args.sim_device,
        config_path=args.config,
    )
    print(
        f"Loaded torch checkpoint: steps={policy.global_step} updates={policy.update_idx} "
        f"device={policy.device} sim_device={args.sim_device}"
    )
    print_summary(
        summary.episodes,
        int(round(summary.success_rate * summary.episodes)),
        [summary.mean_gates_cleared],
        [summary.mean_completion],
        [summary.mean_reward],
        [summary.mean_episode_steps],
        Counter(summary.crash_distribution),
        std_reward=summary.std_reward,
        use_preaggregated=True,
    )


def print_summary(
    episodes: int,
    successes: int,
    gates: list[int],
    completions: list[float],
    rewards: list[float],
    steps: list[int],
    crash_types: Counter[str],
    *,
    std_reward: float | None = None,
    use_preaggregated: bool = False,
) -> None:
    mean_gates = gates[0] if use_preaggregated else float(np.mean(gates))
    mean_completion = completions[0] if use_preaggregated else float(np.mean(completions))
    mean_reward = rewards[0] if use_preaggregated else float(np.mean(rewards))
    reward_std = std_reward if std_reward is not None else float(np.std(rewards))
    mean_steps = steps[0] if use_preaggregated else float(np.mean(steps))
    median_steps = steps[0] if use_preaggregated else float(np.median(steps))
    print("\n" + "=" * 68)
    print(f"{'Metric':<28} {'Value':>12}")
    print("-" * 68)
    print(f"{'Episodes':<28} {episodes:>12}")
    print(f"{'Success Rate':<28} {successes / episodes:>12.1%}")
    print(f"{'Mean Gates Cleared':<28} {mean_gates:>12.2f}")
    print(f"{'Mean Completion':<28} {mean_completion:>12.2f}")
    print(f"{'Mean Reward':<28} {mean_reward:>12.2f}")
    print(f"{'Std Reward':<28} {reward_std:>12.2f}")
    print(f"{'Mean Episode Steps':<28} {mean_steps:>12.1f}")
    print(f"{'Median Episode Steps':<28} {median_steps:>12.1f}")
    print("-" * 68)
    print("Crash Distribution:")
    for crash, count in crash_types.most_common():
        print(f"  {crash:<24} {count:>4}  ({count / episodes:.0%})")
    print("=" * 68)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if model_path.suffix == ".pt":
        evaluate_torch(args)
        return
    config_path = Path(args.config) if args.config else Path("configs/default.toml")
    evaluate_sb3(args, config_path)


if __name__ == "__main__":
    main()
