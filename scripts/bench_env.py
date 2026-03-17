from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dronesim.config import load_config
from dronesim.envs.warp_vec_env import WarpVecDroneRaceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vectorized env step throughput")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--backend-device", type=str, default=None, help="Override sim.backend_device")
    parser.add_argument("--num-envs", type=int, default=None, help="Override ppo.num_envs")
    parser.add_argument("--sim-hz", type=int, default=None, help="Override sim.sim_hz")
    parser.add_argument("--policy-hz", type=int, default=None, help="Override sim.policy_hz")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.backend_device is not None:
        config.sim.backend_device = args.backend_device
    if args.num_envs is not None:
        config.ppo.num_envs = args.num_envs
    if args.sim_hz is not None:
        config.sim.sim_hz = args.sim_hz
    if args.policy_hz is not None:
        config.sim.policy_hz = args.policy_hz

    env = WarpVecDroneRaceEnv(config, num_envs=config.ppo.num_envs)
    try:
        env.reset()
        if args.action_mode == "zero":
            actions = np.zeros((config.ppo.num_envs, 4), dtype=np.float32)
        else:
            rng = np.random.default_rng(config.seed)
            actions = rng.uniform(-1.0, 1.0, size=(config.ppo.num_envs, 4)).astype(np.float32)

        for _ in range(args.warmup_steps):
            env.step(actions)

        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        elapsed = time.perf_counter() - start

        total_env_steps = args.steps * config.ppo.num_envs
        policy_fps = total_env_steps / max(elapsed, 1e-9)
        physics_fps = policy_fps * max(1, config.sim.sim_hz // config.sim.policy_hz)

        print(f"backend_device={config.sim.backend_device or config.device}")
        print(f"num_envs={config.ppo.num_envs}")
        print(f"sim_hz={config.sim.sim_hz}")
        print(f"policy_hz={config.sim.policy_hz}")
        print(f"decimation={max(1, config.sim.sim_hz // config.sim.policy_hz)}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"policy_fps={policy_fps:.2f}")
        print(f"physics_fps={physics_fps:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
