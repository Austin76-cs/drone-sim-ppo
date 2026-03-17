"""MuJoCo viewer playback of SB3 or torch-native PPO agents with gate markers."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import RuntimeConfig, load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage
from dronesim.training.torch_ppo import load_torch_policy

try:
    import mujoco
    import mujoco.viewer

    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False


def add_gate_markers(viewer, gates, current_gate_idx):
    """Draw gate rings as visual markers in the viewer."""
    if not hasattr(viewer, "user_scn") or viewer.user_scn is None:
        return

    scn = viewer.user_scn
    scn.ngeom = 0

    for i, gate in enumerate(gates):
        is_current = i == current_gate_idx
        is_cleared = i < current_gate_idx
        n_segments = 16
        radius = gate.radius_m
        tube_radius = 0.03

        if is_cleared:
            color = np.array([0.2, 0.8, 0.2, 0.5], dtype=np.float32)
        elif is_current:
            color = np.array([1.0, 0.3, 0.0, 0.9], dtype=np.float32)
        else:
            color = np.array([0.3, 0.3, 1.0, 0.6], dtype=np.float32)

        for seg in range(n_segments):
            if scn.ngeom >= scn.maxgeom:
                break

            angle = 2 * np.pi * seg / n_segments
            angle_next = 2 * np.pi * (seg + 1) / n_segments
            y1 = radius * np.cos(angle)
            z1 = radius * np.sin(angle)
            y2 = radius * np.cos(angle_next)
            z2 = radius * np.sin(angle_next)

            pos_from = gate.center + np.array([0.0, y1, z1])
            pos_to = gate.center + np.array([0.0, y2, z2])

            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=tube_radius,
                from_=pos_from,
                to=pos_to,
            )
            scn.geoms[scn.ngeom].rgba = color
            scn.ngeom += 1

        if scn.ngeom < scn.maxgeom:
            label_pos = gate.center + np.array([0.0, 0.0, radius + 0.15])
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.05, 0, 0], dtype=np.float64),
                pos=label_pos,
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=color,
            )
            scn.ngeom += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trained agent with gate markers")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip or .pt")
    parser.add_argument("--normalize", type=str, default=None, help="Path to SB3 vec_normalize.pkl")
    parser.add_argument("--config", type=str, default=None, help="Override runtime config path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Force curriculum stage: 0=INTRO, 1=OFFSET, 2=SLALOM",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch checkpoint inference device")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic actions",
    )
    return parser.parse_args()


def build_env(config: RuntimeConfig, stage: int | None) -> DroneRaceEnv:
    env = DroneRaceEnv(config)
    if stage is not None:
        env.stage_controller.force_stage(CurriculumStage(stage))
    return env


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


def load_sb3_model(model_path: Path, config: RuntimeConfig, stage: int | None, normalize: str | None):
    vec_env = DummyVecEnv([lambda: build_env(config, stage)])
    normalize_path = resolve_normalize_path(model_path, normalize)
    if normalize_path is not None:
        vec_env = VecNormalize.load(str(normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats from {normalize_path}")
    elif normalize:
        print(f"VecNormalize file not found at {normalize}; continuing without normalization.")
    model = PPO.load(str(model_path), env=vec_env, device="cpu")
    return model, vec_env


def main() -> None:
    args = parse_args()
    if not HAS_VIEWER:
        print("mujoco.viewer not available.")
        return

    model_path = Path(args.model)
    sb3_model = None
    sb3_env = None
    torch_policy = None
    if model_path.suffix == ".pt":
        torch_policy = load_torch_policy(model_path, config_path=args.config, device=args.device)
        vis_env = build_env(torch_policy.config, args.stage)
        print(
            f"Loaded torch checkpoint: steps={torch_policy.global_step} updates={torch_policy.update_idx} device={torch_policy.device}"
        )
    else:
        config_path = Path(args.config) if args.config else Path("configs/default.toml")
        config = load_config(config_path)
        vis_env = build_env(config, args.stage)
        sb3_model, sb3_env = load_sb3_model(model_path, config, args.stage, args.normalize)
    sim = vis_env.sim

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.drone_body_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0, 0, 1]

        for ep in range(args.episodes):
            seed = vis_env.config.seed + ep * 100
            obs, info = vis_env.reset(seed=seed)
            done = False
            step = 0
            gates = vis_env.current_task.gates if vis_env.current_task else ()
            sleep_time = (1.0 / vis_env.config.sim.policy_hz) / args.speed

            if sb3_env is not None:
                sb3_env.reset()

            print(f"\nEpisode {ep + 1} starting - {len(gates)} gates, stage {info.get('curriculum_stage', '?')}")

            while not done and viewer.is_running():
                if torch_policy is not None:
                    action = torch_policy.predict(obs, deterministic=args.deterministic)[0]
                else:
                    assert sb3_model is not None and sb3_env is not None
                    if hasattr(sb3_env, "normalize_obs"):
                        policy_obs = sb3_env.normalize_obs(obs.reshape(1, -1))
                    else:
                        policy_obs = obs.reshape(1, -1)
                    action, _ = sb3_model.predict(policy_obs, deterministic=args.deterministic)
                    action = action[0]

                obs, _, terminated, truncated, info = vis_env.step(action)
                done = terminated or truncated
                step += 1

                add_gate_markers(viewer, gates, vis_env.gate_index)
                viewer.sync()
                time.sleep(sleep_time)

            crash = info.get("crash_type", "?")
            cleared = info.get("gates_cleared", 0)
            completion = info.get("completion", 0.0)
            print(
                f"Episode {ep + 1} done: steps={step}, gates={cleared}/{len(gates)}, "
                f"completion={completion:.0%}, result={crash}"
            )

            if not viewer.is_running():
                break
            time.sleep(1.0)

    vis_env.close()
    if sb3_env is not None:
        sb3_env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
