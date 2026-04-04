"""MuJoCo viewer playback of a trained agent with visible gates."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv

try:
    import mujoco
    import mujoco.viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False


def add_gate_markers(viewer, gates, current_gate_idx):
    """Draw gate rings as visual markers in the viewer."""
    if not hasattr(viewer, 'user_scn') or viewer.user_scn is None:
        return

    scn = viewer.user_scn
    scn.ngeom = 0

    for i, gate in enumerate(gates):
        is_current = (i == current_gate_idx)
        is_cleared = (i < current_gate_idx)

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

            # Use mjv_connector for line segments between two points
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=tube_radius,
                from_=pos_from,
                to=pos_to,
            )
            scn.geoms[scn.ngeom].rgba = color
            scn.ngeom += 1

        # Small sphere marker above gate
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trained agent with gate markers")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--normalize", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--stage", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Force curriculum stage: 0=INTRO, 1=OFFSET, 2=SLALOM, 3=SPRINT")
    parser.add_argument("--multi-stage", action="store_true",
                        help="Randomly cycle through all stages each episode")
    args = parser.parse_args()

    if not HAS_VIEWER:
        print("mujoco.viewer not available.")
        return

    config = load_config(Path(args.config))

    # Create the actual env we'll visualize
    from dronesim.tasks.curriculum import CurriculumStage, _MULTI_STAGE_WEIGHTS
    vis_env = DroneRaceEnv(config)
    if args.multi_stage:
        vis_env.stage_controller.multi_stage = True
    elif args.stage is not None:
        vis_env.stage_controller.force_stage(CurriculumStage(args.stage))

    # Create a separate vec_env for the model (handles VecNormalize)
    vec_env = DummyVecEnv([lambda: DroneRaceEnv(config)])
    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    # Use the vis_env's underlying MuJoCo model/data for the viewer
    sim = vis_env.sim

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set up camera to track the drone from behind/above
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.drone_body_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180  # behind the drone
        viewer.cam.elevation = -25  # slightly above
        viewer.cam.lookat[:] = [0, 0, 1]

        run_seed = int(time.time()) % 100000
        for ep in range(args.episodes):
            seed = run_seed + ep * 100
            obs, info = vis_env.reset(seed=seed)

            # Normalize obs the same way VecNormalize would
            vec_obs = vec_env.reset()
            if hasattr(vec_env, 'normalize_obs'):
                norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
            else:
                norm_obs = obs.reshape(1, -1)

            done = False
            step = 0
            gates = vis_env.current_task.gates if vis_env.current_task else ()
            sleep_time = (1.0 / config.sim.policy_hz) / args.speed

            print(f"\nEpisode {ep + 1} starting — {len(gates)} gates, "
                  f"stage {info.get('curriculum_stage', '?')}")

            while not done and viewer.is_running():
                action, _ = model.predict(norm_obs, deterministic=True)

                obs, reward, terminated, truncated, info = vis_env.step(action[0])
                done = terminated or truncated
                step += 1

                # Normalize for next prediction
                if hasattr(vec_env, 'normalize_obs'):
                    norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
                else:
                    norm_obs = obs.reshape(1, -1)

                # Draw gate markers
                add_gate_markers(viewer, gates, vis_env.gate_index)

                viewer.sync()
                time.sleep(sleep_time)

            crash = info.get('crash_type', '?')
            cleared = info.get('gates_cleared', 0)
            comp = info.get('completion', 0)
            print(f"Episode {ep + 1} done: steps={step}, gates={cleared}/{len(gates)}, "
                  f"completion={comp:.0%}, result={crash}")

            if not viewer.is_running():
                break

            # Pause between episodes
            time.sleep(1.0)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()
