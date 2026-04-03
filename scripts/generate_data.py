"""Generate perception training data from the drone sim.

Uses the trained PPO policy (v41) to fly realistic trajectories, renders the
drone-mounted camera at each step, and saves (image, heatmap) pairs to HDF5.

Usage:
    python scripts/generate_data.py \\
        --model checkpoints/ppo_drone_v41/best_model.zip \\
        --normalize checkpoints/ppo_drone_v41/vec_normalize.pkl \\
        --output data/perception_train.h5 \\
        --episodes 300 --stage 2
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import h5py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import CurriculumStage

CAM_W = 160
CAM_H = 120
HEATMAP_SIGMA = 8  # Gaussian sigma in pixels


def make_heatmap(
    centers_uv: list[tuple[float, float]], width: int, height: int, sigma: float
) -> np.ndarray:
    """Gaussian heatmap (H, W) float32 with blobs at each visible gate center."""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.zeros((height, width), dtype=np.float32)
    for u, v in centers_uv:
        heatmap += np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
    return np.clip(heatmap, 0.0, 1.0)


def project_gate(
    gate_center: np.ndarray,
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Project a 3-D gate center to image (u, v). Returns None if behind camera or off-screen."""
    # cam_xmat columns = camera x/y/z axes in world. Camera looks along -z_cam.
    p = cam_xmat.T @ (gate_center - cam_pos)
    depth = -p[2]  # positive = in front of camera
    if depth < 0.05:
        return None
    u = fx * p[0] / depth + cx
    v = cy - fy * p[1] / depth  # flip y: camera y=up, image y=down
    # Keep a small margin outside frame (partially visible gates still matter)
    margin = 0.3
    if not (-width * margin < u < width * (1 + margin)):
        return None
    if not (-height * margin < v < height * (1 + margin)):
        return None
    return float(u), float(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drone perception training data")
    parser.add_argument("--model", required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--normalize", default=None, help="Path to vec_normalize.pkl")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--output", default="data/perception_train.h5")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--stage", type=int, default=2, choices=[0, 1, 2, 3],
        help="Curriculum stage for data collection (default: 2=SLALOM)",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Env for rendering (single, not vectorized)
    vis_env = DroneRaceEnv(config)
    vis_env.stage_controller.force_stage(CurriculumStage(args.stage))

    # Separate vec env for the policy (needs VecNormalize)
    def _make():
        e = DroneRaceEnv(config)
        e.stage_controller.force_stage(CurriculumStage(args.stage))
        return e

    vec_env = DummyVecEnv([_make])
    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    print(f"Collecting {args.episodes} episodes -> {args.output}")

    with h5py.File(args.output, "w") as f:
        img_ds = f.create_dataset(
            "images",
            shape=(0, CAM_H, CAM_W, 3),
            maxshape=(None, CAM_H, CAM_W, 3),
            dtype="uint8",
            chunks=(64, CAM_H, CAM_W, 3),
        )
        hm_ds = f.create_dataset(
            "heatmaps",
            shape=(0, CAM_H, CAM_W),
            maxshape=(None, CAM_H, CAM_W),
            dtype="float32",
            chunks=(64, CAM_H, CAM_W),
        )

        for ep in range(args.episodes):
            seed = config.seed + ep * 137
            obs, _ = vis_env.reset(seed=seed)
            vec_env.reset()

            gates = vis_env.current_task.gates if vis_env.current_task else []
            vis_env.sim.set_gate_visuals(gates)

            # Normalize obs the same way VecNormalize would
            if hasattr(vec_env, "normalize_obs"):
                norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
            else:
                norm_obs = obs.reshape(1, -1)

            fx, fy, cx_i, cy_i = vis_env.sim.get_camera_intrinsics(CAM_W, CAM_H)
            ep_images: list[np.ndarray] = []
            ep_heatmaps: list[np.ndarray] = []
            done = False

            while not done:
                # Render camera image
                img = vis_env.sim.render_camera(CAM_W, CAM_H)
                cam_pos, cam_xmat = vis_env.sim.get_camera_extrinsics()

                # Project current + next gate centers to image
                cur_idx = vis_env.gate_index
                visible_uvs: list[tuple[float, float]] = []
                for gi in range(cur_idx, min(cur_idx + 2, len(gates))):
                    uv = project_gate(
                        gates[gi].center, cam_pos, cam_xmat,
                        fx, fy, cx_i, cy_i, CAM_W, CAM_H,
                    )
                    if uv is not None:
                        visible_uvs.append(uv)

                heatmap = make_heatmap(visible_uvs, CAM_W, CAM_H, HEATMAP_SIGMA)
                ep_images.append(img)
                ep_heatmaps.append(heatmap)

                # Step using trained policy
                action, _ = model.predict(norm_obs, deterministic=True)
                obs, _, terminated, truncated, _ = vis_env.step(action[0])
                done = terminated or truncated

                if hasattr(vec_env, "normalize_obs"):
                    norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
                else:
                    norm_obs = obs.reshape(1, -1)

            # Append episode frames to HDF5
            n = len(ep_images)
            old = img_ds.shape[0]
            img_ds.resize(old + n, axis=0)
            img_ds[old:] = np.stack(ep_images)
            hm_ds.resize(old + n, axis=0)
            hm_ds[old:] = np.stack(ep_heatmaps)
            total_frames += n

            if (ep + 1) % 25 == 0 or ep == args.episodes - 1:
                print(f"  Episode {ep + 1:4d}/{args.episodes} — {total_frames:,} frames total")

    vis_env.sim.close_renderer()
    print(f"\nDone. {total_frames:,} frames saved to {args.output}")


if __name__ == "__main__":
    main()
