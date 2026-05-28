"""Validate the Phase 3 EKF pipeline against MuJoCo ground truth.

Runs the drone through a SPRINT course, renders camera frames, runs the
U-Net + GateDetector + GateEstimator + GateFilter pipeline, then compares
estimated gate positions vs actual gate positions.

Usage:
    python scripts/validate_ekf.py \
        --model checkpoints/ppo_drone_v60/best_model.zip \
        --normalize checkpoints/ppo_drone_v60/vec_normalize.pkl \
        --unet checkpoints/unet_v1/best_model.pt \
        --stage 3 --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.ekf.gate_detector import GateDetector
from dronesim.ekf.gate_estimator import GateEstimator
from dronesim.ekf.gate_filter import MultiGateFilter
from dronesim.perception.unet import GateUNet
from dronesim.sim.env import euler_to_rotation_matrix
from dronesim.tasks.curriculum import CurriculumStage, StageController


IMG_W, IMG_H = 160, 120


def make_env(config, stage, seed):
    def _init():
        import copy
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        controller = StageController(cfg.task)
        controller.force_stage(CurriculumStage(stage))
        return DroneRaceEnv(cfg, stage_controller=controller)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--normalize", required=True)
    parser.add_argument("--unet", required=True)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # Load PPO policy
    vec_env = DummyVecEnv([make_env(config, args.stage, config.seed)])
    vec_env = VecNormalize.load(args.normalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model, env=vec_env, device=args.device)

    # Load U-Net
    unet = GateUNet.from_checkpoint(args.unet, device=args.device)

    # Get the underlying env to access MuJoCoSim directly
    inner_env: DroneRaceEnv = vec_env.envs[0]

    # Build EKF components
    fx, fy, cx, cy = inner_env.sim.get_camera_intrinsics(IMG_W, IMG_H)
    gate_radius = config.task.gate_radius_m
    detector = GateDetector(threshold=0.3, min_area_px=20)
    estimator = GateEstimator(fx, fy, cx, cy, gate_radius_m=gate_radius)

    print(f"Camera intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"Running {args.episodes} episodes on stage {args.stage}...")
    print()

    all_errors: list[float] = []
    detection_rate_list: list[float] = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        n_gates = len(inner_env.current_task.gates)
        gate_filter = MultiGateFilter(n_gates=n_gates, measurement_noise=0.5)
        gate_filter.reset()

        # Position gates in the renderer
        inner_env.sim.set_gate_visuals(list(inner_env.current_task.gates))

        step_errors: list[float] = []
        detections = 0
        total_steps = 0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated = vec_env.step(action)
            done = bool(terminated[0] or truncated[0])

            # Render camera frame
            frame_rgb = inner_env.sim.render_camera(IMG_W, IMG_H)
            frame_tensor = torch.from_numpy(
                frame_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(args.device)

            with torch.no_grad():
                heatmap = unet(frame_tensor)[0, 0].cpu().numpy()

            # Get camera pose
            cam_pos, cam_xmat = inner_env.sim.get_camera_extrinsics()
            drone_pos = inner_env.state.pos
            drone_rot = euler_to_rotation_matrix(inner_env.state.euler)

            # Detect gate in heatmap (we look for the current target gate)
            detection = detector.detect(heatmap)
            measurements: list[np.ndarray | None] = [None] * n_gates

            if detection is not None:
                u, v, radius_px = detection
                pos_world = estimator.estimate_world(u, v, radius_px, cam_pos, cam_xmat)
                if pos_world is not None:
                    # Associate measurement with the current gate index
                    cur_idx = inner_env.gate_index
                    measurements[cur_idx] = pos_world
                    detections += 1

            gate_filter.step(measurements)
            total_steps += 1

            # Compute error for current gate vs ground truth
            cur_idx = inner_env.gate_index
            est = gate_filter.get_estimates()[cur_idx]
            if est is not None:
                gt = inner_env.current_task.gates[cur_idx].center
                err = float(np.linalg.norm(est - gt))
                step_errors.append(err)

        ep_mean_err = np.mean(step_errors) if step_errors else float("nan")
        det_rate = detections / max(total_steps, 1)
        all_errors.extend(step_errors)
        detection_rate_list.append(det_rate)

        gates_cleared = inner_env.gates_cleared
        print(f"Episode {ep+1}: gates={gates_cleared}/{n_gates}  "
              f"det_rate={det_rate:.1%}  "
              f"mean_pos_err={ep_mean_err:.2f}m  "
              f"steps={total_steps}")

    print()
    print("=" * 50)
    print(f"Overall mean position error : {np.nanmean(all_errors):.3f} m")
    print(f"Overall detection rate      : {np.mean(detection_rate_list):.1%}")
    print(f"  (errors >0.5m suggest tuning needed for measurement_noise or threshold)")


if __name__ == "__main__":
    main()
