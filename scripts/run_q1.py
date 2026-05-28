"""Entry point for DCL AI Grand Prix Q1 Virtual Qualifier.

Connects to the DCL simulator via MAVLink, runs the trained PPO policy,
and sends attitude/thrust commands to navigate the race course.

Usage:
    python scripts/run_q1.py --model checkpoints/ppo_drone_v105/final_model.zip \
                              --normalize checkpoints/ppo_drone_v105/vec_normalize.pkl

    # Custom connection (if DCL sim uses different port):
    python scripts/run_q1.py --model ... --connection "udpin:0.0.0.0:14560"

    # With known gate positions (JSON file):
    python scripts/run_q1.py --model ... --gates course_gates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dronesim.bridge.q1_runner import Q1Runner


def load_gates(path: str) -> list[np.ndarray]:
    """Load gate positions from a JSON file.

    Expected format: [[x, y, z], [x, y, z], ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [np.array(g, dtype=np.float64) for g in data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DCL AI Grand Prix Q1 — MAVLink PPO Runner"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO model (.zip)",
    )
    parser.add_argument(
        "--normalize", type=str, default=None,
        help="Path to VecNormalize stats (.pkl)",
    )
    parser.add_argument(
        "--connection", type=str, default="udpin:0.0.0.0:14550",
        help="MAVLink connection string (default: udpin:0.0.0.0:14550)",
    )
    parser.add_argument(
        "--command-hz", type=float, default=100.0,
        help="Command send rate in Hz (default 100, matching training policy_hz)",
    )
    parser.add_argument(
        "--max-body-rate", type=float, default=15.0,
        help="Max body rate rad/s (must match training config drone.max_body_rate_rad_s)",
    )
    parser.add_argument(
        "--gates", type=str, default=None,
        help="Path to JSON file with gate positions [[x,y,z], ...]",
    )
    parser.add_argument(
        "--unet", type=str, default=None,
        help="Path to U-Net .pt checkpoint for vision-based gate detection",
    )
    parser.add_argument(
        "--n-gates", type=int, default=10,
        help="Expected number of gates in course (for Kalman filter allocation)",
    )
    parser.add_argument(
        "--gate-radius", type=float, default=0.50,
        help="Physical gate radius in meters (for depth estimation)",
    )
    parser.add_argument(
        "--cam-fov", type=float, default=90.0,
        help="Camera vertical FOV in degrees",
    )
    parser.add_argument(
        "--timeout", type=float, default=480.0,
        help="Max run duration in seconds (Q1 limit: 480)",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    args = parser.parse_args()

    runner = Q1Runner(
        model_path=args.model,
        normalize_path=args.normalize,
        connection_string=args.connection,
        command_hz=args.command_hz,
        max_body_rate=args.max_body_rate,
        deterministic=not args.stochastic,
        unet_path=args.unet,
        n_gates=args.n_gates,
        gate_radius_m=args.gate_radius,
        cam_fov_deg=args.cam_fov,
    )

    if args.gates:
        gates = load_gates(args.gates)
        runner.set_gate_positions(gates)
        print(f"Loaded {len(gates)} gate positions from {args.gates}")

    runner.run(timeout_s=args.timeout)


if __name__ == "__main__":
    main()
