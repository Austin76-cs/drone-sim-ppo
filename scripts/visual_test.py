"""Visual test: watch the trained PPO agent fly through gates in the MuJoCo viewer."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.types import GateSpec


GATE_COLOR_DEFAULT = (0.1, 0.8, 0.2, 0.7)   # green
GATE_COLOR_ACTIVE = (1.0, 0.85, 0.0, 0.85)  # yellow — current target
GATE_COLOR_CLEARED = (0.3, 0.3, 0.3, 0.35)  # dim grey
GATE_SEGMENTS = 24  # number of capsules forming each ring


def _normal_to_quat(normal: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) that rotates +x to the given normal direction."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0])
    dot = np.dot(ref, n)
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    cross = np.cross(ref, n)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


def _draw_gate_ring(
    viewer: mujoco.viewer.Handle,
    gate: GateSpec,
    color: tuple[float, float, float, float],
    gate_idx: int,
) -> None:
    """Draw a gate as a ring of small capsules using viewer user geoms."""
    radius = gate.radius_m
    center = gate.center
    normal = gate.normal / (np.linalg.norm(gate.normal) + 1e-12)

    # Build a local coordinate frame: normal is the "forward" axis
    # Pick an arbitrary up vector that isn't parallel to normal
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(normal, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, normal)

    tube_radius = 0.03  # thickness of the ring segments

    for i in range(GATE_SEGMENTS):
        theta0 = 2.0 * math.pi * i / GATE_SEGMENTS
        theta1 = 2.0 * math.pi * (i + 1) / GATE_SEGMENTS

        p0 = center + radius * (math.cos(theta0) * right + math.sin(theta0) * up)
        p1 = center + radius * (math.cos(theta1) * right + math.sin(theta1) * up)

        mid = 0.5 * (p0 + p1)
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)

        geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[tube_radius, seg_len * 0.5, 0],
            pos=mid,
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32),
        )

        # Orient capsule along the segment direction
        if seg_len > 1e-9:
            seg_dir = seg / seg_len
            # MuJoCo capsule is along z-axis, we need to rotate z -> seg_dir
            z = np.array([0.0, 0.0, 1.0])
            dot = np.dot(z, seg_dir)
            if dot < -0.9999:
                mat = np.diag([-1.0, 1.0, -1.0])
            elif dot > 0.9999:
                mat = np.eye(3)
            else:
                cross = np.cross(z, seg_dir)
                skew = np.array([
                    [0, -cross[2], cross[1]],
                    [cross[2], 0, -cross[0]],
                    [-cross[1], cross[0], 0],
                ])
                mat = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))
            geom.mat[:] = mat

        viewer.user_scn.ngeom += 1
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1] = geom


def _draw_gates(
    viewer: mujoco.viewer.Handle,
    gates: tuple[GateSpec, ...],
    current_gate_idx: int,
    gates_cleared: int,
) -> None:
    """Draw all gates with color coding: cleared/active/upcoming."""
    viewer.user_scn.ngeom = 0
    for i, gate in enumerate(gates):
        if i < gates_cleared:
            color = GATE_COLOR_CLEARED
        elif i == current_gate_idx:
            color = GATE_COLOR_ACTIVE
        else:
            color = GATE_COLOR_DEFAULT
        _draw_gate_ring(viewer, gate, color, i)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize trained PPO agent")
    parser.add_argument("--model", type=str,
                        default="checkpoints/ppo_drone_v4/best_model.zip")
    parser.add_argument("--normalize", type=str,
                        default="checkpoints/ppo_drone_v4/vec_normalize.pkl")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))

    def make_env():
        return DroneRaceEnv(config)

    vec_env = DummyVecEnv([make_env])
    if args.normalize and Path(args.normalize).exists():
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    # Get the underlying DroneRaceEnv for viewer access
    raw_env: DroneRaceEnv = vec_env.envs[0].unwrapped  # type: ignore
    sim = raw_env.sim
    control_dt = 1.0 / config.sim.policy_hz

    obs = vec_env.reset()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = sim.data.qpos[:3]

        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = vec_env.step(action)

            # Draw gates with color coding
            if raw_env.current_task is not None:
                _draw_gates(
                    viewer,
                    raw_env.current_task.gates,
                    raw_env.gate_index,
                    raw_env.gates_cleared,
                )

            if dones[0]:
                info = infos[0]
                print(f"Episode ended: {info.get('crash_type', '?')} | "
                      f"gates={info.get('gates_cleared', 0)} | "
                      f"completion={info.get('completion', 0):.2f}")
                obs = vec_env.reset()

            # Track the drone with the camera
            viewer.cam.lookat[:] = sim.data.qpos[:3]
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
