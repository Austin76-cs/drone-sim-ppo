"""Stress-test model on diverse course types that Q1 might throw at us.

Tests: variable gate counts, wider turns, extreme elevation, larger spacing,
tighter gates, and combinations. Identifies weaknesses for targeted training.

Includes crossing quality diagnostics to detect sloppy gate passes.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dronesim.config import load_config
from dronesim.envs.drone_race_env import DroneRaceEnv
from dronesim.tasks.curriculum import (
    CurriculumStage,
    StageController,
    StageSpec,
    generate_gate_course,
)


@dataclass
class TestScenario:
    name: str
    num_gates: int
    spacing_m: float
    max_yaw_rad: float
    vertical_span_m: float
    gate_radius_m: float
    lateral_span_m: float = 2.5
    max_distance_m: float = 12.0
    height_bias_range: tuple[float, ...] = (-0.6, 0.0, 0.6)
    height_bias_probs: tuple[float, ...] = (0.25, 0.50, 0.25)
    miss_limit: int = 6


SCENARIOS = [
    # Baseline: our standard SPRINT config
    TestScenario(
        name="baseline_10g_50deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50,
    ),
    # Fewer gates (short course)
    TestScenario(
        name="short_5g_50deg",
        num_gates=5, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50, miss_limit=3,
    ),
    TestScenario(
        name="short_7g_50deg",
        num_gates=7, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50, miss_limit=4,
    ),
    # More gates (longer course)
    TestScenario(
        name="long_12g_50deg",
        num_gates=12, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50, miss_limit=7,
    ),
    TestScenario(
        name="long_15g_50deg",
        num_gates=15, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50, miss_limit=9,
    ),
    # Wider turns
    TestScenario(
        name="wide_turns_10g_60deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=1.05,
        vertical_span_m=0.8, gate_radius_m=0.50,
    ),
    TestScenario(
        name="wide_turns_10g_70deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=1.22,
        vertical_span_m=0.8, gate_radius_m=0.50,
    ),
    TestScenario(
        name="wide_turns_10g_80deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=1.40,
        vertical_span_m=0.8, gate_radius_m=0.50,
    ),
    # Extreme elevation changes
    TestScenario(
        name="hilly_10g_50deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=1.5, gate_radius_m=0.50,
        height_bias_range=(-1.0, 0.0, 1.0),
        height_bias_probs=(0.33, 0.34, 0.33),
    ),
    # Larger spacing (bigger course)
    TestScenario(
        name="spread_10g_50deg",
        num_gates=10, spacing_m=8.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50,
        max_distance_m=18.0,
    ),
    # Tighter gates
    TestScenario(
        name="tight_gates_10g_50deg",
        num_gates=10, spacing_m=5.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.40,
    ),
    # Compact spacing (gates close together)
    TestScenario(
        name="compact_10g_50deg",
        num_gates=10, spacing_m=3.0, max_yaw_rad=0.87,
        vertical_span_m=0.8, gate_radius_m=0.50,
        max_distance_m=8.0,
    ),
    # Gentle course but many gates
    TestScenario(
        name="gentle_long_15g_30deg",
        num_gates=15, spacing_m=5.0, max_yaw_rad=0.52,
        vertical_span_m=0.5, gate_radius_m=0.50, miss_limit=9,
    ),
    # Worst case: many gates, tight turns, tight gates, elevation
    TestScenario(
        name="nightmare_12g_70deg_tight",
        num_gates=12, spacing_m=5.0, max_yaw_rad=1.22,
        vertical_span_m=1.5, gate_radius_m=0.40, miss_limit=7,
        height_bias_range=(-1.0, 0.0, 1.0),
        height_bias_probs=(0.33, 0.34, 0.33),
    ),
]


def make_custom_course(scenario: TestScenario, rng: np.random.Generator, gate_depth: float):
    """Generate a course matching the scenario parameters."""
    gates = []
    pos_xy = np.array([2.8, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0], dtype=np.float64)

    for gate_idx in range(scenario.num_gates):
        if gate_idx > 0:
            yaw_delta = rng.uniform(-scenario.max_yaw_rad, scenario.max_yaw_rad)
            c, s = np.cos(yaw_delta), np.sin(yaw_delta)
            direction = np.array([
                c * direction[0] - s * direction[1],
                s * direction[0] + c * direction[1],
            ])
            direction /= np.linalg.norm(direction)

        spacing = scenario.spacing_m * rng.uniform(0.7, 1.5)
        pos_xy = pos_xy + direction * spacing

        height_bias = rng.choice(
            list(scenario.height_bias_range),
            p=list(scenario.height_bias_probs),
        )
        z = float(np.clip(
            1.0 + height_bias + rng.uniform(-scenario.vertical_span_m, scenario.vertical_span_m),
            0.5, 3.5,
        ))

        center = np.array([pos_xy[0], pos_xy[1], z], dtype=np.float64)
        normal = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        size_scale = rng.uniform(0.85, 1.15)
        gate_w = scenario.gate_radius_m * 2.0 * size_scale
        gate_h = scenario.gate_radius_m * 2.0 * size_scale
        radius = min(gate_w, gate_h) / 2.0

        from dronesim.types import GateSpec
        gates.append(GateSpec(
            center=center, normal=normal, radius_m=radius,
            depth_m=gate_depth, width_m=gate_w, height_m=gate_h,
        ))

    return gates


class CustomDroneRaceEnv(DroneRaceEnv):
    """DroneRaceEnv that uses a custom scenario for course generation."""

    def __init__(self, config, scenario: TestScenario):
        super().__init__(config)
        self.scenario = scenario
        # Force SPRINT stage
        self.stage_controller.stage = CurriculumStage.SPRINT

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # Replace the generated course with our custom one
        assert self.current_task is not None

        custom_gates = make_custom_course(
            self.scenario, self.rng, self.config.task.gate_depth_m,
        )

        # Rebuild task with custom gates
        from dronesim.tasks.curriculum import EpisodeTask
        approach_dir = custom_gates[0].normal
        spawn_pos = custom_gates[0].center - approach_dir * 3.0
        spawn_pos[2] = max(0.55, spawn_pos[2])

        self.current_task = EpisodeTask(
            stage=self.current_task.stage,
            spawn_position=spawn_pos,
            spawn_velocity=np.array([approach_dir[0] * 0.2, approach_dir[1] * 0.2, 0.0]),
            spawn_euler=np.array([0.0, 0.0, float(np.arctan2(approach_dir[1], approach_dir[0]))]),
            spawn_omega=np.zeros(3),
            gates=tuple(custom_gates),
            max_steps=int(self.config.sim.episode_seconds * self.config.sim.policy_hz),
            max_distance_m=self.scenario.max_distance_m,
            randomization_scale=1.0,
            gate_pass_margin_m=self.config.task.gate_pass_margin_m,
        )

        # Re-init episode state for new gates
        from dronesim.sim.env import euler_to_quat_wxyz
        quat = euler_to_quat_wxyz(self.current_task.spawn_euler)
        qpos = np.zeros(7, dtype=np.float64)
        qpos[:3] = self.current_task.spawn_position
        qpos[3:7] = quat
        qvel = np.zeros(6, dtype=np.float64)
        qvel[:3] = self.current_task.spawn_velocity

        self.gate_index = 0
        self.gates_cleared = 0
        self.crossing_offsets = []
        self.step_idx = 0
        self.miss_count = 0
        self.state = self.sim.reset(qpos, qvel)
        self.prev_pos = self.state.pos.copy()
        self._set_gate_progress_reference()

        obs = self._build_obs()
        return obs, {"curriculum_stage": 3}


def run_scenario(model, config, scenario: TestScenario, episodes: int, normalize_path: str | None):
    """Run evaluation on a single scenario and return results."""
    def make_env():
        return CustomDroneRaceEnv(config, scenario)

    vec_env = DummyVecEnv([make_env])
    if normalize_path:
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Re-load model for this env
    local_model = PPO.load(model, env=vec_env)

    crash_types: Counter[str] = Counter()
    all_gates: list[int] = []
    all_completions: list[float] = []
    all_crossing_offsets: list[float] = []
    successes = 0

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_info: dict = {}

        while not done:
            action, _ = local_model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_info = infos[0]
            done = bool(dones[0])

        crash_type = ep_info.get("crash_type", "unknown")
        gates = ep_info.get("gates_cleared", 0)
        completion = ep_info.get("completion", 0.0)
        offsets = ep_info.get("crossing_offsets", [])

        crash_types[crash_type] += 1
        all_gates.append(gates)
        all_completions.append(completion)
        all_crossing_offsets.extend(offsets)
        if crash_type == "success":
            successes += 1

    vec_env.close()

    # Crossing quality stats
    if all_crossing_offsets:
        avg_offset = float(np.mean(all_crossing_offsets))
        max_offset = float(np.max(all_crossing_offsets))
        pct_centered = sum(1 for o in all_crossing_offsets if o < 0.5) / len(all_crossing_offsets)
        pct_edge = sum(1 for o in all_crossing_offsets if o > 0.8) / len(all_crossing_offsets)
    else:
        avg_offset = float("nan")
        max_offset = float("nan")
        pct_centered = 0.0
        pct_edge = 0.0

    return {
        "name": scenario.name,
        "num_gates": scenario.num_gates,
        "success_rate": successes / episodes,
        "mean_gates": np.mean(all_gates),
        "mean_completion": np.mean(all_completions),
        "crash_types": dict(crash_types.most_common()),
        "episodes": episodes,
        "avg_crossing_offset": avg_offset,
        "max_crossing_offset": max_offset,
        "pct_centered": pct_centered,
        "pct_edge": pct_edge,
        "total_passes": len(all_crossing_offsets),
    }


def main():
    parser = argparse.ArgumentParser(description="Stress-test model on diverse Q1 course types")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--normalize", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    config = load_config(Path(args.config))

    print("=" * 80)
    print("STRESS TEST: Evaluating model on diverse Q1 course types")
    print(f"Model: {args.model}")
    print(f"Episodes per scenario: {args.episodes}")
    print("=" * 80)

    results = []
    for scenario in SCENARIOS:
        print(f"\n--- {scenario.name} ({scenario.num_gates}g, "
              f"{np.degrees(scenario.max_yaw_rad):.0f}deg, "
              f"r={scenario.gate_radius_m}m, "
              f"spacing={scenario.spacing_m}m) ---")

        result = run_scenario(args.model, config, scenario, args.episodes, args.normalize)
        results.append(result)

        offset_str = f"{result['avg_crossing_offset']:.2f}" if not math.isnan(result['avg_crossing_offset']) else "N/A"
        print(f"  Success: {result['success_rate']:.0%}  "
              f"Gates: {result['mean_gates']:.1f}/{scenario.num_gates}  "
              f"Completion: {result['mean_completion']:.2f}  "
              f"Crashes: {result['crash_types']}")
        print(f"  Crossing quality: avg_offset={offset_str}  "
              f"centered(<0.5)={result['pct_centered']:.0%}  "
              f"edge(>0.8)={result['pct_edge']:.0%}  "
              f"total_passes={result['total_passes']}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Scenario':<35} {'Success':>8} {'Gates':>10} {'Completion':>11} {'AvgOffset':>10} {'Centered':>9} {'Grade':>6}")
    print("-" * 80)

    grades = []
    for r in results:
        sr = r["success_rate"]
        if sr >= 0.90:
            grade = "A"
        elif sr >= 0.75:
            grade = "B"
        elif sr >= 0.50:
            grade = "C"
        elif sr >= 0.25:
            grade = "D"
        else:
            grade = "F"
        grades.append(grade)

        gate_str = f"{r['mean_gates']:.1f}/{r['num_gates']}"
        offset_str = f"{r['avg_crossing_offset']:.2f}" if not math.isnan(r['avg_crossing_offset']) else "N/A"
        print(f"{r['name']:<35} {sr:>7.0%} {gate_str:>10} {r['mean_completion']:>10.2f} "
              f"{offset_str:>10} {r['pct_centered']:>8.0%} {grade:>6}")

    # Crossing quality summary
    all_offsets = [r["avg_crossing_offset"] for r in results if not math.isnan(r["avg_crossing_offset"])]
    total_passes = sum(r["total_passes"] for r in results)
    total_possible = sum(r["num_gates"] * r["episodes"] for r in results)
    overall_centered = sum(r["pct_centered"] * r["total_passes"] for r in results) / max(total_passes, 1)

    print("\n" + "=" * 80)
    print("CROSSING QUALITY DIAGNOSTICS:")
    if all_offsets:
        print(f"  Overall avg crossing offset: {np.mean(all_offsets):.3f}  (0=center, 1=edge)")
        print(f"  Overall centered rate (<0.5): {overall_centered:.0%}")
        print(f"  Total gate passes: {total_passes}/{total_possible} ({total_passes/max(total_possible,1):.0%} of all gates)")

        # RED FLAGS
        flags = []
        if np.mean(all_offsets) > 0.7:
            flags.append("AVG OFFSET > 0.7: passes are near the edge, drone barely fitting through")
        if overall_centered < 0.5:
            flags.append("CENTERED RATE < 50%: most passes are off-center, likely clipping gates")
        if total_passes / max(total_possible, 1) < 0.3:
            flags.append(f"PASS RATE < 30%: only {total_passes}/{total_possible} gates passed, drone missing most gates")

        # Check for high success but low pass quality (the bug we had)
        for r in results:
            if r["success_rate"] >= 0.8 and not math.isnan(r["avg_crossing_offset"]) and r["avg_crossing_offset"] > 0.85:
                flags.append(f"SUSPICIOUS: {r['name']} has {r['success_rate']:.0%} success but avg offset {r['avg_crossing_offset']:.2f} — passes may be too loose")

        if flags:
            print("\n  *** RED FLAGS ***")
            for f in flags:
                print(f"  !!! {f}")
        else:
            print("\n  No red flags detected. Gate passes look clean.")
    else:
        print("  !!! NO GATE PASSES RECORDED — model is not passing through any gates")

    # Identify weaknesses
    print("\n" + "=" * 80)
    weak = [r for r, g in zip(results, grades) if g in ("C", "D", "F")]
    if weak:
        print("WEAKNESSES IDENTIFIED:")
        for r in weak:
            print(f"  - {r['name']}: {r['success_rate']:.0%} success")
    else:
        print("NO MAJOR WEAKNESSES - model generalizes well across all tested scenarios.")

    strong = [r for r, g in zip(results, grades) if g == "A"]
    print(f"\nOVERALL: {len(strong)}/{len(results)} scenarios at A grade (90%+)")
    print("=" * 80)


if __name__ == "__main__":
    main()
