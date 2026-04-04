from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from dronesim.config import TaskConfig
from dronesim.types import GateSpec


class CurriculumStage(IntEnum):
    INTRO = 0
    OFFSET = 1
    SLALOM = 2
    SPRINT = 3


@dataclass(slots=True, frozen=True)
class StageSpec:
    mastery_threshold: float
    max_steps_scale: float
    num_gates: int
    spacing_m: float
    lateral_span_m: float
    vertical_span_m: float
    spawn_backtrack_m: float
    spawn_xy_jitter_m: float
    spawn_tilt_rad: float
    spawn_speed_m_s: float
    spawn_omega_rad_s: float
    max_distance_m: float
    randomization_scale: float


@dataclass(slots=True, frozen=True)
class EpisodeTask:
    stage: CurriculumStage
    spawn_position: NDArray[np.float64]
    spawn_velocity: NDArray[np.float64]
    spawn_euler: NDArray[np.float64]
    spawn_omega: NDArray[np.float64]
    gates: tuple[GateSpec, ...]
    max_steps: int
    max_distance_m: float
    randomization_scale: float
    gate_pass_margin_m: float


@dataclass(slots=True, frozen=True)
class EpisodeSummary:
    stage: CurriculumStage
    success: bool
    terminated: bool
    truncated: bool
    crash_type: str
    completion: float
    score: float
    gates_cleared: int
    steps: int


def _get_stage_spec(stage: CurriculumStage, config: TaskConfig) -> StageSpec:
    if stage == CurriculumStage.INTRO:
        return StageSpec(
            mastery_threshold=config.intro_threshold,
            max_steps_scale=0.55,
            num_gates=3,
            spacing_m=config.base_gate_spacing_m,
            lateral_span_m=0.0,
            vertical_span_m=0.15,
            spawn_backtrack_m=1.8,
            spawn_xy_jitter_m=0.08,
            spawn_tilt_rad=0.18,
            spawn_speed_m_s=0.18,
            spawn_omega_rad_s=0.8,
            max_distance_m=3.5,
            randomization_scale=0.35,
        )
    if stage == CurriculumStage.OFFSET:
        return StageSpec(
            mastery_threshold=config.slalom_threshold,
            max_steps_scale=0.80,
            num_gates=4,
            spacing_m=config.base_gate_spacing_m * 1.05,
            lateral_span_m=0.9,
            vertical_span_m=0.25,
            spawn_backtrack_m=2.0,
            spawn_xy_jitter_m=0.16,
            spawn_tilt_rad=0.24,
            spawn_speed_m_s=0.24,
            spawn_omega_rad_s=1.0,
            max_distance_m=4.5,
            randomization_scale=0.65,
        )
    if stage == CurriculumStage.SLALOM:
        return StageSpec(
            mastery_threshold=config.sprint_threshold,
            max_steps_scale=1.00,
            num_gates=6,
            spacing_m=config.base_gate_spacing_m * 1.10,
            lateral_span_m=1.2,
            vertical_span_m=0.35,
            spawn_backtrack_m=2.3,
            spawn_xy_jitter_m=0.20,
            spawn_tilt_rad=0.28,
            spawn_speed_m_s=0.30,
            spawn_omega_rad_s=1.2,
            max_distance_m=5.5,
            randomization_scale=1.00,
        )
    # SPRINT: competition-realistic stage modeled after A2RL/AI Grand Prix
    return StageSpec(
        mastery_threshold=config.competition_threshold,
        max_steps_scale=1.00,
        num_gates=10,
        spacing_m=config.base_gate_spacing_m * 2.0,
        lateral_span_m=2.5,
        vertical_span_m=0.8,
        spawn_backtrack_m=3.0,
        spawn_xy_jitter_m=0.25,
        spawn_tilt_rad=0.30,
        spawn_speed_m_s=0.40,
        spawn_omega_rad_s=1.5,
        max_distance_m=8.0,
        randomization_scale=1.00,
    )


def generate_gate_course(
    spec: StageSpec,
    stage: CurriculumStage,
    rng: np.random.Generator,
    gate_radius_m: float,
    gate_depth_m: float,
) -> list[GateSpec]:
    """Generate a sequence of gates for a curriculum stage."""
    gates: list[GateSpec] = []
    x_position = 2.8
    prev_offset_y = 0.0

    for gate_idx in range(spec.num_gates):
        if stage == CurriculumStage.INTRO:
            offset_y = 0.0
        elif stage == CurriculumStage.OFFSET:
            offset_y = rng.uniform(-spec.lateral_span_m, spec.lateral_span_m)
        elif stage == CurriculumStage.SPRINT:
            # Competition-style: mix of slalom, random offsets, and occasional
            # double-gates (two gates close together)
            if gate_idx % 3 == 0:
                # Slalom-style alternating
                sign = -1.0 if gate_idx % 2 == 0 else 1.0
                offset_y = sign * rng.uniform(0.8, spec.lateral_span_m)
            else:
                # Random placement across the full lateral range
                offset_y = rng.uniform(-spec.lateral_span_m, spec.lateral_span_m)
            # Ensure meaningful lateral change between consecutive gates
            if abs(offset_y - prev_offset_y) < 0.35:
                offset_y = -prev_offset_y * 0.8 + rng.uniform(-0.3, 0.3)
                offset_y = np.clip(offset_y, -spec.lateral_span_m, spec.lateral_span_m)
        else:
            sign = -1.0 if gate_idx % 2 == 0 else 1.0
            offset_y = sign * rng.uniform(0.45, spec.lateral_span_m)
            if abs(offset_y - prev_offset_y) < 0.20:
                offset_y = sign * spec.lateral_span_m

        # Height: base 1m + stage vertical range + extra climbs/dips on later stages
        if stage in (CurriculumStage.SLALOM, CurriculumStage.SPRINT):
            height_bias = rng.choice([-0.6, 0.0, 0.6], p=[0.25, 0.50, 0.25])
        else:
            height_bias = 0.0
        offset_z = np.clip(
            1.0 + height_bias + rng.uniform(-spec.vertical_span_m, spec.vertical_span_m),
            0.5, 3.0,
        )
        center = np.array([x_position, offset_y, offset_z], dtype=np.float64)

        if stage == CurriculumStage.SPRINT and gate_idx > 0:
            # Angled normals: gate faces the approach direction from previous gate
            approach = center - gates[-1].center
            approach[2] = 0.0  # keep normal horizontal
            norm = np.linalg.norm(approach)
            if norm > 1e-6:
                normal = approach / norm
            else:
                normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Per-gate random radius and spacing for variety
        radius = gate_radius_m * rng.uniform(0.75, 1.35)
        gates.append(GateSpec(center=center, normal=normal, radius_m=radius, depth_m=gate_depth_m))
        x_position += spec.spacing_m * rng.uniform(0.7, 1.5)
        prev_offset_y = offset_y

    return gates


# Sampling weights for multi-stage mode: [INTRO, OFFSET, SLALOM, SPRINT]
_MULTI_STAGE_WEIGHTS = [0.15, 0.20, 0.35, 0.30]


@dataclass(slots=True)
class StageController:
    config: TaskConfig
    stage: CurriculumStage = CurriculumStage.INTRO
    multi_stage: bool = False
    _history: dict[CurriculumStage, deque[float]] = field(init=False, repr=False)
    _episode_counts: dict[CurriculumStage, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        window = max(1, self.config.curriculum_window)
        self._history = {stage: deque(maxlen=window) for stage in CurriculumStage}
        self._episode_counts = {stage: 0 for stage in CurriculumStage}

    def sample_task(
        self,
        rng: np.random.Generator,
        base_episode_steps: int,
    ) -> EpisodeTask:
        if self.multi_stage:
            active_stage = CurriculumStage(
                rng.choice(len(CurriculumStage), p=_MULTI_STAGE_WEIGHTS)
            )
        else:
            active_stage = self.stage
        spec = _get_stage_spec(active_stage, self.config)
        max_steps = max(1, int(base_episode_steps * spec.max_steps_scale))

        gate_radius = (
            self.config.sprint_gate_radius_m
            if active_stage == CurriculumStage.SPRINT
            else self.config.gate_radius_m
        )
        gates = generate_gate_course(
            spec, active_stage, rng,
            gate_radius, self.config.gate_depth_m,
        )

        first_gate = gates[0].center
        spawn_position = first_gate + np.array([
            -spec.spawn_backtrack_m,
            rng.uniform(-spec.spawn_xy_jitter_m, spec.spawn_xy_jitter_m),
            rng.uniform(-spec.spawn_xy_jitter_m, spec.spawn_xy_jitter_m),
        ], dtype=np.float64)
        spawn_position[2] = max(0.55, spawn_position[2])

        spawn_velocity = np.array([
            rng.uniform(0.0, spec.spawn_speed_m_s),
            rng.uniform(-spec.spawn_speed_m_s * 0.5, spec.spawn_speed_m_s * 0.5),
            rng.uniform(-spec.spawn_speed_m_s * 0.35, spec.spawn_speed_m_s * 0.35),
        ], dtype=np.float64)

        spawn_euler = np.array([
            rng.uniform(-spec.spawn_tilt_rad * 0.6, spec.spawn_tilt_rad * 0.6),
            rng.uniform(-spec.spawn_tilt_rad, spec.spawn_tilt_rad),
            rng.uniform(-0.15, 0.15),
        ], dtype=np.float64)

        spawn_omega = rng.uniform(
            -spec.spawn_omega_rad_s, spec.spawn_omega_rad_s, size=3,
        ).astype(np.float64)

        return EpisodeTask(
            stage=active_stage,
            spawn_position=spawn_position,
            spawn_velocity=spawn_velocity,
            spawn_euler=spawn_euler,
            spawn_omega=spawn_omega,
            gates=tuple(gates),
            max_steps=max_steps,
            max_distance_m=spec.max_distance_m,
            randomization_scale=spec.randomization_scale,
            gate_pass_margin_m=self.config.gate_pass_margin_m,
        )

    def record_episode(self, summary: EpisodeSummary) -> None:
        self._episode_counts[summary.stage] += 1
        history = self._history[summary.stage]
        history.append(summary.score)

        if summary.stage != self.stage or self.stage == CurriculumStage.SPRINT:
            return

        min_episodes = max(1, self.config.curriculum_min_episodes)
        if (
            self._episode_counts[self.stage] < min_episodes
            or len(history) < min_episodes
        ):
            return

        mastery = sum(history) / len(history)
        threshold = _get_stage_spec(self.stage, self.config).mastery_threshold
        if mastery >= threshold:
            self.stage = CurriculumStage(int(self.stage) + 1)

    def force_stage(self, stage: CurriculumStage) -> None:
        self.stage = stage

    @property
    def mastery(self) -> float:
        history = self._history[self.stage]
        if not history:
            return 0.0
        return float(sum(history) / len(history))
