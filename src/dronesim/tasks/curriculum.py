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
            vertical_span_m=0.08,
            spawn_backtrack_m=1.7,
            spawn_xy_jitter_m=0.05,
            spawn_tilt_rad=0.10,
            spawn_speed_m_s=0.10,
            spawn_omega_rad_s=0.25,
            max_distance_m=3.8,
            randomization_scale=0.18,
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
        else:
            sign = -1.0 if gate_idx % 2 == 0 else 1.0
            offset_y = sign * rng.uniform(0.45, spec.lateral_span_m)
            if abs(offset_y - prev_offset_y) < 0.20:
                offset_y = sign * spec.lateral_span_m

        offset_z = 1.0 + rng.uniform(-spec.vertical_span_m, spec.vertical_span_m)
        center = np.array([x_position, offset_y, offset_z], dtype=np.float64)
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gates.append(GateSpec(center=center, normal=normal, radius_m=gate_radius_m, depth_m=gate_depth_m))
        x_position += spec.spacing_m
        prev_offset_y = offset_y

    return gates


@dataclass(slots=True)
class StageController:
    config: TaskConfig
    stage: CurriculumStage = CurriculumStage.INTRO
    locked_stage: CurriculumStage | None = None
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
        spec = _get_stage_spec(self.stage, self.config)
        max_steps = max(1, int(base_episode_steps * spec.max_steps_scale))

        gates = generate_gate_course(
            spec, self.stage, rng,
            self.config.gate_radius_m, self.config.gate_depth_m,
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
            stage=self.stage,
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

        if self.locked_stage is not None:
            self.stage = self.locked_stage
            return

        if summary.stage != self.stage or self.stage == CurriculumStage.SLALOM:
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

    def lock_to_stage(self, stage: CurriculumStage) -> None:
        self.locked_stage = stage
        self.stage = stage

    @property
    def mastery(self) -> float:
        history = self._history[self.stage]
        if not history:
            return 0.0
        return float(sum(history) / len(history))
