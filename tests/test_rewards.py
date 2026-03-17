from __future__ import annotations

import math
import unittest

import numpy as np

from dronesim.config import RewardConfig
from dronesim.tasks.rewards import (
    body_frame_gate,
    compute_total_reward,
    control_effort_reward,
    gate_passed,
    gate_proximity_reward,
    gate_relative_geometry,
    lateral_velocity_penalty_reward,
    progress_reward,
    velocity_alignment_reward,
)
from dronesim.types import DroneState, GateSpec


def _make_state(pos=(0, 0, 1), vel=(0, 0, 0), euler=(0, 0, 0), omega=(0, 0, 0)):
    return DroneState(
        pos=np.array(pos, dtype=np.float64),
        vel=np.array(vel, dtype=np.float64),
        euler=np.array(euler, dtype=np.float64),
        omega=np.array(omega, dtype=np.float64),
        motor=np.zeros(4, dtype=np.float64),
    )


def _make_gate(center=(3, 0, 1)):
    return GateSpec(
        center=np.array(center, dtype=np.float64),
        normal=np.array([1, 0, 0], dtype=np.float64),
        radius_m=0.45,
        depth_m=0.15,
    )


class TestGateGeometry(unittest.TestCase):
    def test_forward_error(self):
        state = _make_state(pos=(1, 0, 1))
        gate = _make_gate(center=(3, 0, 1))
        fwd, lat, align = gate_relative_geometry(state, gate)
        self.assertAlmostEqual(fwd, 2.0, places=3)
        self.assertAlmostEqual(lat, 0.0, places=3)

    def test_lateral_error(self):
        state = _make_state(pos=(3, 0.5, 1))
        gate = _make_gate(center=(3, 0, 1))
        fwd, lat, _ = gate_relative_geometry(state, gate)
        self.assertAlmostEqual(fwd, 0.0, places=3)
        self.assertAlmostEqual(lat, 0.5, places=3)

    def test_gate_passed(self):
        state = _make_state(pos=(3.05, 0.1, 1))
        gate = _make_gate(center=(3, 0, 1))
        self.assertTrue(gate_passed(state, gate, 0.12, prev_forward_error=0.5))

    def test_gate_not_passed(self):
        state = _make_state(pos=(1, 0, 1))
        gate = _make_gate(center=(3, 0, 1))
        self.assertFalse(gate_passed(state, gate, 0.12))

    def test_gate_not_passed_without_crossing_event(self):
        state = _make_state(pos=(3.4, 0.1, 1))
        gate = _make_gate(center=(3, 0, 1))
        self.assertFalse(gate_passed(state, gate, 0.12, prev_forward_error=-0.2))


class TestRewardComponents(unittest.TestCase):
    def test_proximity_at_gate(self):
        state = _make_state(pos=(3, 0, 1))
        gate = _make_gate(center=(3, 0, 1))
        r = gate_proximity_reward(state, gate, 0.45)
        self.assertAlmostEqual(r, 1.0, places=2)

    def test_proximity_far(self):
        state = _make_state(pos=(3, 5, 1))
        gate = _make_gate(center=(3, 0, 1))
        r = gate_proximity_reward(state, gate, 0.45)
        self.assertLess(r, 0.01)

    def test_progress_positive(self):
        r = progress_reward(5.0, 4.0, 0.45)
        self.assertGreater(r, 0.0)

    def test_progress_zero_backward(self):
        r = progress_reward(4.0, 5.0, 0.45)
        self.assertEqual(r, 0.0)

    def test_lateral_velocity_penalty_positive(self):
        state = _make_state(pos=(2, 0, 1), vel=(0, 1.5, 0))
        gate = _make_gate(center=(3, 0, 1))
        r = lateral_velocity_penalty_reward(state, gate, 1.2)
        self.assertGreater(r, 0.0)

    def test_control_effort(self):
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        r = control_effort_reward(action)
        self.assertGreater(r, 0.0)

    def test_control_effort_zero(self):
        action = np.zeros(4, dtype=np.float64)
        r = control_effort_reward(action)
        self.assertAlmostEqual(r, 0.0)

    def test_total_reward_not_nan(self):
        state = _make_state(pos=(2, 0.1, 1))
        gate = _make_gate(center=(3, 0, 1))
        action = np.array([0.1, 0.0, -0.1, 0.0], dtype=np.float64)
        cfg = RewardConfig()
        info = compute_total_reward(state, action, gate, 2.0, cfg, False, 0.12)
        self.assertFalse(math.isnan(info.total))

    def test_gate_passage_reward_requires_crossing(self):
        state = _make_state(pos=(3.4, 0.1, 1))
        gate = _make_gate(center=(3, 0, 1))
        action = np.zeros(4, dtype=np.float64)
        cfg = RewardConfig()
        info = compute_total_reward(state, action, gate, -0.2, cfg, False, 0.12)
        self.assertEqual(info.gate_passage, 0.0)

    def test_instability_components_are_positive_when_tilted_and_spinning(self):
        state = _make_state(pos=(2, 0.1, 1), euler=(0.4, -0.3, 0.0), omega=(3.0, 2.0, 1.0))
        gate = _make_gate(center=(3, 0, 1))
        action = np.zeros(4, dtype=np.float64)
        cfg = RewardConfig(attitude_stability=-0.25, angular_rate_stability=-0.20)
        info = compute_total_reward(state, action, gate, 2.0, cfg, False, 0.12)
        self.assertGreater(info.attitude_stability, 0.0)
        self.assertGreater(info.angular_rate_stability, 0.0)


class TestBodyFrameGate(unittest.TestCase):
    def test_ahead(self):
        state = _make_state(pos=(1, 0, 1), euler=(0, 0, 0))
        gate = _make_gate(center=(3, 0, 1))
        rel = body_frame_gate(state, gate)
        self.assertGreater(rel[0], 0)  # gate is in front


if __name__ == "__main__":
    unittest.main()
