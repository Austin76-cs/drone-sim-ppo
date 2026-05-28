"""Q1 competition runner: MAVLink telemetry -> PPO policy -> SET_ATTITUDE_TARGET.

Main control loop:
  1. Poll MAVLink telemetry (ATTITUDE, ODOMETRY, IMU)
  2. (Optional) Process camera frame through vision pipeline for gate detection
  3. Build 34-dim observation (pos, vel, rot, omega, prev_action, gate vectors)
  4. Normalize observation (VecNormalize stats from training)
  5. Run PPO policy inference
  6. Map action to SET_ATTITUDE_TARGET body-rate mode
  7. Send command via MAVLink
  8. Maintain heartbeat

Gate positions can come from:
  - Vision pipeline (U-Net + GateEstimator + GateFilter) — default for competition
  - External JSON file (for testing with known gate layouts)
  - Zeros (fallback — policy flies blind)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from dronesim.bridge.mavlink_client import MAVLinkClient
from dronesim.bridge.obs_builder import ObsNormalizer, ObservationBuilder
from dronesim.sim.env import euler_to_rotation_matrix


class Q1Runner:
    """Runs the trained PPO policy against the DCL Q1 simulator via MAVLink."""

    def __init__(
        self,
        model_path: str | Path,
        normalize_path: str | Path | None = None,
        connection_string: str = "udpin:0.0.0.0:14550",
        command_hz: float = 50.0,
        max_body_rate: float = 15.0,
        deterministic: bool = True,
        unet_path: str | Path | None = None,
        n_gates: int = 10,
        gate_radius_m: float = 0.50,
        cam_fov_deg: float = 90.0,
        gate_pass_distance_m: float = 1.0,
    ) -> None:
        """
        Args:
            model_path: Path to trained PPO .zip checkpoint.
            normalize_path: Path to VecNormalize .pkl (observation stats).
            connection_string: MAVLink UDP connection string.
            command_hz: Target command rate (DCL recommends 50-120 Hz).
            max_body_rate: Max body rate in rad/s (must match training config).
            deterministic: Use deterministic policy (no action noise).
            unet_path: Path to U-Net .pt checkpoint for vision-based gate detection.
                        If None, vision pipeline is disabled.
            n_gates: Expected number of gates in the course.
            gate_radius_m: Physical gate radius in meters (for depth estimation).
            cam_fov_deg: Camera vertical FOV in degrees.
            gate_pass_distance_m: Distance threshold to consider a gate passed.
        """
        self.command_hz = command_hz
        self.command_dt = 1.0 / command_hz
        self.max_body_rate = max_body_rate
        self.deterministic = deterministic
        self.gate_pass_distance_m = gate_pass_distance_m

        # Load policy
        print(f"Loading policy from {model_path}")
        self.model = PPO.load(str(model_path), device="cpu")

        # Load observation normalizer
        self.normalizer: ObsNormalizer | None = None
        if normalize_path is not None:
            print(f"Loading observation stats from {normalize_path}")
            self.normalizer = ObsNormalizer.from_pkl(normalize_path)

        # MAVLink client
        self.client = MAVLinkClient(connection_string=connection_string)

        # Observation builder
        self.obs_builder = ObservationBuilder()

        # Vision pipeline (optional)
        self.vision = None
        if unet_path is not None:
            from dronesim.bridge.vision_pipeline import VisionPipeline
            print(f"Loading vision pipeline from {unet_path}")
            self.vision = VisionPipeline(
                unet_path=unet_path,
                n_gates=n_gates,
                gate_radius_m=gate_radius_m,
                cam_fov_deg=cam_fov_deg,
                device="cpu",
            )

        # Gate tracking state
        self.gate_positions: list[NDArray[np.float64] | None] = []
        self.gate_index = 0
        self._prev_gate_dist = float("inf")
        self._passed_close = False
        self._frame_callback = None

        # Stats
        self._step_count = 0
        self._start_time = 0.0
        self._gates_cleared = 0

    def set_gate_positions(self, positions: list[NDArray[np.float64]]) -> None:
        """Set known gate positions (world frame).

        Call this if gate positions are known ahead of time (e.g., from JSON).
        """
        self.gate_positions = list(positions)

    def set_frame_callback(self, callback) -> None:
        """Set a callback that provides camera frames: () -> NDArray[uint8] or None.

        This is how the vision pipeline gets camera frames. The callback should
        return the latest (H, W, 3) uint8 RGB frame, or None if no frame is available.
        """
        self._frame_callback = callback

    def _policy_action_to_command(
        self, action: NDArray[np.float64]
    ) -> tuple[float, float, float, float]:
        """Convert PPO action [-1,1]^4 to SET_ATTITUDE_TARGET parameters.

        Returns:
            (thrust, roll_rate, pitch_rate, yaw_rate) where thrust is [0,1]
            and rates are in rad/s.
        """
        action = np.clip(action, -1.0, 1.0)
        thrust = (float(action[0]) + 1.0) * 0.5  # [-1,1] -> [0,1]
        roll_rate = float(action[1]) * self.max_body_rate
        pitch_rate = float(action[2]) * self.max_body_rate
        yaw_rate = float(action[3]) * self.max_body_rate
        return thrust, roll_rate, pitch_rate, yaw_rate

    def _check_gate_passage(self) -> None:
        """Advance gate_index when the drone passes close to the current gate."""
        if self.gate_index >= len(self.gate_positions):
            return
        gate_pos = self.gate_positions[self.gate_index]
        if gate_pos is None:
            return

        dist = float(np.linalg.norm(self.client.telemetry.pos - gate_pos))

        if dist < self.gate_pass_distance_m:
            self._passed_close = True

        if self._passed_close and dist > self._prev_gate_dist and dist > self.gate_pass_distance_m * 0.5:
            # Distance increasing after being close — gate passed
            self._gates_cleared += 1
            self.gate_index += 1
            self._passed_close = False
            self._prev_gate_dist = float("inf")
            if self.vision is not None:
                self.vision.current_gate_index = self.gate_index
            print(f"  ** Gate {self._gates_cleared} passed! "
                  f"(target now: {self.gate_index}/{len(self.gate_positions)})")
            return

        self._prev_gate_dist = dist

    def _process_vision(self) -> None:
        """Run vision pipeline on latest camera frame and update gate positions."""
        if self.vision is None or self._frame_callback is None:
            return
        frame = self._frame_callback()
        if frame is None:
            return

        tel = self.client.telemetry
        euler = np.array([tel.roll, tel.pitch, tel.yaw], dtype=np.float64)
        rot_matrix = euler_to_rotation_matrix(euler)

        self.vision.process_frame(frame, tel.pos, rot_matrix)
        estimates = self.vision.get_upcoming_gate_positions(0, len(self.gate_positions) or 10)

        # Update gate_positions with any new estimates from vision
        while len(self.gate_positions) < len(estimates):
            self.gate_positions.append(None)
        for i, est in enumerate(estimates):
            if est is not None:
                self.gate_positions[i] = est

    def _get_observation(self) -> NDArray[np.float32]:
        """Build and optionally normalize the observation vector."""
        obs = self.obs_builder.build(
            self.client.telemetry,
            self.gate_positions,
            self.gate_index,
        )
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        return obs

    def run(self, timeout_s: float = 480.0) -> None:
        """Main competition loop. Runs until timeout or keyboard interrupt.

        Args:
            timeout_s: Maximum run duration in seconds (Q1 limit: 480s = 8 min).
        """
        if not self.client.connect():
            return

        mode = "VISION" if self.vision is not None else (
            "KNOWN GATES" if self.gate_positions else "BLIND"
        )
        print(f"Starting Q1 run — command rate: {self.command_hz} Hz, "
              f"timeout: {timeout_s}s, mode: {mode}")
        print(f"Gate positions loaded: {len(self.gate_positions)}")
        print("Waiting for first telemetry...")

        # Wait for initial telemetry
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            self.client.send_heartbeat()
            self.client.poll_telemetry()
            if self.client.telemetry.has_attitude:
                break
            time.sleep(0.01)

        if not self.client.telemetry.has_attitude:
            print("ERROR: No telemetry received after 10s. Aborting.")
            self.client.close()
            return

        has_odom = self.client.telemetry.has_odometry
        print(f"Telemetry active — ODOMETRY: {'YES' if has_odom else 'NO'}")
        if not has_odom:
            print("WARNING: No ODOMETRY — position/velocity will be zero. "
                  "Policy may not perform well without position data.")

        self.obs_builder.reset()
        self._step_count = 0
        self._start_time = time.monotonic()

        try:
            self._control_loop(timeout_s)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            elapsed = time.monotonic() - self._start_time
            print(f"\nRun complete — {self._step_count} steps in {elapsed:.1f}s "
                  f"({self._step_count / max(elapsed, 0.001):.1f} Hz avg)")
            self.client.close()

    def _control_loop(self, timeout_s: float) -> None:
        """Inner control loop running at command_hz."""
        end_time = time.monotonic() + timeout_s
        next_cmd_time = time.monotonic()

        while time.monotonic() < end_time:
            now = time.monotonic()

            # Rate limiting
            if now < next_cmd_time:
                time.sleep(max(0, next_cmd_time - now - 0.0005))
                continue
            next_cmd_time += self.command_dt

            # 1. Heartbeat
            self.client.send_heartbeat()

            # 2. Poll telemetry
            self.client.poll_telemetry()

            # 3. Vision pipeline (update gate positions from camera)
            self._process_vision()

            # 4. Check gate passage
            self._check_gate_passage()

            # 5. Build observation
            obs = self._get_observation()

            # 6. Policy inference
            action, _ = self.model.predict(
                obs.reshape(1, -1),
                deterministic=self.deterministic,
            )
            action = action.flatten().astype(np.float64)

            # 7. Convert and send command
            thrust, roll_rate, pitch_rate, yaw_rate = self._policy_action_to_command(action)
            self.client.send_attitude_target(thrust, roll_rate, pitch_rate, yaw_rate)

            # 8. Update state
            self.obs_builder.update_prev_action(action)
            self._step_count += 1

            # Periodic status
            if self._step_count % int(self.command_hz * 10) == 0:
                elapsed = now - self._start_time
                pos = self.client.telemetry.pos
                print(f"  [{elapsed:6.1f}s] step={self._step_count:6d}  "
                      f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
                      f"gate={self.gate_index}/{len(self.gate_positions)}  "
                      f"cleared={self._gates_cleared}  thrust={thrust:.2f}")
