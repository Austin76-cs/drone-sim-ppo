"""MAVLink UDP client for DCL Q1 simulator communication.

Handles connection, heartbeat, telemetry reception, and command sending.
Uses pymavlink for low-level MAVLink protocol handling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pymavlink import mavutil


@dataclass
class TelemetryState:
    """Latest telemetry snapshot from the DCL simulator."""

    # ATTITUDE message
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # ODOMETRY message
    pos: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    vel: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    odom_quat: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

    # HIGHRES_IMU message
    accel: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    gyro: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    # Timestamps (monotonic) — 0 means never received
    attitude_time: float = 0.0
    odometry_time: float = 0.0
    imu_time: float = 0.0

    @property
    def has_odometry(self) -> bool:
        return self.odometry_time > 0.0

    @property
    def has_attitude(self) -> bool:
        return self.attitude_time > 0.0


class MAVLinkClient:
    """Manages MAVLink UDP connection to the DCL Q1 simulator."""

    # SET_ATTITUDE_TARGET type_mask:
    # Bit 7 (128) = ignore attitude quaternion, use body rates instead
    _BODY_RATE_TYPEMASK = 128

    def __init__(
        self,
        connection_string: str = "udpin:0.0.0.0:14550",
        source_system: int = 1,
        source_component: int = 1,
    ) -> None:
        self.connection_string = connection_string
        self.source_system = source_system
        self.source_component = source_component
        self.conn: mavutil.mavfile | None = None
        self.telemetry = TelemetryState()
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 0.4  # ~2.5 Hz (spec minimum: 2 Hz)

    def connect(self, timeout: float = 30.0) -> bool:
        """Establish MAVLink connection and wait for simulator heartbeat."""
        print(f"Connecting to {self.connection_string} ...")
        self.conn = mavutil.mavlink_connection(
            self.connection_string,
            source_system=self.source_system,
            source_component=self.source_component,
        )
        print("Waiting for simulator heartbeat...")
        msg = self.conn.wait_heartbeat(timeout=timeout)
        if msg is None:
            print("ERROR: No heartbeat received. Is the DCL simulator running?")
            return False
        print(
            f"Connected — target system={self.conn.target_system}, "
            f"component={self.conn.target_component}"
        )
        return True

    def send_heartbeat(self) -> None:
        """Send heartbeat if interval has elapsed. Call every loop iteration."""
        assert self.conn is not None
        now = time.monotonic()
        if now - self._last_heartbeat >= self._heartbeat_interval:
            self.conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0,
            )
            self._last_heartbeat = now

    def poll_telemetry(self) -> None:
        """Drain all pending MAVLink messages and update telemetry state."""
        assert self.conn is not None
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            now = time.monotonic()

            if mtype == "ATTITUDE":
                self.telemetry.roll = msg.roll
                self.telemetry.pitch = msg.pitch
                self.telemetry.yaw = msg.yaw
                self.telemetry.rollspeed = msg.rollspeed
                self.telemetry.pitchspeed = msg.pitchspeed
                self.telemetry.yawspeed = msg.yawspeed
                self.telemetry.attitude_time = now

            elif mtype == "ODOMETRY":
                self.telemetry.pos[:] = [msg.x, msg.y, msg.z]
                self.telemetry.vel[:] = [msg.vx, msg.vy, msg.vz]
                self.telemetry.odom_quat[:] = msg.q
                self.telemetry.rollspeed = msg.rollspeed
                self.telemetry.pitchspeed = msg.pitchspeed
                self.telemetry.yawspeed = msg.yawspeed
                self.telemetry.odometry_time = now

            elif mtype == "HIGHRES_IMU":
                self.telemetry.accel[:] = [msg.xacc, msg.yacc, msg.zacc]
                self.telemetry.gyro[:] = [msg.xgyro, msg.ygyro, msg.zgyro]
                self.telemetry.imu_time = now

    def send_attitude_target(
        self,
        thrust: float,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
    ) -> None:
        """Send SET_ATTITUDE_TARGET in body-rate mode.

        Args:
            thrust: Normalized thrust in [0, 1].
            roll_rate: Desired body roll rate (rad/s).
            pitch_rate: Desired body pitch rate (rad/s).
            yaw_rate: Desired body yaw rate (rad/s).
        """
        assert self.conn is not None
        self.conn.mav.set_attitude_target_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self.conn.target_system,
            self.conn.target_component,
            self._BODY_RATE_TYPEMASK,
            [1.0, 0.0, 0.0, 0.0],  # quaternion (ignored with typemask=128)
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(np.clip(thrust, 0.0, 1.0)),
        )

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
