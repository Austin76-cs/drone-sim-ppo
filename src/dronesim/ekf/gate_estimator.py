"""GateEstimator: convert 2D gate detection to 3D position in world frame.

Uses the pinhole camera model:
  - Direction ray from (u, v): d = [(u-cx)/fx, (v-cy)/fy, 1] (normalized)
  - Depth from apparent size:  depth = fx * gate_radius_m / pixel_radius

Then transforms from camera frame -> world frame using the drone's current pose.

This gives a noisy 3D gate position that the GateFilter will smooth over time.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GateEstimator:
    """Converts 2D gate detections to 3D world-frame positions.

    Args:
        fx, fy, cx, cy: Camera intrinsic parameters (pixels).
        gate_radius_m: Known physical gate radius in meters.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        gate_radius_m: float = 0.40,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.gate_radius_m = gate_radius_m

    def estimate_world(
        self,
        u: float,
        v: float,
        radius_px: float,
        cam_pos: NDArray[np.float64],
        cam_xmat: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Estimate 3D gate center in world frame from a 2D detection.

        Args:
            u, v: Gate center in pixel coordinates (u=col, v=row).
            radius_px: Apparent gate radius in pixels.
            cam_pos: Camera position in world frame (3,).
            cam_xmat: Camera rotation matrix (3x3), columns are camera X/Y/Z axes
                      in world frame (MuJoCo convention: cam_xmat from get_camera_extrinsics).

        Returns:
            gate_world (3,) or None if radius_px is too small to be reliable.
        """
        if radius_px < 2.0:
            return None

        # Depth estimate from apparent size: Z = fx * R / r_px
        depth = self.fx * self.gate_radius_m / max(radius_px, 1e-3)

        # Ray in camera frame: camera looks along -Z in MuJoCo (OpenGL convention)
        # cam_xmat columns: [right, up, -forward] in world frame
        # So camera -Z axis = cam_xmat[:, 2] points forward from the drone's perspective
        # Pixel to camera-frame normalized coords (OpenCV convention: Z forward)
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        # MuJoCo camera frame: x=right, y=up, z=-forward (OpenGL)
        # Convert to point in camera frame at given depth
        point_cam = np.array([x_cam * depth, -y_cam * depth, -depth], dtype=np.float64)

        # Transform to world frame: p_world = cam_pos + cam_xmat @ point_cam
        gate_world = cam_pos + cam_xmat @ point_cam
        return gate_world

    def estimate_body(
        self,
        u: float,
        v: float,
        radius_px: float,
        cam_pos: NDArray[np.float64],
        cam_xmat: NDArray[np.float64],
        drone_pos: NDArray[np.float64],
        drone_rot: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """Estimate gate position in drone body frame.

        Args:
            drone_pos: Drone position in world frame (3,).
            drone_rot: Drone rotation matrix body->world (3x3).

        Returns:
            gate_body (3,) or None if detection is not reliable.
        """
        gate_world = self.estimate_world(u, v, radius_px, cam_pos, cam_xmat)
        if gate_world is None:
            return None
        rel_world = gate_world - drone_pos
        gate_body = drone_rot.T @ rel_world
        return gate_body
