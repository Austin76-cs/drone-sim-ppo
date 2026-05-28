"""Vision pipeline: camera frame -> gate world positions.

Wires together:
  1. GateUNet      — RGB image -> heatmap
  2. GateDetector  — heatmap -> (u, v, radius_px)
  3. GateEstimator — 2D detection + camera params -> 3D gate position
  4. GateFilter    — Kalman smoothing for stable estimates

The pipeline detects the nearest visible gate and feeds measurements
to the Kalman filter for each tracked gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from dronesim.ekf.gate_detector import GateDetector
from dronesim.ekf.gate_estimator import GateEstimator
from dronesim.ekf.gate_filter import MultiGateFilter


class VisionPipeline:
    """End-to-end: camera frame -> smoothed 3D gate positions."""

    def __init__(
        self,
        unet_path: str | Path,
        n_gates: int = 10,
        gate_radius_m: float = 0.50,
        cam_fov_deg: float = 90.0,
        cam_width: int = 160,
        cam_height: int = 120,
        device: str = "cpu",
        detection_threshold: float = 0.3,
        min_blob_area: int = 20,
        filter_process_noise: float = 0.001,
        filter_measurement_noise: float = 0.5,
    ) -> None:
        from dronesim.perception.unet import GateUNet

        self.device = device
        self.cam_width = cam_width
        self.cam_height = cam_height

        # U-Net
        self.unet = GateUNet.from_checkpoint(str(unet_path), device=device)

        # Gate detector (heatmap -> 2D)
        self.detector = GateDetector(
            threshold=detection_threshold, min_area_px=min_blob_area
        )

        # Gate estimator (2D -> 3D) — compute intrinsics from FOV
        import math
        fovy_rad = math.radians(cam_fov_deg)
        fy = (cam_height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy  # square pixels
        cx, cy = cam_width / 2.0, cam_height / 2.0
        self.estimator = GateEstimator(
            fx=fx, fy=fy, cx=cx, cy=cy, gate_radius_m=gate_radius_m
        )

        # Multi-gate Kalman filter
        self.gate_filter = MultiGateFilter(
            n_gates=n_gates,
            process_noise=filter_process_noise,
            measurement_noise=filter_measurement_noise,
        )

        self._current_gate_idx = 0

    def reset(self, n_gates: int | None = None) -> None:
        """Reset filters for new episode/run."""
        self.gate_filter.reset(n_gates)
        self._current_gate_idx = 0

    @property
    def current_gate_index(self) -> int:
        return self._current_gate_idx

    @current_gate_index.setter
    def current_gate_index(self, value: int) -> None:
        self._current_gate_idx = value

    def _preprocess_frame(self, frame_rgb: NDArray[np.uint8]) -> torch.Tensor:
        """Convert (H, W, 3) uint8 RGB to (1, 3, H, W) float tensor."""
        # Resize if needed
        h, w = frame_rgb.shape[:2]
        if h != self.cam_height or w != self.cam_width:
            from PIL import Image
            img = Image.fromarray(frame_rgb).resize(
                (self.cam_width, self.cam_height), Image.BILINEAR
            )
            frame_rgb = np.array(img)

        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def process_frame(
        self,
        frame_rgb: NDArray[np.uint8],
        drone_pos: NDArray[np.float64],
        drone_rot: NDArray[np.float64],
    ) -> list[NDArray[np.float64] | None]:
        """Process a camera frame and return updated gate position estimates.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB camera image.
            drone_pos: (3,) drone position in world frame.
            drone_rot: (3, 3) drone rotation matrix (body -> world).

        Returns:
            List of gate position estimates in world frame (one per gate).
            None entries mean the gate has never been seen.
        """
        # 1. U-Net inference -> heatmap
        inp = self._preprocess_frame(frame_rgb)
        heatmap = self.unet(inp).squeeze().cpu().numpy()  # (H, W)

        # 2. Detect gate in heatmap
        detection = self.detector.detect(heatmap)

        # 3. Convert 2D detection to 3D world position
        measurement: NDArray[np.float64] | None = None
        if detection is not None:
            u, v, radius_px = detection
            # Approximate camera extrinsics: camera is at drone pos, looking along
            # drone's forward axis (first column of rot_matrix for our convention)
            # Build camera rotation matrix: cam_xmat columns = [right, up, -forward]
            # Our drone rot_matrix: columns are body x, y, z in world frame
            # Assume camera is aligned with body: forward = rot[:, 0], up = -rot[:, 2]
            cam_pos = drone_pos
            cam_xmat = drone_rot  # approximate — will need calibration for DCL cam

            gate_world = self.estimator.estimate_world(
                u, v, radius_px, cam_pos, cam_xmat
            )
            if gate_world is not None:
                measurement = gate_world

        # 4. Feed measurement to the current gate's Kalman filter
        # Build measurement list: only the current gate gets the detection
        measurements: list[NDArray[np.float64] | None] = [
            None
        ] * self.gate_filter.n_gates
        if measurement is not None and self._current_gate_idx < self.gate_filter.n_gates:
            measurements[self._current_gate_idx] = measurement

        # 5. Step all filters and return estimates
        return self.gate_filter.step(measurements)

    def get_upcoming_gate_positions(
        self, gate_index: int, count: int = 4
    ) -> list[NDArray[np.float64] | None]:
        """Get position estimates for the next `count` gates starting at gate_index."""
        estimates = self.gate_filter.get_estimates()
        result: list[NDArray[np.float64] | None] = []
        for i in range(count):
            idx = gate_index + i
            if idx < len(estimates):
                result.append(estimates[idx])
            else:
                result.append(None)
        return result
