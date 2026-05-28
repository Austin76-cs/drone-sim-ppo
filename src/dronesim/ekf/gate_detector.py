"""GateDetector: extract 2D gate center and apparent radius from a U-Net heatmap.

Input:  heatmap (H, W) float32 in [0, 1]  — output of GateUNet (squeezed)
Output: (u, v, radius_px) or None if no gate detected

Algorithm:
  1. Threshold heatmap at `threshold` to get binary blob
  2. Find the largest connected blob (gates appear as single bright regions)
  3. Compute weighted centroid (u, v) using heatmap values as weights
  4. Estimate radius from blob area: radius = sqrt(area / pi)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GateDetector:
    """Converts a U-Net heatmap to a 2D gate detection (pixel center + radius)."""

    def __init__(self, threshold: float = 0.3, min_area_px: int = 20) -> None:
        """
        Args:
            threshold: Heatmap activation threshold for binary mask.
            min_area_px: Minimum blob area in pixels — smaller blobs are noise.
        """
        self.threshold = threshold
        self.min_area_px = min_area_px

    def detect(
        self, heatmap: NDArray[np.float32]
    ) -> tuple[float, float, float] | None:
        """Detect gate in heatmap.

        Args:
            heatmap: (H, W) float32 heatmap from GateUNet.

        Returns:
            (u, v, radius_px) in pixel coordinates, or None if not detected.
            u = column (x), v = row (y).
        """
        mask = heatmap >= self.threshold
        if mask.sum() < self.min_area_px:
            return None

        # Use connected components to find the largest blob
        # Simple flood-fill via scipy if available, else use the whole mask
        try:
            from scipy.ndimage import label
            labeled, n_labels = label(mask)
            if n_labels == 0:
                return None
            # Pick largest blob
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignore background
            best_label = int(np.argmax(sizes))
            blob_mask = labeled == best_label
            if blob_mask.sum() < self.min_area_px:
                return None
        except ImportError:
            # Fallback: use entire mask (fine when only one gate is visible)
            blob_mask = mask

        # Weighted centroid using heatmap intensities within the blob
        weights = heatmap * blob_mask.astype(np.float32)
        total_weight = float(weights.sum())
        if total_weight < 1e-6:
            return None

        rows, cols = np.indices(heatmap.shape)
        u = float((weights * cols).sum() / total_weight)  # pixel x
        v = float((weights * rows).sum() / total_weight)  # pixel y

        # Radius from blob area: treat blob as a circle
        area = float(blob_mask.sum())
        radius_px = float(np.sqrt(area / np.pi))

        return u, v, radius_px
