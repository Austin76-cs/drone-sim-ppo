"""Phase 3: EKF-based gate state estimation.

Pipeline:
  U-Net heatmap -> GateDetector (2D center + radius in pixels)
               -> GateEstimator (3D position in world frame via pinhole model)
               -> GateFilter (Kalman filter per gate, handles missed frames)
               -> body-frame gate vectors for PPO policy
"""
from dronesim.ekf.gate_detector import GateDetector
from dronesim.ekf.gate_estimator import GateEstimator
from dronesim.ekf.gate_filter import GateFilter

__all__ = ["GateDetector", "GateEstimator", "GateFilter"]
