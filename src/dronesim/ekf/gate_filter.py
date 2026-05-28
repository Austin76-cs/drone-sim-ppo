"""GateFilter: Kalman filter for tracking stationary gate positions in world frame.

Gates don't move, so the state model is trivial:
  state:      x = gate_pos (3,) in world frame
  transition: x_{t+1} = x_t  (constant position)
  measurement: z_t = x_t + noise  (noisy 3D estimate from GateEstimator)

One GateFilter instance per gate. The filter:
  - Maintains a position estimate + uncertainty (covariance)
  - Fuses noisy camera measurements when the gate is visible
  - Holds the last estimate when the gate is not visible (missed frame)
  - Resets when a new episode starts (new course layout)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GateFilter:
    """Kalman filter for a single stationary gate.

    Args:
        process_noise: How much uncertainty to add per step (3,) or scalar.
            Small value — gates don't move, so process noise is near zero.
        measurement_noise: Std dev of 3D position measurement (3,) or scalar.
            Larger = trust measurements less, smooth more.
        init_uncertainty: Initial position covariance diagonal (scalar).
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.5,
        init_uncertainty: float = 100.0,
    ) -> None:
        self.Q = np.eye(3) * process_noise        # process noise covariance
        self.R = np.eye(3) * measurement_noise    # measurement noise covariance

        self._x: NDArray[np.float64] | None = None  # state: gate position (3,)
        self._P: NDArray[np.float64] = np.eye(3) * init_uncertainty  # covariance
        self._init_uncertainty = init_uncertainty
        self._has_measurement = False

    @property
    def initialized(self) -> bool:
        """True once we've received at least one measurement."""
        return self._x is not None

    @property
    def position(self) -> NDArray[np.float64] | None:
        """Current best estimate of gate world position, or None if not yet seen."""
        return self._x.copy() if self._x is not None else None

    @property
    def uncertainty(self) -> float:
        """Scalar uncertainty — trace of position covariance."""
        return float(np.trace(self._P))

    def reset(self) -> None:
        """Call at the start of each new episode (new gate layout)."""
        self._x = None
        self._P = np.eye(3) * self._init_uncertainty
        self._has_measurement = False

    def predict(self) -> None:
        """Prediction step: add process noise (gate is stationary, so tiny)."""
        if self._x is not None:
            self._P = self._P + self.Q

    def update(self, measurement: NDArray[np.float64]) -> None:
        """Fuse a new 3D measurement.

        Args:
            measurement: Estimated gate position in world frame (3,).
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        if self._x is None:
            # First measurement — initialize state directly
            self._x = measurement.copy()
            self._P = self.R.copy()
            self._has_measurement = True
            return

        # Standard Kalman update: H = I (we measure position directly)
        S = self._P + self.R                          # innovation covariance
        K = self._P @ np.linalg.inv(S)               # Kalman gain
        innovation = measurement - self._x
        self._x = self._x + K @ innovation
        self._P = (np.eye(3) - K) @ self._P
        self._has_measurement = True

    def step(self, measurement: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
        """Combined predict + optional update. Returns current estimate.

        Args:
            measurement: 3D gate position from GateEstimator, or None if missed.

        Returns:
            Current gate position estimate in world frame, or None if never seen.
        """
        self.predict()
        if measurement is not None:
            self.update(measurement)
        return self.position


class MultiGateFilter:
    """Manages one GateFilter per gate in the current course.

    Usage:
        filter = MultiGateFilter(n_gates=7)
        filter.reset()  # call at start of each episode

        # Each policy step:
        estimates = filter.step(measurements)  # measurements is list of (pos or None)
    """

    def __init__(
        self,
        n_gates: int,
        process_noise: float = 0.001,
        measurement_noise: float = 0.5,
    ) -> None:
        self.n_gates = n_gates
        self._filters = [
            GateFilter(process_noise=process_noise, measurement_noise=measurement_noise)
            for _ in range(n_gates)
        ]

    def reset(self, n_gates: int | None = None) -> None:
        """Reset all filters. Optionally resize to a new gate count."""
        if n_gates is not None and n_gates != self.n_gates:
            self.n_gates = n_gates
            self._filters = [GateFilter() for _ in range(n_gates)]
        else:
            for f in self._filters:
                f.reset()

    def step(
        self, measurements: list[NDArray[np.float64] | None]
    ) -> list[NDArray[np.float64] | None]:
        """Update all gate filters with their respective measurements.

        Args:
            measurements: List of 3D positions (world frame) or None per gate.
                          Length must equal n_gates.

        Returns:
            List of current position estimates (or None if gate never seen).
        """
        assert len(measurements) == self.n_gates
        return [f.step(m) for f, m in zip(self._filters, measurements)]

    def get_estimates(self) -> list[NDArray[np.float64] | None]:
        """Return current position estimates without stepping."""
        return [f.position for f in self._filters]
