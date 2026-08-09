"""Reduced, interpretable autothrust controller primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpeedTrackingAutothrustController:
    """Command along-path acceleration from speed tracking error.

    The controller is intentionally above the FADEC/spool model.  A caller
    supplies physics-based trim thrust and condition-dependent thrust bounds;
    this class adds only the identified speed feedback and target-rate
    feedforward.  Gains are aircraft parameters and have no A320 defaults.
    """

    speed_error_gain_per_s: float
    target_rate_feedforward_gain: float = 0.0
    minimum_acceleration_mps2: float = -2.0
    maximum_acceleration_mps2: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.speed_error_gain_per_s,
            self.target_rate_feedforward_gain,
            self.minimum_acceleration_mps2,
            self.maximum_acceleration_mps2,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("autothrust controller parameters must be finite")
        if self.speed_error_gain_per_s <= 0.0:
            raise ValueError("speed_error_gain_per_s must be positive")
        if self.target_rate_feedforward_gain < 0.0:
            raise ValueError("target_rate_feedforward_gain must be non-negative")
        if self.minimum_acceleration_mps2 >= self.maximum_acceleration_mps2:
            raise ValueError("minimum acceleration must be below maximum acceleration")

    def acceleration_command_mps2(
        self,
        *,
        true_airspeed_mps: ArrayLike,
        target_true_airspeed_mps: ArrayLike,
        target_true_airspeed_rate_mps2: ArrayLike,
    ) -> FloatArray:
        """Return bounded along-path acceleration command."""
        speed, target, target_rate = np.broadcast_arrays(
            np.asarray(true_airspeed_mps, dtype=float),
            np.asarray(target_true_airspeed_mps, dtype=float),
            np.asarray(target_true_airspeed_rate_mps2, dtype=float),
        )
        if np.any(~np.isfinite(speed + target + target_rate)):
            raise ValueError("speed, target, and target rate must be finite")
        command = (
            self.speed_error_gain_per_s * (target - speed)
            + self.target_rate_feedforward_gain * target_rate
        )
        return np.clip(
            command,
            self.minimum_acceleration_mps2,
            self.maximum_acceleration_mps2,
        )

    def commanded_effective_thrust_n(
        self,
        *,
        trim_thrust_n: ArrayLike,
        mass_kg: ArrayLike,
        true_airspeed_mps: ArrayLike,
        target_true_airspeed_mps: ArrayLike,
        target_true_airspeed_rate_mps2: ArrayLike,
        minimum_thrust_n: ArrayLike,
        maximum_thrust_n: ArrayLike,
    ) -> FloatArray:
        """Add feedback to trim thrust and enforce the supplied FADEC limits."""
        trim, mass, minimum, maximum = np.broadcast_arrays(
            np.asarray(trim_thrust_n, dtype=float),
            np.asarray(mass_kg, dtype=float),
            np.asarray(minimum_thrust_n, dtype=float),
            np.asarray(maximum_thrust_n, dtype=float),
        )
        if np.any(~np.isfinite(trim + mass + minimum + maximum)):
            raise ValueError("trim, mass, and thrust bounds must be finite")
        if np.any(mass <= 0.0):
            raise ValueError("mass_kg must be positive")
        if np.any(minimum > maximum):
            raise ValueError("minimum_thrust_n must not exceed maximum_thrust_n")
        acceleration = self.acceleration_command_mps2(
            true_airspeed_mps=true_airspeed_mps,
            target_true_airspeed_mps=target_true_airspeed_mps,
            target_true_airspeed_rate_mps2=target_true_airspeed_rate_mps2,
        )
        return np.clip(trim + mass * acceleration, minimum, maximum)

    def simulate_speed_response_zoh(
        self,
        *,
        time_s: ArrayLike,
        target_true_airspeed_mps: ArrayLike,
        target_true_airspeed_rate_mps2: ArrayLike,
        initial_true_airspeed_mps: float,
    ) -> FloatArray:
        """Roll out the identified closed-loop speed response exactly per step.

        Target and target rate are zero-order held on each sample interval.  The
        acceleration limits are intentionally omitted from this diagnostic
        response: candidate calibration must remain in the unsaturated regime.
        """
        time = np.asarray(time_s, dtype=float)
        target = np.asarray(target_true_airspeed_mps, dtype=float)
        target_rate = np.asarray(target_true_airspeed_rate_mps2, dtype=float)
        if (
            time.ndim != 1
            or target.shape != time.shape
            or target_rate.shape != time.shape
        ):
            raise ValueError(
                "time, target, and target rate must be equal-length vectors"
            )
        if len(time) == 0 or np.any(~np.isfinite(time + target + target_rate)):
            raise ValueError("speed rollout inputs must be non-empty and finite")
        if len(time) > 1 and np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if (
            not np.isfinite(initial_true_airspeed_mps)
            or initial_true_airspeed_mps < 0.0
        ):
            raise ValueError(
                "initial_true_airspeed_mps must be finite and non-negative"
            )

        speed = np.empty(len(time), dtype=float)
        speed[0] = initial_true_airspeed_mps
        gain = self.speed_error_gain_per_s
        for index, interval in enumerate(np.diff(time)):
            equilibrium = (
                target[index]
                + self.target_rate_feedforward_gain * target_rate[index] / gain
            )
            speed[index + 1] = equilibrium + (speed[index] - equilibrium) * np.exp(
                -gain * interval
            )
        return speed
