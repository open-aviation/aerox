"""Time-indexed trajectory references for guidance and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class TrajectoryReference:
    """Node-aligned state reference and analytic first derivatives."""

    time_s: FloatArray
    true_airspeed_mps: FloatArray
    true_airspeed_rate_mps2: FloatArray
    flight_path_angle_rad: FloatArray
    flight_path_angle_rate_rad_s: FloatArray
    heading_rad: FloatArray
    heading_rate_rad_s: FloatArray
    pressure_altitude_m: FloatArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        arrays = {
            "true_airspeed_mps": self.true_airspeed_mps,
            "true_airspeed_rate_mps2": self.true_airspeed_rate_mps2,
            "flight_path_angle_rad": self.flight_path_angle_rad,
            "flight_path_angle_rate_rad_s": self.flight_path_angle_rate_rad_s,
            "heading_rad": self.heading_rad,
            "heading_rate_rad_s": self.heading_rate_rad_s,
            "pressure_altitude_m": self.pressure_altitude_m,
        }
        if (
            time.ndim != 1
            or len(time) < 2
            or np.any(~np.isfinite(time))
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError("time_s must be a finite increasing vector")
        object.__setattr__(self, "time_s", time)
        for name, values in arrays.items():
            array = np.asarray(values, dtype=float)
            if array.shape != time.shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector matching time_s")
            object.__setattr__(self, name, array)
        if np.any(self.true_airspeed_mps <= 1.0):
            raise ValueError("reference true airspeed must exceed 1 m/s")


__all__ = ["TrajectoryReference"]

