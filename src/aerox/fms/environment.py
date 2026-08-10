"""Sparse spatial environment forecasts for online FMS guidance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .route import FloatArray


@dataclass(frozen=True)
class EnvironmentSample:
    """Atmospheric temperature and local wind at one route coordinate."""

    static_temperature_k: float
    wind_north_mps: float
    wind_east_mps: float
    wind_down_mps: float


@dataclass(frozen=True)
class SpatialEnvironmentProfile:
    """Linearly interpolated environmental forecast along route distance."""

    along_path_m: FloatArray
    static_temperature_k: FloatArray
    wind_north_mps: FloatArray
    wind_east_mps: FloatArray
    wind_down_mps: FloatArray

    def __post_init__(self) -> None:
        distance = np.asarray(self.along_path_m, dtype=float)
        arrays = {
            "static_temperature_k": self.static_temperature_k,
            "wind_north_mps": self.wind_north_mps,
            "wind_east_mps": self.wind_east_mps,
            "wind_down_mps": self.wind_down_mps,
        }
        if (
            distance.ndim != 1
            or len(distance) < 2
            or np.any(~np.isfinite(distance))
            or np.any(np.diff(distance) <= 0.0)
        ):
            raise ValueError("environment along-path coordinates must increase")
        object.__setattr__(self, "along_path_m", distance)
        for name, values in arrays.items():
            array = np.asarray(values, dtype=float)
            if array.shape != distance.shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be finite and match along_path_m")
            object.__setattr__(self, name, array)
        if np.any(self.static_temperature_k <= 0.0):
            raise ValueError("environment temperature must be positive")

    def sample(self, along_path_m: float) -> EnvironmentSample:
        if not np.isfinite(along_path_m):
            raise ValueError("environment sample distance must be finite")
        distance = float(
            np.clip(along_path_m, self.along_path_m[0], self.along_path_m[-1])
        )
        return EnvironmentSample(
            static_temperature_k=float(
                np.interp(distance, self.along_path_m, self.static_temperature_k)
            ),
            wind_north_mps=float(
                np.interp(distance, self.along_path_m, self.wind_north_mps)
            ),
            wind_east_mps=float(
                np.interp(distance, self.along_path_m, self.wind_east_mps)
            ),
            wind_down_mps=float(
                np.interp(distance, self.along_path_m, self.wind_down_mps)
            ),
        )

    def sample_arrays(
        self, along_path_m: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Vectorize interpolation of temperature and all wind components."""

        distance = np.asarray(along_path_m, dtype=float)
        if distance.ndim != 1 or np.any(~np.isfinite(distance)):
            raise ValueError("environment sample coordinates must be a finite vector")
        clipped = np.clip(distance, self.along_path_m[0], self.along_path_m[-1])
        return (
            np.interp(clipped, self.along_path_m, self.static_temperature_k),
            np.interp(clipped, self.along_path_m, self.wind_north_mps),
            np.interp(clipped, self.along_path_m, self.wind_east_mps),
            np.interp(clipped, self.along_path_m, self.wind_down_mps),
        )
