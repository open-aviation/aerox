"""Time-ordered selected-altitude and selected-speed command intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox import aero
from aerox.models.atmosphere import (
    true_airspeed_from_calibrated_airspeed,
    true_airspeed_from_mach,
)

FloatArray = NDArray[np.float64]
SelectedSpeedMode = Literal["CAS", "MACH"]


def _validated_time(values: ArrayLike) -> FloatArray:
    time = np.asarray(values, dtype=float)
    if (
        time.ndim != 1
        or len(time) == 0
        or np.any(~np.isfinite(time))
        or np.any(np.diff(time) <= 0.0)
        or time[0] < 0.0
    ):
        raise ValueError(
            "effective command times must be finite and strictly increasing"
        )
    return time


def _command_index(time: FloatArray, elapsed_time_s: float) -> int:
    if not np.isfinite(elapsed_time_s) or elapsed_time_s < 0.0:
        raise ValueError("elapsed_time_s must be finite and non-negative")
    return max(0, int(np.searchsorted(time, elapsed_time_s, side="right") - 1))


@dataclass(frozen=True)
class SelectedAltitudeIntent:
    """Sample-and-hold selected pressure-altitude commands."""

    effective_time_s: FloatArray
    selected_altitude_m: FloatArray

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        altitude = np.asarray(self.selected_altitude_m, dtype=float)
        if altitude.shape != time.shape or np.any(~np.isfinite(altitude)):
            raise ValueError(
                "selected altitudes must be finite and match command times"
            )
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "selected_altitude_m", altitude)

    @classmethod
    def from_samples(
        cls, elapsed_time_s: ArrayLike, selected_altitude_m: ArrayLike
    ) -> SelectedAltitudeIntent:
        """Collapse a sampled selector channel into its change events."""

        time = _validated_time(elapsed_time_s)
        altitude = np.asarray(selected_altitude_m, dtype=float)
        if altitude.shape != time.shape or np.any(~np.isfinite(altitude)):
            raise ValueError("selected altitudes must be finite and match sample times")
        keep = np.r_[True, altitude[1:] != altitude[:-1]]
        return cls(time[keep], altitude[keep])

    def altitude_at(self, elapsed_time_s: float) -> float:
        """Return the active selected altitude at one elapsed time."""

        return float(
            self.selected_altitude_m[
                _command_index(self.effective_time_s, elapsed_time_s)
            ]
        )


@dataclass(frozen=True)
class SelectedSpeedTarget:
    """Resolved active speed command and its original selector representation."""

    mode: SelectedSpeedMode
    true_airspeed_mps: float
    calibrated_airspeed_mps: float
    mach: float
    crossover_altitude_m: float


@dataclass(frozen=True)
class SelectedSpeedIntent:
    """Sample-and-hold CAS/Mach selector commands with crossover switching."""

    effective_time_s: FloatArray
    selected_calibrated_airspeed_mps: FloatArray
    selected_mach: FloatArray

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        cas = np.asarray(self.selected_calibrated_airspeed_mps, dtype=float)
        mach = np.asarray(self.selected_mach, dtype=float)
        if (
            cas.shape != time.shape
            or mach.shape != time.shape
            or np.any(~np.isfinite(cas + mach))
            or np.any(cas <= 1.0)
            or np.any((mach <= 0.05) | (mach >= 1.5))
        ):
            raise ValueError("selected CAS/Mach must be valid and match command times")
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "selected_calibrated_airspeed_mps", cas)
        object.__setattr__(self, "selected_mach", mach)

    @classmethod
    def from_samples(
        cls,
        elapsed_time_s: ArrayLike,
        selected_calibrated_airspeed_mps: ArrayLike,
        selected_mach: ArrayLike,
    ) -> SelectedSpeedIntent:
        """Collapse sampled speed selectors into joint CAS/Mach change events."""

        time = _validated_time(elapsed_time_s)
        cas = np.asarray(selected_calibrated_airspeed_mps, dtype=float)
        mach = np.asarray(selected_mach, dtype=float)
        if cas.shape != time.shape or mach.shape != time.shape:
            raise ValueError("selected speed samples must match elapsed time")
        keep = np.r_[True, (cas[1:] != cas[:-1]) | (mach[1:] != mach[:-1])]
        return cls(time[keep], cas[keep], mach[keep])

    def target_at(
        self,
        elapsed_time_s: float,
        *,
        pressure_altitude_m: float,
        static_temperature_k: float,
    ) -> SelectedSpeedTarget:
        """Resolve the active selector pair to TAS using standard crossover logic."""

        if not np.isfinite(pressure_altitude_m):
            raise ValueError("pressure_altitude_m must be finite")
        if not np.isfinite(static_temperature_k) or static_temperature_k <= 0.0:
            raise ValueError("static_temperature_k must be finite and positive")
        index = _command_index(self.effective_time_s, elapsed_time_s)
        cas = float(self.selected_calibrated_airspeed_mps[index])
        mach = float(self.selected_mach[index])
        crossover = float(aero.crossover_alt(cas, mach))
        if pressure_altitude_m >= crossover:
            mode: SelectedSpeedMode = "MACH"
            speed = float(
                true_airspeed_from_mach(mach, pressure_altitude_m, static_temperature_k)
            )
        else:
            mode = "CAS"
            speed = float(
                true_airspeed_from_calibrated_airspeed(
                    cas, pressure_altitude_m, static_temperature_k
                )
            )
        return SelectedSpeedTarget(mode, speed, cas, mach, crossover)


__all__ = [
    "SelectedAltitudeIntent",
    "SelectedSpeedIntent",
    "SelectedSpeedMode",
    "SelectedSpeedTarget",
]
