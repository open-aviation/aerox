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
SelectedSpeedGuidanceMode = Literal["CAS", "MACH", "CAS_MACH", "MACH_CAS", "SPEED_HOLD"]
SelectedThrustMode = Literal[
    "AUTO",
    "THRUST_FRACTION",
    "THRUST_CEILING_FRACTION",
]
SelectedConfigurationMode = Literal["CLEAN", "DRAG_INCREMENT"]
SelectedVerticalMode = Literal[
    "PATH",
    "ALTITUDE",
    "ALTITUDE_HOLD",
    "VERTICAL_SPEED",
    "FLIGHT_PATH_ANGLE",
]


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
class SelectedVerticalTarget:
    """One active vertical mode and its mode-specific target."""

    mode: SelectedVerticalMode
    altitude_m: float | None
    vertical_speed_mps: float | None
    flight_path_angle_rad: float | None


@dataclass(frozen=True)
class SelectedVerticalIntent:
    """Sample-and-hold vertical guidance modes and targets."""

    effective_time_s: FloatArray
    mode: tuple[SelectedVerticalMode, ...]
    selected_altitude_m: FloatArray
    selected_vertical_speed_mps: FloatArray
    selected_flight_path_angle_rad: FloatArray

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        modes = tuple(self.mode)
        altitude = np.asarray(self.selected_altitude_m, dtype=float)
        vertical_speed = np.asarray(self.selected_vertical_speed_mps, dtype=float)
        angle = np.asarray(self.selected_flight_path_angle_rad, dtype=float)
        valid_modes = {
            "PATH",
            "ALTITUDE",
            "ALTITUDE_HOLD",
            "VERTICAL_SPEED",
            "FLIGHT_PATH_ANGLE",
        }
        if (
            len(modes) != len(time)
            or altitude.shape != time.shape
            or vertical_speed.shape != time.shape
            or angle.shape != time.shape
            or any(mode not in valid_modes for mode in modes)
        ):
            raise ValueError("selected vertical rules must match command times")
        for mode, altitude_value, speed_value, angle_value in zip(
            modes, altitude, vertical_speed, angle, strict=True
        ):
            values = (altitude_value, speed_value, angle_value)
            populated = tuple(np.isfinite(value) for value in values)
            if mode == "PATH":
                valid = populated == (False, False, False)
            elif mode in {"ALTITUDE", "ALTITUDE_HOLD"}:
                valid = populated == (True, False, False)
            elif mode == "VERTICAL_SPEED":
                valid = populated == (False, True, False)
            else:
                valid = populated == (False, False, True) and abs(angle_value) < (
                    np.deg2rad(30.0)
                )
            if not valid:
                raise ValueError(
                    f"{mode} requires only its finite mode-specific vertical target"
                )
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "mode", modes)
        object.__setattr__(self, "selected_altitude_m", altitude)
        object.__setattr__(self, "selected_vertical_speed_mps", vertical_speed)
        object.__setattr__(self, "selected_flight_path_angle_rad", angle)

    def target_at(self, elapsed_time_s: float) -> SelectedVerticalTarget:
        """Return the active vertical rule at one elapsed time."""

        index = _command_index(self.effective_time_s, elapsed_time_s)
        mode = self.mode[index]

        def optional(values: FloatArray) -> float | None:
            value = float(values[index])
            return value if np.isfinite(value) else None

        return SelectedVerticalTarget(
            mode=mode,
            altitude_m=optional(self.selected_altitude_m),
            vertical_speed_mps=optional(self.selected_vertical_speed_mps),
            flight_path_angle_rad=optional(self.selected_flight_path_angle_rad),
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
class SelectedSpeedGuidanceIntent:
    """Sample-and-hold performance modes and optional active selectors."""

    effective_time_s: FloatArray
    mode: tuple[SelectedSpeedGuidanceMode, ...]
    active_selector: tuple[SelectedSpeedMode | None, ...] | None = None

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        modes = tuple(self.mode)
        selectors = (
            tuple(None for _ in modes)
            if self.active_selector is None
            else tuple(self.active_selector)
        )
        valid_modes = {"CAS", "MACH", "CAS_MACH", "MACH_CAS", "SPEED_HOLD"}
        if len(modes) != len(time) or any(mode not in valid_modes for mode in modes):
            raise ValueError(
                "selected speed modes must be valid and match command times"
            )
        if len(selectors) != len(time) or any(
            selector not in {None, "CAS", "MACH"} for selector in selectors
        ):
            raise ValueError(
                "active speed selectors must be CAS, MACH, or automatic and "
                "match command times"
            )
        if any(
            mode in {"CAS", "MACH"} and selector not in {None, mode}
            for mode, selector in zip(modes, selectors, strict=True)
        ):
            raise ValueError("explicit CAS or MACH mode requires the same selector")
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "mode", modes)
        object.__setattr__(self, "active_selector", selectors)

    def mode_at(self, elapsed_time_s: float) -> SelectedSpeedGuidanceMode:
        """Return the active speed-guidance mode at one elapsed time."""

        return self.mode[_command_index(self.effective_time_s, elapsed_time_s)]

    def active_selector_at(self, elapsed_time_s: float) -> SelectedSpeedMode | None:
        """Return the explicit active selector, or ``None`` for crossover logic."""

        index = _command_index(self.effective_time_s, elapsed_time_s)
        return self.active_selector[index]


@dataclass(frozen=True)
class SelectedThrustTarget:
    """Active normalized thrust rule from a sample-and-hold schedule."""

    mode: SelectedThrustMode
    available_thrust_fraction: float | None


@dataclass(frozen=True)
class SelectedThrustIntent:
    """Sample-and-hold normalized thrust rules.

    A finite fraction selects a point between the condition-specific minimum
    and maximum thrust. The mode states whether that point is a target or a
    ceiling. ``NaN`` is the storage value for AUTO segments.
    """

    effective_time_s: FloatArray
    mode: tuple[SelectedThrustMode, ...]
    available_thrust_fraction: FloatArray

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        modes = tuple(self.mode)
        fraction = np.asarray(self.available_thrust_fraction, dtype=float)
        if len(modes) != len(time) or fraction.shape != time.shape:
            raise ValueError("selected thrust rules must match command times")
        fraction_modes = {"THRUST_FRACTION", "THRUST_CEILING_FRACTION"}
        if any(mode not in {"AUTO", *fraction_modes} for mode in modes):
            raise ValueError("selected thrust modes are not valid")
        for mode, value in zip(modes, fraction, strict=True):
            if mode == "AUTO" and not np.isnan(value):
                raise ValueError("AUTO thrust rules require a NaN fraction")
            if mode in fraction_modes and (
                not np.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{mode} rules require a fraction from zero to one")
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "mode", modes)
        object.__setattr__(self, "available_thrust_fraction", fraction)

    def target_at(self, elapsed_time_s: float) -> SelectedThrustTarget:
        """Return the active normalized thrust rule at one elapsed time."""

        index = _command_index(self.effective_time_s, elapsed_time_s)
        mode = self.mode[index]
        fraction = float(self.available_thrust_fraction[index])
        return SelectedThrustTarget(
            mode,
            None if mode == "AUTO" else fraction,
        )


@dataclass(frozen=True)
class SelectedConfigurationTarget:
    """Active effective aerodynamic-configuration rule."""

    mode: SelectedConfigurationMode
    drag_coefficient_increment: float


@dataclass(frozen=True)
class SelectedConfigurationIntent:
    """Sample-and-hold effective configuration-drag rules.

    The increment is relative to the aircraft model's clean drag coefficient.
    It does not identify a specific flap, gear, or speedbrake setting.
    """

    effective_time_s: FloatArray
    mode: tuple[SelectedConfigurationMode, ...]
    drag_coefficient_increment: FloatArray

    def __post_init__(self) -> None:
        time = _validated_time(self.effective_time_s)
        modes = tuple(self.mode)
        increment = np.asarray(self.drag_coefficient_increment, dtype=float)
        if len(modes) != len(time) or increment.shape != time.shape:
            raise ValueError("selected configuration rules must match command times")
        if any(mode not in {"CLEAN", "DRAG_INCREMENT"} for mode in modes):
            raise ValueError("selected configuration modes are not valid")
        if np.any(~np.isfinite(increment)) or np.any(increment < 0.0):
            raise ValueError("configuration drag increments must be non-negative")
        if any(
            mode == "CLEAN" and not np.isclose(value, 0.0)
            for mode, value in zip(modes, increment, strict=True)
        ):
            raise ValueError("CLEAN configuration rules require zero drag increment")
        if any(
            mode == "DRAG_INCREMENT" and value <= 0.0
            for mode, value in zip(modes, increment, strict=True)
        ):
            raise ValueError(
                "DRAG_INCREMENT rules require a positive drag coefficient increment"
            )
        object.__setattr__(self, "effective_time_s", time)
        object.__setattr__(self, "mode", modes)
        object.__setattr__(self, "drag_coefficient_increment", increment)

    def target_at(self, elapsed_time_s: float) -> SelectedConfigurationTarget:
        """Return the active effective configuration rule."""

        index = _command_index(self.effective_time_s, elapsed_time_s)
        return SelectedConfigurationTarget(
            self.mode[index],
            float(self.drag_coefficient_increment[index]),
        )


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
        mode: SelectedSpeedMode | None = None,
    ) -> SelectedSpeedTarget:
        """Resolve one selector pair to TAS.

        When ``mode`` is omitted, the pair is resolved statelessly from the
        crossover altitude. Stateful climb/descent guidance should supply its
        latched CAS or Mach mode explicitly.
        """

        if not np.isfinite(pressure_altitude_m):
            raise ValueError("pressure_altitude_m must be finite")
        if not np.isfinite(static_temperature_k) or static_temperature_k <= 0.0:
            raise ValueError("static_temperature_k must be finite and positive")
        index = _command_index(self.effective_time_s, elapsed_time_s)
        cas = float(self.selected_calibrated_airspeed_mps[index])
        mach = float(self.selected_mach[index])
        crossover = float(aero.crossover_alt(cas, mach))
        resolved_mode: SelectedSpeedMode = (
            "MACH" if pressure_altitude_m >= crossover else "CAS"
        )
        if mode is not None:
            resolved_mode = mode
        if resolved_mode == "MACH":
            speed = float(
                true_airspeed_from_mach(mach, pressure_altitude_m, static_temperature_k)
            )
        else:
            speed = float(
                true_airspeed_from_calibrated_airspeed(
                    cas, pressure_altitude_m, static_temperature_k
                )
            )
        return SelectedSpeedTarget(resolved_mode, speed, cas, mach, crossover)


__all__ = [
    "SelectedAltitudeIntent",
    "SelectedConfigurationIntent",
    "SelectedConfigurationMode",
    "SelectedConfigurationTarget",
    "SelectedSpeedGuidanceIntent",
    "SelectedSpeedGuidanceMode",
    "SelectedSpeedIntent",
    "SelectedSpeedMode",
    "SelectedSpeedTarget",
    "SelectedThrustIntent",
    "SelectedThrustMode",
    "SelectedThrustTarget",
    "SelectedVerticalIntent",
    "SelectedVerticalMode",
    "SelectedVerticalTarget",
]
