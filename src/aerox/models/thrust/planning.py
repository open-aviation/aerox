"""Condition-dependent thrust bounds used by trajectory planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox.models._atmosphere import (
    SEA_LEVEL_PRESSURE_PA,
    atmosphere_at_pressure_altitude,
)

from .engine import DemonstratedClimbN1Capacity, EffectiveN1Envelope, EffectiveThrustMap


@dataclass(frozen=True)
class ConstantMaximumThrust:
    """Explicit constant ceiling useful for generic aircraft and tests."""

    thrust_n: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.thrust_n) or self.thrust_n <= 0.0:
            raise ValueError("maximum thrust must be finite and positive")

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del pressure_altitude_m, true_airspeed_mps, static_temperature_k
        return self.thrust_n


@dataclass(frozen=True)
class PressureScaledMinimumThrust:
    """Pressure-ratio-scaled effective planning-thrust floor."""

    sea_level_thrust_n: float
    pressure_exponent: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sea_level_thrust_n) or self.sea_level_thrust_n < 0.0:
            raise ValueError("sea-level minimum thrust must be finite and non-negative")
        if not np.isfinite(self.pressure_exponent) or self.pressure_exponent <= 0.0:
            raise ValueError(
                "minimum-thrust pressure exponent must be finite and positive"
            )

    def minimum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del true_airspeed_mps
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        pressure_ratio = float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA)
        return self.sea_level_thrust_n * pressure_ratio**self.pressure_exponent


@dataclass(frozen=True)
class QAREffectiveClimbThrustCeiling:
    """QAR operational CLB N1 boundary mapped to effective net thrust."""

    n1_envelope: EffectiveN1Envelope
    thrust_map: EffectiveThrustMap

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        climb_n1 = self.n1_envelope.climb_limit_percent(
            pressure_altitude_m=np.array([pressure_altitude_m]),
            mach=np.array([mach]),
            static_temperature_k=np.array([static_temperature_k]),
        )
        return self.thrust_map.thrust_n_scalar(
            n1_percent=float(climb_n1[0]),
            mach=mach,
            pressure_ratio=float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA),
            static_temperature_k=static_temperature_k,
        )


@dataclass(frozen=True)
class QARDemonstratedClimbThrustCapacity:
    """Map a one-sided QAR demonstrated N1 frontier to effective net thrust."""

    n1_capacity: DemonstratedClimbN1Capacity
    thrust_map: EffectiveThrustMap
    fallback_n1_envelope: EffectiveN1Envelope | None = None

    def planning_n1_percent(
        self,
        *,
        pressure_altitude_m: np.ndarray,
        mach: np.ndarray,
        static_temperature_k: np.ndarray,
    ) -> np.ndarray:
        altitude, mach_array, temperature = np.broadcast_arrays(
            np.asarray(pressure_altitude_m, dtype=float),
            np.asarray(mach, dtype=float),
            np.asarray(static_temperature_k, dtype=float),
        )
        supported = self.n1_capacity.support_mask(
            pressure_altitude_m=altitude,
            mach=mach_array,
            static_temperature_k=temperature,
        )
        result = np.empty(altitude.size, dtype=float)
        flat_altitude = altitude.ravel()
        flat_mach = mach_array.ravel()
        flat_temperature = temperature.ravel()
        flat_supported = supported.ravel()
        if np.any(flat_supported):
            result[flat_supported] = self.n1_capacity.capacity_percent(
                pressure_altitude_m=flat_altitude[flat_supported],
                mach=flat_mach[flat_supported],
                static_temperature_k=flat_temperature[flat_supported],
            )
        if np.any(~flat_supported):
            if self.fallback_n1_envelope is None:
                raise ValueError(
                    "condition is outside demonstrated support and no fallback was supplied"
                )
            result[~flat_supported] = self.fallback_n1_envelope.climb_limit_percent(
                pressure_altitude_m=flat_altitude[~flat_supported],
                mach=flat_mach[~flat_supported],
                static_temperature_k=flat_temperature[~flat_supported],
            )
        return result.reshape(altitude.shape).ravel()

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        capacity_n1 = self.planning_n1_percent(
            pressure_altitude_m=np.array([pressure_altitude_m]),
            mach=np.array([mach]),
            static_temperature_k=np.array([static_temperature_k]),
        )
        return self.thrust_map.thrust_n_scalar(
            n1_percent=float(capacity_n1[0]),
            mach=mach,
            pressure_ratio=float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA),
            static_temperature_k=static_temperature_k,
        )


__all__ = [
    "ConstantMaximumThrust",
    "PressureScaledMinimumThrust",
    "QARDemonstratedClimbThrustCapacity",
    "QAREffectiveClimbThrustCeiling",
]
