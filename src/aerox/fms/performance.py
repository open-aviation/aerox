"""Aircraft-independent performance contracts used by FMS planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import ArrayLike, NDArray
import numpy as np

FloatArray = NDArray[np.float64]


class FuelFlowModel(Protocol):
    """Total-aircraft fuel flow at realized net thrust."""

    def fuel_flow_kg_s(self, total_aircraft_thrust_n: ArrayLike) -> FloatArray: ...

    def fuel_flow_scalar(self, total_aircraft_thrust_n: float) -> float: ...


class MaximumThrustModel(Protocol):
    """Condition-dependent total-aircraft realized-thrust ceiling."""

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float: ...


class MinimumThrustModel(Protocol):
    """Condition-dependent total-aircraft realized-thrust floor."""

    def minimum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float: ...


@dataclass(frozen=True)
class LocalPerformanceLimits:
    """Available local path limits at one altitude, speed, and mass."""

    maximum_thrust_n: float
    level_drag_n: float
    maximum_climb_angle_rad: float
    maximum_descent_angle_rad: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    maximum_specific_excess_force_mps2: float
    minimum_specific_excess_force_mps2: float


class ProfilePerformance(Protocol):
    """Aircraft performance reduced to limits required by spatial planners."""

    def limits(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        mass_kg: float,
        static_temperature_k: float,
    ) -> LocalPerformanceLimits: ...


__all__ = [
    "FuelFlowModel",
    "LocalPerformanceLimits",
    "MaximumThrustModel",
    "MinimumThrustModel",
    "ProfilePerformance",
]
