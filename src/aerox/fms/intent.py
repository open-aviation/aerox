"""Aircraft-independent lateral, vertical, and speed flight intent."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

SpeedKind = Literal["TAS", "CAS", "MACH"]


@dataclass(frozen=True)
class LateralWaypoint:
    """One horizontal route node, independent of altitude and speed intent."""

    north_m: float
    east_m: float
    name: str = ""
    acceptance_radius_m: float = 4_000.0

    def __post_init__(self) -> None:
        values = (self.north_m, self.east_m, self.acceptance_radius_m)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("lateral waypoint values must be finite")
        if self.acceptance_radius_m <= 0.0:
            raise ValueError("waypoint acceptance radius must be positive")


@dataclass(frozen=True)
class SpeedTarget:
    """A speed value that preserves whether intent was TAS, CAS, or Mach."""

    kind: SpeedKind
    value: float

    def __post_init__(self) -> None:
        if self.kind not in {"TAS", "CAS", "MACH"}:
            raise ValueError("speed kind must be TAS, CAS, or MACH")
        if not np.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("speed target must be finite and positive")
        if self.kind == "MACH" and not 0.05 < self.value < 1.5:
            raise ValueError("Mach target must lie between 0.05 and 1.5")
        if self.kind != "MACH" and self.value <= 1.0:
            raise ValueError("TAS/CAS target must exceed 1 m/s")


@dataclass(frozen=True)
class TerminalConstraint:
    """Exact altitude and speed state at a route endpoint."""

    altitude_m: float
    speed: SpeedTarget

    def __post_init__(self) -> None:
        if not np.isfinite(self.altitude_m):
            raise ValueError("terminal altitude must be finite")
        if not isinstance(self.speed, SpeedTarget):
            raise TypeError("terminal speed must be a SpeedTarget")


@dataclass(frozen=True)
class AltitudeConstraint:
    """Preferred altitude with optional lower and upper bounds at a waypoint."""

    waypoint_index: int
    target_altitude_m: float
    minimum_altitude_m: float | None = None
    maximum_altitude_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.waypoint_index, int):
            raise TypeError("altitude-constraint waypoint index must be an integer")
        values = tuple(
            value
            for value in (
                self.target_altitude_m,
                self.minimum_altitude_m,
                self.maximum_altitude_m,
            )
            if value is not None
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("altitude-constraint values must be finite")
        lower = -np.inf if self.minimum_altitude_m is None else self.minimum_altitude_m
        upper = np.inf if self.maximum_altitude_m is None else self.maximum_altitude_m
        if lower > upper:
            raise ValueError("altitude-constraint bounds are reversed")
        if not lower <= self.target_altitude_m <= upper:
            raise ValueError("target altitude must lie inside its bounds")

    @classmethod
    def at(cls, waypoint_index: int, altitude_m: float) -> AltitudeConstraint:
        """Construct an exact altitude constraint."""

        return cls(waypoint_index, altitude_m, altitude_m, altitude_m)

    @classmethod
    def preferred(cls, waypoint_index: int, altitude_m: float) -> AltitudeConstraint:
        """Construct an advisory altitude target without a hard bound."""

        return cls(waypoint_index, altitude_m)


@dataclass(frozen=True)
class SpeedConstraint:
    """Preferred speed with optional bounds of the same speed kind."""

    waypoint_index: int
    target: SpeedTarget
    minimum_value: float | None = None
    maximum_value: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.waypoint_index, int):
            raise TypeError("speed-constraint waypoint index must be an integer")
        if not isinstance(self.target, SpeedTarget):
            raise TypeError("speed-constraint target must be a SpeedTarget")
        values = tuple(
            value
            for value in (self.minimum_value, self.maximum_value)
            if value is not None
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("speed-constraint bounds must be finite and positive")
        lower = -np.inf if self.minimum_value is None else self.minimum_value
        upper = np.inf if self.maximum_value is None else self.maximum_value
        if lower > upper:
            raise ValueError("speed-constraint bounds are reversed")
        if not lower <= self.target.value <= upper:
            raise ValueError("target speed must lie inside its bounds")
        for value in values:
            SpeedTarget(self.target.kind, value)

    @classmethod
    def at(cls, waypoint_index: int, target: SpeedTarget) -> SpeedConstraint:
        """Construct an exact speed constraint."""

        return cls(waypoint_index, target, target.value, target.value)

    @classmethod
    def preferred(cls, waypoint_index: int, target: SpeedTarget) -> SpeedConstraint:
        """Construct an advisory speed target without a hard bound."""

        return cls(waypoint_index, target)


@dataclass(frozen=True)
class CruiseTarget:
    """Preferred cruise state over an inclusive lateral-waypoint range."""

    start_waypoint_index: int
    end_waypoint_index: int
    altitude_m: float
    speed: SpeedTarget

    def __post_init__(self) -> None:
        if not isinstance(self.start_waypoint_index, int) or not isinstance(
            self.end_waypoint_index, int
        ):
            raise TypeError("cruise waypoint indices must be integers")
        if self.start_waypoint_index > self.end_waypoint_index:
            raise ValueError("cruise waypoint range is reversed")
        if not np.isfinite(self.altitude_m):
            raise ValueError("cruise altitude must be finite")
        if not isinstance(self.speed, SpeedTarget):
            raise TypeError("cruise speed must be a SpeedTarget")


@dataclass(frozen=True)
class FlightIntent:
    """Independent route geometry and managed altitude/speed constraints.

    Departure and arrival states are exact. Intermediate preferred targets
    guide profile construction, while optional bounds carry hard at/above,
    at/below, or window semantics without turning lateral-only waypoints into
    artificial state constraints.
    """

    waypoints: tuple[LateralWaypoint, ...]
    departure: TerminalConstraint
    arrival: TerminalConstraint
    altitude_constraints: tuple[AltitudeConstraint, ...] = ()
    speed_constraints: tuple[SpeedConstraint, ...] = ()
    cruise_target: CruiseTarget | None = None
    maximum_climb_flight_path_angle_rad: float = np.deg2rad(4.0)
    maximum_descent_flight_path_angle_rad: float = np.deg2rad(4.0)
    maximum_acceleration_mps2: float = 0.5
    maximum_deceleration_mps2: float = 0.5

    def __post_init__(self) -> None:
        waypoints = tuple(self.waypoints)
        altitude_constraints = tuple(self.altitude_constraints)
        speed_constraints = tuple(self.speed_constraints)
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "altitude_constraints", altitude_constraints)
        object.__setattr__(self, "speed_constraints", speed_constraints)
        if len(waypoints) < 2:
            raise ValueError("flight intent requires at least two lateral waypoints")
        if any(not isinstance(waypoint, LateralWaypoint) for waypoint in waypoints):
            raise TypeError("waypoints must contain LateralWaypoint objects")
        if not isinstance(self.departure, TerminalConstraint) or not isinstance(
            self.arrival, TerminalConstraint
        ):
            raise TypeError("departure and arrival must be TerminalConstraint objects")
        values = (
            self.maximum_climb_flight_path_angle_rad,
            self.maximum_descent_flight_path_angle_rad,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("flight-intent limits must be finite and positive")
        if max(values[:2]) >= np.deg2rad(30.0):
            raise ValueError("flight-path-angle limits must be below 30 degrees")
        for first, second in pairwise(waypoints):
            if (
                np.hypot(second.north_m - first.north_m, second.east_m - first.east_m)
                <= 1.0
            ):
                raise ValueError("consecutive waypoints must be more than 1 m apart")

        waypoint_count = len(waypoints)
        for name, constraints, expected_type in (
            ("altitude", altitude_constraints, AltitudeConstraint),
            ("speed", speed_constraints, SpeedConstraint),
        ):
            if any(not isinstance(item, expected_type) for item in constraints):
                raise TypeError(f"{name} constraints contain an invalid object")
            indices = [item.waypoint_index for item in constraints]
            if any(not 0 < index < waypoint_count - 1 for index in indices):
                raise ValueError(
                    f"{name} constraints must refer to intermediate waypoints"
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"duplicate {name} constraint at one waypoint")
        if self.cruise_target is not None:
            if not isinstance(self.cruise_target, CruiseTarget):
                raise TypeError("cruise_target must be a CruiseTarget")
            if not (
                0
                < self.cruise_target.start_waypoint_index
                <= self.cruise_target.end_waypoint_index
                < waypoint_count - 1
            ):
                raise ValueError("cruise target must span intermediate waypoints")


__all__ = [
    "AltitudeConstraint",
    "CruiseTarget",
    "FlightIntent",
    "LateralWaypoint",
    "SpeedConstraint",
    "SpeedKind",
    "SpeedTarget",
    "TerminalConstraint",
]
