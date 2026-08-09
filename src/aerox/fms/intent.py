"""Aircraft-independent flight intent for the Aerox trajectory manager."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    """One local route node with altitude and true-airspeed constraints."""

    north_m: float
    east_m: float
    altitude_m: float
    true_airspeed_mps: float
    name: str = ""
    acceptance_radius_m: float = 4_000.0

    def __post_init__(self) -> None:
        values = (
            self.north_m,
            self.east_m,
            self.altitude_m,
            self.true_airspeed_mps,
            self.acceptance_radius_m,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("waypoint values must be finite")
        if self.true_airspeed_mps <= 1.0:
            raise ValueError("waypoint true airspeed must exceed 1 m/s")
        if self.acceptance_radius_m <= 0.0:
            raise ValueError("waypoint acceptance radius must be positive")


@dataclass(frozen=True)
class FlightIntent:
    """Ordered route intent and transparent trajectory-management limits.

    These are research-simulator limits supplied by the caller. They are not
    aircraft-type defaults or certified operational constraints.
    """

    waypoints: tuple[Waypoint, ...]
    maximum_climb_flight_path_angle_rad: float = np.deg2rad(4.0)
    maximum_descent_flight_path_angle_rad: float = np.deg2rad(4.0)
    maximum_acceleration_mps2: float = 0.5
    maximum_deceleration_mps2: float = 0.5

    def __post_init__(self) -> None:
        waypoints = tuple(self.waypoints)
        object.__setattr__(self, "waypoints", waypoints)
        if len(waypoints) < 2:
            raise ValueError("flight intent requires at least two waypoints")
        if any(not isinstance(waypoint, Waypoint) for waypoint in waypoints):
            raise TypeError("waypoints must contain Waypoint objects")
        values = (
            self.maximum_climb_flight_path_angle_rad,
            self.maximum_descent_flight_path_angle_rad,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("flight-intent limits must be finite and positive")
        if max(
            self.maximum_climb_flight_path_angle_rad,
            self.maximum_descent_flight_path_angle_rad,
        ) >= np.deg2rad(30.0):
            raise ValueError("flight-path-angle limits must be below 30 degrees")
        for first, second in pairwise(waypoints):
            if (
                np.hypot(second.north_m - first.north_m, second.east_m - first.east_m)
                <= 1.0
            ):
                raise ValueError("consecutive waypoints must be more than 1 m apart")
