"""Spatial route/profile construction and local-path geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from aerox.fms.intent import FlightIntent, SpeedTarget
from aerox.models.atmosphere import (
    isa_atmosphere,
    true_airspeed_from_calibrated_airspeed,
    true_airspeed_from_mach,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialTrajectorySample:
    """One state reference evaluated at along-path distance."""

    along_path_m: float
    north_m: float
    east_m: float
    altitude_m: float
    true_airspeed_mps: float
    ground_track_rad: float
    ground_path_angle_rad: float
    true_airspeed_gradient_per_m: float
    segment_index: int


@dataclass(frozen=True)
class SpatialTrajectoryGrid:
    """Dense named arrays sampled from one spatial trajectory."""

    along_path_m: FloatArray
    north_m: FloatArray
    east_m: FloatArray
    altitude_m: FloatArray
    true_airspeed_mps: FloatArray
    ground_track_rad: FloatArray
    ground_path_angle_rad: FloatArray


@dataclass(frozen=True)
class RouteProjection:
    """Closest point on a route leg and signed lateral displacement."""

    along_path_m: float
    north_m: float
    east_m: float
    cross_track_error_m: float
    distance_m: float
    segment_index: int


@dataclass(frozen=True)
class SpatialTrajectory:
    """Polyline route with altitude/speed profiles parameterized by distance."""

    intent: FlightIntent
    waypoint_along_path_m: FloatArray
    waypoint_altitude_m: FloatArray
    waypoint_true_airspeed_mps: FloatArray
    waypoint_minimum_altitude_m: FloatArray
    waypoint_maximum_altitude_m: FloatArray
    waypoint_minimum_true_airspeed_mps: FloatArray
    waypoint_maximum_true_airspeed_mps: FloatArray
    altitude_target_waypoint_mask: NDArray[np.bool_]
    speed_target_waypoint_mask: NDArray[np.bool_]
    altitude_bound_waypoint_mask: NDArray[np.bool_]
    speed_bound_waypoint_mask: NDArray[np.bool_]
    segment_length_m: FloatArray
    segment_track_rad: FloatArray
    segment_altitude_gradient: FloatArray
    segment_true_airspeed_gradient_per_m: FloatArray
    calm_air_waypoint_time_s: FloatArray
    profile_along_path_m: FloatArray | None = None
    profile_altitude_m: FloatArray | None = None
    profile_true_airspeed_mps: FloatArray | None = None

    def __post_init__(self) -> None:
        waypoint_count = len(self.intent.waypoints)
        fields = {
            "waypoint_along_path_m": (self.waypoint_along_path_m, waypoint_count),
            "waypoint_altitude_m": (self.waypoint_altitude_m, waypoint_count),
            "waypoint_true_airspeed_mps": (
                self.waypoint_true_airspeed_mps,
                waypoint_count,
            ),
            "segment_length_m": (self.segment_length_m, waypoint_count - 1),
            "segment_track_rad": (self.segment_track_rad, waypoint_count - 1),
            "segment_altitude_gradient": (
                self.segment_altitude_gradient,
                waypoint_count - 1,
            ),
            "segment_true_airspeed_gradient_per_m": (
                self.segment_true_airspeed_gradient_per_m,
                waypoint_count - 1,
            ),
            "calm_air_waypoint_time_s": (
                self.calm_air_waypoint_time_s,
                waypoint_count,
            ),
        }
        for name, (value, expected_length) in fields.items():
            array = np.asarray(value, dtype=float)
            if array.shape != (expected_length,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} has invalid shape or non-finite values")
            object.__setattr__(self, name, array)
        bounded_fields = {
            "waypoint_minimum_altitude_m": self.waypoint_minimum_altitude_m,
            "waypoint_maximum_altitude_m": self.waypoint_maximum_altitude_m,
            "waypoint_minimum_true_airspeed_mps": (
                self.waypoint_minimum_true_airspeed_mps
            ),
            "waypoint_maximum_true_airspeed_mps": (
                self.waypoint_maximum_true_airspeed_mps
            ),
        }
        for name, value in bounded_fields.items():
            array = np.asarray(value, dtype=float)
            if array.shape != (waypoint_count,) or np.any(np.isnan(array)):
                raise ValueError(f"{name} has invalid shape or NaN values")
            object.__setattr__(self, name, array)
        mask_fields = (
            "altitude_target_waypoint_mask",
            "speed_target_waypoint_mask",
            "altitude_bound_waypoint_mask",
            "speed_bound_waypoint_mask",
        )
        for name in mask_fields:
            mask = np.asarray(getattr(self, name), dtype=bool)
            if mask.shape != (waypoint_count,):
                raise ValueError(f"{name} has invalid shape")
            object.__setattr__(self, name, mask)
        if np.any(
            self.waypoint_minimum_altitude_m > self.waypoint_maximum_altitude_m
        ) or np.any(
            self.waypoint_minimum_true_airspeed_mps
            > self.waypoint_maximum_true_airspeed_mps
        ):
            raise ValueError("resolved waypoint constraint bounds are reversed")
        if np.any(self.waypoint_true_airspeed_mps <= 1.0):
            raise ValueError("resolved waypoint TAS must exceed 1 m/s")
        if np.any(self.segment_length_m <= 0.0):
            raise ValueError("route segment lengths must be positive")
        if not np.isclose(self.waypoint_along_path_m[0], 0.0) or np.any(
            np.diff(self.waypoint_along_path_m) <= 0.0
        ):
            raise ValueError("waypoint along-path coordinates must increase from zero")
        profile_fields = (
            self.profile_along_path_m,
            self.profile_altitude_m,
            self.profile_true_airspeed_mps,
        )
        if any(value is not None for value in profile_fields):
            if any(value is None for value in profile_fields):
                raise ValueError("all spatial profile arrays must be supplied together")
            profile_s = np.asarray(self.profile_along_path_m, dtype=float)
            profile_altitude = np.asarray(self.profile_altitude_m, dtype=float)
            profile_speed = np.asarray(self.profile_true_airspeed_mps, dtype=float)
            if (
                profile_s.ndim != 1
                or len(profile_s) < 2
                or profile_altitude.shape != profile_s.shape
                or profile_speed.shape != profile_s.shape
                or np.any(~np.isfinite(profile_s + profile_altitude + profile_speed))
                or np.any(np.diff(profile_s) <= 0.0)
                or not np.isclose(profile_s[0], 0.0)
                or not np.isclose(profile_s[-1], self.total_distance_m)
                or np.any(profile_speed <= 1.0)
            ):
                raise ValueError("spatial profile arrays are invalid")
            object.__setattr__(self, "profile_along_path_m", profile_s)
            object.__setattr__(self, "profile_altitude_m", profile_altitude)
            object.__setattr__(self, "profile_true_airspeed_mps", profile_speed)

    @classmethod
    def from_intent(
        cls,
        intent: FlightIntent,
        *,
        static_temperature_k: FloatArray | None = None,
    ) -> SpatialTrajectory:
        """Resolve independent intent into a deterministic spatial target."""

        north = np.array([point.north_m for point in intent.waypoints], dtype=float)
        east = np.array([point.east_m for point in intent.waypoints], dtype=float)
        delta_north = np.diff(north)
        delta_east = np.diff(east)
        length = np.hypot(delta_north, delta_east)
        along_path = np.concatenate(([0.0], np.cumsum(length)))
        waypoint_count = len(intent.waypoints)

        altitude = np.full(waypoint_count, np.nan)
        minimum_altitude = np.full(waypoint_count, -np.inf)
        maximum_altitude = np.full(waypoint_count, np.inf)
        altitude_target_mask = np.zeros(waypoint_count, dtype=bool)
        altitude_bound_mask = np.zeros(waypoint_count, dtype=bool)
        altitude[0] = intent.departure.altitude_m
        altitude[-1] = intent.arrival.altitude_m
        minimum_altitude[[0, -1]] = altitude[[0, -1]]
        maximum_altitude[[0, -1]] = altitude[[0, -1]]
        altitude_target_mask[[0, -1]] = True
        altitude_bound_mask[[0, -1]] = True
        if intent.cruise_target is not None:
            cruise_slice = slice(
                intent.cruise_target.start_waypoint_index,
                intent.cruise_target.end_waypoint_index + 1,
            )
            altitude[cruise_slice] = intent.cruise_target.altitude_m
            altitude_target_mask[cruise_slice] = True
        for constraint in intent.altitude_constraints:
            index = constraint.waypoint_index
            altitude[index] = constraint.target_altitude_m
            altitude_target_mask[index] = True
            if constraint.minimum_altitude_m is not None:
                minimum_altitude[index] = constraint.minimum_altitude_m
                altitude_bound_mask[index] = True
            if constraint.maximum_altitude_m is not None:
                maximum_altitude[index] = constraint.maximum_altitude_m
                altitude_bound_mask[index] = True
        target_indices = np.flatnonzero(altitude_target_mask)
        altitude = np.interp(
            along_path,
            along_path[target_indices],
            altitude[target_indices],
        )

        if static_temperature_k is None:
            temperature = isa_atmosphere(altitude).temperature_k
        else:
            temperature = np.asarray(static_temperature_k, dtype=float)
            if (
                temperature.shape != (waypoint_count,)
                or np.any(~np.isfinite(temperature))
                or np.any(temperature <= 0.0)
            ):
                raise ValueError(
                    "static_temperature_k must be a positive waypoint-aligned vector"
                )

        def tas(target: SpeedTarget, index: int, value: float | None = None) -> float:
            resolved_value = target.value if value is None else value
            if target.kind == "TAS":
                return float(resolved_value)
            if target.kind == "CAS":
                return float(
                    true_airspeed_from_calibrated_airspeed(
                        resolved_value,
                        altitude[index],
                        temperature[index],
                    )
                )
            return float(
                true_airspeed_from_mach(
                    resolved_value,
                    altitude[index],
                    temperature[index],
                )
            )

        speed = np.full(waypoint_count, np.nan)
        minimum_speed = np.full(waypoint_count, 1.0)
        maximum_speed = np.full(waypoint_count, np.inf)
        speed_target_mask = np.zeros(waypoint_count, dtype=bool)
        speed_bound_mask = np.zeros(waypoint_count, dtype=bool)
        terminal_speeds = (intent.departure.speed, intent.arrival.speed)
        for index, target in zip((0, waypoint_count - 1), terminal_speeds, strict=True):
            speed[index] = tas(target, index)
            minimum_speed[index] = maximum_speed[index] = speed[index]
            speed_target_mask[index] = True
            speed_bound_mask[index] = True
        if intent.cruise_target is not None:
            for index in range(
                intent.cruise_target.start_waypoint_index,
                intent.cruise_target.end_waypoint_index + 1,
            ):
                speed[index] = tas(intent.cruise_target.speed, index)
                speed_target_mask[index] = True
        for constraint in intent.speed_constraints:
            index = constraint.waypoint_index
            speed[index] = tas(constraint.target, index)
            speed_target_mask[index] = True
            if constraint.minimum_value is not None:
                minimum_speed[index] = tas(
                    constraint.target, index, constraint.minimum_value
                )
                speed_bound_mask[index] = True
            if constraint.maximum_value is not None:
                maximum_speed[index] = tas(
                    constraint.target, index, constraint.maximum_value
                )
                speed_bound_mask[index] = True
        target_indices = np.flatnonzero(speed_target_mask)
        speed = np.interp(along_path, along_path[target_indices], speed[target_indices])

        altitude_gradient = np.diff(altitude) / length
        gamma = np.arctan(altitude_gradient)
        speed_gradient = np.diff(speed) / length
        segment_time = 2.0 * length / ((speed[:-1] + speed[1:]) * np.cos(gamma))
        return cls(
            intent=intent,
            waypoint_along_path_m=along_path,
            waypoint_altitude_m=altitude,
            waypoint_true_airspeed_mps=speed,
            waypoint_minimum_altitude_m=minimum_altitude,
            waypoint_maximum_altitude_m=maximum_altitude,
            waypoint_minimum_true_airspeed_mps=minimum_speed,
            waypoint_maximum_true_airspeed_mps=maximum_speed,
            altitude_target_waypoint_mask=altitude_target_mask,
            speed_target_waypoint_mask=speed_target_mask,
            altitude_bound_waypoint_mask=altitude_bound_mask,
            speed_bound_waypoint_mask=speed_bound_mask,
            segment_length_m=length,
            segment_track_rad=np.arctan2(delta_east, delta_north),
            segment_altitude_gradient=altitude_gradient,
            segment_true_airspeed_gradient_per_m=speed_gradient,
            calm_air_waypoint_time_s=np.concatenate(([0.0], np.cumsum(segment_time))),
        )

    @property
    def total_distance_m(self) -> float:
        return float(self.waypoint_along_path_m[-1])

    def segment_index_at(self, along_path_m: float) -> int:
        value = float(np.clip(along_path_m, 0.0, self.total_distance_m))
        return min(
            int(np.searchsorted(self.waypoint_along_path_m, value, side="right") - 1),
            len(self.segment_length_m) - 1,
        )

    def with_profile(
        self,
        *,
        along_path_m: FloatArray,
        altitude_m: FloatArray,
        true_airspeed_mps: FloatArray,
    ) -> SpatialTrajectory:
        """Return the same route geometry with a dense feasible state profile."""

        profile_s = np.asarray(along_path_m, dtype=float)
        profile_speed = np.asarray(true_airspeed_mps, dtype=float)
        interval_gamma = np.arctan(
            np.diff(np.asarray(altitude_m, dtype=float)) / np.diff(profile_s)
        )
        horizontal_speed_start = profile_speed[:-1] * np.cos(interval_gamma)
        horizontal_speed_end = profile_speed[1:] * np.cos(interval_gamma)
        segment_time = (
            2.0 * np.diff(profile_s) / (horizontal_speed_start + horizontal_speed_end)
        )
        profile_time = np.concatenate(([0.0], np.cumsum(segment_time)))
        return replace(
            self,
            calm_air_waypoint_time_s=np.interp(
                self.waypoint_along_path_m, profile_s, profile_time
            ),
            profile_along_path_m=profile_s,
            profile_altitude_m=np.asarray(altitude_m, dtype=float),
            profile_true_airspeed_mps=profile_speed,
        )

    def sample(self, along_path_m: float) -> SpatialTrajectorySample:
        """Interpolate the route and profile at one cumulative distance."""

        value = float(np.clip(along_path_m, 0.0, self.total_distance_m))
        index = self.segment_index_at(value)
        fraction = (value - self.waypoint_along_path_m[index]) / self.segment_length_m[
            index
        ]
        first = self.intent.waypoints[index]
        second = self.intent.waypoints[index + 1]
        altitude = self.waypoint_altitude_m[index] + fraction * (
            self.waypoint_altitude_m[index + 1] - self.waypoint_altitude_m[index]
        )
        speed = self.waypoint_true_airspeed_mps[index] + fraction * (
            self.waypoint_true_airspeed_mps[index + 1]
            - self.waypoint_true_airspeed_mps[index]
        )
        altitude_gradient = self.segment_altitude_gradient[index]
        speed_gradient = self.segment_true_airspeed_gradient_per_m[index]
        if self.profile_along_path_m is not None:
            assert self.profile_altitude_m is not None
            assert self.profile_true_airspeed_mps is not None
            profile_index = int(
                np.searchsorted(self.profile_along_path_m, value, side="right") - 1
            )
            profile_index = int(
                np.clip(profile_index, 0, len(self.profile_along_path_m) - 2)
            )
            profile_interval = (
                self.profile_along_path_m[profile_index + 1]
                - self.profile_along_path_m[profile_index]
            )
            altitude = float(
                np.interp(value, self.profile_along_path_m, self.profile_altitude_m)
            )
            speed = float(
                np.interp(
                    value,
                    self.profile_along_path_m,
                    self.profile_true_airspeed_mps,
                )
            )
            altitude_gradient = float(
                (
                    self.profile_altitude_m[profile_index + 1]
                    - self.profile_altitude_m[profile_index]
                )
                / profile_interval
            )
            speed_gradient = float(
                (
                    self.profile_true_airspeed_mps[profile_index + 1]
                    - self.profile_true_airspeed_mps[profile_index]
                )
                / profile_interval
            )
        return SpatialTrajectorySample(
            along_path_m=value,
            north_m=float(first.north_m + fraction * (second.north_m - first.north_m)),
            east_m=float(first.east_m + fraction * (second.east_m - first.east_m)),
            altitude_m=float(altitude),
            true_airspeed_mps=float(speed),
            ground_track_rad=float(self.segment_track_rad[index]),
            ground_path_angle_rad=float(np.arctan(altitude_gradient)),
            true_airspeed_gradient_per_m=float(speed_gradient),
            segment_index=index,
        )

    def sample_grid(self, spacing_m: float = 500.0) -> SpatialTrajectoryGrid:
        """Return named dense arrays for plots and external trajectory tables."""

        if not np.isfinite(spacing_m) or spacing_m <= 0.0:
            raise ValueError("spacing_m must be finite and positive")
        components = [
            np.arange(0.0, self.total_distance_m, spacing_m),
            self.waypoint_along_path_m,
            np.array([self.total_distance_m]),
        ]
        if self.profile_along_path_m is not None:
            components.append(self.profile_along_path_m)
        values = np.unique(np.concatenate(components))
        samples = [self.sample(value) for value in values]
        return SpatialTrajectoryGrid(
            along_path_m=values,
            north_m=np.array([sample.north_m for sample in samples]),
            east_m=np.array([sample.east_m for sample in samples]),
            altitude_m=np.array([sample.altitude_m for sample in samples]),
            true_airspeed_mps=np.array(
                [sample.true_airspeed_mps for sample in samples]
            ),
            ground_track_rad=np.array([sample.ground_track_rad for sample in samples]),
            ground_path_angle_rad=np.array(
                [sample.ground_path_angle_rad for sample in samples]
            ),
        )

    def project(
        self,
        north_m: float,
        east_m: float,
        *,
        first_segment_index: int = 0,
        final_segment_index: int | None = None,
    ) -> RouteProjection:
        """Project a local position onto a bounded range of route segments."""

        if not np.isfinite(north_m) or not np.isfinite(east_m):
            raise ValueError("position must be finite")
        segment_count = len(self.segment_length_m)
        first_index = int(np.clip(first_segment_index, 0, segment_count - 1))
        last_index = (
            segment_count - 1
            if final_segment_index is None
            else int(np.clip(final_segment_index, first_index, segment_count - 1))
        )
        best: RouteProjection | None = None
        for index in range(first_index, last_index + 1):
            first = self.intent.waypoints[index]
            second = self.intent.waypoints[index + 1]
            delta = np.array(
                [second.north_m - first.north_m, second.east_m - first.east_m]
            )
            relative = np.array([north_m - first.north_m, east_m - first.east_m])
            fraction = float(
                np.clip(np.dot(relative, delta) / np.dot(delta, delta), 0.0, 1.0)
            )
            projected = np.array([first.north_m, first.east_m]) + fraction * delta
            residual = np.array([north_m, east_m]) - projected
            distance = float(np.hypot(*residual))
            unit = delta / self.segment_length_m[index]
            signed_cross_track = float(unit[0] * relative[1] - unit[1] * relative[0])
            candidate = RouteProjection(
                along_path_m=float(
                    self.waypoint_along_path_m[index]
                    + fraction * self.segment_length_m[index]
                ),
                north_m=float(projected[0]),
                east_m=float(projected[1]),
                cross_track_error_m=signed_cross_track,
                distance_m=distance,
                segment_index=index,
            )
            if best is None or candidate.distance_m < best.distance_m:
                best = candidate
        assert best is not None
        return best
