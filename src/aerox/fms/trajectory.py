"""Spatial route/profile construction and local-path geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from aerox.fms.intent import FlightIntent

FloatArray = NDArray[np.float64]


def _wrap_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


@dataclass(frozen=True)
class SpatialTrajectorySample:
    """One state reference evaluated at along-path distance."""

    along_path_m: float
    north_m: float
    east_m: float
    altitude_m: float
    true_airspeed_mps: float
    ground_track_rad: float
    flight_path_angle_rad: float
    true_airspeed_rate_mps2: float
    segment_index: int


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
    segment_length_m: FloatArray
    segment_track_rad: FloatArray
    segment_altitude_gradient: FloatArray
    segment_speed_gradient_per_s: FloatArray
    predicted_waypoint_time_s: FloatArray
    profile_along_path_m: FloatArray | None = None
    profile_altitude_m: FloatArray | None = None
    profile_true_airspeed_mps: FloatArray | None = None

    def __post_init__(self) -> None:
        waypoint_count = len(self.intent.waypoints)
        fields = {
            "waypoint_along_path_m": (self.waypoint_along_path_m, waypoint_count),
            "segment_length_m": (self.segment_length_m, waypoint_count - 1),
            "segment_track_rad": (self.segment_track_rad, waypoint_count - 1),
            "segment_altitude_gradient": (
                self.segment_altitude_gradient,
                waypoint_count - 1,
            ),
            "segment_speed_gradient_per_s": (
                self.segment_speed_gradient_per_s,
                waypoint_count - 1,
            ),
            "predicted_waypoint_time_s": (
                self.predicted_waypoint_time_s,
                waypoint_count,
            ),
        }
        for name, (value, expected_length) in fields.items():
            array = np.asarray(value, dtype=float)
            if array.shape != (expected_length,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} has invalid shape or non-finite values")
            object.__setattr__(self, name, array)
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
    def from_intent(cls, intent: FlightIntent) -> SpatialTrajectory:
        """Build and check the first deterministic piecewise-linear profile."""

        north = np.array([point.north_m for point in intent.waypoints], dtype=float)
        east = np.array([point.east_m for point in intent.waypoints], dtype=float)
        altitude = np.array(
            [point.altitude_m for point in intent.waypoints], dtype=float
        )
        speed = np.array(
            [point.true_airspeed_mps for point in intent.waypoints], dtype=float
        )
        delta_north = np.diff(north)
        delta_east = np.diff(east)
        length = np.hypot(delta_north, delta_east)
        along_path = np.concatenate(([0.0], np.cumsum(length)))
        altitude_gradient = np.diff(altitude) / length
        gamma = np.arctan(altitude_gradient)
        if np.any(gamma > intent.maximum_climb_flight_path_angle_rad) or np.any(
            gamma < -intent.maximum_descent_flight_path_angle_rad
        ):
            raise ValueError("waypoint altitude profile exceeds declared angle limits")
        speed_gradient = np.diff(speed) / length
        acceleration_at_ends = np.maximum(speed[:-1], speed[1:]) * speed_gradient
        if np.any(acceleration_at_ends > intent.maximum_acceleration_mps2) or np.any(
            acceleration_at_ends < -intent.maximum_deceleration_mps2
        ):
            raise ValueError(
                "waypoint speed profile exceeds declared acceleration limits"
            )
        segment_time = 2.0 * length / (speed[:-1] + speed[1:])
        return cls(
            intent=intent,
            waypoint_along_path_m=along_path,
            segment_length_m=length,
            segment_track_rad=np.arctan2(delta_east, delta_north),
            segment_altitude_gradient=altitude_gradient,
            segment_speed_gradient_per_s=speed_gradient,
            predicted_waypoint_time_s=np.concatenate(([0.0], np.cumsum(segment_time))),
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
        segment_time = (
            2.0 * np.diff(profile_s) / (profile_speed[:-1] + profile_speed[1:])
        )
        profile_time = np.concatenate(([0.0], np.cumsum(segment_time)))
        return replace(
            self,
            predicted_waypoint_time_s=np.interp(
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
        altitude = first.altitude_m + fraction * (second.altitude_m - first.altitude_m)
        speed = first.true_airspeed_mps + fraction * (
            second.true_airspeed_mps - first.true_airspeed_mps
        )
        altitude_gradient = self.segment_altitude_gradient[index]
        speed_gradient = self.segment_speed_gradient_per_s[index]
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
            flight_path_angle_rad=float(np.arctan(altitude_gradient)),
            true_airspeed_rate_mps2=float(speed * speed_gradient),
            segment_index=index,
        )

    def sampled(self, spacing_m: float = 500.0) -> tuple[FloatArray, ...]:
        """Return dense arrays useful for plots and external trajectory tables."""

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
        return (
            values,
            np.array([sample.north_m for sample in samples]),
            np.array([sample.east_m for sample in samples]),
            np.array([sample.altitude_m for sample in samples]),
            np.array([sample.true_airspeed_mps for sample in samples]),
            np.array([sample.ground_track_rad for sample in samples]),
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


def inherit_state_profile(
    *,
    geometry_route: SpatialTrajectory,
    state_profile: SpatialTrajectory,
    state_along_path_at_geometry_waypoints_m: FloatArray,
    spacing_m: float = 500.0,
) -> SpatialTrajectory:
    """Return ``geometry_route`` with altitude and speed inherited separately.

    ``state_along_path_at_geometry_waypoints_m`` maps every geometry waypoint
    to a monotonically increasing distance on ``state_profile``. This lets an
    FMS add horizontal-only turn nodes without turning them into new observed
    altitude or speed constraints. The returned waypoint values and dense
    profile are sampled only from ``state_profile``.
    """

    mapping = np.asarray(state_along_path_at_geometry_waypoints_m, dtype=float)
    if (
        mapping.shape != geometry_route.waypoint_along_path_m.shape
        or np.any(~np.isfinite(mapping))
        or np.any(np.diff(mapping) <= 0.0)
        or not np.isclose(mapping[0], 0.0)
        or not np.isclose(mapping[-1], state_profile.total_distance_m)
    ):
        raise ValueError(
            "state-profile mapping must increase from zero to the source distance"
        )
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")

    mapped_samples = [state_profile.sample(float(value)) for value in mapping]
    mapped_waypoints = tuple(
        replace(
            waypoint,
            altitude_m=sample.altitude_m,
            true_airspeed_mps=sample.true_airspeed_mps,
        )
        for waypoint, sample in zip(
            geometry_route.intent.waypoints, mapped_samples, strict=True
        )
    )
    mapped_route = SpatialTrajectory.from_intent(
        replace(geometry_route.intent, waypoints=mapped_waypoints)
    )

    target_support = [
        np.arange(0.0, mapped_route.total_distance_m, spacing_m),
        mapped_route.waypoint_along_path_m,
        np.array([mapped_route.total_distance_m]),
    ]
    if state_profile.profile_along_path_m is not None:
        target_support.append(
            np.interp(
                state_profile.profile_along_path_m,
                mapping,
                mapped_route.waypoint_along_path_m,
            )
        )
    target_s = np.unique(np.concatenate(target_support))
    source_s = np.interp(
        target_s,
        mapped_route.waypoint_along_path_m,
        mapping,
    )
    inherited_samples = [state_profile.sample(float(value)) for value in source_s]
    return mapped_route.with_profile(
        along_path_m=target_s,
        altitude_m=np.array([sample.altitude_m for sample in inherited_samples]),
        true_airspeed_mps=np.array(
            [sample.true_airspeed_mps for sample in inherited_samples]
        ),
    )


def ground_track_to_air_heading(
    ground_track_rad: float,
    true_airspeed_mps: float,
    wind_north_mps: float,
    wind_east_mps: float,
) -> tuple[float, float]:
    """Return wind-corrected air heading and attainable along-track groundspeed."""

    values = (ground_track_rad, true_airspeed_mps, wind_north_mps, wind_east_mps)
    if any(not np.isfinite(value) for value in values) or true_airspeed_mps <= 1.0:
        raise ValueError(
            "track, airspeed, and wind must be finite with positive airspeed"
        )
    tangent = np.array([np.cos(ground_track_rad), np.sin(ground_track_rad)])
    right = np.array([-np.sin(ground_track_rad), np.cos(ground_track_rad)])
    wind = np.array([wind_north_mps, wind_east_mps])
    crosswind = float(np.dot(wind, right))
    if abs(crosswind) >= true_airspeed_mps:
        raise ValueError("crosswind magnitude is not attainable at this true airspeed")
    along_air = float(np.sqrt(true_airspeed_mps**2 - crosswind**2))
    air_velocity = along_air * tangent - crosswind * right
    ground_speed = along_air + float(np.dot(wind, tangent))
    if ground_speed <= 0.0:
        raise ValueError("wind leaves no positive along-track groundspeed")
    return _wrap_angle(
        float(np.arctan2(air_velocity[1], air_velocity[0]))
    ), ground_speed
