"""Online spatial path guidance and route-progress management."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox.fms.route import RouteProjection, SpatialTrajectory
from aerox.navigation import (
    ground_path_angle_from_air_motion,
    horizontal_ground_speed_mps,
    wind_corrected_path,
)


def _wrapped_angle_delta(target_rad: float, actual_rad: float) -> float:
    return float(
        np.arctan2(np.sin(target_rad - actual_rad), np.cos(target_rad - actual_rad))
    )


@dataclass(frozen=True)
class GuidanceTarget:
    """One online reference generated from route progress and current state."""

    projection: RouteProjection
    progress_m: float
    lookahead_distance_m: float
    target_along_path_m: float
    profile_target_along_path_m: float
    target_north_m: float
    target_east_m: float
    reference_altitude_m: float
    reference_true_airspeed_mps: float
    reference_true_airspeed_rate_mps2: float
    reference_air_flight_path_angle_rad: float
    reference_ground_track_rad: float
    reference_air_heading_rad: float
    predicted_groundspeed_mps: float


@dataclass
class SpatialPathGuidance:
    """Stateful, monotonic route-progress and look-ahead guidance law.

    `position_feedback=False` and `wind_correction=False` define a transparent
    route-leg-bearing baseline for comparisons. They are not recommended for
    trajectory generation in wind.
    """

    trajectory: SpatialTrajectory
    lookahead_time_s: float = 25.0
    minimum_lookahead_m: float = 2_500.0
    maximum_lookahead_m: float = 6_000.0
    profile_lookahead_time_s: float = 5.0
    maximum_profile_lookahead_m: float = 1_000.0
    maximum_heading_command_rate_rad_s: float = np.deg2rad(1.0)
    position_feedback: bool = True
    wind_correction: bool = True
    _progress_m: float = 0.0
    _previous_heading_command_rad: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.lookahead_time_s,
            self.minimum_lookahead_m,
            self.maximum_lookahead_m,
            self.profile_lookahead_time_s,
            self.maximum_profile_lookahead_m,
            self.maximum_heading_command_rate_rad_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("look-ahead settings must be finite and positive")
        if self.minimum_lookahead_m > self.maximum_lookahead_m:
            raise ValueError("minimum look-ahead must not exceed maximum look-ahead")

    @property
    def progress_m(self) -> float:
        return self._progress_m

    def reset(self, progress_m: float = 0.0) -> None:
        if not np.isfinite(progress_m):
            raise ValueError("progress must be finite")
        self._progress_m = float(
            np.clip(progress_m, 0.0, self.trajectory.total_distance_m)
        )
        self._previous_heading_command_rad = None

    def project_position(self, *, north_m: float, east_m: float) -> RouteProjection:
        """Project a position onto the active route neighborhood."""

        active_segment = self.trajectory.segment_index_at(self._progress_m)
        return self.trajectory.project(
            north_m,
            east_m,
            first_segment_index=max(0, active_segment - 1),
            final_segment_index=min(
                active_segment + 1, len(self.trajectory.segment_length_m) - 1
            ),
        )

    def update(
        self,
        *,
        north_m: float,
        east_m: float,
        true_airspeed_mps: float,
        flight_path_angle_rad: float,
        elapsed_time_s: float,
        wind_north_mps: float = 0.0,
        wind_east_mps: float = 0.0,
        wind_down_mps: float = 0.0,
    ) -> GuidanceTarget:
        """Update path progress and return the next state reference."""

        values = (
            north_m,
            east_m,
            true_airspeed_mps,
            flight_path_angle_rad,
            elapsed_time_s,
            wind_north_mps,
            wind_east_mps,
            wind_down_mps,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("guidance state, wind, and elapsed time must be finite")
        if true_airspeed_mps <= 1.0 or elapsed_time_s <= 0.0:
            raise ValueError("guidance airspeed and elapsed time must be positive")
        projection = self.project_position(north_m=north_m, east_m=east_m)
        self._progress_m = max(self._progress_m, projection.along_path_m)
        self._progress_m = min(self._progress_m, self.trajectory.total_distance_m)
        current_path = self.trajectory.sample(self._progress_m)
        current_tangent = np.array(
            [
                np.cos(current_path.ground_track_rad),
                np.sin(current_path.ground_track_rad),
            ]
        )
        if self.wind_correction:
            current_ground_path_angle = ground_path_angle_from_air_motion(
                ground_track_rad=current_path.ground_track_rad,
                air_flight_path_angle_rad=flight_path_angle_rad,
                true_airspeed_mps=true_airspeed_mps,
                wind_north_mps=wind_north_mps,
                wind_east_mps=wind_east_mps,
                wind_down_mps=wind_down_mps,
            )
            current_solution = wind_corrected_path(
                ground_track_rad=current_path.ground_track_rad,
                ground_path_angle_rad=current_ground_path_angle,
                true_airspeed_mps=true_airspeed_mps,
                wind_north_mps=wind_north_mps,
                wind_east_mps=wind_east_mps,
                wind_down_mps=wind_down_mps,
            )
            current_groundspeed = float(current_solution.horizontal_ground_speed_mps)
        else:
            current_groundspeed = float(
                horizontal_ground_speed_mps(
                    true_airspeed_mps,
                    flight_path_angle_rad,
                    np.dot(
                        current_tangent,
                        np.array([wind_north_mps, wind_east_mps]),
                    ),
                )
            )
        lookahead = float(
            np.clip(
                current_groundspeed * self.lookahead_time_s,
                self.minimum_lookahead_m,
                self.maximum_lookahead_m,
            )
        )
        target_along_path = min(
            self._progress_m + lookahead, self.trajectory.total_distance_m
        )
        target = self.trajectory.sample(target_along_path)
        profile_lookahead = min(
            current_groundspeed * self.profile_lookahead_time_s,
            self.maximum_profile_lookahead_m,
        )
        profile_target_along_path = min(
            self._progress_m + profile_lookahead,
            self.trajectory.total_distance_m,
        )
        profile_target = self.trajectory.sample(profile_target_along_path)
        delta_north = target.north_m - north_m
        delta_east = target.east_m - east_m
        if self.position_feedback and np.hypot(delta_north, delta_east) > 1.0:
            ground_track = float(np.arctan2(delta_east, delta_north))
        else:
            ground_track = self.trajectory.sample(self._progress_m).ground_track_rad
        if self.wind_correction:
            target_solution = wind_corrected_path(
                ground_track_rad=ground_track,
                ground_path_angle_rad=target.ground_path_angle_rad,
                true_airspeed_mps=target.true_airspeed_mps,
                wind_north_mps=wind_north_mps,
                wind_east_mps=wind_east_mps,
                wind_down_mps=wind_down_mps,
            )
            raw_air_heading = float(target_solution.air_heading_rad)
            groundspeed = float(target_solution.horizontal_ground_speed_mps)
        else:
            raw_air_heading = ground_track
            tangent = np.array([np.cos(ground_track), np.sin(ground_track)])
            groundspeed = float(
                horizontal_ground_speed_mps(
                    target.true_airspeed_mps,
                    target.ground_path_angle_rad,
                    np.dot(tangent, np.array([wind_north_mps, wind_east_mps])),
                )
            )
        profile_solution = wind_corrected_path(
            ground_track_rad=profile_target.ground_track_rad,
            ground_path_angle_rad=profile_target.ground_path_angle_rad,
            true_airspeed_mps=profile_target.true_airspeed_mps,
            wind_north_mps=wind_north_mps,
            wind_east_mps=wind_east_mps,
            wind_down_mps=wind_down_mps,
        )
        profile_groundspeed = float(profile_solution.horizontal_ground_speed_mps)
        if self._previous_heading_command_rad is None:
            air_heading = raw_air_heading
        else:
            maximum_change = self.maximum_heading_command_rate_rad_s * elapsed_time_s
            air_heading = self._previous_heading_command_rad + float(
                np.clip(
                    _wrapped_angle_delta(
                        raw_air_heading, self._previous_heading_command_rad
                    ),
                    -maximum_change,
                    maximum_change,
                )
            )
            air_heading = float(np.arctan2(np.sin(air_heading), np.cos(air_heading)))
        self._previous_heading_command_rad = air_heading
        return GuidanceTarget(
            projection=projection,
            progress_m=self._progress_m,
            lookahead_distance_m=lookahead,
            target_along_path_m=target_along_path,
            profile_target_along_path_m=profile_target_along_path,
            target_north_m=target.north_m,
            target_east_m=target.east_m,
            reference_altitude_m=profile_target.altitude_m,
            reference_true_airspeed_mps=profile_target.true_airspeed_mps,
            reference_true_airspeed_rate_mps2=(
                profile_target.true_airspeed_gradient_per_m * profile_groundspeed
            ),
            reference_air_flight_path_angle_rad=(
                float(profile_solution.air_flight_path_angle_rad)
                if self.wind_correction
                else profile_target.ground_path_angle_rad
            ),
            reference_ground_track_rad=ground_track,
            reference_air_heading_rad=air_heading,
            predicted_groundspeed_mps=float(groundspeed),
        )
