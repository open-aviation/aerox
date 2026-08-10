"""Online selected-altitude capture layered over lateral path guidance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from aerox.fms.selected_intent import SelectedAltitudeIntent, SelectedSpeedIntent
from aerox.fms.route import RouteProjection, SpatialTrajectory
from aerox.guidance.path import GuidanceTarget, SpatialPathGuidance
from aerox.navigation import ground_path_angle_from_air_motion, wind_corrected_path

SelectedAltitudeMode = Literal["CLIMB", "DESCENT", "CAPTURE", "ALT_HOLD"]


@dataclass
class SelectedAltitudeGuidance:
    """Track sample-and-hold altitude commands without a prescribed vertical path."""

    lateral_guidance: SpatialPathGuidance
    altitude_intent: SelectedAltitudeIntent
    speed_intent: SelectedSpeedIntent | None = None
    capture_time_constant_s: float = 90.0
    altitude_hold_deadband_m: float = 15.24
    maximum_climb_angle_rad: float = np.deg2rad(3.5)
    maximum_descent_angle_rad: float = np.deg2rad(3.5)
    maximum_flight_path_command_rate_rad_s: float = np.deg2rad(0.5)
    _previous_flight_path_command_rad: float | None = None
    _previous_heading_command_rad: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.capture_time_constant_s,
            self.altitude_hold_deadband_m,
            self.maximum_climb_angle_rad,
            self.maximum_descent_angle_rad,
            self.maximum_flight_path_command_rate_rad_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("selected-altitude guidance settings must be positive")
        if max(
            self.maximum_climb_angle_rad, self.maximum_descent_angle_rad
        ) >= np.deg2rad(30.0):
            raise ValueError(
                "selected-altitude flight-path limits must be below 30 degrees"
            )

    @property
    def trajectory(self) -> SpatialTrajectory:
        return self.lateral_guidance.trajectory

    @property
    def progress_m(self) -> float:
        return self.lateral_guidance.progress_m

    def reset(self, progress_m: float = 0.0) -> None:
        self.lateral_guidance.reset(progress_m)
        self._previous_flight_path_command_rad = None
        self._previous_heading_command_rad = None

    def project_position(self, *, north_m: float, east_m: float) -> RouteProjection:
        return self.lateral_guidance.project_position(north_m=north_m, east_m=east_m)

    def update(
        self,
        *,
        north_m: float,
        east_m: float,
        true_airspeed_mps: float,
        flight_path_angle_rad: float,
        pressure_altitude_m: float | None = None,
        simulation_time_s: float | None = None,
        static_temperature_k: float | None = None,
        elapsed_time_s: float,
        wind_north_mps: float = 0.0,
        wind_east_mps: float = 0.0,
        wind_down_mps: float = 0.0,
    ) -> GuidanceTarget:
        """Return lateral guidance with selected-altitude and speed references."""

        if pressure_altitude_m is None or simulation_time_s is None:
            raise ValueError(
                "selected-altitude guidance requires altitude and simulation time"
            )
        if not np.isfinite(pressure_altitude_m):
            raise ValueError("pressure_altitude_m must be finite")
        base = self.lateral_guidance.update(
            north_m=north_m,
            east_m=east_m,
            true_airspeed_mps=true_airspeed_mps,
            flight_path_angle_rad=flight_path_angle_rad,
            pressure_altitude_m=pressure_altitude_m,
            simulation_time_s=simulation_time_s,
            static_temperature_k=static_temperature_k,
            elapsed_time_s=elapsed_time_s,
            wind_north_mps=wind_north_mps,
            wind_east_mps=wind_east_mps,
            wind_down_mps=wind_down_mps,
        )
        selected_altitude = self.altitude_intent.altitude_at(simulation_time_s)
        altitude_error = selected_altitude - pressure_altitude_m
        if abs(altitude_error) <= self.altitude_hold_deadband_m:
            desired_ground_vertical_speed = 0.0
            mode: SelectedAltitudeMode = "ALT_HOLD"
        else:
            desired_ground_vertical_speed = (
                altitude_error / self.capture_time_constant_s
            )
            maximum_climb_rate = (
                true_airspeed_mps * np.sin(self.maximum_climb_angle_rad) - wind_down_mps
            )
            maximum_descent_rate = (
                true_airspeed_mps * np.sin(self.maximum_descent_angle_rad)
                + wind_down_mps
            )
            clipped_vertical_speed = float(
                np.clip(
                    desired_ground_vertical_speed,
                    -maximum_descent_rate,
                    maximum_climb_rate,
                )
            )
            saturated = not np.isclose(
                clipped_vertical_speed, desired_ground_vertical_speed
            )
            desired_ground_vertical_speed = clipped_vertical_speed
            if saturated:
                mode = "CLIMB" if altitude_error > 0.0 else "DESCENT"
            else:
                mode = "CAPTURE"
        air_vertical_speed = desired_ground_vertical_speed + wind_down_mps
        raw_gamma = float(
            np.arcsin(np.clip(air_vertical_speed / true_airspeed_mps, -0.95, 0.95))
        )
        if self._previous_flight_path_command_rad is None:
            gamma = raw_gamma
        else:
            maximum_change = (
                self.maximum_flight_path_command_rate_rad_s * elapsed_time_s
            )
            gamma = float(
                np.clip(
                    raw_gamma,
                    self._previous_flight_path_command_rad - maximum_change,
                    self._previous_flight_path_command_rad + maximum_change,
                )
            )
        self._previous_flight_path_command_rad = gamma

        speed = base.reference_true_airspeed_mps
        speed_mode = "PROFILE"
        if self.speed_intent is not None:
            if static_temperature_k is None:
                raise ValueError("selected-speed guidance requires static temperature")
            selected_speed = self.speed_intent.target_at(
                simulation_time_s,
                pressure_altitude_m=pressure_altitude_m,
                static_temperature_k=static_temperature_k,
            )
            speed = selected_speed.true_airspeed_mps
            speed_mode = selected_speed.mode
        ground_path_angle = float(
            ground_path_angle_from_air_motion(
                ground_track_rad=base.reference_ground_track_rad,
                air_flight_path_angle_rad=gamma,
                true_airspeed_mps=speed,
                wind_north_mps=wind_north_mps,
                wind_east_mps=wind_east_mps,
                wind_down_mps=wind_down_mps,
            )
        )
        wind_solution = wind_corrected_path(
            ground_track_rad=base.reference_ground_track_rad,
            ground_path_angle_rad=ground_path_angle,
            true_airspeed_mps=speed,
            wind_north_mps=wind_north_mps,
            wind_east_mps=wind_east_mps,
            wind_down_mps=wind_down_mps,
        )
        raw_heading = float(wind_solution.air_heading_rad)
        if self._previous_heading_command_rad is None:
            heading = raw_heading
        else:
            heading_error = float(
                np.arctan2(
                    np.sin(raw_heading - self._previous_heading_command_rad),
                    np.cos(raw_heading - self._previous_heading_command_rad),
                )
            )
            maximum_heading_change = (
                self.lateral_guidance.maximum_heading_command_rate_rad_s
                * elapsed_time_s
            )
            heading = self._previous_heading_command_rad + float(
                np.clip(
                    heading_error,
                    -maximum_heading_change,
                    maximum_heading_change,
                )
            )
            heading = float(np.arctan2(np.sin(heading), np.cos(heading)))
        self._previous_heading_command_rad = heading
        return replace(
            base,
            reference_altitude_m=selected_altitude,
            reference_true_airspeed_mps=speed,
            reference_true_airspeed_rate_mps2=0.0,
            reference_air_flight_path_angle_rad=gamma,
            reference_air_heading_rad=heading,
            predicted_groundspeed_mps=float(wind_solution.horizontal_ground_speed_mps),
            altitude_error_feedback_gain_scale=0.0,
            vertical_mode=mode,
            speed_mode=speed_mode,
        )


__all__ = ["SelectedAltitudeGuidance", "SelectedAltitudeMode"]
