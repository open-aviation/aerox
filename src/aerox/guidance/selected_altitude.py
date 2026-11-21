"""Online vertical, speed, thrust, and configuration guidance."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from aerox.fms.selected_intent import (
    SelectedAltitudeIntent,
    SelectedConfigurationIntent,
    SelectedSpeedGuidanceIntent,
    SelectedSpeedGuidanceMode,
    SelectedSpeedIntent,
    SelectedSpeedMode,
    SelectedThrustIntent,
    SelectedVerticalIntent,
)
from aerox.fms.route import RouteProjection, SpatialTrajectory
from aerox.guidance.path import GuidanceTarget, SpatialPathGuidance, SpeedGuidanceMode
from aerox.navigation import ground_path_angle_from_air_motion, wind_corrected_path

SelectedAltitudeMode = Literal[
    "PATH",
    "CLIMB",
    "DESCENT",
    "CAPTURE",
    "ALT_HOLD",
    "VERTICAL_SPEED",
    "FLIGHT_PATH_ANGLE",
]
VerticalDirection = Literal["CLIMB", "LEVEL", "DESCENT"]


@dataclass
class SelectedAltitudeGuidance:
    """Execute concurrent selected and direct vertical guidance commands."""

    lateral_guidance: SpatialPathGuidance
    altitude_intent: SelectedAltitudeIntent | None
    speed_intent: SelectedSpeedIntent | None = None
    speed_guidance_intent: SelectedSpeedGuidanceIntent | None = None
    thrust_intent: SelectedThrustIntent | None = None
    configuration_intent: SelectedConfigurationIntent | None = None
    vertical_intent: SelectedVerticalIntent | None = None
    capture_time_constant_s: float = 90.0
    altitude_hold_deadband_m: float = 15.24
    maximum_climb_angle_rad: float | None = None
    maximum_descent_angle_rad: float | None = None
    maximum_flight_path_command_rate_rad_s: float = np.deg2rad(0.5)
    _previous_flight_path_command_rad: float | None = field(
        default=None, init=False, repr=False
    )
    _previous_heading_command_rad: float | None = field(
        default=None, init=False, repr=False
    )
    _active_speed_mode: SelectedSpeedMode | None = field(
        default=None, init=False, repr=False
    )
    _speed_guidance_mode: SelectedSpeedGuidanceMode | None = field(
        default=None, init=False, repr=False
    )
    _previous_speed_target_mps: float | None = field(
        default=None, init=False, repr=False
    )
    _previous_speed_reference_mode: SelectedSpeedMode | None = field(
        default=None, init=False, repr=False
    )
    _previous_active_speed_command: float | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        values = (
            self.capture_time_constant_s,
            self.altitude_hold_deadband_m,
            self.maximum_flight_path_command_rate_rad_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("selected-altitude guidance settings must be positive")
        for angle in (
            self.maximum_climb_angle_rad,
            self.maximum_descent_angle_rad,
        ):
            if angle is not None and (
                not np.isfinite(angle) or angle <= 0.0 or angle >= np.deg2rad(30.0)
            ):
                raise ValueError(
                    "configured flight-path limits must be between 0 and 30 degrees"
                )
        if self.speed_guidance_intent is not None and self.speed_intent is None:
            raise ValueError("speed-guidance intent requires selected-speed intent")
        if self.altitude_intent is None and self.vertical_intent is None:
            raise ValueError("selected guidance requires one vertical intent")

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
        self._active_speed_mode = None
        self._speed_guidance_mode = None
        self._previous_speed_target_mps = None
        self._previous_speed_reference_mode = None
        self._previous_active_speed_command = None

    def project_position(self, *, north_m: float, east_m: float) -> RouteProjection:
        return self.lateral_guidance.project_position(north_m=north_m, east_m=east_m)

    def _selected_speed_modes(
        self,
        *,
        vertical_direction: VerticalDirection,
        pressure_altitude_m: float,
        crossover_altitude_m: float,
        commanded_mode: SelectedSpeedGuidanceMode | None = None,
        commanded_selector: SelectedSpeedMode | None = None,
    ) -> tuple[SpeedGuidanceMode, SelectedSpeedMode]:
        """Select a performance mode and its one-way active speed branch."""

        guidance_mode = commanded_mode
        if guidance_mode is None:
            if vertical_direction == "CLIMB":
                guidance_mode = "CAS_MACH"
            elif vertical_direction == "DESCENT":
                guidance_mode = "MACH_CAS"
            else:
                guidance_mode = "SPEED_HOLD"

        mode_changed = guidance_mode != self._speed_guidance_mode
        if commanded_selector is not None:
            self._active_speed_mode = commanded_selector
        elif guidance_mode == "CAS":
            self._active_speed_mode = "CAS"
        elif guidance_mode == "MACH":
            self._active_speed_mode = "MACH"
        elif self._active_speed_mode is None:
            self._active_speed_mode = (
                "MACH" if pressure_altitude_m >= crossover_altitude_m else "CAS"
            )
        elif mode_changed and guidance_mode == "CAS_MACH":
            self._active_speed_mode = (
                "CAS" if pressure_altitude_m < crossover_altitude_m else "MACH"
            )
        elif mode_changed and guidance_mode == "MACH_CAS":
            self._active_speed_mode = (
                "MACH" if pressure_altitude_m > crossover_altitude_m else "CAS"
            )
        elif (
            guidance_mode == "CAS_MACH"
            and self._active_speed_mode == "CAS"
            and pressure_altitude_m >= crossover_altitude_m
        ):
            self._active_speed_mode = "MACH"
        elif (
            guidance_mode == "MACH_CAS"
            and self._active_speed_mode == "MACH"
            and pressure_altitude_m <= crossover_altitude_m
        ):
            self._active_speed_mode = "CAS"
        self._speed_guidance_mode = guidance_mode
        performance_mode: SpeedGuidanceMode = (
            "SPEED_HOLD" if guidance_mode in {"CAS", "MACH"} else guidance_mode
        )
        return performance_mode, self._active_speed_mode

    def _speed_reference_rate(
        self,
        *,
        speed_target_mps: float,
        active_command: float,
        active_mode: SelectedSpeedMode,
        elapsed_time_s: float,
    ) -> float:
        """Differentiate an unchanged CAS/Mach target through the atmosphere."""

        rate = 0.0
        if (
            self._previous_speed_target_mps is not None
            and self._previous_speed_reference_mode == active_mode
            and self._previous_active_speed_command is not None
            and np.isclose(
                active_command,
                self._previous_active_speed_command,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            rate = (speed_target_mps - self._previous_speed_target_mps) / elapsed_time_s
        self._previous_speed_target_mps = speed_target_mps
        self._previous_speed_reference_mode = active_mode
        self._previous_active_speed_command = active_command
        return float(rate)

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
        """Return one concurrent lateral, vertical, speed, and energy target."""

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
        direct_vertical = (
            None
            if self.vertical_intent is None
            else self.vertical_intent.target_at(simulation_time_s)
        )
        altitude_feedback_gain_scale = 0.0
        if direct_vertical is not None and direct_vertical.mode == "PATH":
            selected_altitude = base.reference_altitude_m
            raw_gamma = base.reference_air_flight_path_angle_rad
            desired_ground_vertical_speed = (
                true_airspeed_mps * np.sin(raw_gamma) - wind_down_mps
            )
            mode: SelectedAltitudeMode = "PATH"
            altitude_feedback_gain_scale = base.altitude_error_feedback_gain_scale
            vertical_direction = (
                "CLIMB"
                if desired_ground_vertical_speed > 1e-6
                else "DESCENT"
                if desired_ground_vertical_speed < -1e-6
                else "LEVEL"
            )
        elif direct_vertical is not None and direct_vertical.mode in {
            "VERTICAL_SPEED",
            "FLIGHT_PATH_ANGLE",
        }:
            selected_altitude = pressure_altitude_m
            if direct_vertical.mode == "VERTICAL_SPEED":
                if direct_vertical.vertical_speed_mps is None:
                    raise ValueError("vertical-speed guidance target is missing")
                desired_ground_vertical_speed = direct_vertical.vertical_speed_mps
                air_vertical_speed = desired_ground_vertical_speed + wind_down_mps
                raw_gamma = float(
                    np.arcsin(
                        np.clip(
                            air_vertical_speed / true_airspeed_mps,
                            -0.95,
                            0.95,
                        )
                    )
                )
                mode = "VERTICAL_SPEED"
            else:
                if direct_vertical.flight_path_angle_rad is None:
                    raise ValueError("flight-path-angle guidance target is missing")
                raw_gamma = direct_vertical.flight_path_angle_rad
                desired_ground_vertical_speed = (
                    true_airspeed_mps * np.sin(raw_gamma) - wind_down_mps
                )
                mode = "FLIGHT_PATH_ANGLE"
            vertical_direction = (
                "CLIMB"
                if desired_ground_vertical_speed > 1e-6
                else "DESCENT"
                if desired_ground_vertical_speed < -1e-6
                else "LEVEL"
            )
        else:
            if direct_vertical is not None:
                if direct_vertical.altitude_m is None:
                    raise ValueError("altitude guidance target is missing")
                selected_altitude = direct_vertical.altitude_m
            elif self.altitude_intent is not None:
                selected_altitude = self.altitude_intent.altitude_at(simulation_time_s)
            else:
                raise ValueError("active vertical guidance target is missing")
            altitude_error = selected_altitude - pressure_altitude_m
            if abs(altitude_error) <= self.altitude_hold_deadband_m:
                desired_ground_vertical_speed = 0.0
                mode = "ALT_HOLD"
                vertical_direction = "LEVEL"
            else:
                vertical_direction = "CLIMB" if altitude_error > 0.0 else "DESCENT"
                desired_ground_vertical_speed = (
                    altitude_error / self.capture_time_constant_s
                )
                maximum_climb_rate = np.inf
                if self.maximum_climb_angle_rad is not None:
                    maximum_climb_rate = (
                        true_airspeed_mps * np.sin(self.maximum_climb_angle_rad)
                        - wind_down_mps
                    )
                maximum_descent_rate = np.inf
                if self.maximum_descent_angle_rad is not None:
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
                    current_ground_vertical_speed = (
                        true_airspeed_mps * np.sin(flight_path_angle_rad)
                        - wind_down_mps
                    )
                    closing_speed = (
                        np.sign(altitude_error) * current_ground_vertical_speed
                    )
                    time_to_target_s = (
                        abs(altitude_error) / closing_speed
                        if closing_speed > 0.0
                        else np.inf
                    )
                    mode = (
                        "CAPTURE"
                        if time_to_target_s <= self.capture_time_constant_s
                        else vertical_direction
                    )
            air_vertical_speed = desired_ground_vertical_speed + wind_down_mps
            raw_gamma = float(
                np.arcsin(np.clip(air_vertical_speed / true_airspeed_mps, -0.95, 0.95))
            )
        previous_gamma = (
            flight_path_angle_rad
            if self._previous_flight_path_command_rad is None
            else self._previous_flight_path_command_rad
        )
        maximum_change = self.maximum_flight_path_command_rate_rad_s * elapsed_time_s
        gamma = float(
            np.clip(
                raw_gamma,
                previous_gamma - maximum_change,
                previous_gamma + maximum_change,
            )
        )
        self._previous_flight_path_command_rad = gamma

        speed = base.reference_true_airspeed_mps
        speed_rate = base.reference_true_airspeed_rate_mps2
        speed_guidance_mode: SpeedGuidanceMode = "PROFILE"
        speed_mode = "PROFILE"
        if self.speed_intent is not None:
            if static_temperature_k is None:
                raise ValueError("selected-speed guidance requires static temperature")
            selector_pair = self.speed_intent.target_at(
                simulation_time_s,
                pressure_altitude_m=pressure_altitude_m,
                static_temperature_k=static_temperature_k,
            )
            speed_guidance_mode, active_mode = self._selected_speed_modes(
                vertical_direction=vertical_direction,
                pressure_altitude_m=pressure_altitude_m,
                crossover_altitude_m=selector_pair.crossover_altitude_m,
                commanded_mode=(
                    None
                    if self.speed_guidance_intent is None
                    else self.speed_guidance_intent.mode_at(simulation_time_s)
                ),
                commanded_selector=(
                    None
                    if self.speed_guidance_intent is None
                    else self.speed_guidance_intent.active_selector_at(
                        simulation_time_s
                    )
                ),
            )
            selected_speed = self.speed_intent.target_at(
                simulation_time_s,
                pressure_altitude_m=pressure_altitude_m,
                static_temperature_k=static_temperature_k,
                mode=active_mode,
            )
            speed = selected_speed.true_airspeed_mps
            speed_mode = selected_speed.mode
            active_command = (
                selected_speed.calibrated_airspeed_mps
                if active_mode == "CAS"
                else selected_speed.mach
            )
            speed_rate = self._speed_reference_rate(
                speed_target_mps=speed,
                active_command=active_command,
                active_mode=active_mode,
                elapsed_time_s=elapsed_time_s,
            )
        thrust_mode = "AUTO"
        available_thrust_fraction = None
        if self.thrust_intent is not None:
            thrust_target = self.thrust_intent.target_at(simulation_time_s)
            thrust_mode = thrust_target.mode
            available_thrust_fraction = thrust_target.available_thrust_fraction
        configuration_mode = "CLEAN"
        drag_coefficient_increment = 0.0
        if self.configuration_intent is not None:
            configuration_target = self.configuration_intent.target_at(
                simulation_time_s
            )
            configuration_mode = configuration_target.mode
            drag_coefficient_increment = configuration_target.drag_coefficient_increment
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
            reference_true_airspeed_rate_mps2=speed_rate,
            reference_air_flight_path_angle_rad=gamma,
            reference_air_heading_rad=heading,
            predicted_groundspeed_mps=float(wind_solution.horizontal_ground_speed_mps),
            altitude_error_feedback_gain_scale=altitude_feedback_gain_scale,
            vertical_mode=mode,
            speed_guidance_mode=speed_guidance_mode,
            speed_mode=speed_mode,
            thrust_mode=thrust_mode,
            available_thrust_fraction=available_thrust_fraction,
            configuration_mode=configuration_mode,
            drag_coefficient_increment=drag_coefficient_increment,
        )


__all__ = [
    "SelectedAltitudeGuidance",
    "SelectedAltitudeMode",
    "SelectedSpeedGuidanceMode",
]
