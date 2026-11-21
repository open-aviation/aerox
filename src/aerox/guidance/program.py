"""Online execution of intent-driven guidance programs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerox.fms.guidance_program import (
    GuardMetric,
    GuidanceObservation,
    GuidanceProgram,
    GuidanceProgramExecutor,
    GuidanceSegment,
    LateralMode,
    SpeedCommand,
    SpeedMode,
    ThrustMode,
    VerticalCommand,
    VerticalMode,
)
from aerox.fms.route import RouteProjection, SpatialTrajectory
from aerox.fms.selected_intent import (
    SelectedAltitudeIntent,
    SelectedConfigurationIntent,
    SelectedSpeedGuidanceIntent,
    SelectedSpeedIntent,
    SelectedThrustIntent,
    SelectedVerticalIntent,
)
from aerox.guidance.path import GuidanceTarget, SpatialPathGuidance
from aerox.guidance.selected_altitude import SelectedAltitudeGuidance


def _single_command_intents(
    segment: GuidanceSegment,
    speed_command: SpeedCommand | None = None,
    vertical_command: VerticalCommand | None = None,
) -> tuple[
    SelectedAltitudeIntent | None,
    SelectedVerticalIntent | None,
    SelectedSpeedIntent,
    SelectedSpeedGuidanceIntent,
    SelectedThrustIntent,
    SelectedConfigurationIntent,
]:
    time = np.asarray([0.0])
    speed_rule = segment.speed if speed_command is None else speed_command
    vertical_rule = segment.vertical if vertical_command is None else vertical_command
    altitude = None
    vertical = None
    if vertical_rule.mode in {VerticalMode.ALTITUDE, VerticalMode.ALTITUDE_HOLD}:
        altitude = SelectedAltitudeIntent(
            time,
            np.asarray([vertical_rule.altitude_m], dtype=float),
        )
    else:
        vertical = SelectedVerticalIntent(
            time,
            (vertical_rule.mode.value,),
            np.asarray(
                [
                    np.nan
                    if vertical_rule.altitude_m is None
                    else vertical_rule.altitude_m
                ]
            ),
            np.asarray(
                [
                    np.nan
                    if vertical_rule.vertical_speed_mps is None
                    else vertical_rule.vertical_speed_mps
                ]
            ),
            np.asarray(
                [
                    np.nan
                    if vertical_rule.flight_path_angle_rad is None
                    else vertical_rule.flight_path_angle_rad
                ]
            ),
        )
    return (
        altitude,
        vertical,
        SelectedSpeedIntent(
            time,
            np.asarray([speed_rule.calibrated_airspeed_mps], dtype=float),
            np.asarray([speed_rule.mach], dtype=float),
        ),
        SelectedSpeedGuidanceIntent(
            time,
            (speed_rule.mode.value,),
            (
                None
                if speed_rule.active_selector is None
                else speed_rule.active_selector.value,
            ),
        ),
        SelectedThrustIntent(
            time,
            (segment.thrust.mode.value,),
            np.asarray(
                [
                    np.nan
                    if segment.thrust.available_thrust_fraction is None
                    else segment.thrust.available_thrust_fraction
                ]
            ),
        ),
        SelectedConfigurationIntent(
            time,
            (segment.configuration.mode.value,),
            np.asarray([segment.configuration.drag_coefficient_increment]),
        ),
    )


def _validate_supported_program(program: GuidanceProgram) -> None:
    for segment in program.segments:
        if segment.lateral.mode != LateralMode.ROUTE:
            raise ValueError("online guidance currently supports ROUTE lateral mode")
        if segment.speed.mode == SpeedMode.PROFILE:
            raise ValueError("online guidance requires explicit CAS/Mach selectors")
        if segment.thrust.mode not in {
            ThrustMode.AUTO,
            ThrustMode.THRUST_FRACTION,
            ThrustMode.THRUST_CEILING_FRACTION,
        }:
            raise ValueError(
                "online guidance supports AUTO and normalized thrust-fraction modes"
            )
    if any(stage.command.mode == SpeedMode.PROFILE for stage in program.speed_stages):
        raise ValueError("online guidance requires explicit speed-stage selectors")


@dataclass
class IntentDrivenGuidance:
    """Generate a trajectory by executing flight intent against generated state."""

    program: GuidanceProgram
    lateral_guidance: SpatialPathGuidance
    capture_time_constant_s: float = 90.0
    altitude_hold_deadband_m: float = 15.24
    maximum_climb_angle_rad: float | None = None
    maximum_descent_angle_rad: float | None = None
    maximum_flight_path_command_rate_rad_s: float = np.deg2rad(0.5)
    executor: GuidanceProgramExecutor = field(init=False)
    _selected_guidance: SelectedAltitudeGuidance = field(init=False, repr=False)
    _active_segment_history: list[int] = field(
        default_factory=list, init=False, repr=False
    )
    _active_speed_stage_history: list[int] = field(
        default_factory=list, init=False, repr=False
    )
    _active_vertical_stage_history: list[int] = field(
        default_factory=list, init=False, repr=False
    )
    _requires_speed_error: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.program, GuidanceProgram):
            raise TypeError("program must be a GuidanceProgram")
        if not isinstance(self.lateral_guidance, SpatialPathGuidance):
            raise TypeError("lateral_guidance must be SpatialPathGuidance")
        _validate_supported_program(self.program)
        self._requires_speed_error = (
            any(
                condition.metric == GuardMetric.SPEED_ERROR_MPS
                for segment in self.program.segments[:-1]
                for condition in segment.transition.conditions
            )
            or any(
                condition.metric == GuardMetric.SPEED_ERROR_MPS
                for stage in self.program.speed_stages[:-1]
                for condition in stage.transition.conditions
            )
            or any(
                condition.metric == GuardMetric.SPEED_ERROR_MPS
                for stage in self.program.vertical_stages[:-1]
                for condition in stage.transition.conditions
            )
        )
        self.executor = GuidanceProgramExecutor(self.program)
        (
            altitude,
            vertical,
            speed,
            speed_guidance,
            thrust,
            configuration,
        ) = _single_command_intents(
            self.executor.current_segment,
            self.executor.current_speed,
            self.executor.current_vertical,
        )
        self._selected_guidance = SelectedAltitudeGuidance(
            lateral_guidance=self.lateral_guidance,
            altitude_intent=altitude,
            speed_intent=speed,
            speed_guidance_intent=speed_guidance,
            thrust_intent=thrust,
            configuration_intent=configuration,
            vertical_intent=vertical,
            capture_time_constant_s=self.capture_time_constant_s,
            altitude_hold_deadband_m=self.altitude_hold_deadband_m,
            maximum_climb_angle_rad=self.maximum_climb_angle_rad,
            maximum_descent_angle_rad=self.maximum_descent_angle_rad,
            maximum_flight_path_command_rate_rad_s=(
                self.maximum_flight_path_command_rate_rad_s
            ),
        )

    @property
    def trajectory(self) -> SpatialTrajectory:
        return self.lateral_guidance.trajectory

    @property
    def progress_m(self) -> float:
        return self.lateral_guidance.progress_m

    @property
    def active_segment_history(self) -> tuple[int, ...]:
        """Return the active program segment used at every update since reset."""

        return tuple(self._active_segment_history)

    @property
    def active_speed_stage_history(self) -> tuple[int, ...]:
        """Return the active speed stage used at every update since reset."""

        return tuple(self._active_speed_stage_history)

    @property
    def active_vertical_stage_history(self) -> tuple[int, ...]:
        """Return the active vertical stage used at every update since reset."""

        return tuple(self._active_vertical_stage_history)

    def reset(self, progress_m: float = 0.0) -> None:
        self.executor.reset()
        self._active_segment_history.clear()
        self._active_speed_stage_history.clear()
        self._active_vertical_stage_history.clear()
        self._activate(
            self.executor.current_segment,
            self.executor.current_speed,
            self.executor.current_vertical,
        )
        self._selected_guidance.reset(progress_m)

    def project_position(self, *, north_m: float, east_m: float) -> RouteProjection:
        return self.lateral_guidance.project_position(
            north_m=north_m,
            east_m=east_m,
        )

    def _activate(
        self,
        segment: GuidanceSegment,
        speed_command: SpeedCommand,
        vertical_command: VerticalCommand,
    ) -> None:
        (
            altitude,
            vertical,
            speed,
            speed_guidance,
            thrust,
            configuration,
        ) = _single_command_intents(segment, speed_command, vertical_command)
        self._selected_guidance.vertical_intent = vertical
        self._selected_guidance.altitude_intent = altitude
        self._selected_guidance.speed_intent = speed
        self._selected_guidance.speed_guidance_intent = speed_guidance
        self._selected_guidance.thrust_intent = thrust
        self._selected_guidance.configuration_intent = configuration

    def _observation(
        self,
        *,
        north_m: float,
        east_m: float,
        true_airspeed_mps: float,
        pressure_altitude_m: float,
        simulation_time_s: float,
        static_temperature_k: float,
    ) -> GuidanceObservation:
        projection = self.project_position(north_m=north_m, east_m=east_m)
        along_path = min(
            self.trajectory.total_distance_m,
            max(self.progress_m, projection.along_path_m),
        )
        segment = self.executor.current_segment
        vertical_command = self.executor.current_vertical
        speed_error = None
        if self._requires_speed_error:
            _, _, speed, _, _, _ = _single_command_intents(
                segment,
                self.executor.current_speed,
                vertical_command,
            )
            target = speed.target_at(
                0.0,
                pressure_altitude_m=pressure_altitude_m,
                static_temperature_k=static_temperature_k,
            )
            speed_error = target.true_airspeed_mps - true_airspeed_mps
        return GuidanceObservation(
            elapsed_time_s=simulation_time_s,
            pressure_altitude_m=pressure_altitude_m,
            along_path_m=along_path,
            distance_to_go_m=self.trajectory.total_distance_m - along_path,
            waypoint_index=self.trajectory.segment_index_at(along_path),
            altitude_error_m=(
                None
                if vertical_command.altitude_m is None
                else vertical_command.altitude_m - pressure_altitude_m
            ),
            speed_error_mps=speed_error,
        )

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
        """Evaluate transitions and generate references from the active segment."""

        if (
            pressure_altitude_m is None
            or simulation_time_s is None
            or static_temperature_k is None
        ):
            raise ValueError(
                "intent-driven guidance requires altitude, time, and temperature"
            )
        previous = self.executor.segment_index
        previous_speed = self.executor.speed_stage_index
        previous_vertical = self.executor.vertical_stage_index
        active = self.executor.update(
            self._observation(
                north_m=north_m,
                east_m=east_m,
                true_airspeed_mps=true_airspeed_mps,
                pressure_altitude_m=pressure_altitude_m,
                simulation_time_s=simulation_time_s,
                static_temperature_k=static_temperature_k,
            )
        )
        if (
            self.executor.segment_index != previous
            or self.executor.speed_stage_index != previous_speed
            or self.executor.vertical_stage_index != previous_vertical
        ):
            self._activate(
                active,
                self.executor.current_speed,
                self.executor.current_vertical,
            )
        self._active_segment_history.append(self.executor.segment_index)
        self._active_speed_stage_history.append(self.executor.speed_stage_index)
        self._active_vertical_stage_history.append(self.executor.vertical_stage_index)
        return self._selected_guidance.update(
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


# Compatibility name retained for callers written before the intent terminology
# was introduced. Both names refer to the same implementation.
StateDrivenGuidance = IntentDrivenGuidance


__all__ = ["IntentDrivenGuidance", "StateDrivenGuidance"]
