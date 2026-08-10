"""Closed-loop composition of FMS targets, guidance, and aircraft dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerox.fms.environment import SpatialEnvironmentProfile
from aerox.fms.performance import MaximumThrustModel, MinimumThrustModel
from aerox.fms.route import SpatialTrajectory
from aerox.guidance.inverse import InverseFlightMechanicsController
from aerox.guidance.path import SpatialPathGuidance
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
    PointMassState,
    PointMassTrajectory,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FMSSimulationResult:
    """Closed-loop plant trajectory plus node-aligned FMS diagnostics."""

    trajectory: PointMassTrajectory
    planned_trajectory: SpatialTrajectory
    progress_m: FloatArray
    cross_track_error_m: FloatArray
    target_along_path_m: FloatArray
    target_north_m: FloatArray
    target_east_m: FloatArray
    reference_altitude_m: FloatArray
    reference_true_airspeed_mps: FloatArray
    reference_ground_track_rad: FloatArray
    reference_air_heading_rad: FloatArray
    static_temperature_k: FloatArray
    wind_north_mps: FloatArray
    wind_east_mps: FloatArray
    wind_down_mps: FloatArray
    target_lift_coefficient: FloatArray
    pitch_limited: NDArray[np.bool_]
    bank_limited: NDArray[np.bool_]
    thrust_limited: NDArray[np.bool_]
    completed: bool

    def waypoint_minimum_distance_m(self) -> FloatArray:
        """Return closest generated horizontal approach to every waypoint."""

        north = self.trajectory.north_position_m
        east = self.trajectory.east_position_m
        return np.array(
            [
                np.min(np.hypot(north - point.north_m, east - point.east_m))
                for point in self.planned_trajectory.intent.waypoints
            ],
            dtype=float,
        )


def simulate_fms_trajectory_zoh(
    *,
    plant: PointMassModel,
    controller: InverseFlightMechanicsController,
    guidance: SpatialPathGuidance,
    maximum_time_s: float,
    static_temperature_k: float,
    initial_mass_kg: float,
    initial_true_airspeed_mps: float | None = None,
    initial_flight_path_angle_rad: float = 0.0,
    initial_heading_rad: float | None = None,
    initial_pressure_altitude_m: float | None = None,
    initial_north_position_m: float | None = None,
    initial_east_position_m: float | None = None,
    wind_north_mps: float = 0.0,
    wind_east_mps: float = 0.0,
    wind_down_mps: float = 0.0,
    command_step_s: float = 1.0,
    integration_rate_hz: int = 10,
    destination_capture_radius_m: float | None = None,
    environment_profile: SpatialEnvironmentProfile | None = None,
    maximum_thrust_model: MaximumThrustModel | None = None,
    minimum_thrust_model: MinimumThrustModel | None = None,
    initial_control: PitchBankThrustInput | None = None,
) -> FMSSimulationResult:
    """Generate FMS references online and propagate the physical plant.

    ``initial_control`` seeds the first command's pitch, bank, and thrust rate
    bounds. It is not propagated before time zero and does not replace the
    controller's first command.
    """

    scalar_values = (
        maximum_time_s,
        static_temperature_k,
        initial_mass_kg,
        wind_north_mps,
        wind_east_mps,
        wind_down_mps,
        command_step_s,
    )
    if any(not np.isfinite(value) for value in scalar_values):
        raise ValueError("simulation scalar inputs must be finite")
    if maximum_time_s <= 0.0 or command_step_s <= 0.0:
        raise ValueError("time horizon and command step must be positive")
    if not isinstance(integration_rate_hz, int) or integration_rate_hz < 1:
        raise ValueError("integration_rate_hz must be a positive integer")
    route = guidance.trajectory
    start = route.intent.waypoints[0]
    destination = route.intent.waypoints[-1]
    capture_radius = (
        destination.acceptance_radius_m
        if destination_capture_radius_m is None
        else destination_capture_radius_m
    )
    if not np.isfinite(capture_radius) or capture_radius <= 0.0:
        raise ValueError("destination capture radius must be finite and positive")
    speed0 = (
        start.true_airspeed_mps
        if initial_true_airspeed_mps is None
        else initial_true_airspeed_mps
    )
    altitude0 = (
        start.altitude_m
        if initial_pressure_altitude_m is None
        else initial_pressure_altitude_m
    )
    north0 = (
        start.north_m if initial_north_position_m is None else initial_north_position_m
    )
    east0 = start.east_m if initial_east_position_m is None else initial_east_position_m
    heading0 = (
        route.segment_track_rad[0]
        if initial_heading_rad is None
        else initial_heading_rad
    )
    state = PointMassState(
        true_airspeed_mps=speed0,
        flight_path_angle_rad=initial_flight_path_angle_rad,
        heading_rad=heading0,
        north_position_m=north0,
        east_position_m=east0,
        pressure_altitude_m=altitude0,
        mass_kg=initial_mass_kg,
    )
    initial_projection = route.project(north0, east0)
    guidance.reset(initial_projection.along_path_m)
    maximum_nodes = int(np.floor(maximum_time_s / command_step_s)) + 1
    time_values: list[float] = []
    states: list[FloatArray] = []
    pitch: list[float] = []
    bank: list[float] = []
    thrust: list[float] = []
    angle: list[float] = []
    lift_coefficient: list[float] = []
    lift: list[float] = []
    drag: list[float] = []
    fuel_flow: list[float] = []
    progress: list[float] = []
    cross_track: list[float] = []
    target_s: list[float] = []
    target_north: list[float] = []
    target_east: list[float] = []
    reference_altitude: list[float] = []
    reference_speed: list[float] = []
    reference_track: list[float] = []
    reference_heading: list[float] = []
    temperature_history: list[float] = []
    wind_north_history: list[float] = []
    wind_east_history: list[float] = []
    wind_down_history: list[float] = []
    target_cl: list[float] = []
    pitch_limited: list[bool] = []
    bank_limited: list[bool] = []
    thrust_limited: list[bool] = []
    previous_control = initial_control
    completed = False

    for index in range(maximum_nodes):
        time_values.append(index * command_step_s)
        states.append(state.to_array())
        if environment_profile is None:
            current_temperature_k = static_temperature_k
            current_wind_north_mps = wind_north_mps
            current_wind_east_mps = wind_east_mps
            current_wind_down_mps = wind_down_mps
        else:
            current_projection = guidance.project_position(
                north_m=state.north_position_m,
                east_m=state.east_position_m,
            )
            environment_distance = max(
                guidance.progress_m, current_projection.along_path_m
            )
            environment = environment_profile.sample(environment_distance)
            current_temperature_k = environment.static_temperature_k
            current_wind_north_mps = environment.wind_north_mps
            current_wind_east_mps = environment.wind_east_mps
            current_wind_down_mps = environment.wind_down_mps
        target = guidance.update(
            north_m=state.north_position_m,
            east_m=state.east_position_m,
            true_airspeed_mps=state.true_airspeed_mps,
            flight_path_angle_rad=state.flight_path_angle_rad,
            elapsed_time_s=command_step_s,
            wind_north_mps=current_wind_north_mps,
            wind_east_mps=current_wind_east_mps,
            wind_down_mps=current_wind_down_mps,
        )
        command = controller.command(
            plant=plant,
            state=state,
            reference_true_airspeed_mps=target.reference_true_airspeed_mps,
            reference_true_airspeed_rate_mps2=target.reference_true_airspeed_rate_mps2,
            reference_flight_path_angle_rad=(
                target.reference_air_flight_path_angle_rad
            ),
            # The moving spatial target is a state reference, not a measured
            # trajectory derivative. Differentiating it injects corner and
            # segment-transition impulses, so feedback performs capture while
            # the spatial speed gradient remains analytic feed-forward.
            reference_flight_path_angle_rate_rad_s=0.0,
            reference_heading_rad=target.reference_air_heading_rad,
            reference_heading_rate_rad_s=0.0,
            reference_pressure_altitude_m=target.reference_altitude_m,
            static_temperature_k=current_temperature_k,
            previous_control=previous_control,
            command_step_s=command_step_s,
            wind_north_mps=current_wind_north_mps,
            wind_east_mps=current_wind_east_mps,
            wind_down_mps=current_wind_down_mps,
        )
        control = command.control
        conditioned_thrust_limited = False
        conditioned_minimum_thrust = (
            minimum_thrust_model.minimum_thrust_n(
                pressure_altitude_m=state.pressure_altitude_m,
                true_airspeed_mps=state.true_airspeed_mps,
                static_temperature_k=current_temperature_k,
            )
            if minimum_thrust_model is not None
            else 0.0
        )
        if (
            not np.isfinite(conditioned_minimum_thrust)
            or conditioned_minimum_thrust < 0.0
        ):
            raise ValueError(
                "conditioned minimum thrust must be finite and non-negative"
            )
        conditioned_maximum_thrust = np.inf
        if maximum_thrust_model is not None:
            conditioned_maximum_thrust = maximum_thrust_model.maximum_thrust_n(
                pressure_altitude_m=state.pressure_altitude_m,
                true_airspeed_mps=state.true_airspeed_mps,
                static_temperature_k=current_temperature_k,
            )
            if (
                not np.isfinite(conditioned_maximum_thrust)
                or conditioned_maximum_thrust <= 0.0
            ):
                raise ValueError(
                    "conditioned maximum thrust must be finite and positive"
                )
        if conditioned_minimum_thrust >= conditioned_maximum_thrust:
            raise ValueError("conditioned maximum thrust must exceed minimum thrust")
        bounded_thrust = float(
            np.clip(
                control.net_thrust_n,
                conditioned_minimum_thrust,
                conditioned_maximum_thrust,
            )
        )
        if not np.isclose(bounded_thrust, control.net_thrust_n):
            control = PitchBankThrustInput(
                pitch_angle_rad=control.pitch_angle_rad,
                bank_angle_rad=control.bank_angle_rad,
                net_thrust_n=bounded_thrust,
                static_temperature_k=control.static_temperature_k,
                wind_north_mps=control.wind_north_mps,
                wind_east_mps=control.wind_east_mps,
                wind_down_mps=control.wind_down_mps,
            )
            conditioned_thrust_limited = True
        forces = plant.forces(state, control)
        pitch.append(control.pitch_angle_rad)
        bank.append(control.bank_angle_rad)
        thrust.append(control.net_thrust_n)
        angle.append(forces.angle_of_attack_rad)
        lift_coefficient.append(forces.lift_coefficient)
        lift.append(forces.lift_n)
        drag.append(forces.drag_n)
        fuel_flow.append(forces.fuel_flow_kg_s)
        progress.append(target.progress_m)
        cross_track.append(target.projection.cross_track_error_m)
        target_s.append(target.target_along_path_m)
        target_north.append(target.target_north_m)
        target_east.append(target.target_east_m)
        reference_altitude.append(target.reference_altitude_m)
        reference_speed.append(target.reference_true_airspeed_mps)
        reference_track.append(target.reference_ground_track_rad)
        reference_heading.append(target.reference_air_heading_rad)
        temperature_history.append(current_temperature_k)
        wind_north_history.append(current_wind_north_mps)
        wind_east_history.append(current_wind_east_mps)
        wind_down_history.append(current_wind_down_mps)
        target_cl.append(command.target_lift_coefficient)
        pitch_limited.append(command.pitch_limited)
        bank_limited.append(command.bank_limited)
        thrust_limited.append(command.thrust_limited or conditioned_thrust_limited)
        previous_control = control

        destination_distance = float(
            np.hypot(
                state.north_position_m - destination.north_m,
                state.east_position_m - destination.east_m,
            )
        )
        if destination_distance <= capture_radius and target.progress_m >= (
            route.total_distance_m - capture_radius
        ):
            completed = True
            break
        if index == maximum_nodes - 1:
            break
        state = plant.integrate_zoh_step(
            state,
            control,
            command_step_s,
            integration_rate_hz=integration_rate_hz,
        )

    state_array = np.asarray(states)
    trajectory = PointMassTrajectory(
        time_s=np.asarray(time_values),
        true_airspeed_mps=state_array[:, 0],
        flight_path_angle_rad=state_array[:, 1],
        heading_rad=state_array[:, 2],
        north_position_m=state_array[:, 3],
        east_position_m=state_array[:, 4],
        pressure_altitude_m=state_array[:, 5],
        mass_kg=state_array[:, 6],
        pitch_angle_rad=np.asarray(pitch),
        bank_angle_rad=np.asarray(bank),
        angle_of_attack_rad=np.asarray(angle),
        net_thrust_n=np.asarray(thrust),
        lift_coefficient=np.asarray(lift_coefficient),
        lift_n=np.asarray(lift),
        drag_n=np.asarray(drag),
        fuel_flow_kg_s=np.asarray(fuel_flow),
    )
    return FMSSimulationResult(
        trajectory=trajectory,
        planned_trajectory=route,
        progress_m=np.asarray(progress),
        cross_track_error_m=np.asarray(cross_track),
        target_along_path_m=np.asarray(target_s),
        target_north_m=np.asarray(target_north),
        target_east_m=np.asarray(target_east),
        reference_altitude_m=np.asarray(reference_altitude),
        reference_true_airspeed_mps=np.asarray(reference_speed),
        reference_ground_track_rad=np.asarray(reference_track),
        reference_air_heading_rad=np.asarray(reference_heading),
        static_temperature_k=np.asarray(temperature_history),
        wind_north_mps=np.asarray(wind_north_history),
        wind_east_mps=np.asarray(wind_east_history),
        wind_down_mps=np.asarray(wind_down_history),
        target_lift_coefficient=np.asarray(target_cl),
        pitch_limited=np.asarray(pitch_limited, dtype=bool),
        bank_limited=np.asarray(bank_limited, dtype=bool),
        thrust_limited=np.asarray(thrust_limited, dtype=bool),
        completed=completed,
    )


__all__ = ["FMSSimulationResult", "simulate_fms_trajectory_zoh"]
