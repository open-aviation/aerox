"""Closed-loop simulation of time-indexed trajectory references."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox.guidance.inverse import InverseFlightMechanicsController
from aerox.guidance.reference import TrajectoryReference
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
    PointMassState,
    PointMassTrajectory,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrackingSimulationResult:
    """Closed-loop generated trajectory and command diagnostics."""

    trajectory: PointMassTrajectory
    reference: TrajectoryReference
    target_lift_coefficient: FloatArray
    pitch_limited: NDArray[np.bool_]
    bank_limited: NDArray[np.bool_]
    thrust_limited: NDArray[np.bool_]


def simulate_reference_tracking_zoh(
    *,
    plant: PointMassModel,
    controller: InverseFlightMechanicsController,
    reference: TrajectoryReference,
    static_temperature_k: ArrayLike,
    initial_mass_kg: float,
    initial_true_airspeed_mps: float | None = None,
    initial_flight_path_angle_rad: float | None = None,
    initial_heading_rad: float | None = None,
    initial_pressure_altitude_m: float | None = None,
    wind_north_mps: ArrayLike | None = None,
    wind_east_mps: ArrayLike | None = None,
    wind_down_mps: ArrayLike | None = None,
    initial_north_position_m: float = 0.0,
    initial_east_position_m: float = 0.0,
    integration_rate_hz: int = 10,
    guidance_plant: PointMassModel | None = None,
    guidance_mass_scale: float = 1.0,
) -> TrackingSimulationResult:
    """Generate and integrate bounded controls at the reference sample rate.

    ``guidance_plant`` defaults to the propagated ``plant``. Supplying a
    distinct model permits controlled model-mismatch studies while preserving
    one physical propagated plant. ``guidance_mass_scale`` changes only the
    mass observed by the guidance calculation; the propagated mass is not
    altered.
    """

    time = reference.time_s
    temperature = np.broadcast_to(
        np.asarray(static_temperature_k, dtype=float), time.shape
    )
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(
            "static_temperature_k must broadcast to finite positive values"
        )
    if not np.isfinite(guidance_mass_scale) or guidance_mass_scale <= 0.0:
        raise ValueError("guidance_mass_scale must be finite and positive")
    command_plant = plant if guidance_plant is None else guidance_plant
    zeros = np.zeros_like(time)

    def wind(values: ArrayLike | None, name: str) -> FloatArray:
        array = np.broadcast_to(
            zeros if values is None else np.asarray(values, dtype=float), time.shape
        ).astype(float, copy=False)
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must broadcast to finite values")
        return array

    north_wind = wind(wind_north_mps, "wind_north_mps")
    east_wind = wind(wind_east_mps, "wind_east_mps")
    down_wind = wind(wind_down_mps, "wind_down_mps")
    state = PointMassState(
        true_airspeed_mps=(
            reference.true_airspeed_mps[0]
            if initial_true_airspeed_mps is None
            else initial_true_airspeed_mps
        ),
        flight_path_angle_rad=(
            reference.flight_path_angle_rad[0]
            if initial_flight_path_angle_rad is None
            else initial_flight_path_angle_rad
        ),
        heading_rad=(
            reference.heading_rad[0]
            if initial_heading_rad is None
            else initial_heading_rad
        ),
        north_position_m=initial_north_position_m,
        east_position_m=initial_east_position_m,
        pressure_altitude_m=(
            reference.pressure_altitude_m[0]
            if initial_pressure_altitude_m is None
            else initial_pressure_altitude_m
        ),
        mass_kg=initial_mass_kg,
    )
    states = np.empty((len(time), 7), dtype=float)
    pitch = np.empty(len(time), dtype=float)
    bank = np.empty(len(time), dtype=float)
    thrust = np.empty(len(time), dtype=float)
    target_cl = np.empty(len(time), dtype=float)
    pitch_limited = np.empty(len(time), dtype=bool)
    bank_limited = np.empty(len(time), dtype=bool)
    thrust_limited = np.empty(len(time), dtype=bool)
    angle = np.empty(len(time), dtype=float)
    lift_coefficient = np.empty(len(time), dtype=float)
    lift = np.empty(len(time), dtype=float)
    drag = np.empty(len(time), dtype=float)
    fuel_flow = np.empty(len(time), dtype=float)
    previous: PitchBankThrustInput | None = None

    for index in range(len(time)):
        states[index] = state.to_array()
        step_s = 1.0 if index == 0 else float(time[index] - time[index - 1])
        guidance_state = replace(
            state,
            mass_kg=state.mass_kg * guidance_mass_scale,
        )
        result = controller.command(
            plant=command_plant,
            state=guidance_state,
            reference_true_airspeed_mps=float(reference.true_airspeed_mps[index]),
            reference_true_airspeed_rate_mps2=float(
                reference.true_airspeed_rate_mps2[index]
            ),
            reference_flight_path_angle_rad=float(
                reference.flight_path_angle_rad[index]
            ),
            reference_flight_path_angle_rate_rad_s=float(
                reference.flight_path_angle_rate_rad_s[index]
            ),
            reference_heading_rad=float(reference.heading_rad[index]),
            reference_heading_rate_rad_s=float(reference.heading_rate_rad_s[index]),
            reference_pressure_altitude_m=float(reference.pressure_altitude_m[index]),
            static_temperature_k=float(temperature[index]),
            previous_control=previous,
            command_step_s=step_s,
            wind_north_mps=float(north_wind[index]),
            wind_east_mps=float(east_wind[index]),
            wind_down_mps=float(down_wind[index]),
        )
        control = result.control
        previous = control
        pitch[index] = control.pitch_angle_rad
        bank[index] = control.bank_angle_rad
        thrust[index] = control.net_thrust_n
        target_cl[index] = result.target_lift_coefficient
        pitch_limited[index] = result.pitch_limited
        bank_limited[index] = result.bank_limited
        thrust_limited[index] = result.thrust_limited
        forces = plant.forces(state, control)
        angle[index] = forces.angle_of_attack_rad
        lift_coefficient[index] = forces.lift_coefficient
        lift[index] = forces.lift_n
        drag[index] = forces.drag_n
        fuel_flow[index] = forces.fuel_flow_kg_s
        if index == len(time) - 1:
            break
        interval = float(time[index + 1] - time[index])
        state = plant.integrate_zoh_step(
            state,
            control,
            interval,
            integration_rate_hz=integration_rate_hz,
        )

    trajectory = PointMassTrajectory(
        time_s=time,
        true_airspeed_mps=states[:, 0],
        flight_path_angle_rad=states[:, 1],
        heading_rad=states[:, 2],
        north_position_m=states[:, 3],
        east_position_m=states[:, 4],
        pressure_altitude_m=states[:, 5],
        mass_kg=states[:, 6],
        pitch_angle_rad=pitch,
        bank_angle_rad=bank,
        angle_of_attack_rad=angle,
        net_thrust_n=thrust,
        lift_coefficient=lift_coefficient,
        lift_n=lift,
        drag_n=drag,
        fuel_flow_kg_s=fuel_flow,
    )
    return TrackingSimulationResult(
        trajectory=trajectory,
        reference=reference,
        target_lift_coefficient=target_cl,
        pitch_limited=pitch_limited,
        bank_limited=bank_limited,
        thrust_limited=thrust_limited,
    )


__all__ = ["TrackingSimulationResult", "simulate_reference_tracking_zoh"]
