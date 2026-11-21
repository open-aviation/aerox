"""Numerical simulation of the pitch-bank-thrust point-mass model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .point_mass import (
    PitchBankThrustInput,
    PointMassModel,
    PointMassState,
    PointMassTrajectory,
)

FloatArray = NDArray[np.float64]


def _sample_array(values: ArrayLike, time: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape != time.shape or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector matching time_s")
    return array


def _simulate_point_mass_zoh(
    *,
    plant: PointMassModel,
    observed_heading_rate_rad_s: ArrayLike | None,
    time_s: ArrayLike,
    pitch_angle_rad: ArrayLike,
    bank_angle_rad: ArrayLike,
    net_thrust_n: ArrayLike,
    static_temperature_k: ArrayLike,
    initial_true_airspeed_mps: float,
    initial_flight_path_angle_rad: float,
    initial_heading_rad: float,
    initial_pressure_altitude_m: float,
    initial_mass_kg: float,
    wind_north_mps: ArrayLike | None = None,
    wind_east_mps: ArrayLike | None = None,
    wind_down_mps: ArrayLike | None = None,
    drag_coefficient_increment: ArrayLike | None = None,
    initial_north_position_m: float = 0.0,
    initial_east_position_m: float = 0.0,
    integration_rate_hz: int = 10,
) -> PointMassTrajectory:
    """Integrate the model, optionally overriding heading rate for replay."""

    time = np.asarray(time_s, dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or np.any(~np.isfinite(time))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("time_s must be a finite increasing vector")
    if integration_rate_hz < 1:
        raise ValueError("integration_rate_hz must be a positive integer")
    pitch = _sample_array(pitch_angle_rad, time, "pitch_angle_rad")
    bank = _sample_array(bank_angle_rad, time, "bank_angle_rad")
    thrust = _sample_array(net_thrust_n, time, "net_thrust_n")
    temperature = _sample_array(static_temperature_k, time, "static_temperature_k")
    observed_heading_rate = (
        None
        if observed_heading_rate_rad_s is None
        else _sample_array(
            observed_heading_rate_rad_s,
            time,
            "observed_heading_rate_rad_s",
        )
    )
    zeros = np.zeros_like(time)
    wind_north = _sample_array(
        zeros if wind_north_mps is None else wind_north_mps,
        time,
        "wind_north_mps",
    )
    wind_east = _sample_array(
        zeros if wind_east_mps is None else wind_east_mps,
        time,
        "wind_east_mps",
    )
    wind_down = _sample_array(
        zeros if wind_down_mps is None else wind_down_mps,
        time,
        "wind_down_mps",
    )
    drag_increment = _sample_array(
        zeros if drag_coefficient_increment is None else drag_coefficient_increment,
        time,
        "drag_coefficient_increment",
    )
    if np.any(drag_increment < 0.0):
        raise ValueError("drag_coefficient_increment must be non-negative")

    initial_state = PointMassState(
        true_airspeed_mps=initial_true_airspeed_mps,
        flight_path_angle_rad=initial_flight_path_angle_rad,
        heading_rad=initial_heading_rad,
        north_position_m=initial_north_position_m,
        east_position_m=initial_east_position_m,
        pressure_altitude_m=initial_pressure_altitude_m,
        mass_kg=initial_mass_kg,
    )
    states = np.empty((len(time), 7), dtype=float)
    states[0] = initial_state.to_array()
    angle_of_attack = np.empty(len(time), dtype=float)
    lift_coefficient = np.empty(len(time), dtype=float)
    lift = np.empty(len(time), dtype=float)
    drag = np.empty(len(time), dtype=float)
    fuel_flow = np.empty(len(time), dtype=float)

    def control(index: int) -> PitchBankThrustInput:
        return PitchBankThrustInput(
            pitch_angle_rad=float(pitch[index]),
            bank_angle_rad=float(bank[index]),
            net_thrust_n=float(thrust[index]),
            static_temperature_k=float(temperature[index]),
            wind_north_mps=float(wind_north[index]),
            wind_east_mps=float(wind_east[index]),
            wind_down_mps=float(wind_down[index]),
            drag_coefficient_increment=float(drag_increment[index]),
        )

    for index in range(len(time)):
        held_control = control(index)
        forces = plant.forces(states[index], held_control)
        angle_of_attack[index] = forces.angle_of_attack_rad
        lift_coefficient[index] = forces.lift_coefficient
        lift[index] = forces.lift_n
        drag[index] = forces.drag_n
        fuel_flow[index] = forces.fuel_flow_kg_s
        if index == len(time) - 1:
            break
        final_state = plant.integrate_zoh_step(
            states[index],
            held_control,
            float(time[index + 1] - time[index]),
            integration_rate_hz=integration_rate_hz,
            heading_rate_override_rad_s=(
                None
                if observed_heading_rate is None
                else float(observed_heading_rate[index])
            ),
        )
        states[index + 1] = final_state.to_array()

    return PointMassTrajectory(
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
        angle_of_attack_rad=angle_of_attack,
        net_thrust_n=thrust,
        lift_coefficient=lift_coefficient,
        lift_n=lift,
        drag_n=drag,
        fuel_flow_kg_s=fuel_flow,
    )


def simulate_point_mass_zoh(
    *,
    plant: PointMassModel,
    time_s: ArrayLike,
    pitch_angle_rad: ArrayLike,
    bank_angle_rad: ArrayLike,
    net_thrust_n: ArrayLike,
    static_temperature_k: ArrayLike,
    initial_true_airspeed_mps: float,
    initial_flight_path_angle_rad: float,
    initial_heading_rad: float,
    initial_pressure_altitude_m: float,
    initial_mass_kg: float,
    wind_north_mps: ArrayLike | None = None,
    wind_east_mps: ArrayLike | None = None,
    wind_down_mps: ArrayLike | None = None,
    drag_coefficient_increment: ArrayLike | None = None,
    initial_north_position_m: float = 0.0,
    initial_east_position_m: float = 0.0,
    integration_rate_hz: int = 10,
) -> PointMassTrajectory:
    """Integrate held pitch, bank, and realized-thrust inputs in local NED."""

    return _simulate_point_mass_zoh(
        plant=plant,
        observed_heading_rate_rad_s=None,
        time_s=time_s,
        pitch_angle_rad=pitch_angle_rad,
        bank_angle_rad=bank_angle_rad,
        net_thrust_n=net_thrust_n,
        static_temperature_k=static_temperature_k,
        initial_true_airspeed_mps=initial_true_airspeed_mps,
        initial_flight_path_angle_rad=initial_flight_path_angle_rad,
        initial_heading_rad=initial_heading_rad,
        initial_pressure_altitude_m=initial_pressure_altitude_m,
        initial_mass_kg=initial_mass_kg,
        wind_north_mps=wind_north_mps,
        wind_east_mps=wind_east_mps,
        wind_down_mps=wind_down_mps,
        drag_coefficient_increment=drag_coefficient_increment,
        initial_north_position_m=initial_north_position_m,
        initial_east_position_m=initial_east_position_m,
        integration_rate_hz=integration_rate_hz,
    )
