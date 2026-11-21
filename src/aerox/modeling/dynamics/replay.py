"""Recorded-heading-rate replay for the point-mass model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .point_mass import HeadingRateReplayResult, PointMassModel
from .simulation import _sample_array, _simulate_point_mass_zoh


def simulate_heading_rate_replay_zoh(
    *,
    plant: PointMassModel,
    time_s: ArrayLike,
    observed_heading_rate_rad_s: ArrayLike,
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
) -> HeadingRateReplayResult:
    """Replay with observed heading rate while retaining all other equations."""

    trajectory = _simulate_point_mass_zoh(
        plant=plant,
        observed_heading_rate_rad_s=observed_heading_rate_rad_s,
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
    observed_rate = _sample_array(
        observed_heading_rate_rad_s,
        trajectory.time_s,
        "observed_heading_rate_rad_s",
    )
    denominator = (
        trajectory.mass_kg
        * trajectory.true_airspeed_mps
        * np.cos(trajectory.flight_path_angle_rad)
    )
    modeled_lateral_lift = trajectory.lift_n * np.sin(trajectory.bank_angle_rad)
    coordinated_rate = modeled_lateral_lift / denominator
    residual = denominator * observed_rate - modeled_lateral_lift
    return HeadingRateReplayResult(
        trajectory=trajectory,
        observed_heading_rate_rad_s=observed_rate,
        coordinated_heading_rate_rad_s=coordinated_rate,
        implied_lateral_force_residual_n=residual,
    )
