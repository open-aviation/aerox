"""Measured-state translational force reconstruction for fixed-wing aircraft."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from aerox.models.atmosphere import G0

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TranslationalForces:
    """Effective point-mass forces reconstructed from measured motion.

    ``net_along_path_force_n`` is the non-gravitational force required along
    the air-relative path.  Under the stated point-mass assumptions it equals
    the along-path thrust component minus drag.  ``required_lift_n`` neglects
    normal thrust and side-force components and assumes coordinated bank.
    """

    true_airspeed_rate_mps2: FloatArray
    flight_path_angle_rate_rad_s: FloatArray
    net_along_path_force_n: FloatArray
    required_lift_n: FloatArray
    lift_coefficient: FloatArray
    prior_drag_n: FloatArray
    prior_implied_thrust_n: FloatArray
    specific_excess_power_mps: FloatArray


def centered_linear_derivative(
    values: ArrayLike, time_s: ArrayLike, window: int = 11
) -> FloatArray:
    """Return local least-squares slopes, with unavailable edges marked NaN."""
    data = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if data.ndim != 1 or time.ndim != 1 or data.shape != time.shape:
        raise ValueError("values and time_s must be equal-length vectors")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least three")
    if len(data) < window:
        raise ValueError("window cannot exceed the vector length")
    if np.any(~np.isfinite(data)) or np.any(~np.isfinite(time)):
        raise ValueError("values and time_s must be finite")
    if np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")

    value_windows = sliding_window_view(data, window)
    time_windows = sliding_window_view(time, window)
    centered_time = time_windows - time_windows.mean(axis=1, keepdims=True)
    centered_values = value_windows - value_windows.mean(axis=1, keepdims=True)
    denominator = np.sum(centered_time**2, axis=1)
    slopes = np.sum(centered_time * centered_values, axis=1) / denominator
    result = np.full(len(data), np.nan, dtype=float)
    half_window = window // 2
    result[half_window:-half_window] = slopes
    return result


def reconstruct_translational_forces(
    *,
    time_s: ArrayLike,
    true_airspeed_mps: ArrayLike,
    flight_path_angle_rad: ArrayLike,
    bank_angle_rad: ArrayLike,
    mass_kg: ArrayLike,
    dynamic_pressure_pa: ArrayLike,
    wing_area_m2: float,
    zero_lift_drag_coefficient: float,
    induced_drag_factor: float,
    derivative_window: int = 11,
) -> TranslationalForces:
    """Reconstruct effective lift, ``T-D``, and prior-implied thrust.

    The equations are the longitudinal/vertical point-mass balances:

    ``T_path - D = m * (dV/dt + g sin(gamma))``

    ``L cos(phi) = m * (V dgamma/dt + g cos(gamma))``

    They are useful identification targets, not direct force measurements.
    """
    time = np.asarray(time_s, dtype=float)
    speed = np.asarray(true_airspeed_mps, dtype=float)
    gamma = np.asarray(flight_path_angle_rad, dtype=float)
    bank = np.asarray(bank_angle_rad, dtype=float)
    mass = np.asarray(mass_kg, dtype=float)
    dynamic_pressure = np.asarray(dynamic_pressure_pa, dtype=float)
    arrays = (time, speed, gamma, bank, mass, dynamic_pressure)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("all state inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("all state inputs must have equal length")
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError("all state inputs must be finite")
    if np.any(speed < 0) or np.any(mass <= 0) or np.any(dynamic_pressure <= 0):
        raise ValueError("speed must be non-negative; mass and q must be positive")
    if wing_area_m2 <= 0:
        raise ValueError("wing_area_m2 must be positive")
    if zero_lift_drag_coefficient < 0 or induced_drag_factor < 0:
        raise ValueError("drag-polar coefficients must be non-negative")
    if np.any(np.abs(np.cos(bank)) < 0.1):
        raise ValueError("bank angle is too close to 90 degrees")

    speed_rate = centered_linear_derivative(speed, time, derivative_window)
    gamma_rate = centered_linear_derivative(gamma, time, derivative_window)
    net_force = mass * (speed_rate + G0 * np.sin(gamma))
    lift = mass * (speed * gamma_rate + G0 * np.cos(gamma)) / np.cos(bank)
    lift_coefficient = lift / (dynamic_pressure * wing_area_m2)
    drag_coefficient = (
        zero_lift_drag_coefficient + induced_drag_factor * lift_coefficient**2
    )
    drag = dynamic_pressure * wing_area_m2 * drag_coefficient
    implied_thrust = net_force + drag
    specific_excess_power = speed * net_force / (mass * G0)
    return TranslationalForces(
        true_airspeed_rate_mps2=speed_rate,
        flight_path_angle_rate_rad_s=gamma_rate,
        net_along_path_force_n=net_force,
        required_lift_n=lift,
        lift_coefficient=lift_coefficient,
        prior_drag_n=drag,
        prior_implied_thrust_n=implied_thrust,
        specific_excess_power_mps=specific_excess_power,
    )
