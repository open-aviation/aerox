"""Vertical-intent speed schedules and energy-based top-of-descent planning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aerox import aero
from aerox.fms.profile import PointMassProfilePerformance
from aerox.models.atmosphere import (
    true_airspeed_from_calibrated_airspeed,
    true_airspeed_from_mach,
)

from .trajectory import FloatArray

KNOT_TO_MPS = 0.5144444444444445
FT_TO_M = 0.3048


def _altitude_nodes(
    lower_m: float,
    upper_m: float,
    step_m: float,
    *special_m: float,
) -> FloatArray:
    """Build an integration grid without near-duplicate transition nodes."""

    base = np.arange(lower_m, upper_m, step_m)
    tolerance_m = min(1.0, 0.1 * step_m)
    for value in special_m:
        base = base[np.abs(base - value) > tolerance_m]
    return np.unique(np.r_[base, special_m, lower_m, upper_m])


@dataclass(frozen=True)
class CASMachDescentSchedule:
    """A conventional Mach/CAS descent schedule with a low-altitude limit."""

    descent_cas_mps: float
    descent_mach: float
    low_altitude_cas_mps: float = 250.0 * KNOT_TO_MPS
    low_altitude_limit_m: float = 10_000.0 * FT_TO_M

    def __post_init__(self) -> None:
        values = (
            self.descent_cas_mps,
            self.descent_mach,
            self.low_altitude_cas_mps,
            self.low_altitude_limit_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("descent schedule values must be finite and positive")
        if not 0.2 <= self.descent_mach <= 1.0:
            raise ValueError("descent Mach must lie between 0.2 and 1.0")

    @property
    def crossover_altitude_m(self) -> float:
        """Return the ISA altitude where descent CAS and Mach are equivalent."""

        return float(aero.crossover_alt(self.descent_cas_mps, self.descent_mach))

    def true_airspeed_mps(
        self,
        pressure_altitude_m: FloatArray,
        static_temperature_offset_k: float = 0.0,
    ) -> FloatArray:
        """Evaluate the commanded TAS implied by the piecewise schedule."""

        altitude = np.asarray(pressure_altitude_m, dtype=float)
        cas = np.where(
            altitude <= self.low_altitude_limit_m,
            self.low_altitude_cas_mps,
            self.descent_cas_mps,
        )
        temperature = (
            np.asarray(aero.temperature(altitude), dtype=float)
            + static_temperature_offset_k
        )
        tas_from_cas = np.asarray(
            true_airspeed_from_calibrated_airspeed(cas, altitude, temperature),
            dtype=float,
        )
        tas_from_mach = np.asarray(
            true_airspeed_from_mach(
                self.descent_mach,
                altitude,
                temperature,
            ),
            dtype=float,
        )
        return np.where(
            altitude >= self.crossover_altitude_m,
            tas_from_mach,
            tas_from_cas,
        )


@dataclass(frozen=True)
class CASMachClimbSchedule:
    """A conventional Mach/CAS climb schedule with a low-altitude limit."""

    climb_cas_mps: float
    climb_mach: float
    low_altitude_cas_mps: float = 250.0 * KNOT_TO_MPS
    low_altitude_limit_m: float = 10_000.0 * FT_TO_M

    def __post_init__(self) -> None:
        values = (
            self.climb_cas_mps,
            self.climb_mach,
            self.low_altitude_cas_mps,
            self.low_altitude_limit_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("climb schedule values must be finite and positive")
        if not 0.2 <= self.climb_mach <= 1.0:
            raise ValueError("climb Mach must lie between 0.2 and 1.0")

    @property
    def crossover_altitude_m(self) -> float:
        """Return the ISA altitude where climb CAS and Mach are equivalent."""

        return float(aero.crossover_alt(self.climb_cas_mps, self.climb_mach))

    def true_airspeed_mps(
        self,
        pressure_altitude_m: FloatArray,
        static_temperature_offset_k: float = 0.0,
    ) -> FloatArray:
        """Evaluate the commanded TAS implied by the piecewise schedule."""

        altitude = np.asarray(pressure_altitude_m, dtype=float)
        cas = np.where(
            altitude <= self.low_altitude_limit_m,
            self.low_altitude_cas_mps,
            self.climb_cas_mps,
        )
        temperature = (
            np.asarray(aero.temperature(altitude), dtype=float)
            + static_temperature_offset_k
        )
        tas_from_cas = np.asarray(
            true_airspeed_from_calibrated_airspeed(cas, altitude, temperature),
            dtype=float,
        )
        tas_from_mach = np.asarray(
            true_airspeed_from_mach(
                self.climb_mach,
                altitude,
                temperature,
            ),
            dtype=float,
        )
        return np.where(
            altitude >= self.crossover_altitude_m,
            tas_from_mach,
            tas_from_cas,
        )


@dataclass(frozen=True)
class TopOfClimbPlan:
    """Forward-integrated maximum-climb energy path and route TOC location."""

    top_of_climb_along_path_m: float
    climb_ground_distance_m: float
    route_distance_m: float
    altitude_m: FloatArray
    true_airspeed_mps: FloatArray
    distance_from_origin_m: FloatArray
    maximum_specific_excess_force_mps2: FloatArray
    feasible_within_route: bool


@dataclass(frozen=True)
class TopOfDescentPlan:
    """Backward-integrated idle-path descent and route TOD location."""

    top_of_descent_along_path_m: float
    descent_ground_distance_m: float
    route_distance_m: float
    altitude_m: FloatArray
    true_airspeed_mps: FloatArray
    distance_to_destination_m: FloatArray
    minimum_specific_excess_force_mps2: FloatArray
    feasible_within_route: bool


def plan_top_of_climb(
    *,
    route_distance_m: float,
    initial_altitude_m: float,
    cruise_altitude_m: float,
    schedule: CASMachClimbSchedule,
    performance: PointMassProfilePerformance,
    mass_kg: float,
    altitude_step_m: float = 100.0,
    along_track_wind_mps: float | Callable[[FloatArray], FloatArray] = 0.0,
    static_temperature_offset_k: float = 0.0,
    speed_schedule_temperature_offset_k: float | None = None,
    climb_distance_reserve_m: float = 0.0,
) -> TopOfClimbPlan:
    """Calculate TOC by forward integration of maximum-climb energy."""

    scalar_values = (
        route_distance_m,
        initial_altitude_m,
        cruise_altitude_m,
        mass_kg,
        altitude_step_m,
        static_temperature_offset_k,
        climb_distance_reserve_m,
    )
    if any(not np.isfinite(value) for value in scalar_values):
        raise ValueError("TOC inputs must be finite")
    if route_distance_m <= 0.0 or mass_kg <= 0.0 or altitude_step_m <= 0.0:
        raise ValueError("route distance, mass, and altitude step must be positive")
    if cruise_altitude_m <= initial_altitude_m:
        raise ValueError("cruise altitude must exceed initial altitude")
    if climb_distance_reserve_m < 0.0:
        raise ValueError("climb distance reserve must be non-negative")
    schedule_temperature_offset_k = (
        static_temperature_offset_k
        if speed_schedule_temperature_offset_k is None
        else speed_schedule_temperature_offset_k
    )
    if not np.isfinite(schedule_temperature_offset_k):
        raise ValueError("speed-schedule temperature offset must be finite")

    altitude = _altitude_nodes(
        initial_altitude_m,
        cruise_altitude_m,
        altitude_step_m,
        schedule.low_altitude_limit_m,
        schedule.crossover_altitude_m,
    )
    altitude = altitude[
        (altitude >= initial_altitude_m) & (altitude <= cruise_altitude_m)
    ]
    speed = schedule.true_airspeed_mps(altitude, schedule_temperature_offset_k)
    wind = (
        np.asarray(along_track_wind_mps(altitude), dtype=float)
        if callable(along_track_wind_mps)
        else np.asarray(along_track_wind_mps, dtype=float)
    )
    try:
        wind = np.broadcast_to(wind, altitude.shape)
    except ValueError as error:
        raise ValueError(
            "along-track wind must broadcast over altitude nodes"
        ) from error
    if np.any(~np.isfinite(wind)):
        raise ValueError("along-track wind must be finite")
    temperature = np.asarray(aero.temperature(altitude), dtype=float) + float(
        static_temperature_offset_k
    )
    local = [
        performance.limits(
            pressure_altitude_m=float(h),
            true_airspeed_mps=float(v),
            mass_kg=mass_kg,
            static_temperature_k=float(t),
        )
        for h, v, t in zip(altitude, speed, temperature, strict=True)
    ]
    maximum_excess = np.array(
        [value.maximum_specific_excess_force_mps2 for value in local]
    )
    if np.any(maximum_excess <= 0.0):
        raise ValueError("maximum thrust must permit energy gain throughout climb")

    specific_energy = 9.80665 * altitude + 0.5 * speed**2
    energy_step = np.diff(specific_energy)
    if np.any(energy_step <= 0.0):
        raise ValueError("climb speed schedule must increase in energy with altitude")
    gain_rate = 0.5 * (maximum_excess[:-1] + maximum_excess[1:])
    air_distance_step = energy_step / gain_rate
    mean_speed = 0.5 * (speed[:-1] + speed[1:])
    mean_wind = 0.5 * (wind[:-1] + wind[1:])
    ground_scale = (mean_speed + mean_wind) / mean_speed
    if np.any(ground_scale <= 0.0):
        raise ValueError("headwind magnitude must remain below scheduled airspeed")
    energy_ground_step = air_distance_step * ground_scale

    maximum_climb = np.array([value.maximum_climb_angle_rad for value in local])
    angle_air_step = np.diff(altitude) / np.tan(maximum_climb[:-1])
    angle_ground_step = angle_air_step * ground_scale
    ground_distance_step = np.maximum(energy_ground_step, angle_ground_step)
    distance_from_origin = np.r_[0.0, np.cumsum(ground_distance_step)]
    climb_distance = float(distance_from_origin[-1] + climb_distance_reserve_m)
    return TopOfClimbPlan(
        top_of_climb_along_path_m=climb_distance,
        climb_ground_distance_m=climb_distance,
        route_distance_m=route_distance_m,
        altitude_m=altitude,
        true_airspeed_mps=speed,
        distance_from_origin_m=distance_from_origin,
        maximum_specific_excess_force_mps2=maximum_excess,
        feasible_within_route=climb_distance <= route_distance_m,
    )


def plan_top_of_descent(
    *,
    route_distance_m: float,
    cruise_altitude_m: float,
    destination_altitude_m: float,
    schedule: CASMachDescentSchedule,
    performance: PointMassProfilePerformance,
    mass_kg: float,
    altitude_step_m: float = 100.0,
    along_track_wind_mps: float | Callable[[FloatArray], FloatArray] = 0.0,
    static_temperature_offset_k: float = 0.0,
    speed_schedule_temperature_offset_k: float | None = None,
    descent_distance_reserve_m: float = 0.0,
) -> TopOfDescentPlan:
    """Calculate TOD by integrating the minimum-thrust energy path backward.

    Specific mechanical energy is ``g h + V²/2``. For each altitude interval,
    the routine divides the schedule's energy change by the modeled magnitude
    of minimum-thrust specific excess force. This includes drag, speed changes,
    mass, CAS/Mach crossover, the 10,000-ft speed transition, and along-track
    wind. The declared descent-angle cap remains an independent lower bound on
    distance. A positive reserve can represent forecast/control uncertainty.

    Callers supply the descent planning assumption through either the scalar
    ``performance.minimum_thrust_n`` fallback or its condition-dependent
    ``minimum_thrust_model``.
    """

    scalar_values = (
        route_distance_m,
        cruise_altitude_m,
        destination_altitude_m,
        mass_kg,
        altitude_step_m,
        static_temperature_offset_k,
        descent_distance_reserve_m,
    )
    if any(not np.isfinite(value) for value in scalar_values):
        raise ValueError("TOD inputs must be finite")
    if route_distance_m <= 0.0 or mass_kg <= 0.0 or altitude_step_m <= 0.0:
        raise ValueError("route distance, mass, and altitude step must be positive")
    if cruise_altitude_m <= destination_altitude_m:
        raise ValueError("cruise altitude must exceed destination altitude")
    if descent_distance_reserve_m < 0.0:
        raise ValueError("descent distance reserve must be non-negative")
    schedule_temperature_offset_k = (
        static_temperature_offset_k
        if speed_schedule_temperature_offset_k is None
        else speed_schedule_temperature_offset_k
    )
    if not np.isfinite(schedule_temperature_offset_k):
        raise ValueError("speed-schedule temperature offset must be finite")

    altitude = _altitude_nodes(
        destination_altitude_m,
        cruise_altitude_m,
        altitude_step_m,
        schedule.low_altitude_limit_m,
        schedule.crossover_altitude_m,
    )
    altitude = altitude[
        (altitude >= destination_altitude_m) & (altitude <= cruise_altitude_m)
    ]
    speed = schedule.true_airspeed_mps(altitude, schedule_temperature_offset_k)
    wind = (
        np.asarray(along_track_wind_mps(altitude), dtype=float)
        if callable(along_track_wind_mps)
        else np.asarray(along_track_wind_mps, dtype=float)
    )
    try:
        wind = np.broadcast_to(wind, altitude.shape)
    except ValueError as error:
        raise ValueError(
            "along-track wind must broadcast over altitude nodes"
        ) from error
    if np.any(~np.isfinite(wind)):
        raise ValueError("along-track wind must be finite")
    temperature = np.asarray(aero.temperature(altitude), dtype=float) + float(
        static_temperature_offset_k
    )
    local = [
        performance.limits(
            pressure_altitude_m=float(h),
            true_airspeed_mps=float(v),
            mass_kg=mass_kg,
            static_temperature_k=float(t),
        )
        for h, v, t in zip(altitude, speed, temperature, strict=True)
    ]
    minimum_excess = np.array(
        [value.minimum_specific_excess_force_mps2 for value in local]
    )
    if np.any(minimum_excess >= 0.0):
        raise ValueError("minimum thrust must permit energy loss throughout descent")

    specific_energy = 9.80665 * altitude + 0.5 * speed**2
    energy_step = np.diff(specific_energy)
    if np.any(energy_step <= 0.0):
        raise ValueError("descent speed schedule must increase in energy with altitude")
    loss_rate = -0.5 * (minimum_excess[:-1] + minimum_excess[1:])
    air_distance_step = energy_step / loss_rate
    mean_speed = 0.5 * (speed[:-1] + speed[1:])
    mean_wind = 0.5 * (wind[:-1] + wind[1:])
    ground_scale = (mean_speed + mean_wind) / mean_speed
    if np.any(ground_scale <= 0.0):
        raise ValueError("headwind magnitude must remain below scheduled airspeed")
    energy_ground_step = air_distance_step * ground_scale

    maximum_descent = np.array([value.maximum_descent_angle_rad for value in local])
    angle_air_step = np.diff(altitude) / np.tan(maximum_descent[:-1])
    angle_ground_step = angle_air_step * ground_scale
    ground_distance_step = np.maximum(energy_ground_step, angle_ground_step)
    distance_to_destination = np.r_[0.0, np.cumsum(ground_distance_step)]
    descent_distance = float(distance_to_destination[-1] + descent_distance_reserve_m)
    tod = float(route_distance_m - descent_distance)
    return TopOfDescentPlan(
        top_of_descent_along_path_m=max(0.0, tod),
        descent_ground_distance_m=descent_distance,
        route_distance_m=route_distance_m,
        altitude_m=altitude,
        true_airspeed_mps=speed,
        distance_to_destination_m=distance_to_destination,
        minimum_specific_excess_force_mps2=minimum_excess,
        feasible_within_route=tod >= 0.0,
    )
