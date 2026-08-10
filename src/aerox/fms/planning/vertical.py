"""Vertical-intent speed schedules and energy-based TOC/TOD planning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aerox import aero
from aerox.fms.performance import ProfilePerformance
from aerox.models.atmosphere import (
    G0,
    true_airspeed_from_calibrated_airspeed,
    true_airspeed_from_mach,
)

from ..route import FloatArray

KNOT_TO_MPS = 0.5144444444444445
FT_TO_M = 0.3048
WindProfile = float | Callable[[FloatArray], FloatArray]


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


def _profile_at_altitude(
    profile: WindProfile,
    altitude_m: FloatArray,
    *,
    name: str,
) -> FloatArray:
    values = (
        np.asarray(profile(altitude_m), dtype=float)
        if callable(profile)
        else np.asarray(profile, dtype=float)
    )
    try:
        result = np.broadcast_to(values, altitude_m.shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast over altitude nodes") from error
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(result, dtype=float)


def _climb_ground_distance_steps(
    *,
    altitude_m: FloatArray,
    true_airspeed_mps: FloatArray,
    maximum_specific_excess_force_mps2: FloatArray,
    maximum_air_path_angle_rad: FloatArray,
    along_track_wind_mps: FloatArray,
    cross_track_wind_mps: FloatArray,
    wind_down_mps: FloatArray,
) -> FloatArray:
    altitude_step = np.diff(altitude_m)
    specific_energy = G0 * altitude_m + 0.5 * true_airspeed_mps**2
    energy_step = np.diff(specific_energy)
    if np.any(energy_step <= 0.0):
        raise ValueError("climb speed schedule must increase in energy with altitude")

    mean_speed = 0.5 * (true_airspeed_mps[:-1] + true_airspeed_mps[1:])
    mean_excess = 0.5 * (
        maximum_specific_excess_force_mps2[:-1] + maximum_specific_excess_force_mps2[1:]
    )
    mean_along_wind = 0.5 * (along_track_wind_mps[:-1] + along_track_wind_mps[1:])
    mean_crosswind = 0.5 * (cross_track_wind_mps[:-1] + cross_track_wind_mps[1:])
    mean_down_wind = 0.5 * (wind_down_mps[:-1] + wind_down_mps[1:])
    interval_angle_limit = np.minimum(
        maximum_air_path_angle_rad[:-1], maximum_air_path_angle_rad[1:]
    )

    energy_gain_rate = mean_speed * mean_excess - G0 * mean_down_wind
    maximum_altitude_rate = mean_speed * np.sin(interval_angle_limit) - mean_down_wind
    if np.any(energy_gain_rate <= 0.0):
        raise ValueError("wind and maximum thrust cannot increase climb energy")
    if np.any(maximum_altitude_rate <= 0.0):
        raise ValueError("vertical wind prevents climb at the declared angle limit")

    elapsed_time = np.maximum(
        energy_step / energy_gain_rate,
        altitude_step / maximum_altitude_rate,
    )
    required_air_vertical_speed = altitude_step / elapsed_time + mean_down_wind
    horizontal_air_speed_squared = mean_speed**2 - required_air_vertical_speed**2
    along_air_speed_squared = horizontal_air_speed_squared - mean_crosswind**2
    if np.any(along_air_speed_squared <= 0.0):
        raise ValueError("crosswind prevents climb along the requested route")
    ground_speed = np.sqrt(along_air_speed_squared) + mean_along_wind
    if np.any(ground_speed <= 0.0):
        raise ValueError("headwind prevents positive route progress during climb")
    return np.asarray(ground_speed * elapsed_time, dtype=float)


def _descent_ground_distance_steps(
    *,
    altitude_m: FloatArray,
    true_airspeed_mps: FloatArray,
    minimum_specific_excess_force_mps2: FloatArray,
    maximum_air_path_angle_rad: FloatArray,
    along_track_wind_mps: FloatArray,
    cross_track_wind_mps: FloatArray,
    wind_down_mps: FloatArray,
) -> FloatArray:
    altitude_step = np.diff(altitude_m)
    specific_energy = G0 * altitude_m + 0.5 * true_airspeed_mps**2
    energy_step = np.diff(specific_energy)
    if np.any(energy_step <= 0.0):
        raise ValueError("descent speed schedule must increase in energy with altitude")

    mean_speed = 0.5 * (true_airspeed_mps[:-1] + true_airspeed_mps[1:])
    mean_excess = 0.5 * (
        minimum_specific_excess_force_mps2[:-1] + minimum_specific_excess_force_mps2[1:]
    )
    mean_along_wind = 0.5 * (along_track_wind_mps[:-1] + along_track_wind_mps[1:])
    mean_crosswind = 0.5 * (cross_track_wind_mps[:-1] + cross_track_wind_mps[1:])
    mean_down_wind = 0.5 * (wind_down_mps[:-1] + wind_down_mps[1:])
    interval_angle_limit = np.minimum(
        maximum_air_path_angle_rad[:-1], maximum_air_path_angle_rad[1:]
    )

    energy_loss_rate = -mean_speed * mean_excess + G0 * mean_down_wind
    maximum_altitude_loss_rate = (
        mean_speed * np.sin(interval_angle_limit) + mean_down_wind
    )
    if np.any(energy_loss_rate <= 0.0):
        raise ValueError("wind and minimum thrust cannot decrease descent energy")
    if np.any(maximum_altitude_loss_rate <= 0.0):
        raise ValueError("vertical wind prevents descent at the declared angle limit")

    elapsed_time = np.maximum(
        energy_step / energy_loss_rate,
        altitude_step / maximum_altitude_loss_rate,
    )
    required_air_vertical_speed = altitude_step / elapsed_time - mean_down_wind
    horizontal_air_speed_squared = mean_speed**2 - required_air_vertical_speed**2
    along_air_speed_squared = horizontal_air_speed_squared - mean_crosswind**2
    if np.any(along_air_speed_squared <= 0.0):
        raise ValueError("crosswind prevents descent along the requested route")
    ground_speed = np.sqrt(along_air_speed_squared) + mean_along_wind
    if np.any(ground_speed <= 0.0):
        raise ValueError("headwind prevents positive route progress during descent")
    return np.asarray(ground_speed * elapsed_time, dtype=float)


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
    performance: ProfilePerformance,
    mass_kg: float,
    altitude_step_m: float = 100.0,
    along_track_wind_mps: WindProfile = 0.0,
    cross_track_wind_mps: WindProfile = 0.0,
    wind_down_mps: WindProfile = 0.0,
    static_temperature_offset_k: float = 0.0,
    speed_schedule_temperature_offset_k: float | None = None,
    climb_distance_reserve_m: float = 0.0,
) -> TopOfClimbPlan:
    """Calculate TOC by forward integration of maximum-climb energy.

    Wind is supplied in route coordinates. Positive cross-track wind may act on
    either side because only its squared crab penalty matters; vertical wind
    follows the NED convention and is positive downward.
    """

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
    along_wind = _profile_at_altitude(
        along_track_wind_mps, altitude, name="along-track wind"
    )
    crosswind = _profile_at_altitude(
        cross_track_wind_mps, altitude, name="cross-track wind"
    )
    down_wind = _profile_at_altitude(wind_down_mps, altitude, name="downward wind")
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

    maximum_climb = np.array([value.maximum_climb_angle_rad for value in local])
    ground_distance_step = _climb_ground_distance_steps(
        altitude_m=altitude,
        true_airspeed_mps=speed,
        maximum_specific_excess_force_mps2=maximum_excess,
        maximum_air_path_angle_rad=maximum_climb,
        along_track_wind_mps=along_wind,
        cross_track_wind_mps=crosswind,
        wind_down_mps=down_wind,
    )
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
    performance: ProfilePerformance,
    mass_kg: float,
    altitude_step_m: float = 100.0,
    along_track_wind_mps: WindProfile = 0.0,
    cross_track_wind_mps: WindProfile = 0.0,
    wind_down_mps: WindProfile = 0.0,
    static_temperature_offset_k: float = 0.0,
    speed_schedule_temperature_offset_k: float | None = None,
    descent_distance_reserve_m: float = 0.0,
) -> TopOfDescentPlan:
    """Calculate TOD by integrating the minimum-thrust energy path backward.

    Specific mechanical energy is ``g h + V²/2``. For each altitude interval,
    the routine divides the schedule's energy change by the modeled magnitude
    of minimum-thrust specific excess force. This includes drag, speed changes,
    mass, CAS/Mach crossover, the 10,000-ft speed transition, and along-track
    wind. Wind is supplied in route coordinates, with positive vertical wind
    pointing downward. The declared air-relative descent-angle cap remains an
    independent lower bound on distance. A positive reserve can represent
    forecast/control uncertainty.

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
    along_wind = _profile_at_altitude(
        along_track_wind_mps, altitude, name="along-track wind"
    )
    crosswind = _profile_at_altitude(
        cross_track_wind_mps, altitude, name="cross-track wind"
    )
    down_wind = _profile_at_altitude(wind_down_mps, altitude, name="downward wind")
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

    maximum_descent = np.array([value.maximum_descent_angle_rad for value in local])
    ground_distance_step = _descent_ground_distance_steps(
        altitude_m=altitude,
        true_airspeed_mps=speed,
        minimum_specific_excess_force_mps2=minimum_excess,
        maximum_air_path_angle_rad=maximum_descent,
        along_track_wind_mps=along_wind,
        cross_track_wind_mps=crosswind,
        wind_down_mps=down_wind,
    )
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
