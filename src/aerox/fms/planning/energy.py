"""Energy-shaped altitude and speed profile planning."""

from __future__ import annotations

import numpy as np

from aerox.fms.performance import ProfilePerformance
from aerox.models.atmosphere import G0
from aerox.navigation import ground_path_angle_from_air_motion
from ..environment import SpatialEnvironmentProfile
from ..route import FloatArray, SpatialTrajectory
from .conditions import planning_conditions
from .feasibility import (
    ProfileFeasibilityResult,
    _exponential_progress,
    _monotone_knot_progress,
    _profile_violation_score,
    _progress_from_rate_limit,
    plan_feasible_profile,
)

def plan_energy_shaped_profile(
    *,
    route: SpatialTrajectory,
    performance: ProfilePerformance,
    mass_kg: float,
    static_temperature_k: float,
    spacing_m: float = 500.0,
    iteration_count: int = 8,
    raise_on_infeasible: bool = False,
    environment_profile: SpatialEnvironmentProfile | None = None,
    waypoint_mass_kg: FloatArray | None = None,
) -> ProfileFeasibilityResult:
    """Redistribute each leg's energy change before checking feasibility.

    Altitude and speed waypoint values remain exact. Between anchors, a shared
    monotone progress coordinate is weighted by the local climb/descent,
    acceleration/deceleration, and combined specific-excess-force capacity.
    This lets a climb front-load energy addition as thrust lapses with altitude
    instead of imposing the constant gradients of linear interpolation.

    The method uses the same point-performance assumptions and final checks as
    :func:`plan_feasible_profile`; it does not increase a thrust boundary or
    relax any declared limit.
    """

    if not isinstance(iteration_count, int) or iteration_count < 1:
        raise ValueError("iteration_count must be a positive integer")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    along_path = np.unique(
        np.concatenate(
            (
                np.arange(0.0, route.total_distance_m, spacing_m),
                route.waypoint_along_path_m,
                np.array([route.total_distance_m]),
            )
        )
    )
    altitude = np.interp(
        along_path,
        route.waypoint_along_path_m,
        [point.altitude_m for point in route.intent.waypoints],
    )
    speed = np.interp(
        along_path,
        route.waypoint_along_path_m,
        [point.true_airspeed_mps for point in route.intent.waypoints],
    )
    conditions = planning_conditions(
        route=route,
        along_path_m=along_path,
        mass_kg=mass_kg,
        static_temperature_k=static_temperature_k,
        environment_profile=environment_profile,
        waypoint_mass_kg=waypoint_mass_kg,
    )

    for _ in range(iteration_count):
        updated_altitude = altitude.copy()
        updated_speed = speed.copy()
        for segment_index, (start_s, end_s) in enumerate(
            zip(
                route.waypoint_along_path_m[:-1],
                route.waypoint_along_path_m[1:],
                strict=True,
            )
        ):
            first_index = int(np.searchsorted(along_path, start_s))
            last_index = int(np.searchsorted(along_path, end_s))
            segment_s = along_path[first_index : last_index + 1]
            first = route.intent.waypoints[segment_index]
            last = route.intent.waypoints[segment_index + 1]
            altitude_change = last.altitude_m - first.altitude_m
            speed_change = last.true_airspeed_mps - first.true_airspeed_mps
            if abs(altitude_change) < 1e-12 and abs(speed_change) < 1e-12:
                continue
            local = [
                performance.limits(
                    pressure_altitude_m=float(local_altitude),
                    true_airspeed_mps=float(local_speed),
                    mass_kg=float(local_mass),
                    static_temperature_k=float(temperature),
                )
                for local_altitude, local_speed, local_mass, temperature in zip(
                    altitude[first_index : last_index + 1],
                    speed[first_index : last_index + 1],
                    conditions.mass_kg[first_index : last_index + 1],
                    conditions.static_temperature_k[first_index : last_index + 1],
                    strict=True,
                )
            ]
            progress_rate_limit = np.full(len(segment_s), np.inf)
            local_speed = speed[first_index : last_index + 1]
            local_conditions = conditions.subset(slice(first_index, last_index + 1))
            if altitude_change > 0.0:
                ground_angle_limit = ground_path_angle_from_air_motion(
                    ground_track_rad=local_conditions.ground_track_rad,
                    air_flight_path_angle_rad=np.array(
                        [value.maximum_climb_angle_rad for value in local]
                    ),
                    true_airspeed_mps=local_speed,
                    wind_north_mps=local_conditions.wind_north_mps,
                    wind_east_mps=local_conditions.wind_east_mps,
                    wind_down_mps=local_conditions.wind_down_mps,
                )
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.maximum(np.tan(ground_angle_limit), 0.0) / altitude_change,
                )
            elif altitude_change < 0.0:
                ground_angle_limit = ground_path_angle_from_air_motion(
                    ground_track_rad=local_conditions.ground_track_rad,
                    air_flight_path_angle_rad=-np.array(
                        [value.maximum_descent_angle_rad for value in local]
                    ),
                    true_airspeed_mps=local_speed,
                    wind_north_mps=local_conditions.wind_north_mps,
                    wind_east_mps=local_conditions.wind_east_mps,
                    wind_down_mps=local_conditions.wind_down_mps,
                )
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.maximum(-np.tan(ground_angle_limit), 0.0)
                    / -altitude_change,
                )
            local_motion = local_conditions.path_motion(
                along_path_m=segment_s,
                altitude_m=altitude[first_index : last_index + 1],
                true_airspeed_mps=local_speed,
            )
            local_ground_speed = local_motion.horizontal_ground_speed_mps
            speed_energy_per_progress = local_ground_speed * speed_change
            if speed_change > 0.0:
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.array([value.maximum_acceleration_mps2 for value in local])
                    / speed_energy_per_progress,
                )
            elif speed_change < 0.0:
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.array([value.maximum_deceleration_mps2 for value in local])
                    / -speed_energy_per_progress,
                )
            energy_per_progress = local_ground_speed * (
                speed_change + G0 * altitude_change / local_speed
            )
            positive_energy = energy_per_progress > 1e-12
            negative_energy = energy_per_progress < -1e-12
            maximum_excess = np.array(
                [value.maximum_specific_excess_force_mps2 for value in local]
            )
            minimum_excess = np.array(
                [value.minimum_specific_excess_force_mps2 for value in local]
            )
            vertical_wind_demand = (
                G0 * local_conditions.wind_down_mps / local_speed
            )
            maximum_excess = maximum_excess - vertical_wind_demand
            minimum_excess = minimum_excess - vertical_wind_demand
            progress_rate_limit[positive_energy] = np.minimum(
                progress_rate_limit[positive_energy],
                np.maximum(maximum_excess[positive_energy], 0.0)
                / energy_per_progress[positive_energy],
            )
            progress_rate_limit[negative_energy] = np.minimum(
                progress_rate_limit[negative_energy],
                np.minimum(minimum_excess[negative_energy], 0.0)
                / energy_per_progress[negative_energy],
            )
            progress = _progress_from_rate_limit(segment_s, progress_rate_limit)
            updated_altitude[first_index : last_index + 1] = (
                first.altitude_m + altitude_change * progress
            )
            updated_speed[first_index : last_index + 1] = (
                first.true_airspeed_mps + speed_change * progress
            )
        altitude = updated_altitude
        speed = updated_speed

    # The shared progress above is a stable initialization. Refine altitude and
    # speed independently because real vertical and kinetic energy changes need
    # not occur in lockstep between two sparse anchors.
    for _ in range(iteration_count):
        for segment_index, (start_s, end_s) in enumerate(
            zip(
                route.waypoint_along_path_m[:-1],
                route.waypoint_along_path_m[1:],
                strict=True,
            )
        ):
            first_index = int(np.searchsorted(along_path, start_s))
            last_index = int(np.searchsorted(along_path, end_s))
            segment_slice = slice(first_index, last_index + 1)
            segment_s = along_path[segment_slice]
            first = route.intent.waypoints[segment_index]
            last = route.intent.waypoints[segment_index + 1]
            altitude_change = last.altitude_m - first.altitude_m
            speed_change = last.true_airspeed_mps - first.true_airspeed_mps

            local = [
                performance.limits(
                    pressure_altitude_m=float(local_altitude),
                    true_airspeed_mps=float(local_speed),
                    mass_kg=float(local_mass),
                    static_temperature_k=float(temperature),
                )
                for local_altitude, local_speed, local_mass, temperature in zip(
                    altitude[segment_slice],
                    speed[segment_slice],
                    conditions.mass_kg[segment_slice],
                    conditions.static_temperature_k[segment_slice],
                    strict=True,
                )
            ]
            if abs(altitude_change) >= 1e-12:
                segment_conditions = conditions.subset(segment_slice)
                local_motion = segment_conditions.path_motion(
                    along_path_m=segment_s,
                    altitude_m=altitude[segment_slice],
                    true_airspeed_mps=speed[segment_slice],
                )
                local_ground_speed = local_motion.horizontal_ground_speed_mps
                speed_acceleration = local_ground_speed * np.gradient(
                    speed[segment_slice], segment_s
                )
                rate_limit = np.full(len(segment_s), np.inf)
                if altitude_change > 0.0:
                    available_sine = np.clip(
                        (
                            np.array(
                                [
                                    value.maximum_specific_excess_force_mps2
                                    for value in local
                                ]
                            )
                            - speed_acceleration
                        )
                        / G0,
                        0.0,
                        0.95,
                    )
                    energy_angle = np.arcsin(available_sine)
                    angle_limit = np.minimum(
                        energy_angle,
                        [value.maximum_climb_angle_rad for value in local],
                    )
                    ground_angle_limit = ground_path_angle_from_air_motion(
                        ground_track_rad=segment_conditions.ground_track_rad,
                        air_flight_path_angle_rad=angle_limit,
                        true_airspeed_mps=speed[segment_slice],
                        wind_north_mps=segment_conditions.wind_north_mps,
                        wind_east_mps=segment_conditions.wind_east_mps,
                        wind_down_mps=segment_conditions.wind_down_mps,
                    )
                    rate_limit = (
                        np.maximum(np.tan(ground_angle_limit), 0.0)
                        / altitude_change
                    )
                else:
                    available_sine = np.clip(
                        (
                            speed_acceleration
                            - np.array(
                                [
                                    value.minimum_specific_excess_force_mps2
                                    for value in local
                                ]
                            )
                        )
                        / G0,
                        0.0,
                        0.95,
                    )
                    energy_angle = np.arcsin(available_sine)
                    angle_limit = np.minimum(
                        energy_angle,
                        [value.maximum_descent_angle_rad for value in local],
                    )
                    ground_angle_limit = ground_path_angle_from_air_motion(
                        ground_track_rad=segment_conditions.ground_track_rad,
                        air_flight_path_angle_rad=-angle_limit,
                        true_airspeed_mps=speed[segment_slice],
                        wind_north_mps=segment_conditions.wind_north_mps,
                        wind_east_mps=segment_conditions.wind_east_mps,
                        wind_down_mps=segment_conditions.wind_down_mps,
                    )
                    rate_limit = (
                        np.maximum(-np.tan(ground_angle_limit), 0.0)
                        / -altitude_change
                    )
                progress = _progress_from_rate_limit(segment_s, rate_limit)
                altitude[segment_slice] = first.altitude_m + altitude_change * progress

            local = [
                performance.limits(
                    pressure_altitude_m=float(local_altitude),
                    true_airspeed_mps=float(local_speed),
                    mass_kg=float(local_mass),
                    static_temperature_k=float(temperature),
                )
                for local_altitude, local_speed, local_mass, temperature in zip(
                    altitude[segment_slice],
                    speed[segment_slice],
                    conditions.mass_kg[segment_slice],
                    conditions.static_temperature_k[segment_slice],
                    strict=True,
                )
            ]
            if abs(speed_change) >= 1e-12:
                segment_conditions = conditions.subset(segment_slice)
                local_motion = segment_conditions.path_motion(
                    along_path_m=segment_s,
                    altitude_m=altitude[segment_slice],
                    true_airspeed_mps=speed[segment_slice],
                )
                local_gamma = local_motion.air_flight_path_angle_rad
                gravity_demand = G0 * np.sin(local_gamma)
                local_speed = speed[segment_slice]
                local_ground_speed = local_motion.horizontal_ground_speed_mps
                rate_limit = np.full(len(segment_s), np.inf)
                if speed_change > 0.0:
                    combined_limit = np.maximum(
                        np.array(
                            [
                                value.maximum_specific_excess_force_mps2
                                for value in local
                            ]
                        )
                        - gravity_demand,
                        0.0,
                    ) / (local_ground_speed * speed_change)
                    acceleration_limit = np.array(
                        [value.maximum_acceleration_mps2 for value in local]
                    ) / (local_ground_speed * speed_change)
                    rate_limit = np.minimum(combined_limit, acceleration_limit)
                else:
                    combined_limit = np.maximum(
                        gravity_demand
                        - np.array(
                            [
                                value.minimum_specific_excess_force_mps2
                                for value in local
                            ]
                        ),
                        0.0,
                    ) / (local_ground_speed * -speed_change)
                    deceleration_limit = np.array(
                        [value.maximum_deceleration_mps2 for value in local]
                    ) / (local_ground_speed * -speed_change)
                    rate_limit = np.minimum(combined_limit, deceleration_limit)
                progress = _progress_from_rate_limit(segment_s, rate_limit)
                speed[segment_slice] = first.true_airspeed_mps + speed_change * progress

    # Finish with a bounded, deterministic per-leg search over independent
    # altitude and speed timing. The two shape parameters cannot change an
    # anchor or a physical limit; they only decide where within a leg each
    # already-requested state change occurs.
    from scipy.optimize import differential_evolution

    for segment_index, (start_s, end_s) in enumerate(
        zip(
            route.waypoint_along_path_m[:-1],
            route.waypoint_along_path_m[1:],
            strict=True,
        )
    ):
        first_index = int(np.searchsorted(along_path, start_s))
        last_index = int(np.searchsorted(along_path, end_s))
        segment_slice = slice(first_index, last_index + 1)
        segment_s = along_path[segment_slice]
        fraction = (segment_s - start_s) / (end_s - start_s)
        first = route.intent.waypoints[segment_index]
        last = route.intent.waypoints[segment_index + 1]
        altitude_change = last.altitude_m - first.altitude_m
        speed_change = last.true_airspeed_mps - first.true_airspeed_mps

        def candidate(
            parameters: np.ndarray,
            *,
            local_fraction: FloatArray = fraction,
            first_altitude_m: float = first.altitude_m,
            first_speed_mps: float = first.true_airspeed_mps,
            local_altitude_change: float = altitude_change,
            local_speed_change: float = speed_change,
            local_slice: slice = segment_slice,
            local_along_path: FloatArray = segment_s,
        ) -> tuple[float, FloatArray, FloatArray]:
            altitude_progress = _exponential_progress(
                local_fraction, float(parameters[0])
            )
            speed_progress = _exponential_progress(local_fraction, float(parameters[1]))
            candidate_altitude = (
                first_altitude_m + local_altitude_change * altitude_progress
            )
            candidate_speed = first_speed_mps + local_speed_change * speed_progress
            local = [
                performance.limits(
                    pressure_altitude_m=float(local_altitude),
                    true_airspeed_mps=float(local_speed),
                    mass_kg=float(local_mass),
                    static_temperature_k=float(temperature),
                )
                for local_altitude, local_speed, local_mass, temperature in zip(
                    candidate_altitude,
                    candidate_speed,
                    conditions.mass_kg[local_slice],
                    conditions.static_temperature_k[local_slice],
                    strict=True,
                )
            ]
            score = _profile_violation_score(
                along_path_m=local_along_path,
                altitude_m=candidate_altitude,
                speed_mps=candidate_speed,
                conditions=conditions.subset(local_slice),
                local=local,
            )
            return score, candidate_altitude, candidate_speed

        current_local = [
            performance.limits(
                pressure_altitude_m=float(local_altitude),
                true_airspeed_mps=float(local_speed),
                mass_kg=float(local_mass),
                static_temperature_k=float(temperature),
            )
            for local_altitude, local_speed, local_mass, temperature in zip(
                altitude[segment_slice],
                speed[segment_slice],
                conditions.mass_kg[segment_slice],
                conditions.static_temperature_k[segment_slice],
                strict=True,
            )
        ]
        current_score = _profile_violation_score(
            along_path_m=segment_s,
            altitude_m=altitude[segment_slice],
            speed_mps=speed[segment_slice],
            conditions=conditions.subset(segment_slice),
            local=current_local,
        )
        if current_score <= 1e-9:
            continue
        result = differential_evolution(
            lambda parameters: candidate(parameters)[0],
            bounds=((-8.0, 8.0), (-8.0, 8.0)),
            seed=segment_index,
            popsize=6,
            maxiter=40,
            tol=1e-7,
            polish=True,
            workers=1,
        )
        candidate_score, candidate_altitude, candidate_speed = candidate(result.x)
        if candidate_score < current_score:
            altitude[segment_slice] = candidate_altitude
            speed[segment_slice] = candidate_speed

    # A single exponential cannot represent an early burst followed by a
    # plateau and a later correction. Refine only still-infeasible legs with
    # three positive monotone increments for altitude and speed independently.
    # Softmax increments and PCHIP interpolation preserve exact anchors and
    # monotonicity; this adds timing freedom, not performance authority.
    knot_interval_count = 3
    for segment_index, (start_s, end_s) in enumerate(
        zip(
            route.waypoint_along_path_m[:-1],
            route.waypoint_along_path_m[1:],
            strict=True,
        )
    ):
        first_index = int(np.searchsorted(along_path, start_s))
        last_index = int(np.searchsorted(along_path, end_s))
        segment_slice = slice(first_index, last_index + 1)
        segment_s = along_path[segment_slice]
        fraction = (segment_s - start_s) / (end_s - start_s)
        first = route.intent.waypoints[segment_index]
        last = route.intent.waypoints[segment_index + 1]
        altitude_change = last.altitude_m - first.altitude_m
        speed_change = last.true_airspeed_mps - first.true_airspeed_mps

        def flexible_candidate(
            parameters: np.ndarray,
            *,
            local_fraction: FloatArray = fraction,
            first_altitude_m: float = first.altitude_m,
            first_speed_mps: float = first.true_airspeed_mps,
            local_altitude_change: float = altitude_change,
            local_speed_change: float = speed_change,
            local_slice: slice = segment_slice,
            local_along_path: FloatArray = segment_s,
        ) -> tuple[float, FloatArray, FloatArray]:
            altitude_progress = _monotone_knot_progress(
                local_fraction, parameters[:knot_interval_count]
            )
            speed_progress = _monotone_knot_progress(
                local_fraction, parameters[knot_interval_count:]
            )
            candidate_altitude = (
                first_altitude_m + local_altitude_change * altitude_progress
            )
            candidate_speed = first_speed_mps + local_speed_change * speed_progress
            local = [
                performance.limits(
                    pressure_altitude_m=float(local_altitude),
                    true_airspeed_mps=float(local_speed),
                    mass_kg=float(local_mass),
                    static_temperature_k=float(temperature),
                )
                for local_altitude, local_speed, local_mass, temperature in zip(
                    candidate_altitude,
                    candidate_speed,
                    conditions.mass_kg[local_slice],
                    conditions.static_temperature_k[local_slice],
                    strict=True,
                )
            ]
            score = _profile_violation_score(
                along_path_m=local_along_path,
                altitude_m=candidate_altitude,
                speed_mps=candidate_speed,
                conditions=conditions.subset(local_slice),
                local=local,
            )
            return score, candidate_altitude, candidate_speed

        current_local = [
            performance.limits(
                pressure_altitude_m=float(local_altitude),
                true_airspeed_mps=float(local_speed),
                mass_kg=float(local_mass),
                static_temperature_k=float(temperature),
            )
            for local_altitude, local_speed, local_mass, temperature in zip(
                altitude[segment_slice],
                speed[segment_slice],
                conditions.mass_kg[segment_slice],
                conditions.static_temperature_k[segment_slice],
                strict=True,
            )
        ]
        current_score = _profile_violation_score(
            along_path_m=segment_s,
            altitude_m=altitude[segment_slice],
            speed_mps=speed[segment_slice],
            conditions=conditions.subset(segment_slice),
            local=current_local,
        )
        if current_score <= 1e-9:
            continue
        result = differential_evolution(
            lambda parameters: flexible_candidate(parameters)[0],
            bounds=[(-4.0, 4.0)] * (2 * knot_interval_count),
            seed=100 + segment_index,
            popsize=5,
            maxiter=35,
            tol=1e-7,
            polish=True,
            workers=1,
            x0=np.zeros(2 * knot_interval_count),
        )
        candidate_score, candidate_altitude, candidate_speed = flexible_candidate(
            result.x
        )
        if candidate_score < current_score:
            altitude[segment_slice] = candidate_altitude
            speed[segment_slice] = candidate_speed

    shaped_route = route.with_profile(
        along_path_m=along_path,
        altitude_m=altitude,
        true_airspeed_mps=speed,
    )
    return plan_feasible_profile(
        route=shaped_route,
        performance=performance,
        mass_kg=mass_kg,
        static_temperature_k=static_temperature_k,
        spacing_m=spacing_m,
        raise_on_infeasible=raise_on_infeasible,
        environment_profile=environment_profile,
        waypoint_mass_kg=waypoint_mass_kg,
    )


__all__ = ["plan_energy_shaped_profile"]
