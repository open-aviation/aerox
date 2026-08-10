"""Advisory altitude projection onto the aircraft energy envelope."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aerox.fms.performance import ProfilePerformance
from aerox.models.atmosphere import G0
from ..environment import SpatialEnvironmentProfile
from ..route import FloatArray, SpatialTrajectory
from .conditions import planning_conditions
from .energy import plan_energy_shaped_profile
from .feasibility import ProfileFeasibilityResult

def plan_advisory_energy_profile(
    *,
    route: SpatialTrajectory,
    performance: ProfilePerformance,
    mass_kg: float,
    static_temperature_k: float,
    hard_waypoint_indices: tuple[int, ...] = (0, -1),
    maximum_altitude_adjustment_m: float = 300.0,
    energy_reserve_fraction: float = 0.01,
    energy_reserve_m2ps2: float = 25.0,
    projection_iteration_count: int = 4,
    spacing_m: float = 500.0,
    raise_on_infeasible: bool = False,
    environment_profile: SpatialEnvironmentProfile | None = None,
    waypoint_mass_kg: FloatArray | None = None,
) -> ProfileFeasibilityResult:
    """Project advisory altitude targets onto the aircraft energy envelope.

    :func:`plan_energy_shaped_profile` treats every waypoint altitude and speed
    as exact. This variant is for route samples or advisory state targets:
    caller-designated hard waypoints remain exact, speed targets remain exact,
    and only an integrally unreachable advisory altitude is shifted by the
    minimum energy correction. Following legs absorb that correction whenever
    they have sufficient excess capacity.

    This changes requested advisory state, never thrust, drag, or a declared
    performance limit. The returned trajectory's intent contains the projected
    targets, making every adjustment inspectable by the caller.
    """

    if (
        not isinstance(projection_iteration_count, int)
        or projection_iteration_count < 1
    ):
        raise ValueError("projection_iteration_count must be a positive integer")
    scalar_values = (
        maximum_altitude_adjustment_m,
        energy_reserve_fraction,
        energy_reserve_m2ps2,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in scalar_values):
        raise ValueError("advisory projection settings must be finite and non-negative")
    if maximum_altitude_adjustment_m <= 0.0:
        raise ValueError("maximum_altitude_adjustment_m must be positive")

    waypoint_count = len(route.intent.waypoints)
    hard: set[int] = set()
    for raw_index in hard_waypoint_indices:
        index = raw_index if raw_index >= 0 else waypoint_count + raw_index
        if not 0 <= index < waypoint_count:
            raise ValueError("hard_waypoint_indices contains an out-of-range index")
        hard.add(index)
    original_altitude = np.array(
        [point.altitude_m for point in route.intent.waypoints], dtype=float
    )
    projected_altitude = original_altitude.copy()
    projected_route = route

    for _ in range(projection_iteration_count):
        result = plan_energy_shaped_profile(
            route=projected_route,
            performance=performance,
            mass_kg=mass_kg,
            static_temperature_k=static_temperature_k,
            spacing_m=spacing_m,
            environment_profile=environment_profile,
            waypoint_mass_kg=waypoint_mass_kg,
        )
        along_path = result.along_path_m
        conditions = planning_conditions(
            route=projected_route,
            along_path_m=along_path,
            mass_kg=mass_kg,
            static_temperature_k=static_temperature_k,
            environment_profile=environment_profile,
            waypoint_mass_kg=waypoint_mass_kg,
        )
        local = [
            performance.limits(
                pressure_altitude_m=float(altitude),
                true_airspeed_mps=float(speed),
                mass_kg=float(local_mass),
                static_temperature_k=float(temperature),
            )
            for altitude, speed, local_mass, temperature in zip(
                result.nominal_altitude_m,
                result.nominal_true_airspeed_mps,
                conditions.mass_kg,
                conditions.static_temperature_k,
                strict=True,
            )
        ]
        maximum_excess = np.array(
            [value.maximum_specific_excess_force_mps2 for value in local]
        )
        minimum_excess = np.array(
            [value.minimum_specific_excess_force_mps2 for value in local]
        )
        nominal_motion = conditions.path_motion(
            along_path_m=along_path,
            altitude_m=result.nominal_altitude_m,
            true_airspeed_mps=result.nominal_true_airspeed_mps,
        )
        energy_distance_scale = (
            result.nominal_true_airspeed_mps
            / nominal_motion.horizontal_ground_speed_mps
        )
        vertical_wind_energy_gradient = (
            G0
            * conditions.wind_down_mps
            / nominal_motion.horizontal_ground_speed_mps
        )
        changed = False
        waypoints = projected_route.intent.waypoints
        for segment_index, (start_s, end_s) in enumerate(
            zip(
                projected_route.waypoint_along_path_m[:-1],
                projected_route.waypoint_along_path_m[1:],
                strict=True,
            )
        ):
            selection = (along_path >= start_s) & (along_path <= end_s)
            first = waypoints[segment_index]
            last = waypoints[segment_index + 1]
            required_energy = 0.5 * (
                last.true_airspeed_mps**2 - first.true_airspeed_mps**2
            ) + G0 * (last.altitude_m - first.altitude_m)
            if required_energy >= 0.0:
                available_energy = float(
                    np.trapezoid(
                        maximum_excess[selection]
                        * energy_distance_scale[selection]
                        - vertical_wind_energy_gradient[selection],
                        along_path[selection],
                    )
                )
                deficit = required_energy - available_energy
                direction = -1.0
            else:
                available_energy = float(
                    np.trapezoid(
                        minimum_excess[selection]
                        * energy_distance_scale[selection]
                        - vertical_wind_energy_gradient[selection],
                        along_path[selection],
                    )
                )
                deficit = available_energy - required_energy
                direction = 1.0
            if deficit <= 1e-6:
                continue
            reserve = energy_reserve_m2ps2 + energy_reserve_fraction * abs(
                required_energy
            )
            energy_correction = deficit + reserve
            end_index = segment_index + 1
            start_index = segment_index
            if end_index not in hard:
                target_index = end_index
                altitude_delta = direction * energy_correction / G0
            elif start_index not in hard:
                target_index = start_index
                altitude_delta = -direction * energy_correction / G0
            else:
                continue
            candidate = projected_altitude[target_index] + altitude_delta
            lower = original_altitude[target_index] - maximum_altitude_adjustment_m
            upper = original_altitude[target_index] + maximum_altitude_adjustment_m
            clipped = float(np.clip(candidate, lower, upper))
            if not np.isclose(clipped, projected_altitude[target_index]):
                projected_altitude[target_index] = clipped
                changed = True

        if not changed:
            if raise_on_infeasible and not result.feasible:
                raise ValueError("; ".join(dict.fromkeys(result.violations)))
            return result
        projected_waypoints = tuple(
            replace(point, altitude_m=float(projected_altitude[index]))
            for index, point in enumerate(route.intent.waypoints)
        )
        projected_route = SpatialTrajectory.from_intent(
            replace(route.intent, waypoints=projected_waypoints)
        )

    result = plan_energy_shaped_profile(
        route=projected_route,
        performance=performance,
        mass_kg=mass_kg,
        static_temperature_k=static_temperature_k,
        spacing_m=spacing_m,
        environment_profile=environment_profile,
        waypoint_mass_kg=waypoint_mass_kg,
    )
    if raise_on_infeasible and not result.feasible:
        raise ValueError("; ".join(dict.fromkeys(result.violations)))
    return result


__all__ = ["plan_advisory_energy_profile"]
