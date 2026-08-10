"""Aircraft-aware feasibility envelopes for spatial altitude/speed profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox.fms.performance import LocalPerformanceLimits, ProfilePerformance
from aerox.models.atmosphere import G0
from aerox.navigation import ground_path_angle_from_air_motion
from ..environment import SpatialEnvironmentProfile
from ..route import FloatArray, SpatialTrajectory
from .conditions import PlanningConditions, planning_conditions


@dataclass(frozen=True)
class ProfileFeasibilityResult:
    """Reachable bands and selected dense profile for one spatial route."""

    trajectory: SpatialTrajectory
    along_path_m: FloatArray
    nominal_altitude_m: FloatArray
    nominal_true_airspeed_mps: FloatArray
    minimum_reachable_altitude_m: FloatArray
    maximum_reachable_altitude_m: FloatArray
    minimum_reachable_true_airspeed_mps: FloatArray
    maximum_reachable_true_airspeed_mps: FloatArray
    maximum_thrust_n: FloatArray
    level_drag_n: FloatArray
    maximum_climb_angle_rad: FloatArray
    maximum_descent_angle_rad: FloatArray
    maximum_acceleration_mps2: FloatArray
    maximum_deceleration_mps2: FloatArray
    maximum_specific_excess_force_mps2: FloatArray
    minimum_specific_excess_force_mps2: FloatArray
    feasible: bool
    violations: tuple[str, ...]


def _anchor_indices(
    route: SpatialTrajectory,
    along_path_m: FloatArray,
    waypoint_indices: tuple[int, ...] | None = None,
) -> dict[int, int]:
    waypoint_count = len(route.intent.waypoints)
    if waypoint_indices is None:
        indices = tuple(range(waypoint_count))
    else:
        normalized: set[int] = set()
        for raw_index in waypoint_indices:
            index = raw_index if raw_index >= 0 else waypoint_count + raw_index
            if not 0 <= index < waypoint_count:
                raise ValueError("hard_waypoint_indices contains an out-of-range index")
            normalized.add(index)
        if 0 not in normalized or waypoint_count - 1 not in normalized:
            raise ValueError(
                "hard_waypoint_indices must include the first and last waypoint"
            )
        indices = tuple(sorted(normalized))
    return {
        int(np.searchsorted(along_path_m, route.waypoint_along_path_m[index])): index
        for index in indices
    }


def _progress_from_rate_limit(
    along_path_m: FloatArray, rate_limit_per_m: FloatArray
) -> FloatArray:
    interval_limit = np.minimum(rate_limit_per_m[:-1], rate_limit_per_m[1:])
    interval_limit = np.maximum(interval_limit, 0.0)
    if np.all(np.isinf(interval_limit)):
        return (along_path_m - along_path_m[0]) / (along_path_m[-1] - along_path_m[0])
    finite = interval_limit[np.isfinite(interval_limit)]
    replacement = float(np.max(finite)) if len(finite) else 0.0
    interval_limit = np.where(np.isfinite(interval_limit), interval_limit, replacement)
    cumulative_capacity = np.concatenate(
        ([0.0], np.cumsum(interval_limit * np.diff(along_path_m)))
    )
    total_capacity = float(cumulative_capacity[-1])
    if total_capacity <= 0.0:
        return (along_path_m - along_path_m[0]) / (along_path_m[-1] - along_path_m[0])
    return cumulative_capacity / total_capacity


def _exponential_progress(fraction: FloatArray, shape: float) -> FloatArray:
    if abs(shape) < 1e-8:
        return fraction
    return -np.expm1(-shape * fraction) / -np.expm1(-shape)


def _monotone_knot_progress(
    fraction: FloatArray, interval_logits: FloatArray
) -> FloatArray:
    """Return smooth monotone progress with positive softmax knot increments."""

    from scipy.interpolate import PchipInterpolator

    logits = np.asarray(interval_logits, dtype=float)
    if logits.ndim != 1 or len(logits) < 2 or np.any(~np.isfinite(logits)):
        raise ValueError("interval_logits must contain at least two finite values")
    weights = np.exp(logits - np.max(logits))
    increments = weights / np.sum(weights)
    knot_progress = np.concatenate(([0.0], np.cumsum(increments)))
    knot_fraction = np.linspace(0.0, 1.0, len(knot_progress))
    progress = PchipInterpolator(knot_fraction, knot_progress)(fraction)
    progress[0] = 0.0
    progress[-1] = 1.0
    return np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)


def _profile_violation_score(
    *,
    along_path_m: FloatArray,
    altitude_m: FloatArray,
    speed_mps: FloatArray,
    conditions: PlanningConditions,
    local: list[LocalPerformanceLimits],
) -> float:
    motion = conditions.path_motion(
        along_path_m=along_path_m,
        altitude_m=altitude_m,
        true_airspeed_mps=speed_mps,
    )
    gamma = motion.air_flight_path_angle_rad
    ground_speed = motion.horizontal_ground_speed_mps
    acceleration = ground_speed * np.gradient(speed_mps, along_path_m)
    demand = acceleration + G0 * np.sin(gamma)
    maximum_excess = np.array(
        [value.maximum_specific_excess_force_mps2 for value in local]
    )
    minimum_excess = np.array(
        [value.minimum_specific_excess_force_mps2 for value in local]
    )
    maximum_climb = np.array([value.maximum_climb_angle_rad for value in local])
    maximum_descent = np.array([value.maximum_descent_angle_rad for value in local])
    maximum_acceleration = np.array(
        [value.maximum_acceleration_mps2 for value in local]
    )
    maximum_deceleration = np.array(
        [value.maximum_deceleration_mps2 for value in local]
    )
    normalized_violation = np.concatenate(
        (
            (demand - maximum_excess) / 0.05,
            (minimum_excess - demand) / 0.05,
            (gamma - maximum_climb) / np.deg2rad(0.1),
            (-gamma - maximum_descent) / np.deg2rad(0.1),
            (acceleration - maximum_acceleration) / 0.05,
            (-acceleration - maximum_deceleration) / 0.05,
        )
    )
    return float(max(0.0, np.max(normalized_violation)))


def plan_feasible_profile(
    *,
    route: SpatialTrajectory,
    performance: ProfilePerformance,
    mass_kg: float,
    static_temperature_k: float,
    spacing_m: float = 1_000.0,
    raise_on_infeasible: bool = False,
    environment_profile: SpatialEnvironmentProfile | None = None,
    waypoint_mass_kg: FloatArray | None = None,
    hard_waypoint_indices: tuple[int, ...] | None = None,
) -> ProfileFeasibilityResult:
    """Propagate hard waypoint constraints forward and backward in distance.

    By default, every waypoint altitude and speed is an exact anchor. A caller
    may designate a subset with ``hard_waypoint_indices`` when intermediate
    waypoints define lateral geometry only; the first and last waypoints must
    remain hard. Between anchors, the original profile is retained wherever it
    lies inside the reachable intersection and clipped only when aircraft
    limits require earlier action.
    """

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
    nominal_samples = [route.sample(value) for value in along_path]
    nominal_altitude = np.array([sample.altitude_m for sample in nominal_samples])
    nominal_speed = np.array([sample.true_airspeed_mps for sample in nominal_samples])
    conditions = planning_conditions(
        route=route,
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
            nominal_altitude,
            nominal_speed,
            conditions.mass_kg,
            conditions.static_temperature_k,
            strict=True,
        )
    ]
    maximum_thrust = np.array([value.maximum_thrust_n for value in local])
    level_drag = np.array([value.level_drag_n for value in local])
    maximum_climb = np.array([value.maximum_climb_angle_rad for value in local])
    maximum_descent = np.array([value.maximum_descent_angle_rad for value in local])
    maximum_acceleration = np.array(
        [value.maximum_acceleration_mps2 for value in local]
    )
    maximum_deceleration = np.array(
        [value.maximum_deceleration_mps2 for value in local]
    )
    maximum_specific_excess_force = np.array(
        [value.maximum_specific_excess_force_mps2 for value in local]
    )
    minimum_specific_excess_force = np.array(
        [value.minimum_specific_excess_force_mps2 for value in local]
    )
    nominal_motion = conditions.path_motion(
        along_path_m=along_path,
        altitude_m=nominal_altitude,
        true_airspeed_mps=nominal_speed,
    )
    maximum_climb_ground_angle = ground_path_angle_from_air_motion(
        ground_track_rad=conditions.ground_track_rad,
        air_flight_path_angle_rad=maximum_climb,
        true_airspeed_mps=nominal_speed,
        wind_north_mps=conditions.wind_north_mps,
        wind_east_mps=conditions.wind_east_mps,
        wind_down_mps=conditions.wind_down_mps,
    )
    maximum_descent_ground_angle = ground_path_angle_from_air_motion(
        ground_track_rad=conditions.ground_track_rad,
        air_flight_path_angle_rad=-maximum_descent,
        true_airspeed_mps=nominal_speed,
        wind_north_mps=conditions.wind_north_mps,
        wind_east_mps=conditions.wind_east_mps,
        wind_down_mps=conditions.wind_down_mps,
    )
    maximum_climb_gradient = np.maximum(
        np.tan(maximum_climb_ground_angle), 0.0
    )
    maximum_descent_gradient = np.maximum(
        -np.tan(maximum_descent_ground_angle), 0.0
    )
    air_to_ground_energy_scale = (
        nominal_speed / nominal_motion.horizontal_ground_speed_mps
    )
    anchors = _anchor_indices(route, along_path, hard_waypoint_indices)
    point_count = len(along_path)
    forward_altitude_min = np.full(point_count, -np.inf)
    forward_altitude_max = np.full(point_count, np.inf)
    forward_speed_min = np.full(point_count, 1.0)
    forward_speed_max = np.full(point_count, np.inf)
    first = route.intent.waypoints[0]
    forward_altitude_min[0] = forward_altitude_max[0] = first.altitude_m
    forward_speed_min[0] = forward_speed_max[0] = first.true_airspeed_mps
    violations: list[str] = []

    for index, distance in enumerate(np.diff(along_path)):
        next_index = index + 1
        forward_altitude_min[next_index] = forward_altitude_min[
            index
        ] - distance * maximum_descent_gradient[index]
        forward_altitude_max[next_index] = forward_altitude_max[
            index
        ] + distance * maximum_climb_gradient[index]
        forward_speed_min[next_index] = np.sqrt(
            max(
                1.0,
                forward_speed_min[index] ** 2
                - 2.0
                * air_to_ground_energy_scale[index]
                * maximum_deceleration[index]
                * distance,
            )
        )
        forward_speed_max[next_index] = np.sqrt(
            max(
                1.0,
                forward_speed_max[index] ** 2
                + 2.0
                * air_to_ground_energy_scale[index]
                * maximum_acceleration[index]
                * distance,
            )
        )
        if next_index in anchors:
            waypoint = route.intent.waypoints[anchors[next_index]]
            if not (
                forward_altitude_min[next_index] - 1e-6
                <= waypoint.altitude_m
                <= forward_altitude_max[next_index] + 1e-6
            ):
                violations.append(
                    f"{waypoint.name or anchors[next_index]} altitude is not forward-reachable"
                )
            if not (
                forward_speed_min[next_index] - 1e-6
                <= waypoint.true_airspeed_mps
                <= forward_speed_max[next_index] + 1e-6
            ):
                violations.append(
                    f"{waypoint.name or anchors[next_index]} speed is not forward-reachable"
                )
            forward_altitude_min[next_index] = forward_altitude_max[next_index] = (
                waypoint.altitude_m
            )
            forward_speed_min[next_index] = forward_speed_max[next_index] = (
                waypoint.true_airspeed_mps
            )

    backward_altitude_min = np.full(point_count, -np.inf)
    backward_altitude_max = np.full(point_count, np.inf)
    backward_speed_min = np.full(point_count, 1.0)
    backward_speed_max = np.full(point_count, np.inf)
    last = route.intent.waypoints[-1]
    backward_altitude_min[-1] = backward_altitude_max[-1] = last.altitude_m
    backward_speed_min[-1] = backward_speed_max[-1] = last.true_airspeed_mps
    for index in range(point_count - 2, -1, -1):
        distance = along_path[index + 1] - along_path[index]
        backward_altitude_min[index] = backward_altitude_min[
            index + 1
        ] - distance * maximum_climb_gradient[index]
        backward_altitude_max[index] = backward_altitude_max[
            index + 1
        ] + distance * maximum_descent_gradient[index]
        backward_speed_min[index] = np.sqrt(
            max(
                1.0,
                backward_speed_min[index + 1] ** 2
                - 2.0
                * air_to_ground_energy_scale[index]
                * maximum_acceleration[index]
                * distance,
            )
        )
        backward_speed_max[index] = np.sqrt(
            max(
                1.0,
                backward_speed_max[index + 1] ** 2
                + 2.0
                * air_to_ground_energy_scale[index]
                * maximum_deceleration[index]
                * distance,
            )
        )
        if index in anchors:
            waypoint = route.intent.waypoints[anchors[index]]
            if not (
                backward_altitude_min[index] - 1e-6
                <= waypoint.altitude_m
                <= backward_altitude_max[index] + 1e-6
            ):
                violations.append(
                    f"{waypoint.name or anchors[index]} altitude is not backward-reachable"
                )
            if not (
                backward_speed_min[index] - 1e-6
                <= waypoint.true_airspeed_mps
                <= backward_speed_max[index] + 1e-6
            ):
                violations.append(
                    f"{waypoint.name or anchors[index]} speed is not backward-reachable"
                )
            backward_altitude_min[index] = backward_altitude_max[index] = (
                waypoint.altitude_m
            )
            backward_speed_min[index] = backward_speed_max[index] = (
                waypoint.true_airspeed_mps
            )

    altitude_min = np.maximum(forward_altitude_min, backward_altitude_min)
    altitude_max = np.minimum(forward_altitude_max, backward_altitude_max)
    speed_min = np.maximum(forward_speed_min, backward_speed_min)
    speed_max = np.minimum(forward_speed_max, backward_speed_max)
    if np.any(altitude_min > altitude_max + 1e-6):
        violations.append("altitude reachable bands do not intersect")
    if np.any(speed_min > speed_max + 1e-6):
        violations.append("speed reachable bands do not intersect")
    nominal_gamma = nominal_motion.air_flight_path_angle_rad
    nominal_acceleration = nominal_motion.horizontal_ground_speed_mps * np.gradient(
        nominal_speed, along_path
    )
    required_specific_excess_force = nominal_acceleration + G0 * np.sin(nominal_gamma)
    if np.any(required_specific_excess_force > maximum_specific_excess_force + 1e-9):
        violations.append(
            "combined climb and acceleration demand exceeds available thrust"
        )
    if np.any(required_specific_excess_force < minimum_specific_excess_force - 1e-9):
        violations.append(
            "combined descent and deceleration demand is below minimum thrust"
        )
    if not violations:
        selected_altitude = np.clip(nominal_altitude, altitude_min, altitude_max)
        selected_speed = np.clip(nominal_speed, speed_min, speed_max)
        selected_local = [
            performance.limits(
                pressure_altitude_m=float(altitude),
                true_airspeed_mps=float(speed),
                mass_kg=float(local_mass),
                static_temperature_k=float(temperature),
            )
            for altitude, speed, local_mass, temperature in zip(
                selected_altitude,
                selected_speed,
                conditions.mass_kg,
                conditions.static_temperature_k,
                strict=True,
            )
        ]
        if (
            _profile_violation_score(
                along_path_m=along_path,
                altitude_m=selected_altitude,
                speed_mps=selected_speed,
                conditions=conditions,
                local=selected_local,
            )
            > 1e-9
        ):
            violations.append(
                "selected altitude/speed profile violates coupled performance limits"
            )
    feasible = not violations
    if raise_on_infeasible and not feasible:
        raise ValueError("; ".join(dict.fromkeys(violations)))
    if feasible:
        trajectory = route.with_profile(
            along_path_m=along_path,
            altitude_m=selected_altitude,
            true_airspeed_mps=selected_speed,
        )
    else:
        trajectory = route
    reported_local = selected_local if feasible else local
    maximum_thrust = np.array([value.maximum_thrust_n for value in reported_local])
    level_drag = np.array([value.level_drag_n for value in reported_local])
    maximum_climb = np.array(
        [value.maximum_climb_angle_rad for value in reported_local]
    )
    maximum_descent = np.array(
        [value.maximum_descent_angle_rad for value in reported_local]
    )
    maximum_acceleration = np.array(
        [value.maximum_acceleration_mps2 for value in reported_local]
    )
    maximum_deceleration = np.array(
        [value.maximum_deceleration_mps2 for value in reported_local]
    )
    maximum_specific_excess_force = np.array(
        [value.maximum_specific_excess_force_mps2 for value in reported_local]
    )
    minimum_specific_excess_force = np.array(
        [value.minimum_specific_excess_force_mps2 for value in reported_local]
    )
    return ProfileFeasibilityResult(
        trajectory=trajectory,
        along_path_m=along_path,
        nominal_altitude_m=nominal_altitude,
        nominal_true_airspeed_mps=nominal_speed,
        minimum_reachable_altitude_m=altitude_min,
        maximum_reachable_altitude_m=altitude_max,
        minimum_reachable_true_airspeed_mps=speed_min,
        maximum_reachable_true_airspeed_mps=speed_max,
        maximum_thrust_n=maximum_thrust,
        level_drag_n=level_drag,
        maximum_climb_angle_rad=maximum_climb,
        maximum_descent_angle_rad=maximum_descent,
        maximum_acceleration_mps2=maximum_acceleration,
        maximum_deceleration_mps2=maximum_deceleration,
        maximum_specific_excess_force_mps2=maximum_specific_excess_force,
        minimum_specific_excess_force_mps2=minimum_specific_excess_force,
        feasible=feasible,
        violations=tuple(dict.fromkeys(violations)),
    )


__all__ = ["ProfileFeasibilityResult", "plan_feasible_profile"]
