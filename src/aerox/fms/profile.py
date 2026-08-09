"""Aircraft-aware feasibility envelopes for spatial altitude/speed profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from aerox.models.atmosphere import (
    SEA_LEVEL_PRESSURE_PA,
    atmosphere_at_pressure_altitude,
    unit_reynolds_number,
)
from aerox.modeling.dynamics import PointMassModel
from aerox.models.thrust import (
    DemonstratedClimbN1Capacity,
    EffectiveN1Envelope,
    EffectiveThrustMap,
)
from aerox.models.atmosphere import G0

from .environment import SpatialEnvironmentProfile
from .trajectory import FloatArray, SpatialTrajectory


class MaximumThrustModel(Protocol):
    """Condition-dependent total-aircraft realized-thrust ceiling."""

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float: ...


class MinimumThrustModel(Protocol):
    """Condition-dependent total-aircraft realized-thrust floor."""

    def minimum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float: ...


@dataclass(frozen=True)
class ConstantMaximumThrust:
    """Explicit constant ceiling useful for generic aircraft and tests."""

    thrust_n: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.thrust_n) or self.thrust_n <= 0.0:
            raise ValueError("maximum thrust must be finite and positive")

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del pressure_altitude_m, true_airspeed_mps, static_temperature_k
        return self.thrust_n


@dataclass(frozen=True)
class PressureScaledMinimumThrust:
    """Simple pressure-ratio-scaled effective planning-thrust floor.

    This form is deliberately generic.  Its parameters and interpretation
    belong to the calibration artifact supplied by an aircraft application;
    the class itself does not claim to represent engine idle or a FADEC law.
    """

    sea_level_thrust_n: float
    pressure_exponent: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sea_level_thrust_n) or self.sea_level_thrust_n < 0.0:
            raise ValueError("sea-level minimum thrust must be finite and non-negative")
        if not np.isfinite(self.pressure_exponent) or self.pressure_exponent <= 0.0:
            raise ValueError(
                "minimum-thrust pressure exponent must be finite and positive"
            )

    def minimum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del true_airspeed_mps
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        pressure_ratio = float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA)
        return self.sea_level_thrust_n * pressure_ratio**self.pressure_exponent


@dataclass(frozen=True)
class QAREffectiveClimbThrustCeiling:
    """QAR-derived operational CLB N1 boundary mapped to effective net thrust.

    This composes two previously released conditional A320 models. It remains
    a conservative 90th-percentile operational planning boundary, not a hard
    physical ceiling, CFM56 engine deck, or FADEC law. It satisfies the
    ``MaximumThrustModel`` interface only for legacy planning diagnostics.
    """

    n1_envelope: EffectiveN1Envelope
    thrust_map: EffectiveThrustMap

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        climb_n1 = self.n1_envelope.climb_limit_percent(
            pressure_altitude_m=np.array([pressure_altitude_m]),
            mach=np.array([mach]),
            static_temperature_k=np.array([static_temperature_k]),
        )
        return self.thrust_map.thrust_n_scalar(
            n1_percent=float(climb_n1[0]),
            mach=mach,
            pressure_ratio=float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA),
            static_temperature_k=static_temperature_k,
        )


@dataclass(frozen=True)
class QARDemonstratedClimbThrustCapacity:
    """Map a one-sided QAR demonstrated N1 frontier to effective net thrust.

    If supplied, ``fallback_n1_envelope`` is used outside the demonstrated
    model's feature support. This avoids extrapolation while retaining the
    older conservative operational q90 boundary at unsupported conditions.
    """

    n1_capacity: DemonstratedClimbN1Capacity
    thrust_map: EffectiveThrustMap
    fallback_n1_envelope: EffectiveN1Envelope | None = None

    def planning_n1_percent(
        self,
        *,
        pressure_altitude_m: np.ndarray,
        mach: np.ndarray,
        static_temperature_k: np.ndarray,
    ) -> FloatArray:
        altitude, mach_array, temperature = np.broadcast_arrays(
            np.asarray(pressure_altitude_m, dtype=float),
            np.asarray(mach, dtype=float),
            np.asarray(static_temperature_k, dtype=float),
        )
        supported = self.n1_capacity.support_mask(
            pressure_altitude_m=altitude,
            mach=mach_array,
            static_temperature_k=temperature,
        )
        result = np.empty(altitude.size, dtype=float)
        flat_altitude = altitude.ravel()
        flat_mach = mach_array.ravel()
        flat_temperature = temperature.ravel()
        flat_supported = supported.ravel()
        if np.any(flat_supported):
            result[flat_supported] = self.n1_capacity.capacity_percent(
                pressure_altitude_m=flat_altitude[flat_supported],
                mach=flat_mach[flat_supported],
                static_temperature_k=flat_temperature[flat_supported],
            )
        if np.any(~flat_supported):
            if self.fallback_n1_envelope is None:
                raise ValueError(
                    "condition is outside demonstrated support and no fallback was supplied"
                )
            result[~flat_supported] = self.fallback_n1_envelope.climb_limit_percent(
                pressure_altitude_m=flat_altitude[~flat_supported],
                mach=flat_mach[~flat_supported],
                static_temperature_k=flat_temperature[~flat_supported],
            )
        return result.reshape(altitude.shape).ravel()

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        capacity_n1 = self.planning_n1_percent(
            pressure_altitude_m=np.array([pressure_altitude_m]),
            mach=np.array([mach]),
            static_temperature_k=np.array([static_temperature_k]),
        )
        return self.thrust_map.thrust_n_scalar(
            n1_percent=float(capacity_n1[0]),
            mach=mach,
            pressure_ratio=float(atmosphere.pressure_pa[0] / SEA_LEVEL_PRESSURE_PA),
            static_temperature_k=static_temperature_k,
        )


@dataclass(frozen=True)
class LocalPerformanceLimits:
    """Available local path limits at one altitude, speed, and mass."""

    maximum_thrust_n: float
    level_drag_n: float
    maximum_climb_angle_rad: float
    maximum_descent_angle_rad: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    maximum_specific_excess_force_mps2: float
    minimum_specific_excess_force_mps2: float


@dataclass(frozen=True)
class PointMassProfilePerformance:
    """Reduce an pitch-bank-thrust point-mass model to conservative planning limits.

    The declared caps keep this point-performance reduction inside a chosen
    operational planning envelope. ``minimum_thrust_n`` remains the explicit
    scalar fallback.  When supplied, ``minimum_thrust_model`` takes precedence
    and provides the local condition-dependent planning floor.
    """

    plant: PointMassModel
    maximum_thrust_model: MaximumThrustModel
    minimum_thrust_n: float
    maximum_climb_angle_rad: float
    maximum_descent_angle_rad: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    minimum_thrust_model: MinimumThrustModel | None = None

    def __post_init__(self) -> None:
        values = (
            self.minimum_thrust_n,
            self.maximum_climb_angle_rad,
            self.maximum_descent_angle_rad,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("profile-performance settings must be finite")
        if self.minimum_thrust_n < 0.0:
            raise ValueError("minimum thrust must be non-negative")
        if min(values[1:]) <= 0.0:
            raise ValueError("profile-performance caps must be positive")

    def limits(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        mass_kg: float,
        static_temperature_k: float,
    ) -> LocalPerformanceLimits:
        values = (
            pressure_altitude_m,
            true_airspeed_mps,
            mass_kg,
            static_temperature_k,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("profile-performance condition must be finite")
        if true_airspeed_mps <= 1.0 or mass_kg <= 0.0 or static_temperature_k <= 0.0:
            raise ValueError("speed, mass, and temperature must be positive")
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        density = float(atmosphere.density_kg_m3[0])
        dynamic_pressure = 0.5 * density * true_airspeed_mps**2
        lift_coefficient = mass_kg * G0 / (dynamic_pressure * self.plant.wing_area_m2)
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        reynolds_per_metre = float(
            unit_reynolds_number(
                density,
                true_airspeed_mps,
                static_temperature_k,
            )
        )
        drag_coefficient = self.plant.drag_coefficient_scalar(
            lift_coefficient,
            mach,
            unit_reynolds_number_per_m=reynolds_per_metre,
        )
        drag = dynamic_pressure * self.plant.wing_area_m2 * drag_coefficient
        maximum_thrust = self.maximum_thrust_model.maximum_thrust_n(
            pressure_altitude_m=pressure_altitude_m,
            true_airspeed_mps=true_airspeed_mps,
            static_temperature_k=static_temperature_k,
        )
        minimum_thrust = (
            self.minimum_thrust_model.minimum_thrust_n(
                pressure_altitude_m=pressure_altitude_m,
                true_airspeed_mps=true_airspeed_mps,
                static_temperature_k=static_temperature_k,
            )
            if self.minimum_thrust_model is not None
            else self.minimum_thrust_n
        )
        if not np.isfinite(minimum_thrust) or minimum_thrust < 0.0:
            raise ValueError(
                "conditioned minimum thrust must be finite and non-negative"
            )
        if not np.isfinite(maximum_thrust) or maximum_thrust <= minimum_thrust:
            raise ValueError("conditioned maximum thrust must exceed minimum thrust")
        raw_maximum_acceleration = (maximum_thrust - drag) / mass_kg
        raw_maximum_deceleration = (drag - minimum_thrust) / mass_kg
        raw_climb_sine = (maximum_thrust - drag) / (mass_kg * G0)
        raw_descent_sine = (drag - minimum_thrust) / (mass_kg * G0)
        maximum_climb = min(
            self.maximum_climb_angle_rad,
            float(np.arcsin(np.clip(raw_climb_sine, 0.0, 0.95))),
        )
        maximum_descent = min(
            self.maximum_descent_angle_rad,
            float(np.arcsin(np.clip(raw_descent_sine, 0.0, 0.95))),
        )
        return LocalPerformanceLimits(
            maximum_thrust_n=float(maximum_thrust),
            level_drag_n=float(drag),
            maximum_climb_angle_rad=maximum_climb,
            maximum_descent_angle_rad=maximum_descent,
            maximum_acceleration_mps2=max(
                0.0,
                min(self.maximum_acceleration_mps2, raw_maximum_acceleration),
            ),
            maximum_deceleration_mps2=max(
                0.0,
                min(self.maximum_deceleration_mps2, raw_maximum_deceleration),
            ),
            maximum_specific_excess_force_mps2=float(raw_maximum_acceleration),
            minimum_specific_excess_force_mps2=float((minimum_thrust - drag) / mass_kg),
        )


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


def _planning_conditions(
    *,
    route: SpatialTrajectory,
    along_path_m: FloatArray,
    mass_kg: float,
    static_temperature_k: float,
    environment_profile: SpatialEnvironmentProfile | None,
    waypoint_mass_kg: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    if waypoint_mass_kg is None:
        mass = np.full_like(along_path_m, mass_kg)
    else:
        waypoint_mass = np.asarray(waypoint_mass_kg, dtype=float)
        if (
            waypoint_mass.shape != route.waypoint_along_path_m.shape
            or np.any(~np.isfinite(waypoint_mass))
            or np.any(waypoint_mass <= 0.0)
        ):
            raise ValueError("waypoint_mass_kg must match positive route anchors")
        mass = np.interp(along_path_m, route.waypoint_along_path_m, waypoint_mass)
    if environment_profile is None:
        temperature = np.full_like(along_path_m, static_temperature_k)
    else:
        temperature = np.array(
            [
                environment_profile.sample(float(distance)).static_temperature_k
                for distance in along_path_m
            ]
        )
    return mass, temperature


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
    local: list[LocalPerformanceLimits],
) -> float:
    gamma = np.arctan(np.gradient(altitude_m, along_path_m))
    acceleration = speed_mps * np.gradient(speed_mps, along_path_m)
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
    performance: PointMassProfilePerformance,
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
    planning_mass, planning_temperature = _planning_conditions(
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
            planning_mass,
            planning_temperature,
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
        ] - distance * np.tan(maximum_descent[index])
        forward_altitude_max[next_index] = forward_altitude_max[
            index
        ] + distance * np.tan(maximum_climb[index])
        forward_speed_min[next_index] = np.sqrt(
            max(
                1.0,
                forward_speed_min[index] ** 2
                - 2.0 * maximum_deceleration[index] * distance,
            )
        )
        forward_speed_max[next_index] = np.sqrt(
            max(
                1.0,
                forward_speed_max[index] ** 2
                + 2.0 * maximum_acceleration[index] * distance,
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
        ] - distance * np.tan(maximum_climb[index])
        backward_altitude_max[index] = backward_altitude_max[
            index + 1
        ] + distance * np.tan(maximum_descent[index])
        backward_speed_min[index] = np.sqrt(
            max(
                1.0,
                backward_speed_min[index + 1] ** 2
                - 2.0 * maximum_acceleration[index] * distance,
            )
        )
        backward_speed_max[index] = np.sqrt(
            max(
                1.0,
                backward_speed_max[index + 1] ** 2
                + 2.0 * maximum_deceleration[index] * distance,
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
    nominal_gamma = np.arctan(np.gradient(nominal_altitude, along_path))
    nominal_acceleration = nominal_speed * np.gradient(nominal_speed, along_path)
    required_specific_excess_force = nominal_acceleration + G0 * np.sin(nominal_gamma)
    if np.any(required_specific_excess_force > maximum_specific_excess_force + 1e-9):
        violations.append(
            "combined climb and acceleration demand exceeds available thrust"
        )
    if np.any(required_specific_excess_force < minimum_specific_excess_force - 1e-9):
        violations.append(
            "combined descent and deceleration demand is below minimum thrust"
        )
    feasible = not violations
    if raise_on_infeasible and not feasible:
        raise ValueError("; ".join(dict.fromkeys(violations)))
    if feasible:
        selected_altitude = np.clip(nominal_altitude, altitude_min, altitude_max)
        selected_speed = np.clip(nominal_speed, speed_min, speed_max)
        trajectory = route.with_profile(
            along_path_m=along_path,
            altitude_m=selected_altitude,
            true_airspeed_mps=selected_speed,
        )
    else:
        trajectory = route
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


def plan_energy_shaped_profile(
    *,
    route: SpatialTrajectory,
    performance: PointMassProfilePerformance,
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
    planning_mass, planning_temperature = _planning_conditions(
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
                    planning_mass[first_index : last_index + 1],
                    planning_temperature[first_index : last_index + 1],
                    strict=True,
                )
            ]
            progress_rate_limit = np.full(len(segment_s), np.inf)
            if altitude_change > 0.0:
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.tan([value.maximum_climb_angle_rad for value in local])
                    / altitude_change,
                )
            elif altitude_change < 0.0:
                progress_rate_limit = np.minimum(
                    progress_rate_limit,
                    np.tan([value.maximum_descent_angle_rad for value in local])
                    / -altitude_change,
                )
            local_speed = speed[first_index : last_index + 1]
            speed_energy_per_progress = local_speed * speed_change
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
            energy_per_progress = speed_energy_per_progress + G0 * altitude_change
            positive_energy = energy_per_progress > 1e-12
            negative_energy = energy_per_progress < -1e-12
            maximum_excess = np.array(
                [value.maximum_specific_excess_force_mps2 for value in local]
            )
            minimum_excess = np.array(
                [value.minimum_specific_excess_force_mps2 for value in local]
            )
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
                    planning_mass[segment_slice],
                    planning_temperature[segment_slice],
                    strict=True,
                )
            ]
            if abs(altitude_change) >= 1e-12:
                speed_acceleration = speed[segment_slice] * np.gradient(
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
                    rate_limit = np.tan(angle_limit) / altitude_change
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
                    rate_limit = np.tan(angle_limit) / -altitude_change
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
                    planning_mass[segment_slice],
                    planning_temperature[segment_slice],
                    strict=True,
                )
            ]
            if abs(speed_change) >= 1e-12:
                altitude_gradient = np.gradient(altitude[segment_slice], segment_s)
                gravity_demand = G0 * np.sin(np.arctan(altitude_gradient))
                local_speed = speed[segment_slice]
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
                    ) / (local_speed * speed_change)
                    acceleration_limit = np.array(
                        [value.maximum_acceleration_mps2 for value in local]
                    ) / (local_speed * speed_change)
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
                    ) / (local_speed * -speed_change)
                    deceleration_limit = np.array(
                        [value.maximum_deceleration_mps2 for value in local]
                    ) / (local_speed * -speed_change)
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
                    planning_mass[local_slice],
                    planning_temperature[local_slice],
                    strict=True,
                )
            ]
            score = _profile_violation_score(
                along_path_m=local_along_path,
                altitude_m=candidate_altitude,
                speed_mps=candidate_speed,
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
                planning_mass[segment_slice],
                planning_temperature[segment_slice],
                strict=True,
            )
        ]
        current_score = _profile_violation_score(
            along_path_m=segment_s,
            altitude_m=altitude[segment_slice],
            speed_mps=speed[segment_slice],
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
                    planning_mass[local_slice],
                    planning_temperature[local_slice],
                    strict=True,
                )
            ]
            score = _profile_violation_score(
                along_path_m=local_along_path,
                altitude_m=candidate_altitude,
                speed_mps=candidate_speed,
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
                planning_mass[segment_slice],
                planning_temperature[segment_slice],
                strict=True,
            )
        ]
        current_score = _profile_violation_score(
            along_path_m=segment_s,
            altitude_m=altitude[segment_slice],
            speed_mps=speed[segment_slice],
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


def plan_advisory_energy_profile(
    *,
    route: SpatialTrajectory,
    performance: PointMassProfilePerformance,
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
        planning_mass, planning_temperature = _planning_conditions(
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
                planning_mass,
                planning_temperature,
                strict=True,
            )
        ]
        maximum_excess = np.array(
            [value.maximum_specific_excess_force_mps2 for value in local]
        )
        minimum_excess = np.array(
            [value.minimum_specific_excess_force_mps2 for value in local]
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
                    np.trapezoid(maximum_excess[selection], along_path[selection])
                )
                deficit = required_energy - available_energy
                direction = -1.0
            else:
                available_energy = float(
                    np.trapezoid(minimum_excess[selection], along_path[selection])
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
