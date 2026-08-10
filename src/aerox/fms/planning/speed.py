"""Continuous, fuel-aware spatial speed-intent planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

from aerox.models.atmosphere import (
    calibrated_airspeed_from_mach,
    mach_from_calibrated_airspeed,
    true_airspeed_from_calibrated_airspeed,
)
from aerox.fms.planning.econ_cruise import EconomyCruiseOptimizer
from aerox.models.atmosphere import G0
from aerox.navigation import wind_corrected_path

from ..environment import SpatialEnvironmentProfile
from ..route import FloatArray, SpatialTrajectory
from ..speed_intent import FT_TO_M, KNOT_TO_MPS, ModeConditionedSpeedIntent
from ..performance import FuelFlowModel, ProfilePerformance


@dataclass(frozen=True)
class CruiseSpeedStrategy:
    """Explicit level-flight speed intent supplied to trajectory planning.

    ``phase_conditioned`` preserves the existing selector-conditioned target.
    ``commanded_mach`` and ``commanded_cas`` are direct user/FMS intent.
    ``economy`` is deliberately guarded because its transonic drag prior is
    experimental and has not passed the A320 QAR release gate.
    """

    kind: str = "phase_conditioned"
    commanded_mach: float | None = None
    commanded_cas_kts: float | None = None
    economy_optimizer: EconomyCruiseOptimizer | None = None
    experimental_economy_opt_in: bool = False

    def __post_init__(self) -> None:
        kinds = {"phase_conditioned", "commanded_mach", "commanded_cas", "economy"}
        if self.kind not in kinds:
            raise ValueError(f"cruise strategy must be one of {sorted(kinds)}")
        if self.kind == "commanded_mach":
            if self.commanded_mach is None or not 0.3 < self.commanded_mach < 0.9:
                raise ValueError("commanded cruise Mach must lie between 0.3 and 0.9")
        elif self.commanded_mach is not None:
            raise ValueError("commanded_mach is only valid for commanded_mach strategy")
        if self.kind == "commanded_cas":
            if (
                self.commanded_cas_kts is None
                or not 100.0 < self.commanded_cas_kts < 400.0
            ):
                raise ValueError("commanded cruise CAS must lie between 100 and 400 kt")
        elif self.commanded_cas_kts is not None:
            raise ValueError(
                "commanded_cas_kts is only valid for commanded_cas strategy"
            )
        if self.kind == "economy":
            if self.economy_optimizer is None:
                raise ValueError("economy strategy requires an economy optimizer")
            if not self.experimental_economy_opt_in:
                raise ValueError(
                    "economy strategy requires explicit experimental opt-in"
                )
        elif self.economy_optimizer is not None or self.experimental_economy_opt_in:
            raise ValueError("economy fields are only valid for economy strategy")


@dataclass(frozen=True)
class ContinuousSpeedMassPlan:
    """A mutually consistent speed, energy, thrust, fuel, and mass plan."""

    trajectory: SpatialTrajectory
    along_path_m: FloatArray
    altitude_m: FloatArray
    desired_true_airspeed_mps: FloatArray
    planned_true_airspeed_mps: FloatArray
    planned_mass_kg: FloatArray
    predicted_fuel_flow_kg_s: FloatArray
    predicted_thrust_n: FloatArray
    required_specific_excess_force_mps2: FloatArray
    maximum_specific_excess_force_mps2: FloatArray
    minimum_specific_excess_force_mps2: FloatArray
    cas_target_kts: FloatArray
    mach_target: FloatArray
    active_cas_equivalent_target_kts: FloatArray
    active_kind: np.ndarray
    phase: np.ndarray
    iteration_count: int
    maximum_mass_update_kg: float
    maximum_energy_violation_mps2: float
    minimum_energy_violation_mps2: float
    converged: bool
    cruise_strategy: str
    economy_cruise_target_mach: FloatArray


def _apply_cruise_strategy(
    *,
    strategy: CruiseSpeedStrategy,
    phase: np.ndarray,
    altitude_m: FloatArray,
    temperature_k: FloatArray,
    mass_kg: FloatArray,
    along_wind_mps: FloatArray,
    cas_target_kts: FloatArray,
    mach_target: FloatArray,
    active_cas_target_kts: FloatArray,
    active_kind: np.ndarray,
) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray, FloatArray]:
    level = phase == "LEVEL"
    cas = np.asarray(cas_target_kts, dtype=float).copy()
    mach = np.asarray(mach_target, dtype=float).copy()
    active_cas = np.asarray(active_cas_target_kts, dtype=float).copy()
    kind = np.asarray(active_kind, dtype="U4").copy()
    economy_mach = np.full(len(phase), np.nan)
    if strategy.kind == "phase_conditioned" or not np.any(level):
        return cas, mach, active_cas, kind, economy_mach
    if strategy.kind == "commanded_mach":
        assert strategy.commanded_mach is not None
        mach[level] = strategy.commanded_mach
        active_cas[level] = (
            calibrated_airspeed_from_mach(mach[level], altitude_m[level]) / KNOT_TO_MPS
        )
        kind[level] = "MACH"
    elif strategy.kind == "commanded_cas":
        assert strategy.commanded_cas_kts is not None
        cas[level] = strategy.commanded_cas_kts
        active_cas[level] = strategy.commanded_cas_kts
        mach[level] = mach_from_calibrated_airspeed(
            cas[level] * KNOT_TO_MPS,
            altitude_m[level],
        )
        kind[level] = "CAS"
    else:
        assert strategy.economy_optimizer is not None
        level_indices = np.flatnonzero(level)
        economy_mach[level] = [
            strategy.economy_optimizer.optimize(
                mass_kg=float(mass_kg[index]),
                pressure_altitude_m=float(altitude_m[index]),
                static_temperature_k=float(temperature_k[index]),
                along_track_wind_mps=float(along_wind_mps[index]),
            ).mach
            for index in level_indices
        ]
        mach[level] = economy_mach[level]
        active_cas[level] = (
            calibrated_airspeed_from_mach(mach[level], altitude_m[level]) / KNOT_TO_MPS
        )
        kind[level] = "MACH"
    return cas, mach, active_cas, kind, economy_mach


def _validate_selector_profile(
    *,
    along_path_m: FloatArray,
    selected_cas_kts: FloatArray,
    selected_mach: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    distance = np.asarray(along_path_m, dtype=float)
    cas = np.asarray(selected_cas_kts, dtype=float)
    mach = np.asarray(selected_mach, dtype=float)
    if (
        distance.ndim != 1
        or len(distance) < 2
        or cas.shape != distance.shape
        or mach.shape != distance.shape
        or np.any(~np.isfinite(distance + cas + mach))
        or np.any(np.diff(distance) <= 0.0)
        or np.any(cas <= 0.0)
        or np.any(mach <= 0.0)
    ):
        raise ValueError(
            "selector profiles must be finite, positive, increasing vectors"
        )
    return distance, cas, mach


def _energy_rate_bounds(
    *,
    distance: FloatArray,
    altitude_m: FloatArray,
    speed_mps: FloatArray,
    mass_kg: FloatArray,
    temperature_k: FloatArray,
    ground_track_rad: FloatArray,
    wind_north_mps: FloatArray,
    wind_east_mps: FloatArray,
    wind_down_mps: FloatArray,
    performance: ProfilePerformance,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    local = [
        performance.limits(
            pressure_altitude_m=float(altitude),
            true_airspeed_mps=float(speed),
            mass_kg=float(mass),
            static_temperature_k=float(temperature),
        )
        for altitude, speed, mass, temperature in zip(
            altitude_m, speed_mps, mass_kg, temperature_k, strict=True
        )
    ]
    maximum_excess = np.asarray(
        [item.maximum_specific_excess_force_mps2 for item in local]
    )
    minimum_excess = np.asarray(
        [item.minimum_specific_excess_force_mps2 for item in local]
    )
    maximum_acceleration = np.asarray(
        [item.maximum_acceleration_mps2 for item in local]
    )
    maximum_deceleration = np.asarray(
        [item.maximum_deceleration_mps2 for item in local]
    )
    drag = np.asarray([item.level_drag_n for item in local])
    maximum_thrust = np.asarray([item.maximum_thrust_n for item in local])
    ground_path_angle = np.arctan(np.gradient(altitude_m, distance))
    motion = wind_corrected_path(
        ground_track_rad=ground_track_rad,
        ground_path_angle_rad=ground_path_angle,
        true_airspeed_mps=speed_mps,
        wind_north_mps=wind_north_mps,
        wind_east_mps=wind_east_mps,
        wind_down_mps=wind_down_mps,
    )
    gamma = motion.air_flight_path_angle_rad
    gravity_rate = G0 * np.sin(gamma)
    lower = np.maximum(minimum_excess - gravity_rate, -maximum_deceleration)
    upper = np.minimum(maximum_excess - gravity_rate, maximum_acceleration)
    if np.any(lower > upper + 1e-10):
        raise ValueError(
            "altitude profile alone lies outside the local energy envelope"
        )
    ground_speed = motion.horizontal_ground_speed_mps
    air_to_ground_energy_scale = speed_mps / ground_speed
    return (
        lower * air_to_ground_energy_scale,
        upper * air_to_ground_energy_scale,
        minimum_excess,
        maximum_excess,
        drag,
        maximum_thrust,
    )


def _project_kinetic_energy(
    *,
    target_speed_mps: FloatArray,
    distance: FloatArray,
    lower_energy_gradient_mps2: FloatArray,
    upper_energy_gradient_mps2: FloatArray,
    initial_speed_mps: float,
    terminal_speed_mps: float,
) -> FloatArray:
    """Project a speed target into fixed-endpoint energy-rate bounds."""

    target_energy = 0.5 * np.asarray(target_speed_mps, dtype=float) ** 2
    interval_distance = np.diff(distance)
    lower_change = interval_distance * np.maximum(
        lower_energy_gradient_mps2[:-1], lower_energy_gradient_mps2[1:]
    )
    upper_change = interval_distance * np.minimum(
        upper_energy_gradient_mps2[:-1], upper_energy_gradient_mps2[1:]
    )
    forward_min = np.empty_like(target_energy)
    forward_max = np.empty_like(target_energy)
    forward_min[0] = forward_max[0] = 0.5 * initial_speed_mps**2
    for index in range(len(interval_distance)):
        forward_min[index + 1] = forward_min[index] + lower_change[index]
        forward_max[index + 1] = forward_max[index] + upper_change[index]

    terminal_energy = 0.5 * terminal_speed_mps**2
    if not forward_min[-1] - 1e-6 <= terminal_energy <= forward_max[-1] + 1e-6:
        raise ValueError("terminal speed is not reachable through the energy envelope")
    backward_min = np.empty_like(target_energy)
    backward_max = np.empty_like(target_energy)
    backward_min[-1] = backward_max[-1] = terminal_energy
    for index in range(len(interval_distance) - 1, -1, -1):
        backward_min[index] = backward_min[index + 1] - upper_change[index]
        backward_max[index] = backward_max[index + 1] - lower_change[index]

    reachable_min = np.maximum(forward_min, backward_min)
    reachable_max = np.minimum(forward_max, backward_max)
    if np.any(reachable_min > reachable_max + 1e-6):
        raise ValueError("forward and backward kinetic-energy bands do not intersect")

    energy = np.empty_like(target_energy)
    energy[0] = forward_min[0]
    for index in range(len(interval_distance)):
        low = max(energy[index] + lower_change[index], backward_min[index + 1])
        high = min(energy[index] + upper_change[index], backward_max[index + 1])
        if low > high + 1e-6:
            raise ValueError(
                "no sequential speed solution exists in the energy envelope"
            )
        energy[index + 1] = np.clip(target_energy[index + 1], low, high)
    energy[-1] = terminal_energy
    return np.sqrt(np.maximum(2.0 * energy, 1.0))


def plan_continuous_speed_mass_profile(
    *,
    route: SpatialTrajectory,
    speed_intent: ModeConditionedSpeedIntent,
    selector_along_path_m: FloatArray,
    selected_cas_kts: FloatArray,
    selected_mach: FloatArray,
    top_of_climb_along_path_m: float,
    top_of_descent_along_path_m: float,
    performance: ProfilePerformance,
    fuel_flow_model: FuelFlowModel,
    initial_mass_kg: float,
    environment_profile: SpatialEnvironmentProfile,
    spacing_m: float = 500.0,
    maximum_iterations: int = 12,
    mass_tolerance_kg: float = 1.0,
    boundary_blend_distance_m: float = 25_000.0,
    minimum_cas_target_kts_by_phase: Mapping[str, float] | None = None,
    energy_margin_mps2: float = 0.002,
    selector_coordinate_kind: str = "along_path",
    cruise_strategy: CruiseSpeedStrategy | None = None,
) -> ContinuousSpeedMassPlan:
    """Plan continuous intent speed and predicted mass on a spatial route.

    Selector evidence is exogenous intent input. Aircraft response speed, fuel,
    or guidance mode is never used to form the target. The altitude profile and
    endpoint speeds remain exact while kinetic energy is projected into the
    local thrust/deceleration envelope. Fuel burn then updates mass and the
    intent/energy calculation repeats to a fixed point.
    """

    selector_s, selector_cas, selector_mach = _validate_selector_profile(
        along_path_m=selector_along_path_m,
        selected_cas_kts=selected_cas_kts,
        selected_mach=selected_mach,
    )
    if selector_coordinate_kind not in {"along_path", "elapsed_time"}:
        raise ValueError("selector_coordinate_kind must be along_path or elapsed_time")
    scalars = (
        top_of_climb_along_path_m,
        top_of_descent_along_path_m,
        initial_mass_kg,
        spacing_m,
        mass_tolerance_kg,
        boundary_blend_distance_m,
        energy_margin_mps2,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("continuous speed-plan settings must be finite")
    if (
        not 0.0
        < top_of_climb_along_path_m
        < top_of_descent_along_path_m
        < route.total_distance_m
    ):
        raise ValueError("TOC and TOD must be ordered within the route")
    if initial_mass_kg <= 0.0 or spacing_m <= 0.0 or mass_tolerance_kg <= 0.0:
        raise ValueError("mass, spacing, and tolerance must be positive")
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or maximum_iterations < 1
        or boundary_blend_distance_m <= 0.0
        or energy_margin_mps2 < 0.0
    ):
        raise ValueError("iteration count and blend distance must be positive")
    phase_cas_floor = dict(minimum_cas_target_kts_by_phase or {})
    cruise = cruise_strategy or CruiseSpeedStrategy()
    if not set(phase_cas_floor).issubset({"CLIMB", "LEVEL", "DESCENT"}) or any(
        not np.isfinite(value) or value <= 0.0 for value in phase_cas_floor.values()
    ):
        raise ValueError("phase CAS floors must be positive and use known phases")

    sampled_route = route.sample_grid(spacing_m=spacing_m)
    distance = sampled_route.along_path_m
    altitude = sampled_route.altitude_m
    base_speed = sampled_route.true_airspeed_mps
    track = sampled_route.ground_track_rad
    temperature, wind_north, wind_east, wind_down = environment_profile.sample_arrays(
        distance
    )
    along_wind = wind_north * np.cos(track) + wind_east * np.sin(track)
    phase = np.full(len(distance), "LEVEL", dtype="U7")
    phase[distance < top_of_climb_along_path_m] = "CLIMB"
    phase[distance >= top_of_descent_along_path_m] = "DESCENT"
    mass = np.full(len(distance), initial_mass_kg, dtype=float)
    speed = base_speed.copy()
    maximum_mass_update = np.inf
    converged = False

    for iteration in range(1, maximum_iterations + 1):
        ground_path_angle = np.arctan(np.gradient(altitude, distance))
        motion = wind_corrected_path(
            ground_track_rad=track,
            ground_path_angle_rad=ground_path_angle,
            true_airspeed_mps=speed,
            wind_north_mps=wind_north,
            wind_east_mps=wind_east,
            wind_down_mps=wind_down,
        )
        ground_speed = motion.horizontal_ground_speed_mps
        planned_elapsed_time = np.r_[
            0.0,
            np.cumsum(2.0 * np.diff(distance) / (ground_speed[:-1] + ground_speed[1:])),
        ]
        selector_query = (
            distance
            if selector_coordinate_kind == "along_path"
            else planned_elapsed_time
        )
        selected_cas = np.interp(selector_query, selector_s, selector_cas)
        selected_mach_values = np.interp(selector_query, selector_s, selector_mach)
        prediction = speed_intent.predict(
            selected_cas_kts=selected_cas,
            selected_mach=selected_mach_values,
            pressure_altitude_ft=altitude / FT_TO_M,
            mass_kg=mass,
            phase=phase,
        )
        (
            resolved_cas_target,
            resolved_mach_target,
            active_cas_target,
            resolved_active_kind,
            economy_cruise_mach,
        ) = _apply_cruise_strategy(
            strategy=cruise,
            phase=phase,
            altitude_m=altitude,
            temperature_k=temperature,
            mass_kg=mass,
            along_wind_mps=along_wind,
            cas_target_kts=prediction.cas_target_kts,
            mach_target=prediction.mach_target,
            active_cas_target_kts=prediction.active_cas_equivalent_kts,
            active_kind=prediction.active_kind,
        )
        for phase_name, floor_kts in phase_cas_floor.items():
            apply_floor = (phase == phase_name) & (resolved_active_kind == "CAS")
            active_cas_target[apply_floor] = np.maximum(
                active_cas_target[apply_floor], floor_kts
            )
        desired_speed = np.asarray(
            true_airspeed_from_calibrated_airspeed(
                active_cas_target * KNOT_TO_MPS,
                altitude,
                temperature,
            )
        )
        initial_weight = np.clip(distance / boundary_blend_distance_m, 0.0, 1.0)
        terminal_weight = np.clip(
            (route.total_distance_m - distance) / boundary_blend_distance_m,
            0.0,
            1.0,
        )
        desired_speed = (
            initial_weight * desired_speed + (1.0 - initial_weight) * base_speed[0]
        )
        desired_speed = (
            terminal_weight * desired_speed + (1.0 - terminal_weight) * base_speed[-1]
        )
        desired_speed[0] = base_speed[0]
        desired_speed[-1] = base_speed[-1]

        candidate_speed = speed.copy()
        for _ in range(8):
            lower, upper, _, _, _, _ = _energy_rate_bounds(
                distance=distance,
                altitude_m=altitude,
                speed_mps=candidate_speed,
                mass_kg=mass,
                temperature_k=temperature,
                ground_track_rad=track,
                wind_north_mps=wind_north,
                wind_east_mps=wind_east,
                wind_down_mps=wind_down,
                performance=performance,
            )
            lower = lower + energy_margin_mps2
            upper = upper - energy_margin_mps2
            if np.any(lower > upper):
                raise ValueError("energy margin exhausts the local planning envelope")
            projected = _project_kinetic_energy(
                target_speed_mps=desired_speed,
                distance=distance,
                lower_energy_gradient_mps2=lower,
                upper_energy_gradient_mps2=upper,
                initial_speed_mps=float(base_speed[0]),
                terminal_speed_mps=float(base_speed[-1]),
            )
            if np.max(np.abs(projected - candidate_speed)) < 1e-3:
                candidate_speed = projected
                break
            candidate_speed = projected
        speed = candidate_speed

        lower, upper, minimum_excess, maximum_excess, drag, maximum_thrust = (
            _energy_rate_bounds(
                distance=distance,
                altitude_m=altitude,
                speed_mps=speed,
                mass_kg=mass,
                temperature_k=temperature,
                ground_track_rad=track,
                wind_north_mps=wind_north,
                wind_east_mps=wind_east,
                wind_down_mps=wind_down,
                performance=performance,
            )
        )
        del lower, upper
        kinetic_energy = 0.5 * speed**2
        ground_path_angle = np.arctan(np.gradient(altitude, distance))
        motion = wind_corrected_path(
            ground_track_rad=track,
            ground_path_angle_rad=ground_path_angle,
            true_airspeed_mps=speed,
            wind_north_mps=wind_north,
            wind_east_mps=wind_east,
            wind_down_mps=wind_down,
        )
        gamma = motion.air_flight_path_angle_rad
        ground_speed = motion.horizontal_ground_speed_mps
        demand = ground_speed / speed * np.gradient(
            kinetic_energy, distance
        ) + G0 * np.sin(gamma)
        minimum_thrust = drag + mass * minimum_excess
        thrust = np.clip(drag + mass * demand, minimum_thrust, maximum_thrust)
        fuel_flow = np.asarray(
            [fuel_flow_model.fuel_flow_scalar(float(value)) for value in thrust]
        )
        if fuel_flow.shape != thrust.shape or np.any(~np.isfinite(fuel_flow)):
            raise ValueError("fuel-flow model must return one finite value per point")
        if np.any(fuel_flow < 0.0):
            raise ValueError("fuel-flow model must not return negative flow")
        interval_time = 2.0 * np.diff(distance) / (ground_speed[:-1] + ground_speed[1:])
        interval_fuel_flow = 0.5 * (fuel_flow[:-1] + fuel_flow[1:])
        new_mass = np.r_[
            initial_mass_kg,
            initial_mass_kg - np.cumsum(interval_fuel_flow * interval_time),
        ]
        if np.any(new_mass <= 0.0):
            raise ValueError("predicted fuel burn exhausted aircraft mass")
        maximum_mass_update = float(np.max(np.abs(new_mass - mass)))
        mass = 0.5 * mass + 0.5 * new_mass
        mass[0] = initial_mass_kg
        if maximum_mass_update <= mass_tolerance_kg:
            mass = new_mass
            converged = True
            break

    if not converged:
        raise ValueError("speed/mass fixed-point iteration did not converge")

    lower, upper, minimum_excess, maximum_excess, drag, maximum_thrust = (
        _energy_rate_bounds(
            distance=distance,
            altitude_m=altitude,
            speed_mps=speed,
            mass_kg=mass,
            temperature_k=temperature,
            ground_track_rad=track,
            wind_north_mps=wind_north,
            wind_east_mps=wind_east,
            wind_down_mps=wind_down,
            performance=performance,
        )
    )
    del lower, upper
    kinetic_energy = 0.5 * speed**2
    ground_path_angle = np.arctan(np.gradient(altitude, distance))
    motion = wind_corrected_path(
        ground_track_rad=track,
        ground_path_angle_rad=ground_path_angle,
        true_airspeed_mps=speed,
        wind_north_mps=wind_north,
        wind_east_mps=wind_east,
        wind_down_mps=wind_down,
    )
    gamma = motion.air_flight_path_angle_rad
    ground_speed = motion.horizontal_ground_speed_mps
    demand = ground_speed / speed * np.gradient(kinetic_energy, distance) + G0 * np.sin(
        gamma
    )
    minimum_thrust = drag + mass * minimum_excess
    thrust = np.clip(drag + mass * demand, minimum_thrust, maximum_thrust)
    fuel_flow = np.asarray(
        [fuel_flow_model.fuel_flow_scalar(float(value)) for value in thrust]
    )
    if fuel_flow.shape != thrust.shape or np.any(~np.isfinite(fuel_flow)):
        raise ValueError("fuel-flow model must return one finite value per point")
    if np.any(fuel_flow < 0.0):
        raise ValueError("fuel-flow model must not return negative flow")
    maximum_violation = float(max(0.0, np.max(demand - maximum_excess)))
    minimum_violation = float(max(0.0, np.max(minimum_excess - demand)))
    trajectory = route.with_profile(
        along_path_m=distance,
        altitude_m=altitude,
        true_airspeed_mps=speed,
    )
    return ContinuousSpeedMassPlan(
        trajectory=trajectory,
        along_path_m=distance,
        altitude_m=altitude,
        desired_true_airspeed_mps=desired_speed,
        planned_true_airspeed_mps=speed,
        planned_mass_kg=mass,
        predicted_fuel_flow_kg_s=fuel_flow,
        predicted_thrust_n=thrust,
        required_specific_excess_force_mps2=demand,
        maximum_specific_excess_force_mps2=maximum_excess,
        minimum_specific_excess_force_mps2=minimum_excess,
        cas_target_kts=resolved_cas_target,
        mach_target=resolved_mach_target,
        active_cas_equivalent_target_kts=active_cas_target,
        active_kind=resolved_active_kind,
        phase=phase,
        iteration_count=iteration,
        maximum_mass_update_kg=maximum_mass_update,
        maximum_energy_violation_mps2=maximum_violation,
        minimum_energy_violation_mps2=minimum_violation,
        converged=converged,
        cruise_strategy=cruise.kind,
        economy_cruise_target_mach=economy_cruise_mach,
    )
