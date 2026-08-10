import numpy as np
import pytest

from aerox import aero
from aerox.fms import (
    CASMachClimbSchedule,
    CASMachDescentSchedule,
    FlightIntent,
    LocalPerformanceLimits,
    SpatialTrajectory,
    Waypoint,
    plan_advisory_energy_profile,
    plan_energy_shaped_profile,
    plan_feasible_profile,
    plan_top_of_climb,
    plan_top_of_descent,
)
from aerox.modeling import PointMassProfilePerformance
from aerox.modeling.dynamics import PointMassModel
from aerox.models import EffectiveLiftMap, ParabolicDragPolar
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import ConstantMaximumThrust, PressureScaledMinimumThrust

MASS_KG = 10_000.0


class _FixedEnvelope:
    def limits(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        mass_kg: float,
        static_temperature_k: float,
    ) -> LocalPerformanceLimits:
        del pressure_altitude_m, true_airspeed_mps, mass_kg, static_temperature_k
        return LocalPerformanceLimits(
            maximum_thrust_n=10_000.0,
            level_drag_n=5_000.0,
            maximum_climb_angle_rad=np.deg2rad(1.0),
            maximum_descent_angle_rad=np.deg2rad(1.0),
            maximum_acceleration_mps2=0.1,
            maximum_deceleration_mps2=0.1,
            maximum_specific_excess_force_mps2=0.05,
            minimum_specific_excess_force_mps2=-0.05,
        )


def _plant() -> PointMassModel:
    return PointMassModel(
        wing_area_m2=50.0,
        lift_map=EffectiveLiftMap("linear_alpha", np.array([0.5, 0.1])),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(2, 50_000.0, 1.0, 2.0, 1.0),
    )


def _performance(
    *,
    maximum_acceleration_mps2: float = 0.5,
    maximum_climb_angle_deg: float = 5.0,
    maximum_thrust_n: float = 80_000.0,
) -> PointMassProfilePerformance:
    return PointMassProfilePerformance(
        plant=_plant(),
        maximum_thrust_model=ConstantMaximumThrust(maximum_thrust_n),
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=np.deg2rad(maximum_climb_angle_deg),
        maximum_descent_angle_rad=np.deg2rad(5.0),
        maximum_acceleration_mps2=maximum_acceleration_mps2,
        maximum_deceleration_mps2=0.5,
    )


def test_independent_reachable_bands_cannot_hide_coupled_energy_violation() -> None:
    terminal_speed = np.sqrt(200.0**2 - 2.0 * aero.g0 * 100.0)
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 200.0, "START"),
                Waypoint(10_000.0, 0.0, 1_100.0, terminal_speed, "END"),
            )
        )
    ).with_profile(
        along_path_m=np.array([0.0, 1_000.0, 10_000.0]),
        altitude_m=np.array([1_000.0, 1_100.0, 1_100.0]),
        true_airspeed_mps=np.array([200.0, terminal_speed, terminal_speed]),
    )

    result = plan_feasible_profile(
        route=route,
        performance=_FixedEnvelope(),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        spacing_m=500.0,
    )

    assert not result.feasible
    assert any("coupled performance limits" in item for item in result.violations)


def test_feasible_profile_preserves_hard_waypoint_constraints() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(10_000.0, 0.0, 1_200.0, 110.0, "END"),
            )
        )
    )

    result = plan_feasible_profile(
        route=route,
        performance=_performance(),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        spacing_m=500.0,
    )

    assert result.feasible
    assert result.violations == ()
    assert result.trajectory.profile_along_path_m is not None
    start = result.trajectory.sample(0.0)
    end = result.trajectory.sample(result.trajectory.total_distance_m)
    assert start.altitude_m == pytest.approx(1_000.0)
    assert end.altitude_m == pytest.approx(1_200.0)
    assert start.true_airspeed_mps == pytest.approx(100.0)
    assert end.true_airspeed_mps == pytest.approx(110.0)
    assert np.all(
        result.nominal_altitude_m >= result.minimum_reachable_altitude_m - 1e-9
    )
    assert np.all(
        result.nominal_altitude_m <= result.maximum_reachable_altitude_m + 1e-9
    )


def test_feasible_profile_accepts_geometry_only_intermediate_waypoints() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(5_000.0, 0.0, 1_000.0, 100.0, "GEOMETRY"),
                Waypoint(10_000.0, 0.0, 1_000.0, 100.0, "END"),
            )
        )
    )

    result = plan_feasible_profile(
        route=route,
        performance=_performance(),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        hard_waypoint_indices=(0, -1),
    )

    assert result.feasible
    assert result.violations == ()


def test_feasible_profile_requires_hard_route_boundaries() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(5_000.0, 0.0, 1_000.0, 100.0, "MIDDLE"),
                Waypoint(10_000.0, 0.0, 1_000.0, 100.0, "END"),
            )
        )
    )

    with pytest.raises(ValueError, match="first and last"):
        plan_feasible_profile(
            route=route,
            performance=_performance(),
            mass_kg=MASS_KG,
            static_temperature_k=aero.T0,
            hard_waypoint_indices=(0, 1),
        )


def test_aircraft_limits_reject_unreachable_altitude_anchor() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(10_000.0, 0.0, 2_000.0, 100.0, "TOO-HIGH"),
            ),
            maximum_climb_flight_path_angle_rad=np.deg2rad(10.0),
        )
    )

    result = plan_feasible_profile(
        route=route,
        performance=_performance(maximum_climb_angle_deg=2.0),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
    )

    assert not result.feasible
    assert any(
        "altitude is not forward-reachable" in item for item in result.violations
    )
    with pytest.raises(ValueError, match="altitude"):
        plan_feasible_profile(
            route=route,
            performance=_performance(maximum_climb_angle_deg=2.0),
            mass_kg=MASS_KG,
            static_temperature_k=aero.T0,
            raise_on_infeasible=True,
        )


def test_aircraft_limits_reject_unreachable_speed_anchor() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(10_000.0, 0.0, 1_000.0, 200.0, "TOO-FAST"),
            ),
            maximum_acceleration_mps2=5.0,
        )
    )

    result = plan_feasible_profile(
        route=route,
        performance=_performance(maximum_acceleration_mps2=0.2),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
    )

    assert not result.feasible
    assert any("speed is not forward-reachable" in item for item in result.violations)


def test_coupled_energy_check_rejects_individually_feasible_demands() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(10_000.0, 0.0, 1_700.0, 130.0, "ENERGY-LIMIT"),
            ),
            maximum_climb_flight_path_angle_rad=np.deg2rad(5.0),
            maximum_acceleration_mps2=0.5,
        )
    )

    result = plan_feasible_profile(
        route=route,
        performance=_performance(maximum_thrust_n=17_000.0),
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
    )

    assert not result.feasible
    assert any("combined climb and acceleration" in item for item in result.violations)


class _AltitudeLapsingMaximumThrust:
    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del true_airspeed_mps, static_temperature_k
        return 12_000.0 - 5.0 * (pressure_altitude_m - 1_000.0)


def test_energy_shaping_frontloads_climb_under_lapsing_thrust() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(20_000.0, 0.0, 1_600.0, 100.0, "END"),
            )
        )
    )
    performance = PointMassProfilePerformance(
        plant=_plant(),
        maximum_thrust_model=_AltitudeLapsingMaximumThrust(),
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=np.deg2rad(5.0),
        maximum_descent_angle_rad=np.deg2rad(5.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )

    linear = plan_feasible_profile(
        route=route,
        performance=performance,
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        spacing_m=500.0,
    )
    shaped = plan_energy_shaped_profile(
        route=route,
        performance=performance,
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        spacing_m=500.0,
    )

    assert not linear.feasible
    assert shaped.feasible
    assert shaped.nominal_altitude_m[1] > linear.nominal_altitude_m[1]
    assert shaped.nominal_altitude_m[-1] == pytest.approx(1_600.0)


def test_advisory_projection_relaxes_only_unreachable_internal_altitude() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(10_000.0, 0.0, 1_600.0, 100.0, "ADVISORY"),
                Waypoint(20_000.0, 0.0, 1_700.0, 100.0, "END"),
            )
        )
    )
    performance = _performance(maximum_thrust_n=12_000.0)

    exact = plan_energy_shaped_profile(
        route=route,
        performance=performance,
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        spacing_m=500.0,
    )
    projected = plan_advisory_energy_profile(
        route=route,
        performance=performance,
        mass_kg=MASS_KG,
        static_temperature_k=aero.T0,
        maximum_altitude_adjustment_m=300.0,
        spacing_m=500.0,
    )

    projected_waypoints = projected.trajectory.intent.waypoints
    assert not exact.feasible
    assert projected.feasible
    assert projected_waypoints[0].altitude_m == pytest.approx(1_000.0)
    assert projected_waypoints[-1].altitude_m == pytest.approx(1_700.0)
    assert 1_300.0 <= projected_waypoints[1].altitude_m < 1_600.0
    assert [point.true_airspeed_mps for point in projected_waypoints] == pytest.approx(
        [100.0, 100.0, 100.0]
    )


def test_energy_tod_uses_cas_mach_schedule_and_wind() -> None:
    schedule = CASMachDescentSchedule(
        descent_cas_mps=280.0 * 0.5144444444444445,
        descent_mach=0.78,
    )
    performance = _performance(maximum_thrust_n=80_000.0)

    calm = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        altitude_step_m=100.0,
    )
    tailwind = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        altitude_step_m=100.0,
        along_track_wind_mps=20.0,
    )
    profiled_wind = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        altitude_step_m=100.0,
        along_track_wind_mps=lambda altitude: np.linspace(0.0, 20.0, len(altitude)),
    )
    crosswind = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        altitude_step_m=100.0,
        cross_track_wind_mps=30.0,
    )

    assert 0.0 < schedule.crossover_altitude_m < 20_000.0
    assert calm.feasible_within_route
    assert calm.altitude_m[0] == pytest.approx(1_000.0)
    assert calm.altitude_m[-1] == pytest.approx(11_000.0)
    assert np.all(np.diff(calm.distance_to_destination_m) > 0.0)
    assert calm.descent_ground_distance_m > 0.0
    assert tailwind.descent_ground_distance_m > calm.descent_ground_distance_m
    assert (
        calm.descent_ground_distance_m
        < profiled_wind.descent_ground_distance_m
        < tailwind.descent_ground_distance_m
    )
    assert tailwind.top_of_descent_along_path_m < calm.top_of_descent_along_path_m
    assert crosswind.descent_ground_distance_m < calm.descent_ground_distance_m


def test_cas_mach_schedules_apply_forecast_temperature_offset() -> None:
    altitude = np.array([3_000.0, 11_000.0])
    descent = CASMachDescentSchedule(
        descent_cas_mps=280.0 * 0.5144444444444445,
        descent_mach=0.78,
    )
    climb = CASMachClimbSchedule(
        climb_cas_mps=280.0 * 0.5144444444444445,
        climb_mach=0.78,
    )

    descent_isa = descent.true_airspeed_mps(altitude)
    descent_warm = descent.true_airspeed_mps(altitude, 10.0)
    climb_isa = climb.true_airspeed_mps(altitude)
    climb_warm = climb.true_airspeed_mps(altitude, 10.0)
    assert np.all(descent_warm > descent_isa)
    assert np.all(climb_warm > climb_isa)


def test_conditioned_minimum_thrust_lengthens_energy_descent() -> None:
    schedule = CASMachDescentSchedule(
        descent_cas_mps=280.0 * 0.5144444444444445,
        descent_mach=0.78,
    )
    zero = _performance(maximum_thrust_n=80_000.0)
    conditioned = PointMassProfilePerformance(
        plant=zero.plant,
        maximum_thrust_model=zero.maximum_thrust_model,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=zero.maximum_climb_angle_rad,
        maximum_descent_angle_rad=zero.maximum_descent_angle_rad,
        maximum_acceleration_mps2=zero.maximum_acceleration_mps2,
        maximum_deceleration_mps2=zero.maximum_deceleration_mps2,
        minimum_thrust_model=PressureScaledMinimumThrust(4_000.0, 0.5),
    )

    zero_plan = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=zero,
        mass_kg=MASS_KG,
    )
    conditioned_plan = plan_top_of_descent(
        route_distance_m=500_000.0,
        cruise_altitude_m=11_000.0,
        destination_altitude_m=1_000.0,
        schedule=schedule,
        performance=conditioned,
        mass_kg=MASS_KG,
    )

    assert (
        conditioned_plan.descent_ground_distance_m > zero_plan.descent_ground_distance_m
    )


def test_top_of_descent_handles_crossover_near_regular_grid_node() -> None:
    schedule = CASMachDescentSchedule(
        descent_cas_mps=290.2607039091762 * 0.5144444444444445,
        descent_mach=0.7611665597614297,
        low_altitude_cas_mps=277.89792110269957 * 0.5144444444444445,
    )

    plan = plan_top_of_descent(
        route_distance_m=1_000_000.0,
        cruise_altitude_m=11_275.2,
        destination_altitude_m=2_524.5,
        schedule=schedule,
        performance=_performance(),
        mass_kg=MASS_KG,
        altitude_step_m=50.0,
    )

    assert plan.descent_ground_distance_m > 0.0
    assert np.all(np.diff(plan.distance_to_destination_m) > 0.0)


def test_energy_top_of_climb_uses_schedule_wind_and_reserve() -> None:
    schedule = CASMachClimbSchedule(
        climb_cas_mps=280.0 * 0.5144444444444445,
        climb_mach=0.78,
    )
    performance = _performance(maximum_thrust_n=80_000.0)

    calm = plan_top_of_climb(
        route_distance_m=500_000.0,
        initial_altitude_m=1_000.0,
        cruise_altitude_m=11_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
    )
    headwind = plan_top_of_climb(
        route_distance_m=500_000.0,
        initial_altitude_m=1_000.0,
        cruise_altitude_m=11_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        along_track_wind_mps=-20.0,
    )
    reserved = plan_top_of_climb(
        route_distance_m=500_000.0,
        initial_altitude_m=1_000.0,
        cruise_altitude_m=11_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        climb_distance_reserve_m=10_000.0,
    )
    downdraft = plan_top_of_climb(
        route_distance_m=500_000.0,
        initial_altitude_m=1_000.0,
        cruise_altitude_m=11_000.0,
        schedule=schedule,
        performance=performance,
        mass_kg=MASS_KG,
        wind_down_mps=3.0,
    )

    assert calm.feasible_within_route
    assert np.all(np.diff(calm.distance_from_origin_m) > 0.0)
    assert headwind.climb_ground_distance_m < calm.climb_ground_distance_m
    assert reserved.climb_ground_distance_m == pytest.approx(
        calm.climb_ground_distance_m + 10_000.0
    )
    assert downdraft.climb_ground_distance_m > calm.climb_ground_distance_m
