import numpy as np
import pytest

from aerox.fms import (
    FlightIntent,
    PressureScaledMinimumThrust,
    SpatialEnvironmentProfile,
    SpatialPathGuidance,
    SpatialTrajectory,
    Waypoint,
    ground_track_to_air_heading,
    inherit_state_profile,
    simulate_fms_trajectory_zoh,
)
from aerox.models import EffectiveLiftMap, ParabolicDragPolar
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
)
from aerox.guidance import TrajectoryTrackingController
from aerox.models.openap import OpenAPFuelFlowPrior


def _route() -> SpatialTrajectory:
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 9_000.0, 200.0, "START"),
                Waypoint(20_000.0, 0.0, 9_200.0, 202.0, "TURN"),
                Waypoint(20_000.0, 30_000.0, 9_500.0, 205.0, "END"),
            )
        )
    )


def test_spatial_trajectory_interpolates_route_constraints() -> None:
    route = _route()

    first_leg = route.sample(10_000.0)
    second_leg = route.sample(35_000.0)

    assert route.total_distance_m == pytest.approx(50_000.0)
    assert first_leg.north_m == pytest.approx(10_000.0)
    assert first_leg.east_m == pytest.approx(0.0)
    assert first_leg.altitude_m == pytest.approx(9_100.0)
    assert first_leg.true_airspeed_mps == pytest.approx(201.0)
    assert first_leg.ground_track_rad == pytest.approx(0.0)
    assert second_leg.north_m == pytest.approx(20_000.0)
    assert second_leg.east_m == pytest.approx(15_000.0)
    assert second_leg.altitude_m == pytest.approx(9_350.0)
    assert second_leg.ground_track_rad == pytest.approx(np.pi / 2.0)


def test_geometry_route_can_inherit_separate_state_profile() -> None:
    state_route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "STATE-START"),
                Waypoint(10_000.0, 0.0, 2_000.0, 120.0, "STATE-END"),
            ),
            maximum_climb_flight_path_angle_rad=np.deg2rad(10.0),
        )
    ).with_profile(
        along_path_m=np.array([0.0, 5_000.0, 10_000.0]),
        altitude_m=np.array([1_000.0, 1_400.0, 2_000.0]),
        true_airspeed_mps=np.array([100.0, 108.0, 120.0]),
    )
    geometry_route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 1_000.0, 100.0, "START"),
                Waypoint(5_000.0, 2_000.0, 1_500.0, 110.0, "GEOMETRY-ONLY"),
                Waypoint(10_000.0, 0.0, 2_000.0, 120.0, "END"),
            ),
            maximum_climb_flight_path_angle_rad=np.deg2rad(10.0),
        )
    )

    inherited = inherit_state_profile(
        geometry_route=geometry_route,
        state_profile=state_route,
        state_along_path_at_geometry_waypoints_m=np.array([0.0, 5_000.0, 10_000.0]),
        spacing_m=1_000.0,
    )

    middle = inherited.sample(inherited.waypoint_along_path_m[1])
    assert inherited.total_distance_m > state_route.total_distance_m
    assert middle.north_m == pytest.approx(5_000.0)
    assert middle.east_m == pytest.approx(2_000.0)
    assert middle.altitude_m == pytest.approx(1_400.0)
    assert middle.true_airspeed_mps == pytest.approx(108.0)


def test_spatial_trajectory_rejects_infeasible_vertical_profile() -> None:
    intent = FlightIntent(
        waypoints=(
            Waypoint(0.0, 0.0, 0.0, 100.0),
            Waypoint(1_000.0, 0.0, 500.0, 100.0),
        ),
        maximum_climb_flight_path_angle_rad=np.deg2rad(5.0),
    )

    with pytest.raises(ValueError, match="angle limits"):
        SpatialTrajectory.from_intent(intent)


def test_route_projection_has_signed_cross_track_error() -> None:
    route = _route()

    right = route.project(5_000.0, 750.0, final_segment_index=0)
    left = route.project(5_000.0, -250.0, final_segment_index=0)

    assert right.along_path_m == pytest.approx(5_000.0)
    assert right.cross_track_error_m == pytest.approx(750.0)
    assert left.cross_track_error_m == pytest.approx(-250.0)


def test_wind_correction_reconstructs_requested_ground_track() -> None:
    track = np.deg2rad(35.0)
    heading, groundspeed = ground_track_to_air_heading(track, 200.0, 15.0, -22.0)
    air = 200.0 * np.array([np.cos(heading), np.sin(heading)])
    ground = air + np.array([15.0, -22.0])

    reconstructed_track = np.arctan2(ground[1], ground[0])
    assert np.arctan2(
        np.sin(reconstructed_track - track), np.cos(reconstructed_track - track)
    ) == pytest.approx(0.0, abs=1e-12)
    assert np.hypot(*ground) == pytest.approx(groundspeed)


def test_wind_correction_rejects_infeasible_crosswind() -> None:
    with pytest.raises(ValueError, match="crosswind"):
        ground_track_to_air_heading(0.0, 50.0, 0.0, 60.0)


def test_guidance_looks_through_corner_and_progress_is_monotonic() -> None:
    guidance = SpatialPathGuidance(
        _route(),
        lookahead_time_s=20.0,
        minimum_lookahead_m=4_000.0,
        maximum_lookahead_m=4_000.0,
    )

    before_corner = guidance.update(
        north_m=18_000.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
    )
    after_backtracking_measurement = guidance.update(
        north_m=17_500.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
    )

    assert before_corner.target_north_m == pytest.approx(20_000.0)
    assert before_corner.target_east_m == pytest.approx(2_000.0)
    assert before_corner.reference_ground_track_rad == pytest.approx(np.pi / 4.0)
    assert after_backtracking_measurement.progress_m >= before_corner.progress_m


def test_guidance_rate_limits_polyline_corner_heading_command() -> None:
    guidance = SpatialPathGuidance(
        _route(),
        minimum_lookahead_m=4_000.0,
        maximum_lookahead_m=4_000.0,
        maximum_heading_command_rate_rad_s=np.deg2rad(1.0),
        command_step_s=1.0,
    )
    initial = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
    )
    corner = guidance.update(
        north_m=18_000.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
    )

    assert initial.reference_air_heading_rad == pytest.approx(0.0)
    assert corner.reference_ground_track_rad == pytest.approx(np.pi / 4.0)
    assert corner.reference_air_heading_rad == pytest.approx(np.deg2rad(1.0))


def test_guidance_uses_shorter_profile_than_lateral_lookahead() -> None:
    guidance = SpatialPathGuidance(
        _route(),
        lookahead_time_s=30.0,
        minimum_lookahead_m=6_000.0,
        maximum_lookahead_m=6_000.0,
        profile_lookahead_time_s=5.0,
        maximum_profile_lookahead_m=1_000.0,
    )

    target = guidance.update(
        north_m=10_000.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
    )

    assert target.target_along_path_m == pytest.approx(16_000.0)
    assert target.profile_target_along_path_m == pytest.approx(11_000.0)
    assert target.reference_altitude_m == pytest.approx(9_110.0)


def test_spatial_environment_interpolates_anchor_forecast() -> None:
    environment = SpatialEnvironmentProfile(
        along_path_m=np.array([0.0, 10_000.0]),
        static_temperature_k=np.array([280.0, 260.0]),
        wind_north_mps=np.array([0.0, 10.0]),
        wind_east_mps=np.array([-4.0, 6.0]),
        wind_down_mps=np.array([0.0, 2.0]),
    )

    sample = environment.sample(2_500.0)

    assert sample.static_temperature_k == pytest.approx(275.0)
    assert sample.wind_north_mps == pytest.approx(2.5)
    assert sample.wind_east_mps == pytest.approx(-1.5)
    assert sample.wind_down_mps == pytest.approx(0.5)


def test_fms_initial_control_seeds_first_command_rate_bounds() -> None:
    plant = PointMassModel(
        wing_area_m2=100.0,
        lift_map=EffectiveLiftMap("linear_alpha", np.array([0.50, 0.10])),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(
            engine_count=2,
            rated_thrust_n_per_engine=100_000.0,
            c1=1.0,
            c2=2.0,
            c3=1.0,
        ),
    )
    controller = TrajectoryTrackingController(
        speed_error_gain_per_s=0.05,
        flight_path_angle_error_gain_per_s=0.25,
        heading_error_gain_per_s=0.08,
        altitude_to_flight_path_gain_rad_per_m=2e-4,
        minimum_pitch_rad=np.deg2rad(-2.0),
        maximum_pitch_rad=np.deg2rad(12.0),
        maximum_bank_rad=np.deg2rad(25.0),
        minimum_thrust_n=0.0,
        maximum_thrust_n=200_000.0,
        maximum_pitch_rate_rad_s=np.deg2rad(1.0),
        maximum_bank_rate_rad_s=np.deg2rad(2.0),
        maximum_thrust_rate_n_s=25_000.0,
    )
    seed = PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(2.0),
        bank_angle_rad=0.0,
        net_thrust_n=50_000.0,
        static_temperature_k=250.0,
    )

    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=controller,
        guidance=SpatialPathGuidance(_route()),
        maximum_time_s=2.0,
        static_temperature_k=250.0,
        initial_mass_kg=60_000.0,
        initial_control=seed,
    )

    assert (
        abs(result.trajectory.pitch_angle_rad[0] - seed.pitch_angle_rad)
        <= np.deg2rad(1.0) + 1e-12
    )
    assert (
        abs(result.trajectory.bank_angle_rad[0] - seed.bank_angle_rad)
        <= np.deg2rad(2.0) + 1e-12
    )
    assert abs(result.trajectory.net_thrust_n[0] - seed.net_thrust_n) <= 25_000.0


def test_fms_simulation_applies_conditioned_minimum_thrust() -> None:
    plant = PointMassModel(
        wing_area_m2=100.0,
        lift_map=EffectiveLiftMap("linear_alpha", np.array([0.50, 0.10])),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(
            engine_count=2,
            rated_thrust_n_per_engine=100_000.0,
            c1=1.0,
            c2=2.0,
            c3=1.0,
        ),
    )
    controller = TrajectoryTrackingController(
        speed_error_gain_per_s=0.05,
        flight_path_angle_error_gain_per_s=0.25,
        heading_error_gain_per_s=0.08,
        altitude_to_flight_path_gain_rad_per_m=2e-4,
        minimum_pitch_rad=np.deg2rad(-2.0),
        maximum_pitch_rad=np.deg2rad(12.0),
        maximum_bank_rad=np.deg2rad(25.0),
        minimum_thrust_n=0.0,
        maximum_thrust_n=200_000.0,
        maximum_pitch_rate_rad_s=np.deg2rad(1.0),
        maximum_bank_rate_rad_s=np.deg2rad(2.0),
        maximum_thrust_rate_n_s=25_000.0,
    )
    minimum_thrust = PressureScaledMinimumThrust(
        sea_level_thrust_n=190_000.0,
        pressure_exponent=0.01,
    )

    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=controller,
        guidance=SpatialPathGuidance(_route()),
        maximum_time_s=1.0,
        static_temperature_k=250.0,
        initial_mass_kg=60_000.0,
        minimum_thrust_model=minimum_thrust,
    )

    conditioned_minimum = np.array(
        [
            minimum_thrust.minimum_thrust_n(
                pressure_altitude_m=float(altitude),
                true_airspeed_mps=float(speed),
                static_temperature_k=250.0,
            )
            for altitude, speed in zip(
                result.trajectory.pressure_altitude_m,
                result.trajectory.true_airspeed_mps,
                strict=True,
            )
        ]
    )
    assert np.all(result.trajectory.net_thrust_n >= conditioned_minimum)
    assert np.all(result.thrust_limited)
