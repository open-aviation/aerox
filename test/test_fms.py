import numpy as np
import pytest

from aerox.fms import (
    AltitudeConstraint,
    FlightIntent,
    LateralWaypoint,
    SelectedAltitudeIntent,
    SelectedSpeedIntent,
    SpeedConstraint,
    SpeedTarget,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    TerminalConstraint,
)
from aerox.guidance import (
    InverseFlightMechanicsController,
    SelectedAltitudeGuidance,
    SpatialPathGuidance,
    ground_track_to_air_heading,
)
from aerox.navigation import (
    ground_path_angle_from_air_motion,
    horizontal_ground_speed_mps,
    wind_corrected_path,
)
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
)
from aerox.models import EffectiveLiftMap, ParabolicDragPolar
from aerox.models.atmosphere import (
    true_airspeed_from_calibrated_airspeed,
    true_airspeed_from_mach,
)
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import PressureScaledMinimumThrust
from aerox.simulation import simulate_fms_trajectory_zoh


def _route() -> SpatialTrajectory:
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                LateralWaypoint(0.0, 0.0, "START"),
                LateralWaypoint(20_000.0, 0.0, "TURN"),
                LateralWaypoint(20_000.0, 30_000.0, "END"),
            ),
            departure=TerminalConstraint(9_000.0, SpeedTarget("TAS", 200.0)),
            arrival=TerminalConstraint(9_500.0, SpeedTarget("TAS", 205.0)),
            altitude_constraints=(AltitudeConstraint.at(1, 9_200.0),),
            speed_constraints=(SpeedConstraint.at(1, SpeedTarget("TAS", 202.0)),),
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


def test_lateral_waypoint_does_not_create_a_state_constraint() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                LateralWaypoint(0.0, 0.0, "START"),
                LateralWaypoint(5_000.0, 2_000.0, "GEOMETRY-ONLY"),
                LateralWaypoint(10_000.0, 0.0, "END"),
            ),
            departure=TerminalConstraint(1_000.0, SpeedTarget("TAS", 100.0)),
            arrival=TerminalConstraint(2_000.0, SpeedTarget("TAS", 120.0)),
            maximum_climb_flight_path_angle_rad=np.deg2rad(10.0),
        )
    )
    route = route.with_profile(
        along_path_m=np.array(
            [0.0, route.waypoint_along_path_m[1], route.total_distance_m]
        ),
        altitude_m=np.array([1_000.0, 1_400.0, 2_000.0]),
        true_airspeed_mps=np.array([100.0, 108.0, 120.0]),
    )

    middle = route.sample(route.waypoint_along_path_m[1])
    assert middle.north_m == pytest.approx(5_000.0)
    assert middle.east_m == pytest.approx(2_000.0)
    assert middle.altitude_m == pytest.approx(1_400.0)
    assert middle.true_airspeed_mps == pytest.approx(108.0)
    assert not route.altitude_target_waypoint_mask[1]
    assert not route.speed_target_waypoint_mask[1]


def test_spatial_trajectory_preserves_infeasible_target_for_planner() -> None:
    intent = FlightIntent(
        waypoints=(
            LateralWaypoint(0.0, 0.0),
            LateralWaypoint(1_000.0, 0.0),
        ),
        departure=TerminalConstraint(0.0, SpeedTarget("TAS", 100.0)),
        arrival=TerminalConstraint(500.0, SpeedTarget("TAS", 100.0)),
        maximum_climb_flight_path_angle_rad=np.deg2rad(5.0),
    )

    route = SpatialTrajectory.from_intent(intent)

    assert np.arctan(route.segment_altitude_gradient[0]) > (
        intent.maximum_climb_flight_path_angle_rad
    )


def test_intent_resolves_cas_mach_and_independent_constraint_bounds() -> None:
    temperature = np.array([285.0, 250.0, 220.0, 250.0])
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=tuple(
                LateralWaypoint(float(index) * 10_000.0, 0.0) for index in range(4)
            ),
            departure=TerminalConstraint(0.0, SpeedTarget("CAS", 120.0)),
            arrival=TerminalConstraint(2_000.0, SpeedTarget("CAS", 110.0)),
            altitude_constraints=(
                AltitudeConstraint(1, 5_000.0, minimum_altitude_m=4_500.0),
                AltitudeConstraint(2, 10_000.0, maximum_altitude_m=10_500.0),
            ),
            speed_constraints=(
                SpeedConstraint(
                    1,
                    SpeedTarget("CAS", 140.0),
                    minimum_value=135.0,
                    maximum_value=145.0,
                ),
                SpeedConstraint(
                    2,
                    SpeedTarget("MACH", 0.76),
                    minimum_value=0.74,
                    maximum_value=0.78,
                ),
            ),
        ),
        static_temperature_k=temperature,
    )

    assert route.waypoint_true_airspeed_mps[1] == pytest.approx(
        true_airspeed_from_calibrated_airspeed(140.0, 5_000.0, 250.0)
    )
    assert route.waypoint_true_airspeed_mps[2] == pytest.approx(
        true_airspeed_from_mach(0.76, 10_000.0, 220.0)
    )
    assert route.waypoint_minimum_altitude_m[1] == pytest.approx(4_500.0)
    assert np.isposinf(route.waypoint_maximum_altitude_m[1])
    assert np.isneginf(route.waypoint_minimum_altitude_m[2])
    assert route.waypoint_maximum_altitude_m[2] == pytest.approx(10_500.0)
    assert np.all(route.speed_bound_waypoint_mask)


def test_selected_command_intent_is_sample_and_hold() -> None:
    altitude = SelectedAltitudeIntent.from_samples(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1_000.0, 1_000.0, 2_000.0, 2_000.0, 500.0],
    )
    speed = SelectedSpeedIntent.from_samples(
        [0.0, 1.0, 2.0],
        [140.0, 140.0, 130.0],
        [0.76, 0.76, 0.74],
    )

    assert altitude.effective_time_s == pytest.approx([0.0, 2.0, 4.0])
    assert altitude.altitude_at(3.9) == pytest.approx(2_000.0)
    assert altitude.altitude_at(4.0) == pytest.approx(500.0)
    assert speed.effective_time_s == pytest.approx([0.0, 2.0])
    assert (
        speed.target_at(
            0.0, pressure_altitude_m=1_000.0, static_temperature_k=280.0
        ).mode
        == "CAS"
    )
    assert (
        speed.target_at(
            0.0, pressure_altitude_m=12_000.0, static_temperature_k=220.0
        ).mode
        == "MACH"
    )


def test_selected_altitude_guidance_commands_climb_capture_hold_and_descent() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                LateralWaypoint(0.0, 0.0),
                LateralWaypoint(20_000.0, 0.0),
            ),
            departure=TerminalConstraint(1_000.0, SpeedTarget("TAS", 100.0)),
            arrival=TerminalConstraint(1_000.0, SpeedTarget("TAS", 100.0)),
        )
    )
    guidance = SelectedAltitudeGuidance(
        lateral_guidance=SpatialPathGuidance(route),
        altitude_intent=SelectedAltitudeIntent(
            np.array([0.0, 20.0]), np.array([2_000.0, 500.0])
        ),
        capture_time_constant_s=100.0,
        maximum_flight_path_command_rate_rad_s=np.deg2rad(10.0),
    )
    common = {
        "north_m": 0.0,
        "east_m": 0.0,
        "true_airspeed_mps": 100.0,
        "flight_path_angle_rad": 0.0,
        "static_temperature_k": 280.0,
        "elapsed_time_s": 1.0,
    }

    climb = guidance.update(
        **common, pressure_altitude_m=1_000.0, simulation_time_s=0.0
    )
    capture = guidance.update(
        **common, pressure_altitude_m=1_900.0, simulation_time_s=10.0
    )
    hold = guidance.update(
        **common, pressure_altitude_m=2_000.0, simulation_time_s=10.0
    )
    descent = guidance.update(
        **common, pressure_altitude_m=2_000.0, simulation_time_s=20.0
    )

    assert climb.vertical_mode == "CLIMB"
    assert climb.reference_air_flight_path_angle_rad > 0.0
    assert capture.vertical_mode == "CAPTURE"
    assert hold.vertical_mode == "ALT_HOLD"
    assert descent.vertical_mode == "DESCENT"
    assert descent.reference_air_flight_path_angle_rad < 0.0
    assert descent.reference_altitude_m == pytest.approx(500.0)
    assert descent.altitude_error_feedback_gain_scale == 0.0


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


def test_3d_wind_correction_reconstructs_ground_velocity() -> None:
    track = np.deg2rad(38.0)
    ground_path_angle = np.deg2rad(3.2)
    true_airspeed = 205.0
    wind = np.array([18.0, -27.0, 4.0])

    solution = wind_corrected_path(
        ground_track_rad=track,
        ground_path_angle_rad=ground_path_angle,
        true_airspeed_mps=true_airspeed,
        wind_north_mps=wind[0],
        wind_east_mps=wind[1],
        wind_down_mps=wind[2],
    )
    air_velocity = true_airspeed * np.array(
        [
            np.cos(solution.air_flight_path_angle_rad)
            * np.cos(solution.air_heading_rad),
            np.cos(solution.air_flight_path_angle_rad)
            * np.sin(solution.air_heading_rad),
            -np.sin(solution.air_flight_path_angle_rad),
        ]
    )
    ground_velocity = air_velocity + wind

    assert np.linalg.norm(air_velocity) == pytest.approx(true_airspeed)
    assert np.arctan2(ground_velocity[1], ground_velocity[0]) == pytest.approx(track)
    assert np.arctan2(-ground_velocity[2], np.hypot(*ground_velocity[:2])) == (
        pytest.approx(ground_path_angle)
    )
    assert np.hypot(*ground_velocity[:2]) == pytest.approx(
        solution.horizontal_ground_speed_mps
    )


def test_air_to_ground_path_conversion_inverts_wind_correction() -> None:
    track = np.deg2rad(-20.0)
    requested_ground_angle = np.deg2rad(-2.5)
    solution = wind_corrected_path(
        ground_track_rad=track,
        ground_path_angle_rad=requested_ground_angle,
        true_airspeed_mps=180.0,
        wind_north_mps=-12.0,
        wind_east_mps=21.0,
        wind_down_mps=-3.0,
    )

    reconstructed = ground_path_angle_from_air_motion(
        ground_track_rad=track,
        air_flight_path_angle_rad=solution.air_flight_path_angle_rad,
        true_airspeed_mps=180.0,
        wind_north_mps=-12.0,
        wind_east_mps=21.0,
        wind_down_mps=-3.0,
    )

    assert reconstructed == pytest.approx(requested_ground_angle)


def test_horizontal_ground_speed_uses_flight_path_angle_and_along_wind() -> None:
    groundspeed = horizontal_ground_speed_mps(
        np.array([100.0]), np.array([np.deg2rad(60.0)]), np.array([10.0])
    )

    assert groundspeed[0] == pytest.approx(60.0)


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
        flight_path_angle_rad=0.0,
        elapsed_time_s=1.0,
    )
    after_backtracking_measurement = guidance.update(
        north_m=17_500.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
        flight_path_angle_rad=0.0,
        elapsed_time_s=1.0,
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
    )
    initial = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
        flight_path_angle_rad=0.0,
        elapsed_time_s=0.25,
    )
    corner = guidance.update(
        north_m=18_000.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
        flight_path_angle_rad=0.0,
        elapsed_time_s=0.25,
    )

    assert initial.reference_air_heading_rad == pytest.approx(0.0)
    assert corner.reference_ground_track_rad == pytest.approx(np.pi / 4.0)
    assert corner.reference_air_heading_rad == pytest.approx(np.deg2rad(0.25))


def test_guidance_uses_shorter_profile_than_lateral_lookahead() -> None:
    guidance = SpatialPathGuidance(
        _route(),
        lookahead_time_s=30.0,
        minimum_lookahead_m=6_000.0,
        maximum_lookahead_m=6_000.0,
        profile_lookahead_time_s=5.0,
        maximum_profile_lookahead_m=1_000.0,
    )

    current_air_path_angle = np.deg2rad(3.0)
    target = guidance.update(
        north_m=10_000.0,
        east_m=0.0,
        true_airspeed_mps=200.0,
        flight_path_angle_rad=current_air_path_angle,
        elapsed_time_s=1.0,
    )

    assert target.target_along_path_m == pytest.approx(16_000.0)
    expected_profile_target = 10_000.0 + min(
        200.0 * np.cos(current_air_path_angle) * 5.0,
        1_000.0,
    )
    assert target.profile_target_along_path_m == pytest.approx(expected_profile_target)
    assert target.reference_altitude_m == pytest.approx(
        9_000.0 + 0.01 * expected_profile_target
    )


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
    controller = InverseFlightMechanicsController(
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
    controller = InverseFlightMechanicsController(
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
