import numpy as np
import pytest
from dataclasses import replace

from aerox.fms import (
    ConfigurationCommand,
    ConfigurationMode,
    FlightIntent,
    GuardComparison,
    GuardCondition,
    GuardLogic,
    GuardMetric,
    GuidanceAlternative,
    GuidanceEvidence,
    GuidanceObservation,
    GuidanceProgram,
    GuidanceProgramExecutor,
    GuidanceSegment,
    LateralCommand,
    LateralMode,
    LateralWaypoint,
    SpeedCommand,
    SpeedGuidanceStage,
    SpeedMode,
    SpeedSelectorMode,
    SpeedTarget,
    SpatialTrajectory,
    TerminalConstraint,
    ThrustCommand,
    ThrustMode,
    TransitionGuard,
    VerticalCommand,
    VerticalGuidanceStage,
    VerticalMode,
)
from aerox.guidance import SpatialPathGuidance


def _time_guard(time_s: float) -> TransitionGuard:
    return TransitionGuard(
        conditions=(
            GuardCondition(
                GuardMetric.ELAPSED_TIME_S,
                GuardComparison.GE,
                time_s,
            ),
        )
    )


def _segment(
    segment_id: str,
    *,
    altitude_m: float,
    speed_mode: SpeedMode,
    transition_time_s: float | None,
    evidence: GuidanceEvidence | None = None,
) -> GuidanceSegment:
    return GuidanceSegment(
        segment_id=segment_id,
        lateral=LateralCommand(LateralMode.ROUTE),
        vertical=VerticalCommand(
            VerticalMode.ALTITUDE,
            altitude_m=altitude_m,
            evidence=GuidanceEvidence() if evidence is None else evidence,
        ),
        speed=SpeedCommand(
            speed_mode,
            calibrated_airspeed_mps=140.0,
            mach=0.76,
        ),
        thrust=ThrustCommand(ThrustMode.AUTO),
        transition=(
            None if transition_time_s is None else _time_guard(transition_time_s)
        ),
    )


def _program() -> GuidanceProgram:
    evidence = GuidanceEvidence(
        confidence=0.82,
        source_ids=("qar:selector", "qar:air-data"),
        alternatives=(GuidanceAlternative("ALTITUDE_HOLD", 0.18),),
    )
    return GuidanceProgram(
        program_id="flight-42",
        segments=(
            _segment(
                "climb",
                altitude_m=11_000.0,
                speed_mode=SpeedMode.CAS_MACH,
                transition_time_s=600.0,
                evidence=evidence,
            ),
            _segment(
                "descent",
                altitude_m=1_000.0,
                speed_mode=SpeedMode.MACH_CAS,
                transition_time_s=None,
            ),
        ),
        metadata={"trajectory_source": "QAR", "aircraft_type": "A320"},
    )


def _route() -> SpatialTrajectory:
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                LateralWaypoint(0.0, 0.0),
                LateralWaypoint(100_000.0, 0.0),
            ),
            departure=TerminalConstraint(1_000.0, SpeedTarget("TAS", 160.0)),
            arrival=TerminalConstraint(1_000.0, SpeedTarget("TAS", 160.0)),
        )
    )


def test_guidance_program_round_trips_yaml_with_evidence() -> None:
    program = _program()

    serialized = program.to_yaml()
    restored = GuidanceProgram.from_yaml(serialized)

    assert restored == program
    assert "!!python" not in serialized
    assert restored.segments[0].vertical.evidence.confidence == pytest.approx(0.82)
    assert restored.segments[0].vertical.evidence.alternatives[0].label == (
        "ALTITUDE_HOLD"
    )


def test_configuration_drag_command_compiles_and_executes() -> None:
    program = _program()
    configured_segment = replace(
        program.segments[1],
        configuration=ConfigurationCommand(
            ConfigurationMode.DRAG_INCREMENT,
            drag_coefficient_increment=0.015,
        ),
    )
    configured = replace(
        program,
        segments=(program.segments[0], configured_segment),
    )
    compiled = configured.compile_selected_guidance()
    guidance = compiled.build_guidance(SpatialPathGuidance(_route()))

    before = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=1_000.0,
        simulation_time_s=599.0,
        static_temperature_k=270.0,
        elapsed_time_s=1.0,
    )
    after = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=1_000.0,
        simulation_time_s=600.0,
        static_temperature_k=270.0,
        elapsed_time_s=1.0,
    )

    assert before.configuration_mode == "CLEAN"
    assert before.drag_coefficient_increment == 0.0
    assert after.configuration_mode == "DRAG_INCREMENT"
    assert after.drag_coefficient_increment == pytest.approx(0.015)


def test_transition_guards_support_stateful_multi_condition_advancement() -> None:
    transition = TransitionGuard(
        conditions=(
            GuardCondition(GuardMetric.ALONG_PATH_M, GuardComparison.GE, 50_000.0),
            GuardCondition(GuardMetric.ALTITUDE_ERROR_M, GuardComparison.ABS_LE, 30.0),
        ),
        logic=GuardLogic.ALL,
    )
    program = GuidanceProgram(
        "guard-test",
        (
            GuidanceSegment(
                "capture",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=10_000.0),
                SpeedCommand(
                    SpeedMode.MACH,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
                transition=transition,
            ),
            _segment(
                "hold",
                altitude_m=10_000.0,
                speed_mode=SpeedMode.MACH,
                transition_time_s=None,
            ),
        ),
    )

    assert (
        program.advance(
            0,
            GuidanceObservation(along_path_m=60_000.0, altitude_error_m=100.0),
        )
        == 0
    )
    assert (
        program.advance(
            0,
            GuidanceObservation(along_path_m=60_000.0, altitude_error_m=-20.0),
        )
        == 1
    )


def test_online_executor_records_monotonic_transition_events() -> None:
    executor = GuidanceProgramExecutor(_program())

    assert executor.update(GuidanceObservation(elapsed_time_s=599.0)).segment_id == (
        "climb"
    )
    assert executor.transition_events == ()
    assert executor.update(GuidanceObservation(elapsed_time_s=600.0)).segment_id == (
        "descent"
    )
    assert executor.segment_index == 1
    assert len(executor.transition_events) == 1
    assert executor.transition_events[0].from_segment_index == 0
    assert executor.transition_events[0].to_segment_index == 1
    assert executor.transition_events[0].observation.elapsed_time_s == 600.0

    executor.reset()

    assert executor.segment_index == 0
    assert executor.transition_events == ()


def test_compile_selected_guidance_preserves_timing_targets_and_modes() -> None:
    compiled = _program().compile_selected_guidance()

    assert compiled.effective_time_s == pytest.approx([0.0, 600.0])
    assert compiled.altitude_intent.altitude_at(599.9) == pytest.approx(11_000.0)
    assert compiled.altitude_intent.altitude_at(600.0) == pytest.approx(1_000.0)
    assert compiled.speed_intent.selected_calibrated_airspeed_mps == pytest.approx(
        [140.0, 140.0]
    )
    assert compiled.speed_guidance_intent.mode_at(599.9) == "CAS_MACH"
    assert compiled.speed_guidance_intent.mode_at(600.0) == "MACH_CAS"


def test_independent_speed_channel_round_trips_and_compiles_on_its_own_clock() -> None:
    source = _program()
    program = GuidanceProgram(
        source.program_id,
        source.segments,
        format_version=2,
        metadata=source.metadata,
        speed_stages=(
            SpeedGuidanceStage(
                "speed-low",
                SpeedCommand(
                    SpeedMode.CAS,
                    calibrated_airspeed_mps=120.0,
                    mach=0.70,
                ),
                transition=_time_guard(300.0),
            ),
            SpeedGuidanceStage(
                "speed-high",
                SpeedCommand(
                    SpeedMode.CAS_MACH,
                    calibrated_airspeed_mps=145.0,
                    mach=0.78,
                    active_selector=SpeedSelectorMode.MACH,
                ),
            ),
        ),
    )

    restored = GuidanceProgram.from_yaml(program.to_yaml())
    compiled = restored.compile_selected_guidance()

    assert restored == program
    assert compiled.effective_time_s == pytest.approx([0.0, 600.0])
    assert compiled.speed_effective_time_s == pytest.approx([0.0, 300.0])
    assert compiled.speed_intent.target_at(
        299.0,
        pressure_altitude_m=1_000.0,
        static_temperature_k=270.0,
    ).calibrated_airspeed_mps == pytest.approx(120.0)
    assert compiled.speed_guidance_intent.mode_at(300.0) == "CAS_MACH"
    assert compiled.speed_guidance_intent.active_selector_at(299.0) is None
    assert compiled.speed_guidance_intent.active_selector_at(300.0) == "MACH"

    guidance = compiled.build_guidance(SpatialPathGuidance(_route()))
    target = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=1_000.0,
        simulation_time_s=300.0,
        static_temperature_k=270.0,
        elapsed_time_s=1.0,
    )
    assert target.speed_guidance_mode == "CAS_MACH"
    assert target.speed_mode == "MACH"


def test_explicit_speed_mode_rejects_a_conflicting_active_selector() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        SpeedCommand(
            SpeedMode.CAS,
            calibrated_airspeed_mps=140.0,
            mach=0.76,
            active_selector=SpeedSelectorMode.MACH,
        )


def test_independent_speed_executor_advances_without_joint_segment_transition() -> None:
    source = _program()
    program = GuidanceProgram(
        source.program_id,
        source.segments,
        format_version=2,
        speed_stages=(
            SpeedGuidanceStage(
                "speed-cas",
                SpeedCommand(
                    SpeedMode.CAS,
                    calibrated_airspeed_mps=120.0,
                    mach=0.70,
                ),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.PRESSURE_ALTITUDE_M,
                            GuardComparison.GE,
                            5_000.0,
                        ),
                    )
                ),
            ),
            SpeedGuidanceStage(
                "speed-mach",
                SpeedCommand(
                    SpeedMode.MACH,
                    calibrated_airspeed_mps=145.0,
                    mach=0.78,
                ),
            ),
        ),
    )
    executor = GuidanceProgramExecutor(program)

    active = executor.update(
        GuidanceObservation(elapsed_time_s=100.0, pressure_altitude_m=5_000.0)
    )

    assert active.segment_id == "climb"
    assert executor.segment_index == 0
    assert executor.speed_stage_index == 1
    assert executor.current_speed.mode == SpeedMode.MACH
    assert executor.transition_events == ()
    assert len(executor.speed_transition_events) == 1


def test_independent_vertical_channel_round_trips_and_compiles_on_its_own_clock() -> (
    None
):
    source = _program()
    program = GuidanceProgram(
        source.program_id,
        source.segments,
        format_version=3,
        metadata=source.metadata,
        vertical_stages=(
            VerticalGuidanceStage(
                "vertical-low",
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=2_000.0),
                transition=_time_guard(300.0),
            ),
            VerticalGuidanceStage(
                "vertical-high",
                VerticalCommand(VerticalMode.ALTITUDE_HOLD, altitude_m=9_000.0),
            ),
        ),
    )

    restored = GuidanceProgram.from_yaml(program.to_yaml())
    compiled = restored.compile_selected_guidance()

    assert restored == program
    assert compiled.effective_time_s == pytest.approx([0.0, 600.0])
    assert compiled.vertical_effective_time_s == pytest.approx([0.0, 300.0])
    assert compiled.altitude_intent.altitude_at(299.0) == pytest.approx(2_000.0)
    assert compiled.altitude_intent.altitude_at(300.0) == pytest.approx(9_000.0)

    guidance = compiled.build_guidance(SpatialPathGuidance(_route()))
    target = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=2_000.0,
        simulation_time_s=300.0,
        static_temperature_k=270.0,
        elapsed_time_s=1.0,
    )
    assert target.reference_altitude_m == pytest.approx(9_000.0)


def test_independent_vertical_executor_advances_without_joint_segment_transition() -> (
    None
):
    source = _program()
    program = GuidanceProgram(
        source.program_id,
        source.segments,
        format_version=3,
        vertical_stages=(
            VerticalGuidanceStage(
                "vertical-low",
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=1_000.0),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.ALONG_PATH_M,
                            GuardComparison.GE,
                            20_000.0,
                        ),
                    )
                ),
            ),
            VerticalGuidanceStage(
                "vertical-high",
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=5_000.0),
            ),
        ),
    )
    guidance = program.build_intent_guidance(SpatialPathGuidance(_route()))

    target = guidance.update(
        north_m=20_000.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=1_000.0,
        simulation_time_s=100.0,
        static_temperature_k=270.0,
        elapsed_time_s=1.0,
    )

    assert guidance.executor.segment_index == 0
    assert guidance.executor.vertical_stage_index == 1
    assert guidance.executor.current_vertical.mode == VerticalMode.ALTITUDE
    assert guidance.executor.transition_events == ()
    assert len(guidance.executor.vertical_transition_events) == 1
    assert guidance.active_vertical_stage_history == (1,)
    assert target.reference_altitude_m == pytest.approx(5_000.0)


def test_vertical_executor_recomputes_command_relative_error_after_stage_change() -> (
    None
):
    source = _program()
    program = GuidanceProgram(
        source.program_id,
        source.segments,
        format_version=3,
        vertical_stages=(
            VerticalGuidanceStage(
                "direct",
                VerticalCommand(VerticalMode.VERTICAL_SPEED, vertical_speed_mps=-5.0),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.ALONG_PATH_M,
                            GuardComparison.GE,
                            20_000.0,
                        ),
                    )
                ),
            ),
            VerticalGuidanceStage(
                "capture",
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=1_000.0),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.ALTITUDE_ERROR_M,
                            GuardComparison.ABS_LE,
                            30.0,
                        ),
                    )
                ),
            ),
            VerticalGuidanceStage(
                "hold",
                VerticalCommand(VerticalMode.ALTITUDE_HOLD, altitude_m=1_000.0),
            ),
        ),
    )
    executor = GuidanceProgramExecutor(program)

    executor.update(GuidanceObservation(elapsed_time_s=0.0, along_path_m=20_000.0))

    assert executor.vertical_stage_index == 1
    assert len(executor.vertical_transition_events) == 1

    executor.update(GuidanceObservation(elapsed_time_s=0.0, altitude_error_m=0.0))

    assert executor.vertical_stage_index == 2
    assert len(executor.vertical_transition_events) == 2


def test_compiled_program_executes_flight_path_angle_and_vertical_speed_modes() -> None:
    program = GuidanceProgram(
        "direct-vertical-modes",
        (
            GuidanceSegment(
                "fpa",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(
                    VerticalMode.FLIGHT_PATH_ANGLE,
                    flight_path_angle_rad=np.deg2rad(-3.0),
                ),
                SpeedCommand(
                    SpeedMode.MACH_CAS,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
                transition=_time_guard(10.0),
            ),
            GuidanceSegment(
                "vertical-speed",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(
                    VerticalMode.VERTICAL_SPEED,
                    vertical_speed_mps=-5.0,
                ),
                SpeedCommand(
                    SpeedMode.MACH_CAS,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
            ),
        ),
    )
    compiled = program.compile_selected_guidance()
    guidance = compiled.build_guidance(
        SpatialPathGuidance(_route()),
        maximum_flight_path_command_rate_rad_s=np.deg2rad(10.0),
    )
    common = {
        "north_m": 0.0,
        "east_m": 0.0,
        "true_airspeed_mps": 150.0,
        "flight_path_angle_rad": 0.0,
        "pressure_altitude_m": 5_000.0,
        "static_temperature_k": 260.0,
        "elapsed_time_s": 1.0,
    }

    fpa = guidance.update(**common, simulation_time_s=0.0)
    vertical_speed = guidance.update(**common, simulation_time_s=10.0)

    assert compiled.altitude_intent is None
    assert compiled.vertical_intent is not None
    assert fpa.vertical_mode == "FLIGHT_PATH_ANGLE"
    assert fpa.reference_air_flight_path_angle_rad == pytest.approx(np.deg2rad(-3.0))
    assert vertical_speed.vertical_mode == "VERTICAL_SPEED"
    assert vertical_speed.reference_air_flight_path_angle_rad == pytest.approx(
        np.arcsin(-5.0 / 150.0)
    )


def test_compiled_path_mode_uses_the_bound_route_vertical_profile() -> None:
    route = _route().with_profile(
        along_path_m=np.array([0.0, 100_000.0]),
        altitude_m=np.array([5_000.0, 1_000.0]),
        true_airspeed_mps=np.array([150.0, 150.0]),
    )
    program = GuidanceProgram(
        "path-mode",
        (
            GuidanceSegment(
                "path",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(VerticalMode.PATH),
                SpeedCommand(
                    SpeedMode.MACH_CAS,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
            ),
        ),
    )
    guidance = program.compile_selected_guidance().build_guidance(
        SpatialPathGuidance(route),
        maximum_flight_path_command_rate_rad_s=np.deg2rad(10.0),
    )

    target = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=150.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=5_000.0,
        simulation_time_s=0.0,
        static_temperature_k=260.0,
        elapsed_time_s=1.0,
    )

    assert target.vertical_mode == "PATH"
    assert target.reference_altitude_m < 5_000.0
    assert target.reference_air_flight_path_angle_rad < 0.0
    assert target.altitude_error_feedback_gain_scale == 1.0


def test_normalized_thrust_rule_round_trips_and_compiles() -> None:
    source = _program()
    first = replace(
        source.segments[0],
        thrust=ThrustCommand(
            ThrustMode.THRUST_FRACTION,
            available_thrust_fraction=0.72,
        ),
    )
    program = GuidanceProgram(
        source.program_id,
        (first, source.segments[1]),
        metadata=source.metadata,
    )

    restored = GuidanceProgram.from_yaml(program.to_yaml())
    compiled = restored.compile_selected_guidance()

    assert restored.segments[0].thrust.available_thrust_fraction == pytest.approx(0.72)
    assert compiled.thrust_intent.target_at(599.0).mode == "THRUST_FRACTION"
    assert compiled.thrust_intent.target_at(
        599.0
    ).available_thrust_fraction == pytest.approx(0.72)
    assert compiled.thrust_intent.target_at(600.0).mode == "AUTO"


def test_compiled_program_commands_speed_mode_independently_of_vertical_direction() -> (
    None
):
    program = GuidanceProgram(
        "explicit-speed-mode",
        (
            _segment(
                "descending-with-cas-mach",
                altitude_m=500.0,
                speed_mode=SpeedMode.CAS_MACH,
                transition_time_s=10.0,
            ),
            _segment(
                "climbing-with-mach-cas",
                altitude_m=12_000.0,
                speed_mode=SpeedMode.MACH_CAS,
                transition_time_s=None,
            ),
        ),
    )
    compiled = program.compile_selected_guidance()
    guidance = compiled.build_guidance(
        SpatialPathGuidance(_route()),
        maximum_flight_path_command_rate_rad_s=np.deg2rad(10.0),
    )
    crossover = compiled.speed_intent.target_at(
        0.0,
        pressure_altitude_m=1_000.0,
        static_temperature_k=270.0,
    ).crossover_altitude_m
    common = {
        "north_m": 0.0,
        "east_m": 0.0,
        "true_airspeed_mps": 190.0,
        "flight_path_angle_rad": 0.0,
        "static_temperature_k": 270.0,
        "elapsed_time_s": 1.0,
    }

    descending = guidance.update(
        **common,
        pressure_altitude_m=crossover - 500.0,
        simulation_time_s=0.0,
    )
    climbing = guidance.update(
        **common,
        pressure_altitude_m=crossover + 500.0,
        simulation_time_s=10.0,
    )

    assert descending.vertical_mode == "DESCENT"
    assert descending.speed_guidance_mode == "CAS_MACH"
    assert descending.speed_mode == "CAS"
    assert climbing.vertical_mode == "CLIMB"
    assert climbing.speed_guidance_mode == "MACH_CAS"
    assert climbing.speed_mode == "MACH"


@pytest.mark.parametrize(
    ("mode", "expected_reference"),
    [(SpeedMode.CAS, "CAS"), (SpeedMode.MACH, "MACH")],
)
def test_compiled_program_can_hold_an_explicit_cas_or_mach_reference(
    mode: SpeedMode, expected_reference: str
) -> None:
    program = GuidanceProgram(
        "single-reference",
        (
            _segment(
                "hold",
                altitude_m=1_000.0,
                speed_mode=mode,
                transition_time_s=None,
            ),
        ),
    )
    guidance = program.compile_selected_guidance().build_guidance(
        SpatialPathGuidance(_route())
    )

    target = guidance.update(
        north_m=0.0,
        east_m=0.0,
        true_airspeed_mps=180.0,
        flight_path_angle_rad=0.0,
        pressure_altitude_m=12_000.0 if mode == SpeedMode.CAS else 1_000.0,
        simulation_time_s=0.0,
        static_temperature_k=250.0,
        elapsed_time_s=1.0,
    )

    assert target.speed_guidance_mode == "SPEED_HOLD"
    assert target.speed_mode == expected_reference


def test_compiler_rejects_non_time_guard_without_changing_its_semantics() -> None:
    program = GuidanceProgram(
        "spatial-transition",
        (
            GuidanceSegment(
                "first",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=5_000.0),
                SpeedCommand(
                    SpeedMode.CAS,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.ALONG_PATH_M,
                            GuardComparison.GE,
                            20_000.0,
                        ),
                    )
                ),
            ),
            _segment(
                "second",
                altitude_m=5_000.0,
                speed_mode=SpeedMode.CAS,
                transition_time_s=None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="ELAPSED_TIME_S GE"):
        program.compile_selected_guidance()


def test_intent_driven_guidance_activates_command_from_route_progress() -> None:
    program = GuidanceProgram(
        "spatial-online",
        (
            GuidanceSegment(
                "low",
                LateralCommand(LateralMode.ROUTE),
                VerticalCommand(VerticalMode.ALTITUDE, altitude_m=1_000.0),
                SpeedCommand(
                    SpeedMode.CAS,
                    calibrated_airspeed_mps=140.0,
                    mach=0.76,
                ),
                transition=TransitionGuard(
                    (
                        GuardCondition(
                            GuardMetric.ALONG_PATH_M,
                            GuardComparison.GE,
                            20_000.0,
                        ),
                    )
                ),
            ),
            _segment(
                "high",
                altitude_m=5_000.0,
                speed_mode=SpeedMode.MACH,
                transition_time_s=None,
            ),
        ),
    )
    guidance = program.build_intent_guidance(SpatialPathGuidance(_route()))
    common = {
        "east_m": 0.0,
        "true_airspeed_mps": 180.0,
        "flight_path_angle_rad": 0.0,
        "pressure_altitude_m": 1_000.0,
        "static_temperature_k": 270.0,
        "elapsed_time_s": 1.0,
    }

    before = guidance.update(
        **common,
        north_m=10_000.0,
        simulation_time_s=100.0,
    )
    after = guidance.update(
        **common,
        north_m=20_000.0,
        simulation_time_s=101.0,
    )

    assert before.reference_altitude_m == pytest.approx(1_000.0)
    assert after.reference_altitude_m == pytest.approx(5_000.0)
    assert guidance.active_segment_history == (0, 1)
    assert guidance.executor.transition_events[0].observation.along_path_m == (
        pytest.approx(20_000.0)
    )


def test_program_schema_rejects_unknown_yaml_fields() -> None:
    data = _program().to_dict()
    data["aircraft_coefficients"] = {"drag": 0.1}

    with pytest.raises(ValueError, match="unknown guidance program fields"):
        GuidanceProgram.from_dict(data)
