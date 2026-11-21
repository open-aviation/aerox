import numpy as np
import pytest

from aerox import aero
from aerox.models import (
    DragCondition,
    CalibratedTabulatedLiftSurface,
    EffectiveLiftMap,
    GuardedHybridLiftModel,
    ParabolicDragPolar,
    TabulatedLiftSurface,
    ParabolicWaveDragModel,
    KornLockWaveDrag,
)
from aerox.models.atmosphere import unit_reynolds_number
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
    PointMassState,
    simulate_point_mass_zoh,
    simulate_heading_rate_replay_zoh,
)
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import EffectiveThrustMap, MeasuredN1ThrustAdapter
from aerox.models.atmosphere import G0

MASS_KG = 60_000.0
SPEED_MPS = 150.0
WING_AREA_M2 = 100.0


def _plant() -> PointMassModel:
    dynamic_pressure = 0.5 * aero.rho0 * SPEED_MPS**2
    trim_lift_coefficient = MASS_KG * G0 / (dynamic_pressure * WING_AREA_M2)
    return PointMassModel(
        wing_area_m2=WING_AREA_M2,
        lift_map=EffectiveLiftMap(
            "linear_alpha", np.array([trim_lift_coefficient, 0.0])
        ),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(
            engine_count=2,
            rated_thrust_n_per_engine=100_000.0,
            c1=1.0,
            c2=2.0,
            c3=1.0,
        ),
    )


def _state() -> np.ndarray:
    return np.array([SPEED_MPS, 0.0, 0.0, 0.0, 0.0, 0.0, MASS_KG])


def test_named_state_round_trips_the_integrator_vector() -> None:
    state = PointMassState.from_array(_state())

    assert state.true_airspeed_mps == SPEED_MPS
    assert state.mass_kg == MASS_KG
    np.testing.assert_array_equal(state.to_array(), _state())


def _trim_control(plant: PointMassModel, bank_rad: float = 0.0):
    zero_thrust = PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(4.0),
        bank_angle_rad=bank_rad,
        net_thrust_n=0.0,
        static_temperature_k=aero.T0,
    )
    drag = plant.forces(_state(), zero_thrust).drag_n
    return PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(4.0),
        bank_angle_rad=bank_rad,
        net_thrust_n=drag,
        static_temperature_k=aero.T0,
    )


def test_direct_net_thrust_trim_has_zero_speed_and_path_angle_rates():
    plant = _plant()
    control = _trim_control(plant)

    derivative = plant.derivative(_state(), control)

    np.testing.assert_allclose(derivative[:3], 0.0, atol=1e-12)
    assert derivative[3] == SPEED_MPS
    assert derivative[4] == 0.0
    assert derivative[5] == 0.0
    assert derivative[6] < 0.0


def test_effective_drag_increment_adds_dynamic_pressure_drag() -> None:
    plant = _plant()
    clean = PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(4.0),
        bank_angle_rad=0.0,
        net_thrust_n=0.0,
        static_temperature_k=aero.T0,
    )
    configured = PitchBankThrustInput(
        pitch_angle_rad=clean.pitch_angle_rad,
        bank_angle_rad=clean.bank_angle_rad,
        net_thrust_n=clean.net_thrust_n,
        static_temperature_k=clean.static_temperature_k,
        drag_coefficient_increment=0.01,
    )

    added_drag = (
        plant.forces(_state(), configured).drag_n - plant.forces(_state(), clean).drag_n
    )

    assert added_drag == pytest.approx(
        0.5 * aero.rho0 * SPEED_MPS**2 * WING_AREA_M2 * 0.01
    )


def test_single_zoh_step_matches_the_two_node_trajectory_integrator() -> None:
    plant = _plant()
    control = _trim_control(plant)

    final_state = plant.integrate_zoh_step(
        PointMassState.from_array(_state()),
        control,
        1.0,
        integration_rate_hz=10,
    )
    trajectory = simulate_point_mass_zoh(
        plant=plant,
        time_s=np.array([0.0, 1.0]),
        pitch_angle_rad=np.full(2, control.pitch_angle_rad),
        bank_angle_rad=np.full(2, control.bank_angle_rad),
        net_thrust_n=np.full(2, control.net_thrust_n),
        static_temperature_k=np.full(2, control.static_temperature_k),
        initial_true_airspeed_mps=SPEED_MPS,
        initial_flight_path_angle_rad=0.0,
        initial_heading_rad=0.0,
        initial_pressure_altitude_m=0.0,
        initial_mass_kg=MASS_KG,
        integration_rate_hz=10,
    )

    np.testing.assert_allclose(
        final_state.to_array(),
        np.array(
            [
                trajectory.true_airspeed_mps[-1],
                trajectory.flight_path_angle_rad[-1],
                trajectory.heading_rad[-1],
                trajectory.north_position_m[-1],
                trajectory.east_position_m[-1],
                trajectory.pressure_altitude_m[-1],
                trajectory.mass_kg[-1],
            ]
        ),
    )


def test_point_mass_model_can_apply_opt_in_wave_drag() -> None:
    base = _plant()
    wave_plant = PointMassModel(
        wing_area_m2=base.wing_area_m2,
        lift_map=base.lift_map,
        fuel_flow_prior=base.fuel_flow_prior,
        drag_model=ParabolicWaveDragModel(
            base.drag_model,
            KornLockWaveDrag(np.deg2rad(25.0), 0.12),
        ),
    )
    state = _state()
    state[0] = 0.80 * np.sqrt(1.4 * 287.05 * aero.T0)
    control = PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(4.0),
        bank_angle_rad=0.0,
        net_thrust_n=0.0,
        static_temperature_k=aero.T0,
    )

    assert wave_plant.forces(state, control).drag_n > base.forces(state, control).drag_n


class _ConditionAwareDragSpy:
    def __init__(self) -> None:
        self.last_condition: DragCondition | None = None

    def drag_coefficient_at(self, lift_coefficient, condition):
        self.last_condition = condition
        return np.full_like(np.asarray(lift_coefficient, dtype=float), 0.03)

    def drag_coefficient_scalar_at(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        self.last_condition = condition
        return 0.03


def test_point_mass_model_passes_actual_unit_reynolds_to_drag_model() -> None:
    base = _plant()
    spy = _ConditionAwareDragSpy()
    plant = PointMassModel(
        wing_area_m2=base.wing_area_m2,
        lift_map=base.lift_map,
        fuel_flow_prior=base.fuel_flow_prior,
        drag_model=spy,
    )
    control = PitchBankThrustInput(
        pitch_angle_rad=np.deg2rad(4.0),
        bank_angle_rad=0.0,
        net_thrust_n=0.0,
        static_temperature_k=aero.T0,
    )

    plant.forces(_state(), control)

    assert spy.last_condition is not None
    assert float(spy.last_condition.unit_reynolds_number_per_m) == pytest.approx(
        float(unit_reynolds_number(aero.rho0, SPEED_MPS, aero.T0))
    )


def test_bank_generates_coordinated_heading_rate_and_wind_is_external():
    plant = _plant()
    trim = _trim_control(plant, np.deg2rad(15.0))
    control = PitchBankThrustInput(
        pitch_angle_rad=trim.pitch_angle_rad,
        bank_angle_rad=trim.bank_angle_rad,
        net_thrust_n=trim.net_thrust_n,
        static_temperature_k=trim.static_temperature_k,
        wind_north_mps=5.0,
        wind_east_mps=-3.0,
        wind_down_mps=1.0,
    )

    derivative = plant.derivative(_state(), control)

    assert derivative[2] > 0.0
    assert derivative[3] == SPEED_MPS + 5.0
    assert derivative[4] == -3.0
    assert derivative[5] == -1.0


def test_point_mass_simulation_returns_direct_input_and_3d_states():
    plant = _plant()
    control = _trim_control(plant)
    time = np.arange(4.0)

    trajectory = simulate_point_mass_zoh(
        plant=plant,
        time_s=time,
        pitch_angle_rad=np.full(4, control.pitch_angle_rad),
        bank_angle_rad=np.zeros(4),
        net_thrust_n=np.full(4, control.net_thrust_n),
        static_temperature_k=np.full(4, aero.T0),
        initial_true_airspeed_mps=SPEED_MPS,
        initial_flight_path_angle_rad=0.0,
        initial_heading_rad=0.0,
        initial_pressure_altitude_m=0.0,
        initial_mass_kg=MASS_KG,
    )

    np.testing.assert_allclose(trajectory.net_thrust_n, control.net_thrust_n)
    np.testing.assert_allclose(trajectory.pitch_angle_rad, control.pitch_angle_rad)
    assert trajectory.north_position_m[-1] > 400.0
    assert np.all(np.isfinite(trajectory.true_airspeed_mps))
    assert trajectory.mass_kg[-1] < trajectory.mass_kg[0]


def test_measured_n1_adapter_is_exactly_the_optional_replay_boundary():
    thrust_map = EffectiveThrustMap(
        rated_total_thrust_n=200_000.0,
        pressure_exponent=1.0,
        feature_set="quadratic_mach",
        corrected_fan_speed=False,
        coefficients=np.array([0.25, 0.5, 0.0, 0.0, 0.0]),
    )
    adapter = MeasuredN1ThrustAdapter(thrust_map)
    condition = {
        "n1_percent": 75.0,
        "mach": 0.6,
        "pressure_ratio": 0.5,
        "static_temperature_k": 250.0,
    }

    assert adapter.net_thrust_n_scalar(**condition) == thrust_map.thrust_n_scalar(
        **condition
    )


def test_observed_heading_rate_replay_only_overrides_lateral_kinematics():
    plant = _plant()
    control = _trim_control(plant)
    time = np.arange(4.0)
    common = {
        "plant": plant,
        "time_s": time,
        "pitch_angle_rad": np.full(4, control.pitch_angle_rad),
        "bank_angle_rad": np.zeros(4),
        "net_thrust_n": np.full(4, control.net_thrust_n),
        "static_temperature_k": np.full(4, aero.T0),
        "initial_true_airspeed_mps": SPEED_MPS,
        "initial_flight_path_angle_rad": 0.0,
        "initial_heading_rad": 0.0,
        "initial_pressure_altitude_m": 0.0,
        "initial_mass_kg": MASS_KG,
    }

    physical = simulate_point_mass_zoh(**common)
    replay = simulate_heading_rate_replay_zoh(
        **common,
        observed_heading_rate_rad_s=np.full(4, np.deg2rad(1.0)),
    )

    np.testing.assert_allclose(
        replay.trajectory.true_airspeed_mps, physical.true_airspeed_mps
    )
    np.testing.assert_allclose(
        replay.trajectory.flight_path_angle_rad, physical.flight_path_angle_rad
    )
    np.testing.assert_allclose(
        replay.trajectory.pressure_altitude_m, physical.pressure_altitude_m
    )
    np.testing.assert_allclose(replay.trajectory.mass_kg, physical.mass_kg)
    assert np.isclose(np.rad2deg(replay.trajectory.heading_rad[-1]), 3.0)
    assert np.all(replay.implied_lateral_force_residual_n > 0.0)
    np.testing.assert_allclose(replay.coordinated_heading_rate_rad_s, 0.0)


def test_observed_heading_rate_residual_is_zero_for_coordinated_turn_rate():
    plant = _plant()
    control = _trim_control(plant, bank_rad=np.deg2rad(15.0))
    state = _state()
    coordinated_rate = plant.derivative(state, control)[2]

    replay = simulate_heading_rate_replay_zoh(
        plant=plant,
        time_s=np.arange(2.0),
        observed_heading_rate_rad_s=np.full(2, coordinated_rate),
        pitch_angle_rad=np.full(2, control.pitch_angle_rad),
        bank_angle_rad=np.full(2, control.bank_angle_rad),
        net_thrust_n=np.full(2, control.net_thrust_n),
        static_temperature_k=np.full(2, aero.T0),
        initial_true_airspeed_mps=SPEED_MPS,
        initial_flight_path_angle_rad=0.0,
        initial_heading_rad=0.0,
        initial_pressure_altitude_m=0.0,
        initial_mass_kg=MASS_KG,
    )

    assert abs(replay.implied_lateral_force_residual_n[0]) < 1e-9


def test_pitch_bank_thrust_input_rejects_negative_realized_thrust():
    with np.testing.assert_raises(ValueError):
        PitchBankThrustInput(0.0, 0.0, -1.0, aero.T0)


def test_pitch_aware_surface_scales_pitch_before_subtracting_gamma():
    surface = TabulatedLiftSurface(
        mach=np.array([0.2, 0.8]),
        angle_of_attack_rad=np.deg2rad(np.array([0.0, 10.0])),
        lift_grid=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    calibrated = CalibratedTabulatedLiftSurface(
        surface=surface,
        pitch_scale=1.1,
        decoded_pitch_bias_rad=np.deg2rad(0.5),
        combined_offset_rad=np.deg2rad(1.0),
        mach_offset_rad_per_mach=0.0,
        registration_angle_offset_rad=np.deg2rad(-0.2),
    )

    angle, coefficient = calibrated.angle_and_lift_coefficient_scalar(
        pitch_angle_rad=np.deg2rad(5.0),
        flight_path_angle_rad=np.deg2rad(2.0),
        mach=0.5,
    )

    assert np.isclose(np.rad2deg(angle), 3.75)
    assert np.isclose(coefficient, 0.375)


def test_guarded_hybrid_uses_surface_in_core_and_empirical_outside():
    surface = TabulatedLiftSurface(
        mach=np.array([0.2, 0.8]),
        angle_of_attack_rad=np.deg2rad(np.array([-1.0, 9.0])),
        lift_grid=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
    calibrated = CalibratedTabulatedLiftSurface(
        surface=surface,
        pitch_scale=1.0,
        decoded_pitch_bias_rad=0.0,
        combined_offset_rad=0.0,
        mach_offset_rad_per_mach=0.0,
    )
    empirical = EffectiveLiftMap("linear_alpha", np.array([0.2, 0.0]))
    hybrid = GuardedHybridLiftModel(
        surface_model=calibrated,
        empirical_fallback=empirical,
        lower_alpha_core_rad=np.deg2rad(0.0),
        upper_alpha_core_rad=np.deg2rad(8.0),
        lower_mach_core=0.3,
        upper_mach_core=0.7,
    )

    _, core = hybrid.angle_and_lift_coefficient_scalar(
        pitch_angle_rad=np.deg2rad(4.0), flight_path_angle_rad=0.0, mach=0.5
    )
    _, outside = hybrid.angle_and_lift_coefficient_scalar(
        pitch_angle_rad=np.deg2rad(10.0), flight_path_angle_rad=0.0, mach=0.5
    )

    assert np.isclose(core, 0.5)
    assert np.isclose(outside, 0.2)

    half_weight = GuardedHybridLiftModel(
        surface_model=calibrated,
        empirical_fallback=empirical,
        lower_alpha_core_rad=np.deg2rad(0.0),
        upper_alpha_core_rad=np.deg2rad(8.0),
        lower_mach_core=0.3,
        upper_mach_core=0.7,
        maximum_surface_weight=0.5,
    )
    _, blended = half_weight.angle_and_lift_coefficient_scalar(
        pitch_angle_rad=np.deg2rad(4.0), flight_path_angle_rad=0.0, mach=0.5
    )
    assert np.isclose(blended, 0.35)

    with pytest.raises(ValueError, match="maximum_surface_weight"):
        GuardedHybridLiftModel(
            surface_model=calibrated,
            empirical_fallback=empirical,
            lower_alpha_core_rad=np.deg2rad(0.0),
            upper_alpha_core_rad=np.deg2rad(8.0),
            lower_mach_core=0.3,
            upper_mach_core=0.7,
            maximum_surface_weight=1.1,
        )
