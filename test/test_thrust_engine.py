import numpy as np

from aerox.models.openap import CalibratedFuelFlow, OpenAPFuelFlowPrior
from aerox.models.thrust import (
    AsymmetricFirstOrderN1Actuator,
    AutothrustMode,
    EffectiveN1Envelope,
    EffectiveThrustMap,
    thrust_map_features,
    fan_speed_fraction,
    n1_envelope_features,
)


def test_asymmetric_n1_actuator_has_exact_rise_and_fall_response():
    actuator = AsymmetricFirstOrderN1Actuator(
        rise_time_constant_s=2.0,
        fall_time_constant_s=4.0,
        minimum_n1_percent=20.0,
        maximum_n1_percent=100.0,
    )

    rising = actuator.step(40.0, 80.0, 2.0)
    falling = actuator.step(80.0, 40.0, 2.0)

    assert np.isclose(rising, 80.0 + (40.0 - 80.0) * np.exp(-1.0))
    assert np.isclose(falling, 40.0 + (80.0 - 40.0) * np.exp(-0.5))
    assert rising - 40.0 > 80.0 - falling


def test_asymmetric_n1_actuator_saturates_commands_and_simulates_zoh():
    actuator = AsymmetricFirstOrderN1Actuator(
        rise_time_constant_s=1.0,
        fall_time_constant_s=2.0,
        minimum_n1_percent=20.0,
        maximum_n1_percent=100.0,
    )
    time = np.array([0.0, 0.5, 2.0, 3.0])
    commands = np.array([120.0, 80.0, 10.0, 50.0])

    state = actuator.simulate_zoh(time, commands, initial_n1_percent=40.0)

    assert state.shape == time.shape
    assert np.all((20.0 <= state) & (state <= 100.0))
    assert np.isclose(state[1], actuator.step(40.0, 100.0, 0.5))
    assert np.isclose(state[2], actuator.step(state[1], 80.0, 1.5))
    assert np.isclose(state[3], actuator.step(state[2], 20.0, 1.0))


def test_asymmetric_n1_actuator_rejects_unphysical_parameters_and_state():
    with np.testing.assert_raises(ValueError):
        AsymmetricFirstOrderN1Actuator(0.0, 1.0)

    actuator = AsymmetricFirstOrderN1Actuator(1.0, 1.0)
    with np.testing.assert_raises(ValueError):
        actuator.derivative(120.0, 80.0)
    with np.testing.assert_raises(ValueError):
        actuator.step(80.0, np.nan, 1.0)


def test_effective_n1_envelope_maps_modes_and_normalized_demand():
    envelope = EffectiveN1Envelope(
        feature_set="altitude_linear",
        idle_coefficients=np.array([20.0, 10.0]),
        climb_coefficients=np.array([85.0, 5.0]),
    )
    condition = {
        "pressure_altitude_m": 6_000.0,
        "mach": 0.5,
        "static_temperature_k": 249.15,
    }

    assert np.isclose(
        envelope.commanded_n1_percent(mode=AutothrustMode.FIXED_IDLE, **condition),
        25.0,
    )
    assert np.isclose(
        envelope.commanded_n1_percent(mode=AutothrustMode.FIXED_CLIMB, **condition),
        87.5,
    )
    assert np.isclose(
        envelope.commanded_n1_percent(
            mode=AutothrustMode.VARIABLE_SPEED,
            demand_fraction=0.4,
            **condition,
        ),
        50.0,
    )
    demand = envelope.normalized_demand(
        n1_percent=np.array([25.0, 50.0, 87.5]),
        pressure_altitude_m=np.full(3, 6_000.0),
        mach=np.full(3, 0.5),
        static_temperature_k=np.full(3, 249.15),
    )
    np.testing.assert_allclose(demand, [0.0, 0.4, 1.0])


def test_climb_limit_can_be_used_without_promoting_invalid_idle_surface():
    envelope = EffectiveN1Envelope(
        feature_set="altitude_linear",
        idle_coefficients=np.array([100.0, 0.0]),
        climb_coefficients=np.array([85.0, 5.0]),
    )
    condition = {
        "pressure_altitude_m": np.array([6_000.0]),
        "mach": np.array([0.5]),
        "static_temperature_k": np.array([249.15]),
    }

    np.testing.assert_allclose(envelope.climb_limit_percent(**condition), [87.5])
    with np.testing.assert_raises(ValueError):
        envelope.bounds_percent(**condition)


def test_n1_envelope_temperature_feature_uses_isa_deviation():
    design = n1_envelope_features(
        pressure_altitude_m=np.array([0.0, 12_000.0]),
        mach=np.array([0.5, 0.6]),
        static_temperature_k=np.array([288.15, 226.65]),
        feature_set="altitude_mach_temperature",
    )

    np.testing.assert_allclose(design[:, 1], [0.0, 1.0])
    np.testing.assert_allclose(design[:, -1], [0.0, 0.5])


def test_corrected_fan_speed_uses_inlet_total_temperature():
    raw = fan_speed_fraction(
        np.array([80.0]),
        np.array([0.0]),
        np.array([288.15]),
        corrected=False,
    )
    corrected = fan_speed_fraction(
        np.array([80.0]),
        np.array([0.0]),
        np.array([288.15]),
        corrected=True,
    )

    np.testing.assert_allclose(raw, [0.8])
    np.testing.assert_allclose(corrected, [0.8])


def test_effective_thrust_map_scales_with_pressure_power_and_rated_thrust():
    design = thrust_map_features(
        n1_percent=np.array([80.0, 80.0]),
        mach=np.array([0.6, 0.6]),
        pressure_ratio=np.array([1.0, 0.25]),
        static_temperature_k=np.array([288.15, 288.15]),
        pressure_exponent=1.0,
        feature_set="quadratic_mach",
        corrected_fan_speed=False,
    )
    model = EffectiveThrustMap(
        rated_total_thrust_n=200_000.0,
        pressure_exponent=1.0,
        feature_set="quadratic_mach",
        corrected_fan_speed=False,
        coefficients=np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(design[:, 0], [1.0, 0.25])
    np.testing.assert_allclose(
        model.thrust_n(
            n1_percent=np.array([80.0, 80.0]),
            mach=np.array([0.6, 0.6]),
            pressure_ratio=np.array([1.0, 0.25]),
            static_temperature_k=np.array([288.15, 288.15]),
        ),
        [100_000.0, 25_000.0],
    )
    assert np.isclose(
        model.thrust_n_scalar(
            n1_percent=80.0,
            mach=0.6,
            pressure_ratio=0.25,
            static_temperature_k=288.15,
        ),
        25_000.0,
    )


def test_effective_thrust_map_inverts_n1_and_saturates_bounds():
    model = EffectiveThrustMap(
        rated_total_thrust_n=200_000.0,
        pressure_exponent=1.0,
        feature_set="quadratic_mach",
        corrected_fan_speed=False,
        coefficients=np.array([0.5, 1.0, 0.0, 0.0, 0.0]),
    )
    condition = {
        "mach": 0.6,
        "pressure_ratio": 1.0,
        "static_temperature_k": 288.15,
        "minimum_n1_percent": 20.0,
        "maximum_n1_percent": 100.0,
    }
    target = model.thrust_n_scalar(
        n1_percent=73.0,
        mach=0.6,
        pressure_ratio=1.0,
        static_temperature_k=288.15,
    )

    assert np.isclose(
        model.n1_for_thrust_scalar(thrust_n=target, **condition), 73.0, atol=1e-5
    )
    assert model.n1_for_thrust_scalar(thrust_n=0.0, **condition) == 20.0
    assert model.n1_for_thrust_scalar(thrust_n=1e9, **condition) == 100.0


def test_openap_fuel_prior_matches_pinned_a320_reference():
    prior = OpenAPFuelFlowPrior(
        engine_count=2,
        rated_thrust_n_per_engine=117_900.0,
        c1=1.0453208160586924,
        c2=2.3633720747416573,
        c3=1.2378127479131922,
        engine_scale=1.166 / 1.132,
    )

    # OpenAP's A320/CFM56-5B4 reference at 50 kN total thrust.
    assert np.isclose(
        prior.fuel_flow_kg_s(np.array([50_000.0]))[0], 1.030994, rtol=2e-3
    )
    assert np.isclose(
        prior.fuel_flow_scalar(50_000.0),
        prior.fuel_flow_kg_s(np.array([50_000.0]))[0],
    )


def test_calibrated_fuel_flow_matches_scalar_and_vector_paths():
    prior = OpenAPFuelFlowPrior(2, 117_900.0, 1.0453, 2.3634, 1.2378)
    calibrated = CalibratedFuelFlow(prior, scale=0.9, offset_kg_s=-0.02)

    vector = calibrated.fuel_flow_kg_s(np.array([0.0, 50_000.0]))

    np.testing.assert_allclose(
        vector,
        [calibrated.fuel_flow_scalar(0.0), calibrated.fuel_flow_scalar(50_000.0)],
    )
    assert np.all(vector >= 0.0)
