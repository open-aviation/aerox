import numpy as np
import pytest

from aerox.fms import ModeConditionedSpeedIntent, speed_intent_design, vertical_phase


def test_vertical_phase_uses_declared_deadband() -> None:
    phase = vertical_phase(np.array([251.0, 250.0, -250.0, -251.0]))
    assert phase.tolist() == ["CLIMB", "LEVEL", "LEVEL", "DESCENT"]


def test_phase_constant_model_switches_at_its_crossover() -> None:
    model = ModeConditionedSpeedIntent(
        structure="phase_constant",
        cas_coefficients=np.array([290.0, -10.0, -15.0]),
        mach_coefficients=np.array([0.78, 0.0, -0.02]),
    )
    prediction = model.predict(
        selected_cas_kts=np.full(4, 300.0),
        selected_mach=np.full(4, 0.80),
        pressure_altitude_ft=np.array([5_000.0, 20_000.0, 30_000.0, 37_000.0]),
        mass_kg=np.full(4, 60_000.0),
        phase=np.array(["CLIMB", "LEVEL", "LEVEL", "DESCENT"]),
    )
    np.testing.assert_allclose(prediction.cas_target_kts, [290.0, 280.0, 280.0, 275.0])
    np.testing.assert_allclose(prediction.mach_target, [0.78, 0.78, 0.78, 0.76])
    assert prediction.active_kind[0] == "CAS"
    assert prediction.active_kind[-1] == "MACH"
    assert np.all(prediction.crossover_altitude_ft >= 25_000.0)
    assert np.all(prediction.crossover_altitude_ft <= 35_000.0)
    assert np.all(np.isfinite(prediction.active_cas_equivalent_kts))


def test_selector_model_clips_implausible_commands() -> None:
    model = ModeConditionedSpeedIntent(
        structure="selector",
        cas_coefficients=np.empty(0),
        mach_coefficients=np.empty(0),
    )
    prediction = model.predict(
        selected_cas_kts=np.array([100.0, 400.0]),
        selected_mach=np.array([0.1, 0.95]),
        pressure_altitude_ft=np.array([0.0, 40_000.0]),
        mass_kg=np.array([60_000.0, 60_000.0]),
        phase=np.array(["CLIMB", "LEVEL"]),
    )
    np.testing.assert_allclose(prediction.cas_target_kts, [180.0, 350.0])
    np.testing.assert_allclose(prediction.mach_target, [0.50, 0.86])


def test_speed_intent_design_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        speed_intent_design(
            selector=np.array([300.0]),
            pressure_altitude_ft=np.array([10_000.0]),
            mass_kg=np.array([60_000.0]),
            phase=np.array(["UNKNOWN"]),
            speed_kind="CAS",
            feature_set="affine_selector",
        )
