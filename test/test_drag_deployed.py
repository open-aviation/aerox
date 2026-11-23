"""Drag of deployed flaps and landing gear."""

import numpy as np
import pytest

from aerox import aero, get_available_aircraft, load_aircraft
from aerox.models.drag import DragCondition, build_drag_model
from aerox.models.drag.deployed import G0, GEAR_COEFFICIENT


def _base(model):
    return getattr(model, "base_model", model)


def _condition(flap_deg=0.0, gear=0.0, mach=0.2):
    reynolds = aero.reynolds_per_m(mach, 0.0)
    return DragCondition(
        mach,
        reynolds,
        flap_angle_rad=np.deg2rad(flap_deg),
        gear_extension_fraction=gear,
    )


def test_a320_gear_drag_is_close_to_flight_data():
    # Deiler and Fezans (AIAA 2025-1250) identified 0.0299 from flight data.
    gear = _base(build_drag_model(load_aircraft("A320"))).deployed.gear_parasite
    assert gear == pytest.approx(0.0322, abs=5e-4)
    assert gear == pytest.approx(0.0299, rel=0.10)


def test_gear_formula():
    aircraft = load_aircraft("B738")
    base = _base(build_drag_model(aircraft))
    mtow = aircraft.mtow
    expected = mtow * G0 / base.reference_area_m2 * GEAR_COEFFICIENT * mtow**-0.215
    assert base.deployed.gear_parasite == pytest.approx(expected)


def test_a320_flap_drag_uses_record_values():
    # A320 record: lambda_f 0.9, cf/c 0.18, bf/b 0.78
    base = _base(build_drag_model(load_aircraft("A320")))
    assert base.deployed.flap_parasite_factor == pytest.approx(0.9 * 0.18**1.38 * 0.78)


def test_breakdown_adds_flap_and_gear_terms():
    model = build_drag_model(load_aircraft("A320"))
    clean = model.breakdown(1.2, condition=_condition())
    deployed = model.breakdown(1.2, condition=_condition(30.0, 1.0))
    base = _base(model).deployed

    assert float(clean.flap) == 0.0 and float(clean.gear) == 0.0
    assert float(deployed.flap) == pytest.approx(
        base.flap_parasite_factor * np.sin(np.deg2rad(30.0)) ** 2
    )
    assert float(deployed.gear) == pytest.approx(base.gear_parasite)
    assert float(deployed.span_induced) > float(clean.span_induced)
    assert float(deployed.total) == pytest.approx(
        float(
            deployed.parasite
            + deployed.span_induced
            + deployed.lift_dependent_profile
            + deployed.flap
            + deployed.gear
            + deployed.wave
        )
    )


def test_drag_grows_with_flap_angle_and_gear():
    model = build_drag_model(load_aircraft("A320"))
    values = [
        float(model.cd(1.5, condition=_condition(flap, gear)))
        for flap, gear in [(0, 0), (15, 0), (30, 0), (40, 0), (40, 0.5), (40, 1)]
    ]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_half_extended_gear_gives_half_the_increment():
    model = build_drag_model(load_aircraft("A320"))
    clean = float(model.cd(0.8, condition=_condition()))
    half = float(model.cd(0.8, condition=_condition(gear=0.5)))
    full = float(model.cd(0.8, condition=_condition(gear=1.0)))
    assert half - clean == pytest.approx((full - clean) / 2)


def test_scalar_path_matches_array_path():
    model = build_drag_model(load_aircraft("A320"))
    condition = _condition(25.0, 1.0)
    assert model._drag_coefficient_scalar(1.3, condition) == pytest.approx(
        float(model.drag_coefficient(1.3, condition=condition))
    )


@pytest.mark.parametrize("code", get_available_aircraft())
def test_expression_matches_model_with_flaps_and_gear(code):
    model = build_drag_model(load_aircraft(code))
    expression = model.expression()
    cl = np.array([0.6, 1.2, 1.8])
    flap = np.deg2rad(np.array([10.0, 25.0, 40.0]))
    gear = np.array([0.0, 0.5, 1.0])
    mach = 0.25
    reynolds = aero.reynolds_per_m(mach, 500.0)
    condition = DragCondition(mach, reynolds, flap, gear)
    expected = model.drag_coefficient(cl, condition=condition)
    actual = expression.cd(
        cl, mach, reynolds, flap_angle_rad=flap, gear_extension_fraction=gear
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_spoilers_and_large_flap_angles_are_rejected():
    model = build_drag_model(load_aircraft("A320"))
    with pytest.raises(ValueError, match="spoiler"):
        model.cd(0.5, condition=DragCondition(0.3, spoiler_deflection_rad=0.2))
    with pytest.raises(ValueError, match="flap angle"):
        model.cd(0.5, condition=_condition(70.0))
