"""Drag of deployed flaps, slats and landing gear."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aerox import Flaps, aero, get_available_aircraft, load_aircraft
from aerox.models.drag import DragCondition, build_drag_model
from aerox.models.drag.deployed import G0, GEAR_COEFFICIENT, deployed_drag_from_aircraft


def _base(model):
    return getattr(model, "base_model", model)


def _condition(flap_deg=0.0, gear=0.0, mach=0.2, slat_deg=0.0):
    reynolds = aero.reynolds_per_m(mach, 0.0)
    return DragCondition(
        mach,
        reynolds,
        flap_angle_rad=np.deg2rad(flap_deg),
        gear_extension_fraction=gear,
        slat_angle_rad=np.deg2rad(slat_deg),
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
    base = _base(build_drag_model(load_aircraft("A320")))
    expected = 0.13 * 0.69 * np.cos(np.deg2rad(25))
    assert base.deployed.flap_parasite(np.deg2rad(40)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kind,optimum,profile",
    [
        ("plain", 60, 0.12),
        ("split", 60, 0.23),
        ("single-slotted", 40, 0.13),
        ("double-slotted", 55, 0.23),
        ("triple-slotted", 55, 0.23),
        ("fowler", 30, 0.10),
        ("double-slotted-fowler", 30, 0.15),
    ],
)
def test_flap_profile_at_published_angle(kind, optimum, profile):
    aircraft = replace(
        load_aircraft("A320"),
        flaps=Flaps(type=kind, flapped_area_ratio=0.5, chord_ratio=0.3),
    )
    deployed = deployed_drag_from_aircraft(aircraft, 100.0, 30.0, 0.12, 0.0, 3.0)
    assert deployed.flap_parasite(np.deg2rad(optimum)) == pytest.approx(0.5 * profile)
    assert deployed.flap_parasite(np.deg2rad(optimum / 2)) == pytest.approx(
        0.5 * profile * 0.37
    )


def test_flapped_area_integrates_the_chord_between_stations():
    aircraft = load_aircraft("A319")
    aircraft.wing.taper_ratio = 0.5
    aircraft.flaps = Flaps(type="single-slotted", eta_inboard=0.2, eta_outboard=0.8)
    deployed = deployed_drag_from_aircraft(aircraft, 100.0, 30.0, 0.12, 0.0, 3.0)
    # The mean chord of this symmetric interval equals the wing mean chord.
    assert deployed.flap_parasite(np.deg2rad(40)) == pytest.approx(0.13 * 0.6)


def test_zero_flapped_area_has_no_flap_effect():
    aircraft = replace(
        load_aircraft("A320"),
        flaps=Flaps(type="single-slotted", flapped_area_ratio=0.0),
    )
    deployed = deployed_drag_from_aircraft(aircraft, 100.0, 30.0, 0.12, 0.0, 3.0)
    assert deployed.flap_parasite(np.deg2rad(40)) == 0
    assert deployed.span_induced(2.0, 0.04, np.deg2rad(40)) == 0.16


def test_span_induced_uses_total_lift_and_gasp_lift_shift():
    aircraft = replace(
        load_aircraft("A320"),
        flaps=Flaps(type="single-slotted", flapped_area_ratio=0.5, chord_ratio=0.3),
    )
    deployed = deployed_drag_from_aircraft(aircraft, 100.0, 30.0, 0.12, 0.0, 3.0)
    # AR=9, t/c=.12, 30% chord, no sweep, and the 40-degree table row.
    delta_lift = 1.18 * 0.95 * 1.10 * 0.5
    expected = 0.04 * (1.5 - 0.58 * delta_lift) ** 2 / 0.855
    assert deployed.span_induced(1.5, 0.04, np.deg2rad(40)) == pytest.approx(expected)


def test_slat_uses_wing_profile_and_leaves_span_induced_drag():
    model = build_drag_model(load_aircraft("A320"))
    clean = model.breakdown(1.2, condition=_condition())
    deployed = model.breakdown(1.2, condition=_condition(slat_deg=20.0))
    expected = (
        clean.component_parasite["wing"]
        * (1 + 1.2**2)
        * 0.17
        * 0.65
        * np.cos(np.deg2rad(25))
    )
    assert deployed.slat == pytest.approx(expected)
    assert deployed.span_induced == clean.span_induced
    assert deployed.total - clean.total == pytest.approx(expected)


def test_breakdown_adds_flap_slat_and_gear_terms():
    model = build_drag_model(load_aircraft("A320"))
    clean = model.breakdown(1.2, condition=_condition())
    deployed = model.breakdown(1.2, condition=_condition(30.0, 1.0, slat_deg=22.0))
    base = _base(model).deployed

    assert clean.flap == clean.slat == clean.gear == 0.0
    assert deployed.flap == pytest.approx(0.65 * 0.13 * 0.69 * np.cos(np.deg2rad(25)))
    assert float(deployed.gear) == pytest.approx(base.gear_parasite)
    assert deployed.span_induced < clean.span_induced
    assert float(deployed.total) == pytest.approx(
        float(
            deployed.parasite
            + deployed.span_induced
            + deployed.lift_dependent_profile
            + deployed.flap
            + deployed.slat
            + deployed.gear
            + deployed.wave
        )
    )


def test_flap_profile_drag_grows_with_angle():
    model = build_drag_model(load_aircraft("A320"))
    flap_deg = np.arange(0, 61)
    condition = _condition(flap_deg)
    assert np.all(np.diff(model.breakdown(1.5, condition=condition).flap) > 0.0)


def test_half_extended_gear_gives_half_the_increment():
    model = build_drag_model(load_aircraft("A320"))
    clean = float(model.cd(0.8, condition=_condition()))
    half = float(model.cd(0.8, condition=_condition(gear=0.5)))
    full = float(model.cd(0.8, condition=_condition(gear=1.0)))
    assert half - clean == pytest.approx((full - clean) / 2)


def test_scalar_path_matches_array_path():
    model = build_drag_model(load_aircraft("A320"))
    condition = _condition(25.0, 1.0, slat_deg=22.0)
    assert model._drag_coefficient_scalar(1.3, condition) == pytest.approx(
        float(model.drag_coefficient(1.3, condition=condition))
    )


@pytest.mark.parametrize("code", get_available_aircraft())
def test_expression_matches_model_with_flaps_slats_and_gear(code):
    model = build_drag_model(load_aircraft(code))
    expression = model.expression()
    cl = np.array([0.6, 1.2, 1.8])
    flap = np.deg2rad(np.array([10.0, 25.0, 40.0]))
    gear = np.array([0.0, 0.5, 1.0])
    mach = 0.25
    reynolds = aero.reynolds_per_m(mach, 500.0)
    slat = np.deg2rad(np.array([0.0, 20.0, 30.0]))
    condition = DragCondition(mach, reynolds, flap, gear, slat_angle_rad=slat)
    expected = model.drag_coefficient(cl, condition=condition)
    actual = expression.cd(
        cl,
        mach,
        reynolds,
        flap_angle_rad=flap,
        gear_extension_fraction=gear,
        slat_angle_rad=slat,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_spoilers_and_large_surface_angles_are_rejected():
    model = build_drag_model(load_aircraft("A320"))
    with pytest.raises(ValueError, match="spoiler"):
        model.cd(0.5, condition=DragCondition(0.3, spoiler_deflection_rad=0.2))
    with pytest.raises(ValueError, match="flap angle"):
        model.cd(0.5, condition=_condition(70.0))
    with pytest.raises(ValueError, match="slat angle"):
        model.cd(0.5, condition=_condition(slat_deg=70.0))


def test_clean_drag_matches_the_baseline_for_every_bundled_aircraft():
    # The baseline holds the clean drag of each record at the conditions
    # below. Rebuild the rows of a record when its geometry data changes.
    baseline = Path(__file__).with_name("data") / "clean_drag_baseline.tsv"
    rows = [line.split("\t") for line in baseline.read_text().splitlines()[1:]]
    assert [code for code, *_ in rows] == sorted(get_available_aircraft())
    for code, *expected in rows:
        aircraft = load_aircraft(code)
        model = build_drag_model(aircraft)
        low_speed_cl = [0.2, 0.6, 1.2, 1.7]
        reference_cl = [0.2, 0.5, 0.8]
        low_speed = _base(model).cd(low_speed_cl, 0.2, h=0)
        reference = model.cd(reference_cl, aircraft.drag.reference_mach)
        np.testing.assert_allclose(
            np.concatenate((low_speed, reference)),
            [float(value) for value in expected],
            rtol=0,
            atol=2e-15,
            err_msg=code,
        )


def test_expression_holds_the_flap_angle_inside_the_tabulated_range():
    model = build_drag_model(load_aircraft("A320"))
    expression = model.expression()

    def cd(flap_deg):
        return expression.cd(
            1.2,
            0.2,
            5e6,
            flap_angle_rad=np.deg2rad(flap_deg),
            gear_extension_fraction=0.4,
        )

    assert cd(65.0) == pytest.approx(cd(60.0))
    assert cd(400.0) == pytest.approx(cd(60.0))
    assert cd(-5.0) == pytest.approx(cd(0.0))
    assert cd(60.0) > 0.0


def test_expression_names_the_missing_backend_function():
    class Backend:
        log, sqrt, exp, sin, maximum = np.log, np.sqrt, np.exp, np.sin, np.maximum

    expression = build_drag_model(load_aircraft("A320")).expression()
    assert expression.cd(0.5, 0.2, 5e6, xp=Backend) > 0.0
    with pytest.raises(AttributeError, match="where"):
        expression.cd(0.5, 0.2, 5e6, xp=Backend, flap_angle_rad=0.3)


def test_deployed_constants_are_validated():
    deployed = _base(build_drag_model(load_aircraft("A320"))).deployed
    with pytest.raises(ValueError, match="non-negative"):
        replace(deployed, gear_parasite=-0.01)
    with pytest.raises(ValueError, match="flap family"):
        replace(deployed, flap_family="blown")


def test_clean_drag_stays_infinite_at_extreme_lift():
    model = build_drag_model(load_aircraft("C172"))
    with np.errstate(over="ignore"):
        assert model.cd(1e155, 0.1, 5e6) == np.inf
