"""Turbofan component drag build-up and its nacelle data."""

from __future__ import annotations

import numpy as np
import pytest

from aerox import get_available_aircraft, load_aircraft
from aerox.models.drag import (
    ROUGHNESS_EXCRESCENCE_FRACTION,
    DragCondition,
    build_drag_model,
    build_drag_report,
    build_reference_drag_polar,
    control_surface_gap_drag_area,
    exposed_wing_area,
    fuselage_upsweep_drag_area,
    nacelle_base_drag_area,
    nacelle_form_factor,
    nacelle_interference_factor,
    nacelle_wetted_area,
    resolve_drag_geometry,
    resolve_nacelle_geometry,
    trapezoidal_root_chord,
    uses_component_buildup,
)

TURBOFAN_TERMS = (
    "wing",
    "fuselage",
    "horizontal_tail",
    "vertical_tail",
    "nacelles",
    "pylons",
    "fuselage_upsweep",
    "control_surface_gaps",
    "nacelle_base",
    "roughness_excrescence",
)


def _cf(reynolds: float, mach: float) -> float:
    return 0.455 / (np.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach**2) ** 0.65)


def test_exposed_wing_area_removes_trapezoid_inside_fuselage():
    # S = 100, b = 20, c_r = 7: taper = 2 S / (b c_r) - 1 = 3/7.
    # Buried = c_r w (1 - (1 - taper) w / (2 b)) = 28 (1 - (4/7) / 10).
    assert exposed_wing_area(100.0, 20.0, 4.0, 7.0) == pytest.approx(
        100.0 - 28.0 * (1.0 - 0.4 / 7.0)
    )
    with pytest.raises(ValueError, match="inconsistent"):
        exposed_wing_area(100.0, 20.0, 4.0, 20.0)


def test_trapezoidal_root_chord_recovers_planform_from_mac():
    # Trapezoid with c_r = 6, c_t = 1.5, b = 30: S = 112.5 and MAC = 4.2.
    assert trapezoidal_root_chord(112.5, 30.0, 4.2) == pytest.approx(6.0)
    # A rectangular wing has MAC equal to the mean chord.
    assert trapezoidal_root_chord(100.0, 20.0, 5.0) == pytest.approx(5.0)


def test_nacelle_and_small_term_formulas_match_hand_values():
    assert nacelle_wetted_area(2.0, 4.0) == pytest.approx(22.4)
    assert nacelle_form_factor(2.0, 4.0) == pytest.approx(1.175)
    assert fuselage_upsweep_drag_area(4.0, 4.0) == pytest.approx(0.006 * 4.0 * np.pi)
    assert nacelle_base_drag_area(2.0) == pytest.approx(0.0127 * np.pi * 2.0 * 0.2)
    gaps = control_surface_gap_drag_area(((30.0, np.deg2rad(60.0)), (10.0, 0.0)))
    assert gaps == pytest.approx(0.0002 * (30.0 * 0.25 + 10.0))


def test_nacelle_interference_factor_follows_engine_mount():
    assert nacelle_interference_factor("wing", 2) == pytest.approx(1.3)
    assert nacelle_interference_factor("underwing mounted", 4) == pytest.approx(1.3)
    assert nacelle_interference_factor("rear fuselage", 2) == pytest.approx(1.5)
    assert nacelle_interference_factor("rear", 2) == pytest.approx(1.5)
    assert nacelle_interference_factor(
        "underwing mounted and fin-integrated", 3
    ) == pytest.approx((2 * 1.3 + 1.5) / 3)
    assert nacelle_interference_factor(
        "both sides of rear fuselage and fin-integrated", 3
    ) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="unsupported engine mount"):
        nacelle_interference_factor(None, 2)


def test_nacelle_lookup_uses_measured_ratio_and_estimated_tiers():
    measured = resolve_nacelle_geometry(load_aircraft("A320"))
    assert measured.tier == "A-measured-nacelle"
    assert (measured.max_diameter_m, measured.length_m) == (2.10, 3.43)
    assert measured.imputed_fields == ()

    ratio = resolve_nacelle_geometry(load_aircraft("B738"))
    assert ratio.tier == "B-fan-diameter-ratio"
    assert ratio.fan_diameter_m == pytest.approx(1.549)
    assert ratio.max_diameter_m == pytest.approx(1.25 * 1.549)
    assert ratio.length_m == pytest.approx(1.95 * 1.549)

    long_duct = resolve_nacelle_geometry(load_aircraft("B752"))
    assert long_duct.duct == "long"
    assert long_duct.length_m == pytest.approx(3.1 * long_duct.fan_diameter_m)

    aircraft = load_aircraft("A388")
    estimated = resolve_nacelle_geometry(aircraft)
    assert estimated.tier == "C-estimated-fan-diameter"
    assert estimated.fan_diameter_m == pytest.approx(
        0.00827 * np.sqrt(aircraft.mtow / aircraft.engine.number)
    )
    assert "engine.fan_diameter" in estimated.imputed_fields


def test_every_component_buildup_record_resolves_nacelle_geometry():
    tiers = {}
    for code in get_available_aircraft():
        aircraft = load_aircraft(code)
        if aircraft.engine.type != "turbofan":
            assert not uses_component_buildup(aircraft), code
            continue
        if not uses_component_buildup(aircraft):
            assert code in {"F15", "F16", "HAWK", "T38"}
            continue
        geometry = resolve_drag_geometry(aircraft)
        assert geometry.nacelle is not None, code
        assert geometry.nacelle.max_diameter_m > geometry.nacelle.fan_diameter_m
        tiers[geometry.nacelle.tier] = tiers.get(geometry.nacelle.tier, 0) + 1

    assert tiers == {
        "A-measured-nacelle": 12,
        "B-fan-diameter-ratio": 42,
        "C-estimated-fan-diameter": 38,
    }


def test_a320_buildup_terms_match_hand_computation():
    aircraft = load_aircraft("A320")
    report = build_drag_report(aircraft)
    terms = {term.name: term.value for term in report.parasite_terms}
    mach = report.reference_mach
    unit_reynolds = report.reference_unit_reynolds_number_per_m
    area = 122.0

    assert tuple(terms) == TURBOFAN_TERMS
    nacelle_q1 = (
        2
        * _cf(unit_reynolds * 3.43, mach)
        * (1.0 + 0.35 * 2.10 / 3.43)
        * 2.8
        * 2.10
        * 3.43
        / area
    )
    assert terms["nacelles"] == pytest.approx(1.3 * nacelle_q1)
    assert terms["pylons"] == pytest.approx(0.2 * nacelle_q1)
    assert terms["fuselage_upsweep"] == pytest.approx(
        0.006 * np.pi / 4.0 * 3.95 * 4.14 / area
    )
    gaps = 0.0002 * (
        np.cos(np.deg2rad(25.0)) ** 2 * 0.3 * area
        + np.cos(np.deg2rad(27.945112)) ** 2 * 30.843428
        + np.cos(np.deg2rad(35.004001)) ** 2 * 0.9 * 21.514182
    )
    assert terms["control_surface_gaps"] == pytest.approx(gaps / area)
    assert terms["nacelle_base"] == pytest.approx(
        2 * 0.0127 * np.pi * 1.735 * 0.2 / area
    )
    horizontal_q1 = (
        _cf(unit_reynolds * 2.687454, mach)
        * (1.0 + 2.0 * 0.10 + 60.0 * 0.10**4)
        * 2.0
        * 30.843428
        * 1.025
        / area
    )
    assert terms["horizontal_tail"] == pytest.approx(1.04 * horizontal_q1)

    # Exposed wing: trapezoid with S, b, and MAC of the record.
    taper = np.linspace(0.0, 1.0, 200_001)
    identity = 4.0 / 3.0 * (1 + taper + taper**2) / (1 + taper) ** 2
    taper = taper[np.argmin(np.abs(identity - 4.1935 * 34.1 / area))]
    root = 2.0 * area / (34.1 * (1.0 + taper))
    exposed = area - root * 3.95 * (1.0 - (1.0 - taper) * 3.95 / (2.0 * 34.1))
    wing = (
        _cf(unit_reynolds * 4.1935, mach)
        * (1.0 + 2.0 * 0.12 + 60.0 * 0.12**4)
        * 2.0
        * exposed
        * 1.03
        / area
    )
    assert terms["wing"] == pytest.approx(wing, rel=1e-4)

    computed = sum(
        value for name, value in terms.items() if name != "roughness_excrescence"
    )
    assert terms["roughness_excrescence"] == pytest.approx(
        ROUGHNESS_EXCRESCENCE_FRACTION * computed
    )


def test_report_terms_sum_to_cd0_for_every_record():
    for code in get_available_aircraft():
        aircraft = load_aircraft(code)
        report = build_drag_report(aircraft)
        total = sum(term.value for term in report.parasite_terms)
        assert total == pytest.approx(report.cd0, rel=1e-12), code
        assert all(term.method for term in report.parasite_terms), code
        names = tuple(term.name for term in report.parasite_terms)
        if uses_component_buildup(aircraft):
            assert names == TURBOFAN_TERMS, code
            assert report.roughness_excrescence_fraction == pytest.approx(0.09)
            assert report.unresolved_fraction_of_resolved_drag is None
        else:
            assert names[-1] == "unresolved", code
            assert "nacelles" not in names, code
            assert report.roughness_excrescence_fraction is None


@pytest.mark.parametrize(
    ("code", "cd0", "k"),
    [
        # Values of the lumped model before the turbofan build-up was added.
        ("C172", 0.024368311377929015, 0.05393909107716087),
        ("AT72", 0.021866690727840428, 0.036155148216331645),
        ("F16", 0.01605237299344407, 0.13209398706035375),
        ("T38", 0.025107994109102232, 0.11151857728705869),
    ],
)
def test_non_turbofan_and_high_speed_classes_keep_lumped_model(code, cd0, k):
    report = build_drag_report(load_aircraft(code))

    assert report.cd0 == pytest.approx(cd0, rel=1e-12)
    assert report.k == pytest.approx(k, rel=1e-12)
    assert report.nacelle_geometry_tier is None


def test_every_record_builds_and_runtime_model_reproduces_report():
    codes = get_available_aircraft()
    assert len(codes) == 144
    for code in codes:
        aircraft = load_aircraft(code)
        report = build_drag_report(aircraft)
        model = build_drag_model(aircraft)
        polar = build_reference_drag_polar(aircraft)
        condition = DragCondition(
            aircraft.drag.reference_mach,
            aircraft.drag.reference_unit_reynolds_number_per_m,
        )
        assert polar.zero_lift_drag_coefficient == pytest.approx(
            report.cd0, rel=1e-5
        ), code
        assert polar.induced_drag_factor == pytest.approx(report.k, rel=1e-5), code
        assert np.isfinite(model.drag_coefficient_scalar_at(0.3, condition)), code


def test_runtime_model_uses_stored_roughness_allowance():
    aircraft = load_aircraft("A320")
    base = build_reference_drag_polar(aircraft).zero_lift_drag_coefficient
    aircraft.drag.roughness_excrescence_fraction = 0.13
    raised = build_reference_drag_polar(aircraft).zero_lift_drag_coefficient

    assert raised / base == pytest.approx(1.13 / 1.09)
    aircraft.drag.roughness_excrescence_fraction = None
    with pytest.raises(ValueError, match="roughness_excrescence_fraction"):
        build_reference_drag_polar(aircraft)
