from __future__ import annotations

import numpy as np
import pytest

from aerox import get_available_aircraft, load_aircraft
from aerox.models.drag import (
    GEOMETRY_DRAG_MODEL_ID,
    KORN_LOCK_WAVE_DRAG_MODEL_ID,
    DragCondition,
    build_drag_model,
    build_drag_report,
    build_reference_drag_polar,
)

def test_all_aircraft_have_one_uniform_finite_active_drag_model():
    codes = get_available_aircraft()

    assert len(codes) == 144
    for code in codes:
        aircraft = load_aircraft(code)
        stored = aircraft.drag
        assert stored is not None
        assert stored.model == GEOMETRY_DRAG_MODEL_ID
        assert stored.configuration == "clean"
        assert stored.geometry_tier
        assert stored.confidence in {"medium", "medium-low", "low"}
        values = np.asarray(
            [
                stored.reference_altitude_ft,
                stored.reference_mach,
                stored.reference_unit_reynolds_number_per_m,
                stored.span_efficiency_factor,
                stored.unresolved_fraction_of_resolved_drag,
            ],
            dtype=float,
        )
        assert np.all(np.isfinite(values)), code


def test_wave_model_is_explicitly_scoped_to_subsonic_civil_turbofans():
    enabled = []
    disabled = []
    for code in get_available_aircraft():
        drag = load_aircraft(code).drag
        if drag.wave_drag_model is None:
            disabled.append(code)
            assert drag.wave_drag_source_tier == "not-applicable"
            assert drag.wave_drag_confidence == "not-applicable"
            assert drag.wave_drag_technology_factor is None
        else:
            enabled.append(code)
            assert drag.wave_drag_model == KORN_LOCK_WAVE_DRAG_MODEL_ID
            assert drag.wave_drag_confidence == "low"
            assert drag.wave_drag_technology_factor == pytest.approx(0.95)

    assert len(enabled) == 90
    assert len(disabled) == 54
    assert {"A320", "B738", "RJ85"} & set(enabled) == {"A320", "B738"}
    assert {"C172", "AT72", "F15", "T38", "C17", "K35R"} <= set(disabled)


def test_stored_parameters_match_deterministic_generator():
    for code in get_available_aircraft():
        aircraft = load_aircraft(code)
        generated = build_drag_report(aircraft)
        stored = aircraft.drag

        assert stored.reference_mach == pytest.approx(
            generated.reference_mach, abs=5.1e-7
        ), code
        assert stored.span_efficiency_factor == pytest.approx(
            generated.span_efficiency_factor, abs=5.1e-5
        ), code


def test_active_model_reconstructs_snapshot_and_varies_with_reynolds_number():
    aircraft = load_aircraft("A320")
    stored = aircraft.drag
    model = build_drag_model(aircraft)
    polar = build_reference_drag_polar(aircraft)
    generated = build_drag_report(aircraft)
    reference = DragCondition(
        stored.reference_mach,
        stored.reference_unit_reynolds_number_per_m,
    )
    low_reynolds = DragCondition(
        stored.reference_mach,
        0.7 * stored.reference_unit_reynolds_number_per_m,
    )

    assert polar.zero_lift_drag_coefficient == pytest.approx(generated.cd0, abs=1e-7)
    assert polar.induced_drag_factor == pytest.approx(generated.k, abs=1e-7)
    assert model.drag_coefficient_scalar_at(0.0, reference) == pytest.approx(
        polar.zero_lift_drag_coefficient
    )
    assert model.drag_coefficient_scalar_at(0.4, low_reynolds) > (
        model.drag_coefficient_scalar_at(0.4, reference)
    )


def test_a320_wave_increment_is_derived_and_reference_polar_stays_subcritical():
    aircraft = load_aircraft("A320")
    model = build_drag_model(aircraft)
    condition = DragCondition(
        0.79,
        aircraft.drag.reference_unit_reynolds_number_per_m,
    )
    breakdown = model.breakdown_at(0.55, condition)
    polar = build_reference_drag_polar(aircraft)
    generated = build_drag_report(aircraft)

    assert model.wave_drag.critical_mach(0.0) == pytest.approx(0.7941158832)
    assert (
        model.wave_drag.critical_mach(0.0) - model.wave_drag.critical_mach(1.0)
    ) == pytest.approx(0.1343299538)
    assert breakdown.wave == pytest.approx(0.0004738001)
    assert breakdown.total == pytest.approx(
        breakdown.parasite
        + breakdown.span_induced
        + breakdown.lift_dependent_profile
        + breakdown.wave
    )
    assert polar.zero_lift_drag_coefficient == pytest.approx(generated.cd0)
    assert polar.induced_drag_factor == pytest.approx(generated.k)


def test_a320_model_uses_observed_geometry_and_poll_span_only_term():
    aircraft = load_aircraft("A320")
    parameters = build_drag_report(aircraft)
    expected_span_efficiency = 1.0 / (
        1.03 + 2.0 * (aircraft.fuselage.width / aircraft.wing.span) ** 2
    )

    assert parameters.geometry_tier == "A-observed-core-and-tails"
    assert parameters.imputed_fields == ()
    assert parameters.span_efficiency_factor == pytest.approx(expected_span_efficiency)
    assert parameters.cd0 == pytest.approx(0.01962633, rel=1e-6)
    assert parameters.k == pytest.approx(0.04345381, rel=1e-6)
    assert parameters.reference_unit_reynolds_number_per_m == pytest.approx(
        5.4436e6, rel=2e-5
    )


def test_scaled_defaults_are_explicit_and_do_not_mutate_source_geometry():
    aircraft = load_aircraft("C172")
    assert aircraft.wing.mac is None
    assert aircraft.fuselage.width is None


def test_reference_condition_never_exceeds_published_mmo():
    for code in get_available_aircraft():
        aircraft = load_aircraft(code)
        parameters = build_drag_report(aircraft)
        if aircraft.mmo is not None:
            assert parameters.reference_mach <= aircraft.mmo, code

    parameters = build_drag_report(aircraft)

    assert parameters.geometry_tier == "D-class-scaled"
    assert parameters.confidence == "low"
    assert {"wing.mac", "wing.t/c", "fuselage.width"} <= set(parameters.imputed_fields)
    assert aircraft.wing.mac is None
    assert aircraft.fuselage.width is None


def test_special_aircraft_do_not_use_civil_transport_span_formulation():
    for code in ("F15", "F16", "HAWK", "T38", "C17", "K35R"):
        parameters = build_drag_report(load_aircraft(code))
        assert parameters.geometry_tier == "X-special-scaled"
        assert parameters.confidence == "low"
