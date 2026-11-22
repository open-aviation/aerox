import numpy as np
import pytest

import aerox.models.drag as drag
from aerox import aero, load_aircraft
from aerox.models._atmosphere import unit_reynolds_number
from aerox.models.drag import (
    DragCondition,
    DragModel,
    ParabolicDragPolar,
    build_drag_model,
    build_reference_drag_polar,
)


def test_aero_reynolds_per_m_matches_density_speed_and_viscosity():
    h = np.array([8_000.0, 11_000.0])
    mach = np.array([0.6, 0.78])
    _, density, temperature = aero.atmos(h)
    expected = unit_reynolds_number(density, aero.mach2tas(mach, h), temperature)
    np.testing.assert_allclose(aero.reynolds_per_m(mach, h), expected)
    assert aero.viscosity(0.0) == pytest.approx(1.789e-5, rel=1e-3)


def test_drag_model_accepts_arguments_or_condition():
    model = build_drag_model(load_aircraft("A320"))
    assert isinstance(model, DragModel)
    reynolds = aero.reynolds_per_m(0.78, 11_000)
    by_arguments = model.drag_coefficient(0.5, 0.78, reynolds)
    by_condition = model.drag_coefficient(0.5, condition=DragCondition(0.78, reynolds))
    assert by_arguments == pytest.approx(by_condition)
    assert model._drag_coefficient_scalar(
        0.5, DragCondition(0.78, reynolds)
    ) == pytest.approx(by_arguments)
    with pytest.raises(TypeError):
        model.drag_coefficient(0.5, 0.78, condition=DragCondition(0.78))
    with pytest.raises(TypeError):
        model.drag_coefficient(0.5)


def test_missing_reynolds_number_uses_the_model_reference():
    aircraft = load_aircraft("A320")
    model = build_drag_model(aircraft)
    reference = aircraft.drag.reference_reynolds_per_m
    assert model.drag_coefficient(0.5, 0.7) == pytest.approx(
        model.drag_coefficient(0.5, 0.7, reference)
    )
    assert model._drag_coefficient_scalar(0.5, DragCondition(0.7)) == pytest.approx(
        model.drag_coefficient(0.5, 0.7, reference)
    )


def test_reference_polar_uses_cd0_and_k():
    polar = build_reference_drag_polar(load_aircraft("A320"))
    assert polar.drag_coefficient(0.5) == pytest.approx(polar.cd0 + polar.k * 0.25)
    assert ParabolicDragPolar(cd0=0.02, k=0.04).drag_coefficient(1.0) == 0.06


def test_drag_package_exports_only_the_public_interface():
    assert set(drag.__all__) == {
        "DragBuildReport",
        "DragCondition",
        "DragModel",
        "GeometryDragModel",
        "ParabolicDragPolar",
        "TransonicGeometryDragModel",
        "build_drag_model",
        "build_drag_report",
        "build_reference_drag_polar",
    }
