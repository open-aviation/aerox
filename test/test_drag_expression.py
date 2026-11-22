import math

import numpy as np
import pytest

from aerox import aero, get_available_aircraft, load_aircraft
from aerox.models.drag import (
    DragExpression,
    GeometryDragModel,
    TransonicGeometryDragModel,
    build_drag_model,
)


@pytest.mark.parametrize("code", get_available_aircraft())
def test_expression_matches_the_drag_model(code):
    model = build_drag_model(load_aircraft(code))
    expression = model.expression()
    cl = np.linspace(0.0, 1.0, 6)[:, None]
    mach = np.linspace(0.2, 0.85, 8)[None, :]
    reynolds = aero.reynolds_per_m(mach, 9_000)
    expected = model.drag_coefficient(cl, mach, reynolds)
    np.testing.assert_allclose(expression.cd(cl, mach, reynolds), expected, rtol=1e-12)
    assert expression.has_wave_drag == isinstance(model, TransonicGeometryDragModel)


def test_expression_critical_mach_matches_the_wave_model():
    model = build_drag_model(load_aircraft("A320"))
    assert isinstance(model, TransonicGeometryDragModel)
    cl = np.array([0.0, 0.4, 0.8])
    np.testing.assert_allclose(
        model.expression().critical_mach(cl), model.critical_mach(cl), rtol=1e-12
    )


class _ScalarMath:
    """A minimal math backend on Python floats, standing in for CasADi or JAX."""

    log = staticmethod(math.log)
    sqrt = staticmethod(math.sqrt)
    maximum = staticmethod(max)


def test_expression_evaluates_with_another_math_backend():
    model = build_drag_model(load_aircraft("A320"))
    reynolds = float(aero.reynolds_per_m(0.78, 11_000))
    value = model.expression().cd(0.5, 0.78, reynolds, xp=_ScalarMath)
    assert isinstance(value, float)
    assert value == pytest.approx(float(model.cd(0.5, 0.78, reynolds)), rel=1e-12)


def test_expression_without_wave_drag():
    model = build_drag_model(load_aircraft("C172"))
    assert isinstance(model, GeometryDragModel)
    expression = model.expression()
    assert not expression.has_wave_drag
    assert expression.wave(0.5, 0.2) == 0.0
    with pytest.raises(ValueError, match="no wave drag"):
        expression.critical_mach(0.5)


def test_expression_is_exported():
    assert isinstance(
        build_drag_model(load_aircraft("B738")).expression(), DragExpression
    )
