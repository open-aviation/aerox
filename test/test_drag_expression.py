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
from aerox.models.drag.expression import isa_reynolds_per_m


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


def test_isa_reynolds_number_matches_aero():
    mach = np.array([0.0, 0.3, 0.6, 0.85])[:, None]
    h = np.array([0.0, 5_000.0, 11_000.0, 13_000.0])[None, :]
    for dT in (0.0, 10.0, -30.0, np.array([[5.0], [20.0], [-10.0], [0.0]])):
        np.testing.assert_allclose(
            isa_reynolds_per_m(mach, h, dT),
            aero.reynolds_per_m(mach, h, dT),
            rtol=1e-12,
        )


def test_expression_takes_altitude_instead_of_reynolds_number():
    model = build_drag_model(load_aircraft("A320"))
    expression = model.expression()
    cl, mach = np.array([0.3, 0.5, 0.7]), np.array([0.4, 0.7, 0.8])
    h, dT = np.array([2_000.0, 9_000.0, 12_000.0]), 10.0
    np.testing.assert_allclose(
        expression.cd(cl, mach, h=h, dT=dT),
        model.cd(cl, mach, h=h, dT=dT),
        rtol=1e-12,
    )
    with pytest.raises(TypeError, match="not both"):
        expression.cd(0.5, 0.78, 6e6, h=11_000)
    with pytest.raises(TypeError, match="reynolds_per_m or h"):
        expression.cd(0.5, 0.78)


def test_expression_accepts_the_casadi_module():
    casadi = pytest.importorskip("casadi")
    model = build_drag_model(load_aircraft("A320"))
    cl, mach, h = casadi.SX.sym("cl"), casadi.SX.sym("mach"), casadi.SX.sym("h")
    flap = casadi.SX.sym("flap")
    cd = model.expression().cd(
        cl, mach, h=h, xp=casadi, flap_angle_rad=flap, gear_extension_fraction=1.0
    )
    function = casadi.Function("cd", [cl, mach, h, flap], [cd])
    expected = model.expression().cd(
        0.8, 0.3, h=500.0, flap_angle_rad=0.3, gear_extension_fraction=1.0
    )
    assert float(function(0.8, 0.3, 500.0, 0.3)) == pytest.approx(expected, rel=1e-12)


def test_expression_runs_in_jax_jit_and_grad():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    expression = build_drag_model(load_aircraft("A320")).expression()

    @jax.jit
    def cd(cl, mach, h):
        return expression.cd(cl, mach, h=h, xp=jnp)

    expected = expression.cd(0.5, 0.78, h=11_000.0)
    assert float(cd(0.5, 0.78, 11_000.0)) == pytest.approx(expected, rel=1e-12)
    slope = float(jax.grad(cd)(0.5, 0.78, 11_000.0))
    step = 1e-6
    numeric = (
        expression.cd(0.5 + step, 0.78, h=11_000.0)
        - expression.cd(0.5 - step, 0.78, h=11_000.0)
    ) / (2 * step)
    assert slope == pytest.approx(numeric, rel=1e-6)


def test_expression_is_exported():
    assert isinstance(
        build_drag_model(load_aircraft("B738")).expression(), DragExpression
    )
