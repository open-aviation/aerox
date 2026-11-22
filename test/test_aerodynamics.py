import numpy as np
import pytest

from aerox.models import (
    DragCondition,
    DragModel,
    EffectiveLiftMap,
    ParabolicDragPolar,
    effective_lift_design,
)
from aerox.models.drag.polar import (
    ParabolicWaveDragModel,
)
from aerox.models.drag.reference import (
    HyperbolicTangentWaveDrag,
    MachDependentInducedDrag,
    PollSchumannDragModel,
    TransonicCompressibilityDragModel,
)
from aerox.models.drag.wave import (
    KornLockWaveDrag,
)


def test_effective_lift_design_is_centered_at_four_degrees_and_mach_point_six():
    design = effective_lift_design(
        np.deg2rad(np.array([4.0])),
        np.array([0.6]),
        feature_set="linear_alpha_mach",
    )

    np.testing.assert_allclose(design, [[1.0, 0.0, 0.0, 0.0]])


def test_lift_map_and_drag_polar_apply_physical_forms():
    lift = EffectiveLiftMap(
        feature_set="linear_alpha",
        coefficients=np.array([0.5, 0.1]),
    )
    polar = ParabolicDragPolar(0.02, 0.04)

    coefficient = lift.lift_coefficient(
        np.deg2rad(np.array([4.0, 5.0])), np.array([0.6, 0.6])
    )

    np.testing.assert_allclose(coefficient, [0.5, 0.6])
    assert np.isclose(
        lift.lift_coefficient_scalar(np.deg2rad(5.0), 0.6), coefficient[1]
    )
    np.testing.assert_allclose(
        polar.drag_coefficient(coefficient), 0.02 + 0.04 * coefficient**2
    )
    broadcast = polar.drag_coefficient(coefficient[:, None], np.array([0.6, 0.7]))
    assert broadcast.shape == (2, 2)
    np.testing.assert_allclose(
        broadcast,
        np.broadcast_to(0.02 + 0.04 * coefficient[:, None] ** 2, (2, 2)),
    )
    assert polar._drag_coefficient_scalar(0.6, DragCondition(0.8)) == pytest.approx(
        0.02 + 0.04 * 0.6**2
    )


def test_transonic_drag_model_returns_vectorized_component_breakdown() -> None:
    polar = ParabolicDragPolar(0.02263, 0.04246)
    wave = KornLockWaveDrag(np.deg2rad(25.0), 0.12)
    model = ParabolicWaveDragModel(polar, wave)
    lift = np.array([0.45, 0.55, 0.65])
    mach = np.array([0.70, 0.78, 0.82])

    result = model.breakdown(lift, mach)

    np.testing.assert_allclose(result.parasite, 0.02263)
    np.testing.assert_allclose(result.induced, 0.04246 * lift**2)
    np.testing.assert_allclose(
        result.total, result.parasite + result.induced + result.wave
    )
    assert result.wave[0] == 0.0
    assert result.wave[-1] > result.wave[1] > 0.0
    np.testing.assert_allclose(model.drag_coefficient(lift, mach), result.total)
    for index in range(len(lift)):
        assert model._drag_coefficient_scalar(
            lift[index], DragCondition(mach[index])
        ) == pytest.approx(result.total[index])


def test_mach_dependent_tanh_drag_model_matches_published_equation() -> None:
    induced = MachDependentInducedDrag(
        reference_induced_drag_factor=0.029856,
        compressibility_strength=0.121,
        compressibility_exponent=1.632,
    )
    wave = HyperbolicTangentWaveDrag(
        lift_coefficient_grid=np.array([0.3, 0.4, 0.5]),
        critical_mach_grid=np.array([0.691745, 0.693807, 0.686675]),
        amplitude=0.048626,
        mach_scale=18.637,
        shift=23.637,
    )
    model = TransonicCompressibilityDragModel(0.020053, induced, wave)
    lift = np.array([0.3, 0.4, 0.5])
    mach = np.array([0.68, 0.75, 0.80])

    result = model.breakdown(lift, mach)
    expected_induced = 0.029856 * lift**2 / (1.0 - 0.121 * (mach / 0.3 - 1.0) ** 1.632)
    critical = np.array([0.691745, 0.693807, 0.686675])
    expected_wave = np.where(
        mach >= critical,
        0.048626 * (1.0 + np.tanh(18.637 * mach / critical - 23.637)),
        0.0,
    )

    np.testing.assert_allclose(result.induced, expected_induced)
    np.testing.assert_allclose(result.wave, expected_wave)
    np.testing.assert_allclose(
        result.total, 0.020053 + expected_induced + expected_wave
    )
    assert model._drag_coefficient_scalar(0.4, DragCondition(0.75)) == pytest.approx(
        result.total[1]
    )
    with pytest.raises(ValueError, match="outside wave-drag support"):
        model._drag_coefficient_scalar(0.51, DragCondition(0.78))
    with pytest.raises(ValueError, match="below the compressibility-model support"):
        induced.drag_coefficient_scalar(0.4, 0.2)


def test_poll_schumann_a320_model_reproduces_published_equations() -> None:
    model = PollSchumannDragModel(
        reference_area_m2=122.4,
        wing_span_m=34.1,
        fuselage_width_m=3.95,
        wing_sweep_rad=np.deg2rad(25.0),
        skin_friction_multiplier=8.40,
        technology_factor=0.750,
        design_mach=0.753,
        design_lift_coefficient=0.590,
        wave_creep_coefficient=0.073,
        wave_creep_threshold=0.869,
        reference_reynolds_number=5.59e7,
    )
    lift = np.array([0.3, 0.5, 0.6])
    mach = np.array([0.60, 0.78, 0.82])

    result = model.breakdown(lift, mach)

    assert model.aspect_ratio == pytest.approx(9.5000816993)
    np.testing.assert_allclose(result.parasite, 0.018594902818)
    np.testing.assert_allclose(
        result.induced / lift**2,
        0.043140733931,
    )
    np.testing.assert_allclose(
        result.wave * 1.0e4,
        [0.0, 13.40737657, 49.33454657],
        rtol=1.0e-7,
    )
    assert model._drag_coefficient_scalar(0.5, DragCondition(0.78)) == pytest.approx(
        result.total[1]
    )
    assert isinstance(model, DragModel)
    with pytest.raises(ValueError, match="lift coefficient"):
        model._drag_coefficient_scalar(0.2, DragCondition(0.78))
    with pytest.raises(ValueError, match="Mach"):
        model._drag_coefficient_scalar(0.5, DragCondition(0.83))


def test_condition_interface_converts_poll_unit_reynolds_number():
    polar = ParabolicDragPolar(0.02, 0.04)
    condition = DragCondition(
        mach=np.array([0.7, 0.8]),
        reynolds_per_m=np.array([4.0e6, 5.0e6]),
    )
    np.testing.assert_allclose(
        polar.drag_coefficient(np.array([0.4, 0.5]), condition=condition),
        polar.drag_coefficient(np.array([0.4, 0.5]), condition.mach),
    )

    poll = PollSchumannDragModel(
        reference_area_m2=122.4,
        wing_span_m=34.1,
        fuselage_width_m=3.95,
        wing_sweep_rad=np.deg2rad(25.0),
        skin_friction_multiplier=8.40,
        technology_factor=0.750,
        design_mach=0.753,
        design_lift_coefficient=0.590,
        wave_creep_coefficient=0.073,
        wave_creep_threshold=0.869,
        reference_reynolds_number=5.59e7,
    )
    lift = np.array([0.4, 0.5])
    np.testing.assert_allclose(
        poll.drag_coefficient(lift, condition=condition),
        poll.cd0(np.asarray(condition.reynolds_per_m) * np.sqrt(122.4))
        + poll.k(np.asarray(condition.reynolds_per_m) * np.sqrt(122.4)) * lift**2
        + poll.wave_drag_coefficient(lift, condition.mach),
    )
