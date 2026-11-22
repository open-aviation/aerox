import numpy as np
import pytest

from aerox import load_aircraft
from aerox.models.drag import (
    DragCondition,
    DragModel,
    TransonicGeometryDragModel,
)
from aerox.models.drag.geometry import (
    geometry_drag_model_from_aircraft,
    geometry_drag_model_from_dimensions,
)
from aerox.models.drag.wave import (
    KornLockWaveDrag,
)


def test_a320_geometry_model_has_explicit_components_and_plausible_polar():
    model = geometry_drag_model_from_aircraft(load_aircraft("A320"))
    result = model.breakdown(np.array([0.0, 0.5]), np.array([0.78, 0.78]))

    assert tuple(result.component_parasite) == (
        "wing",
        "fuselage",
        "horizontal_tail",
        "vertical_tail",
    )
    assert result.parasite[0] == pytest.approx(0.01776675, rel=1e-5)
    assert tuple(result.parasite_increments) == ("unresolved",)
    assert result.increment_parasite[0] > 0.0
    np.testing.assert_allclose(
        result.parasite,
        result.resolved_parasite + result.parasite_increments["unresolved"],
    )
    assert result.span_induced[0] == 0.0
    assert result.lift_dependent_profile[1] > 0.0
    effective_k = (result.total[1] - result.total[0]) / 0.5**2
    assert effective_k == pytest.approx(0.0425403, rel=1e-5)
    assert isinstance(model, DragModel)


def test_geometry_model_uses_component_reynolds_number_and_rejects_configuration():
    model = geometry_drag_model_from_aircraft(load_aircraft("A320"))
    low_reynolds = DragCondition(0.7, 3.0e6)
    high_reynolds = DragCondition(0.7, 8.0e6)

    assert model._drag_coefficient_scalar(0.4, low_reynolds) > (
        model._drag_coefficient_scalar(0.4, high_reynolds)
    )
    with pytest.raises(ValueError, match="only when clean"):
        model._drag_coefficient_scalar(
            0.4,
            DragCondition(0.7, 5.0e6, gear_extension_fraction=1.0),
        )


def test_geometry_transonic_model_has_canonical_continuous_wave_increment():
    base_model = geometry_drag_model_from_aircraft(load_aircraft("A320"))
    wave_drag = KornLockWaveDrag(np.deg2rad(25.0), 0.12)
    model = TransonicGeometryDragModel(
        base_model=base_model,
        wave_drag=wave_drag,
    )
    lift = np.array([0.5, 0.5, 0.5])
    critical = float(model.critical_mach(0.5))
    mach = np.array([critical - 1e-6, critical, critical + 0.05])
    result = model.breakdown(lift, mach)

    np.testing.assert_allclose(result.wave[:2], 0.0)
    assert result.wave[2] > 0.0
    np.testing.assert_allclose(
        result.wave,
        wave_drag.drag_coefficient_increment(mach, lift),
    )
    np.testing.assert_allclose(
        result.total,
        result.parasite
        + result.span_induced
        + result.lift_dependent_profile
        + result.wave,
    )


def test_dimension_builder_is_the_single_geometry_construction_path():
    aircraft = load_aircraft("A320")
    direct = geometry_drag_model_from_aircraft(aircraft)
    dimensions = geometry_drag_model_from_dimensions(
        wing_area_m2=aircraft.wing.area,
        wing_span_m=aircraft.wing.span,
        wing_mean_aerodynamic_chord_m=aircraft.wing.mac,
        wing_thickness_to_chord_ratio=aircraft.wing.t_c,
        wing_sweep_rad=np.deg2rad(aircraft.wing.sweep),
        fuselage_length_m=aircraft.fuselage.length,
        fuselage_width_m=aircraft.fuselage.width,
        fuselage_height_m=aircraft.fuselage.height,
        horizontal_tail_area_m2=aircraft.horizontal_tail.area,
        horizontal_tail_mean_aerodynamic_chord_m=aircraft.horizontal_tail.mac,
        vertical_tail_area_m2=aircraft.vertical_tail.area,
        vertical_tail_mean_aerodynamic_chord_m=aircraft.vertical_tail.mac,
        reference_mach=aircraft.kinematic.cruise_mach,
        reference_reynolds_per_m=5.05e6,
        span_efficiency_factor=0.95,
    )
    condition = DragCondition(0.78, 5.05e6)

    assert dimensions._drag_coefficient_scalar(0.5, condition) == pytest.approx(
        direct._drag_coefficient_scalar(0.5, condition)
    )
