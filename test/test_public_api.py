import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

import aerox.models.drag as drag
from aerox import aero, load_aircraft
from aerox.aircraft import load_aircraft_from_dict, parse_source_notes
from aerox.models._atmosphere import unit_reynolds_number
from aerox.models.drag import (
    DragCondition,
    DragModel,
    ParabolicDragPolar,
    build_drag_model,
    build_parabolic_polar,
)
from aerox.models.drag.polar import ParabolicWaveDragModel
from aerox.models.drag.reference import (
    HyperbolicTangentWaveDrag,
    MachDependentInducedDrag,
    PollSchumannDragModel,
    TransonicCompressibilityDragModel,
)
from aerox.models.drag.wave import KornLockWaveDrag
from tools.migrate_aircraft_data import normalize_bundle, validate_simple


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
    polar = build_parabolic_polar(load_aircraft("A320"))
    assert polar.drag_coefficient(0.5) == pytest.approx(polar.cd0 + polar.k * 0.25)
    assert ParabolicDragPolar(cd0=0.02, k=0.04).drag_coefficient(1.0) == 0.06


def test_drag_package_exports_only_the_public_interface():
    assert set(drag.__all__) == {
        "DeployedDrag",
        "DragBuildReport",
        "DragCondition",
        "DragExpression",
        "DragModel",
        "GeometryDragModel",
        "ParabolicDragPolar",
        "TransonicGeometryDragModel",
        "build_drag_model",
        "build_drag_report",
        "build_parabolic_polar",
    }


DATA_ROOT = Path(__file__).parents[1] / "src" / "aerox" / "data" / "aircraft"


def _all_models():
    polar = ParabolicDragPolar(0.02, 0.04)
    transonic = TransonicCompressibilityDragModel(
        0.020053,
        MachDependentInducedDrag(0.029856, 0.121, 1.632),
        HyperbolicTangentWaveDrag(
            lift_coefficient_grid=np.array([0.3, 0.4, 0.5]),
            critical_mach_grid=np.array([0.691745, 0.693807, 0.686675]),
            amplitude=0.048626,
            mach_scale=18.637,
            shift=23.637,
        ),
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
    return {
        "parabolic": polar,
        "parabolic_wave": ParabolicWaveDragModel(
            polar, KornLockWaveDrag(np.deg2rad(25.0), 0.12)
        ),
        "transonic_compressibility": transonic,
        "poll_schumann": poll,
        "geometry_transonic": build_drag_model(load_aircraft("A320")),
        "geometry_subsonic": build_drag_model(load_aircraft("C172")),
    }


@pytest.mark.parametrize("name", list(_all_models()))
@pytest.mark.parametrize(
    "condition",
    [
        DragCondition(0.7, reynolds_per_m=-1.0),
        DragCondition(0.7, gear_extension_fraction=1.1),
        DragCondition(-0.1),
    ],
)
def test_scalar_and_array_paths_reject_invalid_conditions(name, condition):
    model = _all_models()[name]
    with pytest.raises(ValueError):
        model.drag_coefficient(0.4, condition=condition)
    with pytest.raises(ValueError):
        model._drag_coefficient_scalar(0.4, condition)


@pytest.mark.parametrize("name", list(_all_models()))
@pytest.mark.parametrize(
    "deployed",
    [
        {"gear_extension_fraction": 1.0},
        {"flap_angle_rad": 0.3},
        {"spoiler_deflection_rad": 0.1},
    ],
)
def test_clean_models_reject_a_deployed_configuration(name, deployed):
    model = _all_models()[name]
    condition = DragCondition(0.7, **deployed)
    if name.startswith("geometry") and "spoiler_deflection_rad" not in deployed:
        # The geometry models built from a record have flap and gear drag.
        assert model._drag_coefficient_scalar(0.4, condition) > (
            model._drag_coefficient_scalar(0.4, DragCondition(0.7))
        )
        return
    match = "spoiler" if name.startswith("geometry") else "only when clean"
    with pytest.raises(ValueError, match=match):
        model.drag_coefficient(0.4, condition=condition)
    with pytest.raises(ValueError, match=match):
        model._drag_coefficient_scalar(0.4, condition)


def test_cl_keyword_and_cd_shorthand_on_every_model():
    for name, model in _all_models().items():
        mach = 0.75 if name == "transonic_compressibility" else 0.7
        cl = 0.45
        expected = model.drag_coefficient(cl, mach)
        assert model.drag_coefficient(cl=cl, mach=mach) == pytest.approx(expected)
        assert model.cd(cl=cl, mach=mach) == pytest.approx(expected), name
        assert model.breakdown(cl=cl, mach=mach).total == pytest.approx(expected)
    polar = ParabolicDragPolar(cd0=0.02, k=0.04)
    assert polar.cd(cl=0.5) == pytest.approx(0.03)
    for name, model in _all_models().items():
        mach = 0.75 if name == "transonic_compressibility" else 0.7
        assert model._drag_coefficient_scalar(
            cl=0.45, condition=DragCondition(mach)
        ) == pytest.approx(model.cd(cl=0.45, mach=mach)), name


def test_cd_follows_a_subclass_override_of_drag_coefficient():
    @dataclass(frozen=True)
    class ShiftedPolar(ParabolicDragPolar):
        def drag_coefficient(
            self, cl, mach=None, reynolds_per_m=None, *, h=None, dT=None, condition=None
        ):
            base = super().drag_coefficient(
                cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
            )
            return base + 0.01

    polar = ShiftedPolar(cd0=0.02, k=0.04)
    assert polar.drag_coefficient(cl=0.5) == pytest.approx(0.04)
    assert polar.cd(cl=0.5) == pytest.approx(0.04)
    assert list(inspect.signature(polar.cd).parameters) == [
        "cl",
        "mach",
        "reynolds_per_m",
        "h",
        "dT",
        "condition",
    ]


def test_drag_model_protocol_requires_breakdown():
    class CoefficientOnly:
        def drag_coefficient(
            self, lift_coefficient, mach=None, reynolds_per_m=None, *, condition=None
        ):
            return np.asarray(0.03)

        def _drag_coefficient_scalar(self, lift_coefficient, condition):
            return 0.03

    assert not isinstance(CoefficientOnly(), DragModel)
    for model in _all_models().values():
        assert isinstance(model, DragModel)


def _legacy_a320():
    path = DATA_ROOT / "yml" / "turbofan" / "a320.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    drag_section = dict(document["drag"])
    drag_section.pop("reference_reynolds_per_m")
    drag_section["reference_unit_reynolds_number_per_m"] = 4.2e6
    legacy = {**document, "drag": drag_section}
    notes = (DATA_ROOT / "sources" / "turbofan" / "a320.txt").read_text(
        encoding="utf-8"
    )
    legacy_notes = notes.replace(
        "drag.reference_reynolds_per_m", "drag.reference_unit_reynolds_number_per_m"
    )
    return legacy, legacy_notes, notes


def test_loader_accepts_the_legacy_reynolds_key():
    legacy, legacy_notes, notes = _legacy_a320()
    aircraft = load_aircraft_from_dict(legacy, parse_source_notes(legacy_notes))
    assert aircraft.drag.reference_reynolds_per_m == 4.2e6
    sidecar_fields = aircraft.data_provenance["sidecar"]["fields"]
    assert "drag.reference_reynolds_per_m" in sidecar_fields
    assert (
        sidecar_fields["drag.reference_reynolds_per_m"]
        == (parse_source_notes(notes)["fields"]["drag.reference_reynolds_per_m"])
    )


def test_migration_tool_upgrades_the_legacy_reynolds_key():
    legacy, legacy_notes, notes = _legacy_a320()
    normalized, sidecar = normalize_bundle(legacy, legacy_notes)
    assert normalized["drag"]["reference_reynolds_per_m"] == 4.2e6
    assert "reference_unit_reynolds_number_per_m" not in normalized["drag"]
    assert validate_simple(normalized) == []
    assert sidecar == notes


def test_altitude_gives_the_isa_reynolds_number():
    for name, model in _all_models().items():
        mach = 0.75 if name == "transonic_compressibility" else 0.7
        for dT in (None, 10.0):
            reynolds = aero.reynolds_per_m(mach=mach, h=9_000, dT=dT or 0)
            expected = model.cd(cl=0.45, mach=mach, reynolds_per_m=reynolds)
            assert model.cd(cl=0.45, mach=mach, h=9_000, dT=dT) == pytest.approx(
                expected
            ), name
            assert model.breakdown(
                cl=0.45, mach=mach, h=9_000, dT=dT
            ).total == pytest.approx(expected), name


def test_altitude_conflicts_raise():
    model = build_drag_model(load_aircraft("A320"))
    with pytest.raises(TypeError):
        model.cd(cl=0.5, mach=0.78, h=11_000, reynolds_per_m=5e6)
    with pytest.raises(TypeError):
        model.cd(cl=0.5, h=11_000, condition=DragCondition(0.78))
    with pytest.raises(TypeError):
        model.cd(cl=0.5, mach=0.78, dT=5.0)
    with pytest.raises(TypeError):
        model.cd(cl=0.5, h=11_000)
    with pytest.raises(TypeError):
        ParabolicDragPolar(0.02, 0.04).cd(cl=0.5, h=11_000)
