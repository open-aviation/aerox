"""Build aircraft drag models with explicit geometry imputation.

The condition-aware model in :mod:`aerox.models.drag.geometry` requires a
complete wing and fuselage definition. This module resolves incomplete aircraft
records using frozen, class-scaled dimensionless defaults, then evaluates the
physical model at a documented representative cruise condition. The stored
fields are independent inputs; ``CD0`` and ``k`` are derived only when a
low-order reference-polar projection is requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox import aero
from aerox.aircraft import Aircraft
from aerox.models.atmosphere import isa_atmosphere, unit_reynolds_number

from .core import DragCondition
from .geometry import (
    GeometryDragModel,
    TransonicGeometryDragModel,
    geometry_drag_model_from_dimensions,
)
from .polar import ParabolicDragPolar, project_parabolic_polar
from .wave import KornLockWaveDrag

GEOMETRY_DRAG_MODEL_ID = "aerox-geometry-drag-v1"
KORN_LOCK_WAVE_DRAG_MODEL_ID = "aerox-korn-lock-wave-v1"
_HIGH_SPEED_AIRCRAFT = frozenset({"F15", "F16", "HAWK", "T38"})
_SPECIAL_AIRCRAFT = _HIGH_SPEED_AIRCRAFT | frozenset({"C17", "K35R"})


@dataclass(frozen=True)
class GeometryScalingDefaults:
    """Dimensionless defaults used only when a published field is absent."""

    aspect_ratio: float
    mac_to_mean_chord: float
    thickness_to_chord_ratio: float
    sweep_deg: float
    fuselage_fineness_ratio: float
    fuselage_length_to_span: float
    horizontal_tail_area_ratio: float
    horizontal_tail_mac_ratio: float
    vertical_tail_area_ratio: float
    vertical_tail_mac_ratio: float
    span_efficiency_factor: float
    unresolved_fraction_of_resolved_drag: float
    reference_ceiling_fraction: float
    maximum_reference_altitude_ft: float
    default_reference_altitude_ft: float
    default_reference_mach: float


# The civil-jet ratios are database medians of the currently sourced records.
# Turboprop and piston values are deliberately broad conceptual-design priors;
# their sparse public geometry keeps those records in the low-confidence tier.
_DEFAULTS = {
    "turbofan": GeometryScalingDefaults(
        aspect_ratio=8.6522,
        mac_to_mean_chord=1.1241,
        thickness_to_chord_ratio=0.11,
        sweep_deg=25.0,
        fuselage_fineness_ratio=10.27,
        fuselage_length_to_span=1.10,
        horizontal_tail_area_ratio=0.249,
        horizontal_tail_mac_ratio=0.641,
        vertical_tail_area_ratio=0.166,
        vertical_tail_mac_ratio=0.854,
        span_efficiency_factor=0.95,
        unresolved_fraction_of_resolved_drag=0.23,
        reference_ceiling_fraction=0.85,
        maximum_reference_altitude_ft=35_000.0,
        default_reference_altitude_ft=35_000.0,
        default_reference_mach=0.78,
    ),
    "turboprop": GeometryScalingDefaults(
        aspect_ratio=11.13,
        mac_to_mean_chord=1.053,
        thickness_to_chord_ratio=0.15,
        sweep_deg=0.0,
        fuselage_fineness_ratio=11.0,
        fuselage_length_to_span=0.90,
        horizontal_tail_area_ratio=0.24,
        horizontal_tail_mac_ratio=0.65,
        vertical_tail_area_ratio=0.16,
        vertical_tail_mac_ratio=0.85,
        span_efficiency_factor=0.95,
        unresolved_fraction_of_resolved_drag=0.28,
        reference_ceiling_fraction=0.80,
        maximum_reference_altitude_ft=25_000.0,
        default_reference_altitude_ft=20_000.0,
        default_reference_mach=0.45,
    ),
    "piston": GeometryScalingDefaults(
        aspect_ratio=7.46,
        mac_to_mean_chord=1.05,
        thickness_to_chord_ratio=0.15,
        sweep_deg=0.0,
        fuselage_fineness_ratio=6.0,
        fuselage_length_to_span=0.75,
        horizontal_tail_area_ratio=0.22,
        horizontal_tail_mac_ratio=0.65,
        vertical_tail_area_ratio=0.14,
        vertical_tail_mac_ratio=0.85,
        span_efficiency_factor=0.95,
        unresolved_fraction_of_resolved_drag=0.35,
        reference_ceiling_fraction=0.65,
        maximum_reference_altitude_ft=10_000.0,
        default_reference_altitude_ft=8_000.0,
        default_reference_mach=0.22,
    ),
}

_HIGH_SPEED_DEFAULTS = GeometryScalingDefaults(
    aspect_ratio=3.0,
    mac_to_mean_chord=1.20,
    thickness_to_chord_ratio=0.06,
    sweep_deg=35.0,
    fuselage_fineness_ratio=8.5,
    fuselage_length_to_span=1.50,
    horizontal_tail_area_ratio=0.20,
    horizontal_tail_mac_ratio=0.70,
    vertical_tail_area_ratio=0.16,
    vertical_tail_mac_ratio=0.90,
    span_efficiency_factor=0.85,
    unresolved_fraction_of_resolved_drag=0.30,
    reference_ceiling_fraction=0.75,
    maximum_reference_altitude_ft=35_000.0,
    default_reference_altitude_ft=30_000.0,
    default_reference_mach=0.85,
)


@dataclass(frozen=True)
class ResolvedDragGeometry:
    """Observed and scaled geometry used for one aircraft drag model."""

    wing_area_m2: float
    wing_span_m: float
    wing_mac_m: float
    wing_thickness_to_chord_ratio: float
    wing_sweep_deg: float
    fuselage_length_m: float
    fuselage_width_m: float
    fuselage_height_m: float
    horizontal_tail_area_m2: float
    horizontal_tail_mac_m: float
    vertical_tail_area_m2: float
    vertical_tail_mac_m: float
    imputed_fields: tuple[str, ...]
    geometry_tier: str
    confidence: str
    defaults: GeometryScalingDefaults
    span_efficiency_factor: float

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m**2 / self.wing_area_m2


@dataclass(frozen=True)
class ReferenceDragCondition:
    """Representative clean condition used to configure an aircraft model."""

    pressure_altitude_ft: float
    mach: float
    unit_reynolds_number_per_m: float


@dataclass(frozen=True)
class DragBuildReport:
    """Generated configuration and diagnostics for one aircraft drag model."""

    model: str
    configuration: str
    geometry_tier: str
    confidence: str
    cd0: float
    k: float
    reference_altitude_ft: float
    reference_mach: float
    reference_unit_reynolds_number_per_m: float
    span_efficiency_factor: float
    unresolved_fraction_of_resolved_drag: float
    wave_drag_model: str | None
    wave_drag_source_tier: str
    wave_drag_confidence: str
    wave_drag_technology_factor: float | None
    imputed_fields: tuple[str, ...]


def _wave_drag_config(
    aircraft: Aircraft,
) -> tuple[str | None, str, str, float | None]:
    """Return the supported transonic-wave configuration without extrapolation.

    The compact Korn/Lock branch is applicable to subsonic civil turbofans.
    Low-speed propeller aircraft and special military/transport layouts retain
    an explicit not-applicable marker rather than a fabricated coefficient.
    """

    icao = str(aircraft.icao or "").upper()
    engine_type = aircraft.engine.type if aircraft.engine is not None else None
    if engine_type != "turbofan" or icao in _SPECIAL_AIRCRAFT:
        return None, "not-applicable", "not-applicable", None
    if aircraft.mmo is not None and float(aircraft.mmo) >= 1.0:
        return None, "not-applicable", "not-applicable", None
    if icao == "A320":
        return (
            KORN_LOCK_WAVE_DRAG_MODEL_ID,
            "C-public-increment-cross-check",
            "low",
            0.95,
        )
    return KORN_LOCK_WAVE_DRAG_MODEL_ID, "D-class-scaled", "low", 0.95


def _positive(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) and result > 0.0 else None


def _scaling_defaults_for_aircraft(
    aircraft: Aircraft,
) -> tuple[GeometryScalingDefaults, bool]:
    icao = str(aircraft.icao or "").upper()
    high_speed = icao in _HIGH_SPEED_AIRCRAFT or (
        aircraft.mmo is not None and float(aircraft.mmo) >= 1.0
    )
    special = icao in _SPECIAL_AIRCRAFT
    if high_speed:
        return _HIGH_SPEED_DEFAULTS, True
    engine_type = aircraft.engine.type if aircraft.engine is not None else None
    try:
        return _DEFAULTS[str(engine_type)], special
    except KeyError as exc:
        raise ValueError(
            f"aircraft {aircraft.icao or '<unknown>'} has no supported engine class"
        ) from exc


def resolve_drag_geometry(aircraft: Aircraft) -> ResolvedDragGeometry:
    """Resolve one record without mutating or filling its source geometry."""

    if aircraft.wing is None or aircraft.fuselage is None:
        raise ValueError("aircraft wing and fuselage sections are required")
    defaults, special = _scaling_defaults_for_aircraft(aircraft)
    missing: list[str] = []

    area = _positive(aircraft.wing.area)
    span = _positive(aircraft.wing.span)
    if area is None and span is None:
        raise ValueError("at least one of wing area or span is required")
    if area is None:
        area = float(span**2 / defaults.aspect_ratio)
        missing.append("wing.area")
    if span is None:
        span = float(np.sqrt(area * defaults.aspect_ratio))
        missing.append("wing.span")

    mac = _positive(aircraft.wing.mac)
    if mac is None:
        mac = defaults.mac_to_mean_chord * area / span
        missing.append("wing.mac")
    thickness = _positive(aircraft.wing.t_c)
    if thickness is None or not 0.02 <= thickness <= 0.30:
        thickness = defaults.thickness_to_chord_ratio
        missing.append("wing.t/c")
    sweep = aircraft.wing.sweep
    if sweep is None or not np.isfinite(float(sweep)):
        sweep = defaults.sweep_deg
        missing.append("wing.sweep")
    sweep = float(sweep)

    length = _positive(aircraft.fuselage.length)
    if length is None:
        length = defaults.fuselage_length_to_span * span
        missing.append("fuselage.length")
    width = _positive(aircraft.fuselage.width)
    height = _positive(aircraft.fuselage.height)
    if width is None or height is None:
        equivalent_diameter = length / defaults.fuselage_fineness_ratio
        width = equivalent_diameter
        height = equivalent_diameter
        missing.extend(
            field
            for field in ("fuselage.width", "fuselage.height")
            if field not in missing
        )

    horizontal_area = (
        _positive(aircraft.horizontal_tail.area)
        if aircraft.horizontal_tail is not None
        else None
    )
    if horizontal_area is None:
        horizontal_area = defaults.horizontal_tail_area_ratio * area
        missing.append("horizontal_tail.area")
    horizontal_mac = (
        _positive(aircraft.horizontal_tail.mac)
        if aircraft.horizontal_tail is not None
        else None
    )
    if horizontal_mac is None:
        horizontal_mac = defaults.horizontal_tail_mac_ratio * mac
        missing.append("horizontal_tail.mac")

    vertical_area = (
        _positive(aircraft.vertical_tail.area)
        if aircraft.vertical_tail is not None
        else None
    )
    if vertical_area is None:
        vertical_area = defaults.vertical_tail_area_ratio * area
        missing.append("vertical_tail.area")
    vertical_mac = (
        _positive(aircraft.vertical_tail.mac)
        if aircraft.vertical_tail is not None
        else None
    )
    if vertical_mac is None:
        vertical_mac = defaults.vertical_tail_mac_ratio * mac
        missing.append("vertical_tail.mac")

    core_fields = {
        "wing.area",
        "wing.span",
        "wing.mac",
        "wing.t/c",
        "fuselage.length",
        "fuselage.width",
        "fuselage.height",
    }
    tail_fields = {
        "horizontal_tail.area",
        "horizontal_tail.mac",
        "vertical_tail.area",
        "vertical_tail.mac",
    }
    core_missing = len(core_fields.intersection(missing))
    tail_missing = len(tail_fields.intersection(missing))
    if special:
        tier, confidence = "X-special-scaled", "low"
    elif core_missing == 0 and tail_missing == 0:
        tier, confidence = "A-observed-core-and-tails", "medium"
    elif core_missing == 0:
        tier, confidence = "B-observed-core-scaled-tails", "medium-low"
    elif core_missing <= 2:
        tier, confidence = "C-partially-scaled", "low"
    else:
        tier, confidence = "D-class-scaled", "low"

    engine_type = aircraft.engine.type if aircraft.engine is not None else None
    span_efficiency = defaults.span_efficiency_factor
    if engine_type == "turbofan" and not special:
        # Poll-Schumann pure-vortex branch: delta1=0.03 and
        # delta2=2*(maximum fuselage width / wing span)^2.  Profile drag is
        # represented separately below, so this is intentionally not an
        # effective whole-aircraft Oswald correlation.
        span_efficiency = 1.0 / (1.03 + 2.0 * (width / span) ** 2)

    return ResolvedDragGeometry(
        wing_area_m2=area,
        wing_span_m=span,
        wing_mac_m=mac,
        wing_thickness_to_chord_ratio=thickness,
        wing_sweep_deg=sweep,
        fuselage_length_m=length,
        fuselage_width_m=width,
        fuselage_height_m=height,
        horizontal_tail_area_m2=horizontal_area,
        horizontal_tail_mac_m=horizontal_mac,
        vertical_tail_area_m2=vertical_area,
        vertical_tail_mac_m=vertical_mac,
        imputed_fields=tuple(dict.fromkeys(missing)),
        geometry_tier=tier,
        confidence=confidence,
        defaults=defaults,
        span_efficiency_factor=float(span_efficiency),
    )


def derive_reference_drag_condition(
    aircraft: Aircraft, geometry: ResolvedDragGeometry
) -> ReferenceDragCondition:
    """Derive the representative clean condition for an aircraft model."""

    defaults = geometry.defaults
    ceiling = _positive(aircraft.ceiling)
    if ceiling is None:
        altitude_ft = defaults.default_reference_altitude_ft
    else:
        altitude_ft = min(
            defaults.maximum_reference_altitude_ft,
            defaults.reference_ceiling_fraction * ceiling,
        )
    provisional_atmosphere = isa_atmosphere(altitude_ft * aero.ft)

    cruise_mach = (
        _positive(aircraft.kinematic.cruise_mach)
        if aircraft.kinematic is not None
        else None
    )
    if cruise_mach is None and aircraft.kinematic is not None:
        cruise_tas = _positive(aircraft.kinematic.cruise_tas)
        if cruise_tas is not None:
            cruise_mach = (
                cruise_tas * aero.kts / float(provisional_atmosphere.speed_of_sound_mps)
            )
    mach = float(cruise_mach or defaults.default_reference_mach)
    mach = float(np.clip(mach, 0.08, 0.95))
    maximum_operating_mach = _positive(aircraft.mmo)
    if maximum_operating_mach is not None:
        # Some sparse legacy kinematic records contain a class-default cruise
        # Mach above the aircraft's published MMO. The reference condition may
        # reach, but must never exceed, the certified operating envelope.
        mach = min(mach, maximum_operating_mach)

    engine_type = aircraft.engine.type if aircraft.engine is not None else None
    special = str(aircraft.icao or "").upper() in _SPECIAL_AIRCRAFT
    mtow = _positive(aircraft.mtow)
    if engine_type == "turbofan" and not special and mtow is not None:
        # Poll-Schumann design-condition fallback: clean ISA, 0.8 MTOW and a
        # representative transport CL.  Solving lift for pressure avoids an
        # arbitrary common cruise altitude and recovers about 5.05e6 1/m for
        # the A320 reference case.
        reference_lift_coefficient = 0.556
        pressure_pa = (
            2.0
            * 0.80
            * mtow
            * aero.g0
            / (
                aero.gamma
                * mach**2
                * geometry.wing_area_m2
                * reference_lift_coefficient
            )
        )
        pressure_pa = float(np.clip(pressure_pa, 11_000.0, aero.p0))
        altitude_ft = float(aero.h_isa(pressure_pa) / aero.ft)

    altitude_m = altitude_ft * aero.ft
    atmosphere = isa_atmosphere(altitude_m)
    speed = mach * float(atmosphere.speed_of_sound_mps)
    unit_reynolds = float(
        unit_reynolds_number(
            atmosphere.density_kg_m3,
            speed,
            atmosphere.temperature_k,
        )
    )
    return ReferenceDragCondition(float(altitude_ft), mach, unit_reynolds)


def _build_subcritical_geometry_model(
    geometry: ResolvedDragGeometry,
    reference_mach: float,
    reference_unit_reynolds_number_per_m: float,
    *,
    span_efficiency_factor: float | None = None,
    unresolved_fraction_of_resolved_drag: float | None = None,
) -> GeometryDragModel:
    span_efficiency = (
        geometry.span_efficiency_factor
        if span_efficiency_factor is None
        else float(span_efficiency_factor)
    )
    unresolved_fraction = (
        geometry.defaults.unresolved_fraction_of_resolved_drag
        if unresolved_fraction_of_resolved_drag is None
        else float(unresolved_fraction_of_resolved_drag)
    )
    return geometry_drag_model_from_dimensions(
        wing_area_m2=geometry.wing_area_m2,
        wing_span_m=geometry.wing_span_m,
        wing_mean_aerodynamic_chord_m=geometry.wing_mac_m,
        wing_thickness_to_chord_ratio=(geometry.wing_thickness_to_chord_ratio),
        wing_sweep_rad=float(np.deg2rad(geometry.wing_sweep_deg)),
        fuselage_length_m=geometry.fuselage_length_m,
        fuselage_width_m=geometry.fuselage_width_m,
        fuselage_height_m=geometry.fuselage_height_m,
        horizontal_tail_area_m2=geometry.horizontal_tail_area_m2,
        horizontal_tail_mean_aerodynamic_chord_m=(geometry.horizontal_tail_mac_m),
        vertical_tail_area_m2=geometry.vertical_tail_area_m2,
        vertical_tail_mean_aerodynamic_chord_m=geometry.vertical_tail_mac_m,
        reference_mach=reference_mach,
        reference_unit_reynolds_number_per_m=(reference_unit_reynolds_number_per_m),
        span_efficiency_factor=span_efficiency,
        unresolved_fraction_of_resolved_drag=unresolved_fraction,
    )


def build_drag_report(aircraft: Aircraft) -> DragBuildReport:
    """Generate deterministic physical drag parameters for an aircraft."""

    geometry = resolve_drag_geometry(aircraft)
    reference = derive_reference_drag_condition(aircraft, geometry)
    altitude_ft = reference.pressure_altitude_ft
    mach = reference.mach
    unit_reynolds = reference.unit_reynolds_number_per_m
    model = _build_subcritical_geometry_model(geometry, mach, unit_reynolds)
    condition = DragCondition(mach, unit_reynolds)
    cd0 = model.drag_coefficient_scalar_at(0.0, condition)
    k = (
        model.span_induced_drag_factor
        + model.profile_lift_drag_multiplier * cd0 * model.induced_scale
    )
    effective_e = 1.0 / (np.pi * geometry.aspect_ratio * k)
    values = np.asarray([cd0, k, effective_e, unit_reynolds], dtype=float)
    if np.any(~np.isfinite(values)) or not (
        0.005 < cd0 < 0.10 and 0.01 < k < 0.25 and 0.2 < effective_e < 1.2
    ):
        raise ValueError(
            f"aircraft {aircraft.icao or '<unknown>'} produced an implausible model"
        )
    wave_model, wave_tier, wave_confidence, technology_factor = _wave_drag_config(
        aircraft
    )
    return DragBuildReport(
        model=GEOMETRY_DRAG_MODEL_ID,
        configuration="clean",
        geometry_tier=geometry.geometry_tier,
        confidence=geometry.confidence,
        cd0=float(cd0),
        k=float(k),
        reference_altitude_ft=altitude_ft,
        reference_mach=mach,
        reference_unit_reynolds_number_per_m=unit_reynolds,
        span_efficiency_factor=geometry.span_efficiency_factor,
        unresolved_fraction_of_resolved_drag=(
            geometry.defaults.unresolved_fraction_of_resolved_drag
        ),
        wave_drag_model=wave_model,
        wave_drag_source_tier=wave_tier,
        wave_drag_confidence=wave_confidence,
        wave_drag_technology_factor=technology_factor,
        imputed_fields=geometry.imputed_fields,
    )


def _build_subcritical_drag_model(aircraft: Aircraft) -> GeometryDragModel:
    """Reconstruct the subcritical condition-aware model from aircraft data."""

    drag = aircraft.drag
    if drag is None or drag.model != GEOMETRY_DRAG_MODEL_ID:
        raise ValueError(
            f"aircraft {aircraft.icao or '<unknown>'} has no supported "
            "active drag model"
        )
    required = {
        "reference_mach": drag.reference_mach,
        "reference_unit_reynolds_number_per_m": (
            drag.reference_unit_reynolds_number_per_m
        ),
        "span_efficiency_factor": drag.span_efficiency_factor,
        "unresolved_fraction_of_resolved_drag": (
            drag.unresolved_fraction_of_resolved_drag
        ),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("active drag model is missing: " + ", ".join(missing))

    geometry = resolve_drag_geometry(aircraft)
    return _build_subcritical_geometry_model(
        geometry,
        float(drag.reference_mach),
        float(drag.reference_unit_reynolds_number_per_m),
        span_efficiency_factor=float(drag.span_efficiency_factor),
        unresolved_fraction_of_resolved_drag=float(
            drag.unresolved_fraction_of_resolved_drag
        ),
    )


def build_drag_model(
    aircraft: Aircraft,
) -> GeometryDragModel | TransonicGeometryDragModel:
    """Reconstruct the active condition-aware drag model from aircraft data."""

    base = _build_subcritical_drag_model(aircraft)
    drag = aircraft.drag
    if drag is None or drag.wave_drag_model is None:
        return base
    if drag.wave_drag_model != KORN_LOCK_WAVE_DRAG_MODEL_ID:
        raise ValueError(f"unsupported wave drag model: {drag.wave_drag_model}")
    if drag.wave_drag_technology_factor is None:
        raise ValueError("active wave drag model has no technology factor")

    geometry = resolve_drag_geometry(aircraft)
    wave_drag = KornLockWaveDrag(
        wing_sweep_rad=float(np.deg2rad(geometry.wing_sweep_deg)),
        thickness_to_chord_ratio=geometry.wing_thickness_to_chord_ratio,
        technology_factor=float(drag.wave_drag_technology_factor),
    )
    return TransonicGeometryDragModel(
        base_model=base,
        wave_drag=wave_drag,
        minimum_lift_coefficient=-0.5,
        maximum_lift_coefficient=2.5,
        maximum_mach=max(0.95, float(aircraft.mmo or 0.0)),
    )


def build_reference_drag_polar(aircraft: Aircraft) -> ParabolicDragPolar:
    """Project the active physical model onto its reference parabolic polar.

    This adapter supports low-order analyses without storing derived ``CD0`` or
    ``k`` values in aircraft data. New simulations should pass the model from
    :func:`build_drag_model` so Reynolds-number variation is retained.
    """

    drag = aircraft.drag
    if drag is None or drag.reference_mach is None:
        raise ValueError("active drag model has no reference Mach")
    if drag.reference_unit_reynolds_number_per_m is None:
        raise ValueError("active drag model has no reference unit Reynolds number")
    # The low-order adapter is intentionally a subcritical projection. Wave
    # drag is Mach- and CL-dependent and cannot be represented by one CD0/k.
    model = _build_subcritical_drag_model(aircraft)
    condition = DragCondition(
        drag.reference_mach,
        drag.reference_unit_reynolds_number_per_m,
    )
    return project_parabolic_polar(model, condition)
