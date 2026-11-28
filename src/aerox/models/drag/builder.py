"""Build aircraft drag models with explicit geometry imputation.

The condition-aware model in :mod:`aerox.models.drag.geometry` requires a
complete wing and fuselage definition. This module resolves incomplete aircraft
records using frozen, class-scaled dimensionless defaults, then evaluates the
physical model at a documented representative cruise condition. The stored
fields are independent inputs; ``CD0`` and ``k`` are derived only when a
low-order reference-polar projection is requested.

Civil turbofans use a component build-up: exposed wing, fuselage, tails,
nacelles, pylons, upsweep, control-surface gaps, nacelle base, and one
published roughness and excrescence allowance. Turboprop, piston, and
high-speed aircraft keep one lumped unresolved fraction of the resolved drag.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from aerox import aero
from aerox.aircraft import Aircraft
from aerox.models._atmosphere import isa_atmosphere, unit_reynolds_number

from .components import (
    ParasiteDragComponent,
    control_surface_gap_drag_area,
    exposed_wing_area,
    fuselage_upsweep_drag_area,
    nacelle_base_drag_area,
    nacelle_form_factor,
    nacelle_wetted_area,
    trapezoidal_root_chord,
)
from .core import DragCondition
from .deployed import deployed_drag_from_aircraft
from .geometry import (
    GeometryDragModel,
    TransonicGeometryDragModel,
    geometry_drag_model_from_dimensions,
)
from .polar import ParabolicDragPolar, project_parabolic_polar
from .wave import KornLockWaveDrag

GEOMETRY_DRAG_MODEL_ID = "aerox-geometry-drag-v2"
KORN_LOCK_WAVE_DRAG_MODEL_ID = "aerox-korn-lock-wave-v1"
_HIGH_SPEED_AIRCRAFT = frozenset({"F15", "F16", "HAWK", "T38"})
_SPECIAL_AIRCRAFT = _HIGH_SPEED_AIRCRAFT | frozenset({"C17", "K35R"})
NACELLE_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "engines" / "nacelles.csv"
)

# Turbofan component build-up. Sources are given as quoted in the evidence
# collected for this model; "via Scholz" means Scholz, Aircraft Design,
# ch. 13, which quotes Raymer (1992).
#
# Roughness, excrescence, leakage, and miscellaneous items: one published
# allowance on the sum of all computed terms. 0.09 follows Kroo, AA241 notes
# (6-9 % roughness markup on skin friction plus 1.5 % miscellaneous items).
# Published range: 0.03 (Raymer, 2-5 % leakage and protuberance) to 0.13
# (Boeing 737 excrescence, 7 % of total drag; Goldhammer and Plendl, Boeing
# AERO Q1 2013).
ROUGHNESS_EXCRESCENCE_FRACTION = 0.09
# Nacelle interference Q (Raymer via Scholz Table 13.4): 1.3 within about one
# nacelle diameter of the wing, 1.5 mounted directly on the fuselage or wing.
NACELLE_INTERFERENCE_WING_MOUNTED = 1.3
NACELLE_INTERFERENCE_FUSELAGE_MOUNTED = 1.5
# Pylons: 20 % of the nacelle friction and form drag (SUAVE
# parasite_drag_pylon, also RCAIDE).
PYLON_FRACTION_OF_NACELLE_DRAG = 0.20
# Tail interference Q = 1.04 for a conventional empennage (Raymer via Scholz
# Table 13.4; 1.08 and 1.03 there are H-tail and V-tail layouts).
TAIL_INTERFERENCE_FACTOR = 1.04
# Kroo control-surface gap class values: affected area is 0.3 S_wing,
# 1.0 S_horizontal, and 0.9 S_vertical when no layout data is available.
_GAP_AFFECTED_AREA_RATIOS = (0.3, 1.0, 0.9)
# Nacelle ratios from Airbus airport-planning drawings and NASA (Guynn et al.,
# NTRS 20140001089): D_max = 1.25 D_fan; L = 1.95 D_fan (short duct, inlet lip
# to fan-nozzle exit) or 3.1 D_fan (long duct, inlet lip to common nozzle).
NACELLE_MAX_DIAMETER_TO_FAN = 1.25
NACELLE_LENGTH_TO_FAN = {"short": 1.95, "long": 3.1}
# Lowest tier: D_fan = c sqrt(MTOW / engines), MTOW in kg. The square root
# follows from fan area scaling with thrust and thrust with weight. ``c`` is
# the median of D_fan / sqrt(MTOW / engines) over the bundled turbofan records
# whose default engine has a published fan diameter in nacelles.csv.
FAN_DIAMETER_PER_SQRT_MTOW_PER_ENGINE = 0.00827


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
    unresolved_fraction_of_resolved_drag: float | None
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
        # Civil turbofans use the component build-up, not a lumped fraction.
        unresolved_fraction_of_resolved_drag=None,
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
class NacelleGeometry:
    """Nacelle dimensions resolved from the default engine designation."""

    engine: str | None
    family: str | None
    fan_diameter_m: float
    max_diameter_m: float
    length_m: float
    duct: str
    tier: str
    imputed_fields: tuple[str, ...]


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
    component_buildup: bool = False
    wing_exposed_area_m2: float | None = None
    horizontal_tail_sweep_deg: float | None = None
    vertical_tail_sweep_deg: float | None = None
    engine_count: int | None = None
    nacelle_interference_factor: float | None = None
    nacelle: NacelleGeometry | None = None

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m**2 / self.wing_area_m2


@dataclass(frozen=True)
class ReferenceDragCondition:
    """Representative clean condition used to configure an aircraft model."""

    pressure_altitude_ft: float
    mach: float
    reynolds_per_m: float


@dataclass(frozen=True)
class DragTerm:
    """One zero-lift drag term at the reference condition."""

    name: str
    value: float
    method: str


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
    reference_reynolds_per_m: float
    span_efficiency_factor: float
    unresolved_fraction_of_resolved_drag: float | None
    roughness_excrescence_fraction: float | None
    wave_drag_model: str | None
    wave_drag_source_tier: str
    wave_drag_confidence: str
    wave_drag_technology_factor: float | None
    imputed_fields: tuple[str, ...]
    parasite_terms: tuple[DragTerm, ...]
    nacelle_geometry_tier: str | None


# Korn technology factor for each airfoil generation (``wing.airfoil_class``),
# in the form Mdd = kappa/cos(sweep) - (t/c)/cos^2(sweep) - CL/(10 cos^3(sweep)).
# - conventional 0.87: Mason, Configuration Aerodynamics (2019), eq. 7-3
#   (NACA 6-series sections).
# - peaky 0.89: Shevell, supercritical sections 0.06 above conventional
#   (Ciornei 2005, eq. 3.20), and Ciornei Table 3.1.
# - early-supercritical 0.93: Ciornei Table 3.1 (0.04 above peaky); Kroo AA241
#   (supercritical sections typically 0.03 to 0.04 above peaky).
# - supercritical 0.95: Mason eq. 7-3; Torenbeek, Advanced Aircraft Design
#   (2013), section 10.6.4; Gur, Mason and Schetz (2010).
# A record without a class uses 0.95 with its previous source tier.
AIRFOIL_TECHNOLOGY_FACTORS = {
    "conventional": 0.87,
    "peaky": 0.89,
    "early-supercritical": 0.93,
    "supercritical": 0.95,
}
AIRFOIL_CLASS_WAVE_TIER = "C-airfoil-class"


def _wave_drag_config(
    aircraft: Aircraft,
) -> tuple[str | None, str, str, float | None]:
    """Return the supported transonic-wave configuration without extrapolation.

    The compact Korn/Lock branch is applicable to subsonic civil turbofans.
    Low-speed propeller aircraft and special military/transport layouts retain
    an explicit not-applicable marker rather than a fabricated coefficient.
    The technology factor comes from the airfoil class of the wing when the
    record gives one.
    """

    icao = str(aircraft.icao or "").upper()
    engine_type = aircraft.engine.type if aircraft.engine is not None else None
    if engine_type != "turbofan" or icao in _SPECIAL_AIRCRAFT:
        return None, "not-applicable", "not-applicable", None
    if aircraft.mmo is not None and float(aircraft.mmo) >= 1.0:
        return None, "not-applicable", "not-applicable", None
    airfoil_class = aircraft.wing.airfoil_class if aircraft.wing else None
    if icao == "A320":
        return (
            KORN_LOCK_WAVE_DRAG_MODEL_ID,
            "C-public-increment-cross-check",
            "low",
            AIRFOIL_TECHNOLOGY_FACTORS.get(airfoil_class, 0.95),
        )
    if airfoil_class is not None:
        return (
            KORN_LOCK_WAVE_DRAG_MODEL_ID,
            AIRFOIL_CLASS_WAVE_TIER,
            "low",
            AIRFOIL_TECHNOLOGY_FACTORS[airfoil_class],
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


def uses_component_buildup(aircraft: Aircraft) -> bool:
    """Return whether the turbofan component build-up applies to a record.

    Only civil-class turbofans (including the C17 and K35R) use it. Turboprop,
    piston, and high-speed records keep the lumped unresolved fraction.
    """

    defaults, _ = _scaling_defaults_for_aircraft(aircraft)
    return defaults is _DEFAULTS["turbofan"]


@lru_cache(maxsize=1)
def _nacelle_table() -> tuple[dict[str, str], ...]:
    with NACELLE_DATA_PATH.open(encoding="utf-8", newline="") as stream:
        return tuple(csv.DictReader(stream))


def resolve_nacelle_geometry(aircraft: Aircraft) -> NacelleGeometry:
    """Resolve one nacelle of the default engine with its evidence tier.

    Tier ``A-measured-nacelle`` uses manufacturer drawing dimensions. Tier
    ``B-fan-diameter-ratio`` scales a published fan diameter with the nacelle
    ratios. Tier ``C-estimated-fan-diameter`` first estimates the fan diameter
    from MTOW per engine and then applies the short-duct ratios.
    """

    engine = aircraft.engine.default if aircraft.engine is not None else None
    designation = str(engine or "").strip()
    row = next(
        (
            item
            for item in _nacelle_table()
            if designation and re.search(item["engine_pattern"], designation)
        ),
        None,
    )
    if row is not None and row["nacelle_max_diameter_m"] and row["nacelle_length_m"]:
        return NacelleGeometry(
            engine=engine,
            family=row["family"],
            fan_diameter_m=float(row["fan_diameter_m"]),
            max_diameter_m=float(row["nacelle_max_diameter_m"]),
            length_m=float(row["nacelle_length_m"]),
            duct=row["duct"],
            tier=row["geometry_tier"],
            imputed_fields=(),
        )
    if row is not None:
        fan_diameter = float(row["fan_diameter_m"])
        duct = row["duct"]
        family = row["family"]
        tier = row["geometry_tier"]
        imputed: tuple[str, ...] = (
            "engine.nacelle_max_diameter",
            "engine.nacelle_length",
        )
    else:
        mtow = _positive(aircraft.mtow)
        count = aircraft.engine.number if aircraft.engine is not None else None
        if mtow is None or not count:
            raise ValueError(
                f"aircraft {aircraft.icao or '<unknown>'} needs MTOW and engine "
                "count to estimate nacelle geometry"
            )
        fan_diameter = FAN_DIAMETER_PER_SQRT_MTOW_PER_ENGINE * np.sqrt(mtow / count)
        duct = "short"
        family = None
        tier = "C-estimated-fan-diameter"
        imputed = (
            "engine.fan_diameter",
            "engine.nacelle_max_diameter",
            "engine.nacelle_length",
        )
    return NacelleGeometry(
        engine=engine,
        family=family,
        fan_diameter_m=float(fan_diameter),
        max_diameter_m=float(NACELLE_MAX_DIAMETER_TO_FAN * fan_diameter),
        length_m=float(NACELLE_LENGTH_TO_FAN[duct] * fan_diameter),
        duct=duct,
        tier=tier,
        imputed_fields=imputed,
    )


def nacelle_interference_factor(mount: str | None, engine_count: int) -> float:
    """Return the mean nacelle interference factor for the engine layout.

    Wing-mounted nacelles use 1.3 and fuselage-mounted nacelles 1.5. A layout
    with wing nacelles and a fin-integrated engine counts the centre engine as
    fuselage mounted.
    """

    text = str(mount or "").lower()
    wing = "wing" in text
    body = "fuselage" in text or "fin" in text or text.strip() == "rear"
    if engine_count < 1:
        raise ValueError("engine count must be positive")
    if wing and body:
        wing_count = engine_count - 1
    elif wing:
        wing_count = engine_count
    elif body:
        wing_count = 0
    else:
        raise ValueError(f"unsupported engine mount: {mount!r}")
    return float(
        (
            wing_count * NACELLE_INTERFERENCE_WING_MOUNTED
            + (engine_count - wing_count) * NACELLE_INTERFERENCE_FUSELAGE_MOUNTED
        )
        / engine_count
    )


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

    buildup: dict[str, object] = {}
    if uses_component_buildup(aircraft):
        root_chord = _positive(aircraft.wing.root_chord)
        if root_chord is None:
            root_chord = trapezoidal_root_chord(area, span, mac)
            missing.append("wing.root_chord")
        tail_sweeps = []
        for name, tail in (
            ("horizontal_tail", aircraft.horizontal_tail),
            ("vertical_tail", aircraft.vertical_tail),
        ):
            tail_sweep = None if tail is None else tail.sweep
            if tail_sweep is None or not np.isfinite(float(tail_sweep)):
                tail_sweep = sweep
                missing.append(f"{name}.sweep")
            tail_sweeps.append(float(tail_sweep))
        count = aircraft.engine.number if aircraft.engine is not None else None
        if not count:
            raise ValueError(f"aircraft {aircraft.icao} has no engine count")
        nacelle = resolve_nacelle_geometry(aircraft)
        missing.extend(nacelle.imputed_fields)
        buildup = {
            "component_buildup": True,
            "wing_exposed_area_m2": exposed_wing_area(area, span, width, root_chord),
            "horizontal_tail_sweep_deg": tail_sweeps[0],
            "vertical_tail_sweep_deg": tail_sweeps[1],
            "engine_count": int(count),
            "nacelle_interference_factor": nacelle_interference_factor(
                aircraft.engine.mount, int(count)
            ),
            "nacelle": nacelle,
        }

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
        **buildup,
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
        if ceiling is not None:
            # Light wing loadings solve to a pressure below the 11 kPa floor
            # (about 51,100 ft). Like the Mach cap, the reference condition
            # stays inside the published operating envelope.
            altitude_ft = min(altitude_ft, ceiling)

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


def _turbofan_buildup_terms(
    geometry: ResolvedDragGeometry,
) -> tuple[tuple[ParasiteDragComponent, ...], tuple[tuple[str, float], ...]]:
    """Return nacelle and pylon components and the constant drag areas."""

    nacelle = geometry.nacelle
    count = geometry.engine_count
    interference = geometry.nacelle_interference_factor
    if nacelle is None or count is None or interference is None:
        raise ValueError("turbofan build-up requires resolved nacelle geometry")
    wetted = count * nacelle_wetted_area(nacelle.max_diameter_m, nacelle.length_m)
    form_factor = nacelle_form_factor(nacelle.max_diameter_m, nacelle.length_m)
    components = (
        ParasiteDragComponent(
            name="nacelles",
            wetted_area_m2=wetted,
            characteristic_length_m=nacelle.length_m,
            form_factor=form_factor,
            interference_factor=interference,
        ),
        ParasiteDragComponent(
            name="pylons",
            wetted_area_m2=PYLON_FRACTION_OF_NACELLE_DRAG * wetted,
            characteristic_length_m=nacelle.length_m,
            form_factor=form_factor,
        ),
    )
    wing_ratio, horizontal_ratio, vertical_ratio = _GAP_AFFECTED_AREA_RATIOS
    gap_surfaces = (
        (
            wing_ratio * geometry.wing_area_m2,
            float(np.deg2rad(geometry.wing_sweep_deg)),
        ),
        (
            horizontal_ratio * geometry.horizontal_tail_area_m2,
            float(np.deg2rad(geometry.horizontal_tail_sweep_deg)),
        ),
        (
            vertical_ratio * geometry.vertical_tail_area_m2,
            float(np.deg2rad(geometry.vertical_tail_sweep_deg)),
        ),
    )
    drag_areas = (
        (
            "fuselage_upsweep",
            fuselage_upsweep_drag_area(
                geometry.fuselage_width_m, geometry.fuselage_height_m
            ),
        ),
        ("control_surface_gaps", control_surface_gap_drag_area(gap_surfaces)),
        # The nozzle exit diameter is approximated by the fan diameter.
        ("nacelle_base", count * nacelle_base_drag_area(nacelle.fan_diameter_m)),
    )
    return components, drag_areas


def _build_subcritical_geometry_model(
    geometry: ResolvedDragGeometry,
    reference_mach: float,
    reference_reynolds_per_m: float,
    *,
    span_efficiency_factor: float | None = None,
    allowance_fraction: float | None = None,
) -> GeometryDragModel:
    span_efficiency = (
        geometry.span_efficiency_factor
        if span_efficiency_factor is None
        else float(span_efficiency_factor)
    )
    dimensions = {
        "wing_area_m2": geometry.wing_area_m2,
        "wing_span_m": geometry.wing_span_m,
        "wing_mean_aerodynamic_chord_m": geometry.wing_mac_m,
        "wing_thickness_to_chord_ratio": geometry.wing_thickness_to_chord_ratio,
        "wing_sweep_rad": float(np.deg2rad(geometry.wing_sweep_deg)),
        "fuselage_length_m": geometry.fuselage_length_m,
        "fuselage_width_m": geometry.fuselage_width_m,
        "fuselage_height_m": geometry.fuselage_height_m,
        "horizontal_tail_area_m2": geometry.horizontal_tail_area_m2,
        "horizontal_tail_mean_aerodynamic_chord_m": geometry.horizontal_tail_mac_m,
        "vertical_tail_area_m2": geometry.vertical_tail_area_m2,
        "vertical_tail_mean_aerodynamic_chord_m": geometry.vertical_tail_mac_m,
        "reference_mach": reference_mach,
        "reference_reynolds_per_m": reference_reynolds_per_m,
        "span_efficiency_factor": span_efficiency,
    }
    if geometry.component_buildup:
        components, drag_areas = _turbofan_buildup_terms(geometry)
        return geometry_drag_model_from_dimensions(
            **dimensions,
            wing_wetted_planform_area_m2=geometry.wing_exposed_area_m2,
            tail_interference_factor=TAIL_INTERFERENCE_FACTOR,
            additional_components=components,
            additional_drag_areas_m2=drag_areas,
            allowance_fraction=(
                ROUGHNESS_EXCRESCENCE_FRACTION
                if allowance_fraction is None
                else float(allowance_fraction)
            ),
            allowance_name="roughness_excrescence",
        )
    # Turboprop, piston, and high-speed classes: gross wing area, Q = 1, and
    # one lumped unresolved fraction of the resolved component drag.
    return geometry_drag_model_from_dimensions(
        **dimensions,
        allowance_fraction=(
            geometry.defaults.unresolved_fraction_of_resolved_drag
            if allowance_fraction is None
            else float(allowance_fraction)
        ),
        allowance_name="unresolved",
    )


def _term_methods(geometry: ResolvedDragGeometry) -> dict[str, str]:
    friction = "fully turbulent Cf(Re, M) x FF x Q x Swet / Sref"
    if not geometry.component_buildup:
        return {
            "wing": f"{friction}; Swet = 2 S_gross (1 + 0.25 t/c); Q = 1",
            "fuselage": f"{friction}; fineness-ratio FF; Q = 1",
            "horizontal_tail": f"{friction}; t/c 0.10; Q = 1",
            "vertical_tail": f"{friction}; t/c 0.10; Q = 1",
            "unresolved": (
                f"{geometry.defaults.unresolved_fraction_of_resolved_drag:g} x "
                "resolved sum at reference (lumped class prior)"
            ),
        }
    nacelle = geometry.nacelle
    assert nacelle is not None
    return {
        "wing": (
            f"{friction}; Swet = 2 S_exposed (1 + 0.25 t/c); S_exposed = S minus "
            "trapezoid inside fuselage width; Q = 1.0"
        ),
        "fuselage": f"{friction}; fineness-ratio FF; Q = 1",
        "horizontal_tail": (
            f"{friction}; t/c 0.10; Q = {TAIL_INTERFERENCE_FACTOR} "
            "(Raymer conventional tail, via Scholz Table 13.4)"
        ),
        "vertical_tail": (
            f"{friction}; t/c 0.10; Q = {TAIL_INTERFERENCE_FACTOR} "
            "(Raymer conventional tail, via Scholz Table 13.4)"
        ),
        "nacelles": (
            f"{geometry.engine_count} x Cf(Re_L) (1 + 0.35 D/L) (Raymer) x "
            "2.8 D L (FLOPS) x Q "
            f"{geometry.nacelle_interference_factor:.3g} (Raymer, by mount); "
            f"D {nacelle.max_diameter_m:.3f} m, L {nacelle.length_m:.3f} m, "
            f"{nacelle.duct} duct, {nacelle.tier}"
        ),
        "pylons": (
            f"{PYLON_FRACTION_OF_NACELLE_DRAG:g} x nacelle friction and form drag "
            "(SUAVE)"
        ),
        "fuselage_upsweep": "0.006 x fuselage frontal area / Sref (Kroo)",
        "control_surface_gaps": (
            "0.0002 cos^2(sweep) S_affected / Sref (Kroo); S_affected = "
            "0.3 S_wing + 1.0 S_h + 0.9 S_v (Kroo class values)"
        ),
        "nacelle_base": (
            "engines x 0.5 in x pi D_exit x 0.2 / Sref (Kroo); "
            "D_exit approximated by the fan diameter"
        ),
        "roughness_excrescence": (
            f"{ROUGHNESS_EXCRESCENCE_FRACTION:g} x sum of computed terms at "
            "reference (Kroo; published range 0.03 Raymer to 0.13 Boeing 737)"
        ),
    }


def build_drag_report(aircraft: Aircraft) -> DragBuildReport:
    """Generate deterministic physical drag parameters for an aircraft."""

    geometry = resolve_drag_geometry(aircraft)
    reference = derive_reference_drag_condition(aircraft, geometry)
    altitude_ft = reference.pressure_altitude_ft
    mach = reference.mach
    unit_reynolds = reference.reynolds_per_m
    model = _build_subcritical_geometry_model(geometry, mach, unit_reynolds)
    condition = DragCondition(mach, unit_reynolds)
    cd0 = model._drag_coefficient_scalar(0.0, condition)
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
    breakdown = model.breakdown(0.0, condition=condition)
    methods = _term_methods(geometry)
    terms = tuple(
        DragTerm(name, float(np.asarray(value).item()), methods[name])
        for name, value in (
            *breakdown.component_parasite.items(),
            *breakdown.parasite_increments.items(),
        )
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
        reference_reynolds_per_m=unit_reynolds,
        span_efficiency_factor=geometry.span_efficiency_factor,
        unresolved_fraction_of_resolved_drag=(
            geometry.defaults.unresolved_fraction_of_resolved_drag
        ),
        roughness_excrescence_fraction=(
            ROUGHNESS_EXCRESCENCE_FRACTION if geometry.component_buildup else None
        ),
        wave_drag_model=wave_model,
        wave_drag_source_tier=wave_tier,
        wave_drag_confidence=wave_confidence,
        wave_drag_technology_factor=technology_factor,
        imputed_fields=geometry.imputed_fields,
        parasite_terms=terms,
        nacelle_geometry_tier=(
            None if geometry.nacelle is None else geometry.nacelle.tier
        ),
    )


def _build_subcritical_drag_model(aircraft: Aircraft) -> GeometryDragModel:
    """Reconstruct the subcritical condition-aware model from aircraft data."""

    drag = aircraft.drag
    if drag is None or drag.model != GEOMETRY_DRAG_MODEL_ID:
        raise ValueError(
            f"aircraft {aircraft.icao or '<unknown>'} has no supported "
            "active drag model"
        )
    geometry = resolve_drag_geometry(aircraft)
    if geometry.component_buildup:
        allowance_field = "roughness_excrescence_fraction"
        allowance = drag.roughness_excrescence_fraction
    else:
        allowance_field = "unresolved_fraction_of_resolved_drag"
        allowance = drag.unresolved_fraction_of_resolved_drag
    required = {
        "reference_mach": drag.reference_mach,
        "reference_reynolds_per_m": (drag.reference_reynolds_per_m),
        "span_efficiency_factor": drag.span_efficiency_factor,
        allowance_field: allowance,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("active drag model is missing: " + ", ".join(missing))

    model = _build_subcritical_geometry_model(
        geometry,
        float(drag.reference_mach),
        float(drag.reference_reynolds_per_m),
        span_efficiency_factor=float(drag.span_efficiency_factor),
        allowance_fraction=float(allowance),
    )
    deployed = deployed_drag_from_aircraft(
        aircraft,
        model.reference_area_m2,
        geometry.wing_span_m,
        geometry.wing_thickness_to_chord_ratio,
        geometry.wing_sweep_deg,
        geometry.fuselage_width_m,
    )
    return replace(model, deployed=deployed)


def build_drag_model(
    aircraft: Aircraft,
) -> GeometryDragModel | TransonicGeometryDragModel:
    """Build the full drag model of an aircraft from its geometry.

    The zero-lift drag changes with the Mach and Reynolds number, and aircraft
    with a wave drag model also have wave drag above the critical Mach number.
    Flap, slat and landing gear drag apply when the condition has them deployed
    (see :mod:`aerox.models.drag.deployed`).
    For a constant ``CD = cd0 + k CL²``, use :func:`build_parabolic_polar`.
    """

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


def build_parabolic_polar(aircraft: Aircraft) -> ParabolicDragPolar:
    """Reduce the drag model to a two-term parabolic polar, ``CD = cd0 + k CL²``.

    ``cd0`` and ``k`` are taken from the drag model without wave drag, at the
    reference condition stored in the record (``drag.reference_mach`` and
    ``drag.reference_reynolds_per_m``). The polar does not change with the
    Mach or Reynolds number. Use :func:`build_drag_model` when these effects
    matter.
    """

    drag = aircraft.drag
    if drag is None or drag.reference_mach is None:
        raise ValueError("active drag model has no reference Mach")
    if drag.reference_reynolds_per_m is None:
        raise ValueError("active drag model has no reference unit Reynolds number")
    # The low-order adapter is intentionally a subcritical projection. Wave
    # drag is Mach- and CL-dependent and cannot be represented by one CD0/k.
    model = _build_subcritical_drag_model(aircraft)
    condition = DragCondition(
        drag.reference_mach,
        drag.reference_reynolds_per_m,
    )
    return project_parabolic_polar(model, condition)
