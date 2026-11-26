"""
Aircraft Data Classes

This module provides data classes to represent aircraft configuration from YAML files.
These classes map to the structure defined in aerox/data/aircraft/yml/*.yml files.

Usage:
    from aerox.aircraft import Aircraft, load_aircraft, get_available_aircraft

    # Get all available aircraft ICAO codes
    icao_codes = get_available_aircraft()
    print(f"Total aircraft: {len(icao_codes)}")

    # Load a specific aircraft from YAML
    aircraft = load_aircraft("A320")

    # Access properties
    print(aircraft.mtow)  # 78000
    print(aircraft.wing.span)  # 34.1
    print(aircraft.kinematic.cruise_mach)  # 0.79
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

AIRCRAFT_DATA_DIR = Path(__file__).parent / "data" / "aircraft" / "yml"
ENGINE_TYPE_DIRS = ("turbofan", "turboprop", "piston")


@dataclass
class Pax:
    """Passenger capacity configuration."""

    max: Optional[int] = None
    low: Optional[int] = None
    high: Optional[int] = None


@dataclass
class Fuselage:
    """Fuselage dimensions."""

    length: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None


@dataclass
class Wing:
    """Wing configuration."""

    area: Optional[float] = None
    span: Optional[float] = None
    mac: Optional[float] = None
    sweep: Optional[float] = None
    t_c: Optional[float] = field(default=None, metadata={"yaml_key": "t/c"})
    root_chord: Optional[float] = None
    tip_chord: Optional[float] = None
    taper_ratio: Optional[float] = None
    aspect_ratio: Optional[float] = None
    dihedral: Optional[float] = None


@dataclass
class Certification:
    """Aircraft-model certification identity and source reference."""

    model: Optional[str] = None
    authority: Optional[str] = None
    type_certificate: Optional[str] = None
    issue: Optional[str] = None
    issue_date: Optional[str] = None
    type_design_definition: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class CertifiedMassLimits:
    """Maximum certified mass limits in kilograms for a named model."""

    mtow: Optional[float] = None
    mlw: Optional[float] = None
    mzfw: Optional[float] = None
    minimum_weight: Optional[float] = None
    applicability: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class ReferenceGeometry:
    """Published reference dimensions in metres not covered by wing/fuselage."""

    overall_height: Optional[float] = None
    modified_wing_span: Optional[float] = None
    modified_wing_span_applicability: Optional[str] = None
    engine_axis_spacing: Optional[float] = None
    main_landing_gear_track: Optional[float] = None
    nose_to_main_landing_gear: Optional[float] = None
    datum_forward_of_nose: Optional[float] = None
    source_id: Optional[str] = None


@dataclass
class Tail:
    """Tail geometry in metres, square metres, and degrees where applicable."""

    area: Optional[float] = None
    span: Optional[float] = None
    height: Optional[float] = None
    mac: Optional[float] = None
    root_chord: Optional[float] = None
    tip_chord: Optional[float] = None
    taper_ratio: Optional[float] = None
    sweep: Optional[float] = None
    dihedral: Optional[float] = None
    arm: Optional[float] = None
    elevator_area_ratio: Optional[float] = None
    rudder_area_ratio: Optional[float] = None


@dataclass
class ControlSurface:
    """Control-surface geometry; span stations and ratios are nondimensional."""

    span_limits: Optional[List[float]] = None
    chord_ratio: Optional[float] = None
    deflection_limit: Optional[float] = None
    status: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class MassProperties:
    """Reference CG and rigid-body inertias for higher-order simulation.

    CG locations are fractions of mean aerodynamic chord. Inertias use kg m².
    A reference CG is not an operational CG measurement or a certified range.
    """

    reference_cg_mac: Optional[float] = None
    cg_forward_mac: Optional[float] = None
    cg_aft_mac: Optional[float] = None
    reference_mass: Optional[float] = None
    ixx: Optional[float] = None
    iyy: Optional[float] = None
    izz: Optional[float] = None
    ixz: Optional[float] = None
    status: Optional[str] = None
    source_id: Optional[str] = None


FLAP_TYPES = (
    "plain",
    "split",
    "single-slotted",
    "double-slotted",
    "triple-slotted",
    "fowler",
    "double-slotted-fowler",
    "mixed-single-double-slotted",
)


#: Flap fields of records before the high-lift schema. Their definitions
#: differ from the new fields, so they are not translated.
LEGACY_FLAP_KEYS = frozenset({"bf/b", "lambda_f", "cf/c", "Sf/S"})


@dataclass(frozen=True)
class FlapSetting:
    """Representative surface angles, in degrees, at a named setting."""

    flap_angle_deg: float
    slat_angle_deg: float | None = None
    slats_extended: bool | None = None

    def __post_init__(self) -> None:
        for name, angle in (
            ("flap_angle_deg", self.flap_angle_deg),
            ("slat_angle_deg", self.slat_angle_deg),
        ):
            if name == "slat_angle_deg" and angle is None:
                continue
            if isinstance(angle, bool) or not isinstance(angle, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not 0 <= angle <= 60:
                raise ValueError(f"{name} must lie between 0 and 60 degrees")
        if self.slats_extended is not None:
            if not isinstance(self.slats_extended, bool):
                raise ValueError("slats_extended must be a boolean")
            if self.slat_angle_deg is not None:
                raise ValueError("give slats_extended only when the angle is unknown")


@dataclass(kw_only=True)
class Flaps:
    """High-lift geometry and documented operating settings."""

    type: Optional[str] = None
    area: Optional[float] = None
    eta_inboard: Optional[float] = None
    eta_outboard: Optional[float] = None
    flapped_area_ratio: Optional[float] = None
    chord_ratio: Optional[float] = None
    slatted_area_ratio: Optional[float] = None
    settings: Optional[Dict[str, FlapSetting]] = None

    def __post_init__(self) -> None:
        if self.type is not None and self.type not in FLAP_TYPES:
            raise ValueError(f"unsupported flap type: {self.type!r}")
        for name in (
            "area",
            "eta_inboard",
            "eta_outboard",
            "flapped_area_ratio",
            "chord_ratio",
            "slatted_area_ratio",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0 or (name != "area" and value > 1):
                raise ValueError(f"{name} is outside its physical bounds")
        if (
            self.eta_inboard is not None
            and self.eta_outboard is not None
            and self.eta_inboard >= self.eta_outboard
        ):
            raise ValueError("eta_inboard must be smaller than eta_outboard")

    def angles_rad(self, setting: str) -> tuple[float, float | None]:
        """Return flap and slat angles for a named setting, in radians.

        The slat angle is None when the record gives no angle. Setting names
        are case sensitive. An unknown setting raises KeyError.
        """

        entry = (self.settings or {})[setting]
        slat = entry.slat_angle_deg
        return math.radians(entry.flap_angle_deg), (
            None if slat is None else math.radians(slat)
        )


def _load_flaps(data: Optional[Dict[str, Any]]) -> Flaps | None:
    if data is None:
        return None
    values = dict(data)
    legacy = sorted(LEGACY_FLAP_KEYS & values.keys())
    if legacy:
        raise ValueError(
            f"flaps has fields of the old schema: {', '.join(legacy)}. "
            "Use eta_inboard, eta_outboard, flapped_area_ratio and chord_ratio "
            "(see docs/aircraft-data.md)"
        )
    settings = values.pop("settings", None)
    if settings is None:
        return Flaps(**values)
    if not isinstance(settings, dict) or not settings:
        raise ValueError("settings must be a non-empty mapping")
    entries = {}
    for name, setting in settings.items():
        if not isinstance(name, str) or not name.strip() or "." in name:
            raise ValueError("setting names must be non-empty strings without dots")
        if not isinstance(setting, dict):
            raise ValueError(f"setting {name!r} must be a mapping")
        entries[name] = FlapSetting(**setting)
    return Flaps(**values, settings=entries)


@dataclass
class AircraftSubtype:
    """Queryable ICAO EEDB engine identifiers for one certified subtype."""

    subtype: str
    engines: List[str] = field(default_factory=list)


@dataclass
class Engine:
    """Engine configuration."""

    type: Optional[str] = None
    mount: Optional[str] = None
    number: Optional[int] = None
    default: Optional[str] = None
    options: List[str] = field(default_factory=list)


@dataclass
class DragModelConfig:
    """Independent inputs and provenance for the active clean drag model."""

    model: Optional[str] = None
    configuration: Optional[str] = None
    geometry_tier: Optional[str] = None
    confidence: Optional[str] = None
    reference_altitude_ft: Optional[float] = None
    reference_mach: Optional[float] = None
    reference_reynolds_per_m: Optional[float] = None
    span_efficiency_factor: Optional[float] = None
    unresolved_fraction_of_resolved_drag: Optional[float] = None
    roughness_excrescence_fraction: Optional[float] = None
    wave_drag_model: Optional[str] = None
    wave_drag_source_tier: Optional[str] = None
    wave_drag_confidence: Optional[str] = None
    wave_drag_technology_factor: Optional[float] = None


@dataclass
class KinematicParameters:
    """Kinematic performance parameters loaded from an aircraft YAML record."""

    # Takeoff
    takeoff_distance_m: Optional[int] = None  # Changed to int (e.g., 2190 m)
    takeoff_v2: Optional[int] = None  # Changed to int (e.g., 145 knots)

    # Climb to FL50
    climb_to_fl50_ias: Optional[int] = None  # Changed to int (e.g., 175 knots)
    climb_to_fl50_vs: Optional[int] = None  # Changed to int (e.g., 2500 ft/min)

    # Climb to FL150
    climb_to_fl150_ias: Optional[int] = None  # Changed to int (e.g., 290 knots)
    climb_to_fl150_vs: Optional[int] = None  # Changed to int (e.g., 2000 ft/min)

    # Climb to FL240
    climb_to_fl240_ias: Optional[int] = None  # Changed to int (e.g., 290 knots)
    climb_to_fl240_vs: Optional[int] = None  # Changed to int (e.g., 1400 ft/min)

    # Mach climb
    mach_climb_mach: Optional[float] = None  # Keep as float (e.g., 0.78 Mach)
    mach_climb_vs: Optional[int] = None  # Changed to int (e.g., 1000 ft/min)

    # Cruise
    cruise_mach: Optional[float] = None  # Keep as float (e.g., 0.79 Mach)
    cruise_range_nm: Optional[int] = None  # Changed to int (e.g., 2700 nm)
    cruise_tas: Optional[int] = None  # Changed to int (e.g., 450 knots)

    # Descent to FL240
    descent_to_fl240_mach: Optional[float] = None  # Keep as float (e.g., 0.78 Mach)
    descent_to_fl240_vs: Optional[int] = None  # Changed to int (e.g., 1000 ft/min)

    # Descent to FL100
    descent_to_fl100_ias: Optional[int] = None  # Changed to int (e.g., 290 knots)
    descent_to_fl100_vs: Optional[int] = None  # Changed to int (e.g., 3500 ft/min)

    # Approach
    approach_ias: Optional[int] = None  # Changed to int (e.g., 250 knots)
    approach_vs: Optional[int] = None  # Changed to int (e.g., 1500 ft/min)

    # Landing
    landing_distance_m: Optional[int] = None  # Changed to int (e.g., 1440 m)
    landing_ias: Optional[int] = None  # Changed to int (e.g., 137 knots)


@dataclass
class Aircraft:
    """
    Aircraft data class representing the complete aircraft configuration.

    This class maps to the YAML structure in aerox/data/aircraft/yml/*.yml files.

    Attributes:
        aircraft: Full aircraft name (e.g., "Airbus A320")
        icao: ICAO type code (e.g., "A320")
        iata: IATA type code (e.g., "320 / 32S")
        mtow: Maximum takeoff weight (kg)
        mlw: Maximum landing weight (kg)
        oew: Operating empty weight (kg)
        mfc: Maximum fuel capacity (L)
        vmo: Maximum operating speed (KIAS)
        mmo: Maximum operating Mach number
        ceiling: Service ceiling (ft)
        icao_wtc: ICAO wake turbulence category
        eu_wtc: European wake turbulence category
        faa_wtc: FAA wake turbulence category
        parking_area_m2: Parking area (m²)
        pax: Passenger configuration
        fuselage: Fuselage dimensions
        wing: Wing configuration
        certification: Aircraft-model certification record
        certified_mass_limits: Model-level certified mass limits
        geometry: Additional published reference dimensions
        horizontal_tail: Horizontal-tail geometry
        vertical_tail: Vertical-tail geometry
        aileron: Aileron geometry and limits
        rudder: Rudder geometry and limits
        mass_properties: CG and inertia data for higher-order simulation
        data_provenance: Per-field source and confidence metadata
        flaps: Flap configuration
        engine: Engine configuration
        drag: Independent inputs for the active physical drag model
        kinematic: KinematicParameters performance data
    """

    # Basic identification
    aircraft: Optional[str] = None
    icao: Optional[str] = None
    iata: Optional[str] = None
    subtypes: Optional[List[AircraftSubtype]] = None

    # Weight limits (kg)
    mtow: Optional[float] = None
    mlw: Optional[float] = None
    mzfw: Optional[float] = None
    minimum_weight: Optional[float] = None
    oew: Optional[float] = None
    mfc: Optional[float] = None

    # Speed limits
    vmo: Optional[float] = None
    mmo: Optional[float] = None

    # Altitude limit
    ceiling: Optional[float] = None

    # Wake turbulence categories
    icao_wtc: Optional[str] = None
    eu_wtc: Optional[str] = None
    faa_wtc: Optional[str] = None

    # Parking
    parking_area_m2: Optional[float] = None

    # Complex structures
    pax: Optional[Pax] = None
    fuselage: Optional[Fuselage] = None
    wing: Optional[Wing] = None
    certification: Optional[Certification] = None
    certified_mass_limits: Optional[CertifiedMassLimits] = None
    geometry: Optional[ReferenceGeometry] = None
    horizontal_tail: Optional[Tail] = None
    vertical_tail: Optional[Tail] = None
    aileron: Optional[ControlSurface] = None
    rudder: Optional[ControlSurface] = None
    mass_properties: Optional[MassProperties] = None
    flaps: Optional[Flaps] = None
    engine: Optional[Engine] = None
    drag: Optional[DragModelConfig] = None
    kinematic: Optional[KinematicParameters] = None
    data_provenance: Dict[str, Any] = field(default_factory=dict)

    def engines_for_subtype(self, subtype: str) -> List[str]:
        """Return ICAO EEDB query identifiers for an exact subtype."""

        if self.subtypes is None:
            return []
        for item in self.subtypes:
            if item.subtype == subtype:
                return list(item.engines)
        return []


def _load_nested_dataclass(data: Optional[Dict], cls):
    """Helper to load nested dataclass from dictionary."""
    if data is None:
        return None

    # Handle special field name mappings
    field_map = {}
    if hasattr(cls, "__dataclass_fields__"):
        for field_name, field_obj in cls.__dataclass_fields__.items():
            if "yaml_key" in field_obj.metadata:
                yaml_key = field_obj.metadata["yaml_key"]
                field_map[yaml_key] = field_name

    # Convert keys
    converted_data = {}
    for key, value in data.items():
        field_name = field_map.get(key, key.replace("-", "_").replace("/", "_"))
        if cls is Certification and field_name == "issue" and value is not None:
            value = str(value)
        if (
            cls is Certification
            and field_name == "issue_date"
            and value is not None
            and not isinstance(value, str)
            and hasattr(value, "isoformat")
        ):
            value = value.isoformat()
        converted_data[field_name] = value

    return cls(**converted_data)


def _load_subtypes(data: Any) -> Optional[List[AircraftSubtype]]:
    """Load the optional root-level subtype-to-engine mapping."""

    if data is None:
        return None
    result = []
    for row in data:
        subtype, engines = next(iter(row.items()))
        if isinstance(engines, str):
            engines = [engines]
        result.append(AircraftSubtype(subtype=subtype, engines=list(engines)))
    return result


def _load_kinematic_parameters(
    data: Optional[Dict],
) -> Optional[KinematicParameters]:
    """Load kinematic data structure with flat field names."""
    if data is None:
        return None

    kinematic_dict = {}
    for field_name, value in data.items():
        # Convert hyphens to underscores
        field_name_clean = field_name.replace("-", "_")
        kinematic_dict[field_name_clean] = value

    return KinematicParameters(**kinematic_dict)


def parse_source_notes(text: str) -> Dict[str, Any]:
    """Parse the compact, human-readable source note sidecar."""

    result: Dict[str, Any] = {"header": {}, "fields": {}}
    current: Optional[Dict[str, str]] = None
    for line in text.splitlines():
        if line.startswith("Aircraft: "):
            result["header"]["aircraft"] = line.removeprefix("Aircraft: ")
        elif line.startswith("ICAO: "):
            result["header"]["icao"] = line.removeprefix("ICAO: ")
        elif line.startswith("Last reviewed: "):
            result["header"]["last_reviewed"] = line.removeprefix("Last reviewed: ")
        elif line.startswith("FIELD "):
            current = {}
            result["fields"][line.removeprefix("FIELD ")] = current
        elif line == "END FIELD":
            current = None
        elif current is not None and line.startswith("  ") and ": " in line:
            key, value = line.strip().split(": ", 1)
            current[key.lower().replace(" ", "_")] = value
    return result


def _source_ids(provenance: Dict[str, Any], path: str) -> List[str]:
    value = provenance.get("fields", {}).get(path, {}).get("source")
    return (
        []
        if not value or value == "-"
        else [part.strip() for part in value.split(" | ")]
    )


# Record keys renamed after earlier releases: {old key: current key} per group.
LEGACY_KEYS: Dict[str, Dict[str, str]] = {
    "drag": {"reference_unit_reynolds_number_per_m": "reference_reynolds_per_m"},
}


def upgrade_legacy_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of an aircraft record with renamed keys updated.

    A value stored under an old key is kept unless the current key also has a
    value.
    """

    upgraded = dict(data)
    for group, renames in LEGACY_KEYS.items():
        section = upgraded.get(group)
        if not isinstance(section, dict):
            continue
        section = dict(section)
        for old, new in renames.items():
            if old in section:
                value = section.pop(old)
                if section.get(new) is None:
                    section[new] = value
        upgraded[group] = section
    return upgraded


def upgrade_legacy_source_notes(text: str) -> str:
    """Return source-note text with renamed field paths updated."""

    for group, renames in LEGACY_KEYS.items():
        for old, new in renames.items():
            text = text.replace(f"{group}.{old}", f"{group}.{new}")
    return text


def load_aircraft_from_dict(
    data: Dict[str, Any], provenance: Optional[Dict[str, Any]] = None
) -> Aircraft:
    """
    Load an Aircraft object from a dictionary (typically from YAML).

    Args:
        data: Dictionary containing aircraft configuration

    Returns:
        Aircraft: Populated aircraft data class
    """
    data = upgrade_legacy_keys(data)
    provenance = provenance or {"header": {}, "fields": {}}
    fields = provenance.get("fields") or {}
    for group, renames in LEGACY_KEYS.items():
        for old, new in renames.items():
            if f"{group}.{old}" in fields and f"{group}.{new}" not in fields:
                fields = dict(fields)
                fields[f"{group}.{new}"] = fields.pop(f"{group}.{old}")
    provenance = {**provenance, "fields": fields}
    pax = _load_nested_dataclass(data.get("pax"), Pax)
    fuselage = _load_nested_dataclass(data.get("fuselage"), Fuselage)
    wing = _load_nested_dataclass(data.get("wing"), Wing)
    certification = _load_nested_dataclass(data.get("certification"), Certification)
    certified_mass_limits = _load_nested_dataclass(
        data.get("certified_mass_limits"), CertifiedMassLimits
    )
    if certified_mass_limits is None:
        mtow_sources = _source_ids(provenance, "mtow")
        certified_mass_limits = CertifiedMassLimits(
            mtow=data.get("mtow"),
            mlw=data.get("mlw"),
            mzfw=data.get("mzfw"),
            minimum_weight=data.get("minimum_weight"),
            applicability=provenance.get("fields", {})
            .get("mtow", {})
            .get("applicability"),
            source_id=mtow_sources[0] if mtow_sources else None,
        )
    geometry = _load_nested_dataclass(data.get("geometry"), ReferenceGeometry)
    horizontal_tail = _load_nested_dataclass(data.get("horizontal_tail"), Tail)
    vertical_tail = _load_nested_dataclass(data.get("vertical_tail"), Tail)
    aileron = _load_nested_dataclass(data.get("aileron"), ControlSurface)
    rudder = _load_nested_dataclass(data.get("rudder"), ControlSurface)
    mass_properties = _load_nested_dataclass(
        data.get("mass_properties"), MassProperties
    )
    flaps = _load_flaps(data.get("flaps"))
    engine = _load_nested_dataclass(data.get("engine"), Engine)
    drag = _load_nested_dataclass(data.get("drag"), DragModelConfig)
    kinematic = _load_kinematic_parameters(data.get("kinematic"))
    if certification is not None and certification.source_id is None:
        certification_sources = _source_ids(
            provenance, "certification.type_certificate"
        )
        certification.source_id = (
            certification_sources[0] if certification_sources else None
        )
    field_sources = dict(data.get("data_provenance", {}).get("field_sources", {}))
    field_sources.update(
        {
            path: _source_ids(provenance, path)
            for path in provenance.get("fields", {})
            if _source_ids(provenance, path)
        }
    )

    # Extract top-level fields
    aircraft = Aircraft(
        aircraft=data.get("aircraft"),
        icao=data.get("icao"),
        iata=data.get("iata"),
        subtypes=_load_subtypes(data.get("subtypes")),
        mtow=data.get("mtow"),
        mlw=data.get("mlw"),
        mzfw=data.get("mzfw"),
        minimum_weight=data.get("minimum_weight"),
        oew=data.get("oew"),
        mfc=data.get("mfc"),
        vmo=data.get("vmo"),
        mmo=data.get("mmo"),
        ceiling=data.get("ceiling"),
        icao_wtc=data.get("icao_wtc"),
        eu_wtc=data.get("eu_wtc"),
        faa_wtc=data.get("faa_wtc"),
        parking_area_m2=data.get("parking_area_m2"),
        pax=pax,
        fuselage=fuselage,
        wing=wing,
        certification=certification,
        certified_mass_limits=certified_mass_limits,
        geometry=geometry,
        horizontal_tail=horizontal_tail,
        vertical_tail=vertical_tail,
        aileron=aileron,
        rudder=rudder,
        mass_properties=mass_properties,
        flaps=flaps,
        engine=engine,
        drag=drag,
        kinematic=kinematic,
        data_provenance={
            **data.get("data_provenance", {}),
            "field_sources": field_sources,
            "sidecar": provenance,
        },
    )

    return aircraft


def _default_data_dir() -> Path:
    return AIRCRAFT_DATA_DIR


def _iter_aircraft_yaml_files(data_dir: Path):
    for engine_type in ENGINE_TYPE_DIRS:
        folder = data_dir / engine_type
        if folder.exists():
            yield from sorted(folder.glob("*.yml"))


def _find_aircraft_yaml(icao: str, data_dir: Path) -> Path:
    filename = f"{icao.lower()}.yml"
    direct_file = data_dir / filename
    if direct_file.exists():
        return direct_file

    for engine_type in ENGINE_TYPE_DIRS:
        yaml_file = data_dir / engine_type / filename
        if yaml_file.exists():
            return yaml_file

    raise FileNotFoundError(f"Aircraft file not found for ICAO code: {icao}")


def load_aircraft(icao: str, data_dir: Optional[Path] = None) -> Aircraft:
    """
    Load an aircraft from YAML file by ICAO code.

    Args:
        icao: ICAO aircraft type code (e.g., "A320", "B738")
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        Aircraft: Loaded aircraft data class

    Raises:
        FileNotFoundError: If the YAML file doesn't exist

    Example:
        >>> aircraft = load_aircraft("A320")
        >>> print(aircraft.mtow)
        78000
        >>> print(aircraft.wing.span)
        34.1
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    yaml_file = _find_aircraft_yaml(icao, Path(data_dir))

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    provenance = None
    sidecar = (
        Path(data_dir).parent
        / "sources"
        / yaml_file.parent.name
        / f"{yaml_file.stem}.txt"
    )
    if not sidecar.exists():
        # Compatibility for custom data directories created before the folder
        # was renamed. Bundled data has only the canonical ``sources`` folder.
        legacy_sidecar = (
            Path(data_dir).parent
            / "provenance"
            / yaml_file.parent.name
            / f"{yaml_file.stem}.txt"
        )
        if legacy_sidecar.exists():
            sidecar = legacy_sidecar
    if sidecar.exists():
        provenance = parse_source_notes(sidecar.read_text(encoding="utf-8"))

    return load_aircraft_from_dict(data, provenance)


def get_available_aircraft(data_dir: Optional[Path] = None) -> List[str]:
    """
    Get a list of all available aircraft ICAO codes.

    Args:
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        List[str]: List of ICAO codes (uppercase) for all available aircraft

    Example:
        >>> icao_codes = get_available_aircraft()
        >>> len(icao_codes) > 150
        True
        >>> print("A320" in icao_codes)
        True
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    icao_codes = []
    for yaml_file in _iter_aircraft_yaml_files(Path(data_dir)):
        icao = yaml_file.stem.upper()
        icao_codes.append(icao)

    return sorted(icao_codes)


if __name__ == "__main__":
    # Test the data classes
    print("=== Aircraft Data Class Test ===\n")

    # Test loading a specific aircraft
    try:
        a320 = load_aircraft("A320")
        print(f"Aircraft: {a320.aircraft}")
        print(f"ICAO: {a320.icao}")
        print(f"MTOW: {a320.mtow} kg")
        if a320.wing:
            print(f"Wing span: {a320.wing.span} m")
            print(f"Wing area: {a320.wing.area} m²")
        if a320.engine:
            print(f"Engine type: {a320.engine.type}")
            print(f"Engine default: {a320.engine.default}")
        if a320.kinematic:
            print(f"Cruise Mach: {a320.kinematic.cruise_mach}")
            print(f"Cruise range: {a320.kinematic.cruise_range_nm} nm")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}\n")

    # Test getting all available aircraft
    try:
        icao_codes = get_available_aircraft()
        print(f"Found {len(icao_codes)} aircraft types:")
        for icao in icao_codes[:10]:
            print(f"  {icao}")
        print(f"  ... and {len(icao_codes) - 10} more")
    except Exception as e:
        print(f"Error getting available aircraft: {e}")
