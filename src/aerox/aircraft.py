"""
Aircraft Data Classes

This module provides data classes to represent aircraft configuration from YAML files.
These classes map to the structure defined in aerox/data/aircraft/yml/*.yml files.

Usage:
    from aerox.aircraft import Aircraft, load_aircraft, get_available_aircraft

    # Get all available aircraft ICAO codes
    icao_codes = get_available_aircraft()
    print(f"Total aircraft: {len(icao_codes)}")  # 139

    # Load a specific aircraft from YAML
    aircraft = load_aircraft("A320")

    # Access properties
    print(aircraft.mtow)  # 73900
    print(aircraft.wing.span)  # 34.1
    print(aircraft.kinematic.cruise_mach)  # 0.79
"""

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

    area: Optional[int] = None  # Changed to int (e.g., 124 m²)
    span: Optional[float] = None
    mac: Optional[float] = None
    sweep: Optional[int] = None  # Changed to int (e.g., 25 degrees)
    t_c: Optional[float] = field(default=None, metadata={"yaml_key": "t/c"})


@dataclass
class Flaps:
    """Flap configuration."""

    type: Optional[str] = None
    area: Optional[float] = None
    bf_b: Optional[float] = field(default=None, metadata={"yaml_key": "bf/b"})
    lambda_f: Optional[float] = None
    cf_c: Optional[float] = field(default=None, metadata={"yaml_key": "cf/c"})
    Sf_S: Optional[float] = field(default=None, metadata={"yaml_key": "Sf/S"})


@dataclass
class Engine:
    """Engine configuration."""

    type: Optional[str] = None
    mount: Optional[str] = None
    number: Optional[int] = None
    default: Optional[str] = None
    options: List[str] = field(default_factory=list)


@dataclass
class Drag:
    """Drag coefficients."""

    cd0: Optional[float] = None
    k: Optional[float] = None
    e: Optional[float] = None
    gears: Optional[float] = None


@dataclass
class Kinematic:
    """Kinematic performance data with flat field names."""

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
        mfc: Maximum fuel capacity (kg)
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
        flaps: Flap configuration
        engine: Engine configuration
        drag: Drag coefficients
        kinematic: Kinematic performance data
    """

    # Basic identification
    aircraft: Optional[str] = None
    icao: Optional[str] = None
    iata: Optional[str] = None

    # Weight limits (kg)
    mtow: Optional[int] = None
    mlw: Optional[int] = None
    oew: Optional[int] = None
    mfc: Optional[int] = None

    # Speed limits
    vmo: Optional[int] = None
    mmo: Optional[float] = None

    # Altitude limit
    ceiling: Optional[int] = None

    # Wake turbulence categories
    icao_wtc: Optional[str] = None
    eu_wtc: Optional[str] = None
    faa_wtc: Optional[str] = None

    # Parking
    parking_area_m2: Optional[int] = None  # Changed to int (e.g., 1519 m²)

    # Complex structures
    pax: Optional[Pax] = None
    fuselage: Optional[Fuselage] = None
    wing: Optional[Wing] = None
    flaps: Optional[Flaps] = None
    engine: Optional[Engine] = None
    drag: Optional[Drag] = None
    kinematic: Optional[Kinematic] = None


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
        converted_data[field_name] = value

    return cls(**converted_data)


def _load_kinematic(data: Optional[Dict]) -> Optional[Kinematic]:
    """Load kinematic data structure with flat field names."""
    if data is None:
        return None

    kinematic_dict = {}
    for field_name, value in data.items():
        # Convert hyphens to underscores
        field_name_clean = field_name.replace("-", "_")
        kinematic_dict[field_name_clean] = value

    return Kinematic(**kinematic_dict)


def load_aircraft_from_dict(data: Dict[str, Any]) -> Aircraft:
    """
    Load an Aircraft object from a dictionary (typically from YAML).

    Args:
        data: Dictionary containing aircraft configuration

    Returns:
        Aircraft: Populated aircraft data class
    """
    # Extract nested structures
    pax = _load_nested_dataclass(data.get("pax"), Pax)
    fuselage = _load_nested_dataclass(data.get("fuselage"), Fuselage)
    wing = _load_nested_dataclass(data.get("wing"), Wing)
    flaps = _load_nested_dataclass(data.get("flaps"), Flaps)
    engine = _load_nested_dataclass(data.get("engine"), Engine)
    drag = _load_nested_dataclass(data.get("drag"), Drag)
    kinematic = _load_kinematic(data.get("kinematic"))

    # Extract top-level fields
    aircraft = Aircraft(
        aircraft=data.get("aircraft"),
        icao=data.get("icao"),
        iata=data.get("iata"),
        mtow=data.get("mtow"),
        mlw=data.get("mlw"),
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
        flaps=flaps,
        engine=engine,
        drag=drag,
        kinematic=kinematic,
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
        73900
        >>> print(aircraft.wing.span)
        34.1
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    yaml_file = _find_aircraft_yaml(icao, Path(data_dir))

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return load_aircraft_from_dict(data)


def get_available_aircraft(data_dir: Optional[Path] = None) -> List[str]:
    """
    Get a list of all available aircraft ICAO codes.

    Args:
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        List[str]: List of ICAO codes (uppercase) for all available aircraft

    Example:
        >>> icao_codes = get_available_aircraft()
        >>> print(len(icao_codes))
        139
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
