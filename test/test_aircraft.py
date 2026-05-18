from pathlib import Path

import pytest

from aerox import Aircraft, get_available_aircraft, load_aircraft
from aerox.aircraft import (
    Drag,
    Engine,
    Fuselage,
    Kinematic,
    Pax,
    Wing,
    load_aircraft_from_dict,
)


def test_load_aircraft_from_dict_with_nested_sections():
    aircraft = load_aircraft_from_dict(
        {
            "aircraft": "ATR 72",
            "icao": "AT72",
            "mtow": 21500,
            "pax": {"max": 72},
            "fuselage": {"length": 27.2},
            "wing": {"span": 27.1, "t/c": 0.12},
            "engine": {
                "type": "turboprop",
                "mount": "front wing leading mounted",
                "number": 2,
                "default": "PWC PW124B",
                "options": ["PWC PW124B"],
            },
            "drag": {"cd0": 0.02, "k": 0.04, "e": 0.8, "gears": 0.01},
            "kinematic": {"takeoff_distance_m": 1500, "cruise_range_nm": 1500},
        }
    )

    assert isinstance(aircraft, Aircraft)
    assert isinstance(aircraft.pax, Pax)
    assert isinstance(aircraft.fuselage, Fuselage)
    assert isinstance(aircraft.wing, Wing)
    assert isinstance(aircraft.engine, Engine)
    assert isinstance(aircraft.drag, Drag)
    assert isinstance(aircraft.kinematic, Kinematic)
    assert aircraft.icao == "AT72"
    assert aircraft.wing.t_c == 0.12
    assert aircraft.kinematic.takeoff_distance_m == 1500


def test_get_available_aircraft_finds_nested_category_files():
    codes = get_available_aircraft()

    assert codes
    assert codes == sorted(codes)
    assert "AT72" in codes
    assert all(code == code.upper() for code in codes)


def test_load_aircraft_finds_turboprop_yaml_by_icao():
    aircraft = load_aircraft("at72")

    assert aircraft.aircraft == "ATR 72"
    assert aircraft.icao == "AT72"
    assert aircraft.mtow == 21500
    assert aircraft.engine is not None
    assert aircraft.engine.type == "turboprop"
    assert aircraft.kinematic is not None
    assert aircraft.kinematic.cruise_range_nm == 1500


def test_load_aircraft_raises_for_unknown_icao():
    with pytest.raises(FileNotFoundError):
        load_aircraft("NOPE")


def test_custom_data_dir_can_use_flat_or_nested_layout(tmp_path: Path):
    flat_file = tmp_path / "test.yml"
    flat_file.write_text("aircraft: Test Aircraft\nicao: TEST\nmtow: 123\n")

    nested_dir = tmp_path / "turboprop"
    nested_dir.mkdir()
    nested_file = nested_dir / "nest.yml"
    nested_file.write_text("aircraft: Nested Aircraft\nicao: NEST\nmtow: 456\n")

    assert load_aircraft("TEST", data_dir=tmp_path).mtow == 123
    assert load_aircraft("NEST", data_dir=tmp_path).mtow == 456
    assert get_available_aircraft(data_dir=tmp_path) == ["NEST"]
