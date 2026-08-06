from pathlib import Path

import pytest
import yaml

from aerox import Aircraft, get_available_aircraft, load_aircraft
from aerox.aircraft import (
    Certification,
    CertifiedMassLimits,
    ControlSurface,
    Drag,
    Engine,
    Fuselage,
    Kinematic,
    MassProperties,
    AircraftSubtype,
    Pax,
    ReferenceGeometry,
    Tail,
    Wing,
    load_aircraft_from_dict,
)


def test_load_aircraft_from_dict_with_nested_sections():
    aircraft = load_aircraft_from_dict(
        {
            "aircraft": "ATR 72",
            "icao": "AT72",
            "mtow": 21500,
            "subtypes": [
                {"subtype": "ATR 72-212A", "engines": ["PW127M"]}
            ],
            "pax": {"max": 72},
            "fuselage": {"length": 27.2},
            "wing": {"span": 27.1, "t/c": 0.12},
            "horizontal_tail": {"span": 7.5, "area": 12.4},
            "vertical_tail": {"height": 3.2, "area": 8.1},
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
    assert isinstance(aircraft.horizontal_tail, Tail)
    assert isinstance(aircraft.vertical_tail, Tail)
    assert isinstance(aircraft.engine, Engine)
    assert isinstance(aircraft.subtypes[0], AircraftSubtype)
    assert aircraft.engines_for_subtype("ATR 72-212A") == ["PW127M"]
    assert aircraft.engines_for_subtype("ATR 72-600F") == []
    assert isinstance(aircraft.drag, Drag)
    assert isinstance(aircraft.kinematic, Kinematic)
    assert aircraft.icao == "AT72"
    assert aircraft.wing.t_c == 0.12
    assert aircraft.horizontal_tail.span == 7.5
    assert aircraft.vertical_tail.height == 3.2
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


def test_load_a320_certified_and_higher_order_data():
    aircraft = load_aircraft("A320")

    assert aircraft.aircraft == "Airbus A320-214"
    assert aircraft.mtow == 78000
    assert aircraft.wing is not None
    assert aircraft.wing.span == 34.10
    assert aircraft.wing.mac == 4.1935

    assert isinstance(aircraft.certification, Certification)
    assert aircraft.certification.model == "A320-214"
    assert aircraft.certification.type_certificate == "EASA.A.064"
    assert aircraft.certification.source_id == "easa_a064_issue_62"

    assert isinstance(aircraft.certified_mass_limits, CertifiedMassLimits)
    assert aircraft.certified_mass_limits.mzfw == 62500
    assert aircraft.certified_mass_limits.minimum_weight == 37230
    assert aircraft.geometry is None
    assert isinstance(aircraft.horizontal_tail, Tail)
    assert aircraft.horizontal_tail.area == pytest.approx(30.843428)
    assert aircraft.horizontal_tail.elevator_area_ratio == pytest.approx(0.480561)
    assert isinstance(aircraft.vertical_tail, Tail)
    assert aircraft.vertical_tail.area == pytest.approx(21.514182)
    assert aircraft.vertical_tail.rudder_area_ratio == pytest.approx(0.299983)
    assert aircraft.aileron is None
    assert aircraft.mass_properties is None
    assert aircraft.engines_for_subtype("A320-214") == [
        "CFM56-5B4",
        "CFM56-5B4/2",
    ]
    assert aircraft.engine.default == "CFM56-5B4"
    assert (
        aircraft.data_provenance["field_sources"]["mtow"]
        == ["easa_a064_issue_62"]
    )


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


def test_web_editor_preserves_noneditable_certification_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.syspath_prepend(str(Path("tools").resolve()))
    import web_editor

    target_dir = tmp_path / "turbofan"
    target_dir.mkdir()
    target = target_dir / "a320.yml"
    target.write_text(
        """aircraft: Airbus A320-214
icao: A320
mtow: 78000
wing:
  area: 122
  span: 34.1
  mac: 4.1935
certification:
  model: A320-214
  type_certificate: EASA.A.064
data_provenance:
  field_sources:
    wing.mac: [easa_a064_issue_62]
"""
    )
    editable = {
        "aircraft": "Airbus A320-214",
        "icao": "A320",
        "mtow": 78000,
        "wing": {"area": 122, "span": 34.1, "mac": 4.1935},
    }
    monkeypatch.setattr(web_editor, "YAML_BASE_DIR", tmp_path)

    web_editor.save_aircraft_data("A320", editable, "turbofan")

    saved = yaml.safe_load(target.read_text())
    assert saved["certification"]["type_certificate"] == "EASA.A.064"
    assert saved["data_provenance"]["field_sources"]["wing.mac"] == [
        "easa_a064_issue_62"
    ]
    assert saved["wing"]["mac"] == 4.1935


def test_web_editor_preserves_root_subtypes_during_partial_engine_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.syspath_prepend(str(Path("tools").resolve()))
    import web_editor

    target_dir = tmp_path / "turbofan"
    target_dir.mkdir()
    target = target_dir / "a320.yml"
    target.write_text(
        """subtypes:
  - subtype: A320-214
    engines:
    - CFM56-5B4
engine:
  type: turbofan
  mount: wing
  number: 2
  default: CFM56-5B4
  options:
  - CFM56-5B4
"""
    )
    monkeypatch.setattr(web_editor, "YAML_BASE_DIR", tmp_path)

    web_editor.save_aircraft_data(
        "A320", {"engine": {"default": "CFM56-5B4"}}, "turbofan"
    )

    saved = yaml.safe_load(target.read_text())
    assert saved["subtypes"] == [
        {"subtype": "A320-214", "engines": ["CFM56-5B4"]}
    ]


def test_web_editor_form_parser_preserves_decimal_geometry():
    from tools.web_editor import parse_form_data

    parsed = parse_form_data(
        {
            "wing__area": "112.3",
            "wing__sweep": "25.75",
            "horizontal_tail__area": "30.843428",
            "vertical_tail__height": "5.87",
        }
    )
    assert parsed["wing"]["area"] == 112.3
    assert parsed["wing"]["sweep"] == 25.75
    assert parsed["horizontal_tail"]["area"] == 30.843428
    assert parsed["vertical_tail"]["height"] == 5.87


def test_web_editor_form_parser_round_trips_subtypes():
    from tools.web_editor import parse_form_data

    parsed = parse_form_data(
        {
            "engine__type": "turbofan",
            "subtypes": (
                "A320-214 | CFM56-5B4; CFM56-5B4/2\n"
                "A320-232 | V2527-A5"
            ),
        }
    )
    assert parsed["subtypes"] == [
        {
            "subtype": "A320-214",
            "engines": ["CFM56-5B4", "CFM56-5B4/2"],
        },
        {"subtype": "A320-232", "engines": ["V2527-A5"]},
    ]
