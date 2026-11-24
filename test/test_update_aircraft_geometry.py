from __future__ import annotations

import copy
import sys


from tools import update_aircraft_geometry as geometry
from tools.update_aircraft_geometry import (
    GEOMETRY_UNITS,
    derive_geometry,
    render_geometry_section,
)


def _document(**values: float | None) -> dict:
    document: dict = {"wing": {}, "horizontal_tail": {}, "vertical_tail": {}}
    for path in GEOMETRY_UNITS:
        section, field = path.split(".", 1)
        document[section][field] = values.get(path.replace(".", "__"))
    return document


def _run(document: dict, metadata: dict) -> None:
    original_values = {
        path: document[path.split(".")[0]][path.split(".", 1)[1]]
        for path in GEOMETRY_UNITS
    }
    derive_geometry(document, metadata, original_values, copy.deepcopy(metadata))


PUBLISHED_TAPER = {
    "status": "manufacturer-published",
    "source": "published_source",
    "inputs": "Taper Ratio 0.365",
}


def test_consistent_published_taper_keeps_value_and_note_block():
    document = _document(
        wing__root_chord=7.700826, wing__tip_chord=2.810802, wing__taper_ratio=0.365
    )
    metadata = {"wing.taper_ratio": dict(PUBLISHED_TAPER)}
    _run(document, metadata)
    assert document["wing"]["taper_ratio"] == 0.365
    assert metadata["wing.taper_ratio"] == PUBLISHED_TAPER


def test_consistent_aspect_ratio_keeps_value_and_note_block():
    record = {
        "status": "calculated-from-published-inputs",
        "source": "tc_source",
        "formula": "span^2 / area",
    }
    document = _document(
        wing__span=11.68, wing__area=13.46, wing__aspect_ratio=10.1353937
    )
    metadata = {
        "wing.aspect_ratio": dict(record),
        "wing.span": {"status": "inferred-with-uncertainty", "source": "other"},
    }
    _run(document, metadata)
    assert document["wing"]["aspect_ratio"] == 10.1353937
    assert metadata["wing.aspect_ratio"] == record


def test_null_taper_ratio_is_still_derived():
    document = _document(wing__root_chord=8.0, wing__tip_chord=2.0)
    metadata: dict = {}
    _run(document, metadata)
    assert document["wing"]["taper_ratio"] == 0.25
    assert metadata["wing.taper_ratio"]["formula"] == "tip_chord / root_chord"


def test_taper_ratio_without_note_block_is_still_derived():
    document = _document(
        wing__root_chord=8.0, wing__tip_chord=2.0, wing__taper_ratio=0.25
    )
    metadata: dict = {}
    _run(document, metadata)
    assert metadata["wing.taper_ratio"]["formula"] == "tip_chord / root_chord"


def test_inconsistent_taper_ratio_is_corrected():
    document = _document(
        wing__root_chord=8.0, wing__tip_chord=2.0, wing__taper_ratio=0.3
    )
    metadata = {"wing.taper_ratio": dict(PUBLISHED_TAPER)}
    _run(document, metadata)
    assert document["wing"]["taper_ratio"] == 0.25
    assert metadata["wing.taper_ratio"]["previous_value"] == "0.3 nondimensional"


def test_changed_input_provenance_rederives_aspect_ratio():
    document = _document(wing__span=10.0, wing__area=20.0, wing__aspect_ratio=5.0)
    metadata = {
        "wing.aspect_ratio": {"status": "legacy-unverified", "source": "legacy"},
        "wing.span": {"status": "legacy-unverified", "source": "legacy"},
        "wing.area": {"status": "legacy-unverified", "source": "legacy"},
    }
    original_values = {"wing.span": 10.0, "wing.area": 20.0, "wing.aspect_ratio": 5.0}
    original_metadata = copy.deepcopy(metadata)
    # Research upgrades both inputs without changing their values.
    metadata["wing.span"] = {"status": "certified-published", "source": "tc"}
    metadata["wing.area"] = {"status": "certified-published", "source": "tc"}
    derive_geometry(document, metadata, original_values, original_metadata)
    assert metadata["wing.aspect_ratio"]["status"] == (
        "calculated-from-published-inputs"
    )
    assert metadata["wing.aspect_ratio"]["source"] == "tc"


def test_derive_without_original_metadata_keeps_previous_behaviour():
    document = _document(
        wing__root_chord=7.700826, wing__tip_chord=2.810802, wing__taper_ratio=0.365
    )
    metadata = {"wing.taper_ratio": dict(PUBLISHED_TAPER)}
    derive_geometry(document, metadata)
    assert metadata["wing.taper_ratio"]["formula"] == "tip_chord / root_chord"


def test_render_keeps_equal_hand_written_number_format():
    document = _document(wing__tip_chord=1.434, wing__span=12.0)
    note = (
        "WING AND TAIL GEOMETRY\n\n"
        "FIELD wing.tip_chord\n  Value: 1.434000\n  Unit: m\nEND FIELD\n\n"
        "FIELD wing.span\n  Value: 11.5\n  Unit: m\nEND FIELD\n"
    )
    rendered = render_geometry_section(document, note, {})
    assert "  Value: 1.434000\n" in rendered
    assert "  Value: 12.0\n" in rendered


def test_tool_check_reports_no_change_on_bundled_data(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["update_aircraft_geometry.py", "--check"])
    status = geometry.main()
    assert status == 0, capsys.readouterr().out
