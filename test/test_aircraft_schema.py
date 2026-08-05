from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from aerox import load_aircraft
from aerox.aircraft import parse_source_notes
from tools.migrate_aircraft_data import (
    SIMPLE_TEMPLATE,
    dump_document,
    normalize_bundle,
    validate_simple,
)

ROOT = Path("src/aerox/data/aircraft/yml")
SOURCE_ROOT = Path("src/aerox/data/aircraft/provenance")
FILES = sorted(ROOT.glob("*/*.yml"))


@pytest.fixture(scope="module")
def records():
    return [
        (path, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in FILES
    ]


def test_all_aircraft_use_the_original_compact_parameter_layout(records):
    assert len(records) == 144
    for path, document in records:
        assert not validate_simple(document), path
        assert list(document) == list(SIMPLE_TEMPLATE)
        assert "schema_version" not in document
        assert "flight_controls" not in document
        assert "data_quality" not in document


def test_only_small_certification_extension_was_added(records):
    original_sections = {
        "aircraft", "icao", "iata", "mtow", "mlw", "oew", "mfc", "vmo",
        "mmo", "ceiling", "icao_wtc", "eu_wtc", "faa_wtc",
        "parking_area_m2", "pax", "fuselage", "wing", "flaps", "engine",
        "drag", "kinematic",
    }
    additions = {"certification", "mzfw", "minimum_weight"}
    for _, document in records:
        assert set(document) == original_sections | additions
        assert list(document["certification"]) == [
            "model", "authority", "type_certificate", "issue", "issue_date"
        ]


def test_every_aircraft_has_one_short_plain_text_source_note(records):
    sidecars = sorted(SOURCE_ROOT.glob("*/*.txt"))
    assert len(sidecars) == len(records)
    assert {(p.parent.name, p.stem) for p in sidecars} == {
        (p.parent.name, p.stem) for p, _ in records
    }
    for path in sidecars:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("AEROX AIRCRAFT SOURCE NOTES\n")
        assert "{\"" not in text
        assert len(text.splitlines()) < 150


def test_every_aircraft_loads_through_public_api(records):
    for path, document in records:
        aircraft = load_aircraft(path.stem)
        assert aircraft.icao == document["icao"] == path.stem.upper()
        assert aircraft.certified_mass_limits.mtow == document["mtow"]
        assert aircraft.certified_mass_limits.mzfw == document["mzfw"]


def test_compact_migration_is_idempotent(records):
    for path, document in records:
        sidecar_path = SOURCE_ROOT / path.parent.name / f"{path.stem}.txt"
        sidecar = sidecar_path.read_text(encoding="utf-8")
        normalized, normalized_sidecar = normalize_bundle(document, sidecar)
        assert normalized == document
        assert normalized_sidecar == sidecar
        assert dump_document(normalized) == path.read_text(encoding="utf-8")


def test_a320_keeps_tcds_values_and_plain_text_sources():
    aircraft = load_aircraft("A320")
    assert aircraft.mtow == 78000
    assert aircraft.mzfw == 62500
    assert aircraft.minimum_weight == 37230
    assert aircraft.certification.type_certificate == "EASA.A.064"
    assert aircraft.certification.source_id == "easa_a064_issue_62"

    notes = parse_source_notes(
        (SOURCE_ROOT / "turbofan/a320.txt").read_text(encoding="utf-8")
    )
    assert notes["fields"]["mtow"]["source"] == "easa_a064_issue_62"
    assert notes["fields"]["mtow"]["status"] == "certified-published"


def test_basic_mass_relationships_are_reported_without_automatic_changes(records):
    warnings = []
    for path, document in records:
        mtow, mlw, mzfw = document["mtow"], document["mlw"], document["mzfw"]
        if mtow is not None and mlw is not None and mlw > mtow:
            warnings.append((path.stem, "mlw"))
        if mtow is not None and mzfw is not None and mzfw > mtow:
            warnings.append((path.stem, "mzfw"))
    # Known mixed-model/applicability conflicts remain data-review warnings.
    assert {icao for icao, _ in warnings} == {"be33", "be36", "c182", "c310", "c402", "p28r"}


def test_web_editor_preserves_compact_certification_fields(
    records, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import web_editor

    source_path, document = next(item for item in records if item[0].stem == "a320")
    target = tmp_path / "turbofan/a320.yml"
    target.parent.mkdir(parents=True)
    target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(web_editor, "YAML_BASE_DIR", tmp_path)

    edited = copy.deepcopy(document)
    edited["wing"]["mac"] = 4.1936
    web_editor.save_aircraft_data("A320", edited, "turbofan")
    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert saved["wing"]["mac"] == 4.1936
    assert saved["certification"] == document["certification"]
    assert saved["mzfw"] == document["mzfw"]
    assert saved["minimum_weight"] == document["minimum_weight"]
