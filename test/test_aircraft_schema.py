from __future__ import annotations

import copy
import csv
import subprocess
import sys
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
from tools.update_aircraft_geometry import GEOMETRY_UNITS

ROOT = Path("src/aerox/data/aircraft/yml")
SOURCE_ROOT = Path("src/aerox/data/aircraft/provenance")
FILES = sorted(ROOT.glob("*/*.yml"))
RESEARCH_FILES = (
    Path("docs/modeling/geometry-research-airbus-boeing.tsv"),
    Path("docs/modeling/geometry-research-regional-business.tsv"),
    Path("docs/modeling/geometry-research-ga-turboprop-military.tsv"),
)
CAD_MEASUREMENT_FILES = (
    Path("docs/modeling/airbus-cad-geometry-measurements.csv"),
    Path("docs/modeling/boeing-cad-geometry-measurements.csv"),
)


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


def test_only_compact_certification_and_geometry_extensions_were_added(records):
    original_sections = {
        "aircraft", "icao", "iata", "mtow", "mlw", "oew", "mfc", "vmo",
        "mmo", "ceiling", "icao_wtc", "eu_wtc", "faa_wtc",
        "parking_area_m2", "pax", "fuselage", "wing", "flaps", "engine",
        "drag", "kinematic",
    }
    additions = {
        "certification", "mzfw", "minimum_weight", "horizontal_tail",
        "vertical_tail",
    }
    for _, document in records:
        assert set(document) == original_sections | additions
        assert list(document["certification"]) == [
            "model", "authority", "type_certificate", "issue", "issue_date"
        ]
        assert list(document["wing"]) == [
            "area", "span", "mac", "sweep", "t/c", "root_chord",
            "tip_chord", "taper_ratio", "aspect_ratio", "dihedral",
        ]
        assert list(document["horizontal_tail"]) == [
            "area", "span", "mac", "root_chord", "tip_chord",
            "taper_ratio", "sweep", "dihedral", "arm",
            "elevator_area_ratio",
        ]
        assert list(document["vertical_tail"]) == [
            "area", "height", "mac", "root_chord", "tip_chord",
            "taper_ratio", "sweep", "arm", "rudder_area_ratio",
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
        assert len(text.splitlines()) < 600


def test_every_geometry_field_has_matching_plain_text_provenance(records):
    ledger = yaml.safe_load(
        Path("docs/modeling/sources.yml").read_text(encoding="utf-8")
    )
    source_ids = {source["id"] for source in ledger["sources"]}
    allowed_statuses = {
        "certified-published",
        "manufacturer-published",
        "manufacturer-published-planning",
        "calculated-from-published-inputs",
        "inferred-with-uncertainty",
        "legacy-unverified",
        "unresolved",
    }
    for path, document in records:
        notes = parse_source_notes(
            (SOURCE_ROOT / path.parent.name / f"{path.stem}.txt").read_text(
                encoding="utf-8"
            )
        )
        for field, unit in GEOMETRY_UNITS.items():
            record = notes["fields"][field]
            section, name = field.split(".", 1)
            value = document[section][name]
            assert record["unit"] == unit, (path, field)
            assert record["status"] in allowed_statuses, (path, field)
            if value is None:
                assert record["value"] == "null", (path, field)
                assert record["reason_unavailable"], (path, field)
            else:
                assert float(record["value"]) == pytest.approx(value), (path, field)
                sources = [item.strip() for item in record["source"].split("|")]
                assert sources and sources != ["-"], (path, field)
                assert set(sources) <= source_ids, (path, field)


def test_every_aircraft_loads_through_public_api(records):
    for path, document in records:
        aircraft = load_aircraft(path.stem)
        assert aircraft.icao == document["icao"] == path.stem.upper()
        assert aircraft.certified_mass_limits.mtow == document["mtow"]
        assert aircraft.certified_mass_limits.mzfw == document["mzfw"]
        assert document["certification"]["issue"] is None or isinstance(
            document["certification"]["issue"], str
        )
        assert document["certification"]["issue_date"] is None or isinstance(
            document["certification"]["issue_date"], str
        )
        assert aircraft.certification.issue_date == document["certification"]["issue_date"]


def test_compact_migration_is_idempotent(records):
    for path, document in records:
        sidecar_path = SOURCE_ROOT / path.parent.name / f"{path.stem}.txt"
        sidecar = sidecar_path.read_text(encoding="utf-8")
        normalized, normalized_sidecar = normalize_bundle(document, sidecar)
        assert normalized == document
        assert normalized_sidecar == sidecar
        assert dump_document(normalized) == path.read_text(encoding="utf-8")


def test_geometry_updater_is_idempotent_without_research_input():
    result = subprocess.run(
        [sys.executable, "tools/update_aircraft_geometry.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "changed=0" in result.stdout


def test_geometry_updater_is_idempotent_with_all_research_inputs():
    command = [sys.executable, "tools/update_aircraft_geometry.py"]
    for path in RESEARCH_FILES:
        command.extend(["--research", path])
    result = subprocess.run(
        [*command, "--check"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_rows = sum(
        sum(1 for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")) - 1
        for path in RESEARCH_FILES
    )
    assert expected_rows >= 273
    assert f"research_values={expected_rows}" in result.stdout
    assert "changed=0" in result.stdout


def test_pc24_manufacturer_tail_span_is_adopted_with_exact_provenance():
    pc24 = load_aircraft("PC24")
    assert pc24.horizontal_tail.span == pytest.approx(6.8)
    notes = parse_source_notes(
        (SOURCE_ROOT / "turbofan/pc24.txt").read_text(encoding="utf-8")
    )
    record = notes["fields"]["horizontal_tail.span"]
    assert record["status"] == "manufacturer-published-planning"
    assert record["source"] == "pilatus_pc24_gsg_02484_i2r4"
    assert "Figure 0-1-1" in record["page_or_section"]


def test_cad_measurement_inventories_use_stable_ledger_source_ids():
    ledger = yaml.safe_load(
        Path("docs/modeling/sources.yml").read_text(encoding="utf-8")
    )
    source_ids = {source["id"] for source in ledger["sources"]}
    for path in CAD_MEASUREMENT_FILES:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows, path
        assert all(row["source_id"] in source_ids for row in rows), path


def test_variant_mismatched_geometry_remains_evidence_only():
    f15 = load_aircraft("F15")
    assert f15.wing.span == 13.0
    assert f15.wing.area is None
    assert f15.horizontal_tail.area is None
    assert f15.vertical_tail.area is None

    notes = parse_source_notes(
        (SOURCE_ROOT / "turbofan/f15.txt").read_text(encoding="utf-8")
    )
    record = notes["fields"]["vertical_tail.area"]
    assert record["status"] == "unresolved"
    assert record["source"] == "-"
    assert "nasa_tm_72861" in record["searched_source"]
    assert "two fins" in record["notes"]


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


def test_derived_geometry_matches_documented_formulas(records):
    for path, document in records:
        wing = document["wing"]
        if wing["aspect_ratio"] is not None:
            assert wing["area"] is not None, path
            assert wing["span"] is not None, path
            expected = wing["span"] ** 2 / wing["area"]
            assert wing["aspect_ratio"] == pytest.approx(expected, rel=1e-6), path

        for name in ("wing", "horizontal_tail", "vertical_tail"):
            section = document[name]
            if section["taper_ratio"] is not None:
                assert section["root_chord"] is not None, path
                assert section["tip_chord"] is not None, path
                expected = section["tip_chord"] / section["root_chord"]
                assert section["taper_ratio"] == pytest.approx(expected, rel=2e-6), path

            if name != "wing" and all(
                section[field] is not None
                for field in ("area", "root_chord", "tip_chord")
            ):
                dimension = section["span" if name == "horizontal_tail" else "height"]
                if dimension is not None:
                    expected_area = dimension * (
                        section["root_chord"] + section["tip_chord"]
                    ) / 2
                    assert section["area"] == pytest.approx(
                        expected_area, rel=2e-4
                    ), path

            if name != "wing" and section["mac"] is not None:
                assert section["root_chord"] is not None, path
                assert section["taper_ratio"] is not None, path
                taper = section["taper_ratio"]
                expected_mac = (
                    (2 / 3)
                    * section["root_chord"]
                    * (1 + taper + taper**2)
                    / (1 + taper)
                )
                assert section["mac"] == pytest.approx(expected_mac, rel=5e-5), path


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
    edited["horizontal_tail"]["area"] = 30.843429
    web_editor.save_aircraft_data("A320", edited, "turbofan")
    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert saved["wing"]["mac"] == 4.1936
    assert saved["horizontal_tail"]["area"] == 30.843429
    assert saved["certification"] == document["certification"]
    assert saved["mzfw"] == document["mzfw"]
    assert saved["minimum_weight"] == document["minimum_weight"]


def test_web_editor_updates_geometry_provenance_and_derived_values(
    records, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import web_editor

    source_path, document = next(item for item in records if item[0].stem == "a320")
    yaml_root = tmp_path / "aircraft/yml"
    target = yaml_root / "turbofan/a320.yml"
    target.parent.mkdir(parents=True)
    target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar = tmp_path / "aircraft/provenance/turbofan/a320.txt"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        (SOURCE_ROOT / "turbofan/a320.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(web_editor, "YAML_BASE_DIR", yaml_root)

    edited = copy.deepcopy(document)
    edited["wing"]["span"] = 34.2
    web_editor.save_aircraft_data("A320", edited, "turbofan")

    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert saved["wing"]["aspect_ratio"] == pytest.approx(34.2**2 / 122, rel=1e-6)
    notes = parse_source_notes(sidecar.read_text(encoding="utf-8"))
    assert notes["fields"]["wing.span"]["value"] == "34.2"
    assert notes["fields"]["wing.span"]["source"] == "aerox_web_editor_manual_entry"
    assert notes["fields"]["wing.aspect_ratio"]["inputs"] == "span=34.2 m; area=122.0 m2"
    assert notes["fields"]["wing.aspect_ratio"]["status"] == "legacy-unverified"
