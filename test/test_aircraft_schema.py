from __future__ import annotations

import copy
import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aerox import load_aircraft
from aerox.aircraft import parse_source_notes
from tools.extract_airbus_cad_tail_geometry import DRAWINGS, extract
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
    Path("docs/modeling/geometry-research-airbus-tail-batch.tsv"),
    Path("docs/modeling/geometry-research-cad-batch-2.tsv"),
    Path("docs/modeling/geometry-research-cad-batch-3.tsv"),
    Path("docs/modeling/geometry-research-cad-batch-4.tsv"),
    Path("docs/modeling/geometry-research-regional-business.tsv"),
    Path("docs/modeling/geometry-research-ga-turboprop-military.tsv"),
)
CAD_MEASUREMENT_FILES = (
    Path("docs/modeling/airbus-cad-geometry-measurements.csv"),
    Path("docs/modeling/airbus-cad-tail-direct-extraction.csv"),
    Path("docs/modeling/airbus-cad-tail-batch-2.csv"),
    Path("docs/modeling/boeing-cad-geometry-measurements.csv"),
    Path("docs/modeling/boeing-cad-tail-direct-extraction.csv"),
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
        "subtypes", "certification", "mzfw", "minimum_weight", "horizontal_tail",
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
        assert list(document["engine"]) == [
            "type", "mount", "number", "default", "options",
        ]


def test_subtypes_use_exact_uniform_structure_edb_codes_and_provenance(records):
    ledger = yaml.safe_load(
        Path("docs/modeling/sources.yml").read_text(encoding="utf-8")
    )
    source_ids = {source["id"] for source in ledger["sources"]}
    with Path("src/aerox/data/engines/engines.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        edb_engines = {row["name"] for row in csv.DictReader(stream)}
    mapped_aircraft = 0
    for path, document in records:
        mapping = document["subtypes"]
        assert mapping is None or mapping, path
        if mapping is None:
            continue
        mapped_aircraft += 1
        assert all(isinstance(item, dict) and len(item) == 1 for item in mapping)
        assert len({next(iter(item)) for item in mapping}) == len(mapping), path
        for item in mapping:
            subtype, engines = next(iter(item.items()))
            assert isinstance(subtype, str) and subtype.strip()
            engines = [engines] if isinstance(engines, str) else engines
            assert isinstance(engines, list) and engines
            assert len(set(engines)) == len(engines)
            assert all(isinstance(engine, str) and engine.strip() for engine in engines)
            assert set(engines) <= edb_engines

        notes = parse_source_notes(
            (SOURCE_ROOT / path.parent.name / f"{path.stem}.txt").read_text(
                encoding="utf-8"
            )
        )
        record = notes["fields"]["subtypes"]
        assert record["status"] == "certified-published"
        adopted_sources = [part.strip() for part in record["source"].split("|")]
        assert adopted_sources and set(adopted_sources) <= source_ids
        assert "icao_eedb_issue_32" in adopted_sources
    assert mapped_aircraft == 17


def test_airbus_subtypes_are_populated_and_boeing_subtypes_are_null(records):
    by_icao = {document["icao"]: document for _, document in records}
    airbus = {
        "A19N", "A20N", "A21N", "A306", "A318", "A319", "A320", "A321",
        "A332", "A333", "A339", "A343", "A359", "A35K", "A388", "BCS1",
        "BCS3",
    }
    assert all(by_icao[icao]["subtypes"] for icao in airbus)
    assert all(
        document["subtypes"] is None
        for icao, document in by_icao.items()
        if icao.startswith("B") and icao not in {"BCS1", "BCS3"}
    )


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


def test_subtype_updater_is_idempotent():
    result = subprocess.run(
        [sys.executable, "tools/update_aircraft_subtypes.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "researched=17" in result.stdout
    assert "mapped_subtypes=86" in result.stdout
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


def test_cad_archive_inventory_covers_every_stored_manufacturer_drawing():
    path = Path("docs/modeling/cad-archive-inventory.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert len({row["archive_id"] for row in rows}) == 50
    assert all(row["stored_status"] == "original-archive-and-drawing-present" for row in rows)
    assert all(row["semantic_extraction_status"] in {
        "reviewed-and-adopted", "reviewed-evidence-only",
        "reviewed-blocked-no-parameter",
        "stored-not-semantically-extracted",
    } for row in rows)
    # Every archive has an explicit semantic review, including blocked ones.
    assert not any(
        row["semantic_extraction_status"] == "stored-not-semantically-extracted"
        for row in rows
    )
    assert all(row["source_id"] for row in rows)


def test_cad_evidence_inventory_preserves_unmapped_and_rejected_observations():
    path = Path("docs/modeling/cad-evidence-inventory.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 200
    assert any(not row["canonical_path"] for row in rows)
    assert any(row["adoption_status"] == "rejected" for row in rows)
    assert any(row["entity_handles"] for row in rows)


def test_cad_inventory_generator_is_idempotent():
    before = {
        path: path.read_bytes()
        for path in (
            Path("docs/modeling/cad-archive-inventory.csv"),
            Path("docs/modeling/cad-evidence-inventory.csv"),
        )
    }
    result = subprocess.run(
        [sys.executable, "tools/generate_cad_evidence_inventory.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "archives=50" in result.stdout
    assert all(path.read_bytes() == content for path, content in before.items())


def test_direct_airbus_dwg_json_extractor_preserves_reviewed_geometry():
    """Exercise the parser without requiring LibreDWG or the ignored CAD files."""
    drawing = DRAWINGS[0]
    centre = 10000.0
    semispan = drawing.span_m * 500

    def entity(handle, start, end):
        return {
            "entity": "LINE",
            "handle": [0, 2, int(handle, 16), int(handle, 16)],
            "start": [*start, 0.0],
            "end": [*end, 0.0],
        }

    # Use the reviewed A318 chord/sweep and elevator values in a translated
    # synthetic drawing.  The unrelated duplicate object documents why the
    # extractor indexes LINE entities rather than every LibreDWG object.
    root, tip = 3.726893933 * 1000, 1.227873751 * 1000
    sweep = 27.945078238
    le_root = 20000.0
    quarter_shift = math.tan(math.radians(sweep)) * semispan
    le_tip = le_root + quarter_shift - 0.25 * (tip - root)
    te_root, te_tip = le_root + root, le_tip + tip
    ratio = drawing.expected_elevator_ratio
    hinge_root = te_root - ratio * root
    hinge_tip = te_tip - ratio * tip
    objects = [{"object": "SCALE", "handle": [0, 2, int(drawing.leading_right, 16), int(drawing.leading_right, 16)]}]
    for left, right, y_root, y_tip in (
        (drawing.leading_left, drawing.leading_right, le_root, le_tip),
        (drawing.trailing_left, drawing.trailing_right, te_root, te_tip),
        (drawing.hinge_left, drawing.hinge_right, hinge_root, hinge_tip),
    ):
        objects.extend([
            entity(left, (centre - semispan, y_tip), (centre, y_root)),
            entity(right, (centre, y_root), (centre + semispan, y_tip)),
        ])

    result = extract({"OBJECTS": objects}, drawing)
    assert result["root_chord_m"] == pytest.approx(root / 1000)
    assert result["tip_chord_m"] == pytest.approx(tip / 1000)
    assert result["quarter_chord_sweep_deg"] == pytest.approx(sweep)
    assert result["elevator_area_ratio"] == pytest.approx(ratio)
    assert result["symmetry_error_mm"] == pytest.approx(0)


def test_a350_common_tail_uses_published_area_and_consistent_derived_geometry():
    a359 = load_aircraft("A359")
    a35k = load_aircraft("A35K")
    for aircraft in (a359, a35k):
        tail = aircraft.horizontal_tail
        assert tail.area == pytest.approx(82.0)
        assert tail.span == pytest.approx(18.79)
        assert tail.root_chord == pytest.approx(6.262168)
        assert tail.tip_chord == pytest.approx(2.465879)
        assert tail.taper_ratio == pytest.approx(tail.tip_chord / tail.root_chord, rel=2e-6)
        assert tail.area == pytest.approx(
            tail.span * (tail.root_chord + tail.tip_chord) / 2,
            rel=2e-6,
        )
        assert aircraft.vertical_tail.height == pytest.approx(9.42)

    notes = parse_source_notes(
        (SOURCE_ROOT / "turbofan/a35k.txt").read_text(encoding="utf-8")
    )
    area = notes["fields"]["horizontal_tail.area"]
    assert area["status"] == "manufacturer-published"
    assert area["source"] == "airbus_a350_common_htp_2016_02"
    assert area["previous_value"] == "85.707556 m2"


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
