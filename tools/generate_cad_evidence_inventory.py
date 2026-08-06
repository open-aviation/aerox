#!/usr/bin/env python3
"""Generate complete CAD archive and long-form parameter evidence inventories.

The archive table intentionally includes drawings with no accepted measurement.
That makes semantic extraction coverage auditable instead of treating a stored or
successfully converted drawing as though every useful dimension had been mapped.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAD_ROOT = ROOT / "data/cad"
DOCS = ROOT / "docs/modeling"
ARCHIVE_OUTPUT = DOCS / "cad-archive-inventory.csv"
EVIDENCE_OUTPUT = DOCS / "cad-evidence-inventory.csv"

RESEARCH_FILES = (
    DOCS / "geometry-research-airbus-boeing.tsv",
    DOCS / "geometry-research-airbus-tail-batch.tsv",
    DOCS / "geometry-research-cad-batch-2.tsv",
    DOCS / "geometry-research-cad-batch-3.tsv",
    DOCS / "geometry-research-cad-batch-4.tsv",
    DOCS / "geometry-research-regional-business.tsv",
    DOCS / "geometry-research-ga-turboprop-military.tsv",
)

LONG_INPUTS = (DOCS / "airbus-cad-geometry-measurements.csv",)
WIDE_INPUTS = (
    DOCS / "airbus-cad-tail-direct-extraction.csv",
    DOCS / "airbus-cad-tail-batch-2.csv",
    DOCS / "boeing-cad-geometry-measurements.csv",
    DOCS / "boeing-cad-tail-direct-extraction.csv",
)
REVIEW_INPUT = DOCS / "cad-archive-review-observations.csv"

ARCHIVE_COLUMNS = (
    "archive_id",
    "manufacturer",
    "source_id",
    "source_url",
    "archive_relpath",
    "archive_sha256",
    "drawing_relpath",
    "drawing_sha256",
    "drawing_format",
    "represented_icao",
    "configuration",
    "original_units",
    "manufacturer_accuracy",
    "stored_status",
    "semantic_extraction_status",
    "evidence_record_count",
    "mapped_parameter_count",
    "blocking_reason",
    "last_reviewed",
    "notes",
)

EVIDENCE_COLUMNS = (
    "record_id",
    "record_type",
    "archive_id",
    "source_id",
    "target_icao",
    "configuration",
    "view",
    "entity_type",
    "entity_handles",
    "source_parameter_name",
    "physical_definition",
    "normalized_value",
    "normalized_unit",
    "canonical_path",
    "value_status",
    "extraction_method",
    "input_record_ids",
    "formula",
    "uncertainty",
    "applicability",
    "adoption_status",
    "rejection_reason",
    "reviewed_date",
    "notes",
)

WIDE_FIELDS = {
    "span_m": ("horizontal-tail overall span", "horizontal_tail.span", "m"),
    "root_chord_m": ("equivalent root chord", "horizontal_tail.root_chord", "m"),
    "tip_chord_m": ("equivalent tip chord", "horizontal_tail.tip_chord", "m"),
    "taper_ratio": (
        "equivalent taper ratio",
        "horizontal_tail.taper_ratio",
        "nondimensional",
    ),
    "area_m2": ("equivalent projected area", "horizontal_tail.area", "m2"),
    "mac_m": ("equivalent trapezoid MAC", "horizontal_tail.mac", "m"),
    "quarter_chord_sweep_deg": (
        "equivalent quarter-chord sweep",
        "horizontal_tail.sweep",
        "deg",
    ),
    "elevator_area_ratio": (
        "visible projected elevator area ratio",
        "horizontal_tail.elevator_area_ratio",
        "nondimensional",
    ),
    "symmetry_error_mm": ("bilateral symmetry residual", "", "mm"),
    "published_span_minus_straight_endpoints_mm": (
        "terminal-cap span residual",
        "",
        "mm",
    ),
}


def canonical_archive_id(name: str) -> str:
    """Match URL-escaped names, spaces, parentheses, and local underscores."""
    stem = Path(unquote(name)).name.removesuffix(".zip")
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_index() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    ledger = yaml.safe_load((DOCS / "sources.yml").read_text(encoding="utf-8"))
    records = {row["id"]: row for row in ledger["sources"]}
    by_archive: dict[str, str] = {}
    for source_id, row in records.items():
        url = row.get("download_url")
        if url:
            by_archive[canonical_archive_id(urlparse(str(url)).path)] = source_id
    return records, by_archive


def source_table() -> list[dict[str, str]]:
    pattern = re.compile(
        r"^(airbus|boeing)/archives/(.+\.zip) \| (.+?) \| (https://\S+)$"
    )
    rows = []
    for line in (CAD_ROOT / "SOURCES.txt").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        manufacturer, archive, applicability, url = match.groups()
        rows.append(
            {
                "manufacturer": manufacturer,
                "archive": archive,
                "applicability": applicability,
                "url": url,
            }
        )
    if len(rows) != 50:
        raise ValueError(f"expected 50 CAD archive source rows, found {len(rows)}")
    return rows


def research_usage() -> dict[str, set[str]]:
    usage: dict[str, set[str]] = {}
    for path in RESEARCH_FILES:
        with path.open(encoding="utf-8", newline="") as stream:
            lines = (
                line for line in stream if line.strip() and not line.startswith("#")
            )
            for row in csv.DictReader(lines, delimiter="\t"):
                for source_id in row["source_id"].split("|"):
                    source_id = source_id.strip()
                    if source_id:
                        usage.setdefault(source_id, set()).add(row["field"])
    return usage


def archive_for_source(records: dict[str, dict[str, object]], source_id: str) -> str:
    row = records.get(source_id, {})
    url = row.get("download_url")
    if not url:
        return ""
    return canonical_archive_id(urlparse(str(url)).path)


def long_evidence(records: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for path in LONG_INPUTS:
        with path.open(encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream), 1):
                value = row.get("value", "")
                output.append(
                    {
                        "record_id": f"{path.stem}-{index:04d}",
                        "record_type": "raw-entity"
                        if row.get("status") == "raw-cad-entity"
                        else "derived-parameter",
                        "archive_id": archive_for_source(records, row["source_id"]),
                        "source_id": row["source_id"],
                        "target_icao": row["icao"],
                        "configuration": row["configuration"],
                        "view": row.get("view_layer", ""),
                        "entity_type": row.get("measurement", ""),
                        "entity_handles": row.get("entity_handles", ""),
                        "source_parameter_name": row.get("measurement", ""),
                        "physical_definition": row.get("formula", ""),
                        "normalized_value": "" if value.lower() == "null" else value,
                        "normalized_unit": row.get("unit", ""),
                        "canonical_path": row.get("canonical_field", ""),
                        "value_status": row.get("status", ""),
                        "extraction_method": "reviewed CAD entity selection",
                        "input_record_ids": row.get("entity_coordinates_mm", ""),
                        "formula": row.get("formula", ""),
                        "uncertainty": row.get("uncertainty", ""),
                        "applicability": row.get("applicability", ""),
                        "adoption_status": row.get("adoption", "evidence"),
                        "rejection_reason": "",
                        "reviewed_date": "2026-08-06",
                        "notes": "",
                    }
                )

    for path in WIDE_INPUTS:
        with path.open(encoding="utf-8", newline="") as stream:
            for row_index, row in enumerate(csv.DictReader(stream), 1):
                source_id = row["source_id"]
                handles = ";".join(
                    filter(
                        None,
                        (
                            row.get("leading_handles", ""),
                            row.get("trailing_handles", ""),
                            row.get("hinge_handles", ""),
                            row.get("leading_edge_entities", ""),
                            row.get("trailing_edge_entities", ""),
                            row.get("tip_entities", ""),
                            row.get("hinge_entities", ""),
                        ),
                    )
                )
                base_status = row.get("adoption_status") or (
                    "rejected"
                    if "rejected" in row.get("notes", "").lower()
                    else "candidate"
                )
                for field, (label, canonical, unit) in WIDE_FIELDS.items():
                    if not row.get(field):
                        continue
                    output.append(
                        {
                            "record_id": f"{path.stem}-{row_index:03d}-{field}",
                            "record_type": "measured-geometry"
                            if field
                            in {
                                "span_m",
                                "symmetry_error_mm",
                                "published_span_minus_straight_endpoints_mm",
                            }
                            else "derived-parameter",
                            "archive_id": canonical_archive_id(row["archive"])
                            if row.get("archive")
                            else archive_for_source(records, source_id),
                            "source_id": source_id,
                            "target_icao": row["icao"],
                            "configuration": row.get("configuration", ""),
                            "view": "top",
                            "entity_type": "LINE/POLYLINE/SPLINE",
                            "entity_handles": handles,
                            "source_parameter_name": label,
                            "physical_definition": row.get("applicability", ""),
                            "normalized_value": row[field],
                            "normalized_unit": unit,
                            "canonical_path": canonical,
                            "value_status": "inferred-with-uncertainty",
                            "extraction_method": "reviewed CAD entities and documented formula",
                            "input_record_ids": "",
                            "formula": row.get("formula", ""),
                            "uncertainty": row.get("uncertainty", ""),
                            "applicability": row.get("applicability", ""),
                            "adoption_status": base_status,
                            "rejection_reason": row.get("notes", "")
                            if base_status == "rejected"
                            else "",
                            "reviewed_date": "2026-08-06",
                            "notes": row.get("notes", ""),
                        }
                    )
    with REVIEW_INPUT.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), 1):
            output.append(
                {
                    "record_id": f"cad-archive-review-{index:03d}",
                    "record_type": "archive-review",
                    "archive_id": canonical_archive_id(row["archive"]),
                    "source_id": row["source_id"],
                    "target_icao": row["icao"],
                    "configuration": row["configuration"],
                    "view": "",
                    "entity_type": "",
                    "entity_handles": row["entity_handles"],
                    "source_parameter_name": "archive semantic review",
                    "physical_definition": row["physical_definition"],
                    "normalized_value": "",
                    "normalized_unit": "",
                    "canonical_path": "",
                    "value_status": "unresolved",
                    "extraction_method": "manual vector-drawing applicability review",
                    "input_record_ids": "",
                    "formula": "",
                    "uncertainty": "",
                    "applicability": row["configuration"],
                    "adoption_status": row["adoption_status"],
                    "rejection_reason": row["notes"]
                    if row["adoption_status"] == "unresolved"
                    else "",
                    "reviewed_date": "2026-08-06",
                    "notes": row["notes"],
                }
            )
    return output


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    records, by_archive = source_index()
    usage = research_usage()
    evidence = long_evidence(records)
    evidence_counts: dict[str, int] = {}
    numeric_counts: dict[str, int] = {}
    for row in evidence:
        evidence_counts[row["archive_id"]] = (
            evidence_counts.get(row["archive_id"], 0) + 1
        )
        if row["normalized_value"]:
            numeric_counts[row["archive_id"]] = (
                numeric_counts.get(row["archive_id"], 0) + 1
            )

    archive_rows: list[dict[str, str]] = []
    for source in source_table():
        manufacturer = source["manufacturer"]
        archive_name = source["archive"]
        stem = archive_name.removesuffix(".zip")
        archive_id = canonical_archive_id(archive_name)
        archive = CAD_ROOT / manufacturer / "archives" / archive_name
        drawings = sorted((CAD_ROOT / manufacturer / "drawings" / stem).glob("*"))
        if len(drawings) != 1:
            raise ValueError(
                f"{archive_name}: expected one extracted original drawing, found {len(drawings)}"
            )
        drawing = drawings[0]
        source_id = by_archive.get(archive_id, "")
        count = evidence_counts.get(archive_id, 0)
        mapped = len(usage.get(source_id, set())) if source_id else 0
        if mapped:
            semantic = "reviewed-and-adopted"
            blocker = ""
        elif count:
            if numeric_counts.get(archive_id, 0):
                semantic = "reviewed-evidence-only"
                blocker = "Physical/reference-planform definition or exact bundled applicability is unresolved."
            else:
                semantic = "reviewed-blocked-no-parameter"
                blocker = "Archive was reviewed, but no defensible numeric parameter was extracted or separately needed."
        else:
            semantic = "stored-not-semantically-extracted"
            blocker = "No reviewed entity-to-parameter mapping yet; storage or conversion alone is not extraction."
        archive_rows.append(
            {
                "archive_id": archive_id,
                "manufacturer": manufacturer.title(),
                "source_id": source_id,
                "source_url": source["url"],
                "archive_relpath": str(archive.relative_to(ROOT)),
                "archive_sha256": sha256(archive),
                "drawing_relpath": str(drawing.relative_to(ROOT)),
                "drawing_sha256": sha256(drawing),
                "drawing_format": drawing.suffix.removeprefix(".").upper(),
                "represented_icao": source["applicability"],
                "configuration": source["applicability"],
                "original_units": "drawing-specific; retained in evidence rows",
                "manufacturer_accuracy": "+/-0.10 m planning drawing"
                if manufacturer == "airbus"
                else "+/-6 in planning drawing",
                "stored_status": "original-archive-and-drawing-present",
                "semantic_extraction_status": semantic,
                "evidence_record_count": str(count),
                "mapped_parameter_count": str(mapped),
                "blocking_reason": blocker,
                "last_reviewed": "2026-08-06",
                "notes": (
                    "Official archive URL and stable source ledger ID recorded."
                    if source_id
                    else "Official URL is inventoried but a stable source ledger entry still needs to be added."
                ),
            }
        )

    write_csv(ARCHIVE_OUTPUT, ARCHIVE_COLUMNS, archive_rows)
    write_csv(EVIDENCE_OUTPUT, EVIDENCE_COLUMNS, evidence)
    print(f"archives={len(archive_rows)} evidence_records={len(evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
