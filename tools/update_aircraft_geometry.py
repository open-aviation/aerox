#!/usr/bin/env python3
"""Apply researched wing/tail geometry and maintain plain-text source notes.

Research input is a tab-separated file with one field value per row.  The YAML
files remain parameter-only; publication status, applicability, formulas, and
uncertainty are rendered into their matching source-note text files.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        SOURCES_ROOT,
        dump_document,
        normalize_bundle,
        parse_old_provenance,
        validate_simple,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        SOURCES_ROOT,
        dump_document,
        normalize_bundle,
        parse_old_provenance,
        validate_simple,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LEDGER = PROJECT_ROOT / "sources/sources.yml"
# A source-note section header: a line in capitals, other than END FIELD.
_SECTION_HEADER = re.compile(r"^(?!END FIELD$)[A-Z][A-Z0-9 ,/()&-]*$", re.MULTILINE)
DEFAULT_EXCLUSIONS = PROJECT_ROOT / "tools/geometry-runtime-exclusions.txt"

GEOMETRY_UNITS = {
    "wing.area": "m2",
    "wing.span": "m",
    "wing.mac": "m",
    "wing.sweep": "deg",
    "wing.t/c": "nondimensional",
    "wing.root_chord": "m",
    "wing.tip_chord": "m",
    "wing.taper_ratio": "nondimensional",
    "wing.aspect_ratio": "nondimensional",
    "wing.dihedral": "deg",
    "horizontal_tail.area": "m2",
    "horizontal_tail.span": "m",
    "horizontal_tail.mac": "m",
    "horizontal_tail.root_chord": "m",
    "horizontal_tail.tip_chord": "m",
    "horizontal_tail.taper_ratio": "nondimensional",
    "horizontal_tail.sweep": "deg",
    "horizontal_tail.dihedral": "deg",
    "horizontal_tail.arm": "m",
    "horizontal_tail.elevator_area_ratio": "nondimensional",
    "vertical_tail.area": "m2",
    "vertical_tail.height": "m",
    "vertical_tail.mac": "m",
    "vertical_tail.root_chord": "m",
    "vertical_tail.tip_chord": "m",
    "vertical_tail.taper_ratio": "nondimensional",
    "vertical_tail.sweep": "deg",
    "vertical_tail.arm": "m",
    "vertical_tail.rudder_area_ratio": "nondimensional",
}

TSV_COLUMNS = [
    "icao",
    "represented_model",
    "configuration",
    "field",
    "value",
    "unit",
    "status",
    "source_id",
    "source_url",
    "document_title",
    "revision_date",
    "page_section",
    "derivation_formula",
    "input_values",
    "applicability",
    "uncertainty",
    "notes",
]

EXCLUSION_COLUMNS = ["icao", "field", "restore_value", "reason"]

RENDER_KEYS = [
    ("status", "Status"),
    ("source", "Source"),
    ("source_url", "Source URL"),
    ("document_title", "Document title"),
    ("revision_date", "Revision date"),
    ("page_or_section", "Page or section"),
    ("formula", "Formula"),
    ("inputs", "Inputs"),
    ("applicability", "Applicability"),
    ("uncertainty", "Uncertainty"),
    ("previous_value", "Previous value"),
    ("correction_reason", "Correction reason"),
    ("reason_unavailable", "Reason unavailable"),
    ("searched_source", "Searched source"),
    ("search_record", "Search record"),
    ("notes", "Notes"),
]


def _get(document: Mapping[str, Any], path: str) -> Any:
    section, field = path.split(".", 1)
    return document[section][field]


def _set(document: dict[str, Any], path: str, value: Any) -> None:
    section, field = path.split(".", 1)
    document[section][field] = value


def _parse_number(text: str) -> float | None:
    stripped = text.strip()
    if not stripped or stripped.lower() == "null":
        return None
    value = float(stripped)
    if not math.isfinite(value):
        raise ValueError(f"non-finite geometry value: {text}")
    return value


def load_research(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    records: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            lines = (line for line in stream if not line.startswith("#"))
            reader = csv.DictReader(lines, delimiter="\t")
            if reader.fieldnames != TSV_COLUMNS:
                raise ValueError(f"{path}: unexpected TSV columns {reader.fieldnames}")
            for line_number, row in enumerate(reader, start=2):
                icao = row["icao"].strip().upper()
                field = row["field"].strip()
                if field not in GEOMETRY_UNITS:
                    raise ValueError(f"{path}:{line_number}: unsupported field {field}")
                unit = row["unit"].strip()
                if unit != GEOMETRY_UNITS[field]:
                    raise ValueError(
                        f"{path}:{line_number}: {field} uses {unit!r}, expected {GEOMETRY_UNITS[field]!r}"
                    )
                key = (icao, field)
                if key in records:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate value for {icao} {field}"
                    )
                records[key] = {
                    name: (value or "").strip() for name, value in row.items()
                }
    return records


def load_search_records(paths: list[Path]) -> dict[str, list[str]]:
    """Read structured plain-text ``# SEARCH CODE[,/CODE]: ...`` comments."""
    records: dict[str, list[str]] = {}
    pattern = re.compile(r"^# SEARCH ([A-Z0-9,/]+):\s*(.+)$")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            codes = [code for code in re.split(r"[,/]", match.group(1)) if code]
            note = match.group(2).strip()
            for code in codes:
                if note not in records.setdefault(code, []):
                    records[code].append(note)
    return records


def load_exclusions(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    records: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        lines = (line for line in stream if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(lines, delimiter="\t")
        if reader.fieldnames != EXCLUSION_COLUMNS:
            raise ValueError(
                f"{path}: unexpected exclusion columns {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            icao = row["icao"].strip().upper()
            field = row["field"].strip()
            if field not in GEOMETRY_UNITS:
                raise ValueError(f"{path}:{line_number}: unsupported field {field}")
            key = (icao, field)
            if key in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate exclusion for {icao} {field}"
                )
            records[key] = {name: (value or "").strip() for name, value in row.items()}
    return records


def _source_ids(metadata: Mapping[str, str]) -> list[str]:
    source = metadata.get("source", metadata.get("source_id", ""))
    return [
        item.strip()
        for item in source.split("|")
        if item.strip() and item.strip() != "-"
    ]


def _input_metadata(
    path: str, value: Any, existing: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    metadata = dict(existing.get(path, {}))
    if value is not None and not metadata:
        metadata = {
            "status": "legacy-unverified",
            "source": "aerox_legacy_aircraft_database",
        }
    return metadata


def _derived_metadata(
    formula: str,
    inputs: str,
    input_records: list[Mapping[str, str]],
) -> dict[str, str]:
    source_ids: list[str] = []
    statuses = []
    for record in input_records:
        statuses.append(record.get("status", "legacy-unverified"))
        for source_id in _source_ids(record):
            if source_id not in source_ids:
                source_ids.append(source_id)
    published = statuses and all(
        status
        in {
            "certified-published",
            "manufacturer-published",
            "manufacturer-published-planning",
            "calculated-from-published-inputs",
        }
        for status in statuses
    )
    inferred = (
        statuses
        and all(status != "legacy-unverified" for status in statuses)
        and any(status == "inferred-with-uncertainty" for status in statuses)
    )
    return {
        "status": (
            "calculated-from-published-inputs"
            if published
            else "inferred-with-uncertainty"
            if inferred
            else "legacy-unverified"
        ),
        "source": " | ".join(source_ids) or "aerox_legacy_aircraft_database",
        "formula": formula,
        "inputs": inputs,
    }


def _keep_stored(
    path: str,
    exact: float,
    input_paths: tuple[str, ...],
    document: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, str]],
    original_values: Mapping[str, Any],
    original_metadata: Mapping[str, Mapping[str, str]] | None,
    rel_tol: float,
) -> bool:
    """Return True when the note already documents a consistent derived value.

    The stored value and its note block are kept when the value agrees with the
    derivation and neither the field nor its inputs changed in this run.  This
    keeps hand-written provenance, for example a published taper ratio from
    which the chords were derived.
    """
    if original_metadata is None or path not in original_metadata:
        return False
    stored = original_values.get(path)
    if stored is None:
        return False
    for field in (path, *input_paths):
        if _get(document, field) != original_values.get(field) or metadata.get(
            field
        ) != original_metadata.get(field):
            return False
    return math.isclose(float(stored), exact, rel_tol=rel_tol)


def derive_geometry(
    document: dict[str, Any],
    metadata: dict[str, dict[str, str]],
    original_values: Mapping[str, Any] | None = None,
    original_metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Derive aspect and taper ratios from their inputs.

    With ``original_metadata`` (the note blocks before this run), a stored
    derived value that agrees with its inputs keeps its value and note block.
    """
    original_values = original_values or {}
    wing = document["wing"]
    exact = (
        float(wing["span"]) ** 2 / float(wing["area"])
        if wing["span"] is not None and wing["area"] is not None
        else None
    )
    if exact is not None and not _keep_stored(
        "wing.aspect_ratio",
        exact,
        ("wing.span", "wing.area"),
        document,
        metadata,
        original_values,
        original_metadata,
        2e-6,
    ):
        derived = round(exact, 6)
        wing["aspect_ratio"] = derived
        record = _derived_metadata(
            "span^2 / area",
            f"span={wing['span']} m; area={wing['area']} m2",
            [
                _input_metadata("wing.span", wing["span"], metadata),
                _input_metadata("wing.area", wing["area"], metadata),
            ],
        )
        prior_record = metadata.get("wing.aspect_ratio", {})
        for key in (
            "source_url",
            "document_title",
            "revision_date",
            "page_or_section",
            "applicability",
            "uncertainty",
            "searched_source",
            "search_record",
            "notes",
        ):
            if prior_record.get(key):
                record[key] = prior_record[key]
        for key in ("previous_value", "correction_reason"):
            if prior_record.get(key):
                record[key] = prior_record[key]
        previous = original_values.get("wing.aspect_ratio")
        if previous is not None and not math.isclose(
            float(previous), derived, rel_tol=2e-6
        ):
            record["previous_value"] = f"{previous} nondimensional"
            record["correction_reason"] = (
                "Recalculated after a cited wing span or area correction."
            )
        metadata["wing.aspect_ratio"] = record

    for section_name in ("wing", "horizontal_tail", "vertical_tail"):
        section = document[section_name]
        if section["root_chord"] is not None and section["tip_chord"] is not None:
            path = f"{section_name}.taper_ratio"
            exact = float(section["tip_chord"]) / float(section["root_chord"])
            if _keep_stored(
                path,
                exact,
                (f"{section_name}.tip_chord", f"{section_name}.root_chord"),
                document,
                metadata,
                original_values,
                original_metadata,
                3e-6,
            ):
                continue
            derived = round(exact, 6)
            section["taper_ratio"] = derived
            record = _derived_metadata(
                "tip_chord / root_chord",
                f"tip_chord={section['tip_chord']} m; root_chord={section['root_chord']} m",
                [
                    _input_metadata(
                        f"{section_name}.tip_chord", section["tip_chord"], metadata
                    ),
                    _input_metadata(
                        f"{section_name}.root_chord", section["root_chord"], metadata
                    ),
                ],
            )
            prior_record = metadata.get(path, {})
            for key in (
                "source_url",
                "document_title",
                "revision_date",
                "page_or_section",
                "applicability",
                "uncertainty",
                "searched_source",
                "search_record",
                "notes",
            ):
                if prior_record.get(key):
                    record[key] = prior_record[key]
            for key in ("previous_value", "correction_reason"):
                if prior_record.get(key):
                    record[key] = prior_record[key]
            previous = original_values.get(path)
            if previous is not None and not math.isclose(
                float(previous), derived, rel_tol=3e-6
            ):
                record["previous_value"] = f"{previous} nondimensional"
                record["correction_reason"] = (
                    "Recalculated after a cited chord correction."
                )
            metadata[path] = record


def _metadata_from_row(row: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "status": row.get("status", ""),
            "source": row.get("source_id", ""),
            "source_url": row.get("source_url", ""),
            "document_title": row.get("document_title", ""),
            "revision_date": row.get("revision_date", ""),
            "page_or_section": row.get("page_section", ""),
            "formula": row.get("derivation_formula", ""),
            "inputs": row.get("input_values", ""),
            "applicability": row.get("applicability", ""),
            "uncertainty": row.get("uncertainty", ""),
            "notes": row.get("notes", ""),
        }.items()
        if value
    }


def _certification_search_source(fields: Mapping[str, Mapping[str, str]]) -> str:
    return fields.get("certification.type_certificate", {}).get("source", "-")


def render_geometry_section(
    document: Mapping[str, Any],
    old_text: str,
    metadata: Mapping[str, Mapping[str, str]],
) -> str:
    marker = "WING AND TAIL GEOMETRY\n"
    legacy_marker = "LEGACY PARAMETERS\n"
    if marker in old_text:
        # Replace only the geometry section: it ends at the next section
        # header, so the sections after it are kept.
        prefix, body = old_text.split(marker, 1)
        header = _SECTION_HEADER.search(body)
        suffix = body[header.start() :] if header else ""
    elif legacy_marker in old_text:
        prefix, suffix_body = old_text.split(legacy_marker, 1)
        suffix = legacy_marker + suffix_body
    else:
        prefix, suffix = old_text, ""

    existing = parse_old_provenance(old_text)["fields"]
    searched_source = _certification_search_source(existing)
    lines = ["WING AND TAIL GEOMETRY", ""]
    for path, unit in GEOMETRY_UNITS.items():
        value = _get(document, path)
        record = dict(metadata.get(path, existing.get(path, {})))
        if value is not None and not record:
            record = {
                "status": "legacy-unverified",
                "source": "aerox_legacy_aircraft_database",
            }
        if value is None:
            record.setdefault("status", "unresolved")
            record.setdefault("source", "-")
            record.setdefault(
                "reason_unavailable",
                "No exact-model public value has been adopted.",
            )
            record.setdefault("searched_source", searched_source)

        value_text = "null" if value is None else str(value)
        old_value_text = existing.get(path, {}).get("value")
        if value is not None and old_value_text is not None:
            try:
                if float(old_value_text) == float(value):
                    # Keep the hand-written number format, e.g. "1.434000".
                    value_text = old_value_text
            except ValueError:
                pass
        lines.extend(
            [
                f"FIELD {path}",
                f"  Value: {value_text}",
                f"  Unit: {unit}",
            ]
        )
        for key, label in RENDER_KEYS:
            if record.get(key):
                lines.append(f"  {label}: {record[key]}")
        lines.extend(["END FIELD", ""])
    return f"{prefix.rstrip()}\n\n{'\n'.join(lines)}\n{suffix.lstrip()}"


def source_ledger_ids() -> set[str]:
    ledger = yaml.safe_load(SOURCE_LEDGER.read_text(encoding="utf-8"))
    return {record["id"] for record in ledger.get("sources", [])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, action="append", default=[])
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=DEFAULT_EXCLUSIONS,
        help="plain-text runtime applicability exclusions",
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    research = load_research(args.research)
    search_records = load_search_records(args.research)
    exclusions = load_exclusions(args.exclusions)
    ledger_ids = source_ledger_ids()
    unknown_sources = sorted(
        {
            source_id.strip()
            for row in research.values()
            for source_id in row["source_id"].split("|")
            if source_id.strip() and source_id.strip() not in ledger_ids
        }
    )
    if unknown_sources:
        raise ValueError(f"research references unknown source IDs: {unknown_sources}")

    changed: list[tuple[Path, str, Path, str]] = []
    files = sorted(AIRCRAFT_ROOT.glob("*/*.yml"))
    known_icaos = {path.stem.upper() for path in files}
    unexpected = sorted({icao for icao, _ in research} - known_icaos)
    if unexpected:
        raise ValueError(f"research references unknown aircraft: {unexpected}")

    for path in files:
        original = path.read_text(encoding="utf-8")
        document = yaml.safe_load(original)
        document, _ = normalize_bundle(document, "")
        original_values = {field: _get(document, field) for field in GEOMETRY_UNITS}
        sidecar_path = SOURCES_ROOT / path.parent.name / f"{path.stem}.txt"
        old_sidecar = sidecar_path.read_text(encoding="utf-8")
        metadata = {
            key: dict(value)
            for key, value in parse_old_provenance(old_sidecar)["fields"].items()
            if key in GEOMETRY_UNITS
        }
        note_metadata = {key: dict(value) for key, value in metadata.items()}

        searched_sources: list[str] = []
        aircraft_search = " | ".join(search_records.get(path.stem.upper(), []))
        certification_sources = _source_ids(
            {
                "source": _certification_search_source(
                    parse_old_provenance(old_sidecar)["fields"]
                )
            }
        )
        for (icao, _), row in research.items():
            if icao == path.stem.upper():
                for source_id in row["source_id"].split("|"):
                    source_id = source_id.strip()
                    if source_id and source_id not in searched_sources:
                        searched_sources.append(source_id)

        for (icao, field), exclusion in exclusions.items():
            if icao != path.stem.upper() or (icao, field) not in research:
                continue
            restore_value = _parse_number(exclusion["restore_value"])
            _set(document, field, restore_value)
            candidate = research[(icao, field)]
            if restore_value is None:
                metadata[field] = {
                    "status": "unresolved",
                    "source": "-",
                    "reason_unavailable": (
                        "A public candidate was found but was not adopted because its "
                        "model or configuration does not match the bundled record."
                    ),
                    "searched_source": candidate["source_id"],
                    "applicability": candidate.get("applicability", ""),
                    "notes": exclusion["reason"],
                }
            else:
                metadata[field] = {
                    "status": "legacy-unverified",
                    "source": "aerox_legacy_aircraft_database",
                    "searched_source": candidate["source_id"],
                    "applicability": candidate.get("applicability", ""),
                    "notes": exclusion["reason"],
                }

        for (icao, field), row in research.items():
            if icao != path.stem.upper():
                continue
            if (icao, field) in exclusions:
                continue
            previous = _get(document, field)
            replacement = _parse_number(row["value"])
            prior_record = metadata.get(field, {})
            shared_sources = set(_source_ids(prior_record)) & {
                source_id.strip()
                for source_id in row["source_id"].split("|")
                if source_id.strip()
            }
            if (
                previous is not None
                and replacement is not None
                and shared_sources
                and math.isclose(
                    float(previous), float(replacement), rel_tol=1e-5, abs_tol=1e-4
                )
            ):
                # Retain a more precise value already derived from the same evidence.
                replacement = previous
            _set(document, field, replacement)
            record = _metadata_from_row(row)
            for key in ("previous_value", "correction_reason"):
                if prior_record.get(key):
                    record[key] = prior_record[key]
            if (
                previous is not None
                and replacement is not None
                and not math.isclose(
                    float(previous), float(replacement), rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                record["previous_value"] = f"{previous} {GEOMETRY_UNITS[field]}"
                record["correction_reason"] = (
                    "Replaced the legacy aerox value with the cited exact-scope value."
                )
            elif replacement is not None and record.get("previous_value"):
                try:
                    recorded_previous = float(record["previous_value"].split()[0])
                except ValueError:
                    recorded_previous = math.nan
                if math.isclose(
                    recorded_previous,
                    float(replacement),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    record.pop("previous_value", None)
                    record.pop("correction_reason", None)
            metadata[field] = record
        for field in GEOMETRY_UNITS:
            if _get(document, field) is None and searched_sources:
                record = metadata.setdefault(field, {})
                # An existing search list is kept; research sources are added.
                # The certification source is only the default for a new list.
                combined = _source_ids(
                    {"source": record.get("searched_source", "")}
                ) or list(certification_sources)
                for source_id in searched_sources:
                    if source_id not in combined:
                        combined.append(source_id)
                record["searched_source"] = " | ".join(combined)
            if _get(document, field) is None and aircraft_search:
                metadata.setdefault(field, {})["search_record"] = aircraft_search
        derive_geometry(document, metadata, original_values, note_metadata)
        for (icao, field), exclusion in exclusions.items():
            if icao != path.stem.upper() or (icao, field) not in research:
                continue
            if (
                _parse_number(exclusion["restore_value"]) is None
                and _get(document, field) is not None
            ):
                candidate = research[(icao, field)]
                record = metadata[field]
                record["applicability"] = (
                    "Bundled record; calculated from restored legacy values."
                )
                record["searched_source"] = candidate["source_id"]
                record["notes"] = exclusion["reason"]
                for key in (
                    "source_url",
                    "document_title",
                    "revision_date",
                    "page_or_section",
                    "uncertainty",
                ):
                    record.pop(key, None)
        issues = validate_simple(document)
        if issues:
            raise ValueError(f"{path}: " + "; ".join(issues))
        rendered = dump_document(document)
        sidecar = render_geometry_section(document, old_sidecar, metadata)
        if rendered != original or sidecar != old_sidecar:
            changed.append((path, rendered, sidecar_path, sidecar))

    if args.write:
        for path, rendered, sidecar_path, sidecar in changed:
            path.write_text(rendered, encoding="utf-8")
            sidecar_path.write_text(sidecar, encoding="utf-8")
    mode = "write" if args.write else "check"
    print(
        f"aircraft={len(files)} research_values={len(research)} changed={len(changed)} mode={mode}"
    )
    if changed and not args.write:
        print(
            "changed_files=" + ",".join(path.stem.upper() for path, _, _, _ in changed)
        )
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
