#!/usr/bin/env python3
"""Apply unique certified subtype-to-ICAO-EEDB mappings to compact YAML.

The input is a tab-separated plain-text inventory.  TCDSs establish aircraft
subtype applicability; the ICAO Engine Emissions Databank establishes the
queryable engine identifier spelling.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

try:
    from migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        SOURCES_ROOT,
        dump_document,
        normalize_bundle,
        validate_simple,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        SOURCES_ROOT,
        dump_document,
        normalize_bundle,
        validate_simple,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LEDGER = PROJECT_ROOT / "docs/modeling/sources.yml"
DEFAULT_RESEARCH = (
    PROJECT_ROOT / "docs/aircraft-data/airbus-subtype-engine-mapping.tsv",
)
ENGINE_TABLE = PROJECT_ROOT / "src/aerox/data/engines/engines.csv"
OLD_SECTION_MARKER = "MODEL AND ENGINE APPLICABILITY\n"
SECTION_MARKER = "SUBTYPE AND ENGINE DATABANK APPLICABILITY\n"
EEDB_SOURCE_ID = "icao_eedb_issue_32"
REVIEW_DATE = "2026-08-06"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _split_engines(text: str) -> list[str]:
    separator = "|" if "|" in text else ";"
    return _unique([item.strip() for item in text.split(separator) if item.strip()])


def load_research(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and normalize rows while retaining the source document order."""

    result: list[dict[str, Any]] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        header_index = next(
            (index for index, line in enumerate(lines) if line.startswith("icao\t")),
            None,
        )
        if header_index is None:
            raise ValueError(f"{path}: missing tab-separated icao header")
        reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter="\t")
        for line_number, row in enumerate(reader, start=header_index + 2):
            if not row or not (row.get("icao") or "").strip():
                continue
            engine_text = row.get("edb_engine(s)", "") or ""
            source_id = (row.get("source_id") or "").strip()
            if not source_id:
                raise ValueError(f"{path}:{line_number}: missing source_id")
            result.append(
                {
                    "icao": row["icao"].strip().upper(),
                    "subtype": (row.get("subtype") or "").strip(),
                    "engines": _split_engines(engine_text),
                    "source_id": source_id,
                    "page_or_section": (row.get("page_or_section", row.get("page/section", "")) or "").strip(),
                    "applicability": (row.get("applicability") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                    "input": path.relative_to(PROJECT_ROOT).as_posix(),
                    "line": line_number,
                }
            )
    return result


def group_mappings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group exact subtypes for each ICAO in the source document order."""

    grouped: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    for row in rows:
        subtypes = grouped.setdefault(row["icao"], OrderedDict())
        if not row["subtype"]:
            raise ValueError(f"{row['input']}:{row['line']}: missing subtype")
        if not row["engines"]:
            raise ValueError(f"{row['input']}:{row['line']}: missing EEDB engine")
        engines = subtypes.setdefault(row["subtype"], [])
        engines.extend(engine for engine in row["engines"] if engine not in engines)
    return {
        icao: [
            {subtype: engines[0] if len(engines) == 1 else engines}
            for subtype, engines in subtypes.items()
        ]
        for icao, subtypes in grouped.items()
    }


def render_sources(old_text: str, rows: list[dict[str, Any]], mapping: Any) -> str:
    """Append a compact human-readable source block, replacing an old block."""

    base = old_text.split(OLD_SECTION_MARKER, 1)[0]
    base = base.split(SECTION_MARKER, 1)[0].rstrip()
    if not rows:
        return f"{base}\n"
    sources = _unique([row["source_id"] for row in rows])
    pages = _unique([row["page_or_section"] for row in rows])
    lines = [
        "SUBTYPE AND ENGINE DATABANK APPLICABILITY",
        "",
        "FIELD subtypes",
        f"  Value: {len(mapping)} unique certified subtype mapping(s)",
        "  Status: certified-published",
        f"  Source: {' | '.join([*sources, EEDB_SOURCE_ID])}",
        f"  Page or section: {' | '.join(pages)}",
        "  Applicability: Certified subtype applicability with query identifiers normalized to ICAO EEDB Issue 32.",
        "  Evidence inventory: " + " | ".join(_unique([row["input"] for row in rows])),
        "  Engine table: src/aerox/data/engines/engines.csv",
    ]
    lines.append("  Mappings:")
    for item in mapping:
        subtype, engines = next(iter(item.items()))
        engines = [engines] if isinstance(engines, str) else engines
        lines.append(f"    {subtype}:")
        lines.extend(f"      - {engine}" for engine in engines)
    lines.extend(["END FIELD", ""])
    rendered = f"{base}\n\n" + "\n".join(lines)
    rendered = rendered.replace("Last reviewed: 2026-08-05", f"Last reviewed: {REVIEW_DATE}")
    return rendered


def source_ids() -> set[str]:
    ledger = yaml.safe_load(SOURCE_LEDGER.read_text(encoding="utf-8"))
    return {record["id"] for record in ledger.get("sources", [])}


def edb_engine_ids() -> set[str]:
    with ENGINE_TABLE.open(encoding="utf-8", newline="") as stream:
        return {row["name"] for row in csv.DictReader(stream)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = args.research or list(DEFAULT_RESEARCH)
    rows = load_research(inputs)
    mappings = group_mappings(rows)
    unknown_sources = sorted(
        ({row["source_id"] for row in rows} | {EEDB_SOURCE_ID}) - source_ids()
    )
    if unknown_sources:
        raise ValueError(f"research references unknown source IDs: {unknown_sources}")

    files = sorted(AIRCRAFT_ROOT.glob("*/*.yml"))
    by_icao = {path.stem.upper(): path for path in files}
    unknown_aircraft = sorted(set(mappings) - set(by_icao))
    if unknown_aircraft:
        raise ValueError(f"research references unknown aircraft: {unknown_aircraft}")
    unknown_engines = sorted(
        {engine for row in rows for engine in row["engines"]} - edb_engine_ids()
    )
    if unknown_engines:
        raise ValueError(f"subtype mappings use non-ICAO-EEDB engines: {unknown_engines}")

    changed: list[tuple[Path, str, Path, str]] = []
    for icao, path in by_icao.items():
        mapping = mappings.get(icao)
        original = path.read_text(encoding="utf-8")
        document, _ = normalize_bundle(yaml.safe_load(original), "")
        document["subtypes"] = mapping
        issues = validate_simple(document)
        if issues:
            raise ValueError(f"{path}: " + "; ".join(issues))
        rendered = dump_document(document)
        sidecar_path = SOURCES_ROOT / path.parent.name / f"{path.stem}.txt"
        old_sidecar = sidecar_path.read_text(encoding="utf-8")
        aircraft_rows = [row for row in rows if row["icao"] == icao]
        sidecar = render_sources(old_sidecar, aircraft_rows, mapping)
        if rendered != original or sidecar != old_sidecar:
            changed.append((path, rendered, sidecar_path, sidecar))

    if args.write:
        for path, rendered, sidecar_path, sidecar in changed:
            path.write_text(rendered, encoding="utf-8")
            sidecar_path.write_text(sidecar, encoding="utf-8")
    mode = "write" if args.write else "check"
    mapped_subtypes = sum(len(mapping) for mapping in mappings.values())
    print(
        f"aircraft={len(files)} researched={len(mappings)} mapped_subtypes={mapped_subtypes} "
        f"changed={len(changed)} mode={mode}"
    )
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
