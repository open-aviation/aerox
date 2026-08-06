#!/usr/bin/env python3
"""Apply certified aircraft-model to approved-engine mappings to compact YAML.

Research inputs are tab-separated plain-text inventories.  Comment lines may
precede the header.  Airbus inventories use ``engine(s)`` with semicolon
separation; Boeing inventories use ``engines`` with pipe separation.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        PROVENANCE_ROOT,
        dump_document,
        normalize_bundle,
        validate_simple,
    )
except ModuleNotFoundError:  # package-style import used by tests
    from tools.migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        PROVENANCE_ROOT,
        dump_document,
        normalize_bundle,
        validate_simple,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LEDGER = PROJECT_ROOT / "docs/modeling/sources.yml"
DEFAULT_RESEARCH = (
    PROJECT_ROOT / "docs/modeling/airbus-model-engine-mapping-draft.tsv",
    PROJECT_ROOT / "docs/modeling/boeing-model-engine-mapping-draft.tsv",
)
SECTION_MARKER = "MODEL AND ENGINE APPLICABILITY\n"
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
            engine_text = row.get("engines", row.get("engine(s)", "")) or ""
            source_id = (row.get("source_id") or "").strip()
            if not source_id:
                raise ValueError(f"{path}:{line_number}: missing source_id")
            result.append(
                {
                    "icao": row["icao"].strip().upper(),
                    "model": (row.get("model") or "").strip(),
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


def group_mappings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]] | None]:
    """Group exact models for each ICAO; all-unresolved records become null."""

    grouped: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    for row in rows:
        models = grouped.setdefault(row["icao"], OrderedDict())
        if not row["model"]:
            raise ValueError(f"{row['input']}:{row['line']}: missing model")
        if not row["engines"]:
            continue
        engines = models.setdefault(row["model"], [])
        engines.extend(engine for engine in row["engines"] if engine not in engines)
    affected = _unique([row["icao"] for row in rows])
    return {
        icao: (
            [{"model": model, "engines": engines} for model, engines in grouped.get(icao, {}).items()]
            or None
        )
        for icao in affected
    }


def render_provenance(old_text: str, rows: list[dict[str, Any]], mapping: Any) -> str:
    """Append a compact human-readable source block, replacing an old block."""

    base = old_text.split(SECTION_MARKER, 1)[0].rstrip()
    sources = _unique([row["source_id"] for row in rows])
    pages = _unique([row["page_or_section"] for row in rows])
    lines = [
        "MODEL AND ENGINE APPLICABILITY",
        "",
        "FIELD engine.model_engine_options",
        f"  Value: {'null' if mapping is None else str(len(mapping)) + ' certified model mapping(s)'}",
        f"  Status: {'unresolved' if mapping is None else 'certified-published'}",
        f"  Source: {' | '.join(sources)}",
        f"  Page or section: {' | '.join(pages)}",
        "  Applicability: Model-level approved installations; the individual airframe record is required to identify the engine actually installed.",
        "  Evidence inventory: " + " | ".join(_unique([row["input"] for row in rows])),
    ]
    if mapping is None:
        lines.append("  Reason unavailable: The reviewed certification issue does not list an approved engine installation for this model.")
        notes = _unique([row["notes"] for row in rows])
        if notes:
            lines.append("  Notes: " + " | ".join(notes))
    else:
        lines.append("  Mappings:")
        for item in mapping:
            lines.append(f"    {item['model']}:")
            lines.extend(f"      - {engine}" for engine in item["engines"])
    lines.extend(["END FIELD", ""])
    rendered = f"{base}\n\n" + "\n".join(lines)
    rendered = rendered.replace("Last reviewed: 2026-08-05", f"Last reviewed: {REVIEW_DATE}")
    return rendered


def source_ids() -> set[str]:
    ledger = yaml.safe_load(SOURCE_LEDGER.read_text(encoding="utf-8"))
    return {record["id"] for record in ledger.get("sources", [])}


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
    unknown_sources = sorted({row["source_id"] for row in rows} - source_ids())
    if unknown_sources:
        raise ValueError(f"research references unknown source IDs: {unknown_sources}")

    files = sorted(AIRCRAFT_ROOT.glob("*/*.yml"))
    by_icao = {path.stem.upper(): path for path in files}
    unknown_aircraft = sorted(set(mappings) - set(by_icao))
    if unknown_aircraft:
        raise ValueError(f"research references unknown aircraft: {unknown_aircraft}")

    changed: list[tuple[Path, str, Path, str]] = []
    for icao, mapping in mappings.items():
        path = by_icao[icao]
        original = path.read_text(encoding="utf-8")
        document, _ = normalize_bundle(yaml.safe_load(original), "")
        document["engine"]["model_engine_options"] = mapping
        issues = validate_simple(document)
        if issues:
            raise ValueError(f"{path}: " + "; ".join(issues))
        rendered = dump_document(document)
        sidecar_path = PROVENANCE_ROOT / path.parent.name / f"{path.stem}.txt"
        old_sidecar = sidecar_path.read_text(encoding="utf-8")
        aircraft_rows = [row for row in rows if row["icao"] == icao]
        sidecar = render_provenance(old_sidecar, aircraft_rows, mapping)
        if rendered != original or sidecar != old_sidecar:
            changed.append((path, rendered, sidecar_path, sidecar))

    if args.write:
        for path, rendered, sidecar_path, sidecar in changed:
            path.write_text(rendered, encoding="utf-8")
            sidecar_path.write_text(sidecar, encoding="utf-8")
    mode = "write" if args.write else "check"
    mapped_models = sum(len(mapping or []) for mapping in mappings.values())
    print(
        f"aircraft={len(files)} researched={len(mappings)} mapped_models={mapped_models} "
        f"changed={len(changed)} mode={mode}"
    )
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
