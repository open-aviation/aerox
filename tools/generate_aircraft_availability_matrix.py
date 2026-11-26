#!/usr/bin/env python3
"""Generate the bundled-aircraft parameter-availability matrix.

The CSV contains one row per aircraft and one binary column per compact YAML
parameter. A cell is ``1`` when the parameter has a non-null, non-empty value
and ``0`` otherwise. The aircraft name and ICAO code identify each row and are
therefore not repeated as availability columns.
The named high-lift settings table counts as one parameter.
"""

from __future__ import annotations

import argparse
import csv
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from migrate_aircraft_data import AIRCRAFT_ROOT, SIMPLE_TEMPLATE, validate_simple
except ModuleNotFoundError:  # package-style import used by tests
    from tools.migrate_aircraft_data import (
        AIRCRAFT_ROOT,
        SIMPLE_TEMPLATE,
        validate_simple,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "src/aerox/data/aircraft/aircraft-parameter-availability.csv"
)
IDENTIFIER_PATHS = {"aircraft", "icao"}
SUMMARY_COLUMNS = [
    "icao",
    "aircraft",
    "propulsion_category",
    "yaml_file",
    "available_parameter_count",
    "total_parameter_count",
    "availability_percent",
]


def parameter_paths(
    template: Mapping[str, Any] = SIMPLE_TEMPLATE,
    prefix: str = "",
) -> list[str]:
    """Return leaf paths in the canonical compact-schema order."""

    paths: list[str] = []
    for name, value in template.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping):
            paths.extend(parameter_paths(value, path))
        elif path not in IDENTIFIER_PATHS:
            paths.append(path)
    return paths


def get_path(document: Mapping[str, Any], path: str) -> Any:
    value: Any = document
    for name in path.split("."):
        value = value[name]
    return value


def is_available(value: Any) -> bool:
    """Treat null and empty containers/strings as unavailable; zero is data."""

    if value is None:
        return False
    if isinstance(value, (str, list, tuple, dict, set)) and not value:
        return False
    return True


def load_rows(aircraft_root: Path = AIRCRAFT_ROOT) -> list[dict[str, str]]:
    paths = parameter_paths()
    rows: list[dict[str, str]] = []
    seen_icao: set[str] = set()
    for yaml_path in sorted(aircraft_root.glob("*/*.yml")):
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        errors = validate_simple(document)
        if errors:
            raise ValueError(f"{yaml_path}: {'; '.join(errors)}")

        icao = str(document["icao"])
        if icao in seen_icao:
            raise ValueError(f"duplicate ICAO code in aircraft bundle: {icao}")
        seen_icao.add(icao)

        availability = {
            path: "1" if is_available(get_path(document, path)) else "0"
            for path in paths
        }
        available_count = sum(value == "1" for value in availability.values())
        rows.append(
            {
                "icao": icao,
                "aircraft": str(document["aircraft"]),
                "propulsion_category": yaml_path.parent.name,
                "yaml_file": yaml_path.relative_to(PROJECT_ROOT).as_posix(),
                "available_parameter_count": str(available_count),
                "total_parameter_count": str(len(paths)),
                "availability_percent": f"{100 * available_count / len(paths):.2f}",
                **availability,
            }
        )
    return sorted(rows, key=lambda row: row["icao"])


def render_matrix(aircraft_root: Path = AIRCRAFT_ROOT) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[*SUMMARY_COLUMNS, *parameter_paths()],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(load_rows(aircraft_root))
    return output.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aircraft-root", type=Path, default=AIRCRAFT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing CSV differs from freshly generated output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rendered = render_matrix(args.aircraft_root)
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            print(f"stale aircraft availability matrix: {args.output}")
            return 1
        print(f"aircraft availability matrix is current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {len(load_rows(args.aircraft_root))} aircraft x "
        f"{len(parameter_paths())} parameters to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
