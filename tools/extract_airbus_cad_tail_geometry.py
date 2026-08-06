#!/usr/bin/env python3
"""Extract A320-family horizontal-tail priors from Airbus DWG JSON.

LibreDWG's direct JSON writer preserves entities that its R2000 DXF writer
drops from several Airbus drawings.  This tool either reads those JSON files
or invokes ``dwgread -O JSON`` itself.  It deliberately uses reviewed entity
handles: automatic outline recognition is not reliable enough to establish
which of the many drawing lines defines an aerodynamic reference planform.

The published tail span supplies the tip station.  Chords remain in the
drawing's native millimetre scale; the small gap between the straight-line
endpoints and published span represents the rounded terminal cap and is not
used as a scale factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAD_ROOT = PROJECT_ROOT / "data/cad/airbus/drawings"


@dataclass(frozen=True)
class Drawing:
    icao: str
    configuration: str
    source_id: str
    directory: str
    dwg_name: str
    json_name: str
    span_m: float
    leading_left: str
    leading_right: str
    trailing_left: str
    trailing_right: str
    hinge_left: str
    hinge_right: str
    expected_elevator_ratio: float


DRAWINGS = (
    Drawing(
        "A318", "wing-tip-fence", "airbus_a318_autocad_wtf_2012_05",
        "Airbus_A318_WTF-May-2012", "Airbus_A318_WTF.dwg",
        "Airbus_A318_WTF.json", 12.45,
        "63A", "80A", "63B", "80B", "2AC3", "2AAD", 0.485646,
    ),
    Drawing(
        "A319", "wing-tip-fence", "airbus_a319_autocad_wtf_2012_05",
        "Airbus_A319_WTF-May-2012", "Airbus_A319_WTF.dwg",
        "Airbus_A319_WTF.json", 12.45,
        "2B0", "549A", "545A", "5457", "451C", "4508", 0.475596,
    ),
    Drawing(
        "A320", "wing-tip-fence", "airbus_a320_autocad_wtf_2012_05",
        "Airbus_A320_WTF-May-2012", "Airbus_A320_WTF.dwg",
        "Airbus_A320_WTF.json", 12.45,
        "5653", "31F", "55DF", "55E2", "5006", "501A", 0.480561,
    ),
    Drawing(
        "A321", "wing-tip-fence", "airbus_a321_autocad_wtf_2012_05",
        "Airbus_A321_WTF-May-2012", "Airbus_A321_WTF.dwg",
        "Airbus_A321_WTF.json", 12.45,
        "6EF9", "7078", "6EFA", "7079", "689C", "6869", 0.480561,
    ),
    Drawing(
        "A19N", "sharklet outer-geometry candidate",
        "airbus_a319_autocad_sharklet_2014_07",
        "A319_sharklet-001-Jul-2014", "A319_sharklet-001.dwg",
        "A319_sharklet-001.json", 12.45,
        "82F", "D87", "830", "D88", "58F6", "58E2", 0.475596,
    ),
    Drawing(
        "A20N", "sharklet outer-geometry candidate",
        "airbus_a320_autocad_sharklet_2014_07",
        "A320_sharklet-001-Jul-2014_1", "A320_sharklet-001.dwg",
        "A320_sharklet-001.json", 12.45,
        "18E8", "19B8", "6D65", "6D67", "5F0F", "5EF9", 0.480560,
    ),
    Drawing(
        "A21N", "sharklet outer-geometry candidate",
        "airbus_a321_autocad_sharklet_2014_07",
        "A321_sharklet-001-Jul-2014", "A321_sharklet-001.dwg",
        "A321_sharklet-001.json", 12.45,
        "7382", "744D", "834D", "83B3", "6FFA", "6FE5", 0.480561,
    ),
)


def handle_value(entity: dict[str, Any]) -> int:
    handle = entity.get("handle")
    if not isinstance(handle, list) or not handle:
        raise ValueError("entity has no LibreDWG handle")
    return int(handle[-1])


def line_entities(document: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Index LINE entities, ignoring non-entity objects with duplicate handles."""
    entities: dict[int, list[dict[str, Any]]] = {}
    for item in document.get("OBJECTS", []):
        if item.get("entity") != "LINE":
            continue
        handle = handle_value(item)
        entities.setdefault(handle, []).append(item)
    return entities


def line_for(entities: dict[int, list[dict[str, Any]]], handle: str) -> tuple[tuple[float, float], tuple[float, float]]:
    try:
        candidates = entities[int(handle, 16)]
    except KeyError as exc:
        raise ValueError(f"reviewed LINE handle {handle} is absent") from exc
    geometries = {
        (tuple(entity["start"][:2]), tuple(entity["end"][:2]))
        for entity in candidates
    }
    if len(geometries) != 1:
        raise ValueError(f"reviewed LINE handle {handle} has conflicting duplicates")
    return geometries.pop()


def ordered(line: tuple[tuple[float, float], tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return endpoints ordered by the lateral (first-coordinate) station."""
    return tuple(sorted(line, key=lambda point: point[0]))  # type: ignore[return-value]


def interpolate(line: tuple[tuple[float, float], tuple[float, float]], station: float) -> float:
    (x1, y1), (x2, y2) = line
    if x1 == x2:
        raise ValueError("cannot evaluate a line normal to the lateral axis")
    return y1 + (station - x1) * (y2 - y1) / (x2 - x1)


def extract(document: dict[str, Any], drawing: Drawing) -> dict[str, float | str]:
    entities = line_entities(document)
    le_left = ordered(line_for(entities, drawing.leading_left))
    le_right = ordered(line_for(entities, drawing.leading_right))
    te_left = ordered(line_for(entities, drawing.trailing_left))
    te_right = ordered(line_for(entities, drawing.trailing_right))
    hinge_left = ordered(line_for(entities, drawing.hinge_left))
    hinge_right = ordered(line_for(entities, drawing.hinge_right))

    # Inner endpoints are the right endpoint of the left half and left endpoint
    # of the right half.  Each mirrored edge supplies an independent centre.
    centre_candidates = (
        (le_left[1][0] + le_right[0][0]) / 2,
        (te_left[1][0] + te_right[0][0]) / 2,
        (hinge_left[1][0] + hinge_right[0][0]) / 2,
    )
    centre = sum(centre_candidates) / len(centre_candidates)
    symmetry_error_mm = max(centre_candidates) - min(centre_candidates)
    if symmetry_error_mm > 20:
        raise ValueError(
            f"{drawing.icao}: selected edges disagree on symmetry centre by "
            f"{symmetry_error_mm:.3f} mm"
        )

    tip = centre + drawing.span_m * 500
    root_chord = abs(interpolate(te_right, centre) - interpolate(le_right, centre)) / 1000
    tip_chord = abs(interpolate(te_right, tip) - interpolate(le_right, tip)) / 1000
    taper = tip_chord / root_chord
    area = drawing.span_m * (root_chord + tip_chord) / 2
    mac = (2 / 3) * root_chord * (1 + taper + taper * taper) / (1 + taper)

    root_le = interpolate(le_right, centre)
    root_te = interpolate(te_right, centre)
    tip_le = interpolate(le_right, tip)
    tip_te = interpolate(te_right, tip)
    root_quarter = root_le + 0.25 * (root_te - root_le)
    tip_quarter = tip_le + 0.25 * (tip_te - tip_le)
    sweep = math.degrees(math.atan2(abs(tip_quarter - root_quarter), drawing.span_m * 500))

    elevator_root = abs(interpolate(te_right, centre) - interpolate(hinge_right, centre)) / 1000
    elevator_tip = abs(interpolate(te_right, tip) - interpolate(hinge_right, tip)) / 1000
    elevator_ratio = (elevator_root + elevator_tip) / (root_chord + tip_chord)

    endpoint_span_mm = (
        (le_right[1][0] - le_left[0][0])
        + (te_right[1][0] - te_left[0][0])
    ) / 2
    span_gap_mm = drawing.span_m * 1000 - endpoint_span_mm

    if abs(elevator_ratio - drawing.expected_elevator_ratio) > 1e-5:
        raise ValueError(
            f"{drawing.icao}: elevator ratio {elevator_ratio:.6f} does not "
            f"match reviewed value {drawing.expected_elevator_ratio:.6f}"
        )

    return {
        "icao": drawing.icao,
        "configuration": drawing.configuration,
        "source_id": drawing.source_id,
        "span_m": drawing.span_m,
        "root_chord_m": root_chord,
        "tip_chord_m": tip_chord,
        "area_m2": area,
        "taper_ratio": taper,
        "mac_m": mac,
        "quarter_chord_sweep_deg": sweep,
        "elevator_area_ratio": elevator_ratio,
        "symmetry_error_mm": symmetry_error_mm,
        "published_span_minus_straight_endpoints_mm": span_gap_mm,
        "leading_handles": f"{drawing.leading_left};{drawing.leading_right}",
        "trailing_handles": f"{drawing.trailing_left};{drawing.trailing_right}",
        "hinge_handles": f"{drawing.hinge_left};{drawing.hinge_right}",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def selected_drawings(codes: Iterable[str]) -> list[Drawing]:
    requested = {code.upper() for code in codes}
    selected = [item for item in DRAWINGS if not requested or item.icao in requested]
    unknown = requested - {item.icao for item in selected}
    if unknown:
        raise ValueError(f"unsupported ICAO code(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icao", action="append", default=[], help="limit extraction to an ICAO code")
    parser.add_argument("--json-dir", type=Path, help="directory of prior dwgread JSON outputs")
    parser.add_argument("--dwgread", type=Path, help="LibreDWG dwgread executable")
    parser.add_argument("--cad-root", type=Path, default=DEFAULT_CAD_ROOT)
    parser.add_argument("--output", type=Path, help="write CSV here instead of stdout")
    args = parser.parse_args()
    if bool(args.json_dir) == bool(args.dwgread):
        parser.error("provide exactly one of --json-dir or --dwgread")

    rows: list[dict[str, float | str]] = []
    with tempfile.TemporaryDirectory(prefix="aerox-airbus-cad-") as temporary:
        for drawing in selected_drawings(args.icao):
            if args.json_dir:
                json_path = args.json_dir / drawing.json_name
            else:
                json_path = Path(temporary) / drawing.json_name
                source = args.cad_root / drawing.directory / drawing.dwg_name
                subprocess.run(
                    [str(args.dwgread), "-O", "JSON", "-o", str(json_path), str(source)],
                    check=True,
                )
            rows.append(extract(load_json(json_path), drawing))

    fieldnames = list(rows[0]) if rows else []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", encoding="utf-8", newline="")
    else:
        import sys
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
