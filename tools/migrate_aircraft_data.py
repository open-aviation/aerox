#!/usr/bin/env python3
"""Restore and validate Aerox's compact aircraft YAML format.

The aircraft files contain parameters only.  Short source notes for the few
certification fields live in matching plain-text files under ``provenance``.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_ROOT = PROJECT_ROOT / "src/aerox/data/aircraft/yml"
PROVENANCE_ROOT = PROJECT_ROOT / "src/aerox/data/aircraft/provenance"

SIMPLE_TEMPLATE: dict[str, Any] = {
    "aircraft": None,
    "icao": None,
    "iata": None,
    "certification": {
        "model": None,
        "authority": None,
        "type_certificate": None,
        "issue": None,
        "issue_date": None,
    },
    "mtow": None,
    "mlw": None,
    "mzfw": None,
    "minimum_weight": None,
    "oew": None,
    "mfc": None,
    "vmo": None,
    "mmo": None,
    "ceiling": None,
    "icao_wtc": None,
    "eu_wtc": None,
    "faa_wtc": None,
    "parking_area_m2": None,
    "pax": {"max": None, "high": None, "low": None},
    "fuselage": {"length": None, "height": None, "width": None},
    "wing": {
        "area": None,
        "span": None,
        "mac": None,
        "sweep": None,
        "t/c": None,
        "root_chord": None,
        "tip_chord": None,
        "taper_ratio": None,
        "aspect_ratio": None,
        "dihedral": None,
    },
    "horizontal_tail": {
        "area": None,
        "span": None,
        "mac": None,
        "root_chord": None,
        "tip_chord": None,
        "taper_ratio": None,
        "sweep": None,
        "dihedral": None,
        "arm": None,
        "elevator_area_ratio": None,
    },
    "vertical_tail": {
        "area": None,
        "height": None,
        "mac": None,
        "root_chord": None,
        "tip_chord": None,
        "taper_ratio": None,
        "sweep": None,
        "arm": None,
        "rudder_area_ratio": None,
    },
    "flaps": {
        "type": None,
        "area": None,
        "bf/b": None,
        "lambda_f": None,
        "cf/c": None,
        "Sf/S": None,
    },
    "engine": {
        "type": None,
        "mount": None,
        "number": None,
        "default": None,
        "options": None,
    },
    "drag": {"cd0": None, "k": None, "e": None, "gears": None},
    "kinematic": {
        "takeoff_distance_m": None,
        "takeoff_v2": None,
        "climb_to_fl50_ias": None,
        "climb_to_fl50_vs": None,
        "climb_to_fl150_ias": None,
        "climb_to_fl150_vs": None,
        "climb_to_fl240_ias": None,
        "climb_to_fl240_vs": None,
        "mach_climb_mach": None,
        "mach_climb_vs": None,
        "cruise_mach": None,
        "cruise_range_nm": None,
        "cruise_tas": None,
        "descent_to_fl240_mach": None,
        "descent_to_fl240_vs": None,
        "descent_to_fl100_ias": None,
        "descent_to_fl100_vs": None,
        "approach_ias": None,
        "approach_vs": None,
        "landing_distance_m": None,
        "landing_ias": None,
    },
}

CANONICAL_PATHS = {
    "certification.model": "identification.represented_model",
    "certification.authority": "certification.authority",
    "certification.type_certificate": "certification.type_certificate_identifier",
    "certification.issue": "certification.tcds_issue_revision",
    "certification.issue_date": "certification.issue_date",
    "mtow": "mass_limits.maximum_takeoff_mass",
    "mlw": "mass_limits.maximum_landing_mass",
    "mzfw": "mass_limits.maximum_zero_fuel_mass",
    "minimum_weight": "mass_limits.minimum_certified_mass",
    "vmo": "operating_envelope.vmo",
    "mmo": "operating_envelope.mmo",
    "ceiling": "operating_envelope.service_ceiling",
}

UNITS = {
    "mtow": "kg",
    "mlw": "kg",
    "mzfw": "kg",
    "minimum_weight": "kg",
    "vmo": "kt",
    "mmo": "Mach",
    "ceiling": "ft",
}


class Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _get(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _ordered_like_template(data: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, empty in template.items():
        value = data.get(key)
        if isinstance(empty, Mapping):
            value = value if isinstance(value, Mapping) else {}
            result[key] = _ordered_like_template(value, empty)
        else:
            result[key] = copy.deepcopy(value) if key in data else copy.deepcopy(empty)
    return result


def _derive_geometry(document: dict[str, Any]) -> None:
    """Fill only definition-safe ratios from values stored in the same record."""

    wing = document["wing"]
    if wing["aspect_ratio"] is None and wing["span"] is not None and wing["area"] is not None:
        wing["aspect_ratio"] = round(float(wing["span"]) ** 2 / float(wing["area"]), 6)

    for section_name in ("wing", "horizontal_tail", "vertical_tail"):
        section = document[section_name]
        if (
            section["taper_ratio"] is None
            and section["root_chord"] is not None
            and section["tip_chord"] is not None
        ):
            section["taper_ratio"] = round(
                float(section["tip_chord"]) / float(section["root_chord"]), 6
            )


def simple_from_canonical(data: Mapping[str, Any]) -> dict[str, Any]:
    options = _get(data, "propulsion.certified_engine_options") or []
    result = copy.deepcopy(SIMPLE_TEMPLATE)
    result.update(
        {
            "aircraft": _get(data, "identification.name"),
            "icao": _get(data, "identification.icao_code"),
            "iata": " / ".join(_get(data, "identification.iata_codes") or []) or None,
            "mtow": _get(data, "mass_limits.maximum_takeoff_mass"),
            "mlw": _get(data, "mass_limits.maximum_landing_mass"),
            "mzfw": _get(data, "mass_limits.maximum_zero_fuel_mass"),
            "minimum_weight": _get(data, "mass_limits.minimum_certified_mass"),
            "oew": _get(data, "mass_limits.operating_empty_mass"),
            "mfc": _get(data, "mass_limits.maximum_fuel_mass"),
            "vmo": _get(data, "operating_envelope.vmo"),
            "mmo": _get(data, "operating_envelope.mmo"),
            "ceiling": _get(data, "operating_envelope.service_ceiling"),
            "icao_wtc": _get(data, "identification.icao_wake_turbulence_category"),
            "eu_wtc": _get(data, "identification.recat_eu_category"),
            "faa_wtc": _get(data, "identification.faa_wake_category"),
            "parking_area_m2": _get(data, "geometry.parking_area"),
        }
    )
    result["certification"] = {
        "model": _get(data, "identification.represented_model"),
        "authority": _get(data, "certification.authority"),
        "type_certificate": _get(data, "certification.type_certificate_identifier"),
        "issue": _get(data, "certification.tcds_issue_revision"),
        "issue_date": _get(data, "certification.issue_date"),
    }
    result["pax"] = {
        "max": _get(data, "capacity.maximum_passengers"),
        "high": _get(data, "capacity.typical_passengers"),
        "low": _get(data, "capacity.minimum_passengers"),
    }
    result["fuselage"] = {
        "length": _first(_get(data, "geometry.overall_length"), _get(data, "fuselage.length")),
        "height": _get(data, "fuselage.outside_height"),
        "width": _get(data, "fuselage.outside_width"),
    }
    result["wing"] = {
        "area": _get(data, "wing.reference_area"),
        "span": _get(data, "wing.span"),
        "mac": _get(data, "wing.mean_aerodynamic_chord"),
        "sweep": _get(data, "wing.quarter_chord_sweep"),
        "t/c": _get(data, "wing.thickness_to_chord_ratio"),
        "root_chord": _get(data, "wing.root_chord"),
        "tip_chord": _get(data, "wing.tip_chord"),
        "taper_ratio": _get(data, "wing.taper_ratio"),
        "aspect_ratio": _get(data, "wing.aspect_ratio"),
        "dihedral": _get(data, "wing.dihedral"),
    }
    result["horizontal_tail"] = {
        "area": _get(data, "horizontal_tail.area"),
        "span": _get(data, "horizontal_tail.span"),
        "mac": _get(data, "horizontal_tail.mean_aerodynamic_chord"),
        "root_chord": _get(data, "horizontal_tail.root_chord"),
        "tip_chord": _get(data, "horizontal_tail.tip_chord"),
        "taper_ratio": _get(data, "horizontal_tail.taper_ratio"),
        "sweep": _get(data, "horizontal_tail.quarter_chord_sweep"),
        "dihedral": _get(data, "horizontal_tail.dihedral"),
        "arm": _get(data, "horizontal_tail.aerodynamic_moment_arm"),
        "elevator_area_ratio": _get(data, "flight_controls.elevator.area_ratio"),
    }
    result["vertical_tail"] = {
        "area": _get(data, "vertical_tail.area"),
        "height": _first(_get(data, "vertical_tail.height"), _get(data, "vertical_tail.span")),
        "mac": _get(data, "vertical_tail.mean_aerodynamic_chord"),
        "root_chord": _get(data, "vertical_tail.root_chord"),
        "tip_chord": _get(data, "vertical_tail.tip_chord"),
        "taper_ratio": _get(data, "vertical_tail.taper_ratio"),
        "sweep": _get(data, "vertical_tail.quarter_chord_sweep"),
        "arm": _get(data, "vertical_tail.aerodynamic_moment_arm"),
        "rudder_area_ratio": _get(data, "flight_controls.rudder.area_ratio"),
    }
    flap = _get(data, "flight_controls.flaps") or {}
    result["flaps"] = {
        "type": flap.get("type"),
        "area": flap.get("area"),
        "bf/b": flap.get("span_ratio"),
        "lambda_f": flap.get("taper_ratio"),
        "cf/c": flap.get("chord_ratio"),
        "Sf/S": flap.get("area_ratio"),
    }
    result["engine"] = {
        "type": _get(data, "propulsion.propulsion_type"),
        "mount": _get(data, "propulsion.installation_position"),
        "number": _get(data, "propulsion.engine_count"),
        "default": _get(data, "propulsion.default_engine_designation"),
        "options": [item.get("designation") for item in options if item.get("designation")] or None,
    }
    result["drag"] = {
        "cd0": _get(data, "aerodynamics.zero_lift_drag_coefficient"),
        "k": _get(data, "aerodynamics.induced_drag_factor"),
        "e": _get(data, "aerodynamics.oswald_efficiency"),
        "gears": _get(data, "aerodynamics.gear_drag_increment"),
    }
    result["kinematic"] = {
        "takeoff_distance_m": _get(data, "performance.takeoff.distance"),
        "takeoff_v2": _get(data, "performance.takeoff.v2"),
        "climb_to_fl50_ias": _get(data, "performance.climb.to_fl50_ias"),
        "climb_to_fl50_vs": _get(data, "performance.climb.to_fl50_vertical_rate"),
        "climb_to_fl150_ias": _get(data, "performance.climb.to_fl150_ias"),
        "climb_to_fl150_vs": _get(data, "performance.climb.to_fl150_vertical_rate"),
        "climb_to_fl240_ias": _get(data, "performance.climb.to_fl240_ias"),
        "climb_to_fl240_vs": _get(data, "performance.climb.to_fl240_vertical_rate"),
        "mach_climb_mach": _get(data, "performance.climb.mach"),
        "mach_climb_vs": _get(data, "performance.climb.mach_vertical_rate"),
        "cruise_mach": _get(data, "performance.cruise.mach"),
        "cruise_range_nm": _get(data, "performance.cruise.range"),
        "cruise_tas": _get(data, "performance.cruise.true_airspeed"),
        "descent_to_fl240_mach": _get(data, "performance.descent.from_fl240_mach"),
        "descent_to_fl240_vs": _get(data, "performance.descent.from_fl240_vertical_rate"),
        "descent_to_fl100_ias": _get(data, "performance.descent.to_fl100_ias"),
        "descent_to_fl100_vs": _get(data, "performance.descent.to_fl100_vertical_rate"),
        "approach_ias": _get(data, "performance.approach.indicated_airspeed"),
        "approach_vs": _get(data, "performance.approach.descent_rate"),
        "landing_distance_m": _get(data, "performance.landing.distance"),
        "landing_ias": _get(data, "performance.landing.indicated_airspeed"),
    }
    return result


def parse_old_provenance(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"header": {}, "fields": {}}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("Last reviewed: "):
            result["header"]["last_reviewed"] = line.removeprefix("Last reviewed: ")
        elif line.startswith("FIELD "):
            current = {}
            result["fields"][line.removeprefix("FIELD ")] = current
        elif line == "END FIELD":
            current = None
        elif current is not None and line.startswith("  ") and ": " in line:
            key, value = line.strip().split(": ", 1)
            current[key.lower().replace(" ", "_")] = value
    return result


def render_source_notes(
    simple: Mapping[str, Any], canonical: Mapping[str, Any], old_text: str
) -> str:
    old = parse_old_provenance(old_text)
    fields = old["fields"]
    reviewed = old["header"].get("last_reviewed", "2026-08-05")
    applicability = _get(canonical, "certification.model_applicability") or "Exact model or variant scope remains unresolved."
    lines = [
        "AEROX AIRCRAFT SOURCE NOTES",
        f"Aircraft: {simple['aircraft'] or '-'}",
        f"ICAO: {simple['icao'] or '-'}",
        f"Last reviewed: {reviewed}",
        "Source ledger: docs/modeling/sources.yml",
        "",
        "CERTIFICATION SCOPE",
        f"  Model or family: {simple['certification']['model'] or '-'}",
        f"  Authority: {simple['certification']['authority'] or '-'}",
        f"  Type certificate: {simple['certification']['type_certificate'] or '-'}",
        f"  Applicability: {applicability}",
        "",
        "TYPE-CERTIFICATE AND LIMIT PARAMETERS",
    ]
    for simple_path, canonical_path in CANONICAL_PATHS.items():
        value = _get(simple, simple_path)
        metadata = fields.get(canonical_path, {})
        lines.extend(["", f"FIELD {simple_path}", f"  Value: {value if value is not None else 'null'}"])
        if simple_path in UNITS:
            lines.append(f"  Unit: {UNITS[simple_path]}")
        lines.extend(
            [
                f"  Status: {metadata.get('status', 'unresolved' if value is None else 'legacy-unverified')}",
                f"  Source: {metadata.get('sources', '-')}",
            ]
        )
        if metadata.get("applicability") not in {None, "-"}:
            lines.append(f"  Applicability: {metadata['applicability']}")
        if value is None:
            reason = metadata.get("unavailable_reason")
            if reason not in {None, "-"}:
                lines.append(f"  Reason unavailable: {reason}")
        lines.append("END FIELD")
    lines.extend(
        [
            "",
            "LEGACY PARAMETERS",
            "  Parameters not listed above retain the original Aerox database values.",
            "  Unless a source is named above, those values remain legacy-unverified.",
            "",
        ]
    )
    return "\n".join(lines)


def dump_document(document: Mapping[str, Any]) -> str:
    return yaml.dump(
        document,
        Dumper=Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )


def validate_simple(document: Any) -> list[str]:
    if not isinstance(document, Mapping):
        return ["document root must be a mapping"]
    issues: list[str] = []

    def walk(value: Any, template: Any, path: str) -> None:
        if isinstance(template, Mapping):
            if not isinstance(value, Mapping):
                issues.append(f"{path}: expected mapping")
                return
            if list(value) != list(template):
                issues.append(f"{path or '<root>'}: fields/order differ from compact format")
            for key, child in template.items():
                if key in value:
                    walk(value[key], child, f"{path}.{key}" if path else key)
        elif isinstance(value, (Mapping, tuple, set)):
            issues.append(f"{path}: expected scalar, list, or null")

    walk(document, SIMPLE_TEMPLATE, "")
    if document.get("icao") in {None, "", "unknown"}:
        issues.append("icao: missing ICAO code")
    certification = document.get("certification", {})
    if isinstance(certification, Mapping):
        if certification.get("issue") is not None and not isinstance(
            certification["issue"], str
        ):
            issues.append("certification.issue: expected string or null")
        if certification.get("issue_date") is not None and not isinstance(
            certification["issue_date"], str
        ):
            issues.append("certification.issue_date: expected ISO-date string or null")

    geometry_fields = {
        "wing": SIMPLE_TEMPLATE["wing"],
        "horizontal_tail": SIMPLE_TEMPLATE["horizontal_tail"],
        "vertical_tail": SIMPLE_TEMPLATE["vertical_tail"],
    }
    for section_name, fields in geometry_fields.items():
        section = document.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for field_name in fields:
            value = section.get(field_name)
            path = f"{section_name}.{field_name}"
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                issues.append(f"{path}: expected number or null")
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                issues.append(f"{path}: expected finite number")

        for field_name in (
            "area", "span", "height", "mac", "root_chord", "tip_chord",
            "taper_ratio", "aspect_ratio", "arm", "elevator_area_ratio",
            "rudder_area_ratio", "t/c",
        ):
            value = section.get(field_name)
            if isinstance(value, (int, float)) and value <= 0:
                issues.append(f"{section_name}.{field_name}: expected positive value")

        for field_name in ("t/c", "elevator_area_ratio", "rudder_area_ratio"):
            value = section.get(field_name)
            if isinstance(value, (int, float)) and not 0 < float(value) <= 1:
                issues.append(f"{section_name}.{field_name}: expected ratio in (0, 1]")
        taper = section.get("taper_ratio")
        if isinstance(taper, (int, float)) and not 0 < float(taper) <= 2:
            issues.append(f"{section_name}.taper_ratio: implausible planform ratio")
        aspect = section.get("aspect_ratio")
        if isinstance(aspect, (int, float)) and not 1 < float(aspect) < 100:
            issues.append(f"{section_name}.aspect_ratio: implausible aspect ratio")
        for field_name in ("sweep", "dihedral"):
            value = section.get(field_name)
            if isinstance(value, (int, float)) and not -90 < float(value) < 90:
                issues.append(f"{section_name}.{field_name}: angle must be in (-90, 90) deg")

    wing = document.get("wing", {})
    if isinstance(wing, Mapping):
        if all(wing.get(field) is not None for field in ("span", "area", "aspect_ratio")):
            expected = float(wing["span"]) ** 2 / float(wing["area"])
            if not math.isclose(float(wing["aspect_ratio"]), expected, rel_tol=2e-6):
                issues.append("wing.aspect_ratio: does not match span^2 / area")

    for section_name in ("wing", "horizontal_tail", "vertical_tail"):
        section = document.get(section_name, {})
        if isinstance(section, Mapping) and all(
            section.get(field) is not None
            for field in ("root_chord", "tip_chord", "taper_ratio")
        ):
            expected = float(section["tip_chord"]) / float(section["root_chord"])
            if not math.isclose(float(section["taper_ratio"]), expected, rel_tol=3e-6):
                issues.append(
                    f"{section_name}.taper_ratio: does not match tip_chord / root_chord"
                )
    return issues


def normalize_bundle(document: Mapping[str, Any], sidecar_text: str) -> tuple[dict[str, Any], str]:
    if document.get("schema_version") == "2.0.0":
        simple = simple_from_canonical(document)
        sidecar = render_source_notes(simple, document, sidecar_text)
    else:
        simple = _ordered_like_template(document, SIMPLE_TEMPLATE)
        sidecar = sidecar_text
    issue_date = simple["certification"]["issue_date"]
    if issue_date is not None and not isinstance(issue_date, str) and hasattr(issue_date, "isoformat"):
        simple["certification"]["issue_date"] = issue_date.isoformat()
    _derive_geometry(simple)
    return simple, sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=AIRCRAFT_ROOT)
    parser.add_argument("--icao", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {value.upper() for value in args.icao}
    changes: list[tuple[Path, str, Path, str]] = []
    files = [
        path for path in sorted(args.root.glob("*/*.yml"))
        if not selected or path.stem.upper() in selected
    ]
    provenance_root = PROVENANCE_ROOT if args.root.resolve() == AIRCRAFT_ROOT.resolve() else args.root.parent / "provenance"
    for path in files:
        original = path.read_text(encoding="utf-8")
        document = yaml.safe_load(original)
        sidecar_path = provenance_root / path.parent.name / f"{path.stem}.txt"
        old_sidecar = sidecar_path.read_text(encoding="utf-8") if sidecar_path.exists() else ""
        normalized, sidecar = normalize_bundle(document, old_sidecar)
        issues = validate_simple(normalized)
        if issues:
            raise ValueError(f"{path}: " + "; ".join(issues))
        rendered = dump_document(normalized)
        if rendered != original or sidecar != old_sidecar:
            changes.append((path, rendered, sidecar_path, sidecar))
    if args.write:
        for path, rendered, sidecar_path, sidecar in changes:
            path.write_text(rendered, encoding="utf-8")
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(sidecar, encoding="utf-8")
    print(f"aircraft={len(files)} changed={len(changes)} mode={'write' if args.write else 'check'}")
    return 1 if args.check and changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
