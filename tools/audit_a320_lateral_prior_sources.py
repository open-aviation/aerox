#!/usr/bin/env python3
"""Qualify public sources for an A320 lateral-directional model prior.

This audit is deliberately conservative. It records which public sources can
support a model structure or an engineering-generation method, but it does not
turn generic-transport coefficients into A320 coefficients.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

UPDATED = "2026-08-05"
GTM_DESIGNSIM_COMMIT = "9717143270144aca1f5d38d7c24c0fce678d1589"


GEOMETRY_FIELDS = (
    ("wing.area", "wing reference area", True),
    ("wing.span", "wing span", True),
    ("wing.mac", "wing mean aerodynamic chord", True),
    ("wing.sweep", "wing sweep", True),
    ("wing.t/c", "wing thickness ratio", True),
    ("wing.taper_ratio", "wing taper ratio", False),
    ("wing.dihedral", "wing dihedral", False),
    ("fuselage.length", "fuselage length", True),
    ("fuselage.width", "fuselage width", True),
    ("fuselage.height", "fuselage height", True),
    ("geometry.overall_height", "overall aircraft height", False),
    ("geometry.modified_wing_span", "modified/sharklet wing span", False),
    ("geometry.engine_axis_spacing", "engine-axis spacing", False),
    ("geometry.main_landing_gear_track", "main-gear track", False),
    ("geometry.nose_to_main_landing_gear", "nose-to-main-gear distance", False),
    ("geometry.datum_forward_of_nose", "datum distance forward of nose", False),
    ("horizontal_tail.area", "horizontal-tail area", False),
    ("horizontal_tail.span", "horizontal-tail span", False),
    ("horizontal_tail.mac", "horizontal-tail mean chord", False),
    ("horizontal_tail.sweep", "horizontal-tail sweep", False),
    ("horizontal_tail.arm", "horizontal-tail moment arm", False),
    ("vertical_tail.area", "vertical-tail area", False),
    ("vertical_tail.span", "vertical-tail span", False),
    ("vertical_tail.height", "vertical-tail height above horizontal stabilizer", False),
    ("vertical_tail.mac", "vertical-tail mean chord", False),
    ("vertical_tail.sweep", "vertical-tail sweep", False),
    ("vertical_tail.arm", "vertical-tail moment arm", False),
    ("aileron.span_limits", "aileron inboard/outboard stations", False),
    ("aileron.chord_ratio", "aileron chord ratio", False),
    ("aileron.deflection_limit", "aileron deflection limit", False),
    ("rudder.span_limits", "rudder span limits", False),
    ("rudder.chord_ratio", "rudder chord ratio", False),
    ("rudder.deflection_limit", "rudder deflection limit", False),
    ("certified_mass_limits.mtow", "maximum certified takeoff mass", True),
    ("certified_mass_limits.mlw", "maximum certified landing mass", True),
    ("certified_mass_limits.mzfw", "maximum certified zero-fuel mass", True),
    ("certified_mass_limits.minimum_weight", "minimum certified weight", True),
    ("mass_properties.reference_cg_mac", "published planning reference CG", False),
    ("mass_properties.cg_forward_mac", "certified forward CG limit", False),
    ("mass_properties.cg_aft_mac", "certified aft CG limit", False),
    ("mass_properties.reference_mass", "inertia reference mass", False),
    ("mass_properties.ixx", "roll inertia", False),
    ("mass_properties.iyy", "pitch inertia", False),
    ("mass_properties.izz", "yaw inertia", False),
    ("mass_properties.ixz", "roll-yaw cross inertia", False),
)


SOURCE_MATRIX = (
    {
        "source_id": "easa_a064_issue_62",
        "title": "EASA Type Certificate Data Sheet EASA.A.064, Issue 62",
        "organization": "European Union Aviation Safety Agency",
        "url": "https://www.easa.europa.eu/en/document-library/type-certificates/aircraft-cs-25-cs-22-cs-23-cs-vla-cs-lsa/easaa064-airbus-a318-a319-a320-a321-single-aisle",
        "evidence": "current public certification record for the A320-214",
        "a320_specific": True,
        "public": True,
        "lateral_derivatives": "none",
        "current_use": "certified model identity, dimensions, engines, mass limits, datum, and MAC",
        "decision": "adopt-certified-fields",
        "limitation": "CG range is deferred to the approved AFM; tail planforms, controls, and inertias are absent",
    },
    {
        "source_id": "airbus_a320_acap_2026_07",
        "title": "A320 Aircraft Characteristics - Airport and Maintenance Planning",
        "organization": "Airbus",
        "url": "https://www.aircraft.airbus.com/en/customer-care/fleet-wide-care/airport-operations-and-aircraft-characteristics/aircraft-characteristics",
        "evidence": "manufacturer geometry document",
        "a320_specific": True,
        "public": True,
        "lateral_derivatives": "none",
        "current_use": "overall dimensions, vertical-tail reference height, and 26.5%-MAC planning-reference CG",
        "decision": "adopt-geometry-evidence-only",
        "limitation": "planning reference CG is not an operational CG or certified range; no full tail/control planform or inertia tensor",
    },
    {
        "source_id": "faa_a28nm_revision_42",
        "title": "FAA Type Certificate Data Sheet A28NM, Revision 42",
        "organization": "US Federal Aviation Administration",
        "url": "https://downloads.regulations.gov/FAA-2021-0799-0001/attachment_3.pdf",
        "evidence": "public US certification record cross-check",
        "a320_specific": True,
        "public": True,
        "lateral_derivatives": "none",
        "current_use": "independent certification cross-check and AFM-only CG finding",
        "decision": "retain-cross-check-no-extra-parameters",
        "limitation": "Revision 42 is dated 2020 and adds no public tail/control geometry or inertia tensor",
    },
    {
        "source_id": "nasa_gna_2014_0542",
        "title": "A Generic Nonlinear Aerodynamic Model for Aircraft",
        "organization": "NASA Langley Research Center",
        "url": "https://ntrs.nasa.gov/citations/20140011902",
        "evidence": "wind-tunnel-database system identification across eight aircraft",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "CY, Cl, Cn versus beta, normalized p/r, aileron, rudder",
        "current_use": "model structure; GTM coefficients only as a sanity-check reference",
        "decision": "adopt-structure-not-a320-coefficients",
        "limitation": "GTM is a subscale generic transport and Mach is absent from the regressors",
    },
    {
        "source_id": "nasa_gtm_designsim_9717143",
        "title": "GTM_DesignSim: Flight Dynamics Simulation of a Generic Transport Model",
        "organization": "NASA",
        "url": "https://github.com/nasa/GTM_DesignSim",
        "evidence": "open-source nonlinear 6-DOF reference simulation",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "nonlinear lookup/polynomial database and surface increments",
        "current_use": "reference implementation and regression target only",
        "decision": "adopt-reference-no-code-copy",
        "limitation": "5.5 percent research aircraft, MATLAB/Simulink, NASA OSA 1.3",
    },
    {
        "source_id": "nasa_tcm_2011_217169",
        "title": "Development of the Transport Class Model Aircraft Simulation",
        "organization": "NASA Langley Research Center",
        "url": "https://ntrs.nasa.gov/citations/20110014509",
        "evidence": "full-scale generic single-aisle transport simulation report",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "full nonlinear GTM lateral database retained in a scaled 6-DOF plant",
        "current_use": "full-scale architecture, inertia, actuator, and scaling reference",
        "decision": "adopt-reference-not-a320-parameters",
        "limitation": "representative 185,000-lb aircraft; lateral coefficients were not Reynolds-corrected",
    },
    {
        "source_id": "usaf_digital_datcom_1979",
        "title": "The USAF Stability and Control Digital DATCOM, Volume I",
        "organization": "US Air Force Flight Dynamics Laboratory",
        "url": "https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA086557.xhtml",
        "evidence": "government engineering-method manual",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "static, control, and dynamic derivative prediction from geometry",
        "current_use": "preferred A320-specific generation method after geometry completion",
        "decision": "adopt-method-candidate",
        "limitation": "requires tail, control, CG, and planform inputs absent from Aerox",
    },
    {
        "source_id": "nasa_vspaero_2026",
        "title": "OpenVSP VSPAERO stability analysis",
        "organization": "NASA",
        "url": "https://www.nasa.gov/reference/openvsp-vspaero-basics/",
        "evidence": "NASA geometry and aerodynamic-analysis software documentation",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "steady 6-DOF and unsteady roll/yaw stability analyses",
        "current_use": "independent geometry-based cross-check after geometry completion",
        "decision": "adopt-cross-check-candidate",
        "limitation": "requires a defensible 3D geometry and does not replace viscous validation",
    },
    {
        "source_id": "nasa_gtm_roll_yaw_id_2010",
        "title": "Transport Aircraft System Identification Using Roll and Yaw Oscillatory Wind Tunnel Data",
        "organization": "NASA Langley Research Center",
        "url": "https://ntrs.nasa.gov/citations/20100031282",
        "evidence": "forced-oscillation wind-tunnel identification",
        "a320_specific": False,
        "public": True,
        "lateral_derivatives": "static, damping, cross, and unsteady roll/yaw terms",
        "current_use": "dynamic-model and excitation-design reference",
        "decision": "adopt-method-evidence",
        "limitation": "5.5 percent low-speed model; published examples emphasize high angle of attack",
    },
)


GTM_REFERENCE = {
    "source": "nasa_gna_2014_0542",
    "aircraft": "NASA Generic Transport Model",
    "applicability": "generic-transport sanity check only; not an A320 parameter set",
    "rate_normalization": "p_tilde = p*b/(2*V), r_tilde = r*b/(2*V)",
    "angles_and_surface_deflections": "radians",
    "coefficients": {
        "CY": {
            "beta": -1.003,
            "p_tilde": 0.033,
            "r_tilde": 0.952,
            "delta_a": -0.009,
            "delta_r": 0.253,
        },
        "Cl": {
            "beta": -0.109,
            "p_tilde": -0.366,
            "r_tilde": 0.061,
            "delta_a": -0.079,
            "delta_r": 0.021,
        },
        "Cn": {
            "beta": 0.183,
            "p_tilde": -0.022,
            "r_tilde": -0.405,
            "delta_a": -0.009,
            "delta_r": -0.129,
            "beta_squared": 0.184,
            "beta_cubed": -0.377,
        },
    },
    "published_gtm_geometry": {
        "wing_area_ft2": 5.902,
        "mean_aerodynamic_chord_ft": 0.915,
        "wing_span_ft": 6.849,
    },
    "published_validation": {
        "roll_coefficient_r_squared": 0.9961,
        "database_dutch_roll_frequency_rad_s": 5.89,
        "gna_dutch_roll_frequency_rad_s": 5.27,
        "database_dutch_roll_damping_ratio": 0.149,
        "gna_dutch_roll_damping_ratio": 0.179,
    },
    "adopted_as_a320_coefficients": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        type=Path,
        default=Path("src/aerox/data/aircraft/yml/turbofan/a320.yml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-lateral-prior-source-audit.yml"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("docs/modeling/a320-lateral-prior-source-matrix.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-lateral-prior-source-audit.md"),
    )
    return parser.parse_args()


def _get_path(data: dict[str, Any], path: str) -> Any:
    if data.get("schema_version") in {"1.0.0", "2.0.0"}:
        compatibility_paths = {
            "wing.area": "wing.reference_area",
            "wing.span": "wing.span",
            "wing.mac": "wing.mean_aerodynamic_chord",
            "wing.sweep": "wing.quarter_chord_sweep",
            "wing.t/c": "wing.thickness_to_chord_ratio",
            "fuselage.length": "geometry.overall_length",
            "fuselage.width": "fuselage.outside_width",
            "fuselage.height": "fuselage.outside_height",
            "geometry.overall_height": "geometry.overall_height",
            "geometry.modified_wing_span": "wing.modified_span",
            "geometry.engine_axis_spacing": "propulsion.engine_axis_spacing",
            "geometry.main_landing_gear_track": "landing_gear.main_gear_track",
            "geometry.nose_to_main_landing_gear": "landing_gear.nose_to_main_gear_distance",
            "geometry.datum_forward_of_nose": "geometry.nose_to_datum_distance",
            "horizontal_tail.mac": "horizontal_tail.mean_aerodynamic_chord",
            "horizontal_tail.sweep": "horizontal_tail.quarter_chord_sweep",
            "horizontal_tail.arm": "horizontal_tail.aerodynamic_moment_arm",
            "vertical_tail.mac": "vertical_tail.mean_aerodynamic_chord",
            "vertical_tail.sweep": "vertical_tail.quarter_chord_sweep",
            "vertical_tail.arm": "vertical_tail.aerodynamic_moment_arm",
            "aileron.span_limits": "flight_controls.aileron.span_ratio",
            "aileron.chord_ratio": "flight_controls.aileron.chord_ratio",
            "aileron.deflection_limit": "flight_controls.aileron.positive_deflection_limit",
            "rudder.span_limits": "flight_controls.rudder.span_ratio",
            "rudder.chord_ratio": "flight_controls.rudder.chord_ratio",
            "rudder.deflection_limit": "flight_controls.rudder.positive_deflection_limit",
            "certified_mass_limits.mtow": "mass_limits.maximum_takeoff_mass",
            "certified_mass_limits.mlw": "mass_limits.maximum_landing_mass",
            "certified_mass_limits.mzfw": "mass_limits.maximum_zero_fuel_mass",
            "certified_mass_limits.minimum_weight": "mass_limits.minimum_certified_mass",
            "mass_properties.reference_cg_mac": "mass_properties.reference_cg_fraction_mac",
            "mass_properties.cg_forward_mac": "mass_properties.certified_forward_cg_limit",
            "mass_properties.cg_aft_mac": "mass_properties.certified_aft_cg_limit",
        }
        canonical_path = compatibility_paths.get(path, path)
        current: Any = data
        for key in canonical_path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current.get("value") if isinstance(current, dict) and "value" in current else current

    compact_paths = {
        "certified_mass_limits.mtow": "mtow",
        "certified_mass_limits.mlw": "mlw",
        "certified_mass_limits.mzfw": "mzfw",
        "certified_mass_limits.minimum_weight": "minimum_weight",
    }
    current: Any = data
    for key in compact_paths.get(path, path).split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def build_audit(aircraft_path: Path) -> dict[str, Any]:
    aircraft = yaml.safe_load(aircraft_path.read_text())
    fields = []
    for path, description, expected_available in GEOMETRY_FIELDS:
        value = _get_path(aircraft, path)
        available = value is not None
        if expected_available and not available:
            raise ValueError(f"Expected local A320 field is missing: {path}")
        fields.append(
            {
                "field": path,
                "description": description,
                "available": available,
                "value": value,
            }
        )

    available_count = sum(item["available"] for item in fields)
    wing_area = float(_get_path(aircraft, "wing.area"))
    wing_span = float(_get_path(aircraft, "wing.span"))
    a320_aspect_ratio = wing_span**2 / wing_area
    gtm_area = GTM_REFERENCE["published_gtm_geometry"]["wing_area_ft2"]
    gtm_span = GTM_REFERENCE["published_gtm_geometry"]["wing_span_ft"]
    gtm_aspect_ratio = gtm_span**2 / gtm_area

    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "certified-core-fields-added-external-lateral-prior-still-blocked",
        "scope": {
            "purpose": "qualify public primary sources before adding lateral-directional physics",
            "control_interface": "pitch, bank/roll, heading or heading-rate, and direct net thrust remain the user-facing controls",
            "surface_level_controls_required_for_current_interface": False,
        },
        "local_inventory": {
            "source": str(aircraft_path),
            "screened_fields": len(fields),
            "available_fields": available_count,
            "missing_fields": len(fields) - available_count,
            "fields": fields,
            "a320_wing_aspect_ratio": a320_aspect_ratio,
        },
        "generic_transport_comparison": {
            "gtm_wing_aspect_ratio": gtm_aspect_ratio,
            "a320_minus_gtm_aspect_ratio_fraction": a320_aspect_ratio / gtm_aspect_ratio - 1.0,
            "conclusion": "configuration similarity supports a structural prior, not coefficient identity",
        },
        "source_matrix": list(SOURCE_MATRIX),
        "selected_prior": {
            "model_structure": {
                "source": "nasa_gna_2014_0542",
                "equations": {
                    "CY": "CY_beta*beta + CY_p*p_tilde + CY_r*r_tilde + CY_delta_a*delta_a + CY_delta_r*delta_r",
                    "Cl": "Cl_beta*beta + Cl_p*p_tilde + Cl_r*r_tilde + Cl_delta_a*delta_a + Cl_delta_r*delta_r",
                    "Cn": "Cn_beta*beta + Cn_p*p_tilde + Cn_r*r_tilde + Cn_delta_a*delta_a + Cn_delta_r*delta_r + Cn_beta2*beta^2 + Cn_beta3*beta^3",
                },
                "normalization": "p_tilde = p*b/(2*V), r_tilde = r*b/(2*V)",
                "adopted": True,
            },
            "generic_numeric_reference": GTM_REFERENCE,
            "a320_specific_generation": {
                "primary_method": "usaf_digital_datcom_1979",
                "independent_cross_check": "nasa_vspaero_2026",
                "blocked_by": [
                    "tail areas, chords, sweep, and moment arms",
                    "aileron and rudder geometry and limits",
                    "longitudinal CG location",
                    "roll, yaw, and roll-yaw inertias",
                ],
            },
        },
        "decision": {
            "adopt_lateral_model_structure": True,
            "adopt_gtm_coefficients_as_a320": False,
            "release_a320_lateral_coefficients": False,
            "release_surface_level_simulator": False,
            "retain_high_level_attitude_control_interface": True,
            "next_gate": "derive a traceable uncertain tail/control geometry package from manufacturer drawings or other qualified data, then run DATCOM and VSPAERO as independent prior generators",
        },
        "provenance": {
            "gtm_designsim_commit": GTM_DESIGNSIM_COMMIT,
            "web_sources_accessed": UPDATED,
            "audit_code": "tools/audit_a320_lateral_prior_sources.py",
        },
        "limitations": [
            "No public A320-214 stability/control derivative set was found in the qualified primary sources.",
            "The EASA and FAA certification sheets defer the usable CG range to approved flight manuals and do not publish an inertia tensor.",
            "The Airbus planning document provides reference dimensions and a planning CG, not complete aerodynamic tail/control detail or inertias.",
            "NASA GTM/GNA values are generic-transport references and are not measurements of either operator A320.",
            "The 1 Hz QAR and missing direct sideslip prevent independent release of an A320 lateral parameter set.",
        ],
    }


def _write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(audit: dict[str, Any]) -> str:
    inventory = audit["local_inventory"]
    comparison = audit["generic_transport_comparison"]
    matrix_rows = []
    for item in audit["source_matrix"]:
        matrix_rows.append(
            f"| {item['source_id']} | {item['a320_specific']} | {item['lateral_derivatives']} | {item['decision']} |"
        )
    missing = [
        item["description"]
        for item in inventory["fields"]
        if not item["available"]
    ]
    missing_text = "\n".join(f"- {name}" for name in missing)
    inventory_summary = (
        f"Aerox currently has {inventory['available_fields']} of the "
        f"{inventory['screened_fields']} screened geometry/mass-property fields."
    )
    aspect_summary = (
        "The current A320 aspect ratio is "
        f"{inventory['a320_wing_aspect_ratio']:.3f}; the published NASA GTM "
        f"value is {comparison['gtm_wing_aspect_ratio']:.3f}. The "
        f"{comparison['a320_minus_gtm_aspect_ratio_fraction'] * 100:.1f}% "
        "difference is small enough for model-structure analogy but too large, "
        "alongside the missing tail and inertia data, to imply equal coefficients."
    )

    return f"""# A320 lateral-prior source audit

## Outcome

The public-source gate supports a lateral-directional **model structure**, but
not an A320 coefficient release. NASA's generic nonlinear aerodynamic model
provides traceable equations for `CY`, `Cl`, and `Cn` using sideslip, normalized
roll/yaw rates, aileron, and rudder. Its GTM coefficient set remains a
generic-transport sanity check only. It is not assigned to either operator
A320-214.

The certification sub-gate does add authoritative A320-214 data. EASA
`EASA.A.064` Issue 62 supplies the model definition, certified mass limits,
baseline and modified spans, overall dimensions, engine spacing, datum, and
MAC. Airbus's July 2026 planning document independently supports the external
geometry and supplies a 26.5%-MAC planning-reference CG. That CG is not an
operational measurement or a certified range.

The user-facing trajectory control boundary therefore stays at pitch,
bank/roll, heading or heading rate, and direct net thrust. This audit does not
require exposing aileron or rudder as trajectory-optimization controls.

## Local geometry gate

{inventory_summary} The available fields cover wing reference
area/span/MAC/sweep/thickness ratio and
fuselage dimensions, certified masses, additional external dimensions, one
tail span/height reference, and a planning-reference CG. They still do not
define an A320 stability-and-control model. {aspect_summary}

Missing screened inputs:

{missing_text}

## Qualified sources

| Source | A320-specific | Coverage | Decision |
|---|---:|---|---|
{chr(10).join(matrix_rows)}

The most useful numerical reference is the GTM column in NASA AIAA
2014-0542. The report gives the complete coefficient mapping and shows that its
compact roll-coefficient model fits the GTM database with `R^2 = 0.9961`; it
also preserves the GTM Dutch-roll frequency and damping reasonably closely.
Those are reference-model validation results, not A320 validation.

NASA's full-scale Transport Class Model is an important architecture check. It
scales the GTM into a generic 185,000-lb single-aisle transport with explicit
inertias and representative hydraulic actuators. Its lateral coefficients were
retained from the subscale database without the Reynolds corrections applied
to lift and drag, which is another reason not to relabel them as A320 data.

## Selected development path

1. Adopt the NASA GNA lateral equation structure and normalized-rate
   convention.
2. Keep the published GTM numbers in the audit as a generic sanity-check prior,
   with `adopted_as_a320_coefficients: false`.
3. Extend the certified core into a source-backed A320 tail/control geometry
   and mass-property package, carrying uncertainty for every inferred field.
4. Generate two independent A320 priors: USAF Digital DATCOM as the primary
   semi-empirical method and NASA VSPAERO as the geometry-based cross-check.
5. Compare signs, magnitudes, modes, and replay behavior before releasing any
   A320 lateral coefficient.

## Release decision

- Lateral model structure: **adopted**.
- GTM coefficients as A320 coefficients: **rejected**.
- A320 lateral coefficient set: **not released**.
- Surface-level user control interface: **not introduced**.
- Next gate: traceable A320 tail/control geometry, flight CG, and inertias.
"""


def main() -> None:
    args = parse_args()
    audit = build_audit(args.aircraft)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(audit, sort_keys=False))
    _write_matrix(args.matrix, audit["source_matrix"])
    args.report.write_text(_render_report(audit))


if __name__ == "__main__":
    main()
