#!/usr/bin/env python3
"""Build the uncertainty-aware A320 geometry and generic inertia prior.

The committed measurement table contains a small set of coordinates extracted
from Airbus's May 2012 A320 wing-tip-fence AutoCAD drawing. The source drawing
is not redistributed. Passing ``--dxf`` verifies the table against a local
ASCII DXF converted from the manufacturer DWG before regenerating artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

UPDATED = "2026-08-05"
SOURCE_DWG_SHA256 = "1e6c85334d0e24bba3bbbefe143a83f6962e30db087d53b4a427554839ab6912"
AIRBUS_DRAWING_ACCURACY_MM = 100.0
HORIZONTAL_TAIL_SPAN_M = 12.45
VERTICAL_TAIL_REFERENCE_HEIGHT_M = 5.87
WING_SPAN_M = 34.10
WING_MAC_M = 4.1935
FUSELAGE_WIDTH_M = 3.95
REFERENCE_CG_FRACTION_MAC = 0.265
PLANNING_CG_RANGE_FRACTION_MAC = (0.17, 0.40)
REFERENCE_MASS_KG = 60_000.0
QAR_MASS_RANGE_KG = (52_671.0, 70_960.0)


@dataclass(frozen=True)
class Measurement:
    feature: str
    view: str
    handle: str
    entity_type: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    interpretation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("docs/modeling/a320-cad-geometry-measurements.csv"),
    )
    parser.add_argument("--dxf", type=Path, help="optional local ASCII DXF to verify")
    parser.add_argument("--source-dwg", type=Path, help="optional original DWG to hash")
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-geometry-mass-prior.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-geometry-mass-prior.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-geometry-mass-prior"),
    )
    return parser.parse_args()


def load_measurements(path: Path) -> dict[str, Measurement]:
    result: dict[str, Measurement] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            item = Measurement(
                feature=row["feature"],
                view=row["view"],
                handle=row["entity_handle"].upper(),
                entity_type=row["entity_type"],
                p1=(float(row["x1_mm"]), float(row["y1_mm"])),
                p2=(float(row["x2_mm"]), float(row["y2_mm"])),
                interpretation=row["interpretation"],
            )
            if item.feature in result:
                raise ValueError(f"Duplicate measurement feature: {item.feature}")
            result[item.feature] = item
    return result


def _dxf_entities(path: Path) -> dict[str, dict[str, Any]]:
    """Read the handle, type, and endpoint coordinates needed from ASCII DXF."""

    lines = path.read_text(errors="strict").splitlines()
    if len(lines) % 2:
        raise ValueError("DXF must contain group-code/value line pairs")
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines), 2)]
    entities: dict[str, dict[str, Any]] = {}
    current_type: str | None = None
    groups: list[tuple[str, str]] = []

    def store() -> None:
        if current_type not in {"LINE", "LWPOLYLINE"}:
            return
        handle = next((value.upper() for code, value in groups if code == "5"), None)
        if handle is None:
            return
        if current_type == "LINE":
            values = {code: value for code, value in groups}
            points = [
                (float(values["10"]), float(values["20"])),
                (float(values["11"]), float(values["21"])),
            ]
        else:
            points: list[tuple[float, float]] = []
            pending_x: float | None = None
            for code, value in groups:
                if code == "10":
                    pending_x = float(value)
                elif code == "20" and pending_x is not None:
                    points.append((pending_x, float(value)))
                    pending_x = None
        entities[handle] = {"type": current_type, "points": points}

    for code, value in pairs:
        if code == "0":
            store()
            current_type = value
            groups = []
        else:
            groups.append((code, value))
    store()
    return entities


def verify_dxf(path: Path, measurements: dict[str, Measurement]) -> dict[str, Any]:
    entities = _dxf_entities(path)
    maximum_difference = 0.0
    for item in measurements.values():
        # Nose rows intentionally store entity bounding-box extrema rather than
        # polyline endpoints; the values establish each drawing view's datum.
        if "_view_nose_" in item.feature:
            continue
        entity = entities.get(item.handle)
        if entity is None:
            raise ValueError(f"DXF is missing expected handle {item.handle} ({item.feature})")
        if entity["type"] != item.entity_type:
            raise ValueError(f"DXF entity type changed for {item.feature}")
        points = entity["points"]
        if len(points) < 2:
            raise ValueError(f"DXF entity has fewer than two points for {item.feature}")
        expected = (item.p1, item.p2)
        actual = (points[0], points[-1])
        direct = max(math.dist(a, b) for a, b in zip(expected, actual, strict=True))
        reversed_difference = max(
            math.dist(a, b) for a, b in zip(expected, reversed(actual), strict=True)
        )
        maximum_difference = max(maximum_difference, min(direct, reversed_difference))
    if maximum_difference > 0.02:
        raise ValueError(
            f"DXF endpoint verification differs by {maximum_difference:.6f} mm"
        )
    return {
        "verified": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "maximum_endpoint_difference_mm": maximum_difference,
    }


def _line_value(p1: tuple[float, float], p2: tuple[float, float], x: float) -> float:
    return p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])


def _line_x(p1: tuple[float, float], p2: tuple[float, float], y: float) -> float:
    return p1[0] + (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1])


def _trapezoid_mac(root_chord: float, tip_chord: float) -> float:
    taper = tip_chord / root_chord
    return (2.0 / 3.0) * root_chord * (1 + taper + taper**2) / (1 + taper)


def _trapezoid_mac_station(span: float, root_chord: float, tip_chord: float) -> float:
    taper = tip_chord / root_chord
    return (span / 6.0) * (1 + 2 * taper) / (1 + taper)


def _interp_line(p1: tuple[float, float], p2: tuple[float, float], x: np.ndarray) -> np.ndarray:
    return p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])


def _nominal_geometry(m: dict[str, Measurement]) -> dict[str, dict[str, float]]:
    left_le = m["horizontal_tail_leading_edge_left"]
    right_le = m["horizontal_tail_leading_edge_right"]
    left_te = m["horizontal_tail_trailing_edge_left"]
    right_te = m["horizontal_tail_trailing_edge_right"]
    left_hinge = m["elevator_hinge_left"]
    center_x = np.mean([left_le.p1[0], right_le.p1[0], left_te.p1[0], right_te.p2[0]])
    tip_x = center_x - HORIZONTAL_TAIL_SPAN_M * 500.0
    h_le_root = _line_value(left_le.p1, left_le.p2, center_x)
    h_te_root = _line_value(left_te.p1, left_te.p2, center_x)
    h_le_tip = _line_value(left_le.p1, left_le.p2, tip_x)
    h_te_tip = _line_value(left_te.p1, left_te.p2, tip_x)
    h_hinge_root = _line_value(left_hinge.p1, left_hinge.p2, center_x)
    h_hinge_tip = _line_value(left_hinge.p1, left_hinge.p2, tip_x)
    h_root = (h_te_root - h_le_root) / 1000.0
    h_tip = (h_te_tip - h_le_tip) / 1000.0
    h_mac = _trapezoid_mac(h_root, h_tip)
    h_mac_y = _trapezoid_mac_station(HORIZONTAL_TAIL_SPAN_M, h_root, h_tip)
    h_le_mac_mm = _line_value(left_le.p1, left_le.p2, center_x - h_mac_y * 1000.0)
    top_nose_y = min(
        m["top_view_nose_left"].p1[1], m["top_view_nose_right"].p1[1]
    )
    h_ac_station = (h_le_mac_mm - top_nose_y) / 1000.0 + 0.25 * h_mac
    h_q_root = h_le_root + 0.25 * h_root * 1000.0
    h_q_tip = h_le_tip + 0.25 * h_tip * 1000.0
    h_sweep = math.degrees(math.atan2(h_q_tip - h_q_root, HORIZONTAL_TAIL_SPAN_M * 500.0))
    h_area = HORIZONTAL_TAIL_SPAN_M * (h_root + h_tip) / 2.0
    elevator_root = (h_te_root - h_hinge_root) / 1000.0
    elevator_tip = (h_te_tip - h_hinge_tip) / 1000.0
    elevator_area = HORIZONTAL_TAIL_SPAN_M * (elevator_root + elevator_tip) / 2.0

    v_le = m["vertical_tail_leading_edge"]
    v_te = m["vertical_tail_trailing_edge"]
    rudder = m["rudder_hinge"]
    v_tip_y = v_te.p1[1] + 1.843125
    v_root_y = v_tip_y - VERTICAL_TAIL_REFERENCE_HEIGHT_M * 1000.0
    v_le_root = _line_x(v_le.p1, v_le.p2, v_root_y)
    v_te_root = _line_x(v_te.p1, v_te.p2, v_root_y)
    v_le_tip = _line_x(v_le.p1, v_le.p2, v_tip_y)
    v_te_tip = _line_x(v_te.p1, v_te.p2, v_tip_y)
    v_root = (v_te_root - v_le_root) / 1000.0
    v_tip = (v_te_tip - v_le_tip) / 1000.0
    v_mac = _trapezoid_mac(v_root, v_tip)
    v_mac_z = _trapezoid_mac_station(VERTICAL_TAIL_REFERENCE_HEIGHT_M, v_root, v_tip)
    v_le_mac_mm = _line_x(v_le.p1, v_le.p2, v_root_y + v_mac_z * 1000.0)
    side_nose_x = m["side_view_nose_upper"].p1[0]
    v_ac_station = (v_le_mac_mm - side_nose_x) / 1000.0 + 0.25 * v_mac
    v_q_root = v_le_root + 0.25 * v_root * 1000.0
    v_q_tip = v_le_tip + 0.25 * v_tip * 1000.0
    v_sweep = math.degrees(
        math.atan2(v_q_tip - v_q_root, VERTICAL_TAIL_REFERENCE_HEIGHT_M * 1000.0)
    )
    v_area = VERTICAL_TAIL_REFERENCE_HEIGHT_M * (v_root + v_tip) / 2.0
    rudder_root_x = _line_x(rudder.p1, rudder.p2, v_root_y)
    rudder_tip_x = _line_x(rudder.p1, rudder.p2, v_tip_y)
    rudder_root = (v_te_root - rudder_root_x) / 1000.0
    rudder_tip = (v_te_tip - rudder_tip_x) / 1000.0
    rudder_area = VERTICAL_TAIL_REFERENCE_HEIGHT_M * (rudder_root + rudder_tip) / 2.0

    wing_le_inner = m["wing_leading_edge_inboard_right"]
    wing_le_outer = m["wing_leading_edge_outboard_right"]
    wing_te_inner = m["wing_trailing_edge_inboard_right"]
    wing_te_outer = m["wing_trailing_edge_outboard_right"]
    center_x = (left_le.p1[0] + right_le.p1[0]) / 2.0
    nose_y = top_nose_y
    lateral = np.array(
        [0.0, FUSELAGE_WIDTH_M / 2.0, (wing_le_outer.p2[0] - center_x) / 1000.0,
         (wing_le_outer.p1[0] - center_x) / 1000.0, WING_SPAN_M / 2.0]
    )
    le_station = np.array(
        [wing_le_inner.p2[1] - nose_y, wing_le_inner.p2[1] - nose_y,
         wing_le_outer.p2[1] - nose_y, wing_le_outer.p1[1] - nose_y,
         (wing_le_outer.p1[1] + 817.0) - nose_y]
    ) / 1000.0
    te_station = np.array(
        [wing_te_inner.p1[1] - nose_y, wing_te_inner.p1[1] - nose_y,
         wing_te_outer.p1[1] - nose_y, wing_te_outer.p2[1] - nose_y,
         (wing_le_outer.p1[1] + 817.0) - nose_y]
    ) / 1000.0
    grid = np.linspace(0.0, WING_SPAN_M / 2.0, 20_001)
    le_grid = np.interp(grid, lateral, le_station)
    te_grid = np.interp(grid, lateral, te_station)
    chord_grid = te_grid - le_grid
    reconstructed_area = 2.0 * np.trapezoid(chord_grid, grid)
    reconstructed_mac = 2.0 * np.trapezoid(chord_grid**2, grid) / reconstructed_area
    wing_ac_station = (
        2.0 * np.trapezoid(chord_grid * (le_grid + 0.25 * chord_grid), grid)
        / reconstructed_area
    )
    lemac_station = wing_ac_station - 0.25 * reconstructed_mac
    reference_cg_station = lemac_station + REFERENCE_CG_FRACTION_MAC * WING_MAC_M

    # The front-view upper and lower contours bracket the structural reference
    # line. Averaging their outboard slopes suppresses the projected thickness
    # taper and gives an exposed-wing dihedral suitable for a thin-surface model.
    wing_upper = m["wing_upper_outboard_right"]
    wing_lower_inner = m["wing_lower_outboard_inboard"]
    wing_lower_tip = m["wing_lower_outboard_tip"]
    upper_slope = (wing_upper.p1[1] - wing_upper.p2[1]) / (
        wing_upper.p1[0] - wing_upper.p2[0]
    )
    lower_slope = (wing_lower_tip.p2[1] - wing_lower_inner.p1[1]) / (
        wing_lower_tip.p2[0] - wing_lower_inner.p1[0]
    )
    wing_dihedral = math.degrees(math.atan((upper_slope + lower_slope) / 2.0))
    wing_root_drawing_y = np.mean(
        [m["wing_upper_root_right"].p2[1], m["wing_lower_root_right"].p1[1]]
    )

    # The side and front views share the drawing's vertical placement. Tail
    # centerlines are therefore expressed relative to the wing root midplane.
    h_front_lower = m["horizontal_tail_lower_right"]
    h_front_upper = m["horizontal_tail_upper_right"]
    h_lower_slope = (h_front_lower.p2[1] - h_front_lower.p1[1]) / (
        h_front_lower.p2[0] - h_front_lower.p1[0]
    )
    h_upper_slope = (h_front_upper.p1[1] - h_front_upper.p2[1]) / (
        h_front_upper.p1[0] - h_front_upper.p2[0]
    )
    h_lower_root_y = h_front_lower.p1[1] + h_lower_slope * (
        center_x - h_front_lower.p1[0]
    )
    h_upper_root_y = h_front_upper.p2[1] + h_upper_slope * (
        center_x - h_front_upper.p2[0]
    )
    h_dihedral = math.degrees(math.atan((h_lower_slope + h_upper_slope) / 2.0))
    h_root_vertical_offset = (
        (h_lower_root_y + h_upper_root_y) / 2.0 - wing_root_drawing_y
    ) / 1000.0
    v_root_vertical_offset = (v_root_y - wing_root_drawing_y) / 1000.0

    aileron_hinge = m["aileron_hinge_right"]
    wing_te = m["wing_trailing_edge_outboard_right"]
    wing_le = m["wing_leading_edge_outboard_right"]
    ax1, ax2 = sorted((aileron_hinge.p1[0], aileron_hinge.p2[0]))
    aileron_ratios = []
    aileron_chords = []
    for x in (ax1, ax2):
        hinge_y = _line_value(aileron_hinge.p1, aileron_hinge.p2, x)
        te_y = _line_value(wing_te.p1, wing_te.p2, x)
        le_y = _line_value(wing_le.p1, wing_le.p2, x)
        aileron_chords.append((te_y - hinge_y) / 1000.0)
        aileron_ratios.append((te_y - hinge_y) / (te_y - le_y))
    aileron_area = (ax2 - ax1) / 1000.0 * sum(aileron_chords) / 2.0 * 2.0

    return {
        "wing_reference": {
            "reconstructed_area_m2": float(reconstructed_area),
            "reconstructed_mac_m": float(reconstructed_mac),
            "lemac_station_from_nose_m": float(lemac_station),
            "reference_cg_station_from_nose_m": float(reference_cg_station),
            "reconstruction_area_error_fraction": float(reconstructed_area / 122.0 - 1.0),
            "exposed_dihedral_deg": wing_dihedral,
        },
        "horizontal_tail": {
            "area_m2": h_area,
            "span_m": HORIZONTAL_TAIL_SPAN_M,
            "root_chord_m": h_root,
            "tip_chord_m": h_tip,
            "taper_ratio": h_tip / h_root,
            "mean_aerodynamic_chord_m": h_mac,
            "quarter_chord_sweep_deg": h_sweep,
            "aerodynamic_center_station_from_nose_m": h_ac_station,
            "reference_moment_arm_m": h_ac_station - reference_cg_station,
            "dihedral_deg": h_dihedral,
            "root_vertical_offset_from_wing_m": h_root_vertical_offset,
        },
        "vertical_tail": {
            "area_m2": v_area,
            "reference_height_m": VERTICAL_TAIL_REFERENCE_HEIGHT_M,
            "root_chord_m": v_root,
            "tip_chord_m": v_tip,
            "taper_ratio": v_tip / v_root,
            "mean_aerodynamic_chord_m": v_mac,
            "quarter_chord_sweep_deg": v_sweep,
            "aerodynamic_center_station_from_nose_m": v_ac_station,
            "reference_moment_arm_m": v_ac_station - reference_cg_station,
            "root_vertical_offset_from_wing_m": v_root_vertical_offset,
        },
        "flight_controls": {
            "elevator": {
                "area_m2": elevator_area,
                "area_ratio": elevator_area / h_area,
                "root_chord_ratio": elevator_root / h_root,
                "tip_chord_ratio": elevator_tip / h_tip,
            },
            "rudder": {
                "area_m2": rudder_area,
                "area_ratio": rudder_area / v_area,
                "root_chord_ratio": rudder_root / v_root,
                "tip_chord_ratio": rudder_tip / v_tip,
            },
            "aileron": {
                "total_area_m2": aileron_area,
                "inboard_span_station_fraction_semispan": (ax1 - center_x) / (WING_SPAN_M * 500.0),
                "outboard_span_station_fraction_semispan": (ax2 - center_x) / (WING_SPAN_M * 500.0),
                "mean_chord_ratio": float(np.mean(aileron_ratios)),
            },
        },
    }


def _bounded(value: float, relative: float = 0.0, absolute: float = 0.0) -> dict[str, Any]:
    half_width = max(abs(value) * relative, absolute)
    return {
        "nominal": float(value),
        "lower": float(value - half_width),
        "upper": float(value + half_width),
        "uncertainty_type": "engineering-bound",
    }


def _inertia_prior() -> dict[str, Any]:
    slug_ft2_to_kg_m2 = 1.3558179483314004
    tcm_mass = 185_000.0 * 0.45359237
    tcm_span = 6.849 / 0.055 * 0.3048
    scale = (REFERENCE_MASS_KG / tcm_mass) * (WING_SPAN_M / tcm_span) ** 2
    source = {
        "ixx": 1_770_000.0 * slug_ft2_to_kg_m2,
        "iyy": 5_680_000.0 * slug_ft2_to_kg_m2,
        "izz": 7_270_000.0 * slug_ft2_to_kg_m2,
        "ixz": 160_000.0 * slug_ft2_to_kg_m2,
    }
    nominal = {key: value * scale for key, value in source.items()}
    diagonal = {
        key: {
            "nominal_kg_m2": value,
            "shape_factor_lower": 0.65,
            "shape_factor_upper": 1.35,
            "mass_scaling": f"I({{mass_kg}}) = {value:.8g} * mass_kg / {REFERENCE_MASS_KG:.0f} * shape_factor",
        }
        for key, value in nominal.items()
        if key != "ixz"
    }
    return {
        "status": "generic-transport-scaling-prior-not-a320-identification",
        "release_for_physical_a320_simulation": False,
        "reference_mass_kg": REFERENCE_MASS_KG,
        "qar_aggregate_mass_validity_range_kg": list(QAR_MASS_RANGE_KG),
        "source": "nasa_tcm_2011_217169",
        "source_reference_mass_kg": tcm_mass,
        "source_reference_span_m": tcm_span,
        "scaling": "mass ratio times squared wing-span ratio; all-axis use is provisional because the TCM report does not publish a fuselage-length scale",
        "diagonal": diagonal,
        "ixz": {
            "nominal_kg_m2_under_nasa_sign_convention": nominal["ixz"],
            "admissible_range_kg_m2": [-1.35 * abs(nominal["ixz"]), 1.35 * abs(nominal["ixz"])],
            "sign_status": "unresolved-until-aerox-body-axis-product-of-inertia-convention-is-frozen",
        },
        "positive_definite_gate": "ixx > 0, iyy > 0, izz > 0, and abs(ixz) < sqrt(ixx*izz)",
    }


def build_prior(measurement_path: Path) -> dict[str, Any]:
    measurements = load_measurements(measurement_path)
    geometry = _nominal_geometry(measurements)
    wing_reference = geometry["wing_reference"]
    wing_reference["lemac_station_bound_m"] = _bounded(
        wing_reference["lemac_station_from_nose_m"], absolute=0.25
    )
    wing_reference["reference_cg_station_bound_m"] = _bounded(
        wing_reference["reference_cg_station_from_nose_m"], absolute=0.35
    )
    wing_reference["exposed_dihedral_bound_deg"] = _bounded(
        wing_reference["exposed_dihedral_deg"], absolute=1.0
    )
    for section_name in ("horizontal_tail", "vertical_tail"):
        section = geometry[section_name]
        ac_station = section["aerodynamic_center_station_from_nose_m"]
        forward_cg = (
            wing_reference["lemac_station_from_nose_m"]
            + PLANNING_CG_RANGE_FRACTION_MAC[0] * WING_MAC_M
        )
        aft_cg = (
            wing_reference["lemac_station_from_nose_m"]
            + PLANNING_CG_RANGE_FRACTION_MAC[1] * WING_MAC_M
        )
        geometry[section_name] = {
            key: _bounded(
                value,
                relative=0.05 if key == "area_m2" else 0.0,
                absolute=(
                    1.0
                    if key == "quarter_chord_sweep_deg"
                    else 0.75
                    if key == "reference_moment_arm_m"
                    else 0.25
                    if key == "aerodynamic_center_station_from_nose_m"
                    else 0.10
                    if key == "taper_ratio"
                    else 0.15
                ),
            )
            if key not in {"span_m", "reference_height_m"}
            else _bounded(value, absolute=0.01)
            for key, value in section.items()
        }
        geometry[section_name]["planning_cg_moment_arm_range_m"] = {
            "lower": float(ac_station - aft_cg - 0.25),
            "upper": float(ac_station - forward_cg + 0.25),
            "definition": "Airbus ground-clearance planning CG range plus CAD/model-form station allowance; not a certified flight envelope",
        }
    for surface in geometry["flight_controls"].values():
        for key, value in list(surface.items()):
            surface[key] = _bounded(
                value,
                relative=0.08 if "area" in key else 0.0,
                absolute=0.04 if "ratio" in key or "station" in key else 0.10,
            )

    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "configuration": "A320-200 wing-tip-fence geometry, consistent with 2011 QAR aircraft",
        "status": "geometry-prior-available-inertia-remains-generic-and-unreleased",
        "source": {
            "source_id": "airbus_a320_autocad_wtf_2012_05",
            "source_dwg_sha256": SOURCE_DWG_SHA256,
            "drawing_accuracy_bound_m": AIRBUS_DRAWING_ACCURACY_MM / 1000.0,
            "measurement_table": str(measurement_path),
            "conversion": "LibreDWG 0.13.3 dwg2dxf --as r2000",
            "raw_source_redistributed": False,
        },
        "definitions": {
            "tail_planforms": "gross straight-edge trapezoids extrapolated to aircraft centerline/reference root; rounded tips are represented by the published full span/height",
            "positions": "longitudinal station aft of manufacturer-drawing nose; vertical tail offsets are relative to the front-view wing-root midplane",
            "reference_cg": "Airbus airport-planning 26.5% MAC value, not actual flight CG",
            "planning_cg_range_fraction_mac": list(PLANNING_CG_RANGE_FRACTION_MAC),
            "uncertainty": "engineering bounds combine the Airbus 0.10 m drawing-accuracy statement with planform-definition/model-form allowance",
        },
        "geometry": geometry,
        "mass_properties": _inertia_prior(),
        "identification_gate": {
            "source": "nasa_morelli_inertia_id_2021",
            "qar_inertia_identification_supported": False,
            "reason": "the published method requires purpose-designed maneuvers with high body-axis angular rates and simultaneous aerodynamic/inertia estimation; the commercial 1 Hz QAR has neither that excitation design nor independently known aerodynamic moments",
        },
        "release_decision": {
            "geometry_for_datcom_input": True,
            "geometry_for_vspaero_cross_check": True,
            "control_surface_deflection_or_rate_limits": False,
            "a320_inertia_tensor": False,
            "physical_lateral_model": False,
            "reason": "geometry is now sufficient for uncertain aerodynamic prior generation, but flight-specific CG, inertia, and independent coefficient validation remain unresolved",
        },
        "limitations": [
            "The Airbus drawing is published for airport planning and is accurate to within 10 cm; it is not an aerodynamic loft or control-system definition.",
            "Gross tail roots are straight-edge extrapolations beneath the fuselage/fillet, so exposed and wetted areas differ from these reference areas.",
            "Visible control boundaries do not establish hinge moments, gearing, deflection limits, actuator rates, or flight-control-law behavior.",
            "The wing reconstruction is used only to locate a planning CG and check scale; certified MAC is retained and the reconstructed area is not substituted for the Aerox reference area.",
            "The inertia prior is scaled from a NASA generic transport and must not be labelled as A320 data or fitted from the 1 Hz commercial QAR.",
        ],
    }


def _render_report(prior: dict[str, Any]) -> str:
    geometry = prior["geometry"]
    h = geometry["horizontal_tail"]
    v = geometry["vertical_tail"]
    controls = geometry["flight_controls"]
    mass = prior["mass_properties"]
    return f"""# A320 geometry and mass-property prior

## Outcome

The Airbus manufacturer three-view closes the geometry gate for uncertain
DATCOM and VSPAERO inputs. It does **not** close the physical-model release
gate. The derived gross horizontal-tail area is {h['area_m2']['nominal']:.2f}
m2 and the vertical-tail area is {v['area_m2']['nominal']:.2f} m2. Every
derived value is an `inferred-with-uncertainty` prior, not a certified datum.

The drawing is the A320 wing-tip-fence revision dated May 2012. That
configuration is appropriate for the 2011 operator recordings and avoids
silently substituting the later sharklet geometry. Airbus states that its
airport-planning three-views are accurate to within 10 cm.

## Geometry prior

| Quantity | Nominal | Engineering bound |
|---|---:|---:|
| Horizontal-tail area | {h['area_m2']['nominal']:.3f} m2 | {h['area_m2']['lower']:.3f} to {h['area_m2']['upper']:.3f} |
| Horizontal-tail MAC | {h['mean_aerodynamic_chord_m']['nominal']:.3f} m | {h['mean_aerodynamic_chord_m']['lower']:.3f} to {h['mean_aerodynamic_chord_m']['upper']:.3f} |
| Horizontal-tail quarter-chord sweep | {h['quarter_chord_sweep_deg']['nominal']:.2f} deg | {h['quarter_chord_sweep_deg']['lower']:.2f} to {h['quarter_chord_sweep_deg']['upper']:.2f} |
| Horizontal-tail dihedral | {h['dihedral_deg']['nominal']:.2f} deg | {h['dihedral_deg']['lower']:.2f} to {h['dihedral_deg']['upper']:.2f} |
| Horizontal-tail arm at planning CG | {h['reference_moment_arm_m']['nominal']:.3f} m | {h['planning_cg_moment_arm_range_m']['lower']:.3f} to {h['planning_cg_moment_arm_range_m']['upper']:.3f} over planning CG range |
| Vertical-tail area | {v['area_m2']['nominal']:.3f} m2 | {v['area_m2']['lower']:.3f} to {v['area_m2']['upper']:.3f} |
| Vertical-tail MAC | {v['mean_aerodynamic_chord_m']['nominal']:.3f} m | {v['mean_aerodynamic_chord_m']['lower']:.3f} to {v['mean_aerodynamic_chord_m']['upper']:.3f} |
| Vertical-tail quarter-chord sweep | {v['quarter_chord_sweep_deg']['nominal']:.2f} deg | {v['quarter_chord_sweep_deg']['lower']:.2f} to {v['quarter_chord_sweep_deg']['upper']:.2f} |
| Vertical-tail arm at planning CG | {v['reference_moment_arm_m']['nominal']:.3f} m | {v['planning_cg_moment_arm_range_m']['lower']:.3f} to {v['planning_cg_moment_arm_range_m']['upper']:.3f} over planning CG range |
| Elevator area ratio | {controls['elevator']['area_ratio']['nominal']:.3f} | {controls['elevator']['area_ratio']['lower']:.3f} to {controls['elevator']['area_ratio']['upper']:.3f} |
| Rudder area ratio | {controls['rudder']['area_ratio']['nominal']:.3f} | {controls['rudder']['area_ratio']['lower']:.3f} to {controls['rudder']['area_ratio']['upper']:.3f} |
| Aileron semispan stations | {controls['aileron']['inboard_span_station_fraction_semispan']['nominal']:.3f} to {controls['aileron']['outboard_span_station_fraction_semispan']['nominal']:.3f} | plus/minus 0.04 |

The simplified wing outline reconstructs
{geometry['wing_reference']['reconstructed_area_m2']:.2f} m2 and a
{geometry['wing_reference']['reconstructed_mac_m']:.3f} m MAC. Its
{geometry['wing_reference']['reconstruction_area_error_fraction'] * 100:.1f}%
area difference from the 122 m2 reference is small enough to use the drawing
for a planning station estimate, but it is not substituted for the reference
area or certified 4.1935 m MAC.

## CG and inertia boundary

The nominal moment arms use Airbus's 26.5%-MAC airport-planning CG. The same
planning document shows 17% and 40% MAC configurations for ground-clearance
calculations; those values are retained only as a sensitivity range, not as
certified or observed flight CG limits.

No A320 inertia tensor was found. The machine-readable artifact therefore
contains a deliberately wide mass-scaled prior derived from NASA's 185,000-lb
generic Transport Class Model. At the {mass['reference_mass_kg']:.0f} kg
reference mass its diagonal nominal values are:

- Ixx: {mass['diagonal']['ixx']['nominal_kg_m2']:.0f} kg m2
- Iyy: {mass['diagonal']['iyy']['nominal_kg_m2']:.0f} kg m2
- Izz: {mass['diagonal']['izz']['nominal_kg_m2']:.0f} kg m2

Each diagonal carries a 0.65--1.35 shape factor in addition to linear mass
scaling. `Ixz` permits either sign until the product-of-inertia convention is
frozen. These are numerical conditioning and sensitivity priors only.

NASA's flight-data inertia-identification method requires deliberately
designed maneuvers with high body-axis angular rates while aerodynamic and
inertia parameters are estimated together. The 1 Hz commercial QAR does not
meet that excitation or observability requirement, so it is not used to turn
the generic prior into an A320 estimate.

## Decision

- Geometry input for DATCOM: **available with bounds**.
- Geometry input for a VSPAERO cross-check: **available with bounds**.
- Surface deflection/rate limits: **unresolved**.
- A320 inertia tensor: **not released**.
- Physical lateral-directional model: **not released**.
"""


def _plot(prior: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    h = prior["geometry"]["horizontal_tail"]
    v = prior["geometry"]["vertical_tail"]
    controls = prior["geometry"]["flight_controls"]
    mass = prior["mass_properties"]
    colors = {"tail": "#0072B2", "control": "#E69F00", "prior": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    half_span = h["span_m"]["nominal"] / 2
    hr, ht = h["root_chord_m"]["nominal"], h["tip_chord_m"]["nominal"]
    sweep = math.radians(h["quarter_chord_sweep_deg"]["nominal"])
    q_tip = math.tan(sweep) * half_span
    le_tip = q_tip - 0.25 * ht
    axes[0].fill([0, half_span, half_span, 0], [0, le_tip, le_tip + ht, hr], color=colors["tail"], alpha=0.28)
    er = controls["elevator"]["root_chord_ratio"]["nominal"] * hr
    et = controls["elevator"]["tip_chord_ratio"]["nominal"] * ht
    axes[0].fill([0, half_span, half_span, 0], [hr-er, le_tip+ht-et, le_tip+ht, hr], color=colors["control"], alpha=0.75)
    axes[0].set(title="Horizontal tail (half-planform)", xlabel="semispan [m]", ylabel="aft coordinate [m]")
    axes[0].axis("equal"); axes[0].grid(alpha=0.25)

    height = v["reference_height_m"]["nominal"]
    vr, vt = v["root_chord_m"]["nominal"], v["tip_chord_m"]["nominal"]
    vq_tip = math.tan(math.radians(v["quarter_chord_sweep_deg"]["nominal"])) * height
    vle_tip = vq_tip - 0.25 * vt
    axes[1].fill([0, vr, vle_tip+vt, vle_tip], [0, 0, height, height], color=colors["tail"], alpha=0.28)
    rr = controls["rudder"]["root_chord_ratio"]["nominal"] * vr
    rt = controls["rudder"]["tip_chord_ratio"]["nominal"] * vt
    axes[1].fill([vr-rr, vr, vle_tip+vt, vle_tip+vt-rt], [0, 0, height, height], color=colors["control"], alpha=0.75)
    axes[1].set(title="Vertical tail", xlabel="aft coordinate [m]", ylabel="height [m]")
    axes[1].axis("equal"); axes[1].grid(alpha=0.25)

    labels = ["Ixx", "Iyy", "Izz"]
    values = [mass["diagonal"][key.lower()]["nominal_kg_m2"] / 1e6 for key in labels]
    low = [value * 0.65 for value in values]
    high = [value * 1.35 for value in values]
    axes[2].bar(labels, values, color=colors["prior"], alpha=0.8)
    axes[2].errorbar(labels, values, yerr=[np.array(values)-low, np.array(high)-values], fmt="none", color="#222222", capsize=5)
    axes[2].set(title="Generic inertia prior at 60 t", ylabel="inertia [million kg m2]")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].text(0.02, 0.98, "Not A320-identified", transform=axes[2].transAxes, va="top", color="#A33A2B", fontweight="bold")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    measurements = load_measurements(args.measurements)
    verification = {"verified": False, "reason": "no --dxf supplied"}
    if args.dxf:
        verification = verify_dxf(args.dxf, measurements)
    if args.source_dwg:
        digest = hashlib.sha256(args.source_dwg.read_bytes()).hexdigest()
        if digest != SOURCE_DWG_SHA256:
            raise ValueError(f"Unexpected source DWG SHA256: {digest}")
        verification["source_dwg_verified"] = True

    prior = build_prior(args.measurements)
    prior["source"]["external_verification"] = verification
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(prior, sort_keys=False))
    args.report.write_text(_render_report(prior))
    _plot(prior, args.figure_prefix)


if __name__ == "__main__":
    main()
