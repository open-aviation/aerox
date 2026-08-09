#!/usr/bin/env python3
"""Extend the bounded A320 VSPAERO lift table from Mach 0.78 to 0.82.

This is an inviscid numerical-support experiment.  It releases no VSPAERO drag,
wave-drag, buffet, stall, or transonic-fidelity claim.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_a320_geometry_mass_prior import _nominal_geometry, load_measurements
from tools.build_a320_vspaero_prior import _bootstrap_openvsp, _openvsp_root, _run_case

UPDATED = "2026-08-06"
BASE_MACH = 0.78
NEW_MACH_CASES = (0.80, 0.82)
ALPHA_CASES_DEG = (-1.0, 2.0, 5.0, 8.5)
REFERENCE_ALPHA_DEG = 2.0

# Frozen before the new VSPAERO cases are inspected.  The first two gates test
# whether the tabulated lift response remains usable for bounded interpolation;
# the last limits the departure from the established M=0.78 solver slope.
MAX_ABSOLUTE_CL_LINEARITY_ERROR = 0.02
MIN_ADJACENT_CL_SLOPE_PER_RAD = 3.0
MAX_CL_ALPHA_RELATIVE_CHANGE_FROM_M078 = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("docs/modeling/a320-cad-geometry-measurements.csv"),
    )
    parser.add_argument(
        "--geometry-prior",
        type=Path,
        default=Path("docs/modeling/a320-geometry-mass-prior.yml"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.csv"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-mach-extension.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-mach-extension.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-mach-extension"),
    )
    parser.add_argument("--openvsp-root", type=Path)
    parser.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--keep-workdir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return [
            {
                key: int(value)
                if key in {"vortex_loop_equations", "geometry_nodes"}
                else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def _derive(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["mach"], row["alpha_deg"]): row for row in raw_rows}
    rows: list[dict[str, Any]] = []
    for mach in NEW_MACH_CASES:
        reference = indexed[(mach, REFERENCE_ALPHA_DEG)]
        for alpha in ALPHA_CASES_DEG:
            row = indexed[(mach, alpha)]
            delta = np.deg2rad(alpha - REFERENCE_ALPHA_DEG)
            cl_linear = reference["CL"] + reference["CL_alpha"] * delta
            cmm_linear = reference["CMm"] + reference["CMm_alpha"] * delta
            rows.append(
                {
                    "mach": mach,
                    "alpha_deg": alpha,
                    "CL": row["CL"],
                    "CL_reference_linear": cl_linear,
                    "CL_linearity_error": row["CL"] - cl_linear,
                    "CL_alpha_per_rad": row["CL_alpha"],
                    "CL_alpha_relative_change_from_alpha2": abs(
                        row["CL_alpha"] / reference["CL_alpha"] - 1.0
                    ),
                    "CMm": row["CMm"],
                    "CMm_reference_linear": cmm_linear,
                    "CMm_linearity_error": row["CMm"] - cmm_linear,
                    "CMm_alpha_per_rad": row["CMm_alpha"],
                    "CMm_alpha_relative_change_from_alpha2": abs(
                        row["CMm_alpha"] / reference["CMm_alpha"] - 1.0
                    ),
                    "vortex_loop_equations": row["vortex_loop_equations"],
                    "geometry_nodes": row["geometry_nodes"],
                }
            )
    return rows


def build_document(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    openvsp_version: str,
    vspaero_version: str,
    jobs: int,
) -> dict[str, Any]:
    reference = {
        row["alpha_deg"]: row
        for row in baseline_rows
        if np.isclose(row["mach"], BASE_MACH)
    }
    absolute_linearity_error = max(abs(row["CL_linearity_error"]) for row in rows)
    adjacent_slopes = []
    for mach in NEW_MACH_CASES:
        current = sorted(
            (row for row in rows if row["mach"] == mach),
            key=lambda row: row["alpha_deg"],
        )
        adjacent_slopes.extend(
            (right["CL"] - left["CL"])
            / np.deg2rad(right["alpha_deg"] - left["alpha_deg"])
            for left, right in zip(current[:-1], current[1:], strict=True)
        )
    relative_change = max(
        abs(row["CL_alpha_per_rad"] / reference[row["alpha_deg"]]["CL_alpha_per_rad"] - 1.0)
        for row in rows
    )
    gates = {
        "absolute_cl_linearity_error": {
            "value": float(absolute_linearity_error),
            "maximum": MAX_ABSOLUTE_CL_LINEARITY_ERROR,
            "passed": bool(absolute_linearity_error <= MAX_ABSOLUTE_CL_LINEARITY_ERROR),
        },
        "minimum_adjacent_cl_slope_per_rad": {
            "value": float(min(adjacent_slopes)),
            "minimum": MIN_ADJACENT_CL_SLOPE_PER_RAD,
            "passed": bool(min(adjacent_slopes) >= MIN_ADJACENT_CL_SLOPE_PER_RAD),
        },
        "cl_alpha_relative_change_from_mach_078": {
            "value": float(relative_change),
            "maximum": MAX_CL_ALPHA_RELATIVE_CHANGE_FROM_M078,
            "passed": bool(relative_change <= MAX_CL_ALPHA_RELATIVE_CHANGE_FROM_M078),
        },
    }
    passed = all(item["passed"] for item in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "numerical-mach-extension-passed" if passed else "numerical-mach-extension-held",
        "software": {
            "openvsp_version": openvsp_version,
            "vspaero_version": vspaero_version,
            "analysis_method": "thin-surface vortex lattice",
            "mesh": "medium",
            "jobs": jobs,
        },
        "experiment": {
            "baseline_mach": BASE_MACH,
            "new_mach": list(NEW_MACH_CASES),
            "alpha_deg": list(ALPHA_CASES_DEG),
            "baseline_table": "docs/modeling/a320-vspaero-alpha-support.csv",
            "extended_table": "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv",
            "selection_frozen_before_results": True,
        },
        "gates": gates,
        "decision": {
            "release_extended_bounded_lift_surface": passed,
            "maximum_mach": max(NEW_MACH_CASES) if passed else BASE_MACH,
            "release_vspaero_drag": False,
            "claim_transonic_fidelity": False,
            "reason": "The extension is only a bounded lift prior. Thin-surface VLM omits wave drag, shocks, buffet, and viscous separation.",
        },
        "limitations": [
            "Mach 0.80 and 0.82 are above the A320 nominal cruise Mach and are included only to close the observed replay boundary.",
            "No public A320 transonic wind-tunnel or flight-test polar validates these nodes.",
            "The table must retain a smooth empirical fallback at every alpha and Mach boundary.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(document: dict[str, Any]) -> str:
    gates = document["gates"]
    passed = document["decision"]["release_extended_bounded_lift_surface"]
    return f"""# A320 VSPAERO Mach-extension audit

## Outcome

The predeclared numerical screen for the bounded lift-only extension to Mach
0.82 **{'passes' if passed else 'does not pass'}**.

| Gate | Observed | Bound | Result |
|---|---:|---:|---:|
| Absolute CL local-linearity error | {gates['absolute_cl_linearity_error']['value']:.5f} | <= {gates['absolute_cl_linearity_error']['maximum']:.5f} | {'pass' if gates['absolute_cl_linearity_error']['passed'] else 'fail'} |
| Minimum adjacent CL slope (/rad) | {gates['minimum_adjacent_cl_slope_per_rad']['value']:.3f} | >= {gates['minimum_adjacent_cl_slope_per_rad']['minimum']:.3f} | {'pass' if gates['minimum_adjacent_cl_slope_per_rad']['passed'] else 'fail'} |
| Maximum CLalpha change from M=0.78 | {100*gates['cl_alpha_relative_change_from_mach_078']['value']:.2f}% | <= {100*gates['cl_alpha_relative_change_from_mach_078']['maximum']:.1f}% | {'pass' if gates['cl_alpha_relative_change_from_mach_078']['passed'] else 'fail'} |

## Interpretation

Passing authorizes only bounded interpolation of VSPAERO lift through Mach 0.82.
It does not validate transonic A320 aerodynamics: the thin-surface solver has no
shock, wave-drag, buffet, or viscous-separation model. VSPAERO drag remains
excluded, and the empirical lift fallback remains active near all boundaries.

## Frozen next gate

Use the unchanged fitted pitch/angle registration parameters and a Mach core of
0.23--0.79 with a smooth fade to zero surface weight at Mach 0.82. Confirm once
on the independently selected six-flight continuation panel. Do not make the
hybrid the default unless every frozen trajectory gate passes.
"""


def _plot(rows: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {0.78: "#0072B2", 0.80: "#D55E00", 0.82: "#009E73"}
    markers = {0.78: "o", 0.80: "s", 0.82: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for mach in (BASE_MACH, *NEW_MACH_CASES):
        current = sorted(
            (row for row in rows if np.isclose(row["mach"], mach)),
            key=lambda row: row["alpha_deg"],
        )
        axes[0].plot(
            [row["alpha_deg"] for row in current],
            [row["CL"] for row in current],
            marker=markers[mach],
            color=colors[mach],
            label=f"M={mach:.2f}",
        )
        axes[1].plot(
            [row["alpha_deg"] for row in current],
            [row["CL_alpha_per_rad"] for row in current],
            marker=markers[mach],
            color=colors[mach],
        )
    axes[0].set(xlabel="Angle of attack (deg)", ylabel="$C_L$", title="Bounded lift table")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Angle of attack (deg)", ylabel="$C_{L_\\alpha}$ (/rad)", title="Solver-reported local slope")
    for label, axis in zip("AB", axes, strict=True):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", va="top")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A320 VSPAERO lift-only Mach extension")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    baseline_rows = _read_rows(args.baseline)
    baseline_grid = {
        (row["mach"], row["alpha_deg"]) for row in baseline_rows
    }
    expected = {
        (mach, alpha)
        for mach in (0.20, 0.50, BASE_MACH)
        for alpha in ALPHA_CASES_DEG
    }
    if baseline_grid != expected:
        raise ValueError("baseline lift table is not the frozen 3 x 4 grid")

    root = _openvsp_root(args.openvsp_root)
    _bootstrap_openvsp(root)
    import openvsp as vsp  # type: ignore[import-not-found]

    vsp.SetVSPAEROPath(str(root))
    if not vsp.CheckForVSPAERO(vsp.GetVSPAEROPath()):
        raise RuntimeError(f"VSPAERO executable was not found below {root}")
    measurements = load_measurements(args.measurements)
    geometry = _nominal_geometry(measurements)
    geometry_prior = yaml.safe_load(args.geometry_prior.read_text())
    assert geometry_prior["geometry"]["horizontal_tail"]["area_m2"]["nominal"] == geometry["horizontal_tail"]["area_m2"]

    if args.keep_workdir:
        workdir = args.keep_workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup: contextlib.AbstractContextManager[Any] = contextlib.nullcontext(workdir)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="aerox-a320-vspaero-mach-")
    with cleanup as directory:
        raw_rows = []
        for mach in NEW_MACH_CASES:
            for alpha in ALPHA_CASES_DEG:
                row, _ = _run_case(
                    vsp,
                    Path(directory) / f"m{round(100*mach):02d}_a{alpha:+04.1f}",
                    "medium",
                    mach,
                    geometry,
                    measurements,
                    args.jobs,
                    args.verbose,
                    alpha_deg=alpha,
                )
                raw_rows.append(row)

    new_rows = _derive(raw_rows)
    all_rows = sorted(
        [*baseline_rows, *new_rows], key=lambda row: (row["mach"], row["alpha_deg"])
    )
    process = subprocess.run(
        [str(root / "vspaero")], capture_output=True, text=True, check=False, timeout=10
    )
    match = re.search(r"VSPAERO v\.([0-9.]+)", process.stdout + process.stderr)
    document = build_document(
        new_rows,
        baseline_rows,
        openvsp_version=vsp.GetVSPVersion().replace("OpenVSP ", ""),
        vspaero_version=match.group(1) if match else "unknown",
        jobs=args.jobs,
    )
    _write_csv(args.results, all_rows)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_render_report(document))
    _plot(all_rows, args.figure_prefix)
    print(document["status"])


if __name__ == "__main__":
    main()
