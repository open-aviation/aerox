#!/usr/bin/env python3
"""Test the local-linear A320 VSPAERO prior across the fitted angle support."""

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
from tools.build_a320_vspaero_prior import (
    _bootstrap_openvsp,
    _openvsp_root,
    _run_case,
)

UPDATED = "2026-08-05"
MACH_CASES = (0.20, 0.50, 0.78)
ALPHA_CASES_DEG = (-1.0, 2.0, 5.0, 8.5)
REFERENCE_ALPHA_DEG = 2.0
MAX_CL_LINEARITY_ERROR = 0.02
MAX_CL_ALPHA_RELATIVE_CHANGE = 0.10
MAX_CMM_LINEARITY_ERROR = 0.02
MAX_CMM_ALPHA_RELATIVE_CHANGE = 0.15


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
        "--results",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-alpha-support"),
    )
    parser.add_argument("--openvsp-root", type=Path)
    parser.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--keep-workdir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _derived_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["mach"], row["alpha_deg"]): row for row in raw_rows}
    rows: list[dict[str, Any]] = []
    for mach in MACH_CASES:
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


def build_audit(
    rows: list[dict[str, Any]], *, openvsp_version: str, vspaero_version: str, jobs: int
) -> dict[str, Any]:
    maximums = {
        "absolute_cl_linearity_error": float(
            max(abs(row["CL_linearity_error"]) for row in rows)
        ),
        "cl_alpha_relative_change": float(
            max(row["CL_alpha_relative_change_from_alpha2"] for row in rows)
        ),
        "absolute_cmm_linearity_error": float(
            max(abs(row["CMm_linearity_error"]) for row in rows)
        ),
        "cmm_alpha_relative_change": float(
            max(row["CMm_alpha_relative_change_from_alpha2"] for row in rows)
        ),
    }
    gates = {
        "absolute_cl_linearity_error": {
            "value": maximums["absolute_cl_linearity_error"],
            "maximum": MAX_CL_LINEARITY_ERROR,
            "passed": maximums["absolute_cl_linearity_error"] <= MAX_CL_LINEARITY_ERROR,
        },
        "cl_alpha_relative_change": {
            "value": maximums["cl_alpha_relative_change"],
            "maximum": MAX_CL_ALPHA_RELATIVE_CHANGE,
            "passed": maximums["cl_alpha_relative_change"] <= MAX_CL_ALPHA_RELATIVE_CHANGE,
        },
        "absolute_cmm_linearity_error": {
            "value": maximums["absolute_cmm_linearity_error"],
            "maximum": MAX_CMM_LINEARITY_ERROR,
            "passed": maximums["absolute_cmm_linearity_error"] <= MAX_CMM_LINEARITY_ERROR,
        },
        "cmm_alpha_relative_change": {
            "value": maximums["cmm_alpha_relative_change"],
            "maximum": MAX_CMM_ALPHA_RELATIVE_CHANGE,
            "passed": maximums["cmm_alpha_relative_change"] <= MAX_CMM_ALPHA_RELATIVE_CHANGE,
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "numerical-alpha-support-passed" if passed else "numerical-alpha-support-held",
        "software": {
            "openvsp_version": openvsp_version,
            "vspaero_version": vspaero_version,
            "analysis_method": "thin-surface vortex lattice",
            "mesh": "medium",
            "jobs": jobs,
        },
        "experiment": {
            "mach": list(MACH_CASES),
            "alpha_deg": list(ALPHA_CASES_DEG),
            "reference_alpha_deg": REFERENCE_ALPHA_DEG,
            "geometry": "same generated A320 wing-and-tail model as docs/modeling/a320-vspaero-prior.yml",
            "table": "docs/modeling/a320-vspaero-alpha-support.csv",
        },
        "gates": gates,
        "decision": {
            "extend_local_linear_static_prior_alpha_upper_bound_to_deg": 8.5 if passed else None,
            "physical_stall_or_high_alpha_model": False,
            "viscous_drag_model": False,
            "open_angle_lift_validation": False,
            "reason": "This establishes only internal VSPAERO linearity over the fitted clean-flight angle support; it does not add missing airfoil, viscous, interference, or stall physics.",
        },
        "limitations": [
            "The sweep uses the same symmetric thin-surface geometry and omissions as the released VSPAERO prior.",
            "Passing demonstrates numerical support for local-linear interpolation, not certified A320 aerodynamic fidelity.",
            "Angles above 8.5 degrees, high-lift configurations, stall, and separated flow remain outside scope.",
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
    passed = document["status"] == "numerical-alpha-support-passed"
    return f"""# A320 VSPAERO alpha-support audit

## Outcome

The medium-mesh VSPAERO sweep **{'passes' if passed else 'does not pass'}** the
predeclared numerical linearity screen from −1° through 8.5° at Mach 0.20,
0.50, and 0.78.

| Gate | Maximum observed | Limit | Result |
|---|---:|---:|---:|
| Absolute CL linearity error | {gates['absolute_cl_linearity_error']['value']:.5f} | {gates['absolute_cl_linearity_error']['maximum']:.5f} | {'pass' if gates['absolute_cl_linearity_error']['passed'] else 'fail'} |
| CLalpha relative change | {100*gates['cl_alpha_relative_change']['value']:.2f}% | {100*gates['cl_alpha_relative_change']['maximum']:.1f}% | {'pass' if gates['cl_alpha_relative_change']['passed'] else 'fail'} |
| Absolute CMm linearity error | {gates['absolute_cmm_linearity_error']['value']:.5f} | {gates['absolute_cmm_linearity_error']['maximum']:.5f} | {'pass' if gates['absolute_cmm_linearity_error']['passed'] else 'fail'} |
| CMm_alpha relative change | {100*gates['cmm_alpha_relative_change']['value']:.2f}% | {100*gates['cmm_alpha_relative_change']['maximum']:.1f}% | {'pass' if gates['cmm_alpha_relative_change']['passed'] else 'fail'} |

## Boundary

This is a numerical support test for the existing inviscid geometry prior. It
does not validate an A320 stall model, extend the clean-flight model above
8.5°, or authorize VSPAERO drag. Validation flights are not involved.

## Decision

- Extend the local-linear static-prior upper angle bound to 8.5°: **{'yes' if passed else 'no'}**.
- Physical high-angle or stall model: **no**.
- Viscous drag from VSPAERO: **no**.
- Open QAR validation: **no**.
"""


def _plot(rows: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {0.20: "#0072B2", 0.50: "#D55E00", 0.78: "#009E73"}
    markers = {0.20: "o", 0.50: "s", 0.78: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    flat = axes.ravel()
    for mach in MACH_CASES:
        subset = [row for row in rows if row["mach"] == mach]
        alpha = [row["alpha_deg"] for row in subset]
        flat[0].plot(alpha, [row["CL"] for row in subset], marker=markers[mach], color=colors[mach], label=f"M={mach:.2f}")
        flat[0].plot(alpha, [row["CL_reference_linear"] for row in subset], linestyle="--", color=colors[mach], alpha=0.7)
        flat[1].plot(alpha, 100*np.asarray([row["CL_alpha_relative_change_from_alpha2"] for row in subset]), marker=markers[mach], color=colors[mach])
        flat[2].plot(alpha, [row["CMm"] for row in subset], marker=markers[mach], color=colors[mach])
        flat[2].plot(alpha, [row["CMm_reference_linear"] for row in subset], linestyle="--", color=colors[mach], alpha=0.7)
        flat[3].plot(alpha, 100*np.asarray([row["CMm_alpha_relative_change_from_alpha2"] for row in subset]), marker=markers[mach], color=colors[mach])
    flat[0].set(title="Lift linearity", xlabel="Angle of attack (deg)", ylabel="CL")
    flat[0].legend(frameon=False)
    flat[1].axhline(100*MAX_CL_ALPHA_RELATIVE_CHANGE, color="black", linestyle="--", linewidth=1)
    flat[1].set(title="Lift-slope variation", xlabel="Angle of attack (deg)", ylabel="Change from alpha=2° (%)")
    flat[2].set(title="Pitch-moment linearity", xlabel="Angle of attack (deg)", ylabel="CMm")
    flat[3].axhline(100*MAX_CMM_ALPHA_RELATIVE_CHANGE, color="black", linestyle="--", linewidth=1)
    flat[3].set(title="Pitch-slope variation", xlabel="Angle of attack (deg)", ylabel="Change from alpha=2° (%)")
    for label, axis in zip("ABCD", flat, strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 VSPAERO local-linear alpha support")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = _openvsp_root(args.openvsp_root)
    _bootstrap_openvsp(root)
    import openvsp as vsp  # type: ignore[import-not-found]

    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
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
        cleanup = tempfile.TemporaryDirectory(prefix="aerox-a320-vspaero-alpha-")
    with cleanup as directory:
        raw_rows: list[dict[str, Any]] = []
        for mach in MACH_CASES:
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

    openvsp_version = vsp.GetVSPVersion().replace("OpenVSP ", "")
    process = subprocess.run([str(root / "vspaero")], capture_output=True, text=True, check=False, timeout=10)
    match = re.search(r"VSPAERO v\.([0-9.]+)", process.stdout + process.stderr)
    vspaero_version = match.group(1) if match else "unknown"
    rows = _derived_rows(raw_rows)
    document = build_audit(rows, openvsp_version=openvsp_version, vspaero_version=vspaero_version, jobs=args.jobs)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_csv(args.results, rows)
    args.report.write_text(_render_report(document))
    _plot(rows, args.figure_prefix)
    print(document["status"])


if __name__ == "__main__":
    main()
