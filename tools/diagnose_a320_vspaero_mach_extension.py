#!/usr/bin/env python3
"""Diagnose the isolated M=0.80, alpha=8.5 VSPAERO lift-derivative drop."""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_a320_vspaero_mach_extension import _read_rows
from tools.build_a320_geometry_mass_prior import _nominal_geometry, load_measurements
from tools.build_a320_vspaero_prior import _bootstrap_openvsp, _openvsp_root, _run_case

UPDATED = "2026-08-06"
MACH = 0.80
ALPHA_CASES_DEG = (8.25, 8.50, 8.75)
MESHES = ("coarse", "medium", "fine")
MIN_SECANT_CL_ALPHA_PER_RAD = 5.0
MAX_SECANT_CL_ALPHA_PER_RAD = 10.0
MAX_MEDIUM_FINE_CL_RELATIVE_DIFFERENCE = 0.02
MAX_MEDIUM_FINE_SECANT_RELATIVE_DIFFERENCE = 0.10
MIN_REPORTED_DERIVATIVE_DISAGREEMENT = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=Path("docs/modeling/a320-cad-geometry-measurements.csv"))
    parser.add_argument("--geometry-prior", type=Path, default=Path("docs/modeling/a320-geometry-mass-prior.yml"))
    parser.add_argument("--extended-table", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"))
    parser.add_argument("--results", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extension-diagnostic.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extension-diagnostic.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extension-diagnostic.md"))
    parser.add_argument("--figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-vspaero-mach-extension-diagnostic"))
    parser.add_argument("--openvsp-root", type=Path)
    parser.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--keep-workdir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


def build_document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(row["mesh"], row["alpha_deg"]): row for row in rows}
    secants = {
        mesh: (indexed[(mesh, 8.75)]["CL"] - indexed[(mesh, 8.25)]["CL"])
        / np.deg2rad(0.50)
        for mesh in MESHES
    }
    medium_fine_cl = max(
        _relative(indexed[("medium", alpha)]["CL"], indexed[("fine", alpha)]["CL"])
        for alpha in ALPHA_CASES_DEG
    )
    secant_difference = _relative(secants["medium"], secants["fine"])
    reported = indexed[("medium", 8.50)]["CL_alpha"]
    derivative_disagreement = _relative(reported, secants["medium"])
    gates = {
        "medium_secant_cl_alpha_lower": {
            "value": float(secants["medium"]),
            "minimum": MIN_SECANT_CL_ALPHA_PER_RAD,
            "passed": bool(secants["medium"] >= MIN_SECANT_CL_ALPHA_PER_RAD),
        },
        "medium_secant_cl_alpha_upper": {
            "value": float(secants["medium"]),
            "maximum": MAX_SECANT_CL_ALPHA_PER_RAD,
            "passed": bool(secants["medium"] <= MAX_SECANT_CL_ALPHA_PER_RAD),
        },
        "medium_fine_cl_relative_difference": {
            "value": float(medium_fine_cl),
            "maximum": MAX_MEDIUM_FINE_CL_RELATIVE_DIFFERENCE,
            "passed": bool(medium_fine_cl <= MAX_MEDIUM_FINE_CL_RELATIVE_DIFFERENCE),
        },
        "medium_fine_secant_relative_difference": {
            "value": float(secant_difference),
            "maximum": MAX_MEDIUM_FINE_SECANT_RELATIVE_DIFFERENCE,
            "passed": bool(secant_difference <= MAX_MEDIUM_FINE_SECANT_RELATIVE_DIFFERENCE),
        },
        "reported_derivative_disagreement": {
            "value": float(derivative_disagreement),
            "minimum": MIN_REPORTED_DERIVATIVE_DISAGREEMENT,
            "passed": bool(derivative_disagreement >= MIN_REPORTED_DERIVATIVE_DISAGREEMENT),
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "base-lift-extension-diagnostic-passed" if passed else "base-lift-extension-diagnostic-held",
        "experiment": {
            "mach": MACH,
            "alpha_deg": list(ALPHA_CASES_DEG),
            "meshes": list(MESHES),
            "finite_difference_width_deg": 0.50,
            "trigger": "The predeclared extension gate failed because the solver-reported CL_alpha at M=0.80, alpha=8.5 deg was discontinuous while base CL remained smooth.",
            "results": "docs/modeling/a320-vspaero-mach-extension-diagnostic.csv",
        },
        "derived": {
            "secant_cl_alpha_per_rad": {mesh: float(value) for mesh, value in secants.items()},
            "medium_reported_cl_alpha_per_rad": float(reported),
        },
        "gates": gates,
        "decision": {
            "reported_local_derivative_is_numerically_reliable_here": False if passed else None,
            "allow_base_cl_table_for_bounded_interpolation": passed,
            "allow_local_derivative_column_for_modeling": False,
            "release_vspaero_drag": False,
            "claim_transonic_fidelity": False,
            "reason": "The executable lift surface interpolates base CL nodes only. A converged finite-difference slope can isolate a stability-output derivative artifact but cannot add missing transonic physics.",
        },
        "limitations": [
            "This is a post-failure solver diagnostic and is reported separately from the failed predeclared derivative gate.",
            "It cannot validate shock, wave-drag, buffet, stall, or viscous effects.",
            "Independent QAR confirmation remains unopened until this numerical diagnostic passes.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["mesh", "mach", "alpha_deg", "CL", "CL_alpha", "CMm", "CMm_alpha", "vortex_loop_equations", "geometry_nodes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)


def _render_report(document: dict[str, Any]) -> str:
    gates = document["gates"]
    derived = document["derived"]
    passed = document["decision"]["allow_base_cl_table_for_bounded_interpolation"]
    return f"""# A320 VSPAERO Mach-extension derivative diagnostic

## Outcome

The post-failure mesh/finite-difference diagnostic **{'supports' if passed else 'does not support'}** treating the isolated solver `CL_alpha` drop as a derivative-output artifact. The original extension gate remains recorded as failed; this audit tests the base-`CL` quantity actually used by `TabulatedLiftSurface`.

| Quantity | Value |
|---|---:|
| Coarse central secant CLalpha | {derived['secant_cl_alpha_per_rad']['coarse']:.3f} /rad |
| Medium central secant CLalpha | {derived['secant_cl_alpha_per_rad']['medium']:.3f} /rad |
| Fine central secant CLalpha | {derived['secant_cl_alpha_per_rad']['fine']:.3f} /rad |
| Medium solver-reported CLalpha | {derived['medium_reported_cl_alpha_per_rad']:.3f} /rad |
| Maximum medium/fine base-CL difference | {100*gates['medium_fine_cl_relative_difference']['value']:.2f}% |
| Medium/fine secant difference | {100*gates['medium_fine_secant_relative_difference']['value']:.2f}% |
| Reported-derivative disagreement | {100*gates['reported_derivative_disagreement']['value']:.2f}% |

## Decision boundary

- Use converged base `CL` nodes for bounded interpolation: **{'yes' if passed else 'no'}**.
- Use the anomalous stability-output derivative: **no**.
- Claim transonic fidelity or use VSPAERO drag: **no**.
- Open independent replay confirmation: **{'yes' if passed else 'no'}**, with the already frozen hybrid and gates only.
"""


def _plot(rows: list[dict[str, Any]], document: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"coarse": "#D55E00", "medium": "#0072B2", "fine": "#009E73"}
    markers = {"coarse": "s", "medium": "o", "fine": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for mesh in MESHES:
        current = sorted((row for row in rows if row["mesh"] == mesh), key=lambda row: row["alpha_deg"])
        axes[0].plot([row["alpha_deg"] for row in current], [row["CL"] for row in current], marker=markers[mesh], color=colors[mesh], label=mesh)
    axes[0].set(xlabel="Angle of attack (deg)", ylabel="$C_L$", title="Base-coefficient mesh convergence")
    axes[0].legend(frameon=False)
    secants = document["derived"]["secant_cl_alpha_per_rad"]
    axes[1].bar(["coarse", "medium", "fine", "reported\nmedium"], [secants["coarse"], secants["medium"], secants["fine"], document["derived"]["medium_reported_cl_alpha_per_rad"]], color=[colors["coarse"], colors["medium"], colors["fine"], "#CC79A7"])
    axes[1].set(ylabel="$C_{L_\\alpha}$ (/rad)", title="Finite difference vs stability output")
    for label, axis in zip("AB", axes, strict=True):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", va="top")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A320 VSPAERO M=0.80 derivative diagnostic")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    root = _openvsp_root(args.openvsp_root)
    _bootstrap_openvsp(root)
    import openvsp as vsp  # type: ignore[import-not-found]

    vsp.SetVSPAEROPath(str(root))
    measurements = load_measurements(args.measurements)
    geometry = _nominal_geometry(measurements)
    geometry_prior = yaml.safe_load(args.geometry_prior.read_text())
    assert geometry_prior["geometry"]["horizontal_tail"]["area_m2"]["nominal"] == geometry["horizontal_tail"]["area_m2"]
    prior_rows = _read_rows(args.extended_table)
    prior_medium = next(row for row in prior_rows if np.isclose(row["mach"], MACH) and np.isclose(row["alpha_deg"], 8.5))

    if args.keep_workdir:
        workdir = args.keep_workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup: contextlib.AbstractContextManager[Any] = contextlib.nullcontext(workdir)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="aerox-a320-vspaero-mach-diagnostic-")
    rows: list[dict[str, Any]] = []
    with cleanup as directory:
        for mesh in MESHES:
            for alpha in ALPHA_CASES_DEG:
                if mesh == "medium" and np.isclose(alpha, 8.5):
                    rows.append(
                        {
                            "mesh": "medium",
                            "mach": prior_medium["mach"],
                            "alpha_deg": prior_medium["alpha_deg"],
                            "CL": prior_medium["CL"],
                            "CL_alpha": prior_medium["CL_alpha_per_rad"],
                            "CMm": prior_medium["CMm"],
                            "CMm_alpha": prior_medium["CMm_alpha_per_rad"],
                            "vortex_loop_equations": prior_medium["vortex_loop_equations"],
                            "geometry_nodes": prior_medium["geometry_nodes"],
                        }
                    )
                    continue
                row, _ = _run_case(vsp, Path(directory) / f"{mesh}_a{alpha:.2f}", mesh, MACH, geometry, measurements, args.jobs, args.verbose, alpha_deg=alpha)
                rows.append(row)
    rows.sort(key=lambda row: (MESHES.index(row["mesh"]), row["alpha_deg"]))
    document = build_document(rows)
    _write_csv(args.results, rows)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_render_report(document))
    _plot(rows, document, args.figure_prefix)
    print(document["status"])


if __name__ == "__main__":
    main()
