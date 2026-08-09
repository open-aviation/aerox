#!/usr/bin/env python3
"""Compare the QAR effective lift slope with the CAD/VSPAERO static prior."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

UPDATED = "2026-08-05"
MISMATCH_GATE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vspaero-results",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-prior.csv"),
    )
    parser.add_argument(
        "--effective-attitude",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-anchor.yml"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-anchor.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-anchor.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-lift-anchor"),
    )
    return parser.parse_args()


def build_audit(vspaero_results: Path, effective_attitude: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    with vspaero_results.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["mesh"] == "medium"]
    rows.sort(key=lambda row: float(row["mach"]))
    calibration = yaml.safe_load(effective_attitude.read_text())
    lift = calibration["lift"]
    if lift["feature_set"] != "linear_alpha_mach":
        raise ValueError("expected the frozen linear_alpha_mach effective lift map")
    coefficients = lift["coefficients"]
    base_per_degree = float(coefficients["alpha_centered"])
    interaction_per_degree = float(coefficients["alpha_mach_interaction"])
    comparison: list[dict[str, float]] = []
    for row in rows:
        mach = float(row["mach"])
        effective_per_rad = (
            base_per_degree + interaction_per_degree * (mach - 0.6)
        ) * 180.0 / math.pi
        vspaero_per_rad = float(row["CL_alpha"])
        comparison.append(
            {
                "mach": mach,
                "qar_effective_cl_alpha_per_rad": effective_per_rad,
                "vspaero_cl_alpha_per_rad": vspaero_per_rad,
                "qar_excess_fraction_of_vspaero": effective_per_rad / vspaero_per_rad - 1.0,
                "implied_physical_to_effective_angle_scale": effective_per_rad / vspaero_per_rad,
            }
        )
    mismatch = np.array([row["qar_excess_fraction_of_vspaero"] for row in comparison])
    passed = bool(np.all(np.abs(mismatch) <= MISMATCH_GATE))
    audit = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "physical-angle-lift-slope-gate-passed" if passed else "physical-angle-lift-slope-gate-failed",
        "inputs": {
            "qar_effective_lift": str(effective_attitude),
            "vspaero_geometry_prior": str(vspaero_results),
        },
        "comparison": {
            "mach_points": [row["mach"] for row in comparison],
            "maximum_allowed_absolute_mismatch_fraction": MISMATCH_GATE,
            "minimum_qar_excess_fraction": float(np.min(mismatch)),
            "maximum_qar_excess_fraction": float(np.max(mismatch)),
            "mean_qar_excess_fraction": float(np.mean(mismatch)),
            "passed": passed,
            "table": "docs/modeling/a320-vspaero-lift-anchor.csv",
        },
        "decision": {
            "treat_current_effective_alpha_as_physical": passed,
            "replace_empirical_lift_map_in_replay": False,
            "inject_vspaero_drag": False,
            "next_identification": "jointly refit effective pitch/AOA scale and lift intercept with the VSPAERO CL-alpha curve as an uncertain prior, using fit flights only before reopening validation",
            "reason": "the QAR map is fitted against force-balance effective angles; a slope mismatch cannot be assigned uniquely to airfoil fidelity, angle scale, pitch/AOA bias, CG, or omitted configuration effects",
        },
        "limitations": [
            "VSPAERO omits proprietary airfoils, twist, fuselage/nacelle interference, viscosity, and aeroelasticity.",
            "The QAR lift target is reconstructed and conditional on the drag/thrust calibration.",
            "The effective pitch/AOA convention is not an independently decoded physical angle calibration.",
            "The implied angle-scale ratio is a diagnostic, not a released sensor correction.",
        ],
    }
    return audit, comparison


def _write_comparison(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(audit: dict[str, Any], rows: list[dict[str, float]]) -> str:
    comparison = audit["comparison"]
    angle_scales = [row["implied_physical_to_effective_angle_scale"] for row in rows]
    values = "\n".join(
        f"| {row['mach']:.2f} | {row['qar_effective_cl_alpha_per_rad']:.3f} | "
        f"{row['vspaero_cl_alpha_per_rad']:.3f} | "
        f"{100 * row['qar_excess_fraction_of_vspaero']:.1f}% |"
        for row in rows
    )
    return f"""# A320 VSPAERO lift-slope anchor audit

## Outcome

The current QAR effective-angle lift slope is
{100 * comparison['minimum_qar_excess_fraction']:.1f}% to
{100 * comparison['maximum_qar_excess_fraction']:.1f}% higher than the
CAD-planform VSPAERO prior. Every Mach point misses the predeclared ±10% screen,
so the current effective pitch/AOA scale is **not** promoted to a physical angle
calibration.

| Mach | QAR effective CL-alpha [rad-1] | VSPAERO CL-alpha [rad-1] | QAR excess |
|---:|---:|---:|---:|
{values}

## Decision

The existing effective lift map stays in the recorded-flight replay because it
was fitted and validated as part of that closed calibration chain. It is not
silently replaced by the VSPAERO curve, and VSPAERO drag remains excluded.

For the physical higher-order model, the next identification must jointly fit
the pitch/AOA scale and lift intercept while treating the VSPAERO CL-alpha
curve as an uncertain prior. The implied angle rescaling ({min(angle_scales):.2f}
to {max(angle_scales):.2f} across the table)
is diagnostic only: the mismatch is also sensitive to proprietary airfoil and
twist, fuselage/nacelle interference, aeroelasticity, the planning CG, and the
force-balance thrust/drag calibration.

## Safe executable boundary

`TabulatedStaticStabilityPrior` loads the medium-mesh static coefficients in
the VSPAERO `.stab` convention (without yet claiming an Aerox body-axis map),
interpolates only inside Mach 0.20–0.78, and rejects alpha outside -1–6 deg or
beta outside ±3 deg. It has no drag field and no body-rate derivatives. This
makes the public-source prior usable in identification code without changing
the current trajectory plant prematurely.
"""


def _plot(rows: list[dict[str, float]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    mach = np.array([row["mach"] for row in rows])
    effective = np.array([row["qar_effective_cl_alpha_per_rad"] for row in rows])
    vspaero = np.array([row["vspaero_cl_alpha_per_rad"] for row in rows])
    excess = np.array([row["qar_excess_fraction_of_vspaero"] for row in rows]) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), constrained_layout=True)
    axes[0].plot(mach, effective, "o-", color="#D55E00", label="QAR effective-angle fit")
    axes[0].plot(mach, vspaero, "s-", color="#0072B2", label="CAD/VSPAERO prior")
    axes[0].set(xlabel="Mach", ylabel="CL-alpha [per rad]", title="Lift-slope comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].bar([f"{value:.2f}" for value in mach], excess, color="#D55E00", alpha=0.85)
    axes[1].axhline(10.0, color="#A33A2B", linestyle="--", label="±10% gate")
    axes[1].axhline(-10.0, color="#A33A2B", linestyle="--")
    axes[1].set(xlabel="Mach", ylabel="QAR excess over VSPAERO [%]", title="Physical-angle gate: fail")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("A320 effective lift slope versus geometry prior")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    audit, comparison = build_audit(args.vspaero_results, args.effective_attitude)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(audit, sort_keys=False))
    _write_comparison(args.comparison, comparison)
    args.report.write_text(_render_report(audit, comparison))
    _plot(comparison, args.figure_prefix)
    print(audit["status"])


if __name__ == "__main__":
    main()
