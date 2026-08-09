#!/usr/bin/env python3
"""Confirm the frozen A320 VSPAERO lift surface on separated validation flights."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.models import EffectiveLiftMap, TabulatedLiftSurface
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.identify_a320_vspaero_angle_lift import WING_AREA_M2, _flight_arrays
from tools.identify_a320_vspaero_lift_surface import (
    _model_alpha,
    _surface_extrapolated,
)
from tools.replay_a320_point_mass import (
    _load_point_mass_flights,
    _read_horizontal_parameters,
)

UPDATED = "2026-08-05"
MAX_SURFACE_TO_EMPIRICAL_MEAN_RMSE_RATIO = 1.20
MAX_MEAN_SURFACE_CL_RMSE = 0.05
MAX_FLIGHT_SURFACE_CL_RMSE = 0.07
MAX_REGISTRATION_BIAS_DIFFERENCE_CL = 0.02
MIN_MODEL_ALPHA_ENVELOPE_FRACTION = 0.995


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument("--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv"))
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--horizontal-parameters",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--surface-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"),
    )
    parser.add_argument(
        "--lift-surface",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.csv"),
    )
    parser.add_argument(
        "--empirical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-confirmation-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-confirmation.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-confirmation.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-lift-surface-confirmation"),
    )
    return parser.parse_args()


def _read_frozen_surface_parameters(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    document = yaml.safe_load(path.read_text())
    if not document["decision"]["fit_gate_passed"]:
        raise ValueError("surface fit gate did not pass")
    model = document["model"]
    parameters = np.array(
        [
            model["pitch_scale"],
            math.radians(model["combined_offset_deg"]),
            math.radians(model["mach_offset_deg_per_mach"]),
            math.radians(model["registration_angle_difference_deg"]),
        ],
        dtype=float,
    )
    return parameters, document


def _read_empirical_lift_map(path: Path) -> EffectiveLiftMap:
    document = yaml.safe_load(path.read_text())
    lift = document["lift"]
    return EffectiveLiftMap(
        feature_set=lift["feature_set"],
        coefficients=np.array(
            [lift["coefficients"][name] for name in lift["feature_order"]],
            dtype=float,
        ),
    )


def build_confirmation(
    flights: list[Any],
    surface: TabulatedLiftSurface,
    frozen_parameters: np.ndarray,
    empirical: EffectiveLiftMap,
    gamma_bias_rad: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        alpha = _model_alpha(flight, frozen_parameters)
        surface_prediction = _surface_extrapolated(surface, alpha, flight.mach, 1.0)
        empirical_alpha = flight.pitch_zero_rad + gamma_bias_rad - flight.gamma_rad
        empirical_prediction = empirical.raw_lift_coefficient(empirical_alpha, flight.mach)
        surface_residual = surface_prediction - flight.target_cl
        empirical_residual = empirical_prediction - flight.target_cl
        inside = (
            (alpha >= surface.minimum_alpha_rad)
            & (alpha <= surface.maximum_alpha_rad)
            & (flight.mach >= surface.minimum_mach)
            & (flight.mach <= surface.maximum_mach)
        )
        rows.append(
            {
                "flight": flight.key,
                "registration": flight.registration,
                "samples": len(alpha),
                "empirical_cl_rmse": float(np.sqrt(np.mean(empirical_residual**2))),
                "surface_cl_rmse": float(np.sqrt(np.mean(surface_residual**2))),
                "surface_to_empirical_rmse_ratio": float(
                    np.sqrt(np.mean(surface_residual**2))
                    / np.sqrt(np.mean(empirical_residual**2))
                ),
                "empirical_cl_bias": float(np.mean(empirical_residual)),
                "surface_cl_bias": float(np.mean(surface_residual)),
                "model_alpha_in_envelope_fraction": float(np.mean(inside)),
                "model_alpha_min_deg": float(np.rad2deg(np.min(alpha))),
                "model_alpha_max_deg": float(np.rad2deg(np.max(alpha))),
            }
        )
    empirical_mean = float(np.mean([row["empirical_cl_rmse"] for row in rows]))
    surface_mean = float(np.mean([row["surface_cl_rmse"] for row in rows]))
    maximum_surface = float(max(row["surface_cl_rmse"] for row in rows))
    envelope_fraction = float(
        np.average(
            [row["model_alpha_in_envelope_fraction"] for row in rows],
            weights=[row["samples"] for row in rows],
        )
    )
    frame = pd.DataFrame(rows)
    registration_bias = frame.groupby("registration")["surface_cl_bias"].mean()
    registration_bias_difference = float(registration_bias.max() - registration_bias.min())
    registrations = frame.groupby("registration")["flight"].count().to_dict()
    gates = {
        "surface_to_empirical_mean_rmse_ratio": {
            "value": surface_mean / empirical_mean,
            "maximum": MAX_SURFACE_TO_EMPIRICAL_MEAN_RMSE_RATIO,
            "passed": surface_mean / empirical_mean <= MAX_SURFACE_TO_EMPIRICAL_MEAN_RMSE_RATIO,
        },
        "mean_surface_cl_rmse": {
            "value": surface_mean,
            "maximum": MAX_MEAN_SURFACE_CL_RMSE,
            "passed": surface_mean <= MAX_MEAN_SURFACE_CL_RMSE,
        },
        "maximum_flight_surface_cl_rmse": {
            "value": maximum_surface,
            "maximum": MAX_FLIGHT_SURFACE_CL_RMSE,
            "passed": maximum_surface <= MAX_FLIGHT_SURFACE_CL_RMSE,
        },
        "registration_bias_difference_cl": {
            "value": registration_bias_difference,
            "maximum": MAX_REGISTRATION_BIAS_DIFFERENCE_CL,
            "passed": registration_bias_difference <= MAX_REGISTRATION_BIAS_DIFFERENCE_CL,
        },
        "model_alpha_envelope_fraction": {
            "value": envelope_fraction,
            "minimum": MIN_MODEL_ALPHA_ENVELOPE_FRACTION,
            "passed": envelope_fraction >= MIN_MODEL_ALPHA_ENVELOPE_FRACTION,
        },
        "balanced_registration_coverage": {
            "value": registrations,
            "required": {"TAIL-A": 3, "TAIL-B": 3},
            "passed": registrations == {"TAIL-A": 3, "TAIL-B": 3},
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "independent-confirmation-passed" if passed else "independent-confirmation-failed",
        "leakage_control": {
            "parameters_refitted": False,
            "structure_changed_after_opening_validation": False,
            "validation_flights_loaded": True,
            "flights": [flight.key for flight in flights],
            "samples": int(sum(len(flight.mach) for flight in flights)),
        },
        "frozen_model": {
            "parameters": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
            "surface": "docs/modeling/a320-vspaero-alpha-support.csv",
            "empirical_baseline": "docs/modeling/a320-pitch-bank-input-calibration.yml",
            "pitch_scale": float(frozen_parameters[0]),
            "combined_offset_deg": math.degrees(frozen_parameters[1]),
            "mach_offset_deg_per_mach": math.degrees(frozen_parameters[2]),
            "registration_angle_difference_deg": math.degrees(frozen_parameters[3]),
        },
        "comparison": {
            "empirical_mean_flight_cl_rmse": empirical_mean,
            "surface_mean_flight_cl_rmse": surface_mean,
            "surface_to_empirical_mean_rmse_ratio": surface_mean / empirical_mean,
            "metrics": "docs/modeling/a320-vspaero-lift-surface-confirmation-metrics.csv",
        },
        "gates": gates,
        "decision": {
            "confirmation_passed": passed,
            "release_surface_for_clean_replay_integration": passed,
            "release_physical_pitch_sensor_calibration": False,
            "release_tail_specific_physical_aerodynamics": False,
            "release_vspaero_drag": False,
            "reason": "Independent confirmation can release the bounded clean lift surface for controlled replay integration, while the joint angle offsets remain effective and confounded.",
        },
        "limitations": [
            "The six flights were previously used by earlier replay commissioning but were not loaded by the surface fit or residual-structure selection.",
            "Confirmation covers only clean, low-bank samples inside the established Mach and effective-angle envelope.",
            "The lift target is reconstructed rather than directly measured.",
            "Stall, high-lift configurations, sideslip, and physical sensor calibration remain outside the release.",
        ],
    }
    return document, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(document: dict[str, Any]) -> str:
    comparison = document["comparison"]
    gate_rows = "\n".join(
        f"| {name} | {gate['value'] if isinstance(gate['value'], (bool, dict)) else f'{gate['value']:.4f}'} | {'pass' if gate['passed'] else 'fail'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 VSPAERO lift-surface confirmation

## Outcome

The frozen lift-surface model **{'passes' if document['decision']['confirmation_passed'] else 'does not pass'}** the independent six-flight confirmation. No parameter was refitted after opening the validation flights.

Mean flight CL RMSE is {comparison['surface_mean_flight_cl_rmse']:.4f}, versus {comparison['empirical_mean_flight_cl_rmse']:.4f} for the frozen empirical replay map (ratio {comparison['surface_to_empirical_mean_rmse_ratio']:.3f}).

| Gate | Value | Result |
|---|---:|---:|
{gate_rows}

## Decision

- Release the bounded clean lift surface for controlled replay integration: **{'yes' if document['decision']['release_surface_for_clean_replay_integration'] else 'no'}**.
- Release a physical pitch sensor calibration: **no**.
- Interpret the registration nuisance as physical aerodynamics: **no**.
- Adopt VSPAERO drag: **no**.
"""


def _plot(rows: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["flight"].split("/", 1)[1] for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), constrained_layout=True)
    axes[0].bar(x-width/2, [row["empirical_cl_rmse"] for row in rows], width, color="#0072B2", hatch="//", label="empirical")
    axes[0].bar(x+width/2, [row["surface_cl_rmse"] for row in rows], width, color="#D55E00", hatch="..", label="VSPAERO surface")
    axes[0].set_xticks(x, labels, rotation=50, ha="right")
    axes[0].set(title="Frozen-model error", ylabel="CL RMSE")
    axes[0].legend(frameon=False)
    for registration, marker, color in (("TAIL-A", "o", "#0072B2"), ("TAIL-B", "s", "#D55E00")):
        indices = [i for i, row in enumerate(rows) if row["registration"] == registration]
        axes[1].scatter(indices, [rows[i]["surface_cl_bias"] for i in indices], marker=marker, color=color, label=registration)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=50, ha="right")
    axes[1].set(title="Surface residual bias", ylabel="Mean CL residual")
    axes[1].legend(frameon=False)
    improvement = 100*(1-np.asarray([row["surface_to_empirical_rmse_ratio"] for row in rows]))
    axes[2].bar(x, improvement, color=np.where(improvement >= 0, "#009E73", "#E69F00"))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(x, labels, rotation=50, ha="right")
    axes[2].set(title="Per-flight comparison", ylabel="RMSE improvement vs empirical (%)")
    for label, axis in zip("ABC", axes, strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 frozen VSPAERO lift-surface confirmation")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    frozen_parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    validation = panel.loc[panel["role"] == "validation"].copy()
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    loaded = _load_point_mass_flights(validation, raw_dir=args.raw_dir, processed_path=args.processed, vertical=vertical, horizontal=horizontal, tas_scale=tas_scale, wing_area_m2=WING_AREA_M2)
    flights = _flight_arrays(loaded, gamma_bias_rad=math.radians(vertical.effective_gamma_bias_deg))
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    empirical = _read_empirical_lift_map(args.empirical_parameters)
    document, rows = build_confirmation(flights, surface, frozen_parameters, empirical, math.radians(vertical.effective_gamma_bias_deg))
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_csv(args.metrics, rows)
    args.report.write_text(_render_report(document))
    _plot(rows, args.figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])


if __name__ == "__main__":
    main()
