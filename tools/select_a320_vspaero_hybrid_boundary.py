#!/usr/bin/env python3
"""Select a revised guarded-hybrid boundary using A320 calibration flights only."""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.models import GuardedHybridLiftModel, TabulatedLiftSurface
from tools.calibrate_a320_effective_thrust import _read_tas_scale, _read_vertical_calibration
from tools.replay_a320_point_mass import ReplayJob, _load_point_mass_flights, _read_fuel_model, _read_horizontal_parameters, _read_thrust_map
from tools.replay_a320_vspaero_extended_hybrid import ERROR_METRICS, LOWER_ALPHA_CORE_DEG, LOWER_MACH_CORE, UPPER_ALPHA_CORE_DEG
from tools.replay_a320_vspaero_lift_surface import _read_bank, _run_named_job, _surface_adapter
from tools.validate_a320_vspaero_lift_surface import _read_empirical_lift_map, _read_frozen_surface_parameters

UPDATED = "2026-08-06"
UPPER_MACH_CORE_CANDIDATES = (0.73, 0.76, 0.79)
MAXIMUM_SURFACE_WEIGHT_CANDIDATES = (0.55, 0.70, 0.85, 1.00)
MAX_CALIBRATION_ERROR_RATIO = 1.10
MAX_WINDOW_FAILURE_FRACTION = 0.02
MIN_MEAN_SURFACE_WEIGHT = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument("--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv"))
    parser.add_argument("--vertical-parameters", type=Path, default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"))
    parser.add_argument("--horizontal-parameters", type=Path, default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"))
    parser.add_argument("--thrust-parameters", type=Path, default=Path("docs/modeling/a320-effective-thrust-calibration.yml"))
    parser.add_argument("--fuel-parameters", type=Path, default=Path("docs/modeling/a320-effective-fuel-calibration.yml"))
    parser.add_argument("--pitch-bank-parameters", type=Path, default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"))
    parser.add_argument("--surface-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"))
    parser.add_argument("--lift-surface", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"))
    parser.add_argument("--empirical-metrics", type=Path, default=Path("docs/modeling/a320-vspaero-extended-hybrid-calibration-metrics.csv"))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--metrics", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-candidate-metrics.csv"))
    parser.add_argument("--summary", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-candidates.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.md"))
    parser.add_argument("--figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-vspaero-hybrid-boundary-selection"))
    return parser.parse_args()


def _candidate_name(maximum_weight: float, upper_mach_core: float) -> str:
    return f"w{round(100*maximum_weight):03d}_m{round(100*upper_mach_core):03d}"


def _mean_errors(frame: pd.DataFrame) -> dict[str, float]:
    return {name: float(frame[name].mean()) for name in ERROR_METRICS}


def build_document(summary: pd.DataFrame, *, flight_count: int) -> dict[str, Any]:
    eligible = summary.loc[summary["eligible"]].sort_values(
        ["maximum_error_ratio", "mean_surface_weight"],
        ascending=[True, False],
        kind="stable",
    )
    selected = None if eligible.empty else eligible.iloc[0]
    passed = selected is not None
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "revised-hybrid-boundary-selected" if passed else "revised-hybrid-boundary-held",
        "experiment": {
            "data_split": "original development and calibration flights only; all validation and extension-confirmation flights excluded",
            "flights": flight_count,
            "upper_mach_core_candidates": list(UPPER_MACH_CORE_CANDIDATES),
            "maximum_surface_weight_candidates": list(MAXIMUM_SURFACE_WEIGHT_CANDIDATES),
            "fixed_alpha_core_deg": [LOWER_ALPHA_CORE_DEG, UPPER_ALPHA_CORE_DEG],
            "fixed_lower_mach_core": LOWER_MACH_CORE,
            "parameters_refitted": False,
            "candidate_summary": "docs/modeling/a320-vspaero-hybrid-boundary-candidates.csv",
        },
        "selection_gates": {
            "maximum_each_trajectory_rmse_ratio": MAX_CALIBRATION_ERROR_RATIO,
            "maximum_window_failure_fraction": MAX_WINDOW_FAILURE_FRACTION,
            "required_full_successes": flight_count,
            "minimum_recorded_mean_surface_weight": MIN_MEAN_SURFACE_WEIGHT,
        },
        "selected": None
        if selected is None
        else {
            "candidate": str(selected["candidate"]),
            "upper_mach_core": float(selected["upper_mach_core"]),
            "maximum_surface_weight": float(selected["maximum_surface_weight"]),
            "mean_surface_weight": float(selected["mean_surface_weight"]),
            "maximum_error_ratio": float(selected["maximum_error_ratio"]),
            "speed_rmse_ratio": float(selected["speed_rmse_ratio"]),
            "gamma_rmse_ratio": float(selected["gamma_rmse_ratio"]),
            "altitude_rmse_ratio": float(selected["altitude_rmse_ratio"]),
            "horizontal_rmse_ratio": float(selected["horizontal_rmse_ratio"]),
        },
        "decision": {
            "open_sealed_extension_confirmation": passed,
            "make_model_default_now": False,
            "release_vspaero_drag": False,
            "discard_empirical_fallback": False,
        },
        "limitations": [
            "This is a post-failure calibration-only boundary selection; it does not reuse either prior validation panel.",
            "The maximum surface weight is a multi-fidelity trust parameter, not an aerodynamic coefficient.",
            "The sealed six-flight extension panel remains the only confirmation gate for the selected boundary.",
        ],
    }


def _render_report(document: dict[str, Any]) -> str:
    selected = document["selected"]
    if selected is None:
        outcome = "No candidate passed every frozen calibration-only boundary gate. The extension-confirmation panel remains sealed."
        detail = ""
    else:
        outcome = "A revised boundary passed every calibration-only gate and is frozen before opening the extension-confirmation panel."
        detail = f"""
| Selected property | Value |
|---|---:|
| Candidate | `{selected['candidate']}` |
| Upper Mach core | {selected['upper_mach_core']:.2f} |
| Maximum VSPAERO weight | {selected['maximum_surface_weight']:.2f} |
| Mean recorded VSPAERO weight | {100*selected['mean_surface_weight']:.1f}% |
| Worst trajectory RMSE ratio | {selected['maximum_error_ratio']:.3f} |
"""
    return f"""# A320 VSPAERO hybrid boundary selection

## Outcome

{outcome}
{detail}
## Method

Only the original ten development/calibration flights are used. The grid varies
the upper Mach full-weight boundary and a global maximum VSPAERO trust weight;
it does not refit lift, drag, thrust, attitude, or registration parameters. A
candidate must keep every 300 s trajectory RMSE ratio at or below 1.10, retain
at least 50% mean VSPAERO weight, integrate every full flight, and keep window
failures at or below 2%.

The candidate table records all results, including failures. The response-blind
six-flight extension-confirmation panel is not used in this selection.
"""


def _plot(summary: pd.DataFrame, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    weights = list(MAXIMUM_SURFACE_WEIGHT_CANDIDATES)
    cores = list(UPPER_MACH_CORE_CANDIDATES)
    error = np.empty((len(weights), len(cores)))
    usage = np.empty_like(error)
    for i, weight in enumerate(weights):
        for j, core in enumerate(cores):
            row = summary.loc[np.isclose(summary["maximum_surface_weight"], weight) & np.isclose(summary["upper_mach_core"], core)].iloc[0]
            error[i, j] = row["maximum_error_ratio"]
            usage[i, j] = row["mean_surface_weight"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    for axis, matrix, title, fmt in (
        (axes[0], error, "Worst trajectory RMSE ratio", ".3f"),
        (axes[1], 100*usage, "Mean VSPAERO weight (%)", ".1f"),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis_r" if axis is axes[0] else "viridis")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] > np.nanmedian(matrix) else "black", fontsize=9)
        axis.set_xticks(range(len(cores)), [f"{value:.2f}" for value in cores])
        axis.set_yticks(range(len(weights)), [f"{value:.2f}" for value in weights])
        axis.set(xlabel="Upper Mach core", ylabel="Maximum surface weight", title=title)
        fig.colorbar(image, ax=axis, shrink=0.8)
    for label, axis in zip("AB", axes, strict=True):
        axis.text(-0.18, 1.08, label, transform=axis.transAxes, fontweight="bold", va="top")
    fig.suptitle("A320 calibration-only hybrid boundary grid")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    panel = panel.loc[panel["role"] != "validation"].copy()
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    flights = _load_point_mass_flights(panel, raw_dir=args.raw_dir, processed_path=args.processed, vertical=vertical, horizontal=horizontal, tas_scale=_read_tas_scale(args.horizontal_parameters), wing_area_m2=float(aircraft.wing.area))
    empirical = _read_empirical_lift_map(args.pitch_bank_parameters)
    parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    bank = _read_bank(args.pitch_bank_parameters)
    thrust = _read_thrust_map(args.thrust_parameters)
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)

    empirical_rows = pd.read_csv(args.empirical_metrics)
    empirical_rows = empirical_rows.loc[empirical_rows["lift_model"] == "empirical"]
    empirical_errors = _mean_errors(empirical_rows)
    jobs: list[tuple[str, ReplayJob]] = []
    mean_weights: dict[str, float] = {}
    for maximum_weight in MAXIMUM_SURFACE_WEIGHT_CANDIDATES:
        for upper_core in UPPER_MACH_CORE_CANDIDATES:
            candidate = _candidate_name(maximum_weight, upper_core)
            candidate_weights = []
            for flight in flights:
                adapter = _surface_adapter(flight.base.registration, surface, parameters, math.radians(vertical.effective_gamma_bias_deg))
                hybrid = GuardedHybridLiftModel(
                    adapter,
                    empirical,
                    lower_alpha_core_rad=math.radians(LOWER_ALPHA_CORE_DEG),
                    upper_alpha_core_rad=math.radians(UPPER_ALPHA_CORE_DEG),
                    lower_mach_core=LOWER_MACH_CORE,
                    upper_mach_core=upper_core,
                    maximum_surface_weight=maximum_weight,
                )
                candidate_weights.append(np.array([hybrid.surface_weight_scalar(adapter.effective_angle_of_attack_scalar(pitch_angle_rad=pitch, flight_path_angle_rad=gamma, mach=mach), mach) for pitch, gamma, mach in zip(flight.effective_pitch_angle_rad, flight.base.flight_path_angle_rad, flight.base.mach, strict=True)]))
                jobs.append((candidate, ReplayJob(flight=flight, split="calibration", lift_model=hybrid, bank_calibration=bank, thrust_map=thrust, fuel_model=fuel, wing_area_m2=float(aircraft.wing.area), cd0=reference_drag.zero_lift_drag_coefficient, induced_drag_factor=reference_drag.induced_drag_factor)))
            mean_weights[candidate] = float(np.mean(np.concatenate(candidate_weights)))

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_named_job, candidate, job): (candidate, job.flight.key) for candidate, job in jobs}
        for future in as_completed(futures):
            candidate, key = futures[future]
            rows.append(future.result())
            print(f"completed {candidate} {key}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["lift_model", "registration", "recording"], kind="stable")
    summaries = []
    for candidate, current in metrics.groupby("lift_model", sort=True):
        errors = _mean_errors(current)
        ratios = {name: errors[name] / empirical_errors[name] for name in ERROR_METRICS}
        windows = int(current["window_count"].sum())
        failures = int(current["window_failures"].sum())
        failure_fraction = failures / max(windows + failures, 1)
        full_successes = int(current["full_success"].sum())
        weight_code, core_code = candidate.split("_")
        maximum_weight = int(weight_code[1:]) / 100
        upper_core = int(core_code[1:]) / 100
        eligible = all(value <= MAX_CALIBRATION_ERROR_RATIO for value in ratios.values()) and failure_fraction <= MAX_WINDOW_FAILURE_FRACTION and full_successes == len(panel) and mean_weights[candidate] >= MIN_MEAN_SURFACE_WEIGHT
        summaries.append({
            "candidate": candidate,
            "maximum_surface_weight": maximum_weight,
            "upper_mach_core": upper_core,
            "mean_surface_weight": mean_weights[candidate],
            "speed_rmse_ratio": ratios["window_speed_rmse_mps"],
            "gamma_rmse_ratio": ratios["window_gamma_rmse_deg"],
            "altitude_rmse_ratio": ratios["window_altitude_rmse_m"],
            "horizontal_rmse_ratio": ratios["window_horizontal_rmse_m"],
            "maximum_error_ratio": max(ratios.values()),
            "window_failure_fraction": failure_fraction,
            "full_successes": full_successes,
            "eligible": eligible,
        })
    summary = pd.DataFrame(summaries).sort_values(["maximum_surface_weight", "upper_mach_core"], kind="stable")
    document = build_document(summary, flight_count=len(panel))
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    metrics.to_csv(args.metrics, index=False)
    summary.to_csv(args.summary, index=False)
    args.report.write_text(_render_report(document))
    _plot(summary, args.figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])


if __name__ == "__main__":
    main()
