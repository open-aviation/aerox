#!/usr/bin/env python3
"""Gate a smooth VSPAERO/empirical hybrid in the A320 trajectory replay."""

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
from tools.replay_a320_point_mass import (
    ReplayJob,
    _load_point_mass_flights,
    _read_fuel_model,
    _read_horizontal_parameters,
    _read_thrust_map,
)
from tools.replay_a320_vspaero_lift_surface import (
    _read_bank,
    _run_named_job,
    _surface_adapter,
)
from tools.validate_a320_vspaero_lift_surface import (
    _read_empirical_lift_map,
    _read_frozen_surface_parameters,
)

UPDATED = "2026-08-05"
MAX_VALIDATION_WINDOW_RATIO = 1.10
MAX_WINDOW_FAILURE_FRACTION = 0.02
REQUIRED_VALIDATION_FULL_SUCCESSES = 6
MIN_RECORDED_MEAN_SURFACE_WEIGHT = 0.50


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
    parser.add_argument("--lift-surface", type=Path, default=Path("docs/modeling/a320-vspaero-alpha-support.csv"))
    parser.add_argument("--empirical-replay-metrics", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-replay-metrics.csv"))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--metrics", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-replay-metrics.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-replay.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-replay.md"))
    parser.add_argument("--figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-vspaero-hybrid-replay"))
    return parser.parse_args()


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("empirical", "vspaero_hybrid"):
        for split in ("calibration", "validation"):
            current = metrics.loc[(metrics["lift_model"] == model) & (metrics["split"] == split)]
            successful = current.loc[current["full_success"]]
            total_windows = current["window_count"].sum() + current["window_failures"].sum()
            rows.append({
                "lift_model": model,
                "split": split,
                "flights": len(current),
                "window_count": int(current["window_count"].sum()),
                "window_failures": int(current["window_failures"].sum()),
                "window_failure_fraction": float(current["window_failures"].sum()/max(total_windows, 1)),
                "full_success": int(current["full_success"].sum()),
                "window_speed_rmse_mps": float(current["window_speed_rmse_mps"].mean()),
                "window_gamma_rmse_deg": float(current["window_gamma_rmse_deg"].mean()),
                "window_altitude_rmse_m": float(current["window_altitude_rmse_m"].mean()),
                "window_heading_rmse_deg": float(current["window_heading_rmse_deg"].mean()),
                "window_horizontal_rmse_m": float(current["window_horizontal_rmse_m"].mean()),
                "full_speed_rmse_mps": float(successful["full_speed_rmse_mps"].mean()),
                "full_altitude_rmse_m": float(successful["full_altitude_rmse_m"].mean()),
                "full_heading_rmse_deg": float(successful["full_heading_rmse_deg"].mean()),
                "full_horizontal_rmse_m": float(successful["full_horizontal_rmse_m"].mean()),
            })
    return rows


def build_document(metrics: pd.DataFrame, mean_surface_weights: dict[str, float]) -> dict[str, Any]:
    aggregate = _aggregate(metrics)
    indexed = {(row["lift_model"], row["split"]): row for row in aggregate}
    empirical = indexed[("empirical", "validation")]
    hybrid = indexed[("vspaero_hybrid", "validation")]
    ratios = {name: hybrid[name]/empirical[name] for name in ("window_speed_rmse_mps", "window_gamma_rmse_deg", "window_altitude_rmse_m", "window_horizontal_rmse_m")}
    gates = {
        "validation_window_speed_rmse_ratio": {"value": ratios["window_speed_rmse_mps"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_speed_rmse_mps"] <= MAX_VALIDATION_WINDOW_RATIO},
        "validation_window_gamma_rmse_ratio": {"value": ratios["window_gamma_rmse_deg"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_gamma_rmse_deg"] <= MAX_VALIDATION_WINDOW_RATIO},
        "validation_window_altitude_rmse_ratio": {"value": ratios["window_altitude_rmse_m"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_altitude_rmse_m"] <= MAX_VALIDATION_WINDOW_RATIO},
        "hybrid_window_failure_fraction": {"value": hybrid["window_failure_fraction"], "maximum": MAX_WINDOW_FAILURE_FRACTION, "passed": hybrid["window_failure_fraction"] <= MAX_WINDOW_FAILURE_FRACTION},
        "hybrid_validation_full_successes": {"value": hybrid["full_success"], "required": REQUIRED_VALIDATION_FULL_SUCCESSES, "passed": hybrid["full_success"] == REQUIRED_VALIDATION_FULL_SUCCESSES},
        "validation_recorded_mean_surface_weight": {"value": mean_surface_weights["validation"], "minimum": MIN_RECORDED_MEAN_SURFACE_WEIGHT, "passed": mean_surface_weights["validation"] >= MIN_RECORDED_MEAN_SURFACE_WEIGHT},
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "guarded-hybrid-replay-passed" if passed else "guarded-hybrid-replay-held",
        "model": {
            "surface": "confirmed bounded VSPAERO CL(alpha, Mach)",
            "fallback": "frozen empirical linear_alpha_mach lift map",
            "weight_combination": "minimum of C1 smoothstep weights for each alpha and Mach boundary",
            "alpha_surface_support_deg": [-1.0, 8.5],
            "alpha_core_deg": [-0.5, 8.0],
            "mach_surface_support": [0.2, 0.78],
            "mach_core": [0.23, 0.75],
            "parameters_refitted": False,
            "recorded_mean_surface_weight": mean_surface_weights,
        },
        "simulation": {"sampled_input_rate_hz": 1, "integration_rate_hz": 10, "window_s": 300, "metrics": "docs/modeling/a320-vspaero-hybrid-replay-metrics.csv"},
        "aggregate_metrics": aggregate,
        "validation_hybrid_to_empirical_ratios": ratios,
        "gates": gates,
        "decision": {"guarded_hybrid_replay_passed": passed, "make_guarded_hybrid_clean_replay_default": passed, "discard_empirical_fallback": False, "claim_vspaero_full_envelope": False, "reason": "The hybrid is eligible only if it retains material geometry-surface usage, eliminates boundary failures, and keeps validation trajectory errors within 10 percent of the frozen empirical replay."},
        "limitations": ["The empirical fallback remains essential outside confirmed VSPAERO support.", "The blend is a multi-fidelity engineering boundary, not new aerodynamic evidence.", "Stall, high-lift, wave drag, sideslip, and physical control-surface dynamics remain outside scope."],
    }


def _render_report(document: dict[str, Any]) -> str:
    indexed = {(row["lift_model"], row["split"]): row for row in document["aggregate_metrics"]}
    empirical = indexed[("empirical", "validation")]
    hybrid = indexed[("vspaero_hybrid", "validation")]
    gate_rows = "\n".join(f"| {name} | {gate['value']:.4f} | {'pass' if gate['passed'] else 'fail'} |" for name, gate in document["gates"].items())
    return f"""# A320 guarded VSPAERO/empirical lift replay

## Outcome

The no-refit guarded hybrid **{'passes' if document['decision']['guarded_hybrid_replay_passed'] else 'does not pass'}** its controlled trajectory gate. Its mean recorded VSPAERO-surface weight on validation is {100*document['model']['recorded_mean_surface_weight']['validation']:.1f}%.

| Validation 300 s metric | Empirical | Guarded hybrid |
|---|---:|---:|
| TAS RMSE (m/s) | {empirical['window_speed_rmse_mps']:.3f} | {hybrid['window_speed_rmse_mps']:.3f} |
| Flight-path-angle RMSE (deg) | {empirical['window_gamma_rmse_deg']:.3f} | {hybrid['window_gamma_rmse_deg']:.3f} |
| Pressure-altitude RMSE (m) | {empirical['window_altitude_rmse_m']:.1f} | {hybrid['window_altitude_rmse_m']:.1f} |
| Horizontal RMSE (m) | {empirical['window_horizontal_rmse_m']:.1f} | {hybrid['window_horizontal_rmse_m']:.1f} |

| Gate | Value | Result |
|---|---:|---:|
{gate_rows}

## Decision

- Make the guarded hybrid the clean-replay default: **{'yes' if document['decision']['make_guarded_hybrid_clean_replay_default'] else 'no'}**.
- Remove the empirical fallback: **no**.
- Claim VSPAERO coverage outside the sampled surface: **no**.
"""


def _plot(document: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    rows = document["aggregate_metrics"]
    metrics = ["window_speed_rmse_mps", "window_gamma_rmse_deg", "window_altitude_rmse_m", "window_horizontal_rmse_m"]
    titles = ["TAS RMSE (m/s)", "Flight-path angle RMSE (deg)", "Altitude RMSE (m)", "Horizontal RMSE (m)"]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.5), constrained_layout=True)
    for label, metric, title, axis in zip("ABCD", metrics, titles, axes.ravel()[:4], strict=True):
        for index, (model, color, hatch) in enumerate((("empirical", "#0072B2", "//"), ("vspaero_hybrid", "#009E73", ".."))):
            values = [next(row[metric] for row in rows if row["lift_model"] == model and row["split"] == split) for split in ("calibration", "validation")]
            axis.bar(np.arange(2)+(index-0.5)*0.36, values, 0.36, color=color, hatch=hatch, label=model.replace("_", " "))
        axis.set_xticks(np.arange(2), ["calibration", "validation"])
        axis.set_ylabel(title)
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    weights = document["model"]["recorded_mean_surface_weight"]
    axes[1, 1].bar([0, 1], [weights["calibration"], weights["validation"]], color="#009E73")
    axes[1, 1].axhline(MIN_RECORDED_MEAN_SURFACE_WEIGHT, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks([0, 1], ["calibration", "validation"])
    axes[1, 1].set_ylabel("Mean VSPAERO surface weight")
    for index, model in enumerate(("empirical", "vspaero_hybrid")):
        values = [next(row["full_success"] for row in rows if row["lift_model"] == model and row["split"] == split) for split in ("calibration", "validation")]
        axes[1, 2].bar(np.arange(2)+(index-0.5)*0.36, values, 0.36, color=("#0072B2", "#009E73")[index], hatch=("//", "..")[index])
    axes[1, 2].set_xticks([0, 1], ["calibration", "validation"])
    axes[1, 2].set_ylabel("Full-flight successes")
    for label, axis in zip("EF", axes.ravel()[4:], strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("A320 guarded VSPAERO/empirical lift trajectory replay")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
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
    role = {f"{row.registration}/{row.recording}": ("validation" if row.role == "validation" else "calibration") for row in panel.itertuples()}
    jobs: list[tuple[str, ReplayJob]] = []
    weights_by_split: dict[str, list[np.ndarray]] = {"calibration": [], "validation": []}
    for flight in flights:
        adapter = _surface_adapter(flight.base.registration, surface, parameters, math.radians(vertical.effective_gamma_bias_deg))
        hybrid = GuardedHybridLiftModel(adapter, empirical)
        alpha = adapter.pitch_scale*(flight.effective_pitch_angle_rad-adapter.decoded_pitch_bias_rad)+adapter.combined_offset_rad+adapter.mach_offset_rad_per_mach*(flight.base.mach-adapter.mach_reference)+adapter.registration_angle_offset_rad-flight.base.flight_path_angle_rad
        weights_by_split[role[flight.key]].append(np.array([hybrid.surface_weight_scalar(a, m) for a, m in zip(alpha, flight.base.mach, strict=True)]))
        jobs.append(("vspaero_hybrid", ReplayJob(flight=flight, split=role[flight.key], lift_model=hybrid, bank_calibration=bank, thrust_map=thrust, fuel_model=fuel, wing_area_m2=float(aircraft.wing.area), cd0=reference_drag.zero_lift_drag_coefficient, induced_drag_factor=reference_drag.induced_drag_factor)))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_named_job, model, job): job.flight.key for model, job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"completed vspaero_hybrid {futures[future]}", flush=True)
    empirical_rows = pd.read_csv(args.empirical_replay_metrics)
    empirical_rows = empirical_rows.loc[empirical_rows["lift_model"] == "empirical"]
    metrics = pd.concat([empirical_rows, pd.DataFrame(rows)], ignore_index=True).sort_values(["lift_model", "split", "registration", "recording"], kind="stable")
    mean_weights = {split: float(np.mean(np.concatenate(values))) for split, values in weights_by_split.items()}
    document = build_document(metrics, mean_weights)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    metrics.to_csv(args.metrics, index=False)
    args.report.write_text(_render_report(document))
    _plot(document, args.figure_prefix)
    return document


def main() -> None:
    print(run(parse_args())["status"])


if __name__ == "__main__":
    main()
