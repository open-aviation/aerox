#!/usr/bin/env python3
"""Compare frozen empirical and VSPAERO lift models in the A320 3D replay."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from aerox.models import (
    CalibratedTabulatedLiftSurface,
    EffectiveLiftMap,
    TabulatedLiftSurface,
)
from tools.audit_a320_vspaero_angle_lift_residual import REGISTRATION_CONTRAST
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    ReplayJob,
    _load_point_mass_flights,
    _read_fuel_model,
    _read_horizontal_parameters,
    _read_thrust_map,
    _replay_metrics,
)
from tools.validate_a320_vspaero_lift_surface import (
    _read_empirical_lift_map,
    _read_frozen_surface_parameters,
)

UPDATED = "2026-08-05"
MAX_VALIDATION_WINDOW_RATIO = 1.10
MAX_WINDOW_FAILURE_FRACTION = 0.02
REQUIRED_VALIDATION_FULL_SUCCESSES = 6


@dataclass(frozen=True)
class FrozenBankCalibration:
    delay_s: int
    scale_deg_per_code: float
    bias_deg: float


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
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--metrics", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-replay-metrics.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-replay.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-replay.md"))
    parser.add_argument("--figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-vspaero-lift-surface-replay"))
    return parser.parse_args()


def _read_bank(path: Path) -> FrozenBankCalibration:
    document = yaml.safe_load(path.read_text())
    bank = document["bank"]
    return FrozenBankCalibration(
        delay_s=int(bank["trajectory_reference_delay_s"]),
        scale_deg_per_code=float(bank["scale_deg_per_code"]),
        bias_deg=float(bank["bias_deg"]),
    )


def _surface_adapter(
    registration: str,
    surface: TabulatedLiftSurface,
    parameters: np.ndarray,
    decoded_pitch_bias_rad: float,
) -> CalibratedTabulatedLiftSurface:
    return CalibratedTabulatedLiftSurface(
        surface=surface,
        pitch_scale=float(parameters[0]),
        decoded_pitch_bias_rad=decoded_pitch_bias_rad,
        combined_offset_rad=float(parameters[1]),
        mach_offset_rad_per_mach=float(parameters[2]),
        registration_angle_offset_rad=float(
            parameters[3] * REGISTRATION_CONTRAST[registration]
        ),
    )


def _run_named_job(model: str, job: ReplayJob) -> dict[str, Any]:
    return {"lift_model": model, **_replay_metrics(job)}


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("empirical", "vspaero_surface"):
        for split in ("calibration", "validation"):
            current = metrics.loc[
                (metrics["lift_model"] == model) & (metrics["split"] == split)
            ]
            successful = current.loc[current["full_success"]]
            windows = float(current["window_count"].sum())
            failures = float(current["window_failures"].sum())
            rows.append(
                {
                    "lift_model": model,
                    "split": split,
                    "flights": len(current),
                    "window_count": int(windows),
                    "window_failures": int(failures),
                    "window_failure_fraction": failures / max(windows + failures, 1.0),
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
                }
            )
    return rows


def build_document(metrics: pd.DataFrame, frozen: dict[str, Any]) -> dict[str, Any]:
    aggregate = _aggregate(metrics)
    indexed = {(row["lift_model"], row["split"]): row for row in aggregate}
    empirical = indexed[("empirical", "validation")]
    surface = indexed[("vspaero_surface", "validation")]
    ratios = {
        name: surface[name] / empirical[name]
        for name in (
            "window_speed_rmse_mps",
            "window_gamma_rmse_deg",
            "window_altitude_rmse_m",
            "window_horizontal_rmse_m",
        )
    }
    gates = {
        "validation_window_speed_rmse_ratio": {"value": ratios["window_speed_rmse_mps"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_speed_rmse_mps"] <= MAX_VALIDATION_WINDOW_RATIO},
        "validation_window_gamma_rmse_ratio": {"value": ratios["window_gamma_rmse_deg"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_gamma_rmse_deg"] <= MAX_VALIDATION_WINDOW_RATIO},
        "validation_window_altitude_rmse_ratio": {"value": ratios["window_altitude_rmse_m"], "maximum": MAX_VALIDATION_WINDOW_RATIO, "passed": ratios["window_altitude_rmse_m"] <= MAX_VALIDATION_WINDOW_RATIO},
        "surface_window_failure_fraction": {"value": surface["window_failure_fraction"], "maximum": MAX_WINDOW_FAILURE_FRACTION, "passed": surface["window_failure_fraction"] <= MAX_WINDOW_FAILURE_FRACTION},
        "surface_validation_full_successes": {"value": surface["full_success"], "required": REQUIRED_VALIDATION_FULL_SUCCESSES, "passed": surface["full_success"] == REQUIRED_VALIDATION_FULL_SUCCESSES},
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "controlled-replay-integration-passed" if passed else "controlled-replay-integration-held",
        "frozen_inputs": {
            "surface_identification": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
            "surface_confirmation": "docs/modeling/a320-vspaero-lift-surface-confirmation.yml",
            "attitude": "docs/modeling/a320-pitch-bank-input-calibration.yml",
            "thrust": "docs/modeling/a320-effective-thrust-calibration.yml",
            "fuel": "docs/modeling/a320-effective-fuel-calibration.yml",
            "drag": "unchanged Aerox A320 viscous parabolic polar",
            "parameters_refitted": False,
            "surface_pitch_scale": frozen["model"]["pitch_scale"],
            "surface_registration_angle_difference_deg": frozen["model"]["registration_angle_difference_deg"],
        },
        "simulation": {
            "sampled_input_rate_hz": 1,
            "integration_rate_hz": 10,
            "window_s": 300,
            "workers": None,
            "models": ["frozen empirical lift map", "frozen bounded VSPAERO lift surface"],
            "metrics": "docs/modeling/a320-vspaero-lift-surface-replay-metrics.csv",
        },
        "aggregate_metrics": aggregate,
        "validation_surface_to_empirical_ratios": ratios,
        "gates": gates,
        "decision": {
            "controlled_replay_integration_passed": passed,
            "make_surface_default_lift_model": passed,
            "remove_empirical_fallback": False,
            "release_full_envelope": False,
            "reason": "The bounded surface may become the clean-envelope default only if short-window trajectory errors remain competitive and all validation full-flight integrations stay inside the executable boundary.",
        },
        "limitations": [
            "The same measured pitch, bank, N1-derived thrust, weather, fuel, and drag inputs are used for both lift models.",
            "Full-flight open-loop heading and horizontal drift remain limitations of the bank-controlled reduced plant, independent of lift-map choice.",
            "The bounded surface raises rather than extrapolates outside its alpha/Mach support.",
            "High-lift, stall, sideslip, control-surface allocation, and physical rotational dynamics remain outside scope.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(document: dict[str, Any]) -> str:
    indexed = {(row["lift_model"], row["split"]): row for row in document["aggregate_metrics"]}
    empirical = indexed[("empirical", "validation")]
    surface = indexed[("vspaero_surface", "validation")]
    gates = "\n".join(
        f"| {name} | {gate['value']:.4f} | {'pass' if gate['passed'] else 'fail'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 VSPAERO lift-surface trajectory replay

## Outcome

The controlled replay integration **{'passes' if document['decision']['controlled_replay_integration_passed'] else 'does not pass'}**. No aerodynamic, thrust, drag, bank, fuel, or nuisance parameter was refitted.

On the six validation flights, the frozen empirical versus VSPAERO-surface 300-second mean errors are:

| Metric | Empirical | VSPAERO surface |
|---|---:|---:|
| TAS RMSE (m/s) | {empirical['window_speed_rmse_mps']:.3f} | {surface['window_speed_rmse_mps']:.3f} |
| Flight-path-angle RMSE (deg) | {empirical['window_gamma_rmse_deg']:.3f} | {surface['window_gamma_rmse_deg']:.3f} |
| Pressure-altitude RMSE (m) | {empirical['window_altitude_rmse_m']:.1f} | {surface['window_altitude_rmse_m']:.1f} |
| Horizontal RMSE (m) | {empirical['window_horizontal_rmse_m']:.1f} | {surface['window_horizontal_rmse_m']:.1f} |

The surface averages include only windows that remained inside the bounded
executable surface. Because {100*surface['window_failure_fraction']:.1f}% of
validation windows failed the boundary check, these averages are diagnostic
and must not be read as an unbiased model comparison.

| Gate | Value | Result |
|---|---:|---:|
{gates}

## Decision

- Make the bounded surface the clean-envelope default: **{'yes' if document['decision']['make_surface_default_lift_model'] else 'no'}**.
- Remove the empirical model as a comparison/fallback: **no**.
- Release stall, high-lift, or full-envelope aerodynamics: **no**.
"""


def _plot(document: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    rows = document["aggregate_metrics"]
    metrics = ["window_speed_rmse_mps", "window_gamma_rmse_deg", "window_altitude_rmse_m", "window_horizontal_rmse_m"]
    titles = ["TAS RMSE (m/s)", "Flight-path angle RMSE (deg)", "Altitude RMSE (m)", "Horizontal RMSE (m)"]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.5), constrained_layout=True)
    for label, metric, title, axis in zip("ABCD", metrics, titles, axes.ravel()[:4], strict=True):
        x = np.arange(2)
        width = 0.36
        for index, (model, color, hatch) in enumerate((("empirical", "#0072B2", "//"), ("vspaero_surface", "#D55E00", ".."))):
            values = [next(row[metric] for row in rows if row["lift_model"] == model and row["split"] == split) for split in ("calibration", "validation")]
            axis.bar(x + (index-0.5)*width, values, width, color=color, hatch=hatch, label=model.replace("_", " "))
        axis.set_xticks(x, ["calibration", "validation"])
        axis.set_ylabel(title)
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for index, (model, color, hatch) in enumerate(
        (("empirical", "#0072B2", "//"), ("vspaero_surface", "#D55E00", ".."))
    ):
        coverage = [
            1.0
            - next(
                row["window_failure_fraction"]
                for row in rows
                if row["lift_model"] == model and row["split"] == split
            )
            for split in ("calibration", "validation")
        ]
        axes[1, 1].bar(
            np.arange(2) + (index - 0.5) * 0.36,
            coverage,
            0.36,
            color=color,
            hatch=hatch,
        )
        successes = [
            next(
                row["full_success"]
                for row in rows
                if row["lift_model"] == model and row["split"] == split
            )
            for split in ("calibration", "validation")
        ]
        axes[1, 2].bar(
            np.arange(2) + (index - 0.5) * 0.36,
            successes,
            0.36,
            color=color,
            hatch=hatch,
        )
    axes[1, 1].set_xticks(np.arange(2), ["calibration", "validation"])
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_ylabel("Successful window fraction")
    axes[1, 2].set_xticks(np.arange(2), ["calibration", "validation"])
    axes[1, 2].set_ylabel("Full-flight successes")
    for label, axis in zip("EF", axes.ravel()[4:], strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("A320 frozen lift-model trajectory comparison (300 s windows)")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    flights = _load_point_mass_flights(panel, raw_dir=args.raw_dir, processed_path=args.processed, vertical=vertical, horizontal=horizontal, tas_scale=tas_scale, wing_area_m2=float(aircraft.wing.area))
    bank = _read_bank(args.pitch_bank_parameters)
    thrust = _read_thrust_map(args.thrust_parameters)
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    empirical = _read_empirical_lift_map(args.pitch_bank_parameters)
    frozen_parameters, frozen_document = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    role_by_key = {f"{row.registration}/{row.recording}": ("validation" if row.role == "validation" else "calibration") for row in panel.itertuples()}
    jobs: list[tuple[str, ReplayJob]] = []
    for flight in flights:
        common = {
            "flight": flight,
            "split": role_by_key[flight.key],
            "bank_calibration": bank,
            "thrust_map": thrust,
            "fuel_model": fuel,
            "wing_area_m2": float(aircraft.wing.area),
            "cd0": reference_drag.zero_lift_drag_coefficient,
            "induced_drag_factor": reference_drag.induced_drag_factor,
        }
        jobs.append(("empirical", ReplayJob(lift_model=empirical, **common)))
        adapter = _surface_adapter(flight.base.registration, surface, frozen_parameters, math.radians(vertical.effective_gamma_bias_deg))
        jobs.append(("vspaero_surface", ReplayJob(lift_model=adapter, **common)))
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for model, job in jobs:
            rows.append(_run_named_job(model, job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_named_job, model, job): (model, job.flight.key) for model, job in jobs}
            for future in as_completed(futures):
                model, key = futures[future]
                rows.append(future.result())
                print(f"completed {model} {key}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["lift_model", "split", "registration", "recording"], kind="stable")
    document = build_document(metrics, frozen_document)
    document["simulation"]["workers"] = args.workers
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    metrics.to_csv(args.metrics, index=False)
    args.report.write_text(_render_report(document))
    _plot(document, args.figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])


if __name__ == "__main__":
    main()
