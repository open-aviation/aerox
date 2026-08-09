#!/usr/bin/env python3
"""Gate the frozen Mach-extended A320 VSPAERO/empirical hybrid replay."""

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
from tools.replay_a320_vspaero_lift_surface import _read_bank, _run_named_job, _surface_adapter
from tools.validate_a320_vspaero_lift_surface import _read_empirical_lift_map, _read_frozen_surface_parameters

UPDATED = "2026-08-06"
LOWER_ALPHA_CORE_DEG = -0.5
UPPER_ALPHA_CORE_DEG = 8.0
LOWER_MACH_CORE = 0.23
UPPER_MACH_CORE = 0.79
MAX_CALIBRATION_ERROR_RATIO = 1.25
MAX_CONFIRMATION_ERROR_RATIO = 1.10
MAX_WINDOW_FAILURE_FRACTION = 0.02
MIN_MEAN_SURFACE_WEIGHT = 0.50
ERROR_METRICS = (
    "window_speed_rmse_mps",
    "window_gamma_rmse_deg",
    "window_altitude_rmse_m",
    "window_horizontal_rmse_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--vertical-parameters", type=Path, default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"))
    parser.add_argument("--horizontal-parameters", type=Path, default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"))
    parser.add_argument("--thrust-parameters", type=Path, default=Path("docs/modeling/a320-effective-thrust-calibration.yml"))
    parser.add_argument("--fuel-parameters", type=Path, default=Path("docs/modeling/a320-effective-fuel-calibration.yml"))
    parser.add_argument("--pitch-bank-parameters", type=Path, default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"))
    parser.add_argument("--surface-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"))
    parser.add_argument("--boundary-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.yml"))
    parser.add_argument("--lift-surface", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 4))
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--figure-prefix", type=Path)
    return parser.parse_args()


def _defaults(args: argparse.Namespace) -> None:
    stem = f"a320-vspaero-extended-hybrid-{args.stage}"
    if args.panel is None:
        args.panel = Path(
            "docs/modeling/a320-replay-panel.csv"
            if args.stage == "calibration"
            else "docs/modeling/a320-vspaero-extension-confirmation-panel.csv"
        )
    if args.metrics is None:
        args.metrics = Path(f"docs/modeling/{stem}-metrics.csv")
    if args.parameters is None:
        args.parameters = Path(f"docs/modeling/{stem}.yml")
    if args.report is None:
        args.report = Path(f"docs/modeling/{stem}.md")
    if args.figure_prefix is None:
        args.figure_prefix = Path(f"docs/modeling/figures/{stem}")


def _aggregate(metrics: pd.DataFrame, model: str) -> dict[str, Any]:
    current = metrics.loc[metrics["lift_model"] == model]
    successful = current.loc[current["full_success"]]
    windows = int(current["window_count"].sum())
    failures = int(current["window_failures"].sum())
    row: dict[str, Any] = {
        "lift_model": model,
        "flights": len(current),
        "window_count": windows,
        "window_failures": failures,
        "window_failure_fraction": failures / max(windows + failures, 1),
        "full_success": int(current["full_success"].sum()),
    }
    for name in ERROR_METRICS:
        row[name] = float(current[name].mean())
    row.update(
        {
            "full_speed_rmse_mps": float(successful["full_speed_rmse_mps"].mean()),
            "full_altitude_rmse_m": float(successful["full_altitude_rmse_m"].mean()),
            "full_heading_rmse_deg": float(successful["full_heading_rmse_deg"].mean()),
            "full_horizontal_rmse_m": float(successful["full_horizontal_rmse_m"].mean()),
        }
    )
    return row


def build_document(
    metrics: pd.DataFrame,
    *,
    stage: str,
    mean_surface_weight: float,
    panel_path: Path,
    panel_count: int,
    upper_mach_core: float,
    maximum_surface_weight: float,
) -> dict[str, Any]:
    aggregate = [_aggregate(metrics, model) for model in ("empirical", "vspaero_extended_hybrid")]
    empirical, hybrid = aggregate
    ratios = {name: hybrid[name] / empirical[name] for name in ERROR_METRICS}
    maximum_ratio = MAX_CALIBRATION_ERROR_RATIO if stage == "calibration" else MAX_CONFIRMATION_ERROR_RATIO
    gates: dict[str, Any] = {
        name.replace("window_", "trajectory_") + "_ratio": {
            "value": value,
            "maximum": maximum_ratio,
            "passed": value <= maximum_ratio,
        }
        for name, value in ratios.items()
    }
    gates.update(
        {
            "hybrid_window_failure_fraction": {
                "value": hybrid["window_failure_fraction"],
                "maximum": MAX_WINDOW_FAILURE_FRACTION,
                "passed": hybrid["window_failure_fraction"] <= MAX_WINDOW_FAILURE_FRACTION,
            },
            "hybrid_full_successes": {
                "value": hybrid["full_success"],
                "required": panel_count,
                "passed": hybrid["full_success"] == panel_count,
            },
            "recorded_mean_surface_weight": {
                "value": mean_surface_weight,
                "minimum": MIN_MEAN_SURFACE_WEIGHT,
                "passed": mean_surface_weight >= MIN_MEAN_SURFACE_WEIGHT,
            },
        }
    )
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "stage": stage,
        "status": f"extended-hybrid-{stage}-passed" if passed else f"extended-hybrid-{stage}-held",
        "frozen_model": {
            "surface": "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv",
            "identification": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
            "parameters_refitted": False,
            "alpha_surface_support_deg": [-1.0, 8.5],
            "alpha_core_deg": [LOWER_ALPHA_CORE_DEG, UPPER_ALPHA_CORE_DEG],
            "mach_surface_support": [0.20, 0.82],
            "mach_core": [LOWER_MACH_CORE, upper_mach_core],
            "maximum_surface_weight": maximum_surface_weight,
            "fallback": "frozen empirical linear_alpha_mach lift map",
            "recorded_mean_surface_weight": mean_surface_weight,
        },
        "evaluation": {
            "panel": str(panel_path),
            "flights": panel_count,
            "sampled_input_rate_hz": 1,
            "integration_rate_hz": 10,
            "window_s": 300,
        },
        "aggregate_metrics": aggregate,
        "hybrid_to_empirical_ratios": ratios,
        "gates": gates,
        "decision": {
            "stage_passed": passed,
            "open_independent_confirmation": passed if stage == "calibration" else None,
            "make_extended_hybrid_default": passed if stage == "confirmation" else False,
            "release_vspaero_drag": False,
            "discard_empirical_fallback": False,
        },
        "limitations": [
            "The VSPAERO extension is an inviscid geometry prior, not validated transonic A320 aerodynamics.",
            "The empirical fallback remains active near all Mach and alpha boundaries.",
            "Pitch/yaw/roll are commanded attitude inputs; control-surface and rotational dynamics remain outside this replay.",
        ],
    }


def _render_report(document: dict[str, Any]) -> str:
    stage = document["stage"]
    passed = document["decision"]["stage_passed"]
    ratios = document["hybrid_to_empirical_ratios"]
    model = document["frozen_model"]
    limit = MAX_CALIBRATION_ERROR_RATIO if stage == "calibration" else MAX_CONFIRMATION_ERROR_RATIO
    return f"""# A320 Mach-extended guarded-hybrid {stage}

## Outcome

The frozen Mach-extended hybrid **{'passes' if passed else 'does not pass'}** the
{stage} trajectory gate. Mean recorded VSPAERO-surface weight is
{100*model['recorded_mean_surface_weight']:.1f}%.

| 300 s window metric | Hybrid / empirical | Limit | Result |
|---|---:|---:|---:|
| True airspeed RMSE | {ratios['window_speed_rmse_mps']:.3f} | <= {limit:.2f} | {'pass' if ratios['window_speed_rmse_mps'] <= limit else 'fail'} |
| Flight-path-angle RMSE | {ratios['window_gamma_rmse_deg']:.3f} | <= {limit:.2f} | {'pass' if ratios['window_gamma_rmse_deg'] <= limit else 'fail'} |
| Altitude RMSE | {ratios['window_altitude_rmse_m']:.3f} | <= {limit:.2f} | {'pass' if ratios['window_altitude_rmse_m'] <= limit else 'fail'} |
| Horizontal RMSE | {ratios['window_horizontal_rmse_m']:.3f} | <= {limit:.2f} | {'pass' if ratios['window_horizontal_rmse_m'] <= limit else 'fail'} |

## Frozen model

- Lift surface: Mach 0.20--0.82, angle -1.0--8.5 deg.
- Full-weight Mach core: 0.23--{model['mach_core'][1]:.2f}; smooth fade to empirical by Mach 0.82.
- Maximum VSPAERO trust weight: {model['maximum_surface_weight']:.2f}.
- Full-weight angle core: -0.5--8.0 deg; smooth fade at both angle boundaries.
- Previously fitted pitch scale and registration nuisance are unchanged.
- Drag remains the Aerox parabolic polar; VSPAERO drag is not used.

## Decision

- Stage gate passed: **{'yes' if passed else 'no'}**.
- Make the hybrid default: **{'yes' if document['decision']['make_extended_hybrid_default'] else 'no'}**.
- Remove empirical fallback: **no**.
"""


def _plot(document: dict[str, Any], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    ratios = document["hybrid_to_empirical_ratios"]
    labels = ["TAS", "gamma", "altitude", "horizontal"]
    values = [ratios[name] for name in ERROR_METRICS]
    limit = MAX_CALIBRATION_ERROR_RATIO if document["stage"] == "calibration" else MAX_CONFIRMATION_ERROR_RATIO
    aggregate = document["aggregate_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].bar(labels, values, color=["#0072B2" if value <= limit else "#D55E00" for value in values])
    axes[0].axhline(limit, color="black", linestyle="--", linewidth=1, label=f"gate {limit:.2f}")
    axes[0].axhline(1.0, color="0.45", linestyle=":", linewidth=1)
    axes[0].set(ylabel="Hybrid / empirical RMSE", title="300 s trajectory errors")
    axes[0].legend(frameon=False)
    names = ["Empirical", "Extended hybrid"]
    axes[1].bar(names, [row["full_success"] for row in aggregate], color=["#0072B2", "#009E73"])
    axes[1].set(ylabel="Full-flight successes", title=f"Surface weight: {100*document['frozen_model']['recorded_mean_surface_weight']:.1f}%")
    for label, axis in zip("AB", axes, strict=True):
        axis.text(-0.14, 1.06, label, transform=axis.transAxes, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"A320 Mach-extended hybrid {document['stage']}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _defaults(args)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    if args.stage == "calibration":
        panel = panel.loc[panel["role"] != "validation"].copy()
    elif set(panel["role"]) != {"confirmation"}:
        raise ValueError("confirmation panel must contain only frozen confirmation flights")

    aircraft = load_aircraft("A320")

    reference_drag = build_reference_drag_polar(aircraft)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    flights = _load_point_mass_flights(
        panel,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=_read_tas_scale(args.horizontal_parameters),
        wing_area_m2=float(aircraft.wing.area),
    )
    empirical = _read_empirical_lift_map(args.pitch_bank_parameters)
    parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    if not np.isclose(surface.maximum_mach, 0.82):
        raise ValueError("extended lift surface must end at Mach 0.82")
    bank = _read_bank(args.pitch_bank_parameters)
    thrust = _read_thrust_map(args.thrust_parameters)
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    if args.stage == "confirmation":
        boundary = yaml.safe_load(args.boundary_parameters.read_text())
        if not boundary["decision"]["open_sealed_extension_confirmation"]:
            raise ValueError("selected boundary does not authorize confirmation")
        upper_mach_core = float(boundary["selected"]["upper_mach_core"])
        maximum_surface_weight = float(
            boundary["selected"]["maximum_surface_weight"]
        )
    else:
        upper_mach_core = UPPER_MACH_CORE
        maximum_surface_weight = 1.0

    jobs: list[tuple[str, ReplayJob]] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        common = dict(
            flight=flight,
            split=args.stage,
            bank_calibration=bank,
            thrust_map=thrust,
            fuel_model=fuel,
            wing_area_m2=float(aircraft.wing.area),
            cd0=reference_drag.zero_lift_drag_coefficient,
            induced_drag_factor=reference_drag.induced_drag_factor,
        )
        jobs.append(("empirical", ReplayJob(lift_model=empirical, **common)))
        adapter = _surface_adapter(
            flight.base.registration,
            surface,
            parameters,
            math.radians(vertical.effective_gamma_bias_deg),
        )
        hybrid = GuardedHybridLiftModel(
            adapter,
            empirical,
            lower_alpha_core_rad=math.radians(LOWER_ALPHA_CORE_DEG),
            upper_alpha_core_rad=math.radians(UPPER_ALPHA_CORE_DEG),
            lower_mach_core=LOWER_MACH_CORE,
            upper_mach_core=upper_mach_core,
            maximum_surface_weight=maximum_surface_weight,
        )
        alpha = np.array(
            [
                adapter.effective_angle_of_attack_scalar(
                    pitch_angle_rad=pitch,
                    flight_path_angle_rad=gamma,
                    mach=mach,
                )
                for pitch, gamma, mach in zip(
                    flight.effective_pitch_angle_rad,
                    flight.base.flight_path_angle_rad,
                    flight.base.mach,
                    strict=True,
                )
            ]
        )
        weights.append(
            np.array(
                [hybrid.surface_weight_scalar(angle, mach) for angle, mach in zip(alpha, flight.base.mach, strict=True)]
            )
        )
        jobs.append(("vspaero_extended_hybrid", ReplayJob(lift_model=hybrid, **common)))

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_named_job, model, job): (model, job.flight.key)
            for model, job in jobs
        }
        for future in as_completed(futures):
            model, key = futures[future]
            rows.append(future.result())
            print(f"completed {model} {key}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["lift_model", "registration", "recording"], kind="stable")
    document = build_document(
        metrics,
        stage=args.stage,
        mean_surface_weight=float(np.mean(np.concatenate(weights))),
        panel_path=args.panel,
        panel_count=len(panel),
        upper_mach_core=upper_mach_core,
        maximum_surface_weight=maximum_surface_weight,
    )
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
