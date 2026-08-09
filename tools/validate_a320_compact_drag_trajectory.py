#!/usr/bin/env python3
"""Compare the compact A320 drag candidate on the long-range QAR trajectory."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.fms import (
    PointMassProfilePerformance,
    SpatialPathGuidance,
    plan_energy_shaped_profile,
    simulate_fms_trajectory_zoh,
)
from tools.generate_a320_comparison_trajectories import _controller
from tools.historical_a320_drag import historical_a320_compact_drag_candidate
from tools.validate_a320_fms_long_range_qar import (
    COMMAND_STEP_S,
    _load_holdout,
    build_long_range_intent,
)
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    INTEGRATION_RATE_HZ,
    _build_plant,
    _evaluate,
    _gates,
    _initial_trim_control,
)

UPDATED = "2026-08-08"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "thrust-parameters": "docs/modeling/a320-effective-thrust-calibration.yml",
        "fuel-parameters": "docs/modeling/a320-effective-fuel-calibration.yml",
        "pitch-bank-parameters": "docs/modeling/a320-pitch-bank-input-calibration.yml",
        "surface-parameters": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
        "boundary-parameters": "docs/modeling/a320-vspaero-hybrid-boundary-selection.yml",
        "lift-surface": "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv",
        "n1-envelope": "docs/modeling/a320-autothrust-envelope.yml",
        "climb-capacity": "docs/modeling/a320-demonstrated-climb-capacity.yml",
        "baseline-parameters": "docs/fms/a320-fms-long-range-qar-intent.yml",
        "baseline-errors": "docs/fms/a320-fms-long-range-qar-intent-errors.csv",
        "errors": "docs/fms/a320-fms-compact-drag-trajectory-errors.csv",
        "history": "docs/fms/a320-fms-compact-drag-trajectory-history.csv",
        "parameters": "docs/fms/a320-fms-compact-drag-trajectory.yml",
        "report": "docs/fms/a320-fms-compact-drag-trajectory.md",
        "figure-prefix": "docs/fms/figures/a320-fms-compact-drag-trajectory-comparison",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _history(result: Any) -> pd.DataFrame:
    trajectory = result.trajectory
    return pd.DataFrame(
        {
            "elapsed_time_s": trajectory.time_s,
            "north_position_m": trajectory.north_position_m,
            "east_position_m": trajectory.east_position_m,
            "pressure_altitude_m": trajectory.pressure_altitude_m,
            "true_airspeed_mps": trajectory.true_airspeed_mps,
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_angle_deg": np.rad2deg(trajectory.bank_angle_rad),
            "effective_alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "net_thrust_kn": trajectory.net_thrust_n / 1_000.0,
            "drag_kn": trajectory.drag_n / 1_000.0,
            "fuel_flow_kg_s": trajectory.fuel_flow_kg_s,
            "mass_kg": trajectory.mass_kg,
            "pitch_limited": result.pitch_limited,
            "bank_limited": result.bank_limited,
            "thrust_limited": result.thrust_limited,
        }
    )


def _comparison(
    baseline: dict[str, Any], compact: dict[str, Any], qar_fuel_burn_kg: float
) -> dict[str, dict[str, float]]:
    keys = (
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "arrival_time_error_s",
        "any_control_limited_fraction",
        "fuel_burn_kg",
    )
    result = {
        key: {
            "parabolic_baseline": float(baseline[key]),
            "compact_physical": float(compact[key]),
            "compact_minus_baseline": float(compact[key] - baseline[key]),
        }
        for key in keys
    }
    result["fuel_burn_kg"].update(
        {
            "qar": qar_fuel_burn_kg,
            "parabolic_error": float(baseline["fuel_burn_kg"] - qar_fuel_burn_kg),
            "compact_error": float(compact["fuel_burn_kg"] - qar_fuel_burn_kg),
        }
    )
    return result


def _plot(
    *,
    qar: Any,
    route: Any,
    anchors: np.ndarray,
    result: Any,
    compact_errors: pd.DataFrame,
    baseline_errors: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    compact_metrics: dict[str, Any],
    prefix: Path,
) -> None:
    trajectory = result.trajectory
    time_h = qar.reference.time_s / 3_600.0
    generated_time_h = trajectory.time_s / 3_600.0
    anchor_time_h = qar.reference.time_s[anchors] / 3_600.0
    colors = {
        "qar": "#000000",
        "intent": "#E69F00",
        "baseline": "#56B4E9",
        "compact": "#009E73",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.3), constrained_layout=True)
    axes[0, 0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=colors["qar"],
        linestyle=":",
        linewidth=1.4,
        label="withheld QAR",
    )
    axes[0, 0].plot(
        [point.east_m / 1_000.0 for point in route.intent.waypoints],
        [point.north_m / 1_000.0 for point in route.intent.waypoints],
        color=colors["intent"],
        linestyle="--",
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="600-s intent",
    )
    axes[0, 0].plot(
        trajectory.east_position_m / 1_000.0,
        trajectory.north_position_m / 1_000.0,
        color=colors["compact"],
        linewidth=1.4,
        label="compact generated",
    )
    axes[0, 0].set_xlabel("East displacement (km)")
    axes[0, 0].set_ylabel("North displacement (km)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_title("(a) Horizontal trajectory")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        time_h,
        qar.reference.pressure_altitude_m / 0.3048,
        color=colors["qar"],
        linestyle=":",
        linewidth=1.2,
        label="QAR",
    )
    axes[0, 1].plot(
        anchor_time_h,
        qar.reference.pressure_altitude_m[anchors] / 0.3048,
        color=colors["intent"],
        linestyle="--",
        marker="o",
        markersize=2.5,
        label="vertical intent",
    )
    axes[0, 1].plot(
        generated_time_h,
        trajectory.pressure_altitude_m / 0.3048,
        color=colors["compact"],
        linewidth=1.2,
        label="generated",
    )
    axes[0, 1].set_xlabel("Elapsed time (h)")
    axes[0, 1].set_ylabel("Pressure altitude (ft)")
    axes[0, 1].set_title("(b) Vertical intent and trajectory")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(
        time_h,
        qar.reference.true_airspeed_mps,
        color=colors["qar"],
        linestyle=":",
        linewidth=1.2,
        label="QAR",
    )
    axes[0, 2].plot(
        anchor_time_h,
        qar.reference.true_airspeed_mps[anchors],
        color=colors["intent"],
        linestyle="--",
        marker="o",
        markersize=2.5,
        label="speed intent",
    )
    axes[0, 2].plot(
        generated_time_h,
        trajectory.true_airspeed_mps,
        color=colors["compact"],
        linewidth=1.2,
        label="generated",
    )
    axes[0, 2].set_xlabel("Elapsed time (h)")
    axes[0, 2].set_ylabel("True airspeed (m/s)")
    axes[0, 2].set_title("(c) Speed intent and trajectory")
    axes[0, 2].legend(frameon=False)

    axes[1, 0].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1, 0].plot(
        baseline_errors["elapsed_time_s"] / 3_600.0,
        baseline_errors["altitude_error_m"],
        color=colors["baseline"],
        linewidth=1.0,
        label="parabolic baseline",
    )
    axes[1, 0].plot(
        compact_errors["elapsed_time_s"] / 3_600.0,
        compact_errors["altitude_error_m"],
        color=colors["compact"],
        linewidth=1.0,
        label="compact physical",
    )
    axes[1, 0].set_xlabel("Elapsed time (h)")
    axes[1, 0].set_ylabel("Altitude error (m)")
    axes[1, 0].set_title("(d) Generated minus QAR altitude")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1, 1].plot(
        baseline_errors["elapsed_time_s"] / 3_600.0,
        baseline_errors["true_airspeed_error_mps"],
        color=colors["baseline"],
        linewidth=1.0,
        label="parabolic baseline",
    )
    axes[1, 1].plot(
        compact_errors["elapsed_time_s"] / 3_600.0,
        compact_errors["true_airspeed_error_mps"],
        color=colors["compact"],
        linewidth=1.0,
        label="compact physical",
    )
    axes[1, 1].set_xlabel("Elapsed time (h)")
    axes[1, 1].set_ylabel("True-airspeed error (m/s)")
    axes[1, 1].set_title("(e) Generated minus QAR speed")
    axes[1, 1].legend(frameon=False)

    metric_keys = (
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "fuel_burn_kg",
    )
    labels = ("Path", "Time-pos", "Altitude", "TAS", "Fuel")
    baseline = np.array([baseline_metrics[key] for key in metric_keys], dtype=float)
    compact = np.array([compact_metrics[key] for key in metric_keys], dtype=float)
    x = np.arange(len(metric_keys))
    axes[1, 2].bar(
        x - 0.2,
        np.ones(len(x)),
        0.4,
        color=colors["baseline"],
        edgecolor="black",
        linewidth=0.4,
        label="parabolic baseline",
    )
    axes[1, 2].bar(
        x + 0.2,
        compact / baseline,
        0.4,
        color=colors["compact"],
        edgecolor="black",
        linewidth=0.4,
        label="compact physical",
    )
    axes[1, 2].set_xticks(x, labels)
    axes[1, 2].set_ylabel("Metric / parabolic baseline")
    axes[1, 2].set_title("(f) Same-controller trajectory metrics")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=320)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(path: Path, document: dict[str, Any]) -> None:
    metrics = document["metrics"]
    comparison = document["comparison"]
    decision = document["decision"]
    text = f"""# A320 compact-drag long-range trajectory comparison

## Outcome

The frozen compact physical drag candidate completed the same 4.05-hour
QAR-referenced FMS trajectory with the same intent, guidance, controller,
environment, lift model, and thrust ceiling as the parabolic baseline.

Its altitude RMSE is `{metrics['altitude_rmse_m']:.1f} m`, true-airspeed RMSE is
`{metrics['speed_rmse_mps']:.2f} m/s`, and fuel burn is
`{metrics['fuel_burn_kg']:.1f} kg`. Relative to the original parabolic run,
those changes are `{comparison['altitude_rmse_m']['compact_minus_baseline']:+.1f} m`,
`{comparison['speed_rmse_mps']['compact_minus_baseline']:+.2f} m/s`, and
`{comparison['fuel_burn_kg']['compact_minus_baseline']:+.1f} kg`.

The diagnostic trajectory gate is **{'passed' if decision['passed'] else 'held'}**.
This does not override the separate thrust-surrogate hold and does not change
the default plant.

![Compact drag trajectory comparison](figures/a320-fms-compact-drag-trajectory-comparison.png)

Figure 1. Withheld QAR trajectory, altitude and speed intent anchors, generated
compact-model trajectory, same-controller errors, and normalized comparison
with the prior parabolic run. A
[vector PDF](figures/a320-fms-compact-drag-trajectory-comparison.pdf) is also
provided.

## What the guidance is doing

The orange points are the only state intent supplied to the FMS: position,
altitude, and speed every 600 seconds. The spatial planner connects them and
checks energy feasibility. Online path guidance converts the planned path into
heading, flight-path-angle, and speed/rate references. The existing inverse
flight-mechanics controller then commands pitch, bank, and realized net thrust.
The aircraft plant integrates those controls with wind and fuel burn. No
intermediate QAR state, actual pitch, actual bank, or actual thrust is replayed.

Because net thrust is the plant input, the newly refitted N1 surrogate is not
used by this controller rollout. It is an independent consistency gate for
whether recorded N1 and the new drag law can coexist. The demonstrated climb
ceiling still limits commanded thrust.

## Decision

{decision['reason']}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    qar, registration, recording = _load_holdout(args)
    route, environment, anchors = build_long_range_intent(qar)
    plant = replace(
        _build_plant(args),
        drag_model=historical_a320_compact_drag_candidate(),
    )
    thrust_ceiling = _load_climb_thrust_ceiling(args)
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=thrust_ceiling,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )
    profile = plan_energy_shaped_profile(
        route=route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(np.median(qar.static_temperature_k[anchors])),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[anchors],
    )
    guidance = SpatialPathGuidance(
        profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
        command_step_s=COMMAND_STEP_S,
    )
    controller = _controller()
    initial_gamma, initial_control, _ = _initial_trim_control(
        plant=plant,
        controller=controller,
        trajectory=profile.trajectory,
        environment=environment,
        initial_mass_kg=qar.initial_mass_kg,
    )
    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=controller,
        guidance=guidance,
        maximum_time_s=float(qar.reference.time_s[-1] + 300.0),
        static_temperature_k=float(environment.static_temperature_k[0]),
        initial_mass_kg=qar.initial_mass_kg,
        destination_capture_radius_m=1_500.0,
        integration_rate_hz=INTEGRATION_RATE_HZ,
        environment_profile=environment,
        maximum_thrust_model=thrust_ceiling,
        initial_flight_path_angle_rad=initial_gamma,
        initial_control=initial_control,
    )
    metrics, errors, _ = _evaluate(qar, route, anchors, result)
    gates = _gates(metrics, profile.feasible)
    baseline_document = yaml.safe_load(args.baseline_parameters.read_text())
    baseline_metrics = baseline_document["metrics"]
    qar_fuel_burn = float(qar.mass_kg[0] - qar.mass_kg[-1])
    comparison = _comparison(baseline_metrics, metrics, qar_fuel_burn)
    noninferior = {
        "altitude_rmse": metrics["altitude_rmse_m"]
        <= 1.05 * baseline_metrics["altitude_rmse_m"],
        "speed_rmse": metrics["speed_rmse_mps"]
        <= 1.05 * baseline_metrics["speed_rmse_mps"],
        "path_rmse": metrics["generated_to_qar_path_rmse_m"]
        <= 1.05 * baseline_metrics["generated_to_qar_path_rmse_m"],
        "fuel_burn_error": abs(metrics["fuel_burn_kg"] - qar_fuel_burn)
        <= abs(baseline_metrics["fuel_burn_kg"] - qar_fuel_burn),
    }
    passed = all(noninferior.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": (
            "compact-drag-trajectory-passes-noninferiority-gate"
            if passed
            else "compact-drag-trajectory-held-by-noninferiority-gate"
        ),
        "source": {
            "recording": f"{registration}/{recording}",
            "intent_interval_s": 600,
            "intermediate_qar_states_withheld": True,
        },
        "configuration": {
            "drag_model": "a320_compact_drag_candidate",
            "plant_input": "pitch, bank, realized net thrust",
            "controller_changed": False,
            "lift_model_changed": False,
            "thrust_ceiling_changed": False,
            "n1_surrogate_used_by_rollout": False,
        },
        "profile_feasible": profile.feasible,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "original_gates": gates,
        "decision": {
            "noninferiority_checks": noninferior,
            "passed": passed,
            "default_plant_changed": False,
            "reason": (
                "All same-controller path, altitude, speed, and fuel-burn "
                "noninferiority checks pass; the trajectory diagnostic supports "
                "the compact drag candidate, subject to the separate thrust hold."
                if passed
                else "At least one same-controller trajectory or fuel-burn metric "
                "is more than the frozen tolerance; the compact candidate remains held."
            ),
        },
        "artifacts": {
            "errors": str(args.errors),
            "history": str(args.history),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
            "report": str(args.report),
        },
    }
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    _history(result).to_csv(args.history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_report(args.report, document)
    _plot(
        qar=qar,
        route=route,
        anchors=anchors,
        result=result,
        compact_errors=errors,
        baseline_errors=pd.read_csv(args.baseline_errors),
        baseline_metrics=baseline_metrics,
        compact_metrics=metrics,
        prefix=args.figure_prefix,
    )
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(document["comparison"], sort_keys=False).strip())


if __name__ == "__main__":
    main()
