#!/usr/bin/env python3
"""Run the frozen turn-aware intent through the unchanged A320 FMS plant."""

from __future__ import annotations

import argparse
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

from aerox.fms import (
    FlightIntent,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    Waypoint,
    plan_energy_shaped_profile,
)
from aerox.guidance import SpatialPathGuidance
from aerox.modeling import PointMassProfilePerformance
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.generate_a320_comparison_trajectories import _controller
from tools.validate_a320_fms_long_range_qar import _history, _load_holdout
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    INTEGRATION_RATE_HZ,
    _build_plant,
    _evaluate,
    _gates,
    _initial_trim_control,
    _plot,
    _style_axis,
)

UPDATED = "2026-08-06"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "adaptive-selection": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
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
        "errors": "docs/fms/a320-fms-turn-aware-qar-intent-errors.csv",
        "history": "docs/fms/a320-fms-turn-aware-qar-intent-history.csv",
        "parameters": "docs/fms/a320-fms-turn-aware-qar-intent.yml",
        "report": "docs/fms/a320-fms-turn-aware-qar-intent.md",
        "figure-prefix": "docs/fms/figures/a320-fms-turn-aware-qar-intent-comparison",
        "before-after-figure-prefix": "docs/fms/figures/a320-fms-turn-aware-before-after",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _build_intent(qar: Any, indices: np.ndarray) -> tuple[Any, Any]:
    waypoints = tuple(
        Waypoint(
            north_m=float(qar.north_position_m[index]),
            east_m=float(qar.east_position_m[index]),
            altitude_m=float(qar.reference.pressure_altitude_m[index]),
            true_airspeed_mps=float(qar.reference.true_airspeed_mps[index]),
            name=f"T{number}",
            acceptance_radius_m=1_500.0 if index == indices[-1] else 4_000.0,
        )
        for number, index in enumerate(indices)
    )
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=waypoints,
            maximum_climb_flight_path_angle_rad=math.radians(6.0),
            maximum_descent_flight_path_angle_rad=math.radians(4.0),
            maximum_acceleration_mps2=0.5,
            maximum_deceleration_mps2=0.5,
        )
    )
    environment = SpatialEnvironmentProfile(
        along_path_m=route.waypoint_along_path_m,
        static_temperature_k=qar.static_temperature_k[indices],
        wind_north_mps=qar.wind_north_mps[indices],
        wind_east_mps=qar.wind_east_mps[indices],
        wind_down_mps=np.zeros(len(indices)),
    )
    return route, environment


def _before_after_metrics(
    baseline: dict[str, Any], adaptive: dict[str, Any]
) -> dict[str, Any]:
    keys = (
        "qar_to_sparse_intent_path_rmse_m",
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "terminal_horizontal_error_m",
        "any_control_limited_fraction",
    )
    comparison: dict[str, Any] = {}
    for key in keys:
        before = float(baseline[key])
        after = float(adaptive[key])
        comparison[key] = {
            "before": before,
            "after": after,
            "change": after - before,
            "fractional_improvement": (before - after) / before,
        }
    return comparison


def _plot_before_after(
    *,
    baseline_errors: pd.DataFrame,
    adaptive_errors: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    adaptive_metrics: dict[str, Any],
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    blue, orange, black = "#0072B2", "#D55E00", "#000000"
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    base_time = baseline_errors["elapsed_time_s"] / 60.0
    adaptive_time = adaptive_errors["elapsed_time_s"] / 60.0
    panels = (
        ("horizontal_error_m", "Horizontal error (m)"),
        ("altitude_error_m", "Altitude error (m)"),
        ("true_airspeed_error_mps", "True airspeed error (m/s)"),
    )
    for axis, (column, ylabel) in zip(axes.ravel()[:3], panels, strict=True):
        axis.plot(
            base_time,
            baseline_errors[column],
            color=orange,
            linestyle="--",
            linewidth=1.2,
            label="600-s intent",
        )
        axis.plot(
            adaptive_time,
            adaptive_errors[column],
            color=blue,
            linewidth=1.2,
            label="turn-aware intent",
        )
        axis.set(xlabel="Elapsed time (min)", ylabel=ylabel)
        axis.axhline(0.0, color=black, linewidth=0.7)
        axis.legend(frameon=False, fontsize=7)
    names = ["Path", "Time-pos", "Altitude", "TAS"]
    keys = [
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
    ]
    before = np.array([baseline_metrics[key] for key in keys], dtype=float)
    after = np.array([adaptive_metrics[key] for key in keys], dtype=float)
    normalized = np.column_stack((np.ones(len(keys)), after / before))
    positions = np.arange(len(keys))
    width = 0.36
    axes[1, 1].bar(
        positions - width / 2,
        normalized[:, 0],
        width,
        color=orange,
        alpha=0.85,
        label="600-s intent",
    )
    axes[1, 1].bar(
        positions + width / 2,
        normalized[:, 1],
        width,
        color=blue,
        alpha=0.85,
        label="turn-aware intent",
    )
    axes[1, 1].set(
        ylabel="RMSE normalized to 600-s result",
        xticks=positions,
        xticklabels=names,
        ylim=(0.0, max(1.1, 1.1 * float(np.max(normalized)))),
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 long-range FMS: uniform versus turn-aware intent", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection = yaml.safe_load(args.adaptive_selection.read_text())
    if selection["status"] != "selected-before-adaptive-rollout":
        raise ValueError("adaptive anchor selection was not frozen before rollout")
    indices = np.asarray(selection["anchor_indices"], dtype=int)
    qar, registration, recording = _load_holdout(args)
    route, environment = _build_intent(qar, indices)
    plant = _build_plant(args)
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
        static_temperature_k=float(np.median(qar.static_temperature_k[indices])),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[indices],
    )
    guidance = SpatialPathGuidance(
        profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
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
    metrics, errors, plot_data = _evaluate(qar, route, indices, result)
    gates = _gates(metrics, profile.feasible)
    baseline_document = yaml.safe_load(args.baseline_parameters.read_text())
    baseline_metrics = baseline_document["metrics"]
    comparison = _before_after_metrics(baseline_metrics, metrics)
    passed = all(gate["passed"] for gate in gates.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "passed" if passed else "held",
        "source": {
            "recording": f"{registration}/{recording}",
            "post_holdout_diagnostic": True,
            "adaptive_selection_frozen_before_this_rollout": True,
            "model_controller_or_gate_changed": False,
        },
        "intent": {
            "selection_artifact": str(args.adaptive_selection),
            "anchor_count": len(indices),
            "maximum_time_spacing_s": 600,
            "geometry_chord_tolerance_m": 1_000.0,
            "heading_or_recorded_controls_used_by_guidance": False,
        },
        "profile_feasibility": {
            "feasible": profile.feasible,
            "violations": list(profile.violations),
        },
        "metrics": metrics,
        "comparison_to_uniform_600_s": comparison,
        "gates": gates,
        "decision": {
            "turn_aware_long_range_gate_passed": passed,
            "geometry_refinement_improved_generated_path_rmse": metrics[
                "generated_to_qar_path_rmse_m"
            ]
            < baseline_metrics["generated_to_qar_path_rmse_m"],
            "aircraft_model_or_controller_retuned": False,
            "new_independent_holdout_claimed": False,
        },
        "artifacts": {
            "errors": str(args.errors),
            "history": str(args.history),
            "comparison_figure": f"{args.figure_prefix}.png",
            "before_after_figure": f"{args.before_after_figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/validate_a320_fms_turn_aware_qar.py",
        },
    }
    held = [name for name, gate in gates.items() if not gate["passed"]]
    report = f"""# A320 FMS turn-aware long-range diagnostic

Updated: {UPDATED}

Status: **{document["status"]}**

The frozen geometry-only selector added turn anchors to the 600-second time
grid before this rollout. The A320 plant, demonstrated climb capacity,
controller, integration rates, limits, and gates are unchanged. This is a
post-holdout diagnostic, not a second independent validation claim.

The {len(indices)}-anchor generated path has
{metrics["generated_to_qar_path_rmse_m"]:.1f} m nearest-path RMSE, compared with
{baseline_metrics["generated_to_qar_path_rmse_m"]:.1f} m for uniform 600-second
intent. Time-aligned horizontal, altitude, and TAS RMSE are
{metrics["time_aligned_horizontal_rmse_m"]:.1f} m,
{metrics["altitude_rmse_m"]:.1f} m, and {metrics["speed_rmse_mps"]:.2f} m/s.
Arrival error is {metrics["arrival_time_error_s"]:+.0f} s and terminal position
error is {metrics["terminal_horizontal_error_m"]:.1f} m.

Geometry refinement cuts generated path RMSE by
{100 * comparison["generated_to_qar_path_rmse_m"]["fractional_improvement"]:.1f}%
and altitude RMSE by
{100 * comparison["altitude_rmse_m"]["fractional_improvement"]:.1f}%, so both
gates now pass. Time-aligned horizontal RMSE improves by only
{100 * comparison["time_aligned_horizontal_rmse_m"]["fractional_improvement"]:.1f}%,
showing that the remaining miss is primarily along-track scheduling rather than
lateral path representation. Exact extra altitude/speed anchors also increase
control-limiter occupancy from
{100 * baseline_metrics["any_control_limited_fraction"]:.2f}% to
{100 * metrics["any_control_limited_fraction"]:.2f}%.

Held checks: {", ".join(held) if held else "none"}.

Observed heading and recorded pitch, bank, N1, N2, and TLA are not supplied to
guidance. Added anchors supply only route geometry plus the same altitude,
speed, temperature, and wind intent fields already used by the uniform case.
"""
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    _history(result).to_csv(args.history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(qar, route, indices, result, plot_data, args.figure_prefix)
    _plot_before_after(
        baseline_errors=pd.read_csv(args.baseline_errors),
        adaptive_errors=errors,
        baseline_metrics=baseline_metrics,
        adaptive_metrics=metrics,
        prefix=args.before_after_figure_prefix,
    )
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
