#!/usr/bin/env python3
"""Validate turn geometry with a separately inherited A320 state profile."""

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
    SpatialEnvironmentProfile,
    inherit_state_profile,
    plan_advisory_energy_profile,
    plan_feasible_profile,
)
from aerox.guidance import SpatialPathGuidance
from aerox.modeling import PointMassProfilePerformance
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.generate_a320_comparison_trajectories import _controller
from tools.plot_a320_fms_decoupled_result import render_decoupled_figure
from tools.validate_a320_fms_long_range_qar import (
    _history,
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
    _style_axis,
)
from tools.validate_a320_fms_turn_aware_qar import _build_intent

UPDATED = "2026-08-07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "adaptive-selection": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
        "adaptive-anchors": "docs/fms/a320-fms-turn-aware-intent-anchors.csv",
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
        "uniform-parameters": "docs/fms/a320-fms-long-range-qar-intent.yml",
        "uniform-errors": "docs/fms/a320-fms-long-range-qar-intent-errors.csv",
        "coupled-parameters": "docs/fms/a320-fms-turn-aware-qar-intent.yml",
        "coupled-errors": "docs/fms/a320-fms-turn-aware-qar-intent-errors.csv",
        "exact-parameters": "docs/fms/a320-fms-decoupled-qar-intent.yml",
        "exact-errors": "docs/fms/a320-fms-decoupled-qar-intent-errors.csv",
        "anchor-states": "docs/fms/a320-fms-advisory-anchor-states.csv",
        "errors": "docs/fms/a320-fms-advisory-qar-intent-errors.csv",
        "history": "docs/fms/a320-fms-advisory-qar-intent-history.csv",
        "parameters": "docs/fms/a320-fms-advisory-qar-intent.yml",
        "report": "docs/fms/a320-fms-advisory-qar-intent.md",
        "figure-prefix": "docs/fms/figures/a320-fms-advisory-qar-intent-comparison",
        "three-way-figure-prefix": "docs/fms/figures/a320-fms-advisory-four-way",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _comparison(
    uniform: dict[str, Any],
    coupled: dict[str, Any],
    exact: dict[str, Any],
    decoupled: dict[str, Any],
) -> dict[str, dict[str, float]]:
    keys = (
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "terminal_horizontal_error_m",
        "any_control_limited_fraction",
    )
    return {
        key: {
            "uniform_600_s": float(uniform[key]),
            "coupled_turn_aware": float(coupled[key]),
            "exact_decoupled": float(exact[key]),
            "advisory_decoupled": float(decoupled[key]),
            "advisory_change_from_exact": float(decoupled[key] - exact[key]),
        }
        for key in keys
    }


def _plot_three_way(
    *,
    uniform_errors: pd.DataFrame,
    coupled_errors: pd.DataFrame,
    exact_errors: pd.DataFrame,
    decoupled_errors: pd.DataFrame,
    uniform_metrics: dict[str, Any],
    coupled_metrics: dict[str, Any],
    exact_metrics: dict[str, Any],
    decoupled_metrics: dict[str, Any],
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    orange, blue, magenta, green = "#D55E00", "#0072B2", "#CC79A7", "#009E73"
    scenarios = (
        (uniform_errors, uniform_metrics, "600-s coupled", orange, ":"),
        (coupled_errors, coupled_metrics, "turn-aware coupled", blue, "--"),
        (exact_errors, exact_metrics, "exact-state decoupled", magenta, "-."),
        (decoupled_errors, decoupled_metrics, "advisory-state decoupled", green, "-"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.9), constrained_layout=True)
    panels = (
        ("horizontal_error_m", "Horizontal error (m)"),
        ("altitude_error_m", "Altitude error (m)"),
        ("true_airspeed_error_mps", "True airspeed error (m/s)"),
    )
    for axis, (column, ylabel) in zip(axes.ravel()[:3], panels, strict=True):
        for errors, _, label, color, linestyle in scenarios:
            axis.plot(
                errors["elapsed_time_s"] / 60.0,
                errors[column],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                label=label,
            )
        axis.set(xlabel="Elapsed time (min)", ylabel=ylabel)
        axis.axhline(0.0, color="#000000", linewidth=0.7)
        axis.legend(frameon=False, fontsize=6.7)

    keys = (
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "any_control_limited_fraction",
    )
    labels = ("Path", "Time-pos", "Altitude", "TAS", "Limiter")
    base = np.array([uniform_metrics[key] for key in keys], dtype=float)
    values = np.array(
        [[metrics[key] for key in keys] for _, metrics, *_ in scenarios], dtype=float
    )
    normalized = values / base
    positions = np.arange(len(keys))
    width = 0.19
    for offset, (row, (_, _, label, color, _)) in enumerate(
        zip(normalized, scenarios, strict=True)
    ):
        axes[1, 1].bar(
            positions + (offset - 1.5) * width,
            row,
            width,
            color=color,
            alpha=0.88,
            label=label,
        )
    axes[1, 1].set(
        ylabel="Metric normalized to 600-s result",
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, max(1.1, 1.08 * float(np.max(normalized)))),
    )
    axes[1, 1].legend(frameon=False, fontsize=6.7)
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 long-range FMS: exact versus advisory state intent",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    adaptive_selection = yaml.safe_load(args.adaptive_selection.read_text())
    if adaptive_selection["status"] != "selected-before-adaptive-rollout":
        raise ValueError("turn-aware geometry selection is not frozen")
    adaptive_indices = np.asarray(adaptive_selection["anchor_indices"], dtype=int)
    anchor_table = pd.read_csv(args.adaptive_anchors)
    mandatory_indices = anchor_table.loc[
        anchor_table["selection_class"] == "maximum-time-spacing",
        "source_sample_index",
    ].to_numpy(dtype=int)

    qar, registration, recording = _load_holdout(args)
    state_route, state_environment, loaded_mandatory = build_long_range_intent(qar)
    if not np.array_equal(mandatory_indices, loaded_mandatory):
        raise ValueError("frozen mandatory state anchors do not match the base intent")
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
    state_profile = plan_advisory_energy_profile(
        route=state_route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(
            np.median(qar.static_temperature_k[mandatory_indices])
        ),
        hard_waypoint_indices=(0, len(mandatory_indices) - 1),
        maximum_altitude_adjustment_m=300.0,
        spacing_m=500.0,
        environment_profile=state_environment,
        waypoint_mass_kg=qar.mass_kg[mandatory_indices],
    )

    geometry_route, _ = _build_intent(qar, adaptive_indices)
    state_mapping = np.interp(
        qar.reference.time_s[adaptive_indices],
        qar.reference.time_s[mandatory_indices],
        state_route.waypoint_along_path_m,
    )
    inherited_route = inherit_state_profile(
        geometry_route=geometry_route,
        state_profile=state_profile.trajectory,
        state_along_path_at_geometry_waypoints_m=state_mapping,
        spacing_m=500.0,
    )
    environment = SpatialEnvironmentProfile(
        along_path_m=inherited_route.waypoint_along_path_m,
        static_temperature_k=qar.static_temperature_k[adaptive_indices],
        wind_north_mps=qar.wind_north_mps[adaptive_indices],
        wind_east_mps=qar.wind_east_mps[adaptive_indices],
        wind_down_mps=np.zeros(len(adaptive_indices)),
    )
    inherited_mass = np.interp(
        qar.reference.time_s[adaptive_indices],
        qar.reference.time_s[mandatory_indices],
        qar.mass_kg[mandatory_indices],
    )
    profile = plan_feasible_profile(
        route=inherited_route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(
            np.median(qar.static_temperature_k[adaptive_indices])
        ),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=inherited_mass,
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
    metrics, errors, _ = _evaluate(qar, inherited_route, adaptive_indices, result)
    gates = _gates(metrics, profile.feasible)
    uniform_document = yaml.safe_load(args.uniform_parameters.read_text())
    coupled_document = yaml.safe_load(args.coupled_parameters.read_text())
    exact_document = yaml.safe_load(args.exact_parameters.read_text())
    uniform_metrics = uniform_document["metrics"]
    coupled_metrics = coupled_document["metrics"]
    exact_metrics = exact_document["metrics"]
    comparison = _comparison(uniform_metrics, coupled_metrics, exact_metrics, metrics)
    passed = all(gate["passed"] for gate in gates.values())
    inherited_state_rows = []
    mandatory_set = set(mandatory_indices.tolist())
    projected_state_waypoints = state_profile.trajectory.intent.waypoints
    original_state_waypoints = state_route.intent.waypoints
    for order, (source_index, mapped_s, waypoint) in enumerate(
        zip(
            adaptive_indices,
            state_mapping,
            inherited_route.intent.waypoints,
            strict=True,
        )
    ):
        state_order = int(np.searchsorted(mandatory_indices, source_index))
        is_observed = source_index in mandatory_set
        if is_observed:
            target_altitude = original_state_waypoints[state_order].altitude_m
            target_speed = original_state_waypoints[state_order].true_airspeed_mps
            planned_state = projected_state_waypoints[state_order]
        else:
            target_altitude = np.nan
            target_speed = np.nan
            planned_state = waypoint
        inherited_state_rows.append(
            {
                "anchor_order": order,
                "source_sample_index": source_index,
                "constraint_class": (
                    "observed-state" if is_observed else "geometry-only"
                ),
                "constraint_mode": (
                    "hard"
                    if is_observed and state_order in {0, len(mandatory_indices) - 1}
                    else "advisory"
                    if is_observed
                    else "inherited"
                ),
                "mapped_state_along_path_m": mapped_s,
                "target_altitude_m": target_altitude,
                "target_true_airspeed_mps": target_speed,
                "planned_altitude_m": planned_state.altitude_m,
                "planned_true_airspeed_mps": planned_state.true_airspeed_mps,
                "altitude_adjustment_m": (
                    planned_state.altitude_m - target_altitude if is_observed else 0.0
                ),
            }
        )
    state_adjustments = np.array(
        [
            planned.altitude_m - original.altitude_m
            for planned, original in zip(
                projected_state_waypoints, original_state_waypoints, strict=True
            )
        ]
    )
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "passed" if passed else "held",
        "source": {
            "recording": f"{registration}/{recording}",
            "post_holdout_diagnostic": True,
            "model_controller_limits_or_gates_changed": False,
            "intent_constraint_semantics_changed": True,
        },
        "intent_architecture": {
            "horizontal_geometry_anchor_count": len(adaptive_indices),
            "advisory_altitude_speed_target_count": len(mandatory_indices),
            "hard_state_constraint_count": 2,
            "geometry_only_anchor_count": len(adaptive_indices)
            - len(mandatory_indices),
            "geometry_only_state_source": "inherited from the 25-target advisory aircraft-aware energy-shaped profile",
            "geometry_only_qar_altitude_speed_exposed": False,
            "core_operation": "aerox.fms.inherit_state_profile",
        },
        "exact_state_profile_feasibility": {
            "feasible": exact_document["base_state_profile_feasibility"]["feasible"],
            "violations": exact_document["base_state_profile_feasibility"][
                "violations"
            ],
        },
        "advisory_state_projection": {
            "maximum_allowed_altitude_adjustment_m": 300.0,
            "maximum_absolute_altitude_adjustment_m": float(
                np.max(np.abs(state_adjustments))
            ),
            "adjusted_target_count": int(np.sum(np.abs(state_adjustments) > 1e-6)),
            "speed_targets_changed": False,
            "hard_waypoint_indices": [0, len(mandatory_indices) - 1],
        },
        "base_state_profile_feasibility": {
            "feasible": state_profile.feasible,
            "violations": list(state_profile.violations),
        },
        "remapped_profile_feasibility": {
            "feasible": profile.feasible,
            "violations": list(profile.violations),
        },
        "metrics": metrics,
        "comparison": comparison,
        "gates": gates,
        "decision": {
            "advisory_decoupled_turn_aware_gate_passed": passed,
            "geometry_state_constraint_decoupling_implemented": True,
            "advisory_state_projection_implemented": True,
            "new_independent_holdout_claimed": False,
        },
        "artifacts": {
            "inherited_anchor_states": str(args.anchor_states),
            "errors": str(args.errors),
            "history": str(args.history),
            "comparison_figure": f"{args.figure_prefix}.png",
            "three_way_figure": f"{args.three_way_figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/validate_a320_fms_decoupled_intent.py",
        },
    }
    held = [name for name, gate in gates.items() if not gate["passed"]]
    report = f"""# A320 FMS advisory-state intent diagnostic

Updated: {UPDATED}

Status: **{document["status"]}**

The FMS separates 43 horizontal geometry nodes from 25 sampled QAR state
targets. Start and destination remain hard. Intermediate QAR states are now
explicitly advisory because they are validation samples, not known procedure
constraints. The aircraft-energy projection changes
{int(np.sum(np.abs(state_adjustments) > 1e-6))} altitude target(s), by at most
{np.max(np.abs(state_adjustments)):.1f} m, and changes no speed target. The
plant, controller, thrust boundary, limits, integration rates, and gates remain
unchanged.

Generated path, time-position, altitude, and TAS RMSE are
{metrics["generated_to_qar_path_rmse_m"]:.1f} m,
{metrics["time_aligned_horizontal_rmse_m"]:.1f} m,
{metrics["altitude_rmse_m"]:.1f} m, and {metrics["speed_rmse_mps"]:.2f} m/s.
Control-limiter occupancy is {100 * metrics["any_control_limited_fraction"]:.2f}%.

Compared with the exact-state decoupled case, time-position RMSE changes by
{comparison["time_aligned_horizontal_rmse_m"]["advisory_change_from_exact"]:+.1f} m,
altitude RMSE by
{comparison["altitude_rmse_m"]["advisory_change_from_exact"]:+.1f} m, and
limiter occupancy by
{100 * comparison["any_control_limited_fraction"]["advisory_change_from_exact"]:+.2f}
percentage points.

Held checks: {", ".join(held) if held else "none"}.

This remains a post-holdout architectural diagnostic and makes no new
independent-validation claim.
"""
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inherited_state_rows).to_csv(args.anchor_states, index=False)
    errors.to_csv(args.errors, index=False)
    history = _history(result)
    history.to_csv(args.history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    render_decoupled_figure(
        qar=qar,
        indices=adaptive_indices,
        errors=errors,
        history=history,
        anchor_states=pd.DataFrame(inherited_state_rows),
        figure_prefix=args.figure_prefix,
    )
    _plot_three_way(
        uniform_errors=pd.read_csv(args.uniform_errors),
        coupled_errors=pd.read_csv(args.coupled_errors),
        exact_errors=pd.read_csv(args.exact_errors),
        decoupled_errors=errors,
        uniform_metrics=uniform_metrics,
        coupled_metrics=coupled_metrics,
        exact_metrics=exact_metrics,
        decoupled_metrics=metrics,
        prefix=args.three_way_figure_prefix,
    )
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
