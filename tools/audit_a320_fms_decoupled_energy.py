#!/usr/bin/env python3
"""Audit local and integral energy feasibility of the decoupled A320 plan."""

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
    plan_advisory_energy_profile,
)
from aerox.modeling import PointMassProfilePerformance
from aerox.models.atmosphere import G0
from tools.validate_a320_fms_long_range_qar import _load_holdout
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import _build_plant, _style_axis
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
        "nodes": "docs/fms/a320-fms-decoupled-energy-nodes.csv",
        "segments": "docs/fms/a320-fms-decoupled-energy-segments.csv",
        "parameters": "docs/fms/a320-fms-decoupled-energy-audit.yml",
        "report": "docs/fms/a320-fms-decoupled-energy-audit.md",
        "figure-prefix": "docs/fms/figures/a320-fms-decoupled-energy-audit",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _build_plan(args: argparse.Namespace) -> tuple[Any, ...]:
    adaptive_selection = yaml.safe_load(args.adaptive_selection.read_text())
    adaptive_indices = np.asarray(adaptive_selection["anchor_indices"], dtype=int)
    anchors = pd.read_csv(args.adaptive_anchors)
    mandatory_indices = anchors.loc[
        anchors["selection_class"] == "maximum-time-spacing",
        "source_sample_index",
    ].to_numpy(dtype=int)
    qar, _, _ = _load_holdout(args)
    route, environment = _build_intent(qar, adaptive_indices)
    plant = _build_plant(args)
    ceiling = _load_climb_thrust_ceiling(args)
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=ceiling,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )
    profile = plan_advisory_energy_profile(
        route=route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(
            np.median(qar.static_temperature_k[adaptive_indices])
        ),
        maximum_altitude_adjustment_m=300.0,
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[adaptive_indices],
    )
    state_mapping = route.waypoint_along_path_m.copy()
    inherited_mass = qar.mass_kg[adaptive_indices]
    return (
        qar,
        adaptive_indices,
        mandatory_indices,
        route,
        route,
        state_mapping,
        environment,
        inherited_mass,
        performance,
        profile,
        profile,
    )


def _diagnostics(
    plan: tuple[Any, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    (
        qar,
        adaptive_indices,
        mandatory_indices,
        state_route,
        route,
        state_mapping,
        environment,
        inherited_mass,
        performance,
        state_profile,
        profile,
    ) = plan
    along = profile.along_path_m
    altitude = profile.nominal_altitude_m
    speed = profile.nominal_true_airspeed_mps
    mass = np.interp(along, route.waypoint_along_path_m, inherited_mass)
    temperature = np.array(
        [environment.sample(float(value)).static_temperature_k for value in along]
    )
    local = [
        performance.limits(
            pressure_altitude_m=float(h),
            true_airspeed_mps=float(v),
            mass_kg=float(m),
            static_temperature_k=float(t),
        )
        for h, v, m, t in zip(altitude, speed, mass, temperature, strict=True)
    ]
    maximum_excess = np.array(
        [value.maximum_specific_excess_force_mps2 for value in local]
    )
    minimum_excess = np.array(
        [value.minimum_specific_excess_force_mps2 for value in local]
    )
    gamma = np.arctan(np.gradient(altitude, along))
    acceleration = speed * np.gradient(speed, along)
    demand = acceleration + G0 * np.sin(gamma)
    upper_margin = maximum_excess - demand
    lower_margin = demand - minimum_excess
    source_s = np.interp(along, route.waypoint_along_path_m, state_mapping)
    state_leg = np.clip(
        np.searchsorted(state_route.waypoint_along_path_m, source_s, side="right") - 1,
        0,
        len(mandatory_indices) - 2,
    )
    nodes = pd.DataFrame(
        {
            "along_path_m": along,
            "source_state_along_path_m": source_s,
            "state_leg": state_leg,
            "altitude_m": altitude,
            "true_airspeed_mps": speed,
            "flight_path_angle_deg": np.rad2deg(gamma),
            "acceleration_mps2": acceleration,
            "energy_demand_mps2": demand,
            "maximum_specific_excess_force_mps2": maximum_excess,
            "minimum_specific_excess_force_mps2": minimum_excess,
            "upper_energy_margin_mps2": upper_margin,
            "lower_energy_margin_mps2": lower_margin,
        }
    )
    mandatory_order = np.searchsorted(adaptive_indices, mandatory_indices)
    mandatory_target_s = route.waypoint_along_path_m[mandatory_order]
    segment_rows: list[dict[str, Any]] = []
    for leg in range(len(mandatory_indices) - 1):
        start_s = mandatory_target_s[leg]
        end_s = mandatory_target_s[leg + 1]
        selection = (along >= start_s) & (along <= end_s)
        local_s = along[selection]
        required_energy = 0.5 * (
            qar.reference.true_airspeed_mps[mandatory_indices[leg + 1]] ** 2
            - qar.reference.true_airspeed_mps[mandatory_indices[leg]] ** 2
        ) + G0 * (
            qar.reference.pressure_altitude_m[mandatory_indices[leg + 1]]
            - qar.reference.pressure_altitude_m[mandatory_indices[leg]]
        )
        if required_energy >= 0.0:
            available_energy = float(np.trapezoid(maximum_excess[selection], local_s))
            integral_margin = available_energy - required_energy
        else:
            available_energy = float(np.trapezoid(minimum_excess[selection], local_s))
            integral_margin = required_energy - available_energy
        segment_rows.append(
            {
                "state_leg": leg,
                "start_anchor": f"L{leg}",
                "end_anchor": f"L{leg + 1}",
                "distance_m": end_s - start_s,
                "required_specific_energy_m2ps2": required_energy,
                "available_specific_energy_m2ps2": available_energy,
                "integral_energy_margin_m2ps2": integral_margin,
                "minimum_upper_margin_mps2": float(np.min(upper_margin[selection])),
                "minimum_lower_margin_mps2": float(np.min(lower_margin[selection])),
                "upper_violation_node_count": int(
                    np.sum(upper_margin[selection] < 0.0)
                ),
                "lower_violation_node_count": int(
                    np.sum(lower_margin[selection] < 0.0)
                ),
            }
        )
    segments = pd.DataFrame(segment_rows)
    worst = int(np.argmin(upper_margin))
    positive_energy = segments["required_specific_energy_m2ps2"] >= 0.0
    metrics = {
        "base_state_profile_feasible": state_profile.feasible,
        "remapped_profile_feasible": profile.feasible,
        "minimum_pointwise_upper_margin_mps2": float(np.min(upper_margin)),
        "minimum_pointwise_lower_margin_mps2": float(np.min(lower_margin)),
        "upper_violation_node_count": int(np.sum(upper_margin < 0.0)),
        "lower_violation_node_count": int(np.sum(lower_margin < 0.0)),
        "worst_upper_margin_along_path_m": float(along[worst]),
        "worst_upper_margin_state_leg": int(state_leg[worst]),
        "minimum_positive_leg_integral_margin_m2ps2": float(
            segments.loc[positive_energy, "integral_energy_margin_m2ps2"].min()
        ),
        "all_positive_energy_legs_have_integral_capacity": bool(
            (segments.loc[positive_energy, "integral_energy_margin_m2ps2"] >= 0.0).all()
        ),
        "diagnosis": (
            "pointwise scheduling/continuity deficit"
            if (
                segments.loc[positive_energy, "integral_energy_margin_m2ps2"] >= 0.0
            ).all()
            else "at least one state leg lacks integral energy capacity"
        ),
    }
    return nodes, segments, metrics


def _plot(nodes: pd.DataFrame, segments: pd.DataFrame, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    blue, orange, green, black = "#0072B2", "#D55E00", "#009E73", "#000000"
    distance_km = nodes["along_path_m"] / 1_000.0
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.8), constrained_layout=True)
    axes[0, 0].plot(
        distance_km,
        nodes["energy_demand_mps2"],
        color=orange,
        label="planned demand",
    )
    axes[0, 0].plot(
        distance_km,
        nodes["maximum_specific_excess_force_mps2"],
        color=blue,
        label="maximum available",
    )
    axes[0, 0].set(
        xlabel="Along-path distance (km)", ylabel="Specific energy rate (m/s²)"
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].plot(
        distance_km,
        nodes["upper_energy_margin_mps2"],
        color=green,
        linewidth=1.3,
    )
    axes[0, 1].axhline(0.0, color=black, linestyle=":", linewidth=1.0)
    axes[0, 1].set(
        xlabel="Along-path distance (km)", ylabel="Upper energy margin (m/s²)"
    )
    colors = np.where(segments["integral_energy_margin_m2ps2"] >= 0.0, green, orange)
    axes[1, 0].bar(
        segments["state_leg"],
        segments["integral_energy_margin_m2ps2"] / 1_000.0,
        color=colors,
    )
    axes[1, 0].axhline(0.0, color=black, linewidth=0.8)
    axes[1, 0].set(xlabel="600-s state leg", ylabel="Integral energy margin (kJ/kg)")
    axes[1, 1].plot(
        distance_km,
        nodes["altitude_m"] / 1_000.0,
        color=blue,
        label="altitude",
    )
    speed_axis = axes[1, 1].twinx()
    speed_axis.plot(
        distance_km,
        nodes["true_airspeed_mps"],
        color=orange,
        linestyle="--",
        label="TAS",
    )
    axes[1, 1].set(xlabel="Along-path distance (km)", ylabel="Pressure altitude (km)")
    speed_axis.set_ylabel("True airspeed (m/s)")
    lines = axes[1, 1].lines + speed_axis.lines
    axes[1, 1].legend(
        lines, [line.get_label() for line in lines], frameon=False, fontsize=7
    )
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    speed_axis.spines["top"].set_visible(False)
    fig.suptitle("A320 decoupled-intent energy feasibility audit", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    nodes, segments, metrics = _diagnostics(_build_plan(args))
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "diagnosed",
        "source": {
            "recording": "TAIL-A/FLT0003",
            "model_controller_thrust_boundary_or_gate_changed": False,
            "post_holdout_diagnostic": True,
        },
        "metrics": metrics,
        "decision": {
            "exact_state_anchor_endpoints_integrally_reachable": metrics[
                "all_positive_energy_legs_have_integral_capacity"
            ],
            "exact_endpoint_preserving_planner_change_sufficient": metrics[
                "all_positive_energy_legs_have_integral_capacity"
            ],
            "intent_constraint_relaxation_or_capacity_change_required": not metrics[
                "all_positive_energy_legs_have_integral_capacity"
            ],
        },
        "artifacts": {
            "nodes": str(args.nodes),
            "segments": str(args.segments),
            "figure": f"{args.figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/audit_a320_fms_decoupled_energy.py",
        },
    }
    report = f"""# A320 decoupled-intent energy-feasibility audit

Updated: {UPDATED}

The current aircraft-aware plan has a minimum pointwise upper energy margin of
{metrics["minimum_pointwise_upper_margin_mps2"]:.5f} m/s² across
{metrics["upper_violation_node_count"]} violating nodes. Its worst node lies on
state leg {metrics["worst_upper_margin_state_leg"]}.

The minimum integrated positive-energy-leg margin is
{metrics["minimum_positive_leg_integral_margin_m2ps2"] / 1000:.2f} kJ/kg.
Diagnosis: **{metrics["diagnosis"]}**. No model, controller, demonstrated thrust
boundary, or gate is changed by this audit.
"""
    args.nodes.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(args.nodes, index=False)
    segments.to_csv(args.segments, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(nodes, segments, args.figure_prefix)
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
