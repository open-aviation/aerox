#!/usr/bin/env python3
"""Validate QAR-backed A320 spatial profile feasibility and FMS integration."""

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
    PointMassProfilePerformance,
    FlightIntent,
    QARDemonstratedClimbThrustCapacity,
    QAREffectiveClimbThrustCeiling,
    SpatialTrajectory,
    Waypoint,
    plan_feasible_profile,
)
from aerox.models.atmosphere import isa_atmosphere
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    N1_ENVELOPE_FEATURE_SETS,
    DemonstratedClimbN1Capacity,
    EffectiveN1Envelope,
    EffectiveThrustMap,
)
from tools.generate_a320_comparison_trajectories import _build_plants
from tools.generate_a320_fms_trajectory import _metrics, _run, build_intent

UPDATED = "2026-08-06"
INITIAL_MASS_KG = 65_000.0
SPACING_M = 500.0
DECLARED_CAPS = {
    "maximum_climb_angle_deg": 3.0,
    "maximum_descent_angle_deg": 3.0,
    "maximum_acceleration_mps2": 0.25,
    "maximum_deceleration_mps2": 0.25,
}
GATE_LIMITS = {
    "cross_track_rmse_m": 1_500.0,
    "terminal_altitude_error_m": 50.0,
    "terminal_speed_error_mps": 2.0,
    "constraint_tolerance": 1e-9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--pitch-bank-parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--surface-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"),
    )
    parser.add_argument(
        "--boundary-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.yml"),
    )
    parser.add_argument(
        "--lift-surface",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"),
    )
    parser.add_argument(
        "--thrust-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.yml"),
    )
    parser.add_argument(
        "--fuel-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-fuel-calibration.yml"),
    )
    parser.add_argument(
        "--n1-envelope",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--climb-capacity",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity.yml"),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("docs/fms/a320-fms-profile-feasibility.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-profile-feasibility.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-profile-feasibility.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-profile-feasibility"),
    )
    return parser.parse_args()


def _coefficient_vector(values: dict[str, float], order: tuple[str, ...]) -> np.ndarray:
    return np.array([values[name] for name in order], dtype=float)


def _load_climb_thrust_ceiling(
    args: argparse.Namespace,
) -> QAREffectiveClimbThrustCeiling | QARDemonstratedClimbThrustCapacity:
    envelope_document = yaml.safe_load(args.n1_envelope.read_text())
    envelope_model = envelope_document["model"]
    envelope_feature_set = envelope_model["feature_set"]
    envelope_order = N1_ENVELOPE_FEATURE_SETS[envelope_feature_set]
    envelope = EffectiveN1Envelope(
        feature_set=envelope_feature_set,
        idle_coefficients=_coefficient_vector(
            envelope_model["idle_coefficients"], envelope_order
        ),
        climb_coefficients=_coefficient_vector(
            envelope_model["climb_coefficients"], envelope_order
        ),
    )
    thrust_document = yaml.safe_load(args.thrust_parameters.read_text())
    thrust_model = thrust_document["model"]
    thrust_feature_set = thrust_model["feature_set"]
    thrust_order = THRUST_MAP_FEATURE_SETS[thrust_feature_set]
    thrust = EffectiveThrustMap(
        rated_total_thrust_n=float(thrust_model["rated_total_thrust_n"]),
        pressure_exponent=float(thrust_model["pressure_exponent"]),
        feature_set=thrust_feature_set,
        corrected_fan_speed=bool(thrust_model["corrected_fan_speed"]),
        coefficients=_coefficient_vector(
            thrust_model["scenarios"]["nominal"]["coefficients"], thrust_order
        ),
    )
    capacity_path = getattr(
        args,
        "climb_capacity",
        Path("docs/modeling/a320-demonstrated-climb-capacity.yml"),
    )
    if capacity_path.exists():
        capacity_document = yaml.safe_load(capacity_path.read_text())
        if capacity_document["release_decision"]["planner_integration_authorized"]:
            capacity_model = capacity_document["model"]
            capacity = DemonstratedClimbN1Capacity(
                feature_set=str(capacity_model["feature_set"]),
                bootstrap_coefficients=np.asarray(
                    capacity_model["bootstrap_coefficients"], dtype=float
                ),
                lower_confidence_quantile=float(
                    capacity_model["lower_confidence_quantile"]
                ),
                **{
                    name: float(value)
                    for name, value in capacity_model["domain"].items()
                },
            )
            return QARDemonstratedClimbThrustCapacity(
                capacity, thrust, fallback_n1_envelope=envelope
            )
    return QAREffectiveClimbThrustCeiling(envelope, thrust)


def _stress_route() -> SpatialTrajectory:
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                Waypoint(0.0, 0.0, 9_500.0, 215.0, "STRESS-START"),
                Waypoint(20_000.0, 0.0, 11_500.0, 215.0, "UNREACHABLE-ALT"),
            ),
            maximum_climb_flight_path_angle_rad=math.radians(10.0),
        )
    )


def _constraint_error(result: Any) -> tuple[float, float]:
    route = result.trajectory
    altitude_error = []
    speed_error = []
    for distance, waypoint in zip(
        route.waypoint_along_path_m, route.intent.waypoints, strict=True
    ):
        sample = route.sample(float(distance))
        altitude_error.append(abs(sample.altitude_m - waypoint.altitude_m))
        speed_error.append(abs(sample.true_airspeed_mps - waypoint.true_airspeed_mps))
    return max(altitude_error), max(speed_error)


def _required_profiles(result: Any) -> tuple[np.ndarray, np.ndarray]:
    distance = result.along_path_m
    altitude = result.nominal_altitude_m
    speed = result.nominal_true_airspeed_mps
    altitude_gradient = np.gradient(altitude, distance)
    speed_gradient = np.gradient(speed, distance)
    return np.arctan(altitude_gradient), speed * speed_gradient


def _build_document(
    result: Any,
    stress: Any,
    fms_metrics: dict[str, Any],
) -> dict[str, Any]:
    altitude_constraint_error, speed_constraint_error = _constraint_error(result)
    required_gamma, required_acceleration = _required_profiles(result)
    climb_margin = result.maximum_climb_angle_rad - np.maximum(required_gamma, 0.0)
    descent_margin = result.maximum_descent_angle_rad - np.maximum(-required_gamma, 0.0)
    acceleration_margin = result.maximum_acceleration_mps2 - np.maximum(
        required_acceleration, 0.0
    )
    deceleration_margin = result.maximum_deceleration_mps2 - np.maximum(
        -required_acceleration, 0.0
    )
    gates = {
        "base_profile_feasible": {"value": result.feasible, "passed": result.feasible},
        "stress_profile_rejected": {
            "value": stress.feasible,
            "passed": not stress.feasible,
        },
        "altitude_constraints_preserved": {
            "value": altitude_constraint_error,
            "maximum": GATE_LIMITS["constraint_tolerance"],
            "passed": altitude_constraint_error <= GATE_LIMITS["constraint_tolerance"],
        },
        "speed_constraints_preserved": {
            "value": speed_constraint_error,
            "maximum": GATE_LIMITS["constraint_tolerance"],
            "passed": speed_constraint_error <= GATE_LIMITS["constraint_tolerance"],
        },
        "climb_thrust_above_level_drag": {
            "minimum_margin_n": float(
                np.min(result.maximum_thrust_n - result.level_drag_n)
            ),
            "passed": bool(np.all(result.maximum_thrust_n > result.level_drag_n)),
        },
        "climb_angle_margin": {
            "minimum_margin_deg": float(np.rad2deg(np.min(climb_margin))),
            "passed": bool(np.all(climb_margin >= -1e-12)),
        },
        "descent_angle_margin": {
            "minimum_margin_deg": float(np.rad2deg(np.min(descent_margin))),
            "passed": bool(np.all(descent_margin >= -1e-12)),
        },
        "acceleration_margin": {
            "minimum_margin_mps2": float(np.min(acceleration_margin)),
            "passed": bool(np.all(acceleration_margin >= -1e-12)),
        },
        "deceleration_margin": {
            "minimum_margin_mps2": float(np.min(deceleration_margin)),
            "passed": bool(np.all(deceleration_margin >= -1e-12)),
        },
        "fms_completed": {
            "value": fms_metrics["completed"],
            "passed": fms_metrics["completed"],
        },
        "fms_cross_track": {
            "value_m": fms_metrics["cross_track_rmse_m"],
            "maximum_m": GATE_LIMITS["cross_track_rmse_m"],
            "passed": fms_metrics["cross_track_rmse_m"]
            <= GATE_LIMITS["cross_track_rmse_m"],
        },
        "fms_terminal_altitude": {
            "value_m": abs(fms_metrics["final_altitude_error_m"]),
            "maximum_m": GATE_LIMITS["terminal_altitude_error_m"],
            "passed": abs(fms_metrics["final_altitude_error_m"])
            <= GATE_LIMITS["terminal_altitude_error_m"],
        },
        "fms_terminal_speed": {
            "value_mps": abs(fms_metrics["final_speed_error_mps"]),
            "maximum_mps": GATE_LIMITS["terminal_speed_error_mps"],
            "passed": abs(fms_metrics["final_speed_error_mps"])
            <= GATE_LIMITS["terminal_speed_error_mps"],
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 pooled type-level model",
        "status": "passed" if passed else "held",
        "profile_model": {
            "coordinate": "cumulative horizontal distance s",
            "algorithm": "forward/backward reachable-band intersection with exact waypoint anchors",
            "spacing_m": SPACING_M,
            "mass_kg": INITIAL_MASS_KG,
            "maximum_thrust": "QAR effective CLB N1 ceiling composed with QAR effective N1-to-thrust map",
            "minimum_thrust_n": 0.0,
            "minimum_thrust_status": "mathematical lower bound; QAR idle boundary not released",
            "declared_caps": DECLARED_CAPS,
        },
        "profile_metrics": {
            "minimum_climb_thrust_n": float(np.min(result.maximum_thrust_n)),
            "maximum_climb_thrust_n": float(np.max(result.maximum_thrust_n)),
            "minimum_level_drag_n": float(np.min(result.level_drag_n)),
            "maximum_level_drag_n": float(np.max(result.level_drag_n)),
            "minimum_available_climb_angle_deg": float(
                np.rad2deg(np.min(result.maximum_climb_angle_rad))
            ),
            "minimum_available_descent_angle_deg": float(
                np.rad2deg(np.min(result.maximum_descent_angle_rad))
            ),
            "minimum_available_acceleration_mps2": float(
                np.min(result.maximum_acceleration_mps2)
            ),
            "minimum_available_deceleration_mps2": float(
                np.min(result.maximum_deceleration_mps2)
            ),
            "maximum_required_climb_angle_deg": float(
                np.rad2deg(np.max(np.maximum(required_gamma, 0.0)))
            ),
            "maximum_required_descent_angle_deg": float(
                np.rad2deg(np.max(np.maximum(-required_gamma, 0.0)))
            ),
            "maximum_required_acceleration_mps2": float(
                np.max(np.maximum(required_acceleration, 0.0))
            ),
            "maximum_required_deceleration_mps2": float(
                np.max(np.maximum(-required_acceleration, 0.0))
            ),
            "stress_violations": list(stress.violations),
        },
        "fms_metrics": fms_metrics,
        "gates": gates,
        "decision": {
            "profile_feasibility_gate_passed": passed,
            "qar_used_for_climb_thrust_ceiling": True,
            "qar_trajectory_used_as_reference": False,
            "idle_thrust_boundary_released": False,
            "airbus_fms_or_fadec_identified": False,
        },
        "artifacts": {
            "samples": "docs/fms/a320-fms-profile-feasibility.csv",
            "figure": "docs/fms/figures/a320-fms-profile-feasibility.png",
            "report": "docs/fms/a320-fms-profile-feasibility.md",
            "code": "tools/validate_a320_fms_profile_feasibility.py",
        },
    }


def _table(result: Any) -> pd.DataFrame:
    required_gamma, required_acceleration = _required_profiles(result)
    return pd.DataFrame(
        {
            "along_path_m": result.along_path_m,
            "nominal_altitude_m": result.nominal_altitude_m,
            "selected_altitude_m": result.trajectory.profile_altitude_m,
            "minimum_reachable_altitude_m": result.minimum_reachable_altitude_m,
            "maximum_reachable_altitude_m": result.maximum_reachable_altitude_m,
            "nominal_true_airspeed_mps": result.nominal_true_airspeed_mps,
            "selected_true_airspeed_mps": result.trajectory.profile_true_airspeed_mps,
            "minimum_reachable_true_airspeed_mps": result.minimum_reachable_true_airspeed_mps,
            "maximum_reachable_true_airspeed_mps": result.maximum_reachable_true_airspeed_mps,
            "maximum_climb_thrust_n": result.maximum_thrust_n,
            "level_drag_n": result.level_drag_n,
            "required_flight_path_angle_deg": np.rad2deg(required_gamma),
            "maximum_climb_angle_deg": np.rad2deg(result.maximum_climb_angle_rad),
            "maximum_descent_angle_deg": np.rad2deg(result.maximum_descent_angle_rad),
            "required_acceleration_mps2": required_acceleration,
            "maximum_acceleration_mps2": result.maximum_acceleration_mps2,
            "maximum_deceleration_mps2": result.maximum_deceleration_mps2,
        }
    )


def _style_axis(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot(result: Any, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    distance_km = result.along_path_m / 1_000.0
    blue, orange, green, purple = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
    required_gamma, required_acceleration = _required_profiles(result)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    axes[0, 0].fill_between(
        distance_km,
        result.minimum_reachable_altitude_m,
        result.maximum_reachable_altitude_m,
        color="#56B4E9",
        alpha=0.22,
        label="forward/backward reachable band",
    )
    axes[0, 0].plot(
        distance_km,
        result.nominal_altitude_m,
        color=blue,
        linewidth=1.8,
        label="selected/requested profile",
    )
    axes[0, 0].scatter(
        result.trajectory.waypoint_along_path_m / 1_000.0,
        [point.altitude_m for point in result.trajectory.intent.waypoints],
        color="black",
        facecolors="white",
        zorder=4,
        label="exact waypoint anchors",
    )
    axes[0, 0].set(xlabel="Along-path distance (km)", ylabel="Pressure altitude (m)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].fill_between(
        distance_km,
        result.minimum_reachable_true_airspeed_mps,
        result.maximum_reachable_true_airspeed_mps,
        color="#E69F00",
        alpha=0.22,
        label="forward/backward reachable band",
    )
    axes[0, 1].plot(
        distance_km,
        result.nominal_true_airspeed_mps,
        color=orange,
        linewidth=1.8,
        label="selected/requested profile",
    )
    axes[0, 1].scatter(
        result.trajectory.waypoint_along_path_m / 1_000.0,
        [point.true_airspeed_mps for point in result.trajectory.intent.waypoints],
        color="black",
        facecolors="white",
        zorder=4,
        label="exact waypoint anchors",
    )
    axes[0, 1].set(xlabel="Along-path distance (km)", ylabel="True airspeed (m/s)")
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].plot(
        distance_km,
        result.maximum_thrust_n / 1_000.0,
        color=green,
        linewidth=1.8,
        label="QAR-backed effective CLB ceiling",
    )
    axes[1, 0].plot(
        distance_km,
        result.level_drag_n / 1_000.0,
        color=purple,
        linestyle="--",
        linewidth=1.8,
        label="modeled level drag",
    )
    axes[1, 0].fill_between(
        distance_km,
        result.level_drag_n / 1_000.0,
        result.maximum_thrust_n / 1_000.0,
        color="#009E73",
        alpha=0.13,
    )
    axes[1, 0].set(xlabel="Along-path distance (km)", ylabel="Force (kN)")
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].plot(
        distance_km,
        np.rad2deg(required_gamma),
        color=blue,
        linewidth=1.8,
        label="required flight-path angle (deg)",
    )
    axes[1, 1].plot(
        distance_km,
        np.rad2deg(result.maximum_climb_angle_rad),
        color=green,
        linestyle="--",
        linewidth=1.5,
        label="available climb angle (deg)",
    )
    axes[1, 1].plot(
        distance_km,
        -np.rad2deg(result.maximum_descent_angle_rad),
        color=green,
        linestyle=":",
        linewidth=1.5,
        label="available descent angle (deg)",
    )
    acceleration_axis = axes[1, 1].twinx()
    acceleration_axis.plot(
        distance_km,
        required_acceleration,
        color=orange,
        linewidth=1.4,
        alpha=0.85,
        label="required acceleration",
    )
    acceleration_axis.set_ylabel("Acceleration (m/s²)")
    axes[1, 1].set(xlabel="Along-path distance (km)", ylabel="Flight-path angle (deg)")
    lines = axes[1, 1].lines + acceleration_axis.lines
    axes[1, 1].legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        fontsize=6,
        loc="best",
    )
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 FMS profile feasibility with QAR-backed climb-thrust ceiling", fontsize=13
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _report(document: dict[str, Any]) -> str:
    metrics = document["profile_metrics"]
    fms = document["fms_metrics"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if value['passed'] else 'hold'} |"
        for name, value in document["gates"].items()
    )
    return f"""# A320 FMS profile-feasibility gate

Updated: {UPDATED}

Status: **{document["status"]}**

## Result

The five-waypoint spatial altitude and speed profile is reachable under the
declared planning caps and the condition-dependent QAR-backed effective CLB
thrust ceiling. The modeled ceiling spans
{metrics["minimum_climb_thrust_n"] / 1000:.1f}–{metrics["maximum_climb_thrust_n"] / 1000:.1f} kN,
while modeled level drag spans
{metrics["minimum_level_drag_n"] / 1000:.1f}–{metrics["maximum_level_drag_n"] / 1000:.1f} kN.

The largest requested climb and descent angles are
{metrics["maximum_required_climb_angle_deg"]:.3f} and
{metrics["maximum_required_descent_angle_deg"]:.3f} degrees. The largest
requested acceleration and deceleration are
{metrics["maximum_required_acceleration_mps2"]:.4f} and
{metrics["maximum_required_deceleration_mps2"]:.4f} m/s². Exact waypoint
anchors are preserved. A deliberately unreachable altitude anchor is rejected
with: `{"; ".join(metrics["stress_violations"])}`.

The feasible dense profile was then passed through the unchanged online FMS
guidance and physical A320 plant. It completed with {fms["cross_track_rmse_m"]:.1f} m
cross-track RMSE, {fms["final_altitude_error_m"]:.1f} m final altitude error,
and {fms["final_speed_error_mps"]:.2f} m/s final speed error.

## QAR use

QAR is used here for the effective CLB N1 boundary and the effective
N1-to-total-net-thrust map. The route itself is synthetic, and no QAR heading
or trajectory samples are used as guidance references. This keeps the gate
independent of trajectory tracing.

The QAR audit did not release an idle N1/thrust boundary. Zero net thrust is
therefore retained only as an explicit mathematical lower bound, and the
3 degree descent and 0.25 m/s² deceleration caps are declared planning values.
Consequently, climb and positive-acceleration checks have stronger evidence
than descent and deceleration checks.

## Gates

| Check | Result |
|---|---|
{gate_rows}

## Next validation use of QAR

For response-blind FMS validation, sparse waypoint, altitude, and speed anchors
will be extracted from a held A320 QAR segment. Samples between anchors will be
withheld, the FMS will generate the trajectory independently, and the generated
states will be compared with those withheld samples. Observed heading history
will not be supplied to guidance.

## Artifacts

- Samples: `docs/fms/a320-fms-profile-feasibility.csv`
- Machine-readable result: `docs/fms/a320-fms-profile-feasibility.yml`
- Figure: `docs/fms/figures/a320-fms-profile-feasibility.{{png,pdf}}`
- Reproducer: `tools/validate_a320_fms_profile_feasibility.py`
"""


def main() -> None:
    args = parse_args()
    plant, _, _ = _build_plants(args)
    route = SpatialTrajectory.from_intent(build_intent())
    temperature = float(
        isa_atmosphere(np.array([route.intent.waypoints[0].altitude_m])).temperature_k[
            0
        ]
    )
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=_load_climb_thrust_ceiling(args),
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(DECLARED_CAPS["maximum_climb_angle_deg"]),
        maximum_descent_angle_rad=math.radians(
            DECLARED_CAPS["maximum_descent_angle_deg"]
        ),
        maximum_acceleration_mps2=DECLARED_CAPS["maximum_acceleration_mps2"],
        maximum_deceleration_mps2=DECLARED_CAPS["maximum_deceleration_mps2"],
    )
    result = plan_feasible_profile(
        route=route,
        performance=performance,
        mass_kg=INITIAL_MASS_KG,
        static_temperature_k=temperature,
        spacing_m=SPACING_M,
    )
    stress = plan_feasible_profile(
        route=_stress_route(),
        performance=performance,
        mass_kg=INITIAL_MASS_KG,
        static_temperature_k=temperature,
        spacing_m=SPACING_M,
    )
    fms = _run(result.trajectory, plant, position_feedback=True, wind_correction=True)
    document = _build_document(result, stress, _metrics(fms))
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    _table(result).to_csv(args.samples, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(result, args.figure_prefix)
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(document["profile_metrics"], sort_keys=False).strip())
    if not document["decision"]["profile_feasibility_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
