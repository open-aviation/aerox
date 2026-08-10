#!/usr/bin/env python3
"""Validate Aerox FMS generation against samples withheld between QAR anchors."""

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

from aerox import load_aircraft
from aerox.fms import (
    FlightIntent,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    Waypoint,
    plan_energy_shaped_profile,
)
from aerox.guidance import InverseFlightMechanicsController, SpatialPathGuidance
from aerox.modeling import PointMassProfilePerformance
from aerox.modeling.dynamics import (
    PitchBankThrustInput,
    PointMassModel,
)
from aerox.models.atmosphere import atmosphere_at_pressure_altitude
from aerox.models.drag import build_drag_model
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.generate_a320_comparison_trajectories import _controller
from tools.replay_a320_point_mass import _read_fuel_model
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_qar_guidance_robustness import (
    QARGuidanceReference,
    _build_nominal_model,
    _load_reference,
)

UPDATED = "2026-08-06"
ANCHOR_INTERVAL_S = 180
INTEGRATION_RATE_HZ = 10
GATE_LIMITS = {
    "qar_to_sparse_intent_path_rmse_m": 3_000.0,
    "generated_to_qar_path_rmse_m": 3_000.0,
    "time_aligned_horizontal_rmse_m": 6_000.0,
    "altitude_rmse_m": 250.0,
    "speed_rmse_mps": 8.0,
    "arrival_time_error_s": 120.0,
    "intermediate_waypoint_distance_m": 4_000.0,
    "destination_distance_m": 1_500.0,
    "terminal_altitude_error_m": 100.0,
    "terminal_speed_error_mps": 3.0,
    "limited_fraction": 0.10,
    "minimum_mach": 0.20,
    "maximum_mach": 0.82,
    "minimum_alpha_deg": -1.0,
    "maximum_alpha_deg": 8.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-extension-confirmation-panel.csv"),
    )
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
        "--errors",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-intent-errors.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-intent.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-intent.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-qar-intent-comparison"),
    )
    parser.add_argument(
        "--baseline-authority-history",
        type=Path,
        default=Path("docs/fms/a320-fms-alpha-energy-authority-history.csv"),
    )
    parser.add_argument(
        "--initialization-history",
        type=Path,
        default=Path("docs/fms/a320-fms-initialization-correction-history.csv"),
    )
    parser.add_argument(
        "--initialization-figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-initialization-correction"),
    )
    return parser.parse_args()


def _anchor_indices(sample_count: int) -> np.ndarray:
    indices = np.arange(0, sample_count, ANCHOR_INTERVAL_S, dtype=int)
    return np.unique(np.concatenate((indices, np.array([sample_count - 1]))))


def build_sparse_intent(
    qar: QARGuidanceReference,
) -> tuple[SpatialTrajectory, SpatialEnvironmentProfile, np.ndarray]:
    """Expose only frozen 180-second anchors to the FMS."""

    indices = _anchor_indices(len(qar.reference.time_s))
    waypoints = tuple(
        Waypoint(
            north_m=float(qar.north_position_m[index]),
            east_m=float(qar.east_position_m[index]),
            altitude_m=float(qar.reference.pressure_altitude_m[index]),
            true_airspeed_mps=float(qar.reference.true_airspeed_mps[index]),
            name=f"A{number}",
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
    return route, environment, indices


def _build_plant(args: argparse.Namespace) -> PointMassModel:
    aircraft = load_aircraft("A320")
    return PointMassModel(
        wing_area_m2=float(aircraft.wing.area),
        lift_map=_build_nominal_model(args, "TAIL-A"),
        drag_model=build_drag_model(aircraft),
        fuel_flow_prior=_read_fuel_model(args.thrust_parameters, args.fuel_parameters),
    )


def _initial_trim_control(
    *,
    plant: PointMassModel,
    controller: InverseFlightMechanicsController,
    trajectory: SpatialTrajectory,
    environment: SpatialEnvironmentProfile,
    initial_mass_kg: float,
) -> tuple[float, PitchBankThrustInput, float]:
    """Return planned initial gamma and a first-command rate-limit seed."""

    start = trajectory.intent.waypoints[0]
    reference = trajectory.sample(0.0)
    local_environment = environment.sample(0.0)
    state = np.array(
        [
            start.true_airspeed_mps,
            reference.flight_path_angle_rad,
            trajectory.segment_track_rad[0],
            start.north_m,
            start.east_m,
            start.altitude_m,
            initial_mass_kg,
        ]
    )
    command = controller.command(
        plant=plant,
        state=state,
        reference_true_airspeed_mps=reference.true_airspeed_mps,
        reference_true_airspeed_rate_mps2=reference.true_airspeed_rate_mps2,
        reference_flight_path_angle_rad=reference.flight_path_angle_rad,
        reference_flight_path_angle_rate_rad_s=0.0,
        reference_heading_rad=reference.ground_track_rad,
        reference_heading_rate_rad_s=0.0,
        reference_pressure_altitude_m=reference.altitude_m,
        static_temperature_k=local_environment.static_temperature_k,
        wind_north_mps=local_environment.wind_north_mps,
        wind_east_mps=local_environment.wind_east_mps,
        wind_down_mps=local_environment.wind_down_mps,
    )
    effective_alpha = plant.forces(state, command.control).angle_of_attack_rad
    return reference.flight_path_angle_rad, command.control, effective_alpha


def _point_to_polyline_distance(
    point_north_m: np.ndarray,
    point_east_m: np.ndarray,
    line_north_m: np.ndarray,
    line_east_m: np.ndarray,
) -> np.ndarray:
    first = np.column_stack((line_north_m[:-1], line_east_m[:-1]))
    delta = np.column_stack((np.diff(line_north_m), np.diff(line_east_m)))
    length_squared = np.sum(delta**2, axis=1)
    result = np.empty(len(point_north_m))
    for index, point in enumerate(zip(point_north_m, point_east_m, strict=True)):
        relative = np.asarray(point) - first
        fraction = np.clip(np.sum(relative * delta, axis=1) / length_squared, 0.0, 1.0)
        projection = first + fraction[:, None] * delta
        result[index] = float(np.min(np.linalg.norm(projection - point, axis=1)))
    return result


def _evaluate(
    qar: QARGuidanceReference,
    route: SpatialTrajectory,
    anchor_indices: np.ndarray,
    result: Any,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    trajectory = result.trajectory
    qar_time = qar.reference.time_s
    withheld = np.ones(len(qar_time), dtype=bool)
    withheld[anchor_indices] = False
    common = withheld & (qar_time <= trajectory.time_s[-1])
    generated_north = np.interp(
        qar_time[common], trajectory.time_s, trajectory.north_position_m
    )
    generated_east = np.interp(
        qar_time[common], trajectory.time_s, trajectory.east_position_m
    )
    generated_altitude = np.interp(
        qar_time[common], trajectory.time_s, trajectory.pressure_altitude_m
    )
    generated_speed = np.interp(
        qar_time[common], trajectory.time_s, trajectory.true_airspeed_mps
    )
    north_error = generated_north - qar.north_position_m[common]
    east_error = generated_east - qar.east_position_m[common]
    horizontal_error = np.hypot(north_error, east_error)
    altitude_error = generated_altitude - qar.reference.pressure_altitude_m[common]
    speed_error = generated_speed - qar.reference.true_airspeed_mps[common]
    intent_path_error = np.array(
        [
            route.project(
                float(north),
                float(east),
                first_segment_index=0,
                final_segment_index=len(route.segment_length_m) - 1,
            ).distance_m
            for north, east in zip(
                qar.north_position_m[withheld],
                qar.east_position_m[withheld],
                strict=True,
            )
        ]
    )
    generated_path_error = _point_to_polyline_distance(
        qar.north_position_m[withheld],
        qar.east_position_m[withheld],
        trajectory.north_position_m,
        trajectory.east_position_m,
    )
    waypoint_distance = result.waypoint_minimum_distance_m()
    atmosphere = atmosphere_at_pressure_altitude(
        trajectory.pressure_altitude_m, result.static_temperature_k
    )
    mach = trajectory.true_airspeed_mps / atmosphere.speed_of_sound_mps
    any_limited = result.pitch_limited | result.bank_limited | result.thrust_limited
    metrics = {
        "completed": result.completed,
        "anchor_count": len(anchor_indices),
        "withheld_sample_count": int(np.sum(common)),
        "qar_duration_s": float(qar_time[-1]),
        "generated_duration_s": float(trajectory.time_s[-1]),
        "arrival_time_error_s": float(trajectory.time_s[-1] - qar_time[-1]),
        "qar_to_sparse_intent_path_rmse_m": float(
            np.sqrt(np.mean(intent_path_error**2))
        ),
        "generated_to_qar_path_rmse_m": float(
            np.sqrt(np.mean(generated_path_error**2))
        ),
        "time_aligned_horizontal_rmse_m": float(np.sqrt(np.mean(horizontal_error**2))),
        "time_aligned_horizontal_p95_m": float(np.percentile(horizontal_error, 95.0)),
        "altitude_rmse_m": float(np.sqrt(np.mean(altitude_error**2))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "terminal_horizontal_error_m": float(
            np.hypot(
                trajectory.north_position_m[-1] - qar.north_position_m[-1],
                trajectory.east_position_m[-1] - qar.east_position_m[-1],
            )
        ),
        "terminal_altitude_error_m": float(
            trajectory.pressure_altitude_m[-1] - qar.reference.pressure_altitude_m[-1]
        ),
        "terminal_speed_error_mps": float(
            trajectory.true_airspeed_mps[-1] - qar.reference.true_airspeed_mps[-1]
        ),
        "waypoint_minimum_distance_m": {
            point.name: float(distance)
            for point, distance in zip(
                route.intent.waypoints, waypoint_distance, strict=True
            )
        },
        "any_control_limited_fraction": float(np.mean(any_limited)),
        "pitch_limited_fraction": float(np.mean(result.pitch_limited)),
        "bank_limited_fraction": float(np.mean(result.bank_limited)),
        "thrust_limited_fraction": float(np.mean(result.thrust_limited)),
        "minimum_mach": float(np.min(mach)),
        "maximum_mach": float(np.max(mach)),
        "minimum_alpha_deg": float(np.rad2deg(np.min(trajectory.angle_of_attack_rad))),
        "maximum_alpha_deg": float(np.rad2deg(np.max(trajectory.angle_of_attack_rad))),
        "maximum_absolute_bank_deg": float(
            np.rad2deg(np.max(np.abs(trajectory.bank_angle_rad)))
        ),
        "fuel_burn_kg": float(trajectory.mass_kg[0] - trajectory.mass_kg[-1]),
    }
    errors = pd.DataFrame(
        {
            "elapsed_time_s": qar_time[common],
            "north_error_m": north_error,
            "east_error_m": east_error,
            "horizontal_error_m": horizontal_error,
            "altitude_error_m": altitude_error,
            "true_airspeed_error_mps": speed_error,
        }
    )
    plot_data = {
        "common": common,
        "horizontal_error": horizontal_error,
        "altitude_error": altitude_error,
        "speed_error": speed_error,
        "intent_path_error": intent_path_error,
        "generated_path_error": generated_path_error,
    }
    return metrics, errors, plot_data


def _gates(metrics: dict[str, Any], profile_feasible: bool) -> dict[str, Any]:
    intermediate = list(metrics["waypoint_minimum_distance_m"].values())[1:-1]
    destination = list(metrics["waypoint_minimum_distance_m"].values())[-1]
    checks = {
        "profile_feasible": profile_feasible,
        "fms_completed": metrics["completed"],
        "qar_to_sparse_intent_path": metrics["qar_to_sparse_intent_path_rmse_m"]
        <= GATE_LIMITS["qar_to_sparse_intent_path_rmse_m"],
        "generated_to_qar_path": metrics["generated_to_qar_path_rmse_m"]
        <= GATE_LIMITS["generated_to_qar_path_rmse_m"],
        "time_aligned_horizontal": metrics["time_aligned_horizontal_rmse_m"]
        <= GATE_LIMITS["time_aligned_horizontal_rmse_m"],
        "altitude": metrics["altitude_rmse_m"] <= GATE_LIMITS["altitude_rmse_m"],
        "speed": metrics["speed_rmse_mps"] <= GATE_LIMITS["speed_rmse_mps"],
        "arrival_time": abs(metrics["arrival_time_error_s"])
        <= GATE_LIMITS["arrival_time_error_s"],
        "intermediate_waypoints": max(intermediate)
        <= GATE_LIMITS["intermediate_waypoint_distance_m"],
        "destination": destination <= GATE_LIMITS["destination_distance_m"],
        "terminal_altitude": abs(metrics["terminal_altitude_error_m"])
        <= GATE_LIMITS["terminal_altitude_error_m"],
        "terminal_speed": abs(metrics["terminal_speed_error_mps"])
        <= GATE_LIMITS["terminal_speed_error_mps"],
        "controller_limiters": metrics["any_control_limited_fraction"]
        <= GATE_LIMITS["limited_fraction"],
        "mach_envelope": GATE_LIMITS["minimum_mach"] <= metrics["minimum_mach"]
        and metrics["maximum_mach"] <= GATE_LIMITS["maximum_mach"],
        "alpha_envelope": GATE_LIMITS["minimum_alpha_deg"]
        <= metrics["minimum_alpha_deg"]
        and metrics["maximum_alpha_deg"] <= GATE_LIMITS["maximum_alpha_deg"],
    }
    return {name: {"passed": bool(passed)} for name, passed in checks.items()}


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


def _plot(
    qar: QARGuidanceReference,
    route: SpatialTrajectory,
    anchor_indices: np.ndarray,
    result: Any,
    plot_data: dict[str, np.ndarray],
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    trajectory = result.trajectory
    common = plot_data["common"]
    time_min = qar.reference.time_s[common] / 60.0
    black, blue, orange, green = "#000000", "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    axes[0, 0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=black,
        linestyle=":",
        linewidth=1.8,
        label="withheld QAR path",
    )
    axes[0, 0].plot(
        [point.east_m / 1_000.0 for point in route.intent.waypoints],
        [point.north_m / 1_000.0 for point in route.intent.waypoints],
        color=orange,
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markerfacecolor="white",
        label="sparse intent",
    )
    axes[0, 0].plot(
        trajectory.east_position_m / 1_000.0,
        trajectory.north_position_m / 1_000.0,
        color=blue,
        linewidth=1.8,
        label="FMS generated",
    )
    axes[0, 0].set(
        xlabel="Local east displacement (km)",
        ylabel="Local north displacement (km)",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].plot(time_min, plot_data["horizontal_error"], color=orange)
    axes[0, 1].set(
        xlabel="Elapsed time (min)", ylabel="Time-aligned horizontal error (m)"
    )
    axes[0, 2].plot(
        qar.reference.time_s / 60.0,
        qar.reference.pressure_altitude_m,
        color=black,
        linestyle=":",
        linewidth=1.8,
        label="withheld QAR",
    )
    axes[0, 2].plot(
        trajectory.time_s / 60.0,
        trajectory.pressure_altitude_m,
        color=blue,
        linewidth=1.7,
        label="FMS generated",
    )
    axes[0, 2].scatter(
        qar.reference.time_s[anchor_indices] / 60.0,
        qar.reference.pressure_altitude_m[anchor_indices],
        facecolors="white",
        edgecolors=orange,
        s=28,
        zorder=4,
        label="intent anchors",
    )
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (m)")
    axes[0, 2].legend(frameon=False, fontsize=7)
    axes[1, 0].plot(
        qar.reference.time_s / 60.0,
        qar.reference.true_airspeed_mps,
        color=black,
        linestyle=":",
        linewidth=1.8,
        label="withheld QAR",
    )
    axes[1, 0].plot(
        trajectory.time_s / 60.0,
        trajectory.true_airspeed_mps,
        color=blue,
        linewidth=1.7,
        label="FMS generated",
    )
    axes[1, 0].scatter(
        qar.reference.time_s[anchor_indices] / 60.0,
        qar.reference.true_airspeed_mps[anchor_indices],
        facecolors="white",
        edgecolors=orange,
        s=28,
        zorder=4,
        label="intent anchors",
    )
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="True airspeed (m/s)")
    axes[1, 0].legend(frameon=False, fontsize=7)
    withheld_time = np.delete(qar.reference.time_s, anchor_indices) / 60.0
    axes[1, 1].plot(
        withheld_time,
        plot_data["intent_path_error"],
        color=orange,
        linestyle="--",
        linewidth=1.5,
        label="QAR to sparse intent",
    )
    axes[1, 1].plot(
        withheld_time,
        plot_data["generated_path_error"],
        color=green,
        linewidth=1.6,
        label="QAR to generated path",
    )
    axes[1, 1].set(xlabel="Elapsed time (min)", ylabel="Nearest-path distance (m)")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 2].plot(
        trajectory.time_s / 60.0,
        np.rad2deg(trajectory.pitch_angle_rad),
        color=blue,
        linewidth=1.5,
        label="pitch",
    )
    axes[1, 2].plot(
        trajectory.time_s / 60.0,
        np.rad2deg(trajectory.bank_angle_rad),
        color=orange,
        linestyle="--",
        linewidth=1.5,
        label="bank",
    )
    axes[1, 2].set(xlabel="Elapsed time (min)", ylabel="Pitch / bank angle (deg)")
    axes[1, 2].legend(frameon=False, fontsize=7)
    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 FMS generation from sparse QAR intent; intermediate samples withheld",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _initialization_diagnostics(
    *,
    baseline_path: Path,
    result: Any,
    initial_gamma_rad: float,
    initial_control: PitchBankThrustInput,
    initial_effective_alpha_rad: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    baseline = pd.read_csv(baseline_path)
    trajectory = result.trajectory
    profile_target_s = np.minimum(
        result.progress_m + np.minimum(trajectory.true_airspeed_mps * 5.0, 1_000.0),
        result.planned_trajectory.total_distance_m,
    )
    reference_gamma = np.array(
        [
            result.planned_trajectory.sample(float(value)).ground_path_angle_rad
            for value in profile_target_s
        ]
    )
    corrected = pd.DataFrame(
        {
            "elapsed_time_s": trajectory.time_s,
            "reference_flight_path_angle_deg": np.rad2deg(reference_gamma),
            "realized_flight_path_angle_deg": np.rad2deg(
                trajectory.flight_path_angle_rad
            ),
            "realized_effective_alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "altitude_error_m": result.reference_altitude_m
            - trajectory.pressure_altitude_m,
            "true_airspeed_error_mps": result.reference_true_airspeed_mps
            - trajectory.true_airspeed_mps,
            "thrust_limited": result.thrust_limited,
        }
    )
    baseline_alpha = baseline["realized_effective_alpha_deg"].to_numpy()
    corrected_alpha = corrected["realized_effective_alpha_deg"].to_numpy()
    initialization = {
        "method": (
            "initialize plant gamma from the energy-shaped profile at s=0 and seed "
            "the first command's rate bounds with inverse-flight-mechanics trim"
        ),
        "controller_gains_changed": False,
        "profile_or_intent_changed": False,
        "alpha_boundary_changed": False,
        "initial_flight_path_angle_deg": float(np.rad2deg(initial_gamma_rad)),
        "initial_trim_pitch_deg": float(np.rad2deg(initial_control.pitch_angle_rad)),
        "initial_trim_bank_deg": float(np.rad2deg(initial_control.bank_angle_rad)),
        "initial_trim_net_thrust_kn": float(initial_control.net_thrust_n / 1_000.0),
        "initial_trim_effective_alpha_deg": float(
            np.rad2deg(initial_effective_alpha_rad)
        ),
        "baseline_maximum_effective_alpha_deg": float(np.max(baseline_alpha)),
        "corrected_maximum_effective_alpha_deg": float(np.max(corrected_alpha)),
        "baseline_alpha_above_8_5_sample_count": int(
            np.sum(baseline_alpha > GATE_LIMITS["maximum_alpha_deg"])
        ),
        "corrected_alpha_above_8_5_sample_count": int(
            np.sum(corrected_alpha > GATE_LIMITS["maximum_alpha_deg"])
        ),
        "baseline_source": str(baseline_path),
        "passed": bool(np.max(corrected_alpha) <= GATE_LIMITS["maximum_alpha_deg"]),
    }
    return baseline, corrected, initialization


def _plot_initialization_correction(
    baseline: pd.DataFrame,
    corrected: pd.DataFrame,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    blue, orange, green, black = "#0072B2", "#D55E00", "#009E73", "#000000"
    baseline_time = baseline["elapsed_time_s"].to_numpy()
    corrected_time = corrected["elapsed_time_s"].to_numpy()
    baseline_start = baseline_time <= 60.0
    corrected_start = corrected_time <= 60.0
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)

    axes[0, 0].plot(
        baseline_time[baseline_start],
        baseline.loc[baseline_start, "realized_effective_alpha_deg"],
        color=orange,
        linestyle="--",
        linewidth=1.7,
        label="zero-gamma initialization",
    )
    axes[0, 0].plot(
        corrected_time[corrected_start],
        corrected.loc[corrected_start, "realized_effective_alpha_deg"],
        color=blue,
        linewidth=1.7,
        label="planned-gamma + trim seed",
    )
    axes[0, 0].axhline(
        GATE_LIMITS["maximum_alpha_deg"],
        color=black,
        linestyle=":",
        linewidth=1.3,
        label="validated support",
    )
    axes[0, 0].set(xlabel="Elapsed time (s)", ylabel="Effective angle of attack (deg)")
    axes[0, 0].legend(frameon=False, fontsize=7)

    axes[0, 1].plot(
        baseline_time[baseline_start],
        baseline.loc[baseline_start, "realized_flight_path_angle_deg"],
        color=orange,
        linestyle="--",
        linewidth=1.5,
        label="baseline realized γ",
    )
    axes[0, 1].plot(
        corrected_time[corrected_start],
        corrected.loc[corrected_start, "realized_flight_path_angle_deg"],
        color=blue,
        linewidth=1.5,
        label="corrected realized γ",
    )
    axes[0, 1].plot(
        corrected_time[corrected_start],
        corrected.loc[corrected_start, "reference_flight_path_angle_deg"],
        color=green,
        linestyle=":",
        linewidth=1.5,
        label="planned γ",
    )
    axes[0, 1].set(xlabel="Elapsed time (s)", ylabel="Flight-path angle (deg)")
    axes[0, 1].legend(frameon=False, fontsize=7)

    axes[1, 0].plot(
        baseline_time / 60.0,
        baseline["altitude_error_m"],
        color=orange,
        linestyle="--",
        linewidth=1.4,
        label="baseline",
    )
    axes[1, 0].plot(
        corrected_time / 60.0,
        corrected["altitude_error_m"],
        color=blue,
        linewidth=1.4,
        label="corrected",
    )
    axes[1, 0].axhline(0.0, color=black, linewidth=0.8)
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="Altitude error (m)")
    axes[1, 0].legend(frameon=False, fontsize=7)

    axes[1, 1].plot(
        baseline_time / 60.0,
        baseline["true_airspeed_error_mps"],
        color=orange,
        linestyle="--",
        linewidth=1.4,
        label="baseline",
    )
    axes[1, 1].plot(
        corrected_time / 60.0,
        corrected["true_airspeed_error_mps"],
        color=blue,
        linewidth=1.4,
        label="corrected",
    )
    axes[1, 1].axhline(0.0, color=black, linewidth=0.8)
    axes[1, 1].set(xlabel="Elapsed time (min)", ylabel="True airspeed error (m/s)")
    axes[1, 1].legend(frameon=False, fontsize=7)

    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 FMS initialization correction; controller gains unchanged",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _document(
    metrics: dict[str, Any],
    gates: dict[str, Any],
    profile: Any,
    anchor_indices: np.ndarray,
    qar: QARGuidanceReference,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "passed" if passed else "held",
        "source": {
            "recording": "TAIL-A/FLT0004",
            "segment": "longest contiguous clean segment",
            "already_consumed_for_aircraft_model_validation": True,
            "new_independent_aircraft_model_confirmation": False,
            "duration_s": float(qar.reference.time_s[-1]),
            "sample_count": len(qar.reference.time_s),
            "raw_coordinates_persisted": False,
        },
        "intent_extraction": {
            "rule": "sample 0, every 180 seconds, and final sample",
            "anchor_interval_s": ANCHOR_INTERVAL_S,
            "anchor_indices": anchor_indices.tolist(),
            "anchor_count": len(anchor_indices),
            "exposed_to_fms": [
                "zero-referenced local north/east anchor",
                "smoothed pressure-altitude anchor",
                "smoothed true-airspeed anchor",
                "temperature and wind at anchors",
                "initial mass",
            ],
            "withheld_from_fms": [
                "all intermediate position samples",
                "all heading samples",
                "all intermediate altitude and speed samples",
                "recorded pitch, bank, N1, N2, and TLA",
            ],
        },
        "profile_feasibility": {
            "feasible": profile.feasible,
            "violations": list(profile.violations),
            "planner": "energy-shaped independent altitude/speed scheduling with monotone multi-knot refinement",
            "maximum_thrust": "uncertainty-bounded demonstrated CLB N1 capacity inside support; older operational q90 fallback outside support; effective thrust map downstream",
            "minimum_thrust_status": "unreleased; zero mathematical lower bound with declared descent/deceleration caps",
        },
        "initialization_correction": initialization,
        "metrics": metrics,
        "gates": gates,
        "decision": {
            "response_blind_sparse_intent_gate_passed": passed,
            "qar_used_as_withheld_reference": True,
            "qar_heading_used_by_guidance": False,
            "intermediate_qar_samples_used_by_guidance": False,
            "airbus_fms_autopilot_or_fadec_identified": False,
            "initialization_correction_passed": initialization["passed"],
        },
        "artifacts": {
            "errors": "docs/fms/a320-fms-qar-intent-errors.csv",
            "figure": "docs/fms/figures/a320-fms-qar-intent-comparison.png",
            "initialization_history": "docs/fms/a320-fms-initialization-correction-history.csv",
            "initialization_figure": "docs/fms/figures/a320-fms-initialization-correction.png",
            "report": "docs/fms/a320-fms-qar-intent.md",
            "code": "tools/validate_a320_fms_qar_intent.py",
        },
    }


def _report(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    initialization = document["initialization_correction"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if gate['passed'] else 'hold'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 FMS response-blind sparse-QAR-intent gate

Updated: {UPDATED}

Status: **{document["status"]}**

## Result

Nine sparse anchors, selected every 180 seconds plus the final sample, were
extracted from the 1,423-sample clean segment of `TAIL-A/FLT0004`. The Aerox
FMS received only local anchor position, altitude, speed, sparse environment,
and initial mass. It received no observed heading and no intermediate state or
control samples.

The sparse intent approximates the withheld QAR path with
{metrics["qar_to_sparse_intent_path_rmse_m"]:.1f} m nearest-path RMSE. The
generated trajectory has {metrics["generated_to_qar_path_rmse_m"]:.1f} m
nearest-path RMSE and {metrics["time_aligned_horizontal_rmse_m"]:.1f} m
time-aligned horizontal RMSE. Altitude and TAS RMSE are
{metrics["altitude_rmse_m"]:.1f} m and {metrics["speed_rmse_mps"]:.2f} m/s.
Generated arrival differs by {metrics["arrival_time_error_s"]:+.0f} s.

The final horizontal, altitude, and TAS errors are
{metrics["terminal_horizontal_error_m"]:.1f} m,
{metrics["terminal_altitude_error_m"]:.1f} m, and
{metrics["terminal_speed_error_mps"]:.2f} m/s. Modeled fuel burn is
{metrics["fuel_burn_kg"]:.1f} kg.

Planning uses the released complete-flight-bootstrap demonstrated CLB capacity
and independent altitude/speed energy shaping with monotone multi-knot timing.
All path, altitude, speed, arrival, waypoint, terminal, and Mach gates now pass.
The initialization correction reduces maximum effective alpha from
{initialization["baseline_maximum_effective_alpha_deg"]:.2f}° to
{initialization["corrected_maximum_effective_alpha_deg"]:.2f}° without changing
the controller gains, intent, profile, or 8.5° boundary. The alpha gate now
passes. The overall gate remains held by the small coupled-energy feasibility
deficit and {100 * metrics["thrust_limited_fraction"]:.1f}% thrust-limiter
occupancy.

## Initialization correction

The physical state now starts at the energy-shaped profile's initial
flight-path angle, {initialization["initial_flight_path_angle_deg"]:.2f}°. An
inverse-flight-mechanics trim command seeds the first pitch/bank/thrust rate
bounds; it is not propagated before time zero and does not replace the first
controller command. The prior zero-gamma rollout exceeded 8.5° for
{initialization["baseline_alpha_above_8_5_sample_count"]} samples. The corrected
rollout exceeds it for
{initialization["corrected_alpha_above_8_5_sample_count"]} samples.

## What this validates

This is a response-blind test of intent extraction, profile feasibility,
spatial guidance, wind correction, and closed-loop trajectory generation.
Because this flight was already consumed by the earlier aircraft-model
guidance gate, it is not a new independent validation of lift, drag, thrust, or
fuel parameters.

QAR remains the withheld comparison truth. Only the nine frozen anchors are
used to create intent. Observed heading, intermediate trajectory points,
recorded attitude, N1, N2, and TLA are not passed to guidance.

## Gates

| Check | Result |
|---|---|
{gate_rows}

## Data boundary

Raw latitude/longitude and sample-level geodetic trajectories are external and
not persisted. The CSV artifact contains error histories only. The figure uses
zero-referenced local displacement, with no recoverable geographic origin.

## Artifacts

- Error history: `docs/fms/a320-fms-qar-intent-errors.csv`
- Machine-readable result: `docs/fms/a320-fms-qar-intent.yml`
- Comparison figure: `docs/fms/figures/a320-fms-qar-intent-comparison.{{png,pdf}}`
- Initialization history: `docs/fms/a320-fms-initialization-correction-history.csv`
- Before/after initialization figure: `docs/fms/figures/a320-fms-initialization-correction.{{png,pdf}}`
- Reproducer: `tools/validate_a320_fms_qar_intent.py`
"""


def main() -> None:
    args = parse_args()
    qar, _ = _load_reference(args)
    route, environment, anchor_indices = build_sparse_intent(qar)
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
        static_temperature_k=float(np.median(qar.static_temperature_k[anchor_indices])),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[anchor_indices],
    )
    guidance = SpatialPathGuidance(
        profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
    )
    controller = _controller()
    initial_gamma, initial_control, initial_effective_alpha = _initial_trim_control(
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
    metrics, errors, plot_data = _evaluate(qar, route, anchor_indices, result)
    baseline, initialization_history, initialization = _initialization_diagnostics(
        baseline_path=args.baseline_authority_history,
        result=result,
        initial_gamma_rad=initial_gamma,
        initial_control=initial_control,
        initial_effective_alpha_rad=initial_effective_alpha,
    )
    gates = _gates(metrics, profile.feasible)
    document = _document(
        metrics,
        gates,
        profile,
        anchor_indices,
        qar,
        initialization,
    )
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    initialization_history.to_csv(args.initialization_history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(qar, route, anchor_indices, result, plot_data, args.figure_prefix)
    _plot_initialization_correction(
        baseline,
        initialization_history,
        args.initialization_figure_prefix,
    )
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())
    if not document["decision"]["response_blind_sparse_intent_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
