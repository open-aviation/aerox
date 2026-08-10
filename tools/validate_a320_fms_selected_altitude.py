#!/usr/bin/env python3
"""Replay A320 QAR selector intent through Aerox guidance and dynamics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
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
    LateralWaypoint,
    SelectedAltitudeIntent,
    SelectedSpeedIntent,
    SpeedTarget,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    TerminalConstraint,
)
from aerox.guidance import SelectedAltitudeGuidance, SpatialPathGuidance
from aerox.models.atmosphere import (
    calibrated_airspeed_from_true_airspeed,
    mach_from_true_airspeed,
)
from aerox.models.thrust import PressureScaledMinimumThrust
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.audit_a320_fms_intent_channels import _load_timeline
from tools.generate_a320_comparison_trajectories import _controller
from tools.replay_a320_point_mass import _read_fuel_model
from tools.research_support import native_value
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import _build_plant

UPDATED = "2026-08-10"
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
INTEGRATION_RATE_HZ = 10
COLORS = {
    "black": "#000000",
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#E69F00",
}


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
        "descent-thrust-parameters": "docs/modeling/a320-descent-planning-thrust.yml",
        "history": "docs/fms/a320-fms-selected-altitude-history.csv",
        "errors": "docs/fms/a320-fms-selected-altitude-errors.csv",
        "parameters": "docs/fms/a320-fms-selected-altitude.yml",
        "report": "docs/fms/a320-fms-selected-altitude.md",
        "figure-dir": "docs/fms/figures/a320-fms-selected-altitude",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _load_indices(path: Path, sample_count: int) -> np.ndarray:
    document = yaml.safe_load(path.read_text())
    indices = np.asarray(document["anchor_indices"], dtype=int)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or indices[0] != 0
        or indices[-1] != sample_count - 1
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError("adaptive lateral-intent indices are invalid")
    return indices


def _lateral_route(qar: Any, indices: np.ndarray) -> SpatialTrajectory:
    """Build the lateral carrier without prescribing a vertical trajectory."""

    waypoints = tuple(
        LateralWaypoint(
            north_m=float(qar.north_position_m[index]),
            east_m=float(qar.east_position_m[index]),
            name=f"P{number}",
            acceptance_radius_m=1_500.0 if index == indices[-1] else 4_000.0,
        )
        for number, index in enumerate(indices)
    )
    initial_altitude = float(qar.reference.pressure_altitude_m[0])
    initial_speed = float(qar.reference.true_airspeed_mps[0])
    placeholder = TerminalConstraint(
        initial_altitude,
        SpeedTarget("TAS", initial_speed),
    )
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=waypoints,
            departure=placeholder,
            arrival=placeholder,
            maximum_climb_flight_path_angle_rad=math.radians(6.0),
            maximum_descent_flight_path_angle_rad=math.radians(4.0),
            maximum_acceleration_mps2=0.5,
            maximum_deceleration_mps2=0.5,
        )
    )


def _environment(
    qar: Any, route: SpatialTrajectory, indices: np.ndarray
) -> SpatialEnvironmentProfile:
    return SpatialEnvironmentProfile(
        along_path_m=route.waypoint_along_path_m,
        static_temperature_k=qar.static_temperature_k[indices],
        wind_north_mps=qar.wind_north_mps[indices],
        wind_east_mps=qar.wind_east_mps[indices],
        wind_down_mps=np.zeros(len(indices)),
    )


def _load_minimum_thrust(path: Path) -> PressureScaledMinimumThrust:
    document = yaml.safe_load(path.read_text())
    release = document["release_gates"]
    if not release["all_passed"] or not release["planner_integration_authorized"]:
        raise ValueError("descent planning-thrust prior is not released")
    model = document["model"]
    return PressureScaledMinimumThrust(
        sea_level_thrust_n=float(model["sea_level_thrust_n"]),
        pressure_exponent=float(model["pressure_exponent"]),
    )


def _cumulative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    return np.r_[
        0.0,
        np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(time_s)),
    ]


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _mode_fractions(values: np.ndarray) -> dict[str, float]:
    counts = Counter(str(value) for value in values)
    return {name: count / len(values) for name, count in sorted(counts.items())}


def _history_and_metrics(
    *,
    qar: Any,
    timeline: pd.DataFrame,
    route: SpatialTrajectory,
    result: Any,
    openap_fuel_flow_kg_s: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trajectory = result.trajectory
    if len(trajectory.time_s) != len(qar.reference.time_s) or not np.allclose(
        trajectory.time_s, qar.reference.time_s
    ):
        raise ValueError("generated and QAR histories must share the exact 1 Hz grid")
    altitude = trajectory.pressure_altitude_m
    qar_altitude = qar.reference.pressure_altitude_m
    generated_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            trajectory.true_airspeed_mps,
            altitude,
            result.static_temperature_k,
        )
    )
    generated_mach = np.asarray(
        mach_from_true_airspeed(
            trajectory.true_airspeed_mps,
            altitude,
            result.static_temperature_k,
        )
    )
    qar_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            qar_altitude,
            qar.static_temperature_k,
        )
    )
    qar_mach = np.asarray(
        mach_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            qar_altitude,
            qar.static_temperature_k,
        )
    )
    north_error = trajectory.north_position_m - qar.north_position_m
    east_error = trajectory.east_position_m - qar.east_position_m
    horizontal_error = np.hypot(north_error, east_error)
    altitude_error = altitude - qar_altitude
    tas_error = trajectory.true_airspeed_mps - qar.reference.true_airspeed_mps
    cas_error = generated_cas - qar_cas
    mach_error = generated_mach - qar_mach
    calibrated_fuel = trajectory.fuel_flow_kg_s
    qar_fuel = qar.fuel_flow_kg_s
    time_s = trajectory.time_s
    any_limited = result.pitch_limited | result.bank_limited | result.thrust_limited
    selected_altitude = timeline["selected_altitude_ft"].to_numpy() * FT_TO_M
    qar_path_error = np.asarray(
        [
            route.project(float(north), float(east)).distance_m
            for north, east in zip(
                qar.north_position_m, qar.east_position_m, strict=True
            )
        ]
    )

    history = pd.DataFrame(
        {
            "elapsed_time_s": time_s,
            "qar_north_position_m": qar.north_position_m,
            "qar_east_position_m": qar.east_position_m,
            "generated_north_position_m": trajectory.north_position_m,
            "generated_east_position_m": trajectory.east_position_m,
            "qar_pressure_altitude_m": qar_altitude,
            "selected_altitude_m": selected_altitude,
            "generated_pressure_altitude_m": altitude,
            "qar_true_airspeed_mps": qar.reference.true_airspeed_mps,
            "selected_calibrated_airspeed_mps": timeline[
                "selected_speed_kts"
            ].to_numpy()
            * KNOT_TO_MPS,
            "selected_mach": timeline["selected_mach"].to_numpy(),
            "generated_true_airspeed_mps": trajectory.true_airspeed_mps,
            "qar_calibrated_airspeed_mps": qar_cas,
            "generated_calibrated_airspeed_mps": generated_cas,
            "qar_mach": qar_mach,
            "generated_mach": generated_mach,
            "vertical_mode": result.vertical_mode,
            "speed_mode": result.speed_mode,
            "reference_flight_path_angle_deg": np.rad2deg(
                result.reference_air_flight_path_angle_rad
            ),
            "flight_path_angle_deg": np.rad2deg(trajectory.flight_path_angle_rad),
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_angle_deg": np.rad2deg(trajectory.bank_angle_rad),
            "effective_alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "net_thrust_kn": trajectory.net_thrust_n / 1_000.0,
            "drag_kn": trajectory.drag_n / 1_000.0,
            "qar_fuel_flow_kg_s": qar_fuel,
            "aerox_fuel_flow_kg_s": calibrated_fuel,
            "openap_fuel_flow_kg_s": openap_fuel_flow_kg_s,
            "qar_cumulative_fuel_kg": _cumulative(qar_fuel, time_s),
            "aerox_cumulative_fuel_kg": _cumulative(calibrated_fuel, time_s),
            "openap_cumulative_fuel_kg": _cumulative(openap_fuel_flow_kg_s, time_s),
            "pitch_limited": result.pitch_limited,
            "bank_limited": result.bank_limited,
            "thrust_limited": result.thrust_limited,
        }
    )
    errors = pd.DataFrame(
        {
            "elapsed_time_s": time_s,
            "north_error_m": north_error,
            "east_error_m": east_error,
            "horizontal_error_m": horizontal_error,
            "altitude_error_m": altitude_error,
            "true_airspeed_error_mps": tas_error,
            "calibrated_airspeed_error_mps": cas_error,
            "mach_error": mach_error,
        }
    )
    metrics = {
        "sample_count": len(time_s),
        "recording_duration_s": float(time_s[-1]),
        "recording_interval_completed": bool(time_s[-1] == qar.reference.time_s[-1]),
        "horizontal_rmse_m": _rmse(horizontal_error),
        "horizontal_p95_m": float(np.percentile(horizontal_error, 95.0)),
        "qar_to_lateral_intent_path_rmse_m": _rmse(qar_path_error),
        "generated_cross_track_rmse_m": _rmse(result.cross_track_error_m),
        "generated_cross_track_p95_m": float(
            np.percentile(np.abs(result.cross_track_error_m), 95.0)
        ),
        "altitude_rmse_m": _rmse(altitude_error),
        "true_airspeed_rmse_mps": _rmse(tas_error),
        "calibrated_airspeed_rmse_mps": _rmse(cas_error),
        "mach_rmse": _rmse(mach_error),
        "final_horizontal_error_m": float(horizontal_error[-1]),
        "final_altitude_error_m": float(altitude_error[-1]),
        "final_true_airspeed_error_mps": float(tas_error[-1]),
        "altitude_selector_change_count": int(np.sum(np.diff(selected_altitude) != 0)),
        "speed_selector_change_count": int(
            np.sum(
                (np.diff(timeline["selected_speed_kts"].to_numpy()) != 0)
                | (np.diff(timeline["selected_mach"].to_numpy()) != 0)
            )
        ),
        "vertical_mode_fraction": _mode_fractions(result.vertical_mode),
        "speed_mode_fraction": _mode_fractions(result.speed_mode),
        "any_control_limited_fraction": float(np.mean(any_limited)),
        "pitch_limited_fraction": float(np.mean(result.pitch_limited)),
        "bank_limited_fraction": float(np.mean(result.bank_limited)),
        "thrust_limited_fraction": float(np.mean(result.thrust_limited)),
        "minimum_mach": float(np.min(generated_mach)),
        "maximum_mach": float(np.max(generated_mach)),
        "minimum_alpha_deg": float(np.min(np.rad2deg(trajectory.angle_of_attack_rad))),
        "maximum_alpha_deg": float(np.max(np.rad2deg(trajectory.angle_of_attack_rad))),
        "maximum_absolute_bank_deg": float(
            np.max(np.abs(np.rad2deg(trajectory.bank_angle_rad)))
        ),
        "qar_fuel_burn_kg": float(_cumulative(qar_fuel, time_s)[-1]),
        "aerox_fuel_burn_kg": float(_cumulative(calibrated_fuel, time_s)[-1]),
        "openap_fuel_burn_kg": float(_cumulative(openap_fuel_flow_kg_s, time_s)[-1]),
        "aerox_fuel_flow_rmse_kg_s": _rmse(calibrated_fuel - qar_fuel),
        "openap_fuel_flow_rmse_kg_s": _rmse(openap_fuel_flow_kg_s - qar_fuel),
    }
    return history, errors, metrics


def _style_axis(axis: Any, panel: str, *, grid: bool = True) -> None:
    axis.text(
        -0.12,
        1.06,
        panel,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    if grid:
        axis.grid(True, color="#D0D0D0", alpha=0.55, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(fig: Any, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")


def _plot_all(
    *,
    qar: Any,
    timeline: pd.DataFrame,
    route: SpatialTrajectory,
    result: Any,
    history: pd.DataFrame,
    errors: pd.DataFrame,
    figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
        }
    )
    time_min = timeline["elapsed_time_s"].to_numpy() / 60.0
    generated = result.trajectory

    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=COLORS["black"],
        linestyle=":",
        label="QAR",
    )
    axis.plot(
        generated.east_position_m / 1_000.0,
        generated.north_position_m / 1_000.0,
        color=COLORS["blue"],
        label="Aerox generated",
    )
    axis.plot(
        [point.east_m / 1_000.0 for point in route.intent.waypoints],
        [point.north_m / 1_000.0 for point in route.intent.waypoints],
        color=COLORS["orange"],
        linestyle="none",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        label="lateral pseudo-waypoints",
    )
    axis.set(xlabel="East position (km)", ylabel="North position (km)")
    axis.axis("equal")
    _style_axis(axis, "a")
    axis.legend(frameon=False)
    _save(fig, figure_dir, "01-horizontal-trajectory")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    axis.plot(
        time_min,
        history["qar_pressure_altitude_m"] / FT_TO_M,
        color=COLORS["black"],
        linestyle=":",
        label="QAR altitude",
    )
    axis.step(
        time_min,
        history["selected_altitude_m"] / FT_TO_M,
        where="post",
        color=COLORS["orange"],
        label="ALT_SEL intent",
    )
    axis.plot(
        time_min,
        history["generated_pressure_altitude_m"] / FT_TO_M,
        color=COLORS["blue"],
        label="Aerox generated",
    )
    axis.set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (ft)")
    _style_axis(axis, "a")
    axis.legend(frameon=False, ncol=3)
    _save(fig, figure_dir, "02-altitude-and-selected-intent")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        time_min,
        history["qar_calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=COLORS["black"],
        linestyle=":",
        label="QAR",
    )
    axes[0].step(
        time_min,
        history["selected_calibrated_airspeed_mps"] / KNOT_TO_MPS,
        where="post",
        color=COLORS["orange"],
        label="selected",
    )
    axes[0].plot(
        time_min,
        history["generated_calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=COLORS["blue"],
        label="Aerox generated",
    )
    axes[0].set(ylabel="Calibrated airspeed (kt)")
    _style_axis(axes[0], "a")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(
        time_min,
        history["qar_mach"],
        color=COLORS["black"],
        linestyle=":",
        label="QAR",
    )
    axes[1].step(
        time_min,
        history["selected_mach"],
        where="post",
        color=COLORS["orange"],
        label="selected",
    )
    axes[1].plot(
        time_min,
        history["generated_mach"],
        color=COLORS["blue"],
        label="Aerox generated",
    )
    axes[1].set(xlabel="Elapsed time (min)", ylabel="Mach number")
    _style_axis(axes[1], "b")
    _save(fig, figure_dir, "03-speed-and-selected-intent")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 2, figsize=(9.2, 6.3), sharex=True, constrained_layout=True
    )
    for axis, column, label, panel, color in (
        (axes[0, 0], "pitch_angle_deg", "Pitch angle (deg)", "a", "blue"),
        (axes[0, 1], "bank_angle_deg", "Bank angle (deg)", "b", "orange"),
        (axes[1, 0], "net_thrust_kn", "Net thrust (kN)", "c", "green"),
        (axes[1, 1], "drag_kn", "Drag (kN)", "d", "purple"),
    ):
        axis.plot(time_min, history[column], color=COLORS[color])
        axis.set_ylabel(label)
        _style_axis(axis, panel)
    axes[1, 0].set_xlabel("Elapsed time (min)")
    axes[1, 1].set_xlabel("Elapsed time (min)")
    _save(fig, figure_dir, "04-generated-controls-and-forces")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True
    )
    for label, column, color, style in (
        ("QAR", "qar_fuel_flow_kg_s", "black", ":"),
        ("Aerox calibrated", "aerox_fuel_flow_kg_s", "blue", "-"),
        ("OpenAP prior", "openap_fuel_flow_kg_s", "orange", "--"),
    ):
        axes[0].plot(
            time_min, history[column], color=COLORS[color], linestyle=style, label=label
        )
    axes[0].set_ylabel("Fuel flow (kg s$^{-1}$)")
    _style_axis(axes[0], "a")
    axes[0].legend(frameon=False, ncol=3)
    for label, column, color, style in (
        ("QAR", "qar_cumulative_fuel_kg", "black", ":"),
        ("Aerox calibrated", "aerox_cumulative_fuel_kg", "blue", "-"),
        ("OpenAP prior", "openap_cumulative_fuel_kg", "orange", "--"),
    ):
        axes[1].plot(
            time_min, history[column], color=COLORS[color], linestyle=style, label=label
        )
    axes[1].set(xlabel="Elapsed time (min)", ylabel="Cumulative fuel (kg)")
    _style_axis(axes[1], "b")
    _save(fig, figure_dir, "05-fuel-comparison")
    plt.close(fig)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.0, 7.4), sharex=True, constrained_layout=True
    )
    axes[0].plot(time_min, errors["horizontal_error_m"] / 1_000.0, color=COLORS["blue"])
    axes[0].set_ylabel("Horizontal error (km)")
    _style_axis(axes[0], "a")
    axes[1].plot(time_min, errors["altitude_error_m"] / FT_TO_M, color=COLORS["orange"])
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("Altitude error (ft)")
    _style_axis(axes[1], "b")
    limiter_count = (
        history[["pitch_limited", "bank_limited", "thrust_limited"]]
        .to_numpy(dtype=int)
        .sum(axis=1)
    )
    axes[2].step(time_min, limiter_count, where="post", color=COLORS["purple"])
    axes[2].set(
        xlabel="Elapsed time (min)",
        ylabel="Active limiters",
        yticks=[0, 1, 2, 3],
    )
    _style_axis(axes[2], "c")
    _save(fig, figure_dir, "06-errors-and-limiters")
    plt.close(fig)


def _document(
    *,
    registration: str,
    recording: str,
    qar: Any,
    indices: np.ndarray,
    altitude_intent: SelectedAltitudeIntent,
    speed_intent: SelectedSpeedIntent,
    guidance: SelectedAltitudeGuidance,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "diagnostic",
        "source": {
            "recording": f"{registration}/{recording}",
            "duration_s": float(qar.reference.time_s[-1]),
            "sample_count": len(qar.reference.time_s),
            "raw_coordinates_persisted": False,
        },
        "intent_boundary": {
            "lateral_intent": "turn-aware pseudo-waypoints from frozen QAR geometry",
            "lateral_geometry_node_count": len(indices),
            "vertical_intent": "sample-and-hold ALT_SEL pressure-altitude commands",
            "altitude_command_event_count": len(altitude_intent.effective_time_s),
            "speed_intent": "sample-and-hold SPEEDP_SEL/MACH_SEL with CAS-Mach crossover",
            "speed_command_event_count": len(speed_intent.effective_time_s),
            "terminal_QAR_altitude_used_as_intent": False,
            "terminal_QAR_speed_used_as_intent": False,
            "artificial_altitude_profile_constructed": False,
            "top_of_climb_or_descent_constructed": False,
            "future_selector_commands_replayed": True,
            "recorded_attitude_or_thrust_controls_used": False,
            "recorded_aircraft_response_used_by_feedback": False,
            "final_QAR_state_role": "evaluation only; recording ends during descent",
        },
        "guidance": {
            "vertical_law": "selected-altitude error to bounded vertical speed, then wind-corrected flight-path angle",
            "capture_time_constant_s": guidance.capture_time_constant_s,
            "altitude_hold_deadband_ft": guidance.altitude_hold_deadband_m / FT_TO_M,
            "maximum_climb_angle_deg": math.degrees(guidance.maximum_climb_angle_rad),
            "maximum_descent_angle_deg": math.degrees(
                guidance.maximum_descent_angle_rad
            ),
            "maximum_flight_path_command_rate_deg_s": math.degrees(
                guidance.maximum_flight_path_command_rate_rad_s
            ),
            "command_rate_hz": 1,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
        },
        "metrics": metrics,
        "interpretation": {
            "causal_live_fms": False,
            "reason": "the complete historical selector timeline is supplied as replay intent",
            "trajectory_generated_by_plant": True,
            "state_anchors_or_resets_after_initialization": 0,
        },
        "artifacts": {
            "history": "docs/fms/a320-fms-selected-altitude-history.csv",
            "errors": "docs/fms/a320-fms-selected-altitude-errors.csv",
            "figures": "docs/fms/figures/a320-fms-selected-altitude",
            "code": "tools/validate_a320_fms_selected_altitude.py",
        },
    }


def _report(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    intent = document["intent_boundary"]
    return f"""# A320 selected-altitude FMS replay

Updated: {UPDATED}

This experiment uses the recorded `ALT_SEL` channel as the active altitude
target. It uses the recorded selected CAS/Mach pair as speed intent and the
previously frozen pseudo-waypoints as lateral intent. It does not calculate a
TOC, TOD, cruise-altitude profile, or terminal vertical constraint.

The selector commands are exogenous replay inputs, while every aircraft state
after initialization is generated by the Aerox plant. This is therefore a
trajectory-generation test of the guidance and aircraft model, not a causal
prediction of unknown future pilot/FMS selections.

## Intent boundary

- Altitude command events: {intent["altitude_command_event_count"]}
- Speed command events: {intent["speed_command_event_count"]}
- Lateral pseudo-waypoints: {intent["lateral_geometry_node_count"]}
- Final QAR state used only for evaluation: yes
- Artificial altitude profile or TOC/TOD: no

## Results over the common {metrics["recording_duration_s"]:.0f} s interval

- Horizontal RMSE: {metrics["horizontal_rmse_m"] / 1000:.2f} km
- Generated cross-track RMSE: {metrics["generated_cross_track_rmse_m"]:.0f} m
- Altitude RMSE: {metrics["altitude_rmse_m"] / FT_TO_M:.0f} ft
- CAS RMSE: {metrics["calibrated_airspeed_rmse_mps"] / KNOT_TO_MPS:.1f} kt
- Mach RMSE: {metrics["mach_rmse"]:.4f}
- QAR / Aerox / OpenAP fuel: {metrics["qar_fuel_burn_kg"]:.1f} / {metrics["aerox_fuel_burn_kg"]:.1f} / {metrics["openap_fuel_burn_kg"]:.1f} kg
- Any controller limiter active: {100 * metrics["any_control_limited_fraction"]:.1f}%

The final altitude error is diagnostic only because the recording ends during
descent while the active selected altitude is still below the aircraft.
"""


def main() -> None:
    args = parse_args()
    qar, timeline, registration, recording = _load_timeline(args)
    indices = _load_indices(args.adaptive_selection, len(qar.reference.time_s))
    route = _lateral_route(qar, indices)
    environment = _environment(qar, route, indices)
    elapsed_time = timeline["elapsed_time_s"].to_numpy(dtype=float)
    altitude_intent = SelectedAltitudeIntent.from_samples(
        elapsed_time,
        timeline["selected_altitude_ft"].to_numpy(dtype=float) * FT_TO_M,
    )
    speed_intent = SelectedSpeedIntent.from_samples(
        elapsed_time,
        timeline["selected_speed_kts"].to_numpy(dtype=float) * KNOT_TO_MPS,
        timeline["selected_mach"].to_numpy(dtype=float),
    )
    guidance = SelectedAltitudeGuidance(
        lateral_guidance=SpatialPathGuidance(
            route,
            lookahead_time_s=45.0,
            minimum_lookahead_m=2_500.0,
            maximum_lookahead_m=10_000.0,
            maximum_heading_command_rate_rad_s=math.radians(1.0),
        ),
        altitude_intent=altitude_intent,
        speed_intent=speed_intent,
        maximum_climb_angle_rad=math.radians(1.5),
        maximum_descent_angle_rad=math.radians(3.0),
    )
    plant = _build_plant(args)
    maximum_thrust = _load_climb_thrust_ceiling(args)
    minimum_thrust = _load_minimum_thrust(args.descent_thrust_parameters)
    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=_controller(),
        guidance=guidance,
        maximum_time_s=float(qar.reference.time_s[-1]),
        static_temperature_k=float(qar.static_temperature_k[0]),
        initial_mass_kg=qar.initial_mass_kg,
        initial_true_airspeed_mps=float(qar.reference.true_airspeed_mps[0]),
        initial_flight_path_angle_rad=float(qar.reference.flight_path_angle_rad[0]),
        initial_heading_rad=float(qar.reference.heading_rad[0]),
        initial_pressure_altitude_m=float(qar.reference.pressure_altitude_m[0]),
        initial_north_position_m=float(qar.north_position_m[0]),
        initial_east_position_m=float(qar.east_position_m[0]),
        integration_rate_hz=INTEGRATION_RATE_HZ,
        environment_profile=environment,
        maximum_thrust_model=maximum_thrust,
        minimum_thrust_model=minimum_thrust,
        terminate_at_destination=False,
    )
    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    openap_fuel_flow = np.asarray(
        fuel_model.prior.fuel_flow_kg_s(result.trajectory.net_thrust_n), dtype=float
    )
    history, errors, metrics = _history_and_metrics(
        qar=qar,
        timeline=timeline,
        route=route,
        result=result,
        openap_fuel_flow_kg_s=openap_fuel_flow,
    )
    document = _document(
        registration=registration,
        recording=recording,
        qar=qar,
        indices=indices,
        altitude_intent=altitude_intent,
        speed_intent=speed_intent,
        guidance=guidance,
        metrics=metrics,
    )
    for path in (args.history, args.errors, args.parameters, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history, index=False)
    errors.to_csv(args.errors, index=False)
    args.parameters.write_text(
        yaml.safe_dump(native_value(document), sort_keys=False, width=100)
    )
    args.report.write_text(_report(document))
    _plot_all(
        qar=qar,
        timeline=timeline,
        route=route,
        result=result,
        history=history,
        errors=errors,
        figure_dir=args.figure_dir,
    )
    print(
        yaml.safe_dump(
            native_value(
                {
                    "recording": document["source"]["recording"],
                    "metrics": metrics,
                    "parameters": str(args.parameters),
                }
            ),
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
