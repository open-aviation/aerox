#!/usr/bin/env python3
"""Generate the long A320 vertical trajectory from intent, without state anchors."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero
from aerox.fms import (
    CASMachClimbSchedule,
    CASMachDescentSchedule,
    CruiseSpeedStrategy,
    FlightIntent,
    ModeConditionedSpeedIntent,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    Waypoint,
    plan_continuous_speed_mass_profile,
    plan_feasible_profile,
    plan_top_of_climb,
    plan_top_of_descent,
)
from aerox.guidance import SpatialPathGuidance
from aerox.modeling import PointMassProfilePerformance
from aerox.models.atmosphere import (
    calibrated_airspeed_from_true_airspeed,
    mach_from_true_airspeed,
    true_airspeed_from_mach,
)
from aerox.models.thrust import PressureScaledMinimumThrust
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.generate_a320_comparison_trajectories import _controller
from tools.replay_a320_point_mass import _read_fuel_model
from tools.validate_a320_fms_long_range_qar import _history, _load_holdout
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    GATE_LIMITS,
    INTEGRATION_RATE_HZ,
    _build_plant,
    _evaluate,
    _gates,
    _initial_trim_control,
    _style_axis,
)

UPDATED = "2026-08-08"
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
CLIMB_ENERGY_CAPACITY_FRACTION = 0.8
DESCENT_ENERGY_CAPACITY_FRACTION = 0.95
BOUNDARY_SPEED_BLEND_DISTANCE_M = 25_000.0
CRUISE_SPEED_TRANSITION_DISTANCE_M = 25_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument("--speed-intent-parameters", type=Path)
    parser.add_argument("--speed-intent-prior-profile", type=Path)
    parser.add_argument("--speed-intent-prior-history", type=Path)
    parser.add_argument("--continuous-speed-mass-plan", action="store_true")
    parser.add_argument(
        "--cruise-strategy",
        choices=("phase_conditioned", "commanded_mach"),
        default="phase_conditioned",
    )
    parser.add_argument("--commanded-cruise-mach", type=float)
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "adaptive-selection": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
        "intent-parameters": "docs/fms/a320-fms-qar-intent-channel-audit.yml",
        "descent-thrust-parameters": "docs/modeling/a320-descent-planning-thrust.yml",
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
        "baseline-parameters": "docs/fms/a320-fms-advisory-qar-intent.yml",
        "selector-timeline": "docs/fms/a320-fms-qar-intent-channel-timeline.csv",
        "errors": "docs/fms/a320-fms-vertical-intent-errors.csv",
        "history": "docs/fms/a320-fms-vertical-intent-history.csv",
        "profile": "docs/fms/a320-fms-vertical-intent-profile.csv",
        "constraints": "docs/fms/a320-fms-vertical-intent-constraints.csv",
        "parameters": "docs/fms/a320-fms-vertical-intent.yml",
        "report": "docs/fms/a320-fms-vertical-intent.md",
        "figure-prefix": "docs/fms/figures/a320-fms-vertical-intent-comparison",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _load_mode_conditioned_speed_intent(
    path: Path,
) -> tuple[ModeConditionedSpeedIntent, dict[str, Any]]:
    document = yaml.safe_load(path.read_text())
    if not document["decision"]["release_for_fms_integration"]:
        raise ValueError("mode-conditioned speed intent is not released")
    model = document["model"]
    bounds = model["bounds"]
    return (
        ModeConditionedSpeedIntent(
            structure=str(model["structure"]),
            cas_coefficients=np.asarray(model["cas_coefficients"], dtype=float),
            mach_coefficients=np.asarray(model["mach_coefficients"], dtype=float),
            minimum_cas_kts=float(bounds["cas_kts"][0]),
            maximum_cas_kts=float(bounds["cas_kts"][1]),
            minimum_mach=float(bounds["mach"][0]),
            maximum_mach=float(bounds["mach"][1]),
            minimum_crossover_altitude_ft=float(bounds["crossover_altitude_ft"][0]),
            maximum_crossover_altitude_ft=float(bounds["crossover_altitude_ft"][1]),
        ),
        document,
    )


def _mode_conditioned_schedule(
    *,
    model: ModeConditionedSpeedIntent,
    intent_metrics: dict[str, Any],
    planning_mass_kg: float | np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate constant phase schedule values without using aircraft response."""

    original = intent_metrics["schedule"]
    cruise_altitude_ft = float(intent_metrics["cruise_selected_altitude_ft"])
    labels = (
        "climb_cas",
        "climb_mach",
        "cruise_mach",
        "descent_mach",
        "descent_cas",
        "low_altitude_cas",
    )
    altitude_ft = np.asarray(
        [
            15_000.0,
            cruise_altitude_ft,
            cruise_altitude_ft,
            cruise_altitude_ft,
            20_000.0,
            10_000.0,
        ]
    )
    selected_cas_kts = np.asarray(
        [
            original["climb_selected_cas_kts"],
            original["climb_selected_cas_kts"],
            original["climb_selected_cas_kts"],
            original["descent_selected_cas_kts"],
            original["descent_selected_cas_kts"],
            original["low_altitude_selected_cas_kts"],
        ],
        dtype=float,
    )
    selected_mach = np.asarray(
        [
            original["climb_selected_mach"],
            original["climb_selected_mach"],
            original["cruise_selected_mach"],
            original["descent_selected_mach"],
            original["descent_selected_mach"],
            original["descent_selected_mach"],
        ],
        dtype=float,
    )
    phase = np.asarray(["CLIMB", "CLIMB", "LEVEL", "DESCENT", "DESCENT", "DESCENT"])
    planning_mass = np.broadcast_to(
        np.asarray(planning_mass_kg, dtype=float), (len(labels),)
    )
    if np.any(~np.isfinite(planning_mass)) or np.any(planning_mass <= 0.0):
        raise ValueError("speed-intent planning masses must be finite and positive")
    prediction = model.predict(
        selected_cas_kts=selected_cas_kts,
        selected_mach=selected_mach,
        pressure_altitude_ft=altitude_ft,
        mass_kg=planning_mass,
        phase=phase,
    )
    descent_cas_kts = float(prediction.cas_target_kts[4])
    raw_low_altitude_cas_kts = float(prediction.cas_target_kts[5])
    applied_low_altitude_cas_kts = min(raw_low_altitude_cas_kts, descent_cas_kts)
    schedule = dict(original)
    schedule.update(
        {
            "climb_observed_cas_kts": float(prediction.cas_target_kts[0]),
            "climb_observed_mach": float(prediction.mach_target[1]),
            "cruise_observed_mach": float(prediction.mach_target[2]),
            "descent_observed_mach": float(prediction.mach_target[3]),
            "descent_observed_cas_kts": descent_cas_kts,
            "low_altitude_observed_cas_kts": applied_low_altitude_cas_kts,
        }
    )
    evaluation = {
        label: {
            "phase": str(phase[index]),
            "pressure_altitude_ft": float(altitude_ft[index]),
            "planning_mass_kg": float(planning_mass[index]),
            "selected_cas_evidence_kts": float(selected_cas_kts[index]),
            "selected_mach_evidence": float(selected_mach[index]),
            "predicted_cas_kts": float(prediction.cas_target_kts[index]),
            "predicted_mach": float(prediction.mach_target[index]),
            "guarded_crossover_altitude_ft": float(
                prediction.crossover_altitude_ft[index]
            ),
        }
        for index, label in enumerate(labels)
    }
    evaluation["schedule_consistency"] = {
        "rule": "low-altitude CAS must not exceed descent CAS",
        "raw_low_altitude_cas_kts": raw_low_altitude_cas_kts,
        "applied_low_altitude_cas_kts": applied_low_altitude_cas_kts,
        "adjustment_kts": applied_low_altitude_cas_kts - raw_low_altitude_cas_kts,
        "derived_from_QAR_response": False,
    }
    return schedule, evaluation


def _speed_intent_planning_masses(
    *,
    profile_path: Path,
    history_path: Path,
) -> np.ndarray:
    """Interpolate first-pass predicted mass at declared second-pass points."""

    profile = pd.read_csv(profile_path)
    history = pd.read_csv(history_path)
    required_profile = {
        "predicted_elapsed_time_s",
        "pressure_altitude_m",
        "phase",
    }
    required_history = {"elapsed_time_s", "mass_kg"}
    if not required_profile <= set(profile) or not required_history <= set(history):
        raise ValueError("prior profile/history lacks speed-intent mass columns")

    def phase_time(phase: str, altitude_ft: float | None, location: str) -> float:
        subset = profile[profile["phase"] == phase]
        if subset.empty:
            raise ValueError(f"prior profile has no {phase} phase")
        if altitude_ft is not None:
            target_m = altitude_ft * FT_TO_M
            index = (subset["pressure_altitude_m"] - target_m).abs().idxmin()
            return float(profile.loc[index, "predicted_elapsed_time_s"])
        if location == "start":
            return float(subset["predicted_elapsed_time_s"].min())
        if location == "end":
            return float(subset["predicted_elapsed_time_s"].max())
        return float(subset["predicted_elapsed_time_s"].median())

    evaluation_time_s = np.asarray(
        [
            phase_time("climb", 15_000.0, "middle"),
            phase_time("climb", None, "end"),
            phase_time("cruise", None, "middle"),
            phase_time("descent", None, "start"),
            phase_time("descent", 20_000.0, "middle"),
            phase_time("descent", 10_000.0, "middle"),
        ]
    )
    return np.interp(
        evaluation_time_s,
        history["elapsed_time_s"].to_numpy(dtype=float),
        history["mass_kg"].to_numpy(dtype=float),
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


def _geometry_route(qar: Any, indices: np.ndarray) -> SpatialTrajectory:
    initial_altitude = float(qar.reference.pressure_altitude_m[0])
    initial_speed = float(qar.reference.true_airspeed_mps[0])
    waypoints = tuple(
        Waypoint(
            north_m=float(qar.north_position_m[index]),
            east_m=float(qar.east_position_m[index]),
            altitude_m=initial_altitude,
            true_airspeed_mps=initial_speed,
            name=f"G{number}",
            acceptance_radius_m=1_500.0 if index == indices[-1] else 4_000.0,
        )
        for number, index in enumerate(indices)
    )
    return SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=waypoints,
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


def _compose_energy_profile(
    *,
    geometry: SpatialTrajectory,
    toc: Any,
    tod: Any,
    initial_speed_mps: float,
    cruise_speed_mps: float,
    terminal_speed_mps: float,
    spacing_m: float = 500.0,
) -> SpatialTrajectory:
    climb_s = toc.distance_from_origin_m.copy()
    climb_s *= toc.top_of_climb_along_path_m / climb_s[-1]
    climb_altitude = toc.altitude_m.copy()
    climb_speed = toc.true_airspeed_mps.copy()
    initial_blend = np.clip(climb_s / BOUNDARY_SPEED_BLEND_DISTANCE_M, 0.0, 1.0)
    climb_speed = (
        1.0 - initial_blend
    ) * initial_speed_mps + initial_blend * climb_speed
    climb_speed[0] = initial_speed_mps

    descent_distance = tod.distance_to_destination_m.copy()
    descent_distance *= tod.descent_ground_distance_m / descent_distance[-1]
    descent_s = (geometry.total_distance_m - descent_distance)[::-1]
    descent_altitude = tod.altitude_m[::-1]
    descent_speed = tod.true_airspeed_mps[::-1].copy()
    terminal_blend = np.clip(
        (descent_s - (geometry.total_distance_m - BOUNDARY_SPEED_BLEND_DISTANCE_M))
        / BOUNDARY_SPEED_BLEND_DISTANCE_M,
        0.0,
        1.0,
    )
    descent_speed = (
        1.0 - terminal_blend
    ) * descent_speed + terminal_blend * terminal_speed_mps
    descent_speed[-1] = terminal_speed_mps

    cruise_transition_start = (
        toc.top_of_climb_along_path_m + CRUISE_SPEED_TRANSITION_DISTANCE_M
    )
    descent_transition_start = (
        tod.top_of_descent_along_path_m - CRUISE_SPEED_TRANSITION_DISTANCE_M
    )
    if cruise_transition_start >= descent_transition_start:
        raise ValueError("cruise segment is too short for speed transitions")
    raw_s = np.r_[
        climb_s,
        cruise_transition_start,
        descent_transition_start,
        tod.top_of_descent_along_path_m,
        descent_s[1:],
    ]
    raw_altitude = np.r_[
        climb_altitude,
        climb_altitude[-1],
        climb_altitude[-1],
        climb_altitude[-1],
        descent_altitude[1:],
    ]
    raw_speed = np.r_[
        climb_speed,
        cruise_speed_mps,
        cruise_speed_mps,
        descent_speed[0],
        descent_speed[1:],
    ]
    support = np.unique(
        np.r_[
            np.arange(0.0, geometry.total_distance_m, spacing_m),
            geometry.waypoint_along_path_m,
            raw_s,
            geometry.total_distance_m,
        ]
    )
    altitude = np.interp(support, raw_s, raw_altitude)
    speed = np.interp(support, raw_s, raw_speed)
    mapped_waypoints = tuple(
        replace(
            waypoint,
            altitude_m=float(np.interp(distance, support, altitude)),
            true_airspeed_mps=float(np.interp(distance, support, speed)),
        )
        for waypoint, distance in zip(
            geometry.intent.waypoints,
            geometry.waypoint_along_path_m,
            strict=True,
        )
    )
    mapped = SpatialTrajectory.from_intent(
        replace(geometry.intent, waypoints=mapped_waypoints)
    )
    return mapped.with_profile(
        along_path_m=support,
        altitude_m=altitude,
        true_airspeed_mps=speed,
    )


def _planned_profile_table(
    route: SpatialTrajectory,
    toc_s: float,
    tod_s: float,
    environment: SpatialEnvironmentProfile,
) -> pd.DataFrame:
    sampled_route = route.sample_grid(spacing_m=500.0)
    s = sampled_route.along_path_m
    north = sampled_route.north_m
    east = sampled_route.east_m
    altitude = sampled_route.altitude_m
    speed = sampled_route.true_airspeed_mps
    time = np.r_[0.0, np.cumsum(2.0 * np.diff(s) / (speed[:-1] + speed[1:]))]
    forecast_temperature = np.array(
        [environment.sample(float(distance)).static_temperature_k for distance in s]
    )
    calibrated_airspeed = np.asarray(
        calibrated_airspeed_from_true_airspeed(speed, altitude, forecast_temperature),
        dtype=float,
    )
    mach = np.asarray(
        mach_from_true_airspeed(speed, altitude, forecast_temperature), dtype=float
    )
    phase = np.full(len(s), "cruise", dtype="U7")
    phase[s < toc_s] = "climb"
    phase[s >= tod_s] = "descent"
    return pd.DataFrame(
        {
            "predicted_elapsed_time_s": time,
            "along_path_m": s,
            "north_m": north,
            "east_m": east,
            "pressure_altitude_m": altitude,
            "true_airspeed_mps": speed,
            "calibrated_airspeed_mps": calibrated_airspeed,
            "mach": mach,
            "phase": phase,
        }
    )


def _comparison_table(qar: Any, result: Any) -> pd.DataFrame:
    time = qar.reference.time_s
    trajectory = result.trajectory
    common = time <= trajectory.time_s[-1]
    qar_altitude = qar.reference.pressure_altitude_m[common]
    qar_speed = qar.reference.true_airspeed_mps[common]
    generated_altitude = np.interp(
        time[common], trajectory.time_s, trajectory.pressure_altitude_m
    )
    generated_speed = np.interp(
        time[common], trajectory.time_s, trajectory.true_airspeed_mps
    )
    generated_temperature = np.interp(
        time[common], trajectory.time_s, result.static_temperature_k
    )
    return pd.DataFrame(
        {
            "elapsed_time_s": time[common],
            "qar_north_m": qar.north_position_m[common],
            "qar_east_m": qar.east_position_m[common],
            "qar_pressure_altitude_m": qar_altitude,
            "qar_true_airspeed_mps": qar_speed,
            "qar_calibrated_airspeed_mps": calibrated_airspeed_from_true_airspeed(
                qar_speed, qar_altitude, qar.static_temperature_k[common]
            ),
            "qar_mach": mach_from_true_airspeed(
                qar_speed, qar_altitude, qar.static_temperature_k[common]
            ),
            "generated_north_m": np.interp(
                time[common], trajectory.time_s, trajectory.north_position_m
            ),
            "generated_east_m": np.interp(
                time[common], trajectory.time_s, trajectory.east_position_m
            ),
            "generated_pressure_altitude_m": generated_altitude,
            "generated_true_airspeed_mps": generated_speed,
            "generated_calibrated_airspeed_mps": calibrated_airspeed_from_true_airspeed(
                generated_speed,
                generated_altitude,
                generated_temperature,
            ),
            "generated_mach": mach_from_true_airspeed(
                generated_speed,
                generated_altitude,
                generated_temperature,
            ),
        }
    )


def _plot(
    *,
    qar: Any,
    route: SpatialTrajectory,
    result: Any,
    profile: pd.DataFrame,
    intent_document: dict[str, Any],
    toc_s: float,
    tod_s: float,
    prefix: Path,
    title: str = "A320 trajectory from anchor-free vertical intent",
) -> None:
    import matplotlib.pyplot as plt

    black, blue, orange, green, magenta = (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
    )
    trajectory = result.trajectory
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    axes[0, 0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=black,
        linewidth=1.0,
        label="QAR reference",
    )
    axes[0, 0].plot(
        [point.east_m / 1_000.0 for point in route.intent.waypoints],
        [point.north_m / 1_000.0 for point in route.intent.waypoints],
        color=green,
        linestyle=":",
        marker=".",
        markersize=3,
        label="lateral intent",
    )
    axes[0, 0].plot(
        trajectory.east_position_m / 1_000.0,
        trajectory.north_position_m / 1_000.0,
        color=orange,
        linestyle="--",
        linewidth=1.1,
        label="generated",
    )
    axes[0, 0].set(xlabel="Local east (km)", ylabel="Local north (km)")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(frameon=False, fontsize=6.7)

    axes[0, 1].plot(
        qar.reference.time_s / 60.0,
        qar.reference.pressure_altitude_m / FT_TO_M / 1_000.0,
        color=black,
        label="QAR reference",
    )
    axes[0, 1].plot(
        profile["predicted_elapsed_time_s"] / 60.0,
        profile["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=green,
        linestyle=":",
        label="intent plan",
    )
    axes[0, 1].plot(
        trajectory.time_s / 60.0,
        trajectory.pressure_altitude_m / FT_TO_M / 1_000.0,
        color=orange,
        linestyle="--",
        label="generated",
    )
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (kft)")

    cruise = intent_document["metrics"]["cruise_selected_altitude_ft"] / 1_000.0
    axes[0, 1].axhline(
        cruise,
        color=magenta,
        linestyle="-.",
        linewidth=1.0,
        label="selected cruise target",
    )
    axes[0, 1].legend(frameon=False, fontsize=6.3)

    qar_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            qar.reference.pressure_altitude_m,
            qar.static_temperature_k,
        ),
        dtype=float,
    )
    qar_mach = np.asarray(
        mach_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            qar.reference.pressure_altitude_m,
            qar.static_temperature_k,
        ),
        dtype=float,
    )
    generated_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            trajectory.true_airspeed_mps,
            trajectory.pressure_altitude_m,
            result.static_temperature_k,
        ),
        dtype=float,
    )
    generated_mach = np.asarray(
        mach_from_true_airspeed(
            trajectory.true_airspeed_mps,
            trajectory.pressure_altitude_m,
            result.static_temperature_k,
        ),
        dtype=float,
    )
    axes[0, 2].plot(
        qar.reference.time_s / 60.0,
        qar_cas / KNOT_TO_MPS,
        color=black,
        label="QAR reference",
    )
    axes[0, 2].plot(
        profile["predicted_elapsed_time_s"] / 60.0,
        profile["calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=green,
        linestyle=":",
        label="FMS intent profile",
    )
    axes[0, 2].plot(
        trajectory.time_s / 60.0,
        generated_cas / KNOT_TO_MPS,
        color=orange,
        linestyle="--",
        label="generated",
    )
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Calibrated airspeed (kt)")
    axes[0, 2].legend(frameon=False, fontsize=6.7)

    qar_s = np.r_[
        0.0,
        np.cumsum(
            np.hypot(np.diff(qar.north_position_m), np.diff(qar.east_position_m))
        ),
    ]
    axes[1, 0].plot(
        qar.reference.time_s / 60.0,
        qar_mach,
        color=black,
        label="QAR reference",
    )
    axes[1, 0].plot(
        profile["predicted_elapsed_time_s"] / 60.0,
        profile["mach"],
        color=green,
        linestyle=":",
        label="FMS intent profile",
    )
    axes[1, 0].plot(
        trajectory.time_s / 60.0,
        generated_mach,
        color=orange,
        linestyle="--",
        label="generated",
    )
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="Mach number")
    axes[1, 0].legend(frameon=False, fontsize=6.7)

    axes[1, 1].plot(
        qar_s / 1_000.0,
        qar.reference.pressure_altitude_m / FT_TO_M / 1_000.0,
        color=black,
        label="QAR reference",
    )
    axes[1, 1].plot(
        profile["along_path_m"] / 1_000.0,
        profile["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=green,
        linestyle=":",
        label="intent plan",
    )
    axes[1, 1].plot(
        result.progress_m / 1_000.0,
        trajectory.pressure_altitude_m / FT_TO_M / 1_000.0,
        color=orange,
        linestyle="--",
        label="generated",
    )
    axes[1, 1].axvline(
        toc_s / 1_000.0, color=blue, linestyle="-.", label="calculated TOC"
    )
    axes[1, 1].axvline(
        tod_s / 1_000.0, color=magenta, linestyle="-.", label="calculated TOD"
    )
    axes[1, 1].set(xlabel="Along-path distance (km)", ylabel="Pressure altitude (kft)")
    axes[1, 1].legend(frameon=False, fontsize=6.5)

    common = qar.reference.time_s <= trajectory.time_s[-1]
    generated_altitude = np.interp(
        qar.reference.time_s[common], trajectory.time_s, trajectory.pressure_altitude_m
    )
    axes[1, 2].plot(
        qar.reference.time_s[common] / 60.0,
        (generated_altitude - qar.reference.pressure_altitude_m[common]) / FT_TO_M,
        color=blue,
        label="altitude error (ft)",
    )
    speed_axis = axes[1, 2].twinx()
    generated_cas_common = np.interp(
        qar.reference.time_s[common], trajectory.time_s, generated_cas
    )
    speed_axis.plot(
        qar.reference.time_s[common] / 60.0,
        (generated_cas_common - qar_cas[common]) / KNOT_TO_MPS,
        color=orange,
        linestyle="--",
        label="CAS error (kt)",
    )
    axes[1, 2].axhline(0.0, color=black, linewidth=0.7)
    axes[1, 2].set(xlabel="Elapsed time (min)", ylabel="Altitude error (ft)")
    speed_axis.set_ylabel("Calibrated airspeed error (kt)")
    lines = axes[1, 2].lines[:1] + speed_axis.lines
    axes[1, 2].legend(
        lines, [line.get_label() for line in lines], frameon=False, fontsize=6.7
    )

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    figure.suptitle(title, fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cruise_strategy == "commanded_mach":
        if args.commanded_cruise_mach is None:
            raise ValueError("commanded_mach strategy requires --commanded-cruise-mach")
        if not args.continuous_speed_mass_plan:
            raise ValueError("explicit cruise strategy requires continuous planning")
        cruise_strategy = CruiseSpeedStrategy(
            kind="commanded_mach", commanded_mach=args.commanded_cruise_mach
        )
    else:
        if args.commanded_cruise_mach is not None:
            raise ValueError(
                "--commanded-cruise-mach is only valid with commanded_mach strategy"
            )
        cruise_strategy = CruiseSpeedStrategy()
    adaptive = yaml.safe_load(args.adaptive_selection.read_text())
    if adaptive["status"] != "selected-before-adaptive-rollout":
        raise ValueError("lateral pseudo-waypoint selection is not frozen")
    indices = np.asarray(adaptive["anchor_indices"], dtype=int)
    intent_document = yaml.safe_load(args.intent_parameters.read_text())
    intent_metrics = intent_document["metrics"]

    qar, registration, recording = _load_holdout(args)
    speed_intent_evaluation: dict[str, Any] | None = None
    speed_intent_document: dict[str, Any] | None = None
    if args.speed_intent_parameters is None:
        schedule_values = intent_metrics["schedule"]
        speed_schedule_source = (
            "stable flown CAS/Mach medians from the frozen intent audit"
        )
    else:
        speed_intent_model, speed_intent_document = _load_mode_conditioned_speed_intent(
            args.speed_intent_parameters
        )
        prior_paths = (
            args.speed_intent_prior_profile,
            args.speed_intent_prior_history,
        )
        if (prior_paths[0] is None) != (prior_paths[1] is None):
            raise ValueError(
                "speed-intent prior profile and history must be supplied together"
            )
        if prior_paths[0] is None:
            speed_intent_planning_mass = float(qar.initial_mass_kg)
            speed_intent_mass_source = "constant initial mass (first planning pass)"
        else:
            speed_intent_planning_mass = _speed_intent_planning_masses(
                profile_path=prior_paths[0],
                history_path=prior_paths[1],
            )
            speed_intent_mass_source = "phase-point mass interpolated from first-pass Aerox predicted fuel history"
        schedule_values, speed_intent_evaluation = _mode_conditioned_schedule(
            model=speed_intent_model,
            intent_metrics=intent_metrics,
            planning_mass_kg=speed_intent_planning_mass,
        )
        speed_schedule_source = (
            "frozen development-LOFO affine selector surrogate evaluated from "
            "planned phase, altitude, Aerox planning mass, and QAR selector evidence"
        )
    if cruise_strategy.kind == "commanded_mach":
        schedule_values = dict(schedule_values)
        assert cruise_strategy.commanded_mach is not None
        schedule_values["cruise_observed_mach"] = cruise_strategy.commanded_mach
        speed_schedule_source += (
            f"; explicit commanded cruise Mach {cruise_strategy.commanded_mach:.3f} "
            "replaces only the level-flight surrogate target"
        )
    geometry = _geometry_route(qar, indices)
    environment = _environment(qar, geometry, indices)
    plant = _build_plant(args)
    maximum_thrust = _load_climb_thrust_ceiling(args)
    minimum_thrust = _load_minimum_thrust(args.descent_thrust_parameters)
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=maximum_thrust,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
        minimum_thrust_model=minimum_thrust,
    )
    climb_schedule = CASMachClimbSchedule(
        climb_cas_mps=schedule_values["climb_observed_cas_kts"] * KNOT_TO_MPS,
        climb_mach=schedule_values["climb_observed_mach"],
    )
    descent_schedule = CASMachDescentSchedule(
        descent_cas_mps=schedule_values["descent_observed_cas_kts"] * KNOT_TO_MPS,
        descent_mach=schedule_values["descent_observed_mach"],
        low_altitude_cas_mps=schedule_values["low_altitude_observed_cas_kts"]
        * KNOT_TO_MPS,
    )
    cruise_altitude_m = intent_metrics["cruise_selected_altitude_ft"] * FT_TO_M
    temperature_offset = qar.static_temperature_k - aero.temperature(
        qar.reference.pressure_altitude_m
    )
    cruise_state = (
        np.abs(qar.reference.pressure_altitude_m - cruise_altitude_m) <= 250.0 * FT_TO_M
    )
    cruise_indices = np.flatnonzero(cruise_state)
    if len(cruise_indices) < 120:
        raise ValueError("stable cruise is too short to infer forecast temperature")
    first_cruise = int(cruise_indices[0])
    last_cruise = int(cruise_indices[-1])
    phase_temperature_offset_k = {
        "climb": float(np.median(temperature_offset[:first_cruise])),
        "cruise": float(np.median(temperature_offset[cruise_state])),
        "descent": float(np.median(temperature_offset[last_cruise + 1 :])),
    }
    cruise_mach = schedule_values["cruise_observed_mach"]
    cruise_speed_mps = float(
        true_airspeed_from_mach(
            cruise_mach,
            cruise_altitude_m,
            aero.temperature(cruise_altitude_m) + phase_temperature_offset_k["cruise"],
        )
    )
    unreserved_toc = plan_top_of_climb(
        route_distance_m=geometry.total_distance_m,
        initial_altitude_m=float(qar.reference.pressure_altitude_m[0]),
        cruise_altitude_m=cruise_altitude_m,
        schedule=climb_schedule,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        altitude_step_m=50.0,
        static_temperature_offset_k=0.0,
        speed_schedule_temperature_offset_k=phase_temperature_offset_k["climb"],
    )
    toc = plan_top_of_climb(
        route_distance_m=geometry.total_distance_m,
        initial_altitude_m=float(qar.reference.pressure_altitude_m[0]),
        cruise_altitude_m=cruise_altitude_m,
        schedule=climb_schedule,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        altitude_step_m=50.0,
        static_temperature_offset_k=0.0,
        speed_schedule_temperature_offset_k=phase_temperature_offset_k["climb"],
        climb_distance_reserve_m=unreserved_toc.climb_ground_distance_m
        * (1.0 / CLIMB_ENERGY_CAPACITY_FRACTION - 1.0),
    )
    unreserved_tod = plan_top_of_descent(
        route_distance_m=geometry.total_distance_m,
        cruise_altitude_m=cruise_altitude_m,
        destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
        schedule=descent_schedule,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        altitude_step_m=50.0,
        static_temperature_offset_k=phase_temperature_offset_k["descent"],
    )
    tod = plan_top_of_descent(
        route_distance_m=geometry.total_distance_m,
        cruise_altitude_m=cruise_altitude_m,
        destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
        schedule=descent_schedule,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        altitude_step_m=50.0,
        static_temperature_offset_k=phase_temperature_offset_k["descent"],
        descent_distance_reserve_m=unreserved_tod.descent_ground_distance_m
        * (1.0 / DESCENT_ENERGY_CAPACITY_FRACTION - 1.0),
    )
    if not toc.feasible_within_route or not tod.feasible_within_route:
        raise ValueError("calculated climb or descent does not fit within route")
    if toc.top_of_climb_along_path_m >= tod.top_of_descent_along_path_m:
        raise ValueError("calculated climb and descent paths overlap")

    geometry_constraint_s = np.array(
        [
            0.0,
            toc.top_of_climb_along_path_m,
            tod.top_of_descent_along_path_m,
            geometry.total_distance_m,
        ]
    )
    geometry_samples = [
        geometry.sample(float(value)) for value in geometry_constraint_s
    ]
    toc_speed = float(
        climb_schedule.true_airspeed_mps(
            np.array([cruise_altitude_m]), phase_temperature_offset_k["climb"]
        )[0]
    )
    tod_speed = float(
        descent_schedule.true_airspeed_mps(
            np.array([cruise_altitude_m]), phase_temperature_offset_k["descent"]
        )[0]
    )
    constraint_waypoints = tuple(
        Waypoint(
            north_m=sample.north_m,
            east_m=sample.east_m,
            altitude_m=altitude,
            true_airspeed_mps=speed,
            name=name,
            acceptance_radius_m=1_500.0 if name == "TERMINAL" else 4_000.0,
        )
        for sample, altitude, speed, name in zip(
            geometry_samples,
            (
                float(qar.reference.pressure_altitude_m[0]),
                cruise_altitude_m,
                cruise_altitude_m,
                float(qar.reference.pressure_altitude_m[-1]),
            ),
            (
                float(qar.reference.true_airspeed_mps[0]),
                toc_speed,
                tod_speed,
                float(qar.reference.true_airspeed_mps[-1]),
            ),
            ("INITIAL", "CALCULATED-TOC", "CALCULATED-TOD", "TERMINAL"),
            strict=True,
        )
    )
    energy_route = _compose_energy_profile(
        geometry=geometry,
        toc=toc,
        tod=tod,
        initial_speed_mps=float(qar.reference.true_airspeed_mps[0]),
        cruise_speed_mps=cruise_speed_mps,
        terminal_speed_mps=float(qar.reference.true_airspeed_mps[-1]),
    )
    assert energy_route.profile_along_path_m is not None
    assert energy_route.profile_altitude_m is not None
    continuous_plan = None
    waypoint_mass_kg = np.full(len(indices), qar.initial_mass_kg)
    if args.continuous_speed_mass_plan:
        if args.speed_intent_parameters is None:
            raise ValueError(
                "continuous speed/mass planning requires --speed-intent-parameters"
            )
        selector_timeline = pd.read_csv(args.selector_timeline)
        continuous_plan = plan_continuous_speed_mass_profile(
            route=energy_route,
            speed_intent=speed_intent_model,
            selector_along_path_m=selector_timeline["elapsed_time_s"].to_numpy(),
            selected_cas_kts=selector_timeline["selected_speed_kts"].to_numpy(),
            selected_mach=selector_timeline["selected_mach"].to_numpy(),
            top_of_climb_along_path_m=toc.top_of_climb_along_path_m,
            top_of_descent_along_path_m=tod.top_of_descent_along_path_m,
            performance=performance,
            fuel_flow_model=_read_fuel_model(
                args.thrust_parameters, args.fuel_parameters
            ),
            initial_mass_kg=qar.initial_mass_kg,
            environment_profile=environment,
            spacing_m=500.0,
            maximum_iterations=20,
            minimum_cas_target_kts_by_phase={
                "CLIMB": schedule_values["climb_observed_cas_kts"]
            },
            selector_coordinate_kind="elapsed_time",
            cruise_strategy=cruise_strategy,
        )
        energy_route = continuous_plan.trajectory
        waypoint_mass_kg = np.interp(
            energy_route.waypoint_along_path_m,
            continuous_plan.along_path_m,
            continuous_plan.planned_mass_kg,
        )
    final_profile = plan_feasible_profile(
        route=energy_route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(np.median(environment.static_temperature_k)),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=waypoint_mass_kg,
        hard_waypoint_indices=(0, -1),
    )
    profile_gamma = np.arctan(
        np.gradient(final_profile.nominal_altitude_m, final_profile.along_path_m)
    )
    profile_acceleration = final_profile.nominal_true_airspeed_mps * np.gradient(
        final_profile.nominal_true_airspeed_mps, final_profile.along_path_m
    )
    profile_energy_demand = profile_acceleration + 9.80665 * np.sin(profile_gamma)
    maximum_energy_violation = float(
        max(
            0.0,
            np.max(
                profile_energy_demand - final_profile.maximum_specific_excess_force_mps2
            ),
        )
    )
    minimum_energy_violation = float(
        max(
            0.0,
            np.max(
                final_profile.minimum_specific_excess_force_mps2 - profile_energy_demand
            ),
        )
    )
    route = final_profile.trajectory
    guidance = SpatialPathGuidance(
        route,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
    )
    controller = _controller()
    initial_gamma, initial_control, _ = _initial_trim_control(
        plant=plant,
        controller=controller,
        trajectory=route,
        environment=environment,
        initial_mass_kg=qar.initial_mass_kg,
    )
    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=controller,
        guidance=guidance,
        maximum_time_s=float(qar.reference.time_s[-1] + 1_800.0),
        static_temperature_k=float(environment.static_temperature_k[0]),
        initial_mass_kg=qar.initial_mass_kg,
        destination_capture_radius_m=1_500.0,
        integration_rate_hz=INTEGRATION_RATE_HZ,
        environment_profile=environment,
        maximum_thrust_model=maximum_thrust,
        minimum_thrust_model=minimum_thrust,
        initial_flight_path_angle_rad=initial_gamma,
        initial_control=initial_control,
    )
    boundary_indices = np.array([0, len(qar.reference.time_s) - 1], dtype=int)
    metrics, errors, _ = _evaluate(qar, route, boundary_indices, result)
    airdata_comparison = _comparison_table(qar, result)
    metrics["calibrated_airspeed_rmse_mps"] = float(
        np.sqrt(
            np.mean(
                (
                    airdata_comparison["generated_calibrated_airspeed_mps"]
                    - airdata_comparison["qar_calibrated_airspeed_mps"]
                )
                ** 2
            )
        )
    )
    metrics["mach_rmse"] = float(
        np.sqrt(
            np.mean(
                (airdata_comparison["generated_mach"] - airdata_comparison["qar_mach"])
                ** 2
            )
        )
    )
    gates = _gates(metrics, final_profile.feasible)
    baseline = yaml.safe_load(args.baseline_parameters.read_text())
    baseline_metrics = baseline["metrics"]
    profile_table = _planned_profile_table(
        route,
        toc.top_of_climb_along_path_m,
        tod.top_of_descent_along_path_m,
        environment,
    )
    if continuous_plan is not None:
        plan_s = continuous_plan.along_path_m
        profile_s = profile_table["along_path_m"].to_numpy()
        profile_table["desired_true_airspeed_mps"] = np.interp(
            profile_s, plan_s, continuous_plan.desired_true_airspeed_mps
        )
        profile_table["planned_mass_kg"] = np.interp(
            profile_s, plan_s, continuous_plan.planned_mass_kg
        )
        profile_table["predicted_fuel_flow_kg_s"] = np.interp(
            profile_s, plan_s, continuous_plan.predicted_fuel_flow_kg_s
        )
        profile_table["predicted_thrust_n"] = np.interp(
            profile_s, plan_s, continuous_plan.predicted_thrust_n
        )
        profile_table["cas_target_kts"] = np.interp(
            profile_s, plan_s, continuous_plan.cas_target_kts
        )
        profile_table["active_cas_equivalent_target_kts"] = np.interp(
            profile_s,
            plan_s,
            continuous_plan.active_cas_equivalent_target_kts,
        )
        profile_table["mach_target"] = np.interp(
            profile_s, plan_s, continuous_plan.mach_target
        )
    constraints = pd.DataFrame(
        {
            "name": [point.name for point in constraint_waypoints],
            "geometry_along_path_m": geometry_constraint_s,
            "state_route_along_path_m": geometry_constraint_s,
            "pressure_altitude_m": [point.altitude_m for point in constraint_waypoints],
            "true_airspeed_mps": [
                point.true_airspeed_mps for point in constraint_waypoints
            ],
            "source": (
                "QAR initial boundary",
                "energy-calculated from selected cruise target",
                "energy-calculated from selected cruise target and terminal",
                "QAR terminal boundary",
            ),
        }
    )
    passed = all(gate["passed"] for gate in gates.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "passed" if passed else "held",
        "source": {
            "recording": f"{registration}/{recording}",
            "post_holdout_diagnostic": True,
            "clean_reference_starts_mid_climb": True,
        },
        "intent_boundary": {
            "lateral_geometry_nodes": len(indices),
            "lateral_geometry_source": "frozen turn-aware pseudo-waypoints",
            "periodic_qar_altitude_speed_anchors": 0,
            "qar_state_boundaries": ["initial", "terminal"],
            "intermediate_qar_states_used_by_planner_or_guidance": False,
            "observed_heading_or_recorded_controls_used_by_guidance": False,
            "selected_cruise_altitude_ft": intent_metrics[
                "cruise_selected_altitude_ft"
            ],
            "inferred_climb_schedule": {
                "cas_kts": schedule_values["climb_observed_cas_kts"],
                "mach": schedule_values["climb_observed_mach"],
            },
            "inferred_descent_schedule": {
                "cas_kts": schedule_values["descent_observed_cas_kts"],
                "mach": schedule_values["descent_observed_mach"],
                "low_altitude_cas_kts": schedule_values[
                    "low_altitude_observed_cas_kts"
                ],
            },
            "inferred_cruise_mach": cruise_mach,
            "cruise_speed_strategy": {
                "kind": cruise_strategy.kind,
                "commanded_mach": cruise_strategy.commanded_mach,
                "released": cruise_strategy.kind != "economy",
                "actual_QAR_speed_used_to_form_target": False,
            },
            "cruise_speed_transition_distance_km": (
                CRUISE_SPEED_TRANSITION_DISTANCE_M / 1_000.0
            ),
            "speed_schedule_source": speed_schedule_source,
            "mode_conditioned_speed_intent": (
                {
                    "parameters": str(args.speed_intent_parameters),
                    "structure": speed_intent_document["model"]["structure"],
                    "planning_mass_source": speed_intent_mass_source,
                    "prior_profile": (
                        str(args.speed_intent_prior_profile)
                        if args.speed_intent_prior_profile is not None
                        else None
                    ),
                    "prior_history": (
                        str(args.speed_intent_prior_history)
                        if args.speed_intent_prior_history is not None
                        else None
                    ),
                    "evaluation_points": speed_intent_evaluation,
                    "actual_CAS_or_Mach_used_to_form_schedule": False,
                    "inferred_speed_mode_used_to_form_schedule": False,
                    "continuous_spatial_evaluation": continuous_plan is not None,
                    "selector_timeline": (
                        str(args.selector_timeline)
                        if continuous_plan is not None
                        else None
                    ),
                    "selector_spatialization": (
                        "recorded intent timeline mapped through Aerox predicted elapsed time"
                        if continuous_plan is not None
                        else None
                    ),
                }
                if speed_intent_document is not None
                else None
            ),
            "selected_speed_fields_claimed_as_active_targets": False,
            "discarded_ALT_SEL_steps_as_spatial_constraints": True,
            "reason": "several selected values lie below the simultaneous climbing state or above the simultaneous descending state",
        },
        "energy_plan": {
            "calculated_top_of_climb_along_path_km": toc.top_of_climb_along_path_m
            / 1_000.0,
            "unreserved_top_of_climb_along_path_km": unreserved_toc.top_of_climb_along_path_m
            / 1_000.0,
            "climb_energy_capacity_fraction": CLIMB_ENERGY_CAPACITY_FRACTION,
            "calculated_top_of_descent_along_path_km": tod.top_of_descent_along_path_m
            / 1_000.0,
            "unreserved_top_of_descent_along_path_km": unreserved_tod.top_of_descent_along_path_m
            / 1_000.0,
            "calculated_descent_distance_km": tod.descent_ground_distance_m / 1_000.0,
            "unreserved_descent_distance_km": unreserved_tod.descent_ground_distance_m
            / 1_000.0,
            "descent_energy_capacity_fraction": DESCENT_ENERGY_CAPACITY_FRACTION,
            "boundary_speed_blend_distance_km": BOUNDARY_SPEED_BLEND_DISTANCE_M
            / 1_000.0,
            "hard_vertical_constraint_waypoint_indices": [0, len(indices) - 1],
            "maximum_energy_capacity_violation_mps2": maximum_energy_violation,
            "minimum_energy_capacity_violation_mps2": minimum_energy_violation,
            "TOC_or_TOD_copied_from_QAR": False,
            "planning_wind_mps": 0.0,
            "planning_temperature": (
                "released CLB capacity under ISA; speed schedule and descent "
                "performance use recorded forecast-surrogate phase medians"
            ),
            "planning_static_temperature_offset_k": phase_temperature_offset_k,
            "planning_mass": (
                "iterated from calibrated Aerox predicted fuel flow along the spatial plan"
                if continuous_plan is not None
                else "constant initial mass; no periodic QAR mass anchors"
            ),
            "maximum_thrust": "released demonstrated CLB capacity",
            "minimum_thrust": "released effective pressure-scaled descent planning prior",
            "analytic_energy_profile_constructed": True,
            "geometry_mapped_profile_feasible": final_profile.feasible,
            "violations": list(final_profile.violations),
            "continuous_speed_mass_plan": (
                {
                    "iteration_count": continuous_plan.iteration_count,
                    "maximum_mass_update_kg": continuous_plan.maximum_mass_update_kg,
                    "predicted_fuel_burn_kg": qar.initial_mass_kg
                    - continuous_plan.planned_mass_kg[-1],
                    "maximum_energy_violation_mps2": continuous_plan.maximum_energy_violation_mps2,
                    "minimum_energy_violation_mps2": continuous_plan.minimum_energy_violation_mps2,
                    "QAR_response_used_to_form_targets": False,
                    "QAR_fuel_or_mass_used_in_iteration": False,
                    "minimum_climb_CAS_target_kts": schedule_values[
                        "climb_observed_cas_kts"
                    ],
                    "minimum_climb_CAS_source": "frozen mode-conditioned phase intent evaluation",
                    "energy_numerical_margin_mps2": 0.002,
                    "cruise_strategy": continuous_plan.cruise_strategy,
                    "commanded_cruise_mach": cruise_strategy.commanded_mach,
                }
                if continuous_plan is not None
                else None
            ),
        },
        "environment": {
            "recorded_temperature_and_wind_used_as_forecast_surrogate": True,
            "profile_feasibility_checked_against_ISA_planning_assumption": True,
            "environment_nodes": len(indices),
            "environment_values_are_altitude_or_speed_constraints": False,
        },
        "numerics": {
            "command_rate_hz": 1,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "profile_spacing_m": 500.0,
        },
        "metrics": metrics,
        "comparison_to_baseline": {
            key: {
                "baseline": float(baseline_metrics[key]),
                "current": float(metrics[key]),
                "change": float(metrics[key] - baseline_metrics[key]),
            }
            for key in (
                "time_aligned_horizontal_rmse_m",
                "altitude_rmse_m",
                "speed_rmse_mps",
                "arrival_time_error_s",
                "any_control_limited_fraction",
            )
        },
        "gate_limits": GATE_LIMITS,
        "gates": gates,
        "decision": {
            "anchor_free_vertical_intent_gate_passed": passed,
            "trajectory_generated_without_periodic_QAR_state_anchors": True,
            "airbus_FMS_autopilot_or_FADEC_identified": False,
            "mode_conditioned_speed_intent_integrated": (
                speed_intent_document is not None
            ),
            "continuous_speed_and_predicted_mass_integrated": (
                continuous_plan is not None
            ),
            "new_independent_holdout_claimed": False,
        },
        "artifacts": {
            "errors": str(args.errors),
            "trajectory_comparison": str(
                args.errors.with_name(f"{args.errors.stem}-trajectory.csv")
            ),
            "history": str(args.history),
            "profile": str(args.profile),
            "constraints": str(args.constraints),
            "figure": f"{args.figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/validate_a320_fms_vertical_intent.py",
        },
    }
    held = [name for name, gate in gates.items() if not gate["passed"]]
    if speed_intent_document is None:
        report_title = "A320 FMS trajectory from anchor-free vertical intent"
        figure_title = "A320 trajectory from anchor-free vertical intent"
        schedule_explanation = f"""Because active selected versus managed speed mode is
unobservable, the speed fields are retained only as candidate evidence. The
coarse flown schedule is inferred from stable phase medians: climb
{schedule_values["climb_observed_cas_kts"]:.0f} kt / Mach
{schedule_values["climb_observed_mach"]:.3f}, cruise Mach
{schedule_values["cruise_observed_mach"]:.3f}, and descent
{schedule_values["descent_observed_cas_kts"]:.0f} kt / Mach
{schedule_values["descent_observed_mach"]:.3f}."""
    else:
        report_title = (
            "A320 FMS trajectory from continuous speed and mass intent"
            if continuous_plan is not None
            else "A320 FMS trajectory from mode-conditioned speed intent"
        )
        figure_title = (
            "A320 trajectory from continuous fuel-aware speed intent"
            if continuous_plan is not None
            else "A320 trajectory from frozen phase-conditioned speed intent"
        )
        schedule_explanation = f"""The speed schedule no longer uses flown CAS or Mach
medians. The frozen affine surrogate evaluates the recorded selector evidence
at declared planned phase/altitude points. Its mass source is
`{speed_intent_mass_source}`.
It supplies climb {schedule_values["climb_observed_cas_kts"]:.0f} kt / Mach
{schedule_values["climb_observed_mach"]:.3f}, cruise Mach
{schedule_values["cruise_observed_mach"]:.3f}, and descent
{schedule_values["descent_observed_cas_kts"]:.0f} kt / Mach
{schedule_values["descent_observed_mach"]:.3f}. Actual CAS, actual Mach, and
inferred speed mode do not form these targets."""
        if continuous_plan is not None:
            schedule_explanation += f"""

The surrogate is evaluated continuously every 500 m. A response-blind
{schedule_values["climb_observed_cas_kts"]:.0f}-kt climb CAS floor from the
same frozen phase-intent evaluation guards inactive low selector values. The
speed target is projected into the joint climb/acceleration energy envelope,
and calibrated predicted fuel flow is iterated to a self-consistent mass
profile. No QAR fuel or mass samples enter that iteration."""
            if cruise_strategy.kind == "commanded_mach":
                schedule_explanation += f"""

During level flight, explicit intent Mach
{cruise_strategy.commanded_mach:.3f} replaces the affine surrogate target. It
does not alter climb or descent schedules and is not estimated from QAR
response."""
    report = f"""# {report_title}

Updated: {UPDATED}

Status: **{document["status"]}**

This rollout removes all periodic QAR altitude and speed anchors. The only QAR
aircraft states exposed as constraints are the initial and terminal boundaries
of the clean reference. The 43 frozen pseudo-waypoints supply lateral geometry
only. Recorded wind and temperature are used as a forecast surrogate and never
as altitude or speed constraints.

The stable `ALT_SEL` mode supplies the {intent_metrics["cruise_selected_altitude_ft"]:.0f}-ft
cruise target. Other altitude-selection steps are not treated as spatial
constraints because some are inconsistent with the simultaneous climb/descent
state. {schedule_explanation}

The FMS calculates an unreserved TOC at
{unreserved_toc.top_of_climb_along_path_m / 1000:.1f} km and applies a fixed
{CLIMB_ENERGY_CAPACITY_FRACTION:.0%} climb-energy-capacity planning margin,
placing commanded TOC at {toc.top_of_climb_along_path_m / 1000:.1f} km. TOD is
calculated at
descent-planning prior plus a fixed
{(1.0 / DESCENT_ENERGY_CAPACITY_FRACTION - 1.0):.1%} distance reserve. Neither
location is copied from QAR. The unreserved and commanded descent distances are
{unreserved_tod.descent_ground_distance_m / 1000:.1f} km and
{tod.descent_ground_distance_m / 1000:.1f} km, respectively.

The generated trajectory has altitude RMSE {metrics["altitude_rmse_m"]:.1f} m,
TAS RMSE {metrics["speed_rmse_mps"]:.2f} m/s, CAS RMSE
{metrics["calibrated_airspeed_rmse_mps"]:.2f} m/s, Mach RMSE
{metrics["mach_rmse"]:.4f}, time-aligned horizontal RMSE
{metrics["time_aligned_horizontal_rmse_m"]:.1f} m, and arrival-time error
{metrics["arrival_time_error_s"]:+.0f} s. Held checks: {", ".join(held) if held else "none"}.

This is a deliberately hard test of intent sufficiency. A held result means the
available selected channels and terminal boundary do not fully encode the
flown vertical trajectory; it is not permission to restore periodic QAR state
anchors. The next improvements should infer active guidance regimes from
intent-side signals, add explicit procedure altitude/speed constraints, and
validate the frozen algorithm on another flight.
"""

    for path in (
        args.errors,
        args.history,
        args.profile,
        args.constraints,
        args.parameters,
        args.report,
        args.figure_prefix,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    _history(result).to_csv(args.history, index=False)
    profile_table.to_csv(args.profile, index=False)
    constraints.to_csv(args.constraints, index=False)
    document = _native(document)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    airdata_comparison.to_csv(
        args.errors.with_name(f"{args.errors.stem}-trajectory.csv"), index=False
    )
    _plot(
        qar=qar,
        route=route,
        result=result,
        profile=profile_table,
        intent_document=intent_document,
        toc_s=toc.top_of_climb_along_path_m,
        tod_s=tod.top_of_descent_along_path_m,
        prefix=args.figure_prefix,
        title=figure_title,
    )
    return document


def main() -> None:
    document = run(parse_args())
    print(
        yaml.safe_dump(
            {"status": document["status"], "metrics": document["metrics"]},
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
