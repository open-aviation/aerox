#!/usr/bin/env python3
"""Audit QAR intent channels, CAS/Mach schedules, and energy-based TOD."""

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
from scipy.optimize import brentq
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero
from aerox.flightdata.a320_qar import (
    FT_TO_M,
    KNOT_TO_MPS,
    TIME_COLUMN,
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.fms import (
    PointMassProfilePerformance,
    CASMachDescentSchedule,
    plan_top_of_descent,
)
from aerox.models.atmosphere import (
    calibrated_airspeed_from_true_airspeed,
    mach_from_true_airspeed,
)
from tools.calibrate_a320_effective_thrust import RAW_FILES
from tools.validate_a320_fms_long_range_qar import _load_holdout
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import _build_plant, _style_axis

UPDATED = "2026-08-07"
INTENT_CHANNELS = (
    "ALT_SEL",
    "SPEEDP_SEL",
    "MACH_SEL",
    "SPEEDV_SEL..ft.mn.",
    "FPA_SEL",
    "HEADING_SEL",
    "TRACK_SEL",
)


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
        "timeline": "docs/fms/a320-fms-qar-intent-channel-timeline.csv",
        "descent-profile": "docs/fms/a320-fms-energy-tod-profile.csv",
        "parameters": "docs/fms/a320-fms-qar-intent-channel-audit.yml",
        "report": "docs/fms/a320-fms-qar-intent-channel-audit.md",
        "figure-prefix": "docs/fms/figures/a320-fms-qar-intent-channel-audit",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    time = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time)
    order = np.argsort(time[keep], kind="stable")
    indices = np.flatnonzero(keep)[order]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[indices]


def _processed_origin_time_s(
    processed_path: Path, recording: str, decoded: Any
) -> float:
    """Return raw-recorder time corresponding to processed-flight time zero."""

    processed = pd.read_parquet(
        processed_path,
        columns=["flight_id", "timestamp"],
        filters=[("flight_id", "==", recording)],
        engine="pyarrow",
    )
    if len(processed) == 0:
        raise ValueError(f"processed flight {recording} is absent")
    timestamp = processed["timestamp"].min().to_datetime64()
    index = int(np.searchsorted(decoded.timestamp_utc, timestamp))
    if index >= len(decoded.timestamp_utc) or decoded.timestamp_utc[index] != timestamp:
        raise ValueError("processed flight origin does not align with raw QAR time")
    return float(decoded.time_s[index])


def _first_sustained(
    condition: np.ndarray,
    *,
    start: int,
    window: int = 90,
    required_fraction: float = 0.9,
) -> int:
    values = np.asarray(condition, dtype=bool)
    counts = np.convolve(values.astype(int), np.ones(window, dtype=int), mode="valid")
    candidates = np.flatnonzero(
        (np.arange(len(counts)) >= start)
        & (counts >= math.ceil(required_fraction * window))
    )
    if len(candidates) == 0:
        raise ValueError("no sustained phase transition found")
    return int(candidates[0])


def _mode(values: np.ndarray, *, decimals: int = 0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("cannot estimate a mode from no finite values")
    rounded = np.round(finite, decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    return float(unique[np.argmax(counts)])


def _load_timeline(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, str, str]:
    qar, registration, recording = _load_holdout(args)
    required = list(dict.fromkeys((*raw_columns(), *INTENT_CHANNELS)))
    raw = pd.read_parquet(
        args.raw_dir / RAW_FILES[registration],
        columns=required,
        filters=[("recording", "==", recording)],
        engine="pyarrow",
    )
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    processed_origin_time_s = _processed_origin_time_s(
        args.processed, recording, decoded
    )
    source_time = (
        processed_origin_time_s
        + qar.source_start_time_s
        + qar.reference.time_s
    )
    raw_index = np.searchsorted(decoded.time_s, source_time)
    if np.any(raw_index >= len(decoded.time_s)) or not np.allclose(
        decoded.time_s[raw_index], source_time, atol=1e-6
    ):
        raise ValueError("QAR intent channels do not align with the frozen reference")
    held: dict[str, np.ndarray] = {}
    for channel in INTENT_CHANNELS:
        full = bounded_sample_and_hold(_ordered_raw(raw, channel), decoded.time_s, 1.0)
        held[channel] = full[raw_index]

    altitude = qar.reference.pressure_altitude_m
    actual_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            altitude,
            qar.static_temperature_k,
        ),
        dtype=float,
    )
    actual_mach = np.asarray(
        mach_from_true_airspeed(
            qar.reference.true_airspeed_mps,
            altitude,
            qar.static_temperature_k,
        ),
        dtype=float,
    )
    along_path = np.r_[
        0.0,
        np.cumsum(
            np.hypot(np.diff(qar.north_position_m), np.diff(qar.east_position_m))
        ),
    ]
    north_gradient = np.gradient(qar.north_position_m)
    east_gradient = np.gradient(qar.east_position_m)
    horizontal_norm = np.hypot(north_gradient, east_gradient)
    along_track_wind = (
        qar.wind_north_mps * north_gradient + qar.wind_east_mps * east_gradient
    ) / horizontal_norm
    altitude_ft = altitude / FT_TO_M
    vertical_rate_ft_min = (
        savgol_filter(altitude_ft, 121, 2, deriv=1, delta=1.0, mode="interp") * 60.0
    )
    timeline = pd.DataFrame(
        {
            "elapsed_time_s": qar.reference.time_s,
            "raw_source_time_s": source_time,
            "along_path_m": along_path,
            "pressure_altitude_ft": altitude_ft,
            "vertical_rate_ft_min": vertical_rate_ft_min,
            "actual_cas_kts": actual_cas / KNOT_TO_MPS,
            "actual_mach": actual_mach,
            "selected_altitude_ft": held["ALT_SEL"],
            "selected_speed_kts": held["SPEEDP_SEL"],
            "selected_mach": held["MACH_SEL"],
            "selected_vertical_speed_ft_min": held["SPEEDV_SEL..ft.mn."],
            "selected_heading_deg": held["HEADING_SEL"],
            "along_track_wind_mps": along_track_wind,
        }
    )
    return qar, timeline, registration, recording


def _analyze(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, Any, dict[str, Any]]:
    qar, timeline, registration, recording = _load_timeline(args)
    altitude = timeline["pressure_altitude_ft"].to_numpy()
    vertical_rate = timeline["vertical_rate_ft_min"].to_numpy()
    selected_altitude = timeline["selected_altitude_ft"].to_numpy()
    cruise_altitude_ft = _mode(selected_altitude[selected_altitude > 30_000.0])
    toc = _first_sustained(
        (np.abs(altitude - cruise_altitude_ft) <= 250.0)
        & (np.abs(vertical_rate) <= 250.0),
        start=300,
        window=120,
    )
    tod = _first_sustained(
        vertical_rate <= -300.0,
        start=toc + 600,
        window=90,
    )

    initial_crossover_ft = float(
        aero.crossover_alt(290.0 * KNOT_TO_MPS, 0.78) / FT_TO_M
    )
    climb_cas_mask = (
        (np.arange(len(timeline)) < toc)
        & (altitude >= 12_000.0)
        & (altitude <= initial_crossover_ft - 2_000.0)
        & (vertical_rate >= 300.0)
    )
    climb_mach_mask = (
        (np.arange(len(timeline)) < toc)
        & (altitude >= initial_crossover_ft)
        & (vertical_rate >= 100.0)
    )
    descent_cas_mask = (
        (np.arange(len(timeline)) >= tod)
        & (altitude >= 12_000.0)
        & (altitude <= initial_crossover_ft - 2_000.0)
        & (vertical_rate <= -300.0)
    )
    descent_mach_mask = (
        (np.arange(len(timeline)) >= tod)
        & (altitude >= initial_crossover_ft)
        & (vertical_rate <= -200.0)
    )
    low_descent_mask = (
        (np.arange(len(timeline)) >= tod)
        & (altitude >= 4_000.0)
        & (altitude <= 9_500.0)
        & (vertical_rate <= -200.0)
    )
    cruise_mask = (
        (np.arange(len(timeline)) >= toc + 120)
        & (np.arange(len(timeline)) < tod - 120)
        & (np.abs(altitude - cruise_altitude_ft) <= 250.0)
        & (np.abs(vertical_rate) <= 250.0)
    )

    schedule_values = {
        "climb_selected_cas_kts": _mode(
            timeline.loc[climb_cas_mask, "selected_speed_kts"].to_numpy()
        ),
        "climb_observed_cas_kts": float(
            np.median(timeline.loc[climb_cas_mask, "actual_cas_kts"])
        ),
        "climb_selected_mach": _mode(
            timeline.loc[climb_mach_mask, "selected_mach"].to_numpy(), decimals=3
        ),
        "climb_observed_mach": float(
            np.median(timeline.loc[climb_mach_mask, "actual_mach"])
        ),
        "cruise_selected_mach": _mode(
            timeline.loc[cruise_mask, "selected_mach"].to_numpy(), decimals=3
        ),
        "cruise_observed_mach": float(
            np.median(timeline.loc[cruise_mask, "actual_mach"])
        ),
        "cruise_observed_mach_q25": float(
            np.quantile(timeline.loc[cruise_mask, "actual_mach"], 0.25)
        ),
        "cruise_observed_mach_q75": float(
            np.quantile(timeline.loc[cruise_mask, "actual_mach"], 0.75)
        ),
        "descent_selected_cas_kts": _mode(
            timeline.loc[descent_cas_mask, "selected_speed_kts"].to_numpy()
        ),
        "descent_observed_cas_kts": float(
            np.median(timeline.loc[descent_cas_mask, "actual_cas_kts"])
        ),
        "descent_selected_mach": _mode(
            timeline.loc[descent_mach_mask, "selected_mach"].to_numpy(), decimals=3
        ),
        "descent_observed_mach": float(
            np.median(timeline.loc[descent_mach_mask, "actual_mach"])
        ),
        "low_altitude_selected_cas_kts": _mode(
            timeline.loc[low_descent_mask, "selected_speed_kts"].to_numpy()
        ),
        "low_altitude_observed_cas_kts": float(
            np.median(timeline.loc[low_descent_mask, "actual_cas_kts"])
        ),
    }
    schedule = CASMachDescentSchedule(
        descent_cas_mps=schedule_values["descent_selected_cas_kts"] * KNOT_TO_MPS,
        descent_mach=schedule_values["descent_selected_mach"],
        # This clean reference ends near 8,300 ft and does not expose a
        # conventional terminal speed reduction. Do not turn its later 269-kt
        # value into an invented low-altitude constraint for the TOD path.
        low_altitude_cas_mps=min(
            schedule_values["low_altitude_selected_cas_kts"],
            schedule_values["descent_selected_cas_kts"],
        )
        * KNOT_TO_MPS,
    )
    plant = _build_plant(args)
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=_load_climb_thrust_ceiling(args),
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )
    route_distance = float(timeline["along_path_m"].iloc[-1])
    tod_plan = plan_top_of_descent(
        route_distance_m=route_distance,
        cruise_altitude_m=cruise_altitude_ft * FT_TO_M,
        destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
        schedule=schedule,
        performance=performance,
        mass_kg=float(qar.mass_kg[toc]),
        altitude_step_m=50.0,
    )
    observed_remaining = route_distance - float(timeline["along_path_m"].iloc[tod])
    observed_descent_wind = float(np.median(timeline.loc[tod:, "along_track_wind_mps"]))
    wind_sensitivity: dict[str, float] = {}
    for wind_mps in (-20.0, 0.0, 20.0):
        wind_plan = plan_top_of_descent(
            route_distance_m=route_distance,
            cruise_altitude_m=cruise_altitude_ft * FT_TO_M,
            destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
            schedule=schedule,
            performance=performance,
            mass_kg=float(qar.mass_kg[toc]),
            altitude_step_m=50.0,
            along_track_wind_mps=wind_mps,
        )
        wind_sensitivity[f"{wind_mps:+.0f}_mps"] = wind_plan.descent_ground_distance_m

    def distance_residual(minimum_thrust_n: float) -> float:
        trial = plan_top_of_descent(
            route_distance_m=route_distance,
            cruise_altitude_m=cruise_altitude_ft * FT_TO_M,
            destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
            schedule=schedule,
            performance=replace(performance, minimum_thrust_n=minimum_thrust_n),
            mass_kg=float(qar.mass_kg[toc]),
            altitude_step_m=50.0,
        )
        return trial.descent_ground_distance_m - observed_remaining

    upper_idle_n = 25_000.0
    implied_idle_n = (
        float(brentq(distance_residual, 0.0, upper_idle_n))
        if distance_residual(0.0) * distance_residual(upper_idle_n) < 0.0
        else None
    )
    observed_wind_plan = plan_top_of_descent(
        route_distance_m=route_distance,
        cruise_altitude_m=cruise_altitude_ft * FT_TO_M,
        destination_altitude_m=float(qar.reference.pressure_altitude_m[-1]),
        schedule=schedule,
        performance=performance,
        mass_kg=float(qar.mass_kg[toc]),
        altitude_step_m=50.0,
        along_track_wind_mps=observed_descent_wind,
    )
    metrics = {
        "registration": registration,
        "recording": recording,
        "sample_count": len(timeline),
        "channel_coverage_fraction": {
            channel: float(np.mean(np.isfinite(timeline[column])))
            for channel, column in {
                "ALT_SEL": "selected_altitude_ft",
                "SPEEDP_SEL": "selected_speed_kts",
                "MACH_SEL": "selected_mach",
                "SPEEDV_SEL": "selected_vertical_speed_ft_min",
                "HEADING_SEL": "selected_heading_deg",
            }.items()
        },
        "cruise_selected_altitude_ft": cruise_altitude_ft,
        "observed_top_of_climb_time_s": float(timeline["elapsed_time_s"].iloc[toc]),
        "observed_top_of_descent_time_s": float(timeline["elapsed_time_s"].iloc[tod]),
        "observed_top_of_descent_along_path_m": float(
            timeline["along_path_m"].iloc[tod]
        ),
        "observed_descent_distance_m": observed_remaining,
        "calculated_top_of_descent_along_path_m": tod_plan.top_of_descent_along_path_m,
        "calculated_descent_distance_m": tod_plan.descent_ground_distance_m,
        "calculated_minus_observed_descent_distance_m": (
            tod_plan.descent_ground_distance_m - observed_remaining
        ),
        "calculated_tod_path_error_m": (
            tod_plan.top_of_descent_along_path_m
            - float(timeline["along_path_m"].iloc[tod])
        ),
        "tod_idle_thrust_assumption_n": performance.minimum_thrust_n,
        "tod_wind_assumption_mps": 0.0,
        "tod_wind_sensitivity_descent_distance_m": wind_sensitivity,
        "observed_descent_median_along_track_wind_mps": observed_descent_wind,
        "observed_wind_diagnostic_descent_distance_m": (
            observed_wind_plan.descent_ground_distance_m
        ),
        "observed_wind_diagnostic_tod_path_error_m": (
            observed_wind_plan.top_of_descent_along_path_m
            - float(timeline["along_path_m"].iloc[tod])
        ),
        "tod_mass_source": "QAR mass at detected top of climb",
        "single_flight_implied_minimum_net_thrust_n": implied_idle_n,
        "single_flight_implied_thrust_is_released_idle_model": False,
        "schedule": schedule_values,
        "tod_planning_low_altitude_cas_kts": (
            schedule.low_altitude_cas_mps / KNOT_TO_MPS
        ),
        "descent_crossover_altitude_ft": schedule.crossover_altitude_m / FT_TO_M,
    }
    timeline["phase"] = "cruise"
    timeline.loc[: toc - 1, "phase"] = "climb"
    timeline.loc[tod:, "phase"] = "descent"
    return qar, timeline, tod_plan, metrics


def _plot(
    timeline: pd.DataFrame,
    tod_plan: Any,
    metrics: dict[str, Any],
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    black, blue, orange, green, magenta = (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
    )
    time_min = timeline["elapsed_time_s"] / 60.0
    toc_min = metrics["observed_top_of_climb_time_s"] / 60.0
    tod_min = metrics["observed_top_of_descent_time_s"] / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.9), constrained_layout=True)
    axes[0, 0].plot(
        time_min,
        timeline["pressure_altitude_ft"] / 1_000.0,
        color=black,
        label="QAR altitude",
    )
    axes[0, 0].step(
        time_min,
        timeline["selected_altitude_ft"] / 1_000.0,
        where="post",
        color=orange,
        linestyle="--",
        label="ALT_SEL candidate",
    )
    axes[0, 0].axvline(toc_min, color=green, linestyle=":", label="detected TOC")
    axes[0, 0].axvline(tod_min, color=magenta, linestyle=":", label="observed TOD")
    axes[0, 0].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (kft)")
    axes[0, 0].legend(frameon=False, fontsize=6.7)

    axes[0, 1].plot(time_min, timeline["actual_cas_kts"], color=black, label="QAR CAS")
    axes[0, 1].plot(
        time_min,
        timeline["selected_speed_kts"],
        color=blue,
        linestyle="--",
        label="SPEEDP_SEL candidate",
    )
    axes[0, 1].axvline(tod_min, color=magenta, linestyle=":")
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="Calibrated airspeed (kt)")
    axes[0, 1].legend(frameon=False, fontsize=6.7)

    axes[1, 0].plot(time_min, timeline["actual_mach"], color=black, label="QAR Mach")
    axes[1, 0].plot(
        time_min,
        timeline["selected_mach"],
        color=green,
        linestyle="--",
        label="MACH_SEL candidate",
    )
    axes[1, 0].axvline(tod_min, color=magenta, linestyle=":")
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="Mach number")
    axes[1, 0].legend(frameon=False, fontsize=6.7)

    route_distance = float(timeline["along_path_m"].iloc[-1])
    descent = timeline["phase"] == "descent"
    actual_remaining = route_distance - timeline.loc[descent, "along_path_m"].to_numpy()
    axes[1, 1].plot(
        actual_remaining / 1_000.0,
        timeline.loc[descent, "pressure_altitude_ft"] / 1_000.0,
        color=black,
        label="observed descent",
    )
    axes[1, 1].plot(
        tod_plan.distance_to_destination_m / 1_000.0,
        tod_plan.altitude_m / FT_TO_M / 1_000.0,
        color=magenta,
        linestyle="--",
        label="calculated idle path",
    )
    axes[1, 1].set(
        xlabel="Distance to destination (km)",
        ylabel="Pressure altitude (kft)",
    )
    axes[1, 1].legend(frameon=False, fontsize=6.7)
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 QAR intent channels and energy-based TOD", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _, timeline, tod_plan, metrics = _analyze(args)
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {metrics['registration']}",
        "status": "candidate-intent-channels-audited",
        "source": {
            "recording": f"{metrics['registration']}/{metrics['recording']}",
            "post_holdout_diagnostic": True,
            "raw_selected_channels_used": list(INTENT_CHANNELS),
            "active_selected_vs_managed_mode_observable": False,
            "qar_trajectory_used_for_benchmark_only": True,
        },
        "intent_use": {
            "lateral": "turn-aware geometry waypoints and geometry-only pseudo-waypoints",
            "vertical": "ALT_SEL plus candidate CAS/Mach schedule",
            "pseudo_waypoints_create_vertical_constraints": False,
            "SPEEDP_SEL_claimed_as_active_autothrust_target": False,
            "observed_CAS_and_Mach_use_recorded_static_temperature": True,
        },
        "metrics": metrics,
        "tod_method": {
            "algorithm": "backward integration of specific mechanical energy",
            "equation": "E = g h + V^2/2; dE/ds = (T_min - D)/m",
            "includes": [
                "drag polar",
                "mass",
                "CAS/Mach crossover",
                "10000-ft CAS transition",
                "descent-angle cap",
                "optional along-track wind and distance reserve",
            ],
            "current_limitations": [
                "zero net thrust is a provisional lower bound, not a released idle model",
                "primary calculation uses zero wind",
                "selected-versus-managed mode is not observable",
                "single-flight implied thrust is confounded with wind, speed schedule, and pilot interventions",
            ],
            "core_operation": "aerox.fms.plan_top_of_descent",
        },
        "artifacts": {
            "timeline": str(args.timeline),
            "descent_profile": str(args.descent_profile),
            "figure": f"{args.figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/audit_a320_fms_intent_channels.py",
        },
    }
    schedule = metrics["schedule"]
    report = f"""# A320 QAR intent-channel and top-of-descent audit

Updated: {UPDATED}

The long `TAIL-A/FLT0003` flight contains nearly complete `ALT_SEL`,
`SPEEDP_SEL`, and `MACH_SEL` channels. They are useful intent evidence, but the
QAR lacks the published FMA bits needed to determine active selected versus
managed speed mode. Aerox therefore does not call `SPEEDP_SEL` an active
autothrust command.

The extracted candidate climb schedule is {schedule["climb_selected_cas_kts"]:.0f}
kt / Mach {schedule["climb_selected_mach"]:.3f}; the candidate descent schedule
is {schedule["descent_selected_cas_kts"]:.0f} kt / Mach
{schedule["descent_selected_mach"]:.3f}. The channel later shows
{schedule["low_altitude_selected_cas_kts"]:.0f} kt below 10,000 ft, but this
clean reference ends near 8,300 ft and contains no terminal speed reduction;
the TOD benchmark therefore retains
{metrics["tod_planning_low_altitude_cas_kts"]:.0f} kt through its endpoint. The
corresponding observed stable-flight medians remain in the YAML for comparison.

The stable flown cruise median is Mach
{schedule["cruise_observed_mach"]:.3f}, with interquartile range
{schedule["cruise_observed_mach_q25"]:.3f}--
{schedule["cruise_observed_mach_q75"]:.3f}. It is retained as inferred
cruise-speed intent for the post-holdout trajectory diagnostic, not as a
decoded active managed-speed command.

The first physics-based TOD calculation integrates the minimum-thrust specific
energy path backward from the destination. It predicts
{metrics["calculated_descent_distance_m"] / 1000:.1f} km, versus
{metrics["observed_descent_distance_m"] / 1000:.1f} km observed, a difference of
{metrics["calculated_minus_observed_descent_distance_m"] / 1000:+.1f} km. This
is a provisional benchmark because zero net thrust is still only a declared
lower bound; it is not an identified A320 idle-thrust model.

The same calculation gives descent distances of
{metrics["tod_wind_sensitivity_descent_distance_m"]["-20_mps"] / 1000:.1f},
{metrics["tod_wind_sensitivity_descent_distance_m"]["+0_mps"] / 1000:.1f}, and
{metrics["tod_wind_sensitivity_descent_distance_m"]["+20_mps"] / 1000:.1f} km
for -20, 0, and +20 m/s along-track wind. Matching this one observed descent
with zero wind would imply
{metrics["single_flight_implied_minimum_net_thrust_n"] / 1000:.1f} kN minimum
net thrust, but that value is confounded and is not released as an idle model.
The recorded descent has a median {metrics["observed_descent_median_along_track_wind_mps"]:+.1f}
m/s along-track wind; using that only as a post-flight diagnostic changes the
predicted distance to
{metrics["observed_wind_diagnostic_descent_distance_m"] / 1000:.1f} km. An
operational cruise calculation must instead use forecast wind along the
remaining route.

Turn-aware pseudo-waypoints remain lateral geometry inputs only. They inherit
the vertical plan and create no altitude or speed constraints.
"""
    args.timeline.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(args.timeline, index=False)
    pd.DataFrame(
        {
            "pressure_altitude_m": tod_plan.altitude_m,
            "true_airspeed_mps": tod_plan.true_airspeed_mps,
            "distance_to_destination_m": tod_plan.distance_to_destination_m,
            "minimum_specific_excess_force_mps2": (
                tod_plan.minimum_specific_excess_force_mps2
            ),
        }
    ).to_csv(args.descent_profile, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(timeline, tod_plan, metrics, args.figure_prefix)
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
