#!/usr/bin/env python3
"""Identify and validate an effective A320 descent-planning thrust prior.

The target is the lower operational descent boundary mapped through the frozen
QAR effective-thrust map.  It is not claimed to be physical engine idle or a
FADEC law.  Model selection sees only the ten development/calibration flights;
the six validation flights are loaded after the structure is frozen.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero
from aerox.fms import (
    PointMassProfilePerformance,
    CASMachDescentSchedule,
    PressureScaledMinimumThrust,
    plan_top_of_descent,
)
from aerox.models.atmosphere import (
    SEA_LEVEL_PRESSURE_PA,
    atmosphere_at_pressure_altitude,
)
from tools.audit_a320_autothrust_envelope import (
    _boundary_points,
    _load_flights,
)
from tools.audit_a320_fms_intent_channels import (
    _analyze as analyze_long_holdout,
)
from tools.audit_a320_fms_intent_channels import (
    _first_sustained,
)
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import _build_plant, _style_axis

UPDATED = "2026-08-07"
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
PRESSURE_EXPONENTS = (0.0, 0.5, 1.0, 1.5)
NEAR_BEST_FRACTION = 0.02
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 320214
GATE_LIMITS = {
    "maximum_validation_mean_flight_boundary_rmse_kn": 6.0,
    "maximum_validation_mean_flight_absolute_bias_kn": 3.0,
    "minimum_validation_descent_improved_fraction": 0.8,
    "minimum_validation_median_distance_error_reduction_fraction": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "raw-dir": "data/a320_raw",
        "processed": "data/a320/a320.parquet",
        "panel": "docs/modeling/a320-replay-panel.csv",
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
        "points": "docs/modeling/a320-descent-planning-thrust-points.csv",
        "candidate-metrics": "docs/modeling/a320-descent-planning-thrust-candidates.csv",
        "flight-metrics": "docs/modeling/a320-descent-planning-thrust-flight-metrics.csv",
        "parameters": "docs/modeling/a320-descent-planning-thrust.yml",
        "report": "docs/modeling/a320-descent-planning-thrust.md",
        "figure-prefix": "docs/modeling/figures/a320-descent-planning-thrust",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _pressure_ratio(points: pd.DataFrame) -> np.ndarray:
    atmosphere = atmosphere_at_pressure_altitude(
        points["pressure_altitude_m"].to_numpy(dtype=float),
        points["static_temperature_k"].to_numpy(dtype=float),
    )
    return np.asarray(atmosphere.pressure_pa / SEA_LEVEL_PRESSURE_PA, dtype=float)


def _flight_equal_weights(points: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(points), dtype=float)
    recordings = points["recording"].unique()
    for recording in recordings:
        current = points["recording"].eq(recording).to_numpy()
        weights[current] = 1.0 / (len(recordings) * np.count_nonzero(current))
    return weights


def _fit_coefficient(points: pd.DataFrame, exponent: float) -> float:
    design = points["pressure_ratio"].to_numpy(dtype=float) ** exponent
    target = points["target_effective_thrust_n"].to_numpy(dtype=float)
    weights = _flight_equal_weights(points)
    coefficient = np.sum(weights * design * target) / np.sum(weights * design**2)
    return max(0.0, float(coefficient))


def _predict(points: pd.DataFrame, coefficient: float, exponent: float) -> np.ndarray:
    return coefficient * points["pressure_ratio"].to_numpy(dtype=float) ** exponent


def _per_flight_errors(
    points: pd.DataFrame, prediction: np.ndarray
) -> tuple[list[float], list[float]]:
    rmses: list[float] = []
    biases: list[float] = []
    target = points["target_effective_thrust_n"].to_numpy(dtype=float)
    for recording in points["recording"].unique():
        current = points["recording"].eq(recording).to_numpy()
        residual = prediction[current] - target[current]
        rmses.append(float(np.sqrt(np.mean(residual**2))))
        biases.append(float(np.mean(residual)))
    return rmses, biases


def _candidate_metrics(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    recordings = points["recording"].unique()
    for exponent in PRESSURE_EXPONENTS:
        fold_rmse: list[float] = []
        fold_bias: list[float] = []
        for recording in recordings:
            training = points.loc[~points["recording"].eq(recording)]
            held = points.loc[points["recording"].eq(recording)]
            coefficient = _fit_coefficient(training, exponent)
            rmses, biases = _per_flight_errors(
                held, _predict(held, coefficient, exponent)
            )
            fold_rmse.extend(rmses)
            fold_bias.extend(biases)
        rows.append(
            {
                "pressure_exponent": exponent,
                "feature_count": 1,
                "lofo_mean_flight_rmse_kn": np.mean(fold_rmse) / 1_000.0,
                "lofo_median_flight_rmse_kn": np.median(fold_rmse) / 1_000.0,
                "lofo_worst_flight_rmse_kn": np.max(fold_rmse) / 1_000.0,
                "lofo_mean_flight_absolute_bias_kn": (
                    np.mean(np.abs(fold_bias)) / 1_000.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    limit = (1.0 + NEAR_BEST_FRACTION) * candidates["lofo_mean_flight_rmse_kn"].min()
    return (
        candidates.loc[candidates["lofo_mean_flight_rmse_kn"] <= limit]
        .sort_values(["lofo_mean_flight_rmse_kn", "pressure_exponent"], kind="stable")
        .iloc[0]
    )


def _bootstrap_coefficients(points: pd.DataFrame, exponent: float) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    recordings = points["recording"].unique()
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(recordings, size=len(recordings), replace=True)
        groups = [points.loc[points["recording"].eq(item)] for item in sampled]
        resampled = pd.concat(groups, ignore_index=True)
        # Give repeated flight draws independent bootstrap-cluster identities.
        cursor = 0
        for draw, group in enumerate(groups):
            stop = cursor + len(group)
            resampled.loc[cursor : stop - 1, "recording"] = f"draw-{draw}"
            cursor = stop
        values[index] = _fit_coefficient(resampled, exponent)
    return values


def _mode(values: np.ndarray, *, decimals: int, fallback: float) -> tuple[float, bool]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return fallback, True
    rounded = np.round(finite, decimals)
    unique, counts = np.unique(rounded, return_counts=True)
    return float(unique[np.argmax(counts)]), False


def _descent_episode(
    frame: pd.DataFrame,
    *,
    role: str,
    registration: str,
    performance_zero: PointMassProfilePerformance,
    performance_model: PointMassProfilePerformance,
) -> dict[str, Any]:
    current = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    altitude_ft = current["altitude"].to_numpy(dtype=float)
    altitude_m = altitude_ft * FT_TO_M
    vertical_rate_ft_min = (
        savgol_filter(altitude_ft, 121, 2, deriv=1, delta=1.0, mode="interp") * 60.0
    )
    peak_altitude_ft = float(np.nanmax(altitude_ft))
    first_high = int(np.flatnonzero(altitude_ft >= peak_altitude_ft - 500.0)[0])
    tod = _first_sustained(
        vertical_rate_ft_min <= -300.0,
        start=first_high + 120,
        window=60,
        required_fraction=0.85,
    )
    terminal = np.flatnonzero(
        (np.arange(len(current)) > tod)
        & (altitude_ft <= 5_000.0)
        & (vertical_rate_ft_min < 0.0)
    )
    if len(terminal) == 0:
        raise ValueError("descent does not reach the common 5,000-ft endpoint")
    end = int(terminal[0])
    indices = np.arange(len(current))
    nominal_crossover_ft = float(
        aero.crossover_alt(280.0 * KNOT_TO_MPS, 0.78) / FT_TO_M
    )
    cas_mask = (
        (indices >= tod)
        & (indices <= end)
        & (altitude_ft >= 12_000.0)
        & (altitude_ft <= nominal_crossover_ft - 2_000.0)
        & (vertical_rate_ft_min <= -200.0)
    )
    mach_mask = (
        (indices >= tod)
        & (indices <= end)
        & (altitude_ft >= nominal_crossover_ft)
        & (vertical_rate_ft_min <= -100.0)
    )
    cas_kts, cas_fallback = _mode(
        current.loc[cas_mask, "CAS_sel"].to_numpy(), decimals=0, fallback=280.0
    )
    mach, mach_fallback = _mode(
        current.loc[mach_mask, "mach_sel"].to_numpy(), decimals=3, fallback=0.78
    )
    if not 180.0 <= cas_kts <= 350.0:
        cas_kts, cas_fallback = 280.0, True
    if not 0.5 <= mach <= 0.85:
        mach, mach_fallback = 0.78, True
    schedule = CASMachDescentSchedule(
        descent_cas_mps=cas_kts * KNOT_TO_MPS,
        descent_mach=mach,
        low_altitude_cas_mps=min(cas_kts, 250.0) * KNOT_TO_MPS,
    )
    time = current["timestamp"].to_numpy(dtype="datetime64[ns]")
    time_s = (time - time[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    ground_speed = current["groundspeed"].to_numpy(dtype=float) * KNOT_TO_MPS
    observed_distance_m = float(
        np.sum(
            0.5
            * (ground_speed[tod:end] + ground_speed[tod + 1 : end + 1])
            * np.diff(time_s[tod : end + 1])
        )
    )
    descent_slice = slice(tod, end + 1)
    wind_mps = float(
        np.nanmedian(-current["head_wind"].to_numpy(dtype=float)[descent_slice])
        * KNOT_TO_MPS
    )
    temperature_offset_k = float(
        np.nanmedian(
            current["temperature"].to_numpy(dtype=float)[descent_slice]
            - aero.temperature(altitude_m[descent_slice])
        )
    )
    common = {
        "route_distance_m": 2_000_000.0,
        "cruise_altitude_m": float(altitude_m[tod]),
        "destination_altitude_m": float(altitude_m[end]),
        "schedule": schedule,
        "mass_kg": float(current["gross_weight"].iloc[tod]),
        "altitude_step_m": 100.0,
        "along_track_wind_mps": wind_mps,
        "static_temperature_offset_k": temperature_offset_k,
    }
    zero_distance = plan_top_of_descent(
        performance=performance_zero, **common
    ).descent_ground_distance_m
    model_distance = plan_top_of_descent(
        performance=performance_model, **common
    ).descent_ground_distance_m
    return {
        "registration": registration,
        "recording": str(current["flight_id"].iloc[0]),
        "role": role,
        "split": "validation" if role == "validation" else "fit_lofo_structure",
        "observed_tod_index": tod,
        "common_endpoint_ft": altitude_ft[end],
        "observed_descent_distance_km": observed_distance_m / 1_000.0,
        "zero_thrust_predicted_distance_km": zero_distance / 1_000.0,
        "model_predicted_distance_km": model_distance / 1_000.0,
        "zero_thrust_absolute_error_km": abs(zero_distance - observed_distance_m)
        / 1_000.0,
        "model_absolute_error_km": abs(model_distance - observed_distance_m) / 1_000.0,
        "model_improves_distance": abs(model_distance - observed_distance_m)
        < abs(zero_distance - observed_distance_m),
        "selected_descent_cas_kts": cas_kts,
        "selected_descent_mach": mach,
        "cas_fallback_used": cas_fallback,
        "mach_fallback_used": mach_fallback,
        "median_along_track_wind_mps": wind_mps,
        "median_isa_temperature_offset_k": temperature_offset_k,
    }


def _plot(
    points: pd.DataFrame,
    candidates: pd.DataFrame,
    flight_metrics: pd.DataFrame,
    coefficient: float,
    exponent: float,
    bootstrap: np.ndarray,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    black, blue, orange, green = "#000000", "#0072B2", "#D55E00", "#009E73"
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=True)
    for split, color, marker, label in (
        ("fit", blue, "o", "fit boundary"),
        ("validation", orange, "^", "frozen validation"),
    ):
        current = points.loc[points["split"].eq(split)]
        axes[0, 0].scatter(
            current["pressure_ratio"],
            current["target_effective_thrust_n"] / 1_000.0,
            s=13,
            alpha=0.55,
            color=color,
            marker=marker,
            label=label,
        )
    delta = np.linspace(
        points["pressure_ratio"].min(), points["pressure_ratio"].max(), 200
    )
    curves = bootstrap[:, None] * delta[None, :] ** exponent / 1_000.0
    axes[0, 0].fill_between(
        delta,
        np.quantile(curves, 0.05, axis=0),
        np.quantile(curves, 0.95, axis=0),
        color=green,
        alpha=0.2,
        label="flight-bootstrap 90% band",
    )
    axes[0, 0].plot(
        delta,
        coefficient * delta**exponent / 1_000.0,
        color=black,
        linewidth=1.8,
        label="selected planning prior",
    )
    axes[0, 0].set(
        xlabel="Ambient pressure ratio, δ (–)", ylabel="Effective thrust (kN)"
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    axes[0, 1].plot(
        candidates["pressure_exponent"],
        candidates["lofo_mean_flight_rmse_kn"],
        color=blue,
        marker="o",
        label="fit-flight LOFO",
    )
    axes[0, 1].axvline(exponent, color=orange, linestyle="--", label="selected")
    axes[0, 1].set(
        xlabel="Pressure exponent (–)",
        ylabel="Mean per-flight boundary RMSE (kN)",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    validation = flight_metrics.loc[flight_metrics["split"].eq("validation")]
    limit = 1.05 * max(
        validation["observed_descent_distance_km"].max(),
        validation["model_predicted_distance_km"].max(),
    )
    axes[1, 0].plot([0.0, limit], [0.0, limit], color=black, linestyle=":", label="1:1")
    axes[1, 0].scatter(
        validation["observed_descent_distance_km"],
        validation["zero_thrust_predicted_distance_km"],
        color=blue,
        marker="o",
        label="zero-thrust lower bound",
    )
    axes[1, 0].scatter(
        validation["observed_descent_distance_km"],
        validation["model_predicted_distance_km"],
        color=orange,
        marker="^",
        label="planning prior",
    )
    axes[1, 0].set(
        xlabel="Observed descent distance (km)",
        ylabel="Predicted descent distance (km)",
        xlim=(0.0, limit),
        ylim=(0.0, limit),
    )
    axes[1, 0].legend(frameon=False, fontsize=7)

    x = np.arange(len(validation))
    axes[1, 1].plot(
        x,
        validation["zero_thrust_absolute_error_km"],
        color=blue,
        marker="o",
        label="zero-thrust lower bound",
    )
    axes[1, 1].plot(
        x,
        validation["model_absolute_error_km"],
        color=orange,
        marker="^",
        linestyle="--",
        label="planning prior",
    )
    axes[1, 1].set_xticks(x, validation["recording"], rotation=35, ha="right")
    axes[1, 1].set(
        xlabel="Frozen validation flight", ylabel="Absolute distance error (km)"
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    figure.suptitle("A320 effective descent-planning thrust validation", fontsize=13)
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
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[~panel["role"].eq("validation")]
    validation_selection = panel.loc[panel["role"].eq("validation")]
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected ten fit and six validation flights")

    # Leakage boundary: derive boundary points and select structure on fit flights only.
    fit_flights = _load_flights(
        fit_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    fit_points = _boundary_points(fit_flights, "fit")
    fit_points = fit_points.loc[fit_points["boundary"].eq("idle")].copy()
    thrust_map = _load_climb_thrust_ceiling(args).thrust_map
    fit_points["pressure_ratio"] = _pressure_ratio(fit_points)
    fit_points["target_effective_thrust_n"] = thrust_map.thrust_n(
        n1_percent=fit_points["target_n1_percent"].to_numpy(dtype=float),
        mach=fit_points["mach"].to_numpy(dtype=float),
        pressure_ratio=fit_points["pressure_ratio"].to_numpy(dtype=float),
        static_temperature_k=fit_points["static_temperature_k"].to_numpy(dtype=float),
    )
    candidates = _candidate_metrics(fit_points)
    chosen = _select_candidate(candidates)
    exponent = float(chosen["pressure_exponent"])
    coefficient = _fit_coefficient(fit_points, exponent)
    bootstrap = _bootstrap_coefficients(fit_points, exponent)

    # Structure, estimator, and gates are frozen before validation is loaded.
    validation_flights = _load_flights(
        validation_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    validation_points = _boundary_points(validation_flights, "validation")
    validation_points = validation_points.loc[
        validation_points["boundary"].eq("idle")
    ].copy()
    validation_points["pressure_ratio"] = _pressure_ratio(validation_points)
    validation_points["target_effective_thrust_n"] = thrust_map.thrust_n(
        n1_percent=validation_points["target_n1_percent"].to_numpy(dtype=float),
        mach=validation_points["mach"].to_numpy(dtype=float),
        pressure_ratio=validation_points["pressure_ratio"].to_numpy(dtype=float),
        static_temperature_k=validation_points["static_temperature_k"].to_numpy(
            dtype=float
        ),
    )
    validation_prediction = _predict(validation_points, coefficient, exponent)
    validation_rmse, validation_bias = _per_flight_errors(
        validation_points, validation_prediction
    )

    plant = _build_plant(args)
    maximum = _load_climb_thrust_ceiling(args)
    common_performance = {
        "plant": plant,
        "maximum_thrust_model": maximum,
        "minimum_thrust_n": 0.0,
        "maximum_climb_angle_rad": math.radians(6.0),
        "maximum_descent_angle_rad": math.radians(4.0),
        "maximum_acceleration_mps2": 0.5,
        "maximum_deceleration_mps2": 0.5,
    }
    performance_zero = PointMassProfilePerformance(**common_performance)
    performance_model = PointMassProfilePerformance(
        **common_performance,
        minimum_thrust_model=PressureScaledMinimumThrust(coefficient, exponent),
    )
    columns = [
        "flight_id",
        "timestamp",
        "altitude",
        "groundspeed",
        "gross_weight",
        "CAS_sel",
        "mach_sel",
        "head_wind",
        "temperature",
    ]
    processed = pd.read_parquet(
        args.processed,
        columns=columns,
        filters=[("flight_id", "in", panel["recording"].tolist())],
        engine="pyarrow",
    )
    panel_index = panel.set_index("recording")
    episode_rows = []
    for recording, frame in processed.groupby("flight_id", sort=True):
        metadata = panel_index.loc[str(recording)]
        episode_rows.append(
            _descent_episode(
                frame,
                role=str(metadata["role"]),
                registration=str(metadata["registration"]),
                performance_zero=performance_zero,
                performance_model=performance_model,
            )
        )
    flight_metrics = pd.DataFrame(episode_rows)
    validation_flight_metrics = flight_metrics.loc[
        flight_metrics["split"].eq("validation")
    ]
    zero_median = float(
        validation_flight_metrics["zero_thrust_absolute_error_km"].median()
    )
    model_median = float(validation_flight_metrics["model_absolute_error_km"].median())
    distance_reduction = 1.0 - model_median / zero_median
    improved_fraction = float(
        validation_flight_metrics["model_improves_distance"].mean()
    )
    gates = {
        "validation_mean_flight_boundary_rmse_kn": np.mean(validation_rmse) / 1_000.0,
        "validation_mean_flight_absolute_bias_kn": np.mean(np.abs(validation_bias))
        / 1_000.0,
        "validation_descent_improved_fraction": improved_fraction,
        "validation_median_distance_error_reduction_fraction": distance_reduction,
    }
    passed = (
        gates["validation_mean_flight_boundary_rmse_kn"]
        <= GATE_LIMITS["maximum_validation_mean_flight_boundary_rmse_kn"]
        and gates["validation_mean_flight_absolute_bias_kn"]
        <= GATE_LIMITS["maximum_validation_mean_flight_absolute_bias_kn"]
        and improved_fraction
        >= GATE_LIMITS["minimum_validation_descent_improved_fraction"]
        and distance_reduction
        >= GATE_LIMITS["minimum_validation_median_distance_error_reduction_fraction"]
    )

    # The long flight is a post-fit benchmark and never participates in release.
    long_qar, long_timeline, _, long_metrics = analyze_long_holdout(args)
    long_schedule = CASMachDescentSchedule(
        descent_cas_mps=long_metrics["schedule"]["descent_selected_cas_kts"]
        * KNOT_TO_MPS,
        descent_mach=long_metrics["schedule"]["descent_selected_mach"],
        low_altitude_cas_mps=long_metrics["tod_planning_low_altitude_cas_kts"]
        * KNOT_TO_MPS,
    )
    long_plan = plan_top_of_descent(
        route_distance_m=float(long_timeline["along_path_m"].iloc[-1]),
        cruise_altitude_m=long_metrics["cruise_selected_altitude_ft"] * FT_TO_M,
        destination_altitude_m=float(long_timeline["pressure_altitude_ft"].iloc[-1])
        * FT_TO_M,
        schedule=long_schedule,
        performance=performance_model,
        mass_kg=float(
            long_qar.mass_kg[int(long_metrics["observed_top_of_climb_time_s"])]
        ),
        altitude_step_m=50.0,
        along_track_wind_mps=long_metrics[
            "observed_descent_median_along_track_wind_mps"
        ],
    )
    long_error_km = (
        long_plan.descent_ground_distance_m
        - long_metrics["observed_descent_distance_m"]
    ) / 1_000.0

    all_points = pd.concat([fit_points, validation_points], ignore_index=True)
    document: dict[str, Any] = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "released-effective-descent-planning-prior" if passed else "rejected",
        "interpretation": {
            "output": "effective total-aircraft minimum planning thrust",
            "physical_engine_idle": False,
            "fadec_law": False,
            "use": "energy-path and top-of-descent planning with uncertainty reserve",
        },
        "leakage_control": {
            "fit_flights": [flight.key for flight in fit_flights],
            "structure_frozen_before_validation_load": True,
            "validation_flights": [flight.key for flight in validation_flights],
            "long_flight_used_for_fit_selection_or_release": False,
        },
        "model": {
            "equation": "T_min = sea_level_thrust_n * pressure_ratio ** pressure_exponent",
            "sea_level_thrust_n": coefficient,
            "pressure_exponent": exponent,
            "bootstrap_complete_flight_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_coefficient_p05_n": float(np.quantile(bootstrap, 0.05)),
            "bootstrap_coefficient_p50_n": float(np.quantile(bootstrap, 0.50)),
            "bootstrap_coefficient_p95_n": float(np.quantile(bootstrap, 0.95)),
            "core_class": "aerox.fms.PressureScaledMinimumThrust",
        },
        "selection": {
            "candidate_pressure_exponents": list(PRESSURE_EXPONENTS),
            "cross_validation": "leave-one-complete-fit-flight-out",
            "primary_score": "mean per-flight mapped-boundary thrust RMSE",
            "near_best_fraction": NEAR_BEST_FRACTION,
            "selected": {key: float(value) for key, value in chosen.items()},
        },
        "release_gates": {
            "limits": GATE_LIMITS,
            "observed": gates,
            "all_passed": bool(passed),
            "planner_integration_authorized": bool(passed),
        },
        "descent_distance_validation": {
            "common_terminal_altitude_ft": 5_000.0,
            "zero_thrust_median_absolute_error_km": zero_median,
            "model_median_absolute_error_km": model_median,
            "uses_recorded_wind_and_selected_CAS_Mach_candidates": True,
        },
        "post_fit_long_flight_benchmark": {
            "recording": "TAIL-A/FLT0003",
            "observed_descent_distance_km": long_metrics["observed_descent_distance_m"]
            / 1_000.0,
            "model_predicted_descent_distance_km": long_plan.descent_ground_distance_m
            / 1_000.0,
            "model_minus_observed_distance_km": long_error_km,
            "used_recorded_median_wind_for_diagnostic": True,
        },
        "limitations": [
            "The FMA selected-versus-managed and fixed-idle state is not observable.",
            "The lower operational N1 quantile mixes true idle and speed-maintaining operation.",
            "Mapped thrust remains conditional on the frozen drag polar and QAR effective-thrust map.",
            "Full descent distance also contains altitude constraints, level-offs, and interventions.",
            "Use forecast wind operationally; recorded wind is validation evidence only.",
        ],
        "sources": [
            "docs/modeling/a320-autothrust-envelope.yml",
            "docs/modeling/a320-effective-thrust-calibration.yml",
            "docs/modeling/a320-replay-panel.csv",
            "data/a320_raw/tail-a_2011.parquet",
            "data/a320_raw/tail-b_2011.parquet",
        ],
    }
    report = f"""# A320 effective descent-planning thrust prior

Updated: {UPDATED}

Aerox releases a one-parameter **effective planning prior**, not a physical
engine-idle or FADEC model:

`T_min = {coefficient:.1f} * pressure_ratio^{exponent:.1f} N`

The structure was selected using complete-flight leave-one-out validation on
ten development/calibration flights. Six validation flights were loaded only
after the structure, estimator, and gates were frozen. The mapped lower-boundary
validation RMSE is {gates["validation_mean_flight_boundary_rmse_kn"]:.2f} kN and
the mean absolute per-flight bias is
{gates["validation_mean_flight_absolute_bias_kn"]:.2f} kN.

For the six frozen validation descents, using QAR candidate CAS/Mach intent and
recorded wind, the median absolute descent-distance error falls from
{zero_median:.1f} km with the zero-thrust lower bound to {model_median:.1f} km.
The model improves {improved_fraction:.0%} of validation flights. All release
gates **{"pass" if passed else "do not pass"}**.

The untouched long `TAIL-A/FLT0003` benchmark is evaluated only after release:
the modeled descent distance is {long_plan.descent_ground_distance_m / 1000:.1f}
km versus {long_metrics["observed_descent_distance_m"] / 1000:.1f} km observed
({long_error_km:+.1f} km). Recorded median wind is used only for this post-flight
diagnostic; an operational Aerox FMS must use forecast wind.

The main limitation remains mode observability. The lower operational N1
quantile can mix fixed idle, speed-maintaining thrust, level-offs, and pilot or
automation interventions. Accordingly this model is suitable as a calibrated
energy/TOD planning prior with a distance reserve, but it must not be presented
as a CFM56 idle deck or Airbus autothrust law.
"""

    for path in (
        args.points,
        args.candidate_metrics,
        args.flight_metrics,
        args.parameters,
        args.report,
        args.figure_prefix,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    all_points.to_csv(args.points, index=False)
    candidates.to_csv(args.candidate_metrics, index=False)
    flight_metrics.to_csv(args.flight_metrics, index=False)
    document = _native(document)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(
        all_points,
        candidates,
        flight_metrics,
        coefficient,
        exponent,
        bootstrap,
        args.figure_prefix,
    )
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["release_gates"], sort_keys=False).strip())


if __name__ == "__main__":
    main()
