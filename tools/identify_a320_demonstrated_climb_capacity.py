#!/usr/bin/env python3
"""Identify an uncertainty-bounded A320 demonstrated CLB capacity frontier.

The model is deliberately an operational lower confidence bound on a high-N1
frontier.  It is neither a physical engine maximum nor an Airbus/CFM56 rating
schedule.  Structure selection sees ten development flights only; six prior
validation flights, one FLT0004 diagnostic, and two predeclared untouched
TAIL-B flights are loaded in that order after the method is frozen.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero, load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    N1_ENVELOPE_FEATURE_SETS,
    DemonstratedClimbN1Capacity,
    EffectiveN1Envelope,
    EffectiveThrustMap,
    n1_envelope_features,
)
from aerox.modeling.force_reconstruction import centered_linear_derivative
from tools.audit_a320_autothrust_envelope import (
    ALTITUDE_BIN_FT,
    DOMINANT_CLB_CODE,
    FT_TO_M,
    MINIMUM_BIN_SAMPLES,
    MINIMUM_CLIMB_RATE_FT_S,
    EnvelopeFlight,
    _base_mask,
)
from tools.audit_a320_autothrust_envelope import (
    _load_flights as _load_envelope_flights,
)
from tools.calibrate_a320_effective_thrust import (
    _load_flights as _load_propulsion_flights,
)
from tools.calibrate_a320_effective_thrust import (
    _predict as _predict_thrust,
)
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
    _target_thrust_n,
)

UPDATED = "2026-08-06"
WITHIN_BIN_FRONTIER_QUANTILE = 0.99
ACROSS_FLIGHT_FRONTIER_QUANTILE = 0.90
LOWER_CONFIDENCE_QUANTILE = 0.10
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_SEED = 20260806
N1_DERIVATIVE_WINDOW_S = 11
MAXIMUM_ABSOLUTE_N1_RATE_PERCENT_S = 0.25
MAXIMUM_ENGINE_DISAGREEMENT_PERCENT = 1.0
MINIMUM_STABLE_RUN_S = 20.0
UPPER_CONDITION_FRACTION = 0.20
FIT_NEAR_BEST_FRACTION = 0.02
HIGH_N1_MAP_MARGIN_PERCENT = 0.50
FRONTIER_SUPPORT_TOLERANCE_PERCENT = 0.25

DIAGNOSTIC_SELECTION = pd.DataFrame(
    [{"registration": "TAIL-A", "recording": "FLT0004", "role": "diagnostic"}]
)
CONFIRMATION_SELECTION = pd.DataFrame(
    [
        {"registration": "TAIL-B", "recording": "FLT0006", "role": "confirmation"},
        {"registration": "TAIL-B", "recording": "FLT0007", "role": "confirmation"},
        {"registration": "TAIL-B", "recording": "FLT0008", "role": "confirmation"},
        {"registration": "TAIL-B", "recording": "FLT0009", "role": "confirmation"},
    ]
)
HOLDOUT_SELECTION = pd.DataFrame(
    [
        {"registration": "TAIL-B", "recording": "FLT0001", "role": "holdout"},
        {"registration": "TAIL-B", "recording": "FLT0002", "role": "holdout"},
    ]
)

GATE_LIMITS = {
    "validation_worst_supported_stable_fraction": {"minimum": 0.90},
    "holdout_worst_supported_stable_fraction": {"minimum": 0.90},
    "validation_frontier_quantile_coverage": {"minimum": 0.75, "maximum": 0.98},
    "holdout_frontier_quantile_coverage": {"minimum": 0.75, "maximum": 0.99},
    "validation_frontier_pinball_loss_percent": {"maximum": 0.05},
    "holdout_frontier_pinball_loss_percent": {"maximum": 0.05},
    "validation_frontier_rmse_percent": {"maximum": 1.0},
    "holdout_frontier_rmse_percent": {"maximum": 1.0},
    "validation_registration_frontier_bias_gap_percent": {"maximum": 0.25},
    "validation_median_high_n1_thrust_rmse_kn": {"maximum": 15.0},
    "holdout_median_high_n1_thrust_rmse_kn": {"maximum": 15.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv")
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
        "--n1-envelope",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity-points.csv"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path(
            "docs/modeling/a320-demonstrated-climb-capacity-flight-metrics.csv"
        ),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-demonstrated-climb-capacity"),
    )
    return parser.parse_args()


def _stable_capacity_mask(flight: EnvelopeFlight) -> np.ndarray:
    n1_rate = centered_linear_derivative(
        flight.n1_percent, flight.time_s, N1_DERIVATIVE_WINDOW_S
    )
    candidate = (
        _base_mask(flight)
        & np.isclose(flight.flap_code, 0.0)
        & (flight.altitude_rate_ft_s >= MINIMUM_CLIMB_RATE_FT_S)
        & (np.abs(n1_rate) <= MAXIMUM_ABSOLUTE_N1_RATE_PERCENT_S)
        & (flight.n1_engine_difference_percent <= MAXIMUM_ENGINE_DISAGREEMENT_PERCENT)
    )
    selected = np.zeros(len(candidate), dtype=bool)
    indices = np.flatnonzero(candidate)
    if not len(indices):
        return selected
    breaks = np.where((np.diff(indices) > 1) | (np.diff(flight.time_s[indices]) > 1.5))[
        0
    ]
    for run in np.split(indices, breaks + 1):
        if (
            len(run)
            and flight.time_s[run[-1]] - flight.time_s[run[0]] >= MINIMUM_STABLE_RUN_S
        ):
            selected[run] = True
    return selected


def _isa_deviation_k(
    pressure_altitude_m: np.ndarray, static_temperature_k: np.ndarray
) -> np.ndarray:
    isa = np.maximum(aero.T0 + aero.beta * pressure_altitude_m, 216.65)
    return static_temperature_k - isa


def _frontier_points(flights: Iterable[EnvelopeFlight], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        stable = _stable_capacity_mask(flight)
        altitude_ft = flight.pressure_altitude_m / FT_TO_M
        bins = np.floor(altitude_ft / ALTITUDE_BIN_FT).astype(int)
        for altitude_bin in np.unique(bins[stable]):
            current = stable & (bins == altitude_bin)
            count = int(np.count_nonzero(current))
            if count < MINIMUM_BIN_SAMPLES:
                continue
            values = flight.n1_percent[current]
            condition_threshold = np.quantile(values, 1.0 - UPPER_CONDITION_FRACTION)
            local = current & (flight.n1_percent >= condition_threshold)
            rows.append(
                {
                    "registration": flight.registration,
                    "recording": flight.recording,
                    "role": flight.role,
                    "split": split,
                    "altitude_bin_ft": int(altitude_bin * ALTITUDE_BIN_FT),
                    "support_samples": count,
                    "pressure_altitude_m": float(
                        np.median(flight.pressure_altitude_m[local])
                    ),
                    "mach": float(np.median(flight.mach[local])),
                    "static_temperature_k": float(
                        np.median(flight.static_temperature_k[local])
                    ),
                    "isa_temperature_deviation_k": float(
                        np.median(
                            _isa_deviation_k(
                                flight.pressure_altitude_m[local],
                                flight.static_temperature_k[local],
                            )
                        )
                    ),
                    "target_n1_percent": float(
                        np.quantile(values, WITHIN_BIN_FRONTIER_QUANTILE)
                    ),
                }
            )
    if not rows:
        raise ValueError(f"no demonstrated-capacity points for {split}")
    return (
        pd.DataFrame(rows)
        .sort_values(["registration", "recording", "altitude_bin_ft"], kind="stable")
        .reset_index(drop=True)
    )


def _design(points: pd.DataFrame, feature_set: str) -> np.ndarray:
    return n1_envelope_features(
        pressure_altitude_m=points["pressure_altitude_m"].to_numpy(dtype=float),
        mach=points["mach"].to_numpy(dtype=float),
        static_temperature_k=points["static_temperature_k"].to_numpy(dtype=float),
        feature_set=feature_set,
    )


def _flight_equal_weights(points: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(points), dtype=float)
    recordings = points["recording"].unique()
    for recording in recordings:
        current = points["recording"].eq(recording).to_numpy()
        weights[current] = 1.0 / (len(recordings) * np.count_nonzero(current))
    return weights


def _fit_quantile_frontier(points: pd.DataFrame, feature_set: str) -> np.ndarray:
    design = _design(points, feature_set)
    target = points["target_n1_percent"].to_numpy(dtype=float)
    weights = _flight_equal_weights(points)
    count, feature_count = design.shape
    # target - X beta = positive_residual - negative_residual
    objective = np.concatenate(
        (
            np.zeros(feature_count),
            ACROSS_FLIGHT_FRONTIER_QUANTILE * weights,
            (1.0 - ACROSS_FLIGHT_FRONTIER_QUANTILE) * weights,
        )
    )
    equality = np.column_stack((design, np.eye(count), -np.eye(count)))
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=target,
        bounds=([(None, None)] * feature_count + [(0.0, None)] * (2 * count)),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"quantile frontier fit failed: {result.message}")
    return np.asarray(result.x[:feature_count], dtype=float)


def _candidate_metrics(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recordings = points["recording"].unique()
    for feature_set, names in N1_ENVELOPE_FEATURE_SETS.items():
        residuals: list[np.ndarray] = []
        fold_rmse: list[float] = []
        for held_recording in recordings:
            training = points.loc[points["recording"] != held_recording]
            held = points.loc[points["recording"] == held_recording]
            coefficients = _fit_quantile_frontier(training, feature_set)
            residual = _design(held, feature_set) @ coefficients - held[
                "target_n1_percent"
            ].to_numpy(dtype=float)
            residuals.append(residual)
            fold_rmse.append(float(np.sqrt(np.mean(residual**2))))
        combined = np.concatenate(residuals)
        rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(names),
                "lofo_mean_flight_frontier_rmse_percent": float(np.mean(fold_rmse)),
                "lofo_frontier_bias_percent": float(np.mean(combined)),
                "lofo_frontier_underprediction_fraction": float(
                    np.mean(combined < 0.0)
                ),
                "lofo_frontier_residual_p05_percent": float(
                    np.quantile(combined, 0.05)
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    best = candidates["lofo_mean_flight_frontier_rmse_percent"].min()
    eligible = candidates.loc[
        candidates["lofo_mean_flight_frontier_rmse_percent"]
        <= (1.0 + FIT_NEAR_BEST_FRACTION) * best
    ]
    return eligible.sort_values(
        ["feature_count", "lofo_mean_flight_frontier_rmse_percent"],
        kind="stable",
    ).iloc[0]


def _bootstrap_coefficients(points: pd.DataFrame, feature_set: str) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    recordings = points["recording"].unique()
    coefficients: list[np.ndarray] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        parts: list[pd.DataFrame] = []
        for index, recording in enumerate(
            rng.choice(recordings, len(recordings), replace=True)
        ):
            current = points.loc[points["recording"] == recording].copy()
            current["recording"] = f"{recording}-bootstrap-{index}"
            parts.append(current)
        coefficients.append(
            _fit_quantile_frontier(pd.concat(parts, ignore_index=True), feature_set)
        )
    return np.vstack(coefficients)


def _capacity_domain(flights: Iterable[EnvelopeFlight]) -> dict[str, float]:
    altitude: list[np.ndarray] = []
    mach: list[np.ndarray] = []
    deviation: list[np.ndarray] = []
    for flight in flights:
        stable = _stable_capacity_mask(flight)
        altitude.append(flight.pressure_altitude_m[stable])
        mach.append(flight.mach[stable])
        deviation.append(
            _isa_deviation_k(
                flight.pressure_altitude_m[stable],
                flight.static_temperature_k[stable],
            )
        )
    return {
        "minimum_pressure_altitude_m": float(np.min(np.concatenate(altitude))),
        "maximum_pressure_altitude_m": float(np.max(np.concatenate(altitude))),
        "minimum_mach": float(np.min(np.concatenate(mach))),
        "maximum_mach": float(np.max(np.concatenate(mach))),
        "minimum_isa_temperature_deviation_k": float(np.min(np.concatenate(deviation))),
        "maximum_isa_temperature_deviation_k": float(np.max(np.concatenate(deviation))),
    }


def _load_old_envelope(path: Path) -> EffectiveN1Envelope:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    feature_set = str(model["feature_set"])
    order = N1_ENVELOPE_FEATURE_SETS[feature_set]
    return EffectiveN1Envelope(
        feature_set=feature_set,
        idle_coefficients=np.array(
            [model["idle_coefficients"][name] for name in order]
        ),
        climb_coefficients=np.array(
            [model["climb_coefficients"][name] for name in order]
        ),
    )


def _load_thrust_map(path: Path) -> EffectiveThrustMap:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    feature_set = str(model["feature_set"])
    order = THRUST_MAP_FEATURE_SETS[feature_set]
    return EffectiveThrustMap(
        rated_total_thrust_n=float(model["rated_total_thrust_n"]),
        pressure_exponent=float(model["pressure_exponent"]),
        feature_set=feature_set,
        corrected_fan_speed=bool(model["corrected_fan_speed"]),
        coefficients=np.array(
            [model["scenarios"]["nominal"]["coefficients"][name] for name in order]
        ),
    )


def _capacity_values(
    model: DemonstratedClimbN1Capacity,
    flight: EnvelopeFlight,
    mask: np.ndarray,
) -> np.ndarray:
    return model.capacity_percent(
        pressure_altitude_m=flight.pressure_altitude_m[mask],
        mach=flight.mach[mask],
        static_temperature_k=flight.static_temperature_k[mask],
    )


def _thrust_metrics_by_flight(
    *,
    selection: pd.DataFrame,
    envelope_flights: list[EnvelopeFlight],
    capacity: DemonstratedClimbN1Capacity,
    thrust_map: EffectiveThrustMap,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    vertical = _read_vertical_calibration(args.vertical_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    if aircraft.wing is None or aircraft.drag is None:
        raise ValueError("A320 definition lacks wing or drag data")
    propulsion_flights = _load_propulsion_flights(
        selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=float(aircraft.wing.area),
    )
    envelope_by_recording = {flight.recording: flight for flight in envelope_flights}
    result: dict[str, dict[str, float]] = {}
    for flight in propulsion_flights:
        envelope = envelope_by_recording[flight.recording]
        stable = _stable_capacity_mask(envelope)
        raw_indices = np.searchsorted(envelope.time_s, flight.time_s)
        valid = (raw_indices < len(envelope.time_s)) & np.isclose(
            envelope.time_s[np.minimum(raw_indices, len(envelope.time_s) - 1)],
            flight.time_s,
        )
        select = np.zeros(len(flight.time_s), dtype=bool)
        select[valid] = stable[raw_indices[valid]]
        support = capacity.support_mask(
            pressure_altitude_m=flight.pressure_altitude_m,
            mach=flight.mach,
            static_temperature_k=flight.static_temperature_k,
        )
        select &= support
        if not np.any(select):
            raise ValueError(f"no aligned high-power samples for {flight.key}")
        capacity_n1 = capacity.capacity_percent(
            pressure_altitude_m=flight.pressure_altitude_m[select],
            mach=flight.mach[select],
            static_temperature_k=flight.static_temperature_k[select],
        )
        high = flight.n1_percent[select] >= capacity_n1 - HIGH_N1_MAP_MARGIN_PERCENT
        selected_indices = np.flatnonzero(select)[high]
        if len(selected_indices) < 20:
            raise ValueError(f"too few high-N1 thrust samples for {flight.key}")
        prediction = _predict_thrust(thrust_map, flight)[selected_indices]
        target = _target_thrust_n(
            flight,
            cd0=reference_drag.zero_lift_drag_coefficient,
            induced_drag_factor=reference_drag.induced_drag_factor,
            cd0_scale=1.0,
            k_scale=1.0,
        )[selected_indices]
        residual = prediction - target
        result[flight.recording] = {
            "high_n1_thrust_samples": len(selected_indices),
            "high_n1_thrust_rmse_kn": float(np.sqrt(np.mean(residual**2)) / 1000.0),
            "high_n1_thrust_bias_kn": float(np.mean(residual) / 1000.0),
            "high_n1_target_thrust_median_kn": float(np.median(target) / 1000.0),
            "high_n1_predicted_thrust_median_kn": float(np.median(prediction) / 1000.0),
        }
    return result


def _flight_metrics(
    *,
    flights: list[EnvelopeFlight],
    points: pd.DataFrame,
    capacity: DemonstratedClimbN1Capacity,
    old_envelope: EffectiveN1Envelope,
    thrust_metrics: dict[str, dict[str, float]],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        raw_stable = _stable_capacity_mask(flight)
        support = capacity.support_mask(
            pressure_altitude_m=flight.pressure_altitude_m,
            mach=flight.mach,
            static_temperature_k=flight.static_temperature_k,
        )
        stable = raw_stable & support
        if np.count_nonzero(stable) < MINIMUM_BIN_SAMPLES:
            raise ValueError(f"too few supported stable samples for {flight.key}")
        predicted = _capacity_values(capacity, flight, stable)
        observed = flight.n1_percent[stable]
        excess = observed - predicted
        old = old_envelope.climb_limit_percent(
            pressure_altitude_m=flight.pressure_altitude_m[stable],
            mach=flight.mach[stable],
            static_temperature_k=flight.static_temperature_k[stable],
        )
        current_points = points.loc[points["recording"] == flight.recording].copy()
        point_support = capacity.support_mask(
            pressure_altitude_m=current_points["pressure_altitude_m"],
            mach=current_points["mach"],
            static_temperature_k=current_points["static_temperature_k"],
        )
        current_points = current_points.loc[point_support]
        if current_points.empty:
            raise ValueError(f"no supported frontier points for {flight.key}")
        point_prediction = capacity.capacity_percent(
            pressure_altitude_m=current_points["pressure_altitude_m"],
            mach=current_points["mach"],
            static_temperature_k=current_points["static_temperature_k"],
        )
        point_residual = point_prediction - current_points[
            "target_n1_percent"
        ].to_numpy(dtype=float)
        row = {
            "registration": flight.registration,
            "recording": flight.recording,
            "role": flight.role,
            "split": split,
            "stable_clb_samples": int(np.count_nonzero(raw_stable)),
            "supported_stable_samples": int(np.count_nonzero(stable)),
            "supported_stable_fraction": float(
                np.count_nonzero(stable) / np.count_nonzero(raw_stable)
            ),
            "frontier_points": len(current_points),
            "sample_coverage_fraction": float(np.mean(excess <= 0.0)),
            "n1_excess_p95_percent": float(np.quantile(excess, 0.95)),
            "n1_excess_max_percent": float(np.max(excess)),
            "frontier_rmse_percent": float(np.sqrt(np.mean(point_residual**2))),
            "frontier_bias_percent": float(np.mean(point_residual)),
            "frontier_quantile_coverage_fraction": float(
                np.mean(point_residual >= 0.0)
            ),
            "frontier_pinball_loss_percent": float(
                np.mean(
                    np.where(
                        point_residual < 0.0,
                        ACROSS_FLIGHT_FRONTIER_QUANTILE * -point_residual,
                        (1.0 - ACROSS_FLIGHT_FRONTIER_QUANTILE) * point_residual,
                    )
                )
            ),
            "frontier_supported_fraction": float(
                np.mean(point_residual <= FRONTIER_SUPPORT_TOLERANCE_PERCENT)
            ),
            "frontier_overclaim_p95_percent": float(np.quantile(point_residual, 0.95)),
            "frontier_overclaim_max_percent": float(np.max(point_residual)),
            "capacity_increment_over_old_median_percent": float(
                np.median(predicted - old)
            ),
            "capacity_increment_over_old_p05_percent": float(
                np.quantile(predicted - old, 0.05)
            ),
        }
        row.update(thrust_metrics[flight.recording])
        rows.append(row)
    return pd.DataFrame(rows)


def _registration_frontier_bias_gap(metrics: pd.DataFrame) -> float:
    by_registration = metrics.groupby("registration")["frontier_bias_percent"].median()
    return float(by_registration.max() - by_registration.min())


def _point_weighted_mean(metrics: pd.DataFrame, column: str) -> float:
    return float(
        np.average(
            metrics[column].to_numpy(dtype=float),
            weights=metrics["frontier_points"].to_numpy(dtype=float),
        )
    )


def _point_weighted_rmse(metrics: pd.DataFrame) -> float:
    return float(
        np.sqrt(
            np.average(
                metrics["frontier_rmse_percent"].to_numpy(dtype=float) ** 2,
                weights=metrics["frontier_points"].to_numpy(dtype=float),
            )
        )
    )


def _gate_results(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    validation = metrics.loc[metrics["split"] == "validation"]
    holdout = metrics.loc[metrics["split"] == "holdout"]
    values = {
        "validation_worst_supported_stable_fraction": float(
            validation["supported_stable_fraction"].min()
        ),
        "holdout_worst_supported_stable_fraction": float(
            holdout["supported_stable_fraction"].min()
        ),
        "validation_frontier_quantile_coverage": _point_weighted_mean(
            validation, "frontier_quantile_coverage_fraction"
        ),
        "holdout_frontier_quantile_coverage": _point_weighted_mean(
            holdout, "frontier_quantile_coverage_fraction"
        ),
        "validation_frontier_pinball_loss_percent": _point_weighted_mean(
            validation, "frontier_pinball_loss_percent"
        ),
        "holdout_frontier_pinball_loss_percent": _point_weighted_mean(
            holdout, "frontier_pinball_loss_percent"
        ),
        "validation_frontier_rmse_percent": _point_weighted_rmse(validation),
        "holdout_frontier_rmse_percent": _point_weighted_rmse(holdout),
        "validation_registration_frontier_bias_gap_percent": (
            _registration_frontier_bias_gap(validation)
        ),
        "validation_median_high_n1_thrust_rmse_kn": float(
            validation["high_n1_thrust_rmse_kn"].median()
        ),
        "holdout_median_high_n1_thrust_rmse_kn": float(
            holdout["high_n1_thrust_rmse_kn"].median()
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        limits = GATE_LIMITS[name]
        passed = True
        if "minimum" in limits:
            passed &= value >= limits["minimum"]
        if "maximum" in limits:
            passed &= value <= limits["maximum"]
        results[name] = {"value": value, **limits, "passed": bool(passed)}
    return results


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["number"]):
        if column in {
            "feature_count",
            "altitude_bin_ft",
            "support_samples",
            "stable_clb_samples",
            "supported_stable_samples",
            "frontier_points",
            "high_n1_thrust_samples",
        }:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(8)
    return result


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    strings = frame.astype(str)
    headers = list(strings.columns)
    rows = strings.values.tolist()
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            headers[index].ljust(widths[index]) for index in range(len(headers))
        )
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _plot(
    *,
    points: pd.DataFrame,
    metrics: pd.DataFrame,
    capacity: DemonstratedClimbN1Capacity,
    old_envelope: EffectiveN1Envelope,
    thrust_map: EffectiveThrustMap,
    figure_prefix: Path,
) -> None:
    del thrust_map
    colors = {
        "fit": "#0072B2",
        "validation": "#E69F00",
        "diagnostic": "#CC79A7",
        "confirmation": "#56B4E9",
        "holdout": "#009E73",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for split, current in points.groupby("split", sort=False):
        supported = capacity.support_mask(
            pressure_altitude_m=current["pressure_altitude_m"],
            mach=current["mach"],
            static_temperature_k=current["static_temperature_k"],
        )
        current = current.loc[supported]
        prediction = capacity.capacity_percent(
            pressure_altitude_m=current["pressure_altitude_m"],
            mach=current["mach"],
            static_temperature_k=current["static_temperature_k"],
        )
        ax.scatter(
            current["target_n1_percent"],
            prediction,
            s=18,
            alpha=0.7,
            color=colors[split],
            label=split,
        )
    limits = (87.5, 91.5)
    ax.plot(limits, limits, color="black", linewidth=1.0, linestyle="--")
    ax.set(
        xlim=limits,
        ylim=limits,
        xlabel="Observed bin frontier N1 (%)",
        ylabel="Demonstrated capacity N1 (%)",
    )
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    order = metrics["recording"].tolist()
    x = np.arange(len(order))
    for split, current in metrics.groupby("split", sort=False):
        indices = current.index.to_numpy()
        ax.scatter(
            indices,
            current["sample_coverage_fraction"] * 100.0,
            s=35,
            color=colors[split],
            label=split,
        )
    ax.axhline(95.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, order, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Stable CLB samples below capacity (%)")
    ax.set_ylim(50.0, 101.0)

    ax = axes[1, 0]
    altitude_m = np.linspace(
        capacity.minimum_pressure_altitude_m,
        capacity.maximum_pressure_altitude_m,
        180,
    )
    fraction = (altitude_m - altitude_m.min()) / np.ptp(altitude_m)
    mach = capacity.minimum_mach + fraction * (
        capacity.maximum_mach - capacity.minimum_mach
    )
    deviation = 0.5 * (
        capacity.minimum_isa_temperature_deviation_k
        + capacity.maximum_isa_temperature_deviation_k
    )
    temperature = np.maximum(aero.T0 + aero.beta * altitude_m, 216.65) + deviation
    new = capacity.capacity_percent(
        pressure_altitude_m=altitude_m,
        mach=mach,
        static_temperature_k=temperature,
    )
    old = old_envelope.climb_limit_percent(
        pressure_altitude_m=altitude_m,
        mach=mach,
        static_temperature_k=temperature,
    )
    ax.plot(
        altitude_m / FT_TO_M / 1000.0, new, color="#009E73", label="Demonstrated LCB"
    )
    ax.plot(
        altitude_m / FT_TO_M / 1000.0,
        old,
        color="#0072B2",
        linestyle="--",
        label="Old operational q90",
    )
    ax.set(xlabel="Pressure altitude (kft)", ylabel="CLB N1 boundary (%)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for split, current in metrics.groupby("split", sort=False):
        ax.scatter(
            current["capacity_increment_over_old_median_percent"],
            current["high_n1_thrust_rmse_kn"],
            s=40,
            color=colors[split],
            label=split,
        )
    ax.axhline(
        GATE_LIMITS["validation_median_high_n1_thrust_rmse_kn"]["maximum"],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    ax.set(
        xlabel="Median capacity increment over old q90 (N1 points)",
        ylabel="High-N1 thrust-map RMSE (kN)",
    )

    for label, ax in zip("ABCD", axes.ravel(), strict=True):
        ax.text(
            -0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=11
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.5)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(figure_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    candidates: pd.DataFrame,
    chosen: pd.Series,
    metrics: pd.DataFrame,
    gates: dict[str, dict[str, Any]],
) -> None:
    compact_candidates = _round_frame(
        candidates[
            [
                "feature_set",
                "feature_count",
                "lofo_mean_flight_frontier_rmse_percent",
                "lofo_frontier_underprediction_fraction",
            ]
        ]
    )
    compact_metrics = _round_frame(
        metrics[
            [
                "registration",
                "recording",
                "split",
                "sample_coverage_fraction",
                "supported_stable_fraction",
                "frontier_supported_fraction",
                "frontier_overclaim_p95_percent",
                "frontier_rmse_percent",
                "capacity_increment_over_old_median_percent",
                "high_n1_thrust_rmse_kn",
            ]
        ]
    )
    gate_frame = _round_frame(
        pd.DataFrame(
            [{"gate": name, **values} for name, values in gates.items()]
        ).fillna("")
    )
    passed = all(item["passed"] for item in gates.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# A320 demonstrated climb-capacity gate

## Decision

The gate **{"passes" if passed else "does not pass"}**. The released quantity is a
one-sided lower confidence bound on an operationally demonstrated, stable CLB
N1 frontier. It is not maximum available thrust, an Airbus FADEC schedule, a
CFM56 rating deck, or evidence about TOGA/FLEX/MCT.

The selected `{chosen["feature_set"]}` surface is fit only to the ten existing
development/calibration flights. Six complete validation flights were loaded
after selection. `FLT0004` remains a diagnostic. An exploratory gate had
incorrectly required the conservative bound to exceed ordinary operating N1;
that gate stayed closed. `FLT0006`, `FLT0007`, `FLT0008`, and `FLT0009` are
therefore retained as confirmation data for the corrected quantile-calibration
gate, while the predeclared `TAIL-B/FLT0001` and `TAIL-B/FLT0002` flights are
the untouched final holdout.

## Frozen estimator

- Eligible samples are airborne, clean-flap, raw TLA code `{DOMINANT_CLB_CODE:g}`
  climbs above {MINIMUM_CLIMB_RATE_FT_S:g} ft/s, with paired-engine N1 agreement
  within {MAXIMUM_ENGINE_DISAGREEMENT_PERCENT:g} point.
- Mean N1 must be stable within
  {MAXIMUM_ABSOLUTE_N1_RATE_PERCENT_S:g} point/s for a contiguous
  {MINIMUM_STABLE_RUN_S:g}-second run.
- Each complete-flight/{ALTITUDE_BIN_FT:g}-ft bin contributes its
  {WITHIN_BIN_FRONTIER_QUANTILE:.0%} N1 frontier.
- A flight-equal {ACROSS_FLIGHT_FRONTIER_QUANTILE:.0%} quantile regression fits
  the conditional upper frontier.
- {BOOTSTRAP_REPLICATES} complete-flight bootstrap fits quantify uncertainty;
  planning uses their {LOWER_CONFIDENCE_QUANTILE:.0%} prediction quantile.
- Requests outside the development-flight altitude, Mach, or ISA-temperature-
  deviation range used by the selected feature set are rejected rather than
  extrapolated.
- Independent frontier points must calibrate as a high conditional quantile:
  the observed-frontier-below-bound fraction and q90 pinball loss are gated in
  aggregate, while complete-flight results remain visible. Ordinary-sample
  coverage is diagnostic, not a release gate.

{_markdown_table(compact_candidates)}

## Complete-flight results

{_markdown_table(compact_metrics)}

## Release gates

{_markdown_table(gate_frame)}

The high-N1 thrust check is intentionally separate from N1-frontier coverage.
It compares the already-frozen N1-to-effective-thrust map with QAR force/energy
targets only for stable samples within {HIGH_N1_MAP_MARGIN_PERCENT:g} N1 point
of the demonstrated bound. No thrust-map coefficient is refit here.

## Scope and next use

If released, this model may replace the old 90th-percentile operational CLB
surface only inside its declared support for A320 energy-profile planning. The
old surface remains useful as an operational demand model; neither surface is
a physical maximum. The immediate downstream test is the unchanged frozen
nine-anchor `FLT0004` intent and controller, with only the planning-capacity
source changed and with a new actual-versus-generated trajectory figure.
"""
    )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[
        panel["role"] != "validation", ["registration", "recording", "role"]
    ].copy()
    validation_selection = panel.loc[
        panel["role"] == "validation", ["registration", "recording", "role"]
    ].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected ten fit and six validation flights")
    consumed = set(panel["recording"])
    if consumed & set(DIAGNOSTIC_SELECTION["recording"]):
        raise ValueError("diagnostic flight overlaps the replay panel")
    if consumed & set(CONFIRMATION_SELECTION["recording"]):
        raise ValueError("confirmation flights overlap the replay panel")
    if consumed & set(HOLDOUT_SELECTION["recording"]):
        raise ValueError("holdout flights overlap the replay panel")
    if set(CONFIRMATION_SELECTION["recording"]) & set(HOLDOUT_SELECTION["recording"]):
        raise ValueError("confirmation and holdout flights overlap")

    fit_flights = _load_envelope_flights(
        fit_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    fit_points = _frontier_points(fit_flights, "fit")
    candidates = _candidate_metrics(fit_points)
    chosen = _select_candidate(candidates)
    feature_set = str(chosen["feature_set"])
    central_coefficients = _fit_quantile_frontier(fit_points, feature_set)
    bootstrap_coefficients = _bootstrap_coefficients(fit_points, feature_set)
    domain = _capacity_domain(fit_flights)
    capacity = DemonstratedClimbN1Capacity(
        feature_set=feature_set,
        bootstrap_coefficients=bootstrap_coefficients,
        lower_confidence_quantile=LOWER_CONFIDENCE_QUANTILE,
        **domain,
    )
    old_envelope = _load_old_envelope(args.n1_envelope)
    thrust_map = _load_thrust_map(args.thrust_parameters)

    # All structure, masks, quantiles, thresholds, and feature choice are frozen.
    validation_flights = _load_envelope_flights(
        validation_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    validation_points = _frontier_points(validation_flights, "validation")
    diagnostic_flights = _load_envelope_flights(
        DIAGNOSTIC_SELECTION, raw_dir=args.raw_dir, processed_path=args.processed
    )
    diagnostic_points = _frontier_points(diagnostic_flights, "diagnostic")
    confirmation_flights = _load_envelope_flights(
        CONFIRMATION_SELECTION, raw_dir=args.raw_dir, processed_path=args.processed
    )
    confirmation_points = _frontier_points(confirmation_flights, "confirmation")
    # The corrected gate is frozen before these predeclared flights are opened.
    holdout_flights = _load_envelope_flights(
        HOLDOUT_SELECTION, raw_dir=args.raw_dir, processed_path=args.processed
    )
    holdout_points = _frontier_points(holdout_flights, "holdout")

    flight_groups = [
        ("fit", fit_selection, fit_flights, fit_points),
        ("validation", validation_selection, validation_flights, validation_points),
        ("diagnostic", DIAGNOSTIC_SELECTION, diagnostic_flights, diagnostic_points),
        (
            "confirmation",
            CONFIRMATION_SELECTION,
            confirmation_flights,
            confirmation_points,
        ),
        ("holdout", HOLDOUT_SELECTION, holdout_flights, holdout_points),
    ]
    metric_frames: list[pd.DataFrame] = []
    for split, selection, flights, points in flight_groups:
        thrust_metrics = _thrust_metrics_by_flight(
            selection=selection,
            envelope_flights=flights,
            capacity=capacity,
            thrust_map=thrust_map,
            args=args,
        )
        metric_frames.append(
            _flight_metrics(
                flights=flights,
                points=points,
                capacity=capacity,
                old_envelope=old_envelope,
                thrust_metrics=thrust_metrics,
                split=split,
            )
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    points = pd.concat(
        (
            fit_points,
            validation_points,
            diagnostic_points,
            confirmation_points,
            holdout_points,
        ),
        ignore_index=True,
    )
    gates = _gate_results(metrics)
    passed = all(item["passed"] for item in gates.values())
    feature_names = N1_ENVELOPE_FEATURE_SETS[feature_set]
    document: dict[str, Any] = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "released-demonstrated-clb-capacity" if passed else "gate-held",
        "interpretation": {
            "quantity": "one-sided lower confidence bound on demonstrated stable CLB N1 frontier",
            "not": [
                "physical maximum available thrust",
                "Airbus FADEC schedule",
                "CFM56 engine rating deck",
                "TOGA FLEX or MCT model",
            ],
        },
        "leakage_control": {
            "fit_flights_loaded_for_selection": [flight.key for flight in fit_flights],
            "method_frozen_before_validation_load": True,
            "validation_flights_loaded_after_freeze": [
                flight.key for flight in validation_flights
            ],
            "diagnostic_loaded_after_validation": [
                flight.key for flight in diagnostic_flights
            ],
            "exploratory_confirmation_flights": [
                flight.key for flight in confirmation_flights
            ],
            "corrected_gate_frozen_before_final_holdout_load": True,
            "predeclared_holdout_rule": "next two long AFZ archive flights selected from recording IDs and row counts only after exploratory holdout was consumed",
            "holdout_flights_loaded_last_once": [
                flight.key for flight in holdout_flights
            ],
        },
        "model": {
            "feature_set": feature_set,
            "feature_order": list(feature_names),
            "central_frontier_coefficients": dict(
                zip(feature_names, central_coefficients.tolist(), strict=True)
            ),
            "bootstrap_coefficients": bootstrap_coefficients.tolist(),
            "lower_confidence_quantile": LOWER_CONFIDENCE_QUANTILE,
            "domain": domain,
        },
        "method": {
            "sample_rate_hz": 1,
            "dominant_clb_tla_code": DOMINANT_CLB_CODE,
            "clean_flap_code": 0,
            "minimum_climb_rate_ft_s": MINIMUM_CLIMB_RATE_FT_S,
            "maximum_engine_n1_disagreement_percent": MAXIMUM_ENGINE_DISAGREEMENT_PERCENT,
            "n1_derivative_window_s": N1_DERIVATIVE_WINDOW_S,
            "maximum_absolute_n1_rate_percent_s": MAXIMUM_ABSOLUTE_N1_RATE_PERCENT_S,
            "minimum_stable_run_s": MINIMUM_STABLE_RUN_S,
            "altitude_bin_ft": ALTITUDE_BIN_FT,
            "minimum_bin_samples": MINIMUM_BIN_SAMPLES,
            "within_bin_frontier_quantile": WITHIN_BIN_FRONTIER_QUANTILE,
            "across_flight_frontier_quantile": ACROSS_FLIGHT_FRONTIER_QUANTILE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "complete flight",
            "planning_bound": f"pointwise bootstrap q{LOWER_CONFIDENCE_QUANTILE:g}",
        },
        "selection": _native(chosen.to_dict()),
        "gates": gates,
        "release_decision": {
            "passed": passed,
            "planner_integration_authorized": passed,
        },
        "provenance": {
            "qar": "operator A320 raw QAR archive at data/a320_raw",
            "effective_thrust_map": str(args.thrust_parameters),
            "old_operational_clb_surface": str(args.n1_envelope),
            "source_id": "aerox_a320_demonstrated_climb_capacity_2026_08_06",
        },
        "artifacts": {
            "runner": "tools/identify_a320_demonstrated_climb_capacity.py",
            "points": str(args.points),
            "candidate_metrics": str(args.candidate_metrics),
            "flight_metrics": str(args.flight_metrics),
            "report": str(args.report),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
        "limitations": [
            "TLA and N1 recorder conversions remain provisional.",
            "Bleed, anti-ice, engine health, and commanded rating are absent.",
            "The effective thrust target remains conditional on the drag polar and QAR derivatives.",
            "The tropical QAR temperature domain does not support ISA-temperature extrapolation.",
        ],
    }

    args.points.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(points).to_csv(args.points, index=False)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(metrics).to_csv(args.flight_metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(_native(document), sort_keys=False))
    _write_report(
        args.report,
        candidates=candidates,
        chosen=chosen,
        metrics=metrics,
        gates=gates,
    )
    _plot(
        points=points,
        metrics=metrics,
        capacity=capacity,
        old_envelope=old_envelope,
        thrust_map=thrust_map,
        figure_prefix=args.figure_prefix,
    )
    return candidates, metrics, document


def main() -> None:
    candidates, metrics, document = run(parse_args())
    print(candidates.to_string(index=False))
    print(metrics.to_string(index=False))
    print(yaml.safe_dump(document["gates"], sort_keys=False))


if __name__ == "__main__":
    main()
