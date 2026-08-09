#!/usr/bin/env python3
"""Calibrate an effective A320 N1-to-thrust map with flight-level validation.

Model structure is selected using only the ten development/calibration flights.
The six validation flights are loaded only after the structure and all fitting
rules have been frozen.  Thrust targets remain conditional on a drag-polar
prior; fuel flow is never used to fit or select the thrust map.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from aerox import aero, load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    KNOT_TO_MPS,
    decode_a320_flight,
    raw_columns,
)
from aerox.models.atmosphere import (
    dynamic_pressure_from_calibrated_airspeed,
    mach_from_calibrated_airspeed,
)
from aerox.modeling.identification import (
    delay_samples,
    robust_weighted_linear_fit,
)
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    EffectiveThrustMap,
    thrust_map_features,
)
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.modeling.force_reconstruction import reconstruct_translational_forces


RAW_FILES = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}
PROCESSED_COLUMNS = ["flight_id", "timestamp", "TAS", "temperature"]
DERIVATIVE_WINDOW_S = 21
FIT_THINNING_S = 5
MAX_ABSOLUTE_BANK_DEG = 10.0
MIN_TRUE_AIRSPEED_MPS = 70.0
MAX_MACH = 0.85
MIN_STATIC_TEMPERATURE_K = 180.0
MAX_STATIC_TEMPERATURE_K = 330.0
PRESSURE_EXPONENTS = (0.5, 0.75, 1.0, 1.25)
DRAG_SCENARIOS = {
    "nominal": (1.0, 1.0),
    "cd0_low": (0.8, 1.0),
    "cd0_high": (1.2, 1.0),
    "k_low": (1.0, 0.8),
    "k_high": (1.0, 1.2),
}


@dataclass(frozen=True)
class VerticalCalibration:
    pitch_scale_deg_per_code: float
    angle_of_attack_scale_deg_per_code: float
    effective_gamma_bias_deg: float
    delay_s: int


@dataclass(frozen=True)
class PropulsionFlight:
    registration: str
    recording: str
    role: str
    sample_index: np.ndarray
    time_s: np.ndarray
    true_airspeed_mps: np.ndarray
    flight_path_angle_rad: np.ndarray
    pressure_altitude_m: np.ndarray
    mass_kg: np.ndarray
    bank_angle_rad: np.ndarray
    angle_of_attack_rad: np.ndarray
    n1_percent: np.ndarray
    mach: np.ndarray
    pressure_ratio: np.ndarray
    static_temperature_k: np.ndarray
    fuel_flow_kg_s: np.ndarray
    net_along_path_force_n: np.ndarray
    parasite_drag_basis_n: np.ndarray
    induced_drag_basis_n: np.ndarray
    lift_coefficient: np.ndarray
    phase: np.ndarray
    n1_engine_abs_difference_percent: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.registration}/{self.recording}"


@dataclass(frozen=True)
class ModelSettings:
    pressure_exponent: float
    feature_set: str
    corrected_fan_speed: bool


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
        "--engine-database",
        type=Path,
        default=Path("old/scripts/root/db/engines.csv"),
    )
    parser.add_argument(
        "--fuel-model-database",
        type=Path,
        default=Path("../openap/openap/data/fuel/fuel_models.csv"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-candidates.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.md"),
    )
    return parser.parse_args()


def _read_vertical_calibration(path: Path) -> VerticalCalibration:
    data = yaml.safe_load(path.read_text())
    parameters = data["model"]["parameters"]
    return VerticalCalibration(
        pitch_scale_deg_per_code=float(parameters["pitch_scale_deg_per_code"]["value"]),
        angle_of_attack_scale_deg_per_code=float(
            parameters["angle_of_attack_scale_deg_per_code"]["value"]
        ),
        effective_gamma_bias_deg=float(parameters["effective_gamma_bias_deg"]["value"]),
        delay_s=int(data["model"]["trajectory_reference_delay_s"]["value"]),
    )


def _read_tas_scale(path: Path) -> float:
    data = yaml.safe_load(path.read_text())
    return float(data["model"]["parameters"]["tas_scale"]["value"])


def _read_csv_row(path: Path, column: str, value: str) -> dict[str, str]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row[column] == value]
    if len(rows) != 1:
        raise ValueError(
            f"expected one row with {column}={value!r} in {path}; found {len(rows)}"
        )
    return rows[0]


def _build_flight(
    *,
    registration: str,
    recording: str,
    role: str,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    vertical: VerticalCalibration,
    tas_scale: float,
    wing_area_m2: float,
) -> PropulsionFlight:
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    current = processed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = current["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
    if np.any(raw_indices >= len(decoded.timestamp_utc)) or not np.all(
        decoded.timestamp_utc[raw_indices] == timestamps
    ):
        raise ValueError(f"raw/processed mismatch for {registration}/{recording}")

    time_s = (timestamps - timestamps[0]).astype("timedelta64[ns]").astype(
        np.int64
    ) / 1e9
    pitch_code = (
        decoded.channel("pitch", allow_provisional=True)[raw_indices] / DEG_TO_RAD / 0.5
    )
    aoa_code = (
        decoded.channel("angle_of_attack", allow_provisional=True)[raw_indices]
        / DEG_TO_RAD
        / 0.25
    )
    gamma_deg = (
        vertical.pitch_scale_deg_per_code * delay_samples(pitch_code, vertical.delay_s)
        - vertical.angle_of_attack_scale_deg_per_code
        * delay_samples(aoa_code, vertical.delay_s)
        + vertical.effective_gamma_bias_deg
    )

    start = vertical.delay_s
    select = slice(start, None)
    sample_index = np.arange(len(timestamps), dtype=int)[select]
    time_s = time_s[select]
    gamma_deg = gamma_deg[select]
    gamma = np.deg2rad(gamma_deg)
    true_airspeed = (
        tas_scale * current["TAS"].to_numpy(dtype=float)[select] * KNOT_TO_MPS
    )
    temperature = current["temperature"].to_numpy(dtype=float)[select]
    cas = decoded.channel("calibrated_airspeed")[raw_indices][select]
    pressure_altitude = decoded.channel("pressure_altitude")[raw_indices][select]
    mass = decoded.channel("gross_weight")[raw_indices][select]
    bank = decoded.channel("roll", allow_provisional=True)[raw_indices][select]
    flap = decoded.channel("flap", allow_unresolved=True)[raw_indices][select]
    airborne = decoded.airborne[raw_indices][select]
    n1_1 = decoded.channel("n1_1", allow_provisional=True)[raw_indices][select]
    n1_2 = decoded.channel("n1_2", allow_provisional=True)[raw_indices][select]
    angle_of_attack = decoded.channel("angle_of_attack", allow_provisional=True)[
        raw_indices
    ][select]
    n1 = 0.5 * (n1_1 + n1_2)
    fuel_flow = (
        decoded.channel("fuel_flow_1")[raw_indices][select]
        + decoded.channel("fuel_flow_2")[raw_indices][select]
    )

    dynamic_pressure = dynamic_pressure_from_calibrated_airspeed(cas, pressure_altitude)
    mach = mach_from_calibrated_airspeed(cas, pressure_altitude)
    forces = reconstruct_translational_forces(
        time_s=time_s,
        true_airspeed_mps=true_airspeed,
        flight_path_angle_rad=gamma,
        bank_angle_rad=bank,
        mass_kg=mass,
        dynamic_pressure_pa=dynamic_pressure,
        wing_area_m2=wing_area_m2,
        zero_lift_drag_coefficient=0.0,
        induced_drag_factor=0.0,
        derivative_window=DERIVATIVE_WINDOW_S,
    )
    q_s = dynamic_pressure * wing_area_m2
    induced_basis = forces.required_lift_n**2 / q_s
    pressure_ratio = np.asarray(aero.pressure(pressure_altitude), dtype=float) / aero.p0
    clean = (
        np.isfinite(
            forces.net_along_path_force_n
            + forces.required_lift_n
            + n1
            + fuel_flow
            + mach
            + pressure_ratio
            + temperature
            + angle_of_attack
        )
        & airborne
        & np.isclose(flap, 0.0)
        & (np.abs(bank) < np.deg2rad(MAX_ABSOLUTE_BANK_DEG))
        & (true_airspeed > MIN_TRUE_AIRSPEED_MPS)
        & (mach < MAX_MACH)
        & (temperature > MIN_STATIC_TEMPERATURE_K)
        & (temperature < MAX_STATIC_TEMPERATURE_K)
        & (n1 >= 0.0)
        & (pressure_ratio > 0.0)
    )
    if np.count_nonzero(clean) < 100:
        raise ValueError(f"too few clean samples for {registration}/{recording}")
    phase = np.full(len(gamma_deg), "level", dtype="U7")
    phase[gamma_deg > 1.0] = "climb"
    phase[gamma_deg < -1.0] = "descent"
    return PropulsionFlight(
        registration=registration,
        recording=recording,
        role=role,
        sample_index=sample_index[clean],
        time_s=time_s[clean],
        true_airspeed_mps=true_airspeed[clean],
        flight_path_angle_rad=gamma[clean],
        pressure_altitude_m=pressure_altitude[clean],
        mass_kg=mass[clean],
        bank_angle_rad=bank[clean],
        angle_of_attack_rad=angle_of_attack[clean],
        n1_percent=n1[clean],
        mach=mach[clean],
        pressure_ratio=pressure_ratio[clean],
        static_temperature_k=temperature[clean],
        fuel_flow_kg_s=fuel_flow[clean],
        net_along_path_force_n=forces.net_along_path_force_n[clean],
        parasite_drag_basis_n=q_s[clean],
        induced_drag_basis_n=induced_basis[clean],
        lift_coefficient=forces.lift_coefficient[clean],
        phase=phase[clean],
        n1_engine_abs_difference_percent=np.abs(n1_1[clean] - n1_2[clean]),
    )


def _load_flights(
    selection: pd.DataFrame,
    *,
    raw_dir: Path,
    processed_path: Path,
    vertical: VerticalCalibration,
    tas_scale: float,
    wing_area_m2: float,
) -> list[PropulsionFlight]:
    identifiers = selection["recording"].astype(str).tolist()
    processed = pd.read_parquet(
        processed_path,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "in", identifiers)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame
        for recording, frame in processed.groupby("flight_id", sort=False)
    }
    flights: list[PropulsionFlight] = []
    for registration, current_selection in selection.groupby("registration", sort=True):
        current_ids = current_selection["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current_ids)],
            engine="pyarrow",
        )
        roles = current_selection.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            flights.append(
                _build_flight(
                    registration=str(registration),
                    recording=recording,
                    role=str(roles[recording]),
                    raw=frame,
                    processed=processed_groups[recording],
                    vertical=vertical,
                    tas_scale=tas_scale,
                    wing_area_m2=wing_area_m2,
                )
            )
    loaded = {flight.recording for flight in flights}
    if loaded != set(identifiers):
        raise ValueError(f"flight load mismatch: {sorted(set(identifiers) - loaded)}")
    return flights


def _target_thrust_n(
    flight: PropulsionFlight,
    *,
    cd0: float,
    induced_drag_factor: float,
    cd0_scale: float,
    k_scale: float,
) -> np.ndarray:
    return (
        flight.net_along_path_force_n
        + cd0_scale * cd0 * flight.parasite_drag_basis_n
        + k_scale * induced_drag_factor * flight.induced_drag_basis_n
    )


def _phase_equal_weights(phases: np.ndarray) -> np.ndarray:
    """Give each phase present in one flight equal total weight."""
    values = np.asarray(phases)
    weights = np.zeros(len(values), dtype=float)
    unique = np.unique(values)
    for phase in unique:
        mask = values == phase
        weights[mask] = 1.0 / (len(unique) * np.count_nonzero(mask))
    return weights


def _training_arrays(
    flights: Iterable[PropulsionFlight],
    *,
    cd0: float,
    induced_drag_factor: float,
    cd0_scale: float,
    k_scale: float,
) -> dict[str, np.ndarray]:
    selected = list(flights)
    arrays: dict[str, list[np.ndarray]] = {
        "n1": [],
        "mach": [],
        "delta": [],
        "temperature": [],
        "target": [],
        "weights": [],
    }
    for flight in selected:
        thin = flight.sample_index % FIT_THINNING_S == 0
        arrays["n1"].append(flight.n1_percent[thin])
        arrays["mach"].append(flight.mach[thin])
        arrays["delta"].append(flight.pressure_ratio[thin])
        arrays["temperature"].append(flight.static_temperature_k[thin])
        arrays["target"].append(
            _target_thrust_n(
                flight,
                cd0=cd0,
                induced_drag_factor=induced_drag_factor,
                cd0_scale=cd0_scale,
                k_scale=k_scale,
            )[thin]
        )
        arrays["weights"].append(_phase_equal_weights(flight.phase[thin]))
    result = {name: np.concatenate(values) for name, values in arrays.items()}
    # Every flight has equal total weight; phases are equal within each flight.
    result["weights"] /= len(selected)
    return result


def _fit_map(
    flights: Iterable[PropulsionFlight],
    *,
    settings: ModelSettings,
    rated_total_thrust_n: float,
    cd0: float,
    induced_drag_factor: float,
    cd0_scale: float,
    k_scale: float,
) -> tuple[EffectiveThrustMap, Any]:
    data = _training_arrays(
        flights,
        cd0=cd0,
        induced_drag_factor=induced_drag_factor,
        cd0_scale=cd0_scale,
        k_scale=k_scale,
    )
    design = thrust_map_features(
        n1_percent=data["n1"],
        mach=data["mach"],
        pressure_ratio=data["delta"],
        static_temperature_k=data["temperature"],
        pressure_exponent=settings.pressure_exponent,
        feature_set=settings.feature_set,
        corrected_fan_speed=settings.corrected_fan_speed,
    )
    fit = robust_weighted_linear_fit(
        design,
        data["target"] / rated_total_thrust_n,
        data["weights"],
    )
    model = EffectiveThrustMap(
        rated_total_thrust_n=rated_total_thrust_n,
        pressure_exponent=settings.pressure_exponent,
        feature_set=settings.feature_set,
        corrected_fan_speed=settings.corrected_fan_speed,
        coefficients=fit.coefficients,
    )
    return model, fit


def _predict(model: EffectiveThrustMap, flight: PropulsionFlight) -> np.ndarray:
    return model.thrust_n(
        n1_percent=flight.n1_percent,
        mach=flight.mach,
        pressure_ratio=flight.pressure_ratio,
        static_temperature_k=flight.static_temperature_k,
    )


def _raw_predict(model: EffectiveThrustMap, flight: PropulsionFlight) -> np.ndarray:
    return model.raw_thrust_n(
        n1_percent=flight.n1_percent,
        mach=flight.mach,
        pressure_ratio=flight.pressure_ratio,
        static_temperature_k=flight.static_temperature_k,
    )


def _cross_validated_predictions(
    flights: list[PropulsionFlight],
    *,
    settings: ModelSettings,
    rated_total_thrust_n: float,
    cd0: float,
    induced_drag_factor: float,
    cd0_scale: float,
    k_scale: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[float],
    dict[str, EffectiveThrustMap],
]:
    predictions: dict[str, np.ndarray] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    conditions: list[float] = []
    models: dict[str, EffectiveThrustMap] = {}
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        model, fit = _fit_map(
            training,
            settings=settings,
            rated_total_thrust_n=rated_total_thrust_n,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
            cd0_scale=cd0_scale,
            k_scale=k_scale,
        )
        predictions[held_out.key] = _predict(model, held_out)
        raw_predictions[held_out.key] = _raw_predict(model, held_out)
        conditions.append(float(fit.normal_matrix_condition))
        models[held_out.key] = model
    return predictions, raw_predictions, conditions, models


def _candidate_row(
    flights: list[PropulsionFlight],
    *,
    settings: ModelSettings,
    rated_total_thrust_n: float,
    cd0: float,
    induced_drag_factor: float,
) -> dict[str, Any]:
    predictions, raw_predictions, conditions, fold_models = (
        _cross_validated_predictions(
            flights,
            settings=settings,
            rated_total_thrust_n=rated_total_thrust_n,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
            cd0_scale=1.0,
            k_scale=1.0,
        )
    )
    rmses: list[float] = []
    biases: list[float] = []
    monotonic: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    for flight in flights:
        target = _target_thrust_n(
            flight,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
            cd0_scale=1.0,
            k_scale=1.0,
        )
        residual = predictions[flight.key] - target
        rmses.append(float(np.sqrt(np.mean(residual**2))))
        biases.append(float(np.mean(residual)))
        raw = raw_predictions[flight.key]
        negative.append(raw < 0.0)

        # Local monotonicity in the observed envelope, evaluated on held-out
        # inputs through a symmetric N1 perturbation of that flight's fold map.
        fold_model = fold_models[flight.key]
        plus = fold_model.raw_thrust_n(
            n1_percent=flight.n1_percent + 0.1,
            mach=flight.mach,
            pressure_ratio=flight.pressure_ratio,
            static_temperature_k=flight.static_temperature_k,
        )
        minus = fold_model.raw_thrust_n(
            n1_percent=np.maximum(0.0, flight.n1_percent - 0.1),
            mach=flight.mach,
            pressure_ratio=flight.pressure_ratio,
            static_temperature_k=flight.static_temperature_k,
        )
        monotonic.append(plus >= minus)
    return {
        "pressure_exponent": settings.pressure_exponent,
        "feature_set": settings.feature_set,
        "feature_count": len(THRUST_MAP_FEATURE_SETS[settings.feature_set]),
        "corrected_fan_speed": settings.corrected_fan_speed,
        "lofo_mean_flight_rmse_kn": float(np.mean(rmses) / 1000.0),
        "lofo_median_flight_rmse_kn": float(np.median(rmses) / 1000.0),
        "lofo_worst_flight_rmse_kn": float(np.max(rmses) / 1000.0),
        "lofo_mean_flight_bias_kn": float(np.mean(biases) / 1000.0),
        "observed_local_monotonic_fraction": float(np.mean(np.concatenate(monotonic))),
        "raw_negative_prediction_fraction": float(np.mean(np.concatenate(negative))),
        "median_fold_normal_condition": float(np.median(conditions)),
    }


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    """Select the simplest near-best physically monotone fit-only candidate."""
    eligible = candidates.loc[candidates["observed_local_monotonic_fraction"] >= 0.995]
    if eligible.empty:
        best_fraction = candidates["observed_local_monotonic_fraction"].max()
        eligible = candidates.loc[
            candidates["observed_local_monotonic_fraction"] == best_fraction
        ]
    score_limit = 1.01 * eligible["lofo_mean_flight_rmse_kn"].min()
    near_best = eligible.loc[eligible["lofo_mean_flight_rmse_kn"] <= score_limit]
    return near_best.sort_values(
        ["feature_count", "lofo_mean_flight_rmse_kn", "pressure_exponent"],
        kind="stable",
    ).iloc[0]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _flight_metrics(
    flight: PropulsionFlight,
    *,
    split: str,
    prediction: np.ndarray,
    raw_prediction: np.ndarray,
    scenario_predictions: list[np.ndarray],
    fuel_prior: OpenAPFuelFlowPrior,
    rated_total_thrust_n: float,
    cd0: float,
    induced_drag_factor: float,
) -> dict[str, Any]:
    target = _target_thrust_n(
        flight,
        cd0=cd0,
        induced_drag_factor=induced_drag_factor,
        cd0_scale=1.0,
        k_scale=1.0,
    )
    residual = prediction - target
    predicted_fuel = fuel_prior.fuel_flow_kg_s(prediction)
    fuel_residual = predicted_fuel - flight.fuel_flow_kg_s
    spread = np.ptp(np.vstack(scenario_predictions), axis=0)
    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "split": split,
        "samples": len(target),
        "thrust_rmse_kn": float(np.sqrt(np.mean(residual**2)) / 1000.0),
        "thrust_mae_kn": float(np.mean(np.abs(residual)) / 1000.0),
        "thrust_bias_kn": float(np.mean(residual) / 1000.0),
        "thrust_correlation": _safe_correlation(prediction, target),
        "target_thrust_p05_kn": float(np.quantile(target, 0.05) / 1000.0),
        "target_thrust_p50_kn": float(np.quantile(target, 0.50) / 1000.0),
        "target_thrust_p95_kn": float(np.quantile(target, 0.95) / 1000.0),
        "predicted_thrust_p50_kn": float(np.median(prediction) / 1000.0),
        "raw_negative_prediction_fraction": float(np.mean(raw_prediction < 0.0)),
        "above_rated_prediction_fraction": float(
            np.mean(prediction > rated_total_thrust_n)
        ),
        "drag_sensitivity_spread_p50_kn": float(np.median(spread) / 1000.0),
        "drag_sensitivity_spread_p95_kn": float(np.quantile(spread, 0.95) / 1000.0),
        "fuel_flow_rmse_kg_s": float(np.sqrt(np.mean(fuel_residual**2))),
        "fuel_flow_bias_kg_s": float(np.mean(fuel_residual)),
        "fuel_flow_correlation": _safe_correlation(
            predicted_fuel, flight.fuel_flow_kg_s
        ),
        "n1_engine_abs_difference_p95_percent": float(
            np.quantile(flight.n1_engine_abs_difference_percent, 0.95)
        ),
    }


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["number"]):
        if column in {"samples", "feature_count"}:
            rounded[column] = rounded[column].astype(int)
        else:
            rounded[column] = rounded[column].round(8)
    return rounded


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
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
        max(len(str(headers[index])), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    result = [
        "| "
        + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(headers)
        )
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    result.extend(
        "| "
        + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        + " |"
        for row in rows
    )
    return "\n".join(result)


def _write_report(
    path: Path,
    *,
    candidates: pd.DataFrame,
    chosen: pd.Series,
    metrics: pd.DataFrame,
    parameters: dict[str, Any],
) -> None:
    compact = metrics[
        [
            "registration",
            "recording",
            "split",
            "thrust_rmse_kn",
            "thrust_bias_kn",
            "thrust_correlation",
            "fuel_flow_rmse_kg_s",
            "fuel_flow_correlation",
            "drag_sensitivity_spread_p95_kn",
        ]
    ].copy()
    compact.columns = [
        "registration",
        "recording",
        "split",
        "thrust RMSE kN",
        "thrust bias kN",
        "thrust r",
        "fuel RMSE kg/s",
        "fuel r",
        "polar spread p95 kN",
    ]
    compact = _round_frame(compact)
    fit = metrics.loc[metrics["split"] == "fit_lofo"]
    validation = metrics.loc[metrics["split"] == "validation"]
    coefficients = parameters["model"]["scenarios"]["nominal"]["coefficients"]
    coefficient_text = ", ".join(
        f"`{name}={value:.8g}`" for name, value in coefficients.items()
    )
    text = f"""# A320 effective-thrust calibration

## Decision

The first propulsion model is an **effective total-aircraft thrust map** from
mean engine N1, Mach, ambient pressure ratio, and processed static temperature.
It is suitable for the next trajectory-replay experiment inside the observed
envelope, but it is not an independently identified CFM56 engine deck.

The selected fit-only structure is `{chosen["feature_set"]}` with pressure
exponent {chosen["pressure_exponent"]:.2f} and
`corrected_fan_speed={bool(chosen["corrected_fan_speed"])}`. Its leave-one-flight-out
mean RMSE is {chosen["lofo_mean_flight_rmse_kn"]:.2f} kN. Selection required at
least 99.5% local monotonicity in N1, then chose the fewest coefficients within
1% of the best eligible flight-mean RMSE. All {len(candidates)} candidates and
their scores are retained in `a320-effective-thrust-candidates.csv`.

Nominal coefficients, in the feature order documented in the YAML, are:
{coefficient_text}.

## Leakage boundary and fitting method

1. Only ten development/calibration flights were loaded for model selection.
2. Complete-flight leave-one-flight-out fits selected pressure scaling, raw
   versus inlet-temperature-corrected N1, and polynomial complexity.
3. Each flight had equal total fit weight; climb, level, and descent samples
   present within a flight also had equal total weight. Inputs were thinned to
   5 seconds for fitting, while metrics use every clean 1 Hz sample.
4. Negative prior-implied target thrust was retained during robust fitting.
   Only the executable model output is clipped at zero.
5. The structure and procedure were frozen before the six validation flights
   were loaded once. Fuel flow was not used to fit or select the map.

The conditional target is

```text
T_effective = m (dV/dt + g sin(gamma)) + CD0 qS + k L²/(qS).
```

The repository A320 polar (`CD0={parameters["drag_prior"]["cd0"]:.5f}`,
`k={parameters["drag_prior"]["induced_drag_factor"]:.5f}`) is the nominal prior.
Separate maps repeat the complete fit for `CD0` and `k` at ±20%; the two terms
are never jointly estimated from QAR.

## Complete-flight results

{_markdown_table(compact)}

The median per-flight thrust RMSE is {fit["thrust_rmse_kn"].median():.2f} kN in
fit-flight LOFO prediction and {validation["thrust_rmse_kn"].median():.2f} kN on
the six frozen validation flights. Median fuel-flow RMSE is
{fit["fuel_flow_rmse_kg_s"].median():.3f} kg/s and
{validation["fuel_flow_rmse_kg_s"].median():.3f} kg/s, respectively. Fuel is an
independent consistency signal through the pinned OpenAP A320 curve; agreement
does not make the inferred thrust a direct measurement.

## Interpretation and next gate

This map closes only the measured-N1-to-effective-thrust relation. It does not
yet provide the missing engine command: QAR TLA is detent-like and cannot alone
explain autothrust modulation. Command observability and spool dynamics must be
treated separately. The immediate next experiment is a
longitudinal replay using measured N1 as propulsion input, measured/estimated
pitch as guidance input, the nominal drag polar, and measured initial mass. A
replay should be evaluated first on airspeed, altitude, and fuel burn, with the
five polar scenarios carried as a prediction envelope.

Do not extrapolate this polynomial beyond the YAML validity envelope. The N1
decoder, effective gamma, clean-flap interpretation, static-temperature lineage,
thrust-axis alignment, and drag coefficients remain uncertain. The ±20% polar
spread is a sensitivity study, not a calibrated confidence interval.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected ten fit and six validation flights")
    if set(fit_selection["recording"]) & set(validation_selection["recording"]):
        raise ValueError("fit and validation recordings overlap")

    vertical = _read_vertical_calibration(args.vertical_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    if aircraft.wing is None or aircraft.drag is None or aircraft.engine is None:
        raise ValueError("A320 definition lacks wing, drag, or engine data")
    wing_area = float(aircraft.wing.area)
    cd0 = reference_drag.zero_lift_drag_coefficient
    induced_drag_factor = reference_drag.induced_drag_factor
    engine_count = int(aircraft.engine.number)
    engine_name = str(aircraft.engine.default)
    engine = _read_csv_row(args.engine_database, "name", engine_name)
    fuel_reference = _read_csv_row(args.fuel_model_database, "typecode", "A320")
    fuel_reference_engine = _read_csv_row(
        args.engine_database, "name", fuel_reference["engine_type"]
    )
    rated_thrust_per_engine = float(engine["max_thrust"])
    rated_total_thrust = engine_count * rated_thrust_per_engine
    fuel_prior = OpenAPFuelFlowPrior(
        engine_count=engine_count,
        rated_thrust_n_per_engine=rated_thrust_per_engine,
        c1=float(fuel_reference["c1"]),
        c2=float(fuel_reference["c2"]),
        c3=float(fuel_reference["c3"]),
        engine_scale=float(engine["ff_to"]) / float(fuel_reference_engine["ff_to"]),
    )

    # Leakage boundary: candidate selection can only see these ten flights.
    fit_flights = _load_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    candidate_rows: list[dict[str, Any]] = []
    for pressure_exponent in PRESSURE_EXPONENTS:
        for corrected in (False, True):
            for feature_set in THRUST_MAP_FEATURE_SETS:
                candidate_rows.append(
                    _candidate_row(
                        fit_flights,
                        settings=ModelSettings(
                            pressure_exponent=pressure_exponent,
                            feature_set=feature_set,
                            corrected_fan_speed=corrected,
                        ),
                        rated_total_thrust_n=rated_total_thrust,
                        cd0=cd0,
                        induced_drag_factor=induced_drag_factor,
                    )
                )
    candidates = pd.DataFrame(candidate_rows)
    chosen = _select_candidate(candidates)
    settings = ModelSettings(
        pressure_exponent=float(chosen["pressure_exponent"]),
        feature_set=str(chosen["feature_set"]),
        corrected_fan_speed=bool(chosen["corrected_fan_speed"]),
    )

    scenario_models: dict[str, EffectiveThrustMap] = {}
    scenario_fits: dict[str, Any] = {}
    fit_cv_predictions: dict[str, dict[str, np.ndarray]] = {}
    fit_cv_raw: dict[str, dict[str, np.ndarray]] = {}
    fit_cv_conditions: dict[str, list[float]] = {}
    for name, (cd0_scale, k_scale) in DRAG_SCENARIOS.items():
        scenario_models[name], scenario_fits[name] = _fit_map(
            fit_flights,
            settings=settings,
            rated_total_thrust_n=rated_total_thrust,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
            cd0_scale=cd0_scale,
            k_scale=k_scale,
        )
        (
            fit_cv_predictions[name],
            fit_cv_raw[name],
            fit_cv_conditions[name],
            _,
        ) = _cross_validated_predictions(
            fit_flights,
            settings=settings,
            rated_total_thrust_n=rated_total_thrust,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
            cd0_scale=cd0_scale,
            k_scale=k_scale,
        )

    # Structure and fitting rules are now frozen. Load validation exactly once.
    validation_flights = _load_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    metric_rows: list[dict[str, Any]] = []
    for flight in fit_flights:
        metric_rows.append(
            _flight_metrics(
                flight,
                split="fit_lofo",
                prediction=fit_cv_predictions["nominal"][flight.key],
                raw_prediction=fit_cv_raw["nominal"][flight.key],
                scenario_predictions=[
                    fit_cv_predictions[name][flight.key] for name in DRAG_SCENARIOS
                ],
                fuel_prior=fuel_prior,
                rated_total_thrust_n=rated_total_thrust,
                cd0=cd0,
                induced_drag_factor=induced_drag_factor,
            )
        )
    for flight in validation_flights:
        metric_rows.append(
            _flight_metrics(
                flight,
                split="validation",
                prediction=_predict(scenario_models["nominal"], flight),
                raw_prediction=_raw_predict(scenario_models["nominal"], flight),
                scenario_predictions=[
                    _predict(scenario_models[name], flight) for name in DRAG_SCENARIOS
                ],
                fuel_prior=fuel_prior,
                rated_total_thrust_n=rated_total_thrust,
                cd0=cd0,
                induced_drag_factor=induced_drag_factor,
            )
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["split", "registration", "recording"], kind="stable"
    )

    feature_names = THRUST_MAP_FEATURE_SETS[settings.feature_set]
    all_fit_values = {
        "n1_percent": np.concatenate([flight.n1_percent for flight in fit_flights]),
        "mach": np.concatenate([flight.mach for flight in fit_flights]),
        "pressure_ratio": np.concatenate(
            [flight.pressure_ratio for flight in fit_flights]
        ),
        "static_temperature_k": np.concatenate(
            [flight.static_temperature_k for flight in fit_flights]
        ),
    }
    scenario_records: dict[str, Any] = {}
    for name, model in scenario_models.items():
        cd0_scale, k_scale = DRAG_SCENARIOS[name]
        fit = scenario_fits[name]
        scenario_records[name] = {
            "cd0_scale": cd0_scale,
            "k_scale": k_scale,
            "coefficients": dict(zip(feature_names, model.coefficients, strict=True)),
            "fit_weighted_rmse_kn": float(
                fit.base_weighted_rmse * rated_total_thrust / 1000.0
            ),
            "robust_scale_kn": float(fit.robust_scale * rated_total_thrust / 1000.0),
            "normal_matrix_condition": float(fit.normal_matrix_condition),
            "median_lofo_normal_matrix_condition": float(
                np.median(fit_cv_conditions[name])
            ),
        }
    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "status": "provisional-effective-map-validated-conditionally-on-drag-prior",
        "leakage_control": {
            "fit_flights_loaded_for_structure_selection": [
                flight.key for flight in fit_flights
            ],
            "structure_frozen_before_validation_load": True,
            "validation_flights_loaded_once_after_freeze": [
                flight.key for flight in validation_flights
            ],
            "fuel_flow_used_for_fit_or_selection": False,
        },
        "model": {
            "output": "effective total-aircraft thrust along the air-relative path",
            "rated_total_thrust_n": rated_total_thrust,
            "pressure_exponent": settings.pressure_exponent,
            "corrected_fan_speed": settings.corrected_fan_speed,
            "feature_set": settings.feature_set,
            "feature_order": list(feature_names),
            "fan_speed_center_fraction": 0.8,
            "mach_center": 0.6,
            "physical_output_clip_n": [0.0, None],
            "negative_targets_retained_during_fit": True,
            "scenarios": scenario_records,
        },
        "selection": {
            "candidate_count": len(candidates),
            "complete_flight_cross_validation": "leave-one-fit-flight-out",
            "primary_score": "mean per-flight thrust RMSE",
            "monotonicity_minimum": 0.995,
            "simplicity_rule": "fewest coefficients within 1 percent of eligible best",
            "selected_candidate": chosen.to_dict(),
        },
        "drag_prior": {
            "source": "src/aerox/data/aircraft/yml/turbofan/a320.yml",
            "wing_area_m2": wing_area,
            "cd0": cd0,
            "induced_drag_factor": induced_drag_factor,
            "sensitivity": "one-at-a-time plus/minus 20 percent; not a confidence interval",
        },
        "engine_prior": {
            "source": str(args.engine_database),
            "name": engine_name,
            "engine_count": engine_count,
            "rated_thrust_n_per_engine": rated_thrust_per_engine,
            "fuel_flow_takeoff_kg_s_per_engine": float(engine["ff_to"]),
        },
        "fuel_flow_prior": {
            "source_code": "../openap/openap/fuel.py",
            "source_data": str(args.fuel_model_database),
            "source_commit": "472ab3e48d52ae62f9586d2e75e143ea4182a9e9",
            "reference_engine": fuel_reference["engine_type"],
            "c1": float(fuel_reference["c1"]),
            "c2": float(fuel_reference["c2"]),
            "c3": float(fuel_reference["c3"]),
            "engine_scale": fuel_prior.engine_scale,
            "status": "independent-diagnostic-not-fitted-to-this-qar-panel",
        },
        "method": {
            "sample_rate_hz": 1,
            "derivative_window_s": DERIVATIVE_WINDOW_S,
            "fit_thinning_s": FIT_THINNING_S,
            "weighting": "equal flight then equal phase present within flight",
            "loss": "Huber IRLS",
            "phase_definition_deg": {
                "climb": ">1",
                "level": "[-1,1]",
                "descent": "<-1",
            },
            "selection": {
                "airborne": True,
                "flap_code": 0,
                "maximum_absolute_bank_deg": MAX_ABSOLUTE_BANK_DEG,
                "minimum_true_airspeed_mps": MIN_TRUE_AIRSPEED_MPS,
                "maximum_mach": MAX_MACH,
            },
        },
        "validity_envelope_fit_samples": {
            name: {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p99": float(np.quantile(values, 0.99)),
            }
            for name, values in all_fit_values.items()
        },
        "limitations": [
            "QAR identifies effective T-D; thrust is conditional on the drag prior.",
            "N1 decoding and processed static-temperature lineage remain provisional.",
            "The map consumes measured N1; TLA-to-spool dynamics are not modeled.",
            "Thrust-axis inclination, vertical wind, and normal thrust are omitted.",
            "The polynomial must not be extrapolated beyond the validity envelope.",
        ],
    }

    args.candidate_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(metrics).to_csv(args.metrics, index=False)
    args.parameters.write_text(
        yaml.safe_dump(_native(parameters), sort_keys=False, allow_unicode=True)
    )
    _write_report(
        args.report,
        candidates=candidates,
        chosen=chosen,
        metrics=metrics,
        parameters=parameters,
    )
    return candidates, metrics, parameters


if __name__ == "__main__":
    run(parse_args())
