#!/usr/bin/env python3
"""Calibrate recorded attitude inputs and replay the A320 3D force-balance plant.

The core plant consumes effective recorded pitch, effective recorded bank, and
estimated realized net thrust at 1 Hz.  It advances speed, flight-path angle,
heading, horizontal position, pressure altitude, and mass without state resets
in the full-segment replay.  Surface allocation and FADEC dynamics are outside
this experiment.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    FT_TO_M,
    decode_a320_flight,
    raw_columns,
)
from aerox.models import (
    LIFT_FEATURE_SETS,
    CalibratedTabulatedLiftSurface,
    EffectiveLiftMap,
    GuardedHybridLiftModel,
    ParabolicDragPolar,
    effective_lift_design,
)
from aerox.modeling.dynamics import (
    PointMassModel,
    simulate_point_mass_zoh,
)
from aerox.modeling.identification import (
    RobustLinearFit,
    delay_samples,
    robust_weighted_linear_fit,
)
from aerox.modeling.dynamics.kinematics import (
    horizontal_distance_m,
    integrate_wgs84,
)
from aerox.models.openap import CalibratedFuelFlow, OpenAPFuelFlowPrior
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    EffectiveThrustMap,
    MeasuredN1ThrustAdapter,
)
from aerox.modeling.force_reconstruction import centered_linear_derivative
from aerox.models.atmosphere import G0
from tools.calibrate_a320_effective_thrust import (
    FIT_THINNING_S,
    RAW_FILES,
    PropulsionFlight,
    VerticalCalibration,
    _load_flights,
    _phase_equal_weights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.research_support import (
    RESET_WINDOW_S,
    longest_contiguous_slice as _longest_contiguous_slice,
    markdown_table as _markdown_table,
    native_value as _native,
    round_frame as _round_frame,
)

INTEGRATION_RATE_HZ = 10
LIFT_FEATURE_SET = "linear_alpha_mach"
BANK_DELAY_CANDIDATES_S = tuple(range(8))
BANK_DERIVATIVE_WINDOW_S = 21
MIN_BANK_TARGET_DEG = 0.3
MAX_BANK_TARGET_DEG = 20.0
PROCESSED_COLUMNS = [
    "flight_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "u_component_of_wind",
    "v_component_of_wind",
]


@dataclass(frozen=True)
class HorizontalParameters:
    heading_bias_rad: float
    heading_delay_s: int
    wind_north_correction_mps: float
    wind_east_correction_mps: float


@dataclass(frozen=True)
class PointMassReplayFlight:
    base: PropulsionFlight
    effective_pitch_angle_rad: np.ndarray
    calibrated_heading_rad: np.ndarray
    roll_code_full: np.ndarray
    latitude_rad: np.ndarray
    longitude_rad: np.ndarray
    geometric_altitude_m: np.ndarray
    wind_north_mps: np.ndarray
    wind_east_mps: np.ndarray

    @property
    def key(self) -> str:
        return self.base.key

    def bank_angle_rad(self, calibration: BankCalibration) -> np.ndarray:
        code = delay_samples(self.roll_code_full, calibration.delay_s)[
            self.base.sample_index
        ]
        return np.deg2rad(calibration.scale_deg_per_code * code + calibration.bias_deg)


@dataclass(frozen=True)
class LiftFit:
    model: EffectiveLiftMap
    result: RobustLinearFit
    lofo_metrics: pd.DataFrame


@dataclass(frozen=True)
class BankCalibration:
    delay_s: int
    scale_deg_per_code: float
    bias_deg: float
    result: RobustLinearFit
    delay_metrics: pd.DataFrame
    leave_one_flight_out: np.ndarray


@dataclass(frozen=True)
class ReplayJob:
    flight: PointMassReplayFlight
    split: str
    lift_model: EffectiveLiftMap | CalibratedTabulatedLiftSurface | GuardedHybridLiftModel
    bank_calibration: BankCalibration
    thrust_map: EffectiveThrustMap
    fuel_model: CalibratedFuelFlow
    wing_area_m2: float
    cd0: float
    induced_drag_factor: float


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
        "--fuel-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-fuel-calibration.yml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-point-mass-replay-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-point-mass-replay.md"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="flight-level worker processes; use 1 for serial debugging",
    )
    return parser.parse_args()


def _read_horizontal_parameters(path: Path) -> HorizontalParameters:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    parameters = model["parameters"]
    return HorizontalParameters(
        heading_bias_rad=float(parameters["heading_bias_deg"]["value"]) * DEG_TO_RAD,
        heading_delay_s=int(model["trajectory_reference_delay_s"]["value"]),
        wind_north_correction_mps=float(
            parameters["global_wind_north_correction_mps"]["value"]
        ),
        wind_east_correction_mps=float(
            parameters["global_wind_east_correction_mps"]["value"]
        ),
    )


def _load_point_mass_flights(
    selection: pd.DataFrame,
    *,
    raw_dir: Path,
    processed_path: Path,
    vertical: VerticalCalibration,
    horizontal: HorizontalParameters,
    tas_scale: float,
    wing_area_m2: float,
) -> list[PointMassReplayFlight]:
    base_flights = _load_flights(
        selection,
        raw_dir=raw_dir,
        processed_path=processed_path,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=wing_area_m2,
    )
    base_by_key = {flight.key: flight for flight in base_flights}
    identifiers = selection["recording"].astype(str).tolist()
    processed = pd.read_parquet(
        processed_path,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "in", identifiers)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        )
        for recording, frame in processed.groupby("flight_id", sort=False)
    }

    flights: list[PointMassReplayFlight] = []
    for registration, current_selection in selection.groupby("registration", sort=True):
        current_ids = current_selection["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current_ids)],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            key = f"{registration}/{recording}"
            base = base_by_key[key]
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=recording,
            )
            current = processed_groups[recording]
            timestamps = current["timestamp"].to_numpy(dtype="datetime64[ns]")
            raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
            if np.any(raw_indices >= len(decoded.timestamp_utc)) or not np.all(
                decoded.timestamp_utc[raw_indices] == timestamps
            ):
                raise ValueError(f"raw/processed mismatch for {key}")

            pitch_code = (
                decoded.channel("pitch", allow_provisional=True)[raw_indices]
                / DEG_TO_RAD
                / 0.5
            )
            # The vertical fit cannot separate pitch and AOA zero offsets.  We
            # choose zero effective AOA bias, assigning the fitted intercept to
            # pitch.  The lift-map intercept absorbs this explicit convention.
            effective_pitch = np.deg2rad(
                vertical.pitch_scale_deg_per_code
                * delay_samples(pitch_code, vertical.delay_s)
                + vertical.effective_gamma_bias_deg
            )
            heading = (
                delay_samples(
                    decoded.channel("true_heading", allow_provisional=True)[
                        raw_indices
                    ],
                    horizontal.heading_delay_s,
                )
                + horizontal.heading_bias_rad
            )
            roll_code = (
                decoded.channel("roll", allow_provisional=True)[raw_indices]
                / DEG_TO_RAD
                / 0.5
            )
            select = base.sample_index
            flights.append(
                PointMassReplayFlight(
                    base=base,
                    effective_pitch_angle_rad=effective_pitch[select],
                    calibrated_heading_rad=heading[select],
                    roll_code_full=roll_code,
                    latitude_rad=current["latitude"].to_numpy(dtype=float)[select]
                    * DEG_TO_RAD,
                    longitude_rad=current["longitude"].to_numpy(dtype=float)[select]
                    * DEG_TO_RAD,
                    geometric_altitude_m=current["altitude"].to_numpy(dtype=float)[
                        select
                    ]
                    * FT_TO_M,
                    wind_north_mps=current["v_component_of_wind"].to_numpy(dtype=float)[
                        select
                    ]
                    + horizontal.wind_north_correction_mps,
                    wind_east_mps=current["u_component_of_wind"].to_numpy(dtype=float)[
                        select
                    ]
                    + horizontal.wind_east_correction_mps,
                )
            )
    if {flight.key for flight in flights} != set(base_by_key):
        raise ValueError("attitude replay flight load mismatch")
    return flights


def _lift_training_arrays(
    flights: list[PointMassReplayFlight],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design: list[np.ndarray] = []
    target: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        base = flight.base
        alpha = flight.effective_pitch_angle_rad - base.flight_path_angle_rad
        thin = base.sample_index % FIT_THINNING_S == 0
        design.append(
            effective_lift_design(
                alpha[thin], base.mach[thin], feature_set=LIFT_FEATURE_SET
            )
        )
        target.append(base.lift_coefficient[thin])
        weights.append(_phase_equal_weights(base.phase[thin]) / len(flights))
    return np.vstack(design), np.concatenate(target), np.concatenate(weights)


def _fit_lift_once(
    flights: list[PointMassReplayFlight],
) -> tuple[EffectiveLiftMap, RobustLinearFit]:
    design, target, weights = _lift_training_arrays(flights)
    result = robust_weighted_linear_fit(design, target, weights)
    return EffectiveLiftMap(LIFT_FEATURE_SET, result.coefficients), result


def _fit_lift(flights: list[PointMassReplayFlight]) -> LiftFit:
    rows: list[dict[str, Any]] = []
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        model, _ = _fit_lift_once(training)
        alpha = held_out.effective_pitch_angle_rad - held_out.base.flight_path_angle_rad
        residual = (
            model.raw_lift_coefficient(alpha, held_out.base.mach)
            - held_out.base.lift_coefficient
        )
        rows.append(
            {
                "flight": held_out.key,
                "cl_rmse": float(np.sqrt(np.mean(residual**2))),
                "cl_bias": float(np.mean(residual)),
            }
        )
    model, result = _fit_lift_once(flights)
    return LiftFit(model=model, result=result, lofo_metrics=pd.DataFrame(rows))


def _bank_target_rad(flight: PointMassReplayFlight) -> tuple[slice, np.ndarray]:
    segment = _longest_contiguous_slice(flight.base.sample_index)
    time = flight.base.time_s[segment]
    speed = flight.base.true_airspeed_mps[segment]
    gamma = flight.base.flight_path_angle_rad[segment]
    heading = np.unwrap(flight.calibrated_heading_rad[segment])
    heading_rate = centered_linear_derivative(heading, time, BANK_DERIVATIVE_WINDOW_S)
    gamma_rate = centered_linear_derivative(gamma, time, BANK_DERIVATIVE_WINDOW_S)
    # Exact inversion of the reduced coordinated-flight normal and lateral
    # equations; it avoids using the fitted lift map as a bank target.
    bank = np.arctan2(
        speed * np.cos(gamma) * heading_rate,
        speed * gamma_rate + G0 * np.cos(gamma),
    )
    return segment, bank


def _bank_training_arrays(
    flights: list[PointMassReplayFlight], delay_s: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design: list[np.ndarray] = []
    target: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        segment, bank_target = _bank_target_rad(flight)
        indices = flight.base.sample_index[segment]
        code = delay_samples(flight.roll_code_full, delay_s)[indices]
        target_deg = np.rad2deg(bank_target)
        valid = (
            np.isfinite(code + target_deg)
            & (np.abs(target_deg) >= MIN_BANK_TARGET_DEG)
            & (np.abs(target_deg) < MAX_BANK_TARGET_DEG)
            & (indices % FIT_THINNING_S == 0)
        )
        current_target = target_deg[valid]
        design.append(np.column_stack((code[valid], np.ones(len(current_target)))))
        target.append(current_target)
        weights.append(
            np.full(
                len(current_target),
                1.0 / (len(flights) * len(current_target)),
            )
        )
    return np.vstack(design), np.concatenate(target), np.concatenate(weights)


def _fit_bank_once(
    flights: list[PointMassReplayFlight], delay_s: int
) -> RobustLinearFit:
    design, target, weights = _bank_training_arrays(flights, delay_s)
    return robust_weighted_linear_fit(design, target, weights)


def _fit_bank(flights: list[PointMassReplayFlight]) -> BankCalibration:
    rows: list[dict[str, Any]] = []
    fits: dict[int, RobustLinearFit] = {}
    for delay_s in BANK_DELAY_CANDIDATES_S:
        result = _fit_bank_once(flights, delay_s)
        fits[delay_s] = result
        rows.append(
            {
                "delay_s": delay_s,
                "huber_objective": result.huber_objective,
                "weighted_rmse_deg": result.base_weighted_rmse,
                "scale_deg_per_code": result.coefficients[0],
                "bias_deg": result.coefficients[1],
            }
        )
    selected = min(fits, key=lambda delay: fits[delay].huber_objective)
    result = fits[selected]
    leave_one_out = []
    for held_out in flights:
        subset = [flight for flight in flights if flight.key != held_out.key]
        current = _fit_bank_once(subset, selected)
        leave_one_out.append(current.coefficients)
    return BankCalibration(
        delay_s=selected,
        scale_deg_per_code=float(result.coefficients[0]),
        bias_deg=float(result.coefficients[1]),
        result=result,
        delay_metrics=pd.DataFrame(rows),
        leave_one_flight_out=np.asarray(leave_one_out),
    )


def _bank_flight_metrics(
    flight: PointMassReplayFlight, calibration: BankCalibration
) -> dict[str, Any]:
    segment, target = _bank_target_rad(flight)
    predicted = flight.bank_angle_rad(calibration)[segment]
    valid = np.isfinite(target + predicted) & (
        np.abs(target) < MAX_BANK_TARGET_DEG * DEG_TO_RAD
    )
    residual = np.rad2deg(predicted[valid] - target[valid])
    return {
        "flight": flight.key,
        "bank_rmse_deg": float(np.sqrt(np.mean(residual**2))),
        "bank_bias_deg": float(np.mean(residual)),
    }


def _lift_flight_metrics(
    flight: PointMassReplayFlight, model: EffectiveLiftMap
) -> dict[str, Any]:
    alpha = flight.effective_pitch_angle_rad - flight.base.flight_path_angle_rad
    residual = (
        model.raw_lift_coefficient(alpha, flight.base.mach)
        - flight.base.lift_coefficient
    )
    return {
        "flight": flight.key,
        "cl_rmse": float(np.sqrt(np.mean(residual**2))),
        "cl_bias": float(np.mean(residual)),
    }


def _read_thrust_map(path: Path) -> EffectiveThrustMap:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    feature_set = str(model["feature_set"])
    coefficients = model["scenarios"]["nominal"]["coefficients"]
    return EffectiveThrustMap(
        rated_total_thrust_n=float(model["rated_total_thrust_n"]),
        pressure_exponent=float(model["pressure_exponent"]),
        feature_set=feature_set,
        corrected_fan_speed=bool(model["corrected_fan_speed"]),
        coefficients=np.asarray(
            [coefficients[name] for name in THRUST_MAP_FEATURE_SETS[feature_set]], dtype=float
        ),
    )


def _read_fuel_model(thrust_path: Path, fuel_path: Path) -> CalibratedFuelFlow:
    thrust = yaml.safe_load(thrust_path.read_text())
    fuel = yaml.safe_load(fuel_path.read_text())
    engine = thrust["engine_prior"]
    prior_parameters = thrust["fuel_flow_prior"]
    prior = OpenAPFuelFlowPrior(
        engine_count=int(engine["engine_count"]),
        rated_thrust_n_per_engine=float(engine["rated_thrust_n_per_engine"]),
        c1=float(prior_parameters["c1"]),
        c2=float(prior_parameters["c2"]),
        c3=float(prior_parameters["c3"]),
        engine_scale=float(prior_parameters["engine_scale"]),
    )
    return CalibratedFuelFlow(
        prior=prior,
        scale=float(fuel["model"]["scale"]),
        offset_kg_s=float(fuel["model"]["offset_kg_s"]),
    )


def _make_plant(job: ReplayJob) -> PointMassModel:
    return PointMassModel(
        wing_area_m2=job.wing_area_m2,
        lift_map=job.lift_model,
        drag_model=ParabolicDragPolar(job.cd0, job.induced_drag_factor),
        fuel_flow_prior=job.fuel_model,
    )


def _simulate_slice(job: ReplayJob, selection: slice):
    flight = job.flight
    base = flight.base
    adapter = MeasuredN1ThrustAdapter(job.thrust_map)
    net_thrust = adapter.net_thrust_n(
        n1_percent=base.n1_percent[selection],
        mach=base.mach[selection],
        pressure_ratio=base.pressure_ratio[selection],
        static_temperature_k=base.static_temperature_k[selection],
    )
    return simulate_point_mass_zoh(
        plant=_make_plant(job),
        time_s=base.time_s[selection],
        pitch_angle_rad=flight.effective_pitch_angle_rad[selection],
        bank_angle_rad=flight.bank_angle_rad(job.bank_calibration)[selection],
        net_thrust_n=net_thrust,
        static_temperature_k=base.static_temperature_k[selection],
        wind_north_mps=flight.wind_north_mps[selection],
        wind_east_mps=flight.wind_east_mps[selection],
        initial_true_airspeed_mps=float(base.true_airspeed_mps[selection][0]),
        initial_flight_path_angle_rad=float(base.flight_path_angle_rad[selection][0]),
        initial_heading_rad=float(flight.calibrated_heading_rad[selection][0]),
        initial_pressure_altitude_m=float(base.pressure_altitude_m[selection][0]),
        initial_mass_kg=float(base.mass_kg[selection][0]),
        integration_rate_hz=INTEGRATION_RATE_HZ,
    )


def _trajectory_errors(
    job: ReplayJob, selection: slice, trajectory: Any
) -> dict[str, np.ndarray]:
    flight = job.flight
    base = flight.base
    speed = trajectory.true_airspeed_mps - base.true_airspeed_mps[selection]
    gamma = trajectory.flight_path_angle_rad - base.flight_path_angle_rad[selection]
    altitude = trajectory.pressure_altitude_m - base.pressure_altitude_m[selection]
    mass = trajectory.mass_kg - base.mass_kg[selection]
    heading = np.arctan2(
        np.sin(trajectory.heading_rad - flight.calibrated_heading_rad[selection]),
        np.cos(trajectory.heading_rad - flight.calibrated_heading_rad[selection]),
    )
    horizontal_air = trajectory.true_airspeed_mps * np.cos(
        trajectory.flight_path_angle_rad
    )
    north = (
        horizontal_air * np.cos(trajectory.heading_rad)
        + flight.wind_north_mps[selection]
    )
    east = (
        horizontal_air * np.sin(trajectory.heading_rad)
        + flight.wind_east_mps[selection]
    )
    down = -trajectory.true_airspeed_mps * np.sin(trajectory.flight_path_angle_rad)
    predicted = integrate_wgs84(
        trajectory.time_s,
        float(flight.latitude_rad[selection][0]),
        float(flight.longitude_rad[selection][0]),
        float(flight.geometric_altitude_m[selection][0]),
        north,
        east,
        down,
    )
    horizontal = horizontal_distance_m(
        predicted.latitude_rad,
        predicted.longitude_rad,
        flight.latitude_rad[selection],
        flight.longitude_rad[selection],
    )
    measured_horizontal_air = base.true_airspeed_mps[selection] * np.cos(
        base.flight_path_angle_rad[selection]
    )
    measured_heading_baseline = integrate_wgs84(
        trajectory.time_s,
        float(flight.latitude_rad[selection][0]),
        float(flight.longitude_rad[selection][0]),
        float(flight.geometric_altitude_m[selection][0]),
        measured_horizontal_air * np.cos(flight.calibrated_heading_rad[selection])
        + flight.wind_north_mps[selection],
        measured_horizontal_air * np.sin(flight.calibrated_heading_rad[selection])
        + flight.wind_east_mps[selection],
        -base.true_airspeed_mps[selection]
        * np.sin(base.flight_path_angle_rad[selection]),
    )
    measured_heading_horizontal = horizontal_distance_m(
        measured_heading_baseline.latitude_rad,
        measured_heading_baseline.longitude_rad,
        flight.latitude_rad[selection],
        flight.longitude_rad[selection],
    )
    return {
        "speed": speed,
        "gamma": gamma,
        "altitude": altitude,
        "mass": mass,
        "heading": heading,
        "horizontal": horizontal,
        "measured_heading_horizontal": measured_heading_horizontal,
    }


def _windowed_errors(job: ReplayJob, segment: slice) -> dict[str, Any]:
    start = int(segment.start or 0)
    stop = int(segment.stop or len(job.flight.base.time_s))
    residuals: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "speed",
            "gamma",
            "altitude",
            "mass",
            "heading",
            "horizontal",
            "measured_heading_horizontal",
        )
    }
    terminal_horizontal: list[float] = []
    failures = 0
    windows = 0
    for window_start in range(start, stop - 60, RESET_WINDOW_S):
        window_stop = min(window_start + RESET_WINDOW_S + 1, stop)
        current = slice(window_start, window_stop)
        try:
            trajectory = _simulate_slice(job, current)
            errors = _trajectory_errors(job, current, trajectory)
        except (ValueError, FloatingPointError, OverflowError):
            failures += 1
            continue
        windows += 1
        for name, values in errors.items():
            residuals[name].append(values)
        terminal_horizontal.append(float(errors["horizontal"][-1]))
    if windows == 0:
        return {"window_count": 0, "window_failures": failures}
    joined = {name: np.concatenate(values) for name, values in residuals.items()}
    return {
        "window_count": windows,
        "window_failures": failures,
        "window_speed_rmse_mps": float(np.sqrt(np.mean(joined["speed"] ** 2))),
        "window_gamma_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(joined["gamma"] ** 2)))
        ),
        "window_altitude_rmse_m": float(np.sqrt(np.mean(joined["altitude"] ** 2))),
        "window_mass_rmse_kg": float(np.sqrt(np.mean(joined["mass"] ** 2))),
        "window_heading_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(joined["heading"] ** 2)))
        ),
        "window_horizontal_rmse_m": float(np.sqrt(np.mean(joined["horizontal"] ** 2))),
        "window_measured_heading_horizontal_rmse_m": float(
            np.sqrt(np.mean(joined["measured_heading_horizontal"] ** 2))
        ),
        "window_horizontal_terminal_abs_median_m": float(
            np.median(np.abs(terminal_horizontal))
        ),
    }


def _replay_metrics(job: ReplayJob) -> dict[str, Any]:
    flight = job.flight
    segment = _longest_contiguous_slice(flight.base.sample_index)
    length = int(segment.stop) - int(segment.start)
    if length < 120:
        raise ValueError(f"continuous segment too short for {flight.key}")
    row: dict[str, Any] = {
        "registration": flight.base.registration,
        "recording": flight.base.recording,
        "role": flight.base.role,
        "split": job.split,
        "segment_start_sample": int(flight.base.sample_index[segment][0]),
        "segment_duration_s": float(
            flight.base.time_s[segment][-1] - flight.base.time_s[segment][0]
        ),
    }
    row.update(_windowed_errors(job, segment))
    try:
        trajectory = _simulate_slice(job, segment)
        errors = _trajectory_errors(job, segment, trajectory)
    except (ValueError, FloatingPointError, OverflowError) as exception:
        row["full_success"] = False
        row["full_failure"] = type(exception).__name__
        return row
    row["full_success"] = True
    row["full_failure"] = ""
    row.update(
        {
            "full_speed_rmse_mps": float(np.sqrt(np.mean(errors["speed"] ** 2))),
            "full_speed_terminal_error_mps": float(errors["speed"][-1]),
            "full_gamma_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(errors["gamma"] ** 2)))
            ),
            "full_altitude_rmse_m": float(np.sqrt(np.mean(errors["altitude"] ** 2))),
            "full_altitude_terminal_error_m": float(errors["altitude"][-1]),
            "full_heading_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(errors["heading"] ** 2)))
            ),
            "full_heading_terminal_error_deg": float(np.rad2deg(errors["heading"][-1])),
            "full_horizontal_rmse_m": float(
                np.sqrt(np.mean(errors["horizontal"] ** 2))
            ),
            "full_horizontal_terminal_error_m": float(errors["horizontal"][-1]),
            "full_measured_heading_horizontal_rmse_m": float(
                np.sqrt(np.mean(errors["measured_heading_horizontal"] ** 2))
            ),
            "full_mass_rmse_kg": float(np.sqrt(np.mean(errors["mass"] ** 2))),
            "full_mass_terminal_error_kg": float(errors["mass"][-1]),
            "predicted_fuel_burn_kg": float(
                trajectory.mass_kg[0] - trajectory.mass_kg[-1]
            ),
            "recorded_mass_drop_kg": float(
                flight.base.mass_kg[segment][0] - flight.base.mass_kg[segment][-1]
            ),
            "input_thrust_p01_kn": float(
                np.quantile(trajectory.net_thrust_n, 0.01) / 1000.0
            ),
            "input_thrust_p99_kn": float(
                np.quantile(trajectory.net_thrust_n, 0.99) / 1000.0
            ),
        }
    )
    return row


def _run_job(job: ReplayJob) -> dict[str, Any]:
    return _replay_metrics(job)


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("calibration", "validation"):
        current = metrics.loc[metrics["split"] == split]
        successful = current.loc[current["full_success"]]
        rows.append(
            {
                "split": split,
                "flights": len(current),
                "full_success": int(current["full_success"].sum()),
                "300s_speed_rmse_mps": current["window_speed_rmse_mps"].mean(),
                "300s_altitude_rmse_m": current["window_altitude_rmse_m"].mean(),
                "300s_heading_rmse_deg": current["window_heading_rmse_deg"].mean(),
                "300s_horizontal_rmse_m": current["window_horizontal_rmse_m"].mean(),
                "300s_measured_heading_baseline_m": current[
                    "window_measured_heading_horizontal_rmse_m"
                ].mean(),
                "full_speed_rmse_mps": successful["full_speed_rmse_mps"].mean(),
                "full_altitude_rmse_m": successful["full_altitude_rmse_m"].mean(),
                "full_heading_rmse_deg": successful["full_heading_rmse_deg"].mean(),
                "full_horizontal_rmse_m": successful["full_horizontal_rmse_m"].mean(),
                "full_measured_heading_baseline_m": successful[
                    "full_measured_heading_horizontal_rmse_m"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    lift_fit: LiftFit,
    bank_calibration: BankCalibration,
    point_mass_metrics: pd.DataFrame,
    replay_metrics: pd.DataFrame,
) -> None:
    aggregate = _round_frame(_aggregate_metrics(replay_metrics))
    validation_replay = replay_metrics.loc[replay_metrics["split"] == "validation"]
    validation_point_mass = point_mass_metrics.loc[
        point_mass_metrics["split"] == "validation"
    ]
    baseline_window = validation_replay[
        "window_measured_heading_horizontal_rmse_m"
    ].mean()
    baseline_full = validation_replay["full_measured_heading_horizontal_rmse_m"].mean()
    longitudinal_path = path.with_name("a320-longitudinal-replay-metrics.csv")
    comparison = "The preceding longitudinal metrics were not available for comparison."
    if longitudinal_path.exists():
        longitudinal = pd.read_csv(longitudinal_path)
        old_validation = longitudinal.loc[longitudinal["split"] == "validation"]

        def improvement(column: str) -> float:
            old = float(old_validation[column].mean())
            new = float(validation_replay[column].mean())
            return 100.0 * (old - new) / old

        comparison = (
            "Relative to the preceding measured-AOA longitudinal replay on the "
            "same six evaluation flights, the 300-second mean speed, flight-path-"
            f"angle, and altitude RMSE improve by {improvement('window_speed_rmse_mps'):.1f}%, "
            f"{improvement('window_gamma_rmse_deg'):.1f}%, and "
            f"{improvement('window_altitude_rmse_m'):.1f}%, respectively. "
            "The full-segment speed and altitude errors are worse because their "
            "small remaining biases integrate for as long as 3.5 hours."
        )
    compact = replay_metrics[
        [
            "registration",
            "recording",
            "split",
            "segment_duration_s",
            "window_speed_rmse_mps",
            "window_altitude_rmse_m",
            "window_heading_rmse_deg",
            "window_horizontal_rmse_m",
            "window_measured_heading_horizontal_rmse_m",
            "full_speed_rmse_mps",
            "full_altitude_rmse_m",
            "full_heading_rmse_deg",
            "full_horizontal_rmse_m",
            "full_measured_heading_horizontal_rmse_m",
        ]
    ].copy()
    compact.columns = [
        "reg",
        "recording",
        "split",
        "segment s",
        "300s V RMSE m/s",
        "300s h RMSE m",
        "300s heading RMSE deg",
        "300s horizontal RMSE m",
        "300s measured-heading baseline m",
        "full V RMSE m/s",
        "full h RMSE m",
        "full heading RMSE deg",
        "full horizontal RMSE m",
        "full measured-heading baseline m",
    ]
    text = f"""# A320 attitude/net-thrust 3D trajectory replay

## Outcome

This is the first QAR replay through the direct pitch, bank, and realized-net-thrust
plant. The model consumes one-second effective recorded pitch and bank plus
estimated total net thrust, then advances speed, flight-path angle, heading,
position, pressure altitude, and mass at {INTEGRATION_RATE_HZ} Hz internally.

The vertical calibration does not separately observe pitch and AOA zero bias.
This replay explicitly assigns its fitted intercept to pitch (zero effective AOA
bias convention), then refits the already-frozen `{LIFT_FEATURE_SET}` lift
structure on calibration flights only. Its fit weighted `CL` RMSE is
{lift_fit.result.base_weighted_rmse:.4f}; mean leave-one-calibration-flight-out
RMSE is {lift_fit.lofo_metrics.cl_rmse.mean():.4f}. On the six evaluation flights,
mean `CL` RMSE is {validation_point_mass.cl_rmse.mean():.4f}.

Coordinated-turn kinematics select a {bank_calibration.delay_s}-second roll
trajectory alignment and an effective scale of
{bank_calibration.scale_deg_per_code:.6f} deg/code with
{bank_calibration.bias_deg:.6f} deg bias. Mean evaluation-flight bank RMSE is
{validation_point_mass.bank_rmse_deg.mean():.3f} deg. The delay is an alignment to
the processed trajectory, not a claim about physical sensor latency; four and
five seconds are practically tied by the robust objective.

## Aggregate trajectory errors

{_markdown_table(aggregate)}

## Interpretation and decision

{comparison}

The exact-interval measured-heading baseline has {baseline_window:.1f} m mean
300-second horizontal RMSE and {baseline_full:.1f} m mean full-segment horizontal
RMSE on the evaluation flights. Propagating heading from effective bank raises
those values to {validation_replay.window_horizontal_rmse_m.mean():.1f} m and
{validation_replay.full_horizontal_rmse_m.mean():.1f} m. Bank-only coordinated-
turn propagation is therefore the dominant complete-flight horizontal failure.

The plant is useful for short-window force-balance replay and controller
prototyping. It is not released for multi-hour open-loop trajectory prediction.
The subsequent lateral work implements recorded yaw rate only through an
explicit replay adapter; heading and bank do not become independent physical
controls. See `a320-observed-heading-rate-replay.md` for the force-consistency
diagnostic.

![A320 pitch-bank-thrust replay overview](figures/a320-point-mass-replay-overview.png)

Figure 1. Evaluation-flight replay comparison. Circles denote TAIL-A, squares
denote TAIL-B, grey lines connect the same flight, and diamonds show the across-
flight summary. Error bars are mean +/- one standard deviation on linear panels
and median with interquartile range on the log panel (`n=6`). A
[vector PDF](figures/a320-point-mass-replay-overview.pdf) is provided for
detailed inspection.

## Flight-level trajectory errors

{_markdown_table(_round_frame(compact))}

Windowed results reset all states every {RESET_WINDOW_S} seconds. Full-segment
results initialize once at the start of each flight's longest contiguous clean
segment and expose accumulated bias. Horizontal errors are evaluated on WGS 84
by integrating the plant's predicted ground-velocity vector from the observed
initial coordinate; the plant itself retains its local north/east states for
controller-facing use. The measured-heading baseline uses recorded TAS,
flight-path angle, and heading plus the same wind on the exact same intervals;
it isolates the additional drift caused by force-balance and coordinated-turn
state propagation.

## Input and model lineage

- Pitch uses the frozen vertical scale, three-second alignment, and explicit
  zero-AOA-bias convention from the vertical calibration.
- Bank scale/bias and alignment are identified from heading-rate and
  flight-path-angle-rate kinematics without using the lift map as a target.
- Heading, TAS, and horizontal wind use the frozen horizontal calibration.
- Net thrust is supplied directly to the plant. Because QAR has no thrust
  sensor, the recorded-flight adapter evaluates the already-frozen effective
  N1-to-thrust map once at measured conditions; simulated state drift does not
  feed back into that input.
- Fuel flow uses the frozen affine correction around the pinned OpenAP prior.
- `CD0` and `k` remain the repository A320 clean drag polar.

## Evidence boundary

This is a diagnostic held-flight evaluation, not a newly untouched confirmation:
the six evaluation flights had already been opened by preceding commissioning
work. The lift structure, thrust map, fuel model, vertical calibration, and
horizontal calibration were fixed before this replay. Only the necessary
effective-attitude lift coefficients and roll calibration are fitted here.

Pitch and AOA absolute biases remain structurally confounded. Bank scale is an
effective coordinated-turn calibration under zero-sideslip and negligible
side-force assumptions. Vertical wind, thrust inclination, sideslip, Earth-rate
terms, and configuration changes are excluded. Complete-flight prediction claims
require a newly predeclared panel and acceptance thresholds.

## Reproducibility

- Generator: `tools/replay_a320_point_mass.py`
- Command: `uv run python -m tools.replay_a320_point_mass --workers 32`
- Core plant: `src/aerox/modeling/dynamics/point_mass.py`
- Parameters: `docs/modeling/a320-pitch-bank-input-calibration.yml`
- Metrics: `docs/modeling/a320-point-mass-replay-metrics.csv`
- Source ledger: `docs/modeling/sources.yml`

Only aggregate errors and fitted parameters are stored in the repository. Raw
QAR samples and reconstructed tracks remain external because their redistribution
status is unknown.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    wing_area = float(aircraft.wing.area)
    cd0 = reference_drag.zero_lift_drag_coefficient
    induced_drag_factor = reference_drag.induced_drag_factor
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)

    # Fit-only attitude work is complete before evaluation flights are loaded.
    fit_flights = _load_point_mass_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    lift_fit = _fit_lift(fit_flights)
    bank_calibration = _fit_bank(fit_flights)

    validation_flights = _load_point_mass_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    attitude_rows = []
    for split, flights in (
        ("calibration", fit_flights),
        ("validation", validation_flights),
    ):
        for flight in flights:
            attitude_rows.append(
                {
                    "split": split,
                    **_lift_flight_metrics(flight, lift_fit.model),
                    **_bank_flight_metrics(flight, bank_calibration),
                }
            )
    point_mass_metrics = pd.DataFrame(attitude_rows)

    thrust_map = _read_thrust_map(args.thrust_parameters)
    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    jobs = [
        ReplayJob(
            flight=flight,
            split=split,
            lift_model=lift_fit.model,
            bank_calibration=bank_calibration,
            thrust_map=thrust_map,
            fuel_model=fuel_model,
            wing_area_m2=wing_area,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
        )
        for split, flights in (
            ("calibration", fit_flights),
            ("validation", validation_flights),
        )
        for flight in flights
    ]
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for job in jobs:
            rows.append(_run_job(job))
            print(f"completed {job.split} {job.flight.key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                rows.append(future.result())
                print(f"completed {job.split} {job.flight.key}", flush=True)
    replay_metrics = pd.DataFrame(rows).sort_values(
        ["split", "registration", "recording"], kind="stable"
    )

    loo = bank_calibration.leave_one_flight_out
    validation_point_mass = point_mass_metrics.loc[
        point_mass_metrics["split"] == "validation"
    ]
    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "diagnostic-held-flight-attitude-calibration-and-3d-replay",
        "pitch": {
            "definition": "effective_pitch_deg = frozen_pitch_scale * delayed_pitch_code + frozen_gamma_bias",
            "bias_convention": "zero effective AOA bias; vertical intercept assigned to pitch",
            "scale_deg_per_code": vertical.pitch_scale_deg_per_code,
            "trajectory_reference_delay_s": vertical.delay_s,
            "assigned_bias_deg": vertical.effective_gamma_bias_deg,
            "identifiability": "pitch and AOA zero biases are not separately observable",
        },
        "bank": {
            "equation": "effective_bank_deg = scale * delayed_roll_code + bias",
            "target": "atan2(V cos(gamma) dheading/dt, V dgamma/dt + g cos(gamma))",
            "scale_deg_per_code": bank_calibration.scale_deg_per_code,
            "bias_deg": bank_calibration.bias_deg,
            "trajectory_reference_delay_s": bank_calibration.delay_s,
            "one_percent_objective_support_s": bank_calibration.delay_metrics.loc[
                bank_calibration.delay_metrics["huber_objective"]
                <= 1.01 * bank_calibration.delay_metrics["huber_objective"].min(),
                "delay_s",
            ].tolist(),
            "delay_status": "effective trajectory alignment; four and five seconds are practically tied",
            "derivative_window_s": BANK_DERIVATIVE_WINDOW_S,
            "fit_target_minimum_absolute_deg": MIN_BANK_TARGET_DEG,
            "fit_target_maximum_absolute_deg": MAX_BANK_TARGET_DEG,
            "fit_weighted_rmse_deg": bank_calibration.result.base_weighted_rmse,
            "fit_robust_scale_deg": bank_calibration.result.robust_scale,
            "leave_one_flight_out_scale_min": float(loo[:, 0].min()),
            "leave_one_flight_out_scale_max": float(loo[:, 0].max()),
            "leave_one_flight_out_bias_min_deg": float(loo[:, 1].min()),
            "leave_one_flight_out_bias_max_deg": float(loo[:, 1].max()),
            "delay_candidates": bank_calibration.delay_metrics.to_dict(
                orient="records"
            ),
            "validation_mean_flight_rmse_deg": float(
                validation_point_mass.bank_rmse_deg.mean()
            ),
        },
        "lift": {
            "input": "effective_pitch_angle_rad - simulated_flight_path_angle_rad",
            "feature_set": LIFT_FEATURE_SET,
            "feature_order": list(LIFT_FEATURE_SETS[LIFT_FEATURE_SET]),
            "coefficients": dict(
                zip(
                    LIFT_FEATURE_SETS[LIFT_FEATURE_SET],
                    lift_fit.model.coefficients,
                    strict=True,
                )
            ),
            "fit_weighted_cl_rmse": lift_fit.result.base_weighted_rmse,
            "fit_lofo_mean_flight_cl_rmse": float(lift_fit.lofo_metrics.cl_rmse.mean()),
            "validation_mean_flight_cl_rmse": float(validation_point_mass.cl_rmse.mean()),
        },
        "replay": {
            "plant": "aerox.models.PointMassModel",
            "sampled_input_rate_hz": 1,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "reset_window_s": RESET_WINDOW_S,
            "net_thrust_input": "frozen measured-N1 adapter evaluated at recorded conditions",
            "horizontal_metric": "WGS84 integration of predicted ground velocity",
            "aggregate_metrics": _aggregate_metrics(replay_metrics).to_dict(
                orient="records"
            ),
        },
        "split": {
            "fit_flights": [flight.key for flight in fit_flights],
            "evaluation_flights": [flight.key for flight in validation_flights],
            "evaluation_status": "held from these fits but previously opened in earlier commissioning",
        },
        "dependencies": {
            "vertical": str(args.vertical_parameters),
            "horizontal": str(args.horizontal_parameters),
            "thrust": str(args.thrust_parameters),
            "fuel": str(args.fuel_parameters),
        },
        "limitations": [
            "Pitch and AOA zero offsets are structurally confounded.",
            "Bank is an effective coordinated-turn calibration under zero sideslip and negligible side force.",
            "QAR thrust is inferred from measured N1 and remains conditional on the drag polar.",
            "Vertical wind, thrust inclination, configuration changes, and Earth-rate dynamics are omitted.",
            "The evaluation flights are diagnostic, not newly untouched confirmation data.",
        ],
    }
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(_native(parameters), sort_keys=False))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(replay_metrics).to_csv(args.metrics, index=False)
    _write_report(
        args.report,
        lift_fit=lift_fit,
        bank_calibration=bank_calibration,
        point_mass_metrics=point_mass_metrics,
        replay_metrics=replay_metrics,
    )
    return replay_metrics, parameters


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
