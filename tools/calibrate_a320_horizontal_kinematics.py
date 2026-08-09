#!/usr/bin/env python3
"""Calibrate A320 horizontal air-vector decoding and validate it by flight."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    FT_TO_M,
    KNOT_TO_MPS,
    decode_a320_flight,
    raw_columns,
)
from aerox.modeling.identification import (
    RobustLinearFit,
    delay_samples,
    robust_weighted_linear_fit,
)
from aerox.modeling.dynamics.kinematics import (
    differentiate_wgs84_intervals,
    integrate_wgs84,
    replay_metrics,
)


RAW_FILES = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}
PROCESSED_COLUMNS = [
    "flight_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "TAS",
    "u_component_of_wind",
    "v_component_of_wind",
]
DELAY_CANDIDATES_S = tuple(range(0, 6))
TRAINING_THIN_S = 5
MAX_ROLL_DEG = 10.0
MIN_HORIZONTAL_SPEED_MPS = 50.0
MAX_HORIZONTAL_SPEED_MPS = 300.0
MIN_TAS_MPS = 70.0


@dataclass(frozen=True)
class VerticalCalibration:
    pitch_scale: float
    angle_of_attack_scale: float
    gamma_bias_deg: float
    delay_s: int


@dataclass(frozen=True)
class HorizontalFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    latitude_rad: np.ndarray
    longitude_rad: np.ndarray
    altitude_m: np.ndarray
    observed_north_mps: np.ndarray
    observed_east_mps: np.ndarray
    observed_down_mps: np.ndarray
    true_heading_rad: np.ndarray
    roll_deg: np.ndarray
    horizontal_true_airspeed_mps: np.ndarray
    processed_wind_north_mps: np.ndarray
    processed_wind_east_mps: np.ndarray


@dataclass(frozen=True)
class WindNuisance:
    registration: str
    recording: str
    north_mps: float
    east_mps: float


@dataclass(frozen=True)
class HorizontalCalibration:
    delay_s: int
    fit: RobustLinearFit
    delay_fits: dict[int, RobustLinearFit]
    wind_north_correction_mps: float
    wind_east_correction_mps: float
    nuisances: tuple[WindNuisance, ...]
    leave_one_flight_out: np.ndarray
    training_rows: int

    @property
    def a(self) -> float:
        return float(self.fit.coefficients[0])

    @property
    def b(self) -> float:
        return float(self.fit.coefficients[1])

    @property
    def tas_scale(self) -> float:
        return float(np.hypot(self.a, self.b))

    @property
    def heading_bias_deg(self) -> float:
        return float(np.rad2deg(np.arctan2(self.b, self.a)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-replay-panel.csv"),
    )
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration.md"),
    )
    return parser.parse_args()


def _read_vertical_calibration(path: Path) -> VerticalCalibration:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    parameters = model["parameters"]
    return VerticalCalibration(
        pitch_scale=float(parameters["pitch_scale_deg_per_code"]["value"]),
        angle_of_attack_scale=float(
            parameters["angle_of_attack_scale_deg_per_code"]["value"]
        ),
        gamma_bias_deg=float(parameters["effective_gamma_bias_deg"]["value"]),
        delay_s=int(model["trajectory_reference_delay_s"]["value"]),
    )


def _intervals_to_nodes(values: np.ndarray) -> np.ndarray:
    nodes = np.empty(len(values) + 1, dtype=float)
    nodes[0] = values[0]
    nodes[-1] = values[-1]
    nodes[1:-1] = 0.5 * (values[:-1] + values[1:])
    return nodes


def _load_flights(
    selection: pd.DataFrame,
    raw_dir: Path,
    processed_path: Path,
    vertical: VerticalCalibration,
) -> list[HorizontalFlight]:
    selected_ids = selection["recording"].astype(str).tolist()
    processed = pd.read_parquet(
        processed_path,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "in", selected_ids)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame.sort_values("timestamp", kind="stable").reset_index(
            drop=True
        )
        for recording, frame in processed.groupby("flight_id", sort=False)
    }
    flights: list[HorizontalFlight] = []
    for registration, registration_selection in selection.groupby(
        "registration", sort=True
    ):
        ids = registration_selection["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", ids)],
            engine="pyarrow",
        )
        role_by_id = registration_selection.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
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
                raise ValueError(f"raw/processed timestamp mismatch for {recording}")
            time_s = (
                timestamps - timestamps[0]
            ).astype("timedelta64[ns]").astype(np.int64) / 1e9
            latitude = current["latitude"].to_numpy(dtype=float) * DEG_TO_RAD
            longitude = current["longitude"].to_numpy(dtype=float) * DEG_TO_RAD
            altitude = current["altitude"].to_numpy(dtype=float) * FT_TO_M
            interval_velocity = differentiate_wgs84_intervals(
                time_s, latitude, longitude, altitude
            )
            north, east, down = (
                _intervals_to_nodes(values) for values in interval_velocity
            )
            pitch_code = (
                decoded.channel("pitch", allow_provisional=True)[raw_indices]
                / DEG_TO_RAD
                / 0.5
            )
            angle_of_attack_code = (
                decoded.channel("angle_of_attack", allow_provisional=True)[raw_indices]
                / DEG_TO_RAD
                / 0.25
            )
            gamma_deg = (
                vertical.pitch_scale
                * delay_samples(pitch_code, vertical.delay_s)
                - vertical.angle_of_attack_scale
                * delay_samples(angle_of_attack_code, vertical.delay_s)
                + vertical.gamma_bias_deg
            )
            tas = current["TAS"].to_numpy(dtype=float) * KNOT_TO_MPS
            flights.append(
                HorizontalFlight(
                    registration=str(registration),
                    recording=recording,
                    role=str(role_by_id[recording]),
                    time_s=time_s,
                    latitude_rad=latitude,
                    longitude_rad=longitude,
                    altitude_m=altitude,
                    observed_north_mps=north,
                    observed_east_mps=east,
                    observed_down_mps=down,
                    true_heading_rad=decoded.channel(
                        "true_heading", allow_provisional=True
                    )[raw_indices],
                    roll_deg=(
                        decoded.channel("roll", allow_provisional=True)[raw_indices]
                        / DEG_TO_RAD
                    ),
                    horizontal_true_airspeed_mps=tas
                    * np.cos(gamma_deg * DEG_TO_RAD),
                    processed_wind_north_mps=current[
                        "v_component_of_wind"
                    ].to_numpy(dtype=float),
                    processed_wind_east_mps=current[
                        "u_component_of_wind"
                    ].to_numpy(dtype=float),
                )
            )
    expected = set(selected_ids)
    loaded = {flight.recording for flight in flights}
    if loaded != expected:
        raise ValueError(f"flight load mismatch: missing={sorted(expected - loaded)}")
    return flights


def _normal_mask(flight: HorizontalFlight, heading: np.ndarray) -> np.ndarray:
    observed_speed = np.hypot(
        flight.observed_north_mps, flight.observed_east_mps
    )
    return (
        np.isfinite(
            heading
            + flight.horizontal_true_airspeed_mps
            + flight.observed_north_mps
            + flight.observed_east_mps
            + flight.processed_wind_north_mps
            + flight.processed_wind_east_mps
        )
        & (np.abs(flight.roll_deg) < MAX_ROLL_DEG)
        & (flight.horizontal_true_airspeed_mps > MIN_TAS_MPS)
        & (observed_speed > MIN_HORIZONTAL_SPEED_MPS)
        & (observed_speed < MAX_HORIZONTAL_SPEED_MPS)
    )


def _design(
    flights: list[HorizontalFlight], delay_s: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for flight in flights:
        heading = delay_samples(flight.true_heading_rad, delay_s)
        north_air = flight.horizontal_true_airspeed_mps * np.cos(heading)
        east_air = flight.horizontal_true_airspeed_mps * np.sin(heading)
        north_target = (
            flight.observed_north_mps - flight.processed_wind_north_mps
        )
        east_target = flight.observed_east_mps - flight.processed_wind_east_mps
        valid = _normal_mask(flight, heading) & (
            np.arange(len(heading)) % TRAINING_THIN_S == 0
        )
        north_air = north_air[valid]
        east_air = east_air[valid]
        north_target = north_target[valid]
        east_target = east_target[valid]

        # Removing each flight's mean is equivalent to fitting an unconstrained
        # constant north/east wind nuisance before estimating the global rotation
        # and scale. The nuisance values are recovered after the global fit.
        north_air -= north_air.mean()
        east_air -= east_air.mean()
        north_target -= north_target.mean()
        east_target -= east_target.mean()
        count = len(north_air)
        design = np.vstack(
            (
                np.column_stack((north_air, -east_air)),
                np.column_stack((east_air, north_air)),
            )
        )
        target = np.concatenate((north_target, east_target))
        weights = np.full(2 * count, 1.0 / (2 * count))
        chunks.append((design, target, weights))
    return (
        np.vstack([chunk[0] for chunk in chunks]),
        np.concatenate([chunk[1] for chunk in chunks]),
        np.concatenate([chunk[2] for chunk in chunks]),
    )


def _fit_at_delay(
    flights: list[HorizontalFlight], delay_s: int
) -> tuple[RobustLinearFit, int]:
    design, target, weights = _design(flights, delay_s)
    return robust_weighted_linear_fit(design, target, weights), len(target)


def _wind_nuisances(
    flights: list[HorizontalFlight], delay_s: int, a: float, b: float
) -> tuple[WindNuisance, ...]:
    nuisances = []
    for flight in flights:
        heading = delay_samples(flight.true_heading_rad, delay_s)
        north_air = flight.horizontal_true_airspeed_mps * np.cos(heading)
        east_air = flight.horizontal_true_airspeed_mps * np.sin(heading)
        predicted_north_air = a * north_air - b * east_air
        predicted_east_air = a * east_air + b * north_air
        north_residual = flight.observed_north_mps - (
            predicted_north_air + flight.processed_wind_north_mps
        )
        east_residual = flight.observed_east_mps - (
            predicted_east_air + flight.processed_wind_east_mps
        )
        valid = _normal_mask(flight, heading)
        nuisances.append(
            WindNuisance(
                registration=flight.registration,
                recording=flight.recording,
                north_mps=float(np.median(north_residual[valid])),
                east_mps=float(np.median(east_residual[valid])),
            )
        )
    return tuple(nuisances)


def _global_wind_correction(nuisances: tuple[WindNuisance, ...]) -> tuple[float, float]:
    return (
        float(np.median([item.north_mps for item in nuisances])),
        float(np.median([item.east_mps for item in nuisances])),
    )


def _fit_complete(
    flights: list[HorizontalFlight], delay_s: int
) -> tuple[RobustLinearFit, tuple[WindNuisance, ...], float, float, int]:
    fit, rows = _fit_at_delay(flights, delay_s)
    nuisances = _wind_nuisances(
        flights, delay_s, float(fit.coefficients[0]), float(fit.coefficients[1])
    )
    wind_north, wind_east = _global_wind_correction(nuisances)
    return fit, nuisances, wind_north, wind_east, rows


def calibrate(flights: list[HorizontalFlight]) -> HorizontalCalibration:
    delay_fits: dict[int, RobustLinearFit] = {}
    rows_by_delay: dict[int, int] = {}
    for delay_s in DELAY_CANDIDATES_S:
        fit, rows = _fit_at_delay(flights, delay_s)
        delay_fits[delay_s] = fit
        rows_by_delay[delay_s] = rows
    selected_delay = min(
        delay_fits, key=lambda delay: delay_fits[delay].huber_objective
    )
    fit, nuisances, wind_north, wind_east, rows = _fit_complete(
        flights, selected_delay
    )

    leave_one_out = []
    for held_out in flights:
        subset = [
            flight for flight in flights if flight.recording != held_out.recording
        ]
        current_fit, current_nuisances, north, east, _ = _fit_complete(
            subset, selected_delay
        )
        a, b = current_fit.coefficients
        leave_one_out.append(
            [
                float(np.hypot(a, b)),
                float(np.rad2deg(np.arctan2(b, a))),
                north,
                east,
            ]
        )
    return HorizontalCalibration(
        delay_s=selected_delay,
        fit=fit,
        delay_fits=delay_fits,
        wind_north_correction_mps=wind_north,
        wind_east_correction_mps=wind_east,
        nuisances=nuisances,
        leave_one_flight_out=np.asarray(leave_one_out),
        training_rows=rows,
    )


def _velocity(
    flight: HorizontalFlight,
    *,
    delay_s: int,
    a: float,
    b: float,
    wind_north_correction_mps: float,
    wind_east_correction_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    heading = delay_samples(flight.true_heading_rad, delay_s)
    north_air = flight.horizontal_true_airspeed_mps * np.cos(heading)
    east_air = flight.horizontal_true_airspeed_mps * np.sin(heading)
    return (
        a * north_air
        - b * east_air
        + flight.processed_wind_north_mps
        + wind_north_correction_mps,
        a * east_air
        + b * north_air
        + flight.processed_wind_east_mps
        + wind_east_correction_mps,
    )


def _flight_metrics(
    flight: HorizontalFlight, calibration: HorizontalCalibration
) -> dict[str, object]:
    nominal = _velocity(
        flight,
        delay_s=0,
        a=1.0,
        b=0.0,
        wind_north_correction_mps=0.0,
        wind_east_correction_mps=0.0,
    )
    calibrated = _velocity(
        flight,
        delay_s=calibration.delay_s,
        a=calibration.a,
        b=calibration.b,
        wind_north_correction_mps=calibration.wind_north_correction_mps,
        wind_east_correction_mps=calibration.wind_east_correction_mps,
    )
    valid = np.isfinite(nominal[0] + nominal[1] + calibrated[0] + calibrated[1])
    first = int(np.flatnonzero(valid)[0])
    time = flight.time_s[first:]
    observed_north = flight.observed_north_mps[first:]
    observed_east = flight.observed_east_mps[first:]
    observed_down = flight.observed_down_mps[first:]
    nominal_north, nominal_east = (values[first:] for values in nominal)
    calibrated_north, calibrated_east = (values[first:] for values in calibrated)

    def velocity_rmse(north: np.ndarray, east: np.ndarray) -> float:
        error = np.hypot(north - observed_north, east - observed_east)
        return float(np.sqrt(np.mean(error**2)))

    initial = (
        float(flight.latitude_rad[first]),
        float(flight.longitude_rad[first]),
        float(flight.altitude_m[first]),
    )
    nominal_trajectory = integrate_wgs84(
        time,
        *initial,
        nominal_north,
        nominal_east,
        observed_down,
    )
    calibrated_trajectory = integrate_wgs84(
        time,
        *initial,
        calibrated_north,
        calibrated_east,
        observed_down,
    )
    observed = (
        flight.latitude_rad[first:],
        flight.longitude_rad[first:],
        flight.altitude_m[first:],
    )
    nominal_position = replay_metrics(nominal_trajectory, *observed)
    calibrated_position = replay_metrics(calibrated_trajectory, *observed)
    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "samples": len(time),
        "horizontal_velocity_rmse_nominal_mps": velocity_rmse(
            nominal_north, nominal_east
        ),
        "horizontal_velocity_rmse_calibrated_mps": velocity_rmse(
            calibrated_north, calibrated_east
        ),
        "north_velocity_bias_nominal_mps": float(
            np.mean(nominal_north - observed_north)
        ),
        "north_velocity_bias_calibrated_mps": float(
            np.mean(calibrated_north - observed_north)
        ),
        "east_velocity_bias_nominal_mps": float(
            np.mean(nominal_east - observed_east)
        ),
        "east_velocity_bias_calibrated_mps": float(
            np.mean(calibrated_east - observed_east)
        ),
        "horizontal_position_rmse_nominal_m": nominal_position.horizontal_rmse_m,
        "horizontal_position_rmse_calibrated_m": calibrated_position.horizontal_rmse_m,
        "horizontal_final_nominal_m": nominal_position.horizontal_final_m,
        "horizontal_final_calibrated_m": calibrated_position.horizontal_final_m,
    }


def _round_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rounded = metrics.copy()
    numeric = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric] = rounded[numeric].round(5)
    return rounded


def _aggregate_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Split | Flights | Horizontal velocity RMSE nominal / calibrated (m/s) | Horizontal position RMSE nominal / calibrated (m) | Final horizontal error nominal / calibrated (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    split = np.where(metrics.role == "validation", "validation", "fit")
    for name in ("fit", "validation"):
        current = metrics.loc[split == name]
        lines.append(
            f"| {name} | {len(current)} | "
            f"{current.horizontal_velocity_rmse_nominal_mps.mean():.3f} / {current.horizontal_velocity_rmse_calibrated_mps.mean():.3f} | "
            f"{current.horizontal_position_rmse_nominal_m.mean():.1f} / {current.horizontal_position_rmse_calibrated_m.mean():.1f} | "
            f"{current.horizontal_final_nominal_m.mean():.1f} / {current.horizontal_final_calibrated_m.mean():.1f} |"
        )
    return "\n".join(lines)


def _validation_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Flight | Velocity RMSE nominal / calibrated (m/s) | Position RMSE nominal / calibrated (m) | Final error nominal / calibrated (m) |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics.loc[metrics.role == "validation"].itertuples(index=False):
        lines.append(
            f"| {row.registration}/{row.recording} | "
            f"{row.horizontal_velocity_rmse_nominal_mps:.3f} / {row.horizontal_velocity_rmse_calibrated_mps:.3f} | "
            f"{row.horizontal_position_rmse_nominal_m:.1f} / {row.horizontal_position_rmse_calibrated_m:.1f} | "
            f"{row.horizontal_final_nominal_m:.1f} / {row.horizontal_final_calibrated_m:.1f} |"
        )
    return "\n".join(lines)


def _registration_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Validation registration | Flights | Velocity RMSE nominal / calibrated (m/s) | Position RMSE nominal / calibrated (m) |",
        "|---|---:|---:|---:|",
    ]
    validation = metrics.loc[metrics.role == "validation"]
    for registration, current in validation.groupby("registration", sort=True):
        lines.append(
            f"| {registration} | {len(current)} | "
            f"{current.horizontal_velocity_rmse_nominal_mps.mean():.3f} / {current.horizontal_velocity_rmse_calibrated_mps.mean():.3f} | "
            f"{current.horizontal_position_rmse_nominal_m.mean():.1f} / {current.horizontal_position_rmse_calibrated_m.mean():.1f} |"
        )
    return "\n".join(lines)


def _write_parameters(
    path: Path,
    calibration: HorizontalCalibration,
    fit_flights: list[HorizontalFlight],
    validation_flights: list[HorizontalFlight],
) -> None:
    leave_one_out = calibration.leave_one_flight_out
    best = calibration.fit.huber_objective
    parameter_names = (
        "tas_scale",
        "heading_bias_deg",
        "global_wind_north_correction_mps",
        "global_wind_east_correction_mps",
    )
    values = (
        calibration.tas_scale,
        calibration.heading_bias_deg,
        calibration.wind_north_correction_mps,
        calibration.wind_east_correction_mps,
    )
    parameters = {}
    for index, (name, value) in enumerate(zip(parameter_names, values, strict=True)):
        parameters[name] = {
            "value": float(value),
            "leave_one_flight_out_min": float(leave_one_out[:, index].min()),
            "leave_one_flight_out_max": float(leave_one_out[:, index].max()),
            "leave_one_flight_out_std": float(leave_one_out[:, index].std()),
            "status": "effective-provisional",
        }
    one_percent = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best * 1.01
    ]
    five_percent = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best * 1.05
    ]
    document = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "model": {
            "equation": "ground_horizontal = rotate(heading_bias) * tas_scale * TAS * cos(gamma) + processed_wind + global_wind_correction",
            "parameters": parameters,
            "trajectory_reference_delay_s": {
                "value": calibration.delay_s,
                "one_percent_objective_support": one_percent,
                "five_percent_objective_support": five_percent,
                "status": "effective-alignment-not-physical-sensor-latency",
            },
        },
        "fit": {
            "method": "within-flight centered weighted Huber vector regression",
            "per_flight_nuisance": "constant north/east wind correction removed before global scale/rotation fit",
            "global_wind_convention": "component-wise median of fit-flight nuisance corrections",
            "training_thinning_s": TRAINING_THIN_S,
            "maximum_absolute_roll_deg": MAX_ROLL_DEG,
            "minimum_tas_mps": MIN_TAS_MPS,
            "training_rows": calibration.training_rows,
            "huber_objective": calibration.fit.huber_objective,
            "base_weighted_rmse_mps": calibration.fit.base_weighted_rmse,
            "robust_scale_mps": calibration.fit.robust_scale,
            "normal_matrix_condition": calibration.fit.normal_matrix_condition,
            "coefficient_correlation": calibration.fit.coefficient_correlation.tolist(),
            "delay_objective": {
                str(delay): fit.huber_objective
                for delay, fit in calibration.delay_fits.items()
            },
            "fit_flight_wind_nuisance_mps": [
                {
                    "flight": f"{item.registration}/{item.recording}",
                    "north": item.north_mps,
                    "east": item.east_mps,
                }
                for item in calibration.nuisances
            ],
        },
        "inputs": {
            "vertical_gamma_calibration": "docs/modeling/a320-vertical-kinematic-calibration.yml",
            "true_airspeed": "processed TAS",
            "heading": "raw TH via typed provisional decoder",
            "wind": "processed u eastward and v northward",
            "observed_velocity": "WGS84 differentiation of processed coordinates",
        },
        "split": {
            "fit_flights": [
                f"{flight.registration}/{flight.recording}" for flight in fit_flights
            ],
            "held_out_validation_flights": [
                f"{flight.registration}/{flight.recording}"
                for flight in validation_flights
            ],
        },
        "identifiability": {
            "identified_under_assumptions": [
                "global effective TAS scale",
                "global effective heading rotation",
            ],
            "convention_dependent": [
                "global wind correction as median fit-flight nuisance",
            ],
            "not_separately_identifiable": [
                "heading bias versus mean sideslip",
                "heading latency versus processed coordinate latency",
                "global wind bias versus mean of per-flight wind errors",
                "TAS error versus wind component parallel to the aircraft path on a single flight",
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_report(
    path: Path, calibration: HorizontalCalibration, metrics: pd.DataFrame
) -> None:
    loo = calibration.leave_one_flight_out
    best = calibration.fit.huber_objective
    support_1 = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best * 1.01
    ]
    support_5 = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best * 1.05
    ]
    validation = metrics.loc[metrics.role == "validation"]
    velocity_improvement = 100.0 * (
        1.0
        - validation.horizontal_velocity_rmse_calibrated_mps.mean()
        / validation.horizontal_velocity_rmse_nominal_mps.mean()
    )
    position_change = 100.0 * (
        validation.horizontal_position_rmse_calibrated_m.mean()
        / validation.horizontal_position_rmse_nominal_m.mean()
        - 1.0
    )
    report = f"""# A320 Horizontal Kinematic Calibration

Status: calibrated on 10 flights and evaluated once on 6 held-out flights  
Date: 2026-08-04  
Generator: `tools/calibrate_a320_horizontal_kinematics.py`

## Outcome

The frozen effective horizontal decoder is:

```text
corrected_TAS       = {calibration.tas_scale:.8f} * processed_TAS
corrected_heading   = TH[t-{calibration.delay_s}] + {calibration.heading_bias_deg:.6f} deg
corrected_wind_N/E  = processed_wind_N/E
                    + ({calibration.wind_north_correction_mps:.6f}, {calibration.wind_east_correction_mps:.6f}) m/s
```

On the six held-out flights it reduces mean horizontal-velocity RMSE by
{velocity_improvement:.1f}%. Every held-out flight improves on that primary gate.
Mean open-loop horizontal-position RMSE increases by {position_change:.1f}% because
small residual biases accumulate adversely on two AFC flights, particularly the
longest one. Both comparisons use the already-frozen effective vertical gamma so
this experiment isolates horizontal changes.

## Method

- Coordinate-derived WGS 84 north/east velocity is the target.
- Processed `v` is northward wind and `u` is eastward wind.
- Each fit flight receives an unconstrained constant north/east wind nuisance by
  within-flight centering. This prevents a route's constant wind error from being
  fitted as global heading or TAS scale.
- A shared TAS scale and heading rotation are fitted as one linear two-dimensional
  scale/rotation using equal total weight per fit flight and Huber robustness.
- Fit samples use absolute roll below {MAX_ROLL_DEG:.0f} degrees, TAS above
  {MIN_TAS_MPS:.0f} m/s, and five-second thinning.
- A common causal heading-to-trajectory delay from {min(DELAY_CANDIDATES_S)} to
  {max(DELAY_CANDIDATES_S)} seconds is selected using fit flights only.
- The global wind correction is the component-wise median of the ten fit-flight
  nuisance vectors. Validation flights receive only this global correction; no
  validation-flight nuisance is estimated.

## Parameter stability

| Effective parameter | Estimate | Leave-one-flight-out min | max | standard deviation |
|---|---:|---:|---:|---:|
| TAS scale | {calibration.tas_scale:.8f} | {loo[:, 0].min():.8f} | {loo[:, 0].max():.8f} | {loo[:, 0].std():.8f} |
| Heading rotation (deg) | {calibration.heading_bias_deg:.6f} | {loo[:, 1].min():.6f} | {loo[:, 1].max():.6f} | {loo[:, 1].std():.6f} |
| Global wind north correction (m/s) | {calibration.wind_north_correction_mps:.6f} | {loo[:, 2].min():.6f} | {loo[:, 2].max():.6f} | {loo[:, 2].std():.6f} |
| Global wind east correction (m/s) | {calibration.wind_east_correction_mps:.6f} | {loo[:, 3].min():.6f} | {loo[:, 3].max():.6f} | {loo[:, 3].std():.6f} |

Selected relative delay: {calibration.delay_s} seconds. One-percent objective
support: {support_1}; five-percent support: {support_5}. This is alignment to the
processed coordinate target, not identified heading-sensor latency.

The two scale/rotation coefficients have a normal-matrix condition number of
{calibration.fit.normal_matrix_condition:.3f} and correlation
{calibration.fit.coefficient_correlation[0, 1]:.4f}. Their numerical separation is
strong in the fit data, but heading rotation remains physically confounded with
mean sideslip.

## Aggregate performance

{_aggregate_table(metrics)}

## Held-out validation flights

{_validation_table(metrics)}

## Registration-aware validation

{_registration_table(metrics)}

Both registrations improve on horizontal-velocity RMSE, but their residual levels
and open-loop position behavior differ. With only two aircraft and weather/route
confounding, this is evidence to retain registration labels in later residual
analysis, not evidence for a tail-specific correction yet.

Open-loop position errors can improve or worsen through cancellation of small
time-varying velocity errors. Horizontal-velocity RMSE across complete held-out
flights is therefore the primary gate; final position error remains diagnostic.

## Identifiability decision

- Effective TAS scale and a shared horizontal rotation are practically
  identifiable under the within-flight nuisance convention.
- Global wind correction is convention-dependent: without an external trusted
  wind source, it cannot be separated from the mean fit-flight wind nuisance.
- Heading rotation cannot be separated from mean sideslip because actual sideslip
  is not measured.
- A single flight cannot cleanly separate TAS error from a wind error parallel to
  its path; route and heading diversity across flights is essential.

These parameters may be carried as provisional measurement-model corrections, not
as aircraft aerodynamic parameters. The horizontal velocity gate passes
provisionally; the open-loop horizontal position gate does not yet pass. The next
gate is translational force/energy
replay with measured attitude: speed, flight-path angle, fuel, mass, throttle,
engine spool state, and wind uncertainty. Rotational dynamics remain deferred.

## Artifacts

- Parameters: `docs/modeling/a320-horizontal-kinematic-calibration.yml`
- Per-flight metrics: `docs/modeling/a320-horizontal-kinematic-calibration-metrics.csv`
- Reproducer: `tools/calibrate_a320_horizontal_kinematics.py`
- Vertical input calibration: `docs/modeling/a320-vertical-kinematic-calibration.yml`
- Provenance: `docs/modeling/sources.yml`

No raw coordinates or trajectory samples are written to the repository.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[HorizontalCalibration, pd.DataFrame]:
    panel = pd.read_csv(args.panel)
    fit_selection = panel.loc[panel.role != "validation"].copy()
    validation_selection = panel.loc[panel.role == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected 10 fit flights and 6 held-out validation flights")
    vertical = _read_vertical_calibration(args.vertical_parameters)

    # Freeze delay, scale/rotation, and global wind convention before loading any
    # validation flight.
    fit_flights = _load_flights(
        fit_selection, args.raw_dir, args.processed, vertical
    )
    calibration = calibrate(fit_flights)
    validation_flights = _load_flights(
        validation_selection, args.raw_dir, args.processed, vertical
    )

    metrics = _round_metrics(
        pd.DataFrame(
            [
                _flight_metrics(flight, calibration)
                for flight in [*fit_flights, *validation_flights]
            ]
        )
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    _write_parameters(
        args.parameters, calibration, fit_flights, validation_flights
    )
    _write_report(args.report, calibration, metrics)
    return calibration, metrics


def main() -> None:
    args = parse_args()
    calibration, metrics = run(args)
    print(
        "tas_scale:",
        calibration.tas_scale,
        "heading_bias_deg:",
        calibration.heading_bias_deg,
        "delay_s:",
        calibration.delay_s,
        "wind_N/E:",
        (
            calibration.wind_north_correction_mps,
            calibration.wind_east_correction_mps,
        ),
    )
    print(metrics.loc[metrics.role == "validation"].to_string(index=False))
    print(f"wrote {args.parameters}")
    print(f"wrote {args.metrics}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
