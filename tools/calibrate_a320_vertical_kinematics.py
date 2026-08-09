#!/usr/bin/env python3
"""Calibrate and validate an effective A320 pitch/AOA-to-flight-path decoder."""

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


RAW_FILES = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}
PROCESSED_COLUMNS = [
    "flight_id",
    "timestamp",
    "altitude",
    "TAS",
    "flight_path_angle",
]
NOMINAL_COEFFICIENTS = np.array([0.5, 0.25, 0.0])
SMOOTHING_WINDOW_S = 11
TRAINING_THIN_S = 5
MAX_ROLL_DEG = 10.0
MIN_TAS_MPS = 70.0
DELAY_CANDIDATES_S = tuple(range(0, 6))


@dataclass(frozen=True)
class FlightKinematics:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    pitch_code: np.ndarray
    angle_of_attack_code: np.ndarray
    roll_deg: np.ndarray
    true_airspeed_mps: np.ndarray
    altitude_m: np.ndarray
    reference_flight_path_deg: np.ndarray
    recorded_flight_path_deg: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    delay_s: int
    fit: RobustLinearFit
    delay_fits: dict[int, RobustLinearFit]
    leave_one_flight_out: np.ndarray
    training_rows: int

    @property
    def pitch_scale(self) -> float:
        return float(self.fit.coefficients[0])

    @property
    def angle_of_attack_scale(self) -> float:
        return float(self.fit.coefficients[1])

    @property
    def effective_gamma_bias_deg(self) -> float:
        return float(self.fit.coefficients[2])


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
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.md"),
    )
    return parser.parse_args()


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    numerator = np.convolve(values, np.ones(window), mode="same")
    denominator = np.convolve(np.ones(len(values)), np.ones(window), mode="same")
    return numerator / denominator


def _reference_flight_path(
    time_s: np.ndarray, altitude_m: np.ndarray, true_airspeed_mps: np.ndarray
) -> np.ndarray:
    vertical_velocity = np.gradient(altitude_m, time_s)
    ratio = np.divide(
        vertical_velocity,
        true_airspeed_mps,
        out=np.full_like(vertical_velocity, np.nan),
        where=true_airspeed_mps > 0,
    )
    flight_path = np.rad2deg(np.arcsin(np.clip(ratio, -0.3, 0.3)))
    return _moving_average(flight_path, SMOOTHING_WINDOW_S)


def _load_flights(
    selection: pd.DataFrame, raw_dir: Path, processed_path: Path
) -> list[FlightKinematics]:
    flights: list[FlightKinematics] = []
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

    for registration, registration_selection in selection.groupby(
        "registration", sort=True
    ):
        ids = registration_selection["recording"].astype(str).tolist()
        raw_path = raw_dir / RAW_FILES[str(registration)]
        raw = pd.read_parquet(
            raw_path,
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
            roll_deg = (
                decoded.channel("roll", allow_provisional=True)[raw_indices]
                / DEG_TO_RAD
            )
            tas = current["TAS"].to_numpy(dtype=float) * KNOT_TO_MPS
            altitude = current["altitude"].to_numpy(dtype=float) * FT_TO_M
            flights.append(
                FlightKinematics(
                    registration=str(registration),
                    recording=recording,
                    role=str(role_by_id[recording]),
                    time_s=time_s,
                    pitch_code=pitch_code,
                    angle_of_attack_code=angle_of_attack_code,
                    roll_deg=roll_deg,
                    true_airspeed_mps=tas,
                    altitude_m=altitude,
                    reference_flight_path_deg=_reference_flight_path(
                        time_s, altitude, tas
                    ),
                    recorded_flight_path_deg=current[
                        "flight_path_angle"
                    ].to_numpy(dtype=float),
                )
            )
    expected = set(selected_ids)
    loaded = {flight.recording for flight in flights}
    if loaded != expected:
        raise ValueError(f"flight load mismatch: missing={sorted(expected - loaded)}")
    return flights


def _design(
    flights: list[FlightKinematics], delay_s: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for flight in flights:
        pitch = delay_samples(flight.pitch_code, delay_s)
        angle_of_attack = delay_samples(flight.angle_of_attack_code, delay_s)
        target = flight.reference_flight_path_deg
        valid = (
            np.isfinite(pitch + angle_of_attack + target)
            & (np.abs(flight.roll_deg) < MAX_ROLL_DEG)
            & (flight.true_airspeed_mps > MIN_TAS_MPS)
            & (np.arange(len(target)) % TRAINING_THIN_S == 0)
        )
        phase = np.where(target > 1.0, 1, np.where(target < -1.0, -1, 0))
        for phase_code in (-1, 0, 1):
            indices = np.flatnonzero(valid & (phase == phase_code))
            if not len(indices):
                continue
            design = np.column_stack(
                (
                    pitch[indices],
                    -angle_of_attack[indices],
                    np.ones(len(indices)),
                )
            )
            # Every flight/phase stratum has equal total base weight. This stops
            # long cruise segments from dominating climb and descent information.
            weights = np.full(len(indices), 1.0 / len(indices))
            chunks.append((design, target[indices], weights))
    if not chunks:
        raise ValueError("no calibration samples passed the selection rules")
    return (
        np.vstack([chunk[0] for chunk in chunks]),
        np.concatenate([chunk[1] for chunk in chunks]),
        np.concatenate([chunk[2] for chunk in chunks]),
    )


def _fit_at_delay(
    flights: list[FlightKinematics], delay_s: int
) -> tuple[RobustLinearFit, int]:
    design, target, weights = _design(flights, delay_s)
    fit = robust_weighted_linear_fit(design, target, weights)
    return fit, len(target)


def calibrate(flights: list[FlightKinematics]) -> CalibrationResult:
    delay_fits: dict[int, RobustLinearFit] = {}
    row_counts: dict[int, int] = {}
    for delay_s in DELAY_CANDIDATES_S:
        fit, rows = _fit_at_delay(flights, delay_s)
        delay_fits[delay_s] = fit
        row_counts[delay_s] = rows
    selected_delay = min(
        delay_fits, key=lambda delay: delay_fits[delay].huber_objective
    )
    selected_fit = delay_fits[selected_delay]

    leave_one_out = []
    for held_out in flights:
        subset = [
            flight for flight in flights if flight.recording != held_out.recording
        ]
        fit, _ = _fit_at_delay(subset, selected_delay)
        leave_one_out.append(fit.coefficients)
    return CalibrationResult(
        delay_s=selected_delay,
        fit=selected_fit,
        delay_fits=delay_fits,
        leave_one_flight_out=np.vstack(leave_one_out),
        training_rows=row_counts[selected_delay],
    )


def _integrate_altitude(
    time_s: np.ndarray, initial_altitude_m: float, vertical_velocity_mps: np.ndarray
) -> np.ndarray:
    altitude = np.empty(len(time_s), dtype=float)
    altitude[0] = initial_altitude_m
    for index, step in enumerate(np.diff(time_s), start=1):
        mean_velocity = 0.5 * (
            vertical_velocity_mps[index - 1] + vertical_velocity_mps[index]
        )
        altitude[index] = altitude[index - 1] + mean_velocity * step
    return altitude


def _flight_metrics(
    flight: FlightKinematics, calibration: CalibrationResult
) -> dict[str, object]:
    pitch = delay_samples(flight.pitch_code, calibration.delay_s)
    angle_of_attack = delay_samples(
        flight.angle_of_attack_code, calibration.delay_s
    )
    calibrated = (
        calibration.pitch_scale * pitch
        - calibration.angle_of_attack_scale * angle_of_attack
        + calibration.effective_gamma_bias_deg
    )
    nominal = 0.5 * flight.pitch_code - 0.25 * flight.angle_of_attack_code
    reference = flight.reference_flight_path_deg
    valid = np.isfinite(calibrated + nominal + reference)
    first = int(np.flatnonzero(valid)[0])
    time = flight.time_s[first:]
    reference = reference[first:]
    nominal = nominal[first:]
    calibrated = calibrated[first:]
    tas = flight.true_airspeed_mps[first:]
    observed_altitude = flight.altitude_m[first:]
    observed_vertical_velocity = np.gradient(observed_altitude, time)
    nominal_vertical_velocity = tas * np.sin(nominal * DEG_TO_RAD)
    calibrated_vertical_velocity = tas * np.sin(calibrated * DEG_TO_RAD)
    nominal_altitude = _integrate_altitude(
        time, observed_altitude[0], nominal_vertical_velocity
    )
    calibrated_altitude = _integrate_altitude(
        time, observed_altitude[0], calibrated_vertical_velocity
    )
    nominal_gamma_error = nominal - reference
    calibrated_gamma_error = calibrated - reference
    nominal_altitude_error = nominal_altitude - observed_altitude
    calibrated_altitude_error = calibrated_altitude - observed_altitude

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2)))

    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "samples": len(time),
        "gamma_rmse_nominal_deg": rmse(nominal_gamma_error),
        "gamma_rmse_calibrated_deg": rmse(calibrated_gamma_error),
        "gamma_bias_nominal_deg": float(np.mean(nominal_gamma_error)),
        "gamma_bias_calibrated_deg": float(np.mean(calibrated_gamma_error)),
        "vertical_velocity_rmse_nominal_mps": rmse(
            nominal_vertical_velocity - observed_vertical_velocity
        ),
        "vertical_velocity_rmse_calibrated_mps": rmse(
            calibrated_vertical_velocity - observed_vertical_velocity
        ),
        "altitude_rmse_nominal_m": rmse(nominal_altitude_error),
        "altitude_rmse_calibrated_m": rmse(calibrated_altitude_error),
        "altitude_final_nominal_m": float(nominal_altitude_error[-1]),
        "altitude_final_calibrated_m": float(calibrated_altitude_error[-1]),
    }


def _round_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rounded = metrics.copy()
    numeric = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric] = rounded[numeric].round(5)
    return rounded


def _aggregate_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Split | Flights | Gamma RMSE nominal / calibrated (deg) | Vertical-velocity RMSE nominal / calibrated (m/s) | Altitude RMSE nominal / calibrated (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    split = np.where(metrics["role"] == "validation", "validation", "fit")
    for name in ("fit", "validation"):
        current = metrics.loc[split == name]
        lines.append(
            f"| {name} | {len(current)} | "
            f"{current.gamma_rmse_nominal_deg.mean():.3f} / {current.gamma_rmse_calibrated_deg.mean():.3f} | "
            f"{current.vertical_velocity_rmse_nominal_mps.mean():.3f} / {current.vertical_velocity_rmse_calibrated_mps.mean():.3f} | "
            f"{current.altitude_rmse_nominal_m.mean():.1f} / {current.altitude_rmse_calibrated_m.mean():.1f} |"
        )
    return "\n".join(lines)


def _validation_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Flight | Gamma RMSE nominal / calibrated (deg) | Vertical velocity RMSE nominal / calibrated (m/s) | Final altitude error nominal / calibrated (m) |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics.loc[metrics.role == "validation"].itertuples(index=False):
        lines.append(
            f"| {row.registration}/{row.recording} | "
            f"{row.gamma_rmse_nominal_deg:.3f} / {row.gamma_rmse_calibrated_deg:.3f} | "
            f"{row.vertical_velocity_rmse_nominal_mps:.3f} / {row.vertical_velocity_rmse_calibrated_mps:.3f} | "
            f"{row.altitude_final_nominal_m:.1f} / {row.altitude_final_calibrated_m:.1f} |"
        )
    return "\n".join(lines)


def _write_parameters(
    path: Path,
    calibration: CalibrationResult,
    training_flights: list[FlightKinematics],
    validation_flights: list[FlightKinematics],
) -> None:
    leave_one_out = calibration.leave_one_flight_out
    best_objective = calibration.fit.huber_objective
    delay_support = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best_objective * 1.01
    ]
    parameter_names = [
        "pitch_scale_deg_per_code",
        "angle_of_attack_scale_deg_per_code",
        "effective_gamma_bias_deg",
    ]
    parameters = {}
    for index, name in enumerate(parameter_names):
        parameters[name] = {
            "value": float(calibration.fit.coefficients[index]),
            "leave_one_flight_out_min": float(leave_one_out[:, index].min()),
            "leave_one_flight_out_max": float(leave_one_out[:, index].max()),
            "leave_one_flight_out_std": float(leave_one_out[:, index].std()),
            "status": "effective-provisional",
        }
    document = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "model": {
            "equation": "gamma_deg[t] = pitch_scale * pitch_code[t-delay] - aoa_scale * aoa_code[t-delay] + effective_gamma_bias",
            "parameters": parameters,
            "trajectory_reference_delay_s": {
                "value": calibration.delay_s,
                "one_percent_objective_support": delay_support,
                "status": "effective-alignment-not-physical-sensor-latency",
            },
        },
        "target": {
            "definition": "asin(d(altitude)/dt / TAS)",
            "altitude_source": "processed altitude",
            "true_airspeed_source": "processed TAS",
            "smoothing": f"centered {SMOOTHING_WINDOW_S}-second moving average",
            "vertical_wind_assumption_mps": 0,
        },
        "fit": {
            "method": "weighted Huber iteratively reweighted least squares",
            "training_thinning_s": TRAINING_THIN_S,
            "phase_weighting": "equal total weight per flight and climb/level/descent stratum",
            "maximum_absolute_roll_deg": MAX_ROLL_DEG,
            "minimum_true_airspeed_mps": MIN_TAS_MPS,
            "training_rows": calibration.training_rows,
            "huber_objective": calibration.fit.huber_objective,
            "base_weighted_rmse_deg": calibration.fit.base_weighted_rmse,
            "robust_scale_deg": calibration.fit.robust_scale,
            "normal_matrix_condition": calibration.fit.normal_matrix_condition,
            "coefficient_correlation": calibration.fit.coefficient_correlation.tolist(),
            "delay_objective": {
                str(delay): fit.huber_objective
                for delay, fit in calibration.delay_fits.items()
            },
        },
        "split": {
            "fit_flights": [
                f"{flight.registration}/{flight.recording}"
                for flight in training_flights
            ],
            "held_out_validation_flights": [
                f"{flight.registration}/{flight.recording}"
                for flight in validation_flights
            ],
        },
        "identifiability": {
            "separately_identifiable": [
                "effective pitch code coefficient under the stated assumptions",
                "effective angle-of-attack code coefficient under the stated assumptions",
                "effective pitch-minus-AOA bias",
            ],
            "not_separately_identifiable": [
                "pitch bias versus angle-of-attack bias",
                "sensor latency versus processed trajectory latency",
                "angle-of-attack bias versus unmeasured mean vertical wind",
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_report(
    path: Path, calibration: CalibrationResult, metrics: pd.DataFrame
) -> None:
    loo = calibration.leave_one_flight_out
    best = calibration.fit.huber_objective
    support = [
        delay
        for delay, fit in calibration.delay_fits.items()
        if fit.huber_objective <= best * 1.01
    ]
    validation = metrics.loc[metrics.role == "validation"]
    gamma_improvement = 100.0 * (
        1.0
        - validation.gamma_rmse_calibrated_deg.mean()
        / validation.gamma_rmse_nominal_deg.mean()
    )
    velocity_improvement = 100.0 * (
        1.0
        - validation.vertical_velocity_rmse_calibrated_mps.mean()
        / validation.vertical_velocity_rmse_nominal_mps.mean()
    )
    report = f"""# A320 Vertical Kinematic Calibration

Status: calibrated on 10 flights and evaluated once on 6 held-out flights  
Date: 2026-08-04  
Generator: `tools/calibrate_a320_vertical_kinematics.py`

## Outcome

An effective, causal pitch/AOA-to-flight-path decoder was estimated without using
the six validation flights for delay choice, fitting, or model selection:

```text
gamma_deg[t] = {calibration.pitch_scale:.6f} * pitch_code[t-{calibration.delay_s}]
             - {calibration.angle_of_attack_scale:.6f} * aoa_code[t-{calibration.delay_s}]
             + {calibration.effective_gamma_bias_deg:.6f}
```

The nominal hypotheses were 0.5 deg/code for pitch, 0.25 deg/code for AOA,
zero effective bias, and zero delay. On the held-out flights, the calibrated
relation reduces mean flight-path-angle RMSE by {gamma_improvement:.1f}% and
vertical-velocity RMSE by {velocity_improvement:.1f}%.

This is an **effective kinematic calibration**, not a declaration of physical
sensor calibration. Only the difference between pitch and AOA bias is observable
from this relation.

## Method

- Fit roles: one development flight and nine calibration flights.
- Validation: six complete flights, three from each registration.
- Target: `asin(d(altitude)/dt / TAS)`, smoothed with a centered
  {SMOOTHING_WINDOW_S}-second moving average; zero vertical wind is assumed.
- Normal-operation selection: TAS above {MIN_TAS_MPS:.0f} m/s and absolute roll
  below {MAX_ROLL_DEG:.0f} degrees.
- Five-second thinning for serial correlation and equal total weight for every
  flight/phase stratum (climb, level, descent).
- Weighted Huber regression limits outlier influence.
- One common causal delay was searched from {min(DELAY_CANDIDATES_S)} to
  {max(DELAY_CANDIDATES_S)} seconds using fit flights only.

## Parameter stability and observability

| Parameter | Estimate | Leave-one-flight-out min | max | standard deviation |
|---|---:|---:|---:|---:|
| Effective pitch coefficient (deg/code) | {calibration.pitch_scale:.6f} | {loo[:, 0].min():.6f} | {loo[:, 0].max():.6f} | {loo[:, 0].std():.6f} |
| Effective AOA coefficient (deg/code) | {calibration.angle_of_attack_scale:.6f} | {loo[:, 1].min():.6f} | {loo[:, 1].max():.6f} | {loo[:, 1].std():.6f} |
| Effective gamma bias (deg) | {calibration.effective_gamma_bias_deg:.6f} | {loo[:, 2].min():.6f} | {loo[:, 2].max():.6f} | {loo[:, 2].std():.6f} |

Selected trajectory-reference delay: {calibration.delay_s} seconds. Delays within
1% of the best robust objective: {support}. This should be treated as approximately
2--3 seconds of relative alignment, not an identified physical sensor latency.

The robust normal-matrix condition number is
{calibration.fit.normal_matrix_condition:.1f}. The AOA-scale/effective-bias
correlation is {calibration.fit.coefficient_correlation[1, 2]:.3f}; their
uncertainty must remain coupled. Pitch and AOA offsets cannot be separated, and
an unmeasured mean vertical wind is confounded with effective gamma bias.

## Aggregate performance

{_aggregate_table(metrics)}

## Held-out validation flights

{_validation_table(metrics)}

Complete-flight altitude integration is intentionally severe: a residual mean
flight-path bias accumulates for hours. Gamma and vertical-velocity metrics are
the appropriate gate for this decoder; altitude drift remains a diagnostic until
vertical wind and pressure/geometric-altitude behavior are modeled.

## Decision

The nominal scale/bias/delay combination is rejected as an effective vertical
kinematic decoder; this experiment does not independently reject the physical
recorder scale factors. The effective calibration is suitable as a provisional
decoder inside the normal operational envelope and for the next translational-
force experiment, with its leave-one-flight-out ranges carried as uncertainty. It
is not yet suitable as an air-data calibration or certification-envelope model.

The next step is horizontal identifiability: estimate heading/TAS corrections and
wind bias with per-flight nuisance terms on fit flights, then evaluate them once on
the same held-out panel. Aerodynamic fitting remains deferred.

## Artifacts

- Parameters: `docs/modeling/a320-vertical-kinematic-calibration.yml`
- Per-flight metrics: `docs/modeling/a320-vertical-kinematic-calibration-metrics.csv`
- Typed QAR decoder: `src/aerox/flightdata/a320_qar.py`
- Identification primitives: `src/aerox/modeling/identification.py`
- Provenance ledger: `docs/modeling/sources.yml`

No raw coordinates or trajectory samples are written to the repository.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[CalibrationResult, pd.DataFrame]:
    panel = pd.read_csv(args.panel)
    fit_selection = panel.loc[panel.role != "validation"].copy()
    validation_selection = panel.loc[panel.role == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected 10 fit flights and 6 held-out validation flights")

    # The ordering is intentional: parameters and delay are frozen before any
    # validation flight is loaded.
    fit_flights = _load_flights(fit_selection, args.raw_dir, args.processed)
    calibration = calibrate(fit_flights)
    validation_flights = _load_flights(
        validation_selection, args.raw_dir, args.processed
    )

    metric_rows = [
        _flight_metrics(flight, calibration)
        for flight in [*fit_flights, *validation_flights]
    ]
    metrics = _round_metrics(pd.DataFrame(metric_rows))
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
        "fit:",
        calibration.fit.coefficients,
        "delay_s:",
        calibration.delay_s,
    )
    print(metrics.loc[metrics.role == "validation"].to_string(index=False))
    print(f"wrote {args.parameters}")
    print(f"wrote {args.metrics}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
