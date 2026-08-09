#!/usr/bin/env python3
"""Audit A320 pitch-rate, elevator, and stabilizer observability.

Only the ten development/calibration flights are loaded.  The audit checks the
kinematic consistency of body pitch-rate code against short-horizon pitch
change and the common-mode consistency of the two elevator measurements.  It
does not identify a pitching-moment derivative or release a physical surface
scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox.flightdata.a320_qar import DEG_TO_RAD, decode_a320_flight, raw_columns
from aerox.modeling.identification import robust_weighted_linear_fit
from tools.calibrate_a320_effective_thrust import (
    RAW_FILES,
    _read_vertical_calibration,
)


INTEGRATION_HORIZON_S = 5
MAXIMUM_ABSOLUTE_BANK_DEG = 10.0
NOMINAL_RECORDER_SCALE_DEG_PER_CODE = 0.5
MAXIMUM_MEAN_PITCH_DELTA_RMSE_DEG = 0.5
MINIMUM_MEAN_PITCH_DELTA_CORRELATION = 0.85
MINIMUM_NOMINAL_RMSE_IMPROVEMENT = 0.20
MAXIMUM_SCALE_RATIO_COEFFICIENT_OF_VARIATION = 0.10
MINIMUM_ELEVATOR_PAIR_CORRELATION = 0.95
MAXIMUM_CENTERED_ELEVATOR_DISAGREEMENT_DEG = 0.15


@dataclass(frozen=True)
class LongitudinalControlFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    airborne: np.ndarray
    pitch_code: np.ndarray
    roll_rad: np.ndarray
    pitch_rate_code: np.ndarray
    yaw_rate_code: np.ndarray
    elevator_1_code: np.ndarray
    elevator_2_code: np.ndarray
    stabilizer_code: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.registration}/{self.recording}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv")
    )
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-control-flight-metrics.csv"),
    )
    parser.add_argument(
        "--kinematic-metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-rate-kinematic-lofo.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-control-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-control-audit.md"),
    )
    return parser.parse_args()


def _load_fit_flights(args: argparse.Namespace) -> list[LongitudinalControlFlight]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(selection)}")
    roles = selection.set_index(["registration", "recording"])["role"].to_dict()
    flights: list[LongitudinalControlFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            args.raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=recording,
            )
            # Divide out the current provisional decoder factors.  This audit
            # estimates ratios in recorder-code space so the 0.5 hypothesis is
            # not silently treated as ground truth.
            flights.append(
                LongitudinalControlFlight(
                    registration=str(registration),
                    recording=recording,
                    role=str(roles[(str(registration), recording)]),
                    time_s=decoded.time_s,
                    airborne=decoded.airborne,
                    pitch_code=(
                        decoded.channel("pitch", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                    roll_rad=decoded.channel("roll", allow_provisional=True),
                    pitch_rate_code=(
                        decoded.channel("pitch_rate", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                    yaw_rate_code=(
                        decoded.channel("yaw_rate", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                    elevator_1_code=(
                        decoded.channel("elevator_1", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                    elevator_2_code=(
                        decoded.channel("elevator_2", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                    stabilizer_code=(
                        decoded.channel("stabilizer", allow_provisional=True)
                        / DEG_TO_RAD
                        / NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                    ),
                )
            )
    expected = set(selection["recording"])
    loaded = {flight.recording for flight in flights}
    if loaded != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - loaded)}")
    return flights


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    if np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _flight_metrics(flights: list[LongitudinalControlFlight]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        mask = (
            flight.airborne
            & (np.abs(flight.roll_rad / DEG_TO_RAD) <= MAXIMUM_ABSOLUTE_BANK_DEG)
            & np.isfinite(
                flight.elevator_1_code
                + flight.elevator_2_code
                + flight.stabilizer_code
                + flight.pitch_rate_code
            )
        )
        elevator_1 = flight.elevator_1_code[mask]
        elevator_2 = flight.elevator_2_code[mask]
        difference = elevator_1 - elevator_2
        centered_difference = difference - np.median(difference)
        mean_elevator = 0.5 * (elevator_1 + elevator_2)
        stabilizer = flight.stabilizer_code[mask]
        duration_h = (flight.time_s[-1] - flight.time_s[0]) / 3600.0
        rows.append(
            {
                "registration": flight.registration,
                "recording": flight.recording,
                "role": flight.role,
                "clean_low_bank_samples": int(np.count_nonzero(mask)),
                "elevator_pair_correlation": _safe_correlation(
                    elevator_1, elevator_2
                ),
                "elevator_1_minus_2_median_code": float(np.median(difference)),
                "centered_elevator_disagreement_std_code": float(
                    np.std(centered_difference)
                ),
                "centered_elevator_disagreement_std_provisional_deg": float(
                    np.std(centered_difference)
                    * NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                ),
                "mean_elevator_p01_code": float(np.quantile(mean_elevator, 0.01)),
                "mean_elevator_p50_code": float(np.quantile(mean_elevator, 0.50)),
                "mean_elevator_p99_code": float(np.quantile(mean_elevator, 0.99)),
                "stabilizer_p01_code": float(np.quantile(stabilizer, 0.01)),
                "stabilizer_p50_code": float(np.quantile(stabilizer, 0.50)),
                "stabilizer_p99_code": float(np.quantile(stabilizer, 0.99)),
                "stabilizer_unique_values": int(len(np.unique(stabilizer))),
                "stabilizer_changes_per_hour": float(
                    np.count_nonzero(np.diff(stabilizer)) / duration_h
                ),
            }
        )
    return pd.DataFrame(rows)


def _valid_horizon_starts(mask: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0 or horizon >= len(mask):
        raise ValueError("invalid horizon")
    bad = np.concatenate(([0], np.cumsum(~mask, dtype=int)))
    # Intervals use samples start through start+horizon, inclusive.
    count = bad[horizon + 1 :] - bad[: -(horizon + 1)]
    return count == 0


def _pitch_intervals(
    flight: LongitudinalControlFlight,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = INTEGRATION_HORIZON_S
    finite = np.isfinite(
        flight.pitch_code
        + flight.pitch_rate_code
        + flight.yaw_rate_code
        + flight.roll_rad
    )
    low_bank = (
        flight.airborne
        & finite
        & (np.abs(flight.roll_rad / DEG_TO_RAD) <= MAXIMUM_ABSOLUTE_BANK_DEG)
    )
    valid = _valid_horizon_starts(low_bank, horizon)
    duration = flight.time_s[horizon:] - flight.time_s[:-horizon]
    valid &= np.isclose(duration, horizon)

    pitch_term = flight.pitch_rate_code * np.cos(flight.roll_rad)
    yaw_term = -flight.yaw_rate_code * np.sin(flight.roll_rad)
    clean_pitch_term = np.where(finite, pitch_term, 0.0)
    clean_yaw_term = np.where(finite, yaw_term, 0.0)

    def integrate(values: np.ndarray) -> np.ndarray:
        steps = 0.5 * (values[1:] + values[:-1]) * np.diff(flight.time_s)
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        return cumulative[horizon:] - cumulative[:-horizon]

    integrated_pitch_rate = integrate(clean_pitch_term)
    integrated_yaw_rate = integrate(clean_yaw_term)
    pitch_change = flight.pitch_code[horizon:] - flight.pitch_code[:-horizon]
    valid &= np.isfinite(integrated_pitch_rate + integrated_yaw_rate + pitch_change)
    return (
        pitch_change[valid],
        integrated_pitch_rate[valid],
        integrated_yaw_rate[valid],
    )


def _design(
    flight: LongitudinalControlFlight,
    integrated_pitch_rate: np.ndarray,
    integrated_yaw_rate: np.ndarray,
) -> np.ndarray:
    is_afz = float(flight.registration == "TAIL-B")
    return np.column_stack(
        (
            np.ones(len(integrated_pitch_rate)),
            np.full(len(integrated_pitch_rate), is_afz),
            integrated_pitch_rate,
            integrated_yaw_rate,
        )
    )


def _kinematic_lofo(
    flights: list[LongitudinalControlFlight],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    intervals = {flight.key: _pitch_intervals(flight) for flight in flights}
    rows: list[dict[str, Any]] = []
    for held_out in flights:
        designs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for flight in flights:
            if flight.key == held_out.key:
                continue
            target, integrated_pitch, integrated_yaw = intervals[flight.key]
            designs.append(_design(flight, integrated_pitch, integrated_yaw))
            targets.append(target)
            weights.append(np.full(len(target), 1.0 / (len(target) * 9)))
        fit = robust_weighted_linear_fit(
            np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
        )
        target, integrated_pitch, integrated_yaw = intervals[held_out.key]
        design = _design(held_out, integrated_pitch, integrated_yaw)
        prediction = design @ fit.coefficients
        nominal = integrated_pitch + integrated_yaw
        residual = prediction - target
        nominal_residual = nominal - target
        rows.append(
            {
                "registration": held_out.registration,
                "recording": held_out.recording,
                "role": held_out.role,
                "intervals": len(target),
                "training_pitch_rate_to_pitch_scale_ratio": float(
                    fit.coefficients[2]
                ),
                "training_yaw_rate_to_pitch_scale_ratio": float(
                    fit.coefficients[3]
                ),
                "pitch_delta_rmse_code": float(np.sqrt(np.mean(residual**2))),
                "pitch_delta_rmse_conditional_deg": float(
                    np.sqrt(np.mean(residual**2))
                    * NOMINAL_RECORDER_SCALE_DEG_PER_CODE
                ),
                "nominal_equal_scale_pitch_delta_rmse_code": float(
                    np.sqrt(np.mean(nominal_residual**2))
                ),
                "pitch_delta_bias_code": float(np.mean(residual)),
                "pitch_delta_correlation": _safe_correlation(prediction, target),
            }
        )

    designs = []
    targets = []
    weights = []
    for flight in flights:
        target, integrated_pitch, integrated_yaw = intervals[flight.key]
        designs.append(_design(flight, integrated_pitch, integrated_yaw))
        targets.append(target)
        weights.append(np.full(len(target), 1.0 / (len(target) * len(flights))))
    full_fit = robust_weighted_linear_fit(
        np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
    )
    parameters = {
        "feature_order": [
            "constant_pitch_change_code",
            "TAIL-B_pitch_change_offset_code",
            "integrated_pitch_rate_code_cos_roll",
            "integrated_negative_yaw_rate_code_sin_roll",
        ],
        "coefficients": {
            "constant_pitch_change_code": float(full_fit.coefficients[0]),
            "TAIL-B_pitch_change_offset_code": float(full_fit.coefficients[1]),
            "pitch_rate_to_pitch_scale_ratio": float(full_fit.coefficients[2]),
            "yaw_rate_to_pitch_scale_ratio": float(full_fit.coefficients[3]),
        },
        "conditional_effective_pitch_rate_scale_deg_s_per_code": float(
            full_fit.coefficients[2] * NOMINAL_RECORDER_SCALE_DEG_PER_CODE
        ),
        "conditional_effective_yaw_rate_scale_deg_s_per_code": float(
            full_fit.coefficients[3] * NOMINAL_RECORDER_SCALE_DEG_PER_CODE
        ),
        "normal_matrix_condition": float(full_fit.normal_matrix_condition),
    }
    return pd.DataFrame(rows), parameters


def _decision(
    flight_metrics: pd.DataFrame,
    kinematic_metrics: pd.DataFrame,
    model: dict[str, Any],
) -> dict[str, Any]:
    pitch_ratio = float(model["coefficients"]["pitch_rate_to_pitch_scale_ratio"])
    yaw_ratio = float(model["coefficients"]["yaw_rate_to_pitch_scale_ratio"])
    mean_rmse = float(kinematic_metrics["pitch_delta_rmse_conditional_deg"].mean())
    mean_correlation = float(kinematic_metrics["pitch_delta_correlation"].mean())
    calibrated_rmse_code = float(kinematic_metrics["pitch_delta_rmse_code"].mean())
    nominal_rmse_code = float(
        kinematic_metrics["nominal_equal_scale_pitch_delta_rmse_code"].mean()
    )
    nominal_improvement = (nominal_rmse_code - calibrated_rmse_code) / nominal_rmse_code
    pitch_ratio_cv = float(
        kinematic_metrics["training_pitch_rate_to_pitch_scale_ratio"].std()
        / kinematic_metrics["training_pitch_rate_to_pitch_scale_ratio"].mean()
    )
    yaw_ratio_cv = float(
        kinematic_metrics["training_yaw_rate_to_pitch_scale_ratio"].std()
        / kinematic_metrics["training_yaw_rate_to_pitch_scale_ratio"].mean()
    )
    minimum_pair_correlation = float(
        flight_metrics["elevator_pair_correlation"].min()
    )
    maximum_pair_disagreement = float(
        flight_metrics[
            "centered_elevator_disagreement_std_provisional_deg"
        ].max()
    )
    kinematic_passed = bool(
        pitch_ratio > 0.0
        and yaw_ratio > 0.0
        and mean_rmse <= MAXIMUM_MEAN_PITCH_DELTA_RMSE_DEG
        and mean_correlation >= MINIMUM_MEAN_PITCH_DELTA_CORRELATION
        and nominal_improvement >= MINIMUM_NOMINAL_RMSE_IMPROVEMENT
        and pitch_ratio_cv <= MAXIMUM_SCALE_RATIO_COEFFICIENT_OF_VARIATION
        and yaw_ratio_cv <= MAXIMUM_SCALE_RATIO_COEFFICIENT_OF_VARIATION
    )
    elevator_passed = bool(
        minimum_pair_correlation >= MINIMUM_ELEVATOR_PAIR_CORRELATION
        and maximum_pair_disagreement
        <= MAXIMUM_CENTERED_ELEVATOR_DISAGREEMENT_DEG
    )
    return {
        "pitch_rate_kinematic_fit_gate_passed": kinematic_passed,
        "elevator_common_mode_gate_passed": elevator_passed,
        "ready_for_fit_only_pitch_moment_screen": bool(
            kinematic_passed and elevator_passed
        ),
        "release_physical_pitch_rate_scale": False,
        "release_physical_elevator_scale_or_neutral": False,
        "select_independent_confirmatory_panel_before_release": bool(
            kinematic_passed and elevator_passed
        ),
        "observed": {
            "full_fit_pitch_rate_to_pitch_scale_ratio": pitch_ratio,
            "full_fit_yaw_rate_to_pitch_scale_ratio": yaw_ratio,
            "mean_lofo_pitch_delta_rmse_conditional_deg": mean_rmse,
            "mean_lofo_pitch_delta_correlation": mean_correlation,
            "mean_lofo_rmse_improvement_over_equal_scale": nominal_improvement,
            "pitch_rate_scale_ratio_lofo_coefficient_of_variation": pitch_ratio_cv,
            "yaw_rate_scale_ratio_lofo_coefficient_of_variation": yaw_ratio_cv,
            "minimum_elevator_pair_correlation": minimum_pair_correlation,
            "maximum_centered_elevator_disagreement_provisional_deg": (
                maximum_pair_disagreement
            ),
        },
        "gate": {
            "rate_scale_ratios_must_be_positive": True,
            "maximum_mean_pitch_delta_rmse_conditional_deg": (
                MAXIMUM_MEAN_PITCH_DELTA_RMSE_DEG
            ),
            "minimum_mean_pitch_delta_correlation": (
                MINIMUM_MEAN_PITCH_DELTA_CORRELATION
            ),
            "minimum_rmse_improvement_over_equal_scale": (
                MINIMUM_NOMINAL_RMSE_IMPROVEMENT
            ),
            "maximum_scale_ratio_lofo_coefficient_of_variation": (
                MAXIMUM_SCALE_RATIO_COEFFICIENT_OF_VARIATION
            ),
            "minimum_elevator_pair_correlation": (
                MINIMUM_ELEVATOR_PAIR_CORRELATION
            ),
            "maximum_centered_elevator_disagreement_provisional_deg": (
                MAXIMUM_CENTERED_ELEVATOR_DISAGREEMENT_DEG
            ),
        },
    }


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["number"]):
        if column in {"clean_low_bank_samples", "stabilizer_unique_values", "intervals"}:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(8)
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    strings = frame.astype(str)
    headers = [str(column) for column in strings.columns]
    rows = strings.values.tolist()
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
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


def _write_report(
    path: Path,
    *,
    flight_metrics: pd.DataFrame,
    kinematic_metrics: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    compact_flights = _round_frame(
        flight_metrics[
            [
                "registration",
                "recording",
                "elevator_pair_correlation",
                "elevator_1_minus_2_median_code",
                "centered_elevator_disagreement_std_provisional_deg",
                "mean_elevator_p01_code",
                "mean_elevator_p99_code",
                "stabilizer_unique_values",
            ]
        ]
    )
    compact_kinematic = _round_frame(
        kinematic_metrics[
            [
                "registration",
                "recording",
                "intervals",
                "training_pitch_rate_to_pitch_scale_ratio",
                "training_yaw_rate_to_pitch_scale_ratio",
                "pitch_delta_rmse_conditional_deg",
                "nominal_equal_scale_pitch_delta_rmse_code",
                "pitch_delta_correlation",
            ]
        ]
    )
    decision = audit["decision"]
    observed = decision["observed"]
    text = f"""# A320 longitudinal-control observability audit

## Decision

The fit-only pitch-rate kinematic and elevator common-mode gates both pass.
Over {INTEGRATION_HORIZON_S}-second low-bank intervals, the fitted recorder-code
scale ratios relative to pitch code are
{observed['full_fit_pitch_rate_to_pitch_scale_ratio']:.3f} for body pitch rate
and {observed['full_fit_yaw_rate_to_pitch_scale_ratio']:.3f} for body yaw rate.
The current equal 0.5°/code rate hypothesis is therefore not verified.
Calibrating the two rate scales reduces leave-one-flight-out error by
{observed['mean_lofo_rmse_improvement_over_equal_scale']:.1%}; pitch-change
RMSE averages
{observed['mean_lofo_pitch_delta_rmse_conditional_deg']:.3f}° when expressed
using the existing provisional 0.5° pitch scale.

The two elevator channels have at least
{observed['minimum_elevator_pair_correlation']:.3f} correlation on every fit
flight. After removing each flight's median left/right offset, their maximum
disagreement standard deviation is
{observed['maximum_centered_elevator_disagreement_provisional_deg']:.3f}° under
the provisional 0.5°/code conversion.

This is enough to proceed to a **fit-only effective pitching-moment screen**.
It is not enough to release physical pitch-rate/elevator scales, surface
neutral positions, inertia, or stability derivatives. The two registrations
have opposite left/right elevator zero offsets, so those offsets must remain
measurement nuisances rather than aerodynamic asymmetry.

Only the ten development/calibration flights were loaded. No validation or new
confirmatory flight was opened.

## Elevator and stabilizer measurements

{_markdown_table(compact_flights)}

High pair correlation with registration-dependent offset supports a shared
elevator-motion signal after explicit per-channel bias handling. Stabilizer
position has multiple values and changes on every fit flight, but its physical
scale, sign, and neutral remain provisional.

## Pitch-rate kinematic consistency

{_markdown_table(compact_kinematic)}

For each held-out flight, the two rate-scale ratios and two registration measurement
offsets are fitted robustly on the other nine flights with equal total weight
per flight. The kinematic diagnostic is

```text
delta pitch_code = tail offset
                   + q_scale_ratio * integral(q_code cos(roll)) dt
                   - r_scale_ratio * integral(r_code sin(roll)) dt.
```

The Euler relation includes the body-yaw-rate coupling explicitly; omitting it
causes a material scale error even on this low-bank screen. The fit confirms a
usable effective rate channel at 1 Hz; it is not a
high-bandwidth actuator or short-period-mode identification.

## Next gate

Fit an effective low-order pitch-rate model using AOA, pitch rate, dynamic
pressure/Mach, mean elevator motion, and stabilizer position. Compare a
state-only baseline against added control inputs by leave-one-complete-flight-
out prediction. Keep elevator biases as registration/channel measurement
nuisances and keep measured N1/AOA in the translational plant. Freeze that
structure before selecting a new rotational-dynamics confirmatory panel.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    flights = _load_fit_flights(args)
    flight_metrics = _flight_metrics(flights)
    kinematic_metrics, model = _kinematic_lofo(flights)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    model["conditional_on_effective_pitch_scale_deg_per_code"] = (
        vertical.pitch_scale_deg_per_code
    )
    model["conditional_effective_pitch_rate_scale_from_vertical_deg_s_per_code"] = (
        model["coefficients"]["pitch_rate_to_pitch_scale_ratio"]
        * vertical.pitch_scale_deg_per_code
    )
    model["conditional_effective_yaw_rate_scale_from_vertical_deg_s_per_code"] = (
        model["coefficients"]["yaw_rate_to_pitch_scale_ratio"]
        * vertical.pitch_scale_deg_per_code
    )
    decision = _decision(flight_metrics, kinematic_metrics, model)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": (
            "fit-only-longitudinal-controls-observable"
            if decision["ready_for_fit_only_pitch_moment_screen"]
            else "longitudinal-control-observability-gate-failed"
        ),
        "leakage_control": {
            "fit_flights_only": [flight.key for flight in flights],
            "validation_flights_loaded": False,
            "new_confirmatory_flights_loaded": False,
        },
        "method": {
            "sample_rate_hz": 1,
            "integration_horizon_s": INTEGRATION_HORIZON_S,
            "maximum_absolute_bank_deg": MAXIMUM_ABSOLUTE_BANK_DEG,
            "cross_validation": "leave-one-complete-fit-flight-out",
            "fit_weighting": "equal total weight per fit flight with Huber robustness",
            "pitch_rate_model": (
                "delta pitch code = registration offset + q scale ratio * "
                "integral(q code * cos(roll)) - r scale ratio * "
                "integral(r code * sin(roll))"
            ),
            "elevator_pair_test": (
                "correlation and disagreement after per-flight median offset"
            ),
        },
        "model": model,
        "decision": decision,
        "artifacts": {
            "runner": "tools/audit_a320_longitudinal_controls.py",
            "flight_metrics": str(args.flight_metrics),
            "kinematic_metrics": str(args.kinematic_metrics),
            "report": str(args.report),
        },
        "dependencies": {
            "channel_map": "docs/modeling/a320-qar-channel-map.yml",
            "vertical_calibration": str(args.vertical_parameters),
        },
        "limitations": [
            "All physical recorder scales remain provisional.",
            "Elevator neutral and channel biases are not aerodynamic asymmetry.",
            "The 1 Hz data cannot identify high-bandwidth actuator dynamics.",
            "No pitching-moment derivative or inertia is fitted by this audit.",
        ],
    }
    args.flight_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(flight_metrics).to_csv(args.flight_metrics, index=False)
    _round_frame(kinematic_metrics).to_csv(args.kinematic_metrics, index=False)
    args.audit.write_text(yaml.safe_dump(_native(audit), sort_keys=False))
    _write_report(
        args.report,
        flight_metrics=flight_metrics,
        kinematic_metrics=kinematic_metrics,
        audit=audit,
    )
    return flight_metrics, kinematic_metrics, audit


def main() -> None:
    flight_metrics, kinematic_metrics, audit = run(parse_args())
    print(flight_metrics.to_string(index=False))
    print(kinematic_metrics.to_string(index=False))
    print(yaml.safe_dump(audit["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
