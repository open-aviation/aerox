#!/usr/bin/env python3
"""Calibrate a reduced A320 variable-speed autothrust response.

The controller combines physics-based trim thrust with a small identified
speed-feedback law. Candidate gains are selected on complete-flight 300-second
rollouts from ten fit flights; six validation flights are loaded only after the
structure and gains are frozen. Fixed-climb contacts and low-power/open-descent
samples are excluded from controller calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import aero, load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.flightdata.a320_qar import (
    FT_TO_M,
    KNOT_TO_MPS,
    TIME_COLUMN,
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.models.thrust import SpeedTrackingAutothrustController
from aerox.models.thrust import EffectiveN1Envelope, EffectiveThrustMap
from aerox.modeling.force_reconstruction import centered_linear_derivative
from aerox.models.atmosphere import G0
from tools.audit_a320_autothrust_envelope import DOMINANT_CLB_CODE
from tools.calibrate_a320_effective_thrust import (
    MAX_ABSOLUTE_BANK_DEG,
    MAX_MACH,
    MAX_STATIC_TEMPERATURE_K,
    MIN_STATIC_TEMPERATURE_K,
    MIN_TRUE_AIRSPEED_MPS,
    PROCESSED_COLUMNS,
    RAW_FILES,
    VerticalCalibration,
    _load_flights,
    _read_tas_scale,
    _read_vertical_calibration,
)


ROLL_OUT_WINDOW_S = 300
ROLL_OUT_STRIDE_S = 300
TARGET_RATE_WINDOW_S = 21
CLB_EXCLUSION_MARGIN_N1_PERCENT = 2.0
MINIMUM_VARIABLE_MODE_N1_PERCENT = 50.0
MAXIMUM_TARGET_RATE_MPS2 = 2.0
ACCELERATION_LIMIT_MPS2 = 2.0
RESPONSE_TIME_CONSTANTS_S = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
TARGET_RATE_FEEDFORWARD_GAINS = (0.0, 0.5, 1.0)
NEAR_BEST_FRACTION = 0.01
MAXIMUM_VALIDATION_ROLLOUT_RMSE_MPS = 8.0
MINIMUM_VALIDATION_PERSISTENCE_IMPROVEMENT = 0.10
MINIMUM_EVALUABLE_VALIDATION_FLIGHTS = 5


@dataclass(frozen=True)
class SpeedControllerFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    true_airspeed_mps: np.ndarray
    target_true_airspeed_mps: np.ndarray
    target_true_airspeed_rate_mps2: np.ndarray
    pressure_altitude_m: np.ndarray
    flight_path_angle_rad: np.ndarray
    mass_kg: np.ndarray
    mach: np.ndarray
    pressure_ratio: np.ndarray
    static_temperature_k: np.ndarray
    n1_percent: np.ndarray
    clb_n1_ceiling_percent: np.ndarray
    prior_drag_n: np.ndarray
    variable_mode_mask: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.registration}/{self.recording}"


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
        "--envelope-parameters",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-speed-autothrust-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/a320-speed-autothrust-flight-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-speed-autothrust.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-speed-autothrust.md"),
    )
    return parser.parse_args()


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    time = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time)
    order = np.argsort(time[keep], kind="stable")
    indices = np.flatnonzero(keep)[order]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[indices]


def _read_envelope(path: Path) -> EffectiveN1Envelope:
    data = yaml.safe_load(path.read_text())
    if not data["clb_limit_decision"]["passed"]:
        raise ValueError("CLB N1 ceiling must pass before controller calibration")
    model = data["model"]
    order = model["feature_order"]
    return EffectiveN1Envelope(
        feature_set=model["feature_set"],
        idle_coefficients=np.array([model["idle_coefficients"][name] for name in order]),
        climb_coefficients=np.array([model["climb_coefficients"][name] for name in order]),
    )


def _read_thrust_map(path: Path) -> EffectiveThrustMap:
    data = yaml.safe_load(path.read_text())
    model = data["model"]
    order = model["feature_order"]
    scenario = model["scenarios"]["nominal"]
    return EffectiveThrustMap(
        rated_total_thrust_n=float(model["rated_total_thrust_n"]),
        pressure_exponent=float(model["pressure_exponent"]),
        feature_set=str(model["feature_set"]),
        corrected_fan_speed=bool(model["corrected_fan_speed"]),
        coefficients=np.array([scenario["coefficients"][name] for name in order]),
    )


def _aligned_guidance(
    *, raw: pd.DataFrame, processed: pd.DataFrame, registration: str, recording: str
) -> tuple[np.ndarray, np.ndarray]:
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"), registration=registration, recording=recording
    )
    current = processed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = current["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
    if np.any(raw_indices >= len(decoded.time_s)) or not np.all(
        decoded.timestamp_utc[raw_indices] == timestamps
    ):
        raise ValueError(f"raw/processed mismatch for {registration}/{recording}")
    selected = bounded_sample_and_hold(
        _ordered_raw(raw, "SPEEDP_SEL"), decoded.time_s, 1.0
    )[raw_indices]
    tla = 0.5 * (
        decoded.channel("throttle_1", allow_unresolved=True)[raw_indices]
        + decoded.channel("throttle_2", allow_unresolved=True)[raw_indices]
    )
    return selected, tla


def _attach_guidance(
    base_flights: list[Any],
    selection: pd.DataFrame,
    *,
    raw_dir: Path,
    processed_path: Path,
    envelope: EffectiveN1Envelope,
    cd0: float,
    induced_drag_factor: float,
) -> list[SpeedControllerFlight]:
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
    base_by_key = {flight.key: flight for flight in base_flights}
    output: list[SpeedControllerFlight] = []
    required = list(dict.fromkeys((*raw_columns(), "SPEEDP_SEL")))
    for registration, current in selection.groupby("registration", sort=True):
        recordings = current["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=required,
            filters=[("recording", "in", recordings)],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            base = base_by_key[f"{registration}/{recording}"]
            selected_ias, tla = _aligned_guidance(
                raw=frame,
                processed=processed_groups[recording],
                registration=str(registration),
                recording=recording,
            )
            selected_ias = selected_ias[base.sample_index]
            tla = tla[base.sample_index]
            isa_temperature = np.maximum(
                aero.T0 + aero.beta * base.pressure_altitude_m, 216.65
            )
            temperature_deviation = base.static_temperature_k - isa_temperature
            target_tas = np.asarray(
                aero.cas2tas(
                    selected_ias * KNOT_TO_MPS,
                    base.pressure_altitude_m,
                    dT=temperature_deviation,
                ),
                dtype=float,
            )
            target_rate = centered_linear_derivative(
                target_tas, base.time_s, TARGET_RATE_WINDOW_S
            )
            _, clb = envelope.bounds_percent(
                pressure_altitude_m=base.pressure_altitude_m,
                mach=base.mach,
                static_temperature_k=base.static_temperature_k,
            )
            finite = np.isfinite(target_tas + target_rate + tla + clb)
            variable = (
                finite
                & np.isclose(tla, DOMINANT_CLB_CODE)
                & (base.n1_percent <= clb - CLB_EXCLUSION_MARGIN_N1_PERCENT)
                & (base.n1_percent >= MINIMUM_VARIABLE_MODE_N1_PERCENT)
                & (np.abs(target_rate) <= MAXIMUM_TARGET_RATE_MPS2)
            )
            drag = (
                cd0 * base.parasite_drag_basis_n
                + induced_drag_factor * base.induced_drag_basis_n
            )
            output.append(
                SpeedControllerFlight(
                    registration=str(registration),
                    recording=recording,
                    role=base.role,
                    time_s=base.time_s,
                    true_airspeed_mps=base.true_airspeed_mps,
                    target_true_airspeed_mps=target_tas,
                    target_true_airspeed_rate_mps2=target_rate,
                    pressure_altitude_m=base.pressure_altitude_m,
                    flight_path_angle_rad=base.flight_path_angle_rad,
                    mass_kg=base.mass_kg,
                    mach=base.mach,
                    pressure_ratio=base.pressure_ratio,
                    static_temperature_k=base.static_temperature_k,
                    n1_percent=base.n1_percent,
                    clb_n1_ceiling_percent=clb,
                    prior_drag_n=drag,
                    variable_mode_mask=variable,
                )
            )
    return output


def _contiguous_windows(flight: SpeedControllerFlight) -> list[np.ndarray]:
    selected = np.flatnonzero(flight.variable_mode_mask)
    if len(selected) == 0:
        return []
    breaks = np.flatnonzero(
        (np.diff(selected) != 1)
        | (np.diff(flight.time_s[selected]) > 1.1)
    ) + 1
    segments = np.split(selected, breaks)
    windows: list[np.ndarray] = []
    samples = ROLL_OUT_WINDOW_S + 1
    for segment in segments:
        if len(segment) < samples:
            continue
        starts = list(range(0, len(segment) - samples + 1, ROLL_OUT_STRIDE_S))
        final = len(segment) - samples
        if starts[-1] != final and final - starts[-1] >= ROLL_OUT_STRIDE_S // 2:
            starts.append(final)
        windows.extend(segment[start : start + samples] for start in starts)
    return windows


def _rollout_rmse(
    flight: SpeedControllerFlight,
    windows: list[np.ndarray],
    controller: SpeedTrackingAutothrustController | None,
) -> tuple[float, float, int]:
    squared: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for indices in windows:
        if controller is None:
            prediction = np.full(len(indices), flight.true_airspeed_mps[indices[0]])
        else:
            prediction = controller.simulate_speed_response_zoh(
                time_s=flight.time_s[indices],
                target_true_airspeed_mps=flight.target_true_airspeed_mps[indices],
                target_true_airspeed_rate_mps2=(
                    flight.target_true_airspeed_rate_mps2[indices]
                ),
                initial_true_airspeed_mps=float(flight.true_airspeed_mps[indices[0]]),
            )
        residual = prediction - flight.true_airspeed_mps[indices]
        squared.append(residual**2)
        biases.append(residual)
    if not squared:
        return float("nan"), float("nan"), 0
    return (
        float(np.sqrt(np.mean(np.concatenate(squared)))),
        float(np.mean(np.concatenate(biases))),
        len(windows),
    )


def _candidate_metrics(flights: list[SpeedControllerFlight]) -> pd.DataFrame:
    window_map = {flight.key: _contiguous_windows(flight) for flight in flights}
    if any(len(windows) == 0 for windows in window_map.values()):
        missing = [key for key, windows in window_map.items() if not windows]
        raise ValueError(f"fit flights lack 300-second variable-mode windows: {missing}")
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, float | None, float | None]] = [
        ("persistence", None, None)
    ]
    candidates.extend(
        ("speed_feedback", tau, feedforward)
        for tau in RESPONSE_TIME_CONSTANTS_S
        for feedforward in TARGET_RATE_FEEDFORWARD_GAINS
    )
    for name, tau, feedforward in candidates:
        flight_rmse: list[float] = []
        flight_bias: list[float] = []
        windows_total = 0
        controller = (
            None
            if tau is None
            else SpeedTrackingAutothrustController(
                speed_error_gain_per_s=1.0 / tau,
                target_rate_feedforward_gain=float(feedforward),
                minimum_acceleration_mps2=-ACCELERATION_LIMIT_MPS2,
                maximum_acceleration_mps2=ACCELERATION_LIMIT_MPS2,
            )
        )
        for flight in flights:
            rmse, bias, windows = _rollout_rmse(
                flight, window_map[flight.key], controller
            )
            flight_rmse.append(rmse)
            flight_bias.append(bias)
            windows_total += windows
        rows.append(
            {
                "candidate": name,
                "response_time_constant_s": tau,
                "target_rate_feedforward_gain": feedforward,
                "nonzero_feedback_terms": (
                    0 if tau is None else 1 + int(float(feedforward) > 0.0)
                ),
                "complete_flights": len(flights),
                "windows": windows_total,
                "mean_flight_rollout_rmse_mps": float(np.mean(flight_rmse)),
                "median_flight_rollout_rmse_mps": float(np.median(flight_rmse)),
                "worst_flight_rollout_rmse_mps": float(np.max(flight_rmse)),
                "mean_absolute_flight_bias_mps": float(
                    np.mean(np.abs(flight_bias))
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates.loc[candidates["candidate"] == "speed_feedback"]
    limit = (1.0 + NEAR_BEST_FRACTION) * eligible[
        "mean_flight_rollout_rmse_mps"
    ].min()
    return eligible.loc[eligible["mean_flight_rollout_rmse_mps"] <= limit].sort_values(
        [
            "nonzero_feedback_terms",
            "mean_flight_rollout_rmse_mps",
            "response_time_constant_s",
        ],
        kind="stable",
    ).iloc[0]


def _controller_from_row(row: pd.Series) -> SpeedTrackingAutothrustController:
    return SpeedTrackingAutothrustController(
        speed_error_gain_per_s=1.0 / float(row["response_time_constant_s"]),
        target_rate_feedforward_gain=float(row["target_rate_feedforward_gain"]),
        minimum_acceleration_mps2=-ACCELERATION_LIMIT_MPS2,
        maximum_acceleration_mps2=ACCELERATION_LIMIT_MPS2,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _flight_metrics(
    flights: list[SpeedControllerFlight],
    controller: SpeedTrackingAutothrustController,
    thrust_map: EffectiveThrustMap,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        windows = _contiguous_windows(flight)
        rollout_rmse, rollout_bias, window_count = _rollout_rmse(
            flight, windows, controller
        )
        persistence_rmse, _, _ = _rollout_rmse(flight, windows, None)
        mask = flight.variable_mode_mask
        trim = (
            flight.prior_drag_n[mask]
            + flight.mass_kg[mask]
            * G0
            * np.sin(flight.flight_path_angle_rad[mask])
        )
        clb_thrust = thrust_map.thrust_n(
            n1_percent=flight.clb_n1_ceiling_percent[mask],
            mach=flight.mach[mask],
            pressure_ratio=flight.pressure_ratio[mask],
            static_temperature_k=flight.static_temperature_k[mask],
        )
        command_thrust = controller.commanded_effective_thrust_n(
            trim_thrust_n=trim,
            mass_kg=flight.mass_kg[mask],
            true_airspeed_mps=flight.true_airspeed_mps[mask],
            target_true_airspeed_mps=flight.target_true_airspeed_mps[mask],
            target_true_airspeed_rate_mps2=(
                flight.target_true_airspeed_rate_mps2[mask]
            ),
            minimum_thrust_n=np.zeros(np.count_nonzero(mask)),
            maximum_thrust_n=clb_thrust,
        )
        observed_thrust = thrust_map.thrust_n(
            n1_percent=flight.n1_percent[mask],
            mach=flight.mach[mask],
            pressure_ratio=flight.pressure_ratio[mask],
            static_temperature_k=flight.static_temperature_k[mask],
        )
        predicted_n1 = np.array(
            [
                thrust_map.n1_for_thrust_scalar(
                    thrust_n=float(command_thrust[index]),
                    mach=float(flight.mach[mask][index]),
                    pressure_ratio=float(flight.pressure_ratio[mask][index]),
                    static_temperature_k=float(
                        flight.static_temperature_k[mask][index]
                    ),
                    minimum_n1_percent=20.0,
                    maximum_n1_percent=float(
                        flight.clb_n1_ceiling_percent[mask][index]
                    ),
                )
                for index in range(np.count_nonzero(mask))
            ]
        )
        thrust_residual = command_thrust - observed_thrust
        n1_residual = predicted_n1 - flight.n1_percent[mask]
        target_error = (
            flight.target_true_airspeed_mps[mask]
            - flight.true_airspeed_mps[mask]
        )
        rows.append(
            {
                "registration": flight.registration,
                "recording": flight.recording,
                "role": flight.role,
                "split": split,
                "variable_mode_samples": int(np.count_nonzero(mask)),
                "rollout_windows": window_count,
                "rollout_rmse_mps": rollout_rmse,
                "rollout_bias_mps": rollout_bias,
                "persistence_rollout_rmse_mps": persistence_rmse,
                "persistence_relative_improvement": (
                    (persistence_rmse - rollout_rmse) / persistence_rmse
                ),
                "target_speed_error_median_mps": float(np.median(target_error)),
                "target_speed_error_rmse_mps": float(
                    np.sqrt(np.mean(target_error**2))
                ),
                "instantaneous_thrust_rmse_kn": float(
                    np.sqrt(np.mean(thrust_residual**2)) / 1000.0
                ),
                "instantaneous_thrust_bias_kn": float(
                    np.mean(thrust_residual) / 1000.0
                ),
                "instantaneous_thrust_correlation": _safe_correlation(
                    command_thrust, observed_thrust
                ),
                "instantaneous_n1_rmse_percent": float(
                    np.sqrt(np.mean(n1_residual**2))
                ),
                "clb_saturation_fraction": float(
                    np.mean(np.isclose(command_thrust, clb_thrust))
                ),
            }
        )
    return pd.DataFrame(rows)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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
        if column in {
            "nonzero_feedback_terms",
            "complete_flights",
            "windows",
            "variable_mode_samples",
            "rollout_windows",
        }:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(8)
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
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
    candidates: pd.DataFrame,
    chosen: pd.Series,
    metrics: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    top = _round_frame(
        candidates.sort_values("mean_flight_rollout_rmse_mps").head(10)[
            [
                "candidate",
                "response_time_constant_s",
                "target_rate_feedforward_gain",
                "mean_flight_rollout_rmse_mps",
                "worst_flight_rollout_rmse_mps",
            ]
        ]
    )
    compact = _round_frame(
        metrics[
            [
                "registration",
                "recording",
                "split",
                "rollout_windows",
                "rollout_rmse_mps",
                "persistence_rollout_rmse_mps",
                "persistence_relative_improvement",
                "target_speed_error_median_mps",
                "instantaneous_n1_rmse_percent",
                "clb_saturation_fraction",
            ]
        ]
    )
    validation = metrics.loc[metrics["split"] == "validation"]
    evaluable_validation = int(validation["rollout_rmse_mps"].notna().sum())
    text = f"""# A320 variable-speed autothrust surrogate

## Decision

The first dynamic variable-speed surrogate is
`a_cmd = K_v (V_target - V) + K_ff dV_target/dt`, above physics-based trim
thrust and below the validated CLB ceiling. The fit-only selection chooses a
{chosen['response_time_constant_s']:.0f}-second response time
(`K_v={1.0 / chosen['response_time_constant_s']:.6f} 1/s`) and target-rate
feedforward `{chosen['target_rate_feedforward_gain']:.1f}`.

The frozen validation median 300-second rollout RMSE is
{validation['rollout_rmse_mps'].median():.2f} m/s, versus
{validation['persistence_rollout_rmse_mps'].median():.2f} m/s for holding the
window's initial speed. Median relative improvement is
{validation['persistence_relative_improvement'].median():.1%}. The predeclared
gate is {'passed' if decision['passed'] else 'not passed'}.
Only {evaluable_validation}/{len(validation)} validation flights contain a
continuous 300-second window after the frozen mode exclusions; release requires
at least {MINIMUM_EVALUABLE_VALIDATION_FLIGHTS}.

This is an effective closed-loop response, not Airbus source code or a direct
FADEC command. Fixed-climb contacts, N1 below
{MINIMUM_VARIABLE_MODE_N1_PERCENT:.0f}%, and samples within
{CLB_EXCLUSION_MARGIN_N1_PERCENT:.0f} N1 points of the CLB ceiling are excluded.
The slowest tested response being preferred while persistence remains better is
interpreted as failed target/mode observability, not as an identified
180-second Airbus controller.

## Architecture

```text
selected/managed CAS -> target TAS at current atmosphere
speed error + target-rate feedforward -> along-path acceleration command
drag + m g sin(gamma) + m a_cmd -> effective thrust command
CLB ceiling -> thrust saturation
inverse effective-thrust map -> commanded N1
N1/N2 actuator -> actual N1 -> effective thrust
```

The controller's trim term uses the frozen A320 drag polar and measured flight-
path angle during this identification stage. Its speed-response candidates are
judged on multi-step rollouts; instantaneous thrust/N1 reconstruction is a
secondary diagnostic because it omits the unobserved command and spool lag.

## Fit-only candidate screen

{_markdown_table(top)}

## Complete-flight results

{_markdown_table(compact)}

## Limitations and next gate

- `SPEEDP_SEL` is treated as the active/equivalent CAS target. FMA SPEED versus
  MACH mode is not decoded.
- The calibration covers clean, moderate/high-power variable operation. It does
  not cover approach thrust below {MINIMUM_VARIABLE_MODE_N1_PERCENT:.0f}%.
- The CLB ceiling is QAR validated, but idle, FLEX/MCT/TOGA rating, bleed and
  anti-ice corrections, and airborne spool constants remain unavailable.
- The speed-response rollout prescribes the recorded target and atmosphere. A
  coupled longitudinal plant rollout with commanded N1 is the next trajectory
  gate; no validation trajectory may be tuned after this result.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected ten fit and six validation flights")
    vertical: VerticalCalibration = _read_vertical_calibration(
        args.vertical_parameters
    )
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    if aircraft.wing is None or aircraft.drag is None:
        raise ValueError("A320 definition lacks wing or drag data")
    envelope = _read_envelope(args.envelope_parameters)
    thrust_map = _read_thrust_map(args.thrust_parameters)

    fit_base = _load_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=float(aircraft.wing.area),
    )
    fit_flights = _attach_guidance(
        fit_base,
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        envelope=envelope,
        cd0=reference_drag.zero_lift_drag_coefficient,
        induced_drag_factor=reference_drag.induced_drag_factor,
    )
    candidates = _candidate_metrics(fit_flights)
    chosen = _select_candidate(candidates)
    controller = _controller_from_row(chosen)
    fit_metrics = _flight_metrics(
        fit_flights, controller, thrust_map, "fit_selected"
    )

    # Frozen validation use begins only after candidate selection above.
    validation_base = _load_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=float(aircraft.wing.area),
    )
    validation_flights = _attach_guidance(
        validation_base,
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        envelope=envelope,
        cd0=reference_drag.zero_lift_drag_coefficient,
        induced_drag_factor=reference_drag.induced_drag_factor,
    )
    validation_metrics = _flight_metrics(
        validation_flights, controller, thrust_map, "validation"
    )
    metrics = pd.concat((fit_metrics, validation_metrics), ignore_index=True)
    validation_median_rmse = float(
        validation_metrics["rollout_rmse_mps"].median()
    )
    validation_median_improvement = float(
        validation_metrics["persistence_relative_improvement"].median()
    )
    evaluable_validation_flights = int(
        validation_metrics["rollout_rmse_mps"].notna().sum()
    )
    passed = (
        validation_median_rmse <= MAXIMUM_VALIDATION_ROLLOUT_RMSE_MPS
        and validation_median_improvement
        >= MINIMUM_VALIDATION_PERSISTENCE_IMPROVEMENT
        and evaluable_validation_flights >= MINIMUM_EVALUABLE_VALIDATION_FLIGHTS
    )
    decision = {
        "passed": bool(passed),
        "maximum_validation_median_rollout_rmse_mps": (
            MAXIMUM_VALIDATION_ROLLOUT_RMSE_MPS
        ),
        "minimum_validation_median_persistence_improvement": (
            MINIMUM_VALIDATION_PERSISTENCE_IMPROVEMENT
        ),
        "validation_median_rollout_rmse_mps": validation_median_rmse,
        "validation_median_persistence_improvement": validation_median_improvement,
        "minimum_evaluable_validation_flights": MINIMUM_EVALUABLE_VALIDATION_FLIGHTS,
        "evaluable_validation_flights": evaluable_validation_flights,
        "release_for_coupled_trajectory_test": bool(passed),
    }
    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": (
            "variable-speed-response-validated"
            if passed
            else "variable-speed-response-gate-failed"
        ),
        "leakage_control": {
            "fit_flights_loaded_for_selection": [flight.key for flight in fit_flights],
            "structure_and_gains_frozen_before_validation_load": True,
            "validation_flights_loaded_once_after_freeze": [
                flight.key for flight in validation_flights
            ],
        },
        "model": {
            "mode": "variable_speed",
            "acceleration_command": (
                "K_v * (target_true_airspeed - true_airspeed) + "
                "K_ff * target_true_airspeed_rate"
            ),
            "speed_error_gain_per_s": controller.speed_error_gain_per_s,
            "response_time_constant_s": float(chosen["response_time_constant_s"]),
            "target_rate_feedforward_gain": (
                controller.target_rate_feedforward_gain
            ),
            "acceleration_bounds_mps2": [
                controller.minimum_acceleration_mps2,
                controller.maximum_acceleration_mps2,
            ],
            "trim_thrust": "drag_prior + mass*g*sin(flight_path_angle)",
            "upper_limit": "validated CLB N1 ceiling passed through thrust map",
            "output": "effective thrust command, invertible to commanded N1",
        },
        "selection": _native(chosen.to_dict()),
        "method": {
            "sample_rate_hz": 1,
            "rollout_window_s": ROLL_OUT_WINDOW_S,
            "rollout_stride_s": ROLL_OUT_STRIDE_S,
            "target_rate_window_s": TARGET_RATE_WINDOW_S,
            "candidate_response_time_constants_s": list(
                RESPONSE_TIME_CONSTANTS_S
            ),
            "candidate_target_rate_feedforward_gains": list(
                TARGET_RATE_FEEDFORWARD_GAINS
            ),
            "selection_score": "equal-complete-flight mean 300-second TAS RMSE",
            "simplicity_rule": (
                f"fewest nonzero terms within {NEAR_BEST_FRACTION:.0%} of best"
            ),
            "variable_mode_screen": {
                "dominant_tla_code": DOMINANT_CLB_CODE,
                "minimum_n1_percent": MINIMUM_VARIABLE_MODE_N1_PERCENT,
                "clb_exclusion_margin_n1_percent": (
                    CLB_EXCLUSION_MARGIN_N1_PERCENT
                ),
                "maximum_absolute_target_rate_mps2": MAXIMUM_TARGET_RATE_MPS2,
                "clean_low_bank_maximum_deg": MAX_ABSOLUTE_BANK_DEG,
                "minimum_true_airspeed_mps": MIN_TRUE_AIRSPEED_MPS,
                "maximum_mach": MAX_MACH,
                "static_temperature_k": [
                    MIN_STATIC_TEMPERATURE_K,
                    MAX_STATIC_TEMPERATURE_K,
                ],
            },
        },
        "decision": decision,
        "dependencies": {
            "clb_envelope": str(args.envelope_parameters),
            "effective_thrust_map": str(args.thrust_parameters),
            "drag_prior": {
                "cd0": reference_drag.zero_lift_drag_coefficient,
                "induced_drag_factor": reference_drag.induced_drag_factor,
            },
        },
        "artifacts": {
            "runner": "tools/calibrate_a320_speed_autothrust.py",
            "candidate_metrics": str(args.candidate_metrics),
            "flight_metrics": str(args.flight_metrics),
            "report": str(args.report),
        },
        "limitations": [
            "SPEEDP_SEL is assumed to be the active or equivalent CAS target.",
            "FMA SPEED/MACH and fixed/variable thrust modes are not decoded.",
            "The screen excludes N1 below 50 percent and CLB boundary contacts.",
            "Instantaneous N1 reconstruction omits command and spool lag.",
            "A coupled commanded-N1 trajectory rollout remains required.",
        ],
    }

    args.candidate_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(metrics).to_csv(args.flight_metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(_native(parameters), sort_keys=False))
    _write_report(
        args.report,
        candidates=candidates,
        chosen=chosen,
        metrics=metrics,
        decision=decision,
    )
    return candidates, metrics, parameters


def main() -> None:
    candidates, metrics, parameters = run(parse_args())
    print(candidates.sort_values("mean_flight_rollout_rmse_mps").head(10).to_string(index=False))
    print(metrics.to_string(index=False))
    print(yaml.safe_dump(parameters["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
