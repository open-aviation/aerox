#!/usr/bin/env python3
"""Audit whether A320 QAR contains an observable propulsion command.

Only the ten development/calibration flights are loaded. The audit separates
thrust-lever detents, engine-state duplicates, named guidance targets, and
unresolved FCU/FGC bits. It does not fit an executable spool model unless the
predeclared complete-flight observability gate is met.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from aerox import aero
from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    FT_TO_M,
    KNOT_TO_MPS,
    TIME_COLUMN,
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.modeling.identification import (
    delay_samples,
    robust_weighted_linear_fit,
)
from tools.calibrate_a320_effective_thrust import (
    RAW_FILES,
    VerticalCalibration,
    _phase_equal_weights,
    _read_vertical_calibration,
)


HORIZON_S = 10
PAST_RATE_WINDOW_S = 5
FIT_THINNING_S = 5
DOMINANT_CL_DETENT_CODE = 45.0
TRANSITION_THRESHOLD_N1_PERCENT = 2.0
MIN_TRANSITION_RMSE_IMPROVEMENT = 0.15
MIN_IMPROVED_FLIGHTS = 8
MIN_TRANSITION_SIGN_ACCURACY = 0.65

NAMED_CANDIDATE_CHANNELS = (
    "N1C_1...RPM.",
    "N1C_2...RPM.",
    "N1_TFB_1",
    "EPR1",
    "EPR2",
    "SPEEDP_SEL",
    "MACH_SEL",
    "ALT_SEL",
    "SPEEDV_SEL..ft.mn.",
    "FPA_SEL",
    "VC_TRENDS",
    "FD_PITCHO",
)

MODEL_FEATURES = {
    "state_environment": (
        "constant",
        "n1_centered",
        "mach_centered",
        "pressure_ratio_centered",
        "effective_gamma_scaled",
    ),
    "named_guidance": (
        "constant",
        "n1_centered",
        "mach_centered",
        "pressure_ratio_centered",
        "effective_gamma_scaled",
        "selected_ias_error_scaled",
        "selected_mach_error_scaled",
        "past_cas_rate_scaled",
        "past_selected_ias_rate_scaled",
        "selected_altitude_error_scaled",
        "selected_vertical_speed_scaled",
    ),
}


@dataclass(frozen=True)
class CommandFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    phase: np.ndarray
    model_mask: np.ndarray
    future_n1_change_percent: np.ndarray
    state_environment_design: np.ndarray
    named_guidance_design: np.ndarray
    current_n1_percent: np.ndarray
    speed_error_kts: np.ndarray
    screen_values: dict[str, np.ndarray]
    metrics: dict[str, Any]

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
        default=Path("docs/modeling/a320-propulsion-command-flight-metrics.csv"),
    )
    parser.add_argument(
        "--channel-screen",
        type=Path,
        default=Path("docs/modeling/a320-propulsion-command-channel-screen.csv"),
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=Path("docs/modeling/a320-propulsion-command-model-metrics.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-propulsion-command-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-propulsion-command-audit.md"),
    )
    return parser.parse_args()


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    raw_time = frame[TIME_COLUMN].to_numpy(dtype=float)
    keep = np.isfinite(raw_time)
    order = np.argsort(raw_time[keep], kind="stable")
    indices = np.flatnonzero(keep)[order]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[indices]


def _bounded_candidate(
    values: np.ndarray,
    time_s: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    max_hold_s: float = 1.0,
) -> np.ndarray:
    valid = np.isfinite(values) & (values >= minimum) & (values <= maximum)
    return bounded_sample_and_hold(
        np.where(valid, values, np.nan), time_s, max_hold_s
    )


def _future_change(values: np.ndarray, horizon: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    output = np.full(len(data), np.nan, dtype=float)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon < len(data):
        output[:-horizon] = data[horizon:] - data[:-horizon]
    return output


def _past_rate(values: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    output = np.full(len(data), np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    if window < len(data):
        output[window:] = (data[window:] - data[:-window]) / window
    return output


def _stable_code_mask(values: np.ndarray, code: float, horizon: int) -> np.ndarray:
    """Return samples for which a discrete code remains fixed through horizon."""
    equal = np.isfinite(values) & np.isclose(values, code)
    output = np.zeros(len(equal), dtype=bool)
    if horizon >= len(equal):
        return output
    stable = np.ones(len(equal) - horizon, dtype=bool)
    for offset in range(horizon + 1):
        stable &= equal[offset : offset + len(stable)]
    output[:-horizon] = stable
    return output


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    if np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _dominant(values: np.ndarray) -> tuple[float, float, int, int]:
    finite = values[np.isfinite(values)]
    codes, counts = np.unique(finite, return_counts=True)
    index = int(np.argmax(counts))
    return (
        float(codes[index]),
        float(counts[index] / len(finite)),
        int(len(codes)),
        int(np.count_nonzero(np.diff(finite))),
    )


def _build_flight(
    *,
    registration: str,
    recording: str,
    role: str,
    raw: pd.DataFrame,
    vertical: VerticalCalibration,
    screen_channels: tuple[str, ...],
) -> CommandFlight:
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    time_s = decoded.time_s
    ordered = {name: _ordered_raw(raw, name) for name in screen_channels}
    n1_1 = decoded.channel("n1_1", allow_provisional=True)
    n1_2 = decoded.channel("n1_2", allow_provisional=True)
    n1 = 0.5 * (n1_1 + n1_2)
    tla = 0.5 * (
        decoded.channel("throttle_1", allow_unresolved=True)
        + decoded.channel("throttle_2", allow_unresolved=True)
    )
    cas_kts = decoded.channel("calibrated_airspeed") / KNOT_TO_MPS
    altitude_ft = decoded.channel("pressure_altitude") / FT_TO_M
    pressure_ratio = (
        np.asarray(aero.pressure(decoded.channel("pressure_altitude")), dtype=float)
        / aero.p0
    )
    mach = np.asarray(
        aero.cas2mach(
            decoded.channel("calibrated_airspeed"),
            decoded.channel("pressure_altitude"),
        ),
        dtype=float,
    )
    pitch_code = (
        decoded.channel("pitch", allow_provisional=True) / DEG_TO_RAD / 0.5
    )
    aoa_code = (
        decoded.channel("angle_of_attack", allow_provisional=True)
        / DEG_TO_RAD
        / 0.25
    )
    gamma_deg = (
        vertical.pitch_scale_deg_per_code
        * delay_samples(pitch_code, vertical.delay_s)
        - vertical.angle_of_attack_scale_deg_per_code
        * delay_samples(aoa_code, vertical.delay_s)
        + vertical.effective_gamma_bias_deg
    )

    selected_ias = _bounded_candidate(
        ordered["SPEEDP_SEL"], time_s, minimum=80.0, maximum=400.0
    )
    selected_mach = _bounded_candidate(
        ordered["MACH_SEL"], time_s, minimum=0.0, maximum=1.0
    )
    selected_altitude = _bounded_candidate(
        ordered["ALT_SEL"], time_s, minimum=-2_000.0, maximum=60_000.0
    )
    selected_vertical_speed = _bounded_candidate(
        ordered["SPEEDV_SEL..ft.mn."],
        time_s,
        minimum=-10_000.0,
        maximum=10_000.0,
    )
    speed_error = selected_ias - cas_kts
    future_n1 = _future_change(n1, HORIZON_S)
    cas_rate = _past_rate(cas_kts, PAST_RATE_WINDOW_S)
    selected_ias_rate = _past_rate(selected_ias, PAST_RATE_WINDOW_S)
    stable_cl = _stable_code_mask(tla, DOMINANT_CL_DETENT_CODE, HORIZON_S)
    future_airborne = np.zeros(len(time_s), dtype=bool)
    future_airborne[:-HORIZON_S] = decoded.airborne[HORIZON_S:]
    phase = np.full(len(time_s), "level", dtype="U7")
    phase[gamma_deg > 1.0] = "climb"
    phase[gamma_deg < -1.0] = "descent"

    state_environment = np.column_stack(
        (
            np.ones(len(time_s)),
            (n1 - 70.0) / 20.0,
            (mach - 0.5) / 0.2,
            (pressure_ratio - 0.5) / 0.3,
            gamma_deg / 5.0,
        )
    )
    named_guidance = np.column_stack(
        (
            state_environment,
            speed_error / 20.0,
            (selected_mach - mach) / 0.1,
            cas_rate,
            selected_ias_rate,
            (selected_altitude - altitude_ft) / 10_000.0,
            selected_vertical_speed / 3_000.0,
        )
    )
    model_mask = (
        decoded.airborne
        & future_airborne
        & stable_cl
        & np.isfinite(future_n1)
        & np.all(np.isfinite(named_guidance), axis=1)
    )
    if np.count_nonzero(model_mask) < 500:
        raise ValueError(f"too few command-audit samples for {registration}/{recording}")

    airborne_tla = tla[decoded.airborne & np.isfinite(tla)]
    dominant_code, dominant_fraction, unique_codes, changes = _dominant(airborne_tla)
    dominant_mask = (
        decoded.airborne & np.isfinite(n1) & np.isclose(tla, dominant_code)
    )
    duration_h = (time_s[-1] - time_s[0]) / 3600.0
    n1c = bounded_sample_and_hold(
        ordered["N1C_1...RPM."] * 0.25, time_s, 2.0
    )
    # Provisional signed-wrap hypothesis: raw values appear to be a signed
    # representation of a 9-bit quantity. This is tested, not asserted as a
    # documented conversion.
    n1_tfb = bounded_sample_and_hold(
        np.mod(ordered["N1_TFB_1"], 512.0) * 0.25, time_s, 4.0
    )
    comparison = decoded.airborne & np.isfinite(n1 + n1c)
    tfb_comparison = decoded.airborne & np.isfinite(n1 + n1_tfb)
    transition = model_mask & (
        np.abs(future_n1) >= TRANSITION_THRESHOLD_N1_PERCENT
    )
    epr = np.column_stack((ordered["EPR1"], ordered["EPR2"]))[
        decoded.airborne
    ]
    finite_epr = np.isfinite(epr)
    metrics: dict[str, Any] = {
        "registration": registration,
        "recording": recording,
        "role": role,
        "airborne_samples": int(np.count_nonzero(decoded.airborne)),
        "model_samples": int(np.count_nonzero(model_mask)),
        "transition_samples": int(np.count_nonzero(transition)),
        "tla_dominant_code": dominant_code,
        "tla_dominant_fraction": dominant_fraction,
        "tla_unique_codes": unique_codes,
        "tla_changes_per_hour": changes / duration_h,
        "dominant_tla_n1_p01_percent": float(np.quantile(n1[dominant_mask], 0.01)),
        "dominant_tla_n1_p50_percent": float(np.quantile(n1[dominant_mask], 0.50)),
        "dominant_tla_n1_p99_percent": float(np.quantile(n1[dominant_mask], 0.99)),
        "dominant_tla_n1_p01_to_p99_span_percent": float(
            np.quantile(n1[dominant_mask], 0.99)
            - np.quantile(n1[dominant_mask], 0.01)
        ),
        "selected_ias_unique_values": int(
            len(np.unique(selected_ias[decoded.airborne & np.isfinite(selected_ias)]))
        ),
        "selected_ias_changes_per_hour": float(
            np.count_nonzero(
                np.diff(selected_ias[decoded.airborne & np.isfinite(selected_ias)])
            )
            / duration_h
        ),
        "selected_mach_unique_values": int(
            len(
                np.unique(
                    selected_mach[decoded.airborne & np.isfinite(selected_mach)]
                )
            )
        ),
        "n1c_vs_n1_rmse_percent": float(
            np.sqrt(np.mean((n1c[comparison] - n1[comparison]) ** 2))
        ),
        "n1c_vs_n1_correlation": _safe_correlation(n1c[comparison], n1[comparison]),
        "wrapped_n1_tfb_vs_n1_rmse_percent": float(
            np.sqrt(np.mean((n1_tfb[tfb_comparison] - n1[tfb_comparison]) ** 2))
        ),
        "wrapped_n1_tfb_vs_n1_correlation": _safe_correlation(
            n1_tfb[tfb_comparison], n1[tfb_comparison]
        ),
        "wrapped_n1_tfb_vs_future_n1_change_correlation": _safe_correlation(
            n1_tfb[model_mask], future_n1[model_mask]
        ),
        "epr_nonzero_fraction": float(
            np.mean(epr[finite_epr] != 0.0) if np.any(finite_epr) else np.nan
        ),
    }
    return CommandFlight(
        registration=registration,
        recording=recording,
        role=role,
        time_s=time_s,
        phase=phase,
        model_mask=model_mask,
        future_n1_change_percent=future_n1,
        state_environment_design=state_environment,
        named_guidance_design=named_guidance,
        current_n1_percent=n1,
        speed_error_kts=speed_error,
        screen_values=ordered,
        metrics=metrics,
    )


def _load_fit_flights(
    *,
    raw_dir: Path,
    panel_path: Path,
    vertical: VerticalCalibration,
) -> tuple[list[CommandFlight], tuple[str, ...]]:
    panel = pd.read_csv(panel_path, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(selection)}")
    schema_by_registration = {
        registration: tuple(
            pq.ParquetFile(raw_dir / RAW_FILES[registration]).schema_arrow.names
        )
        for registration in RAW_FILES
    }
    first = set(next(iter(schema_by_registration.values())))
    if any(set(names) != first for names in schema_by_registration.values()):
        raise ValueError("registration raw schemas differ")
    cryptic = tuple(sorted(name for name in first if name.startswith(("FCU.", "FGC."))))
    screen_channels = tuple(dict.fromkeys((*NAMED_CANDIDATE_CHANNELS, *cryptic)))
    required = list(dict.fromkeys((*raw_columns(), *screen_channels)))
    flights: list[CommandFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=required,
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        roles = current.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            flights.append(
                _build_flight(
                    registration=str(registration),
                    recording=str(recording),
                    role=str(roles[str(recording)]),
                    raw=frame,
                    vertical=vertical,
                    screen_channels=screen_channels,
                )
            )
    expected = set(selection["recording"])
    loaded = {flight.recording for flight in flights}
    if loaded != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - loaded)}")
    return flights, screen_channels


def _channel_screen(
    flights: list[CommandFlight], screen_channels: tuple[str, ...]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_model_samples = sum(np.count_nonzero(flight.model_mask) for flight in flights)
    for channel in screen_channels:
        values: list[np.ndarray] = []
        current_n1: list[np.ndarray] = []
        future_n1: list[np.ndarray] = []
        speed_error: list[np.ndarray] = []
        changes = 0
        duration_h = 0.0
        for flight in flights:
            raw_values = flight.screen_values[channel]
            mask = flight.model_mask & np.isfinite(raw_values)
            values.append(raw_values[mask])
            current_n1.append(flight.current_n1_percent[mask])
            future_n1.append(flight.future_n1_change_percent[mask])
            speed_error.append(flight.speed_error_kts[mask])
            finite = raw_values[flight.model_mask & np.isfinite(raw_values)]
            changes += int(np.count_nonzero(np.diff(finite)))
            duration_h += (flight.time_s[-1] - flight.time_s[0]) / 3600.0
        joined = np.concatenate(values)
        n1 = np.concatenate(current_n1)
        future = np.concatenate(future_n1)
        error = np.concatenate(speed_error)
        if len(joined):
            q01, q50, q99 = np.quantile(joined, [0.01, 0.50, 0.99])
            unique_count = len(np.unique(joined))
        else:
            q01 = q50 = q99 = float("nan")
            unique_count = 0
        rows.append(
            {
                "raw_channel": channel,
                "interpretation_status": (
                    "named-unresolved" if channel in NAMED_CANDIDATE_CHANNELS else "cryptic-unresolved"
                ),
                "native_coverage_fraction": len(joined) / total_model_samples,
                "unique_values": unique_count,
                "q01": q01,
                "q50": q50,
                "q99": q99,
                "changes_per_hour": changes / duration_h,
                "current_n1_correlation": _safe_correlation(joined, n1),
                "future_10s_n1_change_correlation": _safe_correlation(joined, future),
                "selected_ias_error_correlation": _safe_correlation(joined, error),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["interpretation_status", "raw_channel"], kind="stable"
    )


def _cross_validated_models(
    flights: list[CommandFlight],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in ("zero_change", *MODEL_FEATURES):
        for held_out in flights:
            mask = held_out.model_mask
            target = held_out.future_n1_change_percent[mask]
            if candidate == "zero_change":
                prediction = np.zeros(len(target))
                condition = float("nan")
            else:
                designs: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                weights: list[np.ndarray] = []
                for flight in flights:
                    if flight.key == held_out.key:
                        continue
                    thin = flight.model_mask & (
                        np.arange(len(flight.time_s)) % FIT_THINNING_S == 0
                    )
                    design = (
                        flight.state_environment_design
                        if candidate == "state_environment"
                        else flight.named_guidance_design
                    )
                    designs.append(design[thin])
                    targets.append(flight.future_n1_change_percent[thin])
                    weights.append(
                        _phase_equal_weights(flight.phase[thin]) / (len(flights) - 1)
                    )
                fit = robust_weighted_linear_fit(
                    np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
                )
                held_design = (
                    held_out.state_environment_design
                    if candidate == "state_environment"
                    else held_out.named_guidance_design
                )
                prediction = held_design[mask] @ fit.coefficients
                condition = float(fit.normal_matrix_condition)
            residual = prediction - target
            transition = np.abs(target) >= TRANSITION_THRESHOLD_N1_PERCENT
            rows.append(
                {
                    "candidate": candidate,
                    "registration": held_out.registration,
                    "recording": held_out.recording,
                    "samples": len(target),
                    "transition_samples": int(np.count_nonzero(transition)),
                    "rmse_n1_percent": float(np.sqrt(np.mean(residual**2))),
                    "bias_n1_percent": float(np.mean(residual)),
                    "transition_rmse_n1_percent": float(
                        np.sqrt(np.mean(residual[transition] ** 2))
                    ),
                    "transition_sign_accuracy": float(
                        np.mean(
                            np.sign(prediction[transition])
                            == np.sign(target[transition])
                        )
                    ),
                    "normal_matrix_condition": condition,
                }
            )
    return pd.DataFrame(rows)


def _observability_decision(model_metrics: pd.DataFrame) -> dict[str, Any]:
    pivot = model_metrics.pivot(
        index=["registration", "recording"],
        columns="candidate",
        values="transition_rmse_n1_percent",
    )
    state_mean = float(pivot["state_environment"].mean())
    guidance_mean = float(pivot["named_guidance"].mean())
    improvement = (state_mean - guidance_mean) / state_mean
    improved_flights = int(
        np.count_nonzero(pivot["named_guidance"] < pivot["state_environment"])
    )
    sign_accuracy = float(
        model_metrics.loc[
            model_metrics["candidate"] == "named_guidance",
            "transition_sign_accuracy",
        ].mean()
    )
    passed = (
        improvement >= MIN_TRANSITION_RMSE_IMPROVEMENT
        and improved_flights >= MIN_IMPROVED_FLIGHTS
        and sign_accuracy >= MIN_TRANSITION_SIGN_ACCURACY
    )
    return {
        "passed": bool(passed),
        "status": (
            "named-command-proxy-observable"
            if passed
            else "no-recorded-engine-command-identified"
        ),
        "state_environment_mean_transition_rmse_n1_percent": state_mean,
        "named_guidance_mean_transition_rmse_n1_percent": guidance_mean,
        "relative_transition_rmse_improvement": improvement,
        "held_out_flights_improved": improved_flights,
        "held_out_flights_total": len(pivot),
        "named_guidance_mean_transition_sign_accuracy": sign_accuracy,
        "gate": {
            "minimum_relative_transition_rmse_improvement": MIN_TRANSITION_RMSE_IMPROVEMENT,
            "minimum_improved_flights": MIN_IMPROVED_FLIGHTS,
            "minimum_transition_sign_accuracy": MIN_TRANSITION_SIGN_ACCURACY,
        },
    }


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["number"]):
        if column in {
            "airborne_samples",
            "model_samples",
            "transition_samples",
            "tla_unique_codes",
            "selected_ias_unique_values",
            "selected_mach_unique_values",
            "samples",
            "unique_values",
        }:
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
        max(len(str(headers[index])), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    flight_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    channel_screen: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    compact_flights = flight_metrics[
        [
            "registration",
            "recording",
            "tla_dominant_fraction",
            "dominant_tla_n1_p01_percent",
            "dominant_tla_n1_p99_percent",
            "selected_ias_unique_values",
            "transition_samples",
        ]
    ].copy()
    compact_flights.columns = [
        "reg",
        "recording",
        "CL-detent fraction",
        "N1 p01 at detent %",
        "N1 p99 at detent %",
        "selected-IAS values",
        "N1 transitions",
    ]
    compact_flights = _round_frame(compact_flights)
    aggregate = (
        model_metrics.groupby("candidate", sort=False)
        .agg(
            mean_RMSE=("rmse_n1_percent", "mean"),
            mean_transition_RMSE=("transition_rmse_n1_percent", "mean"),
            mean_transition_sign_accuracy=("transition_sign_accuracy", "mean"),
        )
        .reset_index()
    )
    aggregate = _round_frame(aggregate)
    decision = audit["decision"]
    screen_dynamic = channel_screen.loc[
        channel_screen["future_10s_n1_change_correlation"].notna()
    ].copy()
    screen_dynamic["absolute_future_correlation"] = screen_dynamic[
        "future_10s_n1_change_correlation"
    ].abs()
    top_screen = screen_dynamic.nlargest(10, "absolute_future_correlation")[
        [
            "raw_channel",
            "interpretation_status",
            "native_coverage_fraction",
            "unique_values",
            "future_10s_n1_change_correlation",
        ]
    ]
    top_screen = _round_frame(top_screen)
    text = f"""# A320 propulsion-command observability audit

## Decision

No recorded engine-command signal has been identified. The named FCU guidance
targets fail the predeclared complete-flight observability gate, and no cryptic
FCU/FGC bit may be assigned physical meaning without a parameter dictionary.
The next executable propulsion interface should therefore accept **commanded
N1 or effective thrust**, with an Aerox speed/energy controller above it. TLA is
retained as a detent/mode/limit signal, not interpreted as continuous throttle
percent.

This audit loads only the ten fit flights. No validation flight is opened.

## Why TLA is insufficient

Airbus documents that autothrust can remain active and target approach or
selected speed while the thrust levers stay in the CLB detent. See Airbus
Safety First, [A Focus on the Landing Flare](https://safetyfirst.airbus.com/a-focus-on-the-landing-flare/).
The QAR independently demonstrates the same distinction: within the dominant
raw TLA code {DOMINANT_CL_DETENT_CODE:.0f}, N1 varies over a broad operating
range on every fit flight.

{_markdown_table(compact_flights)}

Across flights the dominant-detent fraction has median
{flight_metrics['tla_dominant_fraction'].median():.3f}; the median p01-to-p99 N1
span inside that same detent is
{flight_metrics['dominant_tla_n1_p01_to_p99_span_percent'].median():.1f}
percentage points. A fixed TLA-to-N1 map is therefore structurally contradicted
by the data.

## Named-command proxy test

The target is the change in mean N1 over the next {HORIZON_S} seconds, evaluated
only while TLA remains continuously at code {DOMINANT_CL_DETENT_CODE:.0f}.
Transition samples require |ΔN1| ≥ {TRANSITION_THRESHOLD_N1_PERCENT:.1f}
percentage points. Every fitted candidate uses leave-one-complete-flight-out
validation, equal flight/phase weighting, 5-second fit thinning, and all valid
1 Hz samples for metrics.

{_markdown_table(aggregate)}

Adding selected IAS/Mach errors, past CAS/target rates, selected altitude error,
and selected vertical speed changes transition RMSE by only
{100 * decision['relative_transition_rmse_improvement']:.2f}% relative to the
state/environment model. It improves
{decision['held_out_flights_improved']}/{decision['held_out_flights_total']}
held flights and reaches
{100 * decision['named_guidance_mean_transition_sign_accuracy']:.1f}% mean
transition-direction accuracy. The gate required at least
{100 * decision['gate']['minimum_relative_transition_rmse_improvement']:.0f}%,
{decision['gate']['minimum_improved_flights']}/{decision['held_out_flights_total']}
flights, and
{100 * decision['gate']['minimum_transition_sign_accuracy']:.0f}% respectively.

This is an observability screen, not an attempt to reproduce proprietary Airbus
autothrust laws. Closed-loop correlations are not treated as commands.

## Other candidate channels

`N1C_1...RPM.` tracks decoded N1 after its previously recorded provisional
scale. `N1_TFB_1` strongly tracks current N1 under a provisional signed-wrap
hypothesis but does not predict future N1 change, so it is not accepted as a
command. EPR1/EPR2 are zero in these flights. The highest raw correlations from
the complete screen are shown only to make the negative search auditable:

{_markdown_table(top_screen)}

All FCU/FGC labels in this table remain unresolved. Correlation cannot supply
their semantics. Full aggregate screening results are in
`a320-propulsion-command-channel-screen.csv`.

## Next implementation boundary

Use the following layered interface:

```text
speed / energy reference
    -> Aerox speed-energy controller
    -> commanded N1 or effective thrust
    -> asymmetric spool/actuator dynamics
    -> frozen N1/Mach/ambient thrust map
    -> calibrated fuel flow
```

Aerox implements commanded N1 as an explicit input with a bounded asymmetric
effective actuator in `src/aerox/models/thrust/engine.py` and couples it to the
longitudinal plant in `src/aerox/models/longitudinal.py`. Do not assign an A320
nominal rise or fall constant until QAR detent-transition evidence or a
traceable engine source supports it. Recorded N1 remains the held-out state
target. Once actuator parameters are bounded, an Aerox controller can generate
commanded N1 from speed and flight-path references. TLA detents provide mode
and upper-limit evidence where decoded; they do not determine the command
inside CLB.

A subsequent fit-only transition audit in `a320-n1-response-audit.md` supports
only low-power-ground sensitivity bounds. It does not provide a nominal
airborne A320 spool constant.

A separate fit-only N2 audit supports retaining N2 as a second effective
propulsion state, but it does not recover the missing command or release
coupled spool parameters.

## Limitations

- N1 and TLA conversions remain provisional/unresolved recorder hypotheses.
- Named selected-speed/altitude fields lack an upstream parameter dictionary;
  range plausibility does not make their units independently verified.
- Cryptic FCU/FGC bit meanings are unknown and are not inferred from correlation.
- The test is linear and deliberately low-capacity. Its failure rules out the
  tested named proxy, not every possible hidden nonlinear controller state.
- Routine closed-loop QAR cannot distinguish proprietary control logic from
  unrecorded modes and disturbances without an actual command channel.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    vertical = _read_vertical_calibration(args.vertical_parameters)
    flights, screen_channels = _load_fit_flights(
        raw_dir=args.raw_dir,
        panel_path=args.panel,
        vertical=vertical,
    )
    flight_metrics = pd.DataFrame([flight.metrics for flight in flights]).sort_values(
        ["registration", "recording"], kind="stable"
    )
    channel_screen = _channel_screen(flights, screen_channels)
    model_metrics = _cross_validated_models(flights)
    decision = _observability_decision(model_metrics)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "status": decision["status"],
        "leakage_control": {
            "fit_flights_only": [flight.key for flight in flights],
            "validation_flights_loaded": False,
        },
        "source_interpretation": {
            "official_airbus_source": {
                "title": "A Focus on the Landing Flare",
                "url": "https://safetyfirst.airbus.com/a-focus-on-the-landing-flare/",
                "evidence": "autothrust targets approach or selected speed while levers remain in CLB detent",
            },
            "qar_evidence": "N1 varies broadly while TLA remains at dominant code 45",
        },
        "method": {
            "future_n1_horizon_s": HORIZON_S,
            "transition_threshold_n1_percent": TRANSITION_THRESHOLD_N1_PERCENT,
            "dominant_cl_detent_code_raw": DOMINANT_CL_DETENT_CODE,
            "fit_thinning_s": FIT_THINNING_S,
            "cross_validation": "leave-one-fit-flight-out",
            "weighting": "equal flight then equal phase present within flight",
            "candidate_feature_order": {
                name: list(features) for name, features in MODEL_FEATURES.items()
            },
            "cryptic_channels_used_as_model_features": False,
        },
        "decision": decision,
        "engine_channel_findings": {
            "n1c": "alternate actual-N1 encoding under provisional scale",
            "n1_tfb": "unresolved; wrapped hypothesis tracks current N1 but not accepted as command",
            "epr": "zero throughout selected fit-flight airborne samples",
            "tla": "detent/mode/limit evidence, not continuous throttle percent",
        },
        "selected_interface": {
            "plant_input": "commanded N1 or effective thrust",
            "controller": "Aerox speed-energy controller",
            "spool_model": "bounded asymmetric first-order candidate with explicit provenance and uncertainty",
            "nominal_time_constants_available": False,
            "implementation": {
                "actuator": "src/aerox/models/thrust/engine.py",
                "coupled_simulator": "src/aerox/models/longitudinal.py",
            },
            "subsequent_response_audit": {
                "record": "docs/modeling/a320-n1-response-audit.yml",
                "low_power_ground_bounds_available": True,
                "airborne_nominal_available": False,
            },
            "subsequent_n2_audit": {
                "record": "docs/modeling/a320-n2-observability-audit.yml",
                "n2_dynamic_state_supported": True,
                "coupled_two_spool_parameters_released": False,
            },
            "recorded_n1_role": "identification and held-out validation target",
        },
        "artifacts": {
            "runner": "tools/audit_a320_propulsion_commands.py",
            "flight_metrics": str(args.flight_metrics),
            "channel_screen": str(args.channel_screen),
            "model_metrics": str(args.model_metrics),
            "report": str(args.report),
        },
    }
    args.flight_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(flight_metrics).to_csv(args.flight_metrics, index=False)
    _round_frame(channel_screen).to_csv(args.channel_screen, index=False)
    _round_frame(model_metrics).to_csv(args.model_metrics, index=False)
    args.audit.write_text(
        yaml.safe_dump(_native(audit), sort_keys=False, allow_unicode=True)
    )
    _write_report(
        args.report,
        flight_metrics=flight_metrics,
        model_metrics=model_metrics,
        channel_screen=channel_screen,
        audit=audit,
    )
    return flight_metrics, channel_screen, model_metrics, audit


if __name__ == "__main__":
    run(parse_args())
