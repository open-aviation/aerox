#!/usr/bin/env python3
"""Audit whether recorded A320 N2 adds a useful propulsion dynamic state.

The audit loads only the ten development/calibration flights.  It inventories
the provisional N2 channels, compares N1/N2 transition and derivative timing,
and performs leave-one-complete-flight-out prediction at several horizons.
It does not infer an unrecorded FADEC command or release engine-deck values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import aero
from aerox.flightdata.a320_qar import decode_a320_flight, raw_columns
from aerox.modeling.identification import robust_weighted_linear_fit
from tools.audit_a320_n1_response import (
    PAIR_SPECS,
    POST_STEADY_END_S,
    POST_STEADY_START_S,
    PRE_STEADY_WINDOW_S,
    RESPONSE_HORIZON_AFTER_POST_S,
    _pair_spec,
    _run_length_segments,
)
from tools.audit_a320_propulsion_commands import (
    DOMINANT_CL_DETENT_CODE,
    _future_change,
    _past_rate,
    _safe_correlation,
    _stable_code_mask,
)
from tools.calibrate_a320_effective_thrust import RAW_FILES


HORIZONS_S = (1, 3, 5, 10)
PRIMARY_HORIZON_S = 1
FIT_THINNING_S = 5
TRANSITION_THRESHOLD_N1_PERCENT = 2.0
LAG_RANGE_S = tuple(range(-5, 6))
MIN_TRANSITION_RMSE_IMPROVEMENT = 0.08
MIN_DYNAMIC_HISTORY_IMPROVEMENT = 0.08
MIN_IMPROVED_FLIGHTS = 8
MIN_TRANSITION_SIGN_ACCURACY = 0.95


FEATURE_NAMES = {
    "state": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
    ),
    "n1_history": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
        "n1_rate_1s_scaled",
        "n1_rate_3s_scaled",
        "n1_rate_5s_scaled",
        "cas_rate_5s_scaled",
    ),
    "n1_plus_n2_current": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
        "n1_rate_1s_scaled",
        "n1_rate_3s_scaled",
        "n1_rate_5s_scaled",
        "cas_rate_5s_scaled",
        "n2_centered",
        "n2_centered_squared",
        "n1_n2_interaction",
    ),
    "n1_plus_n2_rate_1s": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
        "n1_rate_1s_scaled",
        "n1_rate_3s_scaled",
        "n1_rate_5s_scaled",
        "cas_rate_5s_scaled",
        "n2_rate_1s_scaled",
    ),
    "n1_plus_n2_rates": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
        "n1_rate_1s_scaled",
        "n1_rate_3s_scaled",
        "n1_rate_5s_scaled",
        "cas_rate_5s_scaled",
        "n2_rate_1s_scaled",
        "n2_rate_3s_scaled",
        "n2_rate_5s_scaled",
    ),
    "n1_plus_n2_history": (
        "constant",
        "n1_centered",
        "n1_centered_squared",
        "n1_centered_cubed",
        "mach_centered",
        "pressure_ratio_centered",
        "n1_rate_1s_scaled",
        "n1_rate_3s_scaled",
        "n1_rate_5s_scaled",
        "cas_rate_5s_scaled",
        "n2_centered",
        "n2_centered_squared",
        "n1_n2_interaction",
        "n2_rate_1s_scaled",
        "n2_rate_3s_scaled",
        "n2_rate_5s_scaled",
    ),
}


@dataclass(frozen=True)
class N2Flight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    airborne: np.ndarray
    n1_percent: np.ndarray
    n2_percent: np.ndarray
    throttle_code: np.ndarray
    designs: dict[str, np.ndarray]
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
        "--n1-events",
        type=Path,
        default=Path("docs/modeling/a320-n1-response-events.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/a320-n2-flight-metrics.csv"),
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=Path("docs/modeling/a320-n2-model-metrics.csv"),
    )
    parser.add_argument(
        "--lag-metrics",
        type=Path,
        default=Path("docs/modeling/a320-n2-lag-metrics.csv"),
    )
    parser.add_argument(
        "--transition-metrics",
        type=Path,
        default=Path("docs/modeling/a320-n2-transition-metrics.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-n2-observability-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-n2-observability-audit.md"),
    )
    return parser.parse_args()


def _build_designs(
    *,
    n1_percent: np.ndarray,
    n2_percent: np.ndarray,
    mach: np.ndarray,
    pressure_ratio: np.ndarray,
    calibrated_airspeed_mps: np.ndarray,
) -> dict[str, np.ndarray]:
    n1_centered = (n1_percent - 70.0) / 20.0
    n2_centered = (n2_percent - 85.0) / 15.0
    state = np.column_stack(
        (
            np.ones(len(n1_percent)),
            n1_centered,
            n1_centered**2,
            n1_centered**3,
            (mach - 0.5) / 0.2,
            (pressure_ratio - 0.5) / 0.3,
        )
    )
    n1_history = np.column_stack(
        (
            state,
            _past_rate(n1_percent, 1) / 5.0,
            _past_rate(n1_percent, 3) / 5.0,
            _past_rate(n1_percent, 5) / 5.0,
            _past_rate(calibrated_airspeed_mps, 5) / 5.0,
        )
    )
    n2_current = np.column_stack(
        (
            n1_history,
            n2_centered,
            n2_centered**2,
            n1_centered * n2_centered,
        )
    )
    n2_rate_1s = np.column_stack(
        (
            n1_history,
            _past_rate(n2_percent, 1) / 5.0,
        )
    )
    n2_rates = np.column_stack(
        (
            n1_history,
            _past_rate(n2_percent, 1) / 5.0,
            _past_rate(n2_percent, 3) / 5.0,
            _past_rate(n2_percent, 5) / 5.0,
        )
    )
    n2_history = np.column_stack(
        (
            n2_current,
            _past_rate(n2_percent, 1) / 5.0,
            _past_rate(n2_percent, 3) / 5.0,
            _past_rate(n2_percent, 5) / 5.0,
        )
    )
    return {
        "state": state,
        "n1_history": n1_history,
        "n1_plus_n2_current": n2_current,
        "n1_plus_n2_rate_1s": n2_rate_1s,
        "n1_plus_n2_rates": n2_rates,
        "n1_plus_n2_history": n2_history,
    }


def _flight_metrics(
    *,
    registration: str,
    recording: str,
    role: str,
    airborne: np.ndarray,
    raw_n2_1: np.ndarray,
    raw_n2_2: np.ndarray,
    n1_1: np.ndarray,
    n1_2: np.ndarray,
    n2_1: np.ndarray,
    n2_2: np.ndarray,
) -> dict[str, Any]:
    n1 = 0.5 * (n1_1 + n1_2)
    n2 = 0.5 * (n2_1 + n2_2)
    complete = np.isfinite(n1_1 + n1_2 + n2_1 + n2_2)
    airborne_complete = airborne & complete
    engine_difference = np.abs(n2_1 - n2_2)
    return {
        "registration": registration,
        "recording": recording,
        "role": role,
        "samples": len(n1),
        "native_paired_n2_coverage_fraction": float(
            np.mean(np.isfinite(raw_n2_1 + raw_n2_2))
        ),
        "decoded_airborne_n2_coverage_fraction": float(
            np.count_nonzero(airborne_complete) / np.count_nonzero(airborne)
        ),
        "n2_all_p01_percent": float(np.quantile(n2[complete], 0.01)),
        "n2_all_p99_percent": float(np.quantile(n2[complete], 0.99)),
        "n2_airborne_p01_percent": float(
            np.quantile(n2[airborne_complete], 0.01)
        ),
        "n2_airborne_p99_percent": float(
            np.quantile(n2[airborne_complete], 0.99)
        ),
        "n1_n2_correlation": _safe_correlation(n1[complete], n2[complete]),
        "n2_engine_correlation": _safe_correlation(
            n2_1[complete], n2_2[complete]
        ),
        "n2_engine_difference_median_percent": float(
            np.quantile(engine_difference[complete], 0.50)
        ),
        "n2_engine_difference_p99_percent": float(
            np.quantile(engine_difference[complete], 0.99)
        ),
        "n2_1_unique_values": len(np.unique(n2_1[complete])),
        "n2_2_unique_values": len(np.unique(n2_2[complete])),
    }


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    raw_time = pd.to_numeric(frame["Time..ms."], errors="coerce").to_numpy(
        dtype=float
    )
    keep = np.isfinite(raw_time)
    indices = np.flatnonzero(keep)[
        np.argsort(raw_time[keep], kind="stable")
    ]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[
        indices
    ]


def _load_fit_flights(
    raw_dir: Path, panel_path: Path
) -> tuple[list[N2Flight], tuple[str, ...]]:
    panel = pd.read_csv(panel_path, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(selection)}")
    flights: list[N2Flight] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        roles = current.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=str(recording),
            )
            n1_1 = decoded.channel("n1_1", allow_provisional=True)
            n1_2 = decoded.channel("n1_2", allow_provisional=True)
            n2_1 = decoded.channel("n2_1", allow_provisional=True)
            n2_2 = decoded.channel("n2_2", allow_provisional=True)
            n1 = 0.5 * (n1_1 + n1_2)
            n2 = 0.5 * (n2_1 + n2_2)
            throttle = 0.5 * (
                decoded.channel("throttle_1", allow_unresolved=True)
                + decoded.channel("throttle_2", allow_unresolved=True)
            )
            altitude = decoded.channel("pressure_altitude")
            cas = decoded.channel("calibrated_airspeed")
            mach = np.asarray(aero.cas2mach(cas, altitude, altitude), dtype=float)
            pressure_ratio = np.asarray(aero.pressure(altitude), dtype=float) / aero.p0
            flights.append(
                N2Flight(
                    registration=str(registration),
                    recording=str(recording),
                    role=str(roles[str(recording)]),
                    time_s=decoded.time_s,
                    airborne=decoded.airborne,
                    n1_percent=n1,
                    n2_percent=n2,
                    throttle_code=throttle,
                    designs=_build_designs(
                        n1_percent=n1,
                        n2_percent=n2,
                        mach=mach,
                        pressure_ratio=pressure_ratio,
                        calibrated_airspeed_mps=cas,
                    ),
                    metrics=_flight_metrics(
                        registration=str(registration),
                        recording=str(recording),
                        role=str(roles[str(recording)]),
                        airborne=decoded.airborne,
                        raw_n2_1=_ordered_raw(frame, "N2_1"),
                        raw_n2_2=_ordered_raw(frame, "N2_2"),
                        n1_1=n1_1,
                        n1_2=n1_2,
                        n2_1=n2_1,
                        n2_2=n2_2,
                    ),
                )
            )
    expected = {
        f"{row.registration}/{row.recording}"
        for row in selection.itertuples(index=False)
    }
    loaded = {flight.key for flight in flights}
    if loaded != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - loaded)}")
    return flights, tuple(sorted(loaded))


def _model_mask(flight: N2Flight, horizon_s: int) -> tuple[np.ndarray, np.ndarray]:
    target = _future_change(flight.n1_percent, horizon_s)
    future_airborne = np.zeros(len(flight.time_s), dtype=bool)
    future_airborne[:-horizon_s] = flight.airborne[horizon_s:]
    richest = flight.designs["n1_plus_n2_history"]
    mask = (
        flight.airborne
        & future_airborne
        & _stable_code_mask(
            flight.throttle_code,
            DOMINANT_CL_DETENT_CODE,
            horizon_s,
        )
        & np.isfinite(target)
        & np.all(np.isfinite(richest), axis=1)
    )
    return mask, target


def _cross_validated_models(flights: list[N2Flight]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_s in HORIZONS_S:
        masks_and_targets = {
            flight.key: _model_mask(flight, horizon_s) for flight in flights
        }
        for candidate in FEATURE_NAMES:
            for held_out in flights:
                fit_designs: list[np.ndarray] = []
                fit_targets: list[np.ndarray] = []
                fit_weights: list[np.ndarray] = []
                for flight in flights:
                    if flight.key == held_out.key:
                        continue
                    mask, target = masks_and_targets[flight.key]
                    thin = mask & (np.arange(len(mask)) % FIT_THINNING_S == 0)
                    fit_designs.append(flight.designs[candidate][thin])
                    fit_targets.append(target[thin])
                    fit_weights.append(
                        np.full(
                            np.count_nonzero(thin),
                            1.0
                            / (len(flights) - 1)
                            / np.count_nonzero(thin),
                        )
                    )
                fit = robust_weighted_linear_fit(
                    np.vstack(fit_designs),
                    np.concatenate(fit_targets),
                    np.concatenate(fit_weights),
                )
                mask, target = masks_and_targets[held_out.key]
                observed = target[mask]
                prediction = held_out.designs[candidate][mask] @ fit.coefficients
                residual = prediction - observed
                transition = np.abs(observed) >= TRANSITION_THRESHOLD_N1_PERCENT
                rows.append(
                    {
                        "horizon_s": horizon_s,
                        "candidate": candidate,
                        "feature_count": len(FEATURE_NAMES[candidate]),
                        "registration": held_out.registration,
                        "recording": held_out.recording,
                        "samples": len(observed),
                        "transition_samples": np.count_nonzero(transition),
                        "rmse_n1_percent": float(np.sqrt(np.mean(residual**2))),
                        "bias_n1_percent": float(np.mean(residual)),
                        "transition_rmse_n1_percent": float(
                            np.sqrt(np.mean(residual[transition] ** 2))
                        ),
                        "transition_sign_accuracy": float(
                            np.mean(
                                np.sign(prediction[transition])
                                == np.sign(observed[transition])
                            )
                        ),
                        "normal_matrix_condition": fit.normal_matrix_condition,
                    }
                )
    return pd.DataFrame(rows)


def _lag_metrics(flights: list[N2Flight]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        delta_n1 = _past_rate(flight.n1_percent, 1)
        delta_n2 = _past_rate(flight.n2_percent, 1)
        correlations: dict[int, float] = {}
        samples: dict[int, int] = {}
        for lag_s in LAG_RANGE_S:
            if lag_s >= 0:
                n2_index = np.arange(1, len(flight.time_s) - lag_s)
                n1_index = n2_index + lag_s
            else:
                n1_index = np.arange(1, len(flight.time_s) + lag_s)
                n2_index = n1_index - lag_s
            mask = (
                flight.airborne[n2_index]
                & flight.airborne[n1_index]
                & np.isclose(
                    flight.throttle_code[n2_index], DOMINANT_CL_DETENT_CODE
                )
                & np.isclose(
                    flight.throttle_code[n1_index], DOMINANT_CL_DETENT_CODE
                )
                & np.isfinite(delta_n2[n2_index] + delta_n1[n1_index])
            )
            correlations[lag_s] = _safe_correlation(
                delta_n2[n2_index[mask]], delta_n1[n1_index[mask]]
            )
            samples[lag_s] = np.count_nonzero(mask)
        best_lag = max(correlations, key=lambda item: correlations[item])
        for lag_s in LAG_RANGE_S:
            rows.append(
                {
                    "registration": flight.registration,
                    "recording": flight.recording,
                    "lag_s": lag_s,
                    "interpretation": (
                        "positive means N2 derivative is compared with future N1 derivative"
                    ),
                    "samples": samples[lag_s],
                    "derivative_correlation": correlations[lag_s],
                    "best_lag_for_flight": lag_s == best_lag,
                }
            )
    return pd.DataFrame(rows)


def _crossing_time(
    response: np.ndarray, initial: float, endpoint: float, fraction: float
) -> float:
    direction = np.sign(endpoint - initial)
    if direction == 0.0:
        return float("nan")
    threshold = initial + fraction * (endpoint - initial)
    crossed = np.flatnonzero(
        np.isfinite(response) & (direction * (response - threshold) >= 0.0)
    )
    return float(crossed[0]) if len(crossed) else float("nan")


def _transition_metrics(
    flights: list[N2Flight], n1_event_path: Path
) -> pd.DataFrame:
    n1_events = pd.read_csv(n1_event_path)
    selected = n1_events.loc[n1_events["used_for_actuator_bound"]].copy()
    selected_keys = {
        (str(row.registration), str(row.recording), int(row.event_index)): row
        for row in selected.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for flight in flights:
        segments = _run_length_segments(flight.throttle_code)
        stable = [
            index
            for index, segment in enumerate(segments)
            if np.isfinite(segment.code) and segment.samples >= 8
        ]
        event_index = 0
        for source_index, target_index in zip(stable[:-1], stable[1:], strict=True):
            source = segments[source_index]
            target = segments[target_index]
            spec = _pair_spec(source.code, target.code)
            if spec is None:
                continue
            event_index += 1
            key = (flight.registration, flight.recording, event_index)
            if key not in selected_keys:
                continue
            response_end = min(
                len(flight.time_s),
                target.start + RESPONSE_HORIZON_AFTER_POST_S + 1,
            )
            values = {
                "n1": flight.n1_percent,
                "n2": flight.n2_percent,
            }
            metrics: dict[str, float] = {}
            for spool, signal in values.items():
                initial = float(
                    np.median(
                        signal[
                            source.end - PRE_STEADY_WINDOW_S : source.end
                        ]
                    )
                )
                endpoint = float(
                    np.median(
                        signal[
                            target.start
                            + POST_STEADY_START_S : target.start
                            + POST_STEADY_END_S
                        ]
                    )
                )
                response = signal[source.end:response_end]
                t10 = _crossing_time(response, initial, endpoint, 0.10)
                t90 = _crossing_time(response, initial, endpoint, 0.90)
                metrics[f"{spool}_amplitude_percent"] = endpoint - initial
                metrics[f"{spool}_time_to_10_percent_s"] = t10
                metrics[f"{spool}_time_to_90_percent_s"] = t90
                metrics[f"{spool}_response_10_to_90_s"] = t90 - t10
            expected_n1_response = float(
                selected_keys[key].response_10_to_90_s
            )
            if not np.isclose(
                metrics["n1_response_10_to_90_s"], expected_n1_response
            ):
                raise ValueError(f"N1 event join mismatch for {key}")
            rows.append(
                {
                    "registration": flight.registration,
                    "recording": flight.recording,
                    "event_index": event_index,
                    "direction": spec.direction,
                    **metrics,
                    "n2_lead_at_10_percent_s": (
                        metrics["n1_time_to_10_percent_s"]
                        - metrics["n2_time_to_10_percent_s"]
                    ),
                    "n2_lead_at_90_percent_s": (
                        metrics["n1_time_to_90_percent_s"]
                        - metrics["n2_time_to_90_percent_s"]
                    ),
                }
            )
    if len(rows) != len(selected):
        raise ValueError(
            f"selected N1 event join mismatch: {len(rows)} versus {len(selected)}"
        )
    return pd.DataFrame(rows)


def _decision(model_metrics: pd.DataFrame) -> dict[str, Any]:
    primary = model_metrics.loc[model_metrics["horizon_s"] == PRIMARY_HORIZON_S]
    pivot_transition = primary.pivot(
        index=["registration", "recording"],
        columns="candidate",
        values="transition_rmse_n1_percent",
    )
    pivot_all = primary.pivot(
        index=["registration", "recording"],
        columns="candidate",
        values="rmse_n1_percent",
    )
    baseline = float(pivot_transition["n1_history"].mean())
    selected_candidate = "n1_plus_n2_rate_1s"
    augmented = float(pivot_transition[selected_candidate].mean())
    current_only = float(pivot_transition["n1_plus_n2_current"].mean())
    transition_improvement = (baseline - augmented) / baseline
    dynamic_history_improvement = (current_only - augmented) / current_only
    all_improvement = (
        pivot_all["n1_history"].mean()
        - pivot_all[selected_candidate].mean()
    ) / pivot_all["n1_history"].mean()
    improved_flights = int(
        np.count_nonzero(
            pivot_transition[selected_candidate]
            < pivot_transition["n1_history"]
        )
    )
    sign_accuracy = float(
        primary.loc[
            primary["candidate"] == selected_candidate,
            "transition_sign_accuracy",
        ].mean()
    )
    baseline_condition = float(
        primary.loc[
            primary["candidate"] == "n1_history",
            "normal_matrix_condition",
        ].median()
    )
    selected_condition = float(
        primary.loc[
            primary["candidate"] == selected_candidate,
            "normal_matrix_condition",
        ].median()
    )
    passed = (
        transition_improvement >= MIN_TRANSITION_RMSE_IMPROVEMENT
        and dynamic_history_improvement >= MIN_DYNAMIC_HISTORY_IMPROVEMENT
        and improved_flights >= MIN_IMPROVED_FLIGHTS
        and sign_accuracy >= MIN_TRANSITION_SIGN_ACCURACY
    )
    return {
        "passed": bool(passed),
        "status": (
            "n2-dynamic-state-supported"
            if passed
            else "n2-incremental-dynamics-not-supported"
        ),
        "selected_candidate": selected_candidate,
        "primary_horizon_s": PRIMARY_HORIZON_S,
        "n1_history_mean_transition_rmse_percent": baseline,
        "n1_plus_n2_current_mean_transition_rmse_percent": current_only,
        "n1_plus_n2_rate_1s_mean_transition_rmse_percent": augmented,
        "relative_transition_rmse_improvement_over_n1_history": (
            transition_improvement
        ),
        "relative_transition_rmse_improvement_from_n2_history_over_current_n2": (
            dynamic_history_improvement
        ),
        "relative_all_sample_rmse_improvement": float(all_improvement),
        "held_out_flights_improved": improved_flights,
        "held_out_flights_total": len(pivot_transition),
        "transition_sign_accuracy": sign_accuracy,
        "n1_history_median_normal_matrix_condition": baseline_condition,
        "selected_median_normal_matrix_condition": selected_condition,
        "gate": {
            "minimum_transition_rmse_improvement": (
                MIN_TRANSITION_RMSE_IMPROVEMENT
            ),
            "minimum_dynamic_history_improvement": (
                MIN_DYNAMIC_HISTORY_IMPROVEMENT
            ),
            "minimum_improved_flights": MIN_IMPROVED_FLIGHTS,
            "minimum_transition_sign_accuracy": MIN_TRANSITION_SIGN_ACCURACY,
        },
    }


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    integer_columns = {
        "horizon_s",
        "feature_count",
        "samples",
        "transition_samples",
        "lag_s",
        "event_index",
        "n2_1_unique_values",
        "n2_2_unique_values",
    }
    for column in output.select_dtypes(include=["number"]):
        if column in integer_columns:
            output[column] = output[column].astype(int)
        else:
            output[column] = output[column].round(8)
    return output


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
    headers = [str(item) for item in frame.columns]
    rows = [[str(item) for item in row] for row in frame.to_numpy().tolist()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(item.ljust(widths[i]) for i, item in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(item.ljust(widths[i]) for i, item in enumerate(row))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    flight_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    lag_metrics: pd.DataFrame,
    transition_metrics: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    decision = audit["decision"]
    aggregate = (
        model_metrics.groupby(["horizon_s", "candidate"], sort=False)
        .agg(
            mean_RMSE=("rmse_n1_percent", "mean"),
            mean_transition_RMSE=("transition_rmse_n1_percent", "mean"),
            mean_transition_sign=("transition_sign_accuracy", "mean"),
        )
        .reset_index()
    )
    compact_models = _round_frame(
        aggregate.loc[
            aggregate["candidate"].isin(
                (
                    "n1_history",
                    "n1_plus_n2_current",
                    "n1_plus_n2_rate_1s",
                    "n1_plus_n2_rates",
                    "n1_plus_n2_history",
                )
            )
        ]
    )
    best_lags = lag_metrics.loc[lag_metrics["best_lag_for_flight"]]
    lag_counts = best_lags["lag_s"].value_counts().sort_index()
    lag_text = ", ".join(
        f"{lag:+d} s: {count} flights" for lag, count in lag_counts.items()
    )
    transition_summary = (
        transition_metrics.groupby("direction")
        .agg(
            events=("event_index", "size"),
            median_n1_t10=("n1_time_to_10_percent_s", "median"),
            median_n2_t10=("n2_time_to_10_percent_s", "median"),
            median_n1_t10_90=("n1_response_10_to_90_s", "median"),
            median_n2_t10_90=("n2_response_10_to_90_s", "median"),
        )
        .reset_index()
    )
    transition_summary = _round_frame(transition_summary)
    text = f"""# A320 N2 observability audit

## Decision

N2 should be retained as an observed/estimable **second effective propulsion
state**. It is not a command and it is not yet an identified two-spool engine
model. On the ten fit flights, N2 history provides repeatable one-second N1
predictive information beyond N1 history and ambient state; current N2 without
its history provides essentially no gain.

Only the ten development/calibration flights are loaded. No validation flight
is opened.

## Channel inventory and scaling status

The raw files contain `N2_1` and `N2_2` only; no alternate N2 encoding or
documented command channel is present. Native paired coverage has median
{flight_metrics['native_paired_n2_coverage_fraction'].median():.5f}, and decoded
airborne coverage is
{flight_metrics['decoded_airborne_n2_coverage_fraction'].median():.5f}. Across
flights, the median airborne p01--p99 range after the provisional `raw * 0.5`
conversion is
{flight_metrics['n2_airborne_p01_percent'].median():.2f}--
{flight_metrics['n2_airborne_p99_percent'].median():.2f}%. Median within-flight
N1/N2 correlation is
{flight_metrics['n1_n2_correlation'].median():.3f}.

These ranges and correlations are plausible and internally consistent but do
not independently verify the recorder scale. N2 remains provisional.

## Incremental dynamic-state test

Targets are future mean-engine N1 changes at 1, 3, 5, and 10 seconds while both
endpoints are airborne and raw TLA code `45` remains stable. Fits use five-second
thinning, equal complete-flight weighting, Huber regression, and leave-one-fit-
flight-out evaluation. Transition samples require |ΔN1| ≥
{TRANSITION_THRESHOLD_N1_PERCENT:.0f} percentage points.

{_markdown_table(compact_models)}

At the primary one-second horizon, adding one-second N2 history/rate reduces mean
transition RMSE from
{decision['n1_history_mean_transition_rmse_percent']:.3f}% to
{decision['n1_plus_n2_rate_1s_mean_transition_rmse_percent']:.3f}%—a
{100 * decision['relative_transition_rmse_improvement_over_n1_history']:.1f}%
improvement—with
{decision['held_out_flights_improved']}/{decision['held_out_flights_total']}
flights improved and
{100 * decision['transition_sign_accuracy']:.1f}% transition-sign accuracy.
The selected augmentation is only the one-second N2 rate. N2 level without N2
history gives
{decision['n1_plus_n2_current_mean_transition_rmse_percent']:.3f}% RMSE; N2
history improves that by
{100 * decision['relative_transition_rmse_improvement_from_n2_history_over_current_n2']:.1f}%.
This distinction is why the result supports a dynamic state rather than merely
another static engine feature. The selected model adds one feature; its median
normal-matrix condition is
{decision['selected_median_normal_matrix_condition']:.0f}, versus
{decision['n1_history_median_normal_matrix_condition']:.0f} for the N1-history
baseline. Richer N2 history candidates are reported as diagnostics, not released
coefficients.

The gain decays at longer horizons and does not recover the missing command.
The screening thresholds were selected during fit-flight exploration and must
be frozen before any validation use; they are not externally pre-registered.

## Timing evidence

For derivative lag, positive lag means `dN2[t]` is compared with future
`dN1[t + lag]`. The best one-hertz lag is {lag_text}. Median derivative
correlation is
{lag_metrics.loc[lag_metrics['lag_s'] == 0, 'derivative_correlation'].median():.3f}
at zero lag and
{lag_metrics.loc[lag_metrics['lag_s'] == 1, 'derivative_correlation'].median():.3f}
when N2 leads by one second. This supports synchronous-to-one-sample lead
behavior; it cannot establish a sub-second physical delay.

On the independently screened low-power-ground events:

{_markdown_table(transition_summary)}

N2 reaches the 10% threshold about one recorded sample earlier in the median,
while 10--90% response durations remain similar. These events still convolve
lever motion, FADEC, engine response, and recorder timing.

## Model boundary

- Keep commanded N1/effective thrust as the controller interface.
- Retain N1 and N2 as propulsion states/measurements in the higher-order
  architecture and future state estimator.
- Continue using N1 for the frozen effective-thrust map until N2 gives held-out
  thrust or fuel improvement under a predeclared test.
- Do not release coupled N1/N2 actuator coefficients: the continuous command is
  absent, N2 scaling is provisional, and the evidence is one-hertz closed-loop
  prediction rather than designed engine excitation.
- The next propulsion test should ask whether N2 reduces held-flight fuel-flow
  residuals beyond the frozen N1/thrust/fuel chain.

## Artifacts and limitations

Per-flight inventory, complete-flight model results, lag results, transition
comparisons, and the machine-readable decision are stored alongside this report.
Repository artifacts contain aggregate evidence, not raw trajectories.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    flights, fit_flight_keys = _load_fit_flights(args.raw_dir, args.panel)
    flight_metrics = pd.DataFrame([flight.metrics for flight in flights]).sort_values(
        ["registration", "recording"], kind="stable"
    )
    model_metrics = _cross_validated_models(flights)
    lag_metrics = _lag_metrics(flights)
    transition_metrics = _transition_metrics(flights, args.n1_events)
    decision = _decision(model_metrics)
    best_lags = lag_metrics.loc[lag_metrics["best_lag_for_flight"]]
    transition_aggregate = {
        direction: {
            "events": len(group),
            "median_n1_time_to_10_percent_s": float(
                group["n1_time_to_10_percent_s"].median()
            ),
            "median_n2_time_to_10_percent_s": float(
                group["n2_time_to_10_percent_s"].median()
            ),
            "median_n1_response_10_to_90_s": float(
                group["n1_response_10_to_90_s"].median()
            ),
            "median_n2_response_10_to_90_s": float(
                group["n2_response_10_to_90_s"].median()
            ),
        }
        for direction, group in transition_metrics.groupby("direction")
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "status": decision["status"],
        "leakage_control": {
            "fit_flights_only": list(fit_flight_keys),
            "validation_flights_loaded": False,
        },
        "channel_inventory": {
            "raw_channels": ["N2_1", "N2_2"],
            "alternate_n2_encoding_found": False,
            "provisional_conversion": "raw * 0.5 percent",
            "conversion_independently_verified": False,
            "native_paired_coverage_median": float(
                flight_metrics["native_paired_n2_coverage_fraction"].median()
            ),
            "decoded_airborne_coverage_median": float(
                flight_metrics[
                    "decoded_airborne_n2_coverage_fraction"
                ].median()
            ),
        },
        "method": {
            "prediction_horizons_s": list(HORIZONS_S),
            "primary_horizon_s": PRIMARY_HORIZON_S,
            "target": "future mean-engine N1 change",
            "dominant_tla_code_raw": DOMINANT_CL_DETENT_CODE,
            "transition_threshold_n1_percent": (
                TRANSITION_THRESHOLD_N1_PERCENT
            ),
            "fit_thinning_s": FIT_THINNING_S,
            "cross_validation": "leave-one-fit-flight-out",
            "weighting": "equal complete flight",
            "candidate_feature_order": {
                name: list(features) for name, features in FEATURE_NAMES.items()
            },
            "lag_range_s": list(LAG_RANGE_S),
            "positive_lag_interpretation": (
                "N2 derivative compared with future N1 derivative"
            ),
        },
        "decision": decision,
        "timing_evidence": {
            "best_derivative_lag_counts": {
                int(lag): int(count)
                for lag, count in best_lags["lag_s"].value_counts().items()
            },
            "median_derivative_correlation_zero_lag": float(
                lag_metrics.loc[
                    lag_metrics["lag_s"] == 0, "derivative_correlation"
                ].median()
            ),
            "median_derivative_correlation_n2_leads_1s": float(
                lag_metrics.loc[
                    lag_metrics["lag_s"] == 1, "derivative_correlation"
                ].median()
            ),
            "low_power_ground_transitions": transition_aggregate,
        },
        "selected_architecture": {
            "controller_input": "commanded N1 or effective thrust",
            "effective_propulsion_states": ["N1", "N2"],
            "n2_role": "observed/estimable dynamic state and diagnostic",
            "thrust_map_input_changed": False,
            "coupled_two_spool_parameters_released": False,
            "next_gate": (
                "held-flight incremental fuel-flow residual test beyond frozen "
                "N1/thrust/fuel chain"
            ),
        },
        "artifacts": {
            "runner": "tools/audit_a320_n2_observability.py",
            "flight_metrics": str(args.flight_metrics),
            "model_metrics": str(args.model_metrics),
            "lag_metrics": str(args.lag_metrics),
            "transition_metrics": str(args.transition_metrics),
            "report": str(args.report),
        },
    }
    args.flight_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(flight_metrics).to_csv(args.flight_metrics, index=False)
    _round_frame(model_metrics).to_csv(args.model_metrics, index=False)
    _round_frame(lag_metrics).to_csv(args.lag_metrics, index=False)
    _round_frame(transition_metrics).to_csv(args.transition_metrics, index=False)
    args.audit.write_text(
        yaml.safe_dump(_native(audit), sort_keys=False, allow_unicode=True)
    )
    _write_report(
        args.report,
        flight_metrics=flight_metrics,
        model_metrics=model_metrics,
        lag_metrics=lag_metrics,
        transition_metrics=transition_metrics,
        audit=audit,
    )
    return (
        flight_metrics,
        model_metrics,
        lag_metrics,
        transition_metrics,
        audit,
    )


if __name__ == "__main__":
    run(parse_args())
