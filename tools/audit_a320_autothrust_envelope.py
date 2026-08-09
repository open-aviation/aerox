#!/usr/bin/env python3
"""Infer an effective A320 idle-to-CLB N1 envelope from complete flights.

The fit separates the condition-dependent N1 boundaries from the continuous
autothrust demand inside those boundaries.  Structure selection uses only the
ten development/calibration flights.  Six validation flights are loaded once
after the feature set and extraction rules have been frozen in this file.
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
from aerox.flightdata.a320_qar import (
    FT_TO_M,
    KNOT_TO_MPS,
    TIME_COLUMN,
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.modeling.identification import robust_weighted_linear_fit
from aerox.models.thrust import (
    N1_ENVELOPE_FEATURE_SETS,
    EffectiveN1Envelope,
    n1_envelope_features,
)
from aerox.modeling.force_reconstruction import centered_linear_derivative
from tools.calibrate_a320_effective_thrust import PROCESSED_COLUMNS, RAW_FILES


DOMINANT_CLB_CODE = 45.0
HIGH_POWER_CODE = 67.5
ALTITUDE_BIN_FT = 2_000.0
MINIMUM_BIN_SAMPLES = 20
BOUNDARY_QUANTILES = {"idle": 0.10, "climb": 0.90}
BOUNDARY_CONDITION_FRACTION = 0.20
ALTITUDE_RATE_WINDOW_S = 21
MINIMUM_CLIMB_RATE_FT_S = 5.0
MAXIMUM_DESCENT_RATE_FT_S = -5.0
MAXIMUM_ENGINE_N1_DISAGREEMENT_PERCENT = 3.0
CONTACT_TOLERANCE_N1_PERCENT = 2.0
FIT_NEAR_BEST_FRACTION = 0.02
MAXIMUM_VALIDATION_CLIMB_RMSE_PERCENT = 2.0
MAXIMUM_VALIDATION_IDLE_RMSE_PERCENT = 2.0


@dataclass(frozen=True)
class EnvelopeFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    airborne: np.ndarray
    pressure_altitude_m: np.ndarray
    altitude_rate_ft_s: np.ndarray
    mach: np.ndarray
    static_temperature_k: np.ndarray
    n1_percent: np.ndarray
    n1_engine_difference_percent: np.ndarray
    tla_code: np.ndarray
    flap_code: np.ndarray
    selected_ias_kts: np.ndarray

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
        "--points", type=Path, default=Path("docs/modeling/a320-autothrust-envelope-points.csv")
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope-flight-metrics.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.md"),
    )
    return parser.parse_args()


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    time = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time)
    order = np.argsort(time[keep], kind="stable")
    indices = np.flatnonzero(keep)[order]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[indices]


def _build_flight(
    *,
    registration: str,
    recording: str,
    role: str,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
) -> EnvelopeFlight:
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    current = processed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = current["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
    if np.any(raw_indices >= len(decoded.time_s)) or not np.all(
        decoded.timestamp_utc[raw_indices] == timestamps
    ):
        raise ValueError(f"raw/processed mismatch for {registration}/{recording}")
    time_s = (
        (timestamps - timestamps[0])
        .astype("timedelta64[ns]")
        .astype(np.int64)
        / 1e9
    )
    altitude = decoded.channel("pressure_altitude")[raw_indices]
    cas = decoded.channel("calibrated_airspeed")[raw_indices]
    n1_1 = decoded.channel("n1_1", allow_provisional=True)[raw_indices]
    n1_2 = decoded.channel("n1_2", allow_provisional=True)[raw_indices]
    throttle_1 = decoded.channel("throttle_1", allow_unresolved=True)[raw_indices]
    throttle_2 = decoded.channel("throttle_2", allow_unresolved=True)[raw_indices]
    flap = decoded.channel("flap", allow_unresolved=True)[raw_indices]
    selected_ias_full = bounded_sample_and_hold(
        _ordered_raw(raw, "SPEEDP_SEL"), decoded.time_s, 1.0
    )
    return EnvelopeFlight(
        registration=registration,
        recording=recording,
        role=role,
        time_s=time_s,
        airborne=decoded.airborne[raw_indices],
        pressure_altitude_m=altitude,
        altitude_rate_ft_s=centered_linear_derivative(
            altitude / FT_TO_M, time_s, ALTITUDE_RATE_WINDOW_S
        ),
        mach=np.asarray(aero.cas2mach(cas, altitude), dtype=float),
        static_temperature_k=current["temperature"].to_numpy(dtype=float),
        n1_percent=0.5 * (n1_1 + n1_2),
        n1_engine_difference_percent=np.abs(n1_1 - n1_2),
        tla_code=0.5 * (throttle_1 + throttle_2),
        flap_code=flap,
        selected_ias_kts=selected_ias_full[raw_indices],
    )


def _load_flights(
    selection: pd.DataFrame,
    *,
    raw_dir: Path,
    processed_path: Path,
) -> list[EnvelopeFlight]:
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
    required = list(dict.fromkeys((*raw_columns(), "SPEEDP_SEL")))
    flights: list[EnvelopeFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        recordings = current["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=required,
            filters=[("recording", "in", recordings)],
            engine="pyarrow",
        )
        roles = current.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            flights.append(
                _build_flight(
                    registration=str(registration),
                    recording=recording,
                    role=str(roles[recording]),
                    raw=frame,
                    processed=processed_groups[recording],
                )
            )
    loaded = {flight.recording for flight in flights}
    if loaded != set(identifiers):
        raise ValueError(f"flight load mismatch: {sorted(set(identifiers) - loaded)}")
    return flights


def _base_mask(flight: EnvelopeFlight) -> np.ndarray:
    return (
        flight.airborne
        & np.isclose(flight.tla_code, DOMINANT_CLB_CODE)
        & np.isfinite(
            flight.pressure_altitude_m
            + flight.altitude_rate_ft_s
            + flight.mach
            + flight.static_temperature_k
            + flight.n1_percent
        )
        & (flight.mach >= 0.0)
        & (flight.static_temperature_k > 180.0)
        & (flight.static_temperature_k < 330.0)
        & (
            flight.n1_engine_difference_percent
            <= MAXIMUM_ENGINE_N1_DISAGREEMENT_PERCENT
        )
    )


def _boundary_points(flights: list[EnvelopeFlight], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        base = _base_mask(flight)
        for side, phase_mask in (
            ("climb", flight.altitude_rate_ft_s >= MINIMUM_CLIMB_RATE_FT_S),
            ("idle", flight.altitude_rate_ft_s <= MAXIMUM_DESCENT_RATE_FT_S),
        ):
            mask = base & phase_mask
            altitude_ft = flight.pressure_altitude_m / FT_TO_M
            bins = np.floor(altitude_ft / ALTITUDE_BIN_FT).astype(int)
            for altitude_bin in np.unique(bins[mask]):
                current = mask & (bins == altitude_bin)
                count = int(np.count_nonzero(current))
                if count < MINIMUM_BIN_SAMPLES:
                    continue
                values = flight.n1_percent[current]
                target = float(np.quantile(values, BOUNDARY_QUANTILES[side]))
                condition_quantile = (
                    1.0 - BOUNDARY_CONDITION_FRACTION
                    if side == "climb"
                    else BOUNDARY_CONDITION_FRACTION
                )
                condition_threshold = np.quantile(values, condition_quantile)
                local = current & (
                    (flight.n1_percent >= condition_threshold)
                    if side == "climb"
                    else (flight.n1_percent <= condition_threshold)
                )
                rows.append(
                    {
                        "registration": flight.registration,
                        "recording": flight.recording,
                        "role": flight.role,
                        "split": split,
                        "boundary": side,
                        "altitude_bin_ft": int(altitude_bin * ALTITUDE_BIN_FT),
                        "support_samples": count,
                        "pressure_altitude_m": float(
                            np.median(flight.pressure_altitude_m[local])
                        ),
                        "mach": float(np.median(flight.mach[local])),
                        "static_temperature_k": float(
                            np.median(flight.static_temperature_k[local])
                        ),
                        "target_n1_percent": target,
                    }
                )
    points = pd.DataFrame(rows)
    if points.empty or set(points["boundary"]) != {"idle", "climb"}:
        raise ValueError("both idle and climb boundary points are required")
    return points.sort_values(
        ["split", "registration", "recording", "boundary", "altitude_bin_ft"],
        kind="stable",
    ).reset_index(drop=True)


def _point_design(points: pd.DataFrame, feature_set: str) -> np.ndarray:
    return n1_envelope_features(
        pressure_altitude_m=points["pressure_altitude_m"].to_numpy(dtype=float),
        mach=points["mach"].to_numpy(dtype=float),
        static_temperature_k=points["static_temperature_k"].to_numpy(dtype=float),
        feature_set=feature_set,
    )


def _point_weights(points: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(points), dtype=float)
    for boundary in ("idle", "climb"):
        side = points["boundary"].eq(boundary).to_numpy()
        flights = points.loc[side, "recording"].unique()
        for recording in flights:
            current = side & points["recording"].eq(recording).to_numpy()
            weights[current] = 0.5 / (len(flights) * np.count_nonzero(current))
    return weights


def _fit_envelope(
    points: pd.DataFrame, feature_set: str
) -> tuple[EffectiveN1Envelope, dict[str, float]]:
    coefficients: dict[str, np.ndarray] = {}
    conditions: dict[str, float] = {}
    for boundary in ("idle", "climb"):
        current = points.loc[points["boundary"] == boundary]
        fit = robust_weighted_linear_fit(
            _point_design(current, feature_set),
            current["target_n1_percent"].to_numpy(dtype=float),
            _point_weights(current),
        )
        coefficients[boundary] = fit.coefficients
        conditions[boundary] = float(fit.normal_matrix_condition)
    return (
        EffectiveN1Envelope(
            feature_set=feature_set,
            idle_coefficients=coefficients["idle"],
            climb_coefficients=coefficients["climb"],
        ),
        conditions,
    )


def _predict_points(model: EffectiveN1Envelope, points: pd.DataFrame) -> np.ndarray:
    design = _point_design(points, model.feature_set)
    idle = design @ model.idle_coefficients
    climb = design @ model.climb_coefficients
    return np.where(points["boundary"].eq("idle"), idle, climb)


def _candidate_metrics(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recordings = points["recording"].unique()
    for feature_set, features in N1_ENVELOPE_FEATURE_SETS.items():
        fold_rows: list[dict[str, Any]] = []
        for held_recording in recordings:
            training = points.loc[points["recording"] != held_recording]
            held = points.loc[points["recording"] == held_recording]
            model, conditions = _fit_envelope(training, feature_set)
            prediction = _predict_points(model, held)
            for boundary in ("idle", "climb"):
                side = held["boundary"].eq(boundary).to_numpy()
                residual = (
                    prediction[side]
                    - held.loc[side, "target_n1_percent"].to_numpy(dtype=float)
                )
                fold_rows.append(
                    {
                        "boundary": boundary,
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "bias": float(np.mean(residual)),
                        "condition": conditions[boundary],
                    }
                )
        folds = pd.DataFrame(fold_rows)
        full_model, full_conditions = _fit_envelope(points, feature_set)
        full_design = _point_design(points, feature_set)
        idle = full_design @ full_model.idle_coefficients
        climb = full_design @ full_model.climb_coefficients
        rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(features),
                "lofo_mean_flight_boundary_rmse_percent": float(
                    folds["rmse"].mean()
                ),
                "lofo_idle_mean_rmse_percent": float(
                    folds.loc[folds["boundary"] == "idle", "rmse"].mean()
                ),
                "lofo_climb_mean_rmse_percent": float(
                    folds.loc[folds["boundary"] == "climb", "rmse"].mean()
                ),
                "lofo_mean_absolute_bias_percent": float(folds["bias"].abs().mean()),
                "minimum_full_fit_envelope_width_percent": float(
                    np.min(climb - idle)
                ),
                "median_normal_matrix_condition": float(
                    np.median(list(full_conditions.values()))
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates.loc[
        candidates["minimum_full_fit_envelope_width_percent"] > 0.0
    ]
    if eligible.empty:
        raise ValueError("all envelope candidates cross inside the observed domain")
    # Only the CLB boundary is structurally eligible for selection. A descent
    # bin need not contain fixed-idle operation, so its low quantile mixes modes.
    limit = (1.0 + FIT_NEAR_BEST_FRACTION) * eligible[
        "lofo_climb_mean_rmse_percent"
    ].min()
    return eligible.loc[
        eligible["lofo_climb_mean_rmse_percent"] <= limit
    ].sort_values(
        ["feature_count", "lofo_climb_mean_rmse_percent"],
        kind="stable",
    ).iloc[0]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left + right)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    if np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def _flight_metrics(
    flights: list[EnvelopeFlight],
    points: pd.DataFrame,
    model: EffectiveN1Envelope,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight in flights:
        current_points = points.loc[points["recording"] == flight.recording]
        point_prediction = _predict_points(model, current_points)
        point_target = current_points["target_n1_percent"].to_numpy(dtype=float)
        base = _base_mask(flight)
        idle, climb = model.bounds_percent(
            pressure_altitude_m=flight.pressure_altitude_m[base],
            mach=flight.mach[base],
            static_temperature_k=flight.static_temperature_k[base],
        )
        raw_demand = (flight.n1_percent[base] - idle) / (climb - idle)
        climb_phase = flight.altitude_rate_ft_s[base] >= MINIMUM_CLIMB_RATE_FT_S
        descent_phase = flight.altitude_rate_ft_s[base] <= MAXIMUM_DESCENT_RATE_FT_S
        # Recompute CAS from Mach and pressure altitude for the speed-error
        # diagnostic. It is deliberately not used to fit the boundary model.
        cas_kts = np.asarray(
            aero.mach2cas(flight.mach[base], flight.pressure_altitude_m[base]),
            dtype=float,
        ) / KNOT_TO_MPS
        selected_error = flight.selected_ias_kts[base] - cas_kts
        interior = (raw_demand > 0.05) & (raw_demand < 0.95)
        high_power = flight.airborne & np.isclose(flight.tla_code, HIGH_POWER_CODE)
        idle_points = current_points["boundary"].eq("idle").to_numpy()
        climb_points = current_points["boundary"].eq("climb").to_numpy()
        rows.append(
            {
                "registration": flight.registration,
                "recording": flight.recording,
                "role": flight.role,
                "split": split,
                "boundary_points": len(current_points),
                "idle_boundary_rmse_percent": float(
                    np.sqrt(np.mean((point_prediction[idle_points] - point_target[idle_points]) ** 2))
                ),
                "climb_boundary_rmse_percent": float(
                    np.sqrt(np.mean((point_prediction[climb_points] - point_target[climb_points]) ** 2))
                ),
                "airborne_code45_samples": int(np.count_nonzero(base)),
                "inside_envelope_fraction": float(
                    np.mean((raw_demand >= 0.0) & (raw_demand <= 1.0))
                ),
                "raw_demand_p01": float(np.quantile(raw_demand, 0.01)),
                "raw_demand_p50": float(np.quantile(raw_demand, 0.50)),
                "raw_demand_p99": float(np.quantile(raw_demand, 0.99)),
                "climb_upper_contact_fraction": float(
                    np.mean(
                        np.abs(raw_demand[climb_phase] - 1.0)
                        <= CONTACT_TOLERANCE_N1_PERCENT
                        / (climb[climb_phase] - idle[climb_phase])
                    )
                ),
                "descent_idle_contact_fraction": float(
                    np.mean(
                        np.abs(raw_demand[descent_phase])
                        <= CONTACT_TOLERANCE_N1_PERCENT
                        / (climb[descent_phase] - idle[descent_phase])
                    )
                ),
                "interior_demand_speed_error_correlation": _safe_correlation(
                    raw_demand[interior], selected_error[interior]
                ),
                "high_power_code_samples": int(np.count_nonzero(high_power)),
                "high_power_n1_p95_percent": float(
                    np.quantile(flight.n1_percent[high_power], 0.95)
                    if np.any(high_power)
                    else np.nan
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
        if column.endswith("samples") or column in {
            "feature_count",
            "boundary_points",
            "altitude_bin_ft",
            "support_samples",
        }:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(8)
    return result


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
    audit: dict[str, Any],
) -> None:
    compact_candidates = _round_frame(
        candidates[
            [
                "feature_set",
                "feature_count",
                "lofo_mean_flight_boundary_rmse_percent",
                "lofo_idle_mean_rmse_percent",
                "lofo_climb_mean_rmse_percent",
            ]
        ]
    )
    compact_metrics = _round_frame(
        metrics[
            [
                "registration",
                "recording",
                "split",
                "idle_boundary_rmse_percent",
                "climb_boundary_rmse_percent",
                "inside_envelope_fraction",
                "climb_upper_contact_fraction",
                "descent_idle_contact_fraction",
                "high_power_n1_p95_percent",
            ]
        ]
    )
    fit = metrics.loc[metrics["split"] == "fit_lofo"]
    validation = metrics.loc[metrics["split"] == "validation"]
    high = metrics["high_power_n1_p95_percent"].dropna()
    text = f"""# A320 effective FADEC/autothrust envelope

## Decision

The first identifiable FADEC/autothrust surrogate layer is the effective
condition-dependent **CLB N1 ceiling**. Raw TLA code `45` is retained as the
dominant CLB-detent hypothesis. The same experiment rejects the apparent idle
surface: descent low quantiles mix fixed-idle and variable SPEED/MACH operation
because FMA mode is not recorded.

The selected `{chosen['feature_set']}` schedule has
{int(chosen['feature_count'])} features. Its fit-flight leave-one-complete-flight-
out mean boundary RMSE is
{chosen['lofo_climb_mean_rmse_percent']:.2f} N1 percentage points for CLB.
On the six frozen validation flights, median CLB boundary RMSE is
{validation['climb_boundary_rmse_percent'].median():.2f} points. The idle
diagnostic misses its {MAXIMUM_VALIDATION_IDLE_RMSE_PERCENT:.0f}-point gate at
{validation['idle_boundary_rmse_percent'].median():.2f} points. Therefore only
the CLB ceiling is released as a QAR-supported numerical schedule. This does
not identify Airbus's proprietary FADEC schedules or continuous autothrust law.

## Physical and operational interpretation

FAA A320 material states that the thrust lever establishes the upper thrust
limit during autothrust, while autothrust may command the FADEC without lever
motion. FAA SAFO 07007 separately describes six detents as establishing the
maximum FADEC-computed thrust for ambient conditions. Airbus operational
material documents speed targeting with the levers left at CLB. These sources
define the architecture only; all numerical surfaces below come from this QAR.

The intended executable relation is:

```text
N1_idle(h, M, dT) <= N1_command <= N1_CLB(h, M, dT)
u = (N1_command - N1_idle) / (N1_CLB - N1_idle)

fixed idle:       u = 0
fixed climb:      u = 1
variable SPEED:   0 < u < 1
```

The model class emits commanded N1 and supports all three interfaces. In the
current A320 parameter record, only `N1_CLB` is validated; `N1_idle` must be
supplied from a separate engine prior or used strictly as sensitivity. The
previously implemented N1/N2 actuator and N1-to-effective-thrust map remain
downstream.

## Frozen extraction and selection

- Only raw TLA code `45`, paired-engine agreement within
  {MAXIMUM_ENGINE_N1_DISAGREEMENT_PERCENT:.0f} N1 points, and airborne samples
  are eligible.
- A {ALTITUDE_RATE_WINDOW_S}-second local pressure-altitude slope labels climb
  above {MINIMUM_CLIMB_RATE_FT_S:.0f} ft/s and descent below
  {MAXIMUM_DESCENT_RATE_FT_S:.0f} ft/s.
- Per complete flight and {ALTITUDE_BIN_FT:.0f}-ft altitude bin, the
  {BOUNDARY_QUANTILES['climb']:.0%} climb quantile estimates the CLB boundary
  and the {BOUNDARY_QUANTILES['idle']:.0%} descent quantile tests whether idle
  is similarly observable.
- Candidate selection uses only ten fit flights, complete-flight holdout, and
  the simplest non-crossing model within {FIT_NEAR_BEST_FRACTION:.0%} of the
  best CLB boundary RMSE. Validation flights are loaded only after selection.

{_markdown_table(compact_candidates)}

## Complete-flight results

{_markdown_table(compact_metrics)}

The median climb contact fraction within
±{CONTACT_TOLERANCE_N1_PERCENT:.0f} N1 points of the upper boundary is
{metrics['climb_upper_contact_fraction'].median():.3f}. The corresponding
descent contact fraction with the provisional lower surface is only
{metrics['descent_idle_contact_fraction'].median():.3f}, consistent with mixed
fixed and variable modes. The all-sample inside-envelope fraction is retained
in the table only as a diagnostic and is not a validation score.

## What remains unidentifiable

Raw high-power code `67.5` has per-flight p95 N1 spanning
{high.min():.2f}--{high.max():.2f}% across the panel. Flexible-temperature or
rating inputs are not recorded, so TOGA versus FLEX/MCT schedules are not fit.
The continuous variable-speed demand is also still latent. A normalized N1
position `u` can be computed only after adopting a lower-bound prior; it also
includes FADEC and spool lag and must not be treated as the command itself.

The next identification stage should compare dynamic speed/energy controllers
against this frozen envelope. It must use complete-flight rollout, not only
one-step N1 prediction, and should retain fixed-climb, fixed-idle, and variable-
speed modes separately. FMA/autothrust mode words, flex temperature, bleed
configuration, and engine-health corrections remain unobserved.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected ten fit and six validation flights")

    # Leakage boundary: selection and the final fit happen before validation load.
    fit_flights = _load_flights(
        fit_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    fit_points = _boundary_points(fit_flights, "fit")
    candidates = _candidate_metrics(fit_points)
    chosen = _select_candidate(candidates)
    model, conditions = _fit_envelope(fit_points, str(chosen["feature_set"]))
    fit_metrics = _flight_metrics(fit_flights, fit_points, model, "fit_full")

    # One frozen validation use after every structural decision above.
    validation_flights = _load_flights(
        validation_selection, raw_dir=args.raw_dir, processed_path=args.processed
    )
    validation_points = _boundary_points(validation_flights, "validation")
    validation_metrics = _flight_metrics(
        validation_flights, validation_points, model, "validation"
    )
    points = pd.concat((fit_points, validation_points), ignore_index=True)
    metrics = pd.concat((fit_metrics, validation_metrics), ignore_index=True)

    feature_names = N1_ENVELOPE_FEATURE_SETS[model.feature_set]
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "clb-n1-ceiling-validated-idle-and-controller-unidentified",
        "leakage_control": {
            "fit_flights_loaded_for_structure_selection": [
                flight.key for flight in fit_flights
            ],
            "structure_frozen_before_validation_load": True,
            "validation_flights_loaded_once_after_freeze": [
                flight.key for flight in validation_flights
            ],
        },
        "source_interpretation": {
            "dominant_tla_code": DOMINANT_CLB_CODE,
            "label": "CLB-detent hypothesis",
            "status": "operationally-supported-recorder-code-still-unresolved",
            "published_sources": [
                "faa_a320_final_special_condition_autothrust_upper_limit",
                "faa_safo_07007_a320_fadec_ambient_detents",
                "airbus_a320_autothrust_clb_detent",
            ],
        },
        "model": {
            "output": "validated effective CLB and rejected provisional idle N1 boundaries",
            "feature_set": model.feature_set,
            "feature_order": list(feature_names),
            "idle_coefficients": dict(
                zip(feature_names, model.idle_coefficients.tolist(), strict=True)
            ),
            "climb_coefficients": dict(
                zip(feature_names, model.climb_coefficients.tolist(), strict=True)
            ),
            "normal_matrix_condition": conditions,
            "modes": ["fixed_idle", "fixed_climb", "variable_speed"],
            "mode_parameter_status": {
                "fixed_climb": "QAR-validated effective schedule",
                "fixed_idle": "rejected provisional diagnostic; external prior required",
                "variable_speed": "interface only; continuous controller unidentified",
            },
            "variable_speed_demand": "u in [0,1] requires a separate idle prior",
            "downstream": "N1/N2 actuator then effective thrust map",
        },
        "method": {
            "sample_rate_hz": 1,
            "altitude_rate_window_s": ALTITUDE_RATE_WINDOW_S,
            "minimum_climb_rate_ft_s": MINIMUM_CLIMB_RATE_FT_S,
            "maximum_descent_rate_ft_s": MAXIMUM_DESCENT_RATE_FT_S,
            "altitude_bin_ft": ALTITUDE_BIN_FT,
            "minimum_bin_samples": MINIMUM_BIN_SAMPLES,
            "boundary_quantiles": BOUNDARY_QUANTILES,
            "maximum_engine_n1_disagreement_percent": (
                MAXIMUM_ENGINE_N1_DISAGREEMENT_PERCENT
            ),
            "complete_flight_selection": "leave-one-fit-flight-out",
            "simplicity_rule": (
                f"fewest features within {FIT_NEAR_BEST_FRACTION:.0%} of best "
                "non-crossing CLB candidate"
            ),
        },
        "selection": _native(chosen.to_dict()),
        "aggregate_results": {
            "fit_median_idle_boundary_rmse_percent": float(
                fit_metrics["idle_boundary_rmse_percent"].median()
            ),
            "fit_median_climb_boundary_rmse_percent": float(
                fit_metrics["climb_boundary_rmse_percent"].median()
            ),
            "validation_median_idle_boundary_rmse_percent": float(
                validation_metrics["idle_boundary_rmse_percent"].median()
            ),
            "validation_median_climb_boundary_rmse_percent": float(
                validation_metrics["climb_boundary_rmse_percent"].median()
            ),
            "validation_median_inside_envelope_fraction": float(
                validation_metrics["inside_envelope_fraction"].median()
            ),
        },
        "clb_limit_decision": {
            "passed": bool(
                validation_metrics["climb_boundary_rmse_percent"].median()
                <= MAXIMUM_VALIDATION_CLIMB_RMSE_PERCENT
            ),
            "maximum_validation_median_rmse_percent": (
                MAXIMUM_VALIDATION_CLIMB_RMSE_PERCENT
            ),
            "parameters_released": True,
        },
        "idle_limit_decision": {
            "passed": bool(
                validation_metrics["idle_boundary_rmse_percent"].median()
                <= MAXIMUM_VALIDATION_IDLE_RMSE_PERCENT
            ),
            "maximum_validation_median_rmse_percent": (
                MAXIMUM_VALIDATION_IDLE_RMSE_PERCENT
            ),
            "parameters_released": False,
            "reason": "descent low quantiles mix fixed-idle and variable thrust modes",
        },
        "high_power_limit_decision": {
            "raw_code": HIGH_POWER_CODE,
            "model_fitted": False,
            "reason": (
                "unrecorded FLEX/rating input and insufficient condition coverage"
            ),
        },
        "continuous_controller_decision": {
            "model_fitted": False,
            "observed_proxy": "normalized N1 position after adopting an idle prior",
            "reason": (
                "proxy includes FADEC/spool lag and FMA mode is not decoded"
            ),
            "next_gate": "complete-flight rollout of speed/energy controller candidates",
        },
        "artifacts": {
            "runner": "tools/audit_a320_autothrust_envelope.py",
            "points": str(args.points),
            "candidate_metrics": str(args.candidate_metrics),
            "flight_metrics": str(args.flight_metrics),
            "report": str(args.report),
        },
        "limitations": [
            "TLA and N1 recorder conversions remain unresolved/provisional.",
            "Numerical surfaces are effective QAR envelopes, not CFM56 engine-deck schedules.",
            "Processed static-temperature lineage remains provisional.",
            "Bleeds, anti-ice, engine health, flex temperature, and FMA mode are absent.",
        ],
    }

    args.points.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(points).to_csv(args.points, index=False)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(metrics).to_csv(args.flight_metrics, index=False)
    args.audit.write_text(yaml.safe_dump(_native(audit), sort_keys=False))
    _write_report(
        args.report,
        candidates=candidates,
        chosen=chosen,
        metrics=metrics,
        audit=audit,
    )
    return candidates, metrics, audit


def main() -> None:
    candidates, metrics, audit = run(parse_args())
    print(candidates.to_string(index=False))
    print(metrics.to_string(index=False))
    print(yaml.safe_dump(audit["aggregate_results"], sort_keys=False))


if __name__ == "__main__":
    main()
