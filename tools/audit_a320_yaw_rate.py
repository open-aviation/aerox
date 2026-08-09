#!/usr/bin/env python3
"""Audit A320 body-rate support for effective heading propagation.

The fit-only model predicts five-second true-heading increments from the exact
3-2-1 Euler relation using recorded pitch-rate and yaw-rate codes, effective
pitch/bank angles, and short trajectory-reference delays.  The selected model
is then evaluated on the existing six-flight diagnostic validation panel.  It
is an observed-rate replay adapter audit, not lateral force/moment identification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import load_aircraft
from aerox.flightdata.a320_qar import DEG_TO_RAD, decode_a320_flight, raw_columns
from aerox.modeling.identification import (
    RobustLinearFit,
    delay_samples,
    robust_weighted_linear_fit,
)
from aerox.modeling.dynamics.kinematics import horizontal_distance_m, integrate_wgs84
from tools.calibrate_a320_effective_thrust import (
    RAW_FILES,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    PointMassReplayFlight,
    HorizontalParameters,
    _load_point_mass_flights,
    _markdown_table,
    _native,
    _read_horizontal_parameters,
    _round_frame,
)
from tools.research_support import longest_contiguous_slice as _longest_contiguous_slice

INTEGRATION_HORIZON_S = 5
RESET_WINDOW_S = 300
RATE_DELAY_CANDIDATES_S = tuple(range(8))
MAXIMUM_ABSOLUTE_HEADING_CHANGE_DEG = 30.0
PROCESSED_COLUMNS = ["flight_id", "timestamp"]


@dataclass(frozen=True)
class BankParameters:
    scale_deg_per_code: float
    bias_deg: float
    delay_s: int


@dataclass(frozen=True)
class RateFlight:
    replay: PointMassReplayFlight
    pitch_rate_code_full: np.ndarray
    yaw_rate_code_full: np.ndarray
    lateral_acceleration_mps2_full: np.ndarray

    @property
    def key(self) -> str:
        return self.replay.key

    @property
    def registration(self) -> str:
        return self.replay.base.registration


@dataclass(frozen=True)
class HeadingRateCalibration:
    delay_s: int
    result: RobustLinearFit
    delay_metrics: pd.DataFrame
    leave_one_flight_out: pd.DataFrame

    @property
    def coefficients(self) -> np.ndarray:
        return self.result.coefficients


@dataclass(frozen=True)
class IntervalData:
    flight: RateFlight
    design: np.ndarray
    target_heading_change_deg: np.ndarray


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
        "--pitch-bank-parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--longitudinal-control-parameters",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-control-audit.yml"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-audit.yml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-delay-candidates.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-flight-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-audit.md"),
    )
    return parser.parse_args()


def _read_bank_parameters(path: Path) -> BankParameters:
    bank = yaml.safe_load(path.read_text())["bank"]
    return BankParameters(
        scale_deg_per_code=float(bank["scale_deg_per_code"]),
        bias_deg=float(bank["bias_deg"]),
        delay_s=int(bank["trajectory_reference_delay_s"]),
    )


def _effective_bank_rad(
    flight: PointMassReplayFlight, parameters: BankParameters
) -> np.ndarray:
    code = delay_samples(flight.roll_code_full, parameters.delay_s)[
        flight.base.sample_index
    ]
    return np.deg2rad(parameters.scale_deg_per_code * code + parameters.bias_deg)


def _load_rate_flights(
    selection: pd.DataFrame,
    *,
    raw_dir: Path,
    processed_path: Path,
    vertical: Any,
    horizontal: HorizontalParameters,
    tas_scale: float,
    wing_area_m2: float,
) -> list[RateFlight]:
    replay_flights = _load_point_mass_flights(
        selection,
        raw_dir=raw_dir,
        processed_path=processed_path,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area_m2,
    )
    replay_by_key = {flight.key: flight for flight in replay_flights}
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

    flights: list[RateFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        ids = current["recording"].astype(str).tolist()
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", ids)],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            key = f"{registration}/{recording}"
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=recording,
            )
            timestamps = processed_groups[recording]["timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
            raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
            if np.any(raw_indices >= len(decoded.timestamp_utc)) or not np.all(
                decoded.timestamp_utc[raw_indices] == timestamps
            ):
                raise ValueError(f"raw/processed mismatch for {key}")
            flights.append(
                RateFlight(
                    replay=replay_by_key[key],
                    pitch_rate_code_full=(
                        decoded.channel("pitch_rate", allow_provisional=True)[
                            raw_indices
                        ]
                        / DEG_TO_RAD
                        / 0.5
                    ),
                    yaw_rate_code_full=(
                        decoded.channel("yaw_rate", allow_provisional=True)[raw_indices]
                        / DEG_TO_RAD
                        / 0.5
                    ),
                    lateral_acceleration_mps2_full=decoded.channel(
                        "lateral_acceleration"
                    )[raw_indices],
                )
            )
    if {flight.key for flight in flights} != set(replay_by_key):
        raise ValueError("rate-flight load mismatch")
    return flights


def _integrated_horizon(
    values: np.ndarray, time_s: np.ndarray, horizon: int
) -> np.ndarray:
    interval = 0.5 * (values[1:] + values[:-1]) * np.diff(time_s)
    cumulative = np.concatenate(([0.0], np.cumsum(interval)))
    return cumulative[horizon:] - cumulative[:-horizon]


def _interval_data(
    flight: RateFlight,
    *,
    delay_s: int,
    bank: BankParameters,
) -> IntervalData:
    segment = _longest_contiguous_slice(flight.replay.base.sample_index)
    indices = flight.replay.base.sample_index[segment]
    time = flight.replay.base.time_s[segment]
    heading = np.unwrap(flight.replay.calibrated_heading_rad[segment])
    pitch = flight.replay.effective_pitch_angle_rad[segment]
    bank_angle = _effective_bank_rad(flight.replay, bank)[segment]
    pitch_rate_code = delay_samples(flight.pitch_rate_code_full, delay_s)[indices]
    yaw_rate_code = delay_samples(flight.yaw_rate_code_full, delay_s)[indices]
    pitch_term = pitch_rate_code * np.sin(bank_angle) / np.cos(pitch)
    yaw_term = yaw_rate_code * np.cos(bank_angle) / np.cos(pitch)
    integrated_pitch = _integrated_horizon(pitch_term, time, INTEGRATION_HORIZON_S)
    integrated_yaw = _integrated_horizon(yaw_term, time, INTEGRATION_HORIZON_S)
    heading_change = np.rad2deg(
        heading[INTEGRATION_HORIZON_S:] - heading[:-INTEGRATION_HORIZON_S]
    )
    duration = time[INTEGRATION_HORIZON_S:] - time[:-INTEGRATION_HORIZON_S]
    valid = (
        np.isfinite(integrated_pitch + integrated_yaw + heading_change)
        & np.isclose(duration, INTEGRATION_HORIZON_S)
        & (np.abs(heading_change) < MAXIMUM_ABSOLUTE_HEADING_CHANGE_DEG)
    )
    is_afz = float(flight.registration == "TAIL-B")
    design = np.column_stack(
        (
            np.ones(np.count_nonzero(valid)),
            np.full(np.count_nonzero(valid), is_afz),
            integrated_pitch[valid],
            integrated_yaw[valid],
        )
    )
    return IntervalData(
        flight=flight,
        design=design,
        target_heading_change_deg=heading_change[valid],
    )


def _fit_at_delay(
    flights: list[RateFlight], delay_s: int, bank: BankParameters
) -> RobustLinearFit:
    intervals = [
        _interval_data(flight, delay_s=delay_s, bank=bank) for flight in flights
    ]
    design = np.vstack([item.design for item in intervals])
    target = np.concatenate([item.target_heading_change_deg for item in intervals])
    weights = np.concatenate(
        [
            np.full(
                len(item.target_heading_change_deg),
                1.0 / (len(intervals) * len(item.target_heading_change_deg)),
            )
            for item in intervals
        ]
    )
    return robust_weighted_linear_fit(design, target, weights)


def _interval_metrics(
    interval: IntervalData, coefficients: np.ndarray
) -> dict[str, float]:
    prediction = interval.design @ coefficients
    residual = prediction - interval.target_heading_change_deg
    correlation = float(
        np.corrcoef(prediction, interval.target_heading_change_deg)[0, 1]
    )
    return {
        "interval_count": len(residual),
        "heading_increment_rmse_deg": float(np.sqrt(np.mean(residual**2))),
        "heading_increment_bias_deg": float(np.mean(residual)),
        "heading_increment_correlation": correlation,
    }


def _calibrate(
    flights: list[RateFlight], bank: BankParameters
) -> HeadingRateCalibration:
    fits: dict[int, RobustLinearFit] = {}
    candidate_rows = []
    for delay_s in RATE_DELAY_CANDIDATES_S:
        result = _fit_at_delay(flights, delay_s, bank)
        fits[delay_s] = result
        candidate_rows.append(
            {
                "delay_s": delay_s,
                "huber_objective": result.huber_objective,
                "weighted_rmse_deg": result.base_weighted_rmse,
                "robust_scale_deg": result.robust_scale,
                "normal_matrix_condition": result.normal_matrix_condition,
                "global_five_second_bias_deg": result.coefficients[0],
                "afz_additional_five_second_bias_deg": result.coefficients[1],
                "pitch_rate_scale_deg_s_per_code": result.coefficients[2],
                "yaw_rate_scale_deg_s_per_code": result.coefficients[3],
            }
        )
    selected = min(fits, key=lambda delay: fits[delay].huber_objective)
    result = fits[selected]
    fold_rows = []
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        fold = _fit_at_delay(training, selected, bank)
        interval = _interval_data(held_out, delay_s=selected, bank=bank)
        fold_rows.append(
            {
                "flight": held_out.key,
                **_interval_metrics(interval, fold.coefficients),
                "global_five_second_bias_deg": fold.coefficients[0],
                "afz_additional_five_second_bias_deg": fold.coefficients[1],
                "pitch_rate_scale_deg_s_per_code": fold.coefficients[2],
                "yaw_rate_scale_deg_s_per_code": fold.coefficients[3],
            }
        )
    return HeadingRateCalibration(
        delay_s=selected,
        result=result,
        delay_metrics=pd.DataFrame(candidate_rows),
        leave_one_flight_out=pd.DataFrame(fold_rows),
    )


def _heading_rate_deg_s(
    flight: RateFlight,
    *,
    delay_s: int,
    bank: BankParameters,
    coefficients: np.ndarray,
) -> np.ndarray:
    indices = flight.replay.base.sample_index
    pitch = flight.replay.effective_pitch_angle_rad
    bank_angle = _effective_bank_rad(flight.replay, bank)
    pitch_rate_code = delay_samples(flight.pitch_rate_code_full, delay_s)[indices]
    yaw_rate_code = delay_samples(flight.yaw_rate_code_full, delay_s)[indices]
    global_bias, afz_bias, pitch_scale, yaw_scale = coefficients
    bias_rate = (
        global_bias + afz_bias * float(flight.registration == "TAIL-B")
    ) / INTEGRATION_HORIZON_S
    return (
        bias_rate
        + pitch_scale * pitch_rate_code * np.sin(bank_angle) / np.cos(pitch)
        + yaw_scale * yaw_rate_code * np.cos(bank_angle) / np.cos(pitch)
    )


def _integrate_heading(
    time_s: np.ndarray, initial_heading_rad: float, rate_deg_s: np.ndarray
) -> np.ndarray:
    rate = np.deg2rad(rate_deg_s)
    heading = np.empty(len(time_s), dtype=float)
    heading[0] = initial_heading_rad
    heading[1:] = heading[0] + np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(time_s))
    return heading


def _replay_errors(
    flight: RateFlight,
    selection: slice,
    rate_deg_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    replay = flight.replay
    base = replay.base
    time = base.time_s[selection]
    observed_heading = np.unwrap(replay.calibrated_heading_rad[selection])
    predicted_heading = _integrate_heading(
        time, float(observed_heading[0]), rate_deg_s[selection]
    )
    heading_error = np.arctan2(
        np.sin(predicted_heading - observed_heading),
        np.cos(predicted_heading - observed_heading),
    )
    horizontal_speed = base.true_airspeed_mps[selection] * np.cos(
        base.flight_path_angle_rad[selection]
    )
    predicted = integrate_wgs84(
        time,
        float(replay.latitude_rad[selection][0]),
        float(replay.longitude_rad[selection][0]),
        float(replay.geometric_altitude_m[selection][0]),
        horizontal_speed * np.cos(predicted_heading) + replay.wind_north_mps[selection],
        horizontal_speed * np.sin(predicted_heading) + replay.wind_east_mps[selection],
        -base.true_airspeed_mps[selection]
        * np.sin(base.flight_path_angle_rad[selection]),
    )
    baseline = integrate_wgs84(
        time,
        float(replay.latitude_rad[selection][0]),
        float(replay.longitude_rad[selection][0]),
        float(replay.geometric_altitude_m[selection][0]),
        horizontal_speed * np.cos(observed_heading) + replay.wind_north_mps[selection],
        horizontal_speed * np.sin(observed_heading) + replay.wind_east_mps[selection],
        -base.true_airspeed_mps[selection]
        * np.sin(base.flight_path_angle_rad[selection]),
    )
    predicted_horizontal = horizontal_distance_m(
        predicted.latitude_rad,
        predicted.longitude_rad,
        replay.latitude_rad[selection],
        replay.longitude_rad[selection],
    )
    baseline_horizontal = horizontal_distance_m(
        baseline.latitude_rad,
        baseline.longitude_rad,
        replay.latitude_rad[selection],
        replay.longitude_rad[selection],
    )
    return heading_error, predicted_horizontal, baseline_horizontal


def _flight_metrics(
    flight: RateFlight,
    *,
    split: str,
    calibration: HeadingRateCalibration,
    bank: BankParameters,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    interval = _interval_data(flight, delay_s=calibration.delay_s, bank=bank)
    row: dict[str, Any] = {
        "registration": flight.replay.base.registration,
        "recording": flight.replay.base.recording,
        "role": flight.replay.base.role,
        "split": split,
        **_interval_metrics(interval, coefficients),
    }
    segment = _longest_contiguous_slice(flight.replay.base.sample_index)
    start = int(segment.start or 0)
    stop = int(segment.stop or len(flight.replay.base.time_s))
    rate = _heading_rate_deg_s(
        flight,
        delay_s=calibration.delay_s,
        bank=bank,
        coefficients=coefficients,
    )
    window_heading: list[np.ndarray] = []
    window_horizontal: list[np.ndarray] = []
    window_baseline: list[np.ndarray] = []
    for window_start in range(start, stop - 60, RESET_WINDOW_S):
        window_stop = min(window_start + RESET_WINDOW_S + 1, stop)
        current = slice(window_start, window_stop)
        heading, horizontal, baseline = _replay_errors(flight, current, rate)
        window_heading.append(heading)
        window_horizontal.append(horizontal)
        window_baseline.append(baseline)
    joined_heading = np.concatenate(window_heading)
    joined_horizontal = np.concatenate(window_horizontal)
    joined_baseline = np.concatenate(window_baseline)
    full_heading, full_horizontal, full_baseline = _replay_errors(flight, segment, rate)
    row.update(
        {
            "segment_duration_s": float(
                flight.replay.base.time_s[segment][-1]
                - flight.replay.base.time_s[segment][0]
            ),
            "window_count": len(window_heading),
            "window_heading_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(joined_heading**2)))
            ),
            "window_horizontal_rmse_m": float(np.sqrt(np.mean(joined_horizontal**2))),
            "window_measured_heading_baseline_rmse_m": float(
                np.sqrt(np.mean(joined_baseline**2))
            ),
            "full_heading_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(full_heading**2)))
            ),
            "full_heading_terminal_error_deg": float(np.rad2deg(full_heading[-1])),
            "full_horizontal_rmse_m": float(np.sqrt(np.mean(full_horizontal**2))),
            "full_horizontal_terminal_error_m": float(full_horizontal[-1]),
            "full_measured_heading_baseline_rmse_m": float(
                np.sqrt(np.mean(full_baseline**2))
            ),
        }
    )
    return row


def _coefficient_summary(folds: pd.DataFrame, column: str) -> dict[str, float]:
    values = folds[column].to_numpy(dtype=float)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "coefficient_of_variation": float(values.std() / abs(values.mean())),
    }


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("fit_lofo", "validation"):
        current = metrics.loc[metrics["split"] == split]
        rows.append(
            {
                "split": split,
                "flights": len(current),
                "5s_heading_increment_rmse_deg": current[
                    "heading_increment_rmse_deg"
                ].mean(),
                "300s_heading_rmse_deg": current["window_heading_rmse_deg"].mean(),
                "300s_horizontal_rmse_m": current["window_horizontal_rmse_m"].mean(),
                "300s_measured_heading_baseline_m": current[
                    "window_measured_heading_baseline_rmse_m"
                ].mean(),
                "full_heading_rmse_deg_mean": current["full_heading_rmse_deg"].mean(),
                "full_heading_rmse_deg_median": current[
                    "full_heading_rmse_deg"
                ].median(),
                "full_horizontal_rmse_m_mean": current["full_horizontal_rmse_m"].mean(),
                "full_horizontal_rmse_m_median": current[
                    "full_horizontal_rmse_m"
                ].median(),
                "full_measured_heading_baseline_m_mean": current[
                    "full_measured_heading_baseline_rmse_m"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    calibration: HeadingRateCalibration,
    metrics: pd.DataFrame,
    prior_q_scale: float,
    prior_r_scale: float,
) -> None:
    aggregate = _round_frame(_aggregate(metrics))
    validation = metrics.loc[metrics["split"] == "validation"]
    coefficients = calibration.coefficients
    text = f"""# A320 yaw-rate heading-propagation audit

## Decision

The exact Euler observed-rate adapter is a strong **diagnostic replay** path but
does not release physical body-rate scales or a lateral force/moment model.
Using only the ten calibration flights, the fit selects a
{calibration.delay_s}-second trajectory-reference delay. The frozen model
reaches {validation.heading_increment_rmse_deg.mean():.3f} deg mean five-second
heading-increment RMSE on the six evaluation flights.

The fitted equation is:

```text
delta heading = global five-second bias + registration bias
              + q_scale * integral(q_code sin(bank) / cos(pitch)) dt
              + r_scale * integral(r_code cos(bank) / cos(pitch)) dt
```

The effective scales are {coefficients[2]:.6f} deg/s/code for pitch rate and
{coefficients[3]:.6f} deg/s/code for yaw rate. They materially disagree with the
{prior_q_scale:.6f} and {prior_r_scale:.6f} deg/s/code values inferred earlier
from pitch kinematics. The five-second global intercept is
{coefficients[0]:.6f} deg, or {coefficients[0] / INTEGRATION_HORIZON_S:.6f}
deg/s when distributed uniformly. These discrepancies mean that timing,
heading-reference, rate zero, and Euler-angle lineages remain confounded.

## Replay results

{_markdown_table(aggregate)}

The yaw-rate adapter lowers evaluation-flight 300-second heading RMSE from the
bank-only plant's 2.62 deg to {validation.window_heading_rmse_deg.mean():.3f}
deg. Its 300-second horizontal RMSE is
{validation.window_horizontal_rmse_m.mean():.1f} m, close to the
{validation.window_measured_heading_baseline_rmse_m.mean():.1f} m exact-interval
measured-heading baseline.

Over full segments, median heading and horizontal RMSE are
{validation.full_heading_rmse_deg.median():.3f} deg and
{validation.full_horizontal_rmse_m.median():.1f} m. The means are worse
({validation.full_heading_rmse_deg.mean():.3f} deg and
{validation.full_horizontal_rmse_m.mean():.1f} m) because the longest AFC
evaluation flight retains a large accumulated position error. This remains a
material improvement over bank-only propagation, not a complete-flight release.

## Interpretation

Body rates contain repeatable lateral kinematic information that the current
bank-only coordinated-turn approximation omits. The optional **observed
heading-rate replay adapter** is now implemented. It bypasses the plant's bank-
derived heading equation only in recorded-data replay and emits the implied
lateral-force residual as a diagnostic; see
`a320-observed-heading-rate-replay.md`. It is not a yaw command, flight-control
law, or independently calibrated gyro model.

Trajectory generation should continue to use a physically coordinated bank/
turn relationship until sideslip, side force, yaw dynamics, and rate biases are
identified. Measured heading remains the numerical reference baseline.

## Evidence boundary

- Delay and coefficients are selected on ten calibration flights only.
- The six evaluation flights are held from this fit but had already been opened
  by prior commissioning work; this is not independent confirmation.
- Pitch and bank use the frozen effective attitude conventions.
- Heading, TAS, wind, and coordinates use the frozen horizontal calibration.
- Five-second overlapping intervals provide diagnostic sample density but do
  not create independent observations.
- No raw trajectories or coordinates are written to the repository.

## Figures

![A320 observed yaw-rate feasibility](figures/a320-yaw-rate-audit-overview.png)

Figure 2. Fit-only delay selection, held-flight five-second increment errors,
and evaluation-flight replay comparisons. Circles denote TAIL-A, squares denote
TAIL-B, grey lines connect the same flight, and diamonds show the across-flight
summary. Error bars are mean ± one standard deviation on linear panels and
median with interquartile range on the log panel. Panel B contains ten fit LOFO
and six evaluation points; panels C-D contain the six evaluation flights. A
[vector PDF](figures/a320-yaw-rate-audit-overview.pdf) is provided for detailed
inspection.

## Reproducibility

- Command: `uv run python -m tools.audit_a320_yaw_rate`
- Figures: `uv run python -m tools.plot_a320_modeling_results`
- Parameters: `docs/modeling/a320-yaw-rate-audit.yml`
- Delay candidates: `docs/modeling/a320-yaw-rate-delay-candidates.csv`
- Flight metrics: `docs/modeling/a320-yaw-rate-flight-metrics.csv`
- Source ledger: `docs/modeling/sources.yml`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    aircraft = load_aircraft("A320")
    wing_area = float(aircraft.wing.area)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    bank = _read_bank_parameters(args.pitch_bank_parameters)

    fit_flights = _load_rate_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    calibration = _calibrate(fit_flights, bank)
    fold_coefficients = {
        row.flight: np.asarray(
            [
                row.global_five_second_bias_deg,
                row.afz_additional_five_second_bias_deg,
                row.pitch_rate_scale_deg_s_per_code,
                row.yaw_rate_scale_deg_s_per_code,
            ]
        )
        for row in calibration.leave_one_flight_out.itertuples(index=False)
    }

    # Load the diagnostic evaluation flights only after delay/structure freeze.
    validation_flights = _load_rate_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area,
    )
    rows = [
        _flight_metrics(
            flight,
            split="fit_lofo",
            calibration=calibration,
            bank=bank,
            coefficients=fold_coefficients[flight.key],
        )
        for flight in fit_flights
    ]
    rows.extend(
        _flight_metrics(
            flight,
            split="validation",
            calibration=calibration,
            bank=bank,
            coefficients=calibration.coefficients,
        )
        for flight in validation_flights
    )
    metrics = pd.DataFrame(rows).sort_values(
        ["split", "registration", "recording"], kind="stable"
    )

    prior = yaml.safe_load(args.longitudinal_control_parameters.read_text())
    prior_model = prior["model"]
    prior_q_scale = float(
        prior_model[
            "conditional_effective_pitch_rate_scale_from_vertical_deg_s_per_code"
        ]
    )
    prior_r_scale = float(
        prior_model["conditional_effective_yaw_rate_scale_from_vertical_deg_s_per_code"]
    )
    folds = calibration.leave_one_flight_out
    validation = metrics.loc[metrics["split"] == "validation"]
    support = calibration.delay_metrics.loc[
        calibration.delay_metrics["huber_objective"]
        <= 1.01 * calibration.delay_metrics["huber_objective"].min(),
        "delay_s",
    ].tolist()
    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "effective-observed-rate-replay-supported-physical-scales-rejected",
        "model": {
            "target": "five-second calibrated true-heading increment",
            "equation": "delta_psi = c_global + c_registration + sq integral(q_code sin(bank)/cos(pitch)) + sr integral(r_code cos(bank)/cos(pitch))",
            "feature_order": [
                "global_five_second_bias_deg",
                "TAIL-B_additional_five_second_bias_deg",
                "integrated_pitch_rate_code_sin_bank_over_cos_pitch",
                "integrated_yaw_rate_code_cos_bank_over_cos_pitch",
            ],
            "coefficients": {
                "global_five_second_bias_deg": calibration.coefficients[0],
                "TAIL-B_additional_five_second_bias_deg": calibration.coefficients[1],
                "pitch_rate_scale_deg_s_per_code": calibration.coefficients[2],
                "yaw_rate_scale_deg_s_per_code": calibration.coefficients[3],
            },
            "trajectory_reference_delay_s": calibration.delay_s,
            "one_percent_objective_support_s": support,
            "integration_horizon_s": INTEGRATION_HORIZON_S,
            "fit_weighted_rmse_deg": calibration.result.base_weighted_rmse,
            "fit_robust_scale_deg": calibration.result.robust_scale,
            "normal_matrix_condition": calibration.result.normal_matrix_condition,
        },
        "cross_validation": {
            "pitch_rate_scale": _coefficient_summary(
                folds, "pitch_rate_scale_deg_s_per_code"
            ),
            "yaw_rate_scale": _coefficient_summary(
                folds, "yaw_rate_scale_deg_s_per_code"
            ),
            "mean_fit_lofo_heading_increment_rmse_deg": float(
                folds.heading_increment_rmse_deg.mean()
            ),
        },
        "consistency_check": {
            "earlier_pitch_kinematic_pitch_rate_scale_deg_s_per_code": prior_q_scale,
            "earlier_pitch_kinematic_yaw_rate_scale_deg_s_per_code": prior_r_scale,
            "current_to_prior_pitch_rate_scale_ratio": float(
                calibration.coefficients[2] / prior_q_scale
            ),
            "current_to_prior_yaw_rate_scale_ratio": float(
                calibration.coefficients[3] / prior_r_scale
            ),
            "physical_rate_scale_release": False,
        },
        "evaluation": {
            "flight_count": len(validation),
            "mean_five_second_heading_increment_rmse_deg": float(
                validation.heading_increment_rmse_deg.mean()
            ),
            "mean_300s_heading_rmse_deg": float(
                validation.window_heading_rmse_deg.mean()
            ),
            "mean_300s_horizontal_rmse_m": float(
                validation.window_horizontal_rmse_m.mean()
            ),
            "mean_full_heading_rmse_deg": float(
                validation.full_heading_rmse_deg.mean()
            ),
            "median_full_heading_rmse_deg": float(
                validation.full_heading_rmse_deg.median()
            ),
            "mean_full_horizontal_rmse_m": float(
                validation.full_horizontal_rmse_m.mean()
            ),
            "median_full_horizontal_rmse_m": float(
                validation.full_horizontal_rmse_m.median()
            ),
        },
        "decision": {
            "observed_heading_rate_replay_adapter": "supported-diagnostic-only",
            "replace_bank_as_physical_turn_control": False,
            "release_yaw_command_or_yaw_dynamics": False,
            "next": "implement optional replay-only observed heading-rate input and report implied lateral-force residual",
        },
        "split": {
            "fit_flights": [flight.key for flight in fit_flights],
            "evaluation_flights": [flight.key for flight in validation_flights],
            "evaluation_status": "held from this fit but opened in earlier commissioning",
        },
        "dependencies": {
            "vertical": str(args.vertical_parameters),
            "horizontal": str(args.horizontal_parameters),
            "effective_attitude": str(args.pitch_bank_parameters),
            "prior_rate_audit": str(args.longitudinal_control_parameters),
        },
        "limitations": [
            "Heading-reference delay, body-rate delay, rate zero, and scale are not separately identifiable.",
            "Pitch-rate excitation in the heading equation is weak at low bank.",
            "Overlapping five-second intervals are not independent observations.",
            "No sideslip, side force, rudder, yaw moment, or inertia is identified.",
            "Evaluation flights are diagnostic rather than newly untouched confirmation data.",
        ],
    }

    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(_native(parameters), sort_keys=False))
    _round_frame(calibration.delay_metrics).to_csv(args.candidates, index=False)
    _round_frame(metrics).to_csv(args.metrics, index=False)
    _write_report(
        args.report,
        calibration=calibration,
        metrics=metrics,
        prior_q_scale=prior_q_scale,
        prior_r_scale=prior_r_scale,
    )
    return metrics, parameters


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
