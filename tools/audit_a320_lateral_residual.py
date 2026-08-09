#!/usr/bin/env python3
"""Test A320 turn-force residual attribution on lateral acceleration and beta proxy.

The audit uses only calibration flights for delay and structure selection, then
evaluates frozen candidates on the existing six-flight diagnostic panel. The
coordinate/wind-derived horizontal sideslip is explicitly a proxy because the
QAR contains no direct sideslip measurement.
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
from aerox.modeling.identification import (
    RobustLinearFit,
    delay_samples,
    robust_weighted_linear_fit,
)
from aerox.modeling.dynamics.kinematics import differentiate_wgs84_intervals
from aerox.models.atmosphere import G0
from tools.audit_a320_yaw_rate import (
    BankParameters,
    RateFlight,
    _effective_bank_rad,
    _heading_rate_deg_s,
    _load_rate_flights,
    _read_bank_parameters,
)
from tools.calibrate_a320_effective_thrust import (
    RAW_FILES,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    _markdown_table,
    _native,
    _read_fuel_model,
    _read_horizontal_parameters,
    _read_thrust_map,
    _round_frame,
)
from tools.research_support import (
    RESET_WINDOW_S,
    longest_contiguous_slice as _longest_contiguous_slice,
)
from tools.replay_a320_observed_heading_rate import (
    ObservedRateReplayJob,
    _read_lift_model,
    _read_yaw_model,
    _simulate_slice,
    _trapezoidal_equivalent_zoh,
)

LATACC_DELAY_CANDIDATES_S = tuple(range(8))
MODEL_NAMES = ("bias_only", "lateral_acceleration", "beta_proxy", "combined")
MODEL_COMPLEXITY = {
    "bias_only": 2,
    "lateral_acceleration": 3,
    "beta_proxy": 3,
    "combined": 4,
}
FIT_THINNING_S = 5
TURN_EXCITATION_BANK_DEG = 3.0
MAXIMUM_ABSOLUTE_BETA_PROXY_DEG = 10.0
MAXIMUM_ABSOLUTE_LATERAL_ACCELERATION_G = 0.3
SIMPLICITY_TOLERANCE = 0.01
MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT = 0.05
MINIMUM_EVALUATION_FLIGHTS_IMPROVED = 5
MAXIMUM_LOFO_COEFFICIENT_OF_VARIATION = 0.25
MINIMUM_UNIT_MAGNITUDE_SCALE = 0.5
MAXIMUM_UNIT_MAGNITUDE_SCALE = 1.5


@dataclass(frozen=True)
class AttributionJob:
    rate_flight: RateFlight
    replay_job: ObservedRateReplayJob
    delayed_lateral_acceleration_mps2: np.ndarray


@dataclass(frozen=True)
class AttributionSamples:
    registration: str
    recording: str
    role: str
    sample_index: np.ndarray
    residual_weight_fraction: np.ndarray
    lateral_acceleration_basis: np.ndarray
    beta_force_basis: np.ndarray
    beta_proxy_rad: np.ndarray
    turn_excited: np.ndarray

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
        "--pitch-bank-parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--yaw-parameters",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-audit.yml"),
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
        default=Path("docs/modeling/a320-lateral-residual-audit.yml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("docs/modeling/a320-lateral-residual-candidates.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-lateral-residual-flight-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-lateral-residual-audit.md"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="flight-level worker processes; use 1 for serial debugging",
    )
    return parser.parse_args()


def _intervals_to_nodes(values: np.ndarray) -> np.ndarray:
    nodes = np.empty(len(values) + 1, dtype=float)
    nodes[0] = values[0]
    nodes[-1] = values[-1]
    nodes[1:-1] = 0.5 * (values[:-1] + values[1:])
    return nodes


def _sideslip_target_inventory(raw_dir: Path, panel: pd.DataFrame) -> dict[str, Any]:
    rows = 0
    finite = 0
    values: set[float] = set()
    for registration, selection in panel.groupby("registration", sort=True):
        identifiers = selection["recording"].astype(str).tolist()
        frame = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=["recording", "SIDESLIP_TGT"],
            filters=[("recording", "in", identifiers)],
            engine="pyarrow",
        )
        rows += len(frame)
        current = frame["SIDESLIP_TGT"].dropna().to_numpy(dtype=float)
        finite += len(current)
        values.update(np.unique(current).tolist())
    ordered = sorted(values)
    return {
        "raw_channel": "SIDESLIP_TGT",
        "role": "target-not-measurement",
        "panel_rows": rows,
        "finite_samples": finite,
        "unique_values": ordered,
        "all_finite_samples_zero": bool(ordered == [0.0]),
        "usable_for_identification": False,
    }


def _beta_proxy(job: AttributionJob, selection: slice) -> np.ndarray:
    flight = job.rate_flight.replay
    time = flight.base.time_s[selection]
    north_ground, east_ground, _ = differentiate_wgs84_intervals(
        time,
        flight.latitude_rad[selection],
        flight.longitude_rad[selection],
        flight.geometric_altitude_m[selection],
    )
    north_air = _intervals_to_nodes(north_ground) - flight.wind_north_mps[selection]
    east_air = _intervals_to_nodes(east_ground) - flight.wind_east_mps[selection]
    air_direction = np.arctan2(east_air, north_air)
    heading = flight.calibrated_heading_rad[selection]
    return np.arctan2(
        np.sin(air_direction - heading),
        np.cos(air_direction - heading),
    )


def _extract_samples(job: AttributionJob) -> AttributionSamples:
    flight = job.rate_flight.replay
    base = flight.base
    segment = _longest_contiguous_slice(base.sample_index)
    start = int(segment.start or 0)
    stop = int(segment.stop or len(base.time_s))
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_index",
            "target",
            "lateral",
            "beta_basis",
            "beta",
            "turn",
        )
    }
    for window_start in range(start, stop - 60, RESET_WINDOW_S):
        window_stop = min(window_start + RESET_WINDOW_S + 1, stop)
        current = slice(window_start, window_stop)
        replay = _simulate_slice(job.replay_job, current)
        bank = replay.trajectory.bank_angle_rad
        mass = replay.trajectory.mass_kg
        beta = _beta_proxy(job, current)
        target = replay.implied_lateral_force_residual_n / (mass * G0)
        lateral_basis = (
            job.delayed_lateral_acceleration_mps2[current] * np.cos(bank)[:, None] / G0
        )
        beta_basis = base.parasite_drag_basis_n[current] * beta / (mass * G0)
        valid = (
            np.isfinite(target + beta + beta_basis)
            & np.all(np.isfinite(lateral_basis), axis=1)
            & (np.abs(beta) <= np.deg2rad(MAXIMUM_ABSOLUTE_BETA_PROXY_DEG))
            & np.all(
                np.abs(lateral_basis) <= MAXIMUM_ABSOLUTE_LATERAL_ACCELERATION_G,
                axis=1,
            )
        )
        arrays["sample_index"].append(base.sample_index[current][valid])
        arrays["target"].append(target[valid])
        arrays["lateral"].append(lateral_basis[valid])
        arrays["beta_basis"].append(beta_basis[valid])
        arrays["beta"].append(beta[valid])
        arrays["turn"].append(
            np.abs(bank[valid]) >= np.deg2rad(TURN_EXCITATION_BANK_DEG)
        )
    if not arrays["target"]:
        raise ValueError(f"no usable attribution windows for {flight.key}")
    return AttributionSamples(
        registration=base.registration,
        recording=base.recording,
        role=base.role,
        sample_index=np.concatenate(arrays["sample_index"]),
        residual_weight_fraction=np.concatenate(arrays["target"]),
        lateral_acceleration_basis=np.vstack(arrays["lateral"]),
        beta_force_basis=np.concatenate(arrays["beta_basis"]),
        beta_proxy_rad=np.concatenate(arrays["beta"]),
        turn_excited=np.concatenate(arrays["turn"]),
    )


def _design(
    samples: AttributionSamples,
    model: str,
    delay_s: int,
) -> np.ndarray:
    columns = [
        np.ones(len(samples.sample_index)),
        np.full(len(samples.sample_index), samples.registration == "TAIL-B"),
    ]
    if model in {"lateral_acceleration", "combined"}:
        columns.append(samples.lateral_acceleration_basis[:, delay_s])
    if model in {"beta_proxy", "combined"}:
        columns.append(samples.beta_force_basis)
    return np.column_stack(columns)


def _fit(
    samples: list[AttributionSamples],
    model: str,
    delay_s: int,
) -> RobustLinearFit:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in samples:
        thin = flight.sample_index % FIT_THINNING_S == 0
        design = _design(flight, model, delay_s)[thin]
        target = flight.residual_weight_fraction[thin]
        valid = np.isfinite(target + design.sum(axis=1))
        designs.append(design[valid])
        targets.append(target[valid])
        weights.append(
            np.full(
                np.count_nonzero(valid),
                1.0 / (len(samples) * np.count_nonzero(valid)),
            )
        )
    return robust_weighted_linear_fit(
        np.vstack(designs),
        np.concatenate(targets),
        np.concatenate(weights),
    )


def _prediction_metrics(
    samples: AttributionSamples,
    model: str,
    delay_s: int,
    coefficients: np.ndarray,
) -> dict[str, float]:
    target = samples.residual_weight_fraction
    prediction = _design(samples, model, delay_s) @ coefficients
    valid = np.isfinite(target + prediction)
    residual = target[valid] - prediction[valid]
    target_valid = target[valid]
    correlation = (
        float(np.corrcoef(target_valid, prediction[valid])[0, 1])
        if np.std(prediction[valid]) > 0.0
        else np.nan
    )
    turn = samples.turn_excited[valid]
    turn_residual = residual[turn]
    turn_target = target_valid[turn]
    return {
        "sample_count": int(np.count_nonzero(valid)),
        "turn_excited_sample_count": int(np.count_nonzero(turn)),
        "target_rmse_weight_fraction": float(np.sqrt(np.mean(target_valid**2))),
        "residual_rmse_weight_fraction": float(np.sqrt(np.mean(residual**2))),
        "residual_bias_weight_fraction": float(np.mean(residual)),
        "prediction_correlation": correlation,
        "turn_target_rmse_weight_fraction": float(np.sqrt(np.mean(turn_target**2))),
        "turn_residual_rmse_weight_fraction": float(np.sqrt(np.mean(turn_residual**2))),
    }


def _lofo(
    samples: list[AttributionSamples],
    model: str,
    delay_s: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    coefficients: dict[str, np.ndarray] = {}
    for held_out in samples:
        training = [flight for flight in samples if flight.key != held_out.key]
        result = _fit(training, model, delay_s)
        coefficients[held_out.key] = result.coefficients
        rows.append(
            {
                "flight": held_out.key,
                **_prediction_metrics(
                    held_out,
                    model,
                    delay_s,
                    result.coefficients,
                ),
            }
        )
    return pd.DataFrame(rows), coefficients


def _coefficient_summary(values: np.ndarray) -> dict[str, float | bool]:
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "coefficient_of_variation": float(values.std() / abs(values.mean())),
        "sign_consistent": bool(np.all(np.sign(values) == np.sign(values[0]))),
    }


def _improvement(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / baseline)


def _flight_rows(
    samples: list[AttributionSamples],
    *,
    split: str,
    delay_s: int,
    coefficients: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for flight in samples:
        row: dict[str, Any] = {
            "registration": flight.registration,
            "recording": flight.recording,
            "role": flight.role,
            "split": split,
            "beta_proxy_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(flight.beta_proxy_rad**2)))
            ),
            "beta_proxy_p95_abs_deg": float(
                np.rad2deg(np.quantile(np.abs(flight.beta_proxy_rad), 0.95))
            ),
        }
        for model in MODEL_NAMES:
            metrics = _prediction_metrics(
                flight,
                model,
                delay_s,
                coefficients[model][flight.key],
            )
            for name, value in metrics.items():
                row[f"{model}_{name}"] = value
        baseline = row["bias_only_residual_rmse_weight_fraction"]
        baseline_turn = row["bias_only_turn_residual_rmse_weight_fraction"]
        for model in MODEL_NAMES[1:]:
            row[f"{model}_rmse_improvement"] = _improvement(
                baseline,
                row[f"{model}_residual_rmse_weight_fraction"],
            )
            row[f"{model}_turn_rmse_improvement"] = _improvement(
                baseline_turn,
                row[f"{model}_turn_residual_rmse_weight_fraction"],
            )
        row["combined_incremental_over_latacc"] = _improvement(
            row["lateral_acceleration_residual_rmse_weight_fraction"],
            row["combined_residual_rmse_weight_fraction"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_jobs(
    selection: pd.DataFrame,
    *,
    split: str,
    args: argparse.Namespace,
    vertical: Any,
    horizontal: Any,
    tas_scale: float,
    bank: BankParameters,
    yaw_delay_s: int,
    yaw_coefficients: np.ndarray,
    wing_area_m2: float,
    lift_model: Any,
    thrust_map: Any,
    fuel_model: Any,
    cd0: float,
    induced_drag_factor: float,
) -> list[AttributionJob]:
    rate_flights = _load_rate_flights(
        selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=wing_area_m2,
    )
    jobs = []
    for flight in rate_flights:
        base_indices = flight.replay.base.sample_index
        delayed_acceleration = np.column_stack(
            [
                delay_samples(flight.lateral_acceleration_mps2_full, delay_s)[
                    base_indices
                ]
                for delay_s in LATACC_DELAY_CANDIDATES_S
            ]
        )
        replay_job = ObservedRateReplayJob(
            flight=flight.replay,
            split=split,
            heading_rate_rad_s=_trapezoidal_equivalent_zoh(
                np.deg2rad(
                    _heading_rate_deg_s(
                        flight,
                        delay_s=yaw_delay_s,
                        bank=bank,
                        coefficients=yaw_coefficients,
                    )
                )
            ),
            bank_angle_rad=_effective_bank_rad(flight.replay, bank),
            lift_model=lift_model,
            thrust_map=thrust_map,
            fuel_model=fuel_model,
            wing_area_m2=wing_area_m2,
            cd0=cd0,
            induced_drag_factor=induced_drag_factor,
        )
        jobs.append(
            AttributionJob(
                rate_flight=flight,
                replay_job=replay_job,
                delayed_lateral_acceleration_mps2=delayed_acceleration,
            )
        )
    return jobs


def _extract_jobs(jobs: list[AttributionJob], workers: int) -> list[AttributionSamples]:
    samples: list[AttributionSamples] = []
    if workers == 1:
        for job in jobs:
            samples.append(_extract_samples(job))
            print(f"extracted {job.replay_job.split} {job.rate_flight.key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_samples, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                samples.append(future.result())
                print(
                    f"extracted {job.replay_job.split} {job.rate_flight.key}",
                    flush=True,
                )
    return sorted(samples, key=lambda flight: (flight.registration, flight.recording))


def _write_report(
    path: Path,
    *,
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    parameters: dict[str, Any],
) -> None:
    validation = metrics.loc[metrics["split"] == "validation"]
    selected = parameters["selection"]["selected_model"]
    decision = parameters["decision"]
    compact = candidates[
        [
            "candidate",
            "coefficient_count",
            "fit_weighted_rmse_weight_fraction",
            "fit_lofo_mean_rmse_weight_fraction",
            "fit_lofo_improvement_vs_bias_only",
        ]
    ]
    text = f"""# A320 lateral-residual attribution audit

## Decision

The QAR has a verified-unit lateral-acceleration channel but no direct sideslip
measurement. `SIDESLIP_TGT` is present, but it is a target rather than an actual
measurement and is identically zero on all sixteen panel flights. The selected
fit-only diagnostic structure is `{selected}` at a
{parameters["selection"]["lateral_acceleration_delay_s"]}-second lateral-
acceleration alignment.

Direct lateral acceleration {"passes" if decision["lateral_acceleration_explains_residual"] else "does not pass"}
the frozen attribution gate. The coordinate/wind-derived beta proxy
{"adds repeatable predictive information" if decision["beta_proxy_adds_predictive_information"] else "does not add repeatable predictive information"},
but it cannot release a physical sideslip derivative because heading rotation,
wind error, and mean sideslip remain confounded.

## Candidate comparison

{_markdown_table(_round_frame(compact))}

On evaluation flights, the lateral-acceleration candidate improves residual
RMSE by {100.0 * validation.lateral_acceleration_rmse_improvement.mean():.2f}%
on average and improves
{int((validation.lateral_acceleration_rmse_improvement > 0.0).sum())}/6 flights.
On the bank-excited subset it changes RMSE by
{100.0 * validation.lateral_acceleration_turn_rmse_improvement.mean():.2f}% on
average and improves
{int((validation.lateral_acceleration_turn_rmse_improvement > 0.0).sum())}/6
flights.
The combined candidate adds
{100.0 * validation.combined_incremental_over_latacc.mean():.2f}% mean
incremental improvement over lateral acceleration alone.

## Interpretation

The force target and features are normalized by instantaneous aircraft weight:

```text
target       = [m V cos(gamma) observed_heading_rate - L sin(bank)] / (m g)
latacc basis = measured_lateral_acceleration cos(bank) / g
beta basis   = q S beta_proxy / (m g)
```

`beta_proxy` is the horizontal air-vector direction obtained from differentiated
coordinates minus frozen wind, relative to calibrated heading. It is useful for
testing residual structure, not a calibrated sideslip vane. Lateral acceleration
is a measured specific-force channel, but sensor-axis alignment, sign, bias, and
its precise relationship to wind-axis side force still require interpretation.

## Evidence boundary

- Delay and candidate structure are selected on ten calibration flights only.
- Candidate performance on calibration flights is leave-one-flight-out.
- The six evaluation flights are loaded only after the delay and structure are
  frozen; they were opened in prior commissioning, so the result is diagnostic.
- Each flight receives equal fit weight; samples are thinned to one every
  {FIT_THINNING_S} seconds.
- The beta proxy is limited to ±{MAXIMUM_ABSOLUTE_BETA_PROXY_DEG:.0f} deg and
  lateral acceleration to ±{MAXIMUM_ABSOLUTE_LATERAL_ACCELERATION_G:.1f} g.
- No raw sample, coordinate, or trajectory is written to the repository.

## Figures

![A320 lateral-residual attribution](figures/a320-lateral-residual-audit-overview.png)

Figure 4. Fit-only delay and candidate selection, evaluation-flight residual
improvements, and beta-proxy coverage. Circles denote TAIL-A, squares denote
TAIL-B, and diamonds with error bars in panels B-C show mean ± one standard
deviation. Panel D shows individual flights only. A
[vector PDF](figures/a320-lateral-residual-audit-overview.pdf) is provided. The
dashed line in panels B-C is the predeclared 5% mean-improvement gate; panel B
contains ten calibration LOFO flights and panels C-D contain six evaluation
flights.

## Reproducibility

- Command: `uv run python -m tools.audit_a320_lateral_residual --workers 32`
- Parameters: `docs/modeling/a320-lateral-residual-audit.yml`
- Candidates: `docs/modeling/a320-lateral-residual-candidates.csv`
- Flight metrics: `docs/modeling/a320-lateral-residual-flight-metrics.csv`
- Figures: `uv run python -m tools.plot_a320_modeling_results`
- Source ledger: `docs/modeling/sources.yml`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    sideslip_target = _sideslip_target_inventory(args.raw_dir, panel)
    fit_selection = panel.loc[panel["role"] != "validation"]
    validation_selection = panel.loc[panel["role"] == "validation"]
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    wing_area = float(aircraft.wing.area)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    bank = _read_bank_parameters(args.pitch_bank_parameters)
    yaw_delay_s, yaw_coefficients = _read_yaw_model(args.yaw_parameters)
    lift_model = _read_lift_model(args.pitch_bank_parameters)
    thrust_map = _read_thrust_map(args.thrust_parameters)
    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    shared = {
        "args": args,
        "vertical": vertical,
        "horizontal": horizontal,
        "tas_scale": tas_scale,
        "bank": bank,
        "yaw_delay_s": yaw_delay_s,
        "yaw_coefficients": yaw_coefficients,
        "wing_area_m2": wing_area,
        "lift_model": lift_model,
        "thrust_map": thrust_map,
        "fuel_model": fuel_model,
        "cd0": reference_drag.zero_lift_drag_coefficient,
        "induced_drag_factor": reference_drag.induced_drag_factor,
    }

    fit_jobs = _build_jobs(fit_selection, split="calibration", **shared)
    fit_samples = _extract_jobs(fit_jobs, args.workers)

    delay_rows = []
    for delay_s in LATACC_DELAY_CANDIDATES_S:
        result = _fit(fit_samples, "lateral_acceleration", delay_s)
        lofo, _ = _lofo(fit_samples, "lateral_acceleration", delay_s)
        delay_rows.append(
            {
                "delay_s": delay_s,
                "huber_objective": result.huber_objective,
                "fit_weighted_rmse_weight_fraction": result.base_weighted_rmse,
                "fit_lofo_mean_rmse_weight_fraction": lofo[
                    "residual_rmse_weight_fraction"
                ].mean(),
                "lateral_acceleration_scale": result.coefficients[2],
            }
        )
    delay_metrics = pd.DataFrame(delay_rows)
    selected_delay = int(
        delay_metrics.loc[
            delay_metrics["fit_lofo_mean_rmse_weight_fraction"].idxmin(),
            "delay_s",
        ]
    )

    full_fits: dict[str, RobustLinearFit] = {}
    fold_metrics: dict[str, pd.DataFrame] = {}
    fold_coefficients: dict[str, dict[str, np.ndarray]] = {}
    candidate_rows = []
    baseline_lofo_rmse = np.nan
    for model in MODEL_NAMES:
        full_fits[model] = _fit(fit_samples, model, selected_delay)
        fold_metrics[model], fold_coefficients[model] = _lofo(
            fit_samples,
            model,
            selected_delay,
        )
        lofo_rmse = float(fold_metrics[model]["residual_rmse_weight_fraction"].mean())
        if model == "bias_only":
            baseline_lofo_rmse = lofo_rmse
        candidate_rows.append(
            {
                "candidate": model,
                "coefficient_count": MODEL_COMPLEXITY[model],
                "fit_weighted_rmse_weight_fraction": full_fits[
                    model
                ].base_weighted_rmse,
                "fit_lofo_mean_rmse_weight_fraction": lofo_rmse,
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    candidates["fit_lofo_improvement_vs_bias_only"] = 1.0 - (
        candidates["fit_lofo_mean_rmse_weight_fraction"] / baseline_lofo_rmse
    )
    best_rmse = float(candidates.fit_lofo_mean_rmse_weight_fraction.min())
    eligible = candidates.loc[
        candidates.fit_lofo_mean_rmse_weight_fraction
        <= (1.0 + SIMPLICITY_TOLERANCE) * best_rmse
    ].sort_values(["coefficient_count", "fit_lofo_mean_rmse_weight_fraction"])
    selected_model = str(eligible.iloc[0].candidate)

    # Freeze delay, structure, and all coefficients before loading evaluation.
    validation_jobs = _build_jobs(
        validation_selection,
        split="validation",
        **shared,
    )
    validation_samples = _extract_jobs(validation_jobs, args.workers)
    validation_coefficients = {
        model: {
            flight.key: full_fits[model].coefficients for flight in validation_samples
        }
        for model in MODEL_NAMES
    }
    fit_rows = _flight_rows(
        fit_samples,
        split="fit_lofo",
        delay_s=selected_delay,
        coefficients=fold_coefficients,
    )
    validation_rows = _flight_rows(
        validation_samples,
        split="validation",
        delay_s=selected_delay,
        coefficients=validation_coefficients,
    )
    metrics = pd.concat([fit_rows, validation_rows], ignore_index=True)

    latacc_fold_scales = np.asarray(
        [values[2] for values in fold_coefficients["lateral_acceleration"].values()]
    )
    beta_fold_scales = np.asarray(
        [values[-1] for values in fold_coefficients["combined"].values()]
    )
    latacc_summary = _coefficient_summary(latacc_fold_scales)
    beta_summary = _coefficient_summary(beta_fold_scales)
    latacc_fit_improvement = float(
        fit_rows.lateral_acceleration_rmse_improvement.mean()
    )
    latacc_validation_improvement = float(
        validation_rows.lateral_acceleration_rmse_improvement.mean()
    )
    latacc_scale = float(full_fits["lateral_acceleration"].coefficients[2])
    lateral_gate = bool(
        latacc_fit_improvement >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and latacc_validation_improvement >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and int((validation_rows.lateral_acceleration_rmse_improvement > 0.0).sum())
        >= MINIMUM_EVALUATION_FLIGHTS_IMPROVED
        and bool(latacc_summary["sign_consistent"])
        and float(latacc_summary["coefficient_of_variation"])
        <= MAXIMUM_LOFO_COEFFICIENT_OF_VARIATION
        and MINIMUM_UNIT_MAGNITUDE_SCALE
        <= abs(latacc_scale)
        <= MAXIMUM_UNIT_MAGNITUDE_SCALE
    )
    beta_fit_increment = float(fit_rows.combined_incremental_over_latacc.mean())
    beta_validation_increment = float(
        validation_rows.combined_incremental_over_latacc.mean()
    )
    beta_predictive = bool(
        beta_fit_increment >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and beta_validation_increment >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and int((validation_rows.combined_incremental_over_latacc > 0.0).sum())
        >= MINIMUM_EVALUATION_FLIGHTS_IMPROVED
        and bool(beta_summary["sign_consistent"])
    )

    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "fit-only-structure-selection-and-held-flight-diagnostic",
        "target": {
            "equation": "[m V cos(gamma) observed_heading_rate - L sin(bank)] / (m g)",
            "source": "frozen observed-heading-rate force-balance replay",
        },
        "features": {
            "lateral_acceleration": "LATACC specific acceleration cos(bank) / g",
            "beta_proxy": "q S beta_proxy / (m g)",
            "beta_proxy_definition": "coordinate-derived horizontal air-vector direction minus calibrated true heading",
            "direct_sideslip_measurement_available": False,
            "sideslip_target_inventory": sideslip_target,
        },
        "selection": {
            "lateral_acceleration_delay_s": selected_delay,
            "selected_model": selected_model,
            "rule": "lowest-complexity candidate within one percent of best fit LOFO mean RMSE",
            "delay_candidates": _native(delay_metrics.to_dict(orient="records")),
            "candidate_metrics": _native(candidates.to_dict(orient="records")),
        },
        "coefficients": {
            model: full_fits[model].coefficients.tolist() for model in MODEL_NAMES
        },
        "stability": {
            "lateral_acceleration_scale_lofo": latacc_summary,
            "combined_beta_coefficient_lofo": beta_summary,
        },
        "gates": {
            "minimum_fit_lofo_and_validation_improvement": MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT,
            "minimum_evaluation_flights_improved": MINIMUM_EVALUATION_FLIGHTS_IMPROVED,
            "maximum_lofo_coefficient_of_variation": MAXIMUM_LOFO_COEFFICIENT_OF_VARIATION,
            "acceptable_absolute_lateral_acceleration_scale": [
                MINIMUM_UNIT_MAGNITUDE_SCALE,
                MAXIMUM_UNIT_MAGNITUDE_SCALE,
            ],
        },
        "observed": {
            "lateral_acceleration_fit_lofo_mean_improvement": latacc_fit_improvement,
            "lateral_acceleration_validation_mean_improvement": latacc_validation_improvement,
            "lateral_acceleration_validation_flights_improved": int(
                (validation_rows.lateral_acceleration_rmse_improvement > 0.0).sum()
            ),
            "lateral_acceleration_validation_turn_mean_improvement": float(
                validation_rows.lateral_acceleration_turn_rmse_improvement.mean()
            ),
            "lateral_acceleration_validation_turn_flights_improved": int(
                (validation_rows.lateral_acceleration_turn_rmse_improvement > 0.0).sum()
            ),
            "lateral_acceleration_full_fit_scale": latacc_scale,
            "beta_proxy_fit_lofo_increment_over_latacc": beta_fit_increment,
            "beta_proxy_validation_increment_over_latacc": beta_validation_increment,
            "beta_proxy_validation_flights_improved": int(
                (validation_rows.combined_incremental_over_latacc > 0.0).sum()
            ),
        },
        "decision": {
            "lateral_acceleration_explains_residual": lateral_gate,
            "beta_proxy_adds_predictive_information": beta_predictive,
            "release_physical_sideslip_or_side_force_model": False,
            "release_lateral_dynamics": False,
            "next": "retain only supported diagnostic features and require direct sideslip or an external lateral-directional prior before physical model release",
        },
        "split": {
            "fit_flights": [flight.key for flight in fit_samples],
            "evaluation_flights": [flight.key for flight in validation_samples],
            "evaluation_loaded_after_structure_freeze": True,
            "evaluation_status": "diagnostic; previously opened in commissioning",
        },
        "dependencies": {
            "observed_heading_rate_replay": "docs/modeling/a320-observed-heading-rate-replay.yml",
            "horizontal": str(args.horizontal_parameters),
            "attitude": str(args.pitch_bank_parameters),
            "yaw_rate": str(args.yaw_parameters),
        },
        "limitations": [
            "No direct sideslip measurement is present in the A320 QAR files.",
            "Beta proxy is confounded with heading rotation and wind error.",
            "Lateral acceleration is a sensor-frame specific-force measurement, not direct wind-axis side force.",
            "Evaluation flights are diagnostic rather than newly untouched confirmation data.",
        ],
    }

    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(_native(parameters), sort_keys=False))
    candidate_output = pd.concat(
        [
            delay_metrics.assign(candidate="lateral_acceleration_delay"),
            candidates.assign(delay_s=selected_delay),
        ],
        ignore_index=True,
        sort=False,
    )
    _round_frame(candidate_output).to_csv(args.candidates, index=False)
    _round_frame(metrics).to_csv(args.metrics, index=False)
    _write_report(
        args.report,
        candidates=candidates,
        metrics=metrics,
        parameters=parameters,
    )
    return metrics, parameters


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
