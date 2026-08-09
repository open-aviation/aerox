#!/usr/bin/env python3
"""Replay the A320 force-balance plant with the frozen observed-rate adapter.

This diagnostic path overrides only the bank-derived heading derivative. It
retains the physical pitch, bank, lift, drag, realized-thrust, fuel, and
longitudinal equations and reports the horizontal turn-force inconsistency
required by the observed heading rate.
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
from aerox.models import (
    LIFT_FEATURE_SETS,
    EffectiveLiftMap,
    ParabolicDragPolar,
)
from aerox.modeling.dynamics import (
    PointMassModel,
    HeadingRateReplayResult,
    simulate_heading_rate_replay_zoh,
)
from aerox.models.openap import CalibratedFuelFlow
from aerox.models.thrust import EffectiveThrustMap, MeasuredN1ThrustAdapter
from aerox.models.atmosphere import G0
from tools.audit_a320_yaw_rate import (
    BankParameters,
    _effective_bank_rad,
    _heading_rate_deg_s,
    _load_rate_flights,
    _read_bank_parameters,
)
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    PointMassReplayFlight,
    _markdown_table,
    _native,
    _read_fuel_model,
    _read_horizontal_parameters,
    _read_thrust_map,
    _round_frame,
    _trajectory_errors,
)
from tools.research_support import (
    RESET_WINDOW_S,
    longest_contiguous_slice as _longest_contiguous_slice,
)

INTEGRATION_RATE_HZ = 10
TURN_EXCITATION_BANK_DEG = 3.0


@dataclass(frozen=True)
class ObservedRateReplayJob:
    flight: PointMassReplayFlight
    split: str
    heading_rate_rad_s: np.ndarray
    bank_angle_rad: np.ndarray
    lift_model: EffectiveLiftMap
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
        default=Path("docs/modeling/a320-observed-heading-rate-replay.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-observed-heading-rate-replay-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-observed-heading-rate-replay.md"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="flight-level worker processes; use 1 for serial debugging",
    )
    return parser.parse_args()


def _read_lift_model(path: Path) -> EffectiveLiftMap:
    model = yaml.safe_load(path.read_text())["lift"]
    feature_set = str(model["feature_set"])
    coefficients = model["coefficients"]
    return EffectiveLiftMap(
        feature_set,
        np.asarray(
            [coefficients[name] for name in LIFT_FEATURE_SETS[feature_set]],
            dtype=float,
        ),
    )


def _read_yaw_model(path: Path) -> tuple[int, np.ndarray]:
    model = yaml.safe_load(path.read_text())["model"]
    coefficients = model["coefficients"]
    return int(model["trajectory_reference_delay_s"]), np.asarray(
        [
            coefficients["global_five_second_bias_deg"],
            coefficients["TAIL-B_additional_five_second_bias_deg"],
            coefficients["pitch_rate_scale_deg_s_per_code"],
            coefficients["yaw_rate_scale_deg_s_per_code"],
        ],
        dtype=float,
    )


def _trapezoidal_equivalent_zoh(rate_rad_s: np.ndarray) -> np.ndarray:
    """Return interval means that reproduce node-rate trapezoidal integration."""
    result = np.asarray(rate_rad_s, dtype=float).copy()
    if result.ndim != 1 or len(result) < 2 or np.any(~np.isfinite(result)):
        raise ValueError("rate_rad_s must be a finite vector with at least two values")
    result[:-1] = 0.5 * (result[:-1] + result[1:])
    return result


def _make_plant(job: ObservedRateReplayJob) -> PointMassModel:
    return PointMassModel(
        wing_area_m2=job.wing_area_m2,
        lift_map=job.lift_model,
        drag_model=ParabolicDragPolar(job.cd0, job.induced_drag_factor),
        fuel_flow_prior=job.fuel_model,
    )


def _simulate_slice(
    job: ObservedRateReplayJob, selection: slice
) -> HeadingRateReplayResult:
    flight = job.flight
    base = flight.base
    adapter = MeasuredN1ThrustAdapter(job.thrust_map)
    net_thrust = adapter.net_thrust_n(
        n1_percent=base.n1_percent[selection],
        mach=base.mach[selection],
        pressure_ratio=base.pressure_ratio[selection],
        static_temperature_k=base.static_temperature_k[selection],
    )
    return simulate_heading_rate_replay_zoh(
        plant=_make_plant(job),
        time_s=base.time_s[selection],
        observed_heading_rate_rad_s=job.heading_rate_rad_s[selection],
        pitch_angle_rad=flight.effective_pitch_angle_rad[selection],
        bank_angle_rad=job.bank_angle_rad[selection],
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


def _residual_summary(
    residual_n: np.ndarray,
    mass_kg: np.ndarray,
    bank_rad: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    residual_kn = residual_n / 1000.0
    residual_weight_fraction = residual_n / (mass_kg * G0)
    excited = np.abs(bank_rad) >= np.deg2rad(TURN_EXCITATION_BANK_DEG)
    result: dict[str, float | int] = {
        f"{prefix}_lateral_force_residual_rmse_kn": float(
            np.sqrt(np.mean(residual_kn**2))
        ),
        f"{prefix}_lateral_force_residual_bias_kn": float(np.mean(residual_kn)),
        f"{prefix}_lateral_force_residual_p95_abs_kn": float(
            np.quantile(np.abs(residual_kn), 0.95)
        ),
        f"{prefix}_lateral_force_residual_rmse_weight_fraction": float(
            np.sqrt(np.mean(residual_weight_fraction**2))
        ),
        f"{prefix}_turn_excited_samples": int(np.count_nonzero(excited)),
    }
    if np.any(excited):
        result.update(
            {
                f"{prefix}_turn_excited_residual_rmse_kn": float(
                    np.sqrt(np.mean(residual_kn[excited] ** 2))
                ),
                f"{prefix}_turn_excited_residual_p95_abs_weight_fraction": float(
                    np.quantile(np.abs(residual_weight_fraction[excited]), 0.95)
                ),
            }
        )
    else:
        result.update(
            {
                f"{prefix}_turn_excited_residual_rmse_kn": np.nan,
                f"{prefix}_turn_excited_residual_p95_abs_weight_fraction": np.nan,
            }
        )
    return result


def _windowed_metrics(job: ObservedRateReplayJob, segment: slice) -> dict[str, Any]:
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
    force_residuals: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    banks: list[np.ndarray] = []
    failures = 0
    windows = 0
    for window_start in range(start, stop - 60, RESET_WINDOW_S):
        window_stop = min(window_start + RESET_WINDOW_S + 1, stop)
        current = slice(window_start, window_stop)
        try:
            replay = _simulate_slice(job, current)
            errors = _trajectory_errors(job, current, replay.trajectory)
        except (ValueError, FloatingPointError, OverflowError):
            failures += 1
            continue
        windows += 1
        for name, values in errors.items():
            residuals[name].append(values)
        force_residuals.append(replay.implied_lateral_force_residual_n)
        masses.append(replay.trajectory.mass_kg)
        banks.append(replay.trajectory.bank_angle_rad)
    if windows == 0:
        return {"window_count": 0, "window_failures": failures}
    joined = {name: np.concatenate(values) for name, values in residuals.items()}
    result: dict[str, Any] = {
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
    }
    result.update(
        _residual_summary(
            np.concatenate(force_residuals),
            np.concatenate(masses),
            np.concatenate(banks),
            prefix="window",
        )
    )
    return result


def _flight_metrics(job: ObservedRateReplayJob) -> dict[str, Any]:
    flight = job.flight
    segment = _longest_contiguous_slice(flight.base.sample_index)
    if int(segment.stop) - int(segment.start) < 120:
        raise ValueError(f"continuous segment too short for {flight.key}")
    row: dict[str, Any] = {
        "registration": flight.base.registration,
        "recording": flight.base.recording,
        "role": flight.base.role,
        "split": job.split,
        "segment_duration_s": float(
            flight.base.time_s[segment][-1] - flight.base.time_s[segment][0]
        ),
    }
    row.update(_windowed_metrics(job, segment))
    try:
        replay = _simulate_slice(job, segment)
        errors = _trajectory_errors(job, segment, replay.trajectory)
    except (ValueError, FloatingPointError, OverflowError) as exception:
        row["full_success"] = False
        row["full_failure"] = type(exception).__name__
        return row
    row.update(
        {
            "full_success": True,
            "full_failure": "",
            "full_speed_rmse_mps": float(np.sqrt(np.mean(errors["speed"] ** 2))),
            "full_altitude_rmse_m": float(np.sqrt(np.mean(errors["altitude"] ** 2))),
            "full_heading_rmse_deg": float(
                np.rad2deg(np.sqrt(np.mean(errors["heading"] ** 2)))
            ),
            "full_horizontal_rmse_m": float(
                np.sqrt(np.mean(errors["horizontal"] ** 2))
            ),
            "full_measured_heading_horizontal_rmse_m": float(
                np.sqrt(np.mean(errors["measured_heading_horizontal"] ** 2))
            ),
        }
    )
    row.update(
        _residual_summary(
            replay.implied_lateral_force_residual_n,
            replay.trajectory.mass_kg,
            replay.trajectory.bank_angle_rad,
            prefix="full",
        )
    )
    return row


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("calibration", "validation"):
        current = metrics.loc[metrics["split"] == split]
        successful = current.loc[current["full_success"]]
        rows.append(
            {
                "split": split,
                "flights": len(current),
                "full_success": int(current["full_success"].sum()),
                "300s_speed_rmse_mps": current.window_speed_rmse_mps.mean(),
                "300s_altitude_rmse_m": current.window_altitude_rmse_m.mean(),
                "300s_heading_rmse_deg": current.window_heading_rmse_deg.mean(),
                "300s_horizontal_rmse_m": current.window_horizontal_rmse_m.mean(),
                "300s_measured_heading_baseline_m": current[
                    "window_measured_heading_horizontal_rmse_m"
                ].mean(),
                "300s_lateral_residual_rmse_kn": current[
                    "window_lateral_force_residual_rmse_kn"
                ].mean(),
                "300s_lateral_residual_rmse_weight_percent": 100.0
                * current["window_lateral_force_residual_rmse_weight_fraction"].mean(),
                "full_heading_rmse_deg_mean": successful.full_heading_rmse_deg.mean(),
                "full_heading_rmse_deg_median": successful.full_heading_rmse_deg.median(),
                "full_horizontal_rmse_m_mean": successful.full_horizontal_rmse_m.mean(),
                "full_horizontal_rmse_m_median": successful.full_horizontal_rmse_m.median(),
                "full_lateral_residual_rmse_kn": successful[
                    "full_lateral_force_residual_rmse_kn"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _write_report(path: Path, metrics: pd.DataFrame) -> None:
    aggregate = _round_frame(_aggregate(metrics))
    validation = metrics.loc[metrics["split"] == "validation"]
    text = f"""# A320 observed-heading-rate force-balance replay

## Decision

The replay-only heading-rate interface is implemented and all sixteen clean
segments integrate successfully. It bypasses only the heading derivative; bank
remains an input to the lift decomposition, and all longitudinal force-balance
states are unchanged from the attitude/net-thrust plant.

On the six evaluation flights, mean 300-second heading RMSE is
{validation.window_heading_rmse_deg.mean():.3f} deg and horizontal RMSE is
{validation.window_horizontal_rmse_m.mean():.1f} m. Full-segment median heading
and horizontal RMSE are {validation.full_heading_rmse_deg.median():.3f} deg and
{validation.full_horizontal_rmse_m.median():.1f} m. This branch is useful for
recorded-data replay, but it is not a trajectory-generation control.

## Aggregate metrics

{_markdown_table(aggregate)}

## Implied lateral-force residual

The diagnostic is

```text
F_residual = m V cos(gamma) observed_heading_rate - L sin(bank)
```

It is resolved normal to the horizontal air-relative velocity. It measures the
turn force missing from the coordinated bank/lift equation when the observed
heading rate is imposed. It is **not** a body-axis side-force estimate and must
not be assigned directly to rudder, sideslip, or a yawing-moment coefficient.

Across evaluation flights, the mean 300-second residual RMSE is
{validation.window_lateral_force_residual_rmse_kn.mean():.2f} kN, or
{100.0 * validation.window_lateral_force_residual_rmse_weight_fraction.mean():.2f}%
of aircraft weight. For samples with at least {TURN_EXCITATION_BANK_DEG:.0f} deg
absolute bank, the mean 95th percentile absolute residual is
{100.0 * validation.window_turn_excited_residual_p95_abs_weight_fraction.mean():.2f}%
of weight. This nonzero inconsistency is expected from the earlier conflict in
effective body-rate scales and the omitted sideslip/side-force dynamics.

## Evidence boundary

- The heading-rate coefficients and delay are read unchanged from
  `a320-yaw-rate-audit.yml`; no parameter is fitted in this replay.
- Adjacent one-second rate samples are averaged for each held interval, exactly
  reproducing the trapezoidal integration convention used by the frozen audit.
- Pitch, bank, lift, drag, thrust, fuel, atmosphere, and wind use the same
  frozen inputs as the attitude/net-thrust replay.
- Evaluation flights were held out of the yaw-rate fit but had been opened in
  earlier commissioning, so this remains diagnostic evidence.
- Windowed metrics reset all plant states every {RESET_WINDOW_S} seconds. Full
  metrics expose accumulated open-loop bias.
- Only aggregate residuals are stored; no raw QAR samples or trajectories are
  redistributed.

## Figures

![A320 observed-heading-rate force consistency](figures/a320-observed-heading-rate-replay-overview.png)

Figure 3. Flight-level horizontal replay and implied turn-force residuals.
Circles denote TAIL-A, squares denote TAIL-B, grey lines connect the same
evaluation flight, and diamonds show the across-flight summary. Error bars are
mean ± one standard deviation on linear panels and median with interquartile
range on log panels. A
[vector PDF](figures/a320-observed-heading-rate-replay-overview.pdf) is
provided. Panels A-B contain six paired evaluation flights; panels C-D contain
ten calibration and six evaluation flights.

## Reproducibility

- Command: `uv run python -m tools.replay_a320_observed_heading_rate --workers 32`
- Core implementation: `src/aerox/modeling/dynamics/point_mass.py`
- Frozen rate model: `docs/modeling/a320-yaw-rate-audit.yml`
- Parameters: `docs/modeling/a320-observed-heading-rate-replay.yml`
- Metrics: `docs/modeling/a320-observed-heading-rate-replay-metrics.csv`
- Figures: `uv run python -m tools.plot_a320_modeling_results`
- Source ledger: `docs/modeling/sources.yml`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    wing_area = float(aircraft.wing.area)
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    bank: BankParameters = _read_bank_parameters(args.pitch_bank_parameters)
    yaw_delay_s, yaw_coefficients = _read_yaw_model(args.yaw_parameters)
    lift_model = _read_lift_model(args.pitch_bank_parameters)
    thrust_map = _read_thrust_map(args.thrust_parameters)
    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)

    jobs: list[ObservedRateReplayJob] = []
    for split, selection in (
        ("calibration", panel.loc[panel["role"] != "validation"]),
        ("validation", panel.loc[panel["role"] == "validation"]),
    ):
        rate_flights = _load_rate_flights(
            selection,
            raw_dir=args.raw_dir,
            processed_path=args.processed,
            vertical=vertical,
            horizontal=horizontal,
            tas_scale=tas_scale,
            wing_area_m2=wing_area,
        )
        for rate_flight in rate_flights:
            jobs.append(
                ObservedRateReplayJob(
                    flight=rate_flight.replay,
                    split=split,
                    heading_rate_rad_s=_trapezoidal_equivalent_zoh(
                        np.deg2rad(
                            _heading_rate_deg_s(
                                rate_flight,
                                delay_s=yaw_delay_s,
                                bank=bank,
                                coefficients=yaw_coefficients,
                            )
                        )
                    ),
                    bank_angle_rad=_effective_bank_rad(rate_flight.replay, bank),
                    lift_model=lift_model,
                    thrust_map=thrust_map,
                    fuel_model=fuel_model,
                    wing_area_m2=wing_area,
                    cd0=reference_drag.zero_lift_drag_coefficient,
                    induced_drag_factor=reference_drag.induced_drag_factor,
                )
            )

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for job in jobs:
            rows.append(_flight_metrics(job))
            print(f"completed {job.split} {job.flight.key}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_flight_metrics, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                rows.append(future.result())
                print(f"completed {job.split} {job.flight.key}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(
        ["split", "registration", "recording"], kind="stable"
    )
    aggregate = _aggregate(metrics)
    validation = metrics.loc[metrics["split"] == "validation"]
    parameters: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "implemented-replay-only-not-a-trajectory-generation-control",
        "interface": {
            "implementation": "aerox.models.simulate_heading_rate_replay_zoh",
            "physical_simulator_unchanged": "aerox.models.simulate_point_mass_zoh",
            "override": "heading derivative only",
            "rate_interval_convention": "adjacent-sample mean reproduces frozen trapezoidal rate integration",
            "sampled_input_rate_hz": 1,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
        },
        "residual": {
            "equation": "m V cos(gamma) observed_heading_rate - L sin(bank)",
            "frame": "normal to horizontal air-relative velocity",
            "not_equivalent_to": "body-axis side force",
            "turn_excitation_minimum_absolute_bank_deg": TURN_EXCITATION_BANK_DEG,
        },
        "aggregate_metrics": _native(aggregate.to_dict(orient="records")),
        "evaluation": {
            "flight_count": len(validation),
            "full_segment_success": int(validation.full_success.sum()),
            "mean_300s_heading_rmse_deg": float(
                validation.window_heading_rmse_deg.mean()
            ),
            "mean_300s_horizontal_rmse_m": float(
                validation.window_horizontal_rmse_m.mean()
            ),
            "median_full_heading_rmse_deg": float(
                validation.full_heading_rmse_deg.median()
            ),
            "median_full_horizontal_rmse_m": float(
                validation.full_horizontal_rmse_m.median()
            ),
            "mean_300s_lateral_force_residual_rmse_kn": float(
                validation.window_lateral_force_residual_rmse_kn.mean()
            ),
            "mean_300s_lateral_force_residual_rmse_weight_fraction": float(
                validation.window_lateral_force_residual_rmse_weight_fraction.mean()
            ),
        },
        "decision": {
            "recorded_data_replay": "supported",
            "trajectory_generation_control": False,
            "body_axis_side_force_identified": False,
            "next": "lateral residual audit failed release; require direct sideslip or an external lateral-directional prior before adding physical dynamics",
        },
        "dependencies": {
            "vertical": str(args.vertical_parameters),
            "horizontal": str(args.horizontal_parameters),
            "attitude_and_lift": str(args.pitch_bank_parameters),
            "yaw_rate": str(args.yaw_parameters),
            "thrust": str(args.thrust_parameters),
            "fuel": str(args.fuel_parameters),
        },
        "limitations": [
            "Observed heading rate is an effective QAR replay input, not a command.",
            "The residual is trajectory-frame turn force, not body-axis side force.",
            "Sideslip, rudder, yaw moment, and lateral-directional inertias remain unidentified.",
            "Evaluation flights are diagnostic rather than newly untouched confirmation data.",
        ],
    }
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(_native(parameters), sort_keys=False))
    _round_frame(metrics).to_csv(args.metrics, index=False)
    _write_report(args.report, metrics)
    return metrics, parameters


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
