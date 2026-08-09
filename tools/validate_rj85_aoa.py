#!/usr/bin/env python3
"""Evaluate the frozen RJ85/DASHlink AOA decoder on its unopened panel."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tools.calibrate_rj85_aoa import _load_flight, _read_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dashlink"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-calibration.yml"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-confirmation-panel.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-confirmation.md"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-confirmation.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-confirmation-metrics.csv"),
    )
    return parser.parse_args()


def _flight_metrics(
    flight: Any, *, offset: float, scale: float, training_mean: float
) -> dict[str, Any]:
    prediction = offset + scale * flight.raw_aoa_deg
    target = flight.kinematic_aoa_deg
    error = prediction - target
    baseline_error = training_mean - target
    rmse = float(np.sqrt(np.mean(error**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
    return {
        "tail": flight.tail,
        "archive": flight.archive,
        "member": flight.member,
        "samples": len(target),
        "rmse_deg": rmse,
        "mae_deg": float(np.mean(np.abs(error))),
        "bias_deg": float(np.mean(error)),
        "fixed_training_mean_rmse_deg": baseline_rmse,
        "improvement_vs_fixed_training_mean_pct": 100.0
        * (baseline_rmse - rmse)
        / baseline_rmse,
        "raw_target_correlation": float(np.corrcoef(flight.raw_aoa_deg, target)[0, 1]),
    }


def _decision(
    rows: list[dict[str, Any]],
    gates: dict[str, Any],
    *,
    minimum_valid_samples: int = 0,
) -> dict[str, Any]:
    mean_rmse = float(np.mean([row["rmse_deg"] for row in rows]))
    mean_improvement = float(
        np.mean([row["improvement_vs_fixed_training_mean_pct"] for row in rows])
    )
    improved = int(
        np.count_nonzero(
            [row["improvement_vs_fixed_training_mean_pct"] > 0 for row in rows]
        )
    )
    mean_bias = float(np.mean([row["bias_deg"] for row in rows]))
    correlated = int(
        np.count_nonzero([row["raw_target_correlation"] >= 0.8 for row in rows])
    )
    gate_results = {
        "sample_sufficiency": all(
            row.get("samples", minimum_valid_samples) >= minimum_valid_samples
            for row in rows
        ),
        "mean_rmse": mean_rmse <= gates["maximum_mean_rmse_deg"],
        "mean_improvement": mean_improvement
        >= gates["minimum_mean_improvement_vs_fixed_training_mean_pct"],
        "flights_improved": improved
        >= gates["minimum_flights_improved_vs_fixed_training_mean"],
        "mean_bias": abs(mean_bias) <= gates["maximum_absolute_mean_bias_deg"],
        "flight_correlations": correlated
        >= gates["minimum_flights_with_raw_target_correlation_at_least_0_8"],
    }
    passed = all(gate_results.values())
    return {
        "passed": passed,
        "release_status": (
            "confirmed-for-effective-aoa-state-within-filter-envelope"
            if passed
            else "blocked-from-force-and-moment-identification"
        ),
        "mean_rmse_deg": mean_rmse,
        "mean_improvement_vs_fixed_training_mean_pct": mean_improvement,
        "flights_improved_vs_fixed_training_mean": improved,
        "flight_count": len(rows),
        "minimum_samples_in_flight": min(row.get("samples", 0) for row in rows),
        "mean_bias_deg": mean_bias,
        "flights_with_raw_target_correlation_at_least_0_8": correlated,
        "gate_results": gate_results,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(value, 8) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _write_report(
    path: Path, *, result: dict[str, Any], metrics: list[dict[str, Any]]
) -> None:
    decision = result["decision"]
    status = "passes" if decision["passed"] else "fails"
    improved_text = (
        f"{decision['flights_improved_vs_fixed_training_mean']}/"
        f"{decision['flight_count']}"
    )
    correlated_text = (
        f"{decision['flights_with_raw_target_correlation_at_least_0_8']}/"
        f"{decision['flight_count']}"
    )
    release = (
        "The decoder may now be used as an effective AOA state inside the frozen "
        "filter envelope. It remains a recorder/kinematic mapping, not a physical "
        "vane calibration."
        if decision["passed"]
        else "The decoder remains blocked from force and moment identification."
    )
    rows = "\n".join(
        f"| {row['tail']} | {row['member']} | {row['samples']} | "
        f"{row['rmse_deg']:.3f} | {row['bias_deg']:.3f} | "
        f"{row['improvement_vs_fixed_training_mean_pct']:.1f}% | "
        f"{row['raw_target_correlation']:.3f} |"
        for row in metrics
    )
    path.write_text(
        f"""# RJ85/DASHlink corrected-AOA independent confirmation

## Decision

The frozen affine corrected-AOA decoder **{status}** its independent six-flight
confirmation contract. Mean RMSE is {decision["mean_rmse_deg"]:.3f} degrees,
mean improvement over the fixed development-target mean is
{decision["mean_improvement_vs_fixed_training_mean_pct"]:.1f}%, and mean bias is
{decision["mean_bias_deg"]:.3f} degrees. It improves {improved_text} flights;
{correlated_text} have raw/kinematic correlation of at least 0.8.
The smallest confirmation flight contributes
{decision["minimum_samples_in_flight"]:,} clean samples; the frozen minimum is
{result["minimum_valid_samples_per_flight"]:,}.

{release}

No coefficient or filter was refitted. These six confirmation members are now
consumed and cannot be used for retuning.

## Per-flight evidence

| tail | member | samples | RMSE (deg) | bias (deg) | improvement | correlation |
|---|---|---:|---:|---:|---:|---:|
{rows}

## Scope

The result applies above 5,000 ft, above 100 m/s TAS, within 10 degrees of wings
level, and inside the frozen raw/kinematic AOA bounds. Vertical wind, sideslip,
and recorder biases remain effective-model uncertainty. Avro RJ85 remains the
project identity assumption for NASA's de-identified regional jet.
"""
    )


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = yaml.safe_load(args.calibration.read_text())
    contract = calibration["confirmation_contract"]
    if not contract["no_refit"]:
        raise ValueError("confirmation contract must prohibit refitting")
    panel = _read_panel(args.panel)
    if len(panel) != contract["flight_count"]:
        raise ValueError("confirmation panel size differs from frozen contract")
    fit_members = {
        item.split("/", maxsplit=1)[1] for item in calibration["fit_data"]["flights"]
    }
    panel_members = {row["member"] for row in panel}
    if fit_members & panel_members:
        raise ValueError("fit member entered independent confirmation")
    coefficient = calibration["selected_model"]["coefficients"]
    training_mean = calibration["selected_model"]["training_mean_kinematic_aoa_deg"]
    # Load every frozen panel member so an insufficient-sample member is
    # recorded as a gate failure, not silently replaced after response access.
    flights = [
        _load_flight(args.data_dir, row, minimum_clean_samples=1) for row in panel
    ]
    metrics = [
        _flight_metrics(
            flight,
            offset=coefficient["offset_deg"],
            scale=coefficient["raw_aoa_scale"],
            training_mean=training_mean,
        )
        for flight in flights
    ]
    decision = _decision(
        metrics,
        contract["gates"],
        minimum_valid_samples=contract["minimum_valid_samples_per_flight"],
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": calibration["aircraft"],
        "status": decision["release_status"],
        "leakage_control": {
            "fixed_calibration": str(args.calibration),
            "panel": str(args.panel),
            "fit_members_loaded": False,
            "coefficient_refit": False,
            "filter_refit": False,
            "consumed_confirmation_members": sorted(panel_members),
        },
        "fixed_model": calibration["selected_model"],
        "fixed_target_and_filters": calibration["kinematic_target"],
        "fixed_gates": contract["gates"],
        "minimum_valid_samples_per_flight": contract[
            "minimum_valid_samples_per_flight"
        ],
        "decision": decision,
        "artifacts": {
            "runner": "tools/validate_rj85_aoa.py",
            "result": str(args.result),
            "report": str(args.report),
            "metrics": str(args.metrics),
        },
        "limitations": calibration["limitations"],
    }
    result = _native(result)
    _write_csv(args.metrics, metrics)
    args.result.write_text(yaml.safe_dump(result, sort_keys=False))
    _write_report(args.report, result=result, metrics=metrics)
    return metrics, result


def main() -> None:
    _, result = run(parse_args())
    print(yaml.safe_dump(result["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
