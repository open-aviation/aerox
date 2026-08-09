#!/usr/bin/env python3
"""Select and evaluate an independent A320 pitch-dynamics confirmation panel.

Panel selection uses only pre-existing operational/coverage descriptors and is
written before pitch-response evaluation.  Candidate and baseline coefficients,
features, sample rules, and gates are read from the frozen fit-only audit; this
tool never refits or changes them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tools.audit_a320_pitch_moment import (
    CANDIDATES,
    PitchFlight,
    _load_fit_flights,
    _native,
    _round_frame,
    _samples,
    _sign_accuracy,
)
from tools.audit_a320_qar import (
    RAW_AIRCRAFT,
    FlightSummary,
    audit_raw_file,
    select_diverse,
)

CONFIRMATION_FLIGHTS_PER_REGISTRATION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--existing-panel",
        type=Path,
        default=Path("docs/modeling/a320-replay-panel.csv"),
    )
    parser.add_argument(
        "--frozen-audit",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-audit.yml"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-pitch-confirmation-panel.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-confirmation-metrics.csv"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("docs/modeling/a320-pitch-confirmation.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-pitch-confirmation.md"),
    )
    return parser.parse_args()


def _panel_row(summary: FlightSummary, order: int) -> dict[str, Any]:
    return {
        "registration": summary.registration,
        "recording": summary.recording,
        "role": "confirmation",
        "selection_order": order,
        "date": summary.date,
        "airborne_duration_s": round(summary.airborne_duration_s, 1),
        "max_altitude_ft": round(summary.max_altitude_ft, 1),
        "max_cas_kts": round(summary.max_cas_kts, 3),
        "takeoff_gross_weight_kg": round(summary.takeoff_gross_weight_kg, 1),
        "endpoint_distance_nm": round(summary.endpoint_distance_nm, 2),
        "mean_wind_speed_raw": round(summary.mean_wind_speed_raw, 3),
        "surface_activity_raw": round(summary.surface_activity_raw, 3),
        "min_native_coverage": round(summary.min_native_coverage, 5),
        "max_critical_gap_s": round(summary.max_critical_gap_s, 3),
        "selection_method": "unused-pool-medoid-plus-farthest-point",
    }


def _select_panel(args: argparse.Namespace) -> pd.DataFrame:
    existing = pd.read_csv(args.existing_panel, dtype={"recording": str})
    used = set(zip(existing["registration"], existing["recording"], strict=True))
    rows: list[dict[str, Any]] = []
    for registration, filename in RAW_AIRCRAFT.items():
        summaries, _ = audit_raw_file(registration, args.raw_dir / filename)
        eligible = [
            summary
            for summary in summaries
            if summary.operational_flight
            and (summary.registration, summary.recording) not in used
        ]
        choices = select_diverse(eligible, CONFIRMATION_FLIGHTS_PER_REGISTRATION)
        rows.extend(
            _panel_row(summary, order) for order, summary in enumerate(choices, start=1)
        )
    panel = pd.DataFrame(rows)
    expected = CONFIRMATION_FLIGHTS_PER_REGISTRATION * len(RAW_AIRCRAFT)
    if len(panel) != expected:
        raise ValueError(f"expected {expected} confirmation flights")
    selected = set(zip(panel["registration"], panel["recording"], strict=True))
    if selected & used:
        raise ValueError("confirmation panel overlaps the existing replay panel")
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.panel, index=False)
    return panel


def _coefficient_vector(model: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [model["coefficients"][name] for name in model["feature_order"]],
        dtype=float,
    )


def _candidate_from_model(model: dict[str, Any]):
    name = str(model["candidate"])
    selected = [candidate for candidate in CANDIDATES if candidate.name == name]
    if len(selected) != 1:
        raise ValueError(f"frozen candidate {name!r} is not implemented")
    candidate = selected[0]
    if candidate.control_delay_s != int(model["control_delay_s"]):
        raise ValueError("frozen candidate control delay changed")
    if candidate.required_history_s != int(model["required_control_history_s"]):
        raise ValueError("frozen candidate history changed")
    return candidate


def _evaluate_flight(
    flight: PitchFlight,
    *,
    candidate_model: dict[str, Any],
    baseline_model: dict[str, Any],
    mean_aerodynamic_chord_m: float,
) -> dict[str, Any]:
    candidate = _candidate_from_model(candidate_model)
    design, target, starts = _samples(
        flight,
        controls=candidate.controls,
        control_delay_s=candidate.control_delay_s,
        required_history_s=candidate.required_history_s,
        mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
    )
    baseline_design, baseline_target, baseline_starts = _samples(
        flight,
        controls=(),
        control_delay_s=int(baseline_model["control_delay_s"]),
        required_history_s=int(baseline_model["required_control_history_s"]),
        mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
    )
    if not np.array_equal(target, baseline_target) or not np.array_equal(
        starts, baseline_starts
    ):
        raise ValueError("frozen candidate and baseline do not use identical samples")
    if list(candidate_model["feature_order"]) != [
        "aero_loading",
        "TAIL-B_aero_loading_offset",
        "aero_loading_x_aoa_code",
        "aero_loading_x_reduced_pitch_rate_code_x100",
        "aero_loading_x_mean_elevator_code",
        "aero_loading_x_mean_elevator_change_code_per_s",
    ]:
        raise ValueError("frozen candidate feature order changed")
    candidate_prediction = design @ _coefficient_vector(candidate_model)
    baseline_prediction = baseline_design @ _coefficient_vector(baseline_model)
    candidate_rmse = float(np.sqrt(np.mean((candidate_prediction - target) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((baseline_prediction - target) ** 2)))
    persistence_rmse = float(np.sqrt(np.mean(target**2)))
    significant_count, candidate_sign = _sign_accuracy(candidate_prediction, target)
    _, baseline_sign = _sign_accuracy(baseline_prediction, target)
    elevator_motion = (
        flight.mean_elevator_code[starts] - flight.mean_elevator_code[starts - 1]
    )
    active = np.abs(elevator_motion) > 0.0
    candidate_active_rmse = float(
        np.sqrt(np.mean((candidate_prediction[active] - target[active]) ** 2))
    )
    baseline_active_rmse = float(
        np.sqrt(np.mean((baseline_prediction[active] - target[active]) ** 2))
    )
    conditional_scale = float(
        candidate_model["conditional_pitch_rate_scale_deg_s_per_code"]
    )
    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "samples": len(target),
        "elevator_motion_samples": int(np.count_nonzero(active)),
        "significant_rate_change_samples": significant_count,
        "candidate_rmse_code": candidate_rmse,
        "matched_state_rmse_code": baseline_rmse,
        "persistence_rmse_code": persistence_rmse,
        "candidate_rmse_conditional_deg_s2": candidate_rmse * conditional_scale,
        "rmse_improvement_over_matched_state": (
            (baseline_rmse - candidate_rmse) / baseline_rmse
        ),
        "elevator_motion_rmse_improvement_over_matched_state": (
            (baseline_active_rmse - candidate_active_rmse) / baseline_active_rmse
        ),
        "candidate_significant_sign_accuracy": candidate_sign,
        "matched_state_significant_sign_accuracy": baseline_sign,
        "sign_accuracy_increment": candidate_sign - baseline_sign,
    }


def _decision(metrics: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["gate"]
    mean_candidate = float(metrics["candidate_rmse_code"].mean())
    mean_baseline = float(metrics["matched_state_rmse_code"].mean())
    improvement = (mean_baseline - mean_candidate) / mean_baseline
    improved_flights = int(
        np.count_nonzero(
            metrics["candidate_rmse_code"] < metrics["matched_state_rmse_code"]
        )
    )
    sign_increment = float(metrics["sign_accuracy_increment"].mean())
    passed = bool(
        improvement >= float(gate["minimum_mean_rmse_improvement_over_matched_state"])
        and improved_flights >= int(gate["minimum_complete_flights_improved"])
        and sign_increment >= float(gate["minimum_significant_sign_accuracy_increment"])
    )
    return {
        "independent_confirmation_passed": passed,
        "effective_elevator_candidate_confirmed": passed,
        "ready_for_short_window_free_pitch_replay": passed,
        "ready_for_full_flight_free_pitch_replay": False,
        "release_physical_pitching_moment_derivatives": False,
        "observed": {
            "mean_candidate_rmse_code": mean_candidate,
            "mean_matched_state_rmse_code": mean_baseline,
            "mean_persistence_rmse_code": float(
                metrics["persistence_rmse_code"].mean()
            ),
            "mean_rmse_improvement_over_matched_state": improvement,
            "complete_flights_improved": improved_flights,
            "mean_elevator_motion_rmse_improvement": float(
                metrics["elevator_motion_rmse_improvement_over_matched_state"].mean()
            ),
            "mean_significant_sign_accuracy_increment": sign_increment,
        },
        "frozen_gate": gate,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    strings = frame.fillna("").astype(str)
    headers = [str(column) for column in strings.columns]
    rows = strings.values.tolist()
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(
            headers[index].ljust(widths[index]) for index in range(len(headers))
        )
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
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    result: dict[str, Any],
) -> None:
    decision = result["decision"]
    observed = decision["observed"]
    compact_panel = panel[
        [
            "registration",
            "recording",
            "date",
            "airborne_duration_s",
            "max_altitude_ft",
            "surface_activity_raw",
            "min_native_coverage",
        ]
    ]
    compact_metrics = _round_frame(
        metrics[
            [
                "registration",
                "recording",
                "samples",
                "candidate_rmse_code",
                "matched_state_rmse_code",
                "rmse_improvement_over_matched_state",
                "candidate_significant_sign_accuracy",
                "sign_accuracy_increment",
            ]
        ]
    )
    disposition = (
        "passes and may proceed to short-window free-pitch replay"
        if decision["independent_confirmation_passed"]
        else "fails and must not proceed to free-pitch replay"
    )
    if decision["independent_confirmation_passed"]:
        next_gate = """Run measured-input, free-pitch-rate rollouts over short
continuous clean-flap windows. Compare integrated pitch rate and pitch against
measured trajectories, test boundedness, and increase horizon only after
passing. Keep stabilizer measured as a trim/configuration state; do not add its
rejected dynamic coefficient."""
    else:
        next_gate = """Do not run free-pitch rollouts with this candidate and
do not refit it on the six consumed confirmation flights. Return to model
development using the original fit flights: test the identified propulsion
state as a pitching-moment nuisance and quantify elevator excitation after
state residualization. If the next frozen structure still has weak independent
control authority, use documented aerodynamic priors or higher-rate control-
surface data. Any revised structure requires another previously unopened
panel."""
    text = f"""# A320 independent pitch-dynamics confirmation

## Decision

The frozen effective elevator position-plus-motion candidate **{disposition}**.
Across six previously unopened flights, its mean one-second RMSE improvement
over the exactly matched state-only model is
{observed["mean_rmse_improvement_over_matched_state"]:.2%}; it improves
{observed["complete_flights_improved"]} of six complete flights. Mean
significant-change sign accuracy changes by
{observed["mean_significant_sign_accuracy_increment"]:.2%}.

No coefficient, feature, delay, sample rule, or gate was changed after panel
selection. This confirmation concerns an effective 1 Hz closed-loop predictor,
not a physical pitching-moment derivative. Full-flight free-pitch replay remains
blocked.

## Independently selected panel

{_markdown_table(compact_panel)}

The six flights are absent from the original 16-flight replay panel. Selection
uses the original deterministic medoid-plus-farthest-point rule on duration,
altitude, speed, mass, endpoint distance, wind, surface activity, and coverage.
It does not use pitch-response residual or candidate prediction error.

## Frozen-model results

{_markdown_table(compact_metrics)}

The candidate and baseline use identical samples. Prediction is a single matrix
multiplication with coefficients copied from the fit-only artifact; this tool
does not refit either model.

## Next gate

{next_gate}
"""
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frozen = yaml.safe_load(args.frozen_audit.read_text())
    contract = frozen["independent_confirmation_contract"]
    if contract["status"] != "frozen-before-panel-selection":
        raise ValueError("confirmation contract was not frozen before selection")
    expected_panel_size = int(contract["panel_size"])
    if expected_panel_size != CONFIRMATION_FLIGHTS_PER_REGISTRATION * len(RAW_AIRCRAFT):
        raise ValueError("confirmation panel allocation changed")

    panel = _select_panel(args)
    # The panel artifact now exists. Only after this point are the frozen
    # response model and selected recordings combined for evaluation.
    load_args = SimpleNamespace(
        raw_dir=args.raw_dir, processed=args.processed, panel=args.panel
    )
    flights = _load_fit_flights(load_args, expected_count=expected_panel_size)
    candidate_model = contract["candidate_model"]
    baseline_model = contract["matched_state_baseline_model"]
    mean_aerodynamic_chord = float(frozen["method"]["mean_aerodynamic_chord_m"])
    metrics = pd.DataFrame(
        [
            _evaluate_flight(
                flight,
                candidate_model=candidate_model,
                baseline_model=baseline_model,
                mean_aerodynamic_chord_m=mean_aerodynamic_chord,
            )
            for flight in flights
        ]
    )
    decision = _decision(metrics, contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": (
            "independent-effective-pitch-candidate-confirmed"
            if decision["independent_confirmation_passed"]
            else "independent-effective-pitch-confirmation-failed"
        ),
        "leakage_control": {
            "fit_flights_loaded_by_validation_tool": False,
            "original_replay_panel_excluded": True,
            "panel_written_before_response_evaluation": True,
            "model_or_gate_refit_after_panel_selection": False,
            "confirmation_flights": [flight.key for flight in flights],
        },
        "frozen_source": str(args.frozen_audit),
        "model_candidate": candidate_model["candidate"],
        "method": {
            "selection": "unused-pool medoid plus farthest-point",
            "selection_response_blind": True,
            "evaluation": "fixed coefficients; no refit",
            "cross_flight_aggregation": "unweighted mean of complete-flight metrics",
        },
        "decision": decision,
        "artifacts": {
            "runner": "tools/validate_a320_pitch_moment.py",
            "panel": str(args.panel),
            "metrics": str(args.metrics),
            "report": str(args.report),
        },
        "limitations": [
            "Confirmation remains limited to two A320-214 registrations and airline closed-loop operation.",
            "The effective elevator-motion term is not a physical rate derivative.",
            "One-step confirmation does not establish stable multi-step or full-flight replay.",
            "Physical inertia, CG, surface scale/neutral, and pitching-moment derivatives remain unreleased.",
        ],
    }
    _round_frame(metrics).to_csv(args.metrics, index=False)
    args.result.write_text(yaml.safe_dump(_native(result), sort_keys=False))
    _write_report(args.report, panel=panel, metrics=metrics, result=result)
    return panel, metrics, result


def main() -> None:
    panel, metrics, result = run(parse_args())
    print(panel.to_string(index=False))
    print(metrics.to_string(index=False))
    print(yaml.safe_dump(result["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
