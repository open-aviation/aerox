#!/usr/bin/env python3
"""Audit A320 elevator excitation after propulsion-state residualization.

Only the original ten development/calibration flights are loaded.  Effective
thrust and thrust change are tested as pitching-moment nuisance features before
the elevator position-plus-motion input is reconsidered.  Previously held-out
and consumed confirmation flights are never opened by this tool.
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
from aerox.modeling.identification import robust_weighted_linear_fit
from aerox.models.thrust import EffectiveThrustMap
from tools.audit_a320_pitch_moment import (
    PitchFlight,
    _load_fit_flights,
    _native,
    _round_frame,
    _samples,
    _sign_accuracy,
)

MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT = 0.005
MINIMUM_IMPROVED_FLIGHTS = 8
MINIMUM_COEFFICIENT_SIGN_CONSISTENCY = 1.0
MAXIMUM_COEFFICIENT_VARIATION = 0.15


@dataclass(frozen=True)
class ExcitationFlight:
    flight: PitchFlight
    state_design: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    thrust_acceleration_m_s2: np.ndarray
    thrust_acceleration_change_m_s2: np.ndarray
    elevator_position_feature: np.ndarray
    elevator_motion_feature: np.ndarray

    @property
    def key(self) -> str:
        return self.flight.key


@dataclass(frozen=True)
class Candidate:
    name: str
    thrust_position: bool = False
    thrust_change: bool = False
    elevator: bool = False


CANDIDATES = (
    Candidate("state_only"),
    Candidate("state_plus_thrust", thrust_position=True),
    Candidate("state_plus_thrust_change", thrust_change=True),
    Candidate(
        "state_plus_thrust_and_change",
        thrust_position=True,
        thrust_change=True,
    ),
    Candidate("state_plus_elevator", elevator=True),
    Candidate("state_plus_thrust_plus_elevator", True, False, True),
    Candidate("state_plus_thrust_change_plus_elevator", False, True, True),
    Candidate("state_plus_thrust_and_change_plus_elevator", True, True, True),
)


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
        "--confirmation-panel",
        type=Path,
        default=Path("docs/modeling/a320-pitch-confirmation-panel.csv"),
    )
    parser.add_argument(
        "--pitch-audit",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-audit.yml"),
    )
    parser.add_argument(
        "--thrust-calibration",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.yml"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-excitation-candidates.csv"),
    )
    parser.add_argument(
        "--lofo-metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-excitation-lofo.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-pitch-excitation-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-pitch-excitation-audit.md"),
    )
    return parser.parse_args()


def _read_thrust_model(path: Path) -> tuple[EffectiveThrustMap, dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    model = data["model"]
    coefficient_map = model["scenarios"]["nominal"]["coefficients"]
    coefficients = np.asarray(
        [coefficient_map[name] for name in model["feature_order"]], dtype=float
    )
    return (
        EffectiveThrustMap(
            rated_total_thrust_n=float(model["rated_total_thrust_n"]),
            pressure_exponent=float(model["pressure_exponent"]),
            feature_set=str(model["feature_set"]),
            corrected_fan_speed=bool(model["corrected_fan_speed"]),
            coefficients=coefficients,
        ),
        data,
    )


def _build_flight(
    flight: PitchFlight,
    *,
    thrust_model: EffectiveThrustMap,
    mean_aerodynamic_chord_m: float,
) -> ExcitationFlight:
    state, target, starts = _samples(
        flight,
        controls=(),
        control_delay_s=0,
        required_history_s=1,
        mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
    )
    pressure_ratio = (
        np.asarray(aero.pressure(flight.pressure_altitude_m[starts]), dtype=float)
        / aero.p0
    )
    previous_pressure_ratio = (
        np.asarray(aero.pressure(flight.pressure_altitude_m[starts - 1]), dtype=float)
        / aero.p0
    )
    finite = np.isfinite(
        flight.static_temperature_k[starts]
        + flight.static_temperature_k[starts - 1]
        + flight.n1_percent[starts]
        + flight.n1_percent[starts - 1]
        + pressure_ratio
        + previous_pressure_ratio
    )
    state = state[finite]
    target = target[finite]
    starts = starts[finite]
    pressure_ratio = pressure_ratio[finite]
    previous_pressure_ratio = previous_pressure_ratio[finite]
    thrust = thrust_model.thrust_n(
        n1_percent=flight.n1_percent[starts],
        mach=flight.mach[starts],
        pressure_ratio=pressure_ratio,
        static_temperature_k=flight.static_temperature_k[starts],
    )
    previous_thrust = thrust_model.thrust_n(
        n1_percent=flight.n1_percent[starts - 1],
        mach=flight.mach[starts - 1],
        pressure_ratio=previous_pressure_ratio,
        static_temperature_k=flight.static_temperature_k[starts - 1],
    )
    loading = (flight.dynamic_pressure_pa[starts] / 10_000.0) * (
        60_000.0 / flight.mass_kg[starts]
    )
    weights = np.full(len(target), 1.0 / len(target))
    return ExcitationFlight(
        flight=flight,
        state_design=state,
        target=target,
        weights=weights,
        thrust_acceleration_m_s2=thrust / flight.mass_kg[starts],
        thrust_acceleration_change_m_s2=(thrust - previous_thrust)
        / flight.mass_kg[starts],
        elevator_position_feature=(loading * flight.mean_elevator_code[starts]),
        elevator_motion_feature=(
            loading
            * (
                flight.mean_elevator_code[starts]
                - flight.mean_elevator_code[starts - 1]
            )
        ),
    )


def _feature_order(candidate: Candidate) -> list[str]:
    order = [
        "aero_loading",
        "TAIL-B_aero_loading_offset",
        "aero_loading_x_aoa_code",
        "aero_loading_x_reduced_pitch_rate_code_x100",
    ]
    if candidate.thrust_position:
        order.append("effective_thrust_per_mass_m_s2")
    if candidate.thrust_change:
        order.append("effective_thrust_change_per_mass_m_s2")
    if candidate.elevator:
        order.extend(
            [
                "aero_loading_x_mean_elevator_code",
                "aero_loading_x_mean_elevator_change_code_per_s",
            ]
        )
    return order


def _design(flight: ExcitationFlight, candidate: Candidate) -> np.ndarray:
    columns = [flight.state_design]
    if candidate.thrust_position:
        columns.append(flight.thrust_acceleration_m_s2[:, None])
    if candidate.thrust_change:
        columns.append(flight.thrust_acceleration_change_m_s2[:, None])
    if candidate.elevator:
        columns.extend(
            [
                flight.elevator_position_feature[:, None],
                flight.elevator_motion_feature[:, None],
            ]
        )
    return np.column_stack(columns)


def _fit(flights: list[ExcitationFlight], candidate: Candidate):
    designs = [_design(flight, candidate) for flight in flights]
    weights = [flight.weights / len(flights) for flight in flights]
    return robust_weighted_linear_fit(
        np.vstack(designs),
        np.concatenate([flight.target for flight in flights]),
        np.concatenate(weights),
    )


def _lofo(flights: list[ExcitationFlight]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        order = _feature_order(candidate)
        for held_out in flights:
            fit = _fit(
                [flight for flight in flights if flight.key != held_out.key],
                candidate,
            )
            prediction = _design(held_out, candidate) @ fit.coefficients
            significant_count, sign_accuracy = _sign_accuracy(
                prediction, held_out.target
            )
            row: dict[str, Any] = {
                "candidate": candidate.name,
                "registration": held_out.flight.registration,
                "recording": held_out.flight.recording,
                "role": held_out.flight.role,
                "samples": len(held_out.target),
                "significant_rate_change_samples": significant_count,
                "rmse_code": float(
                    np.sqrt(np.mean((prediction - held_out.target) ** 2))
                ),
                "significant_sign_accuracy": sign_accuracy,
                "training_normal_matrix_condition": float(fit.normal_matrix_condition),
            }
            for feature in (
                "effective_thrust_per_mass_m_s2",
                "effective_thrust_change_per_mass_m_s2",
                "aero_loading_x_mean_elevator_code",
                "aero_loading_x_mean_elevator_change_code_per_s",
            ):
                row[f"coefficient__{feature}"] = (
                    float(fit.coefficients[order.index(feature)])
                    if feature in order
                    else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _coefficient_summary(values: pd.Series) -> tuple[float, float, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].astype(float)
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    cv = float(finite.std() / abs(mean)) if mean != 0 else float("inf")
    consistency = float(max(np.mean(finite > 0.0), np.mean(finite < 0.0)))
    return mean, cv, consistency


def _candidate_metrics(lofo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    coefficient_columns = [
        column for column in lofo if column.startswith("coefficient__")
    ]
    for candidate in CANDIDATES:
        current = lofo.loc[lofo["candidate"] == candidate.name]
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "mean_lofo_rmse_code": float(current["rmse_code"].mean()),
            "mean_significant_sign_accuracy": float(
                current["significant_sign_accuracy"].mean()
            ),
            "maximum_training_normal_matrix_condition": float(
                current["training_normal_matrix_condition"].max()
            ),
        }
        for column in coefficient_columns:
            mean, cv, consistency = _coefficient_summary(current[column])
            name = column.removeprefix("coefficient__")
            row[f"{name}__mean"] = mean
            row[f"{name}__cv"] = cv
            row[f"{name}__sign_consistency"] = consistency
        rows.append(row)
    return pd.DataFrame(rows)


def _row(frame: pd.DataFrame, candidate: str) -> pd.Series:
    selected = frame.loc[frame["candidate"] == candidate]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {candidate}")
    return selected.iloc[0]


def _paired_improvement(
    lofo: pd.DataFrame, baseline: str, candidate: str
) -> tuple[float, int, float]:
    left = lofo.loc[lofo["candidate"] == baseline].sort_values(
        ["registration", "recording"]
    )
    right = lofo.loc[lofo["candidate"] == candidate].sort_values(
        ["registration", "recording"]
    )
    base_rmse = float(left["rmse_code"].mean())
    candidate_rmse = float(right["rmse_code"].mean())
    improvement = (base_rmse - candidate_rmse) / base_rmse
    flights_improved = int(
        np.count_nonzero(right["rmse_code"].to_numpy() < left["rmse_code"].to_numpy())
    )
    sign_increment = float(
        right["significant_sign_accuracy"].mean()
        - left["significant_sign_accuracy"].mean()
    )
    return improvement, flights_improved, sign_increment


def _weighted_projection(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    root_weight = np.sqrt(weights)
    if target.ndim == 1:
        coefficients = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None
        )[0]
    else:
        coefficients = np.linalg.lstsq(
            design * root_weight[:, None],
            target * root_weight[:, None],
            rcond=None,
        )[0]
    return design @ coefficients


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    left_centered = left - np.average(left, weights=weights)
    right_centered = right - np.average(right, weights=weights)
    covariance = np.average(left_centered * right_centered, weights=weights)
    variance = np.average(left_centered**2, weights=weights) * np.average(
        right_centered**2, weights=weights
    )
    return float(covariance / np.sqrt(variance))


def _residualized_excitation(
    flights: list[ExcitationFlight], base_candidate: Candidate
) -> dict[str, Any]:
    design = np.vstack([_design(flight, base_candidate) for flight in flights])
    target = np.concatenate([flight.target for flight in flights])
    controls = np.column_stack(
        (
            np.concatenate([flight.elevator_position_feature for flight in flights]),
            np.concatenate([flight.elevator_motion_feature for flight in flights]),
        )
    )
    weights = np.concatenate([flight.weights / len(flights) for flight in flights])
    control_residual = controls - _weighted_projection(design, controls, weights)
    target_residual = target - _weighted_projection(design, target, weights)
    control_mean = np.average(controls, axis=0, weights=weights)
    total_variance = np.average((controls - control_mean) ** 2, axis=0, weights=weights)
    residual_variance = np.average(control_residual**2, axis=0, weights=weights)
    explained = 1.0 - residual_variance / total_variance
    singular_values = np.linalg.svd(
        control_residual * np.sqrt(weights)[:, None], compute_uv=False
    )
    return {
        "base_candidate": base_candidate.name,
        "elevator_position_variance_explained_by_state": float(explained[0]),
        "elevator_motion_variance_explained_by_state": float(explained[1]),
        "elevator_position_remaining_std_fraction": float(
            np.sqrt(residual_variance[0] / total_variance[0])
        ),
        "elevator_motion_remaining_std_fraction": float(
            np.sqrt(residual_variance[1] / total_variance[1])
        ),
        "target_partial_correlation_with_elevator_position": (
            _weighted_correlation(target_residual, control_residual[:, 0], weights)
        ),
        "target_partial_correlation_with_elevator_motion": (
            _weighted_correlation(target_residual, control_residual[:, 1], weights)
        ),
        "residualized_control_pair_correlation": _weighted_correlation(
            control_residual[:, 0], control_residual[:, 1], weights
        ),
        "residualized_control_singular_values": singular_values.tolist(),
        "residualized_control_condition": float(
            singular_values[0] / singular_values[-1]
        ),
    }


def _decision(
    candidates: pd.DataFrame,
    lofo: pd.DataFrame,
    residualization: dict[str, Any],
) -> dict[str, Any]:
    thrust = _row(candidates, "state_plus_thrust")
    thrust_and_change = _row(candidates, "state_plus_thrust_and_change")
    thrust_effect = _paired_improvement(lofo, "state_only", "state_plus_thrust")
    elevator_after_thrust = _paired_improvement(
        lofo, "state_plus_thrust", "state_plus_thrust_plus_elevator"
    )
    elevator_after_both = _paired_improvement(
        lofo,
        "state_plus_thrust_and_change",
        "state_plus_thrust_and_change_plus_elevator",
    )
    position_cv = float(
        _row(candidates, "state_plus_thrust_plus_elevator")[
            "aero_loading_x_mean_elevator_code__cv"
        ]
    )
    motion_cv = float(
        _row(candidates, "state_plus_thrust_plus_elevator")[
            "aero_loading_x_mean_elevator_change_code_per_s__cv"
        ]
    )
    thrust_cv = float(thrust["effective_thrust_per_mass_m_s2__cv"])
    thrust_change_cv = float(
        thrust_and_change["effective_thrust_change_per_mass_m_s2__cv"]
    )
    thrust_material = bool(
        thrust_effect[0] >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and thrust_effect[1] >= MINIMUM_IMPROVED_FLIGHTS
        and thrust["effective_thrust_per_mass_m_s2__sign_consistency"]
        >= MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
        and thrust_cv <= MAXIMUM_COEFFICIENT_VARIATION
    )
    elevator_passed = bool(
        elevator_after_thrust[0] >= MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
        and elevator_after_thrust[1] >= MINIMUM_IMPROVED_FLIGHTS
        and position_cv <= MAXIMUM_COEFFICIENT_VARIATION
        and motion_cv <= MAXIMUM_COEFFICIENT_VARIATION
    )
    return {
        "material_propulsion_moment_nuisance_observed": thrust_material,
        "elevator_after_propulsion_fit_gate_passed": elevator_passed,
        "select_revised_effective_pitch_candidate": False,
        "open_another_confirmation_panel": False,
        "ready_for_free_pitch_replay": False,
        "recommend_external_aerodynamic_prior_or_higher_rate_data": True,
        "release_physical_pitching_moment_derivatives": False,
        "observed": {
            "thrust_incremental_rmse_improvement": thrust_effect[0],
            "thrust_complete_flights_improved": thrust_effect[1],
            "thrust_coefficient_cv": thrust_cv,
            "elevator_incremental_rmse_improvement_after_thrust": (
                elevator_after_thrust[0]
            ),
            "elevator_complete_flights_improved_after_thrust": (
                elevator_after_thrust[1]
            ),
            "elevator_sign_accuracy_increment_after_thrust": (elevator_after_thrust[2]),
            "elevator_position_coefficient_cv_after_thrust": position_cv,
            "elevator_motion_coefficient_cv_after_thrust": motion_cv,
            "thrust_and_change_elevator_rmse_sensitivity": elevator_after_both[0],
            "thrust_and_change_elevator_flights_improved": elevator_after_both[1],
            "thrust_change_coefficient_cv": thrust_change_cv,
            **residualization,
        },
        "gate": {
            "minimum_incremental_rmse_improvement": (
                MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
            ),
            "minimum_improved_complete_flights": MINIMUM_IMPROVED_FLIGHTS,
            "minimum_coefficient_sign_consistency": (
                MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
            ),
            "maximum_coefficient_of_variation": MAXIMUM_COEFFICIENT_VARIATION,
        },
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
    path: Path, *, candidates: pd.DataFrame, audit: dict[str, Any]
) -> None:
    observed = audit["decision"]["observed"]
    compact = _round_frame(
        candidates[
            [
                "candidate",
                "mean_lofo_rmse_code",
                "mean_significant_sign_accuracy",
                "effective_thrust_per_mass_m_s2__cv",
                "effective_thrust_change_per_mass_m_s2__cv",
                "aero_loading_x_mean_elevator_code__cv",
                "aero_loading_x_mean_elevator_change_code_per_s__cv",
            ]
        ]
    )
    text = f"""# A320 fit-only pitch excitation and propulsion-nuisance audit

## Decision

Adding effective thrust per mass to the state model improves complete-flight
LOFO RMSE by only {observed["thrust_incremental_rmse_improvement"]:.2%}. Its
coefficient is stable ({observed["thrust_coefficient_cv"]:.1%} fold variation),
but the effect is below the unchanged 0.5% materiality gate. After this
physically motivated nuisance is included, elevator position plus one-second
motion improves RMSE by
{observed["elevator_incremental_rmse_improvement_after_thrust"]:.2%} on
{observed["elevator_complete_flights_improved_after_thrust"]} of ten flights,
also below the gate.

Adding thrust change raises the elevator sensitivity result to
{observed["thrust_and_change_elevator_rmse_sensitivity"]:.2%}, but it improves
only {observed["thrust_and_change_elevator_flights_improved"]} flights and the
thrust-change coefficient varies by
{observed["thrust_change_coefficient_cv"]:.1%}. This is retained only as a
timing sensitivity, not a new pitching-moment structure.

The negative result is not explained by a rank collapse. After residualizing
against aerodynamic state and thrust, elevator position retains
{observed["elevator_position_remaining_std_fraction"]:.1%} of its standard
deviation and elevator motion retains
{observed["elevator_motion_remaining_std_fraction"]:.1%}. Their residualized
two-column condition number is
{observed["residualized_control_condition"]:.2f}. Partial correlations with the
pitch-rate-change residual are only
{observed["target_partial_correlation_with_elevator_position"]:.3f} and
{observed["target_partial_correlation_with_elevator_motion"]:.3f}.

Therefore no revised A320 control candidate is selected, no further panel is
opened, and free-pitch replay remains blocked. The evidence points to weak and
non-transferable closed-loop association at 1 Hz rather than a simple absence
of algebraically independent elevator variation.

## Candidate comparison

{_markdown_table(compact)}

All candidates use identical clean-flap samples and leave one complete fit
flight out. Effective thrust comes from the already validated, drag-prior-
conditional N1/Mach/pressure model and enters as thrust divided by gross mass.
No previous validation or consumed confirmation flight is loaded.

## Recommended next move

Do not keep adding fit-only regressors to the A320 QAR model. Use a documented
aerodynamic stability/control prior or higher-rate data to anchor pitch control
authority. The RJ85/Dashlink data should now be audited for sample rate,
elevator/command channels, angular rates, attitude, AOA, air data, engine state,
and mass; if those channels are suitable, it is the better system-identification
airframe. The A320 data can remain valuable for low-frequency trajectory,
propulsion, lift/drag, and eventual prior-conditioned validation.
"""
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pitch_audit = yaml.safe_load(args.pitch_audit.read_text())
    mean_aerodynamic_chord = float(pitch_audit["method"]["mean_aerodynamic_chord_m"])
    thrust_model, thrust_data = _read_thrust_model(args.thrust_calibration)
    flights = _load_fit_flights(args)
    confirmation = pd.read_csv(args.confirmation_panel, dtype={"recording": str})
    confirmation_keys = set(
        zip(
            confirmation["registration"],
            confirmation["recording"],
            strict=True,
        )
    )
    loaded_keys = {(flight.registration, flight.recording) for flight in flights}
    if loaded_keys & confirmation_keys:
        raise ValueError("consumed confirmation flight entered the fit audit")
    excitation_flights = [
        _build_flight(
            flight,
            thrust_model=thrust_model,
            mean_aerodynamic_chord_m=mean_aerodynamic_chord,
        )
        for flight in flights
    ]
    lofo = _lofo(excitation_flights)
    candidates = _candidate_metrics(lofo)
    residualization = _residualized_excitation(
        excitation_flights,
        next(
            candidate
            for candidate in CANDIDATES
            if candidate.name == "state_plus_thrust"
        ),
    )
    decision = _decision(candidates, lofo, residualization)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": "fit-only-propulsion-nuisance-does-not-rescue-pitch-control",
        "leakage_control": {
            "fit_flights_only": [flight.key for flight in excitation_flights],
            "previous_validation_flights_loaded": False,
            "consumed_confirmation_flights_loaded": False,
            "consumed_confirmation_flights_excluded": [
                f"{registration}/{recording}"
                for registration, recording in sorted(confirmation_keys)
            ],
        },
        "method": {
            "sample_rate_hz": 1,
            "prediction_horizon_s": 1,
            "cross_validation": "leave-one-complete-fit-flight-out",
            "fit_weighting": "equal total weight per flight with Huber robustness",
            "propulsion_nuisance": (
                "nominal effective thrust divided by gross mass; one-second "
                "change is timing sensitivity only"
            ),
            "residualization": (
                "equal-flight weighted Frisch-Waugh-Lovell projection against "
                "aerodynamic state plus effective thrust"
            ),
        },
        "decision": decision,
        "artifacts": {
            "runner": "tools/audit_a320_pitch_excitation.py",
            "candidate_metrics": str(args.candidate_metrics),
            "lofo_metrics": str(args.lofo_metrics),
            "report": str(args.report),
        },
        "dependencies": {
            "pitch_fit_audit": str(args.pitch_audit),
            "failed_confirmation": "docs/modeling/a320-pitch-confirmation.yml",
            "effective_thrust_calibration": str(args.thrust_calibration),
            "effective_thrust_status": thrust_data["status"],
        },
        "limitations": [
            "Effective thrust remains conditional on the drag prior and provisional N1/temperature lineage.",
            "Thrust-line moment arm, CG, and pitch inertia are unavailable and absorbed into one effective coefficient.",
            "Residualized association in closed-loop data is not causal control identification.",
            "The 1 Hz data cannot identify actuator or short-period dynamics.",
            "No revised model is authorized for confirmation or replay.",
        ],
    }
    args.candidate_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(lofo).to_csv(args.lofo_metrics, index=False)
    args.audit.write_text(yaml.safe_dump(_native(audit), sort_keys=False))
    _write_report(args.report, candidates=candidates, audit=audit)
    return candidates, lofo, audit


def main() -> None:
    candidates, _, audit = run(parse_args())
    print(candidates.to_string(index=False))
    print(yaml.safe_dump(audit["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
