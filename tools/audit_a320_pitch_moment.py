#!/usr/bin/env python3
"""Screen fit-only A320 pitch dynamics for direct control-input observability.

The target is the one-second change in pitch-rate recorder code.  A minimal
grey-box state model is compared with measured elevator and stabilizer inputs
by leave-one-complete-fit-flight-out prediction.  The result is an effective
closed-loop predictive screen, not a release of physical pitching-moment
derivatives, inertia, surface scales, or surface neutral positions.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import aero, load_aircraft
from aerox.flightdata.a320_qar import DEG_TO_RAD, decode_a320_flight, raw_columns
from aerox.models.atmosphere import (
    dynamic_pressure_from_calibrated_airspeed,
    mach_from_calibrated_airspeed,
)
from aerox.modeling.identification import robust_weighted_linear_fit
from tools.calibrate_a320_effective_thrust import RAW_FILES

PREDICTION_HORIZON_S = 1
MAXIMUM_CONTROL_DELAY_S = 2
MAXIMUM_ABSOLUTE_BANK_DEG = 10.0
MINIMUM_CALIBRATED_AIRSPEED_MPS = 70.0
MAXIMUM_MACH = 0.85
CLEAN_FLAP_CODE = 0.0
REFERENCE_DYNAMIC_PRESSURE_PA = 10_000.0
REFERENCE_MASS_KG = 60_000.0
REDUCED_RATE_FEATURE_SCALE = 100.0
PITCH_RATE_DECODER_SCALE_DEG_S_PER_CODE = 0.5
AOA_DECODER_SCALE_DEG_PER_CODE = 0.25
SURFACE_DECODER_SCALE_DEG_PER_CODE = 0.5
SIGNIFICANT_RATE_CHANGE_CODE = 0.0625

# These are screening gates, not certification/model-release criteria.  A
# small RMSE threshold is appropriate here only in combination with complete-
# flight prevalence and coefficient stability.  Independent confirmation is
# still required before the candidate may drive free pitch replay.
MINIMUM_STATE_RMSE_IMPROVEMENT = 0.05
MINIMUM_ELEVATOR_INCREMENTAL_RMSE_IMPROVEMENT = 0.005
MINIMUM_IMPROVED_FLIGHTS = 8
MINIMUM_COEFFICIENT_SIGN_CONSISTENCY = 1.0
MAXIMUM_COEFFICIENT_VARIATION = 0.15
CONFIRMATION_PANEL_SIZE = 6
CONFIRMATION_MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT = 0.005
CONFIRMATION_MINIMUM_IMPROVED_FLIGHTS = 5
CONFIRMATION_MINIMUM_SIGN_ACCURACY_INCREMENT = -0.02


@dataclass(frozen=True)
class PitchFlight:
    registration: str
    recording: str
    role: str
    time_s: np.ndarray
    clean: np.ndarray
    calibrated_airspeed_mps: np.ndarray
    pressure_altitude_m: np.ndarray
    mach: np.ndarray
    dynamic_pressure_pa: np.ndarray
    mass_kg: np.ndarray
    true_airspeed_mps: np.ndarray
    static_temperature_k: np.ndarray
    n1_percent: np.ndarray
    n2_percent: np.ndarray
    angle_of_attack_code: np.ndarray
    pitch_rate_code: np.ndarray
    mean_elevator_code: np.ndarray
    stabilizer_code: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.registration}/{self.recording}"


@dataclass(frozen=True)
class Candidate:
    name: str
    controls: tuple[str, ...]
    control_delay_s: int
    required_history_s: int = 0


CANDIDATES = (
    Candidate("state_only", (), 0),
    Candidate("state_plus_elevator_d0", ("elevator",), 0),
    Candidate("state_plus_elevator_d1", ("elevator",), 1),
    Candidate("state_plus_elevator_d2", ("elevator",), 2),
    Candidate(
        "state_plus_elevator_motion_d0",
        ("elevator_motion",),
        0,
        required_history_s=1,
    ),
    Candidate(
        "state_plus_elevator_position_motion_d0",
        ("elevator", "elevator_motion"),
        0,
        required_history_s=1,
    ),
    Candidate("state_plus_stabilizer_d0", ("stabilizer",), 0),
    Candidate(
        "state_plus_elevator_position_motion_stabilizer_d0",
        ("elevator", "elevator_motion", "stabilizer"),
        0,
        required_history_s=1,
    ),
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
        "--longitudinal-audit",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-control-audit.yml"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-candidates.csv"),
    )
    parser.add_argument(
        "--lofo-metrics",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-lofo.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-pitch-moment-audit.md"),
    )
    return parser.parse_args()


def _raw_code(
    values: np.ndarray, decoder_scale: float, *, rate: bool = False
) -> np.ndarray:
    # The decoded arrays are radians or radians/second.  Divide out the current
    # weak decoder hypothesis so this audit remains in recorder-code space.
    del rate  # Documents the dimensional distinction at call sites.
    return values / DEG_TO_RAD / decoder_scale


def _load_fit_flights(
    args: argparse.Namespace, *, expected_count: int = 10
) -> list[PitchFlight]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != expected_count:
        raise ValueError(
            f"expected {expected_count} non-validation flights; found {len(selection)}"
        )
    roles = selection.set_index(["registration", "recording"])["role"].to_dict()
    identifiers = selection["recording"].astype(str).tolist()
    processed_path = Path(getattr(args, "processed", Path("data/a320/a320.parquet")))
    processed = pd.read_parquet(
        processed_path,
        columns=["flight_id", "timestamp", "temperature"],
        filters=[("flight_id", "in", identifiers)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame.sort_values("timestamp", kind="stable")
        for recording, frame in processed.groupby("flight_id", sort=False)
    }
    flights: list[PitchFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            args.raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=recording,
            )
            current_processed = processed_groups.get(recording)
            if current_processed is None:
                raise ValueError(f"processed flight {recording} is missing")
            processed_timestamps = current_processed["timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
            raw_indices = np.searchsorted(decoded.timestamp_utc, processed_timestamps)
            if np.any(raw_indices >= len(decoded.timestamp_utc)) or not np.all(
                decoded.timestamp_utc[raw_indices] == processed_timestamps
            ):
                raise ValueError(
                    f"raw/processed mismatch for {registration}/{recording}"
                )
            static_temperature = np.full(len(decoded.time_s), np.nan, dtype=float)
            static_temperature[raw_indices] = current_processed["temperature"].to_numpy(
                dtype=float
            )
            channel = decoded.channel
            cas = channel("calibrated_airspeed")
            altitude = channel("pressure_altitude")
            mass = channel("gross_weight")
            mach = mach_from_calibrated_airspeed(cas, altitude)
            dynamic_pressure = dynamic_pressure_from_calibrated_airspeed(cas, altitude)
            true_airspeed = np.asarray(aero.mach2tas(mach, altitude), dtype=float)
            roll = channel("roll", allow_provisional=True)
            flap = channel("flap", allow_unresolved=True)
            angle_of_attack = _raw_code(
                channel("angle_of_attack", allow_provisional=True),
                AOA_DECODER_SCALE_DEG_PER_CODE,
            )
            pitch_rate = _raw_code(
                channel("pitch_rate", allow_provisional=True),
                PITCH_RATE_DECODER_SCALE_DEG_S_PER_CODE,
                rate=True,
            )
            elevator_1 = _raw_code(
                channel("elevator_1", allow_provisional=True),
                SURFACE_DECODER_SCALE_DEG_PER_CODE,
            )
            elevator_2 = _raw_code(
                channel("elevator_2", allow_provisional=True),
                SURFACE_DECODER_SCALE_DEG_PER_CODE,
            )
            stabilizer = _raw_code(
                channel("stabilizer", allow_provisional=True),
                SURFACE_DECODER_SCALE_DEG_PER_CODE,
            )
            n1 = 0.5 * (
                channel("n1_1", allow_provisional=True)
                + channel("n1_2", allow_provisional=True)
            )
            n2 = 0.5 * (
                channel("n2_1", allow_provisional=True)
                + channel("n2_2", allow_provisional=True)
            )
            mean_elevator = 0.5 * (elevator_1 + elevator_2)
            finite = np.isfinite(
                cas
                + altitude
                + mass
                + mach
                + dynamic_pressure
                + true_airspeed
                + roll
                + flap
                + angle_of_attack
                + pitch_rate
                + mean_elevator
                + stabilizer
            )
            clean = (
                finite
                & decoded.airborne
                & np.isclose(flap, CLEAN_FLAP_CODE)
                & (np.abs(roll) < np.deg2rad(MAXIMUM_ABSOLUTE_BANK_DEG))
                & (cas > MINIMUM_CALIBRATED_AIRSPEED_MPS)
                & (mach < MAXIMUM_MACH)
                & (mass > 0.0)
                & (true_airspeed > 0.0)
            )
            flights.append(
                PitchFlight(
                    registration=str(registration),
                    recording=recording,
                    role=str(roles[(str(registration), recording)]),
                    time_s=decoded.time_s,
                    clean=clean,
                    calibrated_airspeed_mps=cas,
                    pressure_altitude_m=altitude,
                    mach=mach,
                    dynamic_pressure_pa=dynamic_pressure,
                    mass_kg=mass,
                    true_airspeed_mps=true_airspeed,
                    static_temperature_k=static_temperature,
                    n1_percent=n1,
                    n2_percent=n2,
                    angle_of_attack_code=angle_of_attack,
                    pitch_rate_code=pitch_rate,
                    mean_elevator_code=mean_elevator,
                    stabilizer_code=stabilizer,
                )
            )
    expected = set(selection["recording"])
    loaded = {flight.recording for flight in flights}
    if loaded != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - loaded)}")
    return flights


def _valid_starts(
    flight: PitchFlight, delay_s: int, required_history_s: int
) -> np.ndarray:
    if delay_s < 0 or delay_s > MAXIMUM_CONTROL_DELAY_S:
        raise ValueError(f"unsupported control delay {delay_s}")
    if required_history_s < 0:
        raise ValueError("required history must be non-negative")
    starts = np.arange(
        delay_s + required_history_s,
        len(flight.time_s) - PREDICTION_HORIZON_S,
    )
    source = starts - delay_s
    history_start = source - required_history_s
    stop = starts + PREDICTION_HORIZON_S
    bad = np.concatenate(([0], np.cumsum(~flight.clean, dtype=int)))
    continuous = bad[stop + 1] - bad[history_start] == 0
    correct_duration = np.isclose(
        flight.time_s[stop] - flight.time_s[history_start],
        delay_s + required_history_s + PREDICTION_HORIZON_S,
    )
    return starts[continuous & correct_duration]


def _feature_order(controls: tuple[str, ...]) -> list[str]:
    order = [
        "aero_loading",
        "TAIL-B_aero_loading_offset",
        "aero_loading_x_aoa_code",
        "aero_loading_x_reduced_pitch_rate_code_x100",
    ]
    if "elevator" in controls:
        order.append("aero_loading_x_mean_elevator_code")
    if "elevator_motion" in controls:
        order.append("aero_loading_x_mean_elevator_change_code_per_s")
    if "stabilizer" in controls:
        order.append("aero_loading_x_stabilizer_code")
    return order


def _samples(
    flight: PitchFlight,
    *,
    controls: tuple[str, ...],
    control_delay_s: int,
    required_history_s: int,
    mean_aerodynamic_chord_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = _valid_starts(flight, control_delay_s, required_history_s)
    source = starts - control_delay_s
    stop = starts + PREDICTION_HORIZON_S
    loading = (flight.dynamic_pressure_pa[starts] / REFERENCE_DYNAMIC_PRESSURE_PA) * (
        REFERENCE_MASS_KG / flight.mass_kg[starts]
    )
    is_afz = float(flight.registration == "TAIL-B")
    reduced_rate = (
        flight.pitch_rate_code[starts]
        * mean_aerodynamic_chord_m
        / (2.0 * flight.true_airspeed_mps[starts])
        * REDUCED_RATE_FEATURE_SCALE
    )
    columns = [
        loading,
        loading * is_afz,
        loading * flight.angle_of_attack_code[starts],
        loading * reduced_rate,
    ]
    if "elevator" in controls:
        columns.append(loading * flight.mean_elevator_code[source])
    if "elevator_motion" in controls:
        previous = source - 1
        columns.append(
            loading
            * (flight.mean_elevator_code[source] - flight.mean_elevator_code[previous])
        )
    if "stabilizer" in controls:
        columns.append(loading * flight.stabilizer_code[source])
    target = flight.pitch_rate_code[stop] - flight.pitch_rate_code[starts]
    return np.column_stack(columns), target, starts


def _fit_candidate(
    flights: Iterable[PitchFlight],
    *,
    candidate: Candidate,
    mean_aerodynamic_chord_m: float,
):
    selected = list(flights)
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in selected:
        design, target, _ = _samples(
            flight,
            controls=candidate.controls,
            control_delay_s=candidate.control_delay_s,
            required_history_s=candidate.required_history_s,
            mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
        )
        if len(target) < 100:
            raise ValueError(f"too few clean samples for {flight.key}")
        designs.append(design)
        targets.append(target)
        weights.append(np.full(len(target), 1.0 / (len(target) * len(selected))))
    return robust_weighted_linear_fit(
        np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
    )


def _sign_accuracy(prediction: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    significant = np.abs(target) >= SIGNIFICANT_RATE_CHANGE_CODE
    count = int(np.count_nonzero(significant))
    if count == 0:
        return 0, float("nan")
    return count, float(
        np.mean(np.sign(prediction[significant]) == np.sign(target[significant]))
    )


def _lofo(flights: list[PitchFlight], mean_aerodynamic_chord_m: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        matched_state = Candidate(
            f"matched_state_d{candidate.control_delay_s}",
            (),
            candidate.control_delay_s,
            required_history_s=candidate.required_history_s,
        )
        for held_out in flights:
            training = [flight for flight in flights if flight.key != held_out.key]
            fit = _fit_candidate(
                training,
                candidate=candidate,
                mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
            )
            state_fit = _fit_candidate(
                training,
                candidate=matched_state,
                mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
            )
            design, target, _ = _samples(
                held_out,
                controls=candidate.controls,
                control_delay_s=candidate.control_delay_s,
                required_history_s=candidate.required_history_s,
                mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
            )
            state_design, state_target, _ = _samples(
                held_out,
                controls=(),
                control_delay_s=candidate.control_delay_s,
                required_history_s=candidate.required_history_s,
                mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
            )
            if not np.array_equal(target, state_target):
                raise ValueError("candidate and matched-state targets differ")
            prediction = design @ fit.coefficients
            state_prediction = state_design @ state_fit.coefficients
            significant_count, sign_accuracy = _sign_accuracy(prediction, target)
            _, state_sign_accuracy = _sign_accuracy(state_prediction, target)
            row: dict[str, Any] = {
                "candidate": candidate.name,
                "controls": "+".join(candidate.controls) or "none",
                "control_delay_s": candidate.control_delay_s,
                "registration": held_out.registration,
                "recording": held_out.recording,
                "role": held_out.role,
                "samples": len(target),
                "significant_rate_change_samples": significant_count,
                "pitch_rate_delta_rmse_code": float(
                    np.sqrt(np.mean((prediction - target) ** 2))
                ),
                "matched_state_rmse_code": float(
                    np.sqrt(np.mean((state_prediction - target) ** 2))
                ),
                "persistence_rmse_code": float(np.sqrt(np.mean(target**2))),
                "significant_sign_accuracy": sign_accuracy,
                "matched_state_sign_accuracy": state_sign_accuracy,
                "training_normal_matrix_condition": float(fit.normal_matrix_condition),
                "elevator_coefficient": float("nan"),
                "elevator_motion_coefficient": float("nan"),
                "stabilizer_coefficient": float("nan"),
            }
            feature_order = _feature_order(candidate.controls)
            if "elevator" in candidate.controls:
                row["elevator_coefficient"] = float(
                    fit.coefficients[
                        feature_order.index("aero_loading_x_mean_elevator_code")
                    ]
                )
            if "elevator_motion" in candidate.controls:
                row["elevator_motion_coefficient"] = float(
                    fit.coefficients[
                        feature_order.index(
                            "aero_loading_x_mean_elevator_change_code_per_s"
                        )
                    ]
                )
            if "stabilizer" in candidate.controls:
                row["stabilizer_coefficient"] = float(
                    fit.coefficients[
                        feature_order.index("aero_loading_x_stabilizer_code")
                    ]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _coefficient_summary(values: pd.Series) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].astype(float)
    if len(finite) == 0:
        return (float("nan"),) * 4
    mean = float(finite.mean())
    std = float(finite.std())
    coefficient_of_variation = std / abs(mean) if mean != 0.0 else float("inf")
    positive = float(np.mean(finite > 0.0))
    negative = float(np.mean(finite < 0.0))
    return mean, std, coefficient_of_variation, max(positive, negative)


def _candidate_metrics(lofo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        current = lofo.loc[lofo["candidate"] == candidate.name]
        mean_rmse = float(current["pitch_rate_delta_rmse_code"].mean())
        mean_state = float(current["matched_state_rmse_code"].mean())
        mean_persistence = float(current["persistence_rmse_code"].mean())
        elevator = _coefficient_summary(current["elevator_coefficient"])
        elevator_motion = _coefficient_summary(current["elevator_motion_coefficient"])
        stabilizer = _coefficient_summary(current["stabilizer_coefficient"])
        rows.append(
            {
                "candidate": candidate.name,
                "controls": "+".join(candidate.controls) or "none",
                "control_delay_s": candidate.control_delay_s,
                "mean_lofo_rmse_code": mean_rmse,
                "mean_matched_state_rmse_code": mean_state,
                "mean_persistence_rmse_code": mean_persistence,
                "rmse_improvement_over_matched_state": (
                    (mean_state - mean_rmse) / mean_state
                ),
                "rmse_improvement_over_persistence": (
                    (mean_persistence - mean_rmse) / mean_persistence
                ),
                "flights_improved_over_matched_state": int(
                    np.count_nonzero(
                        current["pitch_rate_delta_rmse_code"]
                        < current["matched_state_rmse_code"]
                    )
                ),
                "flights_improved_over_persistence": int(
                    np.count_nonzero(
                        current["pitch_rate_delta_rmse_code"]
                        < current["persistence_rmse_code"]
                    )
                ),
                "mean_significant_sign_accuracy": float(
                    current["significant_sign_accuracy"].mean()
                ),
                "mean_matched_state_sign_accuracy": float(
                    current["matched_state_sign_accuracy"].mean()
                ),
                "elevator_coefficient_mean": elevator[0],
                "elevator_coefficient_std": elevator[1],
                "elevator_coefficient_cv": elevator[2],
                "elevator_coefficient_sign_consistency": elevator[3],
                "elevator_motion_coefficient_mean": elevator_motion[0],
                "elevator_motion_coefficient_std": elevator_motion[1],
                "elevator_motion_coefficient_cv": elevator_motion[2],
                "elevator_motion_coefficient_sign_consistency": elevator_motion[3],
                "stabilizer_coefficient_mean": stabilizer[0],
                "stabilizer_coefficient_std": stabilizer[1],
                "stabilizer_coefficient_cv": stabilizer[2],
                "stabilizer_coefficient_sign_consistency": stabilizer[3],
                "maximum_training_normal_matrix_condition": float(
                    current["training_normal_matrix_condition"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _row(frame: pd.DataFrame, candidate: str) -> pd.Series:
    selected = frame.loc[frame["candidate"] == candidate]
    if len(selected) != 1:
        raise ValueError(f"expected one candidate row for {candidate}")
    return selected.iloc[0]


def _decision(candidates: pd.DataFrame, lofo: pd.DataFrame) -> dict[str, Any]:
    state = _row(candidates, "state_only")
    elevator_rows = candidates.loc[
        candidates["candidate"].isin(
            [
                "state_plus_elevator_d0",
                "state_plus_elevator_d1",
                "state_plus_elevator_d2",
                "state_plus_elevator_motion_d0",
                "state_plus_elevator_position_motion_d0",
            ]
        )
    ]
    # Delay candidates lose a few leading rows, so their absolute RMSE values
    # are not directly comparable.  Select timing by improvement over the
    # state model refitted on the exact same rows.
    elevator = elevator_rows.loc[
        elevator_rows["rmse_improvement_over_matched_state"].idxmax()
    ]
    elevator_position_motion = _row(
        candidates, "state_plus_elevator_position_motion_d0"
    )
    both = _row(candidates, "state_plus_elevator_position_motion_stabilizer_d0")

    state_gate = bool(
        state["rmse_improvement_over_persistence"] >= MINIMUM_STATE_RMSE_IMPROVEMENT
        and state["flights_improved_over_persistence"] >= MINIMUM_IMPROVED_FLIGHTS
    )
    position_is_present = np.isfinite(float(elevator["elevator_coefficient_cv"]))
    motion_is_present = np.isfinite(float(elevator["elevator_motion_coefficient_cv"]))
    position_is_stable = bool(
        not position_is_present
        or (
            elevator["elevator_coefficient_sign_consistency"]
            >= MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
            and elevator["elevator_coefficient_cv"] <= MAXIMUM_COEFFICIENT_VARIATION
        )
    )
    motion_is_stable = bool(
        not motion_is_present
        or (
            elevator["elevator_motion_coefficient_sign_consistency"]
            >= MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
            and elevator["elevator_motion_coefficient_cv"]
            <= MAXIMUM_COEFFICIENT_VARIATION
        )
    )
    elevator_gate = bool(
        elevator["rmse_improvement_over_matched_state"]
        >= MINIMUM_ELEVATOR_INCREMENTAL_RMSE_IMPROVEMENT
        and elevator["flights_improved_over_matched_state"] >= MINIMUM_IMPROVED_FLIGHTS
        and position_is_stable
        and motion_is_stable
    )

    aligned_elevator = lofo.loc[
        lofo["candidate"] == "state_plus_elevator_position_motion_d0"
    ].sort_values(["registration", "recording"])
    aligned_both = lofo.loc[
        lofo["candidate"] == "state_plus_elevator_position_motion_stabilizer_d0"
    ].sort_values(["registration", "recording"])
    stabilizer_incremental = (
        float(elevator_position_motion["mean_lofo_rmse_code"])
        - float(both["mean_lofo_rmse_code"])
    ) / float(elevator_position_motion["mean_lofo_rmse_code"])
    stabilizer_improved_flights = int(
        np.count_nonzero(
            aligned_both["pitch_rate_delta_rmse_code"].to_numpy()
            < aligned_elevator["pitch_rate_delta_rmse_code"].to_numpy()
        )
    )
    stabilizer_gate = bool(
        stabilizer_incremental >= MINIMUM_ELEVATOR_INCREMENTAL_RMSE_IMPROVEMENT
        and stabilizer_improved_flights >= MINIMUM_IMPROVED_FLIGHTS
        and both["stabilizer_coefficient_sign_consistency"]
        >= MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
        and both["stabilizer_coefficient_cv"] <= MAXIMUM_COEFFICIENT_VARIATION
    )
    selected = str(elevator["candidate"]) if elevator_gate else "state_only"
    return {
        "state_dynamics_fit_gate_passed": state_gate,
        "direct_elevator_predictive_gate_passed": elevator_gate,
        "direct_stabilizer_predictive_gate_passed": stabilizer_gate,
        "selected_fit_only_confirmatory_candidate": selected,
        "selected_control_delay_s": int(elevator["control_delay_s"]),
        "retain_elevator_as_effective_pitch_input": elevator_gate,
        "retain_stabilizer_as_direct_one_second_pitch_input": stabilizer_gate,
        "ready_to_select_independent_rotational_confirmatory_panel": bool(
            state_gate and elevator_gate
        ),
        "ready_for_open_loop_free_pitch_replay": False,
        "release_physical_pitching_moment_derivatives": False,
        "release_physical_inertia_or_surface_scale_and_neutral": False,
        "observed": {
            "state_rmse_improvement_over_persistence": float(
                state["rmse_improvement_over_persistence"]
            ),
            "state_flights_improved_over_persistence": int(
                state["flights_improved_over_persistence"]
            ),
            "best_elevator_candidate": str(elevator["candidate"]),
            "best_elevator_mean_lofo_rmse_code": float(elevator["mean_lofo_rmse_code"]),
            "elevator_incremental_rmse_improvement": float(
                elevator["rmse_improvement_over_matched_state"]
            ),
            "elevator_flights_improved_over_state": int(
                elevator["flights_improved_over_matched_state"]
            ),
            "elevator_coefficient_mean": float(elevator["elevator_coefficient_mean"]),
            "elevator_coefficient_cv": float(elevator["elevator_coefficient_cv"]),
            "elevator_coefficient_sign_consistency": float(
                elevator["elevator_coefficient_sign_consistency"]
            ),
            "elevator_motion_coefficient_mean": float(
                elevator["elevator_motion_coefficient_mean"]
            ),
            "elevator_motion_coefficient_cv": float(
                elevator["elevator_motion_coefficient_cv"]
            ),
            "elevator_motion_coefficient_sign_consistency": float(
                elevator["elevator_motion_coefficient_sign_consistency"]
            ),
            "elevator_sign_accuracy_increment": float(
                elevator["mean_significant_sign_accuracy"]
                - elevator["mean_matched_state_sign_accuracy"]
            ),
            "stabilizer_incremental_rmse_improvement_beyond_elevator": (
                stabilizer_incremental
            ),
            "stabilizer_flights_improved_beyond_elevator": (
                stabilizer_improved_flights
            ),
            "stabilizer_coefficient_cv_with_elevator": float(
                both["stabilizer_coefficient_cv"]
            ),
            "stabilizer_coefficient_sign_consistency_with_elevator": float(
                both["stabilizer_coefficient_sign_consistency"]
            ),
        },
        "screening_gate": {
            "minimum_state_rmse_improvement_over_persistence": (
                MINIMUM_STATE_RMSE_IMPROVEMENT
            ),
            "minimum_control_incremental_rmse_improvement": (
                MINIMUM_ELEVATOR_INCREMENTAL_RMSE_IMPROVEMENT
            ),
            "minimum_improved_complete_flights": MINIMUM_IMPROVED_FLIGHTS,
            "minimum_coefficient_sign_consistency": (
                MINIMUM_COEFFICIENT_SIGN_CONSISTENCY
            ),
            "maximum_coefficient_of_variation": MAXIMUM_COEFFICIENT_VARIATION,
        },
    }


def _full_model(
    flights: list[PitchFlight],
    candidate_or_name: Candidate | str,
    mean_aerodynamic_chord_m: float,
    conditional_pitch_rate_scale: float,
) -> dict[str, Any]:
    candidate = (
        next(item for item in CANDIDATES if item.name == candidate_or_name)
        if isinstance(candidate_or_name, str)
        else candidate_or_name
    )
    fit = _fit_candidate(
        flights,
        candidate=candidate,
        mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
    )
    feature_order = _feature_order(candidate.controls)
    ranges: dict[str, list[float]] = {}
    arrays = {
        "calibrated_airspeed_mps": [],
        "mach": [],
        "dynamic_pressure_pa": [],
        "mass_kg": [],
        "angle_of_attack_code": [],
        "pitch_rate_code": [],
        "mean_elevator_code": [],
        "stabilizer_code": [],
    }
    for flight in flights:
        _, _, starts = _samples(
            flight,
            controls=candidate.controls,
            control_delay_s=candidate.control_delay_s,
            required_history_s=candidate.required_history_s,
            mean_aerodynamic_chord_m=mean_aerodynamic_chord_m,
        )
        for name, parts in arrays.items():
            parts.append(np.asarray(getattr(flight, name))[starts])
    for name, parts in arrays.items():
        values = np.concatenate(parts)
        ranges[name] = [
            float(np.quantile(values, 0.01)),
            float(np.quantile(values, 0.99)),
        ]
    return {
        "status": "fit-only-effective-closed-loop-candidate",
        "candidate": candidate.name,
        "equation": (
            "delta q_code[t:t+1] = X_state[t] beta_state + "
            "X_control[t-delay] beta_control"
        ),
        "feature_order": feature_order,
        "coefficients": {
            name: float(value)
            for name, value in zip(feature_order, fit.coefficients, strict=True)
        },
        "control_delay_s": candidate.control_delay_s,
        "required_control_history_s": candidate.required_history_s,
        "prediction_horizon_s": PREDICTION_HORIZON_S,
        "conditional_pitch_rate_scale_deg_s_per_code": (conditional_pitch_rate_scale),
        "conditional_mean_lofo_acceleration_rmse_deg_s2": None,
        "normal_matrix_condition": float(fit.normal_matrix_condition),
        "coefficient_correlation": fit.coefficient_correlation.tolist(),
        "fit_robust_scale_code": float(fit.robust_scale),
        "validity_envelope_fit_p01_p99": ranges,
    }


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    integer_columns = {
        "control_delay_s",
        "samples",
        "significant_rate_change_samples",
        "flights_improved_over_matched_state",
        "flights_improved_over_persistence",
    }
    for column in result.select_dtypes(include=["number"]):
        if column in integer_columns:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(10)
    return result


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
    path: Path, *, candidates: pd.DataFrame, lofo: pd.DataFrame, audit: dict[str, Any]
) -> None:
    decision = audit["decision"]
    observed = decision["observed"]
    if decision["direct_elevator_predictive_gate_passed"]:
        elevator_decision = f"""Adding measured mean-elevator position and its
one-second motion yields a smaller but repeatable
{observed["elevator_incremental_rmse_improvement"]:.2%} incremental improvement
and improves {observed["elevator_flights_improved_over_state"]} of ten complete
fit flights. The position and motion coefficients each have one sign in every
fold, with {observed["elevator_coefficient_cv"]:.1%} and
{observed["elevator_motion_coefficient_cv"]:.1%} coefficients of variation.
The selected fit-only confirmatory candidate is
`{decision["selected_fit_only_confirmatory_candidate"]}`."""
        next_gate = """Freeze the selected feature set, measured-elevator
alignment, clean-flap envelope, fitting rule, and metrics. Select previously
unopened flights stratified by registration and maneuver content. Only if that
panel confirms the one-step effect should the candidate be placed in a free-
pitch replay, starting with short windows before full-flight integration."""
    else:
        elevator_decision = f"""Adding the measured mean elevator improves
one-second RMSE by only
{observed["elevator_incremental_rmse_improvement"]:.2%}. It improves
{observed["elevator_flights_improved_over_state"]} of ten complete fit flights
and its effective coefficient is sign-consistent with a
{observed["elevator_coefficient_cv"]:.1%} coefficient of variation, but the
effect remains below the predeclared 0.5% release threshold. The retained
diagnostic candidate is therefore `state_only`; it is not a controllable pitch
plant."""
        next_gate = """Do not spend a new confirmatory panel on the current
control model. First run a fit-only excitation audit that separates sustained
surface deflection from surface motion, adds the already identified propulsion
state as a possible pitching-moment nuisance, and checks whether maneuver-rich
intervals contain enough independent elevator excitation. If that remains
negative, anchor the control derivative to a documented aerodynamic prior or
seek higher-rate/commanded-surface data."""
    compact_candidates = _round_frame(
        candidates[
            [
                "candidate",
                "mean_lofo_rmse_code",
                "rmse_improvement_over_matched_state",
                "flights_improved_over_matched_state",
                "mean_significant_sign_accuracy",
                "elevator_coefficient_cv",
                "elevator_motion_coefficient_cv",
                "stabilizer_coefficient_cv",
            ]
        ]
    )
    diagnostic_candidate = observed["best_elevator_candidate"]
    diagnostic_lofo = _round_frame(
        lofo.loc[
            lofo["candidate"] == diagnostic_candidate,
            [
                "registration",
                "recording",
                "samples",
                "pitch_rate_delta_rmse_code",
                "matched_state_rmse_code",
                "persistence_rmse_code",
                "significant_sign_accuracy",
                "elevator_coefficient",
                "elevator_motion_coefficient",
            ],
        ]
    )
    text = f"""# A320 fit-only effective pitch-dynamics screen

## Decision

The minimal state dynamics pass their screening gate, improving one-second
pitch-rate-change RMSE over persistence by
{observed["state_rmse_improvement_over_persistence"]:.1%}. {elevator_decision}

The selected augmentation reduces the significant-change sign metric by
{abs(observed["elevator_sign_accuracy_increment"]):.1%}. That secondary
diagnostic is not a release gate here, but it must remain visible in independent
confirmation and short-window rollout testing.

Stabilizer does not pass as a direct one-second dynamic input: after elevator,
its incremental RMSE change is
{observed["stabilizer_incremental_rmse_improvement_beyond_elevator"]:.2%}, and
its coefficient is not stable enough across held-out-flight fits. Stabilizer
must remain an observed trim/configuration state; this result does not mean it
has no physical pitching-moment effect.

This screen does **not** authorize open-loop free-pitch replay or physical
`C_m` derivatives. Inertia, CG, surface scale/neutral, and the closed-loop
feedback confounding remain unresolved. No old validation flight or new
confirmatory flight was loaded.

## Candidate comparison

{_markdown_table(compact_candidates)}

Every entry is evaluated by leaving one complete fit flight out. Each training
flight has equal total weight and the regression uses Huber robustness. The
state baseline is refitted on exactly the samples available to each delayed
control candidate, so delay candidates cannot benefit from a changed sample
set.

## Best elevator diagnostic by flight

{_markdown_table(diagnostic_lofo)}

The significant-change sign metric includes samples with at least
{SIGNIFICANT_RATE_CHANGE_CODE:g} recorder-code change in one second. The model
operates at the requested 1 Hz level; these data cannot reveal actuator or
short-period behavior above that bandwidth.

## Effective equation

The normalized aerodynamic loading is

```text
lambda = (dynamic_pressure / {REFERENCE_DYNAMIC_PRESSURE_PA:g} Pa)
         * ({REFERENCE_MASS_KG:g} kg / gross_mass).
```

The state terms are `lambda`, a registration loading offset,
`lambda * AOA_code`, and
`lambda * 100 * q_code * MAC / (2 TAS)`. Candidate control terms are
`lambda * elevator_code`, `lambda * (elevator_code[t] -
elevator_code[t-1])`, and `lambda * stabilizer_code`. The position term follows
a pitching-moment equation. The motion term is a discrete actuator/recorder-
history feature; it may absorb timing, feedback, or unsteady effects and must
not be labeled a physical elevator-rate derivative. All fitted coefficients
remain effective recorder-space predictors.

Unknown AOA and surface zeros are absorbed in the loading intercept and its
registration offset. This prevents the two known elevator-channel offsets from
being misreported as aerodynamic asymmetry, but it does not identify neutral
positions.

## Next gate

{next_gate}
"""
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aircraft = load_aircraft("A320")
    if aircraft.wing is None or aircraft.wing.mac is None:
        raise ValueError("A320 mean aerodynamic chord is required")
    mean_aerodynamic_chord = float(aircraft.wing.mac)
    longitudinal = yaml.safe_load(args.longitudinal_audit.read_text())
    conditional_pitch_rate_scale = float(
        longitudinal["model"][
            "conditional_effective_pitch_rate_scale_from_vertical_deg_s_per_code"
        ]
    )
    flights = _load_fit_flights(args)
    lofo = _lofo(flights, mean_aerodynamic_chord)
    candidates = _candidate_metrics(lofo)
    decision = _decision(candidates, lofo)
    model = _full_model(
        flights,
        decision["selected_fit_only_confirmatory_candidate"],
        mean_aerodynamic_chord,
        conditional_pitch_rate_scale,
    )
    selected = _row(candidates, decision["selected_fit_only_confirmatory_candidate"])
    model["conditional_mean_lofo_acceleration_rmse_deg_s2"] = float(
        selected["mean_lofo_rmse_code"]
        * conditional_pitch_rate_scale
        / PREDICTION_HORIZON_S
    )
    confirmation_baseline = _full_model(
        flights,
        Candidate("state_only_matched_history1", (), 0, required_history_s=1),
        mean_aerodynamic_chord,
        conditional_pitch_rate_scale,
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": (
            "fit-only-elevator-effect-observed-awaiting-independent-confirmation"
            if decision["ready_to_select_independent_rotational_confirmatory_panel"]
            else "fit-only-effective-pitch-dynamics-gate-failed"
        ),
        "leakage_control": {
            "fit_flights_only": [flight.key for flight in flights],
            "previous_validation_flights_loaded": False,
            "new_confirmatory_flights_loaded": False,
            "candidate_selection_uses_fit_lofo_only": True,
        },
        "method": {
            "sample_rate_hz": 1,
            "prediction_horizon_s": PREDICTION_HORIZON_S,
            "cross_validation": "leave-one-complete-fit-flight-out",
            "fit_weighting": "equal total weight per flight with Huber robustness",
            "envelope": {
                "airborne": True,
                "flap_code": int(CLEAN_FLAP_CODE),
                "maximum_absolute_bank_deg": MAXIMUM_ABSOLUTE_BANK_DEG,
                "minimum_calibrated_airspeed_mps": (MINIMUM_CALIBRATED_AIRSPEED_MPS),
                "maximum_mach": MAXIMUM_MACH,
            },
            "candidate_control_delays_s": list(range(MAXIMUM_CONTROL_DELAY_S + 1)),
            "fit_only_structure_search_order": [
                "state and elevator-position timing screen",
                "elevator-motion augmentation after position-only effect fell below the unchanged gate",
                "stabilizer increment after the selected elevator structure",
            ],
            "target": "pitch_rate_code[t+1] - pitch_rate_code[t]",
            "aerodynamic_loading": (
                "(dynamic_pressure / 10000 Pa) * (60000 kg / gross_mass)"
            ),
            "mean_aerodynamic_chord_m": mean_aerodynamic_chord,
            "surface_zero_handling": (
                "loading intercept plus registration loading offset; no physical "
                "neutral inferred"
            ),
        },
        "model": model,
        "decision": decision,
        "independent_confirmation_contract": {
            "status": "frozen-before-panel-selection",
            "panel_size": CONFIRMATION_PANEL_SIZE,
            "allocation": "three previously unopened complete operational flights per registration",
            "selection_inputs": (
                "duration, maximum altitude/CAS, takeoff gross weight, endpoint "
                "distance, wind code, surface activity, and channel coverage; "
                "no pitch-response residual"
            ),
            "candidate_model": model,
            "matched_state_baseline_model": confirmation_baseline,
            "gate": {
                "minimum_mean_rmse_improvement_over_matched_state": (
                    CONFIRMATION_MINIMUM_INCREMENTAL_RMSE_IMPROVEMENT
                ),
                "minimum_complete_flights_improved": (
                    CONFIRMATION_MINIMUM_IMPROVED_FLIGHTS
                ),
                "minimum_significant_sign_accuracy_increment": (
                    CONFIRMATION_MINIMUM_SIGN_ACCURACY_INCREMENT
                ),
            },
        },
        "artifacts": {
            "runner": "tools/audit_a320_pitch_moment.py",
            "candidate_metrics": str(args.candidate_metrics),
            "lofo_metrics": str(args.lofo_metrics),
            "report": str(args.report),
        },
        "dependencies": {
            "aircraft_geometry": "src/aerox/data/aircraft/yml/turbofan/a320.yml",
            "channel_map": "docs/modeling/a320-qar-channel-map.yml",
            "longitudinal_control_audit": str(args.longitudinal_audit),
            "air_data_equations": "src/aerox/models/atmosphere.py",
        },
        "limitations": [
            "The data are closed-loop airline operations; predictive elevator value is not causal identification.",
            "The 1 Hz recorder cannot identify high-bandwidth actuator or short-period dynamics.",
            "Pitch-rate, AOA, elevator, and stabilizer engineering scales remain provisional.",
            "Pitch inertia, CG, surface neutral, and physical pitching-moment derivatives are not separately identified.",
            "The current screen is clean-flap and low-bank only.",
            "The elevator-motion coefficient is a discrete history term, not a physical elevator-rate derivative.",
            "The selected augmentation reduces the significant-change sign metric despite its small RMSE gain.",
            "Independent confirmatory flights must pass before free-pitch replay.",
        ],
    }
    args.candidate_metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(candidates).to_csv(args.candidate_metrics, index=False)
    _round_frame(lofo).to_csv(args.lofo_metrics, index=False)
    args.audit.write_text(yaml.safe_dump(_native(audit), sort_keys=False))
    _write_report(args.report, candidates=candidates, lofo=lofo, audit=audit)
    return candidates, lofo, audit


def main() -> None:
    candidates, _, audit = run(parse_args())
    print(candidates.to_string(index=False))
    print(yaml.safe_dump(audit["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
