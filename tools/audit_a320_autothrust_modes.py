#!/usr/bin/env python3
"""Audit A320 autothrust-mode and speed-target observability.

Only the ten development/calibration flights are loaded.  Published FGC/DMC
bit assignments are compared with the raw schema first.  Because the normal
FGC autothrust words are absent, unresolved recorded bits are screened only as
anonymous selectors.  They are never assigned mode semantics from statistical
correlation alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from aerox import load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.flightdata.a320_qar import (
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.models.thrust import SpeedTrackingAutothrustController
from tools.calibrate_a320_effective_thrust import (
    PROCESSED_COLUMNS,
    RAW_FILES,
    _load_flights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.calibrate_a320_speed_autothrust import (
    _attach_guidance,
    _contiguous_windows,
    _ordered_raw,
    _read_envelope,
    _rollout_rmse,
)


RESPONSE_TIME_CONSTANTS_S = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
MINIMUM_EVALUABLE_FLIGHTS = 8
MINIMUM_TRAIN_FLIGHTS = 7
MINIMUM_MEAN_PERSISTENCE_IMPROVEMENT = 0.10
MINIMUM_IMPROVED_FLIGHTS = 7

SCREEN_PREFIXES = (
    "FGC.",
    "FCU.",
    "DMC.1.270.",
    "DMC.1.271.",
    "FWC.1.126.",
)

# These assignments come from the published parameter-list mirror cited in
# docs/modeling/sources.yml.  The mirror's provenance is incomplete, so the
# assignments are used to inventory presence/absence, not to derive model
# coefficients.
EXPECTED_MODE_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("autothrust_engaged", "FGC.1.270.00.13", "FGC/270/00 bit 13"),
    ("autothrust_active", "FGC.1.270.00.14", "FGC/270/00 bit 14"),
    ("speed_mach_mode", "FGC.1.270.00.18", "FGC/270/00 bit 18"),
    ("retard_mode", "FGC.1.270.00.20", "FGC/270/00 bit 20"),
    ("n1_mode", "FGC.1.270.00.21", "FGC/270/00 bit 21"),
    ("toga_displayed", "FGC.1.271.00.11", "FGC/271/00 bit 11"),
    ("mct_displayed", "FGC.1.271.00.12", "FGC/271/00 bit 12"),
    ("thr_displayed", "FGC.1.271.00.15", "FGC/271/00 bit 15"),
    ("idle_displayed", "FGC.1.271.00.16", "FGC/271/00 bit 16"),
    ("alpha_floor_displayed", "FGC.1.271.00.17", "FGC/271/00 bit 17"),
    ("speed_displayed", "FGC.1.271.00.19", "FGC/271/00 bit 19"),
    ("asym_displayed", "FGC.1.271.00.21", "FGC/271/00 bit 21"),
)

PUBLISHED_AUXILIARY_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("retard_displayed_dmc", "DMC.1.271.01.18", "DMC/271/xx bit 18"),
    ("autothrust_off_warning", "FWC.1.126.10.12", "FWC/126/10 bit 12"),
    ("autopilot_flare", "FGC.1.274.00.25", "FGC/274/00 bit 25"),
    ("reverser_1_unlocked", "DMC.1.270.01.11", "DMC/270/xx bit 11"),
    ("reverser_2_unlocked", "DMC.1.270.01.12", "DMC/270/xx bit 12"),
    ("reverser_1_deployed", "DMC.1.270.01.13", "DMC/270/xx bit 13"),
    ("reverser_2_deployed", "DMC.1.270.01.14", "DMC/270/xx bit 14"),
)

PUBLISHED_SEMANTICS = {
    channel: name for name, channel, _ in PUBLISHED_AUXILIARY_CHANNELS
}


@dataclass(frozen=True)
class ModeAuditFlight:
    speed: Any
    bit_values: dict[str, np.ndarray]

    @property
    def key(self) -> str:
        return self.speed.key


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
        "--envelope-parameters",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-mode-channel-inventory.csv"),
    )
    parser.add_argument(
        "--selector-metrics",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-mode-selector-metrics.csv"),
    )
    parser.add_argument(
        "--fold-metrics",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-mode-selector-lofo.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-mode-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-mode-audit.md"),
    )
    return parser.parse_args()


def _raw_schema(raw_dir: Path) -> tuple[str, ...]:
    schemas = {
        registration: tuple(
            pq.ParquetFile(raw_dir / filename).schema_arrow.names
        )
        for registration, filename in RAW_FILES.items()
    }
    reference = schemas[next(iter(schemas))]
    if any(set(schema) != set(reference) for schema in schemas.values()):
        raise ValueError("registration raw schemas differ")
    return reference


def _load_fit_flights(
    args: argparse.Namespace,
) -> tuple[list[ModeAuditFlight], tuple[str, ...], pd.DataFrame]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(selection)}")

    schema = _raw_schema(args.raw_dir)
    screen_channels = tuple(
        sorted(name for name in schema if name.startswith(SCREEN_PREFIXES))
    )
    vertical = _read_vertical_calibration(args.vertical_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    if aircraft.wing is None or aircraft.drag is None:
        raise ValueError("A320 definition lacks wing or drag data")
    base_flights = _load_flights(
        selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=float(aircraft.wing.area),
    )
    speed_flights = _attach_guidance(
        base_flights,
        selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        envelope=_read_envelope(args.envelope_parameters),
        cd0=reference_drag.zero_lift_drag_coefficient,
        induced_drag_factor=reference_drag.induced_drag_factor,
    )
    base_by_key = {flight.key: flight for flight in base_flights}
    speed_by_key = {flight.key: flight for flight in speed_flights}

    identifiers = selection["recording"].astype(str).tolist()
    processed = pd.read_parquet(
        args.processed,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "in", identifiers)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame
        for recording, frame in processed.groupby("flight_id", sort=False)
    }
    required = list(dict.fromkeys((*raw_columns(), *screen_channels)))
    output: list[ModeAuditFlight] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            args.raw_dir / RAW_FILES[str(registration)],
            columns=required,
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            key = f"{registration}/{recording}"
            base = base_by_key[key]
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=recording,
            )
            current_processed = processed_groups[recording].sort_values(
                "timestamp", kind="stable"
            )
            timestamps = current_processed["timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
            raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
            if np.any(raw_indices >= len(decoded.time_s)) or not np.all(
                decoded.timestamp_utc[raw_indices] == timestamps
            ):
                raise ValueError(f"raw/processed mismatch for {key}")
            bit_values = {
                channel: bounded_sample_and_hold(
                    _ordered_raw(frame, channel), decoded.time_s, 1.0
                )[raw_indices][base.sample_index]
                for channel in screen_channels
            }
            output.append(
                ModeAuditFlight(speed=speed_by_key[key], bit_values=bit_values)
            )
    expected = set(selection["recording"])
    loaded = {flight.speed.recording for flight in output}
    if loaded != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - loaded)}")
    return output, schema, selection


def _inventory(
    flights: list[ModeAuditFlight], schema: tuple[str, ...]
) -> pd.DataFrame:
    expected = {
        channel: ("required-mode", parameter, source)
        for parameter, channel, source in EXPECTED_MODE_CHANNELS
    }
    expected.update(
        {
            channel: ("published-auxiliary", parameter, source)
            for parameter, channel, source in PUBLISHED_AUXILIARY_CHANNELS
        }
    )
    rows: list[dict[str, Any]] = []
    schema_set = set(schema)
    channels = sorted(set(expected) | set(flights[0].bit_values))
    total_baseline = sum(
        np.count_nonzero(flight.speed.variable_mode_mask) for flight in flights
    )
    total_duration_h = sum(
        (flight.speed.time_s[-1] - flight.speed.time_s[0]) / 3600.0
        for flight in flights
    )
    for channel in channels:
        category, parameter, source = expected.get(
            channel,
            (
                "anonymous-screen",
                PUBLISHED_SEMANTICS.get(channel, "unresolved"),
                "not-published-for-mode-use",
            ),
        )
        values: list[np.ndarray] = []
        changes = 0
        if channel in flights[0].bit_values:
            for flight in flights:
                mask = flight.speed.variable_mode_mask
                current = flight.bit_values[channel][mask]
                finite = current[np.isfinite(current)]
                values.append(finite)
                changes += int(np.count_nonzero(np.diff(finite)))
        joined = np.concatenate(values) if values else np.array([], dtype=float)
        unique = np.unique(joined)
        rows.append(
            {
                "raw_channel": channel,
                "category": category,
                "parameter": parameter,
                "published_assignment": source,
                "present_in_both_raw_files": channel in schema_set,
                "baseline_native_coverage_fraction": (
                    len(joined) / total_baseline if total_baseline else np.nan
                ),
                "unique_values_on_baseline_screen": len(unique),
                "one_fraction_on_baseline_screen": (
                    float(np.mean(np.isclose(joined, 1.0)))
                    if len(joined)
                    else np.nan
                ),
                "changes_per_flight_hour": (
                    changes / total_duration_h if total_duration_h else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _selector_definitions(
    flights: list[ModeAuditFlight], screen_channels: tuple[str, ...]
) -> list[tuple[str, str]]:
    selectors = [("baseline", "all")]
    for channel in screen_channels:
        values = np.concatenate(
            [
                flight.bit_values[channel][flight.speed.variable_mode_mask]
                for flight in flights
            ]
        )
        finite = values[np.isfinite(values)]
        if len(finite) and np.any(np.isclose(finite, 0.0)) and np.any(
            np.isclose(finite, 1.0)
        ):
            selectors.extend(((channel, "0"), (channel, "1")))
    return selectors


def _selector_gate(
    flight: ModeAuditFlight, channel: str, state: str
) -> np.ndarray:
    if channel == "baseline":
        return flight.speed.variable_mode_mask.copy()
    return flight.speed.variable_mode_mask & np.isclose(
        flight.bit_values[channel], float(state)
    )


def _response_records(
    flights: list[ModeAuditFlight], selectors: list[tuple[str, str]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    controllers = {
        tau: SpeedTrackingAutothrustController(speed_error_gain_per_s=1.0 / tau)
        for tau in RESPONSE_TIME_CONSTANTS_S
    }
    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for channel, state in selectors:
        records: list[dict[str, Any]] = []
        for flight in flights:
            mask = _selector_gate(flight, channel, state)
            gated = replace(flight.speed, variable_mode_mask=mask)
            windows = _contiguous_windows(gated)
            target_error = (
                gated.target_true_airspeed_mps[mask]
                - gated.true_airspeed_mps[mask]
            )
            record: dict[str, Any] = {
                "registration": gated.registration,
                "recording": gated.recording,
                "key": gated.key,
                "samples": int(np.count_nonzero(mask)),
                "windows": len(windows),
                "target_error_median_mps": (
                    float(np.median(target_error)) if len(target_error) else np.nan
                ),
                "target_error_rmse_mps": (
                    float(np.sqrt(np.mean(target_error**2)))
                    if len(target_error)
                    else np.nan
                ),
            }
            if windows:
                persistence, _, _ = _rollout_rmse(gated, windows, None)
                record["persistence_rmse_mps"] = persistence
                for tau, controller in controllers.items():
                    response, _, _ = _rollout_rmse(
                        gated, windows, controller
                    )
                    record[f"response_{tau:g}_rmse_mps"] = response
            else:
                record["persistence_rmse_mps"] = np.nan
                for tau in RESPONSE_TIME_CONSTANTS_S:
                    record[f"response_{tau:g}_rmse_mps"] = np.nan
            records.append(record)
        output[(channel, state)] = records
    return output


def _leave_one_flight_out(
    responses: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for (channel, state), records in responses.items():
        available = [row for row in records if row["windows"] > 0]
        for held_out in available:
            training = [row for row in available if row["key"] != held_out["key"]]
            if len(training) < MINIMUM_TRAIN_FLIGHTS:
                continue
            scores = {
                tau: float(
                    np.mean(
                        [row[f"response_{tau:g}_rmse_mps"] for row in training]
                    )
                )
                for tau in RESPONSE_TIME_CONSTANTS_S
            }
            selected_tau = min(scores, key=scores.get)
            response = float(held_out[f"response_{selected_tau:g}_rmse_mps"])
            persistence = float(held_out["persistence_rmse_mps"])
            fold_rows.append(
                {
                    "selector_channel": channel,
                    "selector_state": state,
                    "registration": held_out["registration"],
                    "recording": held_out["recording"],
                    "train_flights": len(training),
                    "held_out_windows": held_out["windows"],
                    "selected_response_time_s": selected_tau,
                    "persistence_rmse_mps": persistence,
                    "feedback_rmse_mps": response,
                    "persistence_relative_improvement": (
                        (persistence - response) / persistence
                    ),
                }
            )
        folds = [
            row
            for row in fold_rows
            if row["selector_channel"] == channel
            and row["selector_state"] == state
        ]
        samples = sum(row["samples"] for row in records)
        windows = sum(row["windows"] for row in records)
        target_errors = [
            row["target_error_rmse_mps"]
            for row in records
            if np.isfinite(row["target_error_rmse_mps"])
        ]
        aggregate_rows.append(
            {
                "selector_channel": channel,
                "selector_state": state,
                "interpretation": (
                    "ungated reference"
                    if channel == "baseline"
                    else PUBLISHED_SEMANTICS.get(channel, "anonymous unresolved bit")
                ),
                "selected_samples": samples,
                "selected_windows": windows,
                "evaluable_flights": len(folds),
                "mean_target_error_rmse_mps": (
                    float(np.mean(target_errors)) if target_errors else np.nan
                ),
                "mean_lofo_persistence_rmse_mps": (
                    float(np.mean([row["persistence_rmse_mps"] for row in folds]))
                    if folds
                    else np.nan
                ),
                "mean_lofo_feedback_rmse_mps": (
                    float(np.mean([row["feedback_rmse_mps"] for row in folds]))
                    if folds
                    else np.nan
                ),
                "mean_lofo_persistence_relative_improvement": (
                    float(
                        np.mean(
                            [
                                row["persistence_relative_improvement"]
                                for row in folds
                            ]
                        )
                    )
                    if folds
                    else np.nan
                ),
                "held_out_flights_improved": int(
                    np.count_nonzero(
                        [
                            row["persistence_relative_improvement"] > 0.0
                            for row in folds
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(fold_rows)


def _decision(
    inventory: pd.DataFrame, selector_metrics: pd.DataFrame
) -> dict[str, Any]:
    required = inventory.loc[inventory["category"] == "required-mode"]
    direct_present = int(required["present_in_both_raw_files"].sum())
    candidates = selector_metrics.loc[
        (selector_metrics["selector_channel"] != "baseline")
        & (selector_metrics["evaluable_flights"] >= MINIMUM_EVALUABLE_FLIGHTS)
    ].copy()
    if len(candidates):
        best = candidates.sort_values(
            [
                "mean_lofo_persistence_relative_improvement",
                "held_out_flights_improved",
                "selected_samples",
            ],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        statistical_passed = bool(
            best["mean_lofo_persistence_relative_improvement"]
            >= MINIMUM_MEAN_PERSISTENCE_IMPROVEMENT
            and best["held_out_flights_improved"] >= MINIMUM_IMPROVED_FLIGHTS
        )
        best_record: dict[str, Any] | None = _native(best.to_dict())
    else:
        statistical_passed = False
        best_record = None
    direct_observable = bool(
        required.set_index("parameter").loc[
            ["autothrust_active", "speed_mach_mode"],
            "present_in_both_raw_files",
        ].all()
    )
    return {
        "direct_required_channels_present": direct_present,
        "direct_required_channels_total": len(required),
        "direct_autothrust_mode_observable": direct_observable,
        "best_indirect_selector": best_record,
        "statistical_selector_gate_passed": statistical_passed,
        "gate": {
            "minimum_evaluable_flights": MINIMUM_EVALUABLE_FLIGHTS,
            "minimum_mean_persistence_improvement": (
                MINIMUM_MEAN_PERSISTENCE_IMPROVEMENT
            ),
            "minimum_improved_flights": MINIMUM_IMPROVED_FLIGHTS,
        },
        "speed_target_identity_observable": bool(
            direct_observable and statistical_passed
        ),
        "release_for_controller_refit": False,
        "open_new_confirmatory_panel": bool(statistical_passed),
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
    for column in result.select_dtypes(include=["number"]):
        if column in {
            "unique_values_on_baseline_screen",
            "selected_samples",
            "selected_windows",
            "evaluable_flights",
            "held_out_flights_improved",
            "train_flights",
            "held_out_windows",
        }:
            result[column] = result[column].astype("Int64")
        else:
            result[column] = result[column].round(8)
    return result


def _markdown_table(frame: pd.DataFrame) -> str:
    strings = frame.astype(str)
    headers = [str(column) for column in strings.columns]
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
    inventory: pd.DataFrame,
    selector_metrics: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    required = _round_frame(
        inventory.loc[
            inventory["category"].isin(["required-mode", "published-auxiliary"]),
            [
                "parameter",
                "raw_channel",
                "published_assignment",
                "present_in_both_raw_files",
            ],
        ]
    )
    eligible = selector_metrics.loc[
        selector_metrics["evaluable_flights"] >= MINIMUM_EVALUABLE_FLIGHTS
    ].sort_values(
        "mean_lofo_persistence_relative_improvement",
        ascending=False,
        kind="stable",
    )
    top = _round_frame(
        eligible.head(15)[
            [
                "selector_channel",
                "selector_state",
                "interpretation",
                "evaluable_flights",
                "selected_windows",
                "mean_lofo_persistence_rmse_mps",
                "mean_lofo_feedback_rmse_mps",
                "mean_lofo_persistence_relative_improvement",
                "held_out_flights_improved",
            ]
        ]
    )
    best = decision["best_indirect_selector"]
    if best is None:
        best_text = "No indirect selector is evaluable on eight complete fit flights."
    else:
        best_text = (
            f"The best indirect candidate is `{best['selector_channel']}="
            f"{best['selector_state']}`. Its mean leave-one-flight-out improvement "
            f"over persistence is "
            f"{best['mean_lofo_persistence_relative_improvement']:.1%}, with "
            f"{best['held_out_flights_improved']}/{best['evaluable_flights']} "
            "held-out fit flights improved."
        )
    text = f"""# A320 autothrust mode and target observability audit

## Decision

The normal autothrust mode is **not directly observable** in these QAR files.
None of the {decision['direct_required_channels_total']} published FGC 270/271
mode bits needed to distinguish A/THR active, SPEED/MACH, and fixed-thrust
annunciations is present in either registration's raw schema.

{best_text} The predeclared statistical selector gate is
{'passed' if decision['statistical_selector_gate_passed'] else 'not passed'}.
An anonymous correlated bit is not assigned Airbus mode semantics, and no
controller refit or new confirmatory-flight panel is authorized by this audit.

Only the ten development/calibration flights were loaded. The six previously
opened validation flights were not reused, and no new untouched flight was
opened.

## Published channel inventory

{_markdown_table(required)}

`DMC.1.271.01.18` is the only recorded published autothrust annunciation in
this inventory: RETARD during landing flare. `FGC.1.274.00.25` is autopilot
FLARE, not the normal autothrust SPEED/MACH selector. The recorded FGC labels
273/274 are therefore not substitutes for the absent FGC 270/271 words.

## Anonymous selector screen

{_markdown_table(top)}

Each binary state had to remain continuous for a full 300-second rollout.
Controller response time was selected from
`{', '.join(f'{value:g}' for value in RESPONSE_TIME_CONSTANTS_S)} s` on the
other fit flights, then scored on the held-out complete flight. Persistence
holds the window's initial TAS. At least {MINIMUM_EVALUABLE_FLIGHTS} flights,
{MINIMUM_MEAN_PERSISTENCE_IMPROVEMENT:.0%} mean improvement, and
{MINIMUM_IMPROVED_FLIGHTS} improved flights are required.

The screen is deliberately diagnostic. FCU/FGC/DMC/FWC bits without published
semantics remain anonymous even if correlated with flight phase, N1, or speed
error. Such correlations can arise because the active vertical mode determines
whether autothrust controls speed or supplies fixed thrust.

## Consequence for the model

- Keep measured N1 as the plant input for trajectory-replay identification.
- Keep the validated CLB N1 envelope as a fixed-thrust boundary.
- Do not treat `SPEEDP_SEL` as the active target without a mode/target selector.
- Do not tune another speed-feedback law against the already-opened validation
  flights.
- The clean way to unblock the controller is a parameter export containing FGC
  labels 270/271 (or named FMA/A/THR fields), plus the managed FMGC speed target.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    flights, schema, selection = _load_fit_flights(args)
    screen_channels = tuple(flights[0].bit_values)
    inventory = _inventory(flights, schema)
    selectors = _selector_definitions(flights, screen_channels)
    responses = _response_records(flights, selectors)
    selector_metrics, fold_metrics = _leave_one_flight_out(responses)
    decision = _decision(inventory, selector_metrics)

    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": "Airbus A320-214",
        "status": (
            "mode-and-target-observable"
            if decision["speed_target_identity_observable"]
            else "required-fma-mode-words-absent"
        ),
        "leakage_control": {
            "fit_flights_only": [flight.key for flight in flights],
            "validation_flights_loaded": False,
            "new_confirmatory_flights_loaded": False,
        },
        "source_interpretation": {
            "published_parameter_list_use": (
                "presence/absence and explicitly listed bit semantics only"
            ),
            "anonymous_bits_assigned_mode_names": False,
            "source_ids": [
                "a320_dfdr_parameter_mnemonic_mirror",
                "dlr_a320_atra_flight_test_thesis",
                "faa_a320_autothrust_mode_architecture",
            ],
        },
        "method": {
            "sample_rate_hz": 1,
            "rollout_window_s": 300,
            "candidate_response_time_constants_s": list(
                RESPONSE_TIME_CONSTANTS_S
            ),
            "target_rate_feedforward": 0.0,
            "cross_validation": "leave-one-fit-flight-out response-time selection",
            "candidate_selector_semantics": "anonymous unless published",
            "baseline_variable_mode_screen": (
                "TLA CL detent; N1 50 percent to 2 points below CLB ceiling"
            ),
        },
        "decision": decision,
        "artifacts": {
            "runner": "tools/audit_a320_autothrust_modes.py",
            "inventory": str(args.inventory),
            "selector_metrics": str(args.selector_metrics),
            "fold_metrics": str(args.fold_metrics),
            "report": str(args.report),
        },
        "dependencies": {
            "panel": str(args.panel),
            "speed_response_audit": "docs/modeling/a320-speed-autothrust.yml",
            "clb_envelope": str(args.envelope_parameters),
        },
    }

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(inventory).to_csv(args.inventory, index=False)
    _round_frame(selector_metrics).to_csv(args.selector_metrics, index=False)
    _round_frame(fold_metrics).to_csv(args.fold_metrics, index=False)
    args.audit.write_text(yaml.safe_dump(_native(audit), sort_keys=False))
    _write_report(
        args.report,
        inventory=inventory,
        selector_metrics=selector_metrics,
        decision=decision,
    )
    return inventory, selector_metrics, fold_metrics, audit


def main() -> None:
    inventory, selectors, _, audit = run(parse_args())
    print(
        inventory.loc[
            inventory["category"].isin(["required-mode", "published-auxiliary"])
        ].to_string(index=False)
    )
    print(
        selectors.loc[selectors["evaluable_flights"] >= MINIMUM_EVALUABLE_FLIGHTS]
        .sort_values("mean_lofo_persistence_relative_improvement", ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print(yaml.safe_dump(audit["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
