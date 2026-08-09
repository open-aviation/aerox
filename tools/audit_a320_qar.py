#!/usr/bin/env python3
"""Audit the two A320-214 QAR files and select a replay flight panel.

The raw files are large external inputs. This tool reads only the columns needed
for the first trajectory-replay experiment and writes small, redistributable
summary artifacts; it never modifies or copies the QAR data.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RAW_AIRCRAFT = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}

TIME = "Time..ms."
DATE = "DATE"
UTC = "UTC"
RECORDING = "recording"
GEAR = ("LDGN_ON..ON.OFF.", "LDG_ON_1..ON.OFF.", "LDG_ON_2..ON.OFF.")

SIGNALS = {
    "altitude": "ALT..ft.",
    "radio_altitude_1": "RALT1..ft.",
    "calibrated_airspeed": "CAS..Kts.",
    "groundspeed": "GS..Kts.",
    "latitude": "GPSLAT....",
    "longitude": "GPSLON....",
    "vertical_acceleration": "VRTACC..G.",
    "lateral_acceleration": "LATACC..G.",
    "longitudinal_acceleration": "LONGACC..G.",
    "pitch": "PITCH....",
    "roll": "ROLL....",
    "pitch_rate": "PTCR",
    "roll_rate": "ROLR",
    "yaw_rate": "YAW",
    "angle_of_attack": "AOAI_1",
    "aileron_1": "AIL_1",
    "aileron_2": "AIL_2",
    "elevator_1": "ELEV_1",
    "elevator_2": "ELEV_2",
    "rudder": "RUDD",
    "stabilizer": "STAB",
    "throttle_1": "TLA_1",
    "throttle_2": "TLA_2",
    "flap": "FLAP_RAW",
    "fuel_flow_1": "FF_1_KG..Kg.h.",
    "fuel_flow_2": "FF_2_KG..Kg.h.",
    "fuel_quantity": "WFQ_T_KG..KG.",
    "gross_weight": "GW_1_KG..10.Kg.",
    "n1_1": "N1_1",
    "n1_2": "N1_2",
    "n2_1": "N2_1",
    "n2_2": "N2_2",
    "total_air_temperature": "TAT",
    "wind_direction": "WD",
    "wind_speed": "WS",
}

# These channels are needed to attempt the first replay. Four seconds is a
# deliberately generous bounded-hold limit for channels sampled at 1--2 Hz.
CRITICAL = (
    "altitude",
    "calibrated_airspeed",
    "groundspeed",
    "latitude",
    "longitude",
    "pitch",
    "roll",
    "pitch_rate",
    "roll_rate",
    "yaw_rate",
    "angle_of_attack",
    "aileron_1",
    "aileron_2",
    "elevator_1",
    "elevator_2",
    "rudder",
    "stabilizer",
    "throttle_1",
    "throttle_2",
    "fuel_flow_1",
    "fuel_flow_2",
    "fuel_quantity",
    "gross_weight",
    "n1_1",
    "n1_2",
)
MAX_HOLD_GAP_S = 4.0

PANEL_FEATURES = (
    "airborne_duration_s",
    "max_altitude_ft",
    "max_cas_kts",
    "takeoff_gross_weight_kg",
    "endpoint_distance_nm",
    "mean_wind_speed_raw",
    "surface_activity_raw",
)


@dataclass
class FlightSummary:
    registration: str
    recording: str
    date: str
    airborne_duration_s: float
    ground_before_liftoff: bool
    ground_after_touchdown: bool
    max_altitude_ft: float
    max_cas_kts: float
    takeoff_gross_weight_kg: float
    endpoint_distance_nm: float
    mean_wind_speed_raw: float
    surface_activity_raw: float
    min_native_coverage: float
    max_critical_gap_s: float
    operational_flight: bool
    in_processed_table: bool = False
    identifier_collision: bool = False
    cadences_s: dict[str, float] = field(default_factory=dict, repr=False)
    gaps_s: dict[str, float] = field(default_factory=dict, repr=False)
    coverages: dict[str, float] = field(default_factory=dict, repr=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/a320_raw"),
        help="directory containing the two per-registration Parquet files",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/a320/a320.parquet"),
        help="existing processed table used only for a lineage cross-check",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-qar-audit.md"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-replay-panel.csv"),
    )
    parser.add_argument("--panel-size", type=int, default=16)
    return parser.parse_args()


def _required_columns() -> list[str]:
    return [RECORDING, TIME, DATE, UTC, *GEAR, *SIGNALS.values()]


def _validate_schema(path: Path) -> tuple[int, int, int]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    missing = sorted(set(_required_columns()) - names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return (
        parquet.metadata.num_rows,
        parquet.metadata.num_columns,
        parquet.metadata.num_row_groups,
    )


def _airborne_bounds(frame: pd.DataFrame) -> tuple[int, int] | None:
    gear = frame.loc[:, list(GEAR)].ffill().bfill().median(axis=1)
    airborne_positions = np.flatnonzero(gear.to_numpy(dtype=float) < 0.5)
    if len(airborne_positions) == 0:
        return None
    return int(airborne_positions[0]), int(airborne_positions[-1])


def _native_signal_stats(
    time_s: np.ndarray, values: pd.Series
) -> tuple[float, float, float]:
    valid = values.notna().to_numpy()
    valid_times = time_s[valid]
    if len(valid_times) == 0:
        return math.nan, math.inf, 0.0
    if len(valid_times) == 1:
        cadence = math.nan
    else:
        positive_steps = np.diff(valid_times)
        positive_steps = positive_steps[positive_steps > 0]
        cadence = float(np.median(positive_steps)) if len(positive_steps) else math.nan

    boundary_and_sample_times = np.concatenate(
        ([time_s[0]], valid_times, [time_s[-1]])
    )
    gap = float(np.max(np.diff(np.sort(boundary_and_sample_times))))
    duration = max(float(time_s[-1] - time_s[0]), 1.0)
    if math.isfinite(cadence):
        expected = duration / cadence + 1.0
        coverage = min(1.0, len(valid_times) / expected)
    else:
        coverage = 1.0 / (duration + 1.0)
    return cadence, gap, float(coverage)


def _first_finite(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.iloc[0]) if len(finite) else math.nan


def _parse_flight_date(values: pd.Series) -> str:
    nonempty = values.astype("string").str.strip().replace("", pd.NA).dropna()
    if not len(nonempty):
        return ""
    try:
        return datetime.strptime(f"{nonempty.iloc[0]}/2011", "%d/%m/%Y").date().isoformat()
    except ValueError:
        return str(nonempty.iloc[0])


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle endpoint distance in nautical miles."""
    if not all(math.isfinite(value) for value in (lat1, lon1, lat2, lon2)):
        return math.nan
    lat1r, lon1r, lat2r, lon2r = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + (
        math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    )
    return 3440.065 * 2 * math.asin(math.sqrt(a))


def _surface_activity(frame: pd.DataFrame) -> float:
    columns = [
        SIGNALS[name]
        for name in (
            "aileron_1",
            "aileron_2",
            "elevator_1",
            "elevator_2",
            "rudder",
            "stabilizer",
        )
    ]
    held = frame.loc[:, columns].ffill().bfill()
    changes = held.diff().abs()
    per_surface = changes.quantile(0.95).to_numpy(dtype=float)
    finite = per_surface[np.isfinite(per_surface)]
    return float(np.linalg.norm(finite)) if len(finite) else math.nan


def summarize_flight(registration: str, recording: str, frame: pd.DataFrame) -> FlightSummary | None:
    # A small number of isolated corrupt rows retain a recording identifier but
    # have no time and contain impossible values in otherwise valid flights.
    frame = frame.loc[frame[TIME].notna()]
    frame = frame.sort_values(TIME, kind="stable").reset_index(drop=True)
    bounds = _airborne_bounds(frame)
    if bounds is None:
        return None
    first, last = bounds
    segment = frame.iloc[first : last + 1]
    time_s = segment[TIME].to_numpy(dtype=float) / 1000.0
    if len(time_s) < 2:
        return None
    row_step = float(np.median(np.diff(time_s)))
    duration_s = float(time_s[-1] - time_s[0] + row_step)

    cadences: dict[str, float] = {}
    gaps: dict[str, float] = {}
    coverages: dict[str, float] = {}
    for logical_name, column in SIGNALS.items():
        cadence, gap, coverage = _native_signal_stats(time_s, segment[column])
        cadences[logical_name] = cadence
        gaps[logical_name] = gap
        coverages[logical_name] = coverage

    held = segment.loc[:, list(SIGNALS.values())].ffill().bfill()
    lat = held[SIGNALS["latitude"]]
    lon = held[SIGNALS["longitude"]]
    distance_nm = haversine_nm(
        _first_finite(lat),
        _first_finite(lon),
        _first_finite(lat.iloc[::-1]),
        _first_finite(lon.iloc[::-1]),
    )
    max_altitude = float(held[SIGNALS["altitude"]].max())
    max_cas = float(held[SIGNALS["calibrated_airspeed"]].max())
    gross_weight = _first_finite(held[SIGNALS["gross_weight"]]) * 10.0
    critical_gaps = [gaps[name] for name in CRITICAL]
    critical_coverages = [coverages[name] for name in CRITICAL]
    max_gap = max(critical_gaps)
    min_coverage = min(critical_coverages)

    ground = (
        frame.loc[:, list(GEAR)].ffill().bfill().median(axis=1).to_numpy() >= 0.5
    )
    ground_before = bool(np.any(ground[:first]))
    ground_after = bool(np.any(ground[last + 1 :]))
    operational = bool(
        ground_before
        and ground_after
        and duration_s >= 1_500.0
        and max_altitude >= 10_000.0
        and max_cas >= 180.0
        and math.isfinite(distance_nm)
        and max_gap <= MAX_HOLD_GAP_S
    )

    return FlightSummary(
        registration=registration,
        recording=str(recording),
        date=_parse_flight_date(segment[DATE]),
        airborne_duration_s=duration_s,
        ground_before_liftoff=ground_before,
        ground_after_touchdown=ground_after,
        max_altitude_ft=max_altitude,
        max_cas_kts=max_cas,
        takeoff_gross_weight_kg=gross_weight,
        endpoint_distance_nm=distance_nm,
        mean_wind_speed_raw=float(held[SIGNALS["wind_speed"]].mean()),
        surface_activity_raw=_surface_activity(segment),
        min_native_coverage=min_coverage,
        max_critical_gap_s=max_gap,
        operational_flight=operational,
        cadences_s=cadences,
        gaps_s=gaps,
        coverages=coverages,
    )


def audit_raw_file(
    registration: str, path: Path
) -> tuple[list[FlightSummary], tuple[int, int, int, int]]:
    rows, columns, row_groups = _validate_schema(path)
    frame = pd.read_parquet(path, columns=_required_columns(), engine="pyarrow")
    invalid_time_rows = int(frame[TIME].isna().sum())
    summaries: list[FlightSummary] = []
    for recording, flight in frame.groupby(RECORDING, sort=True, observed=True):
        summary = summarize_flight(registration, str(recording), flight)
        if summary is not None:
            summaries.append(summary)
    return summaries, (rows, columns, row_groups, invalid_time_rows)


def _processed_recordings(path: Path) -> set[str]:
    parquet = pq.ParquetFile(path)
    if "flight_id" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no flight_id column")
    ids = pd.read_parquet(path, columns=["flight_id"])["flight_id"]
    return set(ids.dropna().astype(str).unique())


def _robust_feature_matrix(flights: list[FlightSummary]) -> np.ndarray:
    matrix = np.array(
        [[getattr(flight, name) for name in PANEL_FEATURES] for flight in flights],
        dtype=float,
    )
    median = np.nanmedian(matrix, axis=0)
    matrix = np.where(np.isfinite(matrix), matrix, median)
    lower = np.nanquantile(matrix, 0.25, axis=0)
    upper = np.nanquantile(matrix, 0.75, axis=0)
    scale = upper - lower
    scale = np.where(scale > 0, scale, 1.0)
    return (matrix - median) / scale


def select_diverse(flights: list[FlightSummary], count: int) -> list[FlightSummary]:
    """Select a deterministic medoid-plus-farthest-point panel."""
    if count > len(flights):
        raise ValueError(f"cannot select {count} flights from a pool of {len(flights)}")
    ordered = sorted(flights, key=lambda item: (item.recording, item.date))
    features = _robust_feature_matrix(ordered)
    medoid = int(np.argmin(np.linalg.norm(features, axis=1)))
    selected = [medoid]
    minimum_distance = np.linalg.norm(features - features[medoid], axis=1)
    minimum_distance[medoid] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return [ordered[index] for index in selected]


def build_panel(summaries: list[FlightSummary], panel_size: int) -> pd.DataFrame:
    if panel_size < 4 or panel_size % len(RAW_AIRCRAFT):
        raise ValueError("panel size must be an even integer of at least four")
    per_registration = panel_size // len(RAW_AIRCRAFT)
    pool = [
        item
        for item in summaries
        if item.operational_flight
        and item.in_processed_table
        and not item.identifier_collision
    ]
    rows: list[dict[str, object]] = []
    for registration in RAW_AIRCRAFT:
        choices = select_diverse(
            [item for item in pool if item.registration == registration],
            per_registration,
        )
        validation_orders = set(range(1, per_registration, 2))
        validation_orders = set(sorted(validation_orders)[:3])
        for order, flight in enumerate(choices, start=1):
            if registration == "TAIL-A" and order == 1:
                role = "development"
            elif order - 1 in validation_orders:
                role = "validation"
            else:
                role = "calibration"
            rows.append(
                {
                    "registration": registration,
                    "recording": flight.recording,
                    "role": role,
                    "selection_order": order,
                    "date": flight.date,
                    "airborne_duration_s": round(flight.airborne_duration_s, 1),
                    "max_altitude_ft": round(flight.max_altitude_ft, 1),
                    "max_cas_kts": round(flight.max_cas_kts, 3),
                    "takeoff_gross_weight_kg": round(
                        flight.takeoff_gross_weight_kg, 1
                    ),
                    "endpoint_distance_nm": round(flight.endpoint_distance_nm, 2),
                    "mean_wind_speed_raw": round(flight.mean_wind_speed_raw, 3),
                    "surface_activity_raw": round(flight.surface_activity_raw, 3),
                    "min_native_coverage": round(flight.min_native_coverage, 5),
                    "max_critical_gap_s": round(flight.max_critical_gap_s, 3),
                    "processed_match": flight.in_processed_table,
                }
            )
    return pd.DataFrame(rows)


def _format_int(value: int) -> str:
    return f"{value:,}"


def _signal_table(summaries: list[FlightSummary]) -> str:
    eligible = [item for item in summaries if item.operational_flight]
    lines = [
        "| Logical signal | Raw column | Median cadence (s) | 99th-percentile max gap (s) | Flights within 4 s |",
        "|---|---|---:|---:|---:|",
    ]
    for logical_name, column in SIGNALS.items():
        cadences = np.array(
            [item.cadences_s[logical_name] for item in eligible], dtype=float
        )
        gaps = np.array([item.gaps_s[logical_name] for item in eligible], dtype=float)
        cadence = float(np.nanmedian(cadences))
        gap = float(np.nanquantile(gaps, 0.99))
        passing = int(np.sum(gaps <= MAX_HOLD_GAP_S))
        lines.append(
            f"| {logical_name} | `{column}` | {cadence:.1f} | {gap:.1f} | {passing}/{len(eligible)} |"
        )
    return "\n".join(lines)


def _report(
    summaries: list[FlightSummary],
    metadata: dict[str, tuple[int, int, int, int]],
    processed_ids: set[str],
    panel: pd.DataFrame,
    raw_dir: Path,
    processed_path: Path,
) -> str:
    collisions = sorted(
        {
            item.recording
            for item in summaries
            if item.identifier_collision
        }
    )
    table_rows = []
    for registration in RAW_AIRCRAFT:
        rows, columns, row_groups, invalid_time_rows = metadata[registration]
        aircraft = [item for item in summaries if item.registration == registration]
        operational = [item for item in aircraft if item.operational_flight]
        matched = [item for item in operational if item.in_processed_table]
        table_rows.append(
            f"| {registration} | {_format_int(rows)} | {columns} | {row_groups} | "
            f"{invalid_time_rows} | "
            f"{len(aircraft)} | {len(operational)} | {len(matched)} |"
        )

    unmatched = sorted(
        (item.registration, item.recording)
        for item in summaries
        if item.operational_flight and not item.in_processed_table
    )
    unmatched_text = ", ".join(f"{reg}/{rid}" for reg, rid in unmatched) or "none"
    role_counts = panel.groupby(["registration", "role"]).size().to_dict()
    roles = ", ".join(
        f"{registration} {role}: {count}"
        for (registration, role), count in sorted(role_counts.items())
    )
    collision_text = ", ".join(collisions) or "none"

    return f"""# A320 QAR Audit and Replay Panel

Status: generated baseline audit  
Date: 2026-08-04  
Generator: `tools/audit_a320_qar.py`

## Outcome

The two raw files contain {_format_int(sum(value[0] for value in metadata.values()))}
one-second rows and {sum(1 for item in summaries if item.operational_flight):,}
aircraft-recording pairs that pass the initial complete operational-flight screen.
Of those, {sum(1 for item in summaries if item.operational_flight and item.in_processed_table):,}
pairs have a matching bare identifier in the existing processed table. The selected
16-flight replay panel is balanced across both registrations and is stored in
`docs/modeling/a320-replay-panel.csv`.

Signal availability is sufficient to start channel decoding and kinematic replay.
It is **not** yet sufficient to identify physical coefficients: a number of named
channels retain recorder scaling or coding. The provisional decoding registry is
`docs/modeling/a320-qar-channel-map.yml`.

## Inputs and method

- Raw directory: `{raw_dir}` (external, read-only)
- Processed cross-check: `{processed_path}`
- Airborne interval: first to last sample for which the median of the three
  landing-gear-on-ground discretes is zero.
- Complete operational screen: ground samples before and after the airborne
  interval, at least 1,500 airborne seconds, at least 10,000 ft, at least 180 kt
  CAS, valid GPS endpoints, and no replay-critical native-signal gap above four
  seconds.
- Sparse native channels are assessed at their own cadence. Future replay will
  apply bounded zero-order hold within a recording; missing values are never
  filled across recording boundaries.
- Rows without a finite recorder time are rejected before segmentation. They
  contain impossible mixed-channel values and must not participate in filling.
- The processed table is only a lineage cross-check. Raw per-registration files
  remain authoritative.

## Inventory

| Registration | Rows | Columns | Row groups | Invalid-time rows | Airborne recordings | Complete operational | In processed table |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The processed table contains {len(processed_ids):,} distinct bare `flight_id`
values. It cannot by itself represent registration when identifiers collide.
Collision found across registration files: {collision_text}. The colliding raw
records were excluded from panel selection. The collision was manually checked and
the two `FLT0005` records have the same row count, dates, and trajectory endpoints;
this should nevertheless be treated as ambiguous until source lineage is confirmed.

Complete raw flights absent from the processed table: {unmatched_text}. They are
quarantined from the initial panel until the transformation lineage explains the
omission.

## Signal cadence and bounded-hold coverage

The table is computed over all complete operational raw flights. “Within 4 s” is
an availability test, not proof that the channel is decoded correctly.

{_signal_table(summaries)}

## Selected replay panel

Selection is deterministic. Within each registration it starts with the flight
nearest the robust feature median, then greedily adds flights farthest from the
current set. Features are duration, maximum altitude, maximum CAS, takeoff gross
weight, endpoint distance, recorded wind-speed code, and a surface-activity index.
This creates useful operational diversity without looking at future replay error.

- Panel allocation: {roles}
- The first TAIL-A selection is the golden development flight.
- Development and calibration flights may be used for iteration and fitting.
- Validation flights are held out as complete flights and must not influence model
  selection, parameter tuning, signal decoding, or residual-model training.
- The CSV intentionally contains no trajectory coordinates or raw samples.

## Critical findings before dynamics

1. `PITCH....`, `ROLL....`, `AOAI_1`, `N1_1`, `N1_2`, control-surface,
   throttle, flap, `TAT`, `WD`, `WS`, and `TFUEL` must not be accepted at face
   value as physical units. Several contain clear scale factors, codes, or invalid
   states.
2. A representative-flight consistency check supports provisional factors of
   0.5 for pitch/roll, 0.25 for angle of attack, and 0.5 for N1/N2. These remain
   hypotheses pending independent confirmation and cross-flight validation.
3. Throttle-lever values are a small set of detent-like codes, not a continuous
   zero-to-100 percent thrust command. The propulsion interface can expose a
   normalized command later, but replay must first preserve the recorded detents.
4. `WFQ_T_KG..KG.` is the plausible total fuel-quantity channel for the first
   audit. `TFUEL` has a different unexplained scale and is not interchangeable.
5. The processed table preserves some raw recorder encodings and adds weather
   fields of presently undocumented lineage. It is useful for cross-checking, not
   as ground truth for channel units.

## Next gate

Before aerodynamic or propulsion fitting, implement a typed decoder with unit,
sign, validity, cadence, and latency tests. Then pass the kinematic replay gate:
reconstruct position from decoded attitude, airspeed, and independently sourced
wind without using aerodynamic coefficients. Only after those residuals are
understood should force or moment parameters be estimated.
"""


def write_outputs(report_path: Path, panel_path: Path, report: str, panel: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    panel.to_csv(panel_path, index=False)


def mark_identifier_collisions(summaries: Iterable[FlightSummary]) -> set[str]:
    registrations_by_id: dict[str, set[str]] = {}
    for item in summaries:
        registrations_by_id.setdefault(item.recording, set()).add(item.registration)
    collisions = {
        recording
        for recording, registrations in registrations_by_id.items()
        if len(registrations) > 1
    }
    for item in summaries:
        item.identifier_collision = item.recording in collisions
    return collisions


def main() -> None:
    args = parse_args()
    if args.panel_size != 16:
        # The report prose and agreed experiment currently specify 16 flights.
        raise ValueError("the baseline report currently requires --panel-size 16")

    all_summaries: list[FlightSummary] = []
    metadata: dict[str, tuple[int, int, int, int]] = {}
    for registration, filename in RAW_AIRCRAFT.items():
        path = args.raw_dir / filename
        summaries, file_metadata = audit_raw_file(registration, path)
        all_summaries.extend(summaries)
        metadata[registration] = file_metadata

    processed_ids = _processed_recordings(args.processed)
    for summary in all_summaries:
        summary.in_processed_table = summary.recording in processed_ids
    mark_identifier_collisions(all_summaries)

    panel = build_panel(all_summaries, args.panel_size)
    report = _report(
        all_summaries,
        metadata,
        processed_ids,
        panel,
        args.raw_dir,
        args.processed,
    )
    write_outputs(args.report, args.panel, report, panel)
    print(f"wrote {args.report}")
    print(f"wrote {args.panel}")


if __name__ == "__main__":
    main()
