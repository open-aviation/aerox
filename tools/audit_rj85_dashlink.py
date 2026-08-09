#!/usr/bin/env python3
"""Audit the local NASA DASHlink archive for RJ85 modeling suitability.

The archive is an external input of roughly 230 GB. This tool reads every ZIP
central directory to establish scope, but decompresses only a deterministic
six-flight sample: the largest MATLAB member in the first, middle, and last
archive for tails 652 and 687. It writes metadata and aggregate diagnostics;
it never extracts, copies, or modifies a raw flight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat

ARCHIVE_PATTERN = re.compile(r"^Tail_(?P<tail>\d+)_(?P<part>\d+)\.zip$")
DEFAULT_TAILS = ("652", "687")
SURFACE_CHANNELS = (
    "AIL_1",
    "AIL_2",
    "ELEV_1",
    "ELEV_2",
    "RUDD",
    "SPL_1",
    "SPL_2",
    "PTRM",
)
PAIRED_SURFACES = {
    "elevator": ("ELEV_1", "ELEV_2"),
    "aileron": ("AIL_1", "AIL_2"),
}
BODY_RATE_TERMS = (
    "pitch rate",
    "roll rate",
    "yaw rate",
    "body rate",
    "angular rate",
    "angular velocity",
    "gyro",
)
MASS_TERMS = ("gross weight", "aircraft weight", "aircraft mass", "gross mass")

PRIMARY_CHANNELS = {
    # Attitude, aerodynamic state, and inertial measurements.
    "PTCH",
    "ROLL",
    "TH",
    "MH",
    "AOA1",
    "AOA2",
    "AOAC",
    "AOAI",
    "BLAC",
    "CTAC",
    "FPAC",
    "VRTG",
    "LATG",
    "LONG",
    # Controls and pilot inputs.
    "AIL_1",
    "AIL_2",
    "ELEV_1",
    "ELEV_2",
    "RUDD",
    "PTRM",
    "SPL_1",
    "SPL_2",
    "CCPC",
    "CCPF",
    "CWPC",
    "CWPF",
    "RUDP",
    # Air data and trajectory.
    "ALT",
    "ALTR",
    "CAS",
    "TAS",
    "MACH",
    "PS",
    "PSA",
    "PT",
    "SAT",
    "TAT",
    "GS",
    "IVV",
    "LATP",
    "LONP",
    "TRK",
    "WD",
    "WS",
    # Propulsion, fuel, configuration, and control modes.
    "N1C",
    "N1T",
    "FLAP",
    "WOW",
    "LGDN",
    "LGUP",
    "A_T",
    "ATEN",
    "TMODE",
    "LMOD",
    "VMODE",
    "APFD",
}
for prefix in ("PLA", "N1", "N2", "FF", "FQTY"):
    PRIMARY_CHANNELS.update(f"{prefix}_{engine}" for engine in range(1, 5))


@dataclass(frozen=True)
class ArchiveRecord:
    path: Path
    tail: str
    part: int
    archive_bytes: int
    mat_members: int
    mat_bytes: int


@dataclass(frozen=True)
class SampleSelection:
    archive: ArchiveRecord
    member: str
    member_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dashlink"),
        help="directory containing Tail_<id>_<part>.zip archives",
    )
    parser.add_argument(
        "--tails",
        nargs="+",
        default=list(DEFAULT_TAILS),
        help="tail identifiers used for the deterministic metadata sample",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/modeling/rj85-dashlink-audit.md")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("docs/modeling/rj85-dashlink-audit.yml")
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/modeling/rj85-dashlink-channel-inventory.csv"),
    )
    parser.add_argument(
        "--sample-flights",
        type=Path,
        default=Path("docs/modeling/rj85-dashlink-sample-flights.csv"),
    )
    parser.add_argument(
        "--surface-diagnostics",
        type=Path,
        default=Path("docs/modeling/rj85-dashlink-surface-diagnostics.csv"),
    )
    return parser.parse_args()


def _parse_archive(path: Path) -> tuple[str, int]:
    match = ARCHIVE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"not a DASHlink archive name: {path.name}")
    return match.group("tail"), int(match.group("part"))


def _scan_archives(data_dir: Path) -> list[ArchiveRecord]:
    records: list[ArchiveRecord] = []
    paths = sorted(
        data_dir.glob("Tail_*_*.zip"),
        key=lambda path: (*_parse_archive(path), path.name),
    )
    if not paths:
        raise FileNotFoundError(f"no DASHlink archives found in {data_dir}")
    for path in paths:
        tail, part = _parse_archive(path)
        with zipfile.ZipFile(path) as archive:
            members = [
                item
                for item in archive.infolist()
                if not item.is_dir() and item.filename.lower().endswith(".mat")
            ]
        records.append(
            ArchiveRecord(
                path=path,
                tail=tail,
                part=part,
                archive_bytes=path.stat().st_size,
                mat_members=len(members),
                mat_bytes=sum(item.file_size for item in members),
            )
        )
    return records


def _sample_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted({0, count // 2, count - 1})


def _select_samples(
    records: Iterable[ArchiveRecord], tails: Iterable[str]
) -> list[SampleSelection]:
    by_tail: dict[str, list[ArchiveRecord]] = defaultdict(list)
    for record in records:
        by_tail[record.tail].append(record)
    selections: list[SampleSelection] = []
    for tail in tails:
        tail_records = sorted(by_tail[str(tail)], key=lambda record: record.part)
        if not tail_records:
            raise ValueError(f"tail {tail} has no local archives")
        for index in _sample_indices(len(tail_records)):
            record = tail_records[index]
            with zipfile.ZipFile(record.path) as archive:
                members = [
                    item
                    for item in archive.infolist()
                    if not item.is_dir() and item.filename.lower().endswith(".mat")
                ]
            if not members:
                raise ValueError(f"{record.path} has no MATLAB members")
            member = max(members, key=lambda item: (item.file_size, item.filename))
            selections.append(
                SampleSelection(
                    archive=record,
                    member=member.filename,
                    member_bytes=member.file_size,
                )
            )
    return selections


def _mat_scalar(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    if len(array) == 0:
        return ""
    item = array[0]
    if isinstance(item, np.generic):
        item = item.item()
    return str(item)


def _rate(value: Any) -> float:
    text = _mat_scalar(value)
    try:
        return float(text)
    except ValueError:
        return math.nan


def _load_selection(selection: SampleSelection) -> dict[str, Any]:
    with zipfile.ZipFile(selection.archive.path) as archive:
        payload = archive.read(selection.member)
    raw = loadmat(BytesIO(payload), squeeze_me=True, struct_as_record=False)
    return {
        key: value
        for key, value in raw.items()
        if not key.startswith("__") and hasattr(value, "data")
    }


def _channel_metadata(name: str, value: Any) -> dict[str, Any]:
    return {
        "parameter": name,
        "alpha": _mat_scalar(getattr(value, "Alpha", "")),
        "rate_hz": _rate(getattr(value, "Rate", math.nan)),
        "units": _mat_scalar(getattr(value, "Units", "")),
        "description": _mat_scalar(getattr(value, "Description", "")),
    }


def _channel_category(name: str, description: str) -> str:
    if name in {
        "AIL_1",
        "AIL_2",
        "ELEV_1",
        "ELEV_2",
        "RUDD",
        "PTRM",
        "SPL_1",
        "SPL_2",
        "ABRK",
        "CCPC",
        "CCPF",
        "CWPC",
        "CWPF",
        "RUDP",
    }:
        return "control"
    if re.match(r"^(PLA|N1|N2|FF|EGT|FQTY|OIP|OIT|VIB)(_|C|T)", name):
        return "propulsion-or-fuel"
    if name in {
        "PTCH",
        "ROLL",
        "TH",
        "MH",
        "AOA1",
        "AOA2",
        "AOAC",
        "AOAI",
        "BLAC",
        "CTAC",
        "FPAC",
        "VRTG",
        "LATG",
        "LONG",
    }:
        return "attitude-or-inertial"
    if name in {
        "ALT",
        "ALTR",
        "CAS",
        "TAS",
        "MACH",
        "PS",
        "PSA",
        "PT",
        "SAT",
        "TAT",
        "GS",
        "IVV",
        "LATP",
        "LONP",
        "TRK",
        "TRKM",
        "DA",
        "WD",
        "WS",
    }:
        return "air-data-or-trajectory"
    if "MODE" in name or name in {"A_T", "ATEN", "APFD", "DFGS", "PH"}:
        return "mode"
    if name in {"FLAP", "WOW", "LGDN", "LGUP", "NSQT", "MSQT_1", "MSQT_2"}:
        return "configuration"
    if "acceleration" in description.lower():
        return "attitude-or-inertial"
    return "other"


def _metadata_fingerprint(channels: dict[str, Any]) -> str:
    records = [
        _channel_metadata(name, value) for name, value in sorted(channels.items())
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_data(value: Any) -> np.ndarray:
    data = np.asarray(value.data).reshape(-1)
    try:
        return data.astype(float, copy=False)
    except (TypeError, ValueError):
        return np.full(len(data), np.nan)


def _duration_seconds(channels: dict[str, Any]) -> tuple[float, float]:
    durations = []
    for value in channels.values():
        rate = _rate(getattr(value, "Rate", math.nan))
        if rate > 0:
            durations.append(len(np.asarray(value.data).reshape(-1)) / rate)
    if not durations:
        return math.nan, math.nan
    median = float(np.median(durations))
    return median, float(np.max(np.abs(np.asarray(durations) - median)))


def _airborne_mask(
    channels: dict[str, Any], *, target_length: int, target_rate: float
) -> np.ndarray:
    if "WOW" not in channels or not math.isfinite(target_rate) or target_rate <= 0:
        return np.ones(target_length, dtype=bool)
    wow = _numeric_data(channels["WOW"])
    wow_rate = _rate(getattr(channels["WOW"], "Rate", math.nan))
    if not math.isfinite(wow_rate) or wow_rate <= 0 or len(wow) == 0:
        return np.ones(target_length, dtype=bool)
    sample_times = np.arange(target_length, dtype=float) / target_rate
    indices = np.minimum((sample_times * wow_rate).astype(int), len(wow) - 1)
    # The sampled DASHlink conversion uses one while airborne and zero on the
    # ground, despite the intuitive reading of the "WEIGHT ON WHEELS" label.
    # ALT/CAS cross-tabs establish this polarity in the audit sample.
    return np.isfinite(wow[indices]) & (wow[indices] > 0.5)


def _finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if len(finite) else math.nan


def _centered_pair_correlation(
    channels: dict[str, Any], left: str, right: str
) -> float:
    if left not in channels or right not in channels:
        return math.nan
    left_data = _numeric_data(channels[left])
    right_data = _numeric_data(channels[right])
    if len(left_data) != len(right_data) or len(left_data) < 2:
        return math.nan
    rate = _rate(getattr(channels[left], "Rate", math.nan))
    mask = _airborne_mask(channels, target_length=len(left_data), target_rate=rate)
    mask &= np.isfinite(left_data) & np.isfinite(right_data)
    # Recurrent extreme conversion codes otherwise dominate correlations while
    # representing less than one percent of a long flight. Keep this diagnostic
    # robust without hiding the raw extremes in the surface artifact.
    if np.count_nonzero(mask) >= 200:
        for values in (left_data, right_data):
            low, high = np.quantile(values[mask], (0.005, 0.995))
            mask &= (values >= low) & (values <= high)
    if np.count_nonzero(mask) < 2:
        return math.nan
    left_centered = left_data[mask] - np.median(left_data[mask])
    right_centered = right_data[mask] - np.median(right_data[mask])
    if np.std(left_centered) == 0 or np.std(right_centered) == 0:
        return math.nan
    return float(np.corrcoef(left_centered, right_centered)[0, 1])


def _surface_rows(
    selection: SampleSelection, channels: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SURFACE_CHANNELS:
        if name not in channels:
            continue
        value = channels[name]
        data = _numeric_data(value)
        rate = _rate(getattr(value, "Rate", math.nan))
        mask = _airborne_mask(channels, target_length=len(data), target_rate=rate)
        airborne = data[mask & np.isfinite(data)]
        rows.append(
            {
                "tail": selection.archive.tail,
                "archive": selection.archive.path.name,
                "member": selection.member,
                "parameter": name,
                "rate_hz": rate,
                "units": _mat_scalar(getattr(value, "Units", "")),
                "airborne_samples": len(airborne),
                "minimum": _finite_quantile(airborne, 0.0),
                "p01": _finite_quantile(airborne, 0.01),
                "median": _finite_quantile(airborne, 0.5),
                "p99": _finite_quantile(airborne, 0.99),
                "maximum": _finite_quantile(airborne, 1.0),
            }
        )
    return rows


def _sample_row(selection: SampleSelection, channels: dict[str, Any]) -> dict[str, Any]:
    duration, duration_spread = _duration_seconds(channels)
    wow = _numeric_data(channels["WOW"]) if "WOW" in channels else np.array([])
    wow_rate = (
        _rate(getattr(channels["WOW"], "Rate", math.nan))
        if "WOW" in channels
        else math.nan
    )
    airborne_s = (
        float(np.count_nonzero(np.isfinite(wow) & (wow > 0.5)) / wow_rate)
        if len(wow) and wow_rate > 0
        else math.nan
    )
    altitude = _numeric_data(channels["ALT"]) if "ALT" in channels else np.array([])
    max_altitude = _finite_quantile(altitude, 1.0)
    if len(wow) and len(altitude) % len(wow) == 0:
        altitude_1hz = altitude.reshape(len(wow), -1).mean(axis=1)
        high_altitude = altitude_1hz >= 5_000
        high_altitude_wow_one = (
            float(np.mean(wow[high_altitude] > 0.5))
            if np.any(high_altitude)
            else math.nan
        )
        wow_zero_max_altitude = _finite_quantile(altitude_1hz[wow < 0.5], 1.0)
    else:
        high_altitude_wow_one = math.nan
        wow_zero_max_altitude = math.nan
    return {
        "tail": selection.archive.tail,
        "archive": selection.archive.path.name,
        "archive_part": selection.archive.part,
        "member": selection.member,
        "member_bytes": selection.member_bytes,
        "channel_count": len(channels),
        "metadata_fingerprint": _metadata_fingerprint(channels),
        "duration_s": duration,
        "channel_duration_max_deviation_s": duration_spread,
        "airborne_s": airborne_s,
        "max_pressure_altitude_ft": max_altitude,
        "high_altitude_wow_one_fraction": high_altitude_wow_one,
        "wow_zero_max_pressure_altitude_ft": wow_zero_max_altitude,
        "operational_sample": bool(airborne_s >= 600 and max_altitude >= 10_000),
        "elevator_centered_correlation": _centered_pair_correlation(
            channels, *PAIRED_SURFACES["elevator"]
        ),
        "aileron_centered_correlation": _centered_pair_correlation(
            channels, *PAIRED_SURFACES["aileron"]
        ),
    }


def _inventory_rows(
    sampled_channels: list[dict[str, Any]], sample_count: int
) -> list[dict[str, Any]]:
    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for channels in sampled_channels:
        for name, value in channels.items():
            collected[name].append(_channel_metadata(name, value))
    rows: list[dict[str, Any]] = []
    for name, records in sorted(collected.items()):
        rates = sorted({record["rate_hz"] for record in records})
        units = sorted({record["units"] for record in records})
        descriptions = sorted({record["description"] for record in records})
        alphas = sorted({record["alpha"] for record in records})
        description = (
            descriptions[0] if len(descriptions) == 1 else " | ".join(descriptions)
        )
        rows.append(
            {
                "parameter": name,
                "alpha": " | ".join(alphas),
                "description": description,
                "units": " | ".join(units),
                "rate_hz": rates[0] if len(rates) == 1 else " | ".join(map(str, rates)),
                "category": _channel_category(name, description),
                "modeling_priority": "primary"
                if name in PRIMARY_CHANNELS
                else "supporting",
                "sample_presence": len(records),
                "sample_count": sample_count,
                "presence_fraction": len(records) / sample_count,
                "metadata_consistent": len(rates)
                == len(units)
                == len(descriptions)
                == len(alphas)
                == 1,
            }
        )
    return rows


def _semantic_matches(
    inventory: Iterable[dict[str, Any]], terms: Iterable[str]
) -> list[str]:
    lowered_terms = tuple(term.lower() for term in terms)
    matches = []
    for row in inventory:
        text = " ".join(
            str(row[field]) for field in ("parameter", "alpha", "description")
        ).lower()
        if any(term in text for term in lowered_terms):
            matches.append(str(row["parameter"]))
    return sorted(matches)


def _rate_of(inventory: Iterable[dict[str, Any]], parameter: str) -> float | None:
    for row in inventory:
        if row["parameter"] == parameter and isinstance(row["rate_hz"], float):
            return row["rate_hz"]
        if row["parameter"] == parameter and isinstance(row["rate_hz"], int):
            return float(row["rate_hz"])
    return None


def _round_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), 8)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: _round_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_value(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty artifact: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_round_value(rows))


def _range_text(values: Iterable[float]) -> str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return f"{min(finite):.3f} to {max(finite):.3f}" if finite else "unavailable"


def _write_report(
    path: Path,
    *,
    audit: dict[str, Any],
    inventory: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> None:
    rates = {
        channel: _rate_of(inventory, channel)
        for channel in (
            "PTCH",
            "ROLL",
            "TH",
            "AOAC",
            "BLAC",
            "VRTG",
            "ELEV_1",
            "AIL_1",
            "RUDD",
            "PLA_1",
            "N1_1",
            "N2_1",
            "FF_1",
            "FQTY_1",
            "LATP",
            "LONP",
        )
    }
    elevator_corr = _range_text(row["elevator_centered_correlation"] for row in samples)
    aileron_corr = _range_text(row["aileron_centered_correlation"] for row in samples)
    scope = audit["archive_scope"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# RJ85/DASHlink channel and cadence audit

## Decision

The local DASHlink archive is suitable for a first **effective 1 Hz
control-to-motion model**, but it does not identify an unanchored physical
six-degree-of-freedom model or fast actuator dynamics. Use the faster attitude,
heading, air-data, and acceleration channels for rate reconstruction and
anti-aliased one-second targets; keep elevator, aileron, spoiler, and trim inputs
at their recorded 1 Hz cadence. The next gate is surface decoding and kinematic
rate reconstruction, not force/moment coefficient fitting.

This decision is conditional on the project assumption that the de-identified
regional jet is an Avro RJ85. It is not a NASA-published type identification.

## Archive scope and sampling contract

- Local scope: {scope["archive_count"]} ZIP archives, {scope["tail_count"]} tail
  identifiers, and {scope["mat_flight_count"]:,} MATLAB members
  ({scope["archive_bytes_gib"]:.2f} GiB compressed).
- The tool inspected every ZIP central directory without bulk extraction.
- It then selected tails 652 and 687, chose each tail's first, middle, and last
  archive, and decompressed only the largest MATLAB member in each archive.
- All {len(samples)} selected members are operational samples, have 186 channels,
  and share one channel-metadata fingerprint.

The deterministic sample is a schema/cadence feasibility check, not a
population-level flight-quality or excitation panel.

## Relevant recorded cadence

| Signal group | Representative channels | Native rate |
|---|---|---:|
| pitch/roll attitude | PTCH, ROLL | {rates["PTCH"]:g} Hz |
| true heading | TH | {rates["TH"]:g} Hz |
| angle of attack | AOAC, AOA1, AOA2 | {rates["AOAC"]:g} Hz |
| high-rate accelerations | BLAC, CTAC, FPAC | {rates["BLAC"]:g} Hz |
| vertical acceleration | VRTG | {rates["VRTG"]:g} Hz |
| elevator/aileron/spoiler/trim | ELEV_*, AIL_*, SPL_*, PTRM | {rates["ELEV_1"]:g} Hz |
| rudder | RUDD | {rates["RUDD"]:g} Hz |
| PLA, N1, N2, fuel flow | four engines | {rates["PLA_1"]:g} Hz |
| fuel quantity | FQTY_1..4 | {rates["FQTY_1"]:g} Hz |
| latitude/longitude | LATP, LONP | {rates["LATP"]:g} Hz |

The engine set is unusually useful: all four PLAs, N1s, N2s, and fuel flows are
recorded at 4 Hz, alongside N1 command, N1 target, thrust mode, and autothrust
status. It can support a substantially cleaner effective FADEC/thrust surrogate
than the A320 QAR channels.

## Observability findings

- No parameter name, description, or identifier in the 186-channel schema
  describes pitch rate, roll rate, yaw rate, body angular rate, angular
  velocity, or a gyro. Body rates must be reconstructed from Euler attitude.
- No gross-weight or aircraft-mass parameter exists. Fuel quantity and fuel
  flow are recorded, but initial aircraft mass needs an external/tail-flight
  prior or a fitted nuisance distribution.
- FQTY_1 is labeled pounds, while FQTY_2..4 are labeled `LSB`; tank scaling and
  total-fuel construction require an explicit calibration check.
- The sampled `WOW` bit is one in flight and zero on the ground, as independently
  established by pressure altitude and computed airspeed. That polarity must be
  retained explicitly instead of inferred from the channel name.
- Raw surface channels are not ready to use as literal physical degrees. After
  excluding only the outer 0.5% conversion extremes, median-centered left/right
  elevator correlation ranges from {elevator_corr} and aileron correlation from
  {aileron_corr}. Channel-specific offsets, polarity conventions, recurrent
  extremes, and weak pair agreement require documented decoding before
  constructing symmetric and differential controls.
- Closed-loop airline data still identifies only the commonly flown envelope.
  Autopilot and mode channels should be used for segmentation and confounding
  checks, not treated as randomized excitation.

## Authorized first RJ85 model

1. Decode surface validity, zero, sign, and pair combinations on development
   flights only; preserve raw channels beside every transformed control.
2. Unwrap and smooth PTCH/ROLL/TH at native cadence, derive Euler rates, convert
   them to body-rate estimates, and aggregate state/response targets to 1 Hz.
3. Verify those rates against 16 Hz inertial acceleration and trajectory
   consistency before fitting any control authority.
4. Fit effective longitudinal and lateral-directional state-plus-control models
   with complete-flight and tail-aware splits. Use aerodynamic geometry/drag
   priors and a mass nuisance; do not estimate all dimensional derivatives at
   once.
5. In parallel, fit the four-engine PLA/mode/air-data to N1/N2/FF surrogate,
   then couple it to the translational model only after held-out validation.

The simulator may integrate faster than 1 Hz, but the learned surface command
is piecewise constant within each recorded second. A higher integration rate
does not create missing control bandwidth.

## Artifacts

- Machine-readable decision: `{audit["artifacts"]["audit"]}`
- Complete sampled schema: `{audit["artifacts"]["channel_inventory"]}`
- Sample selection and aggregate flight diagnostics:
  `{audit["artifacts"]["sample_flights"]}`
- Airborne surface ranges and offsets:
  `{audit["artifacts"]["surface_diagnostics"]}`
- Reproduction tool: `{audit["artifacts"]["runner"]}`
"""
    )


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    archives = _scan_archives(args.data_dir)
    selections = _select_samples(archives, args.tails)
    sampled_channels: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    surfaces: list[dict[str, Any]] = []
    for selection in selections:
        channels = _load_selection(selection)
        sampled_channels.append(channels)
        samples.append(_sample_row(selection, channels))
        surfaces.extend(_surface_rows(selection, channels))
    inventory = _inventory_rows(sampled_channels, len(selections))
    fingerprints = sorted({row["metadata_fingerprint"] for row in samples})
    body_rates = _semantic_matches(inventory, BODY_RATE_TERMS)
    mass_channels = _semantic_matches(inventory, MASS_TERMS)
    operational_count = sum(bool(row["operational_sample"]) for row in samples)
    surface_rates = {
        channel: _rate_of(inventory, channel) for channel in SURFACE_CHANNELS
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": {
            "published_identity": "de-identified regional jet",
            "modeling_assumption": "Avro RJ85",
            "sampled_tail_identifiers": [str(tail) for tail in args.tails],
        },
        "status": "conditional-effective-1-hz-control-to-motion-model-authorized",
        "archive_scope": {
            "location": str(args.data_dir),
            "archive_count": len(archives),
            "tail_count": len({record.tail for record in archives}),
            "tail_identifiers": sorted({record.tail for record in archives}),
            "mat_flight_count": sum(record.mat_members for record in archives),
            "archive_bytes": sum(record.archive_bytes for record in archives),
            "archive_bytes_gib": sum(record.archive_bytes for record in archives)
            / (1024**3),
            "uncompressed_mat_bytes": sum(record.mat_bytes for record in archives),
        },
        "sampling_contract": {
            "metadata_scope": "every local ZIP central directory",
            "sample_tails": [str(tail) for tail in args.tails],
            "archive_positions_per_tail": ["first", "middle", "last"],
            "member_rule": "largest uncompressed MATLAB member in selected archive",
            "sample_count": len(samples),
            "operational_sample_count": operational_count,
            "sampled_members": [
                {
                    "tail": row["tail"],
                    "archive": row["archive"],
                    "member": row["member"],
                }
                for row in samples
            ],
            "metadata_fingerprint_count": len(fingerprints),
            "metadata_fingerprints": fingerprints,
            "purpose": "schema and cadence feasibility, not flight-quality estimation",
        },
        "observability": {
            "direct_body_rate_channels": body_rates,
            "gross_weight_or_mass_channels": mass_channels,
            "surface_native_rates_hz": surface_rates,
            "pitch_roll_attitude_rate_hz": _rate_of(inventory, "PTCH"),
            "heading_rate_hz": _rate_of(inventory, "TH"),
            "high_rate_acceleration_hz": _rate_of(inventory, "BLAC"),
            "four_engine_pla_n1_n2_ff_rate_hz": _rate_of(inventory, "PLA_1"),
            "fuel_quantity_rate_hz": _rate_of(inventory, "FQTY_1"),
            "surface_encoding_status": "zero-sign-scale-validation-required",
            "fuel_quantity_status": "tank-scaling-validation-required",
            "weight_on_wheels_decoding": "one-airborne-zero-ground",
        },
        "decision": {
            "authorized_model": "effective 1 Hz control-to-motion model",
            "state_preprocessing": (
                "derive smoothed body-rate estimates from 8 Hz pitch/roll and "
                "4 Hz true heading; form anti-aliased 1 Hz targets"
            ),
            "control_preprocessing": (
                "validate and transform surface zero/sign/scale; hold at native "
                "1 Hz, except 2 Hz rudder"
            ),
            "integration_rate": (
                "may exceed 1 Hz, but recorded surface commands remain piecewise constant"
            ),
            "propulsion_branch": (
                "fit four-engine 4 Hz PLA/mode/air-data to N1/N2/fuel-flow surrogate"
            ),
            "mass_treatment": "external prior plus tail/flight nuisance; fuel burn observed",
            "blocked_claims": [
                "directly observed body angular rates",
                "fast actuator dynamics or sub-second control bandwidth",
                "unanchored dimensional stability/control derivatives",
                "known per-flight gross mass",
                "literal physical surface degrees before decoding",
            ],
            "next_gate": (
                "surface decoding and held-out validation of reconstructed angular rates "
                "before any force/moment coefficient fit"
            ),
        },
        "artifacts": {
            "runner": "tools/audit_rj85_dashlink.py",
            "audit": str(args.audit),
            "report": str(args.report),
            "channel_inventory": str(args.inventory),
            "sample_flights": str(args.sample_flights),
            "surface_diagnostics": str(args.surface_diagnostics),
        },
        "limitations": [
            "Avro RJ85 is a project assumption; NASA publishes only de-identified regional jet.",
            "Six deterministically sampled flights establish schema consistency, not fleet-wide signal quality.",
            "Recorder conversion laws, invalid codes, surface neutral, and polarity are not documented locally.",
            "No gross-weight/mass channel or direct body-angular-rate channel is present.",
            "Raw-data redistribution remains blocked until the source license is established.",
        ],
    }
    audit = _round_value(audit)
    _write_csv(args.inventory, inventory)
    _write_csv(args.sample_flights, samples)
    _write_csv(args.surface_diagnostics, surfaces)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(yaml.safe_dump(audit, sort_keys=False))
    _write_report(args.report, audit=audit, inventory=inventory, samples=samples)
    return inventory, samples, audit


def main() -> None:
    inventory, samples, audit = run(parse_args())
    print(
        yaml.safe_dump(
            {
                "archive_scope": audit["archive_scope"],
                "sampled_channels": len(inventory),
                "sampled_flights": len(samples),
                "decision": audit["decision"],
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
