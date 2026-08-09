"""Typed decoding for the two Airbus A320-214 QAR files.

This module deliberately distinguishes verified engineering conversions from
provisional recorder-code hypotheses. A provisional channel remains tagged as
such after conversion so model-fitting code cannot mistake it for ground truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DateTimeArray = NDArray[np.datetime64]

TIME_COLUMN = "Time..ms."
DATE_COLUMN = "DATE"
UTC_COLUMN = "UTC"
RECORDING_COLUMN = "recording"
GEAR_COLUMNS = (
    "LDGN_ON..ON.OFF.",
    "LDG_ON_1..ON.OFF.",
    "LDG_ON_2..ON.OFF.",
)


class DecodeStatus(StrEnum):
    """Evidence level for a recorder-to-engineering conversion."""

    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ChannelSpec:
    """Declarative conversion and validity rule for one raw QAR channel."""

    logical_name: str
    raw_name: str
    raw_unit: str
    output_unit: str
    scale: float = 1.0
    offset: float = 0.0
    status: DecodeStatus = DecodeStatus.UNRESOLVED
    max_hold_s: float = 1.0
    raw_min: float | None = None
    raw_max: float | None = None

    def convert(self, raw: Sequence[float], time_s: FloatArray) -> FloatArray:
        values = np.asarray(raw, dtype=float)
        valid = np.isfinite(values)
        if self.raw_min is not None:
            valid &= values >= self.raw_min
        if self.raw_max is not None:
            valid &= values <= self.raw_max
        clean = np.where(valid, values, np.nan)
        held = bounded_sample_and_hold(clean, time_s, self.max_hold_s)
        return held * self.scale + self.offset


@dataclass(frozen=True)
class DecodedChannel:
    """Decoded values that retain their conversion evidence and unit."""

    values: FloatArray
    spec: ChannelSpec

    @property
    def unit(self) -> str:
        return self.spec.output_unit

    @property
    def status(self) -> DecodeStatus:
        return self.spec.status


@dataclass(frozen=True)
class DecodedA320Flight:
    """Time-aligned decoded channels for one registration and recording."""

    registration: str
    recording: str
    time_s: FloatArray
    timestamp_utc: DateTimeArray
    airborne: BoolArray
    channels: Mapping[str, DecodedChannel]

    def channel(
        self,
        name: str,
        *,
        allow_provisional: bool = False,
        allow_unresolved: bool = False,
    ) -> FloatArray:
        channel = self.channels[name]
        if channel.status is DecodeStatus.PROVISIONAL and not allow_provisional:
            raise ValueError(
                f"{name} uses a provisional conversion; pass "
                "allow_provisional=True explicitly"
            )
        if channel.status is DecodeStatus.UNRESOLVED and not allow_unresolved:
            raise ValueError(
                f"{name} is unresolved; pass allow_unresolved=True explicitly"
            )
        return channel.values


DEG_TO_RAD = math.pi / 180.0
KNOT_TO_MPS = 1852.0 / 3600.0
FT_TO_M = 0.3048
G0 = 9.80665


def _spec(
    logical_name: str,
    raw_name: str,
    raw_unit: str,
    output_unit: str,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    status: DecodeStatus,
    max_hold_s: float,
    raw_min: float | None = None,
    raw_max: float | None = None,
) -> ChannelSpec:
    return ChannelSpec(
        logical_name=logical_name,
        raw_name=raw_name,
        raw_unit=raw_unit,
        output_unit=output_unit,
        scale=scale,
        offset=offset,
        status=status,
        max_hold_s=max_hold_s,
        raw_min=raw_min,
        raw_max=raw_max,
    )


A320_CHANNELS: dict[str, ChannelSpec] = {
    "latitude": _spec(
        "latitude",
        "GPSLAT....",
        "deg",
        "rad",
        scale=DEG_TO_RAD,
        status=DecodeStatus.VERIFIED,
        max_hold_s=2.0,
        raw_min=-90.0,
        raw_max=90.0,
    ),
    "longitude": _spec(
        "longitude",
        "GPSLON....",
        "deg",
        "rad",
        scale=DEG_TO_RAD,
        status=DecodeStatus.VERIFIED,
        max_hold_s=2.0,
        raw_min=-180.0,
        raw_max=180.0,
    ),
    "pressure_altitude": _spec(
        "pressure_altitude",
        "ALT..ft.",
        "ft",
        "m",
        scale=FT_TO_M,
        status=DecodeStatus.VERIFIED,
        max_hold_s=2.0,
        raw_min=-2_000.0,
        raw_max=60_000.0,
    ),
    "calibrated_airspeed": _spec(
        "calibrated_airspeed",
        "CAS..Kts.",
        "kt",
        "m/s",
        scale=KNOT_TO_MPS,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=500.0,
    ),
    "groundspeed": _spec(
        "groundspeed",
        "GS..Kts.",
        "kt",
        "m/s",
        scale=KNOT_TO_MPS,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=700.0,
    ),
    "vertical_acceleration": _spec(
        "vertical_acceleration",
        "VRTACC..G.",
        "g",
        "m/s^2",
        scale=G0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=-5.0,
        raw_max=5.0,
    ),
    "lateral_acceleration": _spec(
        "lateral_acceleration",
        "LATACC..G.",
        "g",
        "m/s^2",
        scale=G0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=-5.0,
        raw_max=5.0,
    ),
    "longitudinal_acceleration": _spec(
        "longitudinal_acceleration",
        "LONGACC..G.",
        "g",
        "m/s^2",
        scale=G0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=-5.0,
        raw_max=5.0,
    ),
    "pitch": _spec(
        "pitch",
        "PITCH....",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-120.0,
        raw_max=120.0,
    ),
    "roll": _spec(
        "roll",
        "ROLL....",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-180.0,
        raw_max=180.0,
    ),
    "true_heading": _spec(
        "true_heading",
        "TH",
        "deg",
        "rad",
        scale=DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-180.0,
        raw_max=180.0,
    ),
    "flight_path_angle": _spec(
        "flight_path_angle",
        "FLIGHT_PATH",
        "deg",
        "rad",
        scale=DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-30.0,
        raw_max=30.0,
    ),
    "angle_of_attack": _spec(
        "angle_of_attack",
        "AOAI_1",
        "recorder_code",
        "rad",
        scale=0.25 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-80.0,
        raw_max=160.0,
    ),
    "pitch_rate": _spec(
        "pitch_rate",
        "PTCR",
        "recorder_code",
        "rad/s",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-40.0,
        raw_max=40.0,
    ),
    "roll_rate": _spec(
        "roll_rate",
        "ROLR",
        "recorder_code",
        "rad/s",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-80.0,
        raw_max=80.0,
    ),
    "yaw_rate": _spec(
        "yaw_rate",
        "YAW",
        "recorder_code",
        "rad/s",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=-80.0,
        raw_max=80.0,
    ),
    "aileron_1": _spec(
        "aileron_1",
        "AIL_1",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "aileron_2": _spec(
        "aileron_2",
        "AIL_2",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "elevator_1": _spec(
        "elevator_1",
        "ELEV_1",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "elevator_2": _spec(
        "elevator_2",
        "ELEV_2",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "rudder": _spec(
        "rudder",
        "RUDD",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "stabilizer": _spec(
        "stabilizer",
        "STAB",
        "recorder_code",
        "rad",
        scale=0.5 * DEG_TO_RAD,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
    ),
    "throttle_1": _spec(
        "throttle_1",
        "TLA_1",
        "recorder_code",
        "detent_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=1.0,
    ),
    "throttle_2": _spec(
        "throttle_2",
        "TLA_2",
        "recorder_code",
        "detent_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=1.0,
    ),
    "flap": _spec(
        "flap",
        "FLAP_RAW",
        "recorder_code",
        "configuration_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=2.0,
    ),
    "fuel_flow_1": _spec(
        "fuel_flow_1",
        "FF_1_KG..Kg.h.",
        "kg/h",
        "kg/s",
        scale=1.0 / 3600.0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=10_000.0,
    ),
    "fuel_flow_2": _spec(
        "fuel_flow_2",
        "FF_2_KG..Kg.h.",
        "kg/h",
        "kg/s",
        scale=1.0 / 3600.0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=10_000.0,
    ),
    "fuel_quantity": _spec(
        "fuel_quantity",
        "WFQ_T_KG..KG.",
        "kg",
        "kg",
        status=DecodeStatus.VERIFIED,
        max_hold_s=4.0,
        raw_min=0.0,
        raw_max=30_000.0,
    ),
    "gross_weight": _spec(
        "gross_weight",
        "GW_1_KG..10.Kg.",
        "10 kg",
        "kg",
        scale=10.0,
        status=DecodeStatus.VERIFIED,
        max_hold_s=2.0,
        raw_min=3_000.0,
        raw_max=10_000.0,
    ),
    "n1_1": _spec(
        "n1_1",
        "N1_1",
        "recorder_code",
        "%",
        scale=0.5,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=220.0,
    ),
    "n1_2": _spec(
        "n1_2",
        "N1_2",
        "recorder_code",
        "%",
        scale=0.5,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=220.0,
    ),
    "n2_1": _spec(
        "n2_1",
        "N2_1",
        "recorder_code",
        "%",
        scale=0.5,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=240.0,
    ),
    "n2_2": _spec(
        "n2_2",
        "N2_2",
        "recorder_code",
        "%",
        scale=0.5,
        status=DecodeStatus.PROVISIONAL,
        max_hold_s=1.0,
        raw_min=0.0,
        raw_max=240.0,
    ),
    "wind_direction_raw": _spec(
        "wind_direction_raw",
        "WD",
        "recorder_code",
        "recorder_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=4.0,
    ),
    "wind_speed_raw": _spec(
        "wind_speed_raw",
        "WS",
        "recorder_code",
        "recorder_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=4.0,
    ),
    "total_air_temperature_raw": _spec(
        "total_air_temperature_raw",
        "TAT",
        "recorder_code",
        "recorder_code",
        status=DecodeStatus.UNRESOLVED,
        max_hold_s=2.0,
    ),
}


def bounded_sample_and_hold(
    values: Sequence[float],
    time_s: Sequence[float],
    max_hold_s: float,
    *,
    initial_backfill: bool = True,
) -> FloatArray:
    """Forward-fill samples without carrying them beyond ``max_hold_s``.

    An initial value may be backfilled from the first native sample only when it
    lies within the same hold bound. This handles staggered recorder cadences
    without filling through a long outage or across a flight boundary.
    """
    data = np.asarray(values, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if data.ndim != 1 or time.ndim != 1 or len(data) != len(time):
        raise ValueError(
            "values and time_s must be equal-length one-dimensional arrays"
        )
    if max_hold_s < 0:
        raise ValueError("max_hold_s must be non-negative")
    if len(data) == 0:
        return data.copy()
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) < 0):
        raise ValueError("time_s must be finite and nondecreasing")

    valid = np.isfinite(data)
    source_index = np.where(valid, np.arange(len(data)), -1)
    source_index = np.maximum.accumulate(source_index)
    output = np.full(len(data), np.nan, dtype=float)
    has_source = source_index >= 0
    ages = np.full(len(data), np.inf, dtype=float)
    ages[has_source] = time[has_source] - time[source_index[has_source]]
    held = has_source & (ages <= max_hold_s)
    output[held] = data[source_index[held]]

    if initial_backfill and np.any(valid):
        first = int(np.flatnonzero(valid)[0])
        leading = np.arange(first)
        close = time[first] - time[leading] <= max_hold_s
        output[leading[close]] = data[first]
    return output


def _filled_strings(values: Sequence[object]) -> list[str | None]:
    clean = [str(value).strip() if value is not None else "" for value in values]
    clean = [value if value and value.lower() != "nan" else None for value in clean]
    result: list[str | None] = []
    previous: str | None = None
    for value in clean:
        if value is not None:
            previous = value
        result.append(previous)
    following: str | None = None
    for index in range(len(result) - 1, -1, -1):
        if clean[index] is not None:
            following = clean[index]
        if result[index] is None:
            result[index] = following
    return result


def reconstruct_utc_timestamps(
    time_ms: Sequence[float],
    dates: Sequence[object],
    utc_times: Sequence[object],
    *,
    year: int = 2011,
) -> DateTimeArray:
    """Reconstruct one UTC timestamp per row from staggered DATE/UTC samples."""
    elapsed_ms = np.asarray(time_ms, dtype=float)
    if not (len(elapsed_ms) == len(dates) == len(utc_times)):
        raise ValueError("time, date, and UTC arrays must have equal lengths")
    if len(elapsed_ms) == 0 or np.any(~np.isfinite(elapsed_ms)):
        raise ValueError("time_ms must be non-empty and finite")

    filled_dates = _filled_strings(dates)
    base_candidates: list[datetime] = []
    for elapsed, date_text, utc_value in zip(
        elapsed_ms, filled_dates, utc_times, strict=True
    ):
        utc_text = str(utc_value).strip() if utc_value is not None else ""
        if not date_text or not utc_text or utc_text.lower() == "nan":
            continue
        try:
            timestamp = datetime.strptime(
                f"{date_text}/{year} {utc_text}", "%d/%m/%Y %H:%M:%S"
            )
        except ValueError:
            continue
        base_candidates.append(timestamp - timedelta(milliseconds=float(elapsed)))
    if not base_candidates:
        raise ValueError("no valid DATE/UTC anchor was found")

    # Most rows yield one exact base time. Mode is intentionally more robust
    # than an average to isolated corrupt clock samples and date rollover lag.
    base = Counter(base_candidates).most_common(1)[0][0]
    base_ns = np.datetime64(base, "ns")
    elapsed_ns = np.rint(elapsed_ms * 1_000_000.0).astype("timedelta64[ns]")
    return base_ns + elapsed_ns


def raw_columns() -> list[str]:
    """Return the raw columns required by :func:`decode_a320_flight`."""
    return [
        RECORDING_COLUMN,
        TIME_COLUMN,
        DATE_COLUMN,
        UTC_COLUMN,
        *GEAR_COLUMNS,
        *(spec.raw_name for spec in A320_CHANNELS.values()),
    ]


def decode_a320_flight(
    raw: Mapping[str, Sequence[object]],
    *,
    registration: str,
    recording: str,
    year: int = 2011,
) -> DecodedA320Flight:
    """Decode and time-align one raw A320 recording.

    Rows without recorder time are discarded before sorting. The caller must
    already have isolated one registration and recording.
    """
    missing = sorted(set(raw_columns()) - set(raw))
    if missing:
        raise KeyError(f"missing raw A320 QAR columns: {missing}")
    raw_time = np.asarray(raw[TIME_COLUMN], dtype=float)
    keep = np.isfinite(raw_time)
    order = np.argsort(raw_time[keep], kind="stable")
    indices = np.flatnonzero(keep)[order]
    time_ms = raw_time[indices]
    time_s = (time_ms - time_ms[0]) / 1000.0

    timestamps = reconstruct_utc_timestamps(
        time_ms,
        np.asarray(raw[DATE_COLUMN], dtype=object)[indices],
        np.asarray(raw[UTC_COLUMN], dtype=object)[indices],
        year=year,
    )
    gear = np.column_stack(
        [
            bounded_sample_and_hold(
                np.asarray(raw[column], dtype=float)[indices], time_s, 1.0
            )
            for column in GEAR_COLUMNS
        ]
    )
    airborne = np.nanmedian(gear, axis=1) < 0.5

    channels: dict[str, DecodedChannel] = {}
    for logical_name, spec in A320_CHANNELS.items():
        decoded = spec.convert(np.asarray(raw[spec.raw_name])[indices], time_s)
        channels[logical_name] = DecodedChannel(values=decoded, spec=spec)
    return DecodedA320Flight(
        registration=registration,
        recording=recording,
        time_s=time_s,
        timestamp_utc=timestamps,
        airborne=airborne,
        channels=channels,
    )
