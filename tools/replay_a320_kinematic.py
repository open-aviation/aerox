#!/usr/bin/env python3
"""Run the first open-loop kinematic replay of the golden A320 QAR flight."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    FT_TO_M,
    KNOT_TO_MPS,
    decode_a320_flight,
    raw_columns,
)
from aerox.modeling.dynamics.kinematics import (
    KinematicTrajectory,
    air_velocity_from_orientation,
    air_velocity_from_heading_flight_path,
    differentiate_wgs84_intervals,
    ground_velocity_from_track,
    integrate_wgs84,
    integrate_wgs84_intervals,
    replay_metrics,
)


FT_PER_MIN_TO_MPS = FT_TO_M / 60.0

PROCESSED_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "groundspeed",
    "track",
    "vertical_rate",
    "TAS",
    "u_component_of_wind",
    "v_component_of_wind",
    "flight_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=Path("data/a320_raw/tail-a_2011.parquet")
    )
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument("--registration", default="TAIL-A")
    parser.add_argument("--recording", default="FLT0010")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-kinematic-replay.md"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-kinematic-replay-metrics.csv"),
    )
    return parser.parse_args()


def _load_raw(path: Path, recording: str) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=raw_columns(),
        filters=[("recording", "==", recording)],
        engine="pyarrow",
    )


def _load_processed(path: Path, recording: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "==", recording)],
        engine="pyarrow",
    )
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _raw_indices_for_processed(
    raw_timestamps: np.ndarray, processed_timestamps: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(raw_timestamps, processed_timestamps)
    valid = indices < len(raw_timestamps)
    if not np.all(valid) or not np.all(raw_timestamps[indices] == processed_timestamps):
        missing = int(np.sum(~valid | (raw_timestamps[np.minimum(indices, len(raw_timestamps) - 1)] != processed_timestamps)))
        raise ValueError(f"{missing} processed timestamps do not match raw QAR rows")
    return indices


def _integrate(
    time_s: np.ndarray,
    initial_position: tuple[float, float, float],
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> KinematicTrajectory:
    return integrate_wgs84(time_s, *initial_position, *velocity)


def _velocity_errors(
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    north, east, down = velocity
    ref_north, ref_east, ref_down = reference
    horizontal = np.hypot(north - ref_north, east - ref_east)
    vertical = down - ref_down
    return (
        float(np.sqrt(np.mean(horizontal**2))),
        float(np.sqrt(np.mean(vertical**2))),
    )


def _intervals_to_nodes(values: np.ndarray) -> np.ndarray:
    nodes = np.empty(len(values) + 1, dtype=float)
    nodes[0] = values[0]
    nodes[-1] = values[-1]
    nodes[1:-1] = 0.5 * (values[:-1] + values[1:])
    return nodes


def _metric_row(
    name: str,
    description: str,
    trajectory: KinematicTrajectory,
    observed: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, object]:
    metrics = asdict(replay_metrics(trajectory, *observed))
    horizontal_velocity, vertical_velocity = _velocity_errors(
        (
            trajectory.north_velocity_mps,
            trajectory.east_velocity_mps,
            trajectory.down_velocity_mps,
        ),
        reference_velocity,
    )
    return {
        "replay": name,
        "description": description,
        **{key: round(value, 3) for key, value in metrics.items()},
        "horizontal_velocity_rmse_mps": round(horizontal_velocity, 4),
        "vertical_velocity_rmse_mps": round(vertical_velocity, 4),
    }


def _format_metrics(metrics: pd.DataFrame) -> str:
    lines = [
        "| Replay | Horizontal RMSE (m) | Horizontal p95 (m) | Final horizontal (m) | Vertical RMSE (m) | Vertical p95 (m) | Final vertical (m) | Horizontal velocity RMSE (m/s) | Vertical velocity RMSE (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.replay} | {row.horizontal_rmse_m:.1f} | "
            f"{row.horizontal_p95_m:.1f} | {row.horizontal_final_m:.1f} | "
            f"{row.vertical_rmse_m:.1f} | {row.vertical_p95_m:.1f} | "
            f"{row.vertical_final_m:.1f} | {row.horizontal_velocity_rmse_mps:.2f} | "
            f"{row.vertical_velocity_rmse_mps:.2f} |"
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    *,
    registration: str,
    recording: str,
    samples: int,
    duration_s: float,
    gamma_consistency_rmse_deg: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# A320 Golden-Flight Kinematic Replay

Status: first complete-flight open-loop baseline  
Date: 2026-08-04  
Flight: `{registration}/{recording}`  
Generator: `tools/replay_a320_kinematic.py`

## Purpose

This is a frame, unit, timing, and external-forcing test. It does not contain an
aerodynamic, propulsion, mass, or control-response model. The replay integrates
velocity continuously from the observed initial position without position or
altitude resets.

The comparison interval contains {samples:,} synchronized one-second samples
({duration_s / 3600.0:.2f} hours). Every processed timestamp has an exact raw QAR
timestamp match. The shorter processed interval, rather than the full raw airborne
interval, is used because it is the only interval with the current TAS and weather
surrogate.

## Replay variants

1. **Coordinate-difference closure** differentiates the observed coordinates into
   one mean NED velocity per interval and reintegrates them. It tests timestamp
   reconstruction, NED conventions, WGS 84 propagation, and numerical closure.
2. **Processed ground vector** uses processed groundspeed, track, and vertical
   rate. Its drift measures inconsistency between those derived channels and the
   processed coordinates; it is not numerical integration error.
3. **Heading/FPA + airspeed + wind** uses recorded true heading and flight-path
   angle, processed TAS, and processed horizontal wind.
4. **Orientation/AOA + airspeed + wind** forms body-axis air velocity with zero
   sideslip, rotates it using decoded roll, pitch, and true heading, and adds the
   processed horizontal wind.

The processed wind columns are interpreted as `u` eastward and `v` northward in
m/s following the ECMWF convention. Their actual upstream dataset and interpolation
lineage remain unknown, so variants 3 and 4 are diagnostic and cannot yet be an
acceptance test.

## Results

{_format_metrics(metrics)}

Velocity error is measured against node-centered velocities differentiated from
the observed coordinates before integration.
Position errors are full open-loop accumulated errors, so even a small persistent
heading, wind, or vertical-speed bias grows throughout the 2.43-hour flight.

The decoded small-angle consistency check compares recorded flight-path angle with
`pitch - angle_of_attack`; its complete-flight RMSE is
{gamma_consistency_rmse_deg:.2f} degrees. Roll and nonzero sideslip make that
identity approximate, but a large phase-dependent residual is also evidence of
incorrect recorder scale, bias, or latency.

## Interpretation

- Coordinate-difference closure is the numerical floor and must remain near
  floating-point precision.
- Processed-ground-vector drift is a derivation consistency result. On this
  flight, groundspeed/track have small persistent horizontal component differences
  and vertical rate has a small persistent difference relative to differentiated
  coordinates. Both accumulate materially over 2.43 hours.
- The heading/FPA result isolates heading, TAS, wind, and flight-path-angle errors.
- The attitude/AOA result additionally tests the provisional pitch, roll, and AOA
  conversions and the zero-sideslip assumption.
- Neither air-relative replay is expected to stay close for a complete flight yet:
  the wind lineage is unknown, the recorder conversions are provisional, and no
  sensor latency or sideslip has been estimated.

The important outcome is therefore diagnostic separation, not parameter fitting.
The next identification step is to estimate constant scale/bias and short channel
delays on calibration windows while keeping the six validation flights untouched.
Wind bias must be estimated separately from heading and air-data bias to avoid an
unobservable fit.

## Reproducibility and redistribution

- Typed decoder: `src/aerox/flightdata/a320_qar.py`
- WGS 84 kinematics: `src/aerox/modeling/dynamics/kinematics.py`
- Machine-readable results: `docs/modeling/a320-kinematic-replay-metrics.csv`
- Channel evidence: `docs/modeling/a320-qar-channel-map.yml`
- Sources: `docs/modeling/sources.yml`

Only aggregate errors are written to the repository. Raw coordinates and replayed
trajectory samples remain external because the QAR license is unknown.
"""
    path.write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    raw = _load_raw(args.raw, args.recording)
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=args.registration,
        recording=args.recording,
    )
    processed = _load_processed(args.processed, args.recording)
    processed_timestamps = processed["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_indices = _raw_indices_for_processed(
        decoded.timestamp_utc, processed_timestamps
    )
    time_s = (
        processed_timestamps - processed_timestamps[0]
    ).astype("timedelta64[ns]").astype(np.int64) / 1e9

    observed = (
        processed["latitude"].to_numpy(dtype=float) * DEG_TO_RAD,
        processed["longitude"].to_numpy(dtype=float) * DEG_TO_RAD,
        processed["altitude"].to_numpy(dtype=float) * FT_TO_M,
    )
    initial_position = tuple(values[0] for values in observed)
    ground_reference = ground_velocity_from_track(
        processed["groundspeed"].to_numpy(dtype=float) * KNOT_TO_MPS,
        processed["track"].to_numpy(dtype=float) * DEG_TO_RAD,
        processed["vertical_rate"].to_numpy(dtype=float) * FT_PER_MIN_TO_MPS,
    )
    coordinate_intervals = differentiate_wgs84_intervals(time_s, *observed)
    coordinate_closure = integrate_wgs84_intervals(
        time_s, initial_position[0], initial_position[1], initial_position[2],
        *coordinate_intervals
    )
    coordinate_reference = tuple(
        _intervals_to_nodes(values) for values in coordinate_intervals
    )
    coordinate_closure = KinematicTrajectory(
        coordinate_closure.latitude_rad,
        coordinate_closure.longitude_rad,
        coordinate_closure.altitude_m,
        *coordinate_reference,
    )

    tas = processed["TAS"].to_numpy(dtype=float) * KNOT_TO_MPS
    wind_north = processed["v_component_of_wind"].to_numpy(dtype=float)
    wind_east = processed["u_component_of_wind"].to_numpy(dtype=float)
    heading = decoded.channel(
        "true_heading", allow_provisional=True
    )[raw_indices]
    flight_path = decoded.channel(
        "flight_path_angle", allow_provisional=True
    )[raw_indices]
    fpa_air = air_velocity_from_heading_flight_path(tas, heading, flight_path)
    fpa_ground = (
        fpa_air[0] + wind_north,
        fpa_air[1] + wind_east,
        fpa_air[2],
    )

    pitch = decoded.channel("pitch", allow_provisional=True)[raw_indices]
    roll = decoded.channel("roll", allow_provisional=True)[raw_indices]
    alpha = decoded.channel("angle_of_attack", allow_provisional=True)[raw_indices]
    attitude_air = air_velocity_from_orientation(
        tas,
        alpha,
        np.zeros_like(tas),
        roll,
        pitch,
        heading,
    )
    attitude_ground = (
        attitude_air[0] + wind_north,
        attitude_air[1] + wind_east,
        attitude_air[2],
    )

    trajectories = {
        "coordinate-difference closure": coordinate_closure,
        "processed ground vector": _integrate(
            time_s, initial_position, ground_reference
        ),
        "heading/FPA + wind": _integrate(time_s, initial_position, fpa_ground),
        "attitude/AOA + wind": _integrate(
            time_s, initial_position, attitude_ground
        ),
    }
    descriptions = {
        "coordinate-difference closure": "observed geodetic interval velocities",
        "processed ground vector": "processed groundspeed, track, and vertical rate",
        "heading/FPA + wind": "processed TAS and wind; recorded true heading and FPA",
        "attitude/AOA + wind": "processed TAS and wind; provisional attitude/AOA; beta=0",
    }
    rows = [
        _metric_row(
            name,
            descriptions[name],
            trajectory,
            observed,
            coordinate_reference,
        )
        for name, trajectory in trajectories.items()
    ]
    metrics = pd.DataFrame(rows)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)

    gamma_error = np.arctan2(
        -attitude_air[2], np.hypot(attitude_air[0], attitude_air[1])
    ) - flight_path
    gamma_consistency_rmse_deg = float(
        np.sqrt(np.mean(gamma_error**2)) / DEG_TO_RAD
    )
    _write_report(
        args.report,
        metrics,
        registration=args.registration,
        recording=args.recording,
        samples=len(processed),
        duration_s=float(time_s[-1] - time_s[0]),
        gamma_consistency_rmse_deg=gamma_consistency_rmse_deg,
    )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run(args)
    print(metrics.to_string(index=False))
    print(f"wrote {args.metrics}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
