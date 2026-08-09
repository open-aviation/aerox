#!/usr/bin/env python3
"""Calibrate DASHlink corrected AOA against an independent kinematic target.

The six flights chosen by the channel/cadence audit are development data. A
new confirmation panel is selected from ZIP metadata only and written before
any of those members are decompressed. This runner does not evaluate the new
panel.
"""

from __future__ import annotations

import argparse
import csv
import math
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat

from tools.audit_rj85_dashlink import (
    ArchiveRecord,
    _sample_indices,
    _scan_archives,
)

FT_MIN_TO_M_S = 0.00508
KNOT_TO_M_S = 0.514444
FIT_CANDIDATES = ("constant", "identity", "offset", "affine")


@dataclass(frozen=True)
class AoaFlight:
    tail: str
    archive: str
    member: str
    raw_aoa_deg: np.ndarray
    kinematic_aoa_deg: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.tail}/{self.member}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dashlink"))
    parser.add_argument(
        "--fit-panel",
        type=Path,
        default=Path("docs/modeling/rj85-dashlink-sample-flights.csv"),
    )
    parser.add_argument(
        "--confirmation-panel",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-confirmation-panel.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/modeling/rj85-aoa-calibration.md")
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-calibration.yml"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/rj85-aoa-fit-metrics.csv"),
    )
    return parser.parse_args()


def _array(mat: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(mat[name].data, dtype=float).reshape(-1)


def _block_mean(
    values: np.ndarray, native_rate_hz: int, target_rate_hz: int
) -> np.ndarray:
    if native_rate_hz % target_rate_hz:
        raise ValueError("native rate must be an integer multiple of target rate")
    factor = native_rate_hz // target_rate_hz
    if len(values) % factor:
        raise ValueError("channel length does not align to target rate")
    return values.reshape(-1, factor).mean(axis=1)


def _repeat(values: np.ndarray, native_rate_hz: int, target_rate_hz: int) -> np.ndarray:
    if target_rate_hz % native_rate_hz:
        raise ValueError("target rate must be an integer multiple of native rate")
    return np.repeat(values, target_rate_hz // native_rate_hz)


def _load_flight(
    data_dir: Path, row: dict[str, str], *, minimum_clean_samples: int = 5_000
) -> AoaFlight:
    path = data_dir / row["archive"]
    with zipfile.ZipFile(path) as archive:
        mat = loadmat(
            BytesIO(archive.read(row["member"])),
            squeeze_me=True,
            struct_as_record=False,
        )
    pitch = _block_mean(_array(mat, "PTCH"), 8, 4)
    roll = _block_mean(_array(mat, "ROLL"), 8, 4)
    vertical_speed_m_s = _block_mean(_array(mat, "IVV"), 16, 4) * FT_MIN_TO_M_S
    tas_m_s = _array(mat, "TAS") * KNOT_TO_M_S
    raw_aoa = _array(mat, "AOAC")
    altitude_ft = _array(mat, "ALT")
    airborne = _repeat(_array(mat, "WOW"), 1, 4) > 0.5
    lengths = {
        len(pitch),
        len(roll),
        len(vertical_speed_m_s),
        len(tas_m_s),
        len(raw_aoa),
        len(altitude_ft),
        len(airborne),
    }
    if len(lengths) != 1:
        raise ValueError(f"unaligned 4 Hz channels in {path.name}/{row['member']}")
    flight_path_angle = np.rad2deg(
        np.arcsin(np.clip(vertical_speed_m_s / np.maximum(tas_m_s, 1.0), -0.99, 0.99))
    )
    kinematic_aoa = pitch - flight_path_angle
    finite = np.isfinite(raw_aoa) & np.isfinite(kinematic_aoa)
    keep = (
        airborne
        & finite
        & (altitude_ft >= 5_000.0)
        & (tas_m_s >= 100.0)
        & (np.abs(roll) <= 10.0)
        & (raw_aoa >= -10.0)
        & (raw_aoa <= 10.0)
        & (kinematic_aoa >= -10.0)
        & (kinematic_aoa <= 15.0)
    )
    if np.count_nonzero(keep) < minimum_clean_samples:
        raise ValueError(
            f"insufficient clean AOA samples in {path.name}/{row['member']}"
        )
    return AoaFlight(
        tail=row["tail"],
        archive=row["archive"],
        member=row["member"],
        raw_aoa_deg=raw_aoa[keep],
        kinematic_aoa_deg=kinematic_aoa[keep],
    )


def _read_panel(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"tail", "archive", "member"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"invalid DASHlink panel: {path}")
    return rows


def _candidate_design(
    flight: AoaFlight, candidate: str
) -> tuple[np.ndarray, np.ndarray]:
    raw = flight.raw_aoa_deg
    target = flight.kinematic_aoa_deg
    if candidate == "constant":
        return np.ones((len(raw), 1)), target
    if candidate == "identity":
        return np.empty((len(raw), 0)), target - raw
    if candidate == "offset":
        return np.ones((len(raw), 1)), target - raw
    if candidate == "affine":
        return np.column_stack((np.ones(len(raw)), raw)), target
    raise ValueError(f"unknown candidate: {candidate}")


def _fit_equal_flight_huber(flights: list[AoaFlight], candidate: str) -> np.ndarray:
    if candidate == "identity":
        return np.empty(0)
    designs = []
    targets = []
    base_weights = []
    for flight in flights:
        design, target = _candidate_design(flight, candidate)
        designs.append(design)
        targets.append(target)
        base_weights.append(np.full(len(target), 1.0 / len(target)))
    design = np.vstack(designs)
    target = np.concatenate(targets)
    base_weight = np.concatenate(base_weights)
    sqrt_weight = np.sqrt(base_weight)
    coefficient = np.linalg.lstsq(
        design * sqrt_weight[:, None], target * sqrt_weight, rcond=None
    )[0]
    for _ in range(30):
        residual = target - design @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if not math.isfinite(scale) or scale < 1e-8:
            break
        huber = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-12))
        weight = base_weight * huber
        sqrt_weight = np.sqrt(weight)
        updated = np.linalg.lstsq(
            design * sqrt_weight[:, None], target * sqrt_weight, rcond=None
        )[0]
        if np.max(np.abs(updated - coefficient)) < 1e-10:
            coefficient = updated
            break
        coefficient = updated
    return coefficient


def _predict(flight: AoaFlight, candidate: str, coefficient: np.ndarray) -> np.ndarray:
    raw = flight.raw_aoa_deg
    if candidate == "constant":
        return np.full(len(raw), coefficient[0])
    if candidate == "identity":
        return raw
    if candidate == "offset":
        return raw + coefficient[0]
    if candidate == "affine":
        return coefficient[0] + coefficient[1] * raw
    raise ValueError(candidate)


def _metrics(
    flight: AoaFlight, candidate: str, prediction: np.ndarray, fold: int
) -> dict[str, Any]:
    error = prediction - flight.kinematic_aoa_deg
    return {
        "fold": fold,
        "tail": flight.tail,
        "archive": flight.archive,
        "member": flight.member,
        "candidate": candidate,
        "samples": len(error),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "mae_deg": float(np.mean(np.abs(error))),
        "bias_deg": float(np.mean(error)),
        "raw_target_correlation": float(
            np.corrcoef(flight.raw_aoa_deg, flight.kinematic_aoa_deg)[0, 1]
        ),
    }


def _lofo(
    flights: list[AoaFlight],
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    rows: list[dict[str, Any]] = []
    coefficients: dict[str, list[list[float]]] = {name: [] for name in FIT_CANDIDATES}
    for fold, held_out in enumerate(flights):
        train = [flight for flight in flights if flight is not held_out]
        for candidate in FIT_CANDIDATES:
            coefficient = _fit_equal_flight_huber(train, candidate)
            coefficients[candidate].append(coefficient.tolist())
            prediction = _predict(held_out, candidate, coefficient)
            rows.append(_metrics(held_out, candidate, prediction, fold))
    return rows, coefficients


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    constant = {
        row["fold"]: row["rmse_deg"] for row in rows if row["candidate"] == "constant"
    }
    for candidate in FIT_CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        improvements = [
            100.0 * (constant[row["fold"]] - row["rmse_deg"]) / constant[row["fold"]]
            for row in selected
        ]
        result.append(
            {
                "candidate": candidate,
                "mean_lofo_rmse_deg": float(
                    np.mean([row["rmse_deg"] for row in selected])
                ),
                "max_lofo_rmse_deg": float(
                    np.max([row["rmse_deg"] for row in selected])
                ),
                "mean_improvement_vs_constant_pct": float(np.mean(improvements)),
                "flights_improved_vs_constant": int(
                    np.count_nonzero(np.asarray(improvements) > 0)
                ),
                "flight_count": len(selected),
            }
        )
    return result


def _choose_confirmation_archives(
    archives: list[ArchiveRecord], fit_rows: list[dict[str, str]], tails: list[str]
) -> list[dict[str, Any]]:
    used_archives = {row["archive"] for row in fit_rows}
    panel: list[dict[str, Any]] = []
    for tail in tails:
        available = sorted(
            (
                record
                for record in archives
                if record.tail == tail and record.path.name not in used_archives
            ),
            key=lambda record: record.part,
        )
        if len(available) < 3:
            raise ValueError(f"tail {tail} lacks three unused confirmation archives")
        for index in _sample_indices(len(available)):
            record = available[index]
            with zipfile.ZipFile(record.path) as archive:
                members = [
                    item
                    for item in archive.infolist()
                    if not item.is_dir() and item.filename.lower().endswith(".mat")
                ]
            member = max(members, key=lambda item: (item.file_size, item.filename))
            panel.append(
                {
                    "tail": tail,
                    "archive": record.path.name,
                    "archive_part": record.part,
                    "member": member.filename,
                    "member_bytes": member.file_size,
                    "assignment": "independent_confirmation",
                    "selection_rule": "largest member in first/middle/last unused archive",
                }
            )
    return panel


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if isinstance(value, (np.floating, float)):
        return round(float(value), 10) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_report(
    path: Path, calibration: dict[str, Any], candidates: list[dict[str, Any]]
) -> None:
    selected = next(row for row in candidates if row["candidate"] == "affine")
    coefficients = calibration["selected_model"]["coefficients"]
    path.write_text(
        f"""# RJ85/DASHlink corrected-AOA calibration

## Fit-only decision

Select the pooled affine decoder

`alpha_effective_deg = {coefficients["offset_deg"]:.6f} + {coefficients["raw_aoa_scale"]:.6f} * AOAC_raw`

as a **provisional, fit-only** mapping from the DASHlink corrected-AOA channel
to kinematic angle of attack. Across six complete development flights, its
leave-one-flight-out RMSE is {selected["mean_lofo_rmse_deg"]:.3f} degrees and it
improves the fixed training-mean baseline by
{selected["mean_improvement_vs_constant_pct"]:.1f}% on average. It improves all
{selected["flight_count"]} flights.

The raw channel is not already in physical angle degrees: its fitted scale is
close to one-half and the offset is about 4.4 degrees. The target is independently
formed at 4 Hz as pitch minus flight-path angle, with flight-path angle derived
from inertial vertical speed and true airspeed.

## Leakage control

The six flights from the channel/cadence audit are development data. Before any
new response was loaded, a confirmation panel was selected using only tail,
archive part, member name, and uncompressed member size. It contains three new
members per tail from first/middle/last unused archives. This runner writes that
panel but never decompresses or evaluates it.

The selected model, coefficients, filters, and confirmation gates are frozen in
`docs/modeling/rj85-aoa-calibration.yml`. The new flights cannot be used to
retune the mapping after evaluation.

## Fit candidates

| candidate | mean LOFO RMSE (deg) | max RMSE | improvement vs constant | flights improved |
|---|---:|---:|---:|---:|
"""
        + "\n".join(
            f"| {row['candidate']} | {row['mean_lofo_rmse_deg']:.3f} | "
            f"{row['max_lofo_rmse_deg']:.3f} | "
            f"{row['mean_improvement_vs_constant_pct']:.1f}% | "
            f"{row['flights_improved_vs_constant']}/{row['flight_count']} |"
            for row in candidates
        )
        + """

## Interpretation and limits

This calibration makes AOAC usable as an effective aerodynamic state for the
normal, clean, wings-level envelope. It is not a vane calibration, sideslip
model, high-angle-of-attack result, or proof of RJ85 type identity. Vertical
wind and inertial-vertical-speed bias are absorbed into the effective mapping.
Independent confirmation is required before use in force/moment fitting.
"""
    )


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_rows = _read_panel(args.fit_panel)
    flights = [_load_flight(args.data_dir, row) for row in fit_rows]
    lofo_rows, lofo_coefficients = _lofo(flights)
    candidates = _candidate_summary(lofo_rows)
    selected = next(row for row in candidates if row["candidate"] == "affine")
    full_coefficient = _fit_equal_flight_huber(flights, "affine")
    affine_folds = np.asarray(lofo_coefficients["affine"], dtype=float)
    fit_target_mean = float(
        np.mean([np.mean(flight.kinematic_aoa_deg) for flight in flights])
    )
    pass_fit_gate = bool(
        selected["mean_lofo_rmse_deg"] <= 0.30
        and selected["mean_improvement_vs_constant_pct"] >= 50.0
        and selected["flights_improved_vs_constant"] == len(flights)
        and np.all(affine_folds[:, 1] > 0)
        and np.std(affine_folds[:, 1]) / abs(np.mean(affine_folds[:, 1])) <= 0.15
    )
    if not pass_fit_gate:
        raise RuntimeError("affine AOA candidate failed the pre-confirmation fit gate")
    archives = _scan_archives(args.data_dir)
    tails = sorted({flight.tail for flight in flights})
    confirmation_panel = _choose_confirmation_archives(archives, fit_rows, tails)
    fit_keys = {(row["archive"], row["member"]) for row in fit_rows}
    confirmation_keys = {(row["archive"], row["member"]) for row in confirmation_panel}
    if fit_keys & confirmation_keys:
        raise ValueError("fit member entered independent confirmation panel")
    calibration: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-05",
        "aircraft": {
            "published_identity": "de-identified regional jet",
            "modeling_assumption": "Avro RJ85",
        },
        "status": "fit-only-affine-aoa-decoder-selected-awaiting-confirmation",
        "fit_data": {
            "panel": str(args.fit_panel),
            "flights": [flight.key for flight in flights],
            "flight_count": len(flights),
            "sample_rate_hz": 4,
            "equal_total_weight_per_flight": True,
            "cross_validation": "leave-one-complete-flight-out",
        },
        "kinematic_target": {
            "formula": "pitch_deg - asin((IVV_ft_min * 0.00508) / (TAS_kt * 0.514444)) in degrees",
            "filters": {
                "wow_raw": "greater than 0.5 (one means airborne)",
                "minimum_pressure_altitude_ft": 5_000,
                "minimum_tas_m_s": 100,
                "maximum_absolute_roll_deg": 10,
                "raw_aoa_bounds_deg": [-10, 10],
                "kinematic_aoa_bounds_deg": [-10, 15],
            },
            "limitations": [
                "vertical wind is assumed zero",
                "sideslip effects are neglected in the wings-level screen",
                "target inherits pitch, IVV, and TAS recorder errors",
            ],
        },
        "selected_model": {
            "name": "pooled_affine",
            "formula": "effective_aoa_deg = offset_deg + raw_aoa_scale * AOAC_raw",
            "coefficients": {
                "offset_deg": float(full_coefficient[0]),
                "raw_aoa_scale": float(full_coefficient[1]),
            },
            "training_mean_kinematic_aoa_deg": fit_target_mean,
            "fit_gate_passed": pass_fit_gate,
            "lofo_coefficients": {
                "offset_deg": affine_folds[:, 0].tolist(),
                "raw_aoa_scale": affine_folds[:, 1].tolist(),
                "raw_aoa_scale_cv": float(
                    np.std(affine_folds[:, 1]) / abs(np.mean(affine_folds[:, 1]))
                ),
            },
        },
        "confirmation_contract": {
            "panel": str(args.confirmation_panel),
            "selection": "metadata-only before response evaluation",
            "flight_count": len(confirmation_panel),
            "minimum_valid_samples_per_flight": 5_000,
            "no_refit": True,
            "gates": {
                "maximum_mean_rmse_deg": 0.35,
                "minimum_mean_improvement_vs_fixed_training_mean_pct": 50.0,
                "minimum_flights_improved_vs_fixed_training_mean": 5,
                "maximum_absolute_mean_bias_deg": 0.25,
                "minimum_flights_with_raw_target_correlation_at_least_0_8": 5,
            },
        },
        "artifacts": {
            "runner": "tools/calibrate_rj85_aoa.py",
            "calibration": str(args.calibration),
            "report": str(args.report),
            "candidate_metrics": str(args.candidate_metrics),
            "flight_metrics": str(args.flight_metrics),
            "confirmation_panel": str(args.confirmation_panel),
        },
        "limitations": [
            "Fit-only mapping cannot enter force/moment identification until independent confirmation passes.",
            "Mapping is valid only inside the recorded filter envelope.",
            "It is an effective recorder calibration, not a physical AOA-vane calibration.",
            "Avro RJ85 remains a project identity assumption.",
        ],
    }
    calibration = _native(calibration)
    _write_csv(args.candidate_metrics, candidates)
    _write_csv(args.flight_metrics, lofo_rows)
    _write_csv(args.confirmation_panel, confirmation_panel)
    args.calibration.write_text(yaml.safe_dump(calibration, sort_keys=False))
    _write_report(args.report, calibration, candidates)
    return candidates, calibration


def main() -> None:
    candidates, calibration = run(parse_args())
    print(yaml.safe_dump(calibration["selected_model"], sort_keys=False))
    print(yaml.safe_dump(calibration["confirmation_contract"], sort_keys=False))
    print("candidates:")
    for row in candidates:
        print(row)


if __name__ == "__main__":
    main()
