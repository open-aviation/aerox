#!/usr/bin/env python3
"""Freeze geometry-only adaptive anchors for the long-range A320 FMS test."""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_a320_fms_long_range_qar import _load_holdout
from tools.validate_a320_fms_qar_intent import _point_to_polyline_distance, _style_axis

UPDATED = "2026-08-06"
MAXIMUM_INTERVAL_S = 600
GEOMETRY_TOLERANCE_M = 1_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "anchors": "docs/fms/a320-fms-turn-aware-intent-anchors.csv",
        "parameters": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
        "report": "docs/fms/a320-fms-turn-aware-intent-selection.md",
        "figure-prefix": "docs/fms/figures/a320-fms-turn-aware-intent-selection",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _distance_to_chord(
    north_m: np.ndarray,
    east_m: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    points = np.column_stack((north_m[start + 1 : end], east_m[start + 1 : end]))
    first = np.array([north_m[start], east_m[start]])
    delta = np.array([north_m[end] - north_m[start], east_m[end] - east_m[start]])
    length_squared = float(delta @ delta)
    if not len(points):
        return np.empty(0)
    if length_squared <= 1e-12:
        return np.linalg.norm(points - first, axis=1)
    fraction = np.clip(((points - first) @ delta) / length_squared, 0.0, 1.0)
    projection = first + fraction[:, None] * delta
    return np.linalg.norm(points - projection, axis=1)


def select_turn_aware_indices(
    north_m: np.ndarray,
    east_m: np.ndarray,
    *,
    maximum_interval_s: int = MAXIMUM_INTERVAL_S,
    geometry_tolerance_m: float = GEOMETRY_TOLERANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mandatory time anchors plus deterministic RDP geometry anchors."""

    sample_count = len(north_m)
    mandatory = np.unique(
        np.r_[np.arange(0, sample_count, maximum_interval_s), sample_count - 1]
    ).astype(int)
    selected = set(mandatory.tolist())
    for first, last in pairwise(mandatory):
        stack = [(int(first), int(last))]
        while stack:
            start, end = stack.pop()
            distance = _distance_to_chord(north_m, east_m, start, end)
            if not len(distance):
                continue
            local = int(np.argmax(distance))
            if float(distance[local]) <= geometry_tolerance_m:
                continue
            index = start + 1 + local
            selected.add(index)
            stack.extend(((start, index), (index, end)))
    return np.array(sorted(selected), dtype=int), mandatory


def _turn_angle_deg(
    north_m: np.ndarray, east_m: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(indices))
    points = np.column_stack((north_m[indices], east_m[indices]))
    for index in range(1, len(indices) - 1):
        inbound = points[index] - points[index - 1]
        outbound = points[index + 1] - points[index]
        cross = inbound[0] * outbound[1] - inbound[1] * outbound[0]
        dot = float(inbound @ outbound)
        result[index] = abs(np.rad2deg(np.arctan2(cross, dot)))
    return result


def _path_error(qar: Any, indices: np.ndarray) -> np.ndarray:
    return _point_to_polyline_distance(
        qar.north_position_m,
        qar.east_position_m,
        qar.north_position_m[indices],
        qar.east_position_m[indices],
    )


def _plot(
    qar: Any,
    selected: np.ndarray,
    mandatory: np.ndarray,
    base_error: np.ndarray,
    adaptive_error: np.ndarray,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    black, blue, orange, green = "#000000", "#0072B2", "#D55E00", "#009E73"
    time_min = qar.reference.time_s / 60.0
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    axes[0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=black,
        linestyle=":",
        linewidth=1.5,
        label="QAR geometry",
    )
    axes[0].plot(
        qar.east_position_m[mandatory] / 1_000.0,
        qar.north_position_m[mandatory] / 1_000.0,
        color=orange,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markersize=3,
        label="600-s anchors",
    )
    axes[0].plot(
        qar.east_position_m[selected] / 1_000.0,
        qar.north_position_m[selected] / 1_000.0,
        color=blue,
        linewidth=1.3,
        marker="s",
        markerfacecolor="white",
        markersize=3,
        label="turn-aware anchors",
    )
    axes[0].set(
        xlabel="Local east displacement (km)",
        ylabel="Local north displacement (km)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(time_min, base_error, color=orange, linestyle="--", label="600-s")
    axes[1].plot(time_min, adaptive_error, color=green, label="turn-aware")
    axes[1].axhline(
        GEOMETRY_TOLERANCE_M,
        color=black,
        linestyle=":",
        linewidth=1.0,
        label="1,000-m rule",
    )
    axes[1].set(xlabel="Elapsed time (min)", ylabel="Intent path error (m)")
    axes[1].legend(frameon=False, fontsize=7)

    adaptive_only = np.setdiff1d(selected, mandatory)
    axes[2].eventplot(
        qar.reference.time_s[mandatory] / 60.0,
        lineoffsets=1.0,
        linelengths=0.6,
        color=orange,
        label="mandatory time anchors",
    )
    axes[2].eventplot(
        qar.reference.time_s[adaptive_only] / 60.0,
        lineoffsets=2.0,
        linelengths=0.6,
        color=blue,
        label="added geometry anchors",
    )
    axes[2].set(
        xlabel="Elapsed time (min)",
        ylabel="Anchor class",
        yticks=[1.0, 2.0],
        yticklabels=["600-s", "turn-aware"],
        ylim=(0.4, 2.6),
    )
    axes[2].legend(frameon=False, fontsize=7)
    for label, axis in zip("ABC", axes, strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 geometry-only turn-aware intent selection", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    qar, registration, recording = _load_holdout(args)
    selected, mandatory = select_turn_aware_indices(
        qar.north_position_m, qar.east_position_m
    )
    base_error = _path_error(qar, mandatory)
    adaptive_error = _path_error(qar, selected)
    adaptive_only = np.setdiff1d(selected, mandatory)
    turn_angle = _turn_angle_deg(qar.north_position_m, qar.east_position_m, selected)
    anchors = pd.DataFrame(
        {
            "anchor_order": np.arange(len(selected)),
            "source_sample_index": selected,
            "elapsed_time_s": qar.reference.time_s[selected],
            "selection_class": np.where(
                np.isin(selected, mandatory), "maximum-time-spacing", "geometry"
            ),
            "absolute_turn_angle_deg": turn_angle,
        }
    )
    metrics = {
        "base_anchor_count": len(mandatory),
        "adaptive_anchor_count": len(selected),
        "added_geometry_anchor_count": len(adaptive_only),
        "base_path_rmse_m": float(np.sqrt(np.mean(base_error**2))),
        "adaptive_path_rmse_m": float(np.sqrt(np.mean(adaptive_error**2))),
        "base_path_maximum_m": float(np.max(base_error)),
        "adaptive_path_maximum_m": float(np.max(adaptive_error)),
        "maximum_anchor_interval_s": float(
            np.max(np.diff(qar.reference.time_s[selected]))
        ),
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "selected-before-adaptive-rollout",
        "source": {
            "recording": f"{registration}/{recording}",
            "independent_holdout_response_already_opened_by_prior_600_s_test": True,
            "adaptive_rollout_response_opened": False,
            "raw_coordinates_persisted": False,
        },
        "selection_rule": {
            "maximum_time_spacing_s": MAXIMUM_INTERVAL_S,
            "geometry_chord_tolerance_m": GEOMETRY_TOLERANCE_M,
            "algorithm": "segmented deterministic Ramer-Douglas-Peucker",
            "uses_only_zero-referenced_horizontal_path_geometry": True,
            "uses_altitude_speed_attitude_control_or_model_errors": False,
        },
        "anchor_indices": selected.tolist(),
        "metrics": metrics,
        "gates": {
            "maximum_spacing_preserved": metrics["maximum_anchor_interval_s"]
            <= MAXIMUM_INTERVAL_S,
            "geometry_tolerance_met": metrics["adaptive_path_maximum_m"]
            <= GEOMETRY_TOLERANCE_M + 1e-6,
            "adaptive_path_rmse_below_frozen_fms_gate": metrics["adaptive_path_rmse_m"]
            <= 3_000.0,
        },
        "artifacts": {
            "anchors": str(args.anchors),
            "figure": f"{args.figure_prefix}.png",
            "report": str(args.report),
            "code": "tools/select_a320_fms_turn_aware_intent.py",
        },
    }
    if not all(document["gates"].values()):
        raise ValueError("turn-aware intent selection gate failed")
    report = f"""# A320 FMS geometry-only turn-aware intent selection

Updated: {UPDATED}

Status: **selected before adaptive rollout**

The frozen rule retains an anchor at least every {MAXIMUM_INTERVAL_S} seconds,
then applies deterministic segmented Ramer–Douglas–Peucker simplification with
a {GEOMETRY_TOLERANCE_M:.0f}-m chord-error tolerance. It uses only the
zero-referenced horizontal path. Altitude, speed, attitude, engine/control
channels, aircraft-model output, and rollout errors cannot affect selection.

The rule expands {metrics["base_anchor_count"]} uniform anchors to
{metrics["adaptive_anchor_count"]} anchors by adding
{metrics["added_geometry_anchor_count"]} around turns. Intent-path RMSE falls
from {metrics["base_path_rmse_m"]:.1f} m to
{metrics["adaptive_path_rmse_m"]:.1f} m; maximum chord error falls from
{metrics["base_path_maximum_m"]:.1f} m to
{metrics["adaptive_path_maximum_m"]:.1f} m.

This is a post-holdout improvement diagnostic because the flight response was
already opened by the prior 600-second test. The adaptive rollout remains
unseen at this selection stage and no aircraft model or controller parameter is
changed.
"""
    args.anchors.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(args.anchors, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(qar, selected, mandatory, base_error, adaptive_error, args.figure_prefix)
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
