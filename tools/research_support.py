"""Shared, side-effect-free helpers for Aerox research and audit tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RESET_WINDOW_S = 300


def select_lift_candidate(candidates: pd.DataFrame) -> pd.Series:
    """Choose the simplest near-best locally monotone lift candidate."""

    eligible = candidates.loc[
        candidates["observed_local_monotonic_fraction"] >= 0.995
    ]
    if eligible.empty:
        raise ValueError("no lift candidate is locally monotone")
    limit = 1.01 * eligible["lofo_mean_flight_cl_rmse"].min()
    return eligible.loc[eligible["lofo_mean_flight_cl_rmse"] <= limit].sort_values(
        ["feature_count", "lofo_mean_flight_cl_rmse"], kind="stable"
    ).iloc[0]


def select_fuel_candidate(candidates: pd.DataFrame) -> pd.Series:
    """Choose the least complex fuel correction within one percent of best."""

    limit = 1.01 * candidates["lofo_mean_flight_rmse_kg_s"].min()
    return candidates.loc[
        candidates["lofo_mean_flight_rmse_kg_s"] <= limit
    ].sort_values(
        ["parameter_count", "lofo_mean_flight_rmse_kg_s"], kind="stable"
    ).iloc[0]


def integrate_contiguous_samples(
    values: np.ndarray,
    time_s: np.ndarray,
    sample_index: np.ndarray,
) -> float:
    """Integrate adjacent one-hertz samples without bridging filtered gaps."""

    dt = np.diff(time_s)
    adjacent = (
        (np.diff(sample_index) == 1)
        & np.isfinite(dt)
        & (dt > 0.0)
        & (dt <= 1.5)
    )
    trapezoids = 0.5 * (values[:-1] + values[1:]) * dt
    return float(np.sum(trapezoids[adjacent]))


def longest_contiguous_slice(sample_index: np.ndarray) -> slice:
    """Return the longest interval whose source indices stay adjacent."""

    if len(sample_index) == 0:
        raise ValueError("cannot select from an empty flight")
    boundaries = np.flatnonzero(np.diff(sample_index) != 1) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(sample_index)]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return slice(int(starts[best]), int(stops[best]))


def round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the stable numeric formatting used by generated artifacts."""

    rounded = frame.copy()
    integer_columns = {
        "feature_count",
        "segment_start_sample",
        "full_scenario_success_count",
        "window_count",
        "window_failures",
    }
    for column in rounded.select_dtypes(include=["number"]):
        if column in integer_columns:
            rounded[column] = rounded[column].fillna(0).astype(int)
        else:
            rounded[column] = rounded[column].round(8)
    return rounded


def native_value(value: Any) -> Any:
    """Convert NumPy-rich structures to YAML/JSON-native values."""

    if isinstance(value, dict):
        return {key: native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact, deterministic Markdown table without extra packages."""

    strings = frame.astype(str)
    headers = list(strings.columns)
    rows = strings.values.tolist()
    widths = [
        max(len(str(headers[index])), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row))
        + " |"
        for row in rows
    )
    return "\n".join(lines)
