#!/usr/bin/env python3
"""Render the decoupled A320 result with inherited, not QAR, state anchors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.validate_a320_fms_long_range_qar import _load_holdout
from tools.validate_a320_fms_qar_intent import _style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "adaptive-selection": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "errors": "docs/fms/a320-fms-decoupled-qar-intent-errors.csv",
        "history": "docs/fms/a320-fms-decoupled-qar-intent-history.csv",
        "anchor-states": "docs/fms/a320-fms-decoupled-anchor-states.csv",
        "figure-prefix": "docs/fms/figures/a320-fms-decoupled-qar-intent-comparison",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def render_decoupled_figure(
    *,
    qar: object,
    indices: np.ndarray,
    errors: pd.DataFrame,
    history: pd.DataFrame,
    anchor_states: pd.DataFrame,
    figure_prefix: Path,
) -> None:
    """Render one privacy-preserving decoupled-intent comparison figure."""

    import matplotlib.pyplot as plt

    error_time = errors["elapsed_time_s"].to_numpy()
    reference_index = np.searchsorted(qar.reference.time_s, error_time)
    if not np.all(qar.reference.time_s[reference_index] == error_time):
        raise ValueError("persisted errors do not align with the QAR reference")
    generated_north = (
        qar.north_position_m[reference_index] + errors["north_error_m"].to_numpy()
    )
    generated_east = (
        qar.east_position_m[reference_index] + errors["east_error_m"].to_numpy()
    )

    black, blue, orange, green, magenta = (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    axes[0, 0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=black,
        linestyle=":",
        linewidth=1.6,
        label="withheld QAR path",
    )
    axes[0, 0].plot(
        generated_east / 1_000.0,
        generated_north / 1_000.0,
        color=blue,
        linewidth=1.5,
        label="FMS generated",
    )
    axes[0, 0].plot(
        qar.east_position_m[indices] / 1_000.0,
        qar.north_position_m[indices] / 1_000.0,
        color=orange,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markersize=3.5,
        label="horizontal intent",
    )
    axes[0, 0].set(
        xlabel="Local east displacement (km)",
        ylabel="Local north displacement (km)",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(frameon=False, fontsize=7)

    axes[0, 1].plot(
        error_time / 60.0,
        errors["horizontal_error_m"],
        color=orange,
        linewidth=1.4,
    )
    axes[0, 1].set(
        xlabel="Elapsed time (min)", ylabel="Time-aligned horizontal error (m)"
    )

    history_time = history["elapsed_time_s"] / 60.0
    reference_time = qar.reference.time_s / 60.0
    anchor_time = qar.reference.time_s[indices] / 60.0
    axes[0, 2].plot(
        reference_time,
        qar.reference.pressure_altitude_m,
        color=black,
        linestyle=":",
        linewidth=1.6,
        label="withheld QAR",
    )
    axes[0, 2].plot(
        history_time,
        history["pressure_altitude_m"],
        color=blue,
        linewidth=1.5,
        label="FMS generated",
    )
    observed = anchor_states["constraint_class"] == "observed-state"
    geometry = ~observed
    advisory = "target_altitude_m" in anchor_states
    if advisory:
        axes[0, 2].scatter(
            anchor_time[observed],
            anchor_states.loc[observed, "target_altitude_m"],
            color=orange,
            marker="x",
            s=27,
            label="QAR advisory target",
            zorder=4,
        )
        axes[0, 2].scatter(
            anchor_time[observed],
            anchor_states.loc[observed, "planned_altitude_m"],
            facecolors="white",
            edgecolors=magenta,
            s=28,
            label="energy-projected target",
            zorder=4,
        )
    else:
        axes[0, 2].scatter(
            anchor_time[observed],
            anchor_states.loc[observed, "planned_altitude_m"],
            facecolors="white",
            edgecolors=orange,
            s=30,
            label="observed state constraint",
            zorder=4,
        )
    axes[0, 2].scatter(
        anchor_time[geometry],
        anchor_states.loc[geometry, "planned_altitude_m"],
        facecolors="white",
        edgecolors=green,
        marker="s",
        s=25,
        label="inherited planned state",
        zorder=4,
    )
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (m)")
    axes[0, 2].legend(frameon=False, fontsize=6.5)

    axes[1, 0].plot(
        reference_time,
        qar.reference.true_airspeed_mps,
        color=black,
        linestyle=":",
        linewidth=1.6,
        label="withheld QAR",
    )
    axes[1, 0].plot(
        history_time,
        history["true_airspeed_mps"],
        color=blue,
        linewidth=1.5,
        label="FMS generated",
    )
    axes[1, 0].scatter(
        anchor_time[observed],
        anchor_states.loc[observed, "planned_true_airspeed_mps"],
        facecolors="white",
        edgecolors=orange,
        s=30,
        label="QAR advisory speed target" if advisory else "observed state constraint",
        zorder=4,
    )
    axes[1, 0].scatter(
        anchor_time[geometry],
        anchor_states.loc[geometry, "planned_true_airspeed_mps"],
        facecolors="white",
        edgecolors=green,
        marker="s",
        s=25,
        label="inherited planned state",
        zorder=4,
    )
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="True airspeed (m/s)")
    axes[1, 0].legend(frameon=False, fontsize=6.5)

    limiter = history[["pitch_limited", "bank_limited", "thrust_limited"]].any(axis=1)
    rolling_limiter = limiter.rolling(60, center=True, min_periods=1).mean()
    axes[1, 1].plot(history_time, rolling_limiter, color=green, linewidth=1.3)
    axes[1, 1].axhline(0.10, color=black, linestyle=":", linewidth=1.0)
    axes[1, 1].set(
        xlabel="Elapsed time (min)",
        ylabel="60-s controller-limiter fraction",
        ylim=(-0.02, 1.02),
    )

    axes[1, 2].plot(
        history_time,
        history["pitch_angle_deg"],
        color=blue,
        linewidth=1.3,
        label="pitch",
    )
    axes[1, 2].plot(
        history_time,
        history["bank_angle_deg"],
        color=orange,
        linestyle="--",
        linewidth=1.3,
        label="bank",
    )
    axes[1, 2].set(xlabel="Elapsed time (min)", ylabel="Pitch / bank angle (deg)")
    axes[1, 2].legend(frameon=False, fontsize=7)
    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        (
            "A320 FMS with advisory state intent"
            if advisory
            else "A320 FMS with decoupled horizontal and state intent"
        ),
        fontsize=13,
    )
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(figure_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    qar, _, _ = _load_holdout(args)
    selection = yaml.safe_load(args.adaptive_selection.read_text())
    render_decoupled_figure(
        qar=qar,
        indices=np.asarray(selection["anchor_indices"], dtype=int),
        errors=pd.read_csv(args.errors),
        history=pd.read_csv(args.history),
        anchor_states=pd.read_csv(args.anchor_states),
        figure_prefix=args.figure_prefix,
    )


if __name__ == "__main__":
    main()
