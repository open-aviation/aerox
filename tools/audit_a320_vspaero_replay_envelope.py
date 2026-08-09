#!/usr/bin/env python3
"""Separate recorded-envelope gaps from simulated-state escape in A320 replay."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.models import TabulatedLiftSurface
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    _load_point_mass_flights,
    _read_horizontal_parameters,
)
from tools.research_support import (
    RESET_WINDOW_S,
    longest_contiguous_slice as _longest_contiguous_slice,
)
from tools.replay_a320_vspaero_lift_surface import _surface_adapter
from tools.validate_a320_vspaero_lift_surface import _read_frozen_surface_parameters

UPDATED = "2026-08-05"
MIN_RECORDED_WINDOW_COVERAGE = 0.95
MAX_SIMULATED_ESCAPE_GAP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument("--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv"))
    parser.add_argument("--vertical-parameters", type=Path, default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"))
    parser.add_argument("--horizontal-parameters", type=Path, default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"))
    parser.add_argument("--surface-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"))
    parser.add_argument("--lift-surface", type=Path, default=Path("docs/modeling/a320-vspaero-alpha-support.csv"))
    parser.add_argument("--replay-metrics", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-replay-metrics.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("docs/modeling/a320-vspaero-replay-envelope-flight-metrics.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-replay-envelope-audit.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-vspaero-replay-envelope-audit.md"))
    parser.add_argument("--figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-vspaero-replay-envelope-audit"))
    return parser.parse_args()


def _flight_row(
    flight: Any,
    split: str,
    surface: TabulatedLiftSurface,
    parameters: np.ndarray,
    decoded_pitch_bias_rad: float,
    replay_row: pd.Series,
) -> dict[str, Any]:
    adapter = _surface_adapter(
        flight.base.registration,
        surface,
        parameters,
        decoded_pitch_bias_rad,
    )
    base = flight.base
    alpha = (
        adapter.pitch_scale
        * (flight.effective_pitch_angle_rad - adapter.decoded_pitch_bias_rad)
        + adapter.combined_offset_rad
        + adapter.mach_offset_rad_per_mach * (base.mach - adapter.mach_reference)
        + adapter.registration_angle_offset_rad
        - base.flight_path_angle_rad
    )
    mach_low = base.mach < surface.minimum_mach
    mach_high = base.mach > surface.maximum_mach
    alpha_low = alpha < surface.minimum_alpha_rad
    alpha_high = alpha > surface.maximum_alpha_rad
    inside = ~(mach_low | mach_high | alpha_low | alpha_high)
    segment = _longest_contiguous_slice(base.sample_index)
    start = int(segment.start or 0)
    stop = int(segment.stop or len(base.time_s))
    supported_windows = 0
    total_windows = 0
    for window_start in range(start, stop - 60, RESET_WINDOW_S):
        window_stop = min(window_start + RESET_WINDOW_S + 1, stop)
        total_windows += 1
        supported_windows += int(np.all(inside[window_start:window_stop]))
    actual_success = int(replay_row["window_count"])
    actual_failures = int(replay_row["window_failures"])
    return {
        "flight": flight.key,
        "registration": base.registration,
        "split": split,
        "samples": len(alpha),
        "recorded_point_support_fraction": float(np.mean(inside)),
        "recorded_mach_low_fraction": float(np.mean(mach_low)),
        "recorded_mach_high_fraction": float(np.mean(mach_high)),
        "recorded_alpha_low_fraction": float(np.mean(alpha_low)),
        "recorded_alpha_high_fraction": float(np.mean(alpha_high)),
        "recorded_mach_min": float(np.min(base.mach)),
        "recorded_mach_max": float(np.max(base.mach)),
        "recorded_model_alpha_min_deg": float(np.rad2deg(np.min(alpha))),
        "recorded_model_alpha_max_deg": float(np.rad2deg(np.max(alpha))),
        "total_windows": total_windows,
        "recorded_supported_windows": supported_windows,
        "recorded_window_support_fraction": supported_windows / max(total_windows, 1),
        "simulated_successful_windows": actual_success,
        "simulated_failed_windows": actual_failures,
        "simulated_window_success_fraction": actual_success / max(actual_success + actual_failures, 1),
        "full_replay_success": bool(replay_row["full_success"]),
    }


def build_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    aggregate: list[dict[str, Any]] = []
    for split in ("calibration", "validation"):
        current = frame.loc[frame["split"] == split]
        samples = current["samples"].sum()
        total_windows = current["total_windows"].sum()
        recorded_windows = current["recorded_supported_windows"].sum()
        simulated_windows = current["simulated_successful_windows"].sum()
        possible_windows = current["simulated_successful_windows"].sum() + current["simulated_failed_windows"].sum()
        aggregate.append(
            {
                "split": split,
                "recorded_point_support_fraction": float(np.average(current["recorded_point_support_fraction"], weights=current["samples"])),
                "recorded_window_support_fraction": float(recorded_windows / max(total_windows, 1)),
                "simulated_window_success_fraction": float(simulated_windows / max(possible_windows, 1)),
                "full_replay_successes": int(current["full_replay_success"].sum()),
                "samples": int(samples),
                "windows": int(total_windows),
                "mach_low_samples": int(np.sum(current["recorded_mach_low_fraction"] * current["samples"])),
                "mach_high_samples": int(np.sum(current["recorded_mach_high_fraction"] * current["samples"])),
                "alpha_low_samples": int(np.sum(current["recorded_alpha_low_fraction"] * current["samples"])),
                "alpha_high_samples": int(np.sum(current["recorded_alpha_high_fraction"] * current["samples"])),
            }
        )
    validation = next(row for row in aggregate if row["split"] == "validation")
    causes = {name: validation[name] for name in ("mach_low_samples", "mach_high_samples", "alpha_low_samples", "alpha_high_samples")}
    dominant = max(causes, key=causes.get)
    recorded_gate = validation["recorded_window_support_fraction"] >= MIN_RECORDED_WINDOW_COVERAGE
    escape_gap = validation["recorded_window_support_fraction"] - validation["simulated_window_success_fraction"]
    escape_gate = escape_gap <= MAX_SIMULATED_ESCAPE_GAP
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "replay-envelope-expansion-required",
        "surface_support": {"mach": [0.2, 0.78], "alpha_deg": [-1.0, 8.5]},
        "aggregate": aggregate,
        "gates": {
            "validation_recorded_window_support": {"value": validation["recorded_window_support_fraction"], "minimum": MIN_RECORDED_WINDOW_COVERAGE, "passed": recorded_gate},
            "validation_simulated_escape_gap": {"value": escape_gap, "maximum": MAX_SIMULATED_ESCAPE_GAP, "passed": escape_gate},
        },
        "diagnosis": {"validation_outside_sample_counts": causes, "dominant_recorded_boundary": dominant, "recorded_coverage_gap": not recorded_gate, "simulated_state_escape_beyond_recorded_gap": not escape_gate},
        "decision": {
            "expand_mach_support_before_default_replay": causes["mach_low_samples"] + causes["mach_high_samples"] > 0,
            "expand_alpha_support_before_default_replay": causes["alpha_low_samples"] + causes["alpha_high_samples"] > 0,
            "allow_unbounded_extrapolation": False,
            "make_surface_default_now": False,
            "next_model_options": ["generate and validate additional VSPAERO Mach/alpha nodes where numerically credible", "use a smooth guarded blend to the frozen empirical map outside confirmed surface support"],
        },
        "limitations": ["Recorded support is necessary but not sufficient: simulated gamma and Mach can leave the surface even when the recorded window stays inside.", "This audit does not rerun or refit either lift model.", "A hybrid boundary model would require a separately frozen transition band and trajectory gate."],
    }


def _render_report(document: dict[str, Any]) -> str:
    validation = next(row for row in document["aggregate"] if row["split"] == "validation")
    diagnosis = document["diagnosis"]
    return f"""# A320 VSPAERO replay-envelope audit

## Outcome

The trajectory hold is caused by both recorded-envelope coverage and simulated
state escape. Only {100*validation['recorded_window_support_fraction']:.1f}% of
validation windows are fully inside the recorded-input surface, while
{100*validation['simulated_window_success_fraction']:.1f}% complete in the
open-loop simulator. The dominant recorded boundary is
`{diagnosis['dominant_recorded_boundary']}`.

| Split | Recorded point support | Recorded full-window support | Simulated window success | Full-flight successes |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row['split']} | {100*row['recorded_point_support_fraction']:.1f}% | {100*row['recorded_window_support_fraction']:.1f}% | {100*row['simulated_window_success_fraction']:.1f}% | {row['full_replay_successes']} |"
        for row in document["aggregate"]
    ) + """

## Decision

- Make the surface the replay default now: **no**.
- Extrapolate the surface without bounds: **no**.
- Expand sampled support and/or design a guarded empirical blend: **yes**.
"""


def _plot(rows: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["flight"].split("/", 1)[1] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    axes[0].bar(x, [row["recorded_window_support_fraction"] for row in rows], color="#0072B2", label="recorded support")
    axes[0].scatter(x, [row["simulated_window_success_fraction"] for row in rows], color="#D55E00", marker="s", label="simulated success", zorder=3)
    axes[0].axhline(MIN_RECORDED_WINDOW_COVERAGE, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=50, ha="right")
    axes[0].set(title="Window coverage", ylabel="Fraction")
    axes[0].legend(frameon=False)
    width = 0.2
    causes = [("recorded_mach_low_fraction", "Mach low", "#56B4E9"), ("recorded_mach_high_fraction", "Mach high", "#0072B2"), ("recorded_alpha_low_fraction", "alpha low", "#E69F00"), ("recorded_alpha_high_fraction", "alpha high", "#D55E00")]
    for index, (field, label, color) in enumerate(causes):
        axes[1].bar(x+(index-1.5)*width, [row[field] for row in rows], width, label=label, color=color)
    axes[1].set_xticks(x, labels, rotation=50, ha="right")
    axes[1].set(title="Recorded outside-surface samples", ylabel="Sample fraction")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].vlines(x, [row["recorded_mach_min"] for row in rows], [row["recorded_mach_max"] for row in rows], color="#009E73", linewidth=3)
    axes[2].axhspan(0.2, 0.78, color="#009E73", alpha=0.1, label="surface Mach support")
    axes[2].set_xticks(x, labels, rotation=50, ha="right")
    axes[2].set(title="Recorded Mach ranges", ylabel="Mach")
    axes[2].legend(frameon=False)
    for label, axis in zip("ABC", axes, strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 VSPAERO replay-envelope diagnosis")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    aircraft = load_aircraft("A320")
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    flights = _load_point_mass_flights(panel, raw_dir=args.raw_dir, processed_path=args.processed, vertical=vertical, horizontal=horizontal, tas_scale=_read_tas_scale(args.horizontal_parameters), wing_area_m2=float(aircraft.wing.area))
    parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    replay = pd.read_csv(args.replay_metrics)
    replay = replay.loc[replay["lift_model"] == "vspaero_surface"].set_index(["registration", "recording"])
    role = {f"{row.registration}/{row.recording}": ("validation" if row.role == "validation" else "calibration") for row in panel.itertuples()}
    rows = [
        _flight_row(flight, role[flight.key], surface, parameters, math.radians(vertical.effective_gamma_bias_deg), replay.loc[(flight.base.registration, flight.base.recording)])
        for flight in flights
    ]
    document = build_audit(rows)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    pd.DataFrame(rows).to_csv(args.metrics, index=False)
    args.report.write_text(_render_report(document))
    _plot(rows, args.figure_prefix)
    return document


def main() -> None:
    print(run(parse_args())["status"])


if __name__ == "__main__":
    main()
