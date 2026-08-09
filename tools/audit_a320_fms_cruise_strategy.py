#!/usr/bin/env python3
"""Compare phase-conditioned and explicit-Mach A320 FMS trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_a320_fms_mode_conditioned_integration import _spatial_metrics

UPDATED = "2026-08-08"
FT_TO_M = 0.3048
KNOT_TO_MPS = 1852.0 / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-conditioned",
        type=Path,
        default=Path("docs/fms/a320-fms-continuous-speed-mass.yml"),
    )
    parser.add_argument(
        "--commanded-mach",
        type=Path,
        default=Path("docs/fms/a320-fms-commanded-mach.yml"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/fms/a320-fms-cruise-strategy-comparison.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-cruise-strategy-comparison.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-cruise-strategy-comparison.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-cruise-strategy-comparison"),
    )
    return parser.parse_args()


def _artifact(parameters: Path, document: dict[str, Any], key: str) -> Path:
    path = Path(document["artifacts"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _row(
    label: str, document: dict[str, Any], trajectory: pd.DataFrame
) -> dict[str, Any]:
    metrics = document["metrics"]
    spatial = _spatial_metrics(trajectory)
    return {
        "strategy": label,
        "status": document["status"],
        "commanded_cruise_mach": document["intent_boundary"]
        .get("cruise_speed_strategy", {})
        .get("commanded_mach"),
        "generated_duration_s": metrics["generated_duration_s"],
        "arrival_time_error_s": metrics["arrival_time_error_s"],
        "time_aligned_horizontal_rmse_m": metrics[
            "time_aligned_horizontal_rmse_m"
        ],
        "time_aligned_altitude_rmse_m": metrics["altitude_rmse_m"],
        "spatial_altitude_rmse_m": spatial["spatial_altitude_rmse_m"],
        "tas_rmse_mps": metrics["speed_rmse_mps"],
        "cas_rmse_mps": metrics["calibrated_airspeed_rmse_mps"],
        "mach_rmse": metrics["mach_rmse"],
        "control_limited_fraction": metrics["any_control_limited_fraction"],
        "fuel_burn_kg": metrics["fuel_burn_kg"],
    }


def _plot(
    *,
    trajectories: dict[str, pd.DataFrame],
    profiles: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    gate_limits: dict[str, Any],
    prefix: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    labels = {
        "phase_conditioned": "phase-conditioned",
        "commanded_mach": "explicit M0.78",
    }
    colors = {"phase_conditioned": "#0072B2", "commanded_mach": "#D55E00"}
    styles = {"phase_conditioned": "--", "commanded_mach": "-"}
    reference = trajectories["commanded_mach"]
    time_min = reference["elapsed_time_s"].to_numpy(dtype=float) / 60.0
    figure, axes = plt.subplots(
        2, 3, figsize=(183.0 / 25.4, 125.0 / 25.4), constrained_layout=True
    )

    axes[0, 0].plot(
        time_min,
        reference["qar_pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color="black",
        label="QAR",
    )
    axes[0, 1].plot(
        time_min,
        reference["qar_calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color="black",
        label="QAR",
    )
    axes[0, 2].plot(time_min, reference["qar_mach"], color="black", label="QAR")
    for name, trajectory in trajectories.items():
        local_time = trajectory["elapsed_time_s"].to_numpy(dtype=float) / 60.0
        axes[0, 0].plot(
            local_time,
            trajectory["generated_pressure_altitude_m"] / FT_TO_M / 1_000.0,
            color=colors[name],
            linestyle=styles[name],
            label=labels[name],
        )
        axes[0, 1].plot(
            local_time,
            trajectory["generated_calibrated_airspeed_mps"] / KNOT_TO_MPS,
            color=colors[name],
            linestyle=styles[name],
            label=labels[name],
        )
        axes[0, 2].plot(
            local_time,
            trajectory["generated_mach"],
            color=colors[name],
            linestyle=styles[name],
            label=labels[name],
        )
    axes[0, 0].set(
        xlabel="Elapsed time (min)",
        ylabel="Pressure altitude (kft)",
        title="Generated altitude",
    )
    axes[0, 1].set(
        xlabel="Elapsed time (min)", ylabel="CAS (kt)", title="Generated airspeed"
    )
    axes[0, 2].set(
        xlabel="Elapsed time (min)", ylabel="Mach number", title="Cruise target"
    )
    for axis in axes[0]:
        axis.legend(ncols=1)

    for name, trajectory in trajectories.items():
        time = trajectory["elapsed_time_s"].to_numpy(dtype=float) / 60.0
        horizontal_error = np.hypot(
            trajectory["generated_north_m"] - trajectory["qar_north_m"],
            trajectory["generated_east_m"] - trajectory["qar_east_m"],
        )
        altitude_error = (
            trajectory["generated_pressure_altitude_m"]
            - trajectory["qar_pressure_altitude_m"]
        ) / FT_TO_M
        axes[1, 0].plot(
            time,
            horizontal_error / 1_000.0,
            color=colors[name],
            linestyle=styles[name],
            label=labels[name],
        )
        axes[1, 1].plot(
            time,
            altitude_error,
            color=colors[name],
            linestyle=styles[name],
            label=labels[name],
        )
        profile = profiles[name]
        level = profile["phase"] == "cruise"
        axes[0, 2].plot(
            profile.loc[level, "predicted_elapsed_time_s"] / 60.0,
            profile.loc[level, "mach"],
            color=colors[name],
            linewidth=0.7,
            alpha=0.65,
        )
    axes[1, 0].set(
        xlabel="Elapsed time (min)",
        ylabel="Horizontal position error (km)",
        title="Time-aligned horizontal error",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set(
        xlabel="Elapsed time (min)",
        ylabel="Altitude error (ft)",
        title="Time-aligned altitude error",
    )
    axes[1, 0].legend()
    axes[1, 1].legend()

    gate_names = ("horizontal", "altitude", "speed", "arrival", "limiter")
    numerators = {
        "horizontal": "time_aligned_horizontal_rmse_m",
        "altitude": "time_aligned_altitude_rmse_m",
        "speed": "tas_rmse_mps",
        "arrival": "arrival_time_error_s",
        "limiter": "control_limited_fraction",
    }
    limits = {
        "horizontal": gate_limits["time_aligned_horizontal_rmse_m"],
        "altitude": gate_limits["altitude_rmse_m"],
        "speed": gate_limits["speed_rmse_mps"],
        "arrival": gate_limits["arrival_time_error_s"],
        "limiter": gate_limits["limited_fraction"],
    }
    x = np.arange(len(gate_names))
    for name, label_text in labels.items():
        row = metrics[metrics["strategy"] == name].iloc[0]
        normalized = [abs(float(row[numerators[key]])) / limits[key] for key in gate_names]
        axes[1, 2].plot(
            x,
            normalized,
            color=colors[name],
            linestyle=styles[name],
            marker="o" if name == "commanded_mach" else "s",
            label=label_text,
        )
    axes[1, 2].axhline(1.0, color="#777777", linestyle=":", label="gate")
    axes[1, 2].set_xticks(x, gate_names, rotation=18)
    axes[1, 2].set(
        ylabel="Metric / gate limit", title="Unchanged closed-loop gates"
    )
    axes[1, 2].legend()

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.7)
    figure.suptitle("A320 explicit cruise-speed intent comparison", fontsize=11)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "phase_conditioned": args.phase_conditioned,
        "commanded_mach": args.commanded_mach,
    }
    documents = {name: yaml.safe_load(path.read_text()) for name, path in paths.items()}
    trajectories = {
        name: pd.read_csv(_artifact(paths[name], document, "trajectory_comparison"))
        for name, document in documents.items()
    }
    profiles = {
        name: pd.read_csv(_artifact(paths[name], document, "profile"))
        for name, document in documents.items()
    }
    metrics = pd.DataFrame(
        [_row(name, documents[name], trajectories[name]) for name in paths]
    )
    base = metrics.set_index("strategy").loc["phase_conditioned"]
    command = metrics.set_index("strategy").loc["commanded_mach"]

    def improvement(metric: str, *, absolute: bool = False) -> float:
        old = abs(float(base[metric])) if absolute else float(base[metric])
        new = abs(float(command[metric])) if absolute else float(command[metric])
        return 1.0 - new / old

    held_gates = [
        name for name, gate in documents["commanded_mach"]["gates"].items() if not gate["passed"]
    ]
    result = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "recording": "FLT0003",
        "status": "improved-but-held",
        "experiment": {
            "changed_input": "level-flight cruise strategy only",
            "baseline": "phase-conditioned affine selector surrogate",
            "candidate": "explicit commanded Mach 0.78",
            "unchanged": [
                "lateral and vertical intent",
                "climb and descent speed schedules",
                "TOC and TOD algorithms",
                "wind and temperature forecast surrogate",
                "energy projection and fuel/mass iteration",
                "controller and A320 plant",
                "closed-loop gates",
            ],
            "QAR_response_used_to_form_commanded_target": False,
        },
        "results": {
            "time_aligned_horizontal_rmse_improvement_fraction": improvement(
                "time_aligned_horizontal_rmse_m"
            ),
            "time_aligned_altitude_rmse_improvement_fraction": improvement(
                "time_aligned_altitude_rmse_m"
            ),
            "tas_rmse_improvement_fraction": improvement("tas_rmse_mps"),
            "cas_rmse_improvement_fraction": improvement("cas_rmse_mps"),
            "mach_rmse_improvement_fraction": improvement("mach_rmse"),
            "absolute_arrival_error_improvement_fraction": improvement(
                "arrival_time_error_s", absolute=True
            ),
            "commanded_mach_held_gates": held_gates,
            "commanded_mach_release_authorized": len(held_gates) == 0,
        },
        "metrics": metrics.to_dict(orient="records"),
        "artifacts": {
            "metrics": str(args.metrics),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(result, sort_keys=False))
    _plot(
        trajectories=trajectories,
        profiles=profiles,
        metrics=metrics,
        gate_limits=documents["commanded_mach"]["gate_limits"],
        prefix=args.figure_prefix,
    )
    report = f"""# A320 explicit cruise-speed intent comparison

## Outcome

Explicit commanded Mach 0.78 is a clear improvement but remains held. Against
the otherwise unchanged phase-conditioned rollout, it reduces time-aligned
horizontal RMSE from {base['time_aligned_horizontal_rmse_m'] / 1000:.2f} km to
{command['time_aligned_horizontal_rmse_m'] / 1000:.2f} km and altitude RMSE
from {base['time_aligned_altitude_rmse_m']:.1f} m to
{command['time_aligned_altitude_rmse_m']:.1f} m. Absolute arrival error falls
from {abs(base['arrival_time_error_s']):.0f} s to
{abs(command['arrival_time_error_s']):.0f} s.

CAS RMSE improves from {base['cas_rmse_mps']:.2f} to
{command['cas_rmse_mps']:.2f} m/s and Mach RMSE from
{base['mach_rmse']:.4f} to {command['mach_rmse']:.4f}. The remaining held gates
are `{', '.join(held_gates)}`; altitude is close to its unchanged 250 m limit.

![A320 explicit cruise-speed comparison](figures/a320-fms-cruise-strategy-comparison.png)

Figure 1. QAR, phase-conditioned, and explicit-Mach trajectories with direct
time-aligned errors and gate-normalized metrics. A
[vector PDF](figures/a320-fms-cruise-strategy-comparison.pdf) is available.

## Interpretation

Only level-flight intent changed. The explicit M0.78 target is supplied by the
scenario and is not inferred from QAR response. Climb/descent schedules,
TOC/TOD, route, environment, energy and mass iteration, controller, plant, and
gates are unchanged. The result demonstrates that the earlier affine target's
mass-driven cruise decay was a major source of timing drift.

The target should therefore become a first-class Aerox FMS input. The affine
surrogate remains useful when replaying incomplete selector evidence, while the
held ECON optimizer remains an experimental alternative rather than a default.
"""
    args.report.write_text(report)
    return result


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["results"], sort_keys=False))


if __name__ == "__main__":
    main()
