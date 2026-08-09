#!/usr/bin/env python3
"""Audit closed-loop integration of the frozen A320 speed-intent surrogate."""

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


UPDATED = "2026-08-07"
FT_TO_M = 0.3048
KNOT_TO_MPS = 1852.0 / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "baseline": "docs/fms/a320-fms-vertical-intent.yml",
        "first-pass": "docs/fms/a320-fms-mode-conditioned-intent.yml",
        "mass-iterated": "docs/fms/a320-fms-mode-conditioned-mass-intent.yml",
        "continuous": "docs/fms/a320-fms-continuous-speed-mass.yml",
        "intent-audit": "docs/fms/a320-fms-qar-intent-channel-audit.yml",
        "metrics": "docs/fms/a320-fms-mode-conditioned-integration-metrics.csv",
        "parameters": "docs/fms/a320-fms-mode-conditioned-integration.yml",
        "report": "docs/fms/a320-fms-mode-conditioned-integration.md",
        "figure-prefix": "docs/fms/figures/a320-fms-mode-conditioned-integration",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _trajectory_path(parameters: Path, document: dict[str, Any]) -> Path:
    path = Path(document["artifacts"]["trajectory_comparison"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _profile_path(parameters: Path, document: dict[str, Any]) -> Path:
    path = Path(document["artifacts"]["profile"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _distance(north_m: np.ndarray, east_m: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(np.diff(north_m), np.diff(east_m)))]


def _spatial_metrics(trajectory: pd.DataFrame) -> dict[str, float]:
    qar_s = _distance(
        trajectory["qar_north_m"].to_numpy(dtype=float),
        trajectory["qar_east_m"].to_numpy(dtype=float),
    )
    generated_s = _distance(
        trajectory["generated_north_m"].to_numpy(dtype=float),
        trajectory["generated_east_m"].to_numpy(dtype=float),
    )
    generated_altitude = np.interp(
        qar_s,
        generated_s,
        trajectory["generated_pressure_altitude_m"].to_numpy(dtype=float),
    )
    generated_cas = np.interp(
        qar_s,
        generated_s,
        trajectory["generated_calibrated_airspeed_mps"].to_numpy(dtype=float),
    )
    altitude_error = generated_altitude - trajectory[
        "qar_pressure_altitude_m"
    ].to_numpy(dtype=float)
    cas_error = generated_cas - trajectory["qar_calibrated_airspeed_mps"].to_numpy(
        dtype=float
    )
    return {
        "spatial_altitude_rmse_m": float(np.sqrt(np.mean(altitude_error**2))),
        "spatial_cas_rmse_mps": float(np.sqrt(np.mean(cas_error**2))),
        "qar_distance_km": float(qar_s[-1] / 1_000.0),
        "generated_distance_km": float(generated_s[-1] / 1_000.0),
    }


def _case_row(
    label: str, document: dict[str, Any], trajectory: pd.DataFrame
) -> dict[str, Any]:
    metrics = document["metrics"]
    return {
        "case": label,
        "status": document["status"],
        "altitude_rmse_m": float(metrics["altitude_rmse_m"]),
        "spatial_altitude_rmse_m": _spatial_metrics(trajectory)[
            "spatial_altitude_rmse_m"
        ],
        "tas_rmse_mps": float(metrics["speed_rmse_mps"]),
        "cas_rmse_mps": float(metrics["calibrated_airspeed_rmse_mps"]),
        "mach_rmse": float(metrics["mach_rmse"]),
        "arrival_time_error_s": float(metrics["arrival_time_error_s"]),
        "time_aligned_horizontal_rmse_m": float(
            metrics["time_aligned_horizontal_rmse_m"]
        ),
        "generated_to_qar_path_rmse_m": float(metrics["generated_to_qar_path_rmse_m"]),
        "control_limited_fraction": float(metrics["any_control_limited_fraction"]),
        "fuel_burn_kg": float(metrics["fuel_burn_kg"]),
        "profile_feasible": bool(document["gates"]["profile_feasible"]["passed"]),
    }


def _plot(
    *,
    documents: dict[str, dict[str, Any]],
    trajectories: dict[str, pd.DataFrame],
    final_profile: pd.DataFrame,
    metrics: pd.DataFrame,
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
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "flown-median baseline": "#0072B2",
        "initial-mass surrogate": "#CC79A7",
        "mass-iterated surrogate": "#D55E00",
        "continuous speed + mass": "#009E73",
    }
    linestyles = {
        "flown-median baseline": "--",
        "initial-mass surrogate": "-.",
        "mass-iterated surrogate": "-",
        "continuous speed + mass": "-",
    }
    black, green, gray = "#000000", "#56B4E9", "#777777"
    figure, axes = plt.subplots(
        2, 3, figsize=(183.0 / 25.4, 6.6), constrained_layout=True
    )
    reference = trajectories["flown-median baseline"]
    reference_time = reference["elapsed_time_s"].to_numpy(dtype=float) / 60.0
    axes[0, 0].plot(
        reference_time,
        reference["qar_pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=black,
        label="QAR",
    )
    axes[0, 1].plot(
        reference_time,
        reference["qar_calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=black,
        label="QAR",
    )
    axes[0, 2].plot(reference_time, reference["qar_mach"], color=black, label="QAR")
    for label, trajectory in trajectories.items():
        time = trajectory["elapsed_time_s"].to_numpy(dtype=float) / 60.0
        axes[0, 0].plot(
            time,
            trajectory["generated_pressure_altitude_m"] / FT_TO_M / 1_000.0,
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
        axes[0, 1].plot(
            time,
            trajectory["generated_calibrated_airspeed_mps"] / KNOT_TO_MPS,
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
        axes[0, 2].plot(
            time,
            trajectory["generated_mach"],
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
    plan_time = final_profile["predicted_elapsed_time_s"].to_numpy(dtype=float) / 60.0
    axes[0, 0].plot(
        plan_time,
        final_profile["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=green,
        linestyle=":",
        label="continuous intent",
    )
    axes[0, 1].plot(
        plan_time,
        final_profile["calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=green,
        linestyle=":",
        label="continuous intent",
    )
    axes[0, 2].plot(
        plan_time,
        final_profile["mach"],
        color=green,
        linestyle=":",
        label="continuous intent",
    )
    axes[0, 0].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (kft)")
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="CAS (kt)")
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Mach number")
    axes[0, 0].set_title("Generated altitude")
    axes[0, 1].set_title("Generated calibrated airspeed")
    axes[0, 2].set_title("Generated Mach")
    axes[0, 2].legend(ncols=1, loc="lower left", fontsize=5.8)

    final = trajectories["continuous speed + mass"]
    qar_s = _distance(final["qar_north_m"].to_numpy(), final["qar_east_m"].to_numpy())
    generated_s = _distance(
        final["generated_north_m"].to_numpy(), final["generated_east_m"].to_numpy()
    )
    axes[1, 0].plot(
        qar_s / 1_000.0,
        final["qar_pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=black,
        label="QAR",
    )
    axes[1, 0].plot(
        generated_s / 1_000.0,
        final["generated_pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=colors["continuous speed + mass"],
        label="continuous generated",
    )
    axes[1, 0].plot(
        final_profile["along_path_m"] / 1_000.0,
        final_profile["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=green,
        linestyle=":",
        label="intent",
    )
    axes[1, 0].set(xlabel="Along-path distance (km)", ylabel="Pressure altitude (kft)")
    axes[1, 0].set_title("Altitude by route distance")
    axes[1, 0].legend()

    for label, trajectory in trajectories.items():
        qar_distance = _distance(
            trajectory["qar_north_m"].to_numpy(), trajectory["qar_east_m"].to_numpy()
        )
        generated_distance = _distance(
            trajectory["generated_north_m"].to_numpy(),
            trajectory["generated_east_m"].to_numpy(),
        )
        axes[1, 1].plot(
            trajectory["elapsed_time_s"] / 60.0,
            (generated_distance - qar_distance) / 1_000.0,
            color=colors[label],
            linestyle=linestyles[label],
            label=label,
        )
    axes[1, 1].axhline(0.0, color=black, linewidth=0.7)
    axes[1, 1].set(xlabel="Elapsed time (min)", ylabel="Along-path timing error (km)")
    axes[1, 1].set_title("Along-path timing error")
    axes[1, 1].legend()

    gates = {
        "horizontal": 6_000.0,
        "altitude": 250.0,
        "speed": 8.0,
        "arrival": 120.0,
        "limiter": 0.10,
    }
    x = np.arange(len(gates))
    for label, color in colors.items():
        row = metrics[metrics["case"] == label].iloc[0]
        normalized = np.asarray(
            [
                row["time_aligned_horizontal_rmse_m"] / gates["horizontal"],
                row["altitude_rmse_m"] / gates["altitude"],
                row["tas_rmse_mps"] / gates["speed"],
                abs(row["arrival_time_error_s"]) / gates["arrival"],
                row["control_limited_fraction"] / gates["limiter"],
            ]
        )
        axes[1, 2].plot(
            x,
            normalized,
            color=color,
            linestyle=linestyles[label],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[1, 2].axhline(1.0, color=gray, linestyle=":", label="gate")
    axes[1, 2].set_xticks(x, list(gates), rotation=20, ha="right")
    axes[1, 2].set_ylabel("Metric / gate limit")
    axes[1, 2].set_title("Unchanged closed-loop gates")
    axes[1, 2].legend()

    for panel, axis in zip("ABCDEF", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.6)
    figure.suptitle("A320 closed-loop integration of phase-conditioned speed intent")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "flown-median baseline": args.baseline,
        "initial-mass surrogate": args.first_pass,
        "mass-iterated surrogate": args.mass_iterated,
        "continuous speed + mass": args.continuous,
    }
    documents = {
        label: yaml.safe_load(path.read_text()) for label, path in paths.items()
    }
    trajectories = {
        label: pd.read_csv(_trajectory_path(paths[label], document))
        for label, document in documents.items()
    }
    final_profile = pd.read_csv(
        _profile_path(args.continuous, documents["continuous speed + mass"])
    )
    metrics = pd.DataFrame(
        [_case_row(label, documents[label], trajectories[label]) for label in paths]
    )
    final = metrics[metrics["case"] == "continuous speed + mass"].iloc[0]
    phase_point = metrics[metrics["case"] == "mass-iterated surrogate"].iloc[0]
    first = metrics[metrics["case"] == "initial-mass surrogate"].iloc[0]
    baseline = metrics[metrics["case"] == "flown-median baseline"].iloc[0]
    intent_audit = yaml.safe_load(args.intent_audit.read_text())
    observed_tod_km = (
        float(intent_audit["metrics"]["observed_top_of_descent_along_path_m"]) / 1_000.0
    )
    planned_tod_km = float(
        documents["continuous speed + mass"]["energy_plan"][
            "calculated_top_of_descent_along_path_km"
        ]
    )
    passed_gates = [
        name
        for name, gate in documents["continuous speed + mass"]["gates"].items()
        if gate["passed"]
    ]
    held_gates = [
        name
        for name, gate in documents["continuous speed + mass"]["gates"].items()
        if not gate["passed"]
    ]
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "held",
        "scope": "post-holdout closed-loop integration diagnostic",
        "speed_intent": {
            "structure": "frozen affine selector surrogate",
            "actual_CAS_or_Mach_used_to_form_targets": False,
            "first_pass_mass": "constant initial mass",
            "second_pass_mass": "first-pass Aerox predicted fuel/mass history",
            "continuous_mass": "fixed-point integration of calibrated Aerox fuel flow",
            "QAR_mass_anchors_used": False,
            "controller_or_plant_refit": False,
            "inactive_climb_selector_guard": "frozen phase-intent CAS floor",
        },
        "results": {
            "first_to_phase_point_arrival_error_change_s": float(
                phase_point["arrival_time_error_s"] - first["arrival_time_error_s"]
            ),
            "phase_point_to_continuous_arrival_error_change_s": float(
                final["arrival_time_error_s"] - phase_point["arrival_time_error_s"]
            ),
            "phase_point_to_continuous_altitude_rmse_improvement_fraction": float(
                1.0 - final["altitude_rmse_m"] / phase_point["altitude_rmse_m"]
            ),
            "final_time_aligned_altitude_rmse_m": float(final["altitude_rmse_m"]),
            "final_spatial_altitude_rmse_m": float(final["spatial_altitude_rmse_m"]),
            "final_cas_rmse_mps": float(final["cas_rmse_mps"]),
            "final_arrival_time_error_s": float(final["arrival_time_error_s"]),
            "final_control_limited_fraction": float(final["control_limited_fraction"]),
            "planned_minus_observed_tod_km": planned_tod_km - observed_tod_km,
            "baseline_uses_post_response_flown_speed_medians": True,
            "baseline_altitude_rmse_m": float(baseline["altitude_rmse_m"]),
            "final_release_gate_passed": False,
            "passed_gates": passed_gates,
            "held_gates": held_gates,
        },
        "artifacts": {
            "metrics": str(args.metrics),
            "figure": str(args.figure_prefix.with_suffix(".png")),
            "first_pass": str(args.first_pass),
            "mass_iterated": str(args.mass_iterated),
            "continuous": str(args.continuous),
            "reproducer": "tools/audit_a320_fms_mode_conditioned_integration.py",
        },
        "decision": {
            "speed_intent_surrogate_rejected": False,
            "direct_constant_mass_integration_rejected": True,
            "mass_iterated_schedule_retained_as_diagnostic": True,
            "continuous_speed_mass_plan_retained": True,
            "closed_loop_release_authorized": False,
            "next_gate": (
                "infer active guidance regime and add procedure constraints, "
                "then validate on a new flight"
            ),
        },
        "limitations": [
            "The active selected-versus-managed speed regime remains unobserved.",
            "The climb CAS guard is a frozen operational prior, not an identified mode classifier.",
            "QAR temperature and wind remain forecast surrogates.",
            "The comparison is a post-holdout diagnostic, not new independent validation.",
        ],
    }
    report = f"""# A320 FMS mode-conditioned speed-intent integration

Updated: {UPDATED}

Status: **held for closed-loop release**.

The frozen affine speed-intent surrogate was inserted into the existing A320
TOC/TOD profile, unchanged guidance controller, and unchanged attitude/net-
thrust plant. Actual QAR CAS, Mach, and inferred speed mode do not form the
generated targets.

## Progression

The direct first pass evaluated every phase at initial mass and produced a
Mach-0.809 cruise target. It arrived {first["arrival_time_error_s"]:+.0f} s from
QAR and had {first["altitude_rmse_m"]:.0f} m time-aligned altitude RMSE.
Re-evaluating the frozen surrogate from the first pass's own predicted fuel/mass
history moved cruise intent to Mach 0.785, arrival error to
{phase_point["arrival_time_error_s"]:+.0f} s, and altitude RMSE to
{phase_point["altitude_rmse_m"]:.0f} m. No QAR mass anchors were introduced.

The continuous planner then evaluated intent every 500 m, projected it into the
joint altitude/speed energy envelope, and iterated calibrated predicted fuel
flow to a self-consistent mass profile. It arrived
{final["arrival_time_error_s"]:+.0f} s from QAR with
{final["altitude_rmse_m"]:.0f} m altitude RMSE,
{final["cas_rmse_mps"]:.2f} m/s CAS RMSE, and
{final["control_limited_fraction"]:.1%} limiter occupancy. Relative to the
six-point mass-iterated case, limiter occupancy changed from
{phase_point["control_limited_fraction"]:.1%} to
{final["control_limited_fraction"]:.1%}. The controller and plant were not
refit.

## What still fails

The final spatial altitude RMSE is {final["spatial_altitude_rmse_m"]:.0f} m,
inside the 250 m altitude gate, whereas time-aligned altitude RMSE is
{final["altitude_rmse_m"]:.0f} m. The vertical path is therefore substantially
better than the time history suggests. Residual speed/energy distribution
creates along-route timing error that later recovers to the passing
{final["arrival_time_error_s"]:+.0f} s terminal error.

The final CAS RMSE is {final["cas_rmse_mps"]:.2f} m/s and controller-limiter
occupancy is {final["control_limited_fraction"]:.1%}. The continuous profile is
feasible under the released climb and descent capacity. The calculated TOD differs from
the observed diagnostic by {planned_tod_km - observed_tod_km:+.1f} km; TOD is
not the dominant remaining error.

The flown-median baseline is shown only as a post-response upper benchmark. It
uses stable flown CAS/Mach medians and is not a valid future-intent generator.
Its lower {baseline["altitude_rmse_m"]:.0f} m altitude RMSE must not be obtained
by feeding response variables back into the FMS.

## Decision

The continuous target and mass/energy iteration are retained, but closed-loop
release is held. The next implementation should identify active guidance
regimes from intent-side signals and add explicit procedure speed/altitude
constraints before validation on another flight. It should not tune constants
against this QAR trajectory.

## Artifacts

- Case metrics: `docs/fms/a320-fms-mode-conditioned-integration-metrics.csv`
- Machine-readable audit: `docs/fms/a320-fms-mode-conditioned-integration.yml`
- Comparison: `docs/fms/figures/a320-fms-mode-conditioned-integration.{{png,pdf}}`
- Final generated trajectory: `docs/fms/a320-fms-mode-conditioned-mass-intent-errors-trajectory.csv`
- Final controls and mass: `docs/fms/a320-fms-mode-conditioned-mass-intent-history.csv`
- Continuous plan and predicted mass: `docs/fms/a320-fms-continuous-speed-mass-profile.csv`
- Continuous generated trajectory: `docs/fms/a320-fms-continuous-speed-mass-errors-trajectory.csv`
- Reproducer: `tools/audit_a320_fms_mode_conditioned_integration.py`
"""
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(
        documents=documents,
        trajectories=trajectories,
        final_profile=final_profile,
        metrics=metrics,
        prefix=args.figure_prefix,
    )
    print(yaml.safe_dump(document["results"], sort_keys=False).strip())
    return document


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
