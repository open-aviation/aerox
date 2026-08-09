#!/usr/bin/env python3
"""Run a paired A320 long-flight trajectory gate for the active wave model."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerox-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.fms import (
    PointMassProfilePerformance,
    SpatialPathGuidance,
    plan_energy_shaped_profile,
    simulate_fms_trajectory_zoh,
)
from aerox.models import TransonicGeometryDragModel
from tools.generate_a320_comparison_trajectories import _controller
from tools.validate_a320_fms_long_range_qar import (
    COMMAND_STEP_S,
    _load_holdout,
    build_long_range_intent,
)
from tools.validate_a320_fms_profile_feasibility import (
    _load_climb_thrust_ceiling,
)
from tools.validate_a320_fms_qar_intent import (
    INTEGRATION_RATE_HZ,
    _build_plant,
    _evaluate,
    _gates,
    _initial_trim_control,
)

UPDATED = "2026-08-08"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "thrust-parameters": "docs/modeling/a320-effective-thrust-calibration.yml",
        "fuel-parameters": "docs/modeling/a320-effective-fuel-calibration.yml",
        "pitch-bank-parameters": "docs/modeling/a320-pitch-bank-input-calibration.yml",
        "surface-parameters": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
        "boundary-parameters": "docs/modeling/a320-vspaero-hybrid-boundary-selection.yml",
        "lift-surface": "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv",
        "n1-envelope": "docs/modeling/a320-autothrust-envelope.yml",
        "climb-capacity": "docs/modeling/a320-demonstrated-climb-capacity.yml",
        "errors": "docs/fms/a320-fms-wave-drag-trajectory-errors.csv",
        "history": "docs/fms/a320-fms-wave-drag-trajectory-history.csv",
        "parameters": "docs/fms/a320-fms-wave-drag-trajectory.yml",
        "report": "docs/fms/a320-fms-wave-drag-trajectory.md",
        "figure-prefix": "docs/fms/figures/a320-fms-wave-drag-trajectory-comparison",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _history(result: Any, model: str) -> pd.DataFrame:
    trajectory = result.trajectory
    return pd.DataFrame(
        {
            "model": model,
            "elapsed_time_s": trajectory.time_s,
            "north_position_m": trajectory.north_position_m,
            "east_position_m": trajectory.east_position_m,
            "pressure_altitude_m": trajectory.pressure_altitude_m,
            "true_airspeed_mps": trajectory.true_airspeed_mps,
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_angle_deg": np.rad2deg(trajectory.bank_angle_rad),
            "effective_alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "net_thrust_kn": trajectory.net_thrust_n / 1_000.0,
            "drag_kn": trajectory.drag_n / 1_000.0,
            "fuel_flow_kg_s": trajectory.fuel_flow_kg_s,
            "mass_kg": trajectory.mass_kg,
            "pitch_limited": result.pitch_limited,
            "bank_limited": result.bank_limited,
            "thrust_limited": result.thrust_limited,
        }
    )


def _simulate_variant(
    *,
    plant: Any,
    route: Any,
    environment: Any,
    anchors: np.ndarray,
    qar: Any,
    thrust_ceiling: Any,
) -> tuple[Any, Any, dict[str, Any], pd.DataFrame, dict[str, bool]]:
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=thrust_ceiling,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )
    profile = plan_energy_shaped_profile(
        route=route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(np.median(qar.static_temperature_k[anchors])),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[anchors],
    )
    guidance = SpatialPathGuidance(
        profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
        command_step_s=COMMAND_STEP_S,
    )
    controller = _controller()
    initial_gamma, initial_control, _ = _initial_trim_control(
        plant=plant,
        controller=controller,
        trajectory=profile.trajectory,
        environment=environment,
        initial_mass_kg=qar.initial_mass_kg,
    )
    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=controller,
        guidance=guidance,
        maximum_time_s=float(qar.reference.time_s[-1] + 300.0),
        static_temperature_k=float(environment.static_temperature_k[0]),
        initial_mass_kg=qar.initial_mass_kg,
        destination_capture_radius_m=1_500.0,
        integration_rate_hz=INTEGRATION_RATE_HZ,
        environment_profile=environment,
        maximum_thrust_model=thrust_ceiling,
        initial_flight_path_angle_rad=initial_gamma,
        initial_control=initial_control,
    )
    metrics, errors, _ = _evaluate(qar, route, anchors, result)
    gates = _gates(metrics, profile.feasible)
    return result, profile, metrics, errors, gates


def _comparison(
    base: dict[str, Any], wave: dict[str, Any], qar_fuel_burn_kg: float
) -> dict[str, dict[str, float]]:
    keys = (
        "generated_to_qar_path_rmse_m",
        "time_aligned_horizontal_rmse_m",
        "altitude_rmse_m",
        "speed_rmse_mps",
        "arrival_time_error_s",
        "any_control_limited_fraction",
        "fuel_burn_kg",
    )
    comparison = {
        key: {
            "subcritical_base": float(base[key]),
            "active_wave": float(wave[key]),
            "wave_minus_base": float(wave[key] - base[key]),
            "wave_over_base": float(wave[key] / base[key]) if base[key] else 1.0,
        }
        for key in keys
    }
    comparison["fuel_burn_kg"].update(
        {
            "qar": qar_fuel_burn_kg,
            "subcritical_base_error": float(base["fuel_burn_kg"] - qar_fuel_burn_kg),
            "active_wave_error": float(wave["fuel_burn_kg"] - qar_fuel_burn_kg),
        }
    )
    return comparison


def _plot(
    *,
    qar: Any,
    route: Any,
    anchors: np.ndarray,
    base_result: Any,
    wave_result: Any,
    base_errors: pd.DataFrame,
    wave_errors: pd.DataFrame,
    prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    base = base_result.trajectory
    wave = wave_result.trajectory
    time_h = qar.reference.time_s / 3_600.0
    anchor_time_h = qar.reference.time_s[anchors] / 3_600.0
    colors = {"qar": "#000000", "intent": "#E69F00", "base": "#56B4E9", "wave": "#009E73"}
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.4), constrained_layout=True)

    axes[0, 0].plot(qar.east_position_m / 1_000.0, qar.north_position_m / 1_000.0, color=colors["qar"], linestyle=":", lw=1.3, label="withheld QAR")
    axes[0, 0].plot(
        [point.east_m / 1_000.0 for point in route.intent.waypoints],
        [point.north_m / 1_000.0 for point in route.intent.waypoints],
        color=colors["intent"], linestyle="--", marker="o", markersize=2.4, lw=0.9, label="600-s intent",
    )
    axes[0, 0].plot(base.east_position_m / 1_000.0, base.north_position_m / 1_000.0, color=colors["base"], lw=1.2, label="subcritical base")
    axes[0, 0].plot(wave.east_position_m / 1_000.0, wave.north_position_m / 1_000.0, color=colors["wave"], lw=1.2, linestyle="-.", label="active wave")
    axes[0, 0].set_xlabel("East displacement (km)")
    axes[0, 0].set_ylabel("North displacement (km)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_title("a  Horizontal trajectory", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(time_h, qar.reference.pressure_altitude_m / 0.3048, color=colors["qar"], linestyle=":", lw=1.2, label="QAR")
    axes[0, 1].plot(anchor_time_h, qar.reference.pressure_altitude_m[anchors] / 0.3048, color=colors["intent"], linestyle="--", marker="o", markersize=2.4, label="vertical intent")
    axes[0, 1].plot(base.time_s / 3_600.0, base.pressure_altitude_m / 0.3048, color=colors["base"], lw=1.1, label="subcritical base")
    axes[0, 1].plot(wave.time_s / 3_600.0, wave.pressure_altitude_m / 0.3048, color=colors["wave"], lw=1.1, linestyle="-.", label="active wave")
    axes[0, 1].set_xlabel("Elapsed time (h)")
    axes[0, 1].set_ylabel("Pressure altitude (ft)")
    axes[0, 1].set_title("b  Vertical intent and trajectory", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(time_h, qar.reference.true_airspeed_mps, color=colors["qar"], linestyle=":", lw=1.2, label="QAR")
    axes[0, 2].plot(anchor_time_h, qar.reference.true_airspeed_mps[anchors], color=colors["intent"], linestyle="--", marker="o", markersize=2.4, label="speed intent")
    axes[0, 2].plot(base.time_s / 3_600.0, base.true_airspeed_mps, color=colors["base"], lw=1.1, label="subcritical base")
    axes[0, 2].plot(wave.time_s / 3_600.0, wave.true_airspeed_mps, color=colors["wave"], lw=1.1, linestyle="-.", label="active wave")
    axes[0, 2].set_xlabel("Elapsed time (h)")
    axes[0, 2].set_ylabel("True airspeed (m/s)")
    axes[0, 2].set_title("c  Speed intent and trajectory", loc="left", fontweight="bold")
    axes[0, 2].legend(frameon=False)

    for errors, color, style, label in (
        (base_errors, colors["base"], "-", "subcritical base"),
        (wave_errors, colors["wave"], "-.", "active wave"),
    ):
        axes[1, 0].plot(errors["elapsed_time_s"] / 3_600.0, errors["altitude_error_m"], color=color, linestyle=style, lw=1.0, label=label)
        axes[1, 1].plot(errors["elapsed_time_s"] / 3_600.0, errors["true_airspeed_error_mps"], color=color, linestyle=style, lw=1.0, label=label)
    for axis, ylabel, title in (
        (axes[1, 0], "Altitude error (m)", "d  Generated minus QAR altitude"),
        (axes[1, 1], "True-airspeed error (m/s)", "e  Generated minus QAR speed"),
    ):
        axis.axhline(0.0, color="0.45", lw=0.7)
        axis.set_xlabel("Elapsed time (h)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.legend(frameon=False)

    qar_burn = qar.mass_kg[0] - qar.mass_kg
    axes[1, 2].plot(time_h, qar_burn, color=colors["qar"], linestyle=":", lw=1.2, label="QAR")
    axes[1, 2].plot(base.time_s / 3_600.0, base.mass_kg[0] - base.mass_kg, color=colors["base"], lw=1.1, label="subcritical base")
    axes[1, 2].plot(wave.time_s / 3_600.0, wave.mass_kg[0] - wave.mass_kg, color=colors["wave"], linestyle="-.", lw=1.1, label="active wave")
    axes[1, 2].set_xlabel("Elapsed time (h)")
    axes[1, 2].set_ylabel("Cumulative fuel burn (kg)")
    axes[1, 2].set_title("f  Fuel-burn trajectory", loc="left", fontweight="bold")
    axes[1, 2].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("A320 long-flight trajectory — isolated wave-drag effect", fontsize=12, fontweight="bold")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_report(path: Path, document: dict[str, Any]) -> None:
    wave = document["metrics"]["active_wave"]
    change = document["comparison"]
    decision = document["decision"]
    text = f"""# A320 active wave-drag long-flight trajectory gate

The same {document['source']['flight_duration_h']:.2f}-hour withheld QAR flight
was generated twice. Intent, wind,
temperature, lift model, thrust/fuel model, guidance, controller, integration,
and initialization were identical. The only difference was whether the
geometry-conditioned wave increment was enabled in both profile planning and
the aircraft plant.

The active-wave run has altitude RMSE `{wave['altitude_rmse_m']:.1f} m`, speed
RMSE `{wave['speed_rmse_mps']:.2f} m/s`, and fuel burn
`{wave['fuel_burn_kg']:.1f} kg`. Relative to the subcritical base, those changes
are `{change['altitude_rmse_m']['wave_minus_base']:+.1f} m`,
`{change['speed_rmse_mps']['wave_minus_base']:+.2f} m/s`, and
`{change['fuel_burn_kg']['wave_minus_base']:+.1f} kg`.

![A320 wave-drag trajectory comparison](figures/a320-fms-wave-drag-trajectory-comparison.png)

Figure 1. The top row overlays withheld QAR, sparse intent, and both generated
trajectories. The bottom row shows time-aligned altitude and speed residuals
and cumulative fuel burn. Color and line style both distinguish the two model
runs; a vector PDF is provided beside the PNG.

## How guidance generates the trajectory

Only sparse position, altitude, and speed anchors every 600 seconds are used as
intent. The spatial planner turns them into a continuous feasible route and
energy profile. Online guidance converts that profile into heading,
flight-path-angle, and speed/rate references. The inverse flight-mechanics
controller commands pitch, bank, and realized net thrust; the plant integrates
motion, wind, drag, and fuel burn. No intermediate QAR pitch, bank, thrust, or
position is replayed.

## Decision

The paired noninferiority gate **{'passes' if decision['passed'] else 'does not pass'}**.
{decision['reason']}

The subcritical reference `CD0/k` projection remains wave-free because one
parabolic coefficient pair cannot represent a Mach- and lift-dependent rise.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    qar, registration, recording = _load_holdout(args)
    route, environment, anchors = build_long_range_intent(qar)
    wave_plant = _build_plant(args)
    if not isinstance(wave_plant.drag_model, TransonicGeometryDragModel):
        raise TypeError("A320 active plant does not contain the geometry wave model")
    base_plant = replace(wave_plant, drag_model=wave_plant.drag_model.base_prior)
    thrust_ceiling = _load_climb_thrust_ceiling(args)

    base_result, base_profile, base_metrics, base_errors, base_gates = _simulate_variant(
        plant=base_plant,
        route=route,
        environment=environment,
        anchors=anchors,
        qar=qar,
        thrust_ceiling=thrust_ceiling,
    )
    wave_result, wave_profile, wave_metrics, wave_errors, wave_gates = _simulate_variant(
        plant=wave_plant,
        route=route,
        environment=environment,
        anchors=anchors,
        qar=qar,
        thrust_ceiling=thrust_ceiling,
    )
    qar_fuel_burn = float(qar.mass_kg[0] - qar.mass_kg[-1])
    comparison = _comparison(base_metrics, wave_metrics, qar_fuel_burn)
    noninferior = {
        "profile_feasible": bool(wave_profile.feasible),
        "fms_completed": bool(wave_metrics["completed"]),
        "controller_limiters": bool(
            wave_gates["controller_limiters"]["passed"]
        ),
        "mach_envelope": bool(wave_gates["mach_envelope"]["passed"]),
        "alpha_envelope": bool(wave_gates["alpha_envelope"]["passed"]),
        "altitude_rmse": wave_metrics["altitude_rmse_m"] <= 1.05 * base_metrics["altitude_rmse_m"],
        "speed_rmse": wave_metrics["speed_rmse_mps"] <= 1.05 * base_metrics["speed_rmse_mps"],
        "path_rmse": wave_metrics["generated_to_qar_path_rmse_m"] <= 1.05 * base_metrics["generated_to_qar_path_rmse_m"],
        "fuel_burn_error": abs(wave_metrics["fuel_burn_kg"] - qar_fuel_burn) <= abs(base_metrics["fuel_burn_kg"] - qar_fuel_burn),
    }
    passed = all(noninferior.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "active-wave-trajectory-gate-passed" if passed else "active-wave-trajectory-gate-held",
        "source": {
            "recording": f"{registration}/{recording}",
            "flight_duration_h": float(qar.reference.time_s[-1] / 3_600.0),
            "intent_interval_s": 600,
            "intermediate_qar_states_withheld": True,
        },
        "controlled_comparison": {
            "changed": "wave-drag increment in profile planner and plant",
            "unchanged": [
                "intent", "wind", "temperature", "lift model", "thrust model",
                "fuel model", "guidance", "controller", "initialization", "integration",
            ],
            "active_wave_model": "aerox-korn-lock-wave-v1",
            "technology_factor": 0.95,
        },
        "profile_feasible": {
            "subcritical_base": bool(base_profile.feasible),
            "active_wave": bool(wave_profile.feasible),
        },
        "metrics": {"subcritical_base": base_metrics, "active_wave": wave_metrics},
        "comparison": comparison,
        "trajectory_gates": {"subcritical_base": base_gates, "active_wave": wave_gates},
        "decision": {
            "noninferiority_checks": noninferior,
            "passed": passed,
            "reason": (
                "The wave run is feasible, completes without limiter or "
                "envelope violations, and passes the paired path, altitude, "
                "speed, and fuel-error checks. Existing absolute path and "
                "altitude gates fail for both current-model runs and are not "
                "attributed to wave drag."
                if passed
                else "At least one feasibility, envelope, or paired "
                "noninferiority check fails; inspect the component checks "
                "before treating wave drag as trajectory-compatible."
            ),
        },
        "artifacts": {
            "errors": str(args.errors),
            "history": str(args.history),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
            "report": str(args.report),
        },
    }
    errors = pd.concat(
        [base_errors.assign(model="subcritical_base"), wave_errors.assign(model="active_wave")],
        ignore_index=True,
    )
    history = pd.concat(
        [_history(base_result, "subcritical_base"), _history(wave_result, "active_wave")],
        ignore_index=True,
    )
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    history.to_csv(args.history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    _write_report(args.report, document)
    _plot(
        qar=qar,
        route=route,
        anchors=anchors,
        base_result=base_result,
        wave_result=wave_result,
        base_errors=base_errors,
        wave_errors=wave_errors,
        prefix=args.figure_prefix,
    )
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(document["comparison"], sort_keys=False).strip())


if __name__ == "__main__":
    main()
