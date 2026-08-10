#!/usr/bin/env python3
"""Generate and gate a bounded A320 reference-tracking trajectory comparison.

The generated pitch, bank, and total realized net-thrust history is computed
with the released guarded-hybrid clean lift model.  The identical controls are
then replayed through the frozen empirical lift model to expose aerodynamic
model sensitivity without confounding it with a second controller.
"""

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
from aerox.guidance import (
    InverseFlightMechanicsController,
    TrajectoryReference,
)
from aerox.modeling.dynamics import (
    PointMassModel,
    PointMassTrajectory,
    simulate_point_mass_zoh,
)
from aerox.models import (
    CalibratedTabulatedLiftSurface,
    GuardedHybridLiftModel,
    TabulatedLiftSurface,
)
from aerox.models.atmosphere import isa_atmosphere
from aerox.models.drag import build_drag_model
from aerox.simulation import TrackingSimulationResult, simulate_reference_tracking_zoh
from tools.calibrate_a320_effective_thrust import _read_vertical_calibration
from tools.replay_a320_point_mass import _read_fuel_model
from tools.validate_a320_vspaero_lift_surface import (
    _read_empirical_lift_map,
    _read_frozen_surface_parameters,
)

UPDATED = "2026-08-06"
COMMAND_RATE_HZ = 1
INTEGRATION_RATE_HZ = 10
INITIAL_MASS_KG = 65_000.0

# These engineering demonstration gates were declared before the rollout. They
# are not fitted Airbus control-law parameters or certification limits.
GATE_LIMITS = {
    "speed_rmse_mps": 1.0,
    "altitude_rmse_m": 30.0,
    "heading_rmse_deg": 1.0,
    "horizontal_rmse_m": 2_000.0,
    "control_limited_fraction": 0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--pitch-bank-parameters",
        type=Path,
        default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"),
    )
    parser.add_argument(
        "--surface-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"),
    )
    parser.add_argument(
        "--boundary-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.yml"),
    )
    parser.add_argument(
        "--lift-surface",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"),
    )
    parser.add_argument(
        "--thrust-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.yml"),
    )
    parser.add_argument(
        "--fuel-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-fuel-calibration.yml"),
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=Path("docs/modeling/a320-generated-trajectory-comparison.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-generated-trajectory-comparison.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-generated-trajectory-comparison.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-generated-trajectory-comparison"),
    )
    parser.add_argument(
        "--control-figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-generated-control-history"),
    )
    return parser.parse_args()


def _smooth_step(time: np.ndarray, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a quintic step and its first two analytic derivatives."""

    duration = end_s - start_s
    phase = np.clip((time - start_s) / duration, 0.0, 1.0)
    inside = (time > start_s) & (time < end_s)
    value = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    first = np.zeros_like(time)
    second = np.zeros_like(time)
    x = phase[inside]
    first[inside] = (30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4) / duration
    second[inside] = (60.0 * x - 180.0 * x**2 + 120.0 * x**3) / duration**2
    return value, first, second


def build_reference() -> tuple[TrajectoryReference, np.ndarray, np.ndarray]:
    """Build a smooth, deterministic clean-flight climb/turn/speed scenario."""

    time = np.arange(0.0, 1_501.0, 1.0)
    speed_up, speed_up_rate, _ = _smooth_step(time, 120.0, 320.0)
    speed_down, speed_down_rate, _ = _smooth_step(time, 980.0, 1_180.0)
    speed = 215.0 + 8.0 * speed_up - 12.0 * speed_down
    speed_rate = 8.0 * speed_up_rate - 12.0 * speed_down_rate

    climb, climb_rate, climb_acceleration = _smooth_step(time, 240.0, 540.0)
    descent, descent_rate, descent_acceleration = _smooth_step(time, 1_080.0, 1_380.0)
    altitude = 9_500.0 + 600.0 * climb - 400.0 * descent
    altitude_rate = 600.0 * climb_rate - 400.0 * descent_rate
    altitude_acceleration = 600.0 * climb_acceleration - 400.0 * descent_acceleration
    sine_gamma = altitude_rate / speed
    gamma = np.arcsin(sine_gamma)
    gamma_rate = (
        altitude_acceleration * speed - altitude_rate * speed_rate
    ) / (speed**2 * np.sqrt(1.0 - sine_gamma**2))

    right_turn, right_turn_rate, _ = _smooth_step(time, 390.0, 740.0)
    left_turn, left_turn_rate, _ = _smooth_step(time, 1_140.0, 1_410.0)
    heading = np.deg2rad(10.0 + 75.0 * right_turn - 35.0 * left_turn)
    heading_rate = np.deg2rad(75.0 * right_turn_rate - 35.0 * left_turn_rate)

    north_rate = speed * np.cos(gamma) * np.cos(heading)
    east_rate = speed * np.cos(gamma) * np.sin(heading)
    north = np.zeros_like(time)
    east = np.zeros_like(time)
    north[1:] = np.cumsum(0.5 * (north_rate[:-1] + north_rate[1:]) * np.diff(time))
    east[1:] = np.cumsum(0.5 * (east_rate[:-1] + east_rate[1:]) * np.diff(time))
    return (
        TrajectoryReference(
            time_s=time,
            true_airspeed_mps=speed,
            true_airspeed_rate_mps2=speed_rate,
            flight_path_angle_rad=gamma,
            flight_path_angle_rate_rad_s=gamma_rate,
            heading_rad=heading,
            heading_rate_rad_s=heading_rate,
            pressure_altitude_m=altitude,
        ),
        north,
        east,
    )


def _build_plants(args: argparse.Namespace) -> tuple[PointMassModel, PointMassModel, GuardedHybridLiftModel]:
    aircraft = load_aircraft("A320")
    vertical = _read_vertical_calibration(args.vertical_parameters)
    parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    empirical = _read_empirical_lift_map(args.pitch_bank_parameters)
    boundary = yaml.safe_load(args.boundary_parameters.read_text())
    selected = boundary["selected"]
    # Registration contrast was identified as a measurement/reconstruction
    # nuisance. A zero contrast gives the pooled type-level adapter for a new
    # generated trajectory rather than pretending to simulate a specific tail.
    adapter = CalibratedTabulatedLiftSurface(
        surface=surface,
        pitch_scale=float(parameters[0]),
        decoded_pitch_bias_rad=math.radians(vertical.effective_gamma_bias_deg),
        combined_offset_rad=float(parameters[1]),
        mach_offset_rad_per_mach=float(parameters[2]),
        registration_angle_offset_rad=0.0,
    )
    hybrid = GuardedHybridLiftModel(
        adapter,
        empirical,
        lower_alpha_core_rad=math.radians(-0.5),
        upper_alpha_core_rad=math.radians(8.0),
        lower_mach_core=0.23,
        upper_mach_core=float(selected["upper_mach_core"]),
        maximum_surface_weight=float(selected["maximum_surface_weight"]),
    )
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    common = dict(
        wing_area_m2=float(aircraft.wing.area),
        drag_model=build_drag_model(aircraft),
        fuel_flow_prior=fuel,
    )
    return (
        PointMassModel(lift_map=hybrid, **common),
        PointMassModel(lift_map=empirical, **common),
        hybrid,
    )


def _controller() -> InverseFlightMechanicsController:
    return InverseFlightMechanicsController(
        speed_error_gain_per_s=0.04,
        flight_path_angle_error_gain_per_s=0.25,
        heading_error_gain_per_s=0.06,
        altitude_to_flight_path_gain_rad_per_m=2.0e-4,
        minimum_pitch_rad=math.radians(-2.0),
        maximum_pitch_rad=math.radians(12.0),
        maximum_bank_rad=math.radians(25.0),
        minimum_thrust_n=0.0,
        maximum_thrust_n=200_000.0,
        maximum_pitch_rate_rad_s=math.radians(1.0),
        maximum_bank_rate_rad_s=math.radians(2.0),
        maximum_thrust_rate_n_s=25_000.0,
    )


def _wrapped_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(target - actual), np.cos(target - actual))


def _trajectory_metrics(
    trajectory: PointMassTrajectory,
    reference: TrajectoryReference,
    reference_north_m: np.ndarray,
    reference_east_m: np.ndarray,
) -> dict[str, float]:
    horizontal = np.hypot(
        trajectory.north_position_m - reference_north_m,
        trajectory.east_position_m - reference_east_m,
    )
    heading_error = _wrapped_error(reference.heading_rad, trajectory.heading_rad)
    return {
        "speed_rmse_mps": float(np.sqrt(np.mean((trajectory.true_airspeed_mps - reference.true_airspeed_mps) ** 2))),
        "speed_terminal_error_mps": float(trajectory.true_airspeed_mps[-1] - reference.true_airspeed_mps[-1]),
        "flight_path_angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean((trajectory.flight_path_angle_rad - reference.flight_path_angle_rad) ** 2)))),
        "altitude_rmse_m": float(np.sqrt(np.mean((trajectory.pressure_altitude_m - reference.pressure_altitude_m) ** 2))),
        "altitude_terminal_error_m": float(trajectory.pressure_altitude_m[-1] - reference.pressure_altitude_m[-1]),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(heading_error**2)))),
        "heading_terminal_error_deg": float(np.rad2deg(-heading_error[-1])),
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "horizontal_terminal_error_m": float(horizontal[-1]),
        "fuel_burn_kg": float(trajectory.mass_kg[0] - trajectory.mass_kg[-1]),
        "minimum_pitch_deg": float(np.rad2deg(np.min(trajectory.pitch_angle_rad))),
        "maximum_pitch_deg": float(np.rad2deg(np.max(trajectory.pitch_angle_rad))),
        "maximum_absolute_bank_deg": float(np.rad2deg(np.max(np.abs(trajectory.bank_angle_rad)))),
        "minimum_net_thrust_n": float(np.min(trajectory.net_thrust_n)),
        "maximum_net_thrust_n": float(np.max(trajectory.net_thrust_n)),
        "minimum_effective_alpha_deg": float(np.rad2deg(np.min(trajectory.angle_of_attack_rad))),
        "maximum_effective_alpha_deg": float(np.rad2deg(np.max(trajectory.angle_of_attack_rad))),
    }


def _build_document(
    tracking: TrackingSimulationResult,
    empirical: PointMassTrajectory,
    reference_north_m: np.ndarray,
    reference_east_m: np.ndarray,
    surface_weight: np.ndarray,
    hybrid_mach: np.ndarray,
) -> dict[str, Any]:
    reference = tracking.reference
    hybrid_metrics = _trajectory_metrics(tracking.trajectory, reference, reference_north_m, reference_east_m)
    empirical_metrics = _trajectory_metrics(empirical, reference, reference_north_m, reference_east_m)
    limited = {
        "pitch": float(np.mean(tracking.pitch_limited)),
        "bank": float(np.mean(tracking.bank_limited)),
        "thrust": float(np.mean(tracking.thrust_limited)),
    }
    gates = {
        "hybrid_speed_rmse": {"value": hybrid_metrics["speed_rmse_mps"], "maximum": GATE_LIMITS["speed_rmse_mps"]},
        "hybrid_altitude_rmse": {"value": hybrid_metrics["altitude_rmse_m"], "maximum": GATE_LIMITS["altitude_rmse_m"]},
        "hybrid_heading_rmse": {"value": hybrid_metrics["heading_rmse_deg"], "maximum": GATE_LIMITS["heading_rmse_deg"]},
        "hybrid_horizontal_rmse": {"value": hybrid_metrics["horizontal_rmse_m"], "maximum": GATE_LIMITS["horizontal_rmse_m"]},
        "pitch_limited_fraction": {"value": limited["pitch"], "maximum": GATE_LIMITS["control_limited_fraction"]},
        "bank_limited_fraction": {"value": limited["bank"], "maximum": GATE_LIMITS["control_limited_fraction"]},
        "thrust_limited_fraction": {"value": limited["thrust"], "maximum": GATE_LIMITS["control_limited_fraction"]},
        "hybrid_minimum_mach": {"value": float(np.min(hybrid_mach)), "minimum": 0.20},
        "hybrid_maximum_mach": {"value": float(np.max(hybrid_mach)), "maximum": 0.82},
        "hybrid_minimum_effective_alpha": {"value": hybrid_metrics["minimum_effective_alpha_deg"], "minimum": -1.0},
        "hybrid_maximum_effective_alpha": {"value": hybrid_metrics["maximum_effective_alpha_deg"], "maximum": 8.5},
    }
    for gate in gates.values():
        gate["passed"] = bool(
            gate["value"] <= gate.get("maximum", np.inf)
            and gate["value"] >= gate.get("minimum", -np.inf)
        )
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 pooled type-level model",
        "status": "bounded-trajectory-generation-passed" if passed else "bounded-trajectory-generation-held",
        "scenario": {
            "type": "synthetic deterministic clean-flight engineering demonstration",
            "duration_s": float(reference.time_s[-1] - reference.time_s[0]),
            "command_rate_hz": COMMAND_RATE_HZ,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "initial_mass_kg": INITIAL_MASS_KG,
            "reference_speed_range_mps": [float(np.min(reference.true_airspeed_mps)), float(np.max(reference.true_airspeed_mps))],
            "reference_altitude_range_m": [float(np.min(reference.pressure_altitude_m)), float(np.max(reference.pressure_altitude_m))],
            "reference_heading_range_deg": [float(np.rad2deg(np.min(reference.heading_rad))), float(np.rad2deg(np.max(reference.heading_rad)))],
        },
        "controller": {
            "architecture": "inverse flight mechanics feedforward plus bounded state feedback",
            "inputs": ["pitch angle", "bank angle", "total realized net thrust"],
            "identified_airbus_law": False,
            "gain_source": "declared engineering demonstration values; no QAR response fitting",
            "limits": {
                "pitch_deg": [-2.0, 12.0],
                "bank_deg": [-25.0, 25.0],
                "net_thrust_n": [0.0, 200_000.0],
                "pitch_rate_deg_s": 1.0,
                "bank_rate_deg_s": 2.0,
                "net_thrust_rate_n_s": 25_000.0,
            },
            "limited_sample_fraction": limited,
        },
        "models": {
            "command_generation": "released guarded VSPAERO/empirical clean lift model",
            "comparison": "frozen empirical lift plant under identical generated controls",
            "drag": "Aerox A320 parabolic drag polar for both trajectories",
            "fuel": "frozen calibrated OpenAP fuel-flow prior for both trajectories",
            "mean_geometry_surface_weight": float(np.mean(surface_weight)),
            "minimum_geometry_surface_weight": float(np.min(surface_weight)),
            "maximum_geometry_surface_weight": float(np.max(surface_weight)),
        },
        "trajectory_metrics": {
            "released_hybrid_closed_loop": hybrid_metrics,
            "empirical_same_controls_open_loop": empirical_metrics,
        },
        "same_control_model_divergence": {
            "terminal_horizontal_separation_m": float(np.hypot(
                tracking.trajectory.north_position_m[-1] - empirical.north_position_m[-1],
                tracking.trajectory.east_position_m[-1] - empirical.east_position_m[-1],
            )),
            "terminal_altitude_difference_m": float(tracking.trajectory.pressure_altitude_m[-1] - empirical.pressure_altitude_m[-1]),
            "terminal_speed_difference_mps": float(tracking.trajectory.true_airspeed_mps[-1] - empirical.true_airspeed_mps[-1]),
            "fuel_burn_difference_kg": float(hybrid_metrics["fuel_burn_kg"] - empirical_metrics["fuel_burn_kg"]),
        },
        "gates": gates,
        "decision": {
            "bounded_trajectory_generation_demonstrated": passed,
            "airbus_autopilot_or_fadec_identified": False,
            "controls_safe_to_reuse_outside_released_envelope": False,
            "empirical_same_controls_is_a_model_sensitivity_case_not_a_second_controller": True,
        },
        "artifacts": {
            "samples": "docs/modeling/a320-generated-trajectory-comparison.csv",
            "trajectory_figure": "docs/modeling/figures/a320-generated-trajectory-comparison.png",
            "trajectory_figure_vector": "docs/modeling/figures/a320-generated-trajectory-comparison.pdf",
            "control_figure": "docs/modeling/figures/a320-generated-control-history.png",
            "control_figure_vector": "docs/modeling/figures/a320-generated-control-history.pdf",
            "code": "tools/generate_a320_comparison_trajectories.py",
        },
        "limitations": [
            "The reference is synthetic and demonstrates generation; it is not an independent QAR validation flight.",
            "Pitch and bank are realized attitudes; actuator, rotational, sideslip, and proprietary flight-control dynamics are outside this model.",
            "Net thrust is a realized plant input; no TLA-to-thrust or FADEC control law is claimed.",
            "The empirical same-controls trajectory is intentionally open loop and is not expected to track after the lift model changes.",
        ],
    }


def _trajectory_table(
    tracking: TrackingSimulationResult,
    empirical: PointMassTrajectory,
    reference_north_m: np.ndarray,
    reference_east_m: np.ndarray,
    surface_weight: np.ndarray,
    hybrid_mach: np.ndarray,
) -> pd.DataFrame:
    reference = tracking.reference
    hybrid = tracking.trajectory
    return pd.DataFrame({
        "time_s": reference.time_s,
        "reference_tas_mps": reference.true_airspeed_mps,
        "reference_gamma_deg": np.rad2deg(reference.flight_path_angle_rad),
        "reference_heading_deg": np.rad2deg(reference.heading_rad),
        "reference_north_m": reference_north_m,
        "reference_east_m": reference_east_m,
        "reference_altitude_m": reference.pressure_altitude_m,
        "hybrid_tas_mps": hybrid.true_airspeed_mps,
        "hybrid_gamma_deg": np.rad2deg(hybrid.flight_path_angle_rad),
        "hybrid_heading_deg": np.rad2deg(hybrid.heading_rad),
        "hybrid_north_m": hybrid.north_position_m,
        "hybrid_east_m": hybrid.east_position_m,
        "hybrid_altitude_m": hybrid.pressure_altitude_m,
        "hybrid_mass_kg": hybrid.mass_kg,
        "empirical_tas_mps": empirical.true_airspeed_mps,
        "empirical_gamma_deg": np.rad2deg(empirical.flight_path_angle_rad),
        "empirical_heading_deg": np.rad2deg(empirical.heading_rad),
        "empirical_north_m": empirical.north_position_m,
        "empirical_east_m": empirical.east_position_m,
        "empirical_altitude_m": empirical.pressure_altitude_m,
        "empirical_mass_kg": empirical.mass_kg,
        "command_pitch_deg": np.rad2deg(hybrid.pitch_angle_rad),
        "command_bank_deg": np.rad2deg(hybrid.bank_angle_rad),
        "command_net_thrust_n": hybrid.net_thrust_n,
        "hybrid_effective_alpha_deg": np.rad2deg(hybrid.angle_of_attack_rad),
        "empirical_alpha_deg": np.rad2deg(empirical.angle_of_attack_rad),
        "hybrid_lift_coefficient": hybrid.lift_coefficient,
        "empirical_lift_coefficient": empirical.lift_coefficient,
        "target_lift_coefficient": tracking.target_lift_coefficient,
        "hybrid_mach": hybrid_mach,
        "hybrid_geometry_surface_weight": surface_weight,
        "pitch_limited": tracking.pitch_limited,
        "bank_limited": tracking.bank_limited,
        "thrust_limited": tracking.thrust_limited,
    })


def _style_axis(axis: Any, label: str) -> None:
    axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_trajectories(
    tracking: TrackingSimulationResult,
    empirical: PointMassTrajectory,
    reference_north_m: np.ndarray,
    reference_east_m: np.ndarray,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    reference = tracking.reference
    hybrid = tracking.trajectory
    time_min = reference.time_s / 60.0
    colors = {"reference": "#000000", "hybrid": "#0072B2", "empirical": "#D55E00"}
    styles = {"reference": ":", "hybrid": "-", "empirical": "--"}
    labels = {"reference": "reference", "hybrid": "released hybrid", "empirical": "empirical, same controls"}
    series = {
        "reference": (reference_north_m, reference_east_m, reference.pressure_altitude_m, reference.true_airspeed_mps, np.rad2deg(reference.heading_rad)),
        "hybrid": (hybrid.north_position_m, hybrid.east_position_m, hybrid.pressure_altitude_m, hybrid.true_airspeed_mps, np.rad2deg(hybrid.heading_rad)),
        "empirical": (empirical.north_position_m, empirical.east_position_m, empirical.pressure_altitude_m, empirical.true_airspeed_mps, np.rad2deg(empirical.heading_rad)),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for name, values in series.items():
        north, east, altitude, speed, heading = values
        kwargs = dict(color=colors[name], linestyle=styles[name], linewidth=1.8, label=labels[name])
        axes[0, 0].plot(east / 1_000.0, north / 1_000.0, **kwargs)
        axes[0, 1].plot(time_min, altitude, **kwargs)
        axes[0, 2].plot(time_min, speed, **kwargs)
        axes[1, 0].plot(time_min, heading, **kwargs)
    hybrid_horizontal = np.hypot(hybrid.north_position_m - reference_north_m, hybrid.east_position_m - reference_east_m)
    empirical_horizontal = np.hypot(empirical.north_position_m - reference_north_m, empirical.east_position_m - reference_east_m)
    axes[1, 1].plot(time_min, hybrid_horizontal, color=colors["hybrid"], linewidth=1.8, label=labels["hybrid"])
    axes[1, 1].plot(time_min, empirical_horizontal, color=colors["empirical"], linestyle="--", linewidth=1.8, label=labels["empirical"])
    axes[1, 2].plot(time_min, hybrid.mass_kg, color=colors["hybrid"], linewidth=1.8, label=labels["hybrid"])
    axes[1, 2].plot(time_min, empirical.mass_kg, color=colors["empirical"], linestyle="--", linewidth=1.8, label=labels["empirical"])
    axes[0, 0].set(xlabel="East position (km)", ylabel="North position (km)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].set(xlabel="Time (min)", ylabel="Pressure altitude (m)")
    axes[0, 2].set(xlabel="Time (min)", ylabel="True airspeed (m/s)")
    axes[1, 0].set(xlabel="Time (min)", ylabel="Heading (deg)")
    axes[1, 1].set(xlabel="Time (min)", ylabel="Horizontal reference error (m)")
    axes[1, 2].set(xlabel="Time (min)", ylabel="Aircraft mass (kg)")
    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("A320 generated clean-flight trajectory: lift-model comparison", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_controls(
    tracking: TrackingSimulationResult,
    empirical: PointMassTrajectory,
    surface_weight: np.ndarray,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    trajectory = tracking.trajectory
    time_min = trajectory.time_s / 60.0
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
    axes[0, 0].plot(time_min, np.rad2deg(trajectory.pitch_angle_rad), color=blue, linewidth=1.8)
    axes[0, 0].axhline(-2.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0, 0].axhline(12.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0, 0].set(xlabel="Time (min)", ylabel="Commanded pitch (deg)")
    axes[0, 1].plot(time_min, np.rad2deg(trajectory.bank_angle_rad), color=blue, linewidth=1.8)
    axes[0, 1].axhline(-25.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0, 1].axhline(25.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[0, 1].set(xlabel="Time (min)", ylabel="Commanded bank (deg)")
    axes[1, 0].plot(time_min, trajectory.net_thrust_n / 1_000.0, color=blue, linewidth=1.8)
    axes[1, 0].axhline(200.0, color="0.35", linestyle=":", linewidth=1.0)
    axes[1, 0].set(xlabel="Time (min)", ylabel="Total realized net thrust (kN)")
    axes[1, 1].plot(time_min, tracking.target_lift_coefficient, color="black", linestyle=":", linewidth=1.5, label="target CL")
    axes[1, 1].plot(time_min, trajectory.lift_coefficient, color=blue, linewidth=1.8, label="hybrid CL")
    axes[1, 1].plot(time_min, empirical.lift_coefficient, color=orange, linestyle="--", linewidth=1.8, label="empirical CL")
    axes[1, 1].set(xlabel="Time (min)", ylabel="Lift coefficient, CL")
    weight_axis = axes[1, 1].twinx()
    weight_axis.plot(time_min, surface_weight, color=green, linewidth=1.0, alpha=0.7, label="geometry weight")
    weight_axis.set_ylabel("Geometry-surface weight")
    weight_axis.set_ylim(0.0, 0.62)
    lines, labels = axes[1, 1].get_legend_handles_labels()
    other_lines, other_labels = weight_axis.get_legend_handles_labels()
    axes[1, 1].legend(lines + other_lines, labels + other_labels, frameon=False, fontsize=8, loc="best")
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 bounded pitch–bank–net-thrust command history", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _render_report(document: dict[str, Any]) -> str:
    hybrid = document["trajectory_metrics"]["released_hybrid_closed_loop"]
    empirical = document["trajectory_metrics"]["empirical_same_controls_open_loop"]
    gates = "\n".join(
        f"| {name.replace('_', ' ')} | {gate['value']:.4g} | "
        f"{('>= ' + str(gate['minimum'])) if 'minimum' in gate else ('<= ' + str(gate['maximum']))} | "
        f"{'pass' if gate['passed'] else 'fail'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 bounded generated-trajectory comparison

## Outcome

The released guarded-hybrid A320 clean model **{'passes' if document['decision']['bounded_trajectory_generation_demonstrated'] else 'does not pass'}** the predeclared bounded trajectory-generation demonstration. A 1 Hz controller generated realized pitch, bank, and total net thrust while the plant was integrated at 10 Hz. This is a model-based reference-tracking controller, not a neural network and not an identified Airbus autopilot or FADEC law.

| Trajectory metric | Released hybrid, closed loop | Empirical lift, same controls |
|---|---:|---:|
| TAS RMSE (m/s) | {hybrid['speed_rmse_mps']:.3f} | {empirical['speed_rmse_mps']:.3f} |
| Flight-path-angle RMSE (deg) | {hybrid['flight_path_angle_rmse_deg']:.3f} | {empirical['flight_path_angle_rmse_deg']:.3f} |
| Altitude RMSE (m) | {hybrid['altitude_rmse_m']:.2f} | {empirical['altitude_rmse_m']:.2f} |
| Heading RMSE (deg) | {hybrid['heading_rmse_deg']:.3f} | {empirical['heading_rmse_deg']:.3f} |
| Horizontal RMSE (m) | {hybrid['horizontal_rmse_m']:.1f} | {empirical['horizontal_rmse_m']:.1f} |
| Fuel burn (kg) | {hybrid['fuel_burn_kg']:.1f} | {empirical['fuel_burn_kg']:.1f} |

The empirical column is an intentional model-sensitivity replay: it receives the exact pitch, bank, and thrust generated for the hybrid model, with no separate feedback correction. Its error therefore shows how much the trajectory changes when only the lift map is replaced.

## Predeclared gates

| Gate | Value | Required | Result |
|---|---:|---:|---:|
{gates}

Mean geometry-surface weight was {100.0 * document['models']['mean_geometry_surface_weight']:.1f}%. The simulated hybrid envelope was Mach {document['gates']['hybrid_minimum_mach']['value']:.3f}–{document['gates']['hybrid_maximum_mach']['value']:.3f} and effective angle of attack {hybrid['minimum_effective_alpha_deg']:.2f}–{hybrid['maximum_effective_alpha_deg']:.2f} deg.

## Scenario and controller boundary

The deterministic 25-minute reference combines a smooth 600 m climb, a later 400 m descent, two speed changes, and opposing turns. The inverse-flight-mechanics controller adds bounded state feedback and directly commands realized pitch, bank, and net thrust. Its gains are explicit engineering demonstration values and were not selected against the QAR confirmation flights.

The result demonstrates that the released clean model can generate a nontrivial trajectory inside its supported envelope. It does not validate proprietary Airbus control laws, TLA-to-thrust behavior, actuator dynamics, sideslip, stall, high-lift operation, or unrestricted trajectory optimization.

## Artifacts

- Sample-by-sample comparison: `docs/modeling/a320-generated-trajectory-comparison.csv`
- Trajectory figure: `docs/modeling/figures/a320-generated-trajectory-comparison.png` and `.pdf`
- Control/force figure: `docs/modeling/figures/a320-generated-control-history.png` and `.pdf`
- Reproducer: `tools/generate_a320_comparison_trajectories.py`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference, reference_north, reference_east = build_reference()
    hybrid_plant, empirical_plant, hybrid_model = _build_plants(args)
    temperature = isa_atmosphere(reference.pressure_altitude_m).temperature_k
    tracking = simulate_reference_tracking_zoh(
        plant=hybrid_plant,
        controller=_controller(),
        reference=reference,
        static_temperature_k=temperature,
        initial_mass_kg=INITIAL_MASS_KG,
        integration_rate_hz=INTEGRATION_RATE_HZ,
    )
    hybrid = tracking.trajectory
    empirical = simulate_point_mass_zoh(
        plant=empirical_plant,
        time_s=reference.time_s,
        pitch_angle_rad=hybrid.pitch_angle_rad,
        bank_angle_rad=hybrid.bank_angle_rad,
        net_thrust_n=hybrid.net_thrust_n,
        static_temperature_k=temperature,
        initial_true_airspeed_mps=float(reference.true_airspeed_mps[0]),
        initial_flight_path_angle_rad=float(reference.flight_path_angle_rad[0]),
        initial_heading_rad=float(reference.heading_rad[0]),
        initial_pressure_altitude_m=float(reference.pressure_altitude_m[0]),
        initial_mass_kg=INITIAL_MASS_KG,
        integration_rate_hz=INTEGRATION_RATE_HZ,
    )
    hybrid_mach = np.array([
        hybrid.true_airspeed_mps[index]
        / isa_atmosphere(hybrid.pressure_altitude_m[index]).speed_of_sound_mps.item()
        for index in range(len(reference.time_s))
    ])
    surface_weight = np.array([
        hybrid_model.surface_weight_scalar(alpha, mach)
        for alpha, mach in zip(hybrid.angle_of_attack_rad, hybrid_mach, strict=True)
    ])
    document = _build_document(tracking, empirical, reference_north, reference_east, surface_weight, hybrid_mach)
    table = _trajectory_table(tracking, empirical, reference_north, reference_east, surface_weight, hybrid_mach)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.trajectory_csv, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_render_report(document))
    _plot_trajectories(tracking, empirical, reference_north, reference_east, args.figure_prefix)
    _plot_controls(tracking, empirical, surface_weight, args.control_figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])
    for name, gate in document["gates"].items():
        print(f"{name}: {gate['value']:.6g} ({'pass' if gate['passed'] else 'fail'})")


if __name__ == "__main__":
    main()
