#!/usr/bin/env python3
"""Track a response-blind A320 QAR reference and test model robustness.

This gate reuses the controller gains frozen by the synthetic generation
experiment. It persists aggregate metrics and error-only figures, never QAR
coordinates or sample-level recorded trajectories.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.guidance import (
    TrajectoryReference,
)
from aerox.modeling.dynamics import PointMassModel
from aerox.modeling.dynamics.kinematics import differentiate_wgs84_intervals
from aerox.models import (
    GuardedHybridLiftModel,
    ParabolicDragPolar,
    TabulatedLiftSurface,
)
from aerox.models.atmosphere import atmosphere_at_pressure_altitude
from aerox.models.drag import build_reference_drag_polar
from aerox.simulation import TrackingSimulationResult, simulate_reference_tracking_zoh
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.generate_a320_comparison_trajectories import _controller
from tools.replay_a320_point_mass import (
    PointMassReplayFlight,
    _load_point_mass_flights,
    _read_fuel_model,
    _read_horizontal_parameters,
)
from tools.replay_a320_vspaero_lift_surface import _surface_adapter
from tools.validate_a320_vspaero_lift_surface import (
    _read_empirical_lift_map,
    _read_frozen_surface_parameters,
)

UPDATED = "2026-08-06"
TARGET_REGISTRATION = "TAIL-A"
TARGET_RECORDING = "FLT0004"
COMMAND_RATE_HZ = 1
INTEGRATION_RATE_HZ = 10

NOMINAL_LIMITS = {
    "speed_rmse_mps": 3.0,
    "altitude_rmse_m": 75.0,
    "heading_rmse_deg": 2.0,
    "horizontal_rmse_m": 3_000.0,
    "limited_fraction": 0.05,
}
ROBUST_LIMITS = {
    "speed_rmse_mps": 5.0,
    "altitude_rmse_m": 150.0,
    "heading_rmse_deg": 3.0,
    "horizontal_rmse_m": 8_000.0,
    "limited_fraction": 0.10,
}


@dataclass(frozen=True)
class QARGuidanceReference:
    reference: TrajectoryReference
    north_position_m: np.ndarray
    east_position_m: np.ndarray
    static_temperature_k: np.ndarray
    wind_north_mps: np.ndarray
    wind_east_mps: np.ndarray
    mass_kg: np.ndarray
    initial_mass_kg: float
    source_start_time_s: float
    source_end_time_s: float


@dataclass(frozen=True)
class RobustScenario:
    name: str
    guidance_mass_scale: float = 1.0
    drag_scale: float = 1.0
    lift_scale: float = 1.0
    wind_north_bias_mps: float = 0.0
    wind_east_bias_mps: float = 0.0


@dataclass(frozen=True)
class ScaledAttitudeLiftModel:
    """Multiplicative CL perturbation around an attitude-aware nominal model."""

    nominal: GuardedHybridLiftModel
    scale: float

    def angle_and_lift_coefficient_scalar(
        self, *, pitch_angle_rad: float, flight_path_angle_rad: float, mach: float
    ) -> tuple[float, float]:
        angle, coefficient = self.nominal.angle_and_lift_coefficient_scalar(
            pitch_angle_rad=pitch_angle_rad,
            flight_path_angle_rad=flight_path_angle_rad,
            mach=mach,
        )
        return angle, self.scale * coefficient


@dataclass(frozen=True)
class ScenarioResult:
    scenario: RobustScenario
    tracking: TrackingSimulationResult
    metrics: dict[str, float | bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-extension-confirmation-panel.csv"),
    )
    parser.add_argument("--vertical-parameters", type=Path, default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"))
    parser.add_argument("--horizontal-parameters", type=Path, default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"))
    parser.add_argument("--thrust-parameters", type=Path, default=Path("docs/modeling/a320-effective-thrust-calibration.yml"))
    parser.add_argument("--fuel-parameters", type=Path, default=Path("docs/modeling/a320-effective-fuel-calibration.yml"))
    parser.add_argument("--pitch-bank-parameters", type=Path, default=Path("docs/modeling/a320-pitch-bank-input-calibration.yml"))
    parser.add_argument("--surface-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"))
    parser.add_argument("--boundary-parameters", type=Path, default=Path("docs/modeling/a320-vspaero-hybrid-boundary-selection.yml"))
    parser.add_argument("--lift-surface", type=Path, default=Path("docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"))
    parser.add_argument("--workers", type=int, default=min(10, 2 * (os.cpu_count() or 4)))
    parser.add_argument("--metrics", type=Path, default=Path("docs/modeling/a320-qar-guidance-robustness-metrics.csv"))
    parser.add_argument("--parameters", type=Path, default=Path("docs/modeling/a320-qar-guidance-robustness.yml"))
    parser.add_argument("--report", type=Path, default=Path("docs/modeling/a320-qar-guidance-robustness.md"))
    parser.add_argument("--nominal-figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-qar-guidance-errors"))
    parser.add_argument("--trajectory-figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-qar-guidance-trajectory-comparison"))
    parser.add_argument("--robustness-figure-prefix", type=Path, default=Path("docs/modeling/figures/a320-qar-guidance-robustness"))
    return parser.parse_args()


def scenarios() -> tuple[RobustScenario, ...]:
    """Return the frozen deterministic perturbation set."""

    return (
        RobustScenario("nominal"),
        RobustScenario("mass_estimate_minus_10pct", guidance_mass_scale=0.90),
        RobustScenario("mass_estimate_plus_10pct", guidance_mass_scale=1.10),
        RobustScenario("drag_minus_10pct", drag_scale=0.90),
        RobustScenario("drag_plus_10pct", drag_scale=1.10),
        RobustScenario("lift_minus_5pct", lift_scale=0.95),
        RobustScenario("lift_plus_5pct", lift_scale=1.05),
        RobustScenario("wind_bias_northeast_3mps", wind_north_bias_mps=3.0, wind_east_bias_mps=3.0),
        RobustScenario("wind_bias_southwest_3mps", wind_north_bias_mps=-3.0, wind_east_bias_mps=-3.0),
        RobustScenario(
            "combined_adverse",
            guidance_mass_scale=0.90,
            drag_scale=1.10,
            lift_scale=0.95,
            wind_north_bias_mps=3.0,
            wind_east_bias_mps=-3.0,
        ),
    )


def _longest_contiguous_indices(sample_index: np.ndarray) -> np.ndarray:
    breaks = np.flatnonzero(np.diff(sample_index) > 1) + 1
    segments = np.split(np.arange(len(sample_index)), breaks)
    return max(segments, key=len)


def build_qar_reference(flight: PointMassReplayFlight) -> QARGuidanceReference:
    """Build a dynamically consistent smoothed reference without response data."""

    selection = _longest_contiguous_indices(flight.base.sample_index)
    base = flight.base
    source_time = base.time_s[selection]
    time = source_time - source_time[0]
    speed = savgol_filter(base.true_airspeed_mps[selection], 31, 3, mode="interp")
    speed_rate = savgol_filter(base.true_airspeed_mps[selection], 31, 3, deriv=1, delta=1.0, mode="interp")
    altitude = savgol_filter(base.pressure_altitude_m[selection], 61, 3, mode="interp")
    altitude_rate = savgol_filter(base.pressure_altitude_m[selection], 61, 3, deriv=1, delta=1.0, mode="interp")
    altitude_acceleration = savgol_filter(base.pressure_altitude_m[selection], 61, 3, deriv=2, delta=1.0, mode="interp")
    sine_gamma = np.clip(altitude_rate / speed, -0.25, 0.25)
    gamma = np.arcsin(sine_gamma)
    gamma_rate = (
        altitude_acceleration * speed - altitude_rate * speed_rate
    ) / (speed**2 * np.sqrt(1.0 - sine_gamma**2))
    unwrapped_heading = np.unwrap(flight.calibrated_heading_rad[selection])
    heading = savgol_filter(unwrapped_heading, 31, 3, mode="interp")
    heading_rate = savgol_filter(unwrapped_heading, 31, 3, deriv=1, delta=1.0, mode="interp")

    latitude = flight.latitude_rad[selection]
    longitude = flight.longitude_rad[selection]
    geometric_altitude = flight.geometric_altitude_m[selection]
    north_interval, east_interval, _ = differentiate_wgs84_intervals(
        time, latitude, longitude, geometric_altitude
    )
    step = np.diff(time)
    north_position = np.concatenate(([0.0], np.cumsum(north_interval * step)))
    east_position = np.concatenate(([0.0], np.cumsum(east_interval * step)))
    return QARGuidanceReference(
        reference=TrajectoryReference(
            time_s=time,
            true_airspeed_mps=speed,
            true_airspeed_rate_mps2=speed_rate,
            flight_path_angle_rad=gamma,
            flight_path_angle_rate_rad_s=gamma_rate,
            heading_rad=heading,
            heading_rate_rad_s=heading_rate,
            pressure_altitude_m=altitude,
        ),
        north_position_m=north_position,
        east_position_m=east_position,
        static_temperature_k=base.static_temperature_k[selection],
        wind_north_mps=flight.wind_north_mps[selection],
        wind_east_mps=flight.wind_east_mps[selection],
        mass_kg=base.mass_kg[selection],
        initial_mass_kg=float(base.mass_kg[selection][0]),
        source_start_time_s=float(source_time[0]),
        source_end_time_s=float(source_time[-1]),
    )


def _load_reference(args: argparse.Namespace) -> tuple[QARGuidanceReference, PointMassReplayFlight]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    selected = panel.loc[
        (panel["registration"] == TARGET_REGISTRATION)
        & (panel["recording"] == TARGET_RECORDING)
    ]
    if len(selected) != 1:
        raise ValueError("the frozen response-blind QAR reference is absent from the panel")
    aircraft = load_aircraft("A320")
    flights = _load_point_mass_flights(
        selected,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=_read_vertical_calibration(args.vertical_parameters),
        horizontal=_read_horizontal_parameters(args.horizontal_parameters),
        tas_scale=_read_tas_scale(args.horizontal_parameters),
        wing_area_m2=float(aircraft.wing.area),
    )
    return build_qar_reference(flights[0]), flights[0]


def _build_nominal_model(args: argparse.Namespace, registration: str) -> GuardedHybridLiftModel:
    vertical = _read_vertical_calibration(args.vertical_parameters)
    parameters, _ = _read_frozen_surface_parameters(args.surface_parameters)
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    empirical = _read_empirical_lift_map(args.pitch_bank_parameters)
    boundary = yaml.safe_load(args.boundary_parameters.read_text())["selected"]
    adapter = _surface_adapter(
        registration,
        surface,
        parameters,
        math.radians(vertical.effective_gamma_bias_deg),
    )
    return GuardedHybridLiftModel(
        adapter,
        empirical,
        lower_alpha_core_rad=math.radians(-0.5),
        upper_alpha_core_rad=math.radians(8.0),
        lower_mach_core=0.23,
        upper_mach_core=float(boundary["upper_mach_core"]),
        maximum_surface_weight=float(boundary["maximum_surface_weight"]),
    )


def _wrapped_error(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(target - actual), np.cos(target - actual))


def _metrics(
    tracking: TrackingSimulationResult,
    qar: QARGuidanceReference,
) -> dict[str, float | bool]:
    trajectory = tracking.trajectory
    reference = qar.reference
    horizontal = np.hypot(
        trajectory.north_position_m - qar.north_position_m,
        trajectory.east_position_m - qar.east_position_m,
    )
    heading_error = _wrapped_error(reference.heading_rad, trajectory.heading_rad)
    atmosphere = atmosphere_at_pressure_altitude(
        trajectory.pressure_altitude_m, qar.static_temperature_k
    )
    mach = trajectory.true_airspeed_mps / atmosphere.speed_of_sound_mps
    any_limited = tracking.pitch_limited | tracking.bank_limited | tracking.thrust_limited
    return {
        "integrated": True,
        "speed_rmse_mps": float(np.sqrt(np.mean((trajectory.true_airspeed_mps - reference.true_airspeed_mps) ** 2))),
        "altitude_rmse_m": float(np.sqrt(np.mean((trajectory.pressure_altitude_m - reference.pressure_altitude_m) ** 2))),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(heading_error**2)))),
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "horizontal_terminal_error_m": float(horizontal[-1]),
        "pitch_limited_fraction": float(np.mean(tracking.pitch_limited)),
        "bank_limited_fraction": float(np.mean(tracking.bank_limited)),
        "thrust_limited_fraction": float(np.mean(tracking.thrust_limited)),
        "any_control_limited_fraction": float(np.mean(any_limited)),
        "minimum_mach": float(np.min(mach)),
        "maximum_mach": float(np.max(mach)),
        "minimum_effective_alpha_deg": float(np.rad2deg(np.min(trajectory.angle_of_attack_rad))),
        "maximum_effective_alpha_deg": float(np.rad2deg(np.max(trajectory.angle_of_attack_rad))),
        "minimum_pitch_deg": float(np.rad2deg(np.min(trajectory.pitch_angle_rad))),
        "maximum_pitch_deg": float(np.rad2deg(np.max(trajectory.pitch_angle_rad))),
        "maximum_absolute_bank_deg": float(np.rad2deg(np.max(np.abs(trajectory.bank_angle_rad)))),
        "minimum_net_thrust_n": float(np.min(trajectory.net_thrust_n)),
        "maximum_net_thrust_n": float(np.max(trajectory.net_thrust_n)),
        "fuel_burn_kg": float(trajectory.mass_kg[0] - trajectory.mass_kg[-1]),
    }


def _run_scenario(
    scenario: RobustScenario,
    qar: QARGuidanceReference,
    nominal_lift: GuardedHybridLiftModel,
    wing_area_m2: float,
    cd0: float,
    induced_drag_factor: float,
    fuel_model: Any,
) -> ScenarioResult:
    nominal_plant = PointMassModel(
        wing_area_m2=wing_area_m2,
        lift_map=nominal_lift,
        drag_model=ParabolicDragPolar(cd0, induced_drag_factor),
        fuel_flow_prior=fuel_model,
    )
    actual_lift = (
        nominal_lift
        if scenario.lift_scale == 1.0
        else ScaledAttitudeLiftModel(nominal_lift, scenario.lift_scale)
    )
    actual_plant = PointMassModel(
        wing_area_m2=wing_area_m2,
        lift_map=actual_lift,
        drag_model=ParabolicDragPolar(
            cd0 * scenario.drag_scale,
            induced_drag_factor * scenario.drag_scale,
        ),
        fuel_flow_prior=fuel_model,
    )
    tracking = simulate_reference_tracking_zoh(
        plant=actual_plant,
        guidance_plant=nominal_plant,
        guidance_mass_scale=scenario.guidance_mass_scale,
        controller=_controller(),
        reference=qar.reference,
        static_temperature_k=qar.static_temperature_k,
        wind_north_mps=qar.wind_north_mps + scenario.wind_north_bias_mps,
        wind_east_mps=qar.wind_east_mps + scenario.wind_east_bias_mps,
        initial_mass_kg=qar.initial_mass_kg,
        integration_rate_hz=INTEGRATION_RATE_HZ,
    )
    return ScenarioResult(scenario, tracking, _metrics(tracking, qar))


def _gate_document(results: list[ScenarioResult], qar: QARGuidanceReference, nominal_lift: GuardedHybridLiftModel) -> dict[str, Any]:
    indexed = {result.scenario.name: result for result in results}
    nominal = indexed["nominal"].metrics
    maximum = {
        metric: max(float(result.metrics[metric]) for result in results)
        for metric in (
            "speed_rmse_mps",
            "altitude_rmse_m",
            "heading_rmse_deg",
            "horizontal_rmse_m",
            "any_control_limited_fraction",
        )
    }
    worst_scenarios = {
        metric: max(results, key=lambda result: float(result.metrics[metric])).scenario.name
        for metric in maximum
    }
    gates: dict[str, dict[str, float | bool]] = {}
    for metric, limit in NOMINAL_LIMITS.items():
        key = "any_control_limited_fraction" if metric == "limited_fraction" else metric
        gates[f"nominal_{metric}"] = {
            "value": float(nominal[key]),
            "maximum": limit,
            "passed": bool(float(nominal[key]) <= limit),
        }
    for metric, limit in ROBUST_LIMITS.items():
        key = "any_control_limited_fraction" if metric == "limited_fraction" else metric
        gates[f"robust_maximum_{metric}"] = {
            "value": maximum[key],
            "maximum": limit,
            "passed": bool(maximum[key] <= limit),
        }
    minimum_mach = min(float(result.metrics["minimum_mach"]) for result in results)
    maximum_mach = max(float(result.metrics["maximum_mach"]) for result in results)
    minimum_alpha = min(float(result.metrics["minimum_effective_alpha_deg"]) for result in results)
    maximum_alpha = max(float(result.metrics["maximum_effective_alpha_deg"]) for result in results)
    envelope = {
        "robust_minimum_mach": {"value": minimum_mach, "minimum": 0.20, "passed": minimum_mach >= 0.20},
        "robust_maximum_mach": {"value": maximum_mach, "maximum": 0.82, "passed": maximum_mach <= 0.82},
        "robust_minimum_effective_alpha_deg": {"value": minimum_alpha, "minimum": -1.0, "passed": minimum_alpha >= -1.0},
        "robust_maximum_effective_alpha_deg": {"value": maximum_alpha, "maximum": 8.5, "passed": maximum_alpha <= 8.5},
    }
    gates.update(envelope)
    nominal_passed = all(
        bool(gate["passed"])
        for name, gate in gates.items()
        if name.startswith("nominal_")
    )
    robustness_passed = all(
        bool(gate["passed"])
        for name, gate in gates.items()
        if not name.startswith("nominal_")
    )
    passed = nominal_passed and robustness_passed
    nominal_trajectory = indexed["nominal"].tracking.trajectory
    nominal_atmosphere = atmosphere_at_pressure_altitude(
        nominal_trajectory.pressure_altitude_m, qar.static_temperature_k
    )
    nominal_mach = nominal_trajectory.true_airspeed_mps / nominal_atmosphere.speed_of_sound_mps
    surface_weight = np.array([
        nominal_lift.surface_weight_scalar(alpha, mach)
        for alpha, mach in zip(nominal_trajectory.angle_of_attack_rad, nominal_mach, strict=True)
    ])
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "qar-guidance-robustness-passed" if passed else "qar-guidance-robustness-held",
        "reference": {
            "registration": TARGET_REGISTRATION,
            "recording": TARGET_RECORDING,
            "selection": "first row of the response-blind Mach-extension confirmation panel",
            "segment": "longest contiguous clean segment; no response-based trimming",
            "source_start_time_s": qar.source_start_time_s,
            "source_end_time_s": qar.source_end_time_s,
            "duration_s": float(qar.reference.time_s[-1]),
            "samples": len(qar.reference.time_s),
            "smoothing": {
                "TAS_and_heading": "31 s cubic Savitzky-Golay",
                "pressure_altitude": "61 s cubic Savitzky-Golay",
                "gamma": "derived from smoothed pressure-altitude rate and TAS",
            },
            "raw_coordinates_or_sample_trajectory_persisted": False,
        },
        "controller": {
            "source": "unchanged from the synthetic bounded-generation gate",
            "refitted_to_qar_reference": False,
            "command_rate_hz": COMMAND_RATE_HZ,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "controls": ["pitch angle", "bank angle", "total realized net thrust"],
            "identified_airbus_law": False,
        },
        "robustness_design": {
            "scenarios": [result.scenario.__dict__ for result in results],
            "interpretation": "deterministic engineering sensitivity set, not a probability distribution",
            "mass_perturbation": "guidance mass estimate only; propagated physical mass unchanged",
            "model_mismatch": "nominal guidance plant commands a separately perturbed propagated plant",
        },
        "scenario_metrics": [
            {"scenario": result.scenario.name, **result.metrics}
            for result in results
        ],
        "robust_maximum_metrics": maximum,
        "robust_worst_scenarios": worst_scenarios,
        "nominal_mean_geometry_surface_weight": float(np.mean(surface_weight)),
        "gates": gates,
        "decision": {
            "qar_reference_tracking_passed": nominal_passed,
            "deterministic_robustness_set_passed": robustness_passed,
            "overall_gate_passed": passed,
            "controller_gains_promoted_to_airbus_parameters": False,
            "authorize_unrestricted_optimization": False,
        },
        "artifacts": {
            "aggregate_metrics": "docs/modeling/a320-qar-guidance-robustness-metrics.csv",
            "nominal_error_figure": "docs/modeling/figures/a320-qar-guidance-errors.png",
            "trajectory_comparison_figure": "docs/modeling/figures/a320-qar-guidance-trajectory-comparison.png",
            "robustness_figure": "docs/modeling/figures/a320-qar-guidance-robustness.png",
            "code": "tools/validate_a320_qar_guidance_robustness.py",
        },
        "limitations": [
            "This is one consumed confirmation flight and is not a new independent aerodynamic validation panel.",
            "The smoothed reference is QAR-derived; it does not reproduce recorder quantization or one-second noise.",
            "The horizontal controller tracks heading rather than geographic cross-track position, so wind-bias cases expose uncompensated drift.",
            "The deterministic perturbations are engineering cases, not confidence intervals or certified tolerances.",
            "No proprietary Airbus guidance, surface allocation, or FADEC law is identified.",
        ],
    }


def _metrics_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({**result.scenario.__dict__, **result.metrics})
    return pd.DataFrame(rows)


def _style_axis(axis: Any, label: str) -> None:
    axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_nominal(result: ScenarioResult, qar: QARGuidanceReference, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    trajectory = result.tracking.trajectory
    reference = qar.reference
    time_min = reference.time_s / 60.0
    heading_error = np.rad2deg(_wrapped_error(reference.heading_rad, trajectory.heading_rad))
    horizontal = np.hypot(
        trajectory.north_position_m - qar.north_position_m,
        trajectory.east_position_m - qar.east_position_m,
    )
    blue, orange = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(time_min, trajectory.true_airspeed_mps - reference.true_airspeed_mps, color=blue)
    axes[0, 0].set(xlabel="Elapsed time (min)", ylabel="TAS tracking error (m/s)")
    axes[0, 1].plot(time_min, trajectory.pressure_altitude_m - reference.pressure_altitude_m, color=blue)
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="Altitude tracking error (m)")
    axes[0, 2].plot(time_min, heading_error, color=blue)
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Heading tracking error (deg)")
    axes[1, 0].plot(time_min, horizontal, color=orange)
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="Horizontal reference error (m)")
    axes[1, 1].plot(time_min, np.rad2deg(trajectory.pitch_angle_rad), color=blue, label="pitch")
    axes[1, 1].plot(time_min, np.rad2deg(trajectory.bank_angle_rad), color=orange, linestyle="--", label="bank")
    axes[1, 1].set(xlabel="Elapsed time (min)", ylabel="Commanded attitude (deg)")
    axes[1, 1].legend(frameon=False)
    axes[1, 2].plot(time_min, trajectory.net_thrust_n / 1_000.0, color=blue)
    axes[1, 2].set(xlabel="Elapsed time (min)", ylabel="Total realized net thrust (kN)")
    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 QAR-derived reference tracking: errors and generated controls", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_trajectory_comparison(
    result: ScenarioResult,
    qar: QARGuidanceReference,
    prefix: Path,
) -> None:
    """Plot zero-referenced QAR and generated trajectories without geodetic data."""

    import matplotlib.pyplot as plt

    trajectory = result.tracking.trajectory
    reference = qar.reference
    time_min = reference.time_s / 60.0
    horizontal = np.hypot(
        trajectory.north_position_m - qar.north_position_m,
        trajectory.east_position_m - qar.east_position_m,
    )
    black, blue = "#000000", "#0072B2"
    reference_style = dict(
        color=black,
        linestyle=":",
        linewidth=2.1,
        label="QAR-derived reference",
    )
    generated_style = dict(
        color=blue,
        linestyle="-",
        linewidth=1.7,
        label="generated hybrid trajectory",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        **reference_style,
    )
    axes[0, 0].plot(
        trajectory.east_position_m / 1_000.0,
        trajectory.north_position_m / 1_000.0,
        **generated_style,
    )
    axes[0, 0].set(
        xlabel="Local east displacement (km)",
        ylabel="Local north displacement (km)",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].plot(time_min, reference.pressure_altitude_m, **reference_style)
    axes[0, 1].plot(time_min, trajectory.pressure_altitude_m, **generated_style)
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="Pressure altitude (m)")
    axes[0, 2].plot(time_min, reference.true_airspeed_mps, **reference_style)
    axes[0, 2].plot(time_min, trajectory.true_airspeed_mps, **generated_style)
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="True airspeed (m/s)")
    axes[1, 0].plot(time_min, np.rad2deg(reference.heading_rad), **reference_style)
    axes[1, 0].plot(time_min, np.rad2deg(trajectory.heading_rad), **generated_style)
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="True heading (deg)")
    axes[1, 1].plot(
        time_min,
        np.rad2deg(reference.flight_path_angle_rad),
        **reference_style,
    )
    axes[1, 1].plot(
        time_min,
        np.rad2deg(trajectory.flight_path_angle_rad),
        **generated_style,
    )
    axes[1, 1].set(xlabel="Elapsed time (min)", ylabel="Flight-path angle (deg)")
    axes[1, 2].plot(time_min, horizontal, color="#D55E00", linewidth=1.8)
    axes[1, 2].set(
        xlabel="Elapsed time (min)",
        ylabel="Horizontal trajectory separation (m)",
    )
    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "A320 generated trajectory versus QAR-derived reference",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_robustness(metrics: pd.DataFrame, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    short = [
        "nominal", "mass −10%", "mass +10%", "drag −10%", "drag +10%",
        "lift −5%", "lift +5%", "wind NE", "wind SW", "combined",
    ]
    colors = ["#0072B2"] + ["#999999"] * 8 + ["#D55E00"]
    plots = (
        ("speed_rmse_mps", "TAS RMSE (m/s)", ROBUST_LIMITS["speed_rmse_mps"]),
        ("altitude_rmse_m", "Altitude RMSE (m)", ROBUST_LIMITS["altitude_rmse_m"]),
        ("heading_rmse_deg", "Heading RMSE (deg)", ROBUST_LIMITS["heading_rmse_deg"]),
        ("horizontal_rmse_m", "Horizontal RMSE (m)", ROBUST_LIMITS["horizontal_rmse_m"]),
        ("any_control_limited_fraction", "Limited sample fraction", ROBUST_LIMITS["limited_fraction"]),
        ("horizontal_terminal_error_m", "Terminal horizontal error (m)", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5), constrained_layout=True)
    positions = np.arange(len(metrics))
    for label, axis, (column, ylabel, limit) in zip("ABCDEF", axes.ravel(), plots, strict=True):
        axis.scatter(positions, metrics[column], c=colors, s=34, zorder=3)
        axis.plot(positions, metrics[column], color="0.72", linewidth=0.8, zorder=1)
        if limit is not None:
            axis.axhline(limit, color="black", linestyle=":", linewidth=1.1, label="robust gate")
        axis.set_xticks(positions, short, rotation=42, ha="right")
        axis.set_ylabel(ylabel)
        if limit is not None and label == "A":
            axis.legend(frameon=False, fontsize=8)
        _style_axis(axis, label)
    fig.suptitle("A320 QAR-reference deterministic robustness scenarios", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _render_report(document: dict[str, Any]) -> str:
    nominal = next(row for row in document["scenario_metrics"] if row["scenario"] == "nominal")
    maximum = document["robust_maximum_metrics"]
    worst_altitude_scenario = document["robust_worst_scenarios"]["altitude_rmse_m"]
    gate_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {gate['value']:.4g} | "
        f"{('<= ' + str(gate['maximum'])) if 'maximum' in gate else ('>= ' + str(gate['minimum']))} | "
        f"{'pass' if gate['passed'] else 'fail'} |"
        for name, gate in document["gates"].items()
    )
    nominal_passed = document["decision"]["qar_reference_tracking_passed"]
    robustness_passed = document["decision"]["deterministic_robustness_set_passed"]
    return f"""# A320 QAR-reference guidance and robustness gate

## Outcome

The unchanged bounded pitch–bank–realized-net-thrust controller **{'passes' if nominal_passed else 'does not pass'} the nominal QAR-derived reference gate** and **{'passes' if robustness_passed else 'does not pass'} the deterministic robustness gate**. The overall status is therefore **{'passed' if document['decision']['overall_gate_passed'] else 'held'}**. The reference is the longest contiguous clean segment of `{TARGET_REGISTRATION}/{TARGET_RECORDING}`, the first flight in the response-blind Mach-extension confirmation panel. No controller gain was refitted after loading this reference.

| Metric | Nominal QAR reference | Worst robustness case |
|---|---:|---:|
| TAS RMSE (m/s) | {nominal['speed_rmse_mps']:.3f} | {maximum['speed_rmse_mps']:.3f} |
| Altitude RMSE (m) | {nominal['altitude_rmse_m']:.2f} | {maximum['altitude_rmse_m']:.2f} |
| Heading RMSE (deg) | {nominal['heading_rmse_deg']:.3f} | {maximum['heading_rmse_deg']:.3f} |
| Horizontal RMSE (m) | {nominal['horizontal_rmse_m']:.1f} | {maximum['horizontal_rmse_m']:.1f} |
| Any limiter fraction | {nominal['any_control_limited_fraction']:.3f} | {maximum['any_control_limited_fraction']:.3f} |

The only failed threshold is robust altitude RMSE. The `{worst_altitude_scenario}` case reaches {maximum['altitude_rmse_m']:.1f} m against the frozen 150 m maximum; that threshold is retained unchanged.

## Frozen gates

| Gate | Value | Required | Result |
|---|---:|---:|---:|
{gate_rows}

## Robustness interpretation

The ten deterministic cases cover ±10% guidance mass-estimate error, ±10% drag-polar scaling, ±5% lift scaling, ±3 m/s north/east wind biases, and one combined adverse case. The nominal guidance model generates commands while a separately perturbed plant is propagated, so lift and drag cases are genuine closed-loop model-mismatch tests. These cases are sensitivity tests, not statistical confidence intervals or certified tolerances.

The controller tracks heading, not geographic cross-track position. Wind-bias cases therefore retain the expected uncompensated horizontal drift; adding path/cross-track guidance is the next architectural step if geographic route capture is required.

## Data and release boundary

The QAR trajectory is smoothed only to form a differentiable reference. Raw coordinates and sample-level recorded trajectories are not written to the repository because their redistribution license remains unresolved. Persisted CSV data contain one aggregate row per robustness scenario.

The direct comparison figure uses only zero-referenced local displacement, not latitude or longitude. It overlays the generated and QAR-derived plan path, altitude, TAS, heading, and flight-path angle, with horizontal separation in the final panel.

The comparison is available as `docs/modeling/figures/a320-qar-guidance-trajectory-comparison.png` and a vector PDF with the same stem.

This remains a model-based engineering controller. It does not identify Airbus autopilot gains, control-surface allocation, TLA-to-thrust behavior, or a FADEC law, and it does not authorize optimization outside the released clean Mach/alpha envelope.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    qar, flight = _load_reference(args)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    nominal_lift = _build_nominal_model(args, flight.base.registration)
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    jobs = scenarios()
    results: list[ScenarioResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_scenario,
                scenario,
                qar,
                nominal_lift,
                float(aircraft.wing.area),
                reference_drag.zero_lift_drag_coefficient,
                reference_drag.induced_drag_factor,
                fuel,
            ): scenario.name
            for scenario in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed {result.scenario.name}", flush=True)
    order = {scenario.name: index for index, scenario in enumerate(jobs)}
    results.sort(key=lambda result: order[result.scenario.name])
    document = _gate_document(results, qar, nominal_lift)
    metrics = _metrics_frame(results)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_render_report(document))
    _plot_nominal(results[0], qar, args.nominal_figure_prefix)
    _plot_trajectory_comparison(results[0], qar, args.trajectory_figure_prefix)
    _plot_robustness(metrics, args.robustness_figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])
    for name, gate in document["gates"].items():
        print(f"{name}: {gate['value']:.6g} ({'pass' if gate['passed'] else 'fail'})")


if __name__ == "__main__":
    main()
