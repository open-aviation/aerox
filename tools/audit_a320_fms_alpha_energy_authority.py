#!/usr/bin/env python3
"""Separate planned-profile demand from closed-loop alpha/energy demand."""

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

from aerox.fms import (
    PointMassProfilePerformance,
    SpatialPathGuidance,
    plan_energy_shaped_profile,
    simulate_fms_trajectory_zoh,
)
from aerox.models.atmosphere import atmosphere_at_pressure_altitude
from aerox.models.atmosphere import G0
from tools.generate_a320_comparison_trajectories import _controller
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    INTEGRATION_RATE_HZ,
    _build_plant,
    build_sparse_intent,
)
from tools.validate_a320_qar_guidance_robustness import _load_reference

UPDATED = "2026-08-06"
ALPHA_SUPPORT_MAX_DEG = 8.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-extension-confirmation-panel.csv"),
    )
    parser.add_argument(
        "--vertical-parameters",
        type=Path,
        default=Path("docs/modeling/a320-vertical-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--horizontal-parameters",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"),
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
        "--n1-envelope",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
    )
    parser.add_argument(
        "--climb-capacity",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity.yml"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("docs/fms/a320-fms-alpha-energy-authority-history.csv"),
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("docs/fms/a320-fms-alpha-energy-authority-segments.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-alpha-energy-authority.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-alpha-energy-authority.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-alpha-energy-authority"),
    )
    return parser.parse_args()


def _effective_alpha_for_lift(
    *,
    controller: Any,
    plant: Any,
    target_lift_coefficient: float,
    flight_path_angle_rad: float,
    mach: float,
) -> tuple[float, bool]:
    pitch, limited = controller._pitch_for_lift_coefficient(
        plant,
        target_lift_coefficient=target_lift_coefficient,
        flight_path_angle_rad=flight_path_angle_rad,
        mach=mach,
    )
    angle, _ = plant.lift_map.angle_and_lift_coefficient_scalar(
        pitch_angle_rad=pitch,
        flight_path_angle_rad=flight_path_angle_rad,
        mach=mach,
    )
    return float(angle), bool(limited)


def _build_rollout(args: argparse.Namespace) -> tuple[Any, ...]:
    qar, _ = _load_reference(args)
    route, environment, anchor_indices = build_sparse_intent(qar)
    plant = _build_plant(args)
    thrust_ceiling = _load_climb_thrust_ceiling(args)
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
        static_temperature_k=float(np.median(qar.static_temperature_k[anchor_indices])),
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[anchor_indices],
    )
    guidance = SpatialPathGuidance(
        profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
        command_step_s=1.0,
    )
    controller = _controller()
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
    )
    return (
        qar,
        route,
        environment,
        anchor_indices,
        plant,
        thrust_ceiling,
        performance,
        profile,
        controller,
        result,
    )


def _diagnostics(
    qar: Any,
    route: Any,
    environment: Any,
    anchor_indices: np.ndarray,
    plant: Any,
    thrust_ceiling: Any,
    profile: Any,
    controller: Any,
    result: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trajectory = result.trajectory
    atmosphere = atmosphere_at_pressure_altitude(
        trajectory.pressure_altitude_m, result.static_temperature_k
    )
    mach = trajectory.true_airspeed_mps / atmosphere.speed_of_sound_mps
    dynamic_pressure = 0.5 * atmosphere.density_kg_m3 * trajectory.true_airspeed_mps**2
    profile_lookahead_m = np.minimum(
        trajectory.true_airspeed_mps * 5.0,
        1_000.0,
    )
    profile_target_s = np.minimum(
        result.progress_m + profile_lookahead_m,
        profile.trajectory.total_distance_m,
    )
    profile_samples = [
        profile.trajectory.sample(float(value)) for value in profile_target_s
    ]
    profile_altitude = np.array([sample.altitude_m for sample in profile_samples])
    profile_speed = np.array([sample.true_airspeed_mps for sample in profile_samples])
    profile_gamma = np.array(
        [sample.flight_path_angle_rad for sample in profile_samples]
    )
    profile_acceleration = np.array(
        [sample.true_airspeed_rate_mps2 for sample in profile_samples]
    )
    segment_index = np.array(
        [sample.segment_index for sample in profile_samples], dtype=int
    )
    planning_mass = np.interp(
        profile_target_s,
        route.waypoint_along_path_m,
        qar.mass_kg[anchor_indices],
    )
    planning_temperature = np.array(
        [
            environment.sample(float(value)).static_temperature_k
            for value in profile_target_s
        ]
    )
    planning_atmosphere = atmosphere_at_pressure_altitude(
        profile_altitude, planning_temperature
    )
    planning_mach = profile_speed / planning_atmosphere.speed_of_sound_mps
    planning_dynamic_pressure = (
        0.5 * planning_atmosphere.density_kg_m3 * profile_speed**2
    )

    maximum_thrust = np.array(
        [
            thrust_ceiling.maximum_thrust_n(
                pressure_altitude_m=float(altitude),
                true_airspeed_mps=float(speed),
                static_temperature_k=float(temperature),
            )
            for altitude, speed, temperature in zip(
                trajectory.pressure_altitude_m,
                trajectory.true_airspeed_mps,
                result.static_temperature_k,
                strict=True,
            )
        ]
    )
    capacity_saturated = np.isclose(
        trajectory.net_thrust_n,
        maximum_thrust,
        rtol=0.0,
        atol=1.0,
    )

    altitude_error = result.reference_altitude_m - trajectory.pressure_altitude_m
    speed_error = result.reference_true_airspeed_mps - trajectory.true_airspeed_mps
    speed_feedback = controller.speed_error_gain_per_s * speed_error
    altitude_gamma_correction = (
        controller.altitude_to_flight_path_gain_rad_per_m * altitude_error
    )
    gamma_rate_command = controller.flight_path_angle_error_gain_per_s * (
        profile_gamma + altitude_gamma_correction - trajectory.flight_path_angle_rad
    )
    planned_energy_demand = profile_acceleration + G0 * np.sin(profile_gamma)
    commanded_energy_demand = (
        profile_acceleration
        + speed_feedback
        + G0 * np.sin(trajectory.flight_path_angle_rad)
    )
    available_energy = (maximum_thrust - trajectory.drag_n) / trajectory.mass_kg
    realized_energy = (trajectory.net_thrust_n - trajectory.drag_n) / trajectory.mass_kg

    planned_alpha = np.empty(len(trajectory.time_s))
    steady_alpha = np.empty(len(trajectory.time_s))
    controller_target_alpha = np.empty(len(trajectory.time_s))
    planned_alpha_pitch_limited = np.zeros(len(trajectory.time_s), dtype=bool)
    controller_alpha_pitch_limited = np.zeros(len(trajectory.time_s), dtype=bool)
    for index in range(len(trajectory.time_s)):
        planned_cl = (
            planning_mass[index]
            * G0
            * np.cos(profile_gamma[index])
            / (planning_dynamic_pressure[index] * plant.wing_area_m2)
        )
        planned_alpha[index], planned_alpha_pitch_limited[index] = (
            _effective_alpha_for_lift(
                controller=controller,
                plant=plant,
                target_lift_coefficient=float(planned_cl),
                flight_path_angle_rad=float(profile_gamma[index]),
                mach=float(planning_mach[index]),
            )
        )
        steady_cl = (
            trajectory.mass_kg[index]
            * G0
            * np.cos(trajectory.flight_path_angle_rad[index])
            / (
                dynamic_pressure[index]
                * plant.wing_area_m2
                * np.cos(trajectory.bank_angle_rad[index])
            )
        )
        steady_alpha[index], _ = _effective_alpha_for_lift(
            controller=controller,
            plant=plant,
            target_lift_coefficient=float(steady_cl),
            flight_path_angle_rad=float(trajectory.flight_path_angle_rad[index]),
            mach=float(mach[index]),
        )
        (
            controller_target_alpha[index],
            controller_alpha_pitch_limited[index],
        ) = _effective_alpha_for_lift(
            controller=controller,
            plant=plant,
            target_lift_coefficient=float(result.target_lift_coefficient[index]),
            flight_path_angle_rad=float(trajectory.flight_path_angle_rad[index]),
            mach=float(mach[index]),
        )

    planned_alpha_deg = np.rad2deg(planned_alpha)
    steady_alpha_deg = np.rad2deg(steady_alpha)
    controller_target_alpha_deg = np.rad2deg(controller_target_alpha)
    realized_alpha_deg = np.rad2deg(trajectory.angle_of_attack_rad)
    planned_alpha_outside = planned_alpha_deg > ALPHA_SUPPORT_MAX_DEG
    target_alpha_outside = controller_target_alpha_deg > ALPHA_SUPPORT_MAX_DEG
    realized_alpha_outside = realized_alpha_deg > ALPHA_SUPPORT_MAX_DEG

    history = pd.DataFrame(
        {
            "elapsed_time_s": trajectory.time_s,
            "segment": [f"A{index}-A{index + 1}" for index in segment_index],
            "profile_target_along_path_m": profile_target_s,
            "altitude_error_m": altitude_error,
            "true_airspeed_error_mps": speed_error,
            "reference_flight_path_angle_deg": np.rad2deg(profile_gamma),
            "realized_flight_path_angle_deg": np.rad2deg(
                trajectory.flight_path_angle_rad
            ),
            "altitude_gamma_correction_deg": np.rad2deg(altitude_gamma_correction),
            "gamma_rate_command_deg_s": np.rad2deg(gamma_rate_command),
            "planned_energy_demand_mps2": planned_energy_demand,
            "commanded_energy_demand_mps2": commanded_energy_demand,
            "available_energy_mps2": available_energy,
            "realized_energy_mps2": realized_energy,
            "planned_energy_margin_mps2": available_energy - planned_energy_demand,
            "commanded_energy_margin_mps2": available_energy - commanded_energy_demand,
            "realized_net_thrust_kn": trajectory.net_thrust_n / 1_000.0,
            "maximum_net_thrust_kn": maximum_thrust / 1_000.0,
            "thrust_limited": result.thrust_limited,
            "capacity_saturated": capacity_saturated,
            "planned_effective_alpha_deg": planned_alpha_deg,
            "steady_actual_state_effective_alpha_deg": steady_alpha_deg,
            "controller_target_effective_alpha_deg": controller_target_alpha_deg,
            "realized_effective_alpha_deg": realized_alpha_deg,
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_angle_deg": np.rad2deg(trajectory.bank_angle_rad),
        }
    )

    segment_rows: list[dict[str, Any]] = []
    for index in range(len(route.intent.waypoints) - 1):
        selection = segment_index == index
        if not np.any(selection):
            continue
        segment_rows.append(
            {
                "segment": f"A{index}-A{index + 1}",
                "samples": int(np.sum(selection)),
                "minimum_planned_energy_margin_mps2": float(
                    np.min(
                        available_energy[selection] - planned_energy_demand[selection]
                    )
                ),
                "minimum_commanded_energy_margin_mps2": float(
                    np.min(
                        available_energy[selection] - commanded_energy_demand[selection]
                    )
                ),
                "thrust_limited_fraction": float(
                    np.mean(result.thrust_limited[selection])
                ),
                "capacity_saturated_fraction": float(
                    np.mean(capacity_saturated[selection])
                ),
                "maximum_absolute_altitude_error_m": float(
                    np.max(np.abs(altitude_error[selection]))
                ),
                "maximum_absolute_speed_error_mps": float(
                    np.max(np.abs(speed_error[selection]))
                ),
                "maximum_planned_effective_alpha_deg": float(
                    np.max(planned_alpha_deg[selection])
                ),
                "maximum_controller_target_effective_alpha_deg": float(
                    np.max(controller_target_alpha_deg[selection])
                ),
                "maximum_realized_effective_alpha_deg": float(
                    np.max(realized_alpha_deg[selection])
                ),
            }
        )
    segments = pd.DataFrame(segment_rows)

    profile_s = profile.along_path_m
    profile_speed_values = profile.trajectory.profile_true_airspeed_mps
    profile_altitude_values = profile.trajectory.profile_altitude_m
    assert profile_speed_values is not None
    assert profile_altitude_values is not None
    profile_demand = profile_speed_values * np.gradient(
        profile_speed_values, profile_s
    ) + G0 * np.sin(np.arctan(np.gradient(profile_altitude_values, profile_s)))
    profile_margin = profile.maximum_specific_excess_force_mps2 - profile_demand
    alpha_peak = int(np.argmax(realized_alpha_deg))
    limiter_alpha_overlap = realized_alpha_outside & result.thrust_limited
    alpha_excursion_indices = np.flatnonzero(realized_alpha_outside)
    post_startup = trajectory.time_s > 30.0
    metrics = {
        "profile_minimum_energy_margin_mps2": float(np.min(profile_margin)),
        "rollout_minimum_planned_energy_margin_mps2": float(
            np.min(available_energy - planned_energy_demand)
        ),
        "rollout_minimum_commanded_energy_margin_mps2": float(
            np.min(available_energy - commanded_energy_demand)
        ),
        "thrust_limited_fraction": float(np.mean(result.thrust_limited)),
        "capacity_saturated_fraction": float(np.mean(capacity_saturated)),
        "reported_limiter_without_capacity_saturation_fraction": float(
            np.mean(result.thrust_limited & ~capacity_saturated)
        ),
        "planned_effective_alpha_maximum_deg": float(np.max(planned_alpha_deg)),
        "steady_actual_state_effective_alpha_maximum_deg": float(
            np.max(steady_alpha_deg)
        ),
        "controller_target_effective_alpha_maximum_deg": float(
            np.max(controller_target_alpha_deg)
        ),
        "realized_effective_alpha_maximum_deg": float(np.max(realized_alpha_deg)),
        "planned_alpha_above_support_fraction": float(np.mean(planned_alpha_outside)),
        "controller_target_alpha_above_support_fraction": float(
            np.mean(target_alpha_outside)
        ),
        "realized_alpha_above_support_fraction": float(np.mean(realized_alpha_outside)),
        "realized_alpha_above_support_and_thrust_limited_fraction": float(
            np.mean(limiter_alpha_overlap)
        ),
        "alpha_excursion_samples_with_thrust_limiter_fraction": float(
            np.mean(result.thrust_limited[realized_alpha_outside])
            if np.any(realized_alpha_outside)
            else 0.0
        ),
        "alpha_excursion_first_time_s": float(
            trajectory.time_s[alpha_excursion_indices[0]]
            if len(alpha_excursion_indices)
            else 0.0
        ),
        "alpha_excursion_last_time_s": float(
            trajectory.time_s[alpha_excursion_indices[-1]]
            if len(alpha_excursion_indices)
            else 0.0
        ),
        "alpha_excursion_sample_count": len(alpha_excursion_indices),
        "realized_effective_alpha_maximum_after_30s_deg": float(
            np.max(realized_alpha_deg[post_startup])
        ),
        "alpha_peak": {
            "elapsed_time_s": float(trajectory.time_s[alpha_peak]),
            "segment": f"A{segment_index[alpha_peak]}-A{segment_index[alpha_peak] + 1}",
            "planned_effective_alpha_deg": float(planned_alpha_deg[alpha_peak]),
            "steady_actual_state_effective_alpha_deg": float(
                steady_alpha_deg[alpha_peak]
            ),
            "controller_target_effective_alpha_deg": float(
                controller_target_alpha_deg[alpha_peak]
            ),
            "realized_effective_alpha_deg": float(realized_alpha_deg[alpha_peak]),
            "altitude_error_m": float(altitude_error[alpha_peak]),
            "true_airspeed_error_mps": float(speed_error[alpha_peak]),
            "altitude_gamma_correction_deg": float(
                np.rad2deg(altitude_gamma_correction[alpha_peak])
            ),
            "gamma_rate_command_deg_s": float(
                np.rad2deg(gamma_rate_command[alpha_peak])
            ),
            "bank_angle_deg": float(np.rad2deg(trajectory.bank_angle_rad[alpha_peak])),
            "thrust_limited": bool(result.thrust_limited[alpha_peak]),
            "capacity_saturated": bool(capacity_saturated[alpha_peak]),
        },
        "planned_alpha_pitch_bound_fraction": float(
            np.mean(planned_alpha_pitch_limited)
        ),
        "controller_target_alpha_pitch_bound_fraction": float(
            np.mean(controller_alpha_pitch_limited)
        ),
    }
    return history, segments, metrics


def _document(profile: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    planner_alpha_within_support = (
        metrics["planned_effective_alpha_maximum_deg"] <= ALPHA_SUPPORT_MAX_DEG
        and metrics["planned_alpha_pitch_bound_fraction"] == 0.0
    )
    target_alpha_exceeds_support = (
        metrics["controller_target_effective_alpha_maximum_deg"] > ALPHA_SUPPORT_MAX_DEG
    )
    startup_only_alpha_excursion = (
        target_alpha_exceeds_support
        and metrics["alpha_excursion_first_time_s"] == 0.0
        and metrics["alpha_excursion_last_time_s"] <= 5.0
        and metrics["realized_effective_alpha_maximum_after_30s_deg"]
        <= ALPHA_SUPPORT_MAX_DEG
        and metrics["alpha_excursion_samples_with_thrust_limiter_fraction"] == 0.0
    )
    if planner_alpha_within_support and startup_only_alpha_excursion:
        alpha_origin = (
            "closed-loop initialization transient: zero initial gamma is stepped "
            "toward an already-climbing look-ahead profile"
        )
        next_alpha_gate = (
            "initialize gamma from the planned profile and seed a trimmed first "
            "control, then rerun with the 8.5 deg boundary unchanged"
        )
    elif planner_alpha_within_support and target_alpha_exceeds_support:
        alpha_origin = "closed-loop vertical tracking demand, not the planned profile"
        next_alpha_gate = (
            "add an explicit effective-alpha command guard and audit vertical-loop "
            "anti-windup without changing the 8.5 deg boundary"
        )
    elif not planner_alpha_within_support:
        alpha_origin = "planned profile lift demand"
        next_alpha_gate = (
            "add the 8.5 deg effective-alpha constraint to profile feasibility before "
            "controller changes"
        )
    else:
        alpha_origin = "command realization or lift-model mapping"
        next_alpha_gate = (
            "audit pitch-rate realization and the effective-angle mapping before tuning"
        )
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "diagnostic-complete-full-intent-remains-held",
        "source": {
            "recording": "TAIL-A/FLT0004",
            "already_consumed": True,
            "model_controller_threshold_or_gate_refitted": False,
            "intermediate_qar_response_used_by_guidance": False,
            "raw_coordinates_persisted": False,
        },
        "methods": {
            "planned_alpha": "quasi-steady wings-level lift at the energy-shaped profile state and anchor-interpolated planning mass",
            "steady_actual_state_alpha": "quasi-steady lift at realized q, mass, gamma, and bank; zero commanded gamma acceleration",
            "controller_target_alpha": "inverse lift target including altitude/gamma feedback before pitch realization",
            "reported_alpha": "calibrated effective-angle coordinate used by the guarded hybrid lift model; not a released physical AOA calibration",
            "planned_energy": "profile dV/dt + g sin(profile gamma)",
            "commanded_energy": "profile dV/dt + speed feedback + g sin(realized gamma)",
            "available_energy": "(demonstrated-capacity net thrust - realized polar drag) / mass",
            "realized_energy": "(realized net thrust - realized polar drag) / mass",
            "alpha_support_maximum_deg": ALPHA_SUPPORT_MAX_DEG,
        },
        "profile": {
            "feasible": bool(profile.feasible),
            "violations": list(profile.violations),
        },
        "metrics": metrics,
        "diagnosis": {
            "planner_alpha_within_support": planner_alpha_within_support,
            "controller_target_alpha_exceeds_support": target_alpha_exceeds_support,
            "startup_only_alpha_excursion": startup_only_alpha_excursion,
            "alpha_excursion_origin": alpha_origin,
            "energy_deficit_present_before_closed_loop": not profile.feasible,
            "energy_interpretation": (
                "The sparse-anchor profile retains a small pre-controller capacity "
                "deficit. Closed-loop speed and altitude/gamma corrections add demand "
                "while the demonstrated thrust boundary is active; pitch can trade "
                "kinetic and potential energy but cannot create total energy."
            ),
            "next_alpha_gate": next_alpha_gate,
            "next_energy_gate": (
                "after the initialization correction, add saturation-aware total-energy "
                "priority between altitude and speed tracking"
            ),
            "full_sparse_intent_released": False,
        },
        "artifacts": {
            "history": "docs/fms/a320-fms-alpha-energy-authority-history.csv",
            "segments": "docs/fms/a320-fms-alpha-energy-authority-segments.csv",
            "figure": "docs/fms/figures/a320-fms-alpha-energy-authority.png",
            "report": "docs/fms/a320-fms-alpha-energy-authority.md",
            "code": "tools/audit_a320_fms_alpha_energy_authority.py",
        },
    }


def _style_axis(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _shade_limiter(axis: Any, time_min: np.ndarray, limited: np.ndarray) -> None:
    axis.fill_between(
        time_min,
        0.0,
        1.0,
        where=limited,
        transform=axis.get_xaxis_transform(),
        color="#E69F00",
        alpha=0.12,
        linewidth=0.0,
        label="thrust limiter active",
    )


def _plot(history: pd.DataFrame, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    time_min = history["elapsed_time_s"].to_numpy() / 60.0
    limited = history["thrust_limited"].to_numpy(dtype=bool)
    blue, orange, green, pink, black = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#000000",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)

    axes[0, 0].plot(time_min, history["altitude_error_m"], color=blue, linewidth=1.5)
    axes[0, 0].axhline(0.0, color=black, linewidth=0.8)
    _shade_limiter(axes[0, 0], time_min, limited)
    axes[0, 0].set(xlabel="Elapsed time (min)", ylabel="Altitude error (m)")

    axes[0, 1].plot(
        time_min,
        history["true_airspeed_error_mps"],
        color=green,
        linewidth=1.5,
    )
    axes[0, 1].axhline(0.0, color=black, linewidth=0.8)
    _shade_limiter(axes[0, 1], time_min, limited)
    axes[0, 1].set(xlabel="Elapsed time (min)", ylabel="True airspeed error (m/s)")

    axes[0, 2].plot(
        time_min,
        history["available_energy_mps2"],
        color=black,
        linewidth=1.7,
        label="available",
    )
    axes[0, 2].plot(
        time_min,
        history["planned_energy_demand_mps2"],
        color=blue,
        linestyle="--",
        linewidth=1.4,
        label="planned demand",
    )
    axes[0, 2].plot(
        time_min,
        history["commanded_energy_demand_mps2"],
        color=pink,
        linestyle="-.",
        linewidth=1.3,
        label="with feedback",
    )
    axes[0, 2].plot(
        time_min,
        history["realized_energy_mps2"],
        color=green,
        linewidth=1.2,
        label="realized",
    )
    axes[0, 2].set(xlabel="Elapsed time (min)", ylabel="Specific excess force (m/s²)")
    axes[0, 2].legend(frameon=False, fontsize=7, ncol=2)

    axes[1, 0].plot(
        time_min,
        history["maximum_net_thrust_kn"],
        color=black,
        linestyle="--",
        linewidth=1.6,
        label="demonstrated capacity",
    )
    axes[1, 0].plot(
        time_min,
        history["realized_net_thrust_kn"],
        color=orange,
        linewidth=1.5,
        label="realized net thrust",
    )
    _shade_limiter(axes[1, 0], time_min, limited)
    axes[1, 0].set(xlabel="Elapsed time (min)", ylabel="Total net thrust (kN)")
    axes[1, 0].legend(frameon=False, fontsize=7)

    axes[1, 1].plot(
        time_min,
        history["planned_effective_alpha_deg"],
        color=blue,
        linestyle="--",
        linewidth=1.4,
        label="planned quasi-steady",
    )
    axes[1, 1].plot(
        time_min,
        history["steady_actual_state_effective_alpha_deg"],
        color=green,
        linestyle=":",
        linewidth=1.5,
        label="actual-state quasi-steady",
    )
    axes[1, 1].plot(
        time_min,
        history["controller_target_effective_alpha_deg"],
        color=pink,
        linestyle="-.",
        linewidth=1.4,
        label="controller target",
    )
    axes[1, 1].plot(
        time_min,
        history["realized_effective_alpha_deg"],
        color=black,
        linewidth=1.4,
        label="realized",
    )
    axes[1, 1].axhline(
        ALPHA_SUPPORT_MAX_DEG,
        color=orange,
        linestyle="--",
        linewidth=1.3,
        label="validated support",
    )
    axes[1, 1].set(
        xlabel="Elapsed time (min)", ylabel="Effective angle of attack (deg)"
    )
    axes[1, 1].legend(frameon=False, fontsize=6.5, ncol=2)

    axes[1, 2].plot(
        time_min,
        history["pitch_angle_deg"],
        color=blue,
        linewidth=1.4,
        label="pitch",
    )
    axes[1, 2].plot(
        time_min,
        history["realized_flight_path_angle_deg"],
        color=green,
        linestyle="--",
        linewidth=1.4,
        label="flight-path angle",
    )
    axes[1, 2].plot(
        time_min,
        history["altitude_gamma_correction_deg"],
        color=pink,
        linestyle="-.",
        linewidth=1.3,
        label="altitude-loop γ correction",
    )
    axes[1, 2].set(xlabel="Elapsed time (min)", ylabel="Angle (deg)")
    axes[1, 2].legend(frameon=False, fontsize=7)

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 sparse-intent alpha/energy authority audit; controller unchanged",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _report(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    peak = metrics["alpha_peak"]
    diagnosis = document["diagnosis"]
    return f"""# A320 FMS alpha/energy authority audit

Updated: {UPDATED}

Status: **diagnostic complete; full sparse intent remains held**

## Result

The planner and controller failures are related but not identical.

- The energy-shaped sparse profile is already short of the uncertainty-bounded
  demonstrated capacity by {metrics["profile_minimum_energy_margin_mps2"]:.3f}
  m/s² before closed-loop tracking begins.
- The planner's quasi-steady effective alpha peaks at
  {metrics["planned_effective_alpha_maximum_deg"]:.2f}°, inside the frozen
  {ALPHA_SUPPORT_MAX_DEG:.1f}° lift-model support.
- The closed-loop controller target reaches
  {metrics["controller_target_effective_alpha_maximum_deg"]:.2f}°, and the
  realized trajectory reaches {metrics["realized_effective_alpha_maximum_deg"]:.2f}°.
  The alpha excursion therefore originates in **{diagnosis["alpha_excursion_origin"]}**.
- Only {metrics["alpha_excursion_sample_count"]} one-second samples exceed the
  boundary, all from {metrics["alpha_excursion_first_time_s"]:.0f} to
  {metrics["alpha_excursion_last_time_s"]:.0f} s. After 30 s, maximum effective
  alpha is {metrics["realized_effective_alpha_maximum_after_30s_deg"]:.2f}°.
- The thrust limiter is reported for
  {100 * metrics["thrust_limited_fraction"]:.1f}% of the rollout. Of the samples
  above the alpha boundary,
  {100 * metrics["alpha_excursion_samples_with_thrust_limiter_fraction"]:.1f}%
  coincide with the thrust limiter.

At the maximum-alpha sample ({peak["elapsed_time_s"] / 60.0:.2f} min,
{peak["segment"]}), the planned, actual-state quasi-steady, controller-target,
and realized effective alphas are {peak["planned_effective_alpha_deg"]:.2f}°,
{peak["steady_actual_state_effective_alpha_deg"]:.2f}°,
{peak["controller_target_effective_alpha_deg"]:.2f}°, and
{peak["realized_effective_alpha_deg"]:.2f}°. The altitude loop contributes a
{peak["altitude_gamma_correction_deg"]:+.2f}° flight-path-angle correction at
that instant.

## Interpretation

The alpha excursion and energy saturation must not be collapsed into one
failure. Alpha exceeds support only during initialization and has zero overlap
with the thrust limiter. The simulator starts at zero gamma while the first
look-ahead target is already climbing at about 6°, so the vertical controller
requests an immediate gamma acceleration and extra lift.

Separately, pitch changes the split between kinetic and potential energy; it
does not add total flight energy. During the later thrust-limited intervals,
asking for altitude capture and speed capture simultaneously can leave an
unresolved energy error. The current controller has no explicit total-energy
priority or anti-windup state.

"Alpha" here is the calibrated effective-angle coordinate used by the guarded
hybrid lift model. It is useful for enforcing the validated model boundary, but
it is not yet a released physical AOA calibration or a claim that the recorded
A320 approached stall.

## Decision

The next alpha gate is to **{diagnosis["next_alpha_gate"]}**. After that isolated
startup correction, the energy gate is to **{diagnosis["next_energy_gate"]}**.
The demonstrated thrust capacity, drag polar, controller gains, alpha boundary,
and sparse intent were not changed in this audit. The small pre-controller
profile deficit remains a separate planner constraint and the full-flight
intent is not reclassified.

## Data boundary

`FLT0004` was already consumed by earlier aircraft-model validation. Only its
nine frozen anchors enter guidance. No intermediate QAR state or control
response enters the rollout, and no raw coordinates are persisted by this
audit.

## Artifacts

- Node history: `docs/fms/a320-fms-alpha-energy-authority-history.csv`
- Per-leg summary: `docs/fms/a320-fms-alpha-energy-authority-segments.csv`
- Machine-readable result: `docs/fms/a320-fms-alpha-energy-authority.yml`
- Figure: `docs/fms/figures/a320-fms-alpha-energy-authority.{{png,pdf}}`
- Reproducer: `tools/audit_a320_fms_alpha_energy_authority.py`
"""


def main() -> None:
    args = parse_args()
    (
        qar,
        route,
        environment,
        anchor_indices,
        plant,
        thrust_ceiling,
        _,
        profile,
        controller,
        result,
    ) = _build_rollout(args)
    history, segments, metrics = _diagnostics(
        qar,
        route,
        environment,
        anchor_indices,
        plant,
        thrust_ceiling,
        profile,
        controller,
        result,
    )
    document = _document(profile, metrics)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.history, index=False)
    segments.to_csv(args.segments, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(history, args.figure_prefix)
    print(yaml.safe_dump(document["diagnosis"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
