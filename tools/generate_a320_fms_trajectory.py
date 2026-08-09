#!/usr/bin/env python3
"""Generate and gate the first intent-driven Aerox A320 FMS trajectory."""

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
    FlightIntent,
    FMSTrajectoryResult,
    SpatialPathGuidance,
    SpatialTrajectory,
    Waypoint,
    simulate_fms_trajectory_zoh,
)
from aerox.models.atmosphere import atmosphere_at_pressure_altitude, isa_atmosphere
from tools.generate_a320_comparison_trajectories import _build_plants, _controller

UPDATED = "2026-08-06"
WIND_NORTH_MPS = 18.0
WIND_EAST_MPS = -14.0
INITIAL_MASS_KG = 65_000.0
MAXIMUM_TIME_S = 1_250.0
COMMAND_STEP_S = 1.0
INTEGRATION_RATE_HZ = 10

# Frozen before running this example. These are engineering-demonstration
# thresholds, not Airbus or operational limits.
GATE_LIMITS = {
    "final_horizontal_error_m": 1_500.0,
    "cross_track_rmse_m": 1_500.0,
    "final_altitude_error_m": 50.0,
    "final_speed_error_mps": 2.0,
    "intermediate_waypoint_distance_m": 4_000.0,
    "destination_waypoint_distance_m": 1_500.0,
    "minimum_pitch_deg": -2.0,
    "maximum_pitch_deg": 12.0,
    "maximum_absolute_bank_deg": 25.0,
    "minimum_alpha_deg": -1.0,
    "maximum_alpha_deg": 8.5,
    "minimum_mach": 0.20,
    "maximum_mach": 0.82,
    "maximum_limited_fraction": 0.05,
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
        "--samples",
        type=Path,
        default=Path("docs/fms/a320-fms-trajectory.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-trajectory.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-trajectory.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-trajectory-comparison"),
    )
    parser.add_argument(
        "--control-figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-control-history"),
    )
    return parser.parse_args()


def build_intent() -> FlightIntent:
    """Return the frozen local clean-flight waypoint demonstration."""

    return FlightIntent(
        waypoints=(
            Waypoint(0.0, 0.0, 9_500.0, 215.0, "ORIGIN"),
            Waypoint(50_000.0, 0.0, 9_700.0, 220.0, "CLIMB-1"),
            Waypoint(90_000.0, 35_000.0, 10_100.0, 220.0, "CLIMB-2"),
            Waypoint(140_000.0, 20_000.0, 9_900.0, 212.0, "DESCENT-1"),
            Waypoint(
                185_000.0,
                65_000.0,
                9_500.0,
                205.0,
                "DESTINATION",
                acceptance_radius_m=1_500.0,
            ),
        ),
        maximum_climb_flight_path_angle_rad=math.radians(4.0),
        maximum_descent_flight_path_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )


def _run(
    route: SpatialTrajectory,
    plant: Any,
    *,
    position_feedback: bool,
    wind_correction: bool,
) -> FMSTrajectoryResult:
    guidance = SpatialPathGuidance(
        route,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
        command_step_s=COMMAND_STEP_S,
        position_feedback=position_feedback,
        wind_correction=wind_correction,
    )
    temperature = float(
        isa_atmosphere(np.array([route.intent.waypoints[0].altitude_m])).temperature_k[
            0
        ]
    )
    return simulate_fms_trajectory_zoh(
        plant=plant,
        controller=_controller(),
        guidance=guidance,
        maximum_time_s=MAXIMUM_TIME_S,
        static_temperature_k=temperature,
        initial_mass_kg=INITIAL_MASS_KG,
        wind_north_mps=WIND_NORTH_MPS,
        wind_east_mps=WIND_EAST_MPS,
        command_step_s=COMMAND_STEP_S,
        integration_rate_hz=INTEGRATION_RATE_HZ,
        destination_capture_radius_m=1_500.0,
    )


def _mach(result: FMSTrajectoryResult) -> np.ndarray:
    trajectory = result.trajectory
    temperature = float(
        isa_atmosphere(
            np.array([result.planned_trajectory.intent.waypoints[0].altitude_m])
        ).temperature_k[0]
    )
    atmosphere = atmosphere_at_pressure_altitude(
        trajectory.pressure_altitude_m,
        np.full_like(trajectory.time_s, temperature),
    )
    return trajectory.true_airspeed_mps / atmosphere.speed_of_sound_mps


def _metrics(result: FMSTrajectoryResult) -> dict[str, Any]:
    route = result.planned_trajectory
    destination = route.intent.waypoints[-1]
    trajectory = result.trajectory
    waypoint_distance = result.waypoint_minimum_distance_m()
    mach = _mach(result)
    return {
        "completed": result.completed,
        "duration_s": float(trajectory.time_s[-1]),
        "route_distance_m": route.total_distance_m,
        "final_horizontal_error_m": float(
            np.hypot(
                trajectory.north_position_m[-1] - destination.north_m,
                trajectory.east_position_m[-1] - destination.east_m,
            )
        ),
        "cross_track_rmse_m": float(np.sqrt(np.mean(result.cross_track_error_m**2))),
        "cross_track_p95_m": float(
            np.percentile(np.abs(result.cross_track_error_m), 95.0)
        ),
        "final_altitude_error_m": float(
            trajectory.pressure_altitude_m[-1] - destination.altitude_m
        ),
        "final_speed_error_mps": float(
            trajectory.true_airspeed_mps[-1] - destination.true_airspeed_mps
        ),
        "waypoint_minimum_distance_m": {
            point.name: float(value)
            for point, value in zip(
                route.intent.waypoints, waypoint_distance, strict=True
            )
        },
        "fuel_burn_kg": float(trajectory.mass_kg[0] - trajectory.mass_kg[-1]),
        "minimum_pitch_deg": float(np.rad2deg(np.min(trajectory.pitch_angle_rad))),
        "maximum_pitch_deg": float(np.rad2deg(np.max(trajectory.pitch_angle_rad))),
        "maximum_absolute_bank_deg": float(
            np.rad2deg(np.max(np.abs(trajectory.bank_angle_rad)))
        ),
        "minimum_alpha_deg": float(np.rad2deg(np.min(trajectory.angle_of_attack_rad))),
        "maximum_alpha_deg": float(np.rad2deg(np.max(trajectory.angle_of_attack_rad))),
        "minimum_mach": float(np.min(mach)),
        "maximum_mach": float(np.max(mach)),
        "pitch_limited_fraction": float(np.mean(result.pitch_limited)),
        "bank_limited_fraction": float(np.mean(result.bank_limited)),
        "thrust_limited_fraction": float(np.mean(result.thrust_limited)),
    }


def _gates(fms: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    intermediate_distances = list(fms["waypoint_minimum_distance_m"].values())[1:-1]
    destination_distance = list(fms["waypoint_minimum_distance_m"].values())[-1]
    checks = {
        "destination_completed": (bool(fms["completed"]), True),
        "final_horizontal_error": (
            fms["final_horizontal_error_m"] <= GATE_LIMITS["final_horizontal_error_m"],
            GATE_LIMITS["final_horizontal_error_m"],
        ),
        "cross_track_rmse": (
            fms["cross_track_rmse_m"] <= GATE_LIMITS["cross_track_rmse_m"],
            GATE_LIMITS["cross_track_rmse_m"],
        ),
        "final_altitude_error": (
            abs(fms["final_altitude_error_m"]) <= GATE_LIMITS["final_altitude_error_m"],
            GATE_LIMITS["final_altitude_error_m"],
        ),
        "final_speed_error": (
            abs(fms["final_speed_error_mps"]) <= GATE_LIMITS["final_speed_error_mps"],
            GATE_LIMITS["final_speed_error_mps"],
        ),
        "intermediate_waypoints": (
            max(intermediate_distances)
            <= GATE_LIMITS["intermediate_waypoint_distance_m"],
            GATE_LIMITS["intermediate_waypoint_distance_m"],
        ),
        "destination_waypoint": (
            destination_distance <= GATE_LIMITS["destination_waypoint_distance_m"],
            GATE_LIMITS["destination_waypoint_distance_m"],
        ),
        "pitch_envelope": (
            fms["minimum_pitch_deg"] >= GATE_LIMITS["minimum_pitch_deg"]
            and fms["maximum_pitch_deg"] <= GATE_LIMITS["maximum_pitch_deg"],
            [GATE_LIMITS["minimum_pitch_deg"], GATE_LIMITS["maximum_pitch_deg"]],
        ),
        "bank_envelope": (
            fms["maximum_absolute_bank_deg"]
            <= GATE_LIMITS["maximum_absolute_bank_deg"],
            GATE_LIMITS["maximum_absolute_bank_deg"],
        ),
        "alpha_envelope": (
            fms["minimum_alpha_deg"] >= GATE_LIMITS["minimum_alpha_deg"]
            and fms["maximum_alpha_deg"] <= GATE_LIMITS["maximum_alpha_deg"],
            [GATE_LIMITS["minimum_alpha_deg"], GATE_LIMITS["maximum_alpha_deg"]],
        ),
        "mach_envelope": (
            fms["minimum_mach"] >= GATE_LIMITS["minimum_mach"]
            and fms["maximum_mach"] <= GATE_LIMITS["maximum_mach"],
            [GATE_LIMITS["minimum_mach"], GATE_LIMITS["maximum_mach"]],
        ),
        "controller_limiters": (
            max(
                fms["pitch_limited_fraction"],
                fms["bank_limited_fraction"],
                fms["thrust_limited_fraction"],
            )
            <= GATE_LIMITS["maximum_limited_fraction"],
            GATE_LIMITS["maximum_limited_fraction"],
        ),
        "improves_over_leg_bearing_baseline": (
            fms["cross_track_rmse_m"] < baseline["cross_track_rmse_m"],
            baseline["cross_track_rmse_m"],
        ),
    }
    return {
        name: {"passed": bool(value[0]), "limit_or_comparison": value[1]}
        for name, value in checks.items()
    }


def _table(result: FMSTrajectoryResult, scenario: str) -> pd.DataFrame:
    trajectory = result.trajectory
    return pd.DataFrame(
        {
            "scenario": scenario,
            "time_s": trajectory.time_s,
            "north_m": trajectory.north_position_m,
            "east_m": trajectory.east_position_m,
            "altitude_m": trajectory.pressure_altitude_m,
            "true_airspeed_mps": trajectory.true_airspeed_mps,
            "flight_path_angle_deg": np.rad2deg(trajectory.flight_path_angle_rad),
            "heading_deg": np.rad2deg(trajectory.heading_rad),
            "mass_kg": trajectory.mass_kg,
            "pitch_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_deg": np.rad2deg(trajectory.bank_angle_rad),
            "alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "net_thrust_n": trajectory.net_thrust_n,
            "fuel_flow_kg_s": trajectory.fuel_flow_kg_s,
            "progress_m": result.progress_m,
            "cross_track_error_m": result.cross_track_error_m,
            "target_north_m": result.target_north_m,
            "target_east_m": result.target_east_m,
            "reference_altitude_m": result.reference_altitude_m,
            "reference_true_airspeed_mps": result.reference_true_airspeed_mps,
            "reference_ground_track_deg": np.rad2deg(result.reference_ground_track_rad),
            "reference_air_heading_deg": np.rad2deg(result.reference_air_heading_rad),
            "pitch_limited": result.pitch_limited,
            "bank_limited": result.bank_limited,
            "thrust_limited": result.thrust_limited,
        }
    )


def _style_axis(axis: Any, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_comparison(
    fms: FMSTrajectoryResult,
    baseline: FMSTrajectoryResult,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    along, north, east, altitude, speed, _ = fms.planned_trajectory.sampled(250.0)
    colors = {"plan": "#000000", "fms": "#0072B2", "baseline": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    axes[0, 0].plot(
        east / 1_000.0,
        north / 1_000.0,
        color=colors["plan"],
        linestyle=":",
        linewidth=2.0,
        label="spatial plan",
    )
    for result, key, label, style in (
        (fms, "fms", "FMS position + wind feedback", "-"),
        (baseline, "baseline", "leg-bearing baseline", "--"),
    ):
        axes[0, 0].plot(
            result.trajectory.east_position_m / 1_000.0,
            result.trajectory.north_position_m / 1_000.0,
            color=colors[key],
            linestyle=style,
            linewidth=1.8,
            label=label,
        )
    points = fms.planned_trajectory.intent.waypoints
    axes[0, 0].scatter(
        [point.east_m / 1_000.0 for point in points],
        [point.north_m / 1_000.0 for point in points],
        marker="o",
        facecolors="white",
        edgecolors="black",
        s=34,
        zorder=5,
    )
    for point in points:
        axes[0, 0].annotate(
            point.name,
            (point.east_m / 1_000.0, point.north_m / 1_000.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0, 0].quiver(
        58.0,
        105.0,
        WIND_EAST_MPS,
        WIND_NORTH_MPS,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#009E73",
        width=0.004,
    )
    axes[0, 0].text(50.0, 109.0, "wind", color="#009E73", fontsize=8)
    axes[0, 0].set(xlabel="East position (km)", ylabel="North position (km)")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")

    for result, key, label, style in (
        (fms, "fms", "FMS", "-"),
        (baseline, "baseline", "baseline", "--"),
    ):
        time_min = result.trajectory.time_s / 60.0
        axes[0, 1].plot(
            time_min,
            result.cross_track_error_m,
            color=colors[key],
            linestyle=style,
            linewidth=1.7,
            label=label,
        )
        axes[0, 2].plot(
            time_min,
            result.trajectory.pressure_altitude_m,
            color=colors[key],
            linestyle=style,
            linewidth=1.7,
            label=label,
        )
        axes[1, 0].plot(
            time_min,
            result.trajectory.true_airspeed_mps,
            color=colors[key],
            linestyle=style,
            linewidth=1.7,
            label=label,
        )
    axes[0, 2].plot(
        fms.trajectory.time_s / 60.0,
        fms.reference_altitude_m,
        color=colors["plan"],
        linestyle=":",
        linewidth=1.5,
        label="online reference",
    )
    axes[1, 0].plot(
        fms.trajectory.time_s / 60.0,
        fms.reference_true_airspeed_mps,
        color=colors["plan"],
        linestyle=":",
        linewidth=1.5,
        label="online reference",
    )
    axes[0, 1].axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    axes[0, 1].set(xlabel="Time (min)", ylabel="Signed cross-track error (m)")
    axes[0, 2].set(xlabel="Time (min)", ylabel="Pressure altitude (m)")
    axes[1, 0].set(xlabel="Time (min)", ylabel="True airspeed (m/s)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(
        along / 1_000.0,
        altitude,
        color="#009E73",
        linewidth=1.8,
        label="altitude constraint profile",
    )
    right = axes[1, 1].twinx()
    right.plot(
        along / 1_000.0,
        speed,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.8,
        label="speed constraint profile",
    )
    axes[1, 1].set(xlabel="Along-path distance (km)", ylabel="Altitude (m)")
    right.set_ylabel("True airspeed (m/s)")
    lines = axes[1, 1].lines + right.lines
    axes[1, 1].legend(
        lines, [line.get_label() for line in lines], frameon=False, fontsize=7
    )

    correction = np.rad2deg(
        np.arctan2(
            np.sin(fms.reference_air_heading_rad - fms.reference_ground_track_rad),
            np.cos(fms.reference_air_heading_rad - fms.reference_ground_track_rad),
        )
    )
    axes[1, 2].plot(
        fms.trajectory.time_s / 60.0,
        correction,
        color="#009E73",
        linewidth=1.7,
    )
    axes[1, 2].axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
    axes[1, 2].set(xlabel="Time (min)", ylabel="Heading − ground track (deg)")

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "Aerox FMS: intent-driven A320 trajectory in constant wind", fontsize=13
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_controls(fms: FMSTrajectoryResult, prefix: Path) -> None:
    import matplotlib.pyplot as plt

    trajectory = fms.trajectory
    time_min = trajectory.time_s / 60.0
    blue, orange, green, purple = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.5), constrained_layout=True)
    axes[0, 0].plot(
        time_min,
        np.rad2deg(trajectory.pitch_angle_rad),
        color=blue,
        linewidth=1.7,
        label="pitch",
    )
    axes[0, 0].plot(
        time_min,
        np.rad2deg(trajectory.flight_path_angle_rad),
        color=green,
        linestyle="--",
        linewidth=1.4,
        label="flight-path angle",
    )
    axes[0, 0].axhspan(-2.0, 12.0, color="0.85", alpha=0.35)
    axes[0, 0].set(xlabel="Time (min)", ylabel="Angle (deg)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(
        time_min, np.rad2deg(trajectory.bank_angle_rad), color=orange, linewidth=1.7
    )
    axes[0, 1].axhline(-25.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[0, 1].axhline(25.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[0, 1].set(xlabel="Time (min)", ylabel="Bank angle (deg)")
    axes[1, 0].plot(
        time_min, trajectory.net_thrust_n / 1_000.0, color=purple, linewidth=1.7
    )
    axes[1, 0].axhline(200.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[1, 0].set(xlabel="Time (min)", ylabel="Realized net thrust (kN)")
    axes[1, 1].plot(
        time_min,
        np.rad2deg(trajectory.angle_of_attack_rad),
        color=green,
        linewidth=1.7,
        label="effective angle of attack",
    )
    axes[1, 1].axhline(-1.0, color="0.35", linestyle=":", linewidth=0.9)
    axes[1, 1].axhline(8.5, color="0.35", linestyle=":", linewidth=0.9)
    axes[1, 1].set(xlabel="Time (min)", ylabel="Effective angle of attack (deg)")
    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle("A320 controls generated by spatial FMS guidance", fontsize=13)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _report(document: dict[str, Any]) -> str:
    fms = document["metrics"]["fms_position_and_wind_feedback"]
    baseline = document["metrics"]["leg_bearing_no_wind_correction"]
    status = (
        "passed" if document["decision"]["all_predeclared_gates_passed"] else "held"
    )
    waypoint_rows = "\n".join(
        f"| {name} | {distance:.1f} |"
        for name, distance in fms["waypoint_minimum_distance_m"].items()
    )
    gate_rows = "\n".join(
        f"| {name} | {'pass' if gate['passed'] else 'hold'} | {gate['limit_or_comparison']} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 Aerox FMS intent-driven trajectory

Updated: {UPDATED}

Status: **{status}**

## Result

The deterministic Aerox FMS converted five local waypoints with altitude and
true-airspeed constraints into online references and generated the trajectory
through the released A320 pitch–bank–realized-net-thrust plant. In the declared
constant wind, the full position-feedback and wind-correction case had
{fms["cross_track_rmse_m"]:.1f} m cross-track RMSE versus
{baseline["cross_track_rmse_m"]:.1f} m for the route-leg-bearing baseline.

The FMS case {"captured" if fms["completed"] else "did not capture"} the
destination after {fms["duration_s"]:.0f} s, with terminal horizontal,
altitude, and speed errors of {fms["final_horizontal_error_m"]:.1f} m,
{fms["final_altitude_error_m"]:.1f} m, and
{fms["final_speed_error_mps"]:.2f} m/s. It burned {fms["fuel_burn_kg"]:.1f} kg
in this modeled scenario.

## How intent becomes a trajectory

The waypoint route is parameterized by cumulative horizontal distance `s`.
At each 1 Hz guidance update, current north/east position is projected onto the
active route leg and progress is only allowed to move forward. A 2.5–10 km
speed-scaled look-ahead point may lie beyond the next waypoint, so turns begin
before overflight. The bearing from the current position to that point is the
target ground track. Solving the wind triangle converts ground track to the
air-relative heading required by the plant. The air-heading command is limited
to 1 degree/s so a polyline corner cannot create an impulsive bank request.

Altitude and true airspeed are interpolated at the same spatial target. Route
altitude gradient supplies flight-path-angle feed-forward and route speed
gradient supplies acceleration feed-forward. The existing controller closes
heading, flight-path-angle, altitude, and speed error, inverts the lift/force
model, and emits realized pitch, bank, and total net thrust. Those controls are
held for one second while the physical plant is integrated at 10 Hz. The next
reference therefore depends on the generated position; this is online
closed-loop spatial guidance, not a time-indexed trajectory replay.

## Waypoint approaches

| Waypoint | Closest horizontal distance (m) |
|---|---:|
{waypoint_rows}

Intermediate nodes use a 4,000 m fly-by acceptance radius; the destination
uses 1,500 m. The polyline remains the reference centerline. Look-ahead cuts
corners continuously but is not yet an exact radius-constrained tangent-arc
construction.

## Predeclared gates

| Check | Result | Limit or comparator |
|---|---|---|
{gate_rows}

## Interpretation boundary

The route, constraints, wind vector, and controller parameters are synthetic
engineering inputs. The controller is not an identified Airbus avionics law.
Pitch and bank are realized attitudes and net thrust is realized total
propulsive force; actuator allocation, TLA/N1/N2 scheduling, FADEC logic,
sideslip, rotational dynamics, weather variation, geodetic legs, exact fly-by
turn geometry, and vertical feasibility propagation remain outside this gate.

OpenTOP was used only as a read-only design reference for trajectory schemas,
projected coordinates, wind-aware dynamics, and optimizer boundaries. No
OpenTOP code or runtime dependency is used here.

## Artifacts

- Samples: `docs/fms/a320-fms-trajectory.csv`
- Machine-readable result: `docs/fms/a320-fms-trajectory.yml`
- Comparison figure: `docs/fms/figures/a320-fms-trajectory-comparison.{{png,pdf}}`
- Control figure: `docs/fms/figures/a320-fms-control-history.{{png,pdf}}`
- Reproducer: `tools/generate_a320_fms_trajectory.py`
"""


def main() -> None:
    args = parse_args()
    route = SpatialTrajectory.from_intent(build_intent())
    plant, _, _ = _build_plants(args)
    fms = _run(route, plant, position_feedback=True, wind_correction=True)
    baseline = _run(route, plant, position_feedback=False, wind_correction=False)
    fms_metrics = _metrics(fms)
    baseline_metrics = _metrics(baseline)
    gates = _gates(fms_metrics, baseline_metrics)
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 pooled type-level model",
        "scenario": {
            "type": "synthetic local waypoint intent",
            "wind_north_mps": WIND_NORTH_MPS,
            "wind_east_mps": WIND_EAST_MPS,
            "initial_mass_kg": INITIAL_MASS_KG,
            "command_rate_hz": round(1.0 / COMMAND_STEP_S),
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "waypoints": [
                {
                    "name": point.name,
                    "north_m": point.north_m,
                    "east_m": point.east_m,
                    "altitude_m": point.altitude_m,
                    "true_airspeed_mps": point.true_airspeed_mps,
                    "acceptance_radius_m": point.acceptance_radius_m,
                }
                for point in route.intent.waypoints
            ],
        },
        "fms": {
            "spatial_coordinate": "cumulative horizontal along-path distance s",
            "route_geometry": "piecewise-linear centerline with online look-ahead fly-by anticipation",
            "lookahead_time_s": 45.0,
            "lookahead_bounds_m": [2_500.0, 10_000.0],
            "maximum_heading_command_rate_deg_s": 1.0,
            "wind_correction": "analytic wind triangle",
            "position_feedback": "bearing from actual position to spatial look-ahead target",
            "neural_network": False,
        },
        "metrics": {
            "fms_position_and_wind_feedback": fms_metrics,
            "leg_bearing_no_wind_correction": baseline_metrics,
        },
        "gates": gates,
        "decision": {
            "all_predeclared_gates_passed": all(
                gate["passed"] for gate in gates.values()
            ),
            "local_deterministic_fms_milestone_released": all(
                gate["passed"] for gate in gates.values()
            ),
            "airbus_fms_or_autopilot_identified": False,
        },
        "artifacts": {
            "samples": str(args.samples),
            "report": str(args.report),
            "comparison_figure": str(args.figure_prefix.with_suffix(".png")),
            "control_figure": str(args.control_figure_prefix.with_suffix(".png")),
            "code": "tools/generate_a320_fms_trajectory.py",
        },
    }
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [_table(fms, "fms"), _table(baseline, "leg_bearing_baseline")],
        ignore_index=True,
    ).to_csv(args.samples, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot_comparison(fms, baseline, args.figure_prefix)
    _plot_controls(fms, args.control_figure_prefix)
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(document["metrics"], sort_keys=False).strip())
    if not document["decision"]["all_predeclared_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
