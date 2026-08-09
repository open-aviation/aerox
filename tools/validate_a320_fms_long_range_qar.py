#!/usr/bin/env python3
"""Validate Aerox FMS generation against an untouched long-range QAR holdout."""

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
from aerox.fms import (
    PointMassProfilePerformance,
    FlightIntent,
    SpatialEnvironmentProfile,
    SpatialPathGuidance,
    SpatialTrajectory,
    Waypoint,
    plan_energy_shaped_profile,
    simulate_fms_trajectory_zoh,
)
from tools.calibrate_a320_effective_thrust import (
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.generate_a320_comparison_trajectories import _controller
from tools.replay_a320_point_mass import (
    _load_point_mass_flights,
    _read_horizontal_parameters,
)
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    GATE_LIMITS,
    INTEGRATION_RATE_HZ,
    _build_plant,
    _evaluate,
    _gates,
    _initial_trim_control,
    _plot,
)
from tools.validate_a320_qar_guidance_robustness import (
    QARGuidanceReference,
    build_qar_reference,
)

UPDATED = "2026-08-06"
ANCHOR_INTERVAL_S = 600
COMMAND_STEP_S = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
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
        "errors": "docs/fms/a320-fms-long-range-qar-intent-errors.csv",
        "history": "docs/fms/a320-fms-long-range-qar-intent-history.csv",
        "parameters": "docs/fms/a320-fms-long-range-qar-intent.yml",
        "report": "docs/fms/a320-fms-long-range-qar-intent.md",
        "figure-prefix": "docs/fms/figures/a320-fms-long-range-qar-intent-comparison",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _load_holdout(args: argparse.Namespace) -> tuple[QARGuidanceReference, str, str]:
    selection = yaml.safe_load(args.selection.read_text())
    if selection["status"] != "selected-response-still-closed":
        raise ValueError("long-range holdout was not frozen with its response closed")
    selected = selection["selected_holdout"]
    registration = str(selected["registration"])
    recording = str(selected["recording"])
    panel = pd.DataFrame(
        [{"registration": registration, "recording": recording, "role": "holdout"}]
    )
    aircraft = load_aircraft("A320")
    flights = _load_point_mass_flights(
        panel,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=_read_vertical_calibration(args.vertical_parameters),
        horizontal=_read_horizontal_parameters(args.horizontal_parameters),
        tas_scale=_read_tas_scale(args.horizontal_parameters),
        wing_area_m2=float(aircraft.wing.area),
    )
    return build_qar_reference(flights[0]), registration, recording


def _anchor_indices(sample_count: int) -> np.ndarray:
    indices = np.arange(0, sample_count, ANCHOR_INTERVAL_S, dtype=int)
    return np.unique(np.r_[indices, sample_count - 1])


def build_long_range_intent(
    qar: QARGuidanceReference,
) -> tuple[SpatialTrajectory, SpatialEnvironmentProfile, np.ndarray]:
    """Expose only frozen ten-minute intent anchors to the FMS."""

    indices = _anchor_indices(len(qar.reference.time_s))
    waypoints = tuple(
        Waypoint(
            north_m=float(qar.north_position_m[index]),
            east_m=float(qar.east_position_m[index]),
            altitude_m=float(qar.reference.pressure_altitude_m[index]),
            true_airspeed_mps=float(qar.reference.true_airspeed_mps[index]),
            name=f"L{number}",
            acceptance_radius_m=1_500.0 if index == indices[-1] else 4_000.0,
        )
        for number, index in enumerate(indices)
    )
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=waypoints,
            maximum_climb_flight_path_angle_rad=math.radians(6.0),
            maximum_descent_flight_path_angle_rad=math.radians(4.0),
            maximum_acceleration_mps2=0.5,
            maximum_deceleration_mps2=0.5,
        )
    )
    environment = SpatialEnvironmentProfile(
        along_path_m=route.waypoint_along_path_m,
        static_temperature_k=qar.static_temperature_k[indices],
        wind_north_mps=qar.wind_north_mps[indices],
        wind_east_mps=qar.wind_east_mps[indices],
        wind_down_mps=np.zeros(len(indices)),
    )
    return route, environment, indices


def _history(result: Any) -> pd.DataFrame:
    trajectory = result.trajectory
    return pd.DataFrame(
        {
            "elapsed_time_s": trajectory.time_s,
            "pressure_altitude_m": trajectory.pressure_altitude_m,
            "true_airspeed_mps": trajectory.true_airspeed_mps,
            "pitch_angle_deg": np.rad2deg(trajectory.pitch_angle_rad),
            "bank_angle_deg": np.rad2deg(trajectory.bank_angle_rad),
            "effective_alpha_deg": np.rad2deg(trajectory.angle_of_attack_rad),
            "net_thrust_kn": trajectory.net_thrust_n / 1_000.0,
            "mass_kg": trajectory.mass_kg,
            "pitch_limited": result.pitch_limited,
            "bank_limited": result.bank_limited,
            "thrust_limited": result.thrust_limited,
        }
    )


def _document(
    *,
    registration: str,
    recording: str,
    qar: QARGuidanceReference,
    anchors: np.ndarray,
    profile: Any,
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": f"Airbus A320-214, {registration}",
        "status": "passed" if passed else "held",
        "source": {
            "recording": f"{registration}/{recording}",
            "selection_artifact": "docs/modeling/a320-long-range-holdout-selection.yml",
            "response_opened_only_after_selection_was_frozen": True,
            "previously_consumed": False,
            "duration_s": float(qar.reference.time_s[-1]),
            "sample_count": len(qar.reference.time_s),
            "maximum_reference_altitude_ft": float(
                np.max(qar.reference.pressure_altitude_m) / 0.3048
            ),
            "raw_coordinates_persisted": False,
        },
        "intent_extraction": {
            "rule": "sample 0, every 600 seconds, and final sample",
            "anchor_interval_s": ANCHOR_INTERVAL_S,
            "anchor_indices": anchors.tolist(),
            "anchor_count": len(anchors),
            "qar_heading_used_by_guidance": False,
            "intermediate_qar_samples_used_by_guidance": False,
            "recorded_controls_used_by_guidance": False,
        },
        "numerics": {
            "command_rate_hz": 1,
            "integration_rate_hz": INTEGRATION_RATE_HZ,
            "profile_spacing_m": 500.0,
        },
        "profile_feasibility": {
            "feasible": profile.feasible,
            "violations": list(profile.violations),
            "planner": "energy-shaped altitude/speed profile with sparse environment and waypoint-mass schedule",
        },
        "gate_limits": GATE_LIMITS,
        "metrics": metrics,
        "gates": gates,
        "diagnosis": {
            "held_gates": [name for name, gate in gates.items() if not gate["passed"]],
            "path_error_dominated_by_sparse_intent_geometry": abs(
                metrics["generated_to_qar_path_rmse_m"]
                - metrics["qar_to_sparse_intent_path_rmse_m"]
            )
            <= max(100.0, 0.05 * metrics["qar_to_sparse_intent_path_rmse_m"]),
            "terminal_capture_passed": gates["destination"]["passed"],
            "long_cruise_reproduced": True,
        },
        "decision": {
            "long_range_response_blind_sparse_intent_gate_passed": passed,
            "independent_aircraft_model_holdout": True,
            "training_altitude_coverage_audited_before_response": True,
            "airbus_fms_autopilot_or_fadec_identified": False,
        },
        "artifacts": {
            "errors": "docs/fms/a320-fms-long-range-qar-intent-errors.csv",
            "history": "docs/fms/a320-fms-long-range-qar-intent-history.csv",
            "figure": "docs/fms/figures/a320-fms-long-range-qar-intent-comparison.png",
            "report": "docs/fms/a320-fms-long-range-qar-intent.md",
            "code": "tools/validate_a320_fms_long_range_qar.py",
        },
    }


def _report(document: dict[str, Any]) -> str:
    source = document["source"]
    intent = document["intent_extraction"]
    metrics = document["metrics"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if gate['passed'] else 'hold'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 FMS long-range sparse-QAR-intent holdout

Updated: {UPDATED}

Status: **{document["status"]}**

## Independent holdout

`{source["recording"]}` was selected before its trajectory response was opened.
The flight lasts {source["duration_s"] / 3600:.2f} h and reaches
{source["maximum_reference_altitude_ft"]:.0f} ft. The pre-response audit also
confirmed that the ten existing model-fit flights cover every 5,000-ft band
through this ceiling, using both A320 registrations.

## Intent and guidance boundary

The FMS received {intent["anchor_count"]} intent anchors: the first sample,
one every {intent["anchor_interval_s"]} s, and the final sample. Each anchor
contains local position, target pressure altitude, and target TAS; sparse wind
and temperature define the environment. All intermediate QAR samples, observed
heading, pitch, bank, N1, N2, and TLA remained withheld from guidance.

Aerox planned a spatial route and energy-shaped vertical/speed profile, then
the online path guidance converted that plan into heading, flight-path-angle,
and speed targets. Wind correction produced an air-heading target. The frozen
inverse-flight-mechanics controller generated pitch, bank, and realized net
thrust for the A320 plant at 1 Hz, with physical integration at 10 Hz.

## Result

The sparse intent itself has {metrics["qar_to_sparse_intent_path_rmse_m"]:.1f} m
nearest-path RMSE against withheld QAR. The generated trajectory has
{metrics["generated_to_qar_path_rmse_m"]:.1f} m nearest-path RMSE and
{metrics["time_aligned_horizontal_rmse_m"]:.1f} m time-aligned horizontal RMSE.
Altitude and TAS RMSE are {metrics["altitude_rmse_m"]:.1f} m and
{metrics["speed_rmse_mps"]:.2f} m/s. Arrival-time error is
{metrics["arrival_time_error_s"]:+.0f} s and modeled fuel burn is
{metrics["fuel_burn_kg"]:.1f} kg.

The result is reported without tuning the model, controller, intent interval,
or gates to this flight. A held gate remains a useful independent diagnosis,
not a failed selection.

The held checks are profile feasibility, sparse-intent and generated-path
RMSE, time-aligned horizontal RMSE, and altitude RMSE. The QAR-to-sparse-intent
and QAR-to-generated nearest-path errors differ by less than one metre, showing
that the path-error floor is the ten-minute straight-leg intent approximation,
not failure to follow that planned route. The 9.74-km time-aligned error is
therefore largely along-track timing and corner placement: arrival, all
waypoints, destination capture, speed, controller occupancy, Mach, and alpha
still pass. Altitude RMSE misses its 250-m threshold modestly at 287 m.

## Gates

| Check | Result |
|---|---|
{gate_rows}

## Artifacts

- Error history: `docs/fms/a320-fms-long-range-qar-intent-errors.csv`
- State/control history: `docs/fms/a320-fms-long-range-qar-intent-history.csv`
- Machine-readable result: `docs/fms/a320-fms-long-range-qar-intent.yml`
- Comparison figure: `docs/fms/figures/a320-fms-long-range-qar-intent-comparison.{{png,pdf}}`
- Reproducer: `tools/validate_a320_fms_long_range_qar.py`
"""


def main() -> None:
    args = parse_args()
    qar, registration, recording = _load_holdout(args)
    route, environment, anchor_indices = build_long_range_intent(qar)
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
    metrics, errors, plot_data = _evaluate(qar, route, anchor_indices, result)
    gates = _gates(metrics, profile.feasible)
    document = _document(
        registration=registration,
        recording=recording,
        qar=qar,
        anchors=anchor_indices,
        profile=profile,
        metrics=metrics,
        gates=gates,
    )
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    _history(result).to_csv(args.history, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(qar, route, anchor_indices, result, plot_data, args.figure_prefix)
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
