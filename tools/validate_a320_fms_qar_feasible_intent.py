#!/usr/bin/env python3
"""Validate the longest aircraft-feasible sparse-QAR-intent subsection."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.fms import (
    plan_feasible_profile,
)
from aerox.guidance import SpatialPathGuidance, TrajectoryReference
from aerox.modeling import PointMassProfilePerformance
from aerox.simulation import simulate_fms_trajectory_zoh
from tools.generate_a320_comparison_trajectories import _controller
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    INTEGRATION_RATE_HZ,
    QARGuidanceReference,
    _build_plant,
    _evaluate,
    _gates,
    _load_reference,
    _plot,
    build_sparse_intent,
)

UPDATED = "2026-08-06"


@dataclass(frozen=True)
class FeasibleSelection:
    start_anchor_index: int
    end_anchor_index: int
    source_start_sample: int
    source_end_sample: int
    qar: QARGuidanceReference
    route: Any
    environment: Any
    relative_anchor_indices: np.ndarray
    profile: Any


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
    for name, default in (
        (
            "vertical-parameters",
            "docs/modeling/a320-vertical-kinematic-calibration.yml",
        ),
        (
            "horizontal-parameters",
            "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        ),
        ("thrust-parameters", "docs/modeling/a320-effective-thrust-calibration.yml"),
        ("fuel-parameters", "docs/modeling/a320-effective-fuel-calibration.yml"),
        ("pitch-bank-parameters", "docs/modeling/a320-pitch-bank-input-calibration.yml"),
        (
            "surface-parameters",
            "docs/modeling/a320-vspaero-lift-surface-identification.yml",
        ),
        (
            "boundary-parameters",
            "docs/modeling/a320-vspaero-hybrid-boundary-selection.yml",
        ),
        ("lift-surface", "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv"),
        ("n1-envelope", "docs/modeling/a320-autothrust-envelope.yml"),
        (
            "climb-capacity",
            "docs/modeling/a320-demonstrated-climb-capacity.yml",
        ),
    ):
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-feasible-intent-errors.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-feasible-intent.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-feasible-intent.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-qar-feasible-intent-comparison"),
    )
    return parser.parse_args()


def _slice_reference(
    qar: QARGuidanceReference, start: int, end: int
) -> QARGuidanceReference:
    selection = slice(start, end + 1)
    original = qar.reference
    time = original.time_s[selection] - original.time_s[start]
    reference = TrajectoryReference(
        time_s=time,
        true_airspeed_mps=original.true_airspeed_mps[selection],
        true_airspeed_rate_mps2=original.true_airspeed_rate_mps2[selection],
        flight_path_angle_rad=original.flight_path_angle_rad[selection],
        flight_path_angle_rate_rad_s=original.flight_path_angle_rate_rad_s[selection],
        heading_rad=original.heading_rad[selection],
        heading_rate_rad_s=original.heading_rate_rad_s[selection],
        pressure_altitude_m=original.pressure_altitude_m[selection],
    )
    return QARGuidanceReference(
        reference=reference,
        north_position_m=qar.north_position_m[selection] - qar.north_position_m[start],
        east_position_m=qar.east_position_m[selection] - qar.east_position_m[start],
        static_temperature_k=qar.static_temperature_k[selection],
        wind_north_mps=qar.wind_north_mps[selection],
        wind_east_mps=qar.wind_east_mps[selection],
        mass_kg=qar.mass_kg[selection],
        initial_mass_kg=float(qar.mass_kg[start]),
        source_start_time_s=qar.source_start_time_s + float(original.time_s[start]),
        source_end_time_s=qar.source_start_time_s + float(original.time_s[end]),
    )


def select_longest_feasible_span(
    *,
    qar: QARGuidanceReference,
    performance: PointMassProfilePerformance,
) -> tuple[FeasibleSelection, Any]:
    """Select solely from sparse intent feasibility, never rollout errors."""

    full_route, _, full_anchor_indices = build_sparse_intent(qar)
    full_profile = plan_feasible_profile(
        route=full_route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=float(
            np.median(qar.static_temperature_k[full_anchor_indices])
        ),
        spacing_m=500.0,
    )
    candidates: list[FeasibleSelection] = []
    for start_anchor in range(len(full_anchor_indices) - 2):
        for end_anchor in range(start_anchor + 2, len(full_anchor_indices)):
            start_sample = int(full_anchor_indices[start_anchor])
            end_sample = int(full_anchor_indices[end_anchor])
            subsection = _slice_reference(qar, start_sample, end_sample)
            route, environment, relative_indices = build_sparse_intent(subsection)
            profile = plan_feasible_profile(
                route=route,
                performance=performance,
                mass_kg=subsection.initial_mass_kg,
                static_temperature_k=float(
                    np.median(subsection.static_temperature_k[relative_indices])
                ),
                spacing_m=500.0,
            )
            if profile.feasible:
                candidates.append(
                    FeasibleSelection(
                        start_anchor_index=start_anchor,
                        end_anchor_index=end_anchor,
                        source_start_sample=start_sample,
                        source_end_sample=end_sample,
                        qar=subsection,
                        route=route,
                        environment=environment,
                        relative_anchor_indices=relative_indices,
                        profile=profile,
                    )
                )
    if not candidates:
        raise ValueError(
            "no feasible sparse-intent span contains at least three anchors"
        )
    selected = max(
        candidates,
        key=lambda item: (
            item.source_end_sample - item.source_start_sample,
            -item.source_start_sample,
        ),
    )
    return selected, full_profile


def _document(
    selection: FeasibleSelection,
    full_profile: Any,
    metrics: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    all_gates = {
        "full_intent_rejected_before_rollout": {"passed": not full_profile.feasible},
        **gates,
    }
    passed = all(gate["passed"] for gate in all_gates.values())
    return {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "passed" if passed else "held",
        "selection": {
            "rule": "longest contiguous sparse-anchor span passing aircraft profile feasibility; earliest-start tie break; minimum three anchors",
            "uses_rollout_error_for_selection": False,
            "full_anchor_count": 9,
            "selected_start_anchor": selection.start_anchor_index,
            "selected_end_anchor": selection.end_anchor_index,
            "selected_source_sample_range": [
                selection.source_start_sample,
                selection.source_end_sample,
            ],
            "selected_duration_s": float(selection.qar.reference.time_s[-1]),
            "selected_anchor_count": len(selection.relative_anchor_indices),
            "phase_interpretation": "upper-climb transition, cruise, and descent",
        },
        "full_intent_diagnostic": {
            "feasible": full_profile.feasible,
            "violations": list(full_profile.violations),
            "rollout_artifact": "docs/fms/a320-fms-qar-intent.yml",
        },
        "metrics": metrics,
        "gates": all_gates,
        "decision": {
            "feasibility_screened_sparse_intent_gate_passed": passed,
            "full_sparse_intent_gate_reclassified_as_passed": False,
            "qar_used_as_withheld_reference": True,
            "qar_heading_used_by_guidance": False,
            "rollout_error_used_for_span_selection": False,
            "airbus_fms_autopilot_or_fadec_identified": False,
        },
        "artifacts": {
            "errors": "docs/fms/a320-fms-qar-feasible-intent-errors.csv",
            "figure": "docs/fms/figures/a320-fms-qar-feasible-intent-comparison.png",
            "report": "docs/fms/a320-fms-qar-feasible-intent.md",
            "code": "tools/validate_a320_fms_qar_feasible_intent.py",
        },
    }


def _report(document: dict[str, Any]) -> str:
    selection = document["selection"]
    metrics = document["metrics"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if gate['passed'] else 'hold'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 FMS feasibility-screened sparse-QAR-intent gate

Updated: {UPDATED}

Status: **{document["status"]}**

## Selection

The full nine-anchor intent remains held and is not reclassified. Its requested
climb profile is outside the aircraft-aware reachability model, including a
combined climb/acceleration energy violation.

A deterministic pre-rollout screen selected the longest contiguous feasible
span: anchors {selection["selected_start_anchor"]} through
{selection["selected_end_anchor"]}, source samples
{selection["selected_source_sample_range"][0]}–{selection["selected_source_sample_range"][1]},
lasting {selection["selected_duration_s"]:.0f} s. The rule uses only sparse
intent and aircraft limits; it does not inspect rollout error, heading, or any
withheld intermediate trajectory sample.

## Result

The selected five-anchor FMS trajectory has
{metrics["generated_to_qar_path_rmse_m"]:.1f} m nearest-path RMSE and
{metrics["time_aligned_horizontal_rmse_m"]:.1f} m time-aligned horizontal RMSE
against withheld QAR samples. Altitude and TAS RMSE are
{metrics["altitude_rmse_m"]:.1f} m and {metrics["speed_rmse_mps"]:.2f} m/s.
Arrival-time error is {metrics["arrival_time_error_s"]:+.0f} s.

Terminal horizontal, altitude, and TAS errors are
{metrics["terminal_horizontal_error_m"]:.1f} m,
{metrics["terminal_altitude_error_m"]:.1f} m, and
{metrics["terminal_speed_error_mps"]:.2f} m/s. Any limiter is active for
{100 * metrics["any_control_limited_fraction"]:.1f}% of samples.

## Gates

| Check | Result |
|---|---|
{gate_rows}

## Interpretation

This validates the FMS workflow only on the feasibility-screened portion of an
already-consumed A320 QAR flight. It does not validate the rejected initial
climb, does not provide a new independent aerodynamic confirmation, and does
not identify Airbus FMS, autopilot, autothrust, or FADEC logic.

QAR heading and intermediate samples remain withheld from guidance. Raw
coordinates are not persisted; the CSV contains error histories only and the
figure uses zero-referenced local displacement.

## Artifacts

- Error history: `docs/fms/a320-fms-qar-feasible-intent-errors.csv`
- Machine-readable result: `docs/fms/a320-fms-qar-feasible-intent.yml`
- Figure: `docs/fms/figures/a320-fms-qar-feasible-intent-comparison.{{png,pdf}}`
- Full-intent held diagnostic: `docs/fms/a320-fms-qar-intent.md`
- Reproducer: `tools/validate_a320_fms_qar_feasible_intent.py`
"""


def main() -> None:
    args = parse_args()
    qar, _ = _load_reference(args)
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
    selection, full_profile = select_longest_feasible_span(
        qar=qar, performance=performance
    )
    guidance = SpatialPathGuidance(
        selection.profile.trajectory,
        lookahead_time_s=45.0,
        minimum_lookahead_m=2_500.0,
        maximum_lookahead_m=10_000.0,
        maximum_heading_command_rate_rad_s=math.radians(1.0),
    )
    result = simulate_fms_trajectory_zoh(
        plant=plant,
        controller=_controller(),
        guidance=guidance,
        maximum_time_s=float(selection.qar.reference.time_s[-1] + 300.0),
        static_temperature_k=float(selection.environment.static_temperature_k[0]),
        initial_mass_kg=selection.qar.initial_mass_kg,
        destination_capture_radius_m=1_500.0,
        integration_rate_hz=INTEGRATION_RATE_HZ,
        environment_profile=selection.environment,
        maximum_thrust_model=thrust_ceiling,
    )
    metrics, errors, plot_data = _evaluate(
        selection.qar,
        selection.route,
        selection.relative_anchor_indices,
        result,
    )
    gates = _gates(metrics, selection.profile.feasible)
    document = _document(selection, full_profile, metrics, gates)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(
        selection.qar,
        selection.route,
        selection.relative_anchor_indices,
        result,
        plot_data,
        args.figure_prefix,
    )
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())
    if not document["decision"]["feasibility_screened_sparse_intent_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
