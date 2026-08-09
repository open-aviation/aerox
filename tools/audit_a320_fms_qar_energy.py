#!/usr/bin/env python3
"""Audit why the frozen sparse-QAR A320 intent fails energy feasibility."""

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
    plan_energy_shaped_profile,
    plan_feasible_profile,
)
from aerox.models import DragCondition
from aerox.models.atmosphere import (
    SEA_LEVEL_PRESSURE_PA,
    atmosphere_at_pressure_altitude,
    unit_reynolds_number,
)
from aerox.models.openap import OpenAPTwoSpoolMaximumClimbThrust
from aerox.models.atmosphere import G0
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling
from tools.validate_a320_fms_qar_intent import (
    _build_plant,
    _load_reference,
    build_sparse_intent,
)
from tools.validate_a320_qar_guidance_robustness import _longest_contiguous_indices

UPDATED = "2026-08-06"
ENERGY_EDGE_TRIM_S = 30


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
        "--segments",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-energy-segments.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-energy-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-fms-qar-energy-audit.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-fms-qar-energy-audit"),
    )
    return parser.parse_args()


def _level_drag_n(
    *,
    plant: Any,
    altitude_m: np.ndarray,
    speed_mps: np.ndarray,
    mass_kg: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    atmosphere = atmosphere_at_pressure_altitude(altitude_m, temperature_k)
    dynamic_pressure = 0.5 * atmosphere.density_kg_m3 * speed_mps**2
    lift_coefficient = mass_kg * G0 / (dynamic_pressure * plant.wing_area_m2)
    drag_coefficient = plant.drag_model.drag_coefficient_at(
        lift_coefficient,
        DragCondition(
            mach=speed_mps / atmosphere.speed_of_sound_mps,
            unit_reynolds_number_per_m=unit_reynolds_number(
                atmosphere.density_kg_m3,
                speed_mps,
                temperature_k,
            ),
        ),
    )
    return dynamic_pressure * plant.wing_area_m2 * drag_coefficient


def _maximum_thrust_n(
    *,
    ceiling: Any,
    altitude_m: np.ndarray,
    speed_mps: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            ceiling.maximum_thrust_n(
                pressure_altitude_m=float(altitude),
                true_airspeed_mps=float(speed),
                static_temperature_k=float(temperature),
            )
            for altitude, speed, temperature in zip(
                altitude_m, speed_mps, temperature_k, strict=True
            )
        ]
    )


def _required_n1_percent(
    *,
    ceiling: Any,
    required_thrust_n: np.ndarray,
    altitude_m: np.ndarray,
    speed_mps: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    atmosphere = atmosphere_at_pressure_altitude(altitude_m, temperature_k)
    mach = speed_mps / atmosphere.speed_of_sound_mps
    pressure_ratio = atmosphere.pressure_pa / SEA_LEVEL_PRESSURE_PA
    return np.array(
        [
            ceiling.thrust_map.n1_for_thrust_scalar(
                thrust_n=float(thrust),
                mach=float(local_mach),
                pressure_ratio=float(delta),
                static_temperature_k=float(temperature),
                minimum_n1_percent=20.0,
                maximum_n1_percent=110.0,
            )
            for thrust, local_mach, delta, temperature in zip(
                required_thrust_n,
                mach,
                pressure_ratio,
                temperature_k,
                strict=True,
            )
        ]
    )


def _planning_clb_n1_percent(
    *,
    ceiling: Any,
    altitude_m: np.ndarray,
    mach: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    if hasattr(ceiling, "planning_n1_percent"):
        return ceiling.planning_n1_percent(
            pressure_altitude_m=altitude_m,
            mach=mach,
            static_temperature_k=temperature_k,
        )
    return ceiling.n1_envelope.climb_limit_percent(
        pressure_altitude_m=altitude_m,
        mach=mach,
        static_temperature_k=temperature_k,
    )


def _route_ground_speed_mps(
    *,
    speed_mps: np.ndarray,
    track_rad: np.ndarray,
    wind_north_mps: np.ndarray,
    wind_east_mps: np.ndarray,
) -> np.ndarray:
    wind_along = wind_north_mps * np.cos(track_rad) + wind_east_mps * np.sin(track_rad)
    wind_cross = -wind_north_mps * np.sin(track_rad) + wind_east_mps * np.cos(track_rad)
    air_along = np.sqrt(np.maximum(speed_mps**2 - wind_cross**2, 1.0))
    return np.maximum(air_along + wind_along, 1.0)


def _profile_arrays(
    *, profile: Any, route: Any, environment: Any, initial_mass_kg: float
) -> dict[str, np.ndarray]:
    along_path = profile.along_path_m
    altitude = profile.nominal_altitude_m
    speed = profile.nominal_true_airspeed_mps
    samples = [route.sample(float(distance)) for distance in along_path]
    track = np.array([sample.ground_track_rad for sample in samples])
    temperature = np.array(
        [
            environment.sample(float(distance)).static_temperature_k
            for distance in along_path
        ]
    )
    wind_north = np.array(
        [environment.sample(float(distance)).wind_north_mps for distance in along_path]
    )
    wind_east = np.array(
        [environment.sample(float(distance)).wind_east_mps for distance in along_path]
    )
    mass = np.interp(
        along_path,
        route.waypoint_along_path_m,
        np.linspace(initial_mass_kg, initial_mass_kg, len(route.intent.waypoints)),
    )
    return {
        "along_path_m": along_path,
        "altitude_m": altitude,
        "speed_mps": speed,
        "track_rad": track,
        "temperature_k": temperature,
        "wind_north_mps": wind_north,
        "wind_east_mps": wind_east,
        "mass_kg": mass,
    }


def _metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    qar_valid = values["qar_energy_valid"]
    return {
        "current_intent_minimum_margin_mps2": float(
            np.min(values["current_intent_margin_mps2"])
        ),
        "mass_corrected_intent_minimum_margin_mps2": float(
            np.min(values["mass_corrected_intent_margin_mps2"])
        ),
        "temperature_corrected_intent_minimum_margin_mps2": float(
            np.min(values["temperature_corrected_intent_margin_mps2"])
        ),
        "wind_corrected_intent_minimum_margin_mps2": float(
            np.min(values["wind_corrected_intent_margin_mps2"])
        ),
        "combined_corrected_intent_minimum_margin_mps2": float(
            np.min(values["combined_corrected_intent_margin_mps2"])
        ),
        "energy_shaped_intent_minimum_margin_mps2": float(
            np.min(values["energy_shaped_intent_margin_mps2"])
        ),
        "current_intent_maximum_n1_excess_percent": float(
            np.max(values["current_intent_n1_excess_percent"])
        ),
        "combined_corrected_intent_maximum_n1_excess_percent": float(
            np.max(values["combined_corrected_intent_n1_excess_percent"])
        ),
        "qar_to_clb_minimum_margin_mps2": float(
            np.min(values["qar_to_clb_margin_mps2"][qar_valid])
        ),
        "qar_to_clb_p01_margin_mps2": float(
            np.percentile(values["qar_to_clb_margin_mps2"][qar_valid], 1.0)
        ),
        "qar_to_recorded_n1_minimum_margin_mps2": float(
            np.min(values["qar_to_recorded_n1_margin_mps2"][qar_valid])
        ),
        "qar_to_recorded_n1_median_margin_mps2": float(
            np.median(values["qar_to_recorded_n1_margin_mps2"][qar_valid])
        ),
        "qar_fraction_above_clb_capacity": float(
            np.mean(values["qar_to_clb_margin_mps2"][qar_valid] < 0.0)
        ),
        "qar_fraction_above_recorded_n1_capacity": float(
            np.mean(values["qar_to_recorded_n1_margin_mps2"][qar_valid] < 0.0)
        ),
        "recorded_n1_above_clb_fraction": float(
            np.mean(
                values["recorded_n1_percent"][qar_valid]
                > values["clb_n1_percent"][qar_valid]
            )
        ),
        "recorded_n1_minus_clb_p95_percent": float(
            np.percentile(
                (values["recorded_n1_percent"] - values["clb_n1_percent"])[qar_valid],
                95.0,
            )
        ),
        "openap_fast_climb_to_qar_clb_median_ratio": float(
            np.median(
                values["qar_openap_fast_climb_thrust_n"][qar_valid]
                / values["qar_clb_thrust_n"][qar_valid]
            )
        ),
        "openap_fast_climb_above_qar_clb_fraction": float(
            np.mean(
                values["qar_openap_fast_climb_thrust_n"][qar_valid]
                > values["qar_clb_thrust_n"][qar_valid]
            )
        ),
        "median_modeled_level_drag_kn": float(
            np.median(values["qar_level_drag_n"][qar_valid]) / 1_000.0
        ),
        "median_lift_informed_drag_kn": float(
            np.median(values["qar_lift_informed_drag_n"][qar_valid]) / 1_000.0
        ),
        "median_implied_drag_kn": float(
            np.median(values["qar_implied_drag_n"][qar_valid]) / 1_000.0
        ),
        "negative_implied_drag_fraction": float(
            np.mean(values["qar_implied_drag_n"][qar_valid] < 0.0)
        ),
    }


def _segment_table(
    *,
    anchor_indices: np.ndarray,
    route: Any,
    profile_s: np.ndarray,
    values: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in range(len(anchor_indices) - 1):
        first_sample = int(anchor_indices[segment])
        last_sample = int(anchor_indices[segment + 1])
        qar_selection = np.zeros_like(values["qar_energy_valid"], dtype=bool)
        qar_selection[first_sample : last_sample + 1] = True
        qar_selection &= values["qar_energy_valid"]
        midpoint = (first_sample + last_sample) // 2
        first_half = qar_selection.copy()
        first_half[midpoint + 1 :] = False
        second_half = qar_selection.copy()
        second_half[: midpoint + 1] = False
        start_s = route.waypoint_along_path_m[segment]
        end_s = route.waypoint_along_path_m[segment + 1]
        profile_selection = (profile_s >= start_s) & (profile_s <= end_s)
        rows.append(
            {
                "segment": f"A{segment}-A{segment + 1}",
                "source_start_sample": first_sample,
                "source_end_sample": last_sample,
                "duration_s": last_sample - first_sample,
                "distance_km": (end_s - start_s) / 1_000.0,
                "current_intent_peak_demand_mps2": float(
                    np.max(values["current_intent_demand_mps2"][profile_selection])
                ),
                "wind_corrected_intent_peak_demand_mps2": float(
                    np.max(
                        values["wind_corrected_intent_demand_mps2"][profile_selection]
                    )
                ),
                "current_intent_minimum_margin_mps2": float(
                    np.min(values["current_intent_margin_mps2"][profile_selection])
                ),
                "combined_corrected_intent_minimum_margin_mps2": float(
                    np.min(
                        values["combined_corrected_intent_margin_mps2"][
                            profile_selection
                        ]
                    )
                ),
                "energy_shaped_intent_minimum_margin_mps2": float(
                    np.min(
                        values["energy_shaped_intent_margin_mps2"][profile_selection]
                    )
                ),
                "qar_p99_demand_mps2": float(
                    np.percentile(values["qar_demand_mps2"][qar_selection], 99.0)
                ),
                "qar_first_half_mean_demand_mps2": float(
                    np.mean(values["qar_demand_mps2"][first_half])
                ),
                "qar_second_half_mean_demand_mps2": float(
                    np.mean(values["qar_demand_mps2"][second_half])
                ),
                "qar_to_clb_p01_margin_mps2": float(
                    np.percentile(values["qar_to_clb_margin_mps2"][qar_selection], 1.0)
                ),
                "qar_to_recorded_n1_median_margin_mps2": float(
                    np.median(values["qar_to_recorded_n1_margin_mps2"][qar_selection])
                ),
                "recorded_n1_median_percent": float(
                    np.median(values["recorded_n1_percent"][qar_selection])
                ),
                "clb_n1_median_percent": float(
                    np.median(values["clb_n1_percent"][qar_selection])
                ),
                "recorded_n1_above_clb_fraction": float(
                    np.mean(
                        values["recorded_n1_percent"][qar_selection]
                        > values["clb_n1_percent"][qar_selection]
                    )
                ),
                "combined_intent_maximum_n1_excess_percent": float(
                    np.max(
                        values["combined_corrected_intent_n1_excess_percent"][
                            profile_selection
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


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
    axis.axhline(0.0, color="#666666", linewidth=0.8, zorder=0)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot(
    *,
    values: dict[str, np.ndarray],
    segment_table: pd.DataFrame,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    black, blue, orange, green, purple = (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
    )
    route_km = values["profile_along_path_m"] / 1_000.0
    qar_km = values["qar_cumulative_distance_m"] / 1_000.0
    qar_valid = values["qar_energy_valid"]

    def valid_qar(value: np.ndarray) -> np.ndarray:
        return np.where(qar_valid, value, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)

    axes[0, 0].plot(
        route_km,
        values["current_intent_demand_mps2"],
        color=orange,
        linestyle="--",
        label="current sparse intent",
    )
    axes[0, 0].plot(
        route_km,
        values["wind_corrected_intent_demand_mps2"],
        color=green,
        label="wind-aware sparse intent",
    )
    axes[0, 0].plot(
        qar_km,
        valid_qar(values["qar_demand_mps2"]),
        color=black,
        alpha=0.75,
        linewidth=1.2,
        label="withheld QAR",
    )
    axes[0, 0].set(
        xlabel="Cumulative path distance (km)",
        ylabel="Specific excess-force demand (m/s²)",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    axes[0, 1].plot(
        route_km,
        values["current_available_mps2"],
        color=blue,
        linestyle=":",
        linewidth=1.8,
        label="current CLB capacity",
    )
    axes[0, 1].plot(
        route_km,
        values["combined_available_mps2"],
        color=blue,
        linewidth=1.6,
        label="condition-corrected CLB capacity",
    )
    axes[0, 1].plot(
        route_km,
        values["wind_corrected_intent_demand_mps2"],
        color=green,
        label="wind-aware intent demand",
    )
    axes[0, 1].plot(
        route_km,
        values["energy_shaped_intent_demand_mps2"],
        color=purple,
        linestyle="--",
        label="energy-shaped demand",
    )
    axes[0, 1].set(
        xlabel="Sparse-intent distance (km)",
        ylabel="Specific excess force (m/s²)",
    )
    axes[0, 1].legend(frameon=False, fontsize=7)

    axes[0, 2].plot(
        qar_km,
        values["qar_clb_thrust_n"] / 1_000.0,
        color=blue,
        label="CLB ceiling",
    )
    axes[0, 2].plot(
        qar_km,
        values["qar_recorded_n1_thrust_n"] / 1_000.0,
        color=orange,
        linestyle="--",
        label="recorded-N1 effective thrust",
    )
    axes[0, 2].plot(
        qar_km,
        values["qar_openap_fast_climb_thrust_n"] / 1_000.0,
        color=green,
        linestyle=":",
        linewidth=1.7,
        label="OpenAP fast-climb prior",
    )
    axes[0, 2].plot(
        qar_km,
        valid_qar(values["qar_required_thrust_n"]) / 1_000.0,
        color=black,
        alpha=0.75,
        label="QAR energy-required thrust",
    )
    axes[0, 2].set(
        xlabel="QAR cumulative distance (km)", ylabel="Total net thrust (kN)"
    )
    axes[0, 2].legend(frameon=False, fontsize=7)

    axes[1, 0].plot(
        qar_km,
        values["qar_level_drag_n"] / 1_000.0,
        color=blue,
        label="level-lift polar drag",
    )
    axes[1, 0].plot(
        qar_km,
        values["qar_lift_informed_drag_n"] / 1_000.0,
        color=green,
        linestyle="--",
        label="QAR-lift polar drag",
    )
    axes[1, 0].plot(
        qar_km,
        valid_qar(values["qar_implied_drag_n"]) / 1_000.0,
        color=purple,
        alpha=0.75,
        label="N1/energy-implied drag",
    )
    axes[1, 0].set(xlabel="QAR cumulative distance (km)", ylabel="Drag (kN)")
    axes[1, 0].legend(frameon=False, fontsize=7)

    axes[1, 1].plot(
        route_km,
        values["current_intent_margin_mps2"],
        color=orange,
        linestyle="--",
        label="current assumptions",
    )
    axes[1, 1].plot(
        route_km,
        values["combined_corrected_intent_margin_mps2"],
        color=green,
        label="wind + mass + temperature",
    )
    axes[1, 1].plot(
        route_km,
        values["energy_shaped_intent_margin_mps2"],
        color=purple,
        linestyle=":",
        linewidth=1.8,
        label="energy-shaped",
    )
    axes[1, 1].set(
        xlabel="Sparse-intent distance (km)",
        ylabel="CLB energy margin (m/s²)",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)

    positions = np.arange(len(segment_table))
    width = 0.26
    axes[1, 2].bar(
        positions - width,
        segment_table["current_intent_minimum_margin_mps2"],
        width,
        color=orange,
        hatch="//",
        label="current",
    )
    axes[1, 2].bar(
        positions,
        segment_table["combined_corrected_intent_minimum_margin_mps2"],
        width,
        color=green,
        label="condition-corrected",
    )
    axes[1, 2].bar(
        positions + width,
        segment_table["energy_shaped_intent_minimum_margin_mps2"],
        width,
        color=purple,
        label="energy-shaped",
    )
    axes[1, 2].set_xticks(positions, segment_table["segment"], rotation=35, ha="right")
    axes[1, 2].set(xlabel="Sparse-intent segment", ylabel="Minimum CLB margin (m/s²)")
    axes[1, 2].legend(frameon=False, fontsize=7)

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        _style_axis(axis, label)
    fig.suptitle(
        "A320 sparse-QAR intent energy-consistency audit; no model refit",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _report(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    diagnosis = document["diagnosis"]
    rows = "\n".join(
        f"| {item['segment']} | {item['current_intent_minimum_margin_mps2']:.3f} | "
        f"{item['combined_corrected_intent_minimum_margin_mps2']:.3f} | "
        f"{item['qar_to_clb_p01_margin_mps2']:.3f} |"
        for item in document["segments"]
    )
    return f"""# A320 sparse-QAR-intent energy audit

Updated: {UPDATED}

Status: **{document["status"]}**

## Question

Why does the frozen nine-anchor intent from `TAIL-A/FLT0004` fail the
aircraft-aware profile-feasibility check even though the aircraft flew the
underlying QAR trajectory?

This diagnostic does not refit the thrust map, drag polar, planning capacity,
aircraft mass, guidance controller, or any frozen validation threshold.

## Result

The current sparse-intent calculation reaches a minimum CLB energy margin of
{metrics["current_intent_minimum_margin_mps2"]:.3f} m/s². Applying recorded
wind, anchor-interpolated temperature, and QAR mass changes it to
{metrics["combined_corrected_intent_minimum_margin_mps2"]:.3f} m/s². The
condition-corrected intent requires at most
{metrics["combined_corrected_intent_maximum_n1_excess_percent"]:.2f} N1
percentage points above the active planning boundary.

The no-refit energy-shaped planner reduces the minimum deficit to
{metrics["energy_shaped_intent_minimum_margin_mps2"]:.3f} m/s² but remains
{"feasible" if document["energy_shaped_profile"]["feasible"] else "infeasible"}
under that same uncertainty-bounded boundary. It therefore does not reclassify the
full-intent gate.

The continuous QAR reference has a 1st-percentile margin of
{metrics["qar_to_clb_p01_margin_mps2"]:.3f} m/s² against the same released CLB
planning capacity; {100 * metrics["qar_fraction_above_clb_capacity"]:.1f}% of samples
exceed that modeled capacity. Recorded N1 exceeds the planning N1 boundary in
{100 * metrics["recorded_n1_above_clb_fraction"]:.1f}% of samples; its 95th
percentile excess is {metrics["recorded_n1_minus_clb_p95_percent"]:.2f} N1
percentage points. QAR derivative statistics exclude the 30-second endpoint
transients of the longest smoothing filter.

The independent OpenAP/Bartel--Young 4,000 ft/min fast-climb prior is
{metrics["openap_fast_climb_to_qar_clb_median_ratio"]:.3f} times the QAR CLB
thrust at the median condition and exceeds it for only
{100 * metrics["openap_fast_climb_above_qar_clb_fraction"]:.1f}% of samples.
It is therefore not adopted as an upper-thrust bound.

Median level-lift polar drag, QAR-lift polar drag, and N1/energy-implied drag
are {metrics["median_modeled_level_drag_kn"]:.1f},
{metrics["median_lift_informed_drag_kn"]:.1f}, and
{metrics["median_implied_drag_kn"]:.1f} kN.

## Segment diagnosis

| Segment | Current intent margin (m/s²) | Condition-corrected margin (m/s²) | QAR-to-CLB p01 margin (m/s²) |
|---|---:|---:|---:|
{rows}

## Decision

Primary diagnosis: **{diagnosis["primary_cause"]}**.

{diagnosis["interpretation"]}

The full sparse-intent gate remains held. This audit is evidence for the next
change; it is not itself a release of a corrected performance model.

## Artifacts

- Segment diagnostics: `docs/fms/a320-fms-qar-energy-segments.csv`
- Machine-readable result: `docs/fms/a320-fms-qar-energy-audit.yml`
- Figure: `docs/fms/figures/a320-fms-qar-energy-audit.{{png,pdf}}`
- Reproducer: `tools/audit_a320_fms_qar_energy.py`
"""


def main() -> None:
    args = parse_args()
    qar, flight = _load_reference(args)
    plant = _build_plant(args)
    ceiling = _load_climb_thrust_ceiling(args)
    route, environment, anchor_indices = build_sparse_intent(qar)
    performance = PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=ceiling,
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=math.radians(6.0),
        maximum_descent_angle_rad=math.radians(4.0),
        maximum_acceleration_mps2=0.5,
        maximum_deceleration_mps2=0.5,
    )
    median_temperature = float(np.median(qar.static_temperature_k[anchor_indices]))
    profile = plan_feasible_profile(
        route=route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=median_temperature,
        spacing_m=500.0,
    )
    shaped_profile = plan_energy_shaped_profile(
        route=route,
        performance=performance,
        mass_kg=qar.initial_mass_kg,
        static_temperature_k=median_temperature,
        spacing_m=500.0,
        environment_profile=environment,
        waypoint_mass_kg=qar.mass_kg[anchor_indices],
    )
    profile_values = _profile_arrays(
        profile=profile,
        route=route,
        environment=environment,
        initial_mass_kg=qar.initial_mass_kg,
    )
    profile_s = profile_values["along_path_m"]
    profile_values["mass_kg"] = np.interp(
        profile_s, route.waypoint_along_path_m, qar.mass_kg[anchor_indices]
    )
    altitude = profile_values["altitude_m"]
    speed = profile_values["speed_mps"]
    altitude_gradient = np.gradient(altitude, profile_s)
    speed_gradient = np.gradient(speed, profile_s)
    current_demand = speed * speed_gradient + G0 * np.sin(np.arctan(altitude_gradient))
    ground_speed = _route_ground_speed_mps(
        speed_mps=speed,
        track_rad=profile_values["track_rad"],
        wind_north_mps=profile_values["wind_north_mps"],
        wind_east_mps=profile_values["wind_east_mps"],
    )
    wind_demand = ground_speed * speed_gradient + G0 * np.clip(
        ground_speed * altitude_gradient / speed, -0.95, 0.95
    )
    current_available = profile.maximum_specific_excess_force_mps2
    variable_mass_drag = _level_drag_n(
        plant=plant,
        altitude_m=altitude,
        speed_mps=speed,
        mass_kg=profile_values["mass_kg"],
        temperature_k=np.full_like(altitude, median_temperature),
    )
    current_thrust = profile.maximum_thrust_n
    mass_available = (current_thrust - variable_mass_drag) / profile_values["mass_kg"]
    temperature_thrust = _maximum_thrust_n(
        ceiling=ceiling,
        altitude_m=altitude,
        speed_mps=speed,
        temperature_k=profile_values["temperature_k"],
    )
    temperature_drag = _level_drag_n(
        plant=plant,
        altitude_m=altitude,
        speed_mps=speed,
        mass_kg=np.full_like(altitude, qar.initial_mass_kg),
        temperature_k=profile_values["temperature_k"],
    )
    temperature_available = (
        temperature_thrust - temperature_drag
    ) / qar.initial_mass_kg
    combined_drag = _level_drag_n(
        plant=plant,
        altitude_m=altitude,
        speed_mps=speed,
        mass_kg=profile_values["mass_kg"],
        temperature_k=profile_values["temperature_k"],
    )
    combined_available = (temperature_thrust - combined_drag) / profile_values[
        "mass_kg"
    ]
    shaped_demand = shaped_profile.nominal_true_airspeed_mps * np.gradient(
        shaped_profile.nominal_true_airspeed_mps, shaped_profile.along_path_m
    ) + G0 * np.sin(
        np.arctan(
            np.gradient(shaped_profile.nominal_altitude_m, shaped_profile.along_path_m)
        )
    )
    current_temperature = np.full_like(altitude, median_temperature)
    current_required_thrust = (
        profile.level_drag_n + qar.initial_mass_kg * current_demand
    )
    current_required_n1 = _required_n1_percent(
        ceiling=ceiling,
        required_thrust_n=current_required_thrust,
        altitude_m=altitude,
        speed_mps=speed,
        temperature_k=current_temperature,
    )
    current_atmosphere = atmosphere_at_pressure_altitude(altitude, current_temperature)
    current_clb_n1 = _planning_clb_n1_percent(
        ceiling=ceiling,
        altitude_m=altitude,
        mach=speed / current_atmosphere.speed_of_sound_mps,
        temperature_k=current_temperature,
    )
    combined_required_thrust = combined_drag + profile_values["mass_kg"] * wind_demand
    combined_required_n1 = _required_n1_percent(
        ceiling=ceiling,
        required_thrust_n=combined_required_thrust,
        altitude_m=altitude,
        speed_mps=speed,
        temperature_k=profile_values["temperature_k"],
    )
    combined_atmosphere = atmosphere_at_pressure_altitude(
        altitude, profile_values["temperature_k"]
    )
    combined_clb_n1 = _planning_clb_n1_percent(
        ceiling=ceiling,
        altitude_m=altitude,
        mach=speed / combined_atmosphere.speed_of_sound_mps,
        temperature_k=profile_values["temperature_k"],
    )

    selection = _longest_contiguous_indices(flight.base.sample_index)
    base = flight.base
    if len(selection) != len(qar.reference.time_s):
        raise ValueError("QAR reference and propulsion source selections do not align")
    reference = qar.reference
    atmosphere = atmosphere_at_pressure_altitude(
        reference.pressure_altitude_m, qar.static_temperature_k
    )
    mach = reference.true_airspeed_mps / atmosphere.speed_of_sound_mps
    pressure_ratio = atmosphere.pressure_pa / SEA_LEVEL_PRESSURE_PA
    recorded_n1 = base.n1_percent[selection]
    recorded_n1_thrust = ceiling.thrust_map.thrust_n(
        n1_percent=recorded_n1,
        mach=mach,
        pressure_ratio=pressure_ratio,
        static_temperature_k=qar.static_temperature_k,
    )
    clb_n1 = _planning_clb_n1_percent(
        ceiling=ceiling,
        altitude_m=reference.pressure_altitude_m,
        mach=mach,
        temperature_k=qar.static_temperature_k,
    )
    clb_thrust = ceiling.thrust_map.thrust_n(
        n1_percent=clb_n1,
        mach=mach,
        pressure_ratio=pressure_ratio,
        static_temperature_k=qar.static_temperature_k,
    )
    openap_fast_climb = OpenAPTwoSpoolMaximumClimbThrust(
        reference_rate_of_climb_ft_min=4_000.0
    )
    openap_fast_climb_thrust = _maximum_thrust_n(
        ceiling=openap_fast_climb,
        altitude_m=reference.pressure_altitude_m,
        speed_mps=reference.true_airspeed_mps,
        temperature_k=qar.static_temperature_k,
    )
    level_drag = _level_drag_n(
        plant=plant,
        altitude_m=reference.pressure_altitude_m,
        speed_mps=reference.true_airspeed_mps,
        mass_kg=qar.mass_kg,
        temperature_k=qar.static_temperature_k,
    )
    dynamic_pressure = 0.5 * atmosphere.density_kg_m3 * reference.true_airspeed_mps**2
    lift_coefficient = base.lift_coefficient[selection]
    lift_informed_drag_coefficient = plant.drag_model.drag_coefficient_at(
        lift_coefficient,
        DragCondition(
            mach=mach,
            unit_reynolds_number_per_m=unit_reynolds_number(
                atmosphere.density_kg_m3,
                reference.true_airspeed_mps,
                qar.static_temperature_k,
            ),
        ),
    )
    lift_informed_drag = (
        dynamic_pressure * plant.wing_area_m2 * lift_informed_drag_coefficient
    )
    qar_demand = reference.true_airspeed_rate_mps2 + G0 * np.sin(
        reference.flight_path_angle_rad
    )
    required_thrust = lift_informed_drag + qar.mass_kg * qar_demand
    implied_drag = recorded_n1_thrust - qar.mass_kg * qar_demand
    qar_cumulative_distance = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.hypot(np.diff(qar.north_position_m), np.diff(qar.east_position_m))
            ),
        )
    )
    qar_energy_valid = np.ones(len(reference.time_s), dtype=bool)
    qar_energy_valid[:ENERGY_EDGE_TRIM_S] = False
    qar_energy_valid[-ENERGY_EDGE_TRIM_S:] = False

    values = {
        "profile_along_path_m": profile_s,
        "qar_cumulative_distance_m": qar_cumulative_distance,
        "current_intent_demand_mps2": current_demand,
        "wind_corrected_intent_demand_mps2": wind_demand,
        "current_available_mps2": current_available,
        "combined_available_mps2": combined_available,
        "current_intent_margin_mps2": current_available - current_demand,
        "mass_corrected_intent_margin_mps2": mass_available - current_demand,
        "temperature_corrected_intent_margin_mps2": temperature_available
        - current_demand,
        "wind_corrected_intent_margin_mps2": current_available - wind_demand,
        "combined_corrected_intent_margin_mps2": combined_available - wind_demand,
        "energy_shaped_intent_demand_mps2": shaped_demand,
        "energy_shaped_intent_margin_mps2": shaped_profile.maximum_specific_excess_force_mps2
        - shaped_demand,
        "current_intent_n1_excess_percent": current_required_n1 - current_clb_n1,
        "combined_corrected_intent_n1_excess_percent": combined_required_n1
        - combined_clb_n1,
        "qar_demand_mps2": qar_demand,
        "qar_energy_valid": qar_energy_valid,
        "qar_to_clb_margin_mps2": (clb_thrust - lift_informed_drag) / qar.mass_kg
        - qar_demand,
        "qar_to_recorded_n1_margin_mps2": (recorded_n1_thrust - lift_informed_drag)
        / qar.mass_kg
        - qar_demand,
        "recorded_n1_percent": recorded_n1,
        "clb_n1_percent": clb_n1,
        "qar_level_drag_n": level_drag,
        "qar_lift_informed_drag_n": lift_informed_drag,
        "qar_implied_drag_n": implied_drag,
        "qar_clb_thrust_n": clb_thrust,
        "qar_openap_fast_climb_thrust_n": openap_fast_climb_thrust,
        "qar_recorded_n1_thrust_n": recorded_n1_thrust,
        "qar_required_thrust_n": required_thrust,
    }
    metrics = _metrics(values)
    segment_table = _segment_table(
        anchor_indices=anchor_indices,
        route=route,
        profile_s=profile_s,
        values=values,
    )
    worst_segment = str(
        segment_table.loc[
            segment_table["current_intent_minimum_margin_mps2"].idxmin(), "segment"
        ]
    )
    correction_gain = (
        metrics["combined_corrected_intent_minimum_margin_mps2"]
        - metrics["current_intent_minimum_margin_mps2"]
    )
    still_infeasible = metrics["combined_corrected_intent_minimum_margin_mps2"] < 0.0
    demonstrated_capacity = hasattr(ceiling, "planning_n1_percent")
    if still_infeasible:
        if demonstrated_capacity:
            primary_cause = (
                "exact sparse-anchor energy demand remains marginally above the "
                "uncertainty-bounded demonstrated capacity"
            )
            interpretation = (
                f"The earlier audit identified {worst_segment} under median-temperature "
                "piecewise-linear scheduling and A0-A1 after recorded-condition correction. "
                "The released complete-flight-bootstrap CLB capacity and monotone multi-knot "
                "energy shaping remove most of that mismatch without changing an anchor, "
                "controller, or threshold. A small local deficit remains. Polar drag still "
                "agrees with QAR-lift and energy-implied drag, so changing drag is unsupported. "
                "The new bound is demonstrated operational capacity, not physical maximum "
                "available thrust; the full gate remains held."
            )
        else:
            primary_cause = (
                "piecewise-linear energy scheduling and misuse of an operational N1 "
                "quantile as a hard ceiling"
            )
            interpretation = (
                f"Under the original median-temperature assumptions, {worst_segment} "
                "has the smallest margin. After applying the recorded conditions, A0-A1 "
                "becomes limiting because its piecewise-linear profile holds demand nearly "
                "constant while available thrust falls with altitude. The QAR trajectory "
                "instead front-loads energy addition. Polar drag agrees with the "
                "N1/energy-implied drag, so changing drag is not supported."
            )
    else:
        primary_cause = "omitted wind and condition variation"
        interpretation = (
            f"The worst interval is {worst_segment}, but wind, mass, and temperature "
            "correction restore non-negative margin. The planner should consume the "
            "spatial environment and mass profile during feasibility propagation."
        )
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214, TAIL-A",
        "status": "diagnostic-complete-full-intent-remains-held",
        "source": {
            "recording": "TAIL-A/FLT0004",
            "already_consumed": True,
            "model_or_gate_refitted": False,
            "raw_coordinates_persisted": False,
        },
        "methods": {
            "qar_specific_energy_demand": "dV/dt + g sin(gamma)",
            "available_specific_excess_force": "(effective net thrust - polar drag) / mass",
            "current_profile": "piecewise-linear sparse altitude/TAS versus local route distance; initial mass; median anchor temperature; TAS used as path rate",
            "condition_corrected_profile": "recorded wind projected on route; anchor-interpolated mass and temperature",
            "recorded_thrust": "recorded mean N1 through frozen effective thrust map",
            "maximum_thrust": "released demonstrated CLB capacity with operational q90 fallback outside support, through frozen effective thrust map"
            if demonstrated_capacity
            else "frozen effective CLB N1 ceiling through frozen effective thrust map",
            "independent_thrust_prior": "pinned OpenAP/Bartel-Young CFM56-5B4 two-spool model at its 4000 ft/min fast-climb setting; diagnostic only",
            "energy_shaped_profile": "deterministic capacity weighting plus bounded per-leg exponential and monotone multi-knot altitude/speed timing search; exact anchors; no thrust or limit increase",
        },
        "metrics": metrics,
        "energy_shaped_profile": {
            "feasible": shaped_profile.feasible,
            "violations": list(shaped_profile.violations),
        },
        "segments": segment_table.to_dict(orient="records"),
        "diagnosis": {
            "primary_cause": primary_cause,
            "worst_sparse_segment": worst_segment,
            "condition_correction_gain_mps2": float(correction_gain),
            "interpretation": interpretation,
            "full_sparse_intent_released": False,
        },
        "artifacts": {
            "segments": str(args.segments),
            "figure": str(args.figure_prefix.with_suffix(".png")),
            "report": str(args.report),
            "code": "tools/audit_a320_fms_qar_energy.py",
        },
    }
    args.segments.parent.mkdir(parents=True, exist_ok=True)
    segment_table.to_csv(args.segments, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(values=values, segment_table=segment_table, prefix=args.figure_prefix)
    print(yaml.safe_dump(document["diagnosis"], sort_keys=False).strip())
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
