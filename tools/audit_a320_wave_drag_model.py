#!/usr/bin/env python3
"""Audit the vectorized A320 alpha/Mach lift and transonic drag prior."""

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

from aerox import aero, load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.models import (
    EffectiveLiftMap,
    ParabolicDragPolar,
    ParabolicWaveDragModel,
    KornLockWaveDrag,
    atmosphere_at_pressure_altitude,
)
from aerox.models.atmosphere import G0

UPDATED = "2026-08-08"
FT_TO_M = 0.3048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lift-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-lift-calibration.yml"),
    )
    parser.add_argument(
        "--representative",
        type=Path,
        default=Path("docs/fms/a320-economy-cruise-intent-representative.csv"),
    )
    parser.add_argument(
        "--qar-points",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-qar-points.csv"),
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-surface.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-model.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-model.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-wave-drag-model"),
    )
    return parser.parse_args()


def _lift_model(path: Path) -> EffectiveLiftMap:
    model = yaml.safe_load(path.read_text())["lift_model"]
    names = model["feature_order"]
    coefficients = np.asarray(
        [model["coefficients"][name] for name in names], dtype=float
    )
    return EffectiveLiftMap(str(model["feature_set"]), coefficients)


def _models() -> tuple[ParabolicDragPolar, ParabolicWaveDragModel]:
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    polar = reference_drag
    wave = KornLockWaveDrag(
        wing_sweep_rad=np.deg2rad(float(aircraft.wing.sweep)),
        thickness_to_chord_ratio=float(aircraft.wing.t_c),
    )
    return polar, ParabolicWaveDragModel(polar, wave)


def _surface(
    lift_model: EffectiveLiftMap, drag_model: ParabolicWaveDragModel
) -> pd.DataFrame:
    alpha_deg = np.linspace(0.0, 8.0, 81)
    mach = np.linspace(0.55, 0.82, 109)
    mach_grid, alpha_grid = np.meshgrid(mach, alpha_deg, indexing="ij")
    lift = lift_model.lift_coefficient(
        np.deg2rad(alpha_grid.ravel()), mach_grid.ravel()
    ).reshape(mach_grid.shape)
    drag = drag_model.breakdown(lift, mach_grid)
    return pd.DataFrame(
        {
            "mach": mach_grid.ravel(),
            "angle_of_attack_deg": alpha_grid.ravel(),
            "lift_coefficient": lift.ravel(),
            "critical_mach": drag.critical_mach.ravel(),
            "parasitic_drag_coefficient": drag.parasite.ravel(),
            "induced_drag_coefficient": drag.induced.ravel(),
            "wave_drag_coefficient": drag.wave.ravel(),
            "total_drag_coefficient": drag.total.ravel(),
        }
    )


def _qar_points(
    path: Path, polar: ParabolicDragPolar, drag_model: ParabolicWaveDragModel
) -> pd.DataFrame:
    aircraft = load_aircraft("A320")
    frame = pd.read_csv(path)
    altitude_m = frame["pressure_altitude_ft"].to_numpy(dtype=float) * FT_TO_M
    temperature_k = np.asarray(aero.temperature(altitude_m), dtype=float)
    atmosphere = atmosphere_at_pressure_altitude(altitude_m, temperature_k)
    mach = frame["actual_mach"].to_numpy(dtype=float)
    speed = mach * atmosphere.speed_of_sound_mps
    dynamic_pressure = 0.5 * atmosphere.density_kg_m3 * speed**2
    lift = (
        frame["mass_kg"].to_numpy(dtype=float)
        * G0
        / (dynamic_pressure * float(aircraft.wing.area))
    )
    breakdown = drag_model.breakdown(lift, mach)
    base_drag_n = (
        dynamic_pressure
        * float(aircraft.wing.area)
        * polar.drag_coefficient(lift, mach)
    )
    wave_drag_n = (
        dynamic_pressure * float(aircraft.wing.area) * breakdown.wave
    )
    result = frame[
        ["elapsed_time_s", "pressure_altitude_ft", "actual_mach", "mass_kg"]
    ].copy()
    result["lift_coefficient"] = lift
    result["critical_mach"] = breakdown.critical_mach
    result["parabolic_drag_coefficient"] = breakdown.parasite + breakdown.induced
    result["wave_drag_coefficient"] = breakdown.wave
    result["total_drag_coefficient"] = breakdown.total
    result["parabolic_drag_n"] = base_drag_n
    result["wave_drag_n"] = wave_drag_n
    result["wave_drag_fraction"] = wave_drag_n / (base_drag_n + wave_drag_n)
    return result


def _plot(
    *,
    surface: pd.DataFrame,
    qar: pd.DataFrame,
    drag_model: ParabolicWaveDragModel,
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
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    alpha = np.sort(surface["angle_of_attack_deg"].unique())
    mach = np.sort(surface["mach"].unique())

    def grid(column: str) -> np.ndarray:
        return surface.pivot(
            index="mach", columns="angle_of_attack_deg", values=column
        ).loc[mach, alpha].to_numpy()

    figure, axes = plt.subplots(
        2, 2, figsize=(183.0 / 25.4, 126.0 / 25.4), constrained_layout=True
    )
    lift_image = axes[0, 0].pcolormesh(
        alpha, mach, grid("lift_coefficient"), shading="auto", cmap="viridis"
    )
    figure.colorbar(lift_image, ax=axes[0, 0], label="Lift coefficient")
    axes[0, 0].set(
        xlabel="Effective angle of attack (deg)",
        ylabel="Mach number",
        title="Current pitch/angle-based lift map",
    )

    wave_image = axes[0, 1].pcolormesh(
        alpha,
        mach,
        grid("wave_drag_coefficient"),
        shading="auto",
        cmap="magma",
    )
    figure.colorbar(wave_image, ax=axes[0, 1], label="Wave-drag coefficient")
    axes[0, 1].set(
        xlabel="Effective angle of attack (deg)",
        ylabel="Mach number",
        title="Experimental transonic drag rise",
    )

    mach_line = np.linspace(0.60, 0.84, 241)
    for lift, color in ((0.4, "#0072B2"), (0.5, "#009E73"), (0.6, "#D55E00")):
        values = drag_model.breakdown(np.full_like(mach_line, lift), mach_line)
        axes[1, 0].plot(
            mach_line,
            values.total,
            color=color,
            label=f"CL={lift:.1f}, total",
        )
        axes[1, 0].plot(
            mach_line,
            values.parasite + values.induced,
            color=color,
            linestyle=":",
            linewidth=0.8,
        )
        axes[1, 0].axvline(
            float(values.critical_mach[0]), color=color, linestyle="--", linewidth=0.7
        )
    axes[1, 0].set(
        xlabel="Mach number",
        ylabel="Drag coefficient",
        title="CL-dependent critical Mach and drag rise",
    )
    axes[1, 0].legend(ncols=1)

    time_h = qar["elapsed_time_s"] / 3600.0
    axes[1, 1].plot(time_h, qar["actual_mach"], color="black", label="QAR Mach")
    axes[1, 1].plot(
        time_h,
        qar["critical_mach"],
        color="#0072B2",
        linestyle="--",
        label="model critical Mach",
    )
    force_axis = axes[1, 1].twinx()
    force_axis.plot(
        time_h,
        qar["wave_drag_n"] / 1_000.0,
        color="#D55E00",
        alpha=0.85,
        label="wave-drag increment",
    )
    axes[1, 1].set(
        xlabel="Elapsed time (h)",
        ylabel="Mach number",
        title="FLT0003 cruise-envelope diagnostic",
    )
    force_axis.set_ylabel("Wave-drag increment (kN)")
    lines = axes[1, 1].lines + force_axis.lines
    axes[1, 1].legend(
        lines,
        [line.get_label() for line in lines],
        loc="center right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.75,
    )

    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
            color="white" if label in "AB" else "black",
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.55)
    figure.suptitle("A320 angle-based lift and experimental wave-drag model", fontsize=11)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    aircraft = load_aircraft("A320")
    lift_model = _lift_model(args.lift_parameters)
    polar, drag_model = _models()
    surface = _surface(lift_model, drag_model)
    qar = _qar_points(args.representative, polar, drag_model)
    wave = drag_model.wave_drag
    metrics = {
        "sample_count": len(qar),
        "mach_minimum": float(qar["actual_mach"].min()),
        "mach_maximum": float(qar["actual_mach"].max()),
        "lift_coefficient_p05": float(qar["lift_coefficient"].quantile(0.05)),
        "lift_coefficient_median": float(qar["lift_coefficient"].median()),
        "lift_coefficient_p95": float(qar["lift_coefficient"].quantile(0.95)),
        "critical_mach_median": float(qar["critical_mach"].median()),
        "wave_drag_coefficient_median": float(qar["wave_drag_coefficient"].median()),
        "wave_drag_coefficient_p95": float(qar["wave_drag_coefficient"].quantile(0.95)),
        "wave_drag_force_median_kn": float(qar["wave_drag_n"].median() / 1_000.0),
        "wave_drag_force_p95_kn": float(qar["wave_drag_n"].quantile(0.95) / 1_000.0),
        "wave_drag_fraction_median": float(qar["wave_drag_fraction"].median()),
    }
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "implemented-opt-in-not-physically-validated",
        "current_force_path": {
            "angle_of_attack": "calibrated decoded pitch minus flight-path angle",
            "lift": "effective CL(alpha, Mach)",
            "released_drag": "CD0 + k CL^2",
            "experimental_drag": "CD0 + k CL^2 + wave(Mach, CL)",
            "pitch_affects_drag": "indirectly through CL; no independent CD(alpha, Mach) surface",
        },
        "model": {
            "wing_area_m2": float(aircraft.wing.area),
            "wing_sweep_deg": float(aircraft.wing.sweep),
            "thickness_to_chord_ratio": float(
                aircraft.wing.t_c
            ),
            "zero_lift_drag_coefficient": polar.zero_lift_drag_coefficient,
            "induced_drag_factor": polar.induced_drag_factor,
            "technology_factor": wave.technology_factor,
            "wave_equation": "20 * max(M - Mcrit(CL), 0)^4",
            "source": "OpenAP implementation of Gur, Mason, and Schetz (2010), DOI 10.2514/1.47557",
            "implementation": "src/aerox/models/lift/core.py",
            "plant_selection": "PointMassModel.drag_model (opt-in)",
            "vectorized": True,
        },
        "representative_cruise_diagnostic": metrics,
        "release": {
            "default_aircraft_drag_model_changed": False,
            "wave_drag_approved_for_trajectory_release": False,
            "reason": "QAR N1-to-thrust map was identified conditional on the parabolic drag prior; QAR alone does not separately identify transonic drag rise and engine thrust",
        },
        "artifacts": {
            "surface": str(args.surface),
            "qar_points": str(args.qar_points),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }
    args.surface.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(args.surface, index=False)
    qar.to_csv(args.qar_points, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _plot(surface=surface, qar=qar, drag_model=drag_model, prefix=args.figure_prefix)
    report = f"""# A320 angle-based lift and experimental wave drag

## Current model

Lift is already angle based. The attitude plant calculates an effective angle
of attack from calibrated pitch minus flight-path angle and evaluates the
identified `CL(alpha, Mach)` map. Released drag is still the parabolic polar
`CD = CD0 + k CL²`; pitch therefore changes drag only indirectly through lift.

The new opt-in model evaluates
`CD = CD0 + k CL² + 20 max(M - Mcrit(CL), 0)^4` with the A320 sweep and
thickness/chord priors. It is vectorized and shared by the physical plant,
profile planner, and ECON calculation primitives.

![A320 angle-based lift and wave-drag audit](figures/a320-wave-drag-model.png)

Figure 1. Effective lift surface, wave-drag surface, coefficient breakdown, and
FLT0003 cruise-envelope diagnostic. A
[vector PDF](figures/a320-wave-drag-model.pdf) is available.

## Representative magnitude

Across {metrics['sample_count']} thinned FLT0003 cruise points, median inferred
level-flight CL is {metrics['lift_coefficient_median']:.3f}, median critical Mach
is {metrics['critical_mach_median']:.3f}, and the generic prior adds a median
{metrics['wave_drag_force_median_kn']:.2f} kN of drag. Its 95th-percentile
increment is {metrics['wave_drag_force_p95_kn']:.2f} kN.

## Release decision

The component is implemented but not enabled by default. The existing QAR
N1-to-effective-thrust map was identified while assuming the parabolic polar.
Adding wave drag without an independent engine-thrust reference would double
count a term that the effective thrust map may already have absorbed. QAR alone
cannot separate transonic drag rise from installed engine thrust at the same
condition.

The next physical validation should use an independently sourced CFM56-5B4
installed-thrust deck or published/CFD transonic drag data. Until then, the
model is suitable for sensitivity analysis and ECON research, not release as
the default A320 plant.
"""
    args.report.write_text(report)
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["representative_cruise_diagnostic"], sort_keys=False))


if __name__ == "__main__":
    main()
