#!/usr/bin/env python3
"""Fit the bounded VSPAERO CL(alpha, Mach) surface on A320 fit flights only."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.models import TabulatedLiftSurface
from tools.audit_a320_vspaero_angle_lift_residual import REGISTRATION_CONTRAST
from tools.calibrate_a320_effective_thrust import (
    _phase_equal_weights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.identify_a320_vspaero_angle_lift import (
    FlightArrays,
    PITCH_SCALE_BOUNDS,
    WING_AREA_M2,
    _fit_baseline,
    _flight_arrays,
    _predict_baseline,
)
from tools.replay_a320_point_mass import (
    _load_point_mass_flights,
    _read_horizontal_parameters,
)

UPDATED = "2026-08-05"
CL_SURFACE_MULTIPLIERS = (0.90, 1.00, 1.10)
MAX_LOFO_RMSE_RATIO = 1.20
MAX_SCALE_LOFO_RANGE_FRACTION = 0.15
MIN_MODEL_ALPHA_ENVELOPE_FRACTION = 0.995
MAX_REGISTRATION_BIAS_DIFFERENCE_CL = 0.02
MAX_REGISTRATION_NUISANCE_LOFO_RANGE_DEG = 0.10
PARAMETER_LOWER = np.array([0.75, np.deg2rad(0.0), np.deg2rad(-3.0), np.deg2rad(-1.0)])
PARAMETER_UPPER = np.array([1.50, np.deg2rad(4.0), np.deg2rad(3.0), np.deg2rad(1.0)])
INITIAL = np.array([1.05, np.deg2rad(2.1), np.deg2rad(-0.5), np.deg2rad(-0.30)])


@dataclass(frozen=True)
class SurfaceFit:
    parameters: np.ndarray
    residuals: np.ndarray
    weighted_cl_rmse: float
    robust_scale_cl: float
    converged: bool
    iterations: int
    cl_surface_multiplier: float

    @property
    def pitch_scale(self) -> float:
        return float(self.parameters[0])

    @property
    def combined_offset_rad(self) -> float:
        return float(self.parameters[1])

    @property
    def mach_offset_rad_per_mach(self) -> float:
        return float(self.parameters[2])

    @property
    def registration_angle_difference_rad(self) -> float:
        return float(self.parameters[3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/a320/a320.parquet"))
    parser.add_argument("--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv"))
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
        "--lift-surface",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-alpha-support.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"),
    )
    parser.add_argument(
        "--lofo",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-lofo.csv"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-sensitivity.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-lift-surface-identification"),
    )
    return parser.parse_args()


def _model_alpha(flight: FlightArrays, parameters: np.ndarray) -> np.ndarray:
    return (
        parameters[0] * flight.pitch_zero_rad
        + parameters[1]
        + parameters[2] * (flight.mach - 0.6)
        + parameters[3] * REGISTRATION_CONTRAST[flight.registration]
        - flight.gamma_rad
    )


def _surface_extrapolated(
    surface: TabulatedLiftSurface,
    alpha: np.ndarray,
    mach: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Bilinear evaluator used only inside bounded optimization.

    Final support is gated separately. Allowing linear trial extrapolation
    keeps the optimizer differentiable when an intermediate step crosses a
    table edge; no extrapolated model is eligible for release.
    """

    mach_index = np.clip(
        np.searchsorted(surface.mach, mach, side="right") - 1,
        0,
        len(surface.mach) - 2,
    )
    alpha_index = np.clip(
        np.searchsorted(surface.angle_of_attack_rad, alpha, side="right") - 1,
        0,
        len(surface.angle_of_attack_rad) - 2,
    )
    mach_low = surface.mach[mach_index]
    mach_high = surface.mach[mach_index + 1]
    alpha_low = surface.angle_of_attack_rad[alpha_index]
    alpha_high = surface.angle_of_attack_rad[alpha_index + 1]
    mw = (mach - mach_low) / (mach_high - mach_low)
    aw = (alpha - alpha_low) / (alpha_high - alpha_low)
    lower = (1.0 - aw) * surface.lift_grid[mach_index, alpha_index] + aw * surface.lift_grid[mach_index, alpha_index + 1]
    upper = (1.0 - aw) * surface.lift_grid[mach_index + 1, alpha_index] + aw * surface.lift_grid[mach_index + 1, alpha_index + 1]
    raw = (1.0 - mw) * lower + mw * upper

    reference_alpha_index = int(
        np.argmin(np.abs(surface.angle_of_attack_rad - np.deg2rad(2.0)))
    )
    reference = np.interp(mach, surface.mach, surface.lift_grid[:, reference_alpha_index])
    return reference + multiplier * (raw - reference)


def _stack(flights: list[FlightArrays]) -> tuple[np.ndarray, np.ndarray]:
    weights = np.concatenate(
        [_phase_equal_weights(flight.phase) / len(flights) for flight in flights]
    )
    targets = np.concatenate([flight.target_cl for flight in flights])
    return targets, weights


def _predictions(
    flights: list[FlightArrays],
    surface: TabulatedLiftSurface,
    parameters: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    return np.concatenate(
        [
            _surface_extrapolated(
                surface, _model_alpha(flight, parameters), flight.mach, multiplier
            )
            for flight in flights
        ]
    )


def _fit(
    flights: list[FlightArrays],
    surface: TabulatedLiftSurface,
    multiplier: float = 1.0,
) -> SurfaceFit:
    target, base_weights = _stack(flights)
    robust_weights = np.ones(len(target))
    parameters = np.clip(INITIAL, PARAMETER_LOWER, PARAMETER_UPPER)
    converged = False
    iterations = 0
    for iterations in range(1, 16):
        weights = np.sqrt(base_weights * robust_weights)

        def objective(candidate: np.ndarray) -> np.ndarray:
            return weights * (
                _predictions(flights, surface, candidate, multiplier) - target
            )

        result = least_squares(
            objective,
            parameters,
            bounds=(PARAMETER_LOWER, PARAMETER_UPPER),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        updated = result.x
        residuals = _predictions(flights, surface, updated, multiplier) - target
        centered = residuals - np.median(residuals)
        scale = 1.4826 * np.median(np.abs(centered)) + 1e-12
        cutoff = 1.5 * scale
        robust_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residuals), 1e-15))
        if np.max(np.abs(updated - parameters)) <= 1e-9:
            parameters = updated
            converged = bool(result.success)
            break
        parameters = updated
        converged = bool(result.success)
    residuals = _predictions(flights, surface, parameters, multiplier) - target
    return SurfaceFit(
        parameters=parameters,
        residuals=residuals,
        weighted_cl_rmse=float(np.sqrt(np.average(residuals**2, weights=base_weights))),
        robust_scale_cl=float(scale),
        converged=converged,
        iterations=iterations,
        cl_surface_multiplier=multiplier,
    )


def _flight_metrics(
    flight: FlightArrays,
    fit: SurfaceFit,
    surface: TabulatedLiftSurface,
) -> dict[str, Any]:
    alpha = _model_alpha(flight, fit.parameters)
    prediction = _surface_extrapolated(
        surface, alpha, flight.mach, fit.cl_surface_multiplier
    )
    residual = prediction - flight.target_cl
    inside = (
        (alpha >= surface.minimum_alpha_rad)
        & (alpha <= surface.maximum_alpha_rad)
        & (flight.mach >= surface.minimum_mach)
        & (flight.mach <= surface.maximum_mach)
    )
    return {
        "flight": flight.key,
        "registration": flight.registration,
        "samples": len(alpha),
        "surface_cl_rmse": float(np.sqrt(np.mean(residual**2))),
        "surface_cl_bias": float(np.mean(residual)),
        "pitch_scale": fit.pitch_scale,
        "combined_offset_deg": math.degrees(fit.combined_offset_rad),
        "mach_offset_deg_per_mach": math.degrees(fit.mach_offset_rad_per_mach),
        "registration_angle_difference_deg": math.degrees(
            fit.registration_angle_difference_rad
        ),
        "model_alpha_in_envelope_fraction": float(np.mean(inside)),
        "model_alpha_min_deg": float(np.rad2deg(np.min(alpha))),
        "model_alpha_max_deg": float(np.rad2deg(np.max(alpha))),
        "fit_converged": fit.converged,
    }


def build_identification(
    flights: list[FlightArrays], surface: TabulatedLiftSurface
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lofo: list[dict[str, Any]] = []
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        fit = _fit(training, surface)
        row = _flight_metrics(held_out, fit, surface)
        baseline = _fit_baseline(training)
        baseline_residual = _predict_baseline(held_out, baseline) - held_out.target_cl
        row["baseline_cl_rmse"] = float(np.sqrt(np.mean(baseline_residual**2)))
        row["baseline_cl_bias"] = float(np.mean(baseline_residual))
        row["surface_to_baseline_rmse_ratio"] = row["surface_cl_rmse"] / row["baseline_cl_rmse"]
        lofo.append(row)
    nominal = _fit(flights, surface)
    sensitivity: list[dict[str, Any]] = []
    for multiplier in CL_SURFACE_MULTIPLIERS:
        fit = _fit(flights, surface, multiplier)
        sensitivity.append(
            {
                "cl_surface_slope_multiplier": multiplier,
                "pitch_scale": fit.pitch_scale,
                "combined_offset_deg": math.degrees(fit.combined_offset_rad),
                "mach_offset_deg_per_mach": math.degrees(fit.mach_offset_rad_per_mach),
                "registration_angle_difference_deg": math.degrees(fit.registration_angle_difference_rad),
                "fit_weighted_cl_rmse": fit.weighted_cl_rmse,
                "fit_robust_scale_cl": fit.robust_scale_cl,
                "fit_converged": fit.converged,
            }
        )

    baseline_mean = float(np.mean([row["baseline_cl_rmse"] for row in lofo]))
    surface_mean = float(np.mean([row["surface_cl_rmse"] for row in lofo]))
    scales = np.asarray([row["pitch_scale"] for row in lofo])
    reg_nuisance = np.asarray([row["registration_angle_difference_deg"] for row in lofo])
    envelope_fraction = float(np.average(
        [row["model_alpha_in_envelope_fraction"] for row in lofo],
        weights=[row["samples"] for row in lofo],
    ))
    frame = pd.DataFrame(lofo)
    registration_bias = frame.groupby("registration")["surface_cl_bias"].mean()
    registration_bias_difference = float(registration_bias.max() - registration_bias.min())
    gates = {
        "lofo_rmse_ratio": {"value": surface_mean / baseline_mean, "maximum": MAX_LOFO_RMSE_RATIO, "passed": surface_mean / baseline_mean <= MAX_LOFO_RMSE_RATIO},
        "pitch_scale": {"value": nominal.pitch_scale, "lower": PITCH_SCALE_BOUNDS[0], "upper": PITCH_SCALE_BOUNDS[1], "passed": PITCH_SCALE_BOUNDS[0] <= nominal.pitch_scale <= PITCH_SCALE_BOUNDS[1]},
        "lofo_scale_range_fraction": {"value": float((scales.max()-scales.min())/np.median(scales)), "maximum": MAX_SCALE_LOFO_RANGE_FRACTION, "passed": bool((scales.max()-scales.min())/np.median(scales) <= MAX_SCALE_LOFO_RANGE_FRACTION)},
        "model_alpha_envelope_fraction": {"value": envelope_fraction, "minimum": MIN_MODEL_ALPHA_ENVELOPE_FRACTION, "passed": envelope_fraction >= MIN_MODEL_ALPHA_ENVELOPE_FRACTION},
        "registration_bias_difference_cl": {"value": registration_bias_difference, "maximum": MAX_REGISTRATION_BIAS_DIFFERENCE_CL, "passed": registration_bias_difference <= MAX_REGISTRATION_BIAS_DIFFERENCE_CL},
        "registration_nuisance_lofo_range_deg": {"value": float(reg_nuisance.max()-reg_nuisance.min()), "maximum": MAX_REGISTRATION_NUISANCE_LOFO_RANGE_DEG, "passed": bool(reg_nuisance.max()-reg_nuisance.min() <= MAX_REGISTRATION_NUISANCE_LOFO_RANGE_DEG)},
        "optimizer_convergence": {"value": bool(nominal.converged and all(row["fit_converged"] for row in lofo)), "required": True, "passed": bool(nominal.converged and all(row["fit_converged"] for row in lofo))},
    }
    passed = all(gate["passed"] for gate in gates.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "fit-gate-passed-validation-still-closed" if passed else "fit-gate-failed-validation-remains-closed",
        "data_boundary": {"fit_flights_only": True, "validation_flights_loaded": False, "flights": [flight.key for flight in flights], "samples": int(sum(len(flight.mach) for flight in flights))},
        "model": {
            "equation": "CL = bilinear_VSPAERO_CL(scale*pitch_zero + offset0 + offsetM*(M-0.6) + registration_contrast*offset_registration - gamma, M)",
            "surface": "docs/modeling/a320-vspaero-alpha-support.csv",
            "alpha_support_deg": [math.degrees(surface.minimum_alpha_rad), math.degrees(surface.maximum_alpha_rad)],
            "mach_support": [surface.minimum_mach, surface.maximum_mach],
            "pitch_scale": nominal.pitch_scale,
            "combined_offset_deg": math.degrees(nominal.combined_offset_rad),
            "mach_offset_deg_per_mach": math.degrees(nominal.mach_offset_rad_per_mach),
            "registration_angle_difference_deg": math.degrees(nominal.registration_angle_difference_rad),
            "registration_contrast": REGISTRATION_CONTRAST,
            "fit_weighted_cl_rmse": nominal.weighted_cl_rmse,
            "fit_robust_scale_cl": nominal.robust_scale_cl,
            "parameter_bounds": {"pitch_scale": list(PITCH_SCALE_BOUNDS), "combined_offset_deg": [0.0, 4.0], "mach_offset_deg_per_mach": [-3.0, 3.0], "registration_angle_difference_deg": [-1.0, 1.0]},
        },
        "comparison": {"baseline_mean_lofo_cl_rmse": baseline_mean, "surface_mean_lofo_cl_rmse": surface_mean, "surface_to_baseline_lofo_rmse_ratio": surface_mean/baseline_mean, "lofo_table": "docs/modeling/a320-vspaero-lift-surface-lofo.csv", "sensitivity_table": "docs/modeling/a320-vspaero-lift-surface-sensitivity.csv"},
        "gates": gates,
        "decision": {"fit_gate_passed": passed, "open_existing_validation_panel_next": passed, "change_current_replay_plant": False, "physical_pitch_scale_released": False, "tail_specific_physical_aerodynamics": False, "reason": "The registration contrast is a frozen measurement/force-reconstruction nuisance; only an independently confirmed surface model may replace the empirical replay lift map."},
        "limitations": ["The surface is an inviscid symmetric thin-surface geometry prior, not an A320 aerodynamic database.", "The registration correction is not a physical aerodynamic difference between aircraft.", "The lift target remains conditional on reconstructed force balance, mass, air data, and omitted vertical wind.", "Drag remains the separate Aerox viscous polar."],
    }
    return document, lofo, sensitivity


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(document: dict[str, Any]) -> str:
    model = document["model"]
    comparison = document["comparison"]
    gate_rows = "\n".join(
        f"| {name} | {gate['value'] if isinstance(gate['value'], bool) else f'{gate['value']:.4f}'} | {'pass' if gate['passed'] else 'fail'} |"
        for name, gate in document["gates"].items()
    )
    return f"""# A320 VSPAERO lift-surface identification

## Outcome

The revised fit-flight-only model **{'passes' if document['decision']['fit_gate_passed'] else 'does not pass'}** its predeclared gate. It uses the bounded sampled VSPAERO `CL(alpha, Mach)` surface rather than extending one local derivative. The shared pitch scale is {model['pitch_scale']:.4f}; the AFZ−AFC measurement nuisance is {model['registration_angle_difference_deg']:.3f}°.

Mean complete-flight LOFO CL RMSE is {comparison['surface_mean_lofo_cl_rmse']:.4f}, versus {comparison['baseline_mean_lofo_cl_rmse']:.4f} for the empirical baseline (ratio {comparison['surface_to_baseline_lofo_rmse_ratio']:.3f}).

| Gate | Value | Result |
|---|---:|---:|
{gate_rows}

## Boundary and decision

The registration contrast remains a sensor/force-reconstruction nuisance and
does not create separate aerodynamics for TAIL-A and TAIL-B. Drag remains the
existing Aerox viscous polar. The replay plant is unchanged.

- Open the already-separated validation panel next: **{'yes' if document['decision']['open_existing_validation_panel_next'] else 'no'}**.
- Change the replay plant now: **no**.
- Release the pitch scale as a physical sensor calibration: **no**.
"""


def _plot(lofo: list[dict[str, Any]], sensitivity: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["flight"].split("/", 1)[1] for row in lofo]
    x = np.arange(len(lofo))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    flat = axes.ravel()
    width = 0.38
    flat[0].bar(x-width/2, [row["baseline_cl_rmse"] for row in lofo], width, color="#0072B2", hatch="//", label="empirical")
    flat[0].bar(x+width/2, [row["surface_cl_rmse"] for row in lofo], width, color="#D55E00", hatch="..", label="VSPAERO surface")
    flat[0].set_xticks(x, labels, rotation=50, ha="right")
    flat[0].set(title="Complete-flight holdout", ylabel="CL RMSE")
    flat[0].legend(frameon=False)
    flat[1].plot(x, [row["pitch_scale"] for row in lofo], "o-", color="#009E73")
    flat[1].axhspan(*PITCH_SCALE_BOUNDS, color="#009E73", alpha=0.08)
    flat[1].set_xticks(x, labels, rotation=50, ha="right")
    flat[1].set(title="Pitch-scale stability", ylabel="Scale vs current decoder")
    support = np.asarray([row["model_alpha_in_envelope_fraction"] for row in lofo])
    flat[2].bar(x, support, color=np.where(support >= MIN_MODEL_ALPHA_ENVELOPE_FRACTION, "#009E73", "#E69F00"))
    flat[2].axhline(MIN_MODEL_ALPHA_ENVELOPE_FRACTION, color="black", linestyle="--", linewidth=1)
    flat[2].set_xticks(x, labels, rotation=50, ha="right")
    flat[2].set_ylim(0.97, 1.001)
    flat[2].set(title="Bounded surface support", ylabel="In-envelope sample fraction")
    multipliers = [row["cl_surface_slope_multiplier"] for row in sensitivity]
    flat[3].plot(multipliers, [row["pitch_scale"] for row in sensitivity], "o-", color="#CC79A7", label="pitch scale")
    flat[3].set(title="Surface-slope sensitivity", xlabel="Slope multiplier", ylabel="Fitted pitch scale")
    for label, axis in zip("ABCD", flat, strict=True):
        axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 fit-only VSPAERO lift-surface identification")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    flights = _load_point_mass_flights(fit_selection, raw_dir=args.raw_dir, processed_path=args.processed, vertical=vertical, horizontal=horizontal, tas_scale=tas_scale, wing_area_m2=WING_AREA_M2)
    arrays = _flight_arrays(flights, gamma_bias_rad=math.radians(vertical.effective_gamma_bias_deg))
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(args.lift_surface)
    document, lofo, sensitivity = build_identification(arrays, surface)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_csv(args.lofo, lofo)
    _write_csv(args.sensitivity, sensitivity)
    args.report.write_text(_render_report(document))
    _plot(lofo, sensitivity, args.figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])


if __name__ == "__main__":
    main()
