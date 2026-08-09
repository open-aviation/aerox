#!/usr/bin/env python3
"""Fit an A320 effective pitch scale under a VSPAERO lift-slope prior.

Only the established calibration flights are used. The existing validation
flights are not loaded unless a later, separately authorized confirmation gate
is run. The fitted offset combines pitch/AOA zero, wing incidence, camber, and
other model-form effects and must not be called a sensor bias.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox.models import (
    EffectiveLiftMap,
    TabulatedStaticStabilityPrior,
    effective_lift_design,
)
from aerox.modeling.identification import RobustLinearFit, robust_weighted_linear_fit
from tools.calibrate_a320_effective_thrust import (
    FIT_THINNING_S,
    _phase_equal_weights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.replay_a320_point_mass import (
    LIFT_FEATURE_SET,
    _load_point_mass_flights,
    _read_horizontal_parameters,
)

UPDATED = "2026-08-05"
WING_AREA_M2 = 122.0
MACH_RANGE = (0.20, 0.78)
EFFECTIVE_ALPHA_RANGE_DEG = (-1.0, 6.0)
CL_ALPHA_MULTIPLIERS = (0.90, 1.00, 1.10)
MAX_LOFO_RMSE_RATIO = 1.20
MAX_SCALE_LOFO_RANGE_FRACTION = 0.15
PITCH_SCALE_BOUNDS = (0.75, 1.50)
MIN_MODEL_ALPHA_ENVELOPE_FRACTION = 0.95
MAX_REGISTRATION_BIAS_DIFFERENCE_CL = 0.02


@dataclass(frozen=True)
class FlightArrays:
    key: str
    registration: str
    pitch_zero_rad: np.ndarray
    gamma_rad: np.ndarray
    mach: np.ndarray
    target_cl: np.ndarray
    phase: np.ndarray


@dataclass(frozen=True)
class AnchoredLiftFit:
    pitch_scale: float
    combined_offset_rad: float
    mach_offset_rad_per_mach: float
    cl_alpha_multiplier: float
    result: RobustLinearFit


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
        "--horizontal-calibration",
        type=Path,
        default=Path("docs/modeling/a320-horizontal-kinematic-calibration.yml"),
    )
    parser.add_argument(
        "--vspaero-results",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-prior.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-identification.yml"),
    )
    parser.add_argument(
        "--lofo",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-lofo.csv"),
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-sensitivity.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-identification.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-angle-lift-identification"),
    )
    return parser.parse_args()


def _flight_arrays(
    flights: list[Any], *, gamma_bias_rad: float
) -> list[FlightArrays]:
    result: list[FlightArrays] = []
    for flight in flights:
        base = flight.base
        alpha_effective = flight.effective_pitch_angle_rad - base.flight_path_angle_rad
        select = (
            (base.sample_index % FIT_THINNING_S == 0)
            & (base.mach >= MACH_RANGE[0])
            & (base.mach <= MACH_RANGE[1])
            & (np.rad2deg(alpha_effective) >= EFFECTIVE_ALPHA_RANGE_DEG[0])
            & (np.rad2deg(alpha_effective) <= EFFECTIVE_ALPHA_RANGE_DEG[1])
        )
        if np.count_nonzero(select) < 50:
            raise ValueError(f"too few in-envelope samples for {flight.key}")
        result.append(
            FlightArrays(
                key=flight.key,
                registration=flight.key.split("/", 1)[0],
                pitch_zero_rad=(flight.effective_pitch_angle_rad[select] - gamma_bias_rad),
                gamma_rad=base.flight_path_angle_rad[select],
                mach=base.mach[select],
                target_cl=base.lift_coefficient[select],
                phase=base.phase[select],
            )
        )
    return result


def _prior_values(
    prior: TabulatedStaticStabilityPrior,
    mach: np.ndarray,
    cl_alpha_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    cl_reference = np.interp(mach, prior.mach, prior.reference_lift)
    cl_alpha = cl_alpha_multiplier * np.interp(mach, prior.mach, prior.lift_alpha)
    return cl_reference, cl_alpha


def _stack(
    flights: list[FlightArrays], prior: TabulatedStaticStabilityPrior, multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        cl_reference, cl_alpha = _prior_values(prior, flight.mach, multiplier)
        mach_centered = flight.mach - 0.6
        designs.append(
            np.column_stack(
                (
                    cl_alpha * flight.pitch_zero_rad,
                    cl_alpha,
                    cl_alpha * mach_centered,
                )
            )
        )
        targets.append(
            flight.target_cl
            - cl_reference
            + cl_alpha * (flight.gamma_rad + prior.reference_alpha_rad)
        )
        weights.append(_phase_equal_weights(flight.phase) / len(flights))
    return np.vstack(designs), np.concatenate(targets), np.concatenate(weights)


def _fit_anchored(
    flights: list[FlightArrays],
    prior: TabulatedStaticStabilityPrior,
    multiplier: float = 1.0,
) -> AnchoredLiftFit:
    design, target, weights = _stack(flights, prior, multiplier)
    result = robust_weighted_linear_fit(design, target, weights)
    return AnchoredLiftFit(
        pitch_scale=float(result.coefficients[0]),
        combined_offset_rad=float(result.coefficients[1]),
        mach_offset_rad_per_mach=float(result.coefficients[2]),
        cl_alpha_multiplier=multiplier,
        result=result,
    )


def _predict_anchored(
    flight: FlightArrays,
    fit: AnchoredLiftFit,
    prior: TabulatedStaticStabilityPrior,
) -> tuple[np.ndarray, np.ndarray]:
    cl_reference, cl_alpha = _prior_values(prior, flight.mach, fit.cl_alpha_multiplier)
    model_alpha = (
        fit.pitch_scale * flight.pitch_zero_rad
        + fit.combined_offset_rad
        + fit.mach_offset_rad_per_mach * (flight.mach - 0.6)
        - flight.gamma_rad
    )
    prediction = cl_reference + cl_alpha * (model_alpha - prior.reference_alpha_rad)
    return prediction, model_alpha


def _fit_baseline(flights: list[FlightArrays]) -> EffectiveLiftMap:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        alpha = flight.pitch_zero_rad - flight.gamma_rad
        designs.append(effective_lift_design(alpha, flight.mach, feature_set=LIFT_FEATURE_SET))
        targets.append(flight.target_cl)
        weights.append(_phase_equal_weights(flight.phase) / len(flights))
    result = robust_weighted_linear_fit(
        np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
    )
    return EffectiveLiftMap(LIFT_FEATURE_SET, result.coefficients)


def _predict_baseline(flight: FlightArrays, model: EffectiveLiftMap) -> np.ndarray:
    return model.raw_lift_coefficient(
        flight.pitch_zero_rad - flight.gamma_rad, flight.mach
    )


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    residual = prediction - target
    return float(np.sqrt(np.mean(residual**2))), float(np.mean(residual))


def _lofo(
    flights: list[FlightArrays], prior: TabulatedStaticStabilityPrior
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        anchored = _fit_anchored(training, prior)
        baseline = _fit_baseline(training)
        anchored_prediction, model_alpha = _predict_anchored(held_out, anchored, prior)
        baseline_prediction = _predict_baseline(held_out, baseline)
        anchored_rmse, anchored_bias = _error_metrics(
            anchored_prediction, held_out.target_cl
        )
        baseline_rmse, baseline_bias = _error_metrics(
            baseline_prediction, held_out.target_cl
        )
        inside = (
            (model_alpha >= prior.minimum_alpha_rad)
            & (model_alpha <= prior.maximum_alpha_rad)
        )
        rows.append(
            {
                "flight": held_out.key,
                "registration": held_out.registration,
                "samples": len(held_out.mach),
                "baseline_cl_rmse": baseline_rmse,
                "anchored_cl_rmse": anchored_rmse,
                "anchored_to_baseline_rmse_ratio": anchored_rmse / baseline_rmse,
                "baseline_cl_bias": baseline_bias,
                "anchored_cl_bias": anchored_bias,
                "pitch_scale": anchored.pitch_scale,
                "combined_offset_deg": math.degrees(anchored.combined_offset_rad),
                "mach_offset_deg_per_mach": math.degrees(
                    anchored.mach_offset_rad_per_mach
                ),
                "model_alpha_in_envelope_fraction": float(np.mean(inside)),
                "model_alpha_min_deg": float(np.rad2deg(np.min(model_alpha))),
                "model_alpha_max_deg": float(np.rad2deg(np.max(model_alpha))),
            }
        )
    return rows


def _sensitivity(
    flights: list[FlightArrays], prior: TabulatedStaticStabilityPrior
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for multiplier in CL_ALPHA_MULTIPLIERS:
        fit = _fit_anchored(flights, prior, multiplier)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for flight in flights:
            prediction, _ = _predict_anchored(flight, fit, prior)
            predictions.append(prediction)
            targets.append(flight.target_cl)
        rmse, bias = _error_metrics(np.concatenate(predictions), np.concatenate(targets))
        rows.append(
            {
                "cl_alpha_multiplier": multiplier,
                "pitch_scale": fit.pitch_scale,
                "combined_offset_deg": math.degrees(fit.combined_offset_rad),
                "mach_offset_deg_per_mach": math.degrees(
                    fit.mach_offset_rad_per_mach
                ),
                "fit_cl_rmse": rmse,
                "fit_cl_bias": bias,
            }
        )
    return rows


def _registration_bias_difference(rows: list[dict[str, Any]]) -> float:
    frame = pd.DataFrame(rows)
    registration_bias = frame.groupby("registration")["anchored_cl_bias"].mean()
    return float(registration_bias.max() - registration_bias.min())


def build_identification(
    flights: list[FlightArrays], prior: TabulatedStaticStabilityPrior
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, float]]]:
    lofo = _lofo(flights, prior)
    sensitivity = _sensitivity(flights, prior)
    nominal = _fit_anchored(flights, prior)
    baseline_mean = float(np.mean([row["baseline_cl_rmse"] for row in lofo]))
    anchored_mean = float(np.mean([row["anchored_cl_rmse"] for row in lofo]))
    scales = np.array([row["pitch_scale"] for row in lofo])
    scale_range_fraction = float((scales.max() - scales.min()) / np.median(scales))
    envelope_fraction = float(
        np.average(
            [row["model_alpha_in_envelope_fraction"] for row in lofo],
            weights=[row["samples"] for row in lofo],
        )
    )
    registration_bias_difference = _registration_bias_difference(lofo)
    gates = {
        "lofo_rmse_ratio": {
            "value": anchored_mean / baseline_mean,
            "maximum": MAX_LOFO_RMSE_RATIO,
            "passed": anchored_mean / baseline_mean <= MAX_LOFO_RMSE_RATIO,
        },
        "pitch_scale": {
            "value": nominal.pitch_scale,
            "lower": PITCH_SCALE_BOUNDS[0],
            "upper": PITCH_SCALE_BOUNDS[1],
            "passed": PITCH_SCALE_BOUNDS[0] <= nominal.pitch_scale <= PITCH_SCALE_BOUNDS[1],
        },
        "lofo_scale_range_fraction": {
            "value": scale_range_fraction,
            "maximum": MAX_SCALE_LOFO_RANGE_FRACTION,
            "passed": scale_range_fraction <= MAX_SCALE_LOFO_RANGE_FRACTION,
        },
        "model_alpha_envelope_fraction": {
            "value": envelope_fraction,
            "minimum": MIN_MODEL_ALPHA_ENVELOPE_FRACTION,
            "passed": envelope_fraction >= MIN_MODEL_ALPHA_ENVELOPE_FRACTION,
        },
        "registration_bias_difference_cl": {
            "value": registration_bias_difference,
            "maximum": MAX_REGISTRATION_BIAS_DIFFERENCE_CL,
            "passed": registration_bias_difference <= MAX_REGISTRATION_BIAS_DIFFERENCE_CL,
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "fit-gate-passed-validation-still-closed" if passed else "fit-gate-failed-validation-remains-closed",
        "data_boundary": {
            "fit_flights_only": True,
            "validation_flights_loaded": False,
            "flights": [flight.key for flight in flights],
            "registrations": sorted({flight.registration for flight in flights}),
            "samples": int(sum(len(flight.mach) for flight in flights)),
            "thinning_s": FIT_THINNING_S,
            "mach_range": list(MACH_RANGE),
            "effective_alpha_prefilter_deg": list(EFFECTIVE_ALPHA_RANGE_DEG),
        },
        "model": {
            "equation": "CL = CLref(M) + CLalpha_vsp(M) * [scale*pitch_zero + offset0 + offsetM*(M-0.6) - gamma - alpha_ref]",
            "pitch_zero_definition": "effective_pitch minus frozen vertical gamma-bias assignment",
            "pitch_scale": nominal.pitch_scale,
            "combined_offset_deg": math.degrees(nominal.combined_offset_rad),
            "mach_offset_deg_per_mach": math.degrees(nominal.mach_offset_rad_per_mach),
            "combined_offset_interpretation": "joint pitch/AOA zero, wing incidence/camber, and omitted-model offset; not a sensor bias",
            "vspaero_cl_alpha_multiplier": nominal.cl_alpha_multiplier,
            "fit_weighted_cl_rmse": nominal.result.base_weighted_rmse,
            "fit_robust_scale_cl": nominal.result.robust_scale,
            "normal_matrix_condition": nominal.result.normal_matrix_condition,
        },
        "comparison": {
            "baseline": "refitted linear_alpha_mach effective lift map on identical in-envelope fit samples",
            "baseline_mean_lofo_cl_rmse": baseline_mean,
            "anchored_mean_lofo_cl_rmse": anchored_mean,
            "anchored_to_baseline_lofo_rmse_ratio": anchored_mean / baseline_mean,
            "lofo_table": "docs/modeling/a320-vspaero-angle-lift-lofo.csv",
            "sensitivity_table": "docs/modeling/a320-vspaero-angle-lift-sensitivity.csv",
        },
        "gates": gates,
        "decision": {
            "fit_gate_passed": passed,
            "open_existing_validation_panel_next": passed,
            "change_current_replay_plant": False,
            "physical_pitch_scale_released": False,
            "reason": "a fit-only constrained model can qualify for confirmation, but validation and model-form uncertainty are required before changing the replay or physical simulator",
        },
        "limitations": [
            "The combined offset is structurally confounded and is not a decoded pitch or AOA sensor bias.",
            "VSPAERO CLalpha is a geometry prior with omitted proprietary airfoils, twist, interference, viscosity, and aeroelasticity.",
            "The lift target remains conditional on reconstructed force balance and the existing drag/thrust chain.",
            "No elevator effectiveness, pitch-rate derivative, inertia, or rotational dynamics are identified here.",
        ],
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
    gates = document["gates"]
    decision = document["decision"]
    gate_rows = "\n".join(
        f"| {name} | {gate['value']:.4f} | {'pass' if gate['passed'] else 'fail'} |"
        for name, gate in gates.items()
    )
    return f"""# A320 VSPAERO-anchored angle/lift identification

## Outcome

The fit-flight-only constrained model {'passes' if decision['fit_gate_passed'] else 'does not pass'}
its predeclared gate. It estimates a pitch-variation scale of
{model['pitch_scale']:.4f} relative to the current effective decoder and a
combined angle/model offset of {model['combined_offset_deg']:.3f} deg. The
offset is not a sensor calibration.

On identical in-envelope fit samples, mean leave-one-flight-out CL RMSE is
{comparison['anchored_mean_lofo_cl_rmse']:.4f}, versus
{comparison['baseline_mean_lofo_cl_rmse']:.4f} for the unconstrained empirical
lift map (ratio {comparison['anchored_to_baseline_lofo_rmse_ratio']:.3f}).

| Gate | Value | Result |
|---|---:|---:|
{gate_rows}

## Model boundary

The model fixes `CLalpha(M)` to the medium-mesh VSPAERO curve and estimates only
a Mach-independent pitch scale plus constant and Mach-dependent combined
offsets. Ten established calibration flights from both registrations are used;
validation flights are not loaded. A ±10% VSPAERO-slope sensitivity is retained
in the companion table and figure.

## Decision

- Open the existing validation panel next: **{'yes' if decision['open_existing_validation_panel_next'] else 'no'}**.
- Change the current replay plant: **no**.
- Release a physical pitch scale: **no**.
- Identify elevator or rotational dynamics: **no**.

Even if the fit gate passes, the constrained model must first confirm on the
already-separated validation flights. The combined offset includes pitch/AOA
zero, wing incidence and camber, and omitted aerodynamic effects; it must never
be reported as a QAR sensor correction.
"""


def _plot(
    lofo: list[dict[str, Any]], sensitivity: list[dict[str, float]], prefix: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["flight"].split("/", 1)[1] for row in lofo]
    x = np.arange(len(lofo))
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    flat_axes = axes.ravel()
    flat_axes[0].bar(
        x - width / 2,
        [row["baseline_cl_rmse"] for row in lofo],
        width,
        label="empirical baseline",
        color="#0072B2",
        hatch="//",
    )
    flat_axes[0].bar(
        x + width / 2,
        [row["anchored_cl_rmse"] for row in lofo],
        width,
        label="VSPAERO anchored",
        color="#D55E00",
        hatch="..",
    )
    flat_axes[0].set_xticks(x, labels, rotation=50, ha="right")
    flat_axes[0].set(title="Leave-one-flight-out error", ylabel="Lift coefficient RMSE")
    flat_axes[0].legend(frameon=False)
    flat_axes[0].grid(axis="y", alpha=0.25)

    flat_axes[1].plot(
        x,
        [row["pitch_scale"] for row in lofo],
        "o-",
        color="#009E73",
    )
    flat_axes[1].axhspan(*PITCH_SCALE_BOUNDS, color="#009E73", alpha=0.08)
    flat_axes[1].set_xticks(x, labels, rotation=50, ha="right")
    flat_axes[1].set(title="Pitch-scale fold stability", ylabel="Scale vs current decoder")
    flat_axes[1].grid(alpha=0.25)

    support = np.asarray(
        [row["model_alpha_in_envelope_fraction"] for row in lofo], dtype=float
    )
    flat_axes[2].bar(
        x,
        support,
        color=np.where(support >= MIN_MODEL_ALPHA_ENVELOPE_FRACTION, "#009E73", "#E69F00"),
    )
    flat_axes[2].axhline(
        MIN_MODEL_ALPHA_ENVELOPE_FRACTION,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="95% gate",
    )
    flat_axes[2].set_xticks(x, labels, rotation=50, ha="right")
    flat_axes[2].set_ylim(0.6, 1.01)
    flat_axes[2].set(
        title="Local-prior angle support",
        ylabel="Samples inside −1° to 6° envelope",
    )
    flat_axes[2].legend(frameon=False)
    flat_axes[2].grid(axis="y", alpha=0.25)

    registrations = sorted({row["registration"] for row in lofo})
    colors = {registrations[0]: "#0072B2", registrations[1]: "#D55E00"}
    for registration in registrations:
        indices = [i for i, row in enumerate(lofo) if row["registration"] == registration]
        biases = [lofo[i]["anchored_cl_bias"] for i in indices]
        flat_axes[3].scatter(
            indices,
            biases,
            s=34,
            label=registration,
            color=colors[registration],
            marker="o" if registration == registrations[0] else "s",
            zorder=3,
        )
        flat_axes[3].hlines(
            np.mean(biases),
            min(indices) - 0.35,
            max(indices) + 0.35,
            color=colors[registration],
            linewidth=2.0,
        )
    bias_difference = _registration_bias_difference(lofo)
    flat_axes[3].axhline(0.0, color="black", linewidth=0.8)
    flat_axes[3].set_xticks(x, labels, rotation=50, ha="right")
    flat_axes[3].set(
        title=f"Registration residual structure (Δ={bias_difference:.3f} CL)",
        ylabel="Mean lift-coefficient residual",
    )
    flat_axes[3].legend(frameon=False)
    flat_axes[3].grid(axis="y", alpha=0.25)

    for label, axis in zip("ABCD", flat_axes, strict=True):
        axis.text(
            -0.12,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 fit-only VSPAERO-anchored angle/lift identification")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    if len(fit_selection) == 0:
        raise ValueError("no fit flights in panel")
    vertical = _read_vertical_calibration(args.vertical_parameters)
    horizontal = _read_horizontal_parameters(args.horizontal_parameters)
    tas_scale = _read_tas_scale(args.horizontal_calibration)
    flights = _load_point_mass_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        horizontal=horizontal,
        tas_scale=tas_scale,
        wing_area_m2=WING_AREA_M2,
    )
    arrays = _flight_arrays(
        flights, gamma_bias_rad=math.radians(vertical.effective_gamma_bias_deg)
    )
    prior = TabulatedStaticStabilityPrior.from_vspaero_csv(args.vspaero_results)
    document, lofo, sensitivity = build_identification(arrays, prior)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_csv(args.lofo, lofo)
    _write_csv(args.sensitivity, sensitivity)
    args.report.write_text(_render_report(document))
    _plot(lofo, sensitivity, args.figure_prefix)
    return document, lofo


def main() -> None:
    args = parse_args()
    document, _ = run(args)
    print(document["status"])


if __name__ == "__main__":
    main()
