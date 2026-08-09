#!/usr/bin/env python3
"""Validate an intent-side A320 ECON cruise target against stable QAR cruise."""

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

from aerox.fms import EconomyCruiseOptimizer, ModeConditionedSpeedIntent
from aerox.modeling import GuidanceModeThresholds
from aerox.models import (
    KornLockWaveDrag,
    ParabolicDragPolar,
    ParabolicWaveDragModel,
)
from tools.audit_a320_guidance_modes import _load_panel_flight, _read_n1_envelope
from tools.replay_a320_point_mass import _read_fuel_model

UPDATED = "2026-08-07"
FT_TO_M = 0.3048
CANDIDATES = (
    "fixed_m078",
    "qar_selector",
    "affine_selector",
    "economy_parabolic",
    "economy_wave_drag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv")
    )
    parser.add_argument(
        "--n1-envelope",
        type=Path,
        default=Path("docs/modeling/a320-autothrust-envelope.yml"),
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
        "--speed-parameters",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent.yml"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/fms/a320-economy-cruise-intent-flight-metrics.csv"),
    )
    parser.add_argument(
        "--representative",
        type=Path,
        default=Path("docs/fms/a320-economy-cruise-intent-representative.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-economy-cruise-intent.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-economy-cruise-intent.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-economy-cruise-intent"),
    )
    return parser.parse_args()


def _read_speed_model(path: Path) -> ModeConditionedSpeedIntent:
    model = yaml.safe_load(path.read_text())["model"]
    bounds = model["bounds"]
    return ModeConditionedSpeedIntent(
        structure=str(model["structure"]),
        cas_coefficients=np.asarray(model["cas_coefficients"], dtype=float),
        mach_coefficients=np.asarray(model["mach_coefficients"], dtype=float),
        minimum_cas_kts=float(bounds["cas_kts"][0]),
        maximum_cas_kts=float(bounds["cas_kts"][1]),
        minimum_mach=float(bounds["mach"][0]),
        maximum_mach=float(bounds["mach"][1]),
        minimum_crossover_altitude_ft=float(bounds["crossover_altitude_ft"][0]),
        maximum_crossover_altitude_ft=float(bounds["crossover_altitude_ft"][1]),
    )


def _optimizers(args: argparse.Namespace) -> dict[str, EconomyCruiseOptimizer]:
    fuel = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    common: dict[str, Any] = {
        "fuel_flow_model": fuel,
        "wing_area_m2": 122.0,
        "minimum_mach": 0.68,
        "maximum_mach": 0.81,
        "mach_step": 0.0005,
    }
    parabolic = ParabolicDragPolar(0.02263, 0.04246)
    return {
        "economy_parabolic": EconomyCruiseOptimizer(
            drag_model=parabolic,
            **common,
        ),
        "economy_wave_drag": EconomyCruiseOptimizer(
            drag_model=ParabolicWaveDragModel(
                parabolic,
                KornLockWaveDrag(np.deg2rad(25.0), 0.12),
            ),
            **common,
        ),
    }


def _cruise_samples(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (frame["pressure_altitude_ft"] >= 25_000.0)
        & (np.abs(frame["vertical_rate_ft_min"]) <= 250.0)
        & (frame["speed_mode"] == "MACH_HOLD_LIKE")
        & frame["actual_mach"].between(0.60, 0.86)
        & np.isfinite(
            frame[
                [
                    "actual_mach",
                    "selected_speed_kts",
                    "selected_mach",
                    "pressure_altitude_ft",
                    "mass_kg",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
    )
    # Ten-second thinning avoids treating serially correlated QAR samples as
    # independent while retaining the full cruise evolution.
    mask &= np.arange(len(frame)) % 10 == 0
    return frame.loc[mask].reset_index(drop=True)


def _predict(
    samples: pd.DataFrame,
    *,
    speed_model: ModeConditionedSpeedIntent,
    optimizers: dict[str, EconomyCruiseOptimizer],
) -> pd.DataFrame:
    result = samples.copy()
    count = len(result)
    result["fixed_m078"] = 0.78
    result["qar_selector"] = np.clip(
        result["selected_mach"].to_numpy(dtype=float), 0.68, 0.81
    )
    affine = speed_model.predict(
        selected_cas_kts=result["selected_speed_kts"].to_numpy(dtype=float),
        selected_mach=result["selected_mach"].to_numpy(dtype=float),
        pressure_altitude_ft=result["pressure_altitude_ft"].to_numpy(dtype=float),
        mass_kg=result["mass_kg"].to_numpy(dtype=float),
        phase=np.full(count, "LEVEL"),
    )
    result["affine_selector"] = affine.mach_target
    for name, optimizer in optimizers.items():
        result[name] = [
            optimizer.optimize(
                mass_kg=float(row.mass_kg),
                pressure_altitude_m=float(row.pressure_altitude_ft) * FT_TO_M,
                static_temperature_k=_isa_temperature_k(
                    float(row.pressure_altitude_ft) * FT_TO_M
                ),
            ).mach
            for row in result.itertuples(index=False)
        ]
    return result


def _isa_temperature_k(pressure_altitude_m: float) -> float:
    return max(216.65, 288.15 - 0.0065 * pressure_altitude_m)


def _metrics(prediction: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual = prediction["actual_mach"].to_numpy(dtype=float)
    for candidate in CANDIDATES:
        predicted = prediction[candidate].to_numpy(dtype=float)
        error = predicted - actual
        rows.append(
            {
                "registration": str(prediction["registration"].iloc[0]),
                "recording": str(prediction["recording"].iloc[0]),
                "role": str(prediction["role"].iloc[0]),
                "candidate": candidate,
                "sample_count": len(prediction),
                "duration_s": float(prediction["elapsed_time_s"].max())
                - float(prediction["elapsed_time_s"].min()),
                "rmse_mach": float(np.sqrt(np.mean(error**2))),
                "mae_mach": float(np.mean(np.abs(error))),
                "bias_mach": float(np.mean(error)),
                "predicted_mach_p05": float(np.quantile(predicted, 0.05)),
                "predicted_mach_median": float(np.median(predicted)),
                "predicted_mach_p95": float(np.quantile(predicted, 0.95)),
                "lower_bound_fraction": float(np.mean(predicted <= 0.68001)),
                "upper_bound_fraction": float(np.mean(predicted >= 0.80999)),
            }
        )
    return rows


def _summaries(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, role_mask in {
        "development": metrics["role"] != "validation",
        "validation": metrics["role"] == "validation",
    }.items():
        for candidate in CANDIDATES:
            subset = metrics[role_mask & (metrics["candidate"] == candidate)]
            rows.append(
                {
                    "split": split,
                    "candidate": candidate,
                    "eligible_flight_count": len(subset),
                    "mean_flight_rmse_mach": float(subset["rmse_mach"].mean()),
                    "median_flight_rmse_mach": float(subset["rmse_mach"].median()),
                    "maximum_flight_rmse_mach": float(subset["rmse_mach"].max()),
                    "mean_flight_bias_mach": float(subset["bias_mach"].mean()),
                }
            )
    return rows


def _plot(
    *,
    metrics: pd.DataFrame,
    representative: pd.DataFrame,
    wave_optimizer: EconomyCruiseOptimizer,
    no_wave_optimizer: EconomyCruiseOptimizer,
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
    colors = {
        "fixed_m078": "#0072B2",
        "qar_selector": "#999999",
        "affine_selector": "#009E73",
        "economy_parabolic": "#D55E00",
        "economy_wave_drag": "#CC79A7",
    }
    labels = {
        "fixed_m078": "fixed M0.78",
        "qar_selector": "QAR selector",
        "affine_selector": "affine selector",
        "economy_parabolic": "ECON, parabolic drag",
        "economy_wave_drag": "ECON, + wave drag",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(183.0 / 25.4, 126.0 / 25.4), constrained_layout=True
    )

    validation = metrics[metrics["role"] == "validation"]
    for index, candidate in enumerate(CANDIDATES):
        values = validation.loc[validation["candidate"] == candidate, "rmse_mach"]
        axes[0, 0].scatter(
            np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)),
            values,
            color=colors[candidate],
            s=18,
        )
        axes[0, 0].plot(
            [index - 0.22, index + 0.22],
            [values.median()] * 2,
            color="black",
            linewidth=1.2,
        )
    axes[0, 0].set_xticks(range(len(CANDIDATES)), [labels[x] for x in CANDIDATES])
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set(ylabel="Cruise Mach RMSE", title="Frozen-flight validation")

    time_h = representative["elapsed_time_s"] / 3600.0
    axes[0, 1].plot(time_h, representative["actual_mach"], color="black", label="QAR")
    for candidate in ("qar_selector", "affine_selector", "economy_wave_drag"):
        axes[0, 1].plot(
            time_h,
            representative[candidate],
            color=colors[candidate],
            linewidth=0.9,
            label=labels[candidate],
        )
    axes[0, 1].set(
        xlabel="Elapsed time (h)", ylabel="Mach number", title="FLT0003 cruise"
    )
    axes[0, 1].legend(ncols=2)

    mach = np.linspace(0.68, 0.81, 261)
    for optimizer, candidate in (
        (no_wave_optimizer, "economy_parabolic"),
        (wave_optimizer, "economy_wave_drag"),
    ):
        points = [
            optimizer.optimize(
                mass_kg=63_000.0,
                pressure_altitude_m=37_000.0 * FT_TO_M,
                static_temperature_k=216.65,
                along_track_wind_mps=wind,
            )
            for wind in (0.0,)
        ]
        # Re-evaluate the objective by constraining each grid point.
        objective = []
        for value in mach:
            local = EconomyCruiseOptimizer(
                drag_model=optimizer.drag_model,
                fuel_flow_model=optimizer.fuel_flow_model,
                wing_area_m2=optimizer.wing_area_m2,
                minimum_mach=float(value - 0.00001),
                maximum_mach=float(value + 0.00001),
                mach_step=0.00001,
            ).optimize(
                mass_kg=63_000.0,
                pressure_altitude_m=37_000.0 * FT_TO_M,
                static_temperature_k=216.65,
            )
            objective.append(local.fuel_per_ground_distance_kg_m * 1e6)
        axes[1, 0].plot(mach, objective, color=colors[candidate], label=labels[candidate])
        axes[1, 0].axvline(points[0].mach, color=colors[candidate], linestyle=":")
    axes[1, 0].set(
        xlabel="Mach number",
        ylabel="Fuel per distance (kg/1,000 km)",
        title="FL370, 63 t, still-air objective",
    )
    axes[1, 0].legend()

    mass = np.linspace(55_000.0, 72_000.0, 70)
    for wind, color in ((-20.0, "#D55E00"), (0.0, "#0072B2"), (20.0, "#009E73")):
        target = [
            wave_optimizer.optimize(
                mass_kg=float(value),
                pressure_altitude_m=37_000.0 * FT_TO_M,
                static_temperature_k=216.65,
                along_track_wind_mps=wind,
            ).mach
            for value in mass
        ]
        axes[1, 1].plot(mass / 1_000.0, target, color=color, label=f"wind {wind:+.0f} m/s")
    axes[1, 1].set(
        xlabel="Gross mass (t)",
        ylabel="ECON target Mach",
        title="Mass and along-track-wind sensitivity",
    )
    axes[1, 1].legend()

    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        axis.text(0.01, 0.98, label, transform=axis.transAxes, fontweight="bold", va="top")
        axis.grid(color="#DDDDDD", linewidth=0.45, alpha=0.7)
    figure.suptitle("A320 experimental economy-cruise intent gate", fontsize=11)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel)
    envelope = _read_n1_envelope(args.n1_envelope)
    thresholds = GuidanceModeThresholds()
    speed_model = _read_speed_model(args.speed_parameters)
    optimizers = _optimizers(args)
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for row in panel.itertuples(index=False):
        frame = _load_panel_flight(
            raw_dir=args.raw_dir,
            registration=str(row.registration),
            recording=str(row.recording),
            role=str(row.role),
            envelope=envelope,
            thresholds=thresholds,
        )
        cruise = _cruise_samples(frame)
        if len(cruise) < 30:
            continue
        prediction = _predict(
            cruise, speed_model=speed_model, optimizers=optimizers
        )
        predictions.append(prediction)
        metric_rows.extend(_metrics(prediction))
    metrics = pd.DataFrame(metric_rows)
    summaries = _summaries(metrics)

    representative_frame = _load_panel_flight(
        raw_dir=args.raw_dir,
        registration="TAIL-A",
        recording="FLT0003",
        role="post_holdout_diagnostic",
        envelope=envelope,
        thresholds=thresholds,
    )
    representative = _predict(
        _cruise_samples(representative_frame),
        speed_model=speed_model,
        optimizers=optimizers,
    )
    representative_metrics = _metrics(representative)

    summary_frame = pd.DataFrame(summaries)
    validation = summary_frame[summary_frame["split"] == "validation"].set_index(
        "candidate"
    )
    wave_rmse = float(validation.loc["economy_wave_drag", "median_flight_rmse_mach"])
    fixed_rmse = float(validation.loc["fixed_m078", "median_flight_rmse_mach"])
    affine_rmse = float(validation.loc["affine_selector", "median_flight_rmse_mach"])
    no_wave_bound = float(
        metrics.loc[
            (metrics["role"] == "validation")
            & (metrics["candidate"] == "economy_parabolic"),
            "upper_bound_fraction",
        ].mean()
    )
    passed = wave_rmse < fixed_rmse and wave_rmse <= 1.5 * affine_rmse
    status = "passed" if passed else "held"
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 (TAIL-A and TAIL-B)",
        "status": status,
        "decision": {
            "release_for_optional_economy_cruise_intent": passed,
            "wave_drag_validation_median_rmse_mach": wave_rmse,
            "fixed_m078_validation_median_rmse_mach": fixed_rmse,
            "affine_selector_validation_median_rmse_mach": affine_rmse,
            "parabolic_only_mean_upper_bound_fraction": no_wave_bound,
            "gate": "wave ECON improves on fixed M0.78 and is within 1.5x affine-selector RMSE",
            "integration_semantics": "explicit ECON cruise strategy only; never inferred from absent FMA labels",
        },
        "model": {
            "objective": "minimum calibrated fuel flow per along-track ground distance",
            "mach_bounds": [0.68, 0.81],
            "mach_step": 0.0005,
            "wing_area_m2": 122.0,
            "drag_polar": {"cd0": 0.02263, "k": 0.04246},
            "wave_drag": {
                "wing_sweep_deg": 25.0,
                "thickness_to_chord_ratio": 0.12,
                "technology_factor": 0.95,
                "equation_source": "Gur, Mason, and Schetz (2010), DOI 10.2514/1.47557",
                "implementation_source": "OpenAP commit 472ab3e48d52ae62f9586d2e75e143ea4182a9e9",
                "status": "experimental prior; not fitted to QAR response",
            },
            "fuel_model": str(args.fuel_parameters),
        },
        "evaluation": {
            "mask": "pressure altitude >=25,000 ft, |vertical rate| <=250 ft/min, and stable MACH_HOLD_LIKE scoring regime",
            "thinning_s": 10,
            "wind_for_panel_scoring_mps": 0.0,
            "actual_mach_use": "scoring only",
            "summaries": summaries,
            "representative": "TAIL-A/FLT0003 (post-holdout diagnostic)",
            "representative_metrics": representative_metrics,
        },
        "limitations": [
            "QAR contains no unambiguous FMA ECON label, cost index, or forecast wind uplink.",
            "Zero along-track wind is used for panel scoring; sensitivity is reported separately.",
            "Wing thickness ratio and wave-drag technology factor are public-data priors.",
            "Observed Mach variation can include turbulence, mode changes, and tactical pilot inputs that intent should not reproduce.",
        ],
        "artifacts": {
            "flight_metrics": str(args.flight_metrics),
            "representative": str(args.representative),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }

    args.flight_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.flight_metrics, index=False)
    representative.to_csv(args.representative, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _plot(
        metrics=metrics,
        representative=representative,
        wave_optimizer=optimizers["economy_wave_drag"],
        no_wave_optimizer=optimizers["economy_parabolic"],
        prefix=args.figure_prefix,
    )

    table_rows = []
    for candidate in CANDIDATES:
        dev = summary_frame[
            (summary_frame["split"] == "development")
            & (summary_frame["candidate"] == candidate)
        ].iloc[0]
        val = validation.loc[candidate]
        table_rows.append(
            f"| `{candidate}` | {int(dev['eligible_flight_count'])} | "
            f"{dev['median_flight_rmse_mach']:.4f} | {int(val['eligible_flight_count'])} | "
            f"{val['median_flight_rmse_mach']:.4f} |"
        )
    report = f"""# A320 experimental economy-cruise intent

## Outcome

Status: **{status}**. The wave-drag ECON target has a held-out median cruise-Mach
RMSE of {wave_rmse:.4f}, versus {fixed_rmse:.4f} for fixed M0.78 and
{affine_rmse:.4f} for the QAR-selector-conditioned affine target. The
parabolic-polar-only optimizer spends {no_wave_bound:.1%} of validation samples
at its M0.81 upper bound, demonstrating that a subsonic drag polar alone is not
adequate for ECON cruise-speed selection.

| Candidate | Dev flights | Dev median RMSE | Validation flights | Validation median RMSE |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The physics target minimizes calibrated fuel flow divided by ground speed at the
planned mass, pressure altitude, temperature, and along-track wind. It uses the
existing A320 drag polar plus the experimental OpenAP/Gur critical-Mach and
fourth-power wave-drag correction. It consumes no actual Mach, N1, fuel flow, or
trajectory residual when generating a target.

![A320 economy-cruise intent validation](figures/a320-economy-cruise-intent.png)

Figure 1. Flight-level validation, post-holdout FLT0003 comparison, objective
shape, and mass/wind sensitivity. A [vector PDF](figures/a320-economy-cruise-intent.pdf)
is available.

## Interpretation

This is an explicit Aerox FMS cruise strategy, not an inferred Airbus guidance
mode. The QAR lacks an unambiguous ECON/FMA label and cost index, so actual Mach
is used only to score plausibility. The affine selector remains the stronger
replay reference when cockpit selector evidence is available. ECON is useful
when trajectory generation starts from route intent and aircraft state instead.

Panel scoring assumes zero along-track wind because a forecast-wind intent
channel is not available uniformly. The lower-right panel shows how the same
optimizer responds when route wind is supplied: headwind raises target Mach and
tailwind lowers it.

## Sources and limitations

- Wave drag: Gur, Mason, and Schetz (2010), DOI `10.2514/1.47557`, through the
  pinned OpenAP implementation at commit
  `472ab3e48d52ae62f9586d2e75e143ea4182a9e9`.
- Aerox A320 drag polar: `CD0=0.02263`, `k=0.04246`; calibrated fuel-flow model:
  `{args.fuel_parameters}`.
- Sweep 25 deg and thickness/chord 0.12 remain public-data priors. This is not a
  certified aircraft-specific transonic drag model.
"""
    args.report.write_text(report)
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
