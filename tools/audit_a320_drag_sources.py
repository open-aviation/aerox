#!/usr/bin/env python3
"""Reconcile public A320 drag sources and a Reynolds-aware candidate model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.models import (
    ParabolicDragPolar,
    PollSchumannDragModel,
    ParabolicWaveDragModel,
    KornLockWaveDrag,
    isa_atmosphere,
)

UPDATED = "2026-08-08"
POLL_SCHUMANN_DOI = "https://doi.org/10.1017/aer.2024.141"
POLL_SCHUMANN_DLR = "https://elib.dlr.de/212199/"
OPENAP_PAPER_DOI = "https://doi.org/10.1016/j.trc.2020.01.026"
OPENAP_DATA_DOI = "https://doi.org/10.6084/m9.figshare.11424924.v2"
A320_HACK_URL = (
    "https://repository.tudelft.nl/file/"
    "File_8f7273c9-776d-48a3-863a-3d2802b112da"
)

# Summary of the 92 complete A320 rows in the CC-BY OpenAP supplement.  The
# values are retained here rather than vendoring the full 49 MB source archive.
OPENAP_DISTRIBUTION = {
    "sample_count": 92,
    "cd0": {
        "minimum": 0.007010,
        "p05": 0.009681,
        "p25": 0.013435,
        "median": 0.017530,
        "mean": 0.018179456521739128,
        "p75": 0.021790,
        "p95": 0.029903,
        "maximum": 0.038950,
    },
    "k": {
        "minimum": 0.034550,
        "p05": 0.035560,
        "p25": 0.036990,
        "median": 0.038545,
        "mean": 0.03879239130434783,
        "p75": 0.040170,
        "p95": 0.043250,
        "maximum": 0.046690,
    },
    "e": {
        "minimum": 0.659850,
        "p05": 0.712535,
        "p25": 0.767818,
        "median": 0.799895,
        "mean": 0.7981971739130433,
        "p75": 0.835360,
        "p95": 0.868105,
        "maximum": 0.891900,
    },
    "cd0_k_correlation": 0.9999993124559039,
}


@dataclass(frozen=True)
class ModelMetrics:
    sample_count: int
    rmse_drag_coefficient: float
    rmse_drag_counts: float
    rmspe: float
    mean_bias_drag_counts: float
    maximum_absolute_error_drag_counts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-reference",
        type=Path,
        default=Path("docs/modeling/a320-published-transonic-drag-reference.csv"),
    )
    parser.add_argument(
        "--qar-points",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-qar-points.csv"),
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path("docs/modeling/a320-drag-source-reconciliation-surface.csv"),
    )
    parser.add_argument(
        "--qar-comparison",
        type=Path,
        default=Path("docs/modeling/a320-drag-source-reconciliation-qar.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-drag-source-reconciliation.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-drag-source-reconciliation.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-drag-source-reconciliation"),
    )
    return parser.parse_args()


def _poll_schumann_a320() -> PollSchumannDragModel:
    return PollSchumannDragModel(
        reference_area_m2=122.4,
        wing_span_m=34.1,
        fuselage_width_m=3.95,
        wing_sweep_rad=np.deg2rad(25.0),
        skin_friction_multiplier=8.40,
        technology_factor=0.750,
        design_mach=0.753,
        design_lift_coefficient=0.590,
        wave_creep_coefficient=0.073,
        wave_creep_threshold=0.869,
        reference_reynolds_number=5.59e7,
    )


def _models() -> tuple[ParabolicWaveDragModel, ParabolicWaveDragModel, PollSchumannDragModel]:
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    wave = KornLockWaveDrag(
        wing_sweep_rad=np.deg2rad(float(aircraft.wing.sweep)),
        thickness_to_chord_ratio=float(aircraft.wing.t_c),
    )
    current = ParabolicWaveDragModel(
        reference_drag,
        wave,
    )
    openap = ParabolicWaveDragModel(ParabolicDragPolar(0.018, 0.039), wave)
    return current, openap, _poll_schumann_a320()


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> ModelMetrics:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        raise ValueError("no finite points are available for model metrics")
    residual = predicted - observed
    return ModelMetrics(
        sample_count=len(observed),
        rmse_drag_coefficient=float(np.sqrt(np.mean(residual**2))),
        rmse_drag_counts=float(np.sqrt(np.mean(residual**2)) * 1.0e4),
        rmspe=float(np.sqrt(np.mean((residual / observed) ** 2))),
        mean_bias_drag_counts=float(np.mean(residual) * 1.0e4),
        maximum_absolute_error_drag_counts=float(np.max(np.abs(residual)) * 1.0e4),
    )


def _sutherland_viscosity(temperature_k: np.ndarray) -> np.ndarray:
    """Return dynamic viscosity in Pa s using Sutherland's air correlation."""

    return 1.458e-6 * temperature_k**1.5 / (temperature_k + 110.4)


def _aircraft_reynolds_number(
    pressure_altitude_ft: np.ndarray,
    mach: np.ndarray,
    reference_area_m2: float,
) -> np.ndarray:
    atmosphere = isa_atmosphere(pressure_altitude_ft * 0.3048)
    true_airspeed = mach * atmosphere.speed_of_sound_mps
    viscosity = _sutherland_viscosity(atmosphere.temperature_k)
    return (
        np.sqrt(reference_area_m2)
        * atmosphere.density_kg_m3
        * true_airspeed
        / viscosity
    )


def _evaluate_published_reference(
    reference: pd.DataFrame,
    current: ParabolicWaveDragModel,
    openap: ParabolicWaveDragModel,
    poll: PollSchumannDragModel,
) -> tuple[pd.DataFrame, dict[str, ModelMetrics]]:
    result = reference.copy()
    lift = result["lift_coefficient"].to_numpy(dtype=float)
    mach = result["mach"].to_numpy(dtype=float)
    result["aerox_current_drag_coefficient"] = current.drag_coefficient(lift, mach)
    result["openap_legacy_gur_drag_coefficient"] = openap.drag_coefficient(
        lift, mach
    )
    poll_prediction = np.full_like(lift, np.nan)
    poll_support = (
        (lift >= poll.minimum_lift_coefficient)
        & (lift <= poll.maximum_lift_coefficient)
        & (mach >= poll.minimum_mach)
        & (mach <= poll.maximum_mach)
    )
    poll_prediction[poll_support] = poll.drag_coefficient(
        lift[poll_support], mach[poll_support]
    )
    result["poll_schumann_drag_coefficient"] = poll_prediction
    observed = result["published_observed_drag_coefficient"].to_numpy(dtype=float)
    predictions = {
        "obert_krull_tanh": result[
            "published_model_drag_coefficient"
        ].to_numpy(dtype=float),
        "aerox_current": result["aerox_current_drag_coefficient"].to_numpy(
            dtype=float
        ),
        "openap_legacy_gur": result[
            "openap_legacy_gur_drag_coefficient"
        ].to_numpy(dtype=float),
        "poll_schumann": result["poll_schumann_drag_coefficient"].to_numpy(
            dtype=float
        ),
    }
    return result, {
        name: _metrics(observed, prediction) for name, prediction in predictions.items()
    }


def _surface(
    current: ParabolicWaveDragModel,
    openap: ParabolicWaveDragModel,
    poll: PollSchumannDragModel,
) -> pd.DataFrame:
    lift_grid = np.linspace(0.3, 0.7, 81)
    mach_grid = np.linspace(0.5, 0.82, 65)
    lift, mach = np.meshgrid(lift_grid, mach_grid, indexing="ij")
    poll_breakdown = poll.breakdown(lift, mach)
    current_breakdown = current.breakdown(lift, mach)
    openap_breakdown = openap.breakdown(lift, mach)
    return pd.DataFrame(
        {
            "lift_coefficient": lift.ravel(),
            "mach": mach.ravel(),
            "aerox_current_drag_coefficient": current_breakdown.total.ravel(),
            "openap_legacy_gur_drag_coefficient": openap_breakdown.total.ravel(),
            "poll_schumann_drag_coefficient": poll_breakdown.total.ravel(),
            "poll_schumann_zero_lift_drag_coefficient": (
                poll_breakdown.parasite.ravel()
            ),
            "poll_schumann_induced_drag_coefficient": poll_breakdown.induced.ravel(),
            "poll_schumann_wave_drag_coefficient": poll_breakdown.wave.ravel(),
        }
    )


def _evaluate_qar(
    qar: pd.DataFrame,
    current: ParabolicWaveDragModel,
    openap: ParabolicWaveDragModel,
    poll: PollSchumannDragModel,
) -> pd.DataFrame:
    result = qar.copy()
    lift = result["lift_coefficient"].to_numpy(dtype=float)
    mach = result["actual_mach"].to_numpy(dtype=float)
    reynolds = _aircraft_reynolds_number(
        result["pressure_altitude_ft"].to_numpy(dtype=float),
        mach,
        poll.reference_area_m2,
    )
    q_s = (
        result["parabolic_drag_n"].to_numpy(dtype=float)
        / result["parabolic_drag_coefficient"].to_numpy(dtype=float)
    )
    result["poll_schumann_reynolds_number"] = reynolds
    result["dynamic_pressure_area_n"] = q_s
    result["aerox_current_drag_coefficient"] = current.drag_coefficient(lift, mach)
    result["openap_legacy_gur_drag_coefficient"] = openap.drag_coefficient(
        lift, mach
    )
    result["poll_schumann_drag_coefficient"] = poll.drag_coefficient(
        lift, mach, reynolds
    )
    for name in ("aerox_current", "openap_legacy_gur", "poll_schumann"):
        result[f"{name}_drag_n"] = (
            result[f"{name}_drag_coefficient"].to_numpy(dtype=float) * q_s
        )
    result["poll_minus_current_drag_n"] = (
        result["poll_schumann_drag_n"] - result["aerox_current_drag_n"]
    )
    return result


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def _qar_metrics(qar: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "sample_count": len(qar),
        "lift_coefficient": _distribution(
            qar["lift_coefficient"].to_numpy(dtype=float)
        ),
        "mach": _distribution(qar["actual_mach"].to_numpy(dtype=float)),
        "poll_schumann_reynolds_number": _distribution(
            qar["poll_schumann_reynolds_number"].to_numpy(dtype=float)
        ),
    }
    for name in ("aerox_current", "openap_legacy_gur", "poll_schumann"):
        metrics[name] = {
            "drag_coefficient": _distribution(
                qar[f"{name}_drag_coefficient"].to_numpy(dtype=float)
            ),
            "drag_kn": _distribution(
                qar[f"{name}_drag_n"].to_numpy(dtype=float) / 1.0e3
            ),
        }
    metrics["poll_minus_current_drag_kn"] = _distribution(
        qar["poll_minus_current_drag_n"].to_numpy(dtype=float) / 1.0e3
    )
    return metrics


def _point_breakdown(
    model: ParabolicWaveDragModel | PollSchumannDragModel,
    lift: float,
    mach: float,
    reynolds: float | None = None,
) -> dict[str, float]:
    if isinstance(model, PollSchumannDragModel):
        breakdown = model.breakdown(lift, mach, reynolds)
    else:
        breakdown = model.breakdown(lift, mach)
    return {
        "zero_lift_drag_coefficient": float(breakdown.parasite),
        "induced_drag_coefficient": float(breakdown.induced),
        "wave_drag_coefficient": float(breakdown.wave),
        "total_drag_coefficient": float(breakdown.total),
        "lift_to_drag_ratio": lift / float(breakdown.total),
        "critical_mach": float(breakdown.critical_mach),
    }


def _plot(
    *,
    reference: pd.DataFrame,
    surface: pd.DataFrame,
    qar: pd.DataFrame,
    current: ParabolicWaveDragModel,
    openap: ParabolicWaveDragModel,
    poll: PollSchumannDragModel,
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
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "obert": "#000000",
        "tanh": "#0072B2",
        "current": "#D55E00",
        "openap": "#CC79A7",
        "poll": "#009E73",
        "crosscheck": "#E69F00",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(183.0 / 25.4, 128.0 / 25.4), constrained_layout=True
    )

    source_cl = reference[np.isclose(reference["lift_coefficient"], 0.5)]
    axes[0, 0].scatter(
        source_cl["mach"],
        source_cl["published_observed_drag_coefficient"] * 1.0e4,
        s=13,
        facecolors="white",
        edgecolors=colors["obert"],
        linewidths=0.7,
        label="Obert/Krull points",
        zorder=5,
    )
    axes[0, 0].plot(
        source_cl["mach"],
        source_cl["published_model_drag_coefficient"] * 1.0e4,
        color=colors["tanh"],
        linewidth=1.5,
        label="Obert/Krull tanh",
    )
    for column, color, linestyle, label in (
        (
            "aerox_current_drag_coefficient",
            colors["current"],
            "--",
            "Aerox current",
        ),
        (
            "openap_legacy_gur_drag_coefficient",
            colors["openap"],
            ":",
            "OpenAP + Gur/Lock",
        ),
        (
            "poll_schumann_drag_coefficient",
            colors["poll"],
            "-.",
            "Poll-Schumann",
        ),
    ):
        axes[0, 0].plot(
            source_cl["mach"],
            source_cl[column] * 1.0e4,
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=label,
        )
    axes[0, 0].set(
        xlabel="Mach number",
        ylabel="Drag coefficient (counts)",
        title="Total drag at CL=0.5",
    )
    axes[0, 0].legend(loc="upper left", ncols=1)
    axes[0, 0].grid(axis="y", color="#DDDDDD", linewidth=0.45)

    mach_slice = surface[np.isclose(surface["mach"], 0.78)]
    for column, color, linestyle, label in (
        (
            "aerox_current_drag_coefficient",
            colors["current"],
            "--",
            "Aerox current",
        ),
        (
            "openap_legacy_gur_drag_coefficient",
            colors["openap"],
            ":",
            "OpenAP + Gur/Lock",
        ),
        (
            "poll_schumann_drag_coefficient",
            colors["poll"],
            "-.",
            "Poll-Schumann",
        ),
    ):
        axes[0, 1].plot(
            mach_slice["lift_coefficient"],
            mach_slice[column] * 1.0e4,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )
    lift_min = float(qar["lift_coefficient"].min())
    lift_max = float(qar["lift_coefficient"].max())
    axes[0, 1].axvspan(
        lift_min,
        lift_max,
        color="#56B4E9",
        alpha=0.16,
        label="FLT0003 QAR support",
    )
    axes[0, 1].scatter(
        [0.622],
        [0.0345 * 1.0e4],
        marker="D",
        s=25,
        facecolor=colors["crosscheck"],
        edgecolor="black",
        linewidth=0.5,
        label="A320neo weak cross-check",
        zorder=6,
    )
    axes[0, 1].set(
        xlabel="Lift coefficient",
        ylabel="Drag coefficient (counts)",
        title="Cruise polar at Mach 0.78",
    )
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(axis="y", color="#DDDDDD", linewidth=0.45)

    representative_lift = float(np.median(qar["lift_coefficient"]))
    representative_mach = float(np.median(qar["actual_mach"]))
    representative_reynolds = float(
        np.median(qar["poll_schumann_reynolds_number"])
    )
    point_models = (current, openap, poll)
    labels = ("Aerox current", "OpenAP + Gur", "Poll-Schumann")
    components = np.array(
        [
            list(
                _point_breakdown(
                    model,
                    representative_lift,
                    representative_mach,
                    representative_reynolds,
                ).values()
            )[:3]
            for model in point_models
        ]
    )
    bottom = np.zeros(len(labels))
    for values, color, name in zip(
        components.T,
        ("#999999", "#56B4E9", "#E69F00"),
        ("zero-lift", "lift-dependent", "transonic"),
        strict=True,
    ):
        axes[1, 0].bar(
            labels,
            values * 1.0e4,
            bottom=bottom * 1.0e4,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=name,
        )
        bottom += values
    axes[1, 0].set(
        ylabel="Drag coefficient (counts)",
        title=(
            f"Component allocation at CL={representative_lift:.3f}, "
            f"Mach={representative_mach:.3f}"
        ),
    )
    axes[1, 0].tick_params(axis="x", rotation=10)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.45)

    y = np.array([1.0, 0.0])
    for row, name in zip(y, ("cd0", "k"), strict=True):
        summary = OPENAP_DISTRIBUTION[name]
        axes[1, 1].plot(
            [summary["p05"], summary["p95"]],
            [row, row],
            color="#777777",
            linewidth=1.2,
        )
        axes[1, 1].plot(
            [summary["p25"], summary["p75"]],
            [row, row],
            color=colors["openap"],
            linewidth=7.0,
            solid_capstyle="butt",
        )
        axes[1, 1].scatter(
            summary["median"],
            row,
            marker="|",
            s=100,
            color="black",
            zorder=5,
        )
    poll_cd0 = float(poll.zero_lift_drag_coefficient())
    poll_k = float(poll.induced_drag_factor())
    axes[1, 1].scatter(
        [current.base_polar.zero_lift_drag_coefficient, current.base_polar.induced_drag_factor],
        y + 0.12,
        marker="^",
        color=colors["current"],
        s=28,
        label="Aerox current",
        zorder=6,
    )
    axes[1, 1].scatter(
        [poll_cd0, poll_k],
        y - 0.12,
        marker="o",
        facecolor=colors["poll"],
        edgecolor="black",
        linewidth=0.4,
        s=24,
        label="Poll-Schumann",
        zorder=6,
    )
    axes[1, 1].set(
        xlabel="Coefficient value",
        yticks=y,
        yticklabels=("CD0", "k"),
        ylim=(-0.45, 1.45),
        title="OpenAP 92-flight spread (5–95%, IQR)",
    )
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(axis="x", color="#DDDDDD", linewidth=0.45)

    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
    figure.suptitle("A320 drag-source reconciliation and candidate model")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def _metrics_table(metrics: dict[str, ModelMetrics]) -> str:
    labels = {
        "obert_krull_tanh": "Obert/Krull tanh fit",
        "aerox_current": "Aerox current",
        "openap_legacy_gur": "OpenAP surveillance polar + Gur/Lock",
        "poll_schumann": "Poll-Schumann 2025",
    }
    rows = [
        "| Model | n | RMSE (counts) | RMSPE | Mean bias (counts) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in metrics.items():
        rows.append(
            f"| {labels[name]} | {metric.sample_count} | "
            f"{metric.rmse_drag_counts:.2f} | "
            f"{100.0 * metric.rmspe:.2f}% | {metric.mean_bias_drag_counts:+.2f} |"
        )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference = pd.read_csv(args.published_reference)
    qar_input = pd.read_csv(args.qar_points)
    current, openap, poll = _models()
    evaluated_reference, source_metrics = _evaluate_published_reference(
        reference, current, openap, poll
    )
    surface = _surface(current, openap, poll)
    qar = _evaluate_qar(qar_input, current, openap, poll)
    qar_metrics = _qar_metrics(qar)

    reference_lift = 0.5
    reference_mach = 0.78
    source_cl = evaluated_reference[
        np.isclose(evaluated_reference["lift_coefficient"], reference_lift)
    ]
    obert_total = float(
        np.interp(
            reference_mach,
            source_cl["mach"],
            source_cl["published_model_drag_coefficient"],
        )
    )
    representative = {
        "lift_coefficient": reference_lift,
        "mach": reference_mach,
        "obert_krull_total_drag_coefficient": obert_total,
        "aerox_current": _point_breakdown(
            current, reference_lift, reference_mach
        ),
        "openap_legacy_gur": _point_breakdown(
            openap, reference_lift, reference_mach
        ),
        "poll_schumann": _point_breakdown(
            poll,
            reference_lift,
            reference_mach,
            poll.reference_reynolds_number,
        ),
    }
    a320neo_lift = 0.622
    a320neo_mach = 0.78
    a320neo_crosscheck = {
        "aircraft": "A320neo simplified preliminary-design model",
        "lift_coefficient": a320neo_lift,
        "mach": a320neo_mach,
        "reported_hack_drag_coefficient": 0.0372,
        "reported_tank_zero_lift_drag_increment": 0.0027,
        "inferred_a320neo_drag_coefficient": 0.0345,
        "limitations": (
            "A320neo sharklets, preliminary design, simplified parabolic polar, "
            "and coefficients inherited from an earlier OpenAP study"
        ),
        "model_predictions": {
            "aerox_current": float(
                current.drag_coefficient(a320neo_lift, a320neo_mach)
            ),
            "openap_legacy_gur": float(
                openap.drag_coefficient(a320neo_lift, a320neo_mach)
            ),
            "poll_schumann": float(
                poll.drag_coefficient(a320neo_lift, a320neo_mach)
            ),
        },
    }
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 candidate drag model",
        "status": "poll-schumann-leading-candidate-default-held",
        "source_assessment": {
            "aerox_current": {
                "cd0": current.base_polar.zero_lift_drag_coefficient,
                "k": current.base_polar.induced_drag_factor,
                "origin": (
                    "simplified wing-plus-fuselage skin-friction buildup in "
                    "the removed legacy coefficient generator with its 1.05 "
                    "blanket component factor"
                ),
                "assessment": "engineering estimate, not A320 identification",
            },
            "obert_krull": {
                "source": "Obert (2009), digitized and fitted by Krull (2025)",
                "assessment": (
                    "useful total-drag curve; Krull explicitly reports anomalous "
                    "A320 fitted induced-drag parameters and says the source polar "
                    "may be inaccurate"
                ),
            },
            "openap_surveillance": {
                "paper": OPENAP_PAPER_DOI,
                "dataset": OPENAP_DATA_DOI,
                "distribution": OPENAP_DISTRIBUTION,
                "assessment": (
                    "valuable low-Mach population prior; the wide per-flight "
                    "spread and near-perfect CD0-k correlation demonstrate weak "
                    "separate observability"
                ),
            },
            "poll_schumann": {
                "paper": POLL_SCHUMANN_DOI,
                "repository": POLL_SCHUMANN_DLR,
                "model_support": {
                    "lift_coefficient": [
                        poll.minimum_lift_coefficient,
                        poll.maximum_lift_coefficient,
                    ],
                    "mach": [poll.minimum_mach, poll.maximum_mach],
                    "reynolds_number_is_explicit": True,
                },
                "parameters": {
                    "reference_area_m2": poll.reference_area_m2,
                    "wing_span_m": poll.wing_span_m,
                    "fuselage_width_m": poll.fuselage_width_m,
                    "wing_sweep_deg": float(np.rad2deg(poll.wing_sweep_rad)),
                    "skin_friction_multiplier": poll.skin_friction_multiplier,
                    "technology_factor": poll.technology_factor,
                    "design_mach": poll.design_mach,
                    "design_lift_coefficient": poll.design_lift_coefficient,
                    "design_reynolds_number": poll.reference_reynolds_number,
                    "wave_creep_coefficient": poll.wave_creep_coefficient,
                    "wave_creep_threshold": poll.wave_creep_threshold,
                    "wave_rise_coefficient": poll.wave_rise_coefficient,
                },
                "design_cd0": float(poll.zero_lift_drag_coefficient()),
                "design_k": float(poll.induced_drag_factor()),
                "assessment": (
                    "strongest public candidate because it covers the QAR CL "
                    "range and separates Reynolds, lift-dependent, drag-creep, "
                    "and drag-rise effects"
                ),
            },
        },
        "metrics_against_obert_krull_points": {
            name: asdict(metric) for name, metric in source_metrics.items()
        },
        "representative_comparison": representative,
        "a320neo_weak_crosscheck": a320neo_crosscheck,
        "qar_cruise_comparison": qar_metrics,
        "decision": {
            "leading_candidate": "Poll-Schumann 2025",
            "candidate_implemented": True,
            "default_plant_changed": False,
            "reason": (
                "The candidate is substantially better founded than the current "
                "component buildup, but the installed-thrust map was identified "
                "conditional on the current polar and must be refitted before a "
                "trajectory comparison is physically interpretable."
            ),
            "next_gate": (
                "Refit the effective N1-to-thrust map under the frozen "
                "Poll-Schumann candidate, then compare held-out trajectory and "
                "fuel errors against the current model."
            ),
        },
        "artifacts": {
            "surface": str(args.surface),
            "qar_comparison": str(args.qar_comparison),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }

    args.surface.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(args.surface, index=False)
    qar.to_csv(args.qar_comparison, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _plot(
        reference=evaluated_reference,
        surface=surface,
        qar=qar,
        current=current,
        openap=openap,
        poll=poll,
        prefix=args.figure_prefix,
    )

    metric_table = _metrics_table(source_metrics)
    current_point = representative["aerox_current"]["total_drag_coefficient"]
    poll_point = representative["poll_schumann"]["total_drag_coefficient"]
    report = f"""# A320 drag-source reconciliation

## Outcome

The current Aerox drag coefficients should no longer be treated as the leading
physical prior. `CD0=0.02263` and `k=0.04246` came from a simplified
wing-plus-fuselage component buildup, not from A320 flight identification or a
manufacturer aerodynamic deck.

The strongest public replacement candidate found is the Reynolds-aware model
of Poll and Schumann (2025). It covers `CL=0.3` to `0.7`, explicitly separates
zero-lift, lift-dependent, drag-creep, and drag-rise terms, and contains an A320
parameter row. It is implemented and bounded in Aerox, but remains opt-in until
the conditional thrust map is refitted.

![A320 drag-source reconciliation](figures/a320-drag-source-reconciliation.png)

Figure 1. Total-drag, cruise-polar, component, and source-spread comparisons.
The A320neo point is only a weak cross-check because it uses sharklets and a
simplified preliminary-design polar. A
[vector PDF](figures/a320-drag-source-reconciliation.pdf) is also provided.

## What was wrong with the earlier comparison

Krull's report itself notes that the A320 fit stands out negatively: its
induced-drag parameters are implausible, and one stated explanation is that the
underlying Obert polar may be inaccurate. Therefore its fitted component split
must not be interpreted as truth.

However, the *total* Obert/Krull drag near cruise is not obviously wrong. At
`CL=0.5`, `Mach=0.78`, it gives `{obert_total:.5f}`. Poll-Schumann gives
`{poll_point:.5f}`, while current Aerox gives `{current_point:.5f}`. The two
independent total-drag estimates agree closely even though their component
allocations differ.

## Comparison against the 271 Obert/Krull points

{metric_table}

This table should be read as agreement with one uncertain graphical source,
not as ground-truth accuracy. Poll-Schumann does not reproduce the sharpest
high-Mach end of that curve, but it has physical support across the QAR lift
range where the digitized source has no points.

## OpenAP observability warning

The OpenAP supplement contains 92 usable A320 flight estimates. Their median
is `CD0={OPENAP_DISTRIBUTION['cd0']['median']:.5f}` and
`k={OPENAP_DISTRIBUTION['k']['median']:.5f}`, but the 5--95% ranges are
`{OPENAP_DISTRIBUTION['cd0']['p05']:.5f}`--
`{OPENAP_DISTRIBUTION['cd0']['p95']:.5f}` and
`{OPENAP_DISTRIBUTION['k']['p05']:.5f}`--
`{OPENAP_DISTRIBUTION['k']['p95']:.5f}`. Their correlation is
`{OPENAP_DISTRIBUTION['cd0_k_correlation']:.7f}`. Thus surveillance data
strongly constrain total energy loss but do not independently identify both
polar components.

## QAR implication

Across the representative FLT0003 cruise, the Poll-Schumann candidate changes
drag relative to current Aerox by a median of
`{qar_metrics['poll_minus_current_drag_kn']['median']:+.2f} kN` with a 5--95%
range of `{qar_metrics['poll_minus_current_drag_kn']['p05']:+.2f}` to
`{qar_metrics['poll_minus_current_drag_kn']['p95']:+.2f} kN`. This is large
enough that the existing effective N1-to-thrust map cannot be reused unchanged.

## Decision and next gate

- Keep the released plant unchanged for now.
- Use Poll-Schumann as the leading bounded candidate, including its Reynolds
  dependence.
- Refit the effective thrust map with this polar frozen.
- Compare current and candidate combinations on held-out trajectory and fuel
  metrics; do not judge the drag candidate using a thrust map calibrated under
  the old polar.

## Sources

- Poll and Schumann (2025): `{POLL_SCHUMANN_DOI}`; open DLR copy:
  `{POLL_SCHUMANN_DLR}`.
- OpenAP drag-polar paper: `{OPENAP_PAPER_DOI}`; CC-BY supplement:
  `{OPENAP_DATA_DOI}`.
- Krull/Obert source and license details remain in
  `a320-published-transonic-drag-audit.md`.
- A320neo weak cross-check: `{A320_HACK_URL}`.
"""
    args.report.write_text(report)
    return document


def main() -> None:
    document = run(parse_args())
    print(yaml.safe_dump(document["decision"], sort_keys=False))


if __name__ == "__main__":
    main()
