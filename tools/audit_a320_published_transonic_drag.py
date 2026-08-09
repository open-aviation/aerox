#!/usr/bin/env python3
"""Compare Aerox drag models with the published Krull A320 transonic polar."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.models import (
    HyperbolicTangentWaveDrag,
    MachDependentInducedDrag,
    ParabolicDragPolar,
    TransonicCompressibilityDragModel,
    ParabolicWaveDragModel,
    KornLockWaveDrag,
)

UPDATED = "2026-08-08"
SHEET_NAME = "24.123-A320B737"
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SOURCE_DOI = "https://doi.org/10.7910/DVN/2UBNIE"
REPORT_URL = (
    "https://www.fzt.haw-hamburg.de/pers/Scholz/arbeiten/TextKrullMaster.pdf"
)
POSTER_DOI = "https://doi.org/10.48441/4427.2893"
SOURCE_MD5 = "219286b978a47f1a5a57476c0ec99ecc"


@dataclass(frozen=True)
class ModelMetrics:
    rmse_drag_coefficient: float
    rmse_drag_counts: float
    rmspe: float
    mean_bias_drag_counts: float
    maximum_absolute_error_drag_counts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-xlsx",
        type=Path,
        default=Path("data/external/a320-wave-drag/HyperbolicTangent.xlsx"),
    )
    parser.add_argument(
        "--qar-points",
        type=Path,
        default=Path("docs/modeling/a320-wave-drag-qar-points.csv"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/modeling/a320-published-transonic-drag-reference.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-published-transonic-drag-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-published-transonic-drag-audit.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path(
            "docs/modeling/figures/a320-published-transonic-drag-audit"
        ),
    )
    return parser.parse_args()


def _shared_strings(archive: ZipFile) -> list[str]:
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t"))
        for item in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _sheet_path(archive: ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib["name"] == sheet_name:
            relationship_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
            break
    if relationship_id is None:
        raise ValueError(f"sheet {sheet_name!r} is absent")
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relation in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relation.attrib["Id"] == relationship_id:
            return f"xl/{relation.attrib['Target']}"
    raise ValueError(f"relationship {relationship_id!r} is absent")


def _xlsx_cells(path: Path, sheet_name: str) -> dict[str, str | float]:
    """Read cached XLSX values without introducing a runtime Excel dependency."""

    with ZipFile(path) as archive:
        strings = _shared_strings(archive)
        root = ElementTree.fromstring(archive.read(_sheet_path(archive, sheet_name)))
    cells: dict[str, str | float] = {}
    for cell in root.findall(f".//{{{MAIN_NS}}}c"):
        value = cell.find(f"{{{MAIN_NS}}}v")
        if value is None or value.text is None:
            continue
        reference = cell.attrib["r"]
        cells[reference] = (
            strings[int(value.text)]
            if cell.attrib.get("t") == "s"
            else float(value.text)
        )
    return cells


def _number(cells: dict[str, str | float], reference: str) -> float:
    value = cells.get(reference)
    if not isinstance(value, float):
        raise TypeError(f"cell {reference} does not contain a cached number")
    return value


def _published_model(
    cells: dict[str, str | float],
) -> TransonicCompressibilityDragModel:
    lift_grid = np.array([_number(cells, cell) for cell in ("D1", "J1", "P1")])
    critical_mach = np.array(
        [_number(cells, cell) for cell in ("F1", "L1", "R1")]
    )
    return TransonicCompressibilityDragModel(
        zero_lift_drag_coefficient=_number(cells, "X6") * 1.0e-4,
        induced_drag=MachDependentInducedDrag(
            reference_induced_drag_factor=_number(cells, "X5") * 1.0e-4,
            compressibility_strength=_number(cells, "X7"),
            compressibility_exponent=_number(cells, "X8"),
            compressibility_mach=0.3,
        ),
        wave_drag=HyperbolicTangentWaveDrag(
            lift_coefficient_grid=lift_grid,
            critical_mach_grid=critical_mach,
            amplitude=_number(cells, "X2") * 1.0e-4,
            mach_scale=_number(cells, "X3"),
            shift=_number(cells, "X4"),
        ),
    )


def _source_points(cells: dict[str, str | float]) -> pd.DataFrame:
    blocks = (
        ("C", "D", "E", "D1", "F1"),
        ("I", "J", "K", "J1", "L1"),
        ("O", "P", "Q", "P1", "R1"),
    )
    rows: list[dict[str, float]] = []
    for mach_column, observed_column, fit_column, lift_cell, critical_cell in blocks:
        lift = _number(cells, lift_cell)
        critical_mach = _number(cells, critical_cell)
        for row in range(3, 151):
            mach = cells.get(f"{mach_column}{row}")
            observed = cells.get(f"{observed_column}{row}")
            fitted = cells.get(f"{fit_column}{row}")
            if not all(isinstance(value, float) for value in (mach, observed, fitted)):
                continue
            if not 0.3 < mach < 1.0 or not 100.0 < observed < 1_000.0:
                continue
            rows.append(
                {
                    "lift_coefficient": lift,
                    "mach": mach,
                    "published_observed_drag_coefficient": observed * 1.0e-4,
                    "published_tanh_fit_drag_coefficient": fitted * 1.0e-4,
                    "published_critical_mach": critical_mach,
                }
            )
    result = pd.DataFrame(rows).sort_values(["lift_coefficient", "mach"])
    if result.empty or set(result["lift_coefficient"]) != {0.3, 0.4, 0.5}:
        raise ValueError("A320 source curves were not decoded completely")
    return result.reset_index(drop=True)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> ModelMetrics:
    residual = predicted - observed
    return ModelMetrics(
        rmse_drag_coefficient=float(np.sqrt(np.mean(residual**2))),
        rmse_drag_counts=float(np.sqrt(np.mean(residual**2)) * 1.0e4),
        rmspe=float(np.sqrt(np.mean((residual / observed) ** 2))),
        mean_bias_drag_counts=float(np.mean(residual) * 1.0e4),
        maximum_absolute_error_drag_counts=float(np.max(np.abs(residual)) * 1.0e4),
    )


def _evaluate(
    source: pd.DataFrame,
    published: TransonicCompressibilityDragModel,
) -> tuple[pd.DataFrame, dict[str, ModelMetrics]]:
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    polar = reference_drag
    gur_lock = ParabolicWaveDragModel(
        polar,
        KornLockWaveDrag(
            wing_sweep_rad=np.deg2rad(float(aircraft.wing.sweep)),
            thickness_to_chord_ratio=float(aircraft.wing.t_c),
        ),
    )
    result = source.copy()
    lift = result["lift_coefficient"].to_numpy(dtype=float)
    mach = result["mach"].to_numpy(dtype=float)
    published_breakdown = published.breakdown(lift, mach)
    result["published_model_drag_coefficient"] = published_breakdown.total
    result["published_baseline_drag_coefficient"] = (
        published_breakdown.parasite + published_breakdown.induced
    )
    result["published_baseline_gur_lock_drag_coefficient"] = (
        result["published_baseline_drag_coefficient"].to_numpy(dtype=float)
        + gur_lock.wave_drag.drag_coefficient_increment(mach, lift)
    )
    result["aerox_parabolic_drag_coefficient"] = polar.drag_coefficient(lift, mach)
    result["aerox_gur_lock_drag_coefficient"] = gur_lock.drag_coefficient(lift, mach)
    observed = result["published_observed_drag_coefficient"].to_numpy(dtype=float)
    predictions = {
        "published_tanh_cached": result[
            "published_tanh_fit_drag_coefficient"
        ].to_numpy(dtype=float),
        "published_tanh_reproduced": result[
            "published_model_drag_coefficient"
        ].to_numpy(dtype=float),
        "published_baseline": result[
            "published_baseline_drag_coefficient"
        ].to_numpy(dtype=float),
        "published_baseline_gur_lock": result[
            "published_baseline_gur_lock_drag_coefficient"
        ].to_numpy(dtype=float),
        "aerox_parabolic": result["aerox_parabolic_drag_coefficient"].to_numpy(
            dtype=float
        ),
        "aerox_gur_lock": result["aerox_gur_lock_drag_coefficient"].to_numpy(
            dtype=float
        ),
    }
    for name, prediction in predictions.items():
        result[f"{name}_residual_drag_counts"] = (prediction - observed) * 1.0e4
    return result, {
        name: _metrics(observed, prediction) for name, prediction in predictions.items()
    }


def _qar_support(
    path: Path, reference: pd.DataFrame, published: TransonicCompressibilityDragModel
) -> dict[str, float | int]:
    qar = pd.read_csv(path)
    lift = qar["lift_coefficient"].to_numpy(dtype=float)
    mach = qar["actual_mach"].to_numpy(dtype=float)
    minimum_mach = float(reference["mach"].min())
    maximum_mach = float(reference["mach"].max())
    in_support = (
        (lift >= published.wave_drag.minimum_lift_coefficient)
        & (lift <= published.wave_drag.maximum_lift_coefficient)
        & (mach >= minimum_mach)
        & (mach <= maximum_mach)
    )
    return {
        "sample_count": len(qar),
        "lift_coefficient_minimum": float(np.min(lift)),
        "lift_coefficient_median": float(np.median(lift)),
        "lift_coefficient_maximum": float(np.max(lift)),
        "mach_minimum": float(np.min(mach)),
        "mach_maximum": float(np.max(mach)),
        "samples_inside_full_published_support": int(np.sum(in_support)),
        "fraction_inside_full_published_support": float(np.mean(in_support)),
    }


def _plot(
    *,
    reference: pd.DataFrame,
    qar_path: Path,
    prefix: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(183.0 / 25.4, 126.0 / 25.4), constrained_layout=True
    )
    for axis, lift in zip(axes.ravel()[:3], (0.3, 0.4, 0.5), strict=True):
        subset = reference[reference["lift_coefficient"] == lift]
        mach = subset["mach"]
        axis.scatter(
            mach,
            subset["published_observed_drag_coefficient"] * 1.0e4,
            s=13,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
            label="digitized published polar",
        )
        axis.plot(
            mach,
            subset["published_model_drag_coefficient"] * 1.0e4,
            color="#0072B2",
            linewidth=1.6,
            label="published tanh fit",
        )
        axis.plot(
            mach,
            subset["published_baseline_gur_lock_drag_coefficient"] * 1.0e4,
            color="#D55E00",
            linestyle="--",
            linewidth=1.4,
            label="published baseline + Aerox Gur/Lock",
        )
        axis.plot(
            mach,
            subset["published_baseline_drag_coefficient"] * 1.0e4,
            color="#666666",
            linestyle=":",
            linewidth=1.2,
            label="published baseline without wave drag",
        )
        axis.axvline(
            float(subset["published_critical_mach"].iloc[0]),
            color="#009E73",
            linestyle="-.",
            linewidth=0.9,
            label="published critical Mach",
        )
        axis.set(
            xlabel="Mach number",
            ylabel="Drag coefficient (counts)",
            title=f"Published A320 curve at CL={lift:.1f}",
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    axes[0, 0].legend(loc="upper left")

    qar = pd.read_csv(qar_path)
    axes[1, 1].scatter(
        qar["actual_mach"],
        qar["lift_coefficient"],
        s=5,
        color="#0072B2",
        alpha=0.22,
        rasterized=True,
        label="FLT0003 cruise",
    )
    minimum_mach = float(reference["mach"].min())
    maximum_mach = float(reference["mach"].max())
    axes[1, 1].add_patch(
        Rectangle(
            (minimum_mach, 0.3),
            maximum_mach - minimum_mach,
            0.2,
            facecolor="#E69F00",
            edgecolor="#A05A00",
            alpha=0.20,
            linewidth=1.0,
            label="published support",
        )
    )
    axes[1, 1].scatter(
        reference["mach"],
        reference["lift_coefficient"],
        marker="x",
        s=12,
        linewidths=0.6,
        color="black",
        label="published nodes",
    )
    axes[1, 1].set(
        xlabel="Mach number",
        ylabel="Lift coefficient",
        title="Public-polar versus QAR cruise support",
    )
    axes[1, 1].legend(loc="lower left")
    axes[1, 1].grid(color="#DDDDDD", linewidth=0.45)

    for label, axis in zip("ABCD", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
        )
    figure.suptitle(
        "A320 drag models against an independent published transonic polar",
        fontsize=11,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def _metrics_table(metrics: dict[str, ModelMetrics]) -> str:
    labels = {
        "published_tanh_cached": "Published tanh fit",
        "published_tanh_reproduced": "Aerox reproduction of published fit",
        "published_baseline": "Published baseline without wave drag",
        "published_baseline_gur_lock": "Published baseline + Aerox Gur/Lock",
        "aerox_parabolic": "Aerox parabolic",
        "aerox_gur_lock": "Aerox Gur/Lock",
    }
    rows = ["| Model | RMSE (counts) | RMSPE | Mean bias (counts) |", "|---|---:|---:|---:|"]
    for name, metric in metrics.items():
        rows.append(
            f"| {labels[name]} | {metric.rmse_drag_counts:.2f} | "
            f"{100.0 * metric.rmspe:.2f}% | {metric.mean_bias_drag_counts:+.2f} |"
        )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_xlsx.is_file():
        raise FileNotFoundError(
            f"{args.source_xlsx} is absent; download HyperbolicTangent.xlsx from "
            f"{SOURCE_DOI}"
        )
    cells = _xlsx_cells(args.source_xlsx, SHEET_NAME)
    published = _published_model(cells)
    source = _source_points(cells)
    reference, metrics = _evaluate(source, published)
    qar_support = _qar_support(args.qar_points, reference, published)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    current_wave = KornLockWaveDrag(
        wing_sweep_rad=np.deg2rad(float(aircraft.wing.sweep)),
        thickness_to_chord_ratio=float(aircraft.wing.t_c),
    )
    comparison_lift = 0.5
    comparison_mach = 0.78
    published_comparison = published.breakdown(comparison_lift, comparison_mach)
    current_wave_coefficient = float(
        current_wave.drag_coefficient_increment(comparison_mach, comparison_lift)
    )
    representative_comparison = {
        "lift_coefficient": comparison_lift,
        "mach": comparison_mach,
        "published_zero_lift_drag_coefficient": published.zero_lift_drag_coefficient,
        "aerox_zero_lift_drag_coefficient": reference_drag.zero_lift_drag_coefficient,
        "published_effective_induced_drag_factor": float(
            published_comparison.induced / comparison_lift**2
        ),
        "aerox_induced_drag_factor": reference_drag.induced_drag_factor,
        "published_critical_mach": float(published_comparison.critical_mach),
        "aerox_gur_critical_mach": float(current_wave.critical_mach(comparison_lift)),
        "published_wave_drag_counts": float(published_comparison.wave * 1.0e4),
        "aerox_gur_wave_drag_counts": current_wave_coefficient * 1.0e4,
        "aerox_to_published_wave_drag_ratio": float(
            current_wave_coefficient / published_comparison.wave
        ),
    }
    qar_band = reference[
        (reference["mach"] >= qar_support["mach_minimum"])
        & (reference["mach"] <= qar_support["mach_maximum"])
    ]
    qar_band_observed = qar_band[
        "published_observed_drag_coefficient"
    ].to_numpy(dtype=float)
    qar_band_metrics = {
        name: asdict(
            _metrics(
                qar_band_observed,
                qar_band[column].to_numpy(dtype=float),
            )
        )
        for name, column in {
            "published_tanh_reproduced": "published_model_drag_coefficient",
            "published_baseline": "published_baseline_drag_coefficient",
            "published_baseline_gur_lock": "published_baseline_gur_lock_drag_coefficient",
            "aerox_parabolic": "aerox_parabolic_drag_coefficient",
            "aerox_gur_lock": "aerox_gur_lock_drag_coefficient",
        }.items()
    }
    reproduced_difference = float(
        np.max(
            np.abs(
                reference["published_model_drag_coefficient"]
                - reference["published_tanh_fit_drag_coefficient"]
            )
        )
    )
    per_lift_metrics = {
        f"cl_{lift:.1f}": asdict(
            _metrics(
                subset["published_observed_drag_coefficient"].to_numpy(dtype=float),
                subset["aerox_gur_lock_drag_coefficient"].to_numpy(dtype=float),
            )
        )
        for lift, subset in reference.groupby("lift_coefficient", sort=True)
    }
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-200 public reference versus A320-214 QAR envelope",
        "status": "independent-reference-accepted-default-model-held",
        "source": {
            "dataset_author": "Marlis Krull",
            "title": "Identifying Wave Drag for the Generic Drag Polar Equation",
            "report": REPORT_URL,
            "poster_doi": POSTER_DOI,
            "dataset_doi": SOURCE_DOI,
            "dataset_file": "HyperbolicTangent.xlsx",
            "dataset_file_md5": SOURCE_MD5,
            "dataset_terms": "GNU GPL version 3",
            "derived_reference_artifact_license": "GNU GPL version 3",
            "upstream_drag_data": "Obert (2009), digitized by Krull (2025)",
            "source_sheet": SHEET_NAME,
        },
        "published_support": {
            "lift_coefficient": published.wave_drag.lift_coefficient_grid.tolist(),
            "critical_mach": published.wave_drag.critical_mach_grid.tolist(),
            "mach_minimum": float(reference["mach"].min()),
            "mach_maximum": float(reference["mach"].max()),
            "source_point_count": len(reference),
        },
        "published_model": {
            "zero_lift_drag_coefficient": published.zero_lift_drag_coefficient,
            "reference_induced_drag_factor": published.induced_drag.reference_induced_drag_factor,
            "compressibility_strength": published.induced_drag.compressibility_strength,
            "compressibility_exponent": published.induced_drag.compressibility_exponent,
            "compressibility_mach": published.induced_drag.compressibility_mach,
            "wave_amplitude": published.wave_drag.amplitude,
            "wave_mach_scale": published.wave_drag.mach_scale,
            "wave_shift": published.wave_drag.shift,
            "maximum_reproduction_difference": reproduced_difference,
        },
        "comparison_metrics": {
            name: asdict(metric) for name, metric in metrics.items()
        },
        "comparison_metrics_within_qar_mach_band": {
            "source_point_count": len(qar_band),
            "metrics": qar_band_metrics,
        },
        "representative_component_comparison": representative_comparison,
        "aerox_gur_lock_metrics_by_lift": per_lift_metrics,
        "qar_cruise_support": qar_support,
        "decision": {
            "published_model_implemented": True,
            "independent_a320_transonic_reference_available": True,
            "default_plant_changed": False,
            "reason": "The published curves independently test the drag law, but stop at CL=0.5 while the representative QAR cruise lies above that boundary; the A320-200 source also does not isolate A320-214 engine or airframe subvariants.",
            "next_gate": "Seek or generate transonic data through at least CL=0.6, then validate a bounded interpolation before recalibrating thrust and replaying trajectories.",
        },
        "artifacts": {
            "reference": str(args.reference),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.reference, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _plot(reference=reference, qar_path=args.qar_points, prefix=args.figure_prefix)
    metric_table = _metrics_table(metrics)
    report = f"""# A320 independent transonic-drag reference audit

## Result

An independent public A320 transonic drag polar is now available to Aerox. Krull
digitized the A320-200 `CD(Mach)` curves published by Obert at
`CL=0.3`, `0.4`, and `0.5`, then fitted several wave-drag equations. The study's
hyperbolic-tangent form had the best aggregate fit across 16 aircraft. Aerox
reproduces the cached A320 fit to a maximum coefficient difference of
`{reproduced_difference:.3e}`.

![A320 independent transonic drag audit](figures/a320-published-transonic-drag-audit.png)

Figure 1. Digitized public drag curves and the published tanh, Aerox Gur/Lock,
and no-wave models on the same published baseline. Panel D shows the important
support mismatch with the representative FLT0003 QAR cruise. A
[vector PDF](figures/a320-published-transonic-drag-audit.pdf) is also provided.

## Model comparison

{metric_table}

The published model uses constant zero-lift drag, Mach-dependent induced drag,
and a CL-dependent hyperbolic-tangent wave term. This is a materially stronger
reference than inferring drag from the existing QAR effective-thrust map because
it is independent of Aerox's propulsion calibration.

On the same published zero-lift and induced-drag baseline, adding the Aerox
Gur/Lock wave term reduces the full-source RMSE from
{metrics['published_baseline'].rmse_drag_counts:.2f} to
{metrics['published_baseline_gur_lock'].rmse_drag_counts:.2f} counts. Within
the QAR Mach band it reduces RMSE from
{qar_band_metrics['published_baseline']['rmse_drag_counts']:.2f} to
{qar_band_metrics['published_baseline_gur_lock']['rmse_drag_counts']:.2f}
counts, compared with
{qar_band_metrics['published_tanh_reproduced']['rmse_drag_counts']:.2f} counts
for the published tanh model. Thus the Lock term moves in the correct direction
but does not reproduce the observed transonic rise nearly as well.

At `CL=0.5` and `Mach=0.78`, the published fit has critical Mach
{representative_comparison['published_critical_mach']:.3f} and
{representative_comparison['published_wave_drag_counts']:.2f} wave-drag counts.
The Aerox Gur prior gives critical Mach
{representative_comparison['aerox_gur_critical_mach']:.3f} and only
{representative_comparison['aerox_gur_wave_drag_counts']:.2f} counts. However,
Aerox's zero-lift coefficient is
{representative_comparison['aerox_zero_lift_drag_coefficient']:.5f} versus the
published {representative_comparison['published_zero_lift_drag_coefficient']:.5f}.
This explains why the current total polar can overpredict drag even while its
wave component underpredicts the published transonic rise. Baseline and wave
terms must therefore be recalibrated together.

## Support and release decision

The public source contains {len(reference)} points over Mach
`{reference['mach'].min():.3f}` to `{reference['mach'].max():.3f}` and only
`CL=0.3` to `0.5`. The representative QAR cruise has median
`CL={qar_support['lift_coefficient_median']:.3f}`; only
{100.0 * qar_support['fraction_inside_full_published_support']:.1f}% of its
points lie inside the full public support.

Therefore the new model is implemented as a strict bounded reference but is not
selected as the default plant. Extrapolating the fitted critical-Mach curve above
`CL=0.5` would create precision unsupported by the published data. The next
useful gate is independent data through at least `CL=0.6`, followed by a joint
drag/thrust re-identification and full trajectory comparison.

## Provenance

- Krull report: `{REPORT_URL}`.
- Krull and Scholz poster: `{POSTER_DOI}`.
- Associated GPL-3.0 spreadsheet dataset: `{SOURCE_DOI}`.
- Imported file checksum (MD5): `{SOURCE_MD5}`.
- The derived reference CSV retains the dataset's GPL-3.0 terms.
- Original graphical polar: Obert (2009), digitized in the cited dataset.
"""
    args.report.write_text(report)
    return document


def main() -> None:
    document = run(parse_args())
    summary = {
        name: {
            "rmse_drag_counts": values["rmse_drag_counts"],
            "rmspe": values["rmspe"],
        }
        for name, values in document["comparison_metrics"].items()
    }
    summary["qar_cruise_support"] = document["qar_cruise_support"]
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
