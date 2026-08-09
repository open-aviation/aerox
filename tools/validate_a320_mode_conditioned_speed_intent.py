#!/usr/bin/env python3
"""Select and validate leakage-safe A320 phase-conditioned speed intent."""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero
from aerox.fms import (
    AFFINE_SELECTOR_FEATURES,
    PHASE_CONSTANT_FEATURES,
    ModeConditionedSpeedIntent,
    speed_intent_design,
    vertical_phase,
)
from aerox.modeling import GuidanceModeThresholds, robust_weighted_linear_fit
from tools.audit_a320_guidance_modes import _load_panel_flight, _read_n1_envelope

UPDATED = "2026-08-07"
KNOT_TO_MPS = 1852.0 / 3600.0
FT_TO_M = 0.3048
CANDIDATES = ("phase_constant", "selector", "affine_selector")
STABLE_MODES = ("CAS_HOLD_LIKE", "MACH_HOLD_LIKE")


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
        "--lofo",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent-lofo.csv"),
    )
    parser.add_argument(
        "--candidate-metrics",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent-flight-metrics.csv"),
    )
    parser.add_argument(
        "--representative",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent-representative.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/fms/a320-mode-conditioned-speed-intent.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/fms/figures/a320-mode-conditioned-speed-intent"),
    )
    return parser.parse_args()


def _operational_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["speed_mode"].isin(STABLE_MODES).to_numpy()
        & frame["actual_cas_kts"].between(180.0, 350.0).to_numpy()
        & frame["pressure_altitude_ft"].between(3_000.0, 41_000.0).to_numpy()
        & np.isfinite(
            frame[
                [
                    "actual_cas_kts",
                    "actual_mach",
                    "selected_speed_kts",
                    "selected_mach",
                    "pressure_altitude_ft",
                    "mass_kg",
                ]
            ].to_numpy(dtype=float)
        ).all(axis=1)
    )


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["planned_phase"] = vertical_phase(
        result["vertical_rate_ft_min"].to_numpy(dtype=float)
    )
    result["flight_id"] = (
        result["registration"].astype(str) + "/" + result["recording"].astype(str)
    )
    return result


def _fit_coefficients(
    frames: list[pd.DataFrame], *, structure: str, speed_kind: str
) -> tuple[np.ndarray, dict[str, Any]]:
    if structure == "selector":
        return np.empty(0), {"sample_count": 0, "fit_rmse": None}
    mode = "CAS_HOLD_LIKE" if speed_kind == "CAS" else "MACH_HOLD_LIKE"
    selector_column = "selected_speed_kts" if speed_kind == "CAS" else "selected_mach"
    target_column = "actual_cas_kts" if speed_kind == "CAS" else "actual_mach"
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for frame in frames:
        mask = _operational_mask(frame) & (frame["speed_mode"].to_numpy() == mode)
        # Five-second thinning limits serial duplication without moving mode bounds.
        mask &= np.arange(len(frame)) % 5 == 0
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        flight_id = str(frame["flight_id"].iloc[0])
        counts[flight_id] = count
        designs.append(
            speed_intent_design(
                selector=frame.loc[mask, selector_column].to_numpy(dtype=float),
                pressure_altitude_ft=frame.loc[mask, "pressure_altitude_ft"].to_numpy(
                    dtype=float
                ),
                mass_kg=frame.loc[mask, "mass_kg"].to_numpy(dtype=float),
                phase=frame.loc[mask, "planned_phase"].to_numpy(dtype=str),
                speed_kind=speed_kind,
                feature_set=structure,
            )
        )
        targets.append(frame.loc[mask, target_column].to_numpy(dtype=float))
        weights.append(np.full(count, 1.0 / count))
    if len(designs) < 2:
        raise ValueError(f"insufficient complete flights to fit {speed_kind} intent")
    design = np.vstack(designs)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    fit = robust_weighted_linear_fit(design, target, weight)
    return fit.coefficients, {
        "sample_count": len(target),
        "flight_count": len(counts),
        "thinning_s": 5,
        "equal_complete_flight_weighting": True,
        "fit_rmse": fit.base_weighted_rmse,
        "robust_scale": fit.robust_scale,
        "normal_matrix_condition": fit.normal_matrix_condition,
    }


def _fit_model(
    frames: list[pd.DataFrame], structure: str
) -> tuple[ModeConditionedSpeedIntent, dict[str, Any]]:
    cas, cas_fit = _fit_coefficients(frames, structure=structure, speed_kind="CAS")
    mach, mach_fit = _fit_coefficients(frames, structure=structure, speed_kind="MACH")
    return (
        ModeConditionedSpeedIntent(
            structure=structure,
            cas_coefficients=cas,
            mach_coefficients=mach,
        ),
        {"cas": cas_fit, "mach": mach_fit},
    )


def _prediction_frame(
    frame: pd.DataFrame, model: ModeConditionedSpeedIntent
) -> pd.DataFrame:
    prediction = model.predict(
        selected_cas_kts=frame["selected_speed_kts"].to_numpy(dtype=float),
        selected_mach=frame["selected_mach"].to_numpy(dtype=float),
        pressure_altitude_ft=frame["pressure_altitude_ft"].to_numpy(dtype=float),
        mass_kg=frame["mass_kg"].to_numpy(dtype=float),
        phase=frame["planned_phase"].to_numpy(dtype=str),
    )
    result = frame.copy()
    result["intent_cas_target_kts"] = prediction.cas_target_kts
    result["intent_mach_target"] = prediction.mach_target
    result["intent_unconstrained_crossover_altitude_ft"] = (
        prediction.unconstrained_crossover_altitude_ft
    )
    result["intent_crossover_altitude_ft"] = prediction.crossover_altitude_ft
    result["intent_active_kind"] = prediction.active_kind
    result["intent_active_cas_equivalent_kts"] = prediction.active_cas_equivalent_kts
    result["intent_active_mach_equivalent"] = np.asarray(
        aero.cas2mach(
            prediction.active_cas_equivalent_kts * KNOT_TO_MPS,
            frame["pressure_altitude_ft"].to_numpy(dtype=float) * FT_TO_M,
        ),
        dtype=float,
    )
    return result


def _metrics(
    frame: pd.DataFrame, model: ModeConditionedSpeedIntent, *, split: str
) -> dict[str, Any]:
    predicted = _prediction_frame(frame, model)
    mask = _operational_mask(predicted)
    cas_error = predicted["intent_active_cas_equivalent_kts"].to_numpy(
        dtype=float
    ) - predicted["actual_cas_kts"].to_numpy(dtype=float)
    mach_error = predicted["intent_active_mach_equivalent"].to_numpy(
        dtype=float
    ) - predicted["actual_mach"].to_numpy(dtype=float)
    cas_hold = mask & (predicted["speed_mode"].to_numpy() == "CAS_HOLD_LIKE")
    mach_hold = mask & (predicted["speed_mode"].to_numpy() == "MACH_HOLD_LIKE")
    observed_kind = np.where(
        predicted["speed_mode"].to_numpy() == "MACH_HOLD_LIKE", "MACH", "CAS"
    )
    return {
        "registration": str(frame["registration"].iloc[0]),
        "recording": str(frame["recording"].iloc[0]),
        "role": str(frame["role"].iloc[0]),
        "split": split,
        "candidate": model.structure,
        "stable_sample_count": int(np.count_nonzero(mask)),
        "cas_hold_sample_count": int(np.count_nonzero(cas_hold)),
        "mach_hold_sample_count": int(np.count_nonzero(mach_hold)),
        "stable_cas_equivalent_rmse_kts": float(np.sqrt(np.mean(cas_error[mask] ** 2))),
        "stable_cas_equivalent_mae_kts": float(np.mean(np.abs(cas_error[mask]))),
        "stable_cas_equivalent_bias_kts": float(np.mean(cas_error[mask])),
        "cas_hold_rmse_kts": float(np.sqrt(np.mean(cas_error[cas_hold] ** 2)))
        if np.any(cas_hold)
        else float("nan"),
        "mach_hold_rmse": float(np.sqrt(np.mean(mach_error[mach_hold] ** 2)))
        if np.any(mach_hold)
        else float("nan"),
        "predicted_mach_branch_fraction": float(
            np.mean(predicted.loc[mask, "intent_active_kind"] == "MACH")
        ),
        "stable_branch_agreement_fraction": float(
            np.mean(
                predicted.loc[mask, "intent_active_kind"].to_numpy()
                == observed_kind[mask]
            )
        ),
    }


def _candidate_summary(lofo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        subset = lofo[lofo["candidate"] == candidate]
        rows.append(
            {
                "candidate": candidate,
                "lofo_flight_count": len(subset),
                "mean_lofo_stable_cas_equivalent_rmse_kts": float(
                    subset["stable_cas_equivalent_rmse_kts"].mean()
                ),
                "median_lofo_stable_cas_equivalent_rmse_kts": float(
                    subset["stable_cas_equivalent_rmse_kts"].median()
                ),
                "maximum_lofo_stable_cas_equivalent_rmse_kts": float(
                    subset["stable_cas_equivalent_rmse_kts"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "median_lofo_stable_cas_equivalent_rmse_kts", kind="stable"
    )


def _plot(
    *,
    lofo: pd.DataFrame,
    validation: pd.DataFrame,
    representative_predictions: dict[str, pd.DataFrame],
    selected: str,
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
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "phase_constant": "#0072B2",
        "selector": "#D55E00",
        "affine_selector": "#009E73",
    }
    labels = {
        "phase_constant": "phase-constant",
        "selector": "raw selectors",
        "affine_selector": "affine selector surrogate",
    }
    black, gray, magenta = "#000000", "#777777", "#CC79A7"
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(183.0 / 25.4, 9.1),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0]},
    )

    for index, candidate in enumerate(CANDIDATES):
        values = lofo.loc[
            lofo["candidate"] == candidate,
            "stable_cas_equivalent_rmse_kts",
        ].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[0, 0].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=colors[candidate],
            s=13,
            alpha=0.8,
        )
        axes[0, 0].plot(
            [index - 0.22, index + 0.22],
            [np.median(values)] * 2,
            color=black,
            linewidth=1.4,
        )
    axes[0, 0].set_xticks(range(len(CANDIDATES)), [labels[name] for name in CANDIDATES])
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].set_ylabel("Stable CAS-equivalent RMSE (kt)")
    axes[0, 0].set_title("Development leave-one-flight-out selection")

    flight_labels = (
        validation["registration"].astype(str)
        + "\n"
        + validation["recording"].astype(str)
    ).unique()
    width = 0.24
    x = np.arange(len(flight_labels))
    for offset, candidate in enumerate(CANDIDATES):
        subset = validation[validation["candidate"] == candidate]
        axes[0, 1].bar(
            x + (offset - 1) * width,
            subset["stable_cas_equivalent_rmse_kts"],
            width=width,
            color=colors[candidate],
            label=labels[candidate] + (" (selected)" if candidate == selected else ""),
        )
    axes[0, 1].set_xticks(x, flight_labels)
    axes[0, 1].tick_params(axis="x", labelsize=5.8)
    axes[0, 1].set_ylabel("Stable CAS-equivalent RMSE (kt)")
    axes[0, 1].set_title("Frozen six-flight validation")
    axes[0, 1].legend(ncols=1, loc="upper left")

    representative = representative_predictions[selected]
    baseline = representative_predictions["phase_constant"]
    time_min = representative["elapsed_time_s"].to_numpy(dtype=float) / 60.0
    phase_colors = {"CLIMB": "#56B4E9", "LEVEL": "#E6E6E6", "DESCENT": "#F0E442"}
    phase = representative["planned_phase"].to_numpy(dtype=str)
    boundaries = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1, len(phase)]
    for axis in axes[1:, :].ravel():
        for start, stop in pairwise(boundaries):
            axis.axvspan(
                time_min[start],
                time_min[stop - 1],
                color=phase_colors[phase[start]],
                alpha=0.09,
                linewidth=0,
            )

    axes[1, 0].plot(
        time_min,
        representative["pressure_altitude_ft"] / 1_000.0,
        color=black,
        label="QAR pressure altitude",
    )
    axes[1, 0].set_ylabel("Pressure altitude (kft)")
    axes[1, 0].set_title("Post-holdout phase alignment (not a model input target)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        time_min, representative["actual_cas_kts"], color=black, label="QAR CAS"
    )
    axes[1, 1].step(
        time_min,
        representative["selected_speed_kts"],
        where="post",
        color=gray,
        linewidth=0.7,
        linestyle=":",
        label="SPEEDP_SEL evidence",
    )
    axes[1, 1].plot(
        time_min,
        baseline["intent_active_cas_equivalent_kts"],
        color=colors["phase_constant"],
        linestyle="--",
        label="phase-constant intent",
    )
    axes[1, 1].plot(
        time_min,
        representative["intent_active_cas_equivalent_kts"],
        color=colors[selected],
        label="selected affine intent",
    )
    axes[1, 1].set_ylabel("CAS (kt)")
    axes[1, 1].set_title("Intent on the CAS axis")
    axes[1, 1].legend(ncols=2, loc="lower center", fontsize=6.0)

    axes[2, 0].plot(
        time_min, representative["actual_mach"], color=black, label="QAR Mach"
    )
    axes[2, 0].step(
        time_min,
        representative["selected_mach"],
        where="post",
        color=gray,
        linewidth=0.7,
        linestyle=":",
        label="MACH_SEL evidence",
    )
    axes[2, 0].plot(
        time_min,
        baseline["intent_active_mach_equivalent"],
        color=colors["phase_constant"],
        linestyle="--",
        label="phase-constant intent",
    )
    axes[2, 0].plot(
        time_min,
        representative["intent_active_mach_equivalent"],
        color=colors[selected],
        label="selected affine intent",
    )
    axes[2, 0].set(xlabel="Elapsed time (min)", ylabel="Mach number")
    axes[2, 0].set_title("Intent on the Mach axis")
    axes[2, 0].legend(ncols=2, loc="lower center", fontsize=6.0)

    selected_error = (
        representative["intent_active_cas_equivalent_kts"]
        - representative["actual_cas_kts"]
    )
    baseline_error = (
        baseline["intent_active_cas_equivalent_kts"] - baseline["actual_cas_kts"]
    )
    axes[2, 1].plot(
        time_min,
        baseline_error,
        color=colors["phase_constant"],
        linestyle="--",
        label="phase-constant error",
    )
    axes[2, 1].plot(
        time_min,
        selected_error,
        color=colors[selected],
        label="selected-model error",
    )
    axes[2, 1].axhline(0.0, color=black, linewidth=0.7)
    axes[2, 1].axhline(10.0, color=magenta, linewidth=0.6, linestyle=":")
    axes[2, 1].axhline(-10.0, color=magenta, linewidth=0.6, linestyle=":")
    axes[2, 1].set(xlabel="Elapsed time (min)", ylabel="CAS-equivalent error (kt)")
    axes[2, 1].set_title("Post-holdout target residual")
    axes[2, 1].legend()

    for label, axis in zip("ABCDEF", axes.ravel(), strict=True):
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "pad": 1.0,
            },
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.6)
    figure.suptitle(
        "A320 leakage-safe phase-conditioned speed-intent gate", fontsize=12
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    plt.close(figure)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| Candidate | LOFO median (kt) | Validation median (kt) | Validation max (kt) |"
    rule = "|---|---:|---:|---:|"
    rows = []
    for candidate in CANDIDATES:
        fit = frame[
            (frame["candidate"] == candidate) & (frame["split"] == "lofo_summary")
        ].iloc[0]
        val = frame[
            (frame["candidate"] == candidate) & (frame["split"] == "validation_summary")
        ].iloc[0]
        rows.append(
            f"| `{candidate}` | {fit['median_rmse_kts']:.2f} | "
            f"{val['median_rmse_kts']:.2f} | {val['maximum_rmse_kts']:.2f} |"
        )
    return "\n".join([header, rule, *rows])


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = pd.read_csv(args.panel)
    envelope = _read_n1_envelope(args.n1_envelope)
    thresholds = GuidanceModeThresholds()
    frames: list[pd.DataFrame] = []
    for row in panel.itertuples(index=False):
        frames.append(
            _prepare(
                _load_panel_flight(
                    raw_dir=args.raw_dir,
                    registration=str(row.registration),
                    recording=str(row.recording),
                    role=str(row.role),
                    envelope=envelope,
                    thresholds=thresholds,
                )
            )
        )
    fit_frames = [
        frame for frame in frames if str(frame["role"].iloc[0]) != "validation"
    ]
    validation_frames = [
        frame for frame in frames if str(frame["role"].iloc[0]) == "validation"
    ]
    if len(fit_frames) != 10 or len(validation_frames) != 6:
        raise ValueError("expected ten development and six validation flights")

    lofo_rows: list[dict[str, Any]] = []
    for held_index, held in enumerate(fit_frames):
        training = [
            frame for index, frame in enumerate(fit_frames) if index != held_index
        ]
        for candidate in CANDIDATES:
            model, _ = _fit_model(training, candidate)
            lofo_rows.append(_metrics(held, model, split="development_lofo"))
    lofo = pd.DataFrame(lofo_rows)
    candidate_selection = _candidate_summary(lofo)
    selected = str(candidate_selection.iloc[0]["candidate"])

    frozen_models: dict[str, ModeConditionedSpeedIntent] = {}
    fit_diagnostics: dict[str, Any] = {}
    validation_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        model, diagnostics = _fit_model(fit_frames, candidate)
        frozen_models[candidate] = model
        fit_diagnostics[candidate] = diagnostics
        for frame in validation_frames:
            validation_rows.append(_metrics(frame, model, split="validation"))
    validation = pd.DataFrame(validation_rows)

    representative = _prepare(
        _load_panel_flight(
            raw_dir=args.raw_dir,
            registration="TAIL-A",
            recording="FLT0003",
            role="post_holdout_diagnostic",
            envelope=envelope,
            thresholds=thresholds,
        )
    )
    representative_predictions = {
        candidate: _prediction_frame(representative, model)
        for candidate, model in frozen_models.items()
    }
    representative_metrics = pd.DataFrame(
        [
            _metrics(representative, model, split="post_holdout_diagnostic")
            for model in frozen_models.values()
        ]
    )

    summary_rows = []
    for candidate in CANDIDATES:
        lofo_subset = lofo[lofo["candidate"] == candidate]
        validation_subset = validation[validation["candidate"] == candidate]
        summary_rows.extend(
            [
                {
                    "candidate": candidate,
                    "split": "lofo_summary",
                    "flight_count": len(lofo_subset),
                    "mean_rmse_kts": float(
                        lofo_subset["stable_cas_equivalent_rmse_kts"].mean()
                    ),
                    "median_rmse_kts": float(
                        lofo_subset["stable_cas_equivalent_rmse_kts"].median()
                    ),
                    "maximum_rmse_kts": float(
                        lofo_subset["stable_cas_equivalent_rmse_kts"].max()
                    ),
                },
                {
                    "candidate": candidate,
                    "split": "validation_summary",
                    "flight_count": len(validation_subset),
                    "mean_rmse_kts": float(
                        validation_subset["stable_cas_equivalent_rmse_kts"].mean()
                    ),
                    "median_rmse_kts": float(
                        validation_subset["stable_cas_equivalent_rmse_kts"].median()
                    ),
                    "maximum_rmse_kts": float(
                        validation_subset["stable_cas_equivalent_rmse_kts"].max()
                    ),
                },
            ]
        )
    summaries = pd.DataFrame(summary_rows)
    selected_validation = summaries[
        (summaries["candidate"] == selected)
        & (summaries["split"] == "validation_summary")
    ].iloc[0]
    baseline_validation = summaries[
        (summaries["candidate"] == "phase_constant")
        & (summaries["split"] == "validation_summary")
    ].iloc[0]
    improvement = 1.0 - float(selected_validation["median_rmse_kts"]) / float(
        baseline_validation["median_rmse_kts"]
    )
    selected_validation_flights = validation[validation["candidate"] == selected]
    median_branch_agreement = float(
        selected_validation_flights["stable_branch_agreement_fraction"].median()
    )
    selected_model = frozen_models[selected]
    passed = (
        selected != "phase_constant"
        and improvement > 0.0
        and median_branch_agreement >= 0.80
    )
    status = "passed" if passed else "held"
    feature_names = {
        "phase_constant": PHASE_CONSTANT_FEATURES,
        "selector": (),
        "affine_selector": AFFINE_SELECTOR_FEATURES,
    }
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214 (TAIL-A and TAIL-B)",
        "status": status,
        "decision": {
            "selected_structure": selected,
            "selection_metric": "median complete-flight LOFO stable CAS-equivalent RMSE",
            "selection_used_validation_flights": False,
            "strictly_untouched_after_complete_pipeline_freeze": False,
            "validation_status": "confirmation after uniform semantic crossover-guard QC",
            "crossover_guard_selected_from_validation_metrics": False,
            "validation_improvement_over_phase_constant_fraction": improvement,
            "median_validation_branch_agreement_fraction": median_branch_agreement,
            "minimum_validation_branch_agreement_gate_fraction": 0.80,
            "release_for_fms_integration": passed,
        },
        "data_split": {
            "development_flights": len(fit_frames),
            "validation_flights": len(validation_frames),
            "registrations": sorted(panel["registration"].unique().tolist()),
            "validation_recordings": [
                f"{frame['registration'].iloc[0]}/{frame['recording'].iloc[0]}"
                for frame in validation_frames
            ],
            "post_holdout_diagnostic": "TAIL-A/FLT0003",
        },
        "leakage_boundary": {
            "prediction_inputs": [
                "planned phase",
                "pressure altitude",
                "gross mass",
                "SPEEDP_SEL evidence",
                "MACH_SEL evidence",
            ],
            "response_forbidden_from_prediction": [
                "actual CAS",
                "actual Mach",
                "inferred speed mode",
                "recorded N1",
                "fuel flow",
                "trajectory residual",
            ],
            "actual_response_use": "development fitting labels and held-out scoring only",
            "phase_alignment_note": (
                "QAR vertical rate supplies the offline scenario phase; a generated "
                "trajectory would supply the same phase from its TOC/TOD plan"
            ),
        },
        "operational_evaluation_envelope": {
            "minimum_pressure_altitude_ft": 3_000,
            "maximum_pressure_altitude_ft": 41_000,
            "minimum_actual_cas_kts": 180,
            "maximum_actual_cas_kts": 350,
            "behavioural_modes": list(STABLE_MODES),
            "minimum_mode_dwell_s": thresholds.minimum_dwell_s,
        },
        "model": {
            "structure": selected_model.structure,
            "feature_names": list(feature_names[selected]),
            "cas_coefficients": selected_model.cas_coefficients.tolist(),
            "mach_coefficients": selected_model.mach_coefficients.tolist(),
            "bounds": {
                "cas_kts": [
                    selected_model.minimum_cas_kts,
                    selected_model.maximum_cas_kts,
                ],
                "mach": [selected_model.minimum_mach, selected_model.maximum_mach],
                "crossover_altitude_ft": [
                    selected_model.minimum_crossover_altitude_ft,
                    selected_model.maximum_crossover_altitude_ft,
                ],
            },
            "branch_logic": (
                "Mach at or above target-pair crossover guarded to 25,000--35,000 ft; "
                "CAS below"
            ),
            "fit_diagnostics": fit_diagnostics[selected],
        },
        "candidate_summaries": summaries.to_dict(orient="records"),
        "representative_metrics": representative_metrics.to_dict(orient="records"),
        "artifacts": {
            "lofo": str(args.lofo),
            "candidate_metrics": str(args.candidate_metrics),
            "flight_metrics": str(args.flight_metrics),
            "representative": str(args.representative),
            "figure": str(args.figure_prefix.with_suffix(".png")),
            "reproducer": "tools/validate_a320_mode_conditioned_speed_intent.py",
        },
        "limitations": [
            "The labels are effective behaviours, not decoded Airbus FMA modes.",
            "Selected versus managed operation and ECON cost-index logic remain unobservable.",
            "This gate validates target inference, not closed-loop trajectory tracking.",
            "Takeoff, approach below 3,000 ft, flap effects, and terminal constraints are outside the fit.",
        ],
    }
    document = _native(document)
    table = _markdown_table(summaries)
    selected_rep = representative_metrics[
        representative_metrics["candidate"] == selected
    ].iloc[0]
    report = f"""# A320 phase-conditioned speed-intent validation

Updated: {UPDATED}

Status: **{status}**. The structure and coefficients were selected entirely on
ten development flights. The selected candidate is `{selected}`. Its median
six-flight confirmation stable-speed error is
{selected_validation["median_rmse_kts"]:.2f} kt CAS-equivalent, compared with
{baseline_validation["median_rmse_kts"]:.2f} kt for the development-only
phase-constant schedule ({improvement:.1%} relative improvement).
Median validation agreement between the predicted schedule branch and stable
behavioural mode is {median_branch_agreement:.1%}.

## What this gate tests

The Aerox FMS needs a CAS/Mach target before the aircraft responds. For that
reason, the predictor may use only planned climb/level/descent phase, pressure
altitude, gross mass, and the `SPEEDP_SEL`/`MACH_SEL` channels as imperfect
selector evidence. Actual CAS, actual Mach, inferred speed mode, N1, fuel flow,
and trajectory error never enter held-flight predictions.

The inferred `CAS_HOLD_LIKE` and `MACH_HOLD_LIKE` states are used only to choose
stable development labels and stable scoring intervals. They do not switch the
predicted schedule. The switch is the conventional response-blind CAS/Mach
crossover calculated from the two predicted targets, with the same predeclared
25--35 kft operational guard on every candidate. This prevents extrapolated
inactive selector values from silently eliminating either schedule branch.
The guard was added during figure-quality control after the validation
responses had been viewed; it was not tuned from their metrics, and the affine
structure remained the development-LOFO winner. Accordingly, the six-flight
result is confirmation rather than a pristine untouched test of the complete
final pipeline.

## Candidate decision

{table}

The phase-constant candidate is the transferable replacement for the earlier
single-flight post-response medians. The raw-selector candidate takes the held
values literally. The affine candidate calibrates their relation to stable
flown targets using altitude, gross mass, and planned phase. Every fitted
candidate uses equal complete-flight weight and five-second thinning; the
runtime predictor remains a 1 Hz algebraic model.

## Interpretation

The post-holdout `TAIL-A/FLT0003` diagnostic gives {selected_rep["stable_cas_equivalent_rmse_kts"]:.2f}
kt stable CAS-equivalent RMSE for the frozen selected model. This flight was not
used for structure selection or coefficient fitting. Its timeline is plotted
to reveal where a smooth intent explains the response and where acceleration,
wind, controller transients, or missing managed-target logic remain.

This is not a neural network and it is not a recovered Airbus FMGC law. It is a
small, inspectable surrogate whose coefficients and bounds are recorded in the
YAML artifact. The validation gate concerns target inference only. Closed-loop
integration into the TOC/TOD profile and a generated-versus-QAR trajectory
comparison is the next gate if this result passes.

## Evidence boundary

- Offline phase alignment comes from QAR vertical rate. In generation, the FMS
  already knows the corresponding phase from its planned TOC and TOD.
- Scoring is limited to persistent stable-speed behaviour between 3,000 and
  41,000 ft and 180--350 kt; terminal takeoff/approach targets remain boundary
  constraints rather than outputs of this surrogate.
- `MACH_SEL` may remain constant while flown Mach changes, so selector values
  are evidence rather than FMA truth.
- Selected versus managed speed and ECON/cost-index strategy remain
  unobservable in these recordings.

## Artifacts

- Machine-readable decision: `docs/fms/a320-mode-conditioned-speed-intent.yml`
- Development LOFO rows: `docs/fms/a320-mode-conditioned-speed-intent-lofo.csv`
- Candidate summaries: `docs/fms/a320-mode-conditioned-speed-intent-candidates.csv`
- Frozen validation flight metrics: `docs/fms/a320-mode-conditioned-speed-intent-flight-metrics.csv`
- Five-second post-holdout intent timeline: `docs/fms/a320-mode-conditioned-speed-intent-representative.csv`
- Figure: `docs/fms/figures/a320-mode-conditioned-speed-intent.{{png,pdf}}`
- Reproducer: `tools/validate_a320_mode_conditioned_speed_intent.py`
"""

    args.lofo.parent.mkdir(parents=True, exist_ok=True)
    lofo.to_csv(args.lofo, index=False)
    summaries.to_csv(args.candidate_metrics, index=False)
    pd.concat([validation, representative_metrics], ignore_index=True).to_csv(
        args.flight_metrics, index=False
    )
    representative_columns = [
        "elapsed_time_s",
        "pressure_altitude_ft",
        "actual_cas_kts",
        "actual_mach",
        "selected_speed_kts",
        "selected_mach",
        "mass_kg",
        "planned_phase",
        "speed_mode",
        "intent_cas_target_kts",
        "intent_mach_target",
        "intent_unconstrained_crossover_altitude_ft",
        "intent_crossover_altitude_ft",
        "intent_active_kind",
        "intent_active_cas_equivalent_kts",
        "intent_active_mach_equivalent",
    ]
    representative_predictions[selected].loc[::5, representative_columns].to_csv(
        args.representative, index=False
    )
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(
        lofo=lofo,
        validation=validation,
        representative_predictions=representative_predictions,
        selected=selected,
        prefix=args.figure_prefix,
    )
    print(yaml.safe_dump(document["decision"], sort_keys=False).strip())
    return document


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
