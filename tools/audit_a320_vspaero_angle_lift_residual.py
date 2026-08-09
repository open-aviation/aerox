#!/usr/bin/env python3
"""Attribute the failed A320 VSPAERO angle/lift fit gate on fit flights only.

This diagnostic compares a common type model with phase and registration
nuisance terms under complete-flight leave-one-out testing. Registration terms
are measurement/force-reconstruction nuisance parameters; they are never
interpreted as different A320 aerodynamic coefficients.
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

from aerox.models import TabulatedStaticStabilityPrior
from aerox.modeling.identification import RobustLinearFit, robust_weighted_linear_fit
from tools.calibrate_a320_effective_thrust import (
    _phase_equal_weights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.identify_a320_vspaero_angle_lift import (
    FlightArrays,
    WING_AREA_M2,
    _flight_arrays,
    _prior_values,
)
from tools.replay_a320_point_mass import (
    _load_point_mass_flights,
    _read_horizontal_parameters,
)

UPDATED = "2026-08-05"
REGISTRATION_CONTRAST = {"TAIL-A": -0.5, "TAIL-B": 0.5}
CANDIDATES = {
    "common": (),
    "registration": ("registration_contrast",),
    "phase": ("climb_offset", "descent_offset"),
    "registration_phase": (
        "registration_contrast",
        "climb_offset",
        "descent_offset",
    ),
}
MAX_RMSE_RATIO = 1.02
MAX_REGISTRATION_BIAS_DIFFERENCE_CL = 0.02


@dataclass(frozen=True)
class NuisanceFit:
    candidate: str
    feature_names: tuple[str, ...]
    result: RobustLinearFit

    @property
    def pitch_scale(self) -> float:
        return float(self.result.coefficients[0])

    @property
    def combined_offset_rad(self) -> float:
        return float(self.result.coefficients[1])

    @property
    def mach_offset_rad_per_mach(self) -> float:
        return float(self.result.coefficients[2])

    def nuisance_angle_rad(self, name: str) -> float:
        if name not in self.feature_names:
            return 0.0
        return float(self.result.coefficients[3 + self.feature_names.index(name)])


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
        "--vspaero-results",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-prior.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-residual-audit.yml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-residual-candidates.csv"),
    )
    parser.add_argument(
        "--flight-metrics",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-residual-flight-metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-angle-lift-residual-audit.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-angle-lift-residual-audit"),
    )
    return parser.parse_args()


def _nuisance_columns(flight: FlightArrays, names: tuple[str, ...]) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for name in names:
        if name == "registration_contrast":
            if flight.registration not in REGISTRATION_CONTRAST:
                raise ValueError(f"unknown registration: {flight.registration}")
            columns.append(
                np.full(len(flight.mach), REGISTRATION_CONTRAST[flight.registration])
            )
        elif name == "climb_offset":
            columns.append((flight.phase == "climb").astype(float))
        elif name == "descent_offset":
            columns.append((flight.phase == "descent").astype(float))
        else:
            raise ValueError(f"unknown nuisance feature: {name}")
    return columns


def _design_target(
    flight: FlightArrays,
    prior: TabulatedStaticStabilityPrior,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    cl_reference, cl_alpha = _prior_values(prior, flight.mach, 1.0)
    columns = [
        cl_alpha * flight.pitch_zero_rad,
        cl_alpha,
        cl_alpha * (flight.mach - 0.6),
    ]
    columns.extend(cl_alpha * values for values in _nuisance_columns(flight, names))
    target = (
        flight.target_cl
        - cl_reference
        + cl_alpha * (flight.gamma_rad + prior.reference_alpha_rad)
    )
    return np.column_stack(columns), target


def _fit(
    flights: list[FlightArrays],
    prior: TabulatedStaticStabilityPrior,
    candidate: str,
) -> NuisanceFit:
    names = CANDIDATES[candidate]
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for flight in flights:
        design, target = _design_target(flight, prior, names)
        designs.append(design)
        targets.append(target)
        weights.append(_phase_equal_weights(flight.phase) / len(flights))
    result = robust_weighted_linear_fit(
        np.vstack(designs), np.concatenate(targets), np.concatenate(weights)
    )
    return NuisanceFit(candidate, names, result)


def _predict(
    flight: FlightArrays,
    prior: TabulatedStaticStabilityPrior,
    fit: NuisanceFit,
) -> np.ndarray:
    cl_reference, cl_alpha = _prior_values(prior, flight.mach, 1.0)
    model_alpha = (
        fit.pitch_scale * flight.pitch_zero_rad
        + fit.combined_offset_rad
        + fit.mach_offset_rad_per_mach * (flight.mach - 0.6)
        - flight.gamma_rad
    )
    for name, values in zip(
        fit.feature_names, _nuisance_columns(flight, fit.feature_names), strict=True
    ):
        model_alpha += fit.nuisance_angle_rad(name) * values
    return cl_reference + cl_alpha * (model_alpha - prior.reference_alpha_rad)


def _registration_bias_difference(rows: list[dict[str, Any]]) -> float:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("registration")["cl_bias"].mean()
    return float(grouped.max() - grouped.min())


def _evaluate_candidate(
    flights: list[FlightArrays],
    prior: TabulatedStaticStabilityPrior,
    candidate: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds: list[NuisanceFit] = []
    flight_rows: list[dict[str, Any]] = []
    for held_out in flights:
        fit = _fit(
            [flight for flight in flights if flight.key != held_out.key],
            prior,
            candidate,
        )
        folds.append(fit)
        residual = _predict(held_out, prior, fit) - held_out.target_cl
        flight_rows.append(
            {
                "candidate": candidate,
                "flight": held_out.key,
                "registration": held_out.registration,
                "samples": len(residual),
                "cl_rmse": float(np.sqrt(np.mean(residual**2))),
                "cl_bias": float(np.mean(residual)),
                "pitch_scale": fit.pitch_scale,
                "registration_angle_difference_deg": math.degrees(
                    fit.nuisance_angle_rad("registration_contrast")
                ),
                "climb_angle_offset_deg": math.degrees(
                    fit.nuisance_angle_rad("climb_offset")
                ),
                "descent_angle_offset_deg": math.degrees(
                    fit.nuisance_angle_rad("descent_offset")
                ),
            }
        )
    full = _fit(flights, prior, candidate)
    rmse = float(np.mean([row["cl_rmse"] for row in flight_rows]))
    reg_difference = _registration_bias_difference(flight_rows)
    registration_fold_values = np.asarray(
        [math.degrees(fit.nuisance_angle_rad("registration_contrast")) for fit in folds]
    )
    row = {
        "candidate": candidate,
        "feature_count": 3 + len(CANDIDATES[candidate]),
        "lofo_mean_flight_cl_rmse": rmse,
        "lofo_registration_bias_difference_cl": reg_difference,
        "fit_weighted_cl_rmse": full.result.base_weighted_rmse,
        "pitch_scale": full.pitch_scale,
        "combined_offset_deg": math.degrees(full.combined_offset_rad),
        "mach_offset_deg_per_mach": math.degrees(full.mach_offset_rad_per_mach),
        "registration_angle_difference_deg": math.degrees(
            full.nuisance_angle_rad("registration_contrast")
        ),
        "registration_angle_difference_lofo_min_deg": float(
            registration_fold_values.min()
        ),
        "registration_angle_difference_lofo_max_deg": float(
            registration_fold_values.max()
        ),
        "climb_angle_offset_deg": math.degrees(
            full.nuisance_angle_rad("climb_offset")
        ),
        "descent_angle_offset_deg": math.degrees(
            full.nuisance_angle_rad("descent_offset")
        ),
        "normal_matrix_condition": full.result.normal_matrix_condition,
    }
    return row, flight_rows


def build_audit(
    flights: list[FlightArrays], prior: TabulatedStaticStabilityPrior
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    all_flight_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        row, flight_rows = _evaluate_candidate(flights, prior, candidate)
        candidate_rows.append(row)
        all_flight_rows.extend(flight_rows)
    common = next(row for row in candidate_rows if row["candidate"] == "common")
    for row in candidate_rows:
        row["lofo_rmse_ratio_to_common"] = (
            row["lofo_mean_flight_cl_rmse"] / common["lofo_mean_flight_cl_rmse"]
        )
        row["passes_diagnostic_screen"] = (
            row["lofo_rmse_ratio_to_common"] <= MAX_RMSE_RATIO
            and row["lofo_registration_bias_difference_cl"]
            <= MAX_REGISTRATION_BIAS_DIFFERENCE_CL
        )
    eligible = [row for row in candidate_rows if row["passes_diagnostic_screen"]]
    selected = (
        min(
            eligible,
            key=lambda row: (row["feature_count"], row["lofo_mean_flight_cl_rmse"]),
        )
        if eligible
        else None
    )
    selected_name = selected["candidate"] if selected else None
    supports_registration_nuisance = bool(
        selected_name and "registration" in selected_name
    )
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": (
            "fit-only-registration-nuisance-supported"
            if supports_registration_nuisance
            else "fit-only-operating-nuisance-supported"
            if selected_name
            else "fit-only-residual-structure-unresolved"
        ),
        "data_boundary": {
            "fit_flights_only": True,
            "validation_flights_loaded": False,
            "flights": [flight.key for flight in flights],
            "registrations": sorted({flight.registration for flight in flights}),
            "samples": int(sum(len(flight.mach) for flight in flights)),
        },
        "question": "Can phase or tail-specific measurement nuisance terms explain the failed registration residual gate without degrading complete-flight prediction?",
        "candidate_contract": {
            "common_features": ["pitch_scale", "combined_offset", "mach_offset"],
            "nuisance_candidates": {name: list(features) for name, features in CANDIDATES.items()},
            "registration_contrast": REGISTRATION_CONTRAST,
            "maximum_lofo_rmse_ratio_to_common": MAX_RMSE_RATIO,
            "maximum_registration_bias_difference_cl": MAX_REGISTRATION_BIAS_DIFFERENCE_CL,
            "selection": "fewest coefficients passing both diagnostics, then lowest mean flight LOFO RMSE",
        },
        "result": {
            "selected_candidate": selected_name,
            "supports_registration_nuisance": supports_registration_nuisance,
            "selected_metrics": selected,
            "candidate_table": "docs/modeling/a320-vspaero-angle-lift-residual-candidates.csv",
            "flight_table": "docs/modeling/a320-vspaero-angle-lift-residual-flight-metrics.csv",
        },
        "decision": {
            "tail_specific_aerodynamics": False,
            "tail_specific_measurement_nuisance_for_next_fit_gate": supports_registration_nuisance,
            "open_validation": False,
            "change_replay_plant": False,
            "reason": "This post-failure fit-only diagnostic may choose a revised nuisance structure, but the local VSPAERO angle-support failure remains and the revised model has not passed a frozen confirmation gate.",
        },
        "limitations": [
            "Only two aircraft registrations are available, so a registration contrast is not a population-level random-effect estimate.",
            "The registration term confounds sensor zero, mass and air-data scale, vertical-wind omission, and force-reconstruction error.",
            "Phase terms are operating-condition diagnostics, not aerodynamic configuration coefficients.",
            "Validation flights remain unopened by this tool.",
        ],
    }
    return document, candidate_rows, all_flight_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(document: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    selected = document["result"]["selected_candidate"]
    rows = "\n".join(
        "| {candidate} | {lofo_mean_flight_cl_rmse:.4f} | {lofo_rmse_ratio_to_common:.3f} | "
        "{lofo_registration_bias_difference_cl:.4f} | {registration_angle_difference_deg:.3f} | {passes} |".format(
            **row, passes="pass" if row["passes_diagnostic_screen"] else "fail"
        )
        for row in candidates
    )
    interpretation = (
        "A registration contrast is the simplest structure that passes both diagnostics. "
        "It is supported only as a tail-specific measurement/force-reconstruction nuisance."
        if document["result"]["supports_registration_nuisance"]
        else "No tail-specific measurement nuisance is selected by this diagnostic."
    )
    return f"""# A320 VSPAERO angle/lift residual audit

## Outcome

Selected fit-only diagnostic structure: **{selected or 'none'}**. {interpretation}

| Candidate | Mean flight LOFO CL RMSE | Ratio to common | Registration bias difference (CL) | AFZ−AFC angle nuisance (deg) | Screen |
|---|---:|---:|---:|---:|---:|
{rows}

## Interpretation boundary

All candidates keep the VSPAERO `CLalpha(M)` curve fixed. Phase offsets test a
simple operating-condition explanation. The balanced registration contrast
tests whether a persistent two-tail residual remains after the shared pitch
scale, angle/model offset, and Mach offset. It does not represent different
wing incidence or lift curves for the two A320-214 aircraft.

The selected structure must not be installed in the replay plant yet. The
original angle/lift gate also failed because 6.8% of modeled angles were outside
the local −1° to 6° VSPAERO support. That support question must be resolved
independently, after which a revised fit contract can be frozen before any
validation flight is opened.

## Decision

- Tail-specific physical aerodynamics: **no**.
- Tail-specific measurement nuisance in the next fit gate: **{'yes' if document['decision']['tail_specific_measurement_nuisance_for_next_fit_gate'] else 'no'}**.
- Open validation: **no**.
- Change replay plant: **no**.
"""


def _plot(
    candidates: list[dict[str, Any]],
    flight_rows: list[dict[str, Any]],
    selected: str | None,
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    names = [row["candidate"] for row in candidates]
    x = np.arange(len(names))
    colors = ["#009E73" if row["passes_diagnostic_screen"] else "#D55E00" for row in candidates]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    flat = axes.ravel()

    rmse = [row["lofo_mean_flight_cl_rmse"] for row in candidates]
    common_rmse = candidates[0]["lofo_mean_flight_cl_rmse"]
    flat[0].bar(x, rmse, color=colors, hatch=["//", "..", "xx", "--"])
    flat[0].axhline(common_rmse * MAX_RMSE_RATIO, color="black", linestyle="--", linewidth=1)
    flat[0].set_xticks(x, names, rotation=25, ha="right")
    flat[0].set(title="Predictive error", ylabel="Mean flight LOFO CL RMSE")

    bias = [row["lofo_registration_bias_difference_cl"] for row in candidates]
    flat[1].bar(x, bias, color=colors, hatch=["//", "..", "xx", "--"])
    flat[1].axhline(MAX_REGISTRATION_BIAS_DIFFERENCE_CL, color="black", linestyle="--", linewidth=1)
    flat[1].set_xticks(x, names, rotation=25, ha="right")
    flat[1].set(title="Residual tail separation", ylabel="AFZ−AFC absolute bias difference (CL)")

    chosen_rows = [row for row in flight_rows if row["candidate"] == selected]
    for registration, marker, color in (
        ("TAIL-A", "o", "#0072B2"),
        ("TAIL-B", "s", "#D55E00"),
    ):
        subset = [row for row in chosen_rows if row["registration"] == registration]
        indices = [chosen_rows.index(row) for row in subset]
        flat[2].scatter(indices, [row["cl_bias"] for row in subset], marker=marker, color=color, label=registration)
    flat[2].axhline(0, color="black", linewidth=0.8)
    flat[2].set_xticks(
        np.arange(len(chosen_rows)),
        [row["flight"].split("/", 1)[1] for row in chosen_rows],
        rotation=50,
        ha="right",
    )
    flat[2].set(title=f"Selected model residuals: {selected}", ylabel="Mean CL residual")
    flat[2].legend(frameon=False)

    offsets = np.asarray(
        [[row["registration_angle_difference_deg"], row["climb_angle_offset_deg"], row["descent_angle_offset_deg"]] for row in candidates]
    )
    offset_names = ["AFZ−AFC", "climb", "descent"]
    for index, (label, color, marker) in enumerate(
        zip(offset_names, ("#CC79A7", "#0072B2", "#E69F00"), ("o", "s", "^"), strict=True)
    ):
        flat[3].plot(x, offsets[:, index], marker=marker, color=color, label=label)
    flat[3].axhline(0, color="black", linewidth=0.8)
    flat[3].set_xticks(x, names, rotation=25, ha="right")
    flat[3].set(title="Fitted nuisance angles", ylabel="Effective angle offset (deg)")
    flat[3].legend(frameon=False)

    for label, axis in zip("ABCD", flat, strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("A320 fit-only angle/lift residual attribution")
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
    document, candidates, flight_rows = build_audit(arrays, prior)
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _write_csv(args.candidates, candidates)
    _write_csv(args.flight_metrics, flight_rows)
    args.report.write_text(_render_report(document, candidates))
    _plot(candidates, flight_rows, document["result"]["selected_candidate"], args.figure_prefix)
    return document


def main() -> None:
    document = run(parse_args())
    print(document["status"])


if __name__ == "__main__":
    main()
