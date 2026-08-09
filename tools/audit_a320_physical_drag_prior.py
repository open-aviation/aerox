#!/usr/bin/env python3
"""Build, fit, and audit the compact geometry-derived A320 drag prior."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import load_aircraft
from aerox.models import (
    GeometryDragModel,
    PollSchumannDragModel,
    geometry_drag_model_from_aircraft,
)
from tools.historical_a320_drag import HistoricalCompactTransonicDragModel

UPDATED = "2026-08-08"
REFERENCE_LIFT_COEFFICIENT = 0.45
FIXED_EFFECTIVE_ONSET = (0.75, 0.12)
POLL_SCHUMANN_DOI = "https://doi.org/10.1017/aer.2024.141"
NITA_SCHOLZ_DOI = "https://doi.org/10.48441/4427.2602"
OPENVSP_PARASITE_URL = "https://www.nasa.gov/reference/openvsp-parasite-drag-tool/"
OPENAP_DATA_DOI = "https://doi.org/10.6084/m9.figshare.11424924.v2"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}


@dataclass(frozen=True)
class CandidateMetrics:
    candidate: str
    split: str
    sample_count: int
    rmse_drag_counts: float
    mean_bias_drag_counts: float
    maximum_absolute_error_drag_counts: float


@dataclass(frozen=True)
class CandidateParameters:
    candidate: str
    parameter_count: int
    parasite_scale: float
    induced_scale: float
    zero_lift_critical_mach: float
    lift_critical_mach_slope: float
    wave_drag_amplitude: float
    parameter_at_bound: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-reference",
        type=Path,
        default=Path("docs/modeling/a320-published-transonic-drag-reference.csv"),
    )
    parser.add_argument(
        "--components",
        type=Path,
        default=Path("docs/modeling/a320-physical-drag-components.csv"),
    )
    parser.add_argument(
        "--surface",
        type=Path,
        default=Path("docs/modeling/a320-physical-drag-public-comparison.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-physical-drag-candidate-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-physical-drag-prior.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-physical-drag-prior.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-physical-drag-prior"),
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


def _model(
    prior: GeometryDragModel, parameters: np.ndarray
) -> HistoricalCompactTransonicDragModel:
    parasite_scale, induced_scale, onset, slope, amplitude = parameters
    return HistoricalCompactTransonicDragModel(
        base_prior=replace(
            prior,
            parasite_scale=float(parasite_scale),
            induced_scale=float(induced_scale),
        ),
        zero_lift_critical_mach=float(onset),
        lift_critical_mach_slope=float(slope),
        wave_drag_amplitude=float(amplitude),
    )


def _prediction(
    prior: GeometryDragModel, parameters: np.ndarray, frame: pd.DataFrame
) -> np.ndarray:
    return _model(prior, parameters).drag_coefficient(
        frame["lift_coefficient"].to_numpy(dtype=float),
        frame["mach"].to_numpy(dtype=float),
    )


def _block_weights(frame: pd.DataFrame) -> np.ndarray:
    blocks = pd.cut(
        frame["mach"],
        [0.49, 0.64, 0.72, 0.78, 0.821],
        labels=False,
        include_lowest=True,
    )
    counts = frame.groupby(
        [frame["lift_coefficient"], blocks], observed=True
    ).size()
    weights = np.asarray(
        [
            1.0 / np.sqrt(counts.loc[(lift, block)])
            for lift, block in zip(frame["lift_coefficient"], blocks, strict=True)
        ]
    )
    return weights / np.mean(weights)


def _bounds(prior: GeometryDragModel) -> tuple[np.ndarray, np.ndarray]:
    reference = prior.breakdown(0.0, 0.78)
    cd0 = float(reference.parasite)
    effective_k = prior.span_induced_drag_factor + (
        prior.profile_lift_drag_multiplier * cd0
    )
    lower = np.array([0.016 / cd0, 0.035 / effective_k, 0.70, 0.04, 0.0])
    upper = np.array([0.023 / cd0, 0.048 / effective_k, 0.82, 0.18, 0.006])
    return lower, upper


def _fit_candidates(
    prior: GeometryDragModel, development: pd.DataFrame
) -> tuple[list[CandidateParameters], dict[str, np.ndarray]]:
    observed = development["published_observed_drag_coefficient"].to_numpy(
        dtype=float
    )
    weights = _block_weights(development)
    lower, upper = _bounds(prior)
    initial = np.array([1.0, 1.0, 0.75, 0.12, 0.002])
    fitted: dict[str, np.ndarray] = {"G0_geometry_fixed": initial.copy()}
    fitted["G0_geometry_fixed"][4] = 0.0

    def residual_for(indices: tuple[int, ...], fixed: np.ndarray):
        def residual(values: np.ndarray) -> np.ndarray:
            parameters = fixed.copy()
            parameters[list(indices)] = values
            data = (
                _prediction(prior, parameters, development) - observed
            ) / 5.0e-4
            data *= weights
            prior_residual = []
            if 0 in indices:
                prior_residual.append((parameters[0] - 1.0) / 0.12)
            if 1 in indices:
                prior_residual.append((parameters[1] - 1.0) / 0.15)
            if 2 in indices:
                prior_residual.append((parameters[2] - 0.75) / 0.03)
            if 3 in indices:
                prior_residual.append((parameters[3] - 0.12) / 0.04)
            return np.concatenate([data, np.asarray(prior_residual)])

        return residual

    specifications = {
        "G1_baseline_corrected": ((0, 1), np.array([1.0, 1.0, 0.75, 0.12, 0.0])),
        "G2_fixed_onset": ((0, 1, 4), initial.copy()),
        "G3_free_onset": ((0, 1, 2, 3, 4), initial.copy()),
    }
    for name, (indices, fixed) in specifications.items():
        if name == "G2_fixed_onset":
            fixed[2:4] = FIXED_EFFECTIVE_ONSET
        result = least_squares(
            residual_for(indices, fixed),
            fixed[list(indices)],
            bounds=(lower[list(indices)], upper[list(indices)]),
            loss="huber",
            f_scale=1.0,
        )
        parameters = fixed.copy()
        parameters[list(indices)] = result.x
        fitted[name] = parameters

    parameter_rows = []
    parameter_counts = {
        "G0_geometry_fixed": 0,
        "G1_baseline_corrected": 2,
        "G2_fixed_onset": 3,
        "G3_free_onset": 5,
    }
    for name, values in fitted.items():
        active = {
            "G0_geometry_fixed": (),
            "G1_baseline_corrected": (0, 1),
            "G2_fixed_onset": (0, 1, 4),
            "G3_free_onset": (0, 1, 2, 3, 4),
        }[name]
        at_bound = any(
            np.isclose(values[index], lower[index], atol=1e-7)
            or np.isclose(values[index], upper[index], atol=1e-7)
            for index in active
        )
        parameter_rows.append(
            CandidateParameters(
                candidate=name,
                parameter_count=parameter_counts[name],
                parasite_scale=float(values[0]),
                induced_scale=float(values[1]),
                zero_lift_critical_mach=float(values[2]),
                lift_critical_mach_slope=float(values[3]),
                wave_drag_amplitude=float(values[4]),
                parameter_at_bound=at_bound,
            )
        )
    return parameter_rows, fitted


def _metrics(
    candidate: str, split: str, observed: np.ndarray, predicted: np.ndarray
) -> CandidateMetrics:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if not len(observed):
        raise ValueError("no finite public-comparison points")
    residual = (predicted - observed) * 1.0e4
    return CandidateMetrics(
        candidate=candidate,
        split=split,
        sample_count=len(residual),
        rmse_drag_counts=float(np.sqrt(np.mean(residual**2))),
        mean_bias_drag_counts=float(np.mean(residual)),
        maximum_absolute_error_drag_counts=float(np.max(np.abs(residual))),
    )


def _component_table(prior: GeometryDragModel) -> pd.DataFrame:
    condition_mach = 0.78
    unit_reynolds = prior.reference_unit_reynolds_number_per_m
    rows = []
    uncertainty = {
        "wing": 0.15,
        "fuselage": 0.20,
        "horizontal_tail": 0.20,
        "vertical_tail": 0.20,
    }
    for component in prior.components:
        skin_friction = float(
            component.skin_friction_coefficient(unit_reynolds, condition_mach)
        )
        coefficient = float(
            component.drag_coefficient(
                unit_reynolds,
                condition_mach,
                prior.reference_area_m2,
            )
        )
        rows.append(
            {
                "component": component.name,
                "wetted_area_m2": component.wetted_area_m2,
                "characteristic_length_m": component.characteristic_length_m,
                "form_factor": component.form_factor,
                "interference_factor": component.interference_factor,
                "skin_friction_coefficient": skin_friction,
                "zero_lift_drag_coefficient": coefficient,
                "relative_model_uncertainty": uncertainty[component.name],
                "basis": "analytic gross wetted-area estimate",
            }
        )
    rows.append(
        {
            "component": "unresolved",
            "wetted_area_m2": np.nan,
            "characteristic_length_m": np.nan,
            "form_factor": np.nan,
            "interference_factor": np.nan,
            "skin_friction_coefficient": np.nan,
            "zero_lift_drag_coefficient": prior.unresolved_zero_lift_drag_coefficient,
            "relative_model_uncertainty": 0.50,
            "basis": "nacelles, pylons, cooling, leakage, and excrescences",
        }
    )
    return pd.DataFrame(rows)


def _public_comparison(
    reference: pd.DataFrame,
    prior: GeometryDragModel,
    fitted: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[CandidateMetrics]]:
    result = reference[
        ["lift_coefficient", "mach", "published_observed_drag_coefficient"]
    ].copy()
    development = result["lift_coefficient"].isin([0.3, 0.4])
    test = result["lift_coefficient"] == 0.5
    result["split"] = np.where(development, "development", "public_holdout")
    rows: list[CandidateMetrics] = []
    for name, parameters in fitted.items():
        column = f"{name}_drag_coefficient"
        result[column] = _prediction(prior, parameters, result)
        for split, selector in (("development", development), ("public_holdout", test)):
            rows.append(
                _metrics(
                    name,
                    split,
                    result.loc[selector, "published_observed_drag_coefficient"].to_numpy(),
                    result.loc[selector, column].to_numpy(),
                )
            )
    poll = _poll_schumann_a320()
    poll_prediction = np.full(len(result), np.nan)
    poll_support = result["mach"].to_numpy() <= poll.maximum_mach
    poll_prediction[poll_support] = poll.drag_coefficient(
        result.loc[poll_support, "lift_coefficient"].to_numpy(),
        result.loc[poll_support, "mach"].to_numpy(),
    )
    result["poll_schumann_drag_coefficient"] = poll_prediction
    for split, selector in (("development", development), ("public_holdout", test)):
        rows.append(
            _metrics(
                "poll_schumann",
                split,
                result.loc[selector, "published_observed_drag_coefficient"].to_numpy(),
                result.loc[selector, "poll_schumann_drag_coefficient"].to_numpy(),
            )
        )
    return result, rows


def _reference_coefficients(
    prior: GeometryDragModel, selected: HistoricalCompactTransonicDragModel
) -> dict[str, dict[str, float]]:
    base = prior.breakdown(0.0, 0.78)
    selected_base = selected.base_prior.breakdown(0.0, 0.78)

    def effective_k(item: GeometryDragModel, cd0: float) -> float:
        return item.span_induced_drag_factor + (
            item.induced_scale * item.profile_lift_drag_multiplier * cd0
        )

    poll = _poll_schumann_a320()
    return {
        "aerox_legacy": {"cd0": 0.02263, "k_effective": 0.04246},
        "openap_surveillance_median": {"cd0": 0.01753, "k_effective": 0.038545},
        "poll_schumann": {
            "cd0": float(poll.zero_lift_drag_coefficient()),
            "k_effective": float(poll.induced_drag_factor()),
        },
        "geometry_prior": {
            "cd0": float(base.parasite),
            "k_effective": effective_k(prior, float(base.parasite)),
        },
        "selected_compact": {
            "cd0": float(selected_base.parasite),
            "k_effective": effective_k(
                selected.base_prior,
                float(selected_base.parasite),
            ),
        },
    }


def _plot(
    *,
    components: pd.DataFrame,
    comparison: pd.DataFrame,
    coefficients: dict[str, dict[str, float]],
    prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.dpi": 120,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    component_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        "#999999",
    ]
    values = components["zero_lift_drag_coefficient"].to_numpy() * 1.0e4
    errors = values * components["relative_model_uncertainty"].to_numpy()
    axes[0, 0].barh(
        components["component"],
        values,
        xerr=errors,
        color=component_colors,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Zero-lift drag contribution (counts)")
    axes[0, 0].set_title("(a) Geometry component build-up")

    labels = list(coefficients)
    x = np.arange(len(labels))
    cd0 = np.array([coefficients[label]["cd0"] for label in labels]) * 1.0e4
    k = np.array([coefficients[label]["k_effective"] for label in labels]) * 1.0e4
    width = 0.37
    axes[0, 1].bar(
        x - width / 2,
        cd0,
        width,
        label=r"$C_{D0}$",
        color=OKABE_ITO["sky"],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0, 1].bar(
        x + width / 2,
        k,
        width,
        label=r"effective $k$",
        color=OKABE_ITO["vermillion"],
        hatch="//",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0, 1].set_xticks(x, [label.replace("_", "\n") for label in labels])
    axes[0, 1].tick_params(axis="x", labelsize=7)
    axes[0, 1].set_ylabel("Coefficient (counts)")
    axes[0, 1].set_title("(b) Reference subcritical coefficients")
    axes[0, 1].legend(frameon=False, ncols=2)

    lift_colors = {0.3: OKABE_ITO["blue"], 0.4: OKABE_ITO["green"], 0.5: OKABE_ITO["orange"]}
    for lift, group in comparison.groupby("lift_coefficient"):
        color = lift_colors[float(lift)]
        axes[1, 0].scatter(
            group["mach"],
            group["published_observed_drag_coefficient"],
            s=8,
            facecolors="none",
            edgecolors=color,
            linewidths=0.6,
            label=fr"observed $C_L={lift:.1f}$",
        )
        axes[1, 0].plot(
            group["mach"],
            group["G2_fixed_onset_drag_coefficient"],
            color=color,
            linewidth=1.5,
        )
        axes[1, 0].plot(
            group["mach"],
            group["poll_schumann_drag_coefficient"],
            color=color,
            linestyle="--",
            linewidth=0.9,
        )
    axes[1, 0].set_xlabel("Mach number")
    axes[1, 0].set_ylabel(r"Total $C_D$")
    axes[1, 0].set_title("(c) Public polar: compact solid, Poll dashed")
    axes[1, 0].legend(frameon=False, ncols=1, loc="upper left")

    holdout = comparison[comparison["lift_coefficient"] == 0.5]
    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1, 1].plot(
        holdout["mach"],
        1.0e4
        * (
            holdout["G2_fixed_onset_drag_coefficient"]
            - holdout["published_observed_drag_coefficient"]
        ),
        color=OKABE_ITO["blue"],
        linewidth=1.5,
        label="Aerox compact",
    )
    axes[1, 1].plot(
        holdout["mach"],
        1.0e4
        * (
            holdout["poll_schumann_drag_coefficient"]
            - holdout["published_observed_drag_coefficient"]
        ),
        color=OKABE_ITO["vermillion"],
        linestyle="--",
        linewidth=1.2,
        label="Poll–Schumann",
    )
    axes[1, 1].set_xlabel("Mach number")
    axes[1, 1].set_ylabel(r"Residual $C_D$ (counts)")
    axes[1, 1].set_title(r"(d) Public holdout residual, $C_L=0.5$")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=320)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    prior: GeometryDragModel,
    selected: CandidateParameters,
    metrics: list[CandidateMetrics],
    coefficients: dict[str, dict[str, float]],
) -> None:
    selected_dev = next(
        row for row in metrics if row.candidate == selected.candidate and row.split == "development"
    )
    selected_test = next(
        row for row in metrics if row.candidate == selected.candidate and row.split == "public_holdout"
    )
    poll_test = next(
        row for row in metrics if row.candidate == "poll_schumann" and row.split == "public_holdout"
    )
    base = coefficients["geometry_prior"]
    calibrated = coefficients["selected_compact"]
    text = f"""# A320 geometry-derived compact drag prior

## Outcome

Aerox now has a Reynolds-aware component drag build-up and a compact transonic
extension. The A320 reference build-up gives `CD0={base['cd0']:.5f}` and an
effective `k={base['k_effective']:.5f}` before public-curve correction. These
are close to the independent public ranges, but they are no longer opaque
constants: the runtime model retains separate wing, fuselage, horizontal-tail,
vertical-tail, unresolved, span-induced, lift-profile, and wave contributions.

The selected development candidate is `G2_fixed_onset`. It fits three bounded
corrections while retaining a fixed effective onset shape. Its corrected
reference coefficients are `CD0={calibrated['cd0']:.5f}` and
`k={calibrated['k_effective']:.5f}`. On the `CL=0.5` public holdout it reaches
{selected_test.rmse_drag_counts:.2f} drag-count RMSE, versus
{poll_test.rmse_drag_counts:.2f} for Poll–Schumann. This is evidence of better
agreement with one digitized public polar, not proof of manufacturer-level
accuracy.

![A320 physical drag prior](figures/a320-physical-drag-prior.png)

Figure 1. Component build-up, subcritical coefficient cross-checks, published
drag curves, and the public-holdout residual. Error bars in panel (a) are
engineering model-form ranges, not statistical confidence intervals. A
[vector PDF](figures/a320-physical-drag-prior.pdf) is also provided.

## Physical formulation

For each component `i`, the clean zero-lift build-up is

`CD0_i = Cf_i(Re_i, M) * FF_i * Q_i * Swet_i / Sref`,

with `Re_i = (rho V / mu) Li`. The runtime condition therefore carries unit
Reynolds number rather than one arbitrary aircraft-scale value. The subcritical
lift drag is split into `CL²/(pi AR e_span)` and a sweep-conditioned
lift-profile term. The transonic correction is

`CD_wave = A * max((M - (m0 - m1 CL)) / 0.1, 0)^4`.

The five possible correction quantities are `s0`, `sk`, `m0`, `m1`, and `A`.
The selected simpler candidate fixes `m0={selected.zero_lift_critical_mach:.3f}`
and `m1={selected.lift_critical_mach_slope:.3f}`, and fits
`s0={selected.parasite_scale:.4f}`, `sk={selected.induced_scale:.4f}`, and
`A={selected.wave_drag_amplitude:.6f}`. The word *critical* in the code denotes
an effective drag-rise onset; it must not be interpreted as a measured local
airfoil critical Mach.

## Evidence separation

- Geometry defines the component prior.
- The `CL=0.3` and `0.4` digitized curves are development evidence.
- The complete `CL=0.5` curve is a public development holdout.
- QAR is not used to fit drag. It will only be used after the drag law is
  frozen, with a separately refitted thrust surrogate for every drag law.
- OpenAP's surveillance `CD0/k` estimates are used only as a joint cross-check;
  their near-perfect correlation means they do not identify the two terms
  independently.

The selected development RMSE is {selected_dev.rmse_drag_counts:.2f} counts.
Its induced scale reaches the conservative lower `k` bound, so this is not yet
a release calibration. It is the candidate to carry into the thrust/fuel and
trajectory gates.

## OpenVSP cross-check and limitations

The existing `.vsp3` contains only wing and tail surfaces. Its wetted areas
are gross rather than exposed because there is no fuselage intersection. They
agree closely with the analytic surface areas and are retained as a geometry
check, not substituted silently into the runtime prior.

The isolated untwisted wing gives `k_induced` about `0.0431–0.0434`. Adding a
separate lift-profile term to that result would produce a much higher total
than the public range; this exposes model-form sensitivity to omitted twist,
camber, wing-tip fence, and interference. The runtime prior instead constrains
the sum of span and profile contributions and records the split explicitly.

The largest remaining `CD0` gap is engine installation. Current Aerox aircraft
records have engine identity and count but not nacelle length/diameter, pylon
area, cooling drag, leakage, or excrescence data. These remain an explicit
`unresolved` contribution with a wide uncertainty. Non-clean flap, gear, and
spoiler states are rejected by the new model until configuration increments
are independently implemented.

The older longitudinal and translational reconstruction stacks remain
parabolic-only. The pitch-bank-thrust point-mass model, FMS profile feasibility model, and
ECON optimizer now pass the condition-aware interface.

## Next gate

Freeze `G2_fixed_onset`, refit the N1-to-thrust surrogate on exactly the same
clean samples used for the Poll comparison, and compare held-flight thrust,
QAR/OpenAP fuel flow, and full trajectory metrics. The old thrust map must not
be reused because a drag change is algebraically absorbed into inferred
thrust.

## Sources

- NASA OpenVSP Parasite Drag Tool: `{OPENVSP_PARASITE_URL}`.
- Poll and Schumann transport-aircraft drag model: `{POLL_SCHUMANN_DOI}`.
- Nita and Scholz Oswald-factor geometry method: `{NITA_SCHOLZ_DOI}`.
- OpenAP surveillance drag-polar supplement: `{OPENAP_DATA_DOI}`.
- Digitized Obert/Krull provenance: `a320-published-transonic-drag-audit.md`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    aircraft = load_aircraft("A320")
    prior = geometry_drag_model_from_aircraft(aircraft)
    reference = pd.read_csv(args.published_reference)
    development = reference[reference["lift_coefficient"].isin([0.3, 0.4])]
    parameter_rows, fitted = _fit_candidates(prior, development)
    selected = next(row for row in parameter_rows if row.candidate == "G2_fixed_onset")
    selected_model = _model(prior, fitted[selected.candidate])
    components = _component_table(prior)
    comparison, metric_rows = _public_comparison(reference, prior, fitted)
    coefficients = _reference_coefficients(prior, selected_model)

    args.components.parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(args.components, index=False)
    comparison.to_csv(args.surface, index=False)
    pd.DataFrame(asdict(row) for row in metric_rows).to_csv(args.metrics, index=False)
    document: dict[str, Any] = {
        "updated": UPDATED,
        "status": "candidate_pending_qar_thrust_fuel_trajectory_gate",
        "aircraft": "Airbus A320-214",
        "model": {
            "equation": (
                "s0*sum(Cf_i(Re_i,M)*FF_i*Q_i*Swet_i/Sref)+"
                "sk*(1/(pi*AR*e_span)+profile_multiplier*s0*CD0)*CL^2+"
                "A*max((M-(m0-m1*CL))/0.1,0)^4"
            ),
            "reference_unit_reynolds_number_per_m": (
                prior.reference_unit_reynolds_number_per_m
            ),
            "span_efficiency_factor": prior.span_efficiency_factor,
            "profile_lift_drag_multiplier": prior.profile_lift_drag_multiplier,
            "unresolved_zero_lift_drag_coefficient": (
                prior.unresolved_zero_lift_drag_coefficient
            ),
        },
        "candidate_parameters": [asdict(row) for row in parameter_rows],
        "selected_candidate": asdict(selected),
        "reference_coefficients": coefficients,
        "metrics": [asdict(row) for row in metric_rows],
        "fit_protocol": {
            "development_lift_coefficients": [0.3, 0.4],
            "public_holdout_lift_coefficient": 0.5,
            "mach_blocks": [0.49, 0.64, 0.72, 0.78, 0.821],
            "robust_loss": "Huber",
            "nominal_residual_scale_drag_counts": 5.0,
            "qar_used_for_drag_fit": False,
        },
        "openvsp_cross_check": {
            "surface_wetted_area_m2_medium": 354.769131,
            "surface_wetted_area_m2_fine": 355.0803,
            "areas_are_gross_not_exposed": True,
            "isolated_wing_induced_k_medium": 0.043083,
            "isolated_wing_induced_k_fine": 0.043382,
            "release_use": "wide model-form cross-check only",
        },
        "sources": {
            "openvsp_parasite_drag": OPENVSP_PARASITE_URL,
            "poll_schumann": POLL_SCHUMANN_DOI,
            "nita_scholz": NITA_SCHOLZ_DOI,
            "openap_supplement": OPENAP_DATA_DOI,
        },
        "artifacts": {
            "components": str(args.components),
            "public_comparison": str(args.surface),
            "metrics": str(args.metrics),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
            "report": str(args.report),
        },
    }
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    _plot(
        components=components,
        comparison=comparison,
        coefficients=coefficients,
        prefix=args.figure_prefix,
    )
    _write_report(
        args.report,
        prior=prior,
        selected=selected,
        metrics=metric_rows,
        coefficients=coefficients,
    )

    selected_test = next(
        row
        for row in metric_rows
        if row.candidate == selected.candidate and row.split == "public_holdout"
    )
    poll_test = next(
        row
        for row in metric_rows
        if row.candidate == "poll_schumann" and row.split == "public_holdout"
    )
    print(
        f"selected={selected.candidate} holdout_rmse={selected_test.rmse_drag_counts:.3f} "
        f"poll_rmse={poll_test.rmse_drag_counts:.3f}"
    )


if __name__ == "__main__":
    main()
