#!/usr/bin/env python3
"""Refit A320 effective thrust under the frozen compact physical drag model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aerox import aero
from aerox.models import DragCondition
from aerox.modeling.identification import robust_weighted_linear_fit
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    EffectiveThrustMap,
    thrust_map_features,
)
from tools.audit_a320_drag_conditioned_thrust_refit import _poll_schumann_a320
from tools.audit_a320_drag_conditioned_thrust_refit import _support as poll_support
from tools.calibrate_a320_effective_thrust import (
    FIT_THINNING_S,
    PRESSURE_EXPONENTS,
    ModelSettings,
    PropulsionFlight,
    _load_flights,
    _phase_equal_weights,
    _read_tas_scale,
    _read_vertical_calibration,
)
from tools.historical_a320_drag import historical_a320_compact_drag_candidate

UPDATED = "2026-08-08"
VARIANT = "aerox_compact_physical"


@dataclass(frozen=True)
class FitResult:
    model: EffectiveThrustMap
    weighted_rmse_kn: float
    robust_scale_kn: float
    normal_matrix_condition: float
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv")
    )
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
        "--thrust-parameters",
        type=Path,
        default=Path("docs/modeling/a320-effective-thrust-calibration.yml"),
    )
    parser.add_argument(
        "--poll-benchmark",
        type=Path,
        default=Path("docs/modeling/a320-drag-conditioned-thrust-refit.yml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("docs/modeling/a320-compact-drag-thrust-candidates.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-compact-drag-thrust-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-compact-drag-thrust-refit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-compact-drag-thrust-refit.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-compact-drag-thrust-refit"),
    )
    return parser.parse_args()


def _unit_reynolds(flight: PropulsionFlight) -> np.ndarray:
    pressure_pa = flight.pressure_ratio * aero.p0
    density = pressure_pa / (aero.R * flight.static_temperature_k)
    viscosity = (
        1.458e-6
        * flight.static_temperature_k**1.5
        / (flight.static_temperature_k + 110.4)
    )
    return density * flight.true_airspeed_mps / viscosity


def _compact_support(flight: PropulsionFlight) -> np.ndarray:
    model = historical_a320_compact_drag_candidate()
    return (
        np.isfinite(flight.lift_coefficient)
        & np.isfinite(flight.mach)
        & (flight.lift_coefficient >= model.minimum_lift_coefficient)
        & (flight.lift_coefficient <= model.maximum_lift_coefficient)
        & (flight.mach >= model.minimum_mach)
        & (flight.mach <= model.maximum_mach)
    )


def _compact_drag_n(flight: PropulsionFlight) -> np.ndarray:
    model = historical_a320_compact_drag_candidate()
    supported = _compact_support(flight)
    coefficient = np.full_like(flight.lift_coefficient, np.nan)
    coefficient[supported] = model.drag_coefficient_at(
        flight.lift_coefficient[supported],
        DragCondition(
            mach=flight.mach[supported],
            unit_reynolds_number_per_m=_unit_reynolds(flight)[supported],
        ),
    )
    return flight.parasite_drag_basis_n * coefficient


def _target_thrust_n(flight: PropulsionFlight) -> np.ndarray:
    return flight.net_along_path_force_n + _compact_drag_n(flight)


def _fit_map(
    flights: list[PropulsionFlight],
    *,
    settings: ModelSettings,
    rated_total_thrust_n: float,
) -> FitResult:
    arrays: dict[str, list[np.ndarray]] = {
        "n1": [],
        "mach": [],
        "delta": [],
        "temperature": [],
        "target": [],
        "weights": [],
    }
    sample_count = 0
    for flight in flights:
        selected = _compact_support(flight) & (
            flight.sample_index % FIT_THINNING_S == 0
        )
        if np.count_nonzero(selected) < 20:
            raise ValueError(f"too few supported fit points in {flight.key}")
        arrays["n1"].append(flight.n1_percent[selected])
        arrays["mach"].append(flight.mach[selected])
        arrays["delta"].append(flight.pressure_ratio[selected])
        arrays["temperature"].append(flight.static_temperature_k[selected])
        arrays["target"].append(_target_thrust_n(flight)[selected])
        arrays["weights"].append(_phase_equal_weights(flight.phase[selected]))
        sample_count += int(np.count_nonzero(selected))
    data = {name: np.concatenate(values) for name, values in arrays.items()}
    data["weights"] /= len(flights)
    design = thrust_map_features(
        n1_percent=data["n1"],
        mach=data["mach"],
        pressure_ratio=data["delta"],
        static_temperature_k=data["temperature"],
        pressure_exponent=settings.pressure_exponent,
        feature_set=settings.feature_set,
        corrected_fan_speed=settings.corrected_fan_speed,
    )
    fit = robust_weighted_linear_fit(
        design,
        data["target"] / rated_total_thrust_n,
        data["weights"],
    )
    model = EffectiveThrustMap(
        rated_total_thrust_n=rated_total_thrust_n,
        pressure_exponent=settings.pressure_exponent,
        feature_set=settings.feature_set,
        corrected_fan_speed=settings.corrected_fan_speed,
        coefficients=fit.coefficients,
    )
    return FitResult(
        model=model,
        weighted_rmse_kn=float(
            fit.base_weighted_rmse * rated_total_thrust_n / 1_000.0
        ),
        robust_scale_kn=float(fit.robust_scale * rated_total_thrust_n / 1_000.0),
        normal_matrix_condition=float(fit.normal_matrix_condition),
        sample_count=sample_count,
    )


def _predict(model: EffectiveThrustMap, flight: PropulsionFlight) -> np.ndarray:
    return model.thrust_n(
        n1_percent=flight.n1_percent,
        mach=flight.mach,
        pressure_ratio=flight.pressure_ratio,
        static_temperature_k=flight.static_temperature_k,
    )


def _raw_predict(model: EffectiveThrustMap, flight: PropulsionFlight) -> np.ndarray:
    return model.raw_thrust_n(
        n1_percent=flight.n1_percent,
        mach=flight.mach,
        pressure_ratio=flight.pressure_ratio,
        static_temperature_k=flight.static_temperature_k,
    )


def _candidate_row(
    flights: list[PropulsionFlight],
    *,
    settings: ModelSettings,
    rated_total_thrust_n: float,
) -> dict[str, Any]:
    rmses: list[float] = []
    biases: list[float] = []
    monotonic: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    conditions: list[float] = []
    for held_out in flights:
        fit = _fit_map(
            [flight for flight in flights if flight.key != held_out.key],
            settings=settings,
            rated_total_thrust_n=rated_total_thrust_n,
        )
        selected = _compact_support(held_out)
        prediction = _predict(fit.model, held_out)[selected]
        target = _target_thrust_n(held_out)[selected]
        residual = prediction - target
        rmses.append(float(np.sqrt(np.mean(residual**2))))
        biases.append(float(np.mean(residual)))
        conditions.append(fit.normal_matrix_condition)
        raw = _raw_predict(fit.model, held_out)[selected]
        negative.append(raw < 0.0)
        plus = fit.model.raw_thrust_n(
            n1_percent=held_out.n1_percent[selected] + 0.1,
            mach=held_out.mach[selected],
            pressure_ratio=held_out.pressure_ratio[selected],
            static_temperature_k=held_out.static_temperature_k[selected],
        )
        minus = fit.model.raw_thrust_n(
            n1_percent=np.maximum(0.0, held_out.n1_percent[selected] - 0.1),
            mach=held_out.mach[selected],
            pressure_ratio=held_out.pressure_ratio[selected],
            static_temperature_k=held_out.static_temperature_k[selected],
        )
        monotonic.append(plus >= minus)
    return {
        "variant": VARIANT,
        "pressure_exponent": settings.pressure_exponent,
        "feature_set": settings.feature_set,
        "feature_count": len(THRUST_MAP_FEATURE_SETS[settings.feature_set]),
        "corrected_fan_speed": settings.corrected_fan_speed,
        "lofo_mean_flight_rmse_kn": float(np.mean(rmses) / 1_000.0),
        "lofo_median_flight_rmse_kn": float(np.median(rmses) / 1_000.0),
        "lofo_worst_flight_rmse_kn": float(np.max(rmses) / 1_000.0),
        "lofo_mean_flight_bias_kn": float(np.mean(biases) / 1_000.0),
        "observed_local_monotonic_fraction": float(
            np.mean(np.concatenate(monotonic))
        ),
        "raw_negative_prediction_fraction": float(np.mean(np.concatenate(negative))),
        "median_fold_normal_condition": float(np.median(conditions)),
    }


def _select_candidate(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates[
        candidates["observed_local_monotonic_fraction"] >= 0.995
    ]
    if eligible.empty:
        best = candidates["observed_local_monotonic_fraction"].max()
        eligible = candidates[candidates["observed_local_monotonic_fraction"] == best]
    score_limit = 1.01 * eligible["lofo_mean_flight_rmse_kn"].min()
    return eligible[eligible["lofo_mean_flight_rmse_kn"] <= score_limit].sort_values(
        ["feature_count", "lofo_mean_flight_rmse_kn", "pressure_exponent"],
        kind="stable",
    ).iloc[0]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _metric_row(
    flight: PropulsionFlight,
    *,
    split: str,
    support_name: str,
    selected: np.ndarray,
    model: EffectiveThrustMap,
    fuel_prior: OpenAPFuelFlowPrior,
) -> dict[str, Any]:
    prediction = _predict(model, flight)[selected]
    target = _target_thrust_n(flight)[selected]
    measured_fuel = flight.fuel_flow_kg_s[selected]
    predicted_fuel = fuel_prior.fuel_flow_kg_s(prediction)
    thrust_residual = prediction - target
    fuel_residual = predicted_fuel - measured_fuel
    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "split": split,
        "support": support_name,
        "total_clean_samples": len(flight.mach),
        "supported_samples": int(np.count_nonzero(selected)),
        "support_fraction": float(np.mean(selected)),
        "thrust_rmse_kn": float(np.sqrt(np.mean(thrust_residual**2)) / 1_000.0),
        "thrust_bias_kn": float(np.mean(thrust_residual) / 1_000.0),
        "thrust_correlation": _safe_correlation(prediction, target),
        "target_thrust_median_kn": float(np.median(target) / 1_000.0),
        "predicted_thrust_median_kn": float(np.median(prediction) / 1_000.0),
        "openap_fuel_rmse_kg_s": float(np.sqrt(np.mean(fuel_residual**2))),
        "openap_fuel_bias_kg_s": float(np.mean(fuel_residual)),
        "openap_fuel_correlation": _safe_correlation(predicted_fuel, measured_fuel),
    }


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, support), frame in metrics.groupby(["split", "support"], sort=True):
        rows.append(
            {
                "split": str(split),
                "support": str(support),
                "flight_count": len(frame),
                "mean_flight_thrust_rmse_kn": float(frame["thrust_rmse_kn"].mean()),
                "worst_flight_thrust_rmse_kn": float(frame["thrust_rmse_kn"].max()),
                "mean_flight_thrust_bias_kn": float(frame["thrust_bias_kn"].mean()),
                "mean_flight_openap_fuel_rmse_kg_s": float(
                    frame["openap_fuel_rmse_kg_s"].mean()
                ),
                "mean_flight_openap_fuel_bias_kg_s": float(
                    frame["openap_fuel_bias_kg_s"].mean()
                ),
                "mean_support_fraction": float(frame["support_fraction"].mean()),
            }
        )
    return rows


def _coverage(flights: list[PropulsionFlight]) -> dict[str, Any]:
    phase = np.concatenate([flight.phase for flight in flights])
    compact = np.concatenate([_compact_support(flight) for flight in flights])
    poll = np.concatenate(
        [poll_support(flight, _poll_schumann_a320()) for flight in flights]
    )
    result = {}
    for name in ("all", "climb", "level", "descent"):
        selected = np.ones_like(compact, dtype=bool) if name == "all" else phase == name
        result[name] = {
            "sample_count": int(np.count_nonzero(selected)),
            "compact_support_fraction": float(np.mean(compact[selected])),
            "poll_support_fraction": float(np.mean(poll[selected])),
        }
    return result


def _lookup(
    aggregate: list[dict[str, Any]], split: str, support: str
) -> dict[str, Any]:
    return next(
        row
        for row in aggregate
        if row["split"] == split and row["support"] == support
    )


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _plot(
    metrics: pd.DataFrame,
    coverage: dict[str, Any],
    poll_benchmark: dict[str, Any],
    flights: list[PropulsionFlight],
    prefix: Path,
) -> None:
    validation = metrics[
        (metrics["split"] == "validation") & (metrics["support"] == "poll_common")
    ]
    prior_metrics = pd.DataFrame(poll_benchmark["aggregate_metrics"])
    current = prior_metrics[
        (prior_metrics["split"] == "validation")
        & (prior_metrics["variant"] == "aerox_current")
    ].iloc[0]
    poll = prior_metrics[
        (prior_metrics["split"] == "validation")
        & (prior_metrics["variant"] == "poll_schumann")
    ].iloc[0]
    compact = {
        "thrust": validation["thrust_rmse_kn"].mean(),
        "fuel": validation["openap_fuel_rmse_kg_s"].mean(),
    }
    labels = ["Current\nrefit", "Poll–Schumann\nrefit", "Aerox compact\nrefit"]
    colors = ["#56B4E9", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    axes[0, 0].bar(
        labels,
        [
            current["mean_flight_thrust_rmse_kn"],
            poll["mean_flight_thrust_rmse_kn"],
            compact["thrust"],
        ],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0, 0].set_ylabel("Mean validation thrust RMSE (kN)")
    axes[0, 0].set_title("(a) Common-support thrust refit", loc="left", fontsize=10)
    axes[0, 1].bar(
        labels,
        [
            current["mean_flight_openap_fuel_rmse_kg_s"],
            poll["mean_flight_openap_fuel_rmse_kg_s"],
            compact["fuel"],
        ],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0, 1].set_ylabel("Mean validation fuel-flow RMSE (kg/s)")
    axes[0, 1].set_title(
        "(b) Independent OpenAP fuel diagnostic", loc="left", fontsize=10
    )

    phases = ["all", "climb", "level", "descent"]
    x = np.arange(len(phases))
    axes[1, 0].bar(
        x - 0.2,
        [100.0 * coverage[phase]["poll_support_fraction"] for phase in phases],
        0.4,
        color="#D55E00",
        hatch="//",
        edgecolor="black",
        linewidth=0.5,
        label="Poll–Schumann",
    )
    axes[1, 0].bar(
        x + 0.2,
        [100.0 * coverage[phase]["compact_support_fraction"] for phase in phases],
        0.4,
        color="#009E73",
        edgecolor="black",
        linewidth=0.5,
        label="Aerox compact",
    )
    axes[1, 0].set_xticks(x, phases)
    axes[1, 0].set_ylim(0.0, 105.0)
    axes[1, 0].set_ylabel("Clean QAR coverage (%)")
    axes[1, 0].set_title("(c) Drag-model support", loc="left", fontsize=10)
    axes[1, 0].legend(frameon=False)

    lift = np.concatenate([flight.lift_coefficient for flight in flights])
    mach = np.concatenate([flight.mach for flight in flights])
    unit_reynolds = np.concatenate([_unit_reynolds(flight) for flight in flights])
    model = historical_a320_compact_drag_candidate()
    compact_cd = model.drag_coefficient_at(
        lift,
        DragCondition(mach=mach, unit_reynolds_number_per_m=unit_reynolds),
    )
    current_cd = 0.02263 + 0.04246 * lift**2
    sample = np.linspace(0, len(lift) - 1, min(10_000, len(lift))).astype(int)
    scatter = axes[1, 1].scatter(
        mach[sample],
        1.0e4 * (compact_cd[sample] - current_cd[sample]),
        c=lift[sample],
        cmap="viridis",
        s=4,
        alpha=0.35,
        rasterized=True,
    )
    fig.colorbar(scatter, ax=axes[1, 1], label=r"Lift coefficient $C_L$")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set_xlabel("Mach number")
    axes[1, 1].set_ylabel("Compact minus current drag (counts)")
    axes[1, 1].set_title("(d) QAR-envelope drag change", loc="left", fontsize=10)
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=320)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(path: Path, document: dict[str, Any]) -> None:
    validation = _lookup(document["aggregate_metrics"], "validation", "compact_full")
    common = _lookup(document["aggregate_metrics"], "validation", "poll_common")
    gate = document["gate"]
    coverage = document["coverage"]
    text = f"""# A320 compact-drag conditioned thrust refit

## Outcome

The frozen compact physical drag candidate was given its own complete-flight
LOFO-selected N1-to-effective-thrust surrogate. It covers
{100.0 * coverage['all']['compact_support_fraction']:.1f}% of the clean QAR
samples and {100.0 * coverage['descent']['compact_support_fraction']:.1f}% of
clean descent, removing Poll–Schumann's low-CL descent support gap.

On the six frozen validation flights, the full-support compact refit gives
`{validation['mean_flight_thrust_rmse_kn']:.3f} kN` mean thrust RMSE and
`{validation['mean_flight_openap_fuel_rmse_kg_s']:.5f} kg/s` mean OpenAP-versus-
QAR fuel-flow RMSE. On Poll's common support the corresponding values are
`{common['mean_flight_thrust_rmse_kn']:.3f} kN` and
`{common['mean_flight_openap_fuel_rmse_kg_s']:.5f} kg/s`.

The predeclared gate is **{'passed' if gate['passed'] else 'held'}**. The drag
candidate itself remains frozen; fuel flow was not used in drag or thrust
fitting and every drag law has its own thrust refit.

![A320 compact drag thrust refit](figures/a320-compact-drag-thrust-refit.png)

Figure 1. Common-support benchmark, independent fuel diagnostic, phase support,
and QAR-envelope drag change. A
[vector PDF](figures/a320-compact-drag-thrust-refit.pdf) is also provided.

## Leakage controls

- Drag was fitted only to geometry and public aerodynamic curves.
- Propulsion structure was selected only on ten fit flights using complete-
  flight LOFO.
- The six validation flights were loaded only after selection and full refit.
- QAR fuel flow was never an optimization target.
- Metrics are reported both over the compact model's full support and over the
  exact Poll common support.

## Decision

{gate['reason']}

The next gate is a full generated-trajectory comparison using this exact drag
model and thrust refit, with controller gains held fixed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel[panel["role"] != "validation"].copy()
    validation_selection = panel[panel["role"] == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected the frozen ten-fit/six-validation panel")
    calibration = yaml.safe_load(args.thrust_parameters.read_text())
    rated_total_thrust = float(calibration["model"]["rated_total_thrust_n"])
    fuel_config = calibration["fuel_flow_prior"]
    fuel_prior = OpenAPFuelFlowPrior(
        engine_count=int(calibration["engine_prior"]["engine_count"]),
        rated_thrust_n_per_engine=float(
            calibration["engine_prior"]["rated_thrust_n_per_engine"]
        ),
        c1=float(fuel_config["c1"]),
        c2=float(fuel_config["c2"]),
        c3=float(fuel_config["c3"]),
        engine_scale=float(fuel_config["engine_scale"]),
    )
    vertical = _read_vertical_calibration(args.vertical_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    fit_flights = _load_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=122.0,
    )
    candidate_rows = []
    for pressure_exponent in PRESSURE_EXPONENTS:
        for corrected in (False, True):
            for feature_set in THRUST_MAP_FEATURE_SETS:
                candidate_rows.append(
                    _candidate_row(
                        fit_flights,
                        settings=ModelSettings(
                            pressure_exponent=pressure_exponent,
                            feature_set=feature_set,
                            corrected_fan_speed=corrected,
                        ),
                        rated_total_thrust_n=rated_total_thrust,
                    )
                )
    candidates = pd.DataFrame(candidate_rows)
    selected_row = _select_candidate(candidates)
    settings = ModelSettings(
        pressure_exponent=float(selected_row["pressure_exponent"]),
        feature_set=str(selected_row["feature_set"]),
        corrected_fan_speed=bool(selected_row["corrected_fan_speed"]),
    )
    full_fit = _fit_map(
        fit_flights,
        settings=settings,
        rated_total_thrust_n=rated_total_thrust,
    )
    lofo_models = {}
    for held_out in fit_flights:
        lofo_models[held_out.key] = _fit_map(
            [flight for flight in fit_flights if flight.key != held_out.key],
            settings=settings,
            rated_total_thrust_n=rated_total_thrust,
        ).model

    validation_flights = _load_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=122.0,
    )
    poll = _poll_schumann_a320()
    metric_rows = []
    for flight in fit_flights:
        for support_name, support in (
            ("compact_full", _compact_support(flight)),
            ("poll_common", poll_support(flight, poll)),
        ):
            metric_rows.append(
                _metric_row(
                    flight,
                    split="fit_lofo",
                    support_name=support_name,
                    selected=support,
                    model=lofo_models[flight.key],
                    fuel_prior=fuel_prior,
                )
            )
    for flight in validation_flights:
        for support_name, support in (
            ("compact_full", _compact_support(flight)),
            ("poll_common", poll_support(flight, poll)),
        ):
            metric_rows.append(
                _metric_row(
                    flight,
                    split="validation",
                    support_name=support_name,
                    selected=support,
                    model=full_fit.model,
                    fuel_prior=fuel_prior,
                )
            )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["split", "registration", "recording", "support"], kind="stable"
    )
    aggregate = _aggregate(metrics)
    validation_full = _lookup(aggregate, "validation", "compact_full")
    validation_common = _lookup(aggregate, "validation", "poll_common")
    all_flights = fit_flights + validation_flights
    coverage = _coverage(all_flights)
    raw_negative_fraction = float(
        np.mean(
            np.concatenate(
                [_raw_predict(full_fit.model, flight) < 0.0 for flight in validation_flights]
            )
        )
    )
    monotonic = []
    for flight in validation_flights:
        plus = full_fit.model.raw_thrust_n(
            n1_percent=flight.n1_percent + 0.1,
            mach=flight.mach,
            pressure_ratio=flight.pressure_ratio,
            static_temperature_k=flight.static_temperature_k,
        )
        minus = full_fit.model.raw_thrust_n(
            n1_percent=np.maximum(0.0, flight.n1_percent - 0.1),
            mach=flight.mach,
            pressure_ratio=flight.pressure_ratio,
            static_temperature_k=flight.static_temperature_k,
        )
        monotonic.append(plus >= minus)
    monotonic_fraction = float(np.mean(np.concatenate(monotonic)))
    thresholds = {
        "minimum_all_support_fraction": 0.99,
        "minimum_descent_support_fraction": 0.95,
        "maximum_mean_thrust_rmse_kn": 3.6493693277033814,
        "maximum_worst_thrust_rmse_kn": 5.047,
        "maximum_absolute_mean_thrust_bias_kn": 0.5,
        "minimum_local_n1_monotonic_fraction": 0.995,
        "maximum_raw_negative_prediction_fraction": 0.01,
        "maximum_fuel_flow_rmse_kg_s": 0.08893923531893806,
    }
    checks = {
        "all_support": coverage["all"]["compact_support_fraction"]
        >= thresholds["minimum_all_support_fraction"],
        "descent_support": coverage["descent"]["compact_support_fraction"]
        >= thresholds["minimum_descent_support_fraction"],
        "mean_thrust_rmse": validation_full["mean_flight_thrust_rmse_kn"]
        <= thresholds["maximum_mean_thrust_rmse_kn"],
        "worst_thrust_rmse": validation_full["worst_flight_thrust_rmse_kn"]
        <= thresholds["maximum_worst_thrust_rmse_kn"],
        "mean_thrust_bias": abs(validation_full["mean_flight_thrust_bias_kn"])
        <= thresholds["maximum_absolute_mean_thrust_bias_kn"],
        "local_n1_monotonicity": monotonic_fraction
        >= thresholds["minimum_local_n1_monotonic_fraction"],
        "raw_negative_predictions": raw_negative_fraction
        <= thresholds["maximum_raw_negative_prediction_fraction"],
        "fuel_flow_rmse": validation_full["mean_flight_openap_fuel_rmse_kg_s"]
        <= thresholds["maximum_fuel_flow_rmse_kg_s"],
    }
    passed = all(checks.values())
    feature_names = THRUST_MAP_FEATURE_SETS[settings.feature_set]
    document: dict[str, Any] = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": (
            "compact-candidate-passes-thrust-fuel-support-gate"
            if passed
            else "compact-candidate-held-by-thrust-fuel-support-gate"
        ),
        "drag_candidate": "a320_compact_drag_candidate",
        "leakage_control": {
            "drag_fit_uses_qar": False,
            "thrust_structure_selected_on_fit_flights_only": True,
            "complete_flight_lofo": True,
            "fit_flights": [flight.key for flight in fit_flights],
            "validation_flights_loaded_after_selection": [
                flight.key for flight in validation_flights
            ],
            "fuel_flow_used_for_fit_or_selection": False,
        },
        "selected_propulsion_structure": _native(selected_row.to_dict()),
        "full_fit": {
            "rated_total_thrust_n": rated_total_thrust,
            "pressure_exponent": settings.pressure_exponent,
            "feature_set": settings.feature_set,
            "corrected_fan_speed": settings.corrected_fan_speed,
            "feature_order": list(feature_names),
            "coefficients": dict(
                zip(feature_names, full_fit.model.coefficients, strict=True)
            ),
            "sample_count": full_fit.sample_count,
            "weighted_rmse_kn": full_fit.weighted_rmse_kn,
            "robust_scale_kn": full_fit.robust_scale_kn,
            "normal_matrix_condition": full_fit.normal_matrix_condition,
        },
        "coverage": coverage,
        "aggregate_metrics": aggregate,
        "poll_common_support_validation": validation_common,
        "validation_diagnostics": {
            "local_n1_monotonic_fraction": monotonic_fraction,
            "raw_negative_prediction_fraction": raw_negative_fraction,
        },
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": passed,
            "default_plant_changed": False,
            "reason": (
                "All support, thrust, fuel, monotonicity, and positivity checks pass; "
                "the frozen candidate advances to full trajectory comparison."
                if passed
                else "At least one frozen support/thrust/fuel check fails; inspect the "
                "recorded metrics before any trajectory/default-model promotion."
            ),
        },
        "benchmark_source": str(args.poll_benchmark),
        "artifacts": {
            "candidates": str(args.candidates),
            "metrics": str(args.metrics),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
            "report": str(args.report),
        },
    }

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidates, index=False)
    metrics.to_csv(args.metrics, index=False)
    args.parameters.write_text(yaml.safe_dump(_native(document), sort_keys=False))
    poll_benchmark = yaml.safe_load(args.poll_benchmark.read_text())
    _plot(metrics, coverage, poll_benchmark, all_flights, args.figure_prefix)
    _write_report(args.report, document)
    return metrics, document


def main() -> None:
    metrics, document = run(parse_args())
    validation = _lookup(
        document["aggregate_metrics"], "validation", "compact_full"
    )
    print(
        f"passed={document['gate']['passed']} "
        f"validation_thrust_rmse_kn={validation['mean_flight_thrust_rmse_kn']:.4f} "
        f"validation_fuel_rmse={validation['mean_flight_openap_fuel_rmse_kg_s']:.5f} "
        f"rows={len(metrics)}"
    )


if __name__ == "__main__":
    main()
