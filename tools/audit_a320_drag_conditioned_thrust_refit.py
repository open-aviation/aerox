#!/usr/bin/env python3
"""Refit the A320 effective-thrust surrogate under two frozen drag priors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aerox import aero
from aerox.models import PollSchumannDragModel
from aerox.modeling.identification import robust_weighted_linear_fit
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import (
    THRUST_MAP_FEATURE_SETS,
    EffectiveThrustMap,
    thrust_map_features,
)
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

UPDATED = "2026-08-08"
CURRENT_CD0 = 0.02263
CURRENT_K = 0.04246
VARIANTS = ("aerox_current", "poll_schumann")


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
        "--candidates",
        type=Path,
        default=Path(
            "docs/modeling/a320-drag-conditioned-thrust-refit-candidates.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-drag-conditioned-thrust-refit-metrics.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-drag-conditioned-thrust-refit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-drag-conditioned-thrust-refit.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-drag-conditioned-thrust-refit"),
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


def _support(flight: PropulsionFlight, poll: PollSchumannDragModel) -> np.ndarray:
    return (
        (flight.lift_coefficient >= poll.minimum_lift_coefficient)
        & (flight.lift_coefficient <= poll.maximum_lift_coefficient)
        & (flight.mach >= poll.minimum_mach)
        & (flight.mach <= poll.maximum_mach)
    )


def _reynolds_number(flight: PropulsionFlight, reference_area_m2: float) -> np.ndarray:
    pressure_pa = flight.pressure_ratio * aero.p0
    density = pressure_pa / (aero.R * flight.static_temperature_k)
    viscosity = (
        1.458e-6
        * flight.static_temperature_k**1.5
        / (flight.static_temperature_k + 110.4)
    )
    return (
        np.sqrt(reference_area_m2)
        * density
        * flight.true_airspeed_mps
        / viscosity
    )


def _drag_n(
    flight: PropulsionFlight,
    variant: str,
    poll: PollSchumannDragModel,
) -> np.ndarray:
    if variant == "aerox_current":
        coefficient = CURRENT_CD0 + CURRENT_K * flight.lift_coefficient**2
    elif variant == "poll_schumann":
        supported = _support(flight, poll)
        coefficient = np.full_like(flight.lift_coefficient, np.nan)
        coefficient[supported] = poll.drag_coefficient(
            flight.lift_coefficient[supported],
            flight.mach[supported],
            _reynolds_number(flight, poll.reference_area_m2)[supported],
        )
    else:
        raise ValueError(f"unknown drag variant: {variant}")
    return flight.parasite_drag_basis_n * coefficient


def _target_thrust_n(
    flight: PropulsionFlight,
    variant: str,
    poll: PollSchumannDragModel,
) -> np.ndarray:
    return flight.net_along_path_force_n + _drag_n(flight, variant, poll)


def _fit_map(
    flights: list[PropulsionFlight],
    *,
    variant: str,
    poll: PollSchumannDragModel,
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
        selected = _support(flight, poll) & (
            flight.sample_index % FIT_THINNING_S == 0
        )
        if np.count_nonzero(selected) < 20:
            raise ValueError(f"too few supported fit points in {flight.key}")
        arrays["n1"].append(flight.n1_percent[selected])
        arrays["mach"].append(flight.mach[selected])
        arrays["delta"].append(flight.pressure_ratio[selected])
        arrays["temperature"].append(flight.static_temperature_k[selected])
        arrays["target"].append(_target_thrust_n(flight, variant, poll)[selected])
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
    variant: str,
    poll: PollSchumannDragModel,
    settings: ModelSettings,
    rated_total_thrust_n: float,
) -> dict[str, Any]:
    rmses: list[float] = []
    biases: list[float] = []
    monotonic: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    conditions: list[float] = []
    for held_out in flights:
        training = [flight for flight in flights if flight.key != held_out.key]
        fit = _fit_map(
            training,
            variant=variant,
            poll=poll,
            settings=settings,
            rated_total_thrust_n=rated_total_thrust_n,
        )
        selected = _support(held_out, poll)
        prediction = _predict(fit.model, held_out)[selected]
        target = _target_thrust_n(held_out, variant, poll)[selected]
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
        "variant": variant,
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
    eligible = candidates.loc[
        candidates["observed_local_monotonic_fraction"] >= 0.995
    ]
    if eligible.empty:
        best_fraction = candidates["observed_local_monotonic_fraction"].max()
        eligible = candidates.loc[
            candidates["observed_local_monotonic_fraction"] == best_fraction
        ]
    score_limit = 1.01 * eligible["lofo_mean_flight_rmse_kn"].min()
    near_best = eligible.loc[eligible["lofo_mean_flight_rmse_kn"] <= score_limit]
    return near_best.sort_values(
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
    variant: str,
    model: EffectiveThrustMap,
    poll: PollSchumannDragModel,
    fuel_prior: OpenAPFuelFlowPrior,
) -> dict[str, Any]:
    selected = _support(flight, poll)
    prediction = _predict(model, flight)[selected]
    target = _target_thrust_n(flight, variant, poll)[selected]
    measured_fuel = flight.fuel_flow_kg_s[selected]
    predicted_fuel = fuel_prior.fuel_flow_kg_s(prediction)
    thrust_residual = prediction - target
    fuel_residual = predicted_fuel - measured_fuel
    poll_delta = (
        _target_thrust_n(flight, "poll_schumann", poll)
        - _target_thrust_n(flight, "aerox_current", poll)
    )[selected]
    return {
        "registration": flight.registration,
        "recording": flight.recording,
        "role": flight.role,
        "split": split,
        "variant": variant,
        "total_clean_samples": len(flight.mach),
        "supported_samples": int(np.count_nonzero(selected)),
        "support_fraction": float(np.mean(selected)),
        "thrust_rmse_kn": float(np.sqrt(np.mean(thrust_residual**2)) / 1_000.0),
        "thrust_bias_kn": float(np.mean(thrust_residual) / 1_000.0),
        "thrust_correlation": _safe_correlation(prediction, target),
        "target_thrust_median_kn": float(np.median(target) / 1_000.0),
        "predicted_thrust_median_kn": float(np.median(prediction) / 1_000.0),
        "poll_minus_current_target_median_kn": float(
            np.median(poll_delta) / 1_000.0
        ),
        "openap_fuel_rmse_kg_s": float(np.sqrt(np.mean(fuel_residual**2))),
        "openap_fuel_bias_kg_s": float(np.mean(fuel_residual)),
        "openap_fuel_correlation": _safe_correlation(
            predicted_fuel, measured_fuel
        ),
    }


def _aggregate(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, variant), frame in metrics.groupby(["split", "variant"], sort=True):
        rows.append(
            {
                "split": str(split),
                "variant": str(variant),
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


def _coverage(flights: list[PropulsionFlight], poll: PollSchumannDragModel) -> dict[str, Any]:
    phase = np.concatenate([flight.phase for flight in flights])
    support = np.concatenate([_support(flight, poll) for flight in flights])
    result: dict[str, Any] = {
        "all": {
            "sample_count": len(support),
            "support_fraction": float(np.mean(support)),
        }
    }
    for name in ("climb", "level", "descent"):
        selected = phase == name
        result[name] = {
            "sample_count": int(np.count_nonzero(selected)),
            "support_fraction": float(np.mean(support[selected])),
        }
    return result


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


def _paired_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str,
) -> None:
    colors = {"fit_lofo": "#0072B2", "validation": "#D55E00"}
    positions = {"aerox_current": 0.0, "poll_schumann": 1.0}
    for (_, split), current in frame.groupby(["recording", "split"], sort=True):
        values = current.set_index("variant")[value]
        axis.plot(
            [0.0, 1.0],
            [values["aerox_current"], values["poll_schumann"]],
            color=colors[str(split)],
            alpha=0.30,
            linewidth=0.9,
        )
        axis.scatter(
            [positions[name] for name in VARIANTS],
            [values[name] for name in VARIANTS],
            color=colors[str(split)],
            s=18,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
        )
    for split, color in colors.items():
        axis.scatter([], [], color=color, label=split.replace("_", " "))
    axis.set_xticks([0.0, 1.0], ["Current", "Poll-Schumann"])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)


def _plot(
    metrics: pd.DataFrame,
    flights: list[PropulsionFlight],
    poll: PollSchumannDragModel,
    models: dict[str, EffectiveThrustMap],
    fuel_prior: OpenAPFuelFlowPrior,
    prefix: Path,
) -> None:
    style = {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 220,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
        _paired_panel(
            axes[0, 0],
            metrics,
            "thrust_rmse_kn",
            "Thrust RMSE (kN)",
            "A  Same-support N1-map error",
        )
        axes[0, 0].legend(frameon=False, loc="upper left")
        _paired_panel(
            axes[0, 1],
            metrics,
            "openap_fuel_rmse_kg_s",
            "Fuel-flow RMSE (kg/s)",
            "B  Independent OpenAP fuel diagnostic",
        )

        validation = [flight for flight in flights if flight.role == "validation"]
        measured = np.concatenate(
            [flight.fuel_flow_kg_s[_support(flight, poll)] for flight in validation]
        )
        bins = np.linspace(np.quantile(measured, 0.01), np.quantile(measured, 0.99), 15)
        centers = 0.5 * (bins[:-1] + bins[1:])
        colors = {"aerox_current": "#0072B2", "poll_schumann": "#D55E00"}
        labels = {"aerox_current": "Current refit", "poll_schumann": "Poll-Schumann refit"}
        for variant in VARIANTS:
            prediction = np.concatenate(
                [
                    fuel_prior.fuel_flow_kg_s(_predict(models[variant], flight))[
                        _support(flight, poll)
                    ]
                    for flight in validation
                ]
            )
            digit = np.digitize(measured, bins) - 1
            mean_prediction = np.array(
                [
                    np.mean(prediction[digit == index])
                    if np.any(digit == index)
                    else np.nan
                    for index in range(len(centers))
                ]
            )
            axes[1, 0].plot(
                centers,
                mean_prediction,
                marker="o",
                markersize=3,
                color=colors[variant],
                linewidth=1.5,
                label=labels[variant],
            )
        axes[1, 0].plot(
            [bins[0], bins[-1]],
            [bins[0], bins[-1]],
            linestyle="--",
            color="#555555",
            linewidth=1.0,
            label="1:1",
        )
        axes[1, 0].set_xlabel("Measured QAR total fuel flow (kg/s)")
        axes[1, 0].set_ylabel("Mean OpenAP prediction (kg/s)")
        axes[1, 0].set_title("C  Validation fuel-flow calibration", loc="left", fontweight="bold")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend(frameon=False)

        phase = np.concatenate([flight.phase for flight in flights])
        support = np.concatenate([_support(flight, poll) for flight in flights])
        names = ["all", "climb", "level", "descent"]
        fractions = [
            float(np.mean(support))
            if name == "all"
            else float(np.mean(support[phase == name]))
            for name in names
        ]
        bars = axes[1, 1].barh(
            names,
            100.0 * np.array(fractions),
            color=["#009E73", "#56B4E9", "#0072B2", "#CC79A7"],
        )
        axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        axes[1, 1].set_xlim(0.0, 104.0)
        axes[1, 1].set_xlabel("Samples inside CL 0.3–0.7, Mach ≤ 0.82 (%)")
        axes[1, 1].set_title("D  Candidate support coverage", loc="left", fontweight="bold")
        axes[1, 1].grid(axis="x", alpha=0.25)
        figure.suptitle(
            "A320 drag-conditioned effective-thrust refit",
            fontsize=13,
            fontweight="bold",
        )
        prefix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(prefix.with_suffix(".png"), bbox_inches="tight")
        figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def _summary_lookup(
    aggregate: list[dict[str, Any]], split: str, variant: str
) -> dict[str, Any]:
    matches = [
        row
        for row in aggregate
        if row["split"] == split and row["variant"] == variant
    ]
    if len(matches) != 1:
        raise ValueError(f"missing aggregate for {split}/{variant}")
    return matches[0]


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    validation_selection = panel.loc[panel["role"] == "validation"].copy()
    if len(fit_selection) != 10 or len(validation_selection) != 6:
        raise ValueError("expected the frozen ten-fit/six-validation panel")

    calibration = yaml.safe_load(args.thrust_parameters.read_text())
    model_config = calibration["model"]
    rated_total_thrust = float(model_config["rated_total_thrust_n"])
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
    poll = _poll_schumann_a320()

    fit_flights = _load_flights(
        fit_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=poll.reference_area_m2,
    )
    candidate_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for pressure_exponent in PRESSURE_EXPONENTS:
            for corrected in (False, True):
                for feature_set in THRUST_MAP_FEATURE_SETS:
                    candidate_rows.append(
                        _candidate_row(
                            fit_flights,
                            variant=variant,
                            poll=poll,
                            settings=ModelSettings(
                                pressure_exponent=pressure_exponent,
                                feature_set=feature_set,
                                corrected_fan_speed=corrected,
                            ),
                            rated_total_thrust_n=rated_total_thrust,
                        )
                    )
    candidates = pd.DataFrame(candidate_rows)
    selected_rows = {
        variant: _select_candidate(candidates.loc[candidates["variant"] == variant])
        for variant in VARIANTS
    }
    settings_by_variant = {
        variant: ModelSettings(
            pressure_exponent=float(row["pressure_exponent"]),
            feature_set=str(row["feature_set"]),
            corrected_fan_speed=bool(row["corrected_fan_speed"]),
        )
        for variant, row in selected_rows.items()
    }
    full_fits = {
        variant: _fit_map(
            fit_flights,
            variant=variant,
            poll=poll,
            settings=settings_by_variant[variant],
            rated_total_thrust_n=rated_total_thrust,
        )
        for variant in VARIANTS
    }
    lofo_models: dict[str, dict[str, EffectiveThrustMap]] = {
        variant: {} for variant in VARIANTS
    }
    lofo_fit_records: dict[str, list[FitResult]] = {variant: [] for variant in VARIANTS}
    for held_out in fit_flights:
        training = [flight for flight in fit_flights if flight.key != held_out.key]
        for variant in VARIANTS:
            fit = _fit_map(
                training,
                variant=variant,
                poll=poll,
                settings=settings_by_variant[variant],
                rated_total_thrust_n=rated_total_thrust,
            )
            lofo_models[variant][held_out.key] = fit.model
            lofo_fit_records[variant].append(fit)

    # The propulsion structure and fitting rules are frozen above. Validation
    # samples are loaded only for the final comparison.
    validation_flights = _load_flights(
        validation_selection,
        raw_dir=args.raw_dir,
        processed_path=args.processed,
        vertical=vertical,
        tas_scale=tas_scale,
        wing_area_m2=poll.reference_area_m2,
    )
    metric_rows: list[dict[str, Any]] = []
    for flight in fit_flights:
        for variant in VARIANTS:
            metric_rows.append(
                _metric_row(
                    flight,
                    split="fit_lofo",
                    variant=variant,
                    model=lofo_models[variant][flight.key],
                    poll=poll,
                    fuel_prior=fuel_prior,
                )
            )
    for flight in validation_flights:
        for variant in VARIANTS:
            metric_rows.append(
                _metric_row(
                    flight,
                    split="validation",
                    variant=variant,
                    model=full_fits[variant].model,
                    poll=poll,
                    fuel_prior=fuel_prior,
                )
            )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["split", "registration", "recording", "variant"], kind="stable"
    )
    aggregate = _aggregate(metrics)
    validation_current = _summary_lookup(aggregate, "validation", "aerox_current")
    validation_poll = _summary_lookup(aggregate, "validation", "poll_schumann")
    all_flights = fit_flights + validation_flights
    coverage = _coverage(all_flights, poll)
    passed = (
        validation_poll["mean_flight_openap_fuel_rmse_kg_s"]
        < validation_current["mean_flight_openap_fuel_rmse_kg_s"]
        and validation_poll["mean_flight_thrust_rmse_kn"]
        <= 1.05 * validation_current["mean_flight_thrust_rmse_kn"]
    )
    feature_names_by_variant = {
        variant: THRUST_MAP_FEATURE_SETS[settings.feature_set]
        for variant, settings in settings_by_variant.items()
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": (
            "candidate-passes-supported-envelope-refit-gate"
            if passed
            else "candidate-held-by-supported-envelope-refit-gate"
        ),
        "question": (
            "Does the Poll-Schumann drag candidate remain competitive after "
            "its own N1-to-effective-thrust refit on identical supported samples?"
        ),
        "leakage_control": {
            "structure_reused_from": str(args.thrust_parameters),
            "candidate_family_reused_from_existing_calibration": True,
            "structure_selected_separately_for_each_drag_variant": True,
            "structure_selection_uses_fit_flights_only": True,
            "fit_flights": [flight.key for flight in fit_flights],
            "validation_flights_loaded_after_refits": [
                flight.key for flight in validation_flights
            ],
            "fuel_flow_used_for_fit": False,
            "common_sample_support_for_both_variants": True,
        },
        "candidate_support": {
            "lift_coefficient": [
                poll.minimum_lift_coefficient,
                poll.maximum_lift_coefficient,
            ],
            "mach": [poll.minimum_mach, poll.maximum_mach],
            "coverage": coverage,
        },
        "propulsion_structure_selection": {
            "rated_total_thrust_n": rated_total_thrust,
            "candidate_count_per_variant": int(len(candidates) / len(VARIANTS)),
            "complete_flight_cross_validation": "leave-one-fit-flight-out",
            "primary_score": "mean per-flight thrust RMSE",
            "monotonicity_minimum": 0.995,
            "simplicity_rule": "fewest coefficients within 1 percent of eligible best",
            "selected": {
                variant: row.to_dict() for variant, row in selected_rows.items()
            },
        },
        "full_fit": {},
        "aggregate_metrics": aggregate,
        "gate": {
            "fuel_condition": "candidate validation mean-flight RMSE < current",
            "thrust_condition": "candidate validation mean-flight RMSE <= 1.05 * current",
            "passed": passed,
            "default_plant_changed": False,
            "reason": (
                (
                    "The candidate improves the independent fuel diagnostic and "
                    "keeps held-out thrust error within tolerance, but support "
                    "limits still prevent an unrestricted release."
                )
                if passed
                else (
                    "The independent fuel diagnostic improves, but held-out "
                    "thrust-map error exceeds tolerance after separate structure "
                    "selection; the drag/propulsion ambiguity remains unresolved."
                )
            ),
            "next_gate": (
                (
                    "Define and validate the bounded low-speed/high-CL continuation, "
                    "then run held-out full-flight trajectory and fuel comparisons."
                )
                if passed
                else (
                    "Test a minimally richer, physically constrained thrust "
                    "surrogate or obtain independent installed-thrust evidence "
                    "before extending the drag candidate to full flight."
                )
            ),
        },
        "artifacts": {
            "candidates": str(args.candidates),
            "metrics": str(args.metrics),
            "figure_png": str(args.figure_prefix.with_suffix(".png")),
            "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
        },
    }
    for variant, fit in full_fits.items():
        settings = settings_by_variant[variant]
        feature_names = feature_names_by_variant[variant]
        document["full_fit"][variant] = {
            "pressure_exponent": settings.pressure_exponent,
            "feature_set": settings.feature_set,
            "corrected_fan_speed": settings.corrected_fan_speed,
            "feature_order": list(feature_names),
            "sample_count": fit.sample_count,
            "weighted_rmse_kn": fit.weighted_rmse_kn,
            "robust_scale_kn": fit.robust_scale_kn,
            "normal_matrix_condition": fit.normal_matrix_condition,
            "median_lofo_normal_matrix_condition": float(
                np.median(
                    [item.normal_matrix_condition for item in lofo_fit_records[variant]]
                )
            ),
            "coefficients": dict(
                zip(feature_names, fit.model.coefficients, strict=True)
            ),
        }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    rounded_candidates = candidates.copy()
    for column in rounded_candidates.select_dtypes(include=["number"]):
        if column == "feature_count":
            rounded_candidates[column] = rounded_candidates[column].astype(int)
        else:
            rounded_candidates[column] = rounded_candidates[column].round(8)
    rounded_candidates.to_csv(args.candidates, index=False)
    rounded = metrics.copy()
    for column in rounded.select_dtypes(include=["number"]):
        if column in {"total_clean_samples", "supported_samples"}:
            rounded[column] = rounded[column].astype(int)
        else:
            rounded[column] = rounded[column].round(8)
    rounded.to_csv(args.metrics, index=False)
    args.parameters.write_text(
        yaml.safe_dump(_native(document), sort_keys=False, allow_unicode=True)
    )
    _plot(
        metrics,
        all_flights,
        poll,
        {variant: fit.model for variant, fit in full_fits.items()},
        fuel_prior,
        args.figure_prefix,
    )

    thrust_change = (
        100.0
        * (
            validation_poll["mean_flight_thrust_rmse_kn"]
            / validation_current["mean_flight_thrust_rmse_kn"]
            - 1.0
        )
    )
    fuel_change = (
        100.0
        * (
            validation_poll["mean_flight_openap_fuel_rmse_kg_s"]
            / validation_current["mean_flight_openap_fuel_rmse_kg_s"]
            - 1.0
        )
    )
    current_settings = settings_by_variant["aerox_current"]
    poll_settings = settings_by_variant["poll_schumann"]
    if passed:
        decision_text = (
            "The candidate passes this bounded gate: it is not merely rescued by "
            "the old thrust map and is preferable inside its stated support. The "
            "released plant nevertheless remains unchanged because a continuous "
            "low-speed/high-CL continuation is still required."
        )
    else:
        decision_text = (
            "The evidence remains mixed, so the candidate is held. Its independent "
            "fuel diagnostic improves, but its held-out N1-map error exceeds the "
            "predeclared tolerance even after separate structure selection. This "
            "points to either remaining drag error or an under-specified propulsion "
            "surrogate; neither should be hidden by changing the default plant."
        )
    report = f"""# A320 drag-conditioned effective-thrust refit

## Outcome

The Poll-Schumann candidate **{'passes' if passed else 'does not pass'}** the
supported-envelope refit gate. Both drag alternatives use exactly the same
`CL=0.3`--`0.7`, `Mach<=0.82` samples. The existing bounded candidate family
is reselected separately for each polar using only the ten fit flights: current
drag selects `{current_settings.feature_set}` with pressure exponent
`{current_settings.pressure_exponent:g}`, while Poll-Schumann selects
`{poll_settings.feature_set}` with pressure exponent
`{poll_settings.pressure_exponent:g}`. Fuel flow is not used for fitting or
selection.

![A320 drag-conditioned thrust refit](figures/a320-drag-conditioned-thrust-refit.png)

Figure 1. Paired flight errors, an independent OpenAP fuel-flow diagnostic,
validation calibration, and the candidate's phase coverage. A
[vector PDF](figures/a320-drag-conditioned-thrust-refit.pdf) is also provided.

## Held-out result

On the six untouched validation flights, mean per-flight thrust-map RMSE changes
from `{validation_current['mean_flight_thrust_rmse_kn']:.3f}` kN to
`{validation_poll['mean_flight_thrust_rmse_kn']:.3f}` kN
(`{thrust_change:+.1f}%`). Mean OpenAP-versus-QAR fuel-flow RMSE changes from
`{validation_current['mean_flight_openap_fuel_rmse_kg_s']:.4f}` kg/s to
`{validation_poll['mean_flight_openap_fuel_rmse_kg_s']:.4f}` kg/s
(`{fuel_change:+.1f}%`). The fuel diagnostic is independent of the thrust fit,
although it remains an OpenAP prior rather than a CFM56 manufacturer deck.

## Coverage limit

The public candidate covers `{100.0 * coverage['all']['support_fraction']:.1f}%`
of clean samples: `{100.0 * coverage['climb']['support_fraction']:.1f}%` of
climb, `{100.0 * coverage['level']['support_fraction']:.1f}%` of level, and
`{100.0 * coverage['descent']['support_fraction']:.1f}%` of descent samples.
That is sufficient for a meaningful high-speed refit but not for an
unrestricted full-flight release.

## Decision

{decision_text} The next task is a physically continuous, explicitly bounded
low-speed/high-CL continuation, followed by the requested full held-out
trajectory and fuel comparison.

## Reproduction

- Script: `tools/audit_a320_drag_conditioned_thrust_refit.py`
- Structure candidates: `docs/modeling/a320-drag-conditioned-thrust-refit-candidates.csv`
- Metrics: `docs/modeling/a320-drag-conditioned-thrust-refit-metrics.csv`
- Machine-readable result: `docs/modeling/a320-drag-conditioned-thrust-refit.yml`
"""
    args.report.write_text(report)
    return metrics, document


if __name__ == "__main__":
    run(parse_args())
