#!/usr/bin/env python3
"""Render the six-view long-range A320 intent and trajectory validation."""

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

from aerox.models.atmosphere import (
    calibrated_airspeed_from_true_airspeed,
    mach_from_true_airspeed,
)
from tools.replay_a320_point_mass import _read_fuel_model
from tools.validate_a320_fms_long_range_qar import _load_holdout

FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
COLORS = {
    "qar": "#000000",
    "generated": "#D55E00",
    "plan": "#009E73",
    "selector": "#0072B2",
    "secondary": "#CC79A7",
    "limit": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "adaptive-selection": "docs/fms/a320-fms-turn-aware-intent-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "thrust-parameters": "docs/modeling/a320-effective-thrust-calibration.yml",
        "fuel-parameters": "docs/modeling/a320-effective-fuel-calibration.yml",
        "history": "docs/fms/a320-fms-vertical-intent-history.csv",
        "profile": "docs/fms/a320-fms-vertical-intent-profile.csv",
        "errors": "docs/fms/a320-fms-vertical-intent-errors.csv",
        "selector-timeline": "docs/fms/a320-fms-qar-intent-channel-timeline.csv",
        "parameters": "docs/fms/a320-fms-vertical-intent.yml",
        "output-dir": "docs/fms/figures/a320-fms-full-validation",
        "metrics": "docs/fms/a320-fms-full-validation.yml",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _style(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.55)


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _cumulative(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time_s)
    return np.r_[0.0, np.cumsum(increments)]


def _panel_labels(axes: Any) -> None:
    for label, axis in zip("ABCDEF", np.ravel(axes), strict=False):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
        }
    )
    history = pd.read_csv(args.history)
    profile = pd.read_csv(args.profile)
    errors = pd.read_csv(args.errors)
    selectors = pd.read_csv(args.selector_timeline)
    qar, registration, recording = _load_holdout(args)
    generated_time = history["elapsed_time_s"].to_numpy()
    qar_time = qar.reference.time_s
    plan_time = profile["predicted_elapsed_time_s"].to_numpy()
    generated_temperature = history["static_temperature_k"].to_numpy()
    generated_cas = np.asarray(
        calibrated_airspeed_from_true_airspeed(
            history["true_airspeed_mps"],
            history["pressure_altitude_m"],
            generated_temperature,
        )
    )
    generated_mach = np.asarray(
        mach_from_true_airspeed(
            history["true_airspeed_mps"],
            history["pressure_altitude_m"],
            generated_temperature,
        )
    )
    adaptive = yaml.safe_load(args.adaptive_selection.read_text())
    waypoint_indices = np.asarray(adaptive["anchor_indices"], dtype=int)
    prefix = args.output_dir

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.plot(
        qar.east_position_m / 1_000.0,
        qar.north_position_m / 1_000.0,
        color=COLORS["qar"],
        label="QAR trajectory",
    )
    axis.plot(
        history["east_position_m"] / 1_000.0,
        history["north_position_m"] / 1_000.0,
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox generated",
    )
    axis.plot(
        qar.east_position_m[waypoint_indices] / 1_000.0,
        qar.north_position_m[waypoint_indices] / 1_000.0,
        color=COLORS["plan"],
        marker="o",
        linestyle="none",
        markersize=2.8,
        label="Lateral pseudo-waypoints",
    )
    axis.set(xlabel="Local east (km)", ylabel="Local north (km)")
    axis.axis("equal")
    axis.legend(frameon=False)
    _style(axis)
    _save(figure, prefix / "01-lateral-trajectory")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    axis.plot(
        qar_time / 3_600.0,
        qar.reference.pressure_altitude_m / FT_TO_M / 1_000.0,
        color=COLORS["qar"],
        label="QAR altitude",
    )
    axis.step(
        selectors["elapsed_time_s"] / 3_600.0,
        selectors["selected_altitude_ft"] / 1_000.0,
        where="post",
        color=COLORS["selector"],
        alpha=0.45,
        label="ALT_SEL",
    )
    axis.plot(
        plan_time / 3_600.0,
        profile["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=COLORS["plan"],
        linestyle=":",
        label="Aerox intent profile",
    )
    axis.plot(
        generated_time / 3_600.0,
        history["pressure_altitude_m"] / FT_TO_M / 1_000.0,
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox generated",
    )
    axis.set(xlabel="Elapsed time (h)", ylabel="Pressure altitude (kft)")
    axis.legend(frameon=False, ncol=2)
    _style(axis)
    _save(figure, prefix / "02-altitude-and-vertical-intent")
    plt.close(figure)

    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        qar_time / 3_600.0,
        selectors["actual_cas_kts"],
        color=COLORS["qar"],
        label="QAR CAS",
    )
    axes[0].step(
        selectors["elapsed_time_s"] / 3_600.0,
        selectors["selected_speed_kts"],
        where="post",
        color=COLORS["selector"],
        alpha=0.55,
        label="SPEEDP_SEL",
    )
    axes[0].plot(
        plan_time / 3_600.0,
        profile["calibrated_airspeed_mps"] / KNOT_TO_MPS,
        color=COLORS["plan"],
        linestyle=":",
        label="Aerox CAS intent",
    )
    axes[0].plot(
        generated_time / 3_600.0,
        generated_cas / KNOT_TO_MPS,
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox generated",
    )
    axes[0].set(ylabel="Calibrated airspeed (kt)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(
        qar_time / 3_600.0,
        selectors["actual_mach"],
        color=COLORS["qar"],
        label="QAR Mach",
    )
    axes[1].step(
        selectors["elapsed_time_s"] / 3_600.0,
        selectors["selected_mach"],
        where="post",
        color=COLORS["selector"],
        alpha=0.55,
        label="MACH_SEL",
    )
    axes[1].plot(
        plan_time / 3_600.0,
        profile["mach"],
        color=COLORS["plan"],
        linestyle=":",
        label="Aerox Mach intent",
    )
    axes[1].plot(
        generated_time / 3_600.0,
        generated_mach,
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox generated",
    )
    axes[1].set(xlabel="Elapsed time (h)", ylabel="Mach number")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        _style(axis)
    _panel_labels(axes)
    _save(figure, prefix / "03-cas-mach-and-speed-intent")
    plt.close(figure)

    figure, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        generated_time / 3_600.0,
        history["pitch_angle_deg"],
        color=COLORS["generated"],
        label="Pitch",
    )
    axes[0].plot(
        generated_time / 3_600.0,
        history["effective_alpha_deg"],
        color=COLORS["secondary"],
        linestyle="--",
        label="Effective alpha",
    )
    axes[0].set(ylabel="Angle (deg)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        generated_time / 3_600.0,
        history["bank_angle_deg"],
        color=COLORS["selector"],
        label="Bank",
    )
    axes[1].set(ylabel="Bank angle (deg)")
    axes[2].plot(
        generated_time / 3_600.0,
        history["net_thrust_kn"],
        color=COLORS["generated"],
        label="Net thrust",
    )
    axes[2].plot(
        generated_time / 3_600.0,
        history["drag_kn"],
        color=COLORS["plan"],
        linestyle="--",
        label="Modeled drag",
    )
    axes[2].set(xlabel="Elapsed time (h)", ylabel="Force (kN)")
    axes[2].legend(frameon=False)
    for axis in axes:
        _style(axis)
    _panel_labels(axes)
    _save(figure, prefix / "04-pitch-bank-thrust-and-drag")
    plt.close(figure)

    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    aerox_fuel = history["fuel_flow_kg_s"].to_numpy()
    openap_fuel = fuel_model.prior.fuel_flow_kg_s(
        history["net_thrust_kn"].to_numpy() * 1_000.0
    )
    qar_fuel = qar.fuel_flow_kg_s
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        qar_time / 3_600.0, qar_fuel, color=COLORS["qar"], label="QAR measured"
    )
    axes[0].plot(
        generated_time / 3_600.0,
        aerox_fuel,
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox calibrated",
    )
    axes[0].plot(
        generated_time / 3_600.0,
        openap_fuel,
        color=COLORS["selector"],
        linestyle=":",
        label="OpenAP prior",
    )
    axes[0].set(ylabel="Fuel flow (kg/s)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(
        qar_time / 3_600.0,
        _cumulative(qar_fuel, qar_time),
        color=COLORS["qar"],
        label="QAR measured",
    )
    axes[1].plot(
        generated_time / 3_600.0,
        _cumulative(aerox_fuel, generated_time),
        color=COLORS["generated"],
        linestyle="--",
        label="Aerox calibrated",
    )
    axes[1].plot(
        generated_time / 3_600.0,
        _cumulative(openap_fuel, generated_time),
        color=COLORS["selector"],
        linestyle=":",
        label="OpenAP prior",
    )
    axes[1].set(xlabel="Elapsed time (h)", ylabel="Cumulative fuel (kg)")
    for axis in axes:
        _style(axis)
    _panel_labels(axes)
    _save(figure, prefix / "05-fuel-flow-qar-aerox-openap")
    plt.close(figure)

    figure, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    error_time = errors["elapsed_time_s"] / 3_600.0
    axes[0].plot(
        error_time, errors["horizontal_error_m"] / 1_000.0, color=COLORS["selector"]
    )
    axes[0].set(ylabel="Horizontal error (km)")
    axes[1].plot(
        error_time,
        errors["altitude_error_m"] / FT_TO_M,
        color=COLORS["generated"],
        label="Altitude",
    )
    speed_axis = axes[1].twinx()
    speed_axis.plot(
        error_time,
        errors["true_airspeed_error_mps"] / KNOT_TO_MPS,
        color=COLORS["plan"],
        linestyle="--",
        label="TAS",
    )
    axes[1].set(ylabel="Altitude error (ft)")
    speed_axis.set_ylabel("TAS error (kt)")
    axes[2].step(
        generated_time / 3_600.0,
        history["pitch_limited"].astype(int),
        where="post",
        color=COLORS["secondary"],
        label="Pitch",
    )
    axes[2].step(
        generated_time / 3_600.0,
        history["bank_limited"].astype(int) + 1.2,
        where="post",
        color=COLORS["selector"],
        label="Bank",
    )
    axes[2].step(
        generated_time / 3_600.0,
        history["thrust_limited"].astype(int) + 2.4,
        where="post",
        color=COLORS["limit"],
        label="Thrust",
    )
    axes[2].set(
        xlabel="Elapsed time (h)",
        ylabel="Limiter state",
        yticks=[0.5, 1.7, 2.9],
        yticklabels=["Pitch", "Bank", "Thrust"],
        ylim=(-0.1, 3.6),
    )
    for axis in axes:
        _style(axis)
    _style(speed_axis)
    _panel_labels(axes)
    _save(figure, prefix / "06-errors-and-control-limiters")
    plt.close(figure)

    common = qar_time <= generated_time[-1]
    generated_aerox_at_qar = np.interp(qar_time[common], generated_time, aerox_fuel)
    generated_openap_at_qar = np.interp(qar_time[common], generated_time, openap_fuel)
    metrics = {
        "schema_version": 1,
        "aircraft": f"Airbus A320-214, {registration}",
        "recording": f"{registration}/{recording}",
        "intent_source": {
            "altitude": "QAR ALT_SEL cruise target plus exact endpoint states",
            "speed": "QAR SPEEDP_SEL and MACH_SEL stable phase modes",
            "lateral": "frozen geometry-only pseudo-waypoints",
        },
        "fuel": {
            "qar_burn_kg": float(_cumulative(qar_fuel, qar_time)[-1]),
            "aerox_burn_kg": float(_cumulative(aerox_fuel, generated_time)[-1]),
            "openap_burn_kg": float(_cumulative(openap_fuel, generated_time)[-1]),
            "aerox_vs_qar_time_aligned_rmse_kg_s": float(
                np.sqrt(np.mean((generated_aerox_at_qar - qar_fuel[common]) ** 2))
            ),
            "openap_vs_qar_time_aligned_rmse_kg_s": float(
                np.sqrt(np.mean((generated_openap_at_qar - qar_fuel[common]) ** 2))
            ),
        },
        "figures": [
            str(prefix / f"{index:02d}-{name}.png")
            for index, name in enumerate(
                (
                    "lateral-trajectory",
                    "altitude-and-vertical-intent",
                    "cas-mach-and-speed-intent",
                    "pitch-bank-thrust-and-drag",
                    "fuel-flow-qar-aerox-openap",
                    "errors-and-control-limiters",
                ),
                start=1,
            )
        ],
    }
    args.metrics.write_text(yaml.safe_dump(metrics, sort_keys=False))
    print(yaml.safe_dump(metrics, sort_keys=False).strip())


if __name__ == "__main__":
    main()
