#!/usr/bin/env python3
"""Infer persistent effective A320 guidance regimes from QAR trajectories."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
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
from aerox.flightdata.a320_qar import (
    FT_TO_M,
    KNOT_TO_MPS,
    TIME_COLUMN,
    bounded_sample_and_hold,
    decode_a320_flight,
    raw_columns,
)
from aerox.models import (
    N1_ENVELOPE_FEATURE_SETS,
    EffectiveN1Envelope,
)
from aerox.modeling import GuidanceModeThresholds, infer_guidance_modes
from tools.audit_a320_fms_intent_channels import (
    INTENT_CHANNELS,
    _processed_origin_time_s,
)
from tools.audit_a320_fms_intent_channels import (
    _load_timeline as _load_holdout_timeline,
)
from tools.calibrate_a320_effective_thrust import RAW_FILES
from tools.replay_a320_point_mass import _read_fuel_model, _read_thrust_map

UPDATED = "2026-08-07"
FACTOR_COLUMNS = (
    "vertical_mode",
    "speed_mode",
    "thrust_mode",
    "lateral_mode",
    "speed_target_evidence",
    "guidance_family",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    defaults = {
        "panel": "docs/modeling/a320-replay-panel.csv",
        "selection": "docs/modeling/a320-long-range-holdout-selection.yml",
        "vertical-parameters": "docs/modeling/a320-vertical-kinematic-calibration.yml",
        "horizontal-parameters": "docs/modeling/a320-horizontal-kinematic-calibration.yml",
        "thrust-parameters": "docs/modeling/a320-effective-thrust-calibration.yml",
        "fuel-parameters": "docs/modeling/a320-effective-fuel-calibration.yml",
        "pitch-bank-parameters": "docs/modeling/a320-pitch-bank-input-calibration.yml",
        "surface-parameters": "docs/modeling/a320-vspaero-lift-surface-identification.yml",
        "boundary-parameters": "docs/modeling/a320-vspaero-hybrid-boundary-selection.yml",
        "lift-surface": "docs/modeling/a320-vspaero-mach-extended-lift-surface.csv",
        "n1-envelope": "docs/modeling/a320-autothrust-envelope.yml",
        "climb-capacity": "docs/modeling/a320-demonstrated-climb-capacity.yml",
        "flight-metrics": "docs/modeling/a320-guidance-mode-flight-metrics.csv",
        "segments": "docs/modeling/a320-guidance-mode-segments.csv",
        "representative-timeline": "docs/modeling/a320-guidance-mode-representative-timeline.csv",
        "parameters": "docs/modeling/a320-guidance-mode-audit.yml",
        "report": "docs/modeling/a320-guidance-mode-audit.md",
        "figure-prefix": "docs/modeling/figures/a320-guidance-mode-overlay",
    }
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    return parser.parse_args()


def _ordered_raw(frame: pd.DataFrame, column: str) -> np.ndarray:
    time = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time)
    indices = np.flatnonzero(keep)[np.argsort(time[keep], kind="stable")]
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)[indices]


def _read_raw_flight(
    raw_dir: Path, registration: str, recording: str
) -> tuple[pd.DataFrame, Any, dict[str, np.ndarray]]:
    required = list(dict.fromkeys((*raw_columns(), *INTENT_CHANNELS)))
    raw = pd.read_parquet(
        raw_dir / RAW_FILES[registration],
        columns=required,
        filters=[("recording", "==", recording)],
        engine="pyarrow",
    )
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    held = {
        channel: bounded_sample_and_hold(
            _ordered_raw(raw, channel), decoded.time_s, 1.0
        )
        for channel in INTENT_CHANNELS
    }
    return raw, decoded, held


def _track_from_position(north_m: np.ndarray, east_m: np.ndarray) -> np.ndarray:
    north = np.asarray(north_m, dtype=float)
    east = np.asarray(east_m, dtype=float)
    north_rate = np.gradient(north)
    east_rate = np.gradient(east)
    return np.rad2deg(np.arctan2(east_rate, north_rate)) % 360.0


def _track_from_geodetic(latitude_rad: np.ndarray, longitude_rad: np.ndarray) -> np.ndarray:
    latitude = np.interp(
        np.arange(len(latitude_rad)),
        np.flatnonzero(np.isfinite(latitude_rad)),
        latitude_rad[np.isfinite(latitude_rad)],
    )
    longitude = np.interp(
        np.arange(len(longitude_rad)),
        np.flatnonzero(np.isfinite(longitude_rad)),
        longitude_rad[np.isfinite(longitude_rad)],
    )
    radius_m = 6_371_000.0
    north = radius_m * latitude
    east = radius_m * np.unwrap(longitude) * np.cos(float(np.median(latitude)))
    return _track_from_position(north, east)


def _coefficient_vector(values: dict[str, float], order: tuple[str, ...]) -> np.ndarray:
    return np.asarray([values[name] for name in order], dtype=float)


def _read_n1_envelope(path: Path) -> EffectiveN1Envelope:
    document = yaml.safe_load(path.read_text())
    model = document["model"]
    feature_set = str(model["feature_set"])
    order = N1_ENVELOPE_FEATURE_SETS[feature_set]
    return EffectiveN1Envelope(
        feature_set=feature_set,
        idle_coefficients=_coefficient_vector(model["idle_coefficients"], order),
        climb_coefficients=_coefficient_vector(model["climb_coefficients"], order),
    )


def _estimate_frame(
    frame: pd.DataFrame,
    *,
    thresholds: GuidanceModeThresholds,
) -> pd.DataFrame:
    estimate = infer_guidance_modes(
        pressure_altitude_ft=frame["pressure_altitude_ft"].to_numpy(),
        calibrated_airspeed_kts=frame["actual_cas_kts"].to_numpy(),
        mach=frame["actual_mach"].to_numpy(),
        selected_altitude_ft=frame["selected_altitude_ft"].to_numpy(),
        selected_speed_kts=frame["selected_speed_kts"].to_numpy(),
        selected_mach=frame["selected_mach"].to_numpy(),
        mean_n1_percent=frame["mean_n1_percent"].to_numpy(),
        climb_n1_limit_percent=frame["climb_n1_limit_percent"].to_numpy(),
        heading_deg=frame["heading_deg"].to_numpy(),
        track_deg=frame["track_deg"].to_numpy(),
        selected_heading_deg=frame["selected_heading_deg"].to_numpy(),
        selected_track_deg=frame["selected_track_deg"].to_numpy(),
        bank_angle_deg=frame["bank_angle_deg"].to_numpy(),
        thresholds=thresholds,
    )
    result = frame.copy()
    for name in FACTOR_COLUMNS:
        result[name] = getattr(estimate, name)
    result["vertical_rate_ft_min"] = estimate.vertical_rate_ft_min
    result["cas_rate_kts_s"] = estimate.cas_rate_kts_s
    result["cas_rolling_std_kts"] = estimate.cas_rolling_std_kts
    result["mach_rolling_std"] = estimate.mach_rolling_std
    return result


def _longest_valid_slice(valid: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError("flight contains no valid airborne guidance samples")
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, boundaries)
    return max(groups, key=len)


def _load_panel_flight(
    *,
    raw_dir: Path,
    registration: str,
    recording: str,
    role: str,
    envelope: EffectiveN1Envelope,
    thresholds: GuidanceModeThresholds,
) -> pd.DataFrame:
    _, decoded, held = _read_raw_flight(raw_dir, registration, recording)
    altitude_m = decoded.channel("pressure_altitude")
    altitude_ft = altitude_m / FT_TO_M
    cas_kts = decoded.channel("calibrated_airspeed") / KNOT_TO_MPS
    mach = np.asarray(aero.cas2mach(cas_kts * KNOT_TO_MPS, altitude_m), dtype=float)
    n1 = 0.5 * (
        decoded.channel("n1_1", allow_provisional=True)
        + decoded.channel("n1_2", allow_provisional=True)
    )
    heading = np.rad2deg(
        decoded.channel("true_heading", allow_provisional=True)
    ) % 360.0
    bank = np.rad2deg(decoded.channel("roll", allow_provisional=True))
    track = _track_from_geodetic(
        decoded.channel("latitude"), decoded.channel("longitude")
    )
    fuel_flow = (
        decoded.channel("fuel_flow_1") + decoded.channel("fuel_flow_2")
    )
    mass_kg = decoded.channel("gross_weight")
    valid = (
        decoded.airborne
        & np.isfinite(
            altitude_ft + cas_kts + mach + n1 + heading + bank + fuel_flow + mass_kg
        )
        & (cas_kts >= 80.0)
    )
    selection = _longest_valid_slice(valid)
    altitude_m = altitude_m[selection]
    mach = mach[selection]
    isa_temperature = np.asarray(aero.temperature(altitude_m), dtype=float)
    climb_limit = envelope.climb_limit_percent(
        pressure_altitude_m=altitude_m,
        mach=mach,
        static_temperature_k=isa_temperature,
    )
    time_s = decoded.time_s[selection] - decoded.time_s[selection][0]
    frame = pd.DataFrame(
        {
            "registration": registration,
            "recording": recording,
            "role": role,
            "elapsed_time_s": time_s,
            "pressure_altitude_ft": altitude_ft[selection],
            "actual_cas_kts": cas_kts[selection],
            "actual_mach": mach,
            "selected_altitude_ft": held["ALT_SEL"][selection],
            "selected_speed_kts": held["SPEEDP_SEL"][selection],
            "selected_mach": held["MACH_SEL"][selection],
            "selected_vertical_speed_raw": held["SPEEDV_SEL..ft.mn."][selection],
            "selected_fpa_raw": held["FPA_SEL"][selection],
            "mean_n1_percent": n1[selection],
            "climb_n1_limit_percent": climb_limit,
            "fuel_flow_kg_s": fuel_flow[selection],
            "mass_kg": mass_kg[selection],
            "heading_deg": heading[selection],
            "track_deg": track[selection],
            "selected_heading_deg": held["HEADING_SEL"][selection] % 360.0,
            "selected_track_deg": held["TRACK_SEL"][selection] % 360.0,
            "bank_angle_deg": bank[selection],
        }
    )
    return _estimate_frame(frame, thresholds=thresholds)


def _load_representative(
    args: argparse.Namespace,
    *,
    envelope: EffectiveN1Envelope,
    thresholds: GuidanceModeThresholds,
) -> tuple[pd.DataFrame, str, str]:
    qar, timeline, registration, recording = _load_holdout_timeline(args)
    _, decoded, held = _read_raw_flight(args.raw_dir, registration, recording)
    source_time = (
        _processed_origin_time_s(args.processed, recording, decoded)
        + qar.source_start_time_s
        + qar.reference.time_s
    )
    source_index = np.searchsorted(decoded.time_s, source_time)
    if np.any(source_index >= len(decoded.time_s)) or not np.allclose(
        decoded.time_s[source_index], source_time, atol=1e-6
    ):
        raise ValueError("representative engine channels do not align with reference")
    altitude_m = qar.reference.pressure_altitude_m
    n1 = 0.5 * (
        decoded.channel("n1_1", allow_provisional=True)[source_index]
        + decoded.channel("n1_2", allow_provisional=True)[source_index]
    )
    fuel_flow = (
        decoded.channel("fuel_flow_1")[source_index]
        + decoded.channel("fuel_flow_2")[source_index]
    )
    mach = timeline["actual_mach"].to_numpy(dtype=float)
    climb_limit = envelope.climb_limit_percent(
        pressure_altitude_m=altitude_m,
        mach=mach,
        static_temperature_k=qar.static_temperature_k,
    )
    frame = timeline.copy()
    frame["mean_n1_percent"] = n1
    frame["climb_n1_limit_percent"] = climb_limit
    frame["fuel_flow_kg_s"] = fuel_flow
    frame["mass_kg"] = qar.mass_kg
    frame["heading_deg"] = np.rad2deg(qar.reference.heading_rad) % 360.0
    frame["track_deg"] = _track_from_position(
        qar.north_position_m, qar.east_position_m
    )
    frame["selected_heading_deg"] = held["HEADING_SEL"][source_index] % 360.0
    frame["selected_track_deg"] = held["TRACK_SEL"][source_index] % 360.0
    frame["bank_angle_deg"] = np.rad2deg(
        decoded.channel("roll", allow_provisional=True)[source_index]
    )
    frame = _estimate_frame(frame, thresholds=thresholds)

    thrust_map = _read_thrust_map(args.thrust_parameters)
    fuel_model = _read_fuel_model(args.thrust_parameters, args.fuel_parameters)
    thrust = thrust_map.thrust_n(
        n1_percent=n1,
        mach=mach,
        pressure_ratio=np.asarray(aero.pressure(altitude_m), dtype=float) / aero.p0,
        static_temperature_k=qar.static_temperature_k,
    )
    frame["openap_fuel_flow_kg_s"] = fuel_model.prior.fuel_flow_kg_s(thrust)
    frame["aerox_fuel_flow_kg_s"] = fuel_model.fuel_flow_kg_s(thrust)
    return frame, registration, recording


def _mode_fractions(frame: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "registration": str(frame["registration"].iloc[0]),
        "recording": str(frame["recording"].iloc[0]),
        "role": str(frame["role"].iloc[0]),
        "duration_s": float(frame["elapsed_time_s"].iloc[-1]),
        "maximum_altitude_ft": float(frame["pressure_altitude_ft"].max()),
    }
    for factor in FACTOR_COLUMNS:
        counts = frame[factor].value_counts(normalize=True)
        for label, fraction in counts.items():
            row[f"{factor}__{str(label).lower()}_fraction"] = float(fraction)
    high_level = (
        (frame["pressure_altitude_ft"] >= 25_000.0)
        & (np.abs(frame["vertical_rate_ft_min"]) <= 250.0)
    )
    row["high_level_sample_count"] = int(np.count_nonzero(high_level))
    row["high_level_mach_hold_fraction"] = (
        float(np.mean(frame.loc[high_level, "speed_mode"] == "MACH_HOLD_LIKE"))
        if np.any(high_level)
        else float("nan")
    )
    mach_hold = frame["speed_mode"] == "MACH_HOLD_LIKE"
    row["mach_hold_selected_target_consistency_fraction"] = (
        float(
            np.mean(
                frame.loc[mach_hold, "speed_target_evidence"]
                == "SELECTED_MACH_CONSISTENT"
            )
        )
        if np.any(mach_hold)
        else float("nan")
    )
    return row


def _segments(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    time = frame["elapsed_time_s"].to_numpy(dtype=float)
    for factor in FACTOR_COLUMNS:
        labels = frame[factor].to_numpy(dtype=str)
        boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
        for start, stop in pairwise(boundaries):
            rows.append(
                {
                    "registration": str(frame["registration"].iloc[0]),
                    "recording": str(frame["recording"].iloc[0]),
                    "role": str(frame["role"].iloc[0]),
                    "factor": factor,
                    "label": labels[start],
                    "start_time_s": float(time[start]),
                    "end_time_s": float(time[stop - 1]),
                    "duration_s": float(time[stop - 1] - time[start] + 1.0),
                    "sample_count": int(stop - start),
                }
            )
    return rows


def _shade_modes(axis: Any, time_min: np.ndarray, labels: np.ndarray, colors: dict[str, str]) -> None:
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    for start, stop in pairwise(boundaries):
        axis.axvspan(
            time_min[start],
            time_min[stop - 1],
            color=colors.get(str(labels[start]), "#999999"),
            alpha=0.10,
            linewidth=0,
        )


def _plot(frame: pd.DataFrame, registration: str, recording: str, prefix: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    black, blue, orange, green, magenta, gray = (
        "#000000",
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#777777",
    )
    vertical_colors = {
        "CLIMB": blue,
        "ALT_CAPTURE_LIKE": magenta,
        "ALT_HOLD_LIKE": green,
        "LEVEL": "#56B4E9",
        "DESCENT": orange,
        "TRANSITION": gray,
        "UNRESOLVED": "#BBBBBB",
    }
    speed_colors = {
        "CAS_HOLD_LIKE": blue,
        "MACH_HOLD_LIKE": green,
        "ACCELERATING": orange,
        "DECELERATING": magenta,
        "VARIABLE_SPEED": gray,
        "UNRESOLVED": "#BBBBBB",
    }
    thrust_colors = {
        "CLB_THRUST_LIKE": blue,
        "IDLE_THRUST_LIKE": orange,
        "MODULATED_THRUST": gray,
        "UNRESOLVED": "#BBBBBB",
    }
    time_min = frame["elapsed_time_s"].to_numpy(dtype=float) / 60.0
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(183.0 / 25.4, 8.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.0, 1.15, 0.82]},
    )

    _shade_modes(
        axes[0], time_min, frame["vertical_mode"].to_numpy(), vertical_colors
    )
    axes[0].plot(
        time_min,
        frame["pressure_altitude_ft"] / 1_000.0,
        color=black,
        label="QAR altitude",
    )
    axes[0].step(
        time_min,
        frame["selected_altitude_ft"] / 1_000.0,
        where="post",
        color=orange,
        linestyle="--",
        label="ALT_SEL evidence",
    )
    axes[0].set_ylabel("Pressure altitude (kft)")
    axes[0].legend(ncols=2)

    _shade_modes(axes[1], time_min, frame["speed_mode"].to_numpy(), speed_colors)
    axes[1].plot(time_min, frame["actual_cas_kts"], color=black, label="QAR CAS")
    axes[1].step(
        time_min,
        frame["selected_speed_kts"],
        where="post",
        color=blue,
        linestyle="--",
        label="SPEEDP_SEL evidence",
    )
    axes[1].set_ylabel("CAS (kt)")
    axes[1].legend(ncols=2)

    _shade_modes(axes[2], time_min, frame["speed_mode"].to_numpy(), speed_colors)
    axes[2].plot(time_min, frame["actual_mach"], color=black, label="QAR Mach")
    axes[2].step(
        time_min,
        frame["selected_mach"],
        where="post",
        color=green,
        linestyle="--",
        label="MACH_SEL evidence",
    )
    axes[2].set_ylabel("Mach number")
    axes[2].legend(ncols=2)

    _shade_modes(axes[3], time_min, frame["thrust_mode"].to_numpy(), thrust_colors)
    axes[3].plot(
        time_min, frame["mean_n1_percent"], color=blue, label="mean QAR N1"
    )
    axes[3].plot(
        time_min,
        frame["climb_n1_limit_percent"],
        color=magenta,
        linestyle=":",
        label="CLB boundary",
    )
    axes[3].set_ylabel("N1 (%)")
    fuel_axis = axes[3].twinx()
    fuel_axis.plot(
        time_min,
        frame["fuel_flow_kg_s"],
        color=black,
        linewidth=0.7,
        alpha=0.65,
        label="QAR fuel",
    )
    fuel_axis.plot(
        time_min,
        frame["openap_fuel_flow_kg_s"],
        color=orange,
        linestyle="--",
        label="OpenAP prior",
    )
    fuel_axis.plot(
        time_min,
        frame["aerox_fuel_flow_kg_s"],
        color=green,
        label="Aerox fuel",
    )
    fuel_axis.set_ylabel("Total fuel flow (kg/s)")
    lines = axes[3].lines + fuel_axis.lines
    axes[3].legend(lines, [line.get_label() for line in lines], ncols=3)

    ribbon_factors = [
        "vertical_mode",
        "speed_mode",
        "thrust_mode",
        "guidance_family",
        "lateral_mode",
    ]
    family_colors = {
        "CLB_OR_OP_CLB": blue,
        "DES_OR_OP_DES": orange,
        "ALT_HOLD_COMPATIBLE": green,
        "ALT_CAPTURE_COMPATIBLE": magenta,
        "MODE_FAMILY_UNRESOLVED": gray,
        "UNRESOLVED": "#BBBBBB",
    }
    lateral_colors = {
        "HDG_SELECTED_CONSISTENT": blue,
        "TRK_SELECTED_CONSISTENT": green,
        "TURNING": orange,
        "STRAIGHT_UNRESOLVED": gray,
        "UNRESOLVED": "#BBBBBB",
    }
    factor_colors = [
        vertical_colors,
        speed_colors,
        thrust_colors,
        family_colors,
        lateral_colors,
    ]
    image = np.empty((len(ribbon_factors), len(frame), 4), dtype=float)
    for row, (factor, colors) in enumerate(
        zip(ribbon_factors, factor_colors, strict=True)
    ):
        image[row] = np.asarray(
            [to_rgba(colors.get(str(label), "#BBBBBB")) for label in frame[factor]]
        )
    axes[4].imshow(
        image,
        aspect="auto",
        interpolation="nearest",
        extent=[time_min[0], time_min[-1], len(ribbon_factors), 0],
    )
    axes[4].set_yticks(
        np.arange(len(ribbon_factors)) + 0.5,
        ["Vertical", "Speed", "Thrust", "Family", "Lateral"],
    )
    axes[4].set_xlabel("Elapsed clean-segment time (min)")
    axes[4].set_ylabel("Inferred factor")
    axes[4].set_title(
        "Blue: climb/CAS/CLB; green: altitude/Mach hold; orange: descent/idle; "
        "magenta: capture/deceleration; grey: unresolved/modulated. "
        "Lateral blue: heading-selector-consistent; orange: turning",
        loc="left",
        fontsize=6.8,
    )

    for label, axis in zip("ABCDE", axes, strict=True):
        axis.text(
            -0.095,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.grid(axis="y", color="0.90", linewidth=0.5)
    figure.suptitle(
        f"A320 effective guidance regimes: {registration}/{recording}", fontsize=10
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    widths = [
        max(len(str(headers[index])), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = GuidanceModeThresholds()
    envelope = _read_n1_envelope(args.n1_envelope)
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    frames: list[pd.DataFrame] = []
    for row in panel.itertuples(index=False):
        frame = _load_panel_flight(
            raw_dir=args.raw_dir,
            registration=str(row.registration),
            recording=str(row.recording),
            role=str(row.role),
            envelope=envelope,
            thresholds=thresholds,
        )
        frames.append(frame)
        print(f"segmented {row.registration}/{row.recording}", flush=True)
    metrics = pd.DataFrame([_mode_fractions(frame) for frame in frames]).sort_values(
        ["role", "registration", "recording"], kind="stable"
    )
    segments = pd.DataFrame(
        [segment for frame in frames for segment in _segments(frame)]
    ).sort_values(
        ["registration", "recording", "factor", "start_time_s"], kind="stable"
    )
    representative, representative_registration, representative_recording = (
        _load_representative(args, envelope=envelope, thresholds=thresholds)
    )
    representative["registration"] = representative_registration
    representative["recording"] = representative_recording
    representative["role"] = "post_holdout_diagnostic"

    high_level = metrics.loc[metrics["high_level_sample_count"] >= 300]
    long_segments = segments.loc[segments["duration_s"] >= 60.0]
    family_segments = long_segments.loc[long_segments["factor"] == "guidance_family"]
    demonstrated_families = sorted(
        family_segments.loc[
            family_segments["label"] != "MODE_FAMILY_UNRESOLVED", "label"
        ].unique()
    )
    validation = metrics.loc[metrics["role"] == "validation"]
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "effective-behaviour-segmentation-diagnostic-released",
        "scope": {
            "factorial_states": list(FACTOR_COLUMNS),
            "sample_rate_hz": 1,
            "minimum_dwell_s": thresholds.minimum_dwell_s,
            "exact_fma_mode_identified": False,
            "selected_vs_managed_identified": False,
            "economy_strategy_identified": False,
        },
        "thresholds": asdict(thresholds),
        "panel": {
            "flight_count": len(metrics),
            "registration_count": metrics["registration"].nunique(),
            "validation_flight_count": len(validation),
            "high_level_flight_count": len(high_level),
            "roles": metrics["role"].value_counts().to_dict(),
        },
        "results": {
            "demonstrated_persistent_guidance_families": demonstrated_families,
            "median_high_level_mach_hold_fraction": float(
                high_level["high_level_mach_hold_fraction"].median()
            ),
            "median_mach_hold_selected_target_consistency_fraction": float(
                high_level[
                    "mach_hold_selected_target_consistency_fraction"
                ].median()
            ),
            "validation_contains_climb_family": int(
                family_segments.loc[
                    (family_segments["role"] == "validation")
                    & (family_segments["label"] == "CLB_OR_OP_CLB"),
                    "recording",
                ].nunique()
            ),
            "validation_contains_descent_family": int(
                family_segments.loc[
                    (family_segments["role"] == "validation")
                    & (family_segments["label"] == "DES_OR_OP_DES"),
                    "recording",
                ].nunique()
            ),
            "validation_contains_alt_hold_family": int(
                family_segments.loc[
                    (family_segments["role"] == "validation")
                    & (family_segments["label"] == "ALT_HOLD_COMPATIBLE"),
                    "recording",
                ].nunique()
            ),
        },
        "interpretation": {
            "CLB_OR_OP_CLB": "high scheduled N1 plus climb; managed/open distinction unavailable",
            "DES_OR_OP_DES": "low N1 plus descent; geometric-path/speed-control distinction unavailable",
            "ALT_HOLD_COMPATIBLE": "low vertical rate near the recorded selected altitude",
            "selected_target_consistent": "kinematically consistent evidence, not proof that selected mode was active",
            "econ": "not inferred from low fuel flow; cost index and managed target are absent",
        },
        "fuel_comparison": {
            "figure_includes": [
                "QAR measured total fuel flow",
                "raw OpenAP at_thrust prior",
                "calibrated Aerox fuel relation",
            ],
            "shared_thrust_input": "frozen Aerox effective N1-to-thrust map",
            "dedicated_metrics": "docs/modeling/a320-fuel-flow-comparison-metrics.csv",
        },
        "artifacts": {
            "flight_metrics": str(args.flight_metrics),
            "segments": str(args.segments),
            "representative_timeline": str(args.representative_timeline),
            "representative_timeline_thinning_s": 5,
            "figure": str(args.figure_prefix.with_suffix(".png")),
            "report": str(args.report),
            "runner": "tools/audit_a320_guidance_modes.py",
        },
    }

    compact = metrics[
        [
            "registration",
            "recording",
            "role",
            "maximum_altitude_ft",
            "high_level_mach_hold_fraction",
            "mach_hold_selected_target_consistency_fraction",
        ]
    ].copy()
    compact.columns = [
        "reg",
        "recording",
        "role",
        "max altitude ft",
        "high-level Mach-hold fraction",
        "Mach target-consistency fraction",
    ]
    compact["max altitude ft"] = compact["max altitude ft"].round(0)
    compact["high-level Mach-hold fraction"] = compact[
        "high-level Mach-hold fraction"
    ].round(3)
    compact["Mach target-consistency fraction"] = compact[
        "Mach target-consistency fraction"
    ].round(3)
    report = f"""# A320 effective guidance-mode observability and segmentation audit

Updated: {UPDATED}

## Decision

Aerox can infer persistent **effective guidance behaviours** from the available
QAR, but it cannot decode the exact Airbus FMA modes. The released diagnostic
therefore keeps vertical, speed, thrust, lateral, and target-evidence states
separate. It never converts a matching selector value into proof that selected
mode was active.

The 16-flight panel covers both A320-214 registrations and six validation
flights. Persistent segments of at least 60 s demonstrate the following
combined families: {", ".join(demonstrated_families)}. Among the
{len(high_level)} flights with at least five minutes of high-level data, the
median fraction classified as `MACH_HOLD_LIKE` is
{document["results"]["median_high_level_mach_hold_fraction"]:.1%}. The median
fraction of those Mach-hold samples also matching `MACH_SEL` is
{document["results"]["median_mach_hold_selected_target_consistency_fraction"]:.1%};
this remains consistency evidence because the selected/managed FMA bit is absent.
The combination of high local Mach-hold occupancy and much lower selector
consistency is important: the cruise is usually locally stable, but its target
drifts or changes over the complete flight. A single fixed cruise Mach therefore
cannot reproduce all of the observed variation.

## Method

- One-second QAR signals are smoothed only for derivatives and rolling
  variability; the plotted altitude, CAS, Mach, N1, and fuel flow remain the
  measured values.
- Factor states must persist for at least {thresholds.minimum_dwell_s} s. Short
  isolated runs become `UNRESOLVED` unless surrounded by the same state.
- `CLB_THRUST_LIKE` uses the independently validated condition-dependent CLB
  N1 boundary. Its audit-only panel evaluation uses ISA temperature where a
  matched static-temperature channel is unavailable.
- `IDLE_THRUST_LIKE` is behavioural evidence below {thresholds.idle_n1_threshold_percent:.0f}%
  N1 during descent; the physical idle boundary remains unreleased.
- Lateral inference distinguishes turns and selector-consistent straight
  segments. NAV versus HDG/TRK remains unresolved without route/FMA truth.

## Flight evidence

{_markdown_table(compact)}

## What remains unobservable

`CLB_OR_OP_CLB` and `DES_OR_OP_DES` are intentionally compound names. The
current channels cannot tell whether pitch was following an FMGC geometric
path, an active selected speed, or a managed target. Likewise, low fuel flow
does not identify ECON cruise: cost index, the managed speed target, forecast
wind used by the FMS, and RTA intent are absent.

The next validation step is to test whether mode-conditioned speed intent
reduces held-out QAR CAS/Mach error relative to the current smooth schedule.
Only after that gate should these inferred states select different Aerox FMS
guidance laws or controller gains.

## Artifacts

- Per-flight fractions: `docs/modeling/a320-guidance-mode-flight-metrics.csv`
- Persistent segments: `docs/modeling/a320-guidance-mode-segments.csv`
- Representative 5-second derived timeline: `docs/modeling/a320-guidance-mode-representative-timeline.csv`
- Mode and fuel overlay: `docs/modeling/figures/a320-guidance-mode-overlay.{{png,pdf}}`
- Machine-readable audit: `docs/modeling/a320-guidance-mode-audit.yml`
- Reproducer: `tools/audit_a320_guidance_modes.py`
"""

    args.flight_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.flight_metrics, index=False)
    segments.to_csv(args.segments, index=False)
    representative_mode_columns = [
        "elapsed_time_s",
        *FACTOR_COLUMNS,
        "vertical_rate_ft_min",
        "cas_rate_kts_s",
        "cas_rolling_std_kts",
        "mach_rolling_std",
    ]
    representative.loc[::5, representative_mode_columns].to_csv(
        args.representative_timeline, index=False
    )
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(report)
    _plot(
        representative,
        representative_registration,
        representative_recording,
        args.figure_prefix,
    )
    print(yaml.safe_dump(document["results"], sort_keys=False).strip())


if __name__ == "__main__":
    main()
