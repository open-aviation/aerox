#!/usr/bin/env python3
"""Generate review figures for the A320 higher-order modeling results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}
DOUBLE_COLUMN_WIDTH_IN = 183.0 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attitude-metrics",
        type=Path,
        default=Path("docs/modeling/a320-point-mass-replay-metrics.csv"),
    )
    parser.add_argument(
        "--longitudinal-metrics",
        type=Path,
        default=Path("docs/modeling/a320-longitudinal-replay-metrics.csv"),
    )
    parser.add_argument(
        "--yaw-candidates",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-delay-candidates.csv"),
    )
    parser.add_argument(
        "--yaw-metrics",
        type=Path,
        default=Path("docs/modeling/a320-yaw-rate-flight-metrics.csv"),
    )
    parser.add_argument(
        "--observed-replay-metrics",
        type=Path,
        default=Path("docs/modeling/a320-observed-heading-rate-replay-metrics.csv"),
    )
    parser.add_argument(
        "--lateral-residual-candidates",
        type=Path,
        default=Path("docs/modeling/a320-lateral-residual-candidates.csv"),
    )
    parser.add_argument(
        "--lateral-residual-metrics",
        type=Path,
        default=Path("docs/modeling/a320-lateral-residual-flight-metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/modeling/figures"),
    )
    return parser.parse_args()


def _apply_style() -> None:
    """Apply a compact, colorblind-safe publication review style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _flight_marker(registration: str) -> str:
    return "o" if registration == "TAIL-A" else "s"


def _paired_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    columns: list[str],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> None:
    x = np.arange(len(columns), dtype=float)
    for row in frame.itertuples(index=False):
        values = np.asarray([getattr(row, name) for name in columns], dtype=float)
        ax.plot(x, values, color="0.72", linewidth=0.7, zorder=1)
        for index, (value, color) in enumerate(zip(values, colors, strict=True)):
            ax.plot(
                x[index],
                value,
                marker=_flight_marker(row.registration),
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
                zorder=2,
            )
    for index, (column, color) in enumerate(zip(columns, colors, strict=True)):
        values = frame[column].to_numpy(dtype=float)
        if log_scale:
            center = float(np.median(values))
            lower, upper = np.quantile(values, [0.25, 0.75])
            error = np.array([[center - lower], [upper - center]])
        else:
            center = float(np.mean(values))
            error = float(np.std(values))
        ax.errorbar(
            x[index],
            center,
            yerr=error,
            fmt="D",
            color="black",
            markerfacecolor=color,
            markeredgewidth=0.6,
            capsize=2.5,
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if log_scale:
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", color="0.9", linewidth=0.5)
    else:
        ax.grid(axis="y", color="0.92", linewidth=0.5)


def _panel_labels(axes: np.ndarray) -> None:
    for label, ax in zip("ABCD", axes.flat, strict=True):
        ax.text(
            -0.14,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )


def _save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, dpi in ((".pdf", 300), (".png", 300)):
        path = base.with_suffix(suffix)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        outputs.append(path)
        print(f"saved {path}")
    return outputs


def _validation_comparison(
    attitude: pd.DataFrame, longitudinal: pd.DataFrame
) -> pd.DataFrame:
    new = attitude.loc[attitude["split"] == "validation"].copy()
    old = longitudinal.loc[longitudinal["split"] == "validation"].copy()
    return new.merge(
        old,
        on=["registration", "recording"],
        how="inner",
        suffixes=("_attitude", "_longitudinal"),
        validate="one_to_one",
    ).sort_values(["registration", "recording"], kind="stable")


def plot_attitude_replay(
    attitude: pd.DataFrame, longitudinal: pd.DataFrame, output_dir: Path
) -> list[Path]:
    comparison = _validation_comparison(attitude, longitudinal)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.25),
        constrained_layout=True,
    )
    _paired_panel(
        axes[0, 0],
        comparison,
        columns=[
            "window_speed_rmse_mps_longitudinal",
            "window_speed_rmse_mps_attitude",
        ],
        labels=["AOA/N1 plant", "Pitch-bank/thrust"],
        colors=[OKABE_ITO["vermillion"], OKABE_ITO["blue"]],
        title="300-second speed replay",
        ylabel="Speed RMSE (m/s)",
    )
    _paired_panel(
        axes[0, 1],
        comparison,
        columns=[
            "window_altitude_rmse_m_longitudinal",
            "window_altitude_rmse_m_attitude",
        ],
        labels=["AOA/N1 plant", "Pitch-bank/thrust"],
        colors=[OKABE_ITO["vermillion"], OKABE_ITO["blue"]],
        title="300-second altitude replay",
        ylabel="Altitude RMSE (m)",
    )
    _paired_panel(
        axes[1, 0],
        comparison,
        columns=[
            "window_measured_heading_horizontal_rmse_m",
            "window_horizontal_rmse_m",
        ],
        labels=["Measured heading", "Bank propagated"],
        colors=[OKABE_ITO["green"], OKABE_ITO["blue"]],
        title="300-second horizontal replay",
        ylabel="Horizontal RMSE (m)",
    )
    _paired_panel(
        axes[1, 1],
        comparison,
        columns=[
            "full_measured_heading_horizontal_rmse_m",
            "full_horizontal_rmse_m",
        ],
        labels=["Measured heading", "Bank propagated"],
        colors=[OKABE_ITO["green"], OKABE_ITO["blue"]],
        title="Full clean-segment horizontal replay",
        ylabel="Horizontal RMSE (m)",
        log_scale=True,
    )
    _panel_labels(axes)
    figure.suptitle(
        "A320 attitude/net-thrust replay — six evaluation flights",
        fontsize=10,
        fontweight="bold",
    )
    return _save_figure(figure, output_dir / "a320-point-mass-replay-overview")


def _scatter_by_split(ax: plt.Axes, yaw: pd.DataFrame) -> None:
    for index, split in enumerate(("fit_lofo", "validation")):
        current = yaw.loc[yaw["split"] == split]
        jitter = np.linspace(-0.12, 0.12, len(current))
        color = OKABE_ITO["sky_blue"] if split == "fit_lofo" else OKABE_ITO["orange"]
        markers = [_flight_marker(value) for value in current["registration"]]
        for offset, marker, value in zip(
            jitter,
            markers,
            current["heading_increment_rmse_deg"],
            strict=True,
        ):
            ax.plot(
                index + offset,
                value,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
            )
        mean = current["heading_increment_rmse_deg"].mean()
        standard_deviation = current["heading_increment_rmse_deg"].std(ddof=0)
        ax.errorbar(
            index,
            mean,
            yerr=standard_deviation,
            fmt="D",
            color="black",
            markerfacecolor=color,
            capsize=2.5,
            zorder=3,
        )
    ax.set_xticks([0, 1], ["Fit LOFO", "Evaluation"])
    ax.set_ylabel("5-second increment RMSE (deg)")
    ax.set_title("Held-flight heading increments", loc="left")
    ax.grid(axis="y", color="0.92", linewidth=0.5)


def plot_yaw_rate_audit(
    attitude: pd.DataFrame,
    candidates: pd.DataFrame,
    yaw: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    validation_bank = attitude.loc[attitude["split"] == "validation"]
    validation_yaw = yaw.loc[yaw["split"] == "validation"]
    comparison = validation_yaw.merge(
        validation_bank,
        on=["registration", "recording"],
        how="inner",
        suffixes=("_yaw", "_bank"),
        validate="one_to_one",
    ).sort_values(["registration", "recording"], kind="stable")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.25),
        constrained_layout=True,
    )

    axes[0, 0].plot(
        candidates["delay_s"],
        candidates["weighted_rmse_deg"],
        color=OKABE_ITO["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=OKABE_ITO["blue"],
    )
    selected = candidates.loc[candidates["huber_objective"].idxmin()]
    axes[0, 0].plot(
        selected["delay_s"],
        selected["weighted_rmse_deg"],
        marker="D",
        color=OKABE_ITO["vermillion"],
        linestyle="none",
        label=f"Selected: {int(selected['delay_s'])} s",
    )
    axes[0, 0].set_xlabel("Trajectory-reference delay (s)")
    axes[0, 0].set_ylabel("Weighted 5-second RMSE (deg)")
    axes[0, 0].set_title("Fit-only delay screen", loc="left")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", color="0.92", linewidth=0.5)

    _scatter_by_split(axes[0, 1], yaw)
    _paired_panel(
        axes[1, 0],
        comparison,
        columns=["window_heading_rmse_deg_bank", "window_heading_rmse_deg_yaw"],
        labels=["Bank\npropagated", "Observed\nyaw rate"],
        colors=[OKABE_ITO["blue"], OKABE_ITO["orange"]],
        title="300-second heading replay",
        ylabel="Heading RMSE (deg)",
    )
    _paired_panel(
        axes[1, 1],
        comparison,
        columns=[
            "full_measured_heading_baseline_rmse_m",
            "full_horizontal_rmse_m_yaw",
            "full_horizontal_rmse_m_bank",
        ],
        labels=["Measured\nheading", "Observed\nyaw rate", "Bank\npropagated"],
        colors=[
            OKABE_ITO["green"],
            OKABE_ITO["orange"],
            OKABE_ITO["blue"],
        ],
        title="Full clean-segment horizontal replay",
        ylabel="Horizontal RMSE (m)",
        log_scale=True,
    )
    _panel_labels(axes)
    figure.suptitle(
        "A320 observed yaw-rate feasibility",
        fontsize=10,
        fontweight="bold",
    )
    return _save_figure(figure, output_dir / "a320-yaw-rate-audit-overview")


def _split_metric_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    column: str,
    scale: float,
    title: str,
    ylabel: str,
) -> None:
    for index, split in enumerate(("calibration", "validation")):
        current = frame.loc[frame["split"] == split]
        values = scale * current[column].to_numpy(dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(current))
        color = OKABE_ITO["sky_blue"] if split == "calibration" else OKABE_ITO["orange"]
        for offset, value, registration in zip(
            jitter,
            values,
            current["registration"],
            strict=True,
        ):
            ax.plot(
                index + offset,
                value,
                marker=_flight_marker(registration),
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
            )
        ax.errorbar(
            index,
            float(np.mean(values)),
            yerr=float(np.std(values)),
            fmt="D",
            color="black",
            markerfacecolor=color,
            markeredgewidth=0.6,
            capsize=2.5,
            zorder=3,
        )
    ax.set_xticks([0, 1], ["Calibration\n(n=10)", "Evaluation\n(n=6)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="0.92", linewidth=0.5)


def plot_observed_heading_rate_replay(
    attitude: pd.DataFrame,
    observed: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    bank = attitude.loc[attitude["split"] == "validation"]
    replay = observed.loc[observed["split"] == "validation"]
    comparison = replay.merge(
        bank,
        on=["registration", "recording"],
        how="inner",
        suffixes=("_observed", "_bank"),
        validate="one_to_one",
    ).sort_values(["registration", "recording"], kind="stable")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.25),
        constrained_layout=True,
    )
    _paired_panel(
        axes[0, 0],
        comparison,
        columns=[
            "window_measured_heading_horizontal_rmse_m_observed",
            "window_horizontal_rmse_m_observed",
            "window_horizontal_rmse_m_bank",
        ],
        labels=["Measured\nheading", "Observed\nyaw rate", "Bank\npropagated"],
        colors=[OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["blue"]],
        title="300-second force-balance replay",
        ylabel="Horizontal RMSE (m)",
    )
    _paired_panel(
        axes[0, 1],
        comparison,
        columns=[
            "full_measured_heading_horizontal_rmse_m_observed",
            "full_horizontal_rmse_m_observed",
            "full_horizontal_rmse_m_bank",
        ],
        labels=["Measured\nheading", "Observed\nyaw rate", "Bank\npropagated"],
        colors=[OKABE_ITO["green"], OKABE_ITO["orange"], OKABE_ITO["blue"]],
        title="Full clean-segment force-balance replay",
        ylabel="Horizontal RMSE (m)",
        log_scale=True,
    )
    _split_metric_panel(
        axes[1, 0],
        observed,
        column="window_lateral_force_residual_rmse_weight_fraction",
        scale=100.0,
        title="300-second implied turn-force residual",
        ylabel="Residual RMSE (% aircraft weight)",
    )
    _split_metric_panel(
        axes[1, 1],
        observed,
        column="window_turn_excited_residual_p95_abs_weight_fraction",
        scale=100.0,
        title="Turn-excited residual (|bank| ≥ 3°)",
        ylabel="95th percentile |residual| (% weight)",
    )
    _panel_labels(axes)
    figure.suptitle(
        "A320 observed-heading-rate force consistency",
        fontsize=10,
        fontweight="bold",
    )
    return _save_figure(
        figure,
        output_dir / "a320-observed-heading-rate-replay-overview",
    )


def plot_lateral_residual_audit(
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    delay = candidates.loc[candidates["candidate"] == "lateral_acceleration_delay"]
    fit = metrics.loc[metrics["split"] == "fit_lofo"].copy()
    validation = metrics.loc[metrics["split"] == "validation"].copy()
    fit["latacc_percent"] = 100.0 * fit["lateral_acceleration_rmse_improvement"]
    fit["beta_percent"] = 100.0 * fit["beta_proxy_rmse_improvement"]
    fit["combined_percent"] = 100.0 * fit["combined_rmse_improvement"]
    validation["overall_percent"] = (
        100.0 * validation["lateral_acceleration_rmse_improvement"]
    )
    validation["turn_percent"] = (
        100.0 * validation["lateral_acceleration_turn_rmse_improvement"]
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.25),
        constrained_layout=True,
    )

    axes[0, 0].plot(
        delay["delay_s"],
        100.0 * delay["fit_lofo_mean_rmse_weight_fraction"],
        color=OKABE_ITO["blue"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=OKABE_ITO["blue"],
    )
    selected = delay.loc[delay["fit_lofo_mean_rmse_weight_fraction"].idxmin()]
    axes[0, 0].plot(
        selected["delay_s"],
        100.0 * selected["fit_lofo_mean_rmse_weight_fraction"],
        marker="D",
        color=OKABE_ITO["vermillion"],
        linestyle="none",
        label=f"Selected: {int(selected['delay_s'])} s",
    )
    axes[0, 0].set_xlabel("Lateral-acceleration delay (s)")
    axes[0, 0].set_ylabel("Fit LOFO residual RMSE (% weight)")
    axes[0, 0].set_title("Fit-only delay selection", loc="left")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", color="0.92", linewidth=0.5)

    _paired_panel(
        axes[0, 1],
        fit,
        columns=["latacc_percent", "beta_percent", "combined_percent"],
        labels=["Lateral\nacceleration", "Beta\nproxy", "Combined"],
        colors=[OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["orange"]],
        title="Calibration-flight LOFO improvement",
        ylabel="RMSE improvement vs bias-only (%)",
    )
    axes[0, 1].axhline(5.0, color="0.35", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="0.6", linewidth=0.6)

    _paired_panel(
        axes[1, 0],
        validation,
        columns=["overall_percent", "turn_percent"],
        labels=["All valid\nsamples", "|Bank| ≥ 3°"],
        colors=[OKABE_ITO["blue"], OKABE_ITO["vermillion"]],
        title="Evaluation lateral-acceleration result",
        ylabel="RMSE improvement vs bias-only (%)",
    )
    axes[1, 0].axhline(5.0, color="0.35", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(0.0, color="0.6", linewidth=0.6)

    for row in validation.itertuples(index=False):
        axes[1, 1].plot(
            row.beta_proxy_p95_abs_deg,
            100.0 * row.combined_incremental_over_latacc,
            marker=_flight_marker(row.registration),
            markerfacecolor=OKABE_ITO["purple"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="none",
        )
    axes[1, 1].axhline(0.0, color="0.6", linewidth=0.6)
    axes[1, 1].set_xlabel("Beta-proxy 95th percentile |value| (deg)")
    axes[1, 1].set_ylabel("Combined gain over lateral acceleration (%)")
    axes[1, 1].set_title("Evaluation beta-proxy contribution", loc="left")
    axes[1, 1].grid(axis="y", color="0.92", linewidth=0.5)

    _panel_labels(axes)
    figure.suptitle(
        "A320 lateral-residual attribution",
        fontsize=10,
        fontweight="bold",
    )
    return _save_figure(
        figure,
        output_dir / "a320-lateral-residual-audit-overview",
    )


def run(args: argparse.Namespace) -> list[Path]:
    _apply_style()
    attitude = pd.read_csv(args.point_mass_metrics)
    longitudinal = pd.read_csv(args.longitudinal_metrics)
    candidates = pd.read_csv(args.yaw_candidates)
    yaw = pd.read_csv(args.yaw_metrics)
    observed = pd.read_csv(args.observed_replay_metrics)
    lateral_candidates = pd.read_csv(args.lateral_residual_candidates)
    lateral_metrics = pd.read_csv(args.lateral_residual_metrics)
    outputs = plot_attitude_replay(attitude, longitudinal, args.output_dir)
    outputs.extend(plot_yaw_rate_audit(attitude, candidates, yaw, args.output_dir))
    outputs.extend(
        plot_observed_heading_rate_replay(attitude, observed, args.output_dir)
    )
    outputs.extend(
        plot_lateral_residual_audit(
            lateral_candidates,
            lateral_metrics,
            args.output_dir,
        )
    )
    plt.close("all")
    return outputs


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
