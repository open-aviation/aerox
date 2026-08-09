#!/usr/bin/env python3
"""Audit the active A320 wave increment without refitting subcritical drag."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aerox import load_aircraft  # noqa: E402
from aerox.models import DragCondition, build_drag_model  # noqa: E402

DEFAULT_REFERENCE = PROJECT_ROOT / "docs/modeling/a320-published-transonic-drag-reference.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/modeling/a320-incremental-wave-drag-audit.yml"
DEFAULT_POINTS = PROJECT_ROOT / "docs/modeling/a320-incremental-wave-drag-audit.csv"
DEFAULT_REPORT = PROJECT_ROOT / "docs/modeling/a320-incremental-wave-drag-audit.md"
DEFAULT_FIGURE = PROJECT_ROOT / "docs/modeling/figures/a320-incremental-wave-drag-audit"

DEVELOPMENT_LIFT = (0.3, 0.4)
HOLDOUT_LIFT = (0.5,)
LOCK_COEFFICIENT = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def _wave_increment(
    mach: np.ndarray,
    lift: np.ndarray,
    *,
    sweep_rad: float,
    thickness_to_chord_ratio: float,
    technology_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(sweep_rad)
    critical = (
        technology_factor / cosine
        - thickness_to_chord_ratio / cosine**2
        - 0.1 * lift / cosine**3
        - 0.108
    )
    wave = LOCK_COEFFICIENT * np.maximum(mach - critical, 0.0) ** 4
    return wave, critical


def _metrics(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    residual = predicted[mask] - observed[mask]
    return {
        "sample_count": int(np.sum(mask)),
        "rmse_drag_counts": float(1.0e4 * np.sqrt(np.mean(residual**2))),
        "mae_drag_counts": float(1.0e4 * np.mean(np.abs(residual))),
        "bias_drag_counts": float(1.0e4 * np.mean(residual)),
    }


def _fit_technology(
    frame: pd.DataFrame,
    observed: np.ndarray,
    development: np.ndarray,
    *,
    sweep_rad: float,
    thickness_to_chord_ratio: float,
) -> float:
    mach = frame["mach"].to_numpy(dtype=float)
    lift = frame["lift_coefficient"].to_numpy(dtype=float)

    def objective(technology_factor: float) -> float:
        predicted, _ = _wave_increment(
            mach,
            lift,
            sweep_rad=sweep_rad,
            thickness_to_chord_ratio=thickness_to_chord_ratio,
            technology_factor=technology_factor,
        )
        return float(np.mean((predicted[development] - observed[development]) ** 2))

    result = minimize_scalar(
        objective,
        bounds=(0.85, 1.00),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not result.success:
        raise RuntimeError(f"technology-factor fit failed: {result.message}")
    return float(result.x)


def _write_figure(
    frame: pd.DataFrame,
    result: dict[str, Any],
    prefix: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerox-matplotlib")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    colors = {0.3: "#0072B2", 0.4: "#E69F00", 0.5: "#009E73"}
    markers = {0.3: "o", 0.4: "s", 0.5: "^"}
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)

    for lift in (0.3, 0.4, 0.5):
        subset = frame[frame["lift_coefficient"] == lift]
        axes[0, 0].scatter(
            subset["mach"],
            1.0e4 * subset["observed_wave_increment"],
            s=16,
            facecolors="none",
            edgecolors=colors[lift],
            marker=markers[lift],
            linewidths=0.8,
            alpha=0.75,
            label=rf"public increment, $C_L={lift:.1f}$",
        )
        axes[0, 0].plot(
            subset["mach"],
            1.0e4 * subset["active_wave_increment"],
            color=colors[lift],
            lw=1.4,
            label=rf"active model, $C_L={lift:.1f}$",
        )
        axes[0, 0].plot(
            subset["mach"],
            1.0e4 * subset["development_fit_wave_increment"],
            color=colors[lift],
            lw=1.0,
            linestyle="--",
            alpha=0.8,
        )
    axes[0, 0].axhline(0.0, color="0.35", lw=0.7)
    axes[0, 0].set_xlabel("Mach number")
    axes[0, 0].set_ylabel(r"Wave increment, $10^4\,\Delta C_D$ (counts)")
    axes[0, 0].set_title("a  Public incremental rise and compact prior", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=2)

    names = ["No wave", "Active κ=0.95", "Development fit"]
    x = np.arange(len(names))
    width = 0.34
    development_rmse = [
        result["models"][name]["development"]["rmse_drag_counts"]
        for name in ("no_wave", "active", "development_fit")
    ]
    holdout_rmse = [
        result["models"][name]["holdout"]["rmse_drag_counts"]
        for name in ("no_wave", "active", "development_fit")
    ]
    axes[0, 1].bar(x - width / 2, development_rmse, width, color="#56B4E9", label=r"development $C_L=0.3,0.4$")
    axes[0, 1].bar(x + width / 2, holdout_rmse, width, color="#D55E00", label=r"holdout $C_L=0.5$")
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylabel("RMSE (drag counts)")
    axes[0, 1].set_title("b  Increment-only error", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    lift_grid = np.linspace(0.2, 0.7, 101)
    for key, label, color, style in (
        ("active", "Active κ=0.95", "#0072B2", "-"),
        ("development_fit", "Development-fit κ", "#D55E00", "--"),
    ):
        _, critical = _wave_increment(
            np.zeros_like(lift_grid),
            lift_grid,
            sweep_rad=np.deg2rad(result["aircraft"]["wing_sweep_deg"]),
            thickness_to_chord_ratio=result["aircraft"]["thickness_to_chord_ratio"],
            technology_factor=result["models"][key]["technology_factor"],
        )
        axes[1, 0].plot(lift_grid, critical, color=color, linestyle=style, lw=1.6, label=label)
    axes[1, 0].axhline(result["aircraft"]["cruise_mach"], color="0.3", linestyle=":", lw=1.1, label="A320 cruise Mach")
    axes[1, 0].set_xlabel(r"Lift coefficient, $C_L$")
    axes[1, 0].set_ylabel("Critical Mach")
    axes[1, 0].set_title("c  Geometry-conditioned onset", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    diagnostic = pd.DataFrame(result["representative_cruise_sweep"])
    axes[1, 1].plot(diagnostic["mach"], 1.0e4 * diagnostic["base_drag_coefficient"], color="#000000", linestyle="--", label="subcritical base")
    axes[1, 1].plot(diagnostic["mach"], 1.0e4 * diagnostic["wave_drag_coefficient"], color="#D55E00", linestyle=":", label="wave increment")
    axes[1, 1].plot(diagnostic["mach"], 1.0e4 * diagnostic["total_drag_coefficient"], color="#0072B2", label="active total")
    axes[1, 1].set_xlabel("Mach number")
    axes[1, 1].set_ylabel(r"Drag coefficient, $10^4 C_D$ (counts)")
    axes[1, 1].set_title(r"d  Active model at $C_L=0.55$", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    figure.suptitle("A320 incremental wave-drag audit — base polar held fixed", fontsize=12, fontweight="bold")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_report(result: dict[str, Any], path: Path) -> None:
    active = result["models"]["active"]
    fitted = result["models"]["development_fit"]
    text = f"""# A320 incremental wave-drag audit

This audit isolates wave drag as the published observed coefficient minus the
published subcritical baseline. Aerox's Reynolds-dependent parasite drag and
separate span/profile lift drag are held fixed; neither `CD0` nor `k` is fitted
to the transonic curves.

The active model uses the geometry-conditioned Korn/Gur critical-Mach relation
and Lock's fourth-power increment,

`ΔCDwave = 20 max(M - Mcrit(CL), 0)^4`,

with the stored A320 technology factor `κ={active['technology_factor']:.2f}`.
The wing sweep and thickness ratio come from the aircraft YAML. The resulting
zero-lift critical Mach is `{active['zero_lift_critical_mach']:.4f}` and its
lift slope is `{active['lift_critical_mach_slope']:.4f}`.

## Independent public comparison

`CL=0.3` and `0.4` are development curves; `CL=0.5` is held out. The active
prior has development/holdout RMSE of
`{active['development']['rmse_drag_counts']:.2f}` / `{active['holdout']['rmse_drag_counts']:.2f}`
drag counts. Fitting only the technology factor on the development curves gives
`κ={fitted['technology_factor']:.4f}`, but its held-out RMSE is
`{fitted['holdout']['rmse_drag_counts']:.2f}` counts. That fitted value is not
activated: the source itself flags the A320 polar as anomalous, and its
lift-dependence does not transfer cleanly to the held-out curve.

![A320 incremental wave-drag audit](figures/a320-incremental-wave-drag-audit.png)

Figure 1. Open markers are digitized public increments, solid curves are the
active physical prior, and dashed curves show the development-only fit. Panel d
shows that the low-Mach base remains unchanged while the wave term turns on
continuously near cruise.

## Decision

The compact wave branch is active as a low-confidence physical prior, not an
A320 aerodynamic-deck calibration. It may be replaced by aircraft-specific
wind-tunnel, CFD, certification, or sufficiently observable flight-test
evidence. The public curve is never allowed to rewrite the subcritical model.

Reproduce with `uv run python tools/audit_a320_incremental_wave_drag.py`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.reference)
    required = {
        "lift_coefficient",
        "mach",
        "published_observed_drag_coefficient",
        "published_baseline_drag_coefficient",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError("reference is missing columns: " + ", ".join(sorted(missing)))

    aircraft = load_aircraft("A320")
    active_model = build_drag_model(aircraft)
    sweep_rad = np.deg2rad(float(aircraft.wing.sweep))
    thickness = float(aircraft.wing.t_c)
    technology = float(aircraft.drag.wave_drag_technology_factor)
    observed = (
        frame["published_observed_drag_coefficient"].to_numpy(dtype=float)
        - frame["published_baseline_drag_coefficient"].to_numpy(dtype=float)
    )
    mach = frame["mach"].to_numpy(dtype=float)
    lift = frame["lift_coefficient"].to_numpy(dtype=float)
    development = np.isin(lift, DEVELOPMENT_LIFT)
    holdout = np.isin(lift, HOLDOUT_LIFT)
    fitted_technology = _fit_technology(
        frame,
        observed,
        development,
        sweep_rad=sweep_rad,
        thickness_to_chord_ratio=thickness,
    )
    active_wave, active_critical = _wave_increment(
        mach,
        lift,
        sweep_rad=sweep_rad,
        thickness_to_chord_ratio=thickness,
        technology_factor=technology,
    )
    fitted_wave, fitted_critical = _wave_increment(
        mach,
        lift,
        sweep_rad=sweep_rad,
        thickness_to_chord_ratio=thickness,
        technology_factor=fitted_technology,
    )
    zero_wave = np.zeros_like(observed)
    cosine = np.cos(sweep_rad)

    output_frame = frame[["lift_coefficient", "mach"]].copy()
    output_frame["observed_wave_increment"] = observed
    output_frame["active_wave_increment"] = active_wave
    output_frame["development_fit_wave_increment"] = fitted_wave
    output_frame["active_critical_mach"] = active_critical
    output_frame["development_fit_critical_mach"] = fitted_critical
    output_frame["partition"] = np.where(development, "development", "holdout")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "active-low-confidence-physical-wave-prior",
        "aircraft": {
            "type": "Airbus A320-214",
            "wing_sweep_deg": float(aircraft.wing.sweep),
            "thickness_to_chord_ratio": thickness,
            "cruise_mach": float(aircraft.kinematic.cruise_mach),
        },
        "protocol": {
            "target": "published observed CD minus published subcritical baseline CD",
            "development_lift_coefficients": list(DEVELOPMENT_LIFT),
            "holdout_lift_coefficients": list(HOLDOUT_LIFT),
            "subcritical_aerox_model_refitted": False,
            "lock_fourth_power_coefficient": LOCK_COEFFICIENT,
        },
        "models": {},
    }
    for name, factor, prediction in (
        ("no_wave", None, zero_wave),
        ("active", technology, active_wave),
        ("development_fit", fitted_technology, fitted_wave),
    ):
        model_result: dict[str, Any] = {
            "technology_factor": None if factor is None else float(factor),
            "development": _metrics(observed, prediction, development),
            "holdout": _metrics(observed, prediction, holdout),
            "all": _metrics(observed, prediction, np.ones(len(frame), dtype=bool)),
        }
        if factor is not None:
            model_result["zero_lift_critical_mach"] = float(
                factor / cosine - thickness / cosine**2 - 0.108
            )
            model_result["lift_critical_mach_slope"] = float(0.1 / cosine**3)
        result["models"][name] = model_result

    sweep_mach = np.linspace(0.65, 0.84, 191)
    sweep_condition = DragCondition(
        sweep_mach,
        float(aircraft.drag.reference_unit_reynolds_number_per_m),
    )
    breakdown = active_model.breakdown_at(np.full_like(sweep_mach, 0.55), sweep_condition)
    result["representative_cruise_sweep"] = {
        "mach": sweep_mach.tolist(),
        "base_drag_coefficient": (breakdown.total - breakdown.wave).tolist(),
        "wave_drag_coefficient": breakdown.wave.tolist(),
        "total_drag_coefficient": breakdown.total.tolist(),
    }
    result["artifacts"] = {
        "points": str(args.points.relative_to(PROJECT_ROOT)),
        "report": str(args.report.relative_to(PROJECT_ROOT)),
        "figure_png": str(args.figure_prefix.with_suffix(".png").relative_to(PROJECT_ROOT)),
        "figure_pdf": str(args.figure_prefix.with_suffix(".pdf").relative_to(PROJECT_ROOT)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.points.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.points, index=False)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    _write_report(result, args.report)
    _write_figure(output_frame, result, args.figure_prefix)
    print(
        "active development/holdout RMSE="
        f"{result['models']['active']['development']['rmse_drag_counts']:.3f}/"
        f"{result['models']['active']['holdout']['rmse_drag_counts']:.3f} counts; "
        f"development-fit kappa={fitted_technology:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
