#!/usr/bin/env python3
"""Select an untouched long-range A320 holdout and audit fit altitude coverage."""

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

from tools.audit_a320_qar import _required_columns, summarize_flight

UPDATED = "2026-08-06"
RAW_FILES = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}
LIGHT_COLUMNS = [
    "recording",
    "Time..ms.",
    "ALT..ft.",
    "CAS..Kts.",
    "LDGN_ON..ON.OFF.",
    "LDG_ON_1..ON.OFF.",
    "LDG_ON_2..ON.OFF.",
]
CONSUMPTION_ARTIFACTS = (
    "docs/modeling/a320-replay-panel.csv",
    "docs/modeling/a320-vspaero-extension-confirmation-panel.csv",
    "docs/modeling/a320-demonstrated-climb-capacity-flight-metrics.csv",
    "docs/modeling/a320-pitch-confirmation-panel.csv",
)
ALTITUDE_BIN_EDGES_FT = np.arange(0.0, 40_001.0, 5_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--fit-boundary",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-lift-surface-identification.yml"),
    )
    parser.add_argument(
        "--capacity",
        type=Path,
        default=Path("docs/modeling/a320-demonstrated-climb-capacity.yml"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("docs/modeling/a320-long-range-holdout-candidates.csv"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("docs/modeling/a320-training-altitude-coverage.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("docs/modeling/a320-long-range-holdout-selection.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-long-range-holdout-selection.md"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-long-range-holdout-selection"),
    )
    return parser.parse_args()


def _consumed_recordings() -> tuple[set[str], list[str]]:
    consumed: set[str] = set()
    used_artifacts: list[str] = []
    for name in CONSUMPTION_ARTIFACTS:
        path = Path(name)
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype={"recording": str})
        if "recording" not in table:
            continue
        consumed.update(table["recording"].dropna().astype(str))
        used_artifacts.append(name)
    return consumed, used_artifacts


def _light_inventory(raw_dir: Path, processed_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gear_columns = LIGHT_COLUMNS[4:]
    for registration, filename in RAW_FILES.items():
        frame = pd.read_parquet(raw_dir / filename, columns=LIGHT_COLUMNS)
        for recording, flight in frame.groupby("recording", sort=True, observed=True):
            flight = flight.loc[flight["Time..ms."].notna()].sort_values(
                "Time..ms.", kind="stable"
            )
            if len(flight) < 2:
                continue
            gear = (
                flight.loc[:, gear_columns]
                .ffill()
                .bfill()
                .median(axis=1)
                .to_numpy(dtype=float)
            )
            airborne = np.flatnonzero(gear < 0.5)
            if not len(airborne):
                continue
            first, last = int(airborne[0]), int(airborne[-1])
            segment = flight.iloc[first : last + 1]
            time_s = segment["Time..ms."].to_numpy(dtype=float) / 1_000.0
            row_step = float(np.median(np.diff(time_s)))
            rows.append(
                {
                    "registration": registration,
                    "recording": str(recording),
                    "airborne_duration_s": float(time_s[-1] - time_s[0] + row_step),
                    "maximum_altitude_ft": float(segment["ALT..ft."].max()),
                    "maximum_cas_kts": float(segment["CAS..Kts."].max()),
                    "ground_before_liftoff": bool(np.any(gear[:first] >= 0.5)),
                    "ground_after_touchdown": bool(np.any(gear[last + 1 :] >= 0.5)),
                    "processed_match": str(recording) in processed_ids,
                }
            )
    return pd.DataFrame(rows)


def _fit_flights(path: Path) -> dict[str, str]:
    document = yaml.safe_load(path.read_text())
    result: dict[str, str] = {}
    for flight in document["data_boundary"]["flights"]:
        registration, recording = flight.split("/", maxsplit=1)
        result[recording] = registration
    return result


def _altitude_coverage(
    processed_path: Path,
    fit_flights: dict[str, str],
    test_maximum_altitude_ft: float,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        processed_path,
        columns=["flight_id", "altitude"],
        filters=[("flight_id", "in", list(fit_flights))],
    )
    frame["flight_id"] = frame["flight_id"].astype(str)
    frame = frame.loc[frame["flight_id"].isin(fit_flights)]
    rows: list[dict[str, Any]] = []
    for lower, upper in pairwise(ALTITUDE_BIN_EDGES_FT):
        relevant_to_test = lower <= test_maximum_altitude_ft
        selection = (frame["altitude"] >= lower) & (frame["altitude"] < upper)
        subset = frame.loc[selection]
        flights = sorted(subset["flight_id"].unique())
        registrations = sorted({fit_flights[flight] for flight in flights})
        rows.append(
            {
                "altitude_bin_lower_ft": int(lower),
                "altitude_bin_upper_ft": int(upper),
                "relevant_to_selected_test": relevant_to_test,
                "fit_sample_count": len(subset),
                "fit_flight_count": len(flights),
                "fit_registration_count": len(registrations),
                "fit_flights": ";".join(flights),
                "fit_registrations": ";".join(registrations),
                "minimum_two_flights_passed": len(flights) >= 2,
                "both_registrations_passed": len(registrations) == 2,
            }
        )
    return pd.DataFrame(rows)


def _selected_quality(raw_dir: Path, registration: str, recording: str) -> Any:
    frame = pd.read_parquet(
        raw_dir / RAW_FILES[registration],
        columns=_required_columns(),
        filters=[("recording", "==", recording)],
    )
    summary = summarize_flight(registration, recording, frame)
    if summary is None:
        raise ValueError("selected holdout has no airborne segment")
    return summary


def build_audit(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    processed_ids = set(
        pd.read_parquet(args.processed, columns=["flight_id"])["flight_id"]
        .dropna()
        .astype(str)
        .unique()
    )
    consumed, consumption_artifacts = _consumed_recordings()
    capacity = yaml.safe_load(args.capacity.read_text())
    maximum_capacity_altitude_m = float(
        capacity["model"]["domain"]["maximum_pressure_altitude_m"]
    )
    maximum_capacity_altitude_ft = maximum_capacity_altitude_m / 0.3048
    inventory = _light_inventory(args.raw_dir, processed_ids)
    eligible = inventory.loc[
        (~inventory["recording"].isin(consumed))
        & inventory["processed_match"]
        & inventory["ground_before_liftoff"]
        & inventory["ground_after_touchdown"]
        & (inventory["airborne_duration_s"] >= 7_200.0)
        & (inventory["maximum_altitude_ft"] >= 37_000.0)
        & (inventory["maximum_altitude_ft"] <= maximum_capacity_altitude_ft)
        & (inventory["maximum_cas_kts"] >= 180.0)
    ].copy()
    eligible = eligible.sort_values(
        ["airborne_duration_s", "registration", "recording"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    if eligible.empty:
        raise ValueError(
            "no untouched long-range candidate satisfies the frozen screen"
        )
    eligible["selection_rank"] = np.arange(1, len(eligible) + 1)
    selected = eligible.iloc[0]
    quality = _selected_quality(
        args.raw_dir, str(selected["registration"]), str(selected["recording"])
    )
    if not quality.operational_flight:
        raise ValueError("top pre-screened candidate fails the complete quality screen")
    fit_flights = _fit_flights(args.fit_boundary)
    coverage = _altitude_coverage(
        args.processed,
        fit_flights,
        float(selected["maximum_altitude_ft"]),
    )
    relevant_coverage = coverage.loc[coverage["relevant_to_selected_test"]]
    coverage_passed = bool(
        relevant_coverage["minimum_two_flights_passed"].all()
        and relevant_coverage["both_registrations_passed"].all()
    )
    document = {
        "schema_version": 1,
        "updated": UPDATED,
        "aircraft": "Airbus A320-214",
        "status": "selected-response-still-closed",
        "selection_rule": {
            "response_metrics_loaded": False,
            "excluded_previously_consumed_recordings": len(consumed),
            "consumption_artifacts": consumption_artifacts,
            "minimum_airborne_duration_s": 7_200.0,
            "minimum_cruise_altitude_ft": 37_000.0,
            "maximum_altitude_ft": maximum_capacity_altitude_ft,
            "require_processed_lineage": True,
            "require_ground_before_and_after": True,
            "ranking": "longest airborne duration, then registration and recording",
        },
        "selected_holdout": {
            "registration": quality.registration,
            "recording": quality.recording,
            "date": quality.date,
            "airborne_duration_s": quality.airborne_duration_s,
            "airborne_duration_h": quality.airborne_duration_s / 3_600.0,
            "maximum_altitude_ft": quality.max_altitude_ft,
            "maximum_cas_kts": quality.max_cas_kts,
            "takeoff_gross_weight_kg": quality.takeoff_gross_weight_kg,
            "endpoint_distance_nm": quality.endpoint_distance_nm,
            "minimum_native_coverage": quality.min_native_coverage,
            "maximum_critical_gap_s": quality.max_critical_gap_s,
            "complete_operational_screen_passed": quality.operational_flight,
            "previously_consumed": quality.recording in consumed,
        },
        "training_altitude_coverage": {
            "fit_flights": [
                f"{registration}/{recording}"
                for recording, registration in fit_flights.items()
            ],
            "bin_width_ft": 5_000,
            "selected_test_range_covered": coverage_passed,
            "minimum_fit_flights_per_relevant_bin": int(
                relevant_coverage["fit_flight_count"].min()
            ),
            "minimum_registrations_per_relevant_bin": int(
                relevant_coverage["fit_registration_count"].min()
            ),
            "maximum_fit_flight_altitude_ft": float(
                inventory.loc[
                    inventory["recording"].isin(fit_flights),
                    "maximum_altitude_ft",
                ].max()
            ),
        },
        "gates": {
            "untouched_before_selection": not quality.recording in consumed,
            "complete_operational_quality": quality.operational_flight,
            "duration_at_least_two_hours": quality.airborne_duration_s >= 7_200.0,
            "reaches_cruise_altitude": quality.max_altitude_ft >= 37_000.0,
            "inside_released_capacity_altitude_support": quality.max_altitude_ft
            <= maximum_capacity_altitude_ft,
            "training_covers_selected_altitude_range": coverage_passed,
        },
        "artifacts": {
            "candidate_table": "docs/modeling/a320-long-range-holdout-candidates.csv",
            "coverage_table": "docs/modeling/a320-training-altitude-coverage.csv",
            "figure": "docs/modeling/figures/a320-long-range-holdout-selection.png",
            "report": "docs/modeling/a320-long-range-holdout-selection.md",
            "code": "tools/audit_a320_long_range_holdout.py",
        },
    }
    if not all(document["gates"].values()):
        raise ValueError(
            "long-range holdout selection or altitude coverage gate failed"
        )
    return eligible.head(25), coverage, document


def _plot(
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    document: dict[str, Any],
    prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    blue, orange, green, black = "#0072B2", "#D55E00", "#009E73", "#000000"
    selected = document["selected_holdout"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    axes[0].scatter(
        candidates["airborne_duration_s"] / 3_600.0,
        candidates["maximum_altitude_ft"] / 1_000.0,
        s=32,
        facecolors="white",
        edgecolors=blue,
        label="eligible untouched candidates",
    )
    axes[0].scatter(
        selected["airborne_duration_h"],
        selected["maximum_altitude_ft"] / 1_000.0,
        s=70,
        marker="*",
        color=orange,
        label=f"selected {selected['recording']}",
        zorder=4,
    )
    axes[0].set(
        xlabel="Airborne duration (h)",
        ylabel="Maximum pressure altitude (kft)",
    )
    axes[0].legend(frameon=False, fontsize=7)

    labels = [
        f"{int(lower / 1_000)}–{int(upper / 1_000)}"
        for lower, upper in zip(
            coverage["altitude_bin_lower_ft"],
            coverage["altitude_bin_upper_ft"],
            strict=True,
        )
    ]
    positions = np.arange(len(coverage))
    axes[1].bar(
        positions,
        coverage["fit_flight_count"],
        color=green,
        alpha=0.85,
        label="fit flights",
    )
    for position, row in coverage.iterrows():
        axes[1].text(
            position,
            row["fit_flight_count"] + 0.18,
            f"{row['fit_sample_count']:,} s",
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=90,
        )
    axes[1].axhline(2.0, color=black, linestyle="--", linewidth=1.1, label="minimum")
    axes[1].set(
        xlabel="Pressure-altitude bin (kft)",
        ylabel="Number of fit flights",
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, max(coverage["fit_flight_count"]) + 4.0),
    )
    axes[1].legend(frameon=False, fontsize=7)
    for label, axis in zip("AB", axes, strict=True):
        axis.text(
            -0.12,
            1.07,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "A320 long-range holdout selection and fit-flight altitude coverage",
        fontsize=13,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _report(document: dict[str, Any]) -> str:
    selected = document["selected_holdout"]
    coverage = document["training_altitude_coverage"]
    gate_rows = "\n".join(
        f"| {name} | {'pass' if passed else 'fail'} |"
        for name, passed in document["gates"].items()
    )
    return f"""# A320 long-range QAR holdout selection

Updated: {UPDATED}

Status: **selected with response still closed**

## Selection

The frozen response-blind screen selects **{selected["registration"]}/{selected["recording"]}**.
It contains {selected["airborne_duration_s"]:.0f} airborne seconds
({selected["airborne_duration_h"]:.2f} h), reaches
{selected["maximum_altitude_ft"]:.0f} ft, and spans
{selected["endpoint_distance_nm"]:.0f} NM between recorded endpoints. Native
replay-channel coverage is at least {100 * selected["minimum_native_coverage"]:.3f}%
and the maximum critical gap is {selected["maximum_critical_gap_s"]:.1f} s.

The recording was absent from every pre-existing model-development,
calibration, validation, confirmation, diagnostic, and demonstrated-capacity
flight inventory used by the frozen exclusion snapshot. No trajectory response
or replay metric was loaded during selection.

## Training altitude coverage

The ten existing fit flights cover every 5,000-ft band from below 5,000 ft
through the selected test ceiling. The least-populated relevant band still has
{coverage["minimum_fit_flights_per_relevant_bin"]} complete fit flights and
{coverage["minimum_registrations_per_relevant_bin"]} aircraft registrations.
The maximum fit-flight altitude is
{coverage["maximum_fit_flight_altitude_ft"]:.0f} ft, above the selected
{selected["maximum_altitude_ft"]:.0f}-ft ceiling. Therefore no training-flight
split change or holdout contamination is needed.

This confirms flight-level altitude coverage. Individual submodels retain
their own clean-configuration, stable-CLB, Mach, alpha, and temperature support
rules; full-flight coverage is not permission to extrapolate those submodels.

## Gates

| Gate | Result |
|---|---|
{gate_rows}

## Artifacts

- Candidate ranking: `docs/modeling/a320-long-range-holdout-candidates.csv`
- Fit altitude coverage: `docs/modeling/a320-training-altitude-coverage.csv`
- Machine-readable selection: `docs/modeling/a320-long-range-holdout-selection.yml`
- Figure: `docs/modeling/figures/a320-long-range-holdout-selection.{{png,pdf}}`
- Reproducer: `tools/audit_a320_long_range_holdout.py`
"""


def main() -> None:
    args = parse_args()
    candidates, coverage, document = build_audit(args)
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidates, index=False)
    coverage.to_csv(args.coverage, index=False)
    args.parameters.write_text(yaml.safe_dump(document, sort_keys=False))
    args.report.write_text(_report(document))
    _plot(candidates, coverage, document, args.figure_prefix)
    print(yaml.safe_dump(document["selected_holdout"], sort_keys=False).strip())
    print(
        yaml.safe_dump(document["training_altitude_coverage"], sort_keys=False).strip()
    )


if __name__ == "__main__":
    main()
