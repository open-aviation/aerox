#!/usr/bin/env python3
"""Select an unused response-blind panel for the A320 VSPAERO extension gate.

The baseline QAR audit selected the first eight flights per registration with a
deterministic medoid-plus-farthest-point rule.  This tool continues that exact
sequence and freezes selections 9--11 before the extended aerodynamic model is
evaluated.  No aerodynamic residual or trajectory-replay result enters the
selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_a320_qar import (
    RAW_AIRCRAFT,
    _processed_recordings,
    audit_raw_file,
    mark_identifier_collisions,
    select_diverse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--processed", type=Path, default=Path("data/a320/a320.parquet")
    )
    parser.add_argument(
        "--baseline-panel",
        type=Path,
        default=Path("docs/modeling/a320-replay-panel.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/modeling/a320-vspaero-extension-confirmation-panel.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_ids = _processed_recordings(args.processed)
    summaries = []
    for registration, filename in RAW_AIRCRAFT.items():
        current, _ = audit_raw_file(registration, args.raw_dir / filename)
        summaries.extend(current)
    for summary in summaries:
        summary.in_processed_table = summary.recording in processed_ids
    mark_identifier_collisions(summaries)

    baseline = pd.read_csv(args.baseline_panel, dtype={"recording": str})
    baseline_keys = set(
        zip(baseline["registration"], baseline["recording"], strict=True)
    )
    rows: list[dict[str, object]] = []
    for registration in RAW_AIRCRAFT:
        pool = [
            item
            for item in summaries
            if item.registration == registration
            and item.operational_flight
            and item.in_processed_table
            and not item.identifier_collision
        ]
        sequence = select_diverse(pool, 11)
        expected_baseline = list(
            baseline.loc[
                baseline["registration"] == registration, "recording"
            ].astype(str)
        )
        observed_baseline = [flight.recording for flight in sequence[:8]]
        if observed_baseline != expected_baseline:
            raise RuntimeError(
                f"baseline selection sequence changed for {registration}: "
                f"{observed_baseline} != {expected_baseline}"
            )
        continuation = sequence[8:11]
        for selection_order, flight in enumerate(continuation, start=9):
            key = (flight.registration, flight.recording)
            if key in baseline_keys:
                raise RuntimeError(f"continued selection unexpectedly reused {key}")
            rows.append(
                {
                    "registration": flight.registration,
                    "recording": flight.recording,
                    "role": "confirmation",
                    "selection_order": selection_order,
                    "date": flight.date,
                    "airborne_duration_s": round(flight.airborne_duration_s, 1),
                    "max_altitude_ft": round(flight.max_altitude_ft, 1),
                    "max_cas_kts": round(flight.max_cas_kts, 3),
                    "takeoff_gross_weight_kg": round(
                        flight.takeoff_gross_weight_kg, 1
                    ),
                    "endpoint_distance_nm": round(flight.endpoint_distance_nm, 2),
                    "mean_wind_speed_raw": round(flight.mean_wind_speed_raw, 3),
                    "surface_activity_raw": round(flight.surface_activity_raw, 3),
                    "min_native_coverage": round(flight.min_native_coverage, 5),
                    "max_critical_gap_s": round(flight.max_critical_gap_s, 3),
                    "processed_match": flight.in_processed_table,
                    "selection_basis": "baseline sequence continuation; no replay response",
                }
            )

    output = pd.DataFrame(rows)
    if len(output) != 6 or output["registration"].value_counts().nunique() != 1:
        raise RuntimeError("confirmation panel must contain three flights per registration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
