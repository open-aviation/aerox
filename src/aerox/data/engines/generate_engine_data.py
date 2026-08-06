#!/usr/bin/env python3
"""Generate Aerox's queryable engine table from the official ICAO EEDB.

Adapted from OpenAP's ``scripts/gen_engine_data_v27.py``.  The source columns
are addressed by their published Issue 32 names instead of spreadsheet column
positions.  Run from the repository root with::

    uv run --with openpyxl python src/aerox/data/engines/generate_engine_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "icao-edb-issue-32.xlsx"
DEFAULT_CRUISE = HERE / "engine_cruise_performance.csv"
DEFAULT_OUTPUT = HERE / "engines.csv"

COLUMNS = {
    "UID No": "uid",
    "Engine Identification": "name",
    "Manufacturer": "manufacturer",
    "Eng Type": "type",
    "B/P Ratio": "bpr",
    "Pressure Ratio": "pr",
    "Rated Thrust (kN)": "max_thrust",
    "Data Superseded": "superseded",
    "Superseded by UID No": "superseded_by",
    "Current Engine Status": "current_engine_status",
    "Current Engine Status Date": "current_engine_status_date",
    "HC EI T/O (g/kg)": "ei_hc_to",
    "HC EI C/O (g/kg)": "ei_hc_co",
    "HC EI App (g/kg)": "ei_hc_app",
    "HC EI Idle (g/kg)": "ei_hc_idl",
    "CO EI T/O (g/kg)": "ei_co_to",
    "CO EI C/O (g/kg)": "ei_co_co",
    "CO EI App (g/kg)": "ei_co_app",
    "CO EI Idle (g/kg)": "ei_co_idl",
    "NOx EI T/O (g/kg)": "ei_nox_to",
    "NOx EI C/O (g/kg)": "ei_nox_co",
    "NOx EI App (g/kg)": "ei_nox_app",
    "NOx EI Idle (g/kg)": "ei_nox_idl",
    "Fuel Flow T/O (kg/sec)": "ff_to",
    "Fuel Flow C/O (kg/sec)": "ff_co",
    "Fuel Flow App (kg/sec)": "ff_app",
    "Fuel Flow Idle (kg/sec)": "ff_idl",
    "Fuel LTO Cycle (kg)  ": "fuel_lto",
}

REQUIRED_NUMERIC = [
    "bpr", "max_thrust",
    "ei_hc_to", "ei_hc_co", "ei_hc_app", "ei_hc_idl",
    "ei_co_to", "ei_co_co", "ei_co_app", "ei_co_idl",
    "ei_nox_to", "ei_nox_co", "ei_nox_app", "ei_nox_idl",
    "ff_to", "ff_co", "ff_app", "ff_idl", "fuel_lto",
]


def clean_name(value: object) -> str:
    """Retain OpenAP's stable engine identifier cleaning convention."""

    name = str(value).strip()
    for marker in ("SelectOne", "Build Spec", "Block", "series"):
        if marker in name:
            name = name.split()[0].strip()
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    return name


def fuel_shift_curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (x + b) ** 2


def load_edb(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Gaseous Emissions and Smoke")
    missing = sorted(set(COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"ICAO EEDB columns changed; missing: {missing}")
    data = raw[list(COLUMNS)].rename(columns=COLUMNS).copy()
    data = data.dropna(subset=["uid", "name"])
    data["uid"] = data["uid"].astype(str).str.strip()
    data["name"] = data["name"].map(clean_name)
    data = data[data["superseded"].fillna("").astype(str).str.lower() != "yes"]

    for column in REQUIRED_NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=REQUIRED_NUMERIC)
    data["max_thrust"] = (data["max_thrust"] * 1000).round().astype(int)
    data = data.drop_duplicates(subset=["name"], keep="last")
    return data


def add_fuel_coefficients(data: pd.DataFrame) -> pd.DataFrame:
    throttle = np.array([0.07, 0.30, 0.85, 1.00], dtype=float)
    data = data.copy()
    for index, row in data.iterrows():
        fuel_flow = np.array(
            [row["ff_idl"], row["ff_app"], row["ff_co"], row["ff_to"]],
            dtype=float,
        )
        cubic = np.polyfit(throttle, fuel_flow, 3)
        data.loc[index, ["fuel_c3", "fuel_c2", "fuel_c1"]] = cubic[:3]
        shifted, _ = curve_fit(fuel_shift_curve, throttle, fuel_flow)
        data.loc[index, ["fuel_a", "fuel_b"]] = shifted
    return data


def merge_cruise(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    cruise = pd.read_csv(path)
    result = data.merge(cruise, how="left", left_on="name", right_on="engine")
    return result.drop(columns=["engine"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cruise", type=Path, default=DEFAULT_CRUISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = merge_cruise(add_fuel_coefficients(load_edb(args.input)), args.cruise)
    rendered = data.to_csv(index=False, float_format="%g")
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            print(f"stale generated engine table: {args.output}")
            return 1
        print(f"engine table is current: {len(data)} identifiers")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(data)} ICAO EEDB engine identifiers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
