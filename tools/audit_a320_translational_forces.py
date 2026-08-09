#!/usr/bin/env python3
"""Audit A320 translational force targets and their identifiability.

This stage intentionally does not fit drag or thrust coefficients.  It uses
only the development/calibration flights, reconstructs effective point-mass
force requirements, and quantifies whether the available propulsion and drag
regressors can be separated before the held-out flights are opened.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox import aero, load_aircraft
from aerox.models.drag import build_reference_drag_polar
from aerox.flightdata.a320_qar import (
    DEG_TO_RAD,
    KNOT_TO_MPS,
    decode_a320_flight,
    raw_columns,
)
from aerox.models.atmosphere import (
    dynamic_pressure_from_calibrated_airspeed,
    mach_from_calibrated_airspeed,
)
from aerox.modeling.identification import delay_samples
from aerox.modeling.force_reconstruction import reconstruct_translational_forces


RAW_FILES = {
    "TAIL-A": "tail-a_2011.parquet",
    "TAIL-B": "tail-b_2011.parquet",
}
PROCESSED_COLUMNS = [
    "flight_id",
    "timestamp",
    "TAS",
    "temperature",
]
DERIVATIVE_WINDOW_S = 21
OBSERVABILITY_THIN_S = 5
MAX_ABSOLUTE_BANK_DEG = 10.0
MIN_TRUE_AIRSPEED_MPS = 70.0
MAX_MACH = 0.85


@dataclass(frozen=True)
class VerticalCalibration:
    pitch_scale_deg_per_code: float
    angle_of_attack_scale_deg_per_code: float
    effective_gamma_bias_deg: float
    delay_s: int


@dataclass(frozen=True)
class FlightAudit:
    metrics: dict[str, Any]
    observability: np.ndarray
    observability_weights: np.ndarray


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
        "--engine-database",
        type=Path,
        default=Path("old/scripts/root/db/engines.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("docs/modeling/a320-translational-force-audit-metrics.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-translational-force-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-translational-force-audit.md"),
    )
    return parser.parse_args()


def _read_vertical_calibration(path: Path) -> VerticalCalibration:
    data = yaml.safe_load(path.read_text())
    parameters = data["model"]["parameters"]
    return VerticalCalibration(
        pitch_scale_deg_per_code=float(
            parameters["pitch_scale_deg_per_code"]["value"]
        ),
        angle_of_attack_scale_deg_per_code=float(
            parameters["angle_of_attack_scale_deg_per_code"]["value"]
        ),
        effective_gamma_bias_deg=float(
            parameters["effective_gamma_bias_deg"]["value"]
        ),
        delay_s=int(data["model"]["trajectory_reference_delay_s"]["value"]),
    )


def _read_tas_scale(path: Path) -> float:
    data = yaml.safe_load(path.read_text())
    return float(data["model"]["parameters"]["tas_scale"]["value"])


def _read_engine(path: Path, name: str) -> dict[str, Any]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {name} row in {path}; found {len(rows)}")
    row = rows[0]
    return {
        "name": name,
        "manufacturer": row["manufacturer"],
        "maximum_sea_level_static_thrust_n_per_engine": float(row["max_thrust"]),
        "source": str(path),
    }


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {
        f"{prefix}_p05": float(q05),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p95": float(q95),
    }


def _dominant_code(values: np.ndarray) -> tuple[float, float, int, int]:
    finite = values[np.isfinite(values)]
    codes, counts = np.unique(finite, return_counts=True)
    dominant = int(np.argmax(counts))
    changes = int(np.count_nonzero(np.diff(finite)))
    return (
        float(codes[dominant]),
        float(counts[dominant] / len(finite)),
        int(len(codes)),
        changes,
    )


def _flight_audit(
    *,
    registration: str,
    recording: str,
    role: str,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    vertical: VerticalCalibration,
    tas_scale: float,
    wing_area_m2: float,
    cd0: float,
    induced_drag_factor: float,
    total_rated_thrust_n: float,
) -> FlightAudit:
    decoded = decode_a320_flight(
        raw.to_dict(orient="list"),
        registration=registration,
        recording=recording,
    )
    current = processed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = current["timestamp"].to_numpy(dtype="datetime64[ns]")
    raw_indices = np.searchsorted(decoded.timestamp_utc, timestamps)
    if np.any(raw_indices >= len(decoded.timestamp_utc)) or not np.all(
        decoded.timestamp_utc[raw_indices] == timestamps
    ):
        raise ValueError(f"raw/processed timestamp mismatch for {registration}/{recording}")

    time_s = (
        timestamps - timestamps[0]
    ).astype("timedelta64[ns]").astype(np.int64) / 1e9
    pitch_code = (
        decoded.channel("pitch", allow_provisional=True)[raw_indices]
        / DEG_TO_RAD
        / 0.5
    )
    angle_of_attack_code = (
        decoded.channel("angle_of_attack", allow_provisional=True)[raw_indices]
        / DEG_TO_RAD
        / 0.25
    )
    gamma_deg = (
        vertical.pitch_scale_deg_per_code
        * delay_samples(pitch_code, vertical.delay_s)
        - vertical.angle_of_attack_scale_deg_per_code
        * delay_samples(angle_of_attack_code, vertical.delay_s)
        + vertical.effective_gamma_bias_deg
    )

    nominal_tas = current["TAS"].to_numpy(dtype=float) * KNOT_TO_MPS
    true_airspeed = tas_scale * nominal_tas
    calibrated_airspeed = decoded.channel("calibrated_airspeed")[raw_indices]
    pressure_altitude = decoded.channel("pressure_altitude")[raw_indices]
    mass = decoded.channel("gross_weight")[raw_indices]
    bank = decoded.channel("roll", allow_provisional=True)[raw_indices]
    flap = decoded.channel("flap", allow_unresolved=True)[raw_indices]
    fuel_flow = (
        decoded.channel("fuel_flow_1")[raw_indices]
        + decoded.channel("fuel_flow_2")[raw_indices]
    )
    fuel_quantity = decoded.channel("fuel_quantity")[raw_indices]
    n1 = 0.5 * (
        decoded.channel("n1_1", allow_provisional=True)[raw_indices]
        + decoded.channel("n1_2", allow_provisional=True)[raw_indices]
    )
    throttle = 0.5 * (
        decoded.channel("throttle_1", allow_unresolved=True)[raw_indices]
        + decoded.channel("throttle_2", allow_unresolved=True)[raw_indices]
    )
    airborne = decoded.airborne[raw_indices]

    # The delay prefix cannot supply gamma.  Trim it rather than silently
    # backfilling future attitude into an unavailable causal interval.
    start = vertical.delay_s
    selection = slice(start, None)
    time_s = time_s[selection]
    gamma = np.deg2rad(gamma_deg[selection])
    true_airspeed = true_airspeed[selection]
    nominal_tas = nominal_tas[selection]
    calibrated_airspeed = calibrated_airspeed[selection]
    pressure_altitude = pressure_altitude[selection]
    mass = mass[selection]
    bank = bank[selection]
    flap = flap[selection]
    fuel_flow = fuel_flow[selection]
    fuel_quantity = fuel_quantity[selection]
    n1 = n1[selection]
    throttle = throttle[selection]
    airborne = airborne[selection]
    processed_temperature = current["temperature"].to_numpy(dtype=float)[selection]

    dynamic_pressure = dynamic_pressure_from_calibrated_airspeed(
        calibrated_airspeed, pressure_altitude
    )
    mach = mach_from_calibrated_airspeed(calibrated_airspeed, pressure_altitude)
    forces = reconstruct_translational_forces(
        time_s=time_s,
        true_airspeed_mps=true_airspeed,
        flight_path_angle_rad=gamma,
        bank_angle_rad=bank,
        mass_kg=mass,
        dynamic_pressure_pa=dynamic_pressure,
        wing_area_m2=wing_area_m2,
        zero_lift_drag_coefficient=cd0,
        induced_drag_factor=induced_drag_factor,
        derivative_window=DERIVATIVE_WINDOW_S,
    )
    clean = (
        np.isfinite(
            forces.net_along_path_force_n
            + forces.required_lift_n
            + forces.prior_implied_thrust_n
        )
        & airborne
        & (np.abs(bank) < np.deg2rad(MAX_ABSOLUTE_BANK_DEG))
        & (true_airspeed > MIN_TRUE_AIRSPEED_MPS)
        & (mach < MAX_MACH)
        & np.isclose(flap, 0.0)
    )
    if np.count_nonzero(clean) < 100:
        raise ValueError(f"too few clean force samples for {registration}/{recording}")

    # ALT is treated as pressure altitude: pressure/Mach come from the OpenAP
    # ISA conversion at dT=0, while the independent processed static
    # temperature affects only speed of sound in this diagnostic TAS recovery.
    tas_from_cas = mach * np.sqrt(aero.gamma * aero.R * processed_temperature)
    tas_difference = nominal_tas - tas_from_cas

    initial_mass = float(mass[0])
    final_mass = float(mass[-1])
    mass_drop = initial_mass - final_mass
    integrated_fuel_flow = float(np.trapezoid(fuel_flow, time_s))
    fuel_quantity_drop = float(fuel_quantity[0] - fuel_quantity[-1])
    dominant_tla, dominant_fraction, unique_tla, tla_changes = _dominant_code(
        throttle[airborne]
    )

    metrics: dict[str, Any] = {
        "registration": registration,
        "recording": recording,
        "role": role,
        "duration_s": float(time_s[-1] - time_s[0]),
        "force_samples": int(np.count_nonzero(clean)),
        "clean_sample_fraction": float(np.mean(clean)),
        "initial_mass_kg": initial_mass,
        "final_mass_kg": final_mass,
        "mass_drop_kg": mass_drop,
        "integrated_fuel_flow_kg": integrated_fuel_flow,
        "fuel_quantity_drop_kg": fuel_quantity_drop,
        "fuel_flow_minus_mass_drop_percent": float(
            100.0 * (integrated_fuel_flow - mass_drop) / mass_drop
        ),
        "fuel_quantity_minus_mass_drop_percent": float(
            100.0 * (fuel_quantity_drop - mass_drop) / mass_drop
        ),
        "processed_tas_minus_openap_cas_tas_bias_mps": float(
            np.mean(tas_difference)
        ),
        "processed_tas_minus_openap_cas_tas_rmse_mps": float(
            np.sqrt(np.mean(tas_difference**2))
        ),
        "tla_dominant_code": dominant_tla,
        "tla_dominant_fraction": dominant_fraction,
        "tla_unique_codes": unique_tla,
        "tla_changes": tla_changes,
        "negative_prior_implied_thrust_fraction": float(
            np.mean(forces.prior_implied_thrust_n[clean] < 0.0)
        ),
        "above_rated_prior_implied_thrust_fraction": float(
            np.mean(forces.prior_implied_thrust_n[clean] > total_rated_thrust_n)
        ),
    }
    metrics.update(_quantiles(mach[clean], "mach"))
    metrics.update(_quantiles(forces.lift_coefficient[clean], "lift_coefficient"))
    metrics.update(
        _quantiles(forces.net_along_path_force_n[clean] / 1000.0, "net_force_kn")
    )
    metrics.update(_quantiles(forces.prior_drag_n[clean] / 1000.0, "prior_drag_kn"))
    metrics.update(
        _quantiles(
            forces.prior_implied_thrust_n[clean] / 1000.0,
            "prior_implied_thrust_kn",
        )
    )
    metrics.update(_quantiles(n1[clean], "mean_n1_percent"))
    metrics.update(_quantiles(fuel_flow[clean], "total_fuel_flow_kg_s"))

    q_s = dynamic_pressure * wing_area_m2
    induced_basis = forces.required_lift_n**2 / q_s
    propulsion_n1_proxy = (n1 / 100.0) ** 2 * (
        np.asarray(aero.pressure(pressure_altitude), dtype=float) / aero.p0
    )
    feature_matrix = np.column_stack(
        (
            propulsion_n1_proxy,
            fuel_flow,
            q_s,
            induced_basis,
            forces.prior_implied_thrust_n,
        )
    )
    thin = clean & (np.arange(len(clean)) % OBSERVABILITY_THIN_S == 0)
    observations = feature_matrix[thin]
    weights = np.full(len(observations), 1.0 / len(observations))
    return FlightAudit(
        metrics=metrics,
        observability=observations,
        observability_weights=weights,
    )


def _weighted_standardized_matrix(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized_weights = weights / np.sum(weights)
    mean = np.sum(normalized_weights[:, None] * matrix, axis=0)
    centered = matrix - mean
    variance = np.sum(normalized_weights[:, None] * centered**2, axis=0)
    standardized = centered / np.sqrt(variance)
    correlation = standardized.T @ (normalized_weights[:, None] * standardized)
    return standardized, correlation


def _design_audit(
    matrix: np.ndarray,
    weights: np.ndarray,
    column_indices: tuple[int, ...],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    design = matrix[:, column_indices]
    standardized, correlation = _weighted_standardized_matrix(design, weights)
    normalized_weights = weights / np.sum(weights)
    singular_values = np.linalg.svd(
        standardized * np.sqrt(normalized_weights[:, None]),
        compute_uv=False,
    )
    return {
        "columns": list(labels),
        "rows": int(len(design)),
        "rank": int(np.linalg.matrix_rank(standardized)),
        "standardized_condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "weighted_correlation": correlation.tolist(),
    }


def _round_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rounded = metrics.copy()
    for column in rounded.select_dtypes(include=["number"]):
        if column in {"force_samples", "tla_unique_codes", "tla_changes"}:
            rounded[column] = rounded[column].astype(int)
        else:
            rounded[column] = rounded[column].round(6)
    return rounded


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    compact = metrics[
        [
            "registration",
            "recording",
            "force_samples",
            "fuel_flow_minus_mass_drop_percent",
            "lift_coefficient_p50",
            "net_force_kn_p50",
            "prior_drag_kn_p50",
            "prior_implied_thrust_kn_p50",
            "tla_dominant_code",
            "tla_dominant_fraction",
        ]
    ].copy()
    compact.columns = [
        "registration",
        "recording",
        "samples",
        "FF-mass (%)",
        "CL p50",
        "T-D p50 (kN)",
        "drag p50 (kN)",
        "thrust p50 (kN)",
        "dominant TLA code",
        "dominant TLA fraction",
    ]
    compact = _round_metrics(compact)
    fuel_error = metrics["fuel_flow_minus_mass_drop_percent"].abs()
    tas_bias = metrics["processed_tas_minus_openap_cas_tas_bias_mps"]
    tas_rmse = metrics["processed_tas_minus_openap_cas_tas_rmse_mps"]
    negative = metrics["negative_prior_implied_thrust_fraction"]
    above_rated = metrics["above_rated_prior_implied_thrust_fraction"]
    n1_design = audit["observability"]["n1_proxy_drag_design"]
    fuel_design = audit["observability"]["fuel_flow_proxy_drag_design"]
    text = f"""# A320 translational-force and observability audit

## Decision

The QAR supports effective **lift** and **net along-path force** targets well
enough to begin a grey-box propulsion stage, but it does **not** identify
thrust and drag independently without priors. The next model should therefore
freeze the repository A320 clean drag polar initially, fit an altitude/Mach/N1
effective thrust map on these ten fit flights, and use fuel flow as a separate
validation signal. `CD0` and `k` must not yet be optimized freely.

No held-out validation flight was loaded by this audit.

## Inputs and fixed priors

- Air data and atmosphere: NumPy port of sibling OpenAP `aero.py`, commit
  `{audit['software']['openap_aero_commit']}`.
- A320 definition: `{audit['aircraft_prior']['source']}`.
- Wing area: {audit['aircraft_prior']['wing_area_m2']:.3f} m².
- Clean polar: `CD0={audit['aircraft_prior']['cd0']:.5f}` and
  `k={audit['aircraft_prior']['induced_drag_factor']:.5f}`.
- Engines: {audit['aircraft_prior']['engine_count']} ×
  {audit['engine_prior']['name']},
  {audit['engine_prior']['maximum_sea_level_static_thrust_n_per_engine'] / 1000:.3f}
  kN per engine in the repository engine table.
- Effective gamma: frozen vertical calibration with a
  {audit['method']['gamma_delay_s']}-second trajectory-reference alignment.
- TAS: processed TAS with the frozen horizontal scale
  {audit['method']['tas_scale']:.9f}.
- Derivatives: centered local-linear {DERIVATIVE_WINDOW_S}-second window.
- Clean samples: airborne, flap code exactly zero, |bank| <
  {MAX_ABSOLUTE_BANK_DEG:.0f}°, TAS > {MIN_TRUE_AIRSPEED_MPS:.0f} m/s, and
  Mach < {MAX_MACH:.2f}.

Dynamic pressure is recovered from QAR CAS and pressure altitude through the
OpenAP compressible air-data conversion. It does not use the processed
temperature field. Processed temperature is used only for a diagnostic
CAS-to-TAS consistency comparison because its upstream lineage is unresolved.

## Force reconstruction

The audit uses the effective point-mass balances

```text
F_along = T_path - D = m (dV/dt + g sin(gamma))
L cos(bank)              = m (V dgamma/dt + g cos(gamma))
CL                       = L / (q S)
D_prior                  = q S (CD0 + k CL²)
T_prior-implied          = F_along + D_prior
```

These are identification targets, not load-cell measurements. `L` neglects
normal thrust and side force, while `T_path` is the thrust component along the
air-relative flight path.

## Fit-flight results

{_markdown_table(compact)}

Across flights, the absolute integrated-fuel-flow versus recorded-mass-drop
error has median {fuel_error.median():.2f}% and maximum {fuel_error.max():.2f}%.
This strongly supports the mass and fuel-flow scales for longitudinal-energy
work. With processed temperature used only for speed of sound, the processed
TAS minus OpenAP CAS/pressure-altitude TAS bias ranges from
{tas_bias.min():.2f} to {tas_bias.max():.2f} m/s; the median per-flight RMSE is
{tas_rmse.median():.2f} m/s. That is suitable as an initial consistency check,
but temperature and air-data uncertainty must remain in the force budget.

The repository-polar implied thrust is negative for a median
{100 * negative.median():.2f}% of selected samples (maximum
{100 * negative.max():.2f}%), primarily in descent/deceleration; this residual
must be retained rather than clipped during later fitting. No flight has a
material above-rated fraction (maximum {100 * above_rated.max():.3f}%).

## Why thrust and drag are not yet separately observable

For every sample,

```text
F_along[i] = T[i] - CD0 (qS)[i] - k (L²/qS)[i].
```

If `T[i]` is free, any pair (`CD0`, `k`) can be matched exactly by redefining
`T[i]`. N1 and fuel flow constrain propulsion only after an engine map is
specified; neither is thrust itself. TLA is even weaker here: it is categorical
and dominated by one code during normal airborne operation.

As a numerical diagnostic—not a physical engine fit—the equal-flight-weighted,
standardized design condition number is
{n1_design['standardized_condition_number']:.2f} for the provisional
N1²×pressure-ratio proxy plus the two drag bases, and
{fuel_design['standardized_condition_number']:.2f} when total fuel flow is used
as propulsion proxy. The complete matrices are in
`a320-translational-force-audit.yml`. Full column rank does not resolve the
structural thrust-map ambiguity.

## Next identifiable step

1. Keep the Aerox A320 clean polar fixed and attach explicit prior uncertainty.
2. Fit an **effective** thrust map using mean N1, pressure ratio, Mach, and
   possibly N1 rate on the ten fit flights. Do not use TLA as continuous percent.
3. Validate predicted fuel flow independently and inspect residuals by flight,
   phase, registration, and derivative window.
4. Freeze that propulsion map, then open the six held-out flights once.
5. Audit FCU/FGC/autothrust command observability before attempting spool
   dynamics; treat TLA as a detent/limit, not continuous thrust command.
6. Only after thrust residuals are stable, test whether limited hierarchical
   corrections to `CD0` and `k` are supported. Fit spool dynamics only where an
   independent command proxy exists; use TLA transitions as limit/mode evidence.

## Limitations

- The pitch/AOA-to-gamma and roll decoders remain provisional effective
  calibrations.
- Flap zero is treated as clean, but the categorical flap code has not been
  fully decoded.
- Vertical wind, thrust-axis inclination, normal thrust, sideslip, and side
  force are omitted.
- The current drag coefficients are repository priors, not QAR estimates.
- OpenAP ISA temperature-deviation behavior is preserved exactly from the
  pinned implementation; the processed weather-temperature source is unknown
  and is used only for the separate TAS consistency diagnostic.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_csv(args.panel, dtype={"recording": str})
    fit_selection = panel.loc[panel["role"] != "validation"].copy()
    if len(fit_selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(fit_selection)}")
    if set(panel.loc[panel["role"] == "validation", "recording"]) & set(
        fit_selection["recording"]
    ):
        raise ValueError("fit and validation recording identifiers overlap")

    selected_ids = fit_selection["recording"].tolist()
    processed = pd.read_parquet(
        args.processed,
        columns=PROCESSED_COLUMNS,
        filters=[("flight_id", "in", selected_ids)],
        engine="pyarrow",
    )
    processed_groups = {
        str(recording): frame
        for recording, frame in processed.groupby("flight_id", sort=False)
    }
    vertical = _read_vertical_calibration(args.vertical_parameters)
    tas_scale = _read_tas_scale(args.horizontal_parameters)
    aircraft = load_aircraft("A320")
    reference_drag = build_reference_drag_polar(aircraft)
    if aircraft.wing is None or aircraft.drag is None or aircraft.engine is None:
        raise ValueError("A320 aircraft definition lacks wing, drag, or engine data")
    if any(
        value is None
        for value in (
            aircraft.wing.area,
            reference_drag.zero_lift_drag_coefficient,
            reference_drag.induced_drag_factor,
            aircraft.engine.number,
            aircraft.engine.default,
        )
    ):
        raise ValueError("A320 aircraft prior has incomplete required values")
    engine = _read_engine(args.engine_database, str(aircraft.engine.default))
    total_rated_thrust = (
        int(aircraft.engine.number)
        * engine["maximum_sea_level_static_thrust_n_per_engine"]
    )

    results: list[FlightAudit] = []
    for registration, registration_selection in fit_selection.groupby(
        "registration", sort=True
    ):
        ids = registration_selection["recording"].tolist()
        raw_path = args.raw_dir / RAW_FILES[str(registration)]
        raw = pd.read_parquet(
            raw_path,
            columns=raw_columns(),
            filters=[("recording", "in", ids)],
            engine="pyarrow",
        )
        role_by_id = registration_selection.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            recording = str(recording)
            results.append(
                _flight_audit(
                    registration=str(registration),
                    recording=recording,
                    role=str(role_by_id[recording]),
                    raw=frame,
                    processed=processed_groups[recording],
                    vertical=vertical,
                    tas_scale=tas_scale,
                    wing_area_m2=float(aircraft.wing.area),
                    cd0=reference_drag.zero_lift_drag_coefficient,
                    induced_drag_factor=reference_drag.induced_drag_factor,
                    total_rated_thrust_n=total_rated_thrust,
                )
            )

    loaded = {result.metrics["recording"] for result in results}
    if loaded != set(selected_ids):
        raise ValueError(f"fit-flight load mismatch: {sorted(set(selected_ids) - loaded)}")
    metrics = pd.DataFrame([result.metrics for result in results]).sort_values(
        ["registration", "recording"], kind="stable"
    )
    matrix = np.vstack([result.observability for result in results])
    weights = np.concatenate(
        [result.observability_weights for result in results]
    )
    observability = {
        "feature_order": [
            "n1_squared_times_pressure_ratio",
            "total_fuel_flow_kg_s",
            "parasite_drag_basis_qS_n",
            "induced_drag_basis_L2_over_qS_n",
            "repository_polar_implied_thrust_n",
        ],
        "n1_proxy_drag_design": _design_audit(
            matrix,
            weights,
            (0, 2, 3),
            (
                "n1_squared_times_pressure_ratio",
                "parasite_drag_basis_qS_n",
                "induced_drag_basis_L2_over_qS_n",
            ),
        ),
        "fuel_flow_proxy_drag_design": _design_audit(
            matrix,
            weights,
            (1, 2, 3),
            (
                "total_fuel_flow_kg_s",
                "parasite_drag_basis_qS_n",
                "induced_drag_basis_L2_over_qS_n",
            ),
        ),
        "proxy_and_implied_thrust_correlation": _design_audit(
            matrix,
            weights,
            (0, 1, 4),
            (
                "n1_squared_times_pressure_ratio",
                "total_fuel_flow_kg_s",
                "repository_polar_implied_thrust_n",
            ),
        ),
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "software": {
            "openap_aero_source": "../openap/openap/aero.py",
            "openap_aero_commit": "472ab3e48d52ae62f9586d2e75e143ea4182a9e9",
            "aerox_aero_port": "src/aerox/aero.py",
            "force_reconstruction": "src/aerox/modeling/force_reconstruction.py",
            "runner": "tools/audit_a320_translational_forces.py",
        },
        "aircraft_prior": {
            "source": "src/aerox/data/aircraft/yml/turbofan/a320.yml",
            "wing_area_m2": float(aircraft.wing.area),
            "cd0": reference_drag.zero_lift_drag_coefficient,
            "induced_drag_factor": reference_drag.induced_drag_factor,
            "engine_count": int(aircraft.engine.number),
            "default_engine": str(aircraft.engine.default),
            "status": "fixed-repository-prior-not-estimated-from-qar",
        },
        "engine_prior": engine
        | {
            "total_rated_thrust_n": total_rated_thrust,
            "status": "repository-database-prior-not-an-in-flight-thrust-limit",
        },
        "method": {
            "fit_flights_only": True,
            "held_out_flights_loaded": False,
            "fit_flights": [
                f"{row.registration}/{row.recording}"
                for row in fit_selection.itertuples(index=False)
            ],
            "gamma_delay_s": vertical.delay_s,
            "tas_scale": tas_scale,
            "derivative_window_s": DERIVATIVE_WINDOW_S,
            "observability_thinning_s": OBSERVABILITY_THIN_S,
            "dynamic_pressure_source": "QAR CAS and pressure altitude through pinned OpenAP aero conversion",
            "selection": {
                "airborne": True,
                "flap_code": 0,
                "maximum_absolute_bank_deg": MAX_ABSOLUTE_BANK_DEG,
                "minimum_true_airspeed_mps": MIN_TRUE_AIRSPEED_MPS,
                "maximum_mach": MAX_MACH,
            },
        },
        "observability": observability,
        "identifiability": {
            "directly_reconstructed_under_assumptions": [
                "effective net along-path non-gravitational force",
                "effective required lift",
                "lift coefficient given fixed wing area",
            ],
            "prior_dependent": [
                "drag",
                "implied thrust",
            ],
            "not_separately_identifiable_without_additional_model_or_prior": [
                "sample thrust versus CD0 and induced drag factor",
                "thrust-axis inclination versus aerodynamic drag",
                "TLA-to-thrust map versus engine spool dynamics",
                "normal thrust versus lift",
            ],
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    _round_metrics(metrics).to_csv(args.metrics, index=False)
    args.audit.write_text(yaml.safe_dump(audit, sort_keys=False))
    _write_report(args.report, metrics, audit)
    return metrics, audit


def main() -> None:
    args = parse_args()
    metrics, audit = run(args)
    print(f"fit flights: {len(metrics)}")
    print(f"force samples: {int(metrics['force_samples'].sum())}")
    print(
        "median |fuel-flow minus mass drop|:",
        f"{metrics['fuel_flow_minus_mass_drop_percent'].abs().median():.2f}%",
    )
    print(
        "N1/drag standardized condition:",
        f"{audit['observability']['n1_proxy_drag_design']['standardized_condition_number']:.2f}",
    )
    print(f"wrote {args.metrics}, {args.audit}, and {args.report}")


if __name__ == "__main__":
    main()
