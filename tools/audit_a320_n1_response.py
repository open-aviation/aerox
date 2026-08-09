#!/usr/bin/env python3
"""Bound observable A320 N1 response around stable TLA-code transitions.

Only the ten development/calibration flights are loaded.  The output separates
repeatable low-power ground transitions from takeoff staging, CL/high-power,
and reverse-code events.  The audit estimates empirical response times; it does
not identify a FADEC command or claim an airborne CFM56 spool constant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aerox.flightdata.a320_qar import decode_a320_flight, raw_columns
from tools.calibrate_a320_effective_thrust import RAW_FILES


MIN_PLATEAU_S = 8
MIN_PRE_PLATEAU_S = 10
MIN_POST_PLATEAU_S = 20
MAX_TLA_TRANSITION_S = 3
PRE_STEADY_WINDOW_S = 5
POST_STEADY_START_S = 10
POST_STEADY_END_S = 20
RESPONSE_HORIZON_AFTER_POST_S = 20
MIN_N1_AMPLITUDE_PERCENT = 5.0
MAX_STEADY_N1_SLOPE_PERCENT_S = 0.25
MAX_OPPOSING_VARIATION_FRACTION = 0.35
MAX_ENDPOINT_ENGINE_DISAGREEMENT_PERCENT = 3.0
MIN_BOUND_EVENTS = 6
MIN_BOUND_FLIGHTS = 4
MIN_BOUND_REGISTRATIONS = 2
EXPECTED_SAMPLE_INTERVAL_S = 1.0


@dataclass(frozen=True)
class Segment:
    """One run of an unchanged TLA code; ``end`` is exclusive."""

    start: int
    end: int
    code: float

    @property
    def samples(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PairSpec:
    source_code: float
    target_code: float
    regime: str
    direction: str
    eligible_for_actuator_bound: bool


PAIR_SPECS = (
    PairSpec(-5.625, 5.625, "low_power_ground", "rise", True),
    PairSpec(5.625, -5.625, "low_power_ground", "fall", True),
    PairSpec(16.875, 67.5, "takeoff_staging_to_high", "rise", False),
    PairSpec(67.5, 45.0, "high_to_dominant_code", "fall", False),
    PairSpec(45.0, 67.5, "dominant_to_high_code", "rise", False),
    PairSpec(-16.875, -5.625, "reverse_code_to_low_code", "fall", False),
    PairSpec(-45.0, -16.875, "high_reverse_to_reverse_code", "fall", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/a320_raw"))
    parser.add_argument(
        "--panel", type=Path, default=Path("docs/modeling/a320-replay-panel.csv")
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("docs/modeling/a320-n1-response-events.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/modeling/a320-n1-response-summary.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/modeling/a320-n1-response-audit.yml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/modeling/a320-n1-response-audit.md"),
    )
    return parser.parse_args()


def _same_code(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    return bool(np.isfinite(left + right) and np.isclose(left, right))


def _run_length_segments(values: np.ndarray) -> list[Segment]:
    """Return contiguous runs without dropping missing-code intervals."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(data) == 0:
        return []
    starts = [0]
    for index in range(1, len(data)):
        if not _same_code(float(data[index - 1]), float(data[index])):
            starts.append(index)
    ends = [*starts[1:], len(data)]
    return [
        Segment(start, end, float(data[start]))
        for start, end in zip(starts, ends, strict=True)
    ]


def _pair_spec(source: float, target: float) -> PairSpec | None:
    for spec in PAIR_SPECS:
        if np.isclose(source, spec.source_code) and np.isclose(
            target, spec.target_code
        ):
            return spec
    return None


def _linear_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(time_s + values)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    return float(np.polyfit(time_s[valid], values[valid], 1)[0])


def _crossing_time(
    time_s: np.ndarray,
    response: np.ndarray,
    initial: float,
    endpoint: float,
    fraction: float,
) -> float:
    direction = np.sign(endpoint - initial)
    if direction == 0 or not 0.0 < fraction < 1.0:
        return float("nan")
    threshold = initial + fraction * (endpoint - initial)
    crossed = np.flatnonzero(
        np.isfinite(response) & (direction * (response - threshold) >= 0.0)
    )
    return float(time_s[crossed[0]]) if len(crossed) else float("nan")


def _opposing_variation_fraction(response: np.ndarray, amplitude: float) -> float:
    if not np.isfinite(amplitude) or amplitude == 0.0:
        return float("inf")
    smoothed = (
        pd.Series(response)
        .rolling(3, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    movement = np.diff(smoothed)
    opposing = np.maximum(0.0, -np.sign(amplitude) * movement)
    return float(np.nansum(opposing) / abs(amplitude))


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _event_row(
    *,
    registration: str,
    recording: str,
    role: str,
    event_index: int,
    time_s: np.ndarray,
    airborne: np.ndarray,
    throttle_1: np.ndarray,
    throttle_2: np.ndarray,
    n1_1: np.ndarray,
    n1_2: np.ndarray,
    source: Segment,
    target: Segment,
    spec: PairSpec,
) -> dict[str, Any]:
    transition_start = source.end
    post_start = target.start
    transition_samples = post_start - transition_start
    pre_slice = slice(
        max(source.start, source.end - PRE_STEADY_WINDOW_S), source.end
    )
    post_slice = slice(
        post_start + POST_STEADY_START_S,
        min(target.end, post_start + POST_STEADY_END_S),
    )
    response_end = min(
        len(time_s), post_start + RESPONSE_HORIZON_AFTER_POST_S + 1
    )
    response_slice = slice(transition_start, response_end)
    engines = np.column_stack((n1_1, n1_2))
    finite_engine_count = np.sum(np.isfinite(engines), axis=1)
    mean_n1 = np.divide(
        np.nansum(engines, axis=1),
        finite_engine_count,
        out=np.full(len(engines), np.nan, dtype=float),
        where=finite_engine_count > 0,
    )
    baseline_engine = np.array(
        [_finite_median(engines[pre_slice, index]) for index in range(2)]
    )
    endpoint_engine = np.array(
        [_finite_median(engines[post_slice, index]) for index in range(2)]
    )
    baseline = _finite_median(baseline_engine)
    endpoint = _finite_median(endpoint_engine)
    amplitude = endpoint - baseline
    response = mean_n1[response_slice]
    relative_time = time_s[response_slice] - time_s[transition_start]
    t10 = _crossing_time(relative_time, response, baseline, endpoint, 0.10)
    t90 = _crossing_time(relative_time, response, baseline, endpoint, 0.90)
    response_10_90 = t90 - t10 if np.isfinite(t10 + t90) else float("nan")
    equivalent_tau = (
        response_10_90 / np.log(9.0)
        if np.isfinite(response_10_90) and response_10_90 >= 0.0
        else float("nan")
    )
    pre_slope = _linear_slope(time_s[pre_slice], mean_n1[pre_slice])
    post_slope = _linear_slope(time_s[post_slice], mean_n1[post_slice])
    engine_disagreement = float(
        np.max(
            (
                abs(baseline_engine[0] - baseline_engine[1]),
                abs(endpoint_engine[0] - endpoint_engine[1]),
            )
        )
        if np.all(np.isfinite(baseline_engine + endpoint_engine))
        else np.nan
    )
    paired_input_slice = slice(max(source.start, source.end - 5), post_slice.stop)
    paired_inputs_finite = bool(
        np.all(
            np.isfinite(
                throttle_1[paired_input_slice] + throttle_2[paired_input_slice]
            )
        )
    )
    ground_window = ~airborne[paired_input_slice]
    fully_ground = bool(len(ground_window) and np.all(ground_window))
    opposing_fraction = _opposing_variation_fraction(response, amplitude)

    reasons: list[str] = []
    if source.samples < MIN_PRE_PLATEAU_S:
        reasons.append("short_pre_plateau")
    if target.samples < MIN_POST_PLATEAU_S:
        reasons.append("short_post_plateau")
    if transition_samples > MAX_TLA_TRANSITION_S:
        reasons.append("slow_or_interrupted_tla_transition")
    if not paired_inputs_finite:
        reasons.append("missing_paired_tla")
    if spec.regime == "low_power_ground" and not fully_ground:
        reasons.append("not_fully_ground")
    if not np.isfinite(amplitude) or abs(amplitude) < MIN_N1_AMPLITUDE_PERCENT:
        reasons.append("small_n1_amplitude")
    if (
        not np.isfinite(pre_slope)
        or abs(pre_slope) > MAX_STEADY_N1_SLOPE_PERCENT_S
    ):
        reasons.append("unstable_pre_n1")
    if (
        not np.isfinite(post_slope)
        or abs(post_slope) > MAX_STEADY_N1_SLOPE_PERCENT_S
    ):
        reasons.append("unstable_endpoint_n1")
    if not np.isfinite(response_10_90) or response_10_90 < 0.0:
        reasons.append("missing_ordered_threshold_crossing")
    if opposing_fraction > MAX_OPPOSING_VARIATION_FRACTION:
        reasons.append("nonmonotone_response")
    if (
        not np.isfinite(engine_disagreement)
        or engine_disagreement > MAX_ENDPOINT_ENGINE_DISAGREEMENT_PERCENT
    ):
        reasons.append("engine_endpoint_disagreement")

    quality_accepted = not reasons
    return {
        "registration": registration,
        "recording": recording,
        "role": role,
        "event_index": event_index,
        "regime": spec.regime,
        "direction": spec.direction,
        "source_tla_code": spec.source_code,
        "target_tla_code": spec.target_code,
        "eligible_for_actuator_bound": spec.eligible_for_actuator_bound,
        "pre_plateau_s": source.samples,
        "tla_transition_s": transition_samples,
        "post_plateau_s": target.samples,
        "airborne_at_transition": bool(airborne[transition_start]),
        "fully_ground_screening_window": fully_ground,
        "baseline_n1_percent": baseline,
        "endpoint_n1_percent": endpoint,
        "n1_amplitude_percent": amplitude,
        "pre_n1_slope_percent_s": pre_slope,
        "endpoint_n1_slope_percent_s": post_slope,
        "time_to_10_percent_s": t10,
        "time_to_90_percent_s": t90,
        "response_10_to_90_s": response_10_90,
        "first_order_equivalent_tau_s": equivalent_tau,
        "opposing_variation_fraction": opposing_fraction,
        "endpoint_engine_disagreement_percent": engine_disagreement,
        "quality_accepted": quality_accepted,
        "used_for_actuator_bound": bool(
            quality_accepted and spec.eligible_for_actuator_bound
        ),
        "rejection_reasons": ";".join(reasons),
    }


def _extract_flight_events(
    *,
    registration: str,
    recording: str,
    role: str,
    time_s: np.ndarray,
    airborne: np.ndarray,
    throttle_1: np.ndarray,
    throttle_2: np.ndarray,
    n1_1: np.ndarray,
    n1_2: np.ndarray,
) -> list[dict[str, Any]]:
    if len(time_s) < 2 or not np.allclose(
        np.diff(time_s), EXPECTED_SAMPLE_INTERVAL_S
    ):
        raise ValueError(f"{registration}/{recording} is not on the 1 Hz grid")
    mean_tla = 0.5 * (throttle_1 + throttle_2)
    segments = _run_length_segments(mean_tla)
    stable_indices = [
        index
        for index, segment in enumerate(segments)
        if np.isfinite(segment.code) and segment.samples >= MIN_PLATEAU_S
    ]
    rows: list[dict[str, Any]] = []
    for source_index, target_index in zip(
        stable_indices[:-1], stable_indices[1:], strict=True
    ):
        source = segments[source_index]
        target = segments[target_index]
        spec = _pair_spec(source.code, target.code)
        if spec is None:
            continue
        rows.append(
            _event_row(
                registration=registration,
                recording=recording,
                role=role,
                event_index=len(rows) + 1,
                time_s=time_s,
                airborne=airborne,
                throttle_1=throttle_1,
                throttle_2=throttle_2,
                n1_1=n1_1,
                n1_2=n1_2,
                source=source,
                target=target,
                spec=spec,
            )
        )
    return rows


def _load_fit_events(raw_dir: Path, panel_path: Path) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(panel_path, dtype={"recording": str})
    selection = panel.loc[panel["role"] != "validation"].copy()
    if len(selection) != 10:
        raise ValueError(f"expected ten fit flights; found {len(selection)}")
    rows: list[dict[str, Any]] = []
    loaded: list[str] = []
    for registration, current in selection.groupby("registration", sort=True):
        raw = pd.read_parquet(
            raw_dir / RAW_FILES[str(registration)],
            columns=raw_columns(),
            filters=[("recording", "in", current["recording"].tolist())],
            engine="pyarrow",
        )
        roles = current.set_index("recording")["role"].to_dict()
        for recording, frame in raw.groupby("recording", sort=True):
            decoded = decode_a320_flight(
                frame.to_dict(orient="list"),
                registration=str(registration),
                recording=str(recording),
            )
            rows.extend(
                _extract_flight_events(
                    registration=str(registration),
                    recording=str(recording),
                    role=str(roles[str(recording)]),
                    time_s=decoded.time_s,
                    airborne=decoded.airborne,
                    throttle_1=decoded.channel(
                        "throttle_1", allow_unresolved=True
                    ),
                    throttle_2=decoded.channel(
                        "throttle_2", allow_unresolved=True
                    ),
                    n1_1=decoded.channel("n1_1", allow_provisional=True),
                    n1_2=decoded.channel("n1_2", allow_provisional=True),
                )
            )
            loaded.append(f"{registration}/{recording}")
    expected = {
        f"{row.registration}/{row.recording}"
        for row in selection.itertuples(index=False)
    }
    if set(loaded) != expected:
        raise ValueError(f"fit-flight load mismatch: {sorted(expected - set(loaded))}")
    return pd.DataFrame(rows), sorted(loaded)


def _summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in PAIR_SPECS:
        group = events.loc[
            np.isclose(events["source_tla_code"], spec.source_code)
            & np.isclose(events["target_tla_code"], spec.target_code)
        ]
        accepted = group.loc[group["quality_accepted"]]
        flights = accepted[["registration", "recording"]].drop_duplicates()
        passed = (
            len(accepted) >= MIN_BOUND_EVENTS
            and len(flights) >= MIN_BOUND_FLIGHTS
            and accepted["registration"].nunique() >= MIN_BOUND_REGISTRATIONS
        )
        rows.append(
            {
                "regime": spec.regime,
                "direction": spec.direction,
                "source_tla_code": spec.source_code,
                "target_tla_code": spec.target_code,
                "candidate_events": len(group),
                "quality_accepted_events": len(accepted),
                "accepted_flights": len(flights),
                "accepted_registrations": accepted["registration"].nunique(),
                "median_response_10_to_90_s": accepted[
                    "response_10_to_90_s"
                ].median(),
                "minimum_response_10_to_90_s": accepted[
                    "response_10_to_90_s"
                ].min(),
                "maximum_response_10_to_90_s": accepted[
                    "response_10_to_90_s"
                ].max(),
                "repeatability_gate_passed": bool(passed),
                "selected_for_actuator_bound": bool(
                    passed and spec.eligible_for_actuator_bound
                ),
            }
        )
    return pd.DataFrame(rows)


def _direction_bound(events: pd.DataFrame, direction: str) -> dict[str, Any]:
    selected = events.loc[
        events["used_for_actuator_bound"] & (events["direction"] == direction)
    ]
    flights = selected[["registration", "recording"]].drop_duplicates()
    passed = (
        len(selected) >= MIN_BOUND_EVENTS
        and len(flights) >= MIN_BOUND_FLIGHTS
        and selected["registration"].nunique() >= MIN_BOUND_REGISTRATIONS
    )
    if not passed:
        return {
            "passed": False,
            "events": len(selected),
            "flights": len(flights),
            "registrations": selected["registration"].nunique(),
        }
    response = selected["response_10_to_90_s"].to_numpy(dtype=float)
    expanded_lower = max(0.0, float(np.min(response) - EXPECTED_SAMPLE_INTERVAL_S))
    expanded_upper = float(np.max(response) + EXPECTED_SAMPLE_INTERVAL_S)
    return {
        "passed": True,
        "events": len(selected),
        "flights": len(flights),
        "registrations": selected["registration"].nunique(),
        "observed_response_10_to_90_s": {
            "minimum": float(np.min(response)),
            "median": float(np.median(response)),
            "maximum": float(np.max(response)),
        },
        "sampling_expanded_response_10_to_90_s": {
            "lower": expanded_lower,
            "upper": expanded_upper,
        },
        "sampling_expanded_first_order_equivalent_tau_s": {
            "lower": expanded_lower / np.log(9.0),
            "upper": expanded_upper / np.log(9.0),
        },
        "interpretation": (
            "effective low-power ground code-pair response bound; not an "
            "airborne engine-spool confidence interval"
        ),
    }


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    integer_columns = {
        "event_index",
        "pre_plateau_s",
        "tla_transition_s",
        "post_plateau_s",
        "candidate_events",
        "quality_accepted_events",
        "accepted_flights",
        "accepted_registrations",
    }
    for column in output.select_dtypes(include=["number"]):
        if column in integer_columns:
            output[column] = output[column].astype(int)
        else:
            output[column] = output[column].round(8)
    return output


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    strings = frame.astype(str)
    headers = list(strings.columns)
    rows = strings.values.tolist()
    widths = [
        max(len(str(headers[index])), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(headers))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(row))
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    compact = _round_frame(
        summary[
            [
                "regime",
                "direction",
                "candidate_events",
                "quality_accepted_events",
                "accepted_flights",
                "accepted_registrations",
                "median_response_10_to_90_s",
                "repeatability_gate_passed",
                "selected_for_actuator_bound",
            ]
        ].copy()
    )
    compact.columns = [
        "regime",
        "dir",
        "candidate",
        "accepted",
        "flights",
        "regs",
        "median 10-90 s",
        "repeatable",
        "selected",
    ]
    bound_rows = []
    for direction, bound in audit["low_power_ground_bounds"].items():
        if not bound["passed"]:
            continue
        observed = bound["observed_response_10_to_90_s"]
        expanded = bound["sampling_expanded_response_10_to_90_s"]
        tau = bound["sampling_expanded_first_order_equivalent_tau_s"]
        bound_rows.append(
            {
                "direction": direction,
                "events": bound["events"],
                "flights": bound["flights"],
                "observed min/median/max s": (
                    f"{observed['minimum']:.0f}/{observed['median']:.0f}/"
                    f"{observed['maximum']:.0f}"
                ),
                "expanded 10-90 bound s": (
                    f"{expanded['lower']:.0f}--{expanded['upper']:.0f}"
                ),
                "equivalent tau bound s": (
                    f"{tau['lower']:.2f}--{tau['upper']:.2f}"
                ),
            }
        )
    bounds_table = _markdown_table(pd.DataFrame(bound_rows))
    accepted_low = events.loc[events["used_for_actuator_bound"]]
    text = f"""# A320 empirical N1 response audit

## Decision

The fit-flight QAR supports repeatable **low-power ground** N1 response bounds
for the raw TLA-code pairs `-5.625 -> 5.625` and `5.625 -> -5.625`. It does not
support an airborne CFM56-5B4 spool constant. The takeoff input is staged or
ramped, the dominant-code/high-power changes are too small or sparse, and the
actual FADEC/autothrust command remains unrecorded.

Only the ten development/calibration flights are loaded. No validation flight
is opened. Raw TLA meanings and the N1 scale remain unresolved/provisional,
respectively.

## Screen

Stable TLA plateaus are separated by at most {MAX_TLA_TRANSITION_S} seconds.
Accepted events require at least {MIN_PRE_PLATEAU_S} seconds before and
{MIN_POST_PLATEAU_S} seconds after the change, |ΔN1| ≥
{MIN_N1_AMPLITUDE_PERCENT:.0f} percentage points, pre/post drift ≤
{MAX_STEADY_N1_SLOPE_PERCENT_S:.2f} percentage points/s, ordered 10% and 90%
crossings, limited opposing motion, and paired-engine endpoint agreement. The
low-power pair must remain on the ground throughout its screening window.

These deterministic rules were selected during exploratory fit-flight
development and must be frozen before any future validation use. They are not
pre-registered or externally validated.

{_markdown_table(compact)}

## Observable low-power bounds

{bounds_table}

The selected set contains {len(accepted_low)} events. The 1 Hz sampling-expanded
bounds add one second on each side of the observed event range. For comparison
with the implemented first-order actuator only, `tau = t10-90 / ln(9)`. This
conversion does not remove lever motion, FADEC dynamics, recorder latency, or
ground-operating effects, so these are scenario bounds rather than identified
engine parameters or confidence intervals.

## Why no airborne parameter is released

Takeoff commonly holds raw code `16.875` while N1 is already rising, then moves
through one or more short intermediate codes to `67.5`. Most such events fail
the steady-initial-state gate. The few quality-accepted high-power reductions
do not span both registrations and do not meet the event/flight gate. Changes
inside raw code `45` cannot be analyzed because their command is not recorded.

Therefore:

- the actuator API continues to require explicit rise and fall constants;
- the bounds above may be used only for low-power-ground sensitivity cases;
- no default A320 airborne constant is added; and
- recorded N1 remains the propulsion-state target for future controller tests.

A subsequent fit-only audit in `a320-n2-observability-audit.md` supports
retaining recorded N2 as a second effective propulsion state. It does not
release coupled N1/N2 actuator coefficients or change the airborne-constant
decision above.

## Artifacts and limitations

Per-event accepted and rejected evidence is in
`a320-n1-response-events.csv`; grouped evidence is in
`a320-n1-response-summary.csv`; the machine-readable decision is in
`a320-n1-response-audit.yml`.

The audit is limited by one-hertz sampling, unresolved TLA semantics, provisional
N1 scaling, lever/FADEC/engine convolution, ground-only selected events, small
sample sizes, and routine operational inputs rather than designed excitation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    events, fit_flights = _load_fit_events(args.raw_dir, args.panel)
    if events.empty:
        raise ValueError("no candidate TLA-code response events found")
    summary = _summary(events)
    bounds = {
        direction: _direction_bound(events, direction)
        for direction in ("rise", "fall")
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "updated": "2026-08-04",
        "aircraft": "Airbus A320-214",
        "status": "low-power-ground-bounded-airborne-unidentified",
        "leakage_control": {
            "fit_flights_only": fit_flights,
            "validation_flights_loaded": False,
        },
        "channel_status": {
            "tla": "unresolved categorical recorder code",
            "n1": "provisional value-times-0.5 percent conversion",
        },
        "method": {
            "sample_interval_s": EXPECTED_SAMPLE_INTERVAL_S,
            "minimum_plateau_s": MIN_PLATEAU_S,
            "minimum_pre_plateau_s": MIN_PRE_PLATEAU_S,
            "minimum_post_plateau_s": MIN_POST_PLATEAU_S,
            "maximum_tla_transition_s": MAX_TLA_TRANSITION_S,
            "pre_steady_window_s": PRE_STEADY_WINDOW_S,
            "post_steady_window_s": [
                POST_STEADY_START_S,
                POST_STEADY_END_S,
            ],
            "minimum_n1_amplitude_percent": MIN_N1_AMPLITUDE_PERCENT,
            "maximum_steady_n1_slope_percent_s": (
                MAX_STEADY_N1_SLOPE_PERCENT_S
            ),
            "maximum_opposing_variation_fraction": (
                MAX_OPPOSING_VARIATION_FRACTION
            ),
            "maximum_endpoint_engine_disagreement_percent": (
                MAX_ENDPOINT_ENGINE_DISAGREEMENT_PERCENT
            ),
            "repeatability_gate": {
                "minimum_events": MIN_BOUND_EVENTS,
                "minimum_complete_flights": MIN_BOUND_FLIGHTS,
                "minimum_registrations": MIN_BOUND_REGISTRATIONS,
            },
            "first_order_equivalent_relation": "tau = t10_to_90 / ln(9)",
            "sampling_expansion_s_each_side": EXPECTED_SAMPLE_INTERVAL_S,
        },
        "low_power_ground_bounds": bounds,
        "airborne_actuator_decision": {
            "nominal_time_constants_available": False,
            "reason": (
                "no recorded continuous command; accepted high-power events "
                "fail the complete-flight and registration repeatability gate"
            ),
            "model_default_changed": False,
        },
        "subsequent_n2_audit": {
            "record": "docs/modeling/a320-n2-observability-audit.yml",
            "n2_dynamic_state_supported": True,
            "coupled_two_spool_parameters_released": False,
        },
        "artifacts": {
            "runner": "tools/audit_a320_n1_response.py",
            "events": str(args.events),
            "summary": str(args.summary),
            "report": str(args.report),
        },
    }
    args.events.parent.mkdir(parents=True, exist_ok=True)
    _round_frame(events).to_csv(args.events, index=False)
    _round_frame(summary).to_csv(args.summary, index=False)
    args.audit.write_text(
        yaml.safe_dump(_native(audit), sort_keys=False, allow_unicode=True)
    )
    _write_report(
        args.report,
        events=events,
        summary=summary,
        audit=audit,
    )
    return events, summary, audit


if __name__ == "__main__":
    run(parse_args())
