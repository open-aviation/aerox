from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.audit_a320_n1_response import (
    _direction_bound,
    _extract_flight_events,
    _run_length_segments,
)


def test_run_length_segments_preserves_missing_intervals():
    values = np.array([1.0, 1.0, np.nan, np.nan, 2.0, 2.0, 2.0])

    segments = _run_length_segments(values)

    assert [(item.start, item.end) for item in segments] == [(0, 2), (2, 4), (4, 7)]
    assert np.isnan(segments[1].code)


def test_clean_low_power_step_is_selected_for_response_bound():
    time = np.arange(60.0)
    throttle = np.r_[np.full(30, -5.625), np.full(30, 5.625)]
    n1 = np.full(60, 20.0)
    n1[30:] = 80.0 + (20.0 - 80.0) * np.exp(-np.arange(30) / 2.0)

    events = _extract_flight_events(
        registration="TEST-A",
        recording="REC1",
        role="calibration",
        time_s=time,
        airborne=np.zeros(60, dtype=bool),
        throttle_1=throttle,
        throttle_2=throttle,
        n1_1=n1,
        n1_2=n1,
    )

    assert len(events) == 1
    event = events[0]
    assert event["regime"] == "low_power_ground"
    assert event["direction"] == "rise"
    assert event["quality_accepted"]
    assert event["used_for_actuator_bound"]
    assert event["response_10_to_90_s"] == 4.0


def test_repeatable_bound_requires_flights_and_both_registrations():
    events = pd.DataFrame(
        {
            "used_for_actuator_bound": [True] * 6,
            "direction": ["rise"] * 6,
            "registration": ["A", "A", "A", "B", "B", "B"],
            "recording": ["1", "1", "2", "3", "4", "4"],
            "response_10_to_90_s": [3.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        }
    )

    bound = _direction_bound(events, "rise")

    assert bound["passed"]
    assert bound["events"] == 6
    assert bound["flights"] == 4
    assert bound["sampling_expanded_response_10_to_90_s"] == {
        "lower": 2.0,
        "upper": 10.0,
    }


def test_bound_gate_rejects_single_registration():
    events = pd.DataFrame(
        {
            "used_for_actuator_bound": [True] * 6,
            "direction": ["fall"] * 6,
            "registration": ["A"] * 6,
            "recording": ["1", "1", "2", "3", "4", "4"],
            "response_10_to_90_s": [4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        }
    )

    assert not _direction_bound(events, "fall")["passed"]


def test_generated_response_artifacts_preserve_fit_only_scope():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-n1-response-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    events = pd.read_csv(root / "docs/modeling/a320-n1-response-events.csv")
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv",
        dtype={"recording": str},
    )

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert not audit["leakage_control"]["validation_flights_loaded"]
    assert validation.isdisjoint(set(events["recording"]))
    assert not audit["airborne_actuator_decision"][
        "nominal_time_constants_available"
    ]
    for direction in ("rise", "fall"):
        bound = audit["low_power_ground_bounds"][direction]
        selected = events.loc[
            events["used_for_actuator_bound"]
            & (events["direction"] == direction)
        ]
        assert bound["events"] == len(selected)
