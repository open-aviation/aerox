from pathlib import Path
from runpy import run_path

import numpy as np

AUDIT = run_path(Path(__file__).parents[1] / "tools" / "audit_rj85_dashlink.py")
_channel_category = AUDIT["_channel_category"]
_centered_pair_correlation = AUDIT["_centered_pair_correlation"]
_parse_archive = AUDIT["_parse_archive"]
_sample_indices = AUDIT["_sample_indices"]
_semantic_matches = AUDIT["_semantic_matches"]


class Channel:
    def __init__(self, data, rate=1):
        self.data = np.asarray(data)
        self.Rate = rate


def test_archive_name_and_three_position_sampling_are_deterministic():
    assert _parse_archive(Path("Tail_652_8.zip")) == ("652", 8)
    assert _sample_indices(1) == [0]
    assert _sample_indices(2) == [0, 1]
    assert _sample_indices(8) == [0, 4, 7]


def test_semantic_search_does_not_confuse_altitude_rate_or_weight_on_wheels():
    inventory = [
        {"parameter": "ALTR", "alpha": "ALTR", "description": "ALTITUDE RATE"},
        {"parameter": "WOW", "alpha": "WOW", "description": "WEIGHT ON WHEELS"},
        {"parameter": "PTCR", "alpha": "PTCR", "description": "PITCH RATE"},
        {"parameter": "GW", "alpha": "GW", "description": "GROSS WEIGHT"},
    ]

    assert _semantic_matches(inventory, AUDIT["BODY_RATE_TERMS"]) == ["PTCR"]
    assert _semantic_matches(inventory, AUDIT["MASS_TERMS"]) == ["GW"]


def test_centered_pair_correlation_uses_airborne_samples_and_preserves_polarity():
    channels = {
        "WOW": Channel([0, 1, 1, 1, 0]),
        "LEFT": Channel([100, 1, 2, 3, -100]),
        "RIGHT": Channel([100, -1, -2, -3, -100]),
    }

    assert np.isclose(_centered_pair_correlation(channels, "LEFT", "RIGHT"), -1.0)


def test_modeling_channel_categories_cover_control_propulsion_and_state():
    assert _channel_category("ELEV_1", "ELEVATOR POSITION LEFT") == "control"
    assert _channel_category("N2_4", "CORE SPEED 4") == "propulsion-or-fuel"
    assert _channel_category("PTCH", "PITCH ANGLE") == "attitude-or-inertial"
