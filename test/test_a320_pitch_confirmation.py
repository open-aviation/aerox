from pathlib import Path

import pandas as pd
import yaml


def test_confirmation_panel_is_new_balanced_and_response_blind():
    root = Path(__file__).parents[1]
    existing = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )
    panel = pd.read_csv(
        root / "docs/modeling/a320-pitch-confirmation-panel.csv",
        dtype={"recording": str},
    )
    with (root / "docs/modeling/a320-pitch-confirmation.yml").open() as stream:
        result = yaml.safe_load(stream)

    existing_keys = set(
        zip(existing["registration"], existing["recording"], strict=True)
    )
    confirmation_keys = set(zip(panel["registration"], panel["recording"], strict=True))
    assert not (existing_keys & confirmation_keys)
    assert panel.groupby("registration").size().to_dict() == {
        "TAIL-A": 3,
        "TAIL-B": 3,
    }
    assert result["method"]["selection_response_blind"]
    assert result["leakage_control"]["panel_written_before_response_evaluation"]
    assert not result["leakage_control"]["model_or_gate_refit_after_panel_selection"]


def test_frozen_pitch_candidate_fails_effect_size_confirmation_gate():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-pitch-confirmation.yml").open() as stream:
        decision = yaml.safe_load(stream)["decision"]
    observed = decision["observed"]
    gate = decision["frozen_gate"]

    assert observed["complete_flights_improved"] == 5
    assert (
        observed["mean_rmse_improvement_over_matched_state"]
        < gate["minimum_mean_rmse_improvement_over_matched_state"]
    )
    assert (
        observed["mean_significant_sign_accuracy_increment"]
        >= gate["minimum_significant_sign_accuracy_increment"]
    )
    assert not decision["independent_confirmation_passed"]
    assert not decision["ready_for_short_window_free_pitch_replay"]
    assert not decision["ready_for_full_flight_free_pitch_replay"]
