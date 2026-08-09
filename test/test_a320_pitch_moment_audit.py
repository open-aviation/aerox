from pathlib import Path

import pandas as pd
import yaml


def test_generated_pitch_moment_audit_is_fit_only_and_freezes_control_candidate():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-pitch-moment-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )
    lofo = pd.read_csv(root / "docs/modeling/a320-pitch-moment-lofo.csv")

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert set(lofo["recording"]).isdisjoint(validation)
    assert not audit["leakage_control"]["previous_validation_flights_loaded"]
    assert not audit["leakage_control"]["new_confirmatory_flights_loaded"]
    assert audit["decision"]["state_dynamics_fit_gate_passed"]
    assert audit["decision"]["direct_elevator_predictive_gate_passed"]
    assert not audit["decision"]["direct_stabilizer_predictive_gate_passed"]
    assert audit["decision"][
        "ready_to_select_independent_rotational_confirmatory_panel"
    ]
    assert not audit["decision"]["ready_for_open_loop_free_pitch_replay"]
    assert not audit["decision"]["release_physical_pitching_moment_derivatives"]


def test_elevator_position_and_motion_pass_fit_gate_but_remain_nonphysical():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-pitch-moment-audit.yml").open() as stream:
        decision = yaml.safe_load(stream)["decision"]
    observed = decision["observed"]
    gate = decision["screening_gate"]

    assert (
        observed["best_elevator_candidate"] == "state_plus_elevator_position_motion_d0"
    )
    assert observed["elevator_flights_improved_over_state"] == 10
    assert observed["elevator_coefficient_sign_consistency"] == 1.0
    assert observed["elevator_coefficient_cv"] < 0.10
    assert observed["elevator_motion_coefficient_sign_consistency"] == 1.0
    assert observed["elevator_motion_coefficient_cv"] < 0.15
    assert (
        observed["elevator_incremental_rmse_improvement"]
        >= gate["minimum_control_incremental_rmse_improvement"]
    )
    assert observed["stabilizer_incremental_rmse_improvement_beyond_elevator"] < 0
