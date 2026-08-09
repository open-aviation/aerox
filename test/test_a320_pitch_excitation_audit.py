from pathlib import Path

import pandas as pd
import yaml


def test_pitch_excitation_audit_uses_only_original_fit_flights():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-pitch-excitation-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    confirmation = pd.read_csv(
        root / "docs/modeling/a320-pitch-confirmation-panel.csv",
        dtype={"recording": str},
    )
    lofo = pd.read_csv(
        root / "docs/modeling/a320-pitch-excitation-lofo.csv",
        dtype={"recording": str},
    )

    confirmation_ids = set(confirmation["recording"])
    assert set(lofo["recording"]).isdisjoint(confirmation_ids)
    assert not audit["leakage_control"]["previous_validation_flights_loaded"]
    assert not audit["leakage_control"]["consumed_confirmation_flights_loaded"]
    assert len(audit["leakage_control"]["fit_flights_only"]) == 10


def test_propulsion_nuisance_does_not_rescue_elevator_identification():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-pitch-excitation-audit.yml").open() as stream:
        decision = yaml.safe_load(stream)["decision"]
    observed = decision["observed"]
    gate = decision["gate"]

    assert observed["thrust_coefficient_cv"] < 0.10
    assert (
        observed["thrust_incremental_rmse_improvement"]
        < gate["minimum_incremental_rmse_improvement"]
    )
    assert (
        observed["elevator_incremental_rmse_improvement_after_thrust"]
        < gate["minimum_incremental_rmse_improvement"]
    )
    assert observed["elevator_position_remaining_std_fraction"] > 0.60
    assert observed["elevator_motion_remaining_std_fraction"] > 0.99
    assert observed["residualized_control_condition"] < 4.0
    assert not decision["elevator_after_propulsion_fit_gate_passed"]
    assert not decision["select_revised_effective_pitch_candidate"]
    assert not decision["open_another_confirmation_panel"]
    assert decision["recommend_external_aerodynamic_prior_or_higher_rate_data"]
