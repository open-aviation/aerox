from pathlib import Path

import pandas as pd
import yaml


def test_generated_mode_audit_preserves_fit_only_boundary_and_rejects_release():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-autothrust-mode-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    inventory = pd.read_csv(
        root / "docs/modeling/a320-autothrust-mode-channel-inventory.csv"
    )
    folds = pd.read_csv(root / "docs/modeling/a320-autothrust-mode-selector-lofo.csv")
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    required = inventory.loc[inventory["category"] == "required-mode"]
    assert len(required) == 12
    assert not required["present_in_both_raw_files"].any()
    assert set(folds["recording"]).isdisjoint(validation)
    assert not audit["leakage_control"]["validation_flights_loaded"]
    assert not audit["leakage_control"]["new_confirmatory_flights_loaded"]
    assert not audit["decision"]["direct_autothrust_mode_observable"]
    assert not audit["decision"]["statistical_selector_gate_passed"]
    assert not audit["decision"]["speed_target_identity_observable"]
    assert not audit["decision"]["release_for_controller_refit"]
    assert not audit["decision"]["open_new_confirmatory_panel"]


def test_published_retard_fallback_is_present_but_inactive_in_screen():
    root = Path(__file__).parents[1]
    inventory = pd.read_csv(
        root / "docs/modeling/a320-autothrust-mode-channel-inventory.csv"
    ).set_index("raw_channel")

    retard = inventory.loc["DMC.1.271.01.18"]
    assert bool(retard["present_in_both_raw_files"])
    assert retard["parameter"] == "retard_displayed_dmc"
    assert retard["one_fraction_on_baseline_screen"] == 0.0
