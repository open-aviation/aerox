from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.audit_a320_lateral_residual import (
    _intervals_to_nodes,
    _sideslip_target_inventory,
)

MODELING = Path("docs/modeling")


def test_interval_values_are_centered_at_nodes_without_changing_endpoints():
    result = _intervals_to_nodes(np.array([1.0, 3.0, 5.0]))

    assert np.allclose(result, [1.0, 2.0, 4.0, 5.0])


def test_sideslip_target_is_present_but_has_no_panel_excitation():
    panel = pd.read_csv(MODELING / "a320-replay-panel.csv", dtype={"recording": str})

    result = _sideslip_target_inventory(Path("data/a320_raw"), panel)

    assert result["role"] == "target-not-measurement"
    assert result["finite_samples"] == 118_976
    assert result["unique_values"] == [0.0]
    assert result["all_finite_samples_zero"]
    assert not result["usable_for_identification"]


def test_generated_lateral_audit_freezes_structure_before_evaluation():
    audit = yaml.safe_load((MODELING / "a320-lateral-residual-audit.yml").read_text())
    metrics = pd.read_csv(MODELING / "a320-lateral-residual-flight-metrics.csv")
    validation = metrics.loc[metrics["split"] == "validation"]

    assert audit["selection"]["lateral_acceleration_delay_s"] == 3
    assert audit["selection"]["selected_model"] == "lateral_acceleration"
    assert audit["split"]["evaluation_loaded_after_structure_freeze"]
    assert not audit["features"]["direct_sideslip_measurement_available"]
    assert audit["features"]["sideslip_target_inventory"]["all_finite_samples_zero"]
    assert not audit["features"]["sideslip_target_inventory"][
        "usable_for_identification"
    ]
    assert len(metrics) == 16
    assert len(validation) == 6


def test_lateral_acceleration_and_beta_proxy_do_not_pass_release_gates():
    audit = yaml.safe_load((MODELING / "a320-lateral-residual-audit.yml").read_text())
    metrics = pd.read_csv(MODELING / "a320-lateral-residual-flight-metrics.csv")
    validation = metrics.loc[metrics["split"] == "validation"]

    assert (validation["lateral_acceleration_rmse_improvement"] > 0.0).all()
    assert validation["lateral_acceleration_rmse_improvement"].mean() < 0.05
    assert validation["lateral_acceleration_turn_rmse_improvement"].mean() < 0.0
    assert validation["combined_incremental_over_latacc"].mean() < 0.01
    assert not audit["decision"]["lateral_acceleration_explains_residual"]
    assert not audit["decision"]["beta_proxy_adds_predictive_information"]
    assert not audit["decision"]["release_physical_sideslip_or_side_force_model"]
    assert not audit["decision"]["release_lateral_dynamics"]
