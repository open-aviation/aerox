from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.audit_a320_autothrust_envelope import (
    _select_candidate,
)


def test_envelope_candidate_selection_prefers_simplest_near_best_model():
    candidates = pd.DataFrame(
        {
            "feature_set": ["simple", "complex", "crossing"],
            "feature_count": [2, 5, 1],
            "lofo_mean_flight_boundary_rmse_percent": [1.01, 1.0, 0.5],
            "lofo_climb_mean_rmse_percent": [1.01, 1.0, 0.5],
            "minimum_full_fit_envelope_width_percent": [40.0, 40.0, -1.0],
        }
    )

    selected = _select_candidate(candidates)

    assert selected["feature_set"] == "simple"


def test_generated_autothrust_envelope_is_frozen_before_validation():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-autothrust-envelope.yml").open() as stream:
        audit = yaml.safe_load(stream)
    points = pd.read_csv(root / "docs/modeling/a320-autothrust-envelope-points.csv")
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert audit["leakage_control"]["structure_frozen_before_validation_load"]
    assert set(points.loc[points["split"] == "fit", "recording"]).isdisjoint(validation)
    assert set(points.loc[points["split"] == "validation", "recording"]) == validation
    assert audit["model"]["modes"] == [
        "fixed_idle",
        "fixed_climb",
        "variable_speed",
    ]
    assert not audit["high_power_limit_decision"]["model_fitted"]
    assert not audit["continuous_controller_decision"]["model_fitted"]
    assert audit["clb_limit_decision"]["passed"]
    assert not audit["idle_limit_decision"]["passed"]
    assert np.isfinite(
        audit["aggregate_results"]["validation_median_climb_boundary_rmse_percent"]
    )
