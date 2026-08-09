from pathlib import Path

import pandas as pd
import yaml

from tools.calibrate_a320_speed_autothrust import _select_candidate


def test_speed_controller_selection_uses_simplicity_near_best_rule():
    candidates = pd.DataFrame(
        {
            "candidate": ["persistence", "speed_feedback", "speed_feedback"],
            "response_time_constant_s": [None, 30.0, 45.0],
            "target_rate_feedforward_gain": [None, 0.5, 0.0],
            "nonzero_feedback_terms": [0, 2, 1],
            "mean_flight_rollout_rmse_mps": [10.0, 4.0, 4.03],
        }
    )

    selected = _select_candidate(candidates)

    assert selected["response_time_constant_s"] == 45.0
    assert selected["target_rate_feedforward_gain"] == 0.0


def test_generated_speed_autothrust_artifacts_preserve_frozen_validation():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-speed-autothrust.yml").open() as stream:
        parameters = yaml.safe_load(stream)
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )
    metrics = pd.read_csv(
        root / "docs/modeling/a320-speed-autothrust-flight-metrics.csv"
    )

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert parameters["leakage_control"][
        "structure_and_gains_frozen_before_validation_load"
    ]
    assert set(metrics.loc[metrics["split"] == "validation", "recording"]) == validation
    assert set(metrics.loc[metrics["split"] != "validation", "recording"]).isdisjoint(
        validation
    )
    assert parameters["model"]["mode"] == "variable_speed"
    assert not parameters["decision"]["passed"]
    assert not parameters["decision"]["release_for_coupled_trajectory_test"]
    assert parameters["decision"]["evaluable_validation_flights"] == 3
