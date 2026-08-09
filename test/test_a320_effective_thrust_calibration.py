from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd


CALIBRATION = run_path(
    Path(__file__).parents[1] / "tools" / "calibrate_a320_effective_thrust.py"
)
phase_equal_weights = CALIBRATION["_phase_equal_weights"]
select_candidate = CALIBRATION["_select_candidate"]


def test_phase_equal_weights_equalizes_each_present_phase():
    phase = np.array(["climb", "climb", "climb", "level", "descent", "descent"])

    weights = phase_equal_weights(phase)

    np.testing.assert_allclose(weights.sum(), 1.0)
    for name in np.unique(phase):
        np.testing.assert_allclose(weights[phase == name].sum(), 1.0 / 3.0)


def test_candidate_selection_prefers_simpler_model_within_one_percent():
    candidates = pd.DataFrame(
        [
            {
                "feature_count": 7,
                "lofo_mean_flight_rmse_kn": 10.00,
                "observed_local_monotonic_fraction": 1.0,
                "pressure_exponent": 1.0,
            },
            {
                "feature_count": 5,
                "lofo_mean_flight_rmse_kn": 10.09,
                "observed_local_monotonic_fraction": 1.0,
                "pressure_exponent": 0.75,
            },
            {
                "feature_count": 5,
                "lofo_mean_flight_rmse_kn": 9.0,
                "observed_local_monotonic_fraction": 0.9,
                "pressure_exponent": 0.5,
            },
        ]
    )

    selected = select_candidate(candidates)

    assert selected["feature_count"] == 5
    assert selected["pressure_exponent"] == 0.75
