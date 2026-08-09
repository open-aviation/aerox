from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.replay_a320_observed_heading_rate import (
    _read_yaw_model,
    _trapezoidal_equivalent_zoh,
)

MODELING = Path("docs/modeling")


def test_interval_means_reproduce_trapezoidal_node_rate_integration():
    result = _trapezoidal_equivalent_zoh(np.array([0.0, 2.0, 4.0]))

    assert np.allclose(result, [1.0, 3.0, 4.0])


def test_observed_replay_loads_frozen_yaw_model():
    delay_s, coefficients = _read_yaw_model(MODELING / "a320-yaw-rate-audit.yml")

    assert delay_s == 5
    assert np.allclose(
        coefficients,
        [0.1530221928, 0.0034021082, 1.1483353304, 0.9803762189],
    )


def test_generated_observed_replay_preserves_longitudinal_plant_and_boundary():
    observed = pd.read_csv(MODELING / "a320-observed-heading-rate-replay-metrics.csv")
    physical = pd.read_csv(MODELING / "a320-point-mass-replay-metrics.csv")
    comparison = observed.merge(
        physical,
        on=["registration", "recording"],
        suffixes=("_observed", "_physical"),
        validate="one_to_one",
    )
    parameters = yaml.safe_load(
        (MODELING / "a320-observed-heading-rate-replay.yml").read_text()
    )

    assert len(observed) == 16
    assert observed["full_success"].all()
    assert np.allclose(
        comparison["window_speed_rmse_mps_observed"],
        comparison["window_speed_rmse_mps_physical"],
    )
    assert np.allclose(
        comparison["window_altitude_rmse_m_observed"],
        comparison["window_altitude_rmse_m_physical"],
    )
    assert observed["window_lateral_force_residual_rmse_kn"].gt(0.0).all()
    assert not parameters["decision"]["trajectory_generation_control"]
    assert not parameters["decision"]["body_axis_side_force_identified"]
