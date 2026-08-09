from pathlib import Path

import numpy as np
import pandas as pd

from tools.replay_a320_point_mass import (
    _aggregate_metrics,
    _read_fuel_model,
    _read_horizontal_parameters,
    _read_thrust_map,
)

MODELING = Path("docs/modeling")


def test_frozen_replay_adapters_load_documented_models():
    thrust = _read_thrust_map(MODELING / "a320-effective-thrust-calibration.yml")
    fuel = _read_fuel_model(
        MODELING / "a320-effective-thrust-calibration.yml",
        MODELING / "a320-effective-fuel-calibration.yml",
    )
    horizontal = _read_horizontal_parameters(
        MODELING / "a320-horizontal-kinematic-calibration.yml"
    )

    assert thrust.feature_set == "cubic_mach"
    assert thrust.rated_total_thrust_n == 235_800.0
    assert fuel.scale == 0.8932532219831367
    assert horizontal.heading_delay_s == 4
    assert np.isclose(np.rad2deg(horizontal.heading_bias_rad), 0.28771918949784747)


def test_replay_aggregate_keeps_window_and_full_baselines_separate():
    rows = []
    for split, scale in (("calibration", 1.0), ("validation", 2.0)):
        rows.append(
            {
                "split": split,
                "full_success": True,
                "window_speed_rmse_mps": scale,
                "window_altitude_rmse_m": 10.0 * scale,
                "window_heading_rmse_deg": 0.1 * scale,
                "window_horizontal_rmse_m": 100.0 * scale,
                "window_measured_heading_horizontal_rmse_m": 20.0 * scale,
                "full_speed_rmse_mps": 3.0 * scale,
                "full_altitude_rmse_m": 30.0 * scale,
                "full_heading_rmse_deg": 0.3 * scale,
                "full_horizontal_rmse_m": 300.0 * scale,
                "full_measured_heading_horizontal_rmse_m": 40.0 * scale,
            }
        )

    result = _aggregate_metrics(pd.DataFrame(rows)).set_index("split")

    assert result.loc["validation", "300s_horizontal_rmse_m"] == 200.0
    assert result.loc["validation", "300s_measured_heading_baseline_m"] == 40.0
    assert result.loc["validation", "full_horizontal_rmse_m"] == 600.0
    assert result.loc["validation", "full_measured_heading_baseline_m"] == 80.0
