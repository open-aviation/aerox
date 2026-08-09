from pathlib import Path
from runpy import run_path

import numpy as np

CALIBRATION = run_path(Path(__file__).parents[1] / "tools" / "calibrate_rj85_aoa.py")
AoaFlight = CALIBRATION["AoaFlight"]
_block_mean = CALIBRATION["_block_mean"]
_fit_equal_flight_huber = CALIBRATION["_fit_equal_flight_huber"]
_predict = CALIBRATION["_predict"]
_repeat = CALIBRATION["_repeat"]


def test_rate_alignment_uses_block_means_and_discrete_repetition():
    assert np.array_equal(_block_mean(np.arange(8.0), 8, 4), [0.5, 2.5, 4.5, 6.5])
    assert np.array_equal(_repeat(np.array([0.0, 1.0]), 1, 4), [0, 0, 0, 0, 1, 1, 1, 1])


def test_equal_flight_affine_fit_recovers_known_mapping():
    flights = [
        AoaFlight("652", "a.zip", "a.mat", np.arange(5.0), 4.0 + 0.5 * np.arange(5.0)),
        AoaFlight(
            "687", "b.zip", "b.mat", np.arange(20.0), 4.0 + 0.5 * np.arange(20.0)
        ),
    ]

    coefficient = _fit_equal_flight_huber(flights, "affine")

    assert np.allclose(coefficient, [4.0, 0.5])
    assert np.allclose(
        _predict(flights[0], "affine", coefficient), flights[0].kinematic_aoa_deg
    )
