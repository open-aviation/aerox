from pathlib import Path

import numpy as np

from tools.audit_a320_yaw_rate import (
    _integrate_heading,
    _integrated_horizon,
    _read_bank_parameters,
)

MODELING = Path("docs/modeling")


def test_integrated_horizon_uses_trapezoidal_time_integral():
    values = np.array([0.0, 2.0, 4.0, 6.0])
    time_s = np.array([0.0, 1.0, 2.0, 3.0])

    result = _integrated_horizon(values, time_s, horizon=2)

    assert np.allclose(result, [4.0, 8.0])


def test_heading_integration_preserves_initial_heading_and_rate_units():
    time_s = np.array([0.0, 1.0, 2.0])

    result = _integrate_heading(time_s, np.deg2rad(10.0), np.ones(3))

    assert np.allclose(np.rad2deg(result), [10.0, 11.0, 12.0])


def test_yaw_audit_loads_frozen_effective_bank_parameters():
    result = _read_bank_parameters(MODELING / "a320-pitch-bank-input-calibration.yml")

    assert result.delay_s == 4
    assert np.isclose(result.scale_deg_per_code, 0.8870262184616207)
    assert np.isclose(result.bias_deg, -0.07816915206927486)
