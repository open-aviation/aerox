import numpy as np

from aerox.modeling import (
    GuidanceModeThresholds,
    infer_guidance_modes,
)
from aerox.modeling.mode_inference import _persistent


def test_factorial_modes_recover_climb_cruise_descent_behaviour():
    samples = 900
    altitude = np.r_[
        np.linspace(0.0, 30_000.0, 300, endpoint=False),
        np.full(300, 30_000.0),
        np.linspace(30_000.0, 0.0, 300),
    ]
    selected_altitude = np.r_[np.full(600, 30_000.0), np.zeros(300)]
    cas = np.full(samples, 250.0)
    mach = np.r_[np.full(300, 0.55), np.full(300, 0.77), np.full(300, 0.55)]
    n1 = np.r_[np.full(300, 89.0), np.full(300, 72.0), np.full(300, 35.0)]
    bank = np.zeros(samples)
    bank[500:550] = 10.0

    estimate = infer_guidance_modes(
        pressure_altitude_ft=altitude,
        calibrated_airspeed_kts=cas,
        mach=mach,
        selected_altitude_ft=selected_altitude,
        selected_speed_kts=np.full(samples, 250.0),
        selected_mach=np.full(samples, 0.77),
        mean_n1_percent=n1,
        climb_n1_limit_percent=np.full(samples, 90.0),
        heading_deg=np.full(samples, 90.0),
        track_deg=np.full(samples, 90.0),
        selected_heading_deg=np.full(samples, 90.0),
        selected_track_deg=np.zeros(samples),
        bank_angle_deg=bank,
    )

    assert estimate.thrust_mode[150] == "CLB_THRUST_LIKE"
    assert estimate.guidance_family[150] == "CLB_OR_OP_CLB"
    assert estimate.vertical_mode[450] == "ALT_HOLD_LIKE"
    assert estimate.speed_mode[450] == "MACH_HOLD_LIKE"
    assert estimate.speed_target_evidence[450] == "SELECTED_MACH_CONSISTENT"
    assert estimate.lateral_mode[525] == "TURNING"
    assert estimate.thrust_mode[750] == "IDLE_THRUST_LIKE"
    assert estimate.guidance_family[750] == "DES_OR_OP_DES"


def test_persistence_replaces_short_flicker_between_matching_modes():
    labels = np.array(["CLIMB"] * 20 + ["LEVEL"] * 3 + ["CLIMB"] * 20)

    result = _persistent(labels, minimum_dwell=10)

    assert np.all(result == "CLIMB")


def test_threshold_windows_must_be_odd():
    try:
        GuidanceModeThresholds(derivative_window_s=60)
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("even derivative window should be rejected")
