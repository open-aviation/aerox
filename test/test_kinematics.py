import numpy as np

from aerox.modeling.dynamics.kinematics import (
    air_velocity_from_orientation,
    differentiate_wgs84_intervals,
    integrate_wgs84_intervals,
    replay_metrics,
)


def test_orientation_rotation_uses_north_east_down_aerospace_convention():
    speed = np.array([100.0, 100.0, 100.0])
    zero = np.zeros(3)
    heading = np.deg2rad([0.0, 90.0, 0.0])
    pitch = np.deg2rad([0.0, 0.0, 10.0])
    alpha = np.deg2rad([0.0, 0.0, 10.0])

    north, east, down = air_velocity_from_orientation(
        speed, alpha, zero, zero, pitch, heading
    )

    np.testing.assert_allclose(north, [100.0, 0.0, 100.0], atol=1e-12)
    np.testing.assert_allclose(east, [0.0, 100.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(down, [0.0, 0.0, 0.0], atol=1e-12)


def test_wgs84_coordinate_differentiation_and_integration_close():
    time = np.arange(5.0)
    latitude = np.deg2rad(np.array([52.0, 52.0001, 52.0002, 52.0003, 52.0004]))
    longitude = np.deg2rad(np.array([4.0, 4.0002, 4.0004, 4.0006, 4.0008]))
    altitude = np.array([100.0, 102.0, 105.0, 103.0, 101.0])

    interval_velocity = differentiate_wgs84_intervals(
        time, latitude, longitude, altitude
    )
    replay = integrate_wgs84_intervals(
        time, latitude[0], longitude[0], altitude[0], *interval_velocity
    )
    metrics = replay_metrics(replay, latitude, longitude, altitude)

    assert metrics.horizontal_final_m < 1e-6
    assert metrics.horizontal_rmse_m < 1e-6
    assert metrics.vertical_rmse_m < 1e-12
