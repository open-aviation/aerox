import numpy as np

from aerox.models.thrust import SpeedTrackingAutothrustController


def test_speed_controller_adds_feedback_to_trim_and_applies_limits():
    controller = SpeedTrackingAutothrustController(
        speed_error_gain_per_s=0.05,
        target_rate_feedforward_gain=0.5,
    )
    acceleration = controller.acceleration_command_mps2(
        true_airspeed_mps=np.array([200.0]),
        target_true_airspeed_mps=np.array([210.0]),
        target_true_airspeed_rate_mps2=np.array([0.2]),
    )
    thrust = controller.commanded_effective_thrust_n(
        trim_thrust_n=np.array([40_000.0]),
        mass_kg=np.array([60_000.0]),
        true_airspeed_mps=np.array([200.0]),
        target_true_airspeed_mps=np.array([210.0]),
        target_true_airspeed_rate_mps2=np.array([0.2]),
        minimum_thrust_n=np.array([10_000.0]),
        maximum_thrust_n=np.array([70_000.0]),
    )

    np.testing.assert_allclose(acceleration, [0.6])
    np.testing.assert_allclose(thrust, [70_000.0])


def test_speed_response_rollout_uses_exact_first_order_solution():
    controller = SpeedTrackingAutothrustController(speed_error_gain_per_s=0.1)
    time = np.array([0.0, 1.0, 3.0])
    target = np.full(3, 100.0)
    target_rate = np.zeros(3)

    speed = controller.simulate_speed_response_zoh(
        time_s=time,
        target_true_airspeed_mps=target,
        target_true_airspeed_rate_mps2=target_rate,
        initial_true_airspeed_mps=80.0,
    )

    np.testing.assert_allclose(speed, 100.0 - 20.0 * np.exp(-0.1 * time))


def test_speed_controller_rejects_nonpositive_feedback_gain():
    with np.testing.assert_raises(ValueError):
        SpeedTrackingAutothrustController(speed_error_gain_per_s=0.0)
