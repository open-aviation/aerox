import numpy as np

from aerox.modeling.force_reconstruction import (
    centered_linear_derivative,
    reconstruct_translational_forces,
)


def test_centered_derivative_is_exact_for_a_quadratic_at_the_center():
    time = np.arange(11.0)
    values = time**2 + 3.0 * time + 2.0

    derivative = centered_linear_derivative(values, time, window=5)

    np.testing.assert_allclose(derivative[2:-2], 2.0 * time[2:-2] + 3.0)
    assert np.all(np.isnan(derivative[:2]))
    assert np.all(np.isnan(derivative[-2:]))


def test_steady_level_flight_recovers_weight_lift_and_prior_drag():
    samples = 21
    mass = np.full(samples, 60_000.0)
    speed = np.full(samples, 200.0)
    dynamic_pressure = np.full(samples, 10_000.0)
    forces = reconstruct_translational_forces(
        time_s=np.arange(float(samples)),
        true_airspeed_mps=speed,
        flight_path_angle_rad=np.zeros(samples),
        bank_angle_rad=np.zeros(samples),
        mass_kg=mass,
        dynamic_pressure_pa=dynamic_pressure,
        wing_area_m2=120.0,
        cd0=0.02,
        k=0.04,
        derivative_window=5,
    )

    interior = slice(2, -2)
    weight = mass[interior] * 9.80665
    expected_cl = weight / (dynamic_pressure[interior] * 120.0)
    expected_drag = dynamic_pressure[interior] * 120.0 * (0.02 + 0.04 * expected_cl**2)
    np.testing.assert_allclose(forces.net_along_path_force_n[interior], 0.0)
    np.testing.assert_allclose(forces.required_lift_n[interior], weight)
    np.testing.assert_allclose(forces.prior_implied_thrust_n[interior], expected_drag)
