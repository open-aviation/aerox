import numpy as np

from aerox import aero
from aerox.models import EffectiveLiftMap, ParabolicDragPolar
from aerox.modeling.dynamics import PointMassModel
from aerox.guidance import (
    TrajectoryTrackingController,
    TrajectoryReference,
    simulate_trajectory_tracking_zoh,
)
from aerox.models.openap import OpenAPFuelFlowPrior


def _plant() -> PointMassModel:
    return PointMassModel(
        wing_area_m2=100.0,
        lift_map=EffectiveLiftMap("linear_alpha", np.array([0.50, 0.10])),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(
            engine_count=2,
            rated_thrust_n_per_engine=100_000.0,
            c1=1.0,
            c2=2.0,
            c3=1.0,
        ),
    )


def _controller() -> TrajectoryTrackingController:
    return TrajectoryTrackingController(
        speed_error_gain_per_s=0.05,
        flight_path_angle_error_gain_per_s=0.25,
        heading_error_gain_per_s=0.08,
        altitude_to_flight_path_gain_rad_per_m=2e-4,
        minimum_pitch_rad=np.deg2rad(-2.0),
        maximum_pitch_rad=np.deg2rad(12.0),
        maximum_bank_rad=np.deg2rad(25.0),
        minimum_thrust_n=0.0,
        maximum_thrust_n=200_000.0,
        maximum_pitch_rate_rad_s=np.deg2rad(1.0),
        maximum_bank_rate_rad_s=np.deg2rad(2.0),
        maximum_thrust_rate_n_s=25_000.0,
    )


def _reference(time: np.ndarray, *, heading_rad: np.ndarray | None = None):
    zeros = np.zeros_like(time)
    if heading_rad is None:
        heading_rad = zeros
    heading_rate = np.gradient(heading_rad, time)
    return TrajectoryReference(
        time_s=time,
        true_airspeed_mps=np.full_like(time, 150.0),
        true_airspeed_rate_mps2=zeros,
        flight_path_angle_rad=zeros,
        flight_path_angle_rate_rad_s=zeros,
        heading_rad=heading_rad,
        heading_rate_rad_s=heading_rate,
        pressure_altitude_m=zeros,
    )


def test_level_reference_converges_from_state_errors():
    time = np.arange(0.0, 121.0)
    result = simulate_trajectory_tracking_zoh(
        plant=_plant(),
        controller=_controller(),
        reference=_reference(time),
        static_temperature_k=aero.T0,
        initial_mass_kg=60_000.0,
        initial_true_airspeed_mps=145.0,
        initial_pressure_altitude_m=-100.0,
    )
    trajectory = result.trajectory

    assert abs(trajectory.true_airspeed_mps[-1] - 150.0) < 0.1
    assert abs(trajectory.pressure_altitude_m[-1]) < 5.0
    assert np.max(np.abs(np.rad2deg(trajectory.bank_angle_rad))) <= 25.0
    assert np.all(
        (trajectory.net_thrust_n >= 0.0) & (trajectory.net_thrust_n <= 200_000.0)
    )
    assert trajectory.mass_kg[-1] < trajectory.mass_kg[0]


def test_smooth_heading_reference_generates_bounded_bank_and_turn():
    time = np.arange(0.0, 121.0)
    phase = np.clip((time - 20.0) / 80.0, 0.0, 1.0)
    smooth = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    reference = _reference(time, heading_rad=np.deg2rad(30.0) * smooth)
    result = simulate_trajectory_tracking_zoh(
        plant=_plant(),
        controller=_controller(),
        reference=reference,
        static_temperature_k=aero.T0,
        initial_mass_kg=60_000.0,
    )
    trajectory = result.trajectory

    assert np.rad2deg(trajectory.heading_rad[-1]) > 29.0
    assert np.max(trajectory.bank_angle_rad) > np.deg2rad(1.0)
    assert np.max(np.abs(np.diff(trajectory.bank_angle_rad))) <= np.deg2rad(2.0) + 1e-12
    assert (
        np.max(np.abs(np.diff(trajectory.pitch_angle_rad))) <= np.deg2rad(1.0) + 1e-12
    )
    assert np.max(np.abs(np.diff(trajectory.net_thrust_n))) <= 25_000.0 + 1e-9


def test_reference_rejects_mismatched_or_nonpositive_speed():
    time = np.arange(3.0)
    zeros = np.zeros(3)
    try:
        TrajectoryReference(
            time_s=time,
            true_airspeed_mps=np.zeros(3),
            true_airspeed_rate_mps2=zeros,
            flight_path_angle_rad=zeros,
            flight_path_angle_rate_rad_s=zeros,
            heading_rad=zeros,
            heading_rate_rad_s=zeros,
            pressure_altitude_m=zeros,
        )
    except ValueError as error:
        assert "airspeed" in str(error)
    else:
        raise AssertionError("nonpositive reference speed was accepted")


def test_tracking_supports_explicit_guidance_mass_mismatch():
    time = np.arange(0.0, 31.0)
    common = dict(
        plant=_plant(),
        controller=_controller(),
        reference=_reference(time),
        static_temperature_k=aero.T0,
        initial_mass_kg=60_000.0,
    )
    nominal = simulate_trajectory_tracking_zoh(**common)
    biased = simulate_trajectory_tracking_zoh(
        **common,
        guidance_plant=_plant(),
        guidance_mass_scale=0.9,
    )

    assert not np.allclose(
        nominal.trajectory.pitch_angle_rad, biased.trajectory.pitch_angle_rad
    )
    assert not np.allclose(
        nominal.trajectory.pressure_altitude_m,
        biased.trajectory.pressure_altitude_m,
    )

    with np.testing.assert_raises(ValueError):
        simulate_trajectory_tracking_zoh(**common, guidance_mass_scale=0.0)
