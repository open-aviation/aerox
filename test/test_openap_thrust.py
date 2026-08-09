import pytest

from aerox import aero
from aerox.models.openap import OpenAPTwoSpoolMaximumClimbThrust


@pytest.mark.parametrize(
    ("altitude_m", "speed_mps", "temperature_shift_k", "expected_thrust_n"),
    (
        (1_000.0, 110.0, 5.0, 109_774.18294207501),
        (3_500.0, 180.0, -3.0, 77_602.19977449338),
        (8_000.0, 225.0, 8.0, 55_845.12209784101),
    ),
)
def test_openap_thrust_port_matches_pinned_upstream(
    altitude_m: float,
    speed_mps: float,
    temperature_shift_k: float,
    expected_thrust_n: float,
) -> None:
    prior = OpenAPTwoSpoolMaximumClimbThrust()
    temperature = float(aero.temperature(altitude_m)) + temperature_shift_k

    thrust = prior.maximum_thrust_n(
        pressure_altitude_m=altitude_m,
        true_airspeed_mps=speed_mps,
        static_temperature_k=temperature,
    )

    assert thrust == pytest.approx(expected_thrust_n, rel=1e-12)


def test_fast_climb_setting_has_more_thrust_than_slow_climb_below_cruise() -> None:
    slow = OpenAPTwoSpoolMaximumClimbThrust(reference_rate_of_climb_ft_min=1_000.0)
    fast = OpenAPTwoSpoolMaximumClimbThrust(reference_rate_of_climb_ft_min=4_000.0)
    condition = {
        "pressure_altitude_m": 3_000.0,
        "true_airspeed_mps": 180.0,
        "static_temperature_k": 270.0,
    }

    assert fast.maximum_thrust_n(**condition) > slow.maximum_thrust_n(**condition)
