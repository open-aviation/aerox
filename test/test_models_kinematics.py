import pytest

from aerox import load_aircraft
from aerox.models.kinematics import (
    kinematic_parameters_from_aircraft,
    load_kinematic_parameters,
)


def test_load_kinematic_parameters_reads_aircraft_yaml_section() -> None:
    parameters = load_kinematic_parameters("A320")

    assert parameters.cruise_mach == 0.79
    assert parameters.mach_climb_mach == 0.78
    assert parameters.climb_to_fl240_ias == 290


def test_kinematic_parameters_require_populated_aircraft_section() -> None:
    aircraft = load_aircraft("A320")
    aircraft.kinematic = None

    with pytest.raises(ValueError, match="no kinematic parameters"):
        kinematic_parameters_from_aircraft(aircraft)
