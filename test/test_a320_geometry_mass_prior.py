from pathlib import Path

import pytest
import yaml

from tools.build_a320_geometry_mass_prior import build_prior

MEASUREMENTS = Path("docs/modeling/a320-cad-geometry-measurements.csv")
PARAMETERS = Path("docs/modeling/a320-geometry-mass-prior.yml")


def test_manufacturer_cad_geometry_prior_is_physically_consistent():
    prior = build_prior(MEASUREMENTS)
    geometry = prior["geometry"]
    wing = geometry["wing_reference"]
    horizontal = geometry["horizontal_tail"]
    vertical = geometry["vertical_tail"]

    assert wing["reconstructed_area_m2"] == pytest.approx(122.0, rel=0.03)
    assert wing["reconstructed_mac_m"] == pytest.approx(4.1935, rel=0.03)
    assert wing["exposed_dihedral_deg"] == pytest.approx(5.29, abs=0.05)
    assert horizontal["area_m2"]["nominal"] == pytest.approx(30.84, rel=0.01)
    assert vertical["area_m2"]["nominal"] == pytest.approx(21.51, rel=0.01)
    assert 0 < horizontal["taper_ratio"]["nominal"] < 1
    assert 0 < vertical["taper_ratio"]["nominal"] < 1
    assert horizontal["reference_moment_arm_m"]["nominal"] > 15
    assert vertical["reference_moment_arm_m"]["nominal"] > 14
    assert horizontal["dihedral_deg"]["nominal"] == pytest.approx(6.28, abs=0.05)
    assert horizontal["root_vertical_offset_from_wing_m"]["nominal"] > 1.5
    assert vertical["root_vertical_offset_from_wing_m"]["nominal"] > 2.5


def test_visible_control_geometry_does_not_claim_actuator_limits():
    prior = build_prior(MEASUREMENTS)
    controls = prior["geometry"]["flight_controls"]

    assert controls["elevator"]["area_ratio"]["nominal"] == pytest.approx(0.481, abs=0.002)
    assert controls["rudder"]["area_ratio"]["nominal"] == pytest.approx(0.300, abs=0.002)
    assert 0.75 < controls["aileron"]["inboard_span_station_fraction_semispan"]["nominal"] < 0.82
    assert 0.92 < controls["aileron"]["outboard_span_station_fraction_semispan"]["nominal"] < 0.98
    assert not prior["release_decision"]["control_surface_deflection_or_rate_limits"]


def test_generic_inertia_prior_is_bounded_and_not_released_as_a320_data():
    prior = build_prior(MEASUREMENTS)
    mass = prior["mass_properties"]
    diagonal = mass["diagonal"]

    assert not mass["release_for_physical_a320_simulation"]
    assert not prior["release_decision"]["a320_inertia_tensor"]
    assert diagonal["ixx"]["nominal_kg_m2"] < diagonal["iyy"]["nominal_kg_m2"]
    assert diagonal["iyy"]["nominal_kg_m2"] < diagonal["izz"]["nominal_kg_m2"]
    assert abs(mass["ixz"]["admissible_range_kg_m2"][1]) < (
        diagonal["ixx"]["nominal_kg_m2"] * diagonal["izz"]["nominal_kg_m2"]
    ) ** 0.5


def test_generated_prior_matches_reproducible_measurement_build():
    generated = yaml.safe_load(PARAMETERS.read_text())
    rebuilt = build_prior(MEASUREMENTS)

    assert generated["source"]["source_dwg_sha256"] == rebuilt["source"]["source_dwg_sha256"]
    assert generated["geometry"] == rebuilt["geometry"]
    assert generated["mass_properties"] == rebuilt["mass_properties"]
