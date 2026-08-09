import csv
from pathlib import Path

import pytest
import yaml

from tools.build_a320_vspaero_prior import parse_stability_file

PARAMETERS = Path("docs/modeling/a320-vspaero-prior.yml")
RESULTS = Path("docs/modeling/a320-vspaero-prior.csv")
CONVERGENCE = Path("docs/modeling/a320-vspaero-convergence.csv")
MODEL = Path("docs/modeling/a320-vspaero-nominal.vsp3")


def test_stability_parser_extracts_base_and_derivative_conventions(tmp_path):
    stab = tmp_path / "minimal.stab"
    stab.write_text(
        """Coef          Total        Alpha        Beta          p            q            r           Mach         U
CL        0.2000       5.7000       0.0000       0.0000      14.0000       0.0000       0.1000       0.0500
CD        0.0090       0.0900       0.0000       0.0000      -0.1000       0.0000       0.0010       0.0005
CS        0.0000       0.0000      -0.4000       0.0300       0.0000       0.5000       0.0000       0.0000
CMl       0.0000       0.0000      -0.1500      -0.5000       0.0000       0.1000       0.0000       0.0000
CMm      -0.0900      -2.8000       0.0000       0.0000     -40.0000       0.0000      -0.0200      -0.0100
CMn       0.0000       0.0000       0.1700       0.0100       0.0000      -0.1900       0.0000       0.0000
#
SM                      0.4900000 no_unit
"""
    )

    parsed = parse_stability_file(stab)

    assert parsed["CL"] == pytest.approx(0.2)
    assert parsed["CL_alpha"] == pytest.approx(5.7)
    assert parsed["CMm_q"] == pytest.approx(-40.0)
    assert parsed["CS_beta"] == pytest.approx(-0.4)
    assert parsed["CMl_p"] == pytest.approx(-0.5)
    assert parsed["CMn_r"] == pytest.approx(-0.19)
    assert parsed["static_margin"] == pytest.approx(0.49)


def test_generated_mesh_gate_and_release_boundary_are_consistent():
    prior = yaml.safe_load(PARAMETERS.read_text())
    with CONVERGENCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    differences = [float(row["medium_fine_relative_difference"]) for row in rows]
    assert len(rows) == 8
    assert max(differences) == pytest.approx(
        prior["mesh_convergence_gate"]["maximum_medium_fine_relative_difference"]
    )
    assert max(differences) < prior["mesh_convergence_gate"]["threshold_relative"]
    assert prior["release_decision"]["static_geometry_derivative_prior"]
    assert not prior["release_decision"]["dimensional_rate_derivative_prior"]
    assert not prior["release_decision"]["viscous_drag"]
    assert not prior["release_decision"]["physical_a320_six_dof_model"]


def test_nominal_result_has_expected_stable_transport_signs():
    with RESULTS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    nominal = next(
        row for row in rows if row["mesh"] == "medium" and float(row["mach"]) == 0.5
    )

    assert float(nominal["CL_alpha"]) > 0
    assert float(nominal["CMm_alpha"]) < 0
    assert float(nominal["CS_beta"]) < 0
    assert float(nominal["CMl_beta"]) < 0
    assert float(nominal["CMl_p"]) < 0
    assert float(nominal["CMn_beta"]) > 0
    assert float(nominal["CMn_r"]) < 0


def test_openvsp_model_is_inspectable_and_contains_all_lifting_surfaces():
    text = MODEL.read_text()

    assert text.startswith("<?xml version=\"1.0\"?>")
    assert "A320_Main_Wing" in text
    assert "A320_Horizontal_Tail" in text
    assert "A320_Vertical_Tail" in text
    assert "/tmp/" not in text
