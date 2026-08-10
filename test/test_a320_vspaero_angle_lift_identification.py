import csv
from pathlib import Path

import pytest
import yaml

MODELING = Path("docs/modeling")


def test_local_linear_fit_gate_and_residual_attribution_are_frozen():
    identification = yaml.safe_load(
        (MODELING / "a320-vspaero-angle-lift-identification.yml").read_text()
    )
    residual = yaml.safe_load(
        (MODELING / "a320-vspaero-angle-lift-residual-audit.yml").read_text()
    )

    assert not identification["decision"]["fit_gate_passed"]
    assert not identification["data_boundary"]["validation_flights_loaded"]
    assert identification["gates"]["lofo_rmse_ratio"]["passed"]
    assert not identification["gates"]["model_alpha_envelope_fraction"]["passed"]
    assert not identification["gates"]["registration_bias_difference_cl"]["passed"]
    assert residual["result"]["selected_candidate"] == "registration"
    assert residual["result"]["selected_metrics"][
        "registration_angle_difference_deg"
    ] == pytest.approx(-0.2973016904)
    assert not residual["decision"]["tail_specific_aerodynamics"]
    assert not residual["decision"]["open_validation"]


def test_angle_lift_diagnostic_tables_and_figures_exist():
    with (MODELING / "a320-vspaero-angle-lift-residual-candidates.csv").open(
        newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert [row["candidate"] for row in rows] == [
        "common",
        "registration",
        "phase",
        "registration_phase",
    ]
    assert rows[1]["passes_diagnostic_screen"] == "True"
    for stem in (
        "a320-vspaero-angle-lift-identification",
        "a320-vspaero-angle-lift-residual-audit",
    ):
        for suffix in (".png", ".pdf"):
            assert (MODELING / "figures" / f"{stem}{suffix}").stat().st_size > 1_000
