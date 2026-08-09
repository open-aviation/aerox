import csv
from pathlib import Path

import pytest
import yaml

from tools.audit_a320_vspaero_alpha_support import build_audit

MODELING = Path("docs/modeling")


def test_alpha_support_holds_local_derivative_extension_but_retains_surface():
    with (MODELING / "a320-vspaero-alpha-support.csv").open(newline="") as stream:
        rows = [
            {
                key: float(value)
                for key, value in row.items()
                if key not in {"vortex_loop_equations", "geometry_nodes"}
            }
            | {
                "vortex_loop_equations": int(row["vortex_loop_equations"]),
                "geometry_nodes": int(row["geometry_nodes"]),
            }
            for row in csv.DictReader(stream)
        ]
    rebuilt = build_audit(
        rows, openvsp_version="3.51.2", vspaero_version="7.2.2", jobs=64
    )
    generated = yaml.safe_load(
        (MODELING / "a320-vspaero-alpha-support.yml").read_text()
    )

    assert generated == rebuilt
    assert generated["gates"]["absolute_cl_linearity_error"]["passed"]
    assert not generated["gates"]["cl_alpha_relative_change"]["passed"]
    assert not generated["decision"]["physical_stall_or_high_alpha_model"]


def test_surface_fit_and_independent_confirmation_pass_without_overclaiming():
    fit = yaml.safe_load(
        (MODELING / "a320-vspaero-lift-surface-identification.yml").read_text()
    )
    confirmation = yaml.safe_load(
        (MODELING / "a320-vspaero-lift-surface-confirmation.yml").read_text()
    )

    assert fit["decision"]["fit_gate_passed"]
    assert not fit["data_boundary"]["validation_flights_loaded"]
    assert fit["model"]["pitch_scale"] == pytest.approx(1.0517819354)
    assert all(gate["passed"] for gate in fit["gates"].values())
    assert confirmation["decision"]["confirmation_passed"]
    assert not confirmation["leakage_control"]["parameters_refitted"]
    assert confirmation["comparison"]["surface_to_empirical_mean_rmse_ratio"] < 1.0
    assert not confirmation["decision"]["release_physical_pitch_sensor_calibration"]
    assert not confirmation["decision"]["release_vspaero_drag"]


def test_surface_artifacts_include_review_figures():
    for stem in (
        "a320-vspaero-alpha-support",
        "a320-vspaero-lift-surface-identification",
        "a320-vspaero-lift-surface-confirmation",
    ):
        for suffix in (".png", ".pdf"):
            assert (MODELING / "figures" / f"{stem}{suffix}").stat().st_size > 1_000
