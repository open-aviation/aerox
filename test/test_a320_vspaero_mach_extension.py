from pathlib import Path

import pandas as pd
import yaml

from aerox.models import TabulatedLiftSurface


MODELING = Path("docs/modeling")


def test_extension_panel_is_balanced_unused_and_response_blind():
    baseline = pd.read_csv(MODELING / "a320-replay-panel.csv", dtype={"recording": str})
    panel = pd.read_csv(
        MODELING / "a320-vspaero-extension-confirmation-panel.csv",
        dtype={"recording": str},
    )
    baseline_keys = set(
        zip(baseline["registration"], baseline["recording"], strict=True)
    )
    panel_keys = set(zip(panel["registration"], panel["recording"], strict=True))
    assert not baseline_keys & panel_keys
    assert panel.groupby("registration").size().to_dict() == {"TAIL-A": 3, "TAIL-B": 3}
    assert set(panel["role"]) == {"confirmation"}
    assert set(panel["selection_order"]) == {9, 10, 11}
    assert panel["selection_basis"].str.contains("no replay response").all()


def test_mach_extension_records_failed_original_gate_without_drag_claim():
    document = yaml.safe_load(
        (MODELING / "a320-vspaero-mach-extension.yml").read_text()
    )
    assert document["status"] == "numerical-mach-extension-held"
    assert not document["gates"]["cl_alpha_relative_change_from_mach_078"]["passed"]
    assert not document["decision"]["release_vspaero_drag"]
    assert not document["decision"]["claim_transonic_fidelity"]


def test_extended_lift_table_is_a_complete_bounded_surface():
    surface = TabulatedLiftSurface.from_vspaero_alpha_support_csv(
        MODELING / "a320-vspaero-mach-extended-lift-surface.csv"
    )
    assert surface.mach.tolist() == [0.2, 0.5, 0.78, 0.8, 0.82]
    assert surface.maximum_mach == 0.82
    assert surface.lift_coefficient_scalar(surface.maximum_alpha_rad, 0.82) > 1.0


def test_extension_figures_and_preregistration_exist():
    assert (MODELING / "a320-vspaero-extension-confirmation-plan.md").is_file()
    for stem in (
        "a320-vspaero-mach-extension",
        "a320-vspaero-mach-extension-diagnostic",
    ):
        for suffix in (".png", ".pdf"):
            assert (MODELING / "figures" / f"{stem}{suffix}").is_file()


def test_revised_boundary_was_selected_without_confirmation_data():
    document = yaml.safe_load(
        (MODELING / "a320-vspaero-hybrid-boundary-selection.yml").read_text()
    )
    assert document["status"] == "revised-hybrid-boundary-selected"
    assert "calibration flights only" in document["experiment"]["data_split"]
    assert document["selected"]["candidate"] == "w055_m076"
    assert document["selected"]["mean_surface_weight"] >= 0.5
    assert document["selected"]["maximum_error_ratio"] <= 1.1


def test_independent_extended_hybrid_confirmation_passed_all_gates():
    document = yaml.safe_load(
        (MODELING / "a320-vspaero-extended-hybrid-confirmation.yml").read_text()
    )
    assert document["status"] == "extended-hybrid-confirmation-passed"
    assert document["evaluation"]["flights"] == 6
    assert document["frozen_model"]["maximum_surface_weight"] == 0.55
    assert document["frozen_model"]["mach_core"] == [0.23, 0.76]
    assert all(gate["passed"] for gate in document["gates"].values())
    assert document["decision"]["make_extended_hybrid_default"]
    assert not document["decision"]["release_vspaero_drag"]
    assert not document["decision"]["discard_empirical_fallback"]

    release = yaml.safe_load(
        (MODELING / "a320-clean-lift-model-release.yml").read_text()
    )
    assert release["status"] == "released-controlled-clean-replay-default"
    assert release["lift_model"]["fallback_retained"]
    assert not release["drag_model"]["vspaero_drag_used"]


def test_boundary_and_confirmation_figures_exist():
    for stem in (
        "a320-vspaero-hybrid-boundary-selection",
        "a320-vspaero-extended-hybrid-confirmation",
    ):
        for suffix in (".png", ".pdf"):
            assert (MODELING / "figures" / f"{stem}{suffix}").is_file()
