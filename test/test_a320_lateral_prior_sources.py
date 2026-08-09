from pathlib import Path

import yaml

from tools.audit_a320_lateral_prior_sources import build_audit

AIRCRAFT = Path("src/aerox/data/aircraft/yml/turbofan/a320.yml")
MODELING = Path("docs/modeling")


def test_certification_source_gate_adds_core_a320_fields():
    audit = build_audit(AIRCRAFT)
    fields = {item["field"]: item for item in audit["local_inventory"]["fields"]}

    assert audit["aircraft"] == "Airbus A320-214"
    assert audit["local_inventory"]["available_fields"] == 23
    assert fields["wing.span"]["value"] == 34.10
    assert fields["wing.mac"]["value"] == 4.1935
    assert fields["certified_mass_limits.mtow"]["value"] == 78000
    assert fields["certified_mass_limits.mzfw"]["value"] == 62500


def test_drawing_geometry_is_available_but_controls_and_inertias_remain_unresolved():
    audit = build_audit(AIRCRAFT)
    fields = {item["field"]: item for item in audit["local_inventory"]["fields"]}

    assert fields["horizontal_tail.area"]["value"] == 30.843428
    assert fields["horizontal_tail.span"]["value"] == 12.45
    assert fields["vertical_tail.area"]["value"] == 21.514182
    assert fields["vertical_tail.height"]["value"] == 5.87

    for field in (
        "aileron.deflection_limit",
        "rudder.deflection_limit",
        "mass_properties.cg_forward_mac",
        "mass_properties.reference_cg_mac",
        "mass_properties.ixx",
        "mass_properties.iyy",
        "mass_properties.izz",
        "mass_properties.ixz",
    ):
        assert not fields[field]["available"]
        assert fields[field]["value"] is None

    assert not audit["decision"]["adopt_gtm_coefficients_as_a320"]
    assert not audit["decision"]["release_a320_lateral_coefficients"]


def test_generated_source_audit_records_primary_source_decisions():
    audit = yaml.safe_load(
        (MODELING / "a320-lateral-prior-source-audit.yml").read_text()
    )
    sources = {item["source_id"]: item for item in audit["source_matrix"]}

    assert sources["easa_a064_issue_62"]["decision"] == "adopt-certified-fields"
    assert (
        sources["airbus_a320_acap_2026_07"]["decision"]
        == "adopt-geometry-evidence-only"
    )
    assert (
        sources["nasa_gna_2014_0542"]["decision"]
        == "adopt-structure-not-a320-coefficients"
    )
    assert not audit["selected_prior"]["generic_numeric_reference"][
        "adopted_as_a320_coefficients"
    ]
