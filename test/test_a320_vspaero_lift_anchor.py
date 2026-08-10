import csv
from pathlib import Path

import pytest
import yaml

from tools.audit_a320_vspaero_lift_anchor import build_audit

MODELING = Path("docs/modeling")


def test_vspaero_anchor_blocks_physical_effective_angle_promotion():
    audit, rows = build_audit(
        MODELING / "a320-vspaero-prior.csv",
        MODELING / "a320-pitch-bank-input-calibration.yml",
    )

    assert [row["mach"] for row in rows] == [0.2, 0.5, 0.78]
    assert all(row["qar_excess_fraction_of_vspaero"] > 0.10 for row in rows)
    assert not audit["comparison"]["passed"]
    assert not audit["decision"]["treat_current_effective_alpha_as_physical"]
    assert not audit["decision"]["replace_empirical_lift_map_in_replay"]
    assert not audit["decision"]["inject_vspaero_drag"]


def test_generated_lift_anchor_matches_reproducible_audit():
    generated = yaml.safe_load((MODELING / "a320-vspaero-lift-anchor.yml").read_text())
    rebuilt, rebuilt_rows = build_audit(
        MODELING / "a320-vspaero-prior.csv",
        MODELING / "a320-pitch-bank-input-calibration.yml",
    )
    with (MODELING / "a320-vspaero-lift-anchor.csv").open(newline="") as stream:
        generated_rows = list(csv.DictReader(stream))

    assert generated == rebuilt
    assert float(generated_rows[1]["qar_excess_fraction_of_vspaero"]) == pytest.approx(
        rebuilt_rows[1]["qar_excess_fraction_of_vspaero"]
    )
    for suffix in (".png", ".pdf"):
        figure = MODELING / "figures" / f"a320-vspaero-lift-anchor{suffix}"
        assert figure.stat().st_size > 1_000
