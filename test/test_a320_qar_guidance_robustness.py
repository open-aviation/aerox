from pathlib import Path

import pandas as pd
import yaml

from tools.validate_a320_qar_guidance_robustness import scenarios


MODELING = Path("docs/modeling")


def test_robustness_design_is_frozen_and_covers_all_requested_sources():
    cases = scenarios()
    names = {case.name for case in cases}

    assert len(cases) == 10
    assert "nominal" in names
    assert {case.guidance_mass_scale for case in cases} >= {0.9, 1.0, 1.1}
    assert {case.drag_scale for case in cases} >= {0.9, 1.0, 1.1}
    assert {case.lift_scale for case in cases} >= {0.95, 1.0, 1.05}
    assert any(case.wind_north_bias_mps != 0.0 for case in cases)
    assert "combined_adverse" in names


def test_qar_guidance_artifacts_separate_nominal_pass_from_robustness_hold():
    document = yaml.safe_load(
        (MODELING / "a320-qar-guidance-robustness.yml").read_text()
    )
    metrics = pd.read_csv(MODELING / "a320-qar-guidance-robustness-metrics.csv")

    assert document["decision"]["qar_reference_tracking_passed"]
    assert not document["decision"]["deterministic_robustness_set_passed"]
    assert not document["decision"]["overall_gate_passed"]
    assert document["status"] == "qar-guidance-robustness-held"
    assert not document["gates"]["robust_maximum_altitude_rmse_m"]["passed"]
    assert (
        document["robust_worst_scenarios"]["altitude_rmse_m"]
        == "combined_adverse"
    )
    assert len(metrics) == 10
    assert not {"latitude", "longitude", "time_s"} & set(metrics.columns)
    assert set(metrics["name"]) == {case.name for case in scenarios()}
    for stem in (
        "a320-qar-guidance-errors",
        "a320-qar-guidance-trajectory-comparison",
        "a320-qar-guidance-robustness",
    ):
        for suffix in (".png", ".pdf"):
            path = MODELING / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000
