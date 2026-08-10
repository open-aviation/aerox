from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_a320_physical_drag_prior_is_reproducible_and_not_released_early():
    document = yaml.safe_load(
        (ROOT / "docs/modeling/a320-physical-drag-prior.yml").read_text()
    )
    components = pd.read_csv(ROOT / "docs/modeling/a320-physical-drag-components.csv")
    metrics = pd.read_csv(
        ROOT / "docs/modeling/a320-physical-drag-candidate-metrics.csv"
    )

    assert document["status"] == ("candidate_pending_qar_thrust_fuel_trajectory_gate")
    assert document["fit_protocol"]["qar_used_for_drag_fit"] is False
    assert document["selected_candidate"]["candidate"] == "G2_fixed_onset"
    assert set(components["component"]) == {
        "wing",
        "fuselage",
        "horizontal_tail",
        "vertical_tail",
        "unresolved",
    }
    selected = metrics[
        (metrics["candidate"] == "G2_fixed_onset")
        & (metrics["split"] == "public_holdout")
    ].iloc[0]
    poll = metrics[
        (metrics["candidate"] == "poll_schumann")
        & (metrics["split"] == "public_holdout")
    ].iloc[0]
    assert selected["rmse_drag_counts"] < 10.0
    assert selected["rmse_drag_counts"] < poll["rmse_drag_counts"]
    assert document["openvsp_cross_check"]["areas_are_gross_not_exposed"] is True
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
