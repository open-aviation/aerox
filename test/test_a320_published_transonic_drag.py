from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_published_a320_transonic_drag_audit_is_bounded_and_reproduced() -> None:
    document = yaml.safe_load(
        (ROOT / "docs/modeling/a320-published-transonic-drag-audit.yml").read_text()
    )
    reference = pd.read_csv(
        ROOT / "docs/modeling/a320-published-transonic-drag-reference.csv"
    )

    assert document["status"] == "independent-reference-accepted-default-model-held"
    assert document["source"]["dataset_file_md5"] == (
        "219286b978a47f1a5a57476c0ec99ecc"
    )
    assert len(reference) == document["published_support"]["source_point_count"] == 271
    assert sorted(reference["lift_coefficient"].unique()) == [0.3, 0.4, 0.5]
    assert (
        document["comparison_metrics"]["published_tanh_reproduced"]["rmse_drag_counts"]
        < document["comparison_metrics"]["published_baseline_gur_lock"][
            "rmse_drag_counts"
        ]
    )
    assert document["qar_cruise_support"]["samples_inside_full_published_support"] == 0
    assert document["decision"]["default_plant_changed"] is False
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
