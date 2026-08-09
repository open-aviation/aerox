from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_a320_drag_reconciliation_selects_bounded_candidate_without_release() -> None:
    document = yaml.safe_load(
        (ROOT / "docs/modeling/a320-drag-source-reconciliation.yml").read_text()
    )
    surface = pd.read_csv(
        ROOT / "docs/modeling/a320-drag-source-reconciliation-surface.csv"
    )
    qar = pd.read_csv(
        ROOT / "docs/modeling/a320-drag-source-reconciliation-qar.csv"
    )

    assert document["status"] == "poll-schumann-leading-candidate-default-held"
    assert document["decision"]["candidate_implemented"] is True
    assert document["decision"]["default_plant_changed"] is False
    assert document["source_assessment"]["poll_schumann"]["model_support"] == {
        "lift_coefficient": [0.3, 0.7],
        "mach": [0.0, 0.82],
        "reynolds_number_is_explicit": True,
    }
    assert document["source_assessment"]["openap_surveillance"]["distribution"][
        "sample_count"
    ] == 92
    assert (
        document["source_assessment"]["openap_surveillance"]["distribution"][
            "cd0_k_correlation"
        ]
        > 0.999
    )
    assert (
        document["metrics_against_obert_krull_points"]["poll_schumann"][
            "rmse_drag_counts"
        ]
        < document["metrics_against_obert_krull_points"]["aerox_current"][
            "rmse_drag_counts"
        ]
    )
    representative = document["representative_comparison"]
    assert abs(
        representative["obert_krull_total_drag_coefficient"]
        - representative["poll_schumann"]["total_drag_coefficient"]
    ) < 2.0e-4
    assert document["qar_cruise_comparison"]["sample_count"] == len(qar) == 953
    assert qar["poll_minus_current_drag_n"].median() < -3_000.0
    assert surface["poll_schumann_drag_coefficient"].notna().all()
    assert surface["lift_coefficient"].min() == 0.3
    assert surface["lift_coefficient"].max() == 0.7
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
