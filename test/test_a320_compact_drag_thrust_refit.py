from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compact_drag_refit_records_support_gain_and_frozen_hold():
    document = yaml.safe_load(
        (ROOT / "docs/modeling/a320-compact-drag-thrust-refit.yml").read_text()
    )
    metrics = pd.read_csv(ROOT / "docs/modeling/a320-compact-drag-thrust-metrics.csv")

    assert document["status"] == ("compact-candidate-held-by-thrust-fuel-support-gate")
    assert document["leakage_control"]["drag_fit_uses_qar"] is False
    assert document["leakage_control"]["fuel_flow_used_for_fit_or_selection"] is False
    assert document["coverage"]["all"]["compact_support_fraction"] >= 0.99
    assert document["coverage"]["descent"]["compact_support_fraction"] >= 0.95
    assert document["coverage"]["descent"]["poll_support_fraction"] < 0.80
    assert document["gate"]["checks"]["fuel_flow_rmse"] is True
    assert document["gate"]["checks"]["mean_thrust_rmse"] is False
    assert len(metrics) == 32
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
