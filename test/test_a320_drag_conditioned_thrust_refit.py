from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_drag_conditioned_thrust_refit_passes_only_inside_common_support() -> None:
    document = yaml.safe_load(
        (ROOT / "docs/modeling/a320-drag-conditioned-thrust-refit.yml").read_text()
    )
    candidates = pd.read_csv(
        ROOT / "docs/modeling/a320-drag-conditioned-thrust-refit-candidates.csv"
    )
    metrics = pd.read_csv(
        ROOT / "docs/modeling/a320-drag-conditioned-thrust-refit-metrics.csv"
    )

    assert document["status"] == "candidate-passes-supported-envelope-refit-gate"
    assert len(candidates) == 48
    assert candidates.groupby("variant").size().to_dict() == {
        "aerox_current": 24,
        "poll_schumann": 24,
    }
    assert len(metrics) == 32
    assert document["leakage_control"]["fuel_flow_used_for_fit"] is False
    assert (
        document["leakage_control"]["common_sample_support_for_both_variants"] is True
    )
    assert document["candidate_support"]["coverage"]["all"]["support_fraction"] > 0.93
    assert (
        document["candidate_support"]["coverage"]["descent"]["support_fraction"] < 0.80
    )

    aggregate = {
        (row["split"], row["variant"]): row for row in document["aggregate_metrics"]
    }
    current = aggregate[("validation", "aerox_current")]
    candidate = aggregate[("validation", "poll_schumann")]
    assert candidate["mean_flight_thrust_rmse_kn"] <= (
        1.05 * current["mean_flight_thrust_rmse_kn"]
    )
    assert (
        candidate["mean_flight_openap_fuel_rmse_kg_s"]
        < (current["mean_flight_openap_fuel_rmse_kg_s"])
    )
    assert document["gate"]["passed"] is True
    assert document["gate"]["default_plant_changed"] is False
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
