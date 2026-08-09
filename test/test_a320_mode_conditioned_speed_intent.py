from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_mode_conditioned_speed_intent_passes_reserved_flight_gate() -> None:
    document = yaml.safe_load(
        (ROOT / "docs/fms/a320-mode-conditioned-speed-intent.yml").read_text()
    )
    decision = document["decision"]
    assert document["status"] == "passed"
    assert decision["selected_structure"] == "affine_selector"
    assert decision["selection_used_validation_flights"] is False
    assert decision["strictly_untouched_after_complete_pipeline_freeze"] is False
    assert decision["crossover_guard_selected_from_validation_metrics"] is False
    assert decision["validation_improvement_over_phase_constant_fraction"] > 0.0
    assert (
        decision["median_validation_branch_agreement_fraction"]
        >= decision["minimum_validation_branch_agreement_gate_fraction"]
    )
    assert document["model"]["bounds"]["crossover_altitude_ft"] == [25_000.0, 35_000.0]


def test_speed_intent_validation_preserves_complete_flight_split() -> None:
    metrics = pd.read_csv(
        ROOT / "docs/fms/a320-mode-conditioned-speed-intent-flight-metrics.csv"
    )
    validation = metrics[metrics["split"] == "validation"]
    assert validation["candidate"].nunique() == 3
    assert validation[["registration", "recording"]].drop_duplicates().shape[0] == 6
    selected = validation[validation["candidate"] == "affine_selector"]
    assert len(selected) == 6
    assert selected["stable_branch_agreement_fraction"].median() >= 0.80


def test_speed_intent_prediction_boundary_excludes_aircraft_response() -> None:
    document = yaml.safe_load(
        (ROOT / "docs/fms/a320-mode-conditioned-speed-intent.yml").read_text()
    )
    inputs = set(document["leakage_boundary"]["prediction_inputs"])
    forbidden = set(document["leakage_boundary"]["response_forbidden_from_prediction"])
    assert inputs.isdisjoint(forbidden)
    assert {"actual CAS", "actual Mach", "inferred speed mode"} <= forbidden
    assert (ROOT / "docs/fms/figures/a320-mode-conditioned-speed-intent.png").is_file()
    assert (ROOT / "docs/fms/figures/a320-mode-conditioned-speed-intent.pdf").is_file()
