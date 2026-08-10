from pathlib import Path

import pandas as pd
import yaml


def test_generated_longitudinal_control_audit_is_fit_only_and_passes_screen():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-longitudinal-control-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv", dtype={"recording": str}
    )
    metrics = pd.read_csv(root / "docs/modeling/a320-pitch-rate-kinematic-lofo.csv")

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert set(metrics["recording"]).isdisjoint(validation)
    assert not audit["leakage_control"]["validation_flights_loaded"]
    assert not audit["leakage_control"]["new_confirmatory_flights_loaded"]
    assert audit["decision"]["pitch_rate_kinematic_fit_gate_passed"]
    assert audit["decision"]["elevator_common_mode_gate_passed"]
    assert audit["decision"]["ready_for_fit_only_pitch_moment_screen"]
    assert not audit["decision"]["release_physical_pitch_rate_scale"]
    assert not audit["decision"]["release_physical_elevator_scale_or_neutral"]


def test_rate_scales_are_stable_and_elevator_offsets_are_measurement_nuisances():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-longitudinal-control-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    flights = pd.read_csv(
        root / "docs/modeling/a320-longitudinal-control-flight-metrics.csv"
    )
    observed = audit["decision"]["observed"]

    assert observed["mean_lofo_rmse_improvement_over_equal_scale"] > 0.4
    assert observed["pitch_rate_scale_ratio_lofo_coefficient_of_variation"] < 0.01
    assert observed["yaw_rate_scale_ratio_lofo_coefficient_of_variation"] < 0.02
    offsets = flights.groupby("registration")["elevator_1_minus_2_median_code"].median()
    assert offsets["TAIL-A"] < 0.0 < offsets["TAIL-B"]
    assert flights["elevator_pair_correlation"].min() > 0.97
