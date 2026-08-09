from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.audit_a320_n2_observability import (
    FEATURE_NAMES,
    _build_designs,
    _decision,
)


def test_n2_design_candidates_match_declared_feature_order():
    samples = 12
    designs = _build_designs(
        n1_percent=np.linspace(60.0, 80.0, samples),
        n2_percent=np.linspace(80.0, 95.0, samples),
        mach=np.linspace(0.3, 0.7, samples),
        pressure_ratio=np.linspace(0.8, 0.3, samples),
        calibrated_airspeed_mps=np.linspace(120.0, 230.0, samples),
    )

    assert set(designs) == set(FEATURE_NAMES)
    for candidate, feature_names in FEATURE_NAMES.items():
        assert designs[candidate].shape == (samples, len(feature_names))
    assert np.all(np.isfinite(designs["n1_plus_n2_rate_1s"][5:]))


def _synthetic_metrics(selected_transition_rmse: float) -> pd.DataFrame:
    rows = []
    for index in range(10):
        for candidate, transition_rmse, rmse, sign, condition in (
            ("n1_history", 10.0, 1.0, 0.90, 120_000.0),
            ("n1_plus_n2_current", 9.99, 0.999, 0.90, 125_000.0),
            (
                "n1_plus_n2_rate_1s",
                selected_transition_rmse,
                0.9,
                0.97,
                121_000.0,
            ),
        ):
            rows.append(
                {
                    "horizon_s": 1,
                    "registration": "A" if index < 5 else "B",
                    "recording": str(index),
                    "candidate": candidate,
                    "transition_rmse_n1_percent": transition_rmse,
                    "rmse_n1_percent": rmse,
                    "transition_sign_accuracy": sign,
                    "normal_matrix_condition": condition,
                }
            )
    return pd.DataFrame(rows)


def test_n2_dynamic_state_gate_accepts_minimal_history_gain():
    decision = _decision(_synthetic_metrics(9.0))

    assert decision["passed"]
    assert decision["selected_candidate"] == "n1_plus_n2_rate_1s"
    assert decision["held_out_flights_improved"] == 10


def test_n2_dynamic_state_gate_rejects_static_only_gain():
    decision = _decision(_synthetic_metrics(9.9))

    assert not decision["passed"]


def test_generated_n2_artifacts_are_fit_only_and_self_consistent():
    root = Path(__file__).parents[1]
    with (root / "docs/modeling/a320-n2-observability-audit.yml").open() as stream:
        audit = yaml.safe_load(stream)
    model_metrics = pd.read_csv(
        root / "docs/modeling/a320-n2-model-metrics.csv"
    )
    lag_metrics = pd.read_csv(root / "docs/modeling/a320-n2-lag-metrics.csv")
    transition_metrics = pd.read_csv(
        root / "docs/modeling/a320-n2-transition-metrics.csv"
    )
    panel = pd.read_csv(
        root / "docs/modeling/a320-replay-panel.csv",
        dtype={"recording": str},
    )

    validation = set(panel.loc[panel["role"] == "validation", "recording"])
    assert not audit["leakage_control"]["validation_flights_loaded"]
    assert validation.isdisjoint(set(model_metrics["recording"]))
    assert audit["decision"]["passed"]
    assert audit["decision"]["selected_candidate"] == "n1_plus_n2_rate_1s"
    assert set(audit["channel_inventory"]["raw_channels"]) == {"N2_1", "N2_2"}
    assert len(lag_metrics.loc[lag_metrics["best_lag_for_flight"]]) == 10
    assert len(transition_metrics) == 18
