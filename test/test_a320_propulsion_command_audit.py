import numpy as np
import pandas as pd

from tools.audit_a320_propulsion_commands import (
    _future_change,
    _observability_decision,
    _stable_code_mask,
)


def test_future_change_is_forward_difference_without_edge_fill():
    result = _future_change(np.array([1.0, 2.0, 4.0, 7.0]), 2)

    np.testing.assert_allclose(result[:2], [3.0, 5.0])
    assert np.all(np.isnan(result[2:]))


def test_stable_code_mask_requires_the_whole_future_interval():
    values = np.array([45.0, 45.0, 45.0, 20.0, 45.0, 45.0])

    result = _stable_code_mask(values, 45.0, 2)

    np.testing.assert_array_equal(result, [True, False, False, False, False, False])


def test_observability_gate_rejects_small_inconsistent_improvement():
    rows = []
    for index in range(10):
        for candidate, rmse, accuracy in (
            ("state_environment", 8.0, 0.50),
            ("named_guidance", 7.9 if index < 6 else 8.1, 0.55),
        ):
            rows.append(
                {
                    "registration": "A",
                    "recording": str(index),
                    "candidate": candidate,
                    "transition_rmse_n1_percent": rmse,
                    "transition_sign_accuracy": accuracy,
                }
            )

    decision = _observability_decision(pd.DataFrame(rows))

    assert not decision["passed"]
    assert decision["status"] == "no-recorded-engine-command-identified"
