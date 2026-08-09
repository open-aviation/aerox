from pathlib import Path
from runpy import run_path

import numpy as np


AUDIT = run_path(
    Path(__file__).parents[1] / "tools" / "audit_a320_translational_forces.py"
)
design_audit = AUDIT["_design_audit"]
weighted_standardized_matrix = AUDIT["_weighted_standardized_matrix"]


def test_weighted_standardization_has_zero_mean_and_unit_variance():
    matrix = np.array(
        [
            [1.0, 4.0],
            [2.0, 2.0],
            [4.0, 1.0],
        ]
    )
    weights = np.array([0.5, 0.25, 0.25])

    standardized, correlation = weighted_standardized_matrix(matrix, weights)
    normalized = weights / weights.sum()

    np.testing.assert_allclose(
        np.sum(normalized[:, None] * standardized, axis=0), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-12)
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)


def test_design_audit_reports_full_rank_for_independent_columns():
    matrix = np.column_stack(
        (
            np.arange(1.0, 8.0),
            np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0]),
            np.array([3.0, 1.0, 6.0, 2.0, 7.0, 4.0, 9.0]),
        )
    )

    result = design_audit(
        matrix,
        np.ones(len(matrix)),
        (0, 1, 2),
        ("propulsion", "parasite", "induced"),
    )

    assert result["rank"] == 3
    assert result["standardized_condition_number"] >= 1.0
    assert result["columns"] == ["propulsion", "parasite", "induced"]
