from pathlib import Path
from runpy import run_path

VALIDATION = run_path(Path(__file__).parents[1] / "tools" / "validate_rj85_aoa.py")
_decision = VALIDATION["_decision"]


GATES = {
    "maximum_mean_rmse_deg": 0.35,
    "minimum_mean_improvement_vs_fixed_training_mean_pct": 50.0,
    "minimum_flights_improved_vs_fixed_training_mean": 2,
    "maximum_absolute_mean_bias_deg": 0.25,
    "minimum_flights_with_raw_target_correlation_at_least_0_8": 2,
}


def _row(rmse=0.2, improvement=60.0, bias=0.1, correlation=0.9):
    return {
        "rmse_deg": rmse,
        "improvement_vs_fixed_training_mean_pct": improvement,
        "bias_deg": bias,
        "raw_target_correlation": correlation,
    }


def test_confirmation_decision_passes_only_when_every_gate_passes():
    decision = _decision([_row(), _row(bias=-0.1)], GATES)

    assert decision["passed"] is True
    assert all(decision["gate_results"].values())


def test_confirmation_decision_blocks_on_mean_rmse_gate():
    decision = _decision([_row(rmse=0.5), _row(rmse=0.4)], GATES)

    assert decision["passed"] is False
    assert decision["gate_results"]["mean_rmse"] is False
