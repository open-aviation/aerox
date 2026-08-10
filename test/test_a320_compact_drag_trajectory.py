from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compact_drag_long_range_trajectory_is_compared_without_promotion():
    document = yaml.safe_load(
        (ROOT / "docs/fms/a320-fms-compact-drag-trajectory.yml").read_text()
    )

    assert document["status"] == ("compact-drag-trajectory-held-by-noninferiority-gate")
    assert document["configuration"]["drag_model"] == ("a320_compact_drag_candidate")
    assert document["configuration"]["controller_changed"] is False
    assert document["configuration"]["n1_surrogate_used_by_rollout"] is False
    assert document["metrics"]["completed"] is True
    assert document["comparison"]["speed_rmse_mps"]["compact_minus_baseline"] < 0.0
    assert document["decision"]["noninferiority_checks"]["altitude_rmse"] is False
    assert document["decision"]["noninferiority_checks"]["fuel_burn_error"] is False
    assert (ROOT / document["artifacts"]["figure_png"]).is_file()
    assert (ROOT / document["artifacts"]["figure_pdf"]).is_file()
