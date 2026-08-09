from argparse import Namespace
from pathlib import Path

from tools.plot_a320_modeling_results import run

MODELING = Path("docs/modeling")


def test_modeling_result_figures_export_png_and_pdf(tmp_path):
    outputs = run(
        Namespace(
            point_mass_metrics=MODELING / "a320-point-mass-replay-metrics.csv",
            longitudinal_metrics=MODELING / "a320-longitudinal-replay-metrics.csv",
            yaw_candidates=MODELING / "a320-yaw-rate-delay-candidates.csv",
            yaw_metrics=MODELING / "a320-yaw-rate-flight-metrics.csv",
            observed_replay_metrics=MODELING
            / "a320-observed-heading-rate-replay-metrics.csv",
            lateral_residual_candidates=MODELING
            / "a320-lateral-residual-candidates.csv",
            lateral_residual_metrics=MODELING
            / "a320-lateral-residual-flight-metrics.csv",
            output_dir=tmp_path,
        )
    )

    assert {path.suffix for path in outputs} == {".pdf", ".png"}
    assert len(outputs) == 8
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in outputs)
