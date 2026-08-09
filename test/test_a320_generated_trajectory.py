from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.generate_a320_comparison_trajectories import build_reference


MODELING = Path("docs/modeling")


def test_generated_reference_is_smooth_and_nontrivial():
    reference, north, east = build_reference()

    assert len(reference.time_s) == 1_501
    assert np.allclose(np.diff(reference.time_s), 1.0)
    assert np.isclose(np.ptp(reference.true_airspeed_mps), 12.0)
    assert np.isclose(np.ptp(reference.pressure_altitude_m), 600.0)
    assert np.isclose(np.rad2deg(np.ptp(reference.heading_rad)), 75.0)
    assert north[-1] > 100_000.0
    assert east[-1] > 100_000.0
    assert np.max(np.abs(np.diff(reference.true_airspeed_rate_mps2))) < 0.002


def test_generated_comparison_artifacts_record_passing_bounded_run():
    document = yaml.safe_load(
        (MODELING / "a320-generated-trajectory-comparison.yml").read_text()
    )
    samples = pd.read_csv(MODELING / "a320-generated-trajectory-comparison.csv")

    assert document["decision"]["bounded_trajectory_generation_demonstrated"]
    assert all(gate["passed"] for gate in document["gates"].values())
    assert document["decision"]["airbus_autopilot_or_fadec_identified"] is False
    assert len(samples) == 1_501
    assert np.allclose(samples["hybrid_geometry_surface_weight"], 0.55)
    assert (
        document["trajectory_metrics"]["empirical_same_controls_open_loop"][
            "horizontal_rmse_m"
        ]
        > document["trajectory_metrics"]["released_hybrid_closed_loop"][
            "horizontal_rmse_m"
        ]
    )
    for stem in (
        "a320-generated-trajectory-comparison",
        "a320-generated-control-history",
    ):
        for suffix in (".png", ".pdf"):
            path = MODELING / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000
