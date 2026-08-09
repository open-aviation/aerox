import numpy as np
import pandas as pd

from tools.research_support import (
    integrate_contiguous_samples,
    longest_contiguous_slice,
    select_fuel_candidate,
    select_lift_candidate,
)


def test_longest_contiguous_slice_uses_source_sample_adjacency():
    result = longest_contiguous_slice(np.array([2, 3, 7, 8, 9, 10, 15]))

    assert result == slice(2, 6)


def test_fuel_integration_does_not_bridge_filtered_sample_gaps():
    result = integrate_contiguous_samples(
        values=np.full(4, 2.0),
        time_s=np.array([0.0, 1.0, 10.0, 11.0]),
        sample_index=np.array([0, 1, 10, 11]),
    )

    assert result == 4.0


def test_lift_selection_rejects_nonmonotone_and_avoids_unneeded_complexity():
    candidates = pd.DataFrame(
        [
            {
                "feature_set": "complex",
                "feature_count": 5,
                "lofo_mean_flight_cl_rmse": 0.0300,
                "observed_local_monotonic_fraction": 1.0,
            },
            {
                "feature_set": "compact",
                "feature_count": 4,
                "lofo_mean_flight_cl_rmse": 0.0302,
                "observed_local_monotonic_fraction": 1.0,
            },
            {
                "feature_set": "invalid",
                "feature_count": 2,
                "lofo_mean_flight_cl_rmse": 0.0200,
                "observed_local_monotonic_fraction": 0.8,
            },
        ]
    )

    selected = select_lift_candidate(candidates)

    assert selected["feature_set"] == "compact"


def test_fuel_selection_uses_simplest_model_only_when_near_best():
    candidates = pd.DataFrame(
        [
            {
                "structure": "scale",
                "parameter_count": 1,
                "lofo_mean_flight_rmse_kg_s": 0.060,
            },
            {
                "structure": "affine",
                "parameter_count": 2,
                "lofo_mean_flight_rmse_kg_s": 0.055,
            },
        ]
    )

    selected = select_fuel_candidate(candidates)

    assert selected["structure"] == "affine"
