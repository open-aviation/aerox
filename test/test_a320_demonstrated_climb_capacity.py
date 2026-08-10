from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aerox.models.thrust import (
    DemonstratedClimbN1Capacity,
    QARDemonstratedClimbThrustCapacity,
)
from tools.validate_a320_fms_profile_feasibility import _load_climb_thrust_ceiling

ROOT = Path(__file__).parents[1]
PARAMETERS = ROOT / "docs/modeling/a320-demonstrated-climb-capacity.yml"


def test_demonstrated_capacity_support_uses_only_selected_features() -> None:
    model = DemonstratedClimbN1Capacity(
        feature_set="altitude_linear",
        bootstrap_coefficients=np.array([[90.0, 0.0], [92.0, 0.0]]),
        lower_confidence_quantile=0.10,
        minimum_pressure_altitude_m=500.0,
        maximum_pressure_altitude_m=10_000.0,
        minimum_mach=0.3,
        maximum_mach=0.8,
        minimum_isa_temperature_deviation_k=5.0,
        maximum_isa_temperature_deviation_k=20.0,
    )

    supported = model.support_mask(
        pressure_altitude_m=np.array([1_000.0, 11_000.0]),
        mach=np.array([0.1, 0.5]),
        static_temperature_k=np.array([220.0, 220.0]),
    )

    assert supported.tolist() == [True, False]
    assert model.capacity_percent(
        pressure_altitude_m=np.array([1_000.0]),
        mach=np.array([0.1]),
        static_temperature_k=np.array([220.0]),
    )[0] == pytest.approx(90.2)


def test_released_capacity_uses_untouched_final_afz_holdout() -> None:
    document = yaml.safe_load(PARAMETERS.read_text())
    metrics = pd.read_csv(
        ROOT / "docs/modeling/a320-demonstrated-climb-capacity-flight-metrics.csv",
        dtype={"recording": str},
    )

    assert document["release_decision"]["planner_integration_authorized"]
    assert all(gate["passed"] for gate in document["gates"].values())
    assert document["model"]["feature_set"] == "altitude_quadratic"
    assert set(metrics.loc[metrics["split"] == "holdout", "recording"]) == {
        "FLT0001",
        "FLT0002",
    }
    assert set(metrics.loc[metrics["split"] == "fit", "recording"]).isdisjoint(
        set(metrics.loc[metrics["split"] == "holdout", "recording"])
    )
    assert (
        metrics.loc[metrics["split"] == "holdout", "high_n1_thrust_rmse_kn"].median()
        < 3.0
    )
    for suffix in (".png", ".pdf"):
        figure = (
            ROOT / "docs/modeling/figures" / f"a320-demonstrated-climb-capacity{suffix}"
        )
        assert figure.is_file() and figure.stat().st_size > 1_000


def test_fms_loader_prefers_released_demonstrated_capacity() -> None:
    args = Namespace(
        n1_envelope=ROOT / "docs/modeling/a320-autothrust-envelope.yml",
        climb_capacity=PARAMETERS,
        thrust_parameters=ROOT / "docs/modeling/a320-effective-thrust-calibration.yml",
    )

    model = _load_climb_thrust_ceiling(args)

    assert isinstance(model, QARDemonstratedClimbThrustCapacity)
    assert model.fallback_n1_envelope is not None
