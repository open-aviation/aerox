from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FMS_DOCS = ROOT / "docs/fms"


def test_continuous_speed_mass_iteration_improves_but_remains_held() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-mode-conditioned-integration.yml").read_text()
    )
    results = document["results"]
    assert document["status"] == "held"
    assert results["first_to_phase_point_arrival_error_change_s"] == 324.0
    assert results["phase_point_to_continuous_arrival_error_change_s"] == -56.0
    assert results["phase_point_to_continuous_altitude_rmse_improvement_fraction"] < 0.0
    assert results["final_spatial_altitude_rmse_m"] < 200.0
    assert results["final_time_aligned_altitude_rmse_m"] > 350.0
    assert abs(results["final_arrival_time_error_s"]) < 120.0
    assert results["final_cas_rmse_mps"] < 8.0
    assert document["decision"]["closed_loop_release_authorized"] is False


def test_mode_conditioned_integration_uses_generated_mass_not_qar_response() -> None:
    final = yaml.safe_load(
        (FMS_DOCS / "a320-fms-mode-conditioned-mass-intent.yml").read_text()
    )
    speed = final["intent_boundary"]["mode_conditioned_speed_intent"]
    assert speed["actual_CAS_or_Mach_used_to_form_schedule"] is False
    assert speed["inferred_speed_mode_used_to_form_schedule"] is False
    assert "first-pass Aerox predicted fuel history" in speed["planning_mass_source"]
    consistency = speed["evaluation_points"]["schedule_consistency"]
    assert (
        consistency["applied_low_altitude_cas_kts"]
        <= final["intent_boundary"]["inferred_descent_schedule"]["cas_kts"]
    )
    continuous = yaml.safe_load(
        (FMS_DOCS / "a320-fms-continuous-speed-mass.yml").read_text()
    )
    plan = continuous["energy_plan"]["continuous_speed_mass_plan"]
    assert continuous["energy_plan"]["geometry_mapped_profile_feasible"] is True
    assert plan["QAR_response_used_to_form_targets"] is False
    assert plan["QAR_fuel_or_mass_used_in_iteration"] is False
    assert plan["maximum_mass_update_kg"] < 1.0
    assert plan["maximum_energy_violation_mps2"] == 0.0


def test_mode_conditioned_integration_artifacts_cover_all_four_cases() -> None:
    metrics = pd.read_csv(
        FMS_DOCS / "a320-fms-mode-conditioned-integration-metrics.csv"
    )
    assert set(metrics["case"]) == {
        "flown-median baseline",
        "initial-mass surrogate",
        "mass-iterated surrogate",
        "continuous speed + mass",
    }
    assert (FMS_DOCS / "figures/a320-fms-mode-conditioned-integration.png").is_file()
    assert (FMS_DOCS / "figures/a320-fms-mode-conditioned-integration.pdf").is_file()


def test_explicit_commanded_mach_improves_long_trajectory_but_remains_held() -> None:
    command = yaml.safe_load((FMS_DOCS / "a320-fms-commanded-mach.yml").read_text())
    comparison = yaml.safe_load(
        (FMS_DOCS / "a320-fms-cruise-strategy-comparison.yml").read_text()
    )
    strategy = command["intent_boundary"]["cruise_speed_strategy"]
    result = comparison["results"]

    assert strategy["kind"] == "commanded_mach"
    assert strategy["commanded_mach"] == 0.78
    assert strategy["actual_QAR_speed_used_to_form_target"] is False
    assert command["status"] == "held"
    assert result["time_aligned_horizontal_rmse_improvement_fraction"] > 0.60
    assert result["time_aligned_altitude_rmse_improvement_fraction"] > 0.25
    assert result["absolute_arrival_error_improvement_fraction"] > 0.60
    assert result["commanded_mach_held_gates"] == [
        "time_aligned_horizontal",
        "altitude",
    ]
    assert result["commanded_mach_release_authorized"] is False


def test_explicit_cruise_strategy_comparison_artifacts_exist() -> None:
    metrics = pd.read_csv(FMS_DOCS / "a320-fms-cruise-strategy-comparison.csv")
    assert set(metrics["strategy"]) == {"phase_conditioned", "commanded_mach"}
    assert (FMS_DOCS / "figures/a320-fms-cruise-strategy-comparison.png").is_file()
    assert (FMS_DOCS / "figures/a320-fms-cruise-strategy-comparison.pdf").is_file()
