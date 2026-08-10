from pathlib import Path

import pandas as pd
import yaml

from tools.generate_a320_fms_trajectory import build_intent
from tools.select_a320_fms_turn_aware_intent import select_turn_aware_indices

FMS_DOCS = Path("docs/fms")
MODELING_DOCS = Path("docs/modeling")


def test_a320_fms_intent_is_spatial_and_constrained() -> None:
    intent = build_intent()

    assert len(intent.waypoints) == 5
    assert intent.waypoints[0].name == "ORIGIN"
    assert intent.waypoints[-1].name == "DESTINATION"
    assert intent.waypoints[-1].acceptance_radius_m == 1_500.0
    assert intent.departure.altitude_m > 0.0
    assert intent.arrival.altitude_m > 0.0
    assert intent.departure.speed.value > 1.0
    assert intent.arrival.speed.value > 1.0
    assert len(intent.altitude_constraints) == 3
    assert len(intent.speed_constraints) == 3


def test_long_range_holdout_is_frozen_and_training_spans_its_altitude() -> None:
    document = yaml.safe_load(
        Path("docs/modeling/a320-long-range-holdout-selection.yml").read_text()
    )
    coverage = pd.read_csv("docs/modeling/a320-training-altitude-coverage.csv")
    selected = document["selected_holdout"]

    assert document["status"] == "selected-response-still-closed"
    assert selected["registration"] == "TAIL-A"
    assert selected["recording"] == "FLT0003"
    assert selected["airborne_duration_h"] > 4.0
    assert selected["maximum_altitude_ft"] >= 37_000.0
    assert selected["previously_consumed"] is False
    assert all(document["gates"].values())
    relevant = coverage.loc[coverage["relevant_to_selected_test"]]
    assert (relevant["fit_flight_count"] >= 2).all()
    assert (relevant["fit_registration_count"] == 2).all()
    for suffix in (".png", ".pdf"):
        path = (
            Path("docs/modeling/figures") / f"a320-long-range-holdout-selection{suffix}"
        )
        assert path.is_file() and path.stat().st_size > 1_000


def test_long_range_sparse_intent_artifacts_preserve_independent_result() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-long-range-qar-intent.yml").read_text()
    )
    errors = pd.read_csv(FMS_DOCS / "a320-fms-long-range-qar-intent-errors.csv")
    history = pd.read_csv(FMS_DOCS / "a320-fms-long-range-qar-intent-history.csv")
    metrics = document["metrics"]

    assert document["source"]["recording"] == "TAIL-A/FLT0003"
    assert document["source"]["response_opened_only_after_selection_was_frozen"]
    assert document["source"]["maximum_reference_altitude_ft"] >= 37_000.0
    assert document["decision"]["independent_aircraft_model_holdout"]
    assert document["decision"]["training_altitude_coverage_audited_before_response"]
    assert document["decision"]["airbus_fms_autopilot_or_fadec_identified"] is False
    assert document["intent_extraction"]["anchor_interval_s"] == 600
    assert document["intent_extraction"]["qar_heading_used_by_guidance"] is False
    assert document["numerics"] == {
        "command_rate_hz": 1,
        "integration_rate_hz": 10,
        "profile_spacing_m": 500.0,
    }
    assert metrics["completed"]
    assert metrics["generated_duration_s"] > 13_000.0
    assert metrics["maximum_mach"] <= 0.82
    assert metrics["maximum_alpha_deg"] <= 8.5
    assert document["gates"]["controller_limiters"]["passed"]
    assert document["gates"]["destination"]["passed"]
    assert document["gates"]["speed"]["passed"]
    assert document["status"] == "held"
    assert not document["gates"]["qar_to_sparse_intent_path"]["passed"]
    assert set(errors) == {
        "elapsed_time_s",
        "north_error_m",
        "east_error_m",
        "horizontal_error_m",
        "altitude_error_m",
        "true_airspeed_error_mps",
    }
    assert len(history) > 13_000
    assert {"effective_alpha_deg", "net_thrust_kn", "thrust_limited"} <= set(history)
    for suffix in (".png", ".pdf"):
        path = (
            FMS_DOCS / "figures" / f"a320-fms-long-range-qar-intent-comparison{suffix}"
        )
        assert path.is_file() and path.stat().st_size > 1_000


def test_turn_aware_selector_adds_geometry_anchor_without_losing_time_grid() -> None:
    north = pd.Series([0.0, 0.0, 0.0, 1_000.0, 2_000.0]).to_numpy()
    east = pd.Series([0.0, 1_000.0, 2_000.0, 2_000.0, 2_000.0]).to_numpy()

    selected, mandatory = select_turn_aware_indices(
        north,
        east,
        maximum_interval_s=4,
        geometry_tolerance_m=500.0,
    )

    assert mandatory.tolist() == [0, 4]
    assert selected.tolist() == [0, 2, 4]


def test_turn_aware_intent_selection_is_frozen_before_rollout() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-turn-aware-intent-selection.yml").read_text()
    )
    anchors = pd.read_csv(FMS_DOCS / "a320-fms-turn-aware-intent-anchors.csv")

    assert document["status"] == "selected-before-adaptive-rollout"
    assert document["source"]["adaptive_rollout_response_opened"] is False
    assert document["selection_rule"][
        "uses_only_zero-referenced_horizontal_path_geometry"
    ]
    assert (
        document["selection_rule"][
            "uses_altitude_speed_attitude_control_or_model_errors"
        ]
        is False
    )
    assert document["metrics"]["adaptive_anchor_count"] == len(anchors)
    assert document["metrics"]["added_geometry_anchor_count"] > 0
    assert (
        document["metrics"]["adaptive_path_rmse_m"]
        < 0.1 * document["metrics"]["base_path_rmse_m"]
    )
    assert all(document["gates"].values())
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-turn-aware-intent-selection{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_turn_aware_rollout_separates_geometry_gain_from_remaining_holds() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-turn-aware-qar-intent.yml").read_text()
    )
    errors = pd.read_csv(FMS_DOCS / "a320-fms-turn-aware-qar-intent-errors.csv")
    history = pd.read_csv(FMS_DOCS / "a320-fms-turn-aware-qar-intent-history.csv")
    metrics = document["metrics"]
    comparison = document["comparison_to_uniform_600_s"]

    assert document["source"]["post_holdout_diagnostic"]
    assert document["source"]["adaptive_selection_frozen_before_this_rollout"]
    assert document["source"]["model_controller_or_gate_changed"] is False
    assert document["decision"]["new_independent_holdout_claimed"] is False
    assert document["decision"]["geometry_refinement_improved_generated_path_rmse"]
    assert metrics["completed"]
    assert metrics["anchor_count"] == 43
    assert comparison["generated_to_qar_path_rmse_m"]["fractional_improvement"] > 0.94
    assert comparison["altitude_rmse_m"]["fractional_improvement"] > 0.30
    assert document["gates"]["generated_to_qar_path"]["passed"]
    assert document["gates"]["altitude"]["passed"]
    assert document["gates"]["time_aligned_horizontal"]["passed"] is False
    assert document["gates"]["controller_limiters"]["passed"] is False
    assert document["status"] == "held"
    assert len(errors) > 13_000
    assert len(history) > 13_000
    for stem in (
        "a320-fms-turn-aware-qar-intent-comparison",
        "a320-fms-turn-aware-before-after",
    ):
        for suffix in (".png", ".pdf"):
            path = FMS_DOCS / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000


def test_decoupled_geometry_and_state_intent_isolates_profile_hold() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-decoupled-qar-intent.yml").read_text()
    )
    anchors = pd.read_csv(FMS_DOCS / "a320-fms-decoupled-anchor-states.csv")
    metrics = document["metrics"]
    architecture = document["intent_architecture"]

    assert document["source"]["post_holdout_diagnostic"]
    assert document["source"]["model_controller_limits_or_gates_changed"] is False
    assert document["decision"]["geometry_state_constraint_decoupling_implemented"]
    assert document["decision"]["new_independent_holdout_claimed"] is False
    assert architecture["horizontal_geometry_anchor_count"] == 43
    assert architecture["observed_altitude_speed_constraint_count"] == 25
    assert architecture["geometry_only_anchor_count"] == 18
    assert architecture["geometry_only_qar_altitude_speed_exposed"] is False
    assert (anchors["constraint_class"] == "observed-state").sum() == 25
    assert (anchors["constraint_class"] == "geometry-only").sum() == 18
    assert metrics["generated_to_qar_path_rmse_m"] < 250.0
    assert metrics["time_aligned_horizontal_rmse_m"] < 6_000.0
    assert metrics["any_control_limited_fraction"] < 0.10
    assert document["gates"]["profile_feasible"]["passed"] is False
    assert all(
        gate["passed"]
        for name, gate in document["gates"].items()
        if name != "profile_feasible"
    )
    assert document["status"] == "held"
    for stem in (
        "a320-fms-decoupled-qar-intent-comparison",
        "a320-fms-decoupled-three-way",
    ):
        for suffix in (".png", ".pdf"):
            path = FMS_DOCS / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000


def test_decoupled_energy_audit_localizes_integral_capacity_deficit() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-decoupled-energy-audit.yml").read_text()
    )
    segments = pd.read_csv(FMS_DOCS / "a320-fms-decoupled-energy-segments.csv")

    assert (
        document["source"]["model_controller_thrust_boundary_or_gate_changed"] is False
    )
    assert (
        document["metrics"]["all_positive_energy_legs_have_integral_capacity"] is False
    )
    assert document["metrics"]["worst_upper_margin_state_leg"] == 1
    assert (
        segments.loc[segments["state_leg"] == 1, "integral_energy_margin_m2ps2"].iloc[0]
        < 0.0
    )
    assert (
        segments.loc[segments["state_leg"] == 2, "integral_energy_margin_m2ps2"].iloc[0]
        > 0.0
    )
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-decoupled-energy-audit{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_advisory_state_projection_clears_final_long_range_gate() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-advisory-qar-intent.yml").read_text()
    )
    anchors = pd.read_csv(FMS_DOCS / "a320-fms-advisory-anchor-states.csv")
    metrics = document["metrics"]
    projection = document["advisory_state_projection"]

    assert document["source"]["model_controller_limits_or_gates_changed"] is False
    assert document["source"]["intent_constraint_semantics_changed"]
    assert document["decision"]["advisory_decoupled_turn_aware_gate_passed"]
    assert document["decision"]["new_independent_holdout_claimed"] is False
    assert document["exact_state_profile_feasibility"]["feasible"] is False
    assert document["base_state_profile_feasibility"]["feasible"]
    assert document["remapped_profile_feasibility"]["feasible"]
    assert projection["adjusted_target_count"] == 1
    assert projection["maximum_absolute_altitude_adjustment_m"] < 200.0
    assert projection["speed_targets_changed"] is False
    observed = anchors["constraint_class"] == "observed-state"
    adjusted = anchors.loc[observed, "altitude_adjustment_m"].abs() > 1e-6
    assert observed.sum() == 25
    assert (anchors["constraint_mode"] == "hard").sum() == 2
    assert (anchors["constraint_mode"] == "advisory").sum() == 23
    assert adjusted.sum() == 1
    assert anchors.iloc[0]["altitude_adjustment_m"] == 0.0
    assert anchors.iloc[-1]["altitude_adjustment_m"] == 0.0
    assert metrics["completed"]
    assert metrics["arrival_time_error_s"] == 0.0
    assert metrics["time_aligned_horizontal_rmse_m"] < 2_500.0
    assert metrics["any_control_limited_fraction"] < 0.05
    assert all(gate["passed"] for gate in document["gates"].values())
    assert document["status"] == "passed"
    for stem in (
        "a320-fms-advisory-qar-intent-comparison",
        "a320-fms-advisory-four-way",
    ):
        for suffix in (".png", ".pdf"):
            path = FMS_DOCS / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000


def test_qar_intent_channels_seed_vertical_intent_and_energy_tod() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-qar-intent-channel-audit.yml").read_text()
    )
    timeline = pd.read_csv(FMS_DOCS / "a320-fms-qar-intent-channel-timeline.csv")
    descent = pd.read_csv(FMS_DOCS / "a320-fms-energy-tod-profile.csv")
    metrics = document["metrics"]

    assert document["source"]["active_selected_vs_managed_mode_observable"] is False
    assert document["source"]["qar_trajectory_used_for_benchmark_only"]
    assert document["intent_use"][
        "observed_CAS_and_Mach_use_recorded_static_temperature"
    ]
    assert (
        document["intent_use"]["pseudo_waypoints_create_vertical_constraints"] is False
    )
    assert (
        document["intent_use"]["SPEEDP_SEL_claimed_as_active_autothrust_target"]
        is False
    )
    assert all(value == 1.0 for value in metrics["channel_coverage_fraction"].values())
    assert metrics["cruise_selected_altitude_ft"] == 36_992.0
    assert metrics["schedule"]["climb_selected_cas_kts"] == 298.0
    assert metrics["schedule"]["descent_selected_mach"] == 0.76
    assert 0.75 < metrics["schedule"]["cruise_observed_mach"] < 0.80
    assert (
        metrics["observed_descent_distance_m"]
        > metrics["calculated_descent_distance_m"]
    )
    assert metrics["calculated_tod_path_error_m"] < 35_000.0
    assert metrics["single_flight_implied_minimum_net_thrust_n"] > 0.0
    assert metrics["single_flight_implied_thrust_is_released_idle_model"] is False
    assert set(timeline["phase"]) == {"climb", "cruise", "descent"}
    assert timeline["raw_source_time_s"].iloc[0] == 865.0
    assert timeline["selected_speed_kts"].iloc[0] == 290.0
    assert len(timeline) > 13_000
    assert len(descent) > 100
    assert (descent["distance_to_destination_m"].diff().dropna() > 0.0).all()
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-qar-intent-channel-audit{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_descent_planning_thrust_prior_passes_frozen_validation() -> None:
    document = yaml.safe_load(
        (MODELING_DOCS / "a320-descent-planning-thrust.yml").read_text()
    )
    flights = pd.read_csv(
        MODELING_DOCS / "a320-descent-planning-thrust-flight-metrics.csv"
    )
    validation = flights.loc[flights["split"].eq("validation")]

    assert document["status"] == "released-effective-descent-planning-prior"
    assert document["interpretation"]["physical_engine_idle"] is False
    assert document["interpretation"]["fadec_law"] is False
    assert document["leakage_control"]["structure_frozen_before_validation_load"]
    assert (
        document["leakage_control"]["long_flight_used_for_fit_selection_or_release"]
        is False
    )
    assert document["model"]["pressure_exponent"] == 0.5
    assert document["release_gates"]["all_passed"]
    assert document["release_gates"]["planner_integration_authorized"]
    assert len(validation) == 6
    assert validation["model_improves_distance"].all()
    assert (
        validation["model_absolute_error_km"].median()
        < 0.5 * validation["zero_thrust_absolute_error_km"].median()
    )
    for suffix in (".png", ".pdf"):
        path = MODELING_DOCS / "figures" / f"a320-descent-planning-thrust{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_a320_fms_artifacts_record_passing_closed_loop_run() -> None:
    document = yaml.safe_load((FMS_DOCS / "a320-fms-trajectory.yml").read_text())
    samples = pd.read_csv(FMS_DOCS / "a320-fms-trajectory.csv")
    fms = document["metrics"]["fms_position_and_wind_feedback"]
    baseline = document["metrics"]["leg_bearing_no_wind_correction"]

    assert document["decision"]["local_deterministic_fms_milestone_released"]
    assert all(gate["passed"] for gate in document["gates"].values())
    assert document["decision"]["airbus_fms_or_autopilot_identified"] is False
    assert set(samples["scenario"]) == {"fms", "leg_bearing_baseline"}
    assert fms["completed"]
    assert not baseline["completed"]
    assert fms["cross_track_rmse_m"] < 0.1 * baseline["cross_track_rmse_m"]
    assert fms["bank_limited_fraction"] == 0.0

    for stem in ("a320-fms-trajectory-comparison", "a320-fms-control-history"):
        for suffix in (".png", ".pdf"):
            path = FMS_DOCS / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000


def test_a320_fms_profile_feasibility_artifacts_record_passing_gate() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-profile-feasibility.yml").read_text()
    )
    samples = pd.read_csv(FMS_DOCS / "a320-fms-profile-feasibility.csv")

    assert document["decision"]["profile_feasibility_gate_passed"]
    assert document["decision"]["qar_used_for_climb_thrust_ceiling"]
    assert document["decision"]["qar_trajectory_used_as_reference"] is False
    assert document["decision"]["idle_thrust_boundary_released"] is False
    assert all(gate["passed"] for gate in document["gates"].values())
    assert len(samples) > 400
    assert (samples["maximum_climb_thrust_n"] > samples["level_drag_n"]).all()
    assert (samples["selected_altitude_m"] == samples["nominal_altitude_m"]).all()
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-profile-feasibility{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_full_sparse_qar_intent_is_preserved_as_held_diagnostic() -> None:
    document = yaml.safe_load((FMS_DOCS / "a320-fms-qar-intent.yml").read_text())
    errors = pd.read_csv(FMS_DOCS / "a320-fms-qar-intent-errors.csv")
    initialization = pd.read_csv(
        FMS_DOCS / "a320-fms-initialization-correction-history.csv"
    )

    assert document["status"] == "held"
    assert not document["decision"]["response_blind_sparse_intent_gate_passed"]
    assert document["decision"]["initialization_correction_passed"]
    assert document["decision"]["qar_heading_used_by_guidance"] is False
    assert any(
        "combined climb and acceleration" in violation
        for violation in document["profile_feasibility"]["violations"]
    )
    assert document["initialization_correction"]["controller_gains_changed"] is False
    assert document["initialization_correction"]["profile_or_intent_changed"] is False
    assert (
        document["initialization_correction"]["baseline_alpha_above_8_5_sample_count"]
        == 5
    )
    assert (
        document["initialization_correction"]["corrected_alpha_above_8_5_sample_count"]
        == 0
    )
    assert document["gates"]["alpha_envelope"]["passed"]
    assert document["gates"]["profile_feasible"]["passed"] is False
    assert document["gates"]["controller_limiters"]["passed"] is False
    assert "north_m" not in errors and "east_m" not in errors
    assert set(initialization) == {
        "elapsed_time_s",
        "reference_flight_path_angle_deg",
        "realized_flight_path_angle_deg",
        "realized_effective_alpha_deg",
        "pitch_angle_deg",
        "altitude_error_m",
        "true_airspeed_error_mps",
        "thrust_limited",
    }
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-initialization-correction{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_feasibility_screened_sparse_qar_intent_passes_without_heading() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-qar-feasible-intent.yml").read_text()
    )
    errors = pd.read_csv(FMS_DOCS / "a320-fms-qar-feasible-intent-errors.csv")

    assert document["decision"]["feasibility_screened_sparse_intent_gate_passed"]
    assert (
        document["decision"]["full_sparse_intent_gate_reclassified_as_passed"] is False
    )
    assert document["decision"]["qar_heading_used_by_guidance"] is False
    assert document["decision"]["rollout_error_used_for_span_selection"] is False
    assert document["selection"]["selected_source_sample_range"] == [720, 1422]
    assert all(gate["passed"] for gate in document["gates"].values())
    assert set(errors) == {
        "elapsed_time_s",
        "north_error_m",
        "east_error_m",
        "horizontal_error_m",
        "altitude_error_m",
        "true_airspeed_error_mps",
    }
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-qar-feasible-intent-comparison{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_sparse_qar_energy_audit_preserves_held_gate_and_attribution() -> None:
    document = yaml.safe_load((FMS_DOCS / "a320-fms-qar-energy-audit.yml").read_text())
    segments = pd.read_csv(FMS_DOCS / "a320-fms-qar-energy-segments.csv")

    assert document["diagnosis"]["full_sparse_intent_released"] is False
    assert document["source"]["model_or_gate_refitted"] is False
    assert document["source"]["raw_coordinates_persisted"] is False
    assert "demonstrated capacity" in document["diagnosis"]["primary_cause"]
    assert document["metrics"]["combined_corrected_intent_minimum_margin_mps2"] < 0
    assert (
        document["metrics"]["energy_shaped_intent_minimum_margin_mps2"]
        > document["metrics"]["current_intent_minimum_margin_mps2"]
    )
    assert document["energy_shaped_profile"]["feasible"] is False
    assert document["metrics"]["energy_shaped_intent_minimum_margin_mps2"] > -0.02
    assert document["metrics"]["openap_fast_climb_to_qar_clb_median_ratio"] < 1.0
    assert document["metrics"]["openap_fast_climb_above_qar_clb_fraction"] == 0.0
    assert document["metrics"]["negative_implied_drag_fraction"] == 0.0
    assert len(segments) == 8
    assert set(segments["segment"]) == {f"A{i}-A{i + 1}" for i in range(8)}
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-qar-energy-audit{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_alpha_energy_authority_audit_separates_startup_and_saturation() -> None:
    document = yaml.safe_load(
        (FMS_DOCS / "a320-fms-alpha-energy-authority.yml").read_text()
    )
    history = pd.read_csv(FMS_DOCS / "a320-fms-alpha-energy-authority-history.csv")
    segments = pd.read_csv(FMS_DOCS / "a320-fms-alpha-energy-authority-segments.csv")

    assert document["source"]["model_controller_threshold_or_gate_refitted"] is False
    assert document["source"]["raw_coordinates_persisted"] is False
    assert document["diagnosis"]["planner_alpha_within_support"]
    assert document["diagnosis"]["startup_only_alpha_excursion"]
    assert document["diagnosis"]["energy_deficit_present_before_closed_loop"]
    assert document["diagnosis"]["full_sparse_intent_released"] is False
    assert document["metrics"]["alpha_excursion_last_time_s"] <= 5.0
    assert (
        document["metrics"]["alpha_excursion_samples_with_thrust_limiter_fraction"]
        == 0.0
    )
    assert document["metrics"]["planned_effective_alpha_maximum_deg"] <= 8.5
    assert document["metrics"]["realized_effective_alpha_maximum_deg"] > 8.5
    assert set(history["segment"]) == {f"A{i}-A{i + 1}" for i in range(8)}
    assert len(segments) == 8
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-alpha-energy-authority{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000


def test_anchor_free_vertical_intent_generates_complete_long_trajectory() -> None:
    document = yaml.safe_load((FMS_DOCS / "a320-fms-vertical-intent.yml").read_text())
    constraints = pd.read_csv(FMS_DOCS / "a320-fms-vertical-intent-constraints.csv")
    history = pd.read_csv(FMS_DOCS / "a320-fms-vertical-intent-history.csv")
    profile = pd.read_csv(FMS_DOCS / "a320-fms-vertical-intent-profile.csv")
    comparison = pd.read_csv(
        FMS_DOCS / "a320-fms-vertical-intent-errors-trajectory.csv"
    )
    metrics = document["metrics"]
    boundary = document["intent_boundary"]

    assert document["source"]["recording"] == "TAIL-A/FLT0003"
    assert boundary["lateral_geometry_nodes"] == 43
    assert boundary["periodic_qar_altitude_speed_anchors"] == 0
    assert boundary["qar_state_boundaries"] == ["initial", "terminal"]
    assert boundary["intermediate_qar_states_used_by_planner_or_guidance"] is False
    assert boundary["observed_heading_or_recorded_controls_used_by_guidance"] is False
    assert document["energy_plan"]["TOC_or_TOD_copied_from_QAR"] is False
    cruise_mach = boundary.get(
        "inferred_cruise_mach", boundary.get("selected_cruise_mach")
    )
    assert cruise_mach is not None and 0.75 < cruise_mach < 0.80
    assert document["energy_plan"]["hard_vertical_constraint_waypoint_indices"] == [
        0,
        42,
    ]
    assert metrics["completed"]
    assert metrics["anchor_count"] == 2
    assert metrics["generated_duration_s"] > 13_000.0
    assert metrics["terminal_horizontal_error_m"] < 1_500.0
    assert abs(metrics["terminal_altitude_error_m"]) < 100.0
    assert abs(metrics["terminal_speed_error_mps"]) < 3.0
    assert document["gates"]["profile_feasible"]["passed"] is False
    assert document["gates"]["time_aligned_horizontal"]["passed"] is False
    assert document["gates"]["controller_limiters"]["passed"] is False
    assert document["status"] == "held"
    assert list(constraints["name"]) == [
        "INITIAL",
        "CALCULATED-TOC",
        "CALCULATED-TOD",
        "TERMINAL",
    ]
    assert len(history) > 13_000
    assert {"calibrated_airspeed_mps", "mach"} <= set(profile)
    assert {
        "qar_calibrated_airspeed_mps",
        "qar_mach",
        "generated_calibrated_airspeed_mps",
        "generated_mach",
    } <= set(comparison)
    assert metrics["calibrated_airspeed_rmse_mps"] < 10.0
    assert metrics["mach_rmse"] < 0.05
    for suffix in (".png", ".pdf"):
        path = FMS_DOCS / "figures" / f"a320-fms-vertical-intent-comparison{suffix}"
        assert path.is_file() and path.stat().st_size > 1_000
