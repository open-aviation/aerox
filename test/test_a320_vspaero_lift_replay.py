from pathlib import Path

import yaml

MODELING = Path("docs/modeling")


def test_strict_surface_replay_holds_on_coverage_and_altitude():
    replay = yaml.safe_load(
        (MODELING / "a320-vspaero-lift-surface-replay.yml").read_text()
    )

    assert not replay["decision"]["controlled_replay_integration_passed"]
    assert replay["gates"]["validation_window_speed_rmse_ratio"]["passed"]
    assert not replay["gates"]["validation_window_altitude_rmse_ratio"]["passed"]
    assert not replay["gates"]["surface_window_failure_fraction"]["passed"]
    assert not replay["decision"]["make_surface_default_lift_model"]


def test_envelope_audit_attributes_hold_to_recorded_support():
    audit = yaml.safe_load(
        (MODELING / "a320-vspaero-replay-envelope-audit.yml").read_text()
    )

    assert audit["diagnosis"]["dominant_recorded_boundary"] == "mach_high_samples"
    assert audit["diagnosis"]["recorded_coverage_gap"]
    assert not audit["diagnosis"]["simulated_state_escape_beyond_recorded_gap"]
    assert audit["decision"]["expand_mach_support_before_default_replay"]
    assert not audit["decision"]["allow_unbounded_extrapolation"]


def test_guarded_hybrid_restores_execution_but_remains_held():
    hybrid = yaml.safe_load((MODELING / "a320-vspaero-hybrid-replay.yml").read_text())

    assert hybrid["gates"]["hybrid_window_failure_fraction"]["passed"]
    assert hybrid["gates"]["hybrid_validation_full_successes"]["passed"]
    assert (
        hybrid["validation_hybrid_to_empirical_ratios"]["window_speed_rmse_mps"] < 1.0
    )
    assert not hybrid["gates"]["validation_window_altitude_rmse_ratio"]["passed"]
    assert not hybrid["gates"]["validation_recorded_mean_surface_weight"]["passed"]
    assert not hybrid["decision"]["make_guarded_hybrid_clean_replay_default"]


def test_replay_diagnostic_figures_are_exported():
    for stem in (
        "a320-vspaero-lift-surface-replay",
        "a320-vspaero-replay-envelope-audit",
        "a320-vspaero-hybrid-replay",
    ):
        for suffix in (".png", ".pdf"):
            assert (MODELING / "figures" / f"{stem}{suffix}").stat().st_size > 1_000
