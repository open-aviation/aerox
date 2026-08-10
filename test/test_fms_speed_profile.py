import numpy as np
import pytest

from aerox import aero
from aerox.fms import (
    CruiseSpeedStrategy,
    CruiseTarget,
    FlightIntent,
    LateralWaypoint,
    ModeConditionedSpeedIntent,
    SpeedTarget,
    SpatialEnvironmentProfile,
    SpatialTrajectory,
    TerminalConstraint,
    plan_continuous_speed_mass_profile,
)
from aerox.modeling import PointMassProfilePerformance
from aerox.modeling.dynamics import PointMassModel
from aerox.models import EffectiveLiftMap, ParabolicDragPolar
from aerox.models.openap import OpenAPFuelFlowPrior
from aerox.models.thrust import ConstantMaximumThrust


class _LinearFuelFlow:
    def fuel_flow_scalar(self, total_aircraft_thrust_n: float) -> float:
        return 0.1 + total_aircraft_thrust_n / 100_000.0


def _performance() -> PointMassProfilePerformance:
    plant = PointMassModel(
        wing_area_m2=50.0,
        lift_map=EffectiveLiftMap("linear_alpha", np.array([0.5, 0.1])),
        drag_model=ParabolicDragPolar(0.02, 0.04),
        fuel_flow_prior=OpenAPFuelFlowPrior(2, 50_000.0, 1.0, 2.0, 1.0),
    )
    return PointMassProfilePerformance(
        plant=plant,
        maximum_thrust_model=ConstantMaximumThrust(12_000.0),
        minimum_thrust_n=0.0,
        maximum_climb_angle_rad=np.deg2rad(5.0),
        maximum_descent_angle_rad=np.deg2rad(5.0),
        maximum_acceleration_mps2=0.3,
        maximum_deceleration_mps2=0.3,
    )


def test_continuous_speed_plan_iterates_fuel_mass_without_response_inputs() -> None:
    route = SpatialTrajectory.from_intent(
        FlightIntent(
            waypoints=(
                LateralWaypoint(0.0, 0.0, "START"),
                LateralWaypoint(30_000.0, 0.0, "TOC"),
                LateralWaypoint(70_000.0, 0.0, "TOD"),
                LateralWaypoint(100_000.0, 0.0, "END"),
            ),
            departure=TerminalConstraint(1_000.0, SpeedTarget("TAS", 100.0)),
            arrival=TerminalConstraint(1_000.0, SpeedTarget("TAS", 100.0)),
            cruise_target=CruiseTarget(1, 2, 2_000.0, SpeedTarget("TAS", 130.0)),
            maximum_climb_flight_path_angle_rad=np.deg2rad(5.0),
            maximum_descent_flight_path_angle_rad=np.deg2rad(5.0),
        )
    )
    environment = SpatialEnvironmentProfile(
        along_path_m=np.array([0.0, route.total_distance_m]),
        static_temperature_k=np.array([aero.T0, aero.T0]),
        wind_north_mps=np.array([5.0, 5.0]),
        wind_east_mps=np.zeros(2),
        wind_down_mps=np.zeros(2),
    )
    model = ModeConditionedSpeedIntent(
        structure="selector",
        cas_coefficients=np.empty(0),
        mach_coefficients=np.empty(0),
    )

    result = plan_continuous_speed_mass_profile(
        route=route,
        speed_intent=model,
        selector_along_path_m=np.array([0.0, route.total_distance_m]),
        selected_cas_kts=np.array([250.0, 250.0]),
        selected_mach=np.array([0.7, 0.7]),
        top_of_climb_along_path_m=30_000.0,
        top_of_descent_along_path_m=70_000.0,
        performance=_performance(),
        fuel_flow_model=_LinearFuelFlow(),
        initial_mass_kg=10_000.0,
        environment_profile=environment,
        spacing_m=500.0,
        cruise_strategy=CruiseSpeedStrategy(kind="commanded_mach", commanded_mach=0.40),
    )

    assert result.iteration_count > 1
    assert result.maximum_mass_update_kg <= 1.0
    assert np.all(np.diff(result.planned_mass_kg) <= 0.0)
    assert result.planned_mass_kg[-1] < result.planned_mass_kg[0]
    assert result.planned_true_airspeed_mps[0] == pytest.approx(100.0)
    assert result.planned_true_airspeed_mps[-1] == pytest.approx(100.0)
    assert result.maximum_energy_violation_mps2 < 2e-3
    assert result.minimum_energy_violation_mps2 < 2e-3
    level = result.phase == "LEVEL"
    assert result.cruise_strategy == "commanded_mach"
    assert np.all(result.active_kind[level] == "MACH")
    assert np.all(result.mach_target[level] == pytest.approx(0.40))
    assert np.all(np.isnan(result.economy_cruise_target_mach))
    assert np.any(
        np.abs(result.planned_true_airspeed_mps - result.desired_true_airspeed_mps)
        > 0.1
    )


def test_economy_cruise_strategy_requires_explicit_experimental_opt_in() -> None:
    with pytest.raises(ValueError, match="requires an economy optimizer"):
        CruiseSpeedStrategy(kind="economy", experimental_economy_opt_in=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "commanded_mach", "commanded_mach": 0.95}, "cruise Mach"),
        ({"kind": "commanded_cas", "commanded_cas_kts": 90.0}, "cruise CAS"),
        ({"kind": "phase_conditioned", "commanded_mach": 0.78}, "only valid"),
    ],
)
def test_cruise_speed_strategy_rejects_inconsistent_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CruiseSpeedStrategy(**kwargs)
