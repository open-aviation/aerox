import numpy as np
import pytest

from aerox import load_aircraft
from aerox.fms import CruiseSpeedStrategy, EconomyCruiseOptimizer
from aerox.models import (
    KornLockWaveDrag,
    ParabolicDragPolar,
    ParabolicWaveDragModel,
    build_drag_model,
)
from aerox.models.openap import OpenAPFuelFlowPrior

class _SpeedLimitedMaximumThrust:
    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        del pressure_altitude_m, static_temperature_k
        return 100_000.0 if true_airspeed_mps < 235.0 else 1_000.0


class _InvalidFuelFlow:
    def fuel_flow_kg_s(self, total_aircraft_thrust_n: np.ndarray) -> np.ndarray:
        return np.full_like(total_aircraft_thrust_n, np.nan)


def test_wave_drag_matches_openap_gur_equations() -> None:
    model = KornLockWaveDrag(np.deg2rad(25.0), 0.12)
    lift = np.array([0.50, 0.55])
    mach = np.array([0.70, 0.80])
    cosine = np.cos(np.deg2rad(25.0))
    expected_critical = (
        0.95 / cosine - 0.12 / cosine**2 - 0.1 * lift / cosine**3 - 0.108
    )
    expected_wave = 20.0 * np.maximum(mach - expected_critical, 0.0) ** 4
    evaluated_wave, evaluated_critical = model.evaluate(mach, lift)

    np.testing.assert_allclose(model.critical_mach(lift), expected_critical)
    np.testing.assert_allclose(
        model.drag_coefficient_increment(mach, lift), expected_wave
    )
    np.testing.assert_allclose(evaluated_critical, expected_critical)
    np.testing.assert_allclose(evaluated_wave, expected_wave)


def test_economy_optimizer_has_interior_solution_and_headwind_speeds_up() -> None:
    fuel = OpenAPFuelFlowPrior(
        engine_count=2,
        rated_thrust_n_per_engine=60_000.0,
        c1=0.7,
        c2=1.2,
        c3=0.3,
    )
    optimizer = EconomyCruiseOptimizer(
        drag_model=ParabolicWaveDragModel(
            ParabolicDragPolar(0.02263, 0.04246),
            KornLockWaveDrag(np.deg2rad(25.0), 0.12),
        ),
        fuel_flow_model=fuel,
        wing_area_m2=122.0,
    )

    still_air = optimizer.optimize(
        mass_kg=63_000.0,
        pressure_altitude_m=11_275.0,
        static_temperature_k=216.65,
    )
    headwind = optimizer.optimize(
        mass_kg=63_000.0,
        pressure_altitude_m=11_275.0,
        static_temperature_k=216.65,
        along_track_wind_mps=-20.0,
    )

    assert 0.74 < still_air.mach < 0.81
    assert still_air.critical_mach < still_air.mach
    assert still_air.wave_drag_coefficient > 0.0
    assert still_air.total_drag_n > 0.0
    assert headwind.mach > still_air.mach
    assert still_air.fuel_per_ground_distance_kg_m > 0.0
    assert still_air.mach == pytest.approx(
        optimizer.optimize(
            mass_kg=63_000.0,
            pressure_altitude_m=11_275.0,
            static_temperature_k=216.65,
        ).mach
    )
    with pytest.raises(ValueError, match="explicit experimental opt-in"):
        CruiseSpeedStrategy(kind="economy", economy_optimizer=optimizer)
    strategy = CruiseSpeedStrategy(
        kind="economy",
        economy_optimizer=optimizer,
        experimental_economy_opt_in=True,
    )
    assert strategy.kind == "economy"


def test_economy_optimizer_without_wave_drag_reaches_upper_bound() -> None:
    fuel = OpenAPFuelFlowPrior(
        engine_count=2,
        rated_thrust_n_per_engine=60_000.0,
        c1=0.7,
        c2=1.2,
        c3=0.3,
    )
    optimizer = EconomyCruiseOptimizer(
        drag_model=ParabolicDragPolar(0.02263, 0.04246),
        fuel_flow_model=fuel,
        wing_area_m2=122.0,
    )

    point = optimizer.optimize(
        mass_kg=63_000.0,
        pressure_altitude_m=11_275.0,
        static_temperature_k=216.65,
    )

    assert point.mach == pytest.approx(optimizer.maximum_mach)
    assert np.isnan(point.critical_mach)
    assert point.wave_drag_coefficient == 0.0


def test_economy_optimizer_enforces_available_thrust() -> None:
    fuel = OpenAPFuelFlowPrior(2, 60_000.0, 0.7, 1.2, 0.3)
    optimizer = EconomyCruiseOptimizer(
        drag_model=ParabolicDragPolar(0.02263, 0.04246),
        fuel_flow_model=fuel,
        wing_area_m2=122.0,
        maximum_thrust_model=_SpeedLimitedMaximumThrust(),
    )

    point = optimizer.optimize(
        mass_kg=63_000.0,
        pressure_altitude_m=11_275.0,
        static_temperature_k=216.65,
    )

    assert point.true_airspeed_mps < 235.0
    assert point.thrust_margin_n >= 0.0


def test_economy_optimizer_rejects_invalid_fuel_model_output() -> None:
    optimizer = EconomyCruiseOptimizer(
        drag_model=ParabolicDragPolar(0.02263, 0.04246),
        fuel_flow_model=_InvalidFuelFlow(),
        wing_area_m2=122.0,
    )

    with pytest.raises(ValueError, match="finite value per Mach"):
        optimizer.optimize(
            mass_kg=63_000.0,
            pressure_altitude_m=11_275.0,
            static_temperature_k=216.65,
        )


def test_economy_optimizer_accepts_active_condition_aware_drag_model() -> None:
    fuel = OpenAPFuelFlowPrior(
        engine_count=2,
        rated_thrust_n_per_engine=60_000.0,
        c1=0.7,
        c2=1.2,
        c3=0.3,
    )
    aircraft = load_aircraft("A320")
    optimizer = EconomyCruiseOptimizer(
        drag_model=build_drag_model(aircraft),
        fuel_flow_model=fuel,
        wing_area_m2=122.0,
    )

    point = optimizer.optimize(
        mass_kg=63_000.0,
        pressure_altitude_m=11_275.0,
        static_temperature_k=216.65,
    )

    assert optimizer.minimum_mach < point.mach < optimizer.maximum_mach
    assert point.wave_drag_coefficient > 0.0
    assert np.isfinite(point.critical_mach)
