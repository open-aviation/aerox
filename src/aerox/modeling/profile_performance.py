"""Point-mass aircraft adapter for FMS profile-performance limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox.fms.performance import (
    LocalPerformanceLimits,
    MaximumThrustModel,
    MinimumThrustModel,
)
from aerox.models._atmosphere import (
    G0,
    atmosphere_at_pressure_altitude,
    unit_reynolds_number,
)
from aerox.modeling.dynamics import PointMassModel


@dataclass(frozen=True)
class PointMassProfilePerformance:
    """Reduce a pitch-bank-thrust model to conservative planning limits."""

    plant: PointMassModel
    maximum_thrust_model: MaximumThrustModel
    minimum_thrust_n: float
    maximum_climb_angle_rad: float
    maximum_descent_angle_rad: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    minimum_thrust_model: MinimumThrustModel | None = None

    def __post_init__(self) -> None:
        values = (
            self.minimum_thrust_n,
            self.maximum_climb_angle_rad,
            self.maximum_descent_angle_rad,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("profile-performance settings must be finite")
        if self.minimum_thrust_n < 0.0:
            raise ValueError("minimum thrust must be non-negative")
        if min(values[1:]) <= 0.0:
            raise ValueError("profile-performance caps must be positive")

    def limits(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        mass_kg: float,
        static_temperature_k: float,
    ) -> LocalPerformanceLimits:
        values = (
            pressure_altitude_m,
            true_airspeed_mps,
            mass_kg,
            static_temperature_k,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("profile-performance condition must be finite")
        if true_airspeed_mps <= 1.0 or mass_kg <= 0.0 or static_temperature_k <= 0.0:
            raise ValueError("speed, mass, and temperature must be positive")
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([pressure_altitude_m]), np.array([static_temperature_k])
        )
        density = float(atmosphere.density_kg_m3[0])
        dynamic_pressure = 0.5 * density * true_airspeed_mps**2
        lift_coefficient = mass_kg * G0 / (dynamic_pressure * self.plant.wing_area_m2)
        mach = true_airspeed_mps / float(atmosphere.speed_of_sound_mps[0])
        reynolds_per_metre = float(
            unit_reynolds_number(density, true_airspeed_mps, static_temperature_k)
        )
        drag_coefficient = self.plant.drag_coefficient_scalar(
            lift_coefficient,
            mach,
            reynolds_per_m=reynolds_per_metre,
        )
        drag = dynamic_pressure * self.plant.wing_area_m2 * drag_coefficient
        maximum_thrust = self.maximum_thrust_model.maximum_thrust_n(
            pressure_altitude_m=pressure_altitude_m,
            true_airspeed_mps=true_airspeed_mps,
            static_temperature_k=static_temperature_k,
        )
        minimum_thrust = (
            self.minimum_thrust_model.minimum_thrust_n(
                pressure_altitude_m=pressure_altitude_m,
                true_airspeed_mps=true_airspeed_mps,
                static_temperature_k=static_temperature_k,
            )
            if self.minimum_thrust_model is not None
            else self.minimum_thrust_n
        )
        if not np.isfinite(minimum_thrust) or minimum_thrust < 0.0:
            raise ValueError(
                "conditioned minimum thrust must be finite and non-negative"
            )
        if not np.isfinite(maximum_thrust) or maximum_thrust <= minimum_thrust:
            raise ValueError("conditioned maximum thrust must exceed minimum thrust")
        raw_maximum_acceleration = (maximum_thrust - drag) / mass_kg
        raw_maximum_deceleration = (drag - minimum_thrust) / mass_kg
        raw_climb_sine = (maximum_thrust - drag) / (mass_kg * G0)
        raw_descent_sine = (drag - minimum_thrust) / (mass_kg * G0)
        maximum_climb = min(
            self.maximum_climb_angle_rad,
            float(np.arcsin(np.clip(raw_climb_sine, 0.0, 0.95))),
        )
        maximum_descent = min(
            self.maximum_descent_angle_rad,
            float(np.arcsin(np.clip(raw_descent_sine, 0.0, 0.95))),
        )
        return LocalPerformanceLimits(
            maximum_thrust_n=float(maximum_thrust),
            level_drag_n=float(drag),
            maximum_climb_angle_rad=maximum_climb,
            maximum_descent_angle_rad=maximum_descent,
            maximum_acceleration_mps2=max(
                0.0,
                min(self.maximum_acceleration_mps2, raw_maximum_acceleration),
            ),
            maximum_deceleration_mps2=max(
                0.0,
                min(self.maximum_deceleration_mps2, raw_maximum_deceleration),
            ),
            maximum_specific_excess_force_mps2=float(raw_maximum_acceleration),
            minimum_specific_excess_force_mps2=float((minimum_thrust - drag) / mass_kg),
        )


__all__ = ["PointMassProfilePerformance"]
