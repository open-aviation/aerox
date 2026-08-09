"""Inspectible cruise-economy primitives with experimental wave drag.

The compressibility correction follows OpenAP commit
``472ab3e48d52ae62f9586d2e75e143ea4182a9e9`` and its cited implementation of
Gur, Mason, and Schetz (2010), DOI 10.2514/1.47557. OpenAP is licensed under
GNU LGPL v3. The model is deliberately opt-in because wing sweep and thickness
ratio may be uncertain and the correction is not an aircraft-specific CFD or
wind-tunnel calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox.models.atmosphere import (
    G0,
    atmosphere_at_pressure_altitude,
    unit_reynolds_number,
)
from aerox.models.drag import DragCondition, DragModel

FloatArray = NDArray[np.float64]


class CruiseFuelFlowModel(Protocol):
    """Total-aircraft fuel flow evaluated at level-flight net thrust."""

    def fuel_flow_kg_s(self, total_aircraft_thrust_n: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True)
class EconomyCruisePoint:
    """One minimum-fuel-per-ground-distance cruise solution."""

    mach: float
    true_airspeed_mps: float
    ground_speed_mps: float
    lift_coefficient: float
    critical_mach: float
    wave_drag_coefficient: float
    total_drag_n: float
    fuel_flow_kg_s: float
    fuel_per_ground_distance_kg_m: float


@dataclass(frozen=True)
class EconomyCruiseOptimizer:
    """Grid-search an ECON cruise target with one physical drag model."""

    drag_model: DragModel
    fuel_flow_model: CruiseFuelFlowModel
    wing_area_m2: float
    minimum_mach: float = 0.68
    maximum_mach: float = 0.81
    mach_step: float = 0.0005

    def __post_init__(self) -> None:
        values = (
            self.wing_area_m2,
            self.minimum_mach,
            self.maximum_mach,
            self.mach_step,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("economy optimizer parameters must be finite")
        if self.wing_area_m2 <= 0.0 or self.mach_step <= 0.0:
            raise ValueError("wing area and Mach step must be positive")
        if not 0.0 < self.minimum_mach < self.maximum_mach < 1.0:
            raise ValueError("economy Mach bounds must be ordered below one")

    def _drag_components(
        self,
        lift_coefficient: FloatArray,
        mach: FloatArray,
        unit_reynolds_number_per_m: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return total, wave, and critical-Mach arrays from shared models."""

        condition = DragCondition(
            mach=mach,
            unit_reynolds_number_per_m=unit_reynolds_number_per_m,
        )
        total = self.drag_model.drag_coefficient_at(lift_coefficient, condition)
        breakdown_at = getattr(self.drag_model, "breakdown_at", None)
        if breakdown_at is None:
            shape = np.broadcast_shapes(lift_coefficient.shape, mach.shape)
            return (
                total,
                np.full(shape, np.nan, dtype=float),
                np.full(shape, np.nan, dtype=float),
            )
        breakdown = breakdown_at(lift_coefficient, condition)
        wave = getattr(breakdown, "wave", np.full_like(total, np.nan))
        critical = getattr(
            breakdown,
            "critical_mach",
            np.full_like(total, np.nan),
        )
        return total, np.asarray(wave, dtype=float), np.asarray(critical, dtype=float)

    def optimize(
        self,
        *,
        mass_kg: float,
        pressure_altitude_m: float,
        static_temperature_k: float,
        along_track_wind_mps: float = 0.0,
    ) -> EconomyCruisePoint:
        """Minimize calibrated fuel flow per planned ground distance."""

        values = (
            mass_kg,
            pressure_altitude_m,
            static_temperature_k,
            along_track_wind_mps,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("economy planning condition must be finite")
        if mass_kg <= 0.0 or static_temperature_k <= 0.0:
            raise ValueError("mass and temperature must be positive")
        count = int(np.ceil((self.maximum_mach - self.minimum_mach) / self.mach_step))
        mach = np.linspace(self.minimum_mach, self.maximum_mach, count + 1)
        atmosphere = atmosphere_at_pressure_altitude(
            np.asarray([pressure_altitude_m]),
            np.asarray([static_temperature_k]),
        )
        speed = mach * float(atmosphere.speed_of_sound_mps[0])
        density = float(atmosphere.density_kg_m3[0])
        reynolds_per_metre = unit_reynolds_number(
            density,
            speed,
            static_temperature_k,
        )
        dynamic_pressure_area = 0.5 * density * speed**2 * self.wing_area_m2
        lift = mass_kg * G0 / dynamic_pressure_area
        drag_coefficient, wave, critical_mach = self._drag_components(
            lift,
            mach,
            reynolds_per_metre,
        )
        drag = dynamic_pressure_area * drag_coefficient
        fuel_flow = np.asarray(self.fuel_flow_model.fuel_flow_kg_s(drag), dtype=float)
        ground_speed = speed + along_track_wind_mps
        if np.any(ground_speed <= 30.0):
            raise ValueError("economy candidate ground speed is non-positive")
        objective = fuel_flow / ground_speed
        index = int(np.argmin(objective))
        return EconomyCruisePoint(
            mach=float(mach[index]),
            true_airspeed_mps=float(speed[index]),
            ground_speed_mps=float(ground_speed[index]),
            lift_coefficient=float(lift[index]),
            critical_mach=float(critical_mach[index]),
            wave_drag_coefficient=float(wave[index]),
            total_drag_n=float(drag[index]),
            fuel_flow_kg_s=float(fuel_flow[index]),
            fuel_per_ground_distance_kg_m=float(objective[index]),
        )
