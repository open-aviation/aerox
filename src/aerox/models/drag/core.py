"""Shared interfaces and value objects for drag-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DragCondition:
    """Air-relative flow condition and aircraft configuration.

    Reynolds number is supplied per metre so each component can apply its own
    characteristic length. Configuration inputs are explicit so clean-aircraft
    models cannot silently evaluate a deployed configuration.
    """

    mach: ArrayLike
    unit_reynolds_number_per_m: ArrayLike
    flap_angle_rad: ArrayLike = 0.0
    gear_extension_fraction: ArrayLike = 0.0
    spoiler_deflection_rad: ArrayLike = 0.0

    def broadcast_with(
        self, lift_coefficient: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Validate and broadcast the condition with a lift coefficient."""

        lift, mach, unit_reynolds, flap, gear, spoiler = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float),
            np.asarray(self.mach, dtype=float),
            np.asarray(self.unit_reynolds_number_per_m, dtype=float),
            np.asarray(self.flap_angle_rad, dtype=float),
            np.asarray(self.gear_extension_fraction, dtype=float),
            np.asarray(self.spoiler_deflection_rad, dtype=float),
        )
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        if np.any(~np.isfinite(mach)) or np.any(mach < 0.0):
            raise ValueError("mach must be finite and non-negative")
        if np.any(~np.isfinite(unit_reynolds)) or np.any(unit_reynolds <= 0.0):
            raise ValueError("unit_reynolds_number_per_m must be finite and positive")
        if np.any(~np.isfinite(flap)) or np.any(flap < 0.0):
            raise ValueError("flap_angle_rad must be finite and non-negative")
        if np.any(~np.isfinite(gear)) or np.any((gear < 0.0) | (gear > 1.0)):
            raise ValueError("gear_extension_fraction must lie between zero and one")
        if np.any(~np.isfinite(spoiler)) or np.any(spoiler < 0.0):
            raise ValueError("spoiler_deflection_rad must be finite and non-negative")
        return lift, mach, unit_reynolds, flap, gear, spoiler


@runtime_checkable
class DragModel(Protocol):
    """Condition-aware drag-coefficient model used by the flight mechanics."""

    def drag_coefficient_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> FloatArray: ...

    def drag_coefficient_scalar_at(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float: ...


@dataclass(frozen=True)
class DragBreakdown:
    """Common drag-coefficient components returned by compact models."""

    parasite: FloatArray
    induced: FloatArray
    wave: FloatArray
    total: FloatArray
    critical_mach: FloatArray
