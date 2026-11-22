"""Shared interfaces and value objects for drag-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DragCondition:
    """Air-relative flow condition and aircraft configuration.

    Reynolds number is supplied per metre so each component can apply its own
    characteristic length. When it is ``None``, a model that depends on it uses
    its own reference value. Configuration inputs are explicit so
    clean-aircraft models cannot silently evaluate a deployed configuration.
    """

    mach: ArrayLike
    reynolds_per_m: ArrayLike | None = None
    flap_angle_rad: ArrayLike = 0.0
    gear_extension_fraction: ArrayLike = 0.0
    spoiler_deflection_rad: ArrayLike = 0.0

    def with_default_reynolds(self, reynolds_per_m: float) -> DragCondition:
        """Return this condition with a Reynolds number filled in if missing."""

        if self.reynolds_per_m is not None:
            return self
        return replace(self, reynolds_per_m=reynolds_per_m)

    def broadcast_with(
        self, lift_coefficient: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Validate and broadcast the condition with a lift coefficient.

        A missing Reynolds number is returned as NaN.
        """

        reynolds_given = self.reynolds_per_m is not None
        lift, mach, reynolds, flap, gear, spoiler = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float),
            np.asarray(self.mach, dtype=float),
            np.asarray(self.reynolds_per_m if reynolds_given else np.nan, dtype=float),
            np.asarray(self.flap_angle_rad, dtype=float),
            np.asarray(self.gear_extension_fraction, dtype=float),
            np.asarray(self.spoiler_deflection_rad, dtype=float),
        )
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        if np.any(~np.isfinite(mach)) or np.any(mach < 0.0):
            raise ValueError("mach must be finite and non-negative")
        if reynolds_given and (
            np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 0.0)
        ):
            raise ValueError("reynolds_per_m must be finite and positive")
        if np.any(~np.isfinite(flap)) or np.any(flap < 0.0):
            raise ValueError("flap_angle_rad must be finite and non-negative")
        if np.any(~np.isfinite(gear)) or np.any((gear < 0.0) | (gear > 1.0)):
            raise ValueError("gear_extension_fraction must lie between zero and one")
        if np.any(~np.isfinite(spoiler)) or np.any(spoiler < 0.0):
            raise ValueError("spoiler_deflection_rad must be finite and non-negative")
        return lift, mach, reynolds, flap, gear, spoiler


def resolve_condition(
    mach: ArrayLike | None,
    reynolds_per_m: ArrayLike | None,
    condition: DragCondition | None,
) -> DragCondition:
    """Return the condition given either as arguments or as a ``DragCondition``."""

    if condition is not None:
        if mach is not None or reynolds_per_m is not None:
            raise TypeError("give either mach and reynolds_per_m, or condition")
        return condition
    if mach is None:
        raise TypeError("mach is required")
    return DragCondition(mach=mach, reynolds_per_m=reynolds_per_m)


@runtime_checkable
class DragModel(Protocol):
    """Drag-coefficient model used by the flight mechanics.

    Every model is evaluated as ``drag_coefficient(cl, mach, reynolds_per_m)``
    or with ``condition=DragCondition(...)``. ``_drag_coefficient_scalar`` is a
    fast path for scalar conditions used inside integrators.
    """

    def drag_coefficient(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> FloatArray: ...

    def _drag_coefficient_scalar(
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
