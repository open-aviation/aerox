"""Low-order drag polars and their wave-drag composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import DragBreakdown, DragCondition, DragModel, FloatArray
from .wave import KornLockWaveDrag


@dataclass(frozen=True)
class ParabolicDragPolar:
    """Conventional clean parabolic drag polar ``CD = CD0 + k CL²``."""

    zero_lift_drag_coefficient: float
    induced_drag_factor: float

    def __post_init__(self) -> None:
        if self.zero_lift_drag_coefficient < 0 or self.induced_drag_factor < 0:
            raise ValueError("drag-polar coefficients must be non-negative")

    def drag_coefficient(
        self, lift_coefficient: ArrayLike, mach: ArrayLike | None = None
    ) -> FloatArray:
        lift = np.asarray(lift_coefficient, dtype=float)
        if mach is not None:
            lift, mach_array = np.broadcast_arrays(
                lift,
                np.asarray(mach, dtype=float),
            )
            if np.any(~np.isfinite(mach_array)) or np.any(mach_array < 0.0):
                raise ValueError("mach must be finite and non-negative")
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        return self.zero_lift_drag_coefficient + self.induced_drag_factor * lift**2

    def drag_coefficient_scalar(
        self, lift_coefficient: float, mach: float | None = None
    ) -> float:
        """Scalar fast path; Mach is accepted for drag-model interchangeability."""

        if not np.isfinite(lift_coefficient):
            raise ValueError("lift_coefficient must be finite")
        if mach is not None and (not np.isfinite(mach) or mach < 0.0):
            raise ValueError("mach must be finite and non-negative")
        return float(
            self.zero_lift_drag_coefficient
            + self.induced_drag_factor * lift_coefficient**2
        )

    def breakdown_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> DragBreakdown:
        lift, mach, *_ = condition.broadcast_with(lift_coefficient)
        parasite = np.full_like(lift, self.zero_lift_drag_coefficient)
        induced = self.induced_drag_factor * lift**2
        zeros = np.zeros_like(lift)
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=zeros,
            total=parasite + induced,
            critical_mach=np.full_like(mach, np.nan),
        )

    def drag_coefficient_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> FloatArray:
        return self.breakdown_at(lift_coefficient, condition).total

    def drag_coefficient_scalar_at(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        condition.broadcast_with(lift_coefficient)
        return self.drag_coefficient_scalar(lift_coefficient, float(condition.mach))


@dataclass(frozen=True)
class ParabolicWaveDragModel:
    """Clean parabolic polar plus an explicit transonic wave-drag component."""

    base_polar: ParabolicDragPolar
    wave_drag: KornLockWaveDrag

    def breakdown(self, lift_coefficient: ArrayLike, mach: ArrayLike) -> DragBreakdown:
        lift, mach_array = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float), np.asarray(mach, dtype=float)
        )
        if np.any(~np.isfinite(lift + mach_array)) or np.any(mach_array < 0.0):
            raise ValueError("lift coefficient and Mach must be finite")
        parasite = np.full_like(lift, self.base_polar.zero_lift_drag_coefficient)
        induced = self.base_polar.induced_drag_factor * lift**2
        wave, critical = self.wave_drag.evaluate(mach_array, lift)
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=wave,
            total=parasite + induced + wave,
            critical_mach=critical,
        )

    def drag_coefficient(
        self, lift_coefficient: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        return self.breakdown(lift_coefficient, mach).total

    def drag_coefficient_scalar(self, lift_coefficient: float, mach: float) -> float:
        return self.base_polar.drag_coefficient_scalar(
            lift_coefficient, mach
        ) + self.wave_drag.drag_coefficient_increment_scalar(mach, lift_coefficient)

    def breakdown_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> DragBreakdown:
        lift, mach, *_ = condition.broadcast_with(lift_coefficient)
        return self.breakdown(lift, mach)

    def drag_coefficient_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> FloatArray:
        return self.breakdown_at(lift_coefficient, condition).total

    def drag_coefficient_scalar_at(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        condition.broadcast_with(lift_coefficient)
        return self.drag_coefficient_scalar(lift_coefficient, float(condition.mach))


def project_parabolic_polar(
    model: DragModel, condition: DragCondition
) -> ParabolicDragPolar:
    """Project a condition-aware model onto ``CD = CD0 + k CL²``.

    The projection is local to the supplied condition and deliberately omits
    any claim that the resulting coefficients remain valid elsewhere.
    """

    zero_lift = model.drag_coefficient_scalar_at(0.0, condition)
    induced_factor = model.drag_coefficient_scalar_at(1.0, condition) - zero_lift
    return ParabolicDragPolar(zero_lift, induced_factor)
