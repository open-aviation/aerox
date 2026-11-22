"""Low-order drag polars and their wave-drag composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import (
    DragBreakdown,
    DragCondition,
    DragModel,
    FloatArray,
    resolve_condition,
)
from .wave import KornLockWaveDrag


@dataclass(frozen=True)
class ParabolicDragPolar:
    """Conventional clean parabolic drag polar ``CD = cd0 + k CL²``.

    The polar has no Mach or Reynolds number effect, so ``mach`` is optional.
    """

    cd0: float
    k: float

    def __post_init__(self) -> None:
        if self.cd0 < 0 or self.k < 0:
            raise ValueError("drag-polar coefficients must be non-negative")

    def breakdown(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> DragBreakdown:
        if condition is None and mach is None:
            mach = 0.0
        condition = resolve_condition(mach, reynolds_per_m, condition)
        lift, mach_array, *_ = condition.broadcast_with(lift_coefficient)
        parasite = np.full_like(lift, self.cd0)
        induced = self.k * lift**2
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=np.zeros_like(lift),
            total=parasite + induced,
            critical_mach=np.full_like(mach_array, np.nan),
        )

    def drag_coefficient(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            lift_coefficient, mach, reynolds_per_m, condition=condition
        ).total

    def _drag_coefficient_scalar(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        if not np.isfinite(lift_coefficient):
            raise ValueError("lift_coefficient must be finite")
        mach = float(condition.mach)
        if not np.isfinite(mach) or mach < 0.0:
            raise ValueError("mach must be finite and non-negative")
        return float(self.cd0 + self.k * lift_coefficient**2)


@dataclass(frozen=True)
class ParabolicWaveDragModel:
    """Clean parabolic polar plus an explicit transonic wave-drag component."""

    base_polar: ParabolicDragPolar
    wave_drag: KornLockWaveDrag

    def breakdown(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> DragBreakdown:
        condition = resolve_condition(mach, reynolds_per_m, condition)
        lift, mach_array, *_ = condition.broadcast_with(lift_coefficient)
        parasite = np.full_like(lift, self.base_polar.cd0)
        induced = self.base_polar.k * lift**2
        wave, critical = self.wave_drag.evaluate(mach_array, lift)
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=wave,
            total=parasite + induced + wave,
            critical_mach=critical,
        )

    def drag_coefficient(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            lift_coefficient, mach, reynolds_per_m, condition=condition
        ).total

    def _drag_coefficient_scalar(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        mach = float(condition.mach)
        return self.base_polar._drag_coefficient_scalar(
            lift_coefficient, condition
        ) + self.wave_drag.drag_coefficient_increment_scalar(mach, lift_coefficient)


def project_parabolic_polar(
    model: DragModel, condition: DragCondition
) -> ParabolicDragPolar:
    """Project a drag model onto ``CD = cd0 + k CL²``.

    The projection is local to the supplied condition and deliberately omits
    any claim that the resulting coefficients remain valid elsewhere.
    """

    cd0 = model._drag_coefficient_scalar(0.0, condition)
    k = model._drag_coefficient_scalar(1.0, condition) - cd0
    return ParabolicDragPolar(cd0, k)
