"""Low-order drag polars and their wave-drag composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import (
    CdShorthand,
    DragBreakdown,
    DragCondition,
    DragModel,
    FloatArray,
    require_clean,
    resolve_condition,
)
from .wave import KornLockWaveDrag


@dataclass(frozen=True)
class ParabolicDragPolar(CdShorthand):
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
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> DragBreakdown:
        if condition is None and mach is None and h is None:
            mach = 0.0
        condition = resolve_condition(mach, reynolds_per_m, condition, h=h, dT=dT)
        lift, mach_array, _, flap, gear, spoiler = condition.broadcast_with(cl)
        require_clean(flap, gear, spoiler)
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
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
        ).total

    def _drag_coefficient_scalar(self, cl: float, condition: DragCondition) -> float:
        lift, _, _, flap, gear, spoiler = condition.broadcast_with(cl)
        if lift.ndim != 0:
            raise ValueError("scalar drag evaluation requires scalar condition values")
        require_clean(flap, gear, spoiler)
        return float(self.cd0 + self.k * float(cl) ** 2)


@dataclass(frozen=True)
class ParabolicWaveDragModel(CdShorthand):
    """Clean parabolic polar plus an explicit transonic wave-drag component."""

    base_polar: ParabolicDragPolar
    wave_drag: KornLockWaveDrag

    def breakdown(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> DragBreakdown:
        condition = resolve_condition(mach, reynolds_per_m, condition, h=h, dT=dT)
        lift, mach_array, _, flap, gear, spoiler = condition.broadcast_with(cl)
        require_clean(flap, gear, spoiler)
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
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
        ).total

    def _drag_coefficient_scalar(self, cl: float, condition: DragCondition) -> float:
        mach = float(condition.mach)
        return self.base_polar._drag_coefficient_scalar(
            cl, condition
        ) + self.wave_drag.drag_coefficient_increment_scalar(mach, cl)


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
