"""Frozen A320 drag candidates retained only to reproduce 2026-08-08 audits.

These fitted candidates predate the fleet geometry/Korn-Lock model and are not
part of the Aerox runtime API. New simulations must use
``aerox.models.build_drag_model``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox.aircraft import Aircraft, load_aircraft
from aerox.models import DragCondition
from aerox.models.drag import (
    GeometryDragModel,
    geometry_drag_model_from_aircraft,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HistoricalCompactDragBreakdown:
    """Component output of the frozen fitted compact candidate."""

    component_parasite: Mapping[str, FloatArray]
    resolved_parasite: FloatArray
    unresolved_parasite: FloatArray
    parasitic: FloatArray
    span_induced: FloatArray
    lift_dependent_profile: FloatArray
    wave: FloatArray
    critical_mach: FloatArray
    total: FloatArray


@dataclass(frozen=True)
class HistoricalCompactTransonicDragModel:
    """Historical fitted fourth-power candidate used by archived audits."""

    base_prior: GeometryDragModel
    zero_lift_critical_mach: float
    lift_critical_mach_slope: float
    wave_drag_amplitude: float
    wave_mach_width: float = 0.1
    minimum_lift_coefficient: float = 0.0
    maximum_lift_coefficient: float = 1.5
    minimum_mach: float = 0.0
    maximum_mach: float = 0.84

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.zero_lift_critical_mach,
                self.lift_critical_mach_slope,
                self.wave_drag_amplitude,
                self.wave_mach_width,
                self.minimum_lift_coefficient,
                self.maximum_lift_coefficient,
                self.minimum_mach,
                self.maximum_mach,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("historical compact drag parameters must be finite")
        if self.lift_critical_mach_slope < 0.0 or self.wave_drag_amplitude < 0.0:
            raise ValueError("wave slope and amplitude must be non-negative")
        if self.wave_mach_width <= 0.0:
            raise ValueError("wave_mach_width must be positive")
        if self.minimum_lift_coefficient >= self.maximum_lift_coefficient:
            raise ValueError("lift coefficient bounds are invalid")
        if self.minimum_mach < 0.0 or self.minimum_mach >= self.maximum_mach:
            raise ValueError("Mach bounds are invalid")

    def critical_mach(self, lift_coefficient: ArrayLike) -> FloatArray:
        lift = np.asarray(lift_coefficient, dtype=float)
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        return self.zero_lift_critical_mach - self.lift_critical_mach_slope * lift

    def breakdown_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> HistoricalCompactDragBreakdown:
        lift, mach, *_ = condition.broadcast_with(lift_coefficient)
        if np.any(
            (lift < self.minimum_lift_coefficient)
            | (lift > self.maximum_lift_coefficient)
        ):
            raise ValueError("lift coefficient is outside historical-model support")
        if np.any((mach < self.minimum_mach) | (mach > self.maximum_mach)):
            raise ValueError("Mach is outside historical-model support")
        base = self.base_prior.breakdown_at(lift, condition)
        critical = self.critical_mach(lift)
        excess = np.maximum((mach - critical) / self.wave_mach_width, 0.0)
        wave = self.wave_drag_amplitude * excess**4
        return HistoricalCompactDragBreakdown(
            component_parasite=base.component_parasite,
            resolved_parasite=base.resolved_parasite,
            unresolved_parasite=base.unresolved_parasite,
            parasitic=base.parasite,
            span_induced=base.span_induced,
            lift_dependent_profile=base.lift_dependent_profile,
            wave=wave,
            critical_mach=critical,
            total=base.total + wave,
        )

    def breakdown(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike,
        unit_reynolds_number_per_m: ArrayLike | None = None,
    ) -> HistoricalCompactDragBreakdown:
        return self.breakdown_at(
            lift_coefficient,
            DragCondition(
                mach=mach,
                unit_reynolds_number_per_m=(
                    self.base_prior.reference_unit_reynolds_number_per_m
                    if unit_reynolds_number_per_m is None
                    else unit_reynolds_number_per_m
                ),
            ),
        )

    def drag_coefficient_at(
        self, lift_coefficient: ArrayLike, condition: DragCondition
    ) -> FloatArray:
        return self.breakdown_at(lift_coefficient, condition).total

    def drag_coefficient_scalar_at(
        self, lift_coefficient: float, condition: DragCondition
    ) -> float:
        return float(self.drag_coefficient_at(lift_coefficient, condition))

    def drag_coefficient(
        self, lift_coefficient: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        return self.breakdown(lift_coefficient, mach).total

    def drag_coefficient_scalar(self, lift_coefficient: float, mach: float) -> float:
        return float(self.drag_coefficient(lift_coefficient, mach))


def historical_a320_compact_drag_candidate(
    aircraft: Aircraft | None = None,
) -> HistoricalCompactTransonicDragModel:
    """Return the rejected candidate recorded by the historical A320 audits."""

    aircraft = load_aircraft("A320") if aircraft is None else aircraft
    if aircraft.icao != "A320":
        raise ValueError("historical candidate requires an A320 record")
    return HistoricalCompactTransonicDragModel(
        base_prior=geometry_drag_model_from_aircraft(
            aircraft,
            parasite_scale=1.1169917262650704,
            induced_scale=0.8227488253841422,
        ),
        zero_lift_critical_mach=0.75,
        lift_critical_mach_slope=0.12,
        wave_drag_amplitude=0.002863434146558585,
        maximum_mach=float(aircraft.mmo or 0.84),
    )
