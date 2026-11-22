"""Published and comparison drag formulations.

These models support source reconciliation and validation. They are not
selected implicitly as an aircraft's active runtime model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import DragBreakdown, DragCondition, FloatArray, resolve_condition


@dataclass(frozen=True)
class MachDependentInducedDrag:
    """Compressibility correction for the lift-induced drag coefficient.

    The model evaluates ``CL² d / (1 - e (M/Mcomp - 1)^f)``. It is intended
    for the high-subsonic support of the source calibration and deliberately
    rejects Mach numbers below ``Mcomp`` where a non-integer exponent would
    otherwise make the expression ambiguous.
    """

    reference_induced_drag_factor: float
    compressibility_strength: float
    compressibility_exponent: float
    compressibility_mach: float = 0.3

    def __post_init__(self) -> None:
        values = (
            self.reference_induced_drag_factor,
            self.compressibility_strength,
            self.compressibility_exponent,
            self.compressibility_mach,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("induced-drag parameters must be finite")
        if self.reference_induced_drag_factor < 0.0:
            raise ValueError("reference induced-drag factor must be non-negative")
        if self.compressibility_strength < 0.0:
            raise ValueError("compressibility strength must be non-negative")
        if self.compressibility_exponent <= 0.0 or self.compressibility_mach <= 0.0:
            raise ValueError("compressibility exponent and Mach must be positive")

    def drag_coefficient(
        self, lift_coefficient: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        lift, mach_array = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float),
            np.asarray(mach, dtype=float),
        )
        if np.any(~np.isfinite(lift)) or np.any(~np.isfinite(mach_array)):
            raise ValueError("lift coefficient and Mach must be finite")
        if np.any(mach_array < self.compressibility_mach):
            raise ValueError("Mach is below the compressibility-model support")
        denominator = (
            1.0
            - self.compressibility_strength
            * (mach_array / self.compressibility_mach - 1.0)
            ** self.compressibility_exponent
        )
        if np.any(denominator <= 0.0):
            raise ValueError("induced-drag compressibility correction is singular")
        return self.reference_induced_drag_factor * lift**2 / denominator

    def drag_coefficient_scalar(self, lift_coefficient: float, mach: float) -> float:
        return float(self.drag_coefficient(lift_coefficient, mach))


@dataclass(frozen=True)
class HyperbolicTangentWaveDrag:
    """Bounded CL-dependent hyperbolic-tangent wave-drag model."""

    lift_coefficient_grid: FloatArray
    critical_mach_grid: FloatArray
    amplitude: float
    mach_scale: float
    shift: float

    def __post_init__(self) -> None:
        lift = np.asarray(self.lift_coefficient_grid, dtype=float)
        critical_mach = np.asarray(self.critical_mach_grid, dtype=float)
        if lift.ndim != 1 or len(lift) < 2 or np.any(np.diff(lift) <= 0.0):
            raise ValueError("lift coefficient grid must contain increasing points")
        if critical_mach.shape != lift.shape:
            raise ValueError("critical Mach grid must match the lift grid")
        if np.any(~np.isfinite(lift)) or np.any(~np.isfinite(critical_mach)):
            raise ValueError("wave-drag grids must be finite")
        if np.any((critical_mach <= 0.0) | (critical_mach >= 1.0)):
            raise ValueError("critical Mach values must lie between zero and one")
        parameters = self.amplitude, self.mach_scale, self.shift
        if any(not np.isfinite(value) for value in parameters):
            raise ValueError("wave-drag parameters must be finite")
        if self.amplitude < 0.0 or self.mach_scale <= 0.0:
            raise ValueError("wave amplitude must be non-negative and scale positive")
        object.__setattr__(self, "lift_coefficient_grid", lift)
        object.__setattr__(self, "critical_mach_grid", critical_mach)

    @property
    def minimum_lift_coefficient(self) -> float:
        return float(self.lift_coefficient_grid[0])

    @property
    def maximum_lift_coefficient(self) -> float:
        return float(self.lift_coefficient_grid[-1])

    def critical_mach(self, lift_coefficient: ArrayLike) -> FloatArray:
        lift = np.asarray(lift_coefficient, dtype=float)
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        if np.any(
            (lift < self.minimum_lift_coefficient)
            | (lift > self.maximum_lift_coefficient)
        ):
            raise ValueError("lift coefficient is outside wave-drag support")
        return np.interp(
            lift,
            self.lift_coefficient_grid,
            self.critical_mach_grid,
        )

    def drag_coefficient_increment(
        self, mach: ArrayLike, lift_coefficient: ArrayLike
    ) -> FloatArray:
        mach_array, lift = np.broadcast_arrays(
            np.asarray(mach, dtype=float),
            np.asarray(lift_coefficient, dtype=float),
        )
        if np.any(~np.isfinite(mach_array)) or np.any(mach_array < 0.0):
            raise ValueError("Mach must be finite and non-negative")
        critical_mach = self.critical_mach(lift)
        increment = self.amplitude * (
            1.0 + np.tanh(self.mach_scale * mach_array / critical_mach - self.shift)
        )
        return np.where(mach_array >= critical_mach, increment, 0.0)

    def drag_coefficient_increment_scalar(
        self, mach: float, lift_coefficient: float
    ) -> float:
        return float(self.drag_coefficient_increment(mach, lift_coefficient))


@dataclass(frozen=True)
class TransonicCompressibilityDragModel:
    """Constant parasite, Mach-dependent induced, and wave drag components."""

    cd0: float
    induced_drag: MachDependentInducedDrag
    wave_drag: HyperbolicTangentWaveDrag

    def __post_init__(self) -> None:
        if not np.isfinite(self.cd0) or self.cd0 < 0.0:
            raise ValueError("zero-lift drag coefficient must be non-negative")

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
        critical_mach = self.wave_drag.critical_mach(lift)
        parasite = np.full_like(lift, self.cd0)
        induced = self.induced_drag.drag_coefficient(lift, mach_array)
        wave = self.wave_drag.drag_coefficient_increment(mach_array, lift)
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=wave,
            total=parasite + induced + wave,
            critical_mach=critical_mach,
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
        return float(self.drag_coefficient(lift_coefficient, condition=condition))


@dataclass(frozen=True)
class PollSchumannDragModel:
    """Reynolds-aware clean transport-aircraft drag model.

    This implements Equations 30--45 of Poll and Schumann (2025), DOI
    ``10.1017/aer.2024.141``.  The model separates Reynolds-dependent
    zero-lift drag, lift-dependent drag, drag creep, and transonic drag rise.
    Its explicit lift and Mach bounds make source support part of the runtime
    interface instead of permitting silent extrapolation.
    """

    reference_area_m2: float
    wing_span_m: float
    fuselage_width_m: float
    wing_sweep_rad: float
    skin_friction_multiplier: float
    technology_factor: float
    design_mach: float
    design_lift_coefficient: float
    wave_creep_coefficient: float
    wave_creep_threshold: float
    reference_reynolds_number: float
    wave_rise_coefficient: float = 40.0
    minimum_lift_coefficient: float = 0.3
    maximum_lift_coefficient: float = 0.7
    minimum_mach: float = 0.0
    maximum_mach: float = 0.82

    def __post_init__(self) -> None:
        values = (
            self.reference_area_m2,
            self.wing_span_m,
            self.fuselage_width_m,
            self.wing_sweep_rad,
            self.skin_friction_multiplier,
            self.technology_factor,
            self.design_mach,
            self.design_lift_coefficient,
            self.wave_creep_coefficient,
            self.wave_creep_threshold,
            self.reference_reynolds_number,
            self.wave_rise_coefficient,
            self.minimum_lift_coefficient,
            self.maximum_lift_coefficient,
            self.minimum_mach,
            self.maximum_mach,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Poll-Schumann parameters must be finite")
        if (
            min(
                self.reference_area_m2,
                self.wing_span_m,
                self.fuselage_width_m,
                self.skin_friction_multiplier,
                self.technology_factor,
                self.design_mach,
                self.design_lift_coefficient,
                self.wave_creep_coefficient,
                self.wave_creep_threshold,
                self.reference_reynolds_number,
                self.wave_rise_coefficient,
            )
            <= 0.0
        ):
            raise ValueError("Poll-Schumann dimensional parameters must be positive")
        if not 0.0 <= self.wing_sweep_rad < np.deg2rad(70.0):
            raise ValueError("wing sweep must lie between zero and 70 degrees")
        if self.minimum_lift_coefficient >= self.maximum_lift_coefficient:
            raise ValueError("lift coefficient bounds are invalid")
        if self.minimum_mach < 0.0 or self.minimum_mach >= self.maximum_mach:
            raise ValueError("Mach bounds are invalid")
        if not (
            self.minimum_lift_coefficient
            <= self.design_lift_coefficient
            <= self.maximum_lift_coefficient
        ):
            raise ValueError("design lift coefficient is outside model support")
        if not self.minimum_mach <= self.design_mach <= self.maximum_mach:
            raise ValueError("design Mach is outside model support")

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m**2 / self.reference_area_m2

    def _reynolds_number(self, reynolds_number: ArrayLike | None) -> FloatArray:
        reynolds = np.asarray(
            self.reference_reynolds_number
            if reynolds_number is None
            else reynolds_number,
            dtype=float,
        )
        if np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 0.0):
            raise ValueError("reynolds_number must be finite and positive")
        return reynolds

    def cd0(self, reynolds_number: ArrayLike | None = None) -> FloatArray:
        """Return ``CD0 = psi0 * 0.0269 / Re**0.14``."""

        reynolds = self._reynolds_number(reynolds_number)
        return self.skin_friction_multiplier * 0.0269 / reynolds**0.14

    def k(self, reynolds_number: ArrayLike | None = None) -> FloatArray:
        """Return the geometry- and profile-drag-dependent factor ``K``."""

        zero_lift = self.cd0(reynolds_number)
        cosine = np.cos(self.wing_sweep_rad)
        profile_lift_factor = 0.80 * (1.0 - 0.53 * cosine) * zero_lift
        fuselage_correction = 2.0 * (self.fuselage_width_m / self.wing_span_m) ** 2
        return profile_lift_factor + (1.03 + fuselage_correction) / (
            np.pi * self.aspect_ratio
        )

    def crest_critical_mach(self, lift_coefficient: ArrayLike) -> FloatArray:
        lift = np.asarray(lift_coefficient, dtype=float)
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        if np.any(
            (lift < self.minimum_lift_coefficient)
            | (lift > self.maximum_lift_coefficient)
        ):
            raise ValueError("lift coefficient is outside Poll-Schumann support")
        cosine = np.cos(self.wing_sweep_rad)
        critical_mach = self.technology_factor - 0.10 * lift / cosine**2
        if np.any(critical_mach <= 0.0):
            raise ValueError("crest critical Mach is non-positive")
        return critical_mach

    @property
    def design_wave_coordinate(self) -> float:
        cosine = np.cos(self.wing_sweep_rad)
        critical_mach = float(self.crest_critical_mach(self.design_lift_coefficient))
        return self.design_mach * cosine / critical_mach

    def wave_drag_coefficient(
        self, lift_coefficient: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        lift, mach_array = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float),
            np.asarray(mach, dtype=float),
        )
        if np.any(~np.isfinite(mach_array)) or np.any(
            (mach_array < self.minimum_mach) | (mach_array > self.maximum_mach)
        ):
            raise ValueError("Mach is outside Poll-Schumann support")
        cosine = np.cos(self.wing_sweep_rad)
        coordinate = mach_array * cosine / self.crest_critical_mach(lift)
        creep = np.where(
            coordinate >= self.wave_creep_threshold,
            cosine**3
            * self.wave_creep_coefficient
            * (coordinate - self.wave_creep_threshold) ** 2,
            0.0,
        )
        rise = np.where(
            coordinate > self.design_wave_coordinate,
            cosine**3
            * self.wave_rise_coefficient
            * (coordinate - self.design_wave_coordinate) ** 4,
            0.0,
        )
        return creep + rise

    def breakdown(
        self,
        lift_coefficient: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        condition: DragCondition | None = None,
    ) -> DragBreakdown:
        """Evaluate the model.

        The source model uses the aircraft Reynolds number based on
        ``sqrt(reference_area_m2)``. It is computed from ``reynolds_per_m``;
        without it, ``reference_reynolds_number`` is used.
        """

        condition = resolve_condition(mach, reynolds_per_m, condition)
        lift, mach_array, unit_reynolds, flap, gear, spoiler = condition.broadcast_with(
            lift_coefficient
        )
        if np.any(flap > 0.0) or np.any(gear > 0.0) or np.any(spoiler > 0.0):
            raise ValueError("Poll-Schumann is currently supported only when clean")
        if condition.reynolds_per_m is None:
            reynolds = np.full_like(lift, self.reference_reynolds_number)
        else:
            reynolds = unit_reynolds * np.sqrt(self.reference_area_m2)
        critical_mach = self.crest_critical_mach(lift)
        parasite = self.cd0(reynolds)
        induced = self.k(reynolds) * lift**2
        wave = self.wave_drag_coefficient(lift, mach_array)
        return DragBreakdown(
            parasite=parasite,
            induced=induced,
            wave=wave,
            total=parasite + induced + wave,
            critical_mach=critical_mach,
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
        return float(self.drag_coefficient(lift_coefficient, condition=condition))
