"""Transonic wave-drag formulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray


@dataclass(frozen=True)
class KornLockWaveDrag:
    """Korn critical-Mach relation with Lock's fourth-power wave-drag rise."""

    wing_sweep_rad: float
    thickness_to_chord_ratio: float
    technology_factor: float = 0.95

    def __post_init__(self) -> None:
        values = (
            self.wing_sweep_rad,
            self.thickness_to_chord_ratio,
            self.technology_factor,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("wave-drag parameters must be finite")
        if not 0.0 <= self.wing_sweep_rad < np.deg2rad(70.0):
            raise ValueError("wing sweep must lie between zero and 70 degrees")
        if not 0.02 <= self.thickness_to_chord_ratio <= 0.30:
            raise ValueError("thickness-to-chord ratio is outside model support")
        if not 0.5 <= self.technology_factor <= 1.1:
            raise ValueError("technology factor is outside model support")

    def _critical_mach_from_lift(self, lift: FloatArray) -> FloatArray:
        cosine = np.cos(self.wing_sweep_rad)
        return (
            self.technology_factor / cosine
            - self.thickness_to_chord_ratio / cosine**2
            - 0.1 * lift / cosine**3
            - 0.108
        )

    def critical_mach(self, lift_coefficient: ArrayLike) -> FloatArray:
        """Return the sweep-adjusted critical Mach estimate."""

        lift = np.asarray(lift_coefficient, dtype=float)
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift_coefficient must be finite")
        return self._critical_mach_from_lift(lift)

    def evaluate(
        self, mach: ArrayLike, lift_coefficient: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Return wave-drag increment and critical Mach in one vector pass."""

        mach_array, lift = np.broadcast_arrays(
            np.asarray(mach, dtype=float),
            np.asarray(lift_coefficient, dtype=float),
        )
        if np.any(~np.isfinite(mach_array + lift)) or np.any(mach_array < 0.0):
            raise ValueError("Mach and lift coefficient must be finite")
        critical = self._critical_mach_from_lift(lift)
        excess = np.maximum(mach_array - critical, 0.0)
        return 20.0 * excess**4, critical

    def drag_coefficient_increment(
        self, mach: ArrayLike, lift_coefficient: ArrayLike
    ) -> FloatArray:
        """Return Lock fourth-power wave-drag coefficient increment."""

        wave, _ = self.evaluate(mach, lift_coefficient)
        return wave

    def drag_coefficient_increment_scalar(
        self, mach: float, lift_coefficient: float
    ) -> float:
        if not np.isfinite(mach + lift_coefficient) or mach < 0.0:
            raise ValueError("Mach and lift coefficient must be finite")
        lift = np.asarray(lift_coefficient, dtype=float)
        critical = float(self._critical_mach_from_lift(lift))
        excess = max(mach - critical, 0.0)
        return 20.0 * excess**4
