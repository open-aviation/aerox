"""Empirical lift-coefficient maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

LIFT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "linear_alpha": ("constant", "alpha_centered"),
    "linear_alpha_mach": (
        "constant",
        "alpha_centered",
        "mach_centered",
        "alpha_mach_interaction",
    ),
    "quadratic_alpha_mach": (
        "constant",
        "alpha_centered",
        "alpha_centered_squared",
        "mach_centered",
        "alpha_mach_interaction",
    ),
}


def effective_lift_design(
    angle_of_attack_rad: ArrayLike,
    mach: ArrayLike,
    *,
    feature_set: str,
) -> FloatArray:
    """Return a centered design matrix for an effective clean lift map."""
    if feature_set not in LIFT_FEATURE_SETS:
        raise ValueError(f"unknown lift feature set: {feature_set}")
    alpha = np.asarray(angle_of_attack_rad, dtype=float)
    mach_array = np.asarray(mach, dtype=float)
    if alpha.ndim != 1 or mach_array.ndim != 1 or alpha.shape != mach_array.shape:
        raise ValueError("angle_of_attack_rad and mach must be equal-length vectors")
    if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(mach_array)):
        raise ValueError("angle_of_attack_rad and mach must be finite")
    # Degree scaling keeps the low-order fit numerically well conditioned and
    # makes the slope coefficient directly interpretable as CL per degree.
    alpha_centered = np.rad2deg(alpha) - 4.0
    mach_centered = mach_array - 0.6
    available = {
        "constant": np.ones_like(alpha),
        "alpha_centered": alpha_centered,
        "alpha_centered_squared": alpha_centered**2,
        "mach_centered": mach_centered,
        "alpha_mach_interaction": alpha_centered * mach_centered,
    }
    return np.column_stack([available[name] for name in LIFT_FEATURE_SETS[feature_set]])


@dataclass(frozen=True)
class EffectiveLiftMap:
    """Effective clean-configuration lift coefficient map."""

    feature_set: str
    coefficients: FloatArray
    minimum_lift_coefficient: float = -0.5
    maximum_lift_coefficient: float = 2.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = len(LIFT_FEATURE_SETS.get(self.feature_set, ()))
        if coefficients.ndim != 1 or len(coefficients) != expected:
            raise ValueError("coefficient count does not match lift feature set")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        if self.minimum_lift_coefficient >= self.maximum_lift_coefficient:
            raise ValueError("lift coefficient bounds are invalid")
        object.__setattr__(self, "coefficients", coefficients)

    def raw_lift_coefficient(
        self, angle_of_attack_rad: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        design = effective_lift_design(
            angle_of_attack_rad,
            mach,
            feature_set=self.feature_set,
        )
        return design @ self.coefficients

    def raw_lift_coefficient_scalar(
        self, angle_of_attack_rad: float, mach: float
    ) -> float:
        """Scalar fast path for numerical integration."""
        alpha_centered = np.rad2deg(angle_of_attack_rad) - 4.0
        mach_centered = mach - 0.6
        available = {
            "constant": 1.0,
            "alpha_centered": alpha_centered,
            "alpha_centered_squared": alpha_centered**2,
            "mach_centered": mach_centered,
            "alpha_mach_interaction": alpha_centered * mach_centered,
        }
        return float(
            sum(
                coefficient * available[name]
                for coefficient, name in zip(
                    self.coefficients,
                    LIFT_FEATURE_SETS[self.feature_set],
                    strict=True,
                )
            )
        )

    def lift_coefficient(
        self, angle_of_attack_rad: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        return np.clip(
            self.raw_lift_coefficient(angle_of_attack_rad, mach),
            self.minimum_lift_coefficient,
            self.maximum_lift_coefficient,
        )

    def lift_coefficient_scalar(self, angle_of_attack_rad: float, mach: float) -> float:
        return float(
            np.clip(
                self.raw_lift_coefficient_scalar(angle_of_attack_rad, mach),
                self.minimum_lift_coefficient,
                self.maximum_lift_coefficient,
            )
        )


__all__ = ["LIFT_FEATURE_SETS", "EffectiveLiftMap", "effective_lift_design"]
