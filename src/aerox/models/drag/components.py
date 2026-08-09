"""Component-level parasite-drag build-up equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray


def turbulent_flat_plate_skin_friction(
    reynolds_number: ArrayLike, mach: ArrayLike
) -> FloatArray:
    """Return mean compressible turbulent flat-plate skin friction."""

    reynolds, mach_array = np.broadcast_arrays(
        np.asarray(reynolds_number, dtype=float),
        np.asarray(mach, dtype=float),
    )
    if np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 1.0e5):
        raise ValueError("component reynolds_number must exceed 100,000")
    if np.any(~np.isfinite(mach_array)) or np.any(mach_array < 0.0):
        raise ValueError("mach must be finite and non-negative")
    return 0.455 / (np.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach_array**2) ** 0.65)


def laminar_flat_plate_skin_friction(reynolds_number: ArrayLike) -> FloatArray:
    """Return mean laminar flat-plate skin friction."""

    reynolds = np.asarray(reynolds_number, dtype=float)
    if np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 0.0):
        raise ValueError("component reynolds_number must be finite and positive")
    return 1.328 / np.sqrt(reynolds)


def lifting_surface_wetted_area(
    reference_area_m2: float, thickness_to_chord_ratio: float
) -> float:
    """Estimate two-sided lifting-surface wetted area."""

    if not np.isfinite(reference_area_m2) or reference_area_m2 <= 0.0:
        raise ValueError("reference_area_m2 must be finite and positive")
    if not np.isfinite(thickness_to_chord_ratio) or not (
        0.02 <= thickness_to_chord_ratio <= 0.30
    ):
        raise ValueError("thickness_to_chord_ratio is outside model support")
    return 2.0 * reference_area_m2 * (1.0 + 0.25 * thickness_to_chord_ratio)


def lifting_surface_form_factor(thickness_to_chord_ratio: float) -> float:
    """Return the compact Hoerner thickness form factor used by OpenVSP."""

    ratio = float(thickness_to_chord_ratio)
    if not np.isfinite(ratio) or not 0.02 <= ratio <= 0.30:
        raise ValueError("thickness_to_chord_ratio is outside model support")
    return 1.0 + 2.0 * ratio + 60.0 * ratio**4


def fuselage_wetted_area(length_m: float, width_m: float, height_m: float) -> float:
    """Estimate fuselage wetted area from an equivalent circular body."""

    values = np.asarray([length_m, width_m, height_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("fuselage dimensions must be finite and positive")
    diameter = np.sqrt(width_m * height_m)
    fineness = length_m / diameter
    if fineness <= 2.0:
        raise ValueError("fuselage fineness ratio must exceed two")
    return float(
        np.pi
        * diameter
        * length_m
        * (1.0 - 2.0 / fineness) ** (2.0 / 3.0)
        * (1.0 + 1.0 / fineness**2)
    )


def fuselage_form_factor(length_m: float, width_m: float, height_m: float) -> float:
    """Return a fineness-ratio fuselage form factor."""

    values = np.asarray([length_m, width_m, height_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("fuselage dimensions must be finite and positive")
    fineness = length_m / np.sqrt(width_m * height_m)
    if fineness <= 2.0:
        raise ValueError("fuselage fineness ratio must exceed two")
    return float(1.0 + 60.0 / fineness**3 + fineness / 400.0)


@dataclass(frozen=True)
class ParasiteDragComponent:
    """One wetted component in a skin-friction drag build-up."""

    name: str
    wetted_area_m2: float
    characteristic_length_m: float
    form_factor: float
    interference_factor: float = 1.0
    laminar_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component name must not be empty")
        values = np.asarray(
            [
                self.wetted_area_m2,
                self.characteristic_length_m,
                self.form_factor,
                self.interference_factor,
                self.laminar_fraction,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("component parameters must be finite")
        if min(values[:4]) <= 0.0:
            raise ValueError("component geometry and factors must be positive")
        if not 0.0 <= self.laminar_fraction <= 1.0:
            raise ValueError("laminar_fraction must lie between zero and one")

    def skin_friction_coefficient(
        self, unit_reynolds_number_per_m: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        """Return the blended component-average skin-friction coefficient."""

        reynolds = (
            np.asarray(unit_reynolds_number_per_m, dtype=float)
            * self.characteristic_length_m
        )
        turbulent = turbulent_flat_plate_skin_friction(reynolds, mach)
        if self.laminar_fraction == 0.0:
            return turbulent
        laminar = laminar_flat_plate_skin_friction(reynolds)
        return (
            self.laminar_fraction * laminar + (1.0 - self.laminar_fraction) * turbulent
        )

    def drag_coefficient(
        self,
        unit_reynolds_number_per_m: ArrayLike,
        mach: ArrayLike,
        reference_area_m2: float,
    ) -> FloatArray:
        """Return this component's zero-lift coefficient contribution."""

        return (
            self.skin_friction_coefficient(unit_reynolds_number_per_m, mach)
            * self.form_factor
            * self.interference_factor
            * self.wetted_area_m2
            / reference_area_m2
        )

    def drag_coefficient_scalar(
        self,
        unit_reynolds_number_per_m: float,
        mach: float,
        reference_area_m2: float,
    ) -> float:
        """Scalar fast path for recursive trajectory integration."""

        reynolds = unit_reynolds_number_per_m * self.characteristic_length_m
        if not np.isfinite(reynolds) or reynolds <= 1.0e5:
            raise ValueError("component reynolds_number must exceed 100,000")
        if not np.isfinite(mach) or mach < 0.0:
            raise ValueError("mach must be finite and non-negative")
        turbulent = 0.455 / (
            np.log10(reynolds) ** 2.58 * (1.0 + 0.144 * mach**2) ** 0.65
        )
        skin_friction = turbulent
        if self.laminar_fraction > 0.0:
            laminar = 1.328 / np.sqrt(reynolds)
            skin_friction = (
                self.laminar_fraction * laminar
                + (1.0 - self.laminar_fraction) * turbulent
            )
        return float(
            skin_friction
            * self.form_factor
            * self.interference_factor
            * self.wetted_area_m2
            / reference_area_m2
        )
