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


def exposed_wing_area(
    area_m2: float,
    span_m: float,
    fuselage_width_m: float,
    root_chord_m: float,
) -> float:
    """Return the trapezoidal wing area outside the fuselage width.

    The reference trapezoid has centreline chord ``root_chord_m`` and the tip
    chord implied by area and span. The removed part is the exact trapezoid
    area between ``±fuselage_width_m / 2``.
    """

    values = np.asarray([area_m2, span_m, fuselage_width_m, root_chord_m], float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("wing and fuselage dimensions must be finite and positive")
    if fuselage_width_m >= span_m:
        raise ValueError("fuselage width must be smaller than the wing span")
    taper = 2.0 * area_m2 / (span_m * root_chord_m) - 1.0
    if not 0.0 <= taper <= 1.0:
        raise ValueError("root chord is inconsistent with wing area and span")
    buried = (
        root_chord_m
        * fuselage_width_m
        * (1.0 - (1.0 - taper) * fuselage_width_m / (2.0 * span_m))
    )
    return float(area_m2 - buried)


def trapezoidal_root_chord(area_m2: float, span_m: float, mac_m: float) -> float:
    """Return the centreline chord of the trapezoid with the given S, b, and MAC.

    A trapezoid satisfies ``MAC / (S/b) = 4/3 (1 + λ + λ²) / (1 + λ)²``. The
    taper ratio ``λ`` is solved from that identity and clipped to [0, 1].
    """

    values = np.asarray([area_m2, span_m, mac_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("wing area, span, and MAC must be finite and positive")
    ratio = float(np.clip(mac_m * span_m / area_m2, 1.0, 4.0 / 3.0))
    # (4/3 - r) λ² + (4/3 - 2r) λ + (4/3 - r) = 0; the roots are λ and 1/λ.
    a = 4.0 / 3.0 - ratio
    b = 4.0 / 3.0 - 2.0 * ratio
    if a <= 0.0:
        taper = 0.0
    else:
        taper = (-b - np.sqrt(max(b * b - 4.0 * a * a, 0.0))) / (2.0 * a)
    taper = float(np.clip(taper, 0.0, 1.0))
    return float(2.0 * area_m2 / (span_m * (1.0 + taper)))


def nacelle_wetted_area(max_diameter_m: float, length_m: float) -> float:
    """Return one nacelle's wetted area, ``2.8 D L`` (FLOPS, as in Aviary)."""

    values = np.asarray([max_diameter_m, length_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("nacelle dimensions must be finite and positive")
    return float(2.8 * max_diameter_m * length_m)


def nacelle_form_factor(max_diameter_m: float, length_m: float) -> float:
    """Return Raymer's nacelle form factor ``1 + 0.35 / (L / D)``."""

    values = np.asarray([max_diameter_m, length_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("nacelle dimensions must be finite and positive")
    return float(1.0 + 0.35 * max_diameter_m / length_m)


def fuselage_upsweep_drag_area(width_m: float, height_m: float) -> float:
    """Return Kroo's typical upsweep drag area, 0.006 on fuselage frontal area."""

    values = np.asarray([width_m, height_m], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("fuselage dimensions must be finite and positive")
    return float(0.006 * 0.25 * np.pi * width_m * height_m)


def control_surface_gap_drag_area(
    surfaces: tuple[tuple[float, float], ...],
) -> float:
    """Return Kroo's gap drag area ``0.0002 Σ cos²Λ S_affected``.

    ``surfaces`` holds ``(affected_area_m2, sweep_rad)`` pairs.
    """

    total = 0.0
    for area, sweep in surfaces:
        if not np.isfinite(area) or area < 0.0 or not np.isfinite(sweep):
            raise ValueError("gap areas and sweeps must be finite")
        total += 0.0002 * np.cos(sweep) ** 2 * area
    return float(total)


def nacelle_base_drag_area(exit_diameter_m: float) -> float:
    """Return Kroo's nozzle-gap base drag area for one nacelle.

    The base is a 0.5 in gap around the exit circumference with a base drag
    coefficient of 0.2.
    """

    if not np.isfinite(exit_diameter_m) or exit_diameter_m <= 0.0:
        raise ValueError("exit diameter must be finite and positive")
    return float(0.5 * 0.0254 * np.pi * exit_diameter_m * 0.2)


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
