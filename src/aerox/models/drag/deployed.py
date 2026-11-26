"""Low-speed high-lift drag and the Torenbeek landing gear correlation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.interpolate import PchipInterpolator

    from aerox.aircraft import Aircraft

G0 = 9.80665
MAXIMUM_FLAP_ANGLE_RAD = math.radians(60.0)
MAXIMUM_SLAT_ANGLE_RAD = math.radians(60.0)

#: Torenbeek gear coefficient with flaps retracted, in SI units.
GEAR_COEFFICIENT = 5.81e-5

# GASP (NASA CR-152303, Vol. III, section III.1.4.1), tables as published
# in Aviary. Columns are optimum flap angle in degrees, DCDOTE, DCLMTE.
_FLAP_TYPE_CONSTANTS = {
    "plain": (60.0, 0.12, 0.90),
    "split": (60.0, 0.23, 0.80),
    "single-slotted": (40.0, 0.13, 1.18),
    "double-slotted": (55.0, 0.23, 1.40),
    "triple-slotted": (55.0, 0.23, 1.60),
    "fowler": (30.0, 0.10, 1.67),
    "double-slotted-fowler": (30.0, 0.15, 2.25),
}
_FLAP_ANGLES = (0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 44, 50, 55, 60)
_FLAP_EFFICIENCY = (
    _FLAP_ANGLES,
    (
        1,
        0.995,
        0.99,
        0.98,
        0.97,
        0.955,
        0.935,
        0.90,
        0.875,
        0.855,
        0.83,
        0.80,
        0.70,
        0.54,
        0.30,
    ),
)
_FLAP_LIFT_SHIFT = (
    _FLAP_ANGLES,
    (
        0,
        0.16,
        0.285,
        0.375,
        0.435,
        0.48,
        0.52,
        0.55,
        0.575,
        0.58,
        0.59,
        0.60,
        0.62,
        0.635,
        0.65,
    ),
)
_FLAP_DRAG_ANGLE = (
    (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 2.5, 2.75, 3),
    (0, 0.18, 0.37, 0.65, 1, 1.97, 3.44, 4.15, 4.55, 4.82, 5),
)
_FLAP_LIFT_ANGLE = {
    "hinged": (
        _FLAP_ANGLES,
        (
            0,
            0.12,
            0.23,
            0.34,
            0.43,
            0.53,
            0.62,
            0.71,
            0.76,
            0.80,
            0.82,
            0.86,
            0.94,
            0.98,
            1,
        ),
    ),
    "slotted": (
        _FLAP_ANGLES,
        (
            0,
            0.22,
            0.41,
            0.57,
            0.71,
            0.83,
            0.91,
            0.975,
            0.995,
            1,
            0.997,
            0.992,
            0.945,
            0.85,
            0.75,
        ),
    ),
    "fowler": (
        _FLAP_ANGLES,
        (
            0,
            0.25,
            0.46,
            0.65,
            0.80,
            0.92,
            1,
            1.07,
            1.10,
            1.11,
            1.10,
            1.07,
            0.85,
            0.56,
            0.20,
        ),
    ),
}

_CHORD_RATIOS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
_THICKNESS_RATIOS = (
    0,
    0.04,
    0.06,
    0.07,
    0.08,
    0.10,
    0.11,
    0.12,
    0.14,
    0.15,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.28,
)
_FLAP_LIFT_CHORD = {
    "hinged": (0, 0.72, 0.94, 1, 0.95, 0.73),
    "slotted": (0, 0.575, 0.83, 1, 1.065, 1.09),
    "fowler": (0, 0.41, 0.73, 1, 1.22, 1.40),
}


@cache
def _curve(knots: tuple[float, ...], values: tuple[float, ...]) -> PchipInterpolator:
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(knots, values)


def _interpolate(
    table: tuple[tuple[float, ...], tuple[float, ...]], x: Any, xp: Any
) -> Any:
    """Evaluate the same piecewise cubic in NumPy or a symbolic backend."""

    curve = _curve(*table)
    if xp is np:
        return curve(x)
    evaluate = getattr(xp, "if_else", None) or getattr(xp, "where", None)
    if evaluate is None:
        raise AttributeError("flap drag needs a math module with 'where' or 'if_else'")
    pieces = zip(curve.x[:-1], curve.c.T, strict=True)
    start, (a, b, c, d) = next(pieces)
    offset = x - float(start)
    result = ((float(a) * offset + float(b)) * offset + float(c)) * offset + float(d)
    for start, (a, b, c, d) in pieces:
        left = float(start)
        offset = x - left
        value = ((float(a) * offset + float(b)) * offset + float(c)) * offset + float(d)
        result = evaluate(x >= left, value, result)
    return result


def _flap_angle_deg(flap_angle_rad: Any, xp: Any) -> Any:
    """Flap angle in degrees, held inside the tabulated range of 0 to 60.

    The numeric drag models reject angles outside this range. The expression
    does not check its inputs, so trial points of an optimiser are clamped.
    """

    maximum = getattr(xp, "maximum", None) or xp.fmax
    limit = math.degrees(MAXIMUM_FLAP_ANGLE_RAD)
    angle = maximum(flap_angle_rad * (180.0 / math.pi), 0.0)
    return limit - maximum(limit - angle, 0.0)


@dataclass(frozen=True)
class DeployedDrag:
    """Aircraft constants for drag increments from deployed surfaces and gear.

    Each flap profile pair holds the optimum angle in degrees and the drag
    increment at that angle. The lift factor scales the GASP flap lift
    increment. The slat factor scales the clean wing profile drag.
    """

    gear_parasite: float
    flap_profile_factors: tuple[tuple[float, float], ...]
    flap_lift_factor: float
    flap_family: str
    slat_profile_factor: float

    def __post_init__(self) -> None:
        values = (
            self.gear_parasite,
            self.flap_lift_factor,
            self.slat_profile_factor,
            *(value for pair in self.flap_profile_factors for value in pair),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("deployed drag constants must be finite and non-negative")
        if self.flap_family not in _FLAP_LIFT_ANGLE:
            raise ValueError(f"unsupported flap family: {self.flap_family!r}")

    def flap_parasite(self, flap_angle_rad: Any, xp: Any = np) -> Any:
        """GASP minimum profile drag increment from trailing-edge flaps."""

        angle = _flap_angle_deg(flap_angle_rad, xp)
        return sum(
            factor * _interpolate(_FLAP_DRAG_ANGLE, angle / optimum, xp)
            for optimum, factor in self.flap_profile_factors
        )

    def span_induced(
        self, cl: Any, clean_factor: float, flap_angle_rad: Any, xp: Any = np
    ) -> Any:
        """Span-induced drag at total aircraft lift (GASP eq. III.1.58)."""

        if self.flap_lift_factor == 0.0:
            return clean_factor * cl**2 + 0.0 * flap_angle_rad
        angle = _flap_angle_deg(flap_angle_rad, xp)
        delta_cl = self.flap_lift_factor * _interpolate(
            _FLAP_LIFT_ANGLE[self.flap_family], angle, xp
        )
        shift = _interpolate(_FLAP_LIFT_SHIFT, angle, xp) * delta_cl
        efficiency = _interpolate(_FLAP_EFFICIENCY, angle, xp)
        return clean_factor * (cl - shift) ** 2 / efficiency

    def slat_profile(
        self, cl: Any, wing_parasite: Any, slat_angle_rad: Any, xp: Any = np
    ) -> Any:
        """Torenbeek eq. G-62, scaled by the sine of the slat angle."""

        return (
            wing_parasite
            * (1.0 + cl**2)
            * self.slat_profile_factor
            * xp.sin(slat_angle_rad)
        )


def deployed_drag_from_aircraft(
    aircraft: Aircraft,
    reference_area_m2: float,
    wing_span_m: float,
    wing_thickness_to_chord_ratio: float,
    wing_sweep_deg: float,
    fuselage_width_m: float,
) -> DeployedDrag | None:
    """Resolve high-lift geometry, published correlations and class defaults."""

    mtow = aircraft.mtow
    if mtow is None or not np.isfinite(mtow) or mtow <= 0.0:
        return None

    flaps = aircraft.flaps
    propulsion = aircraft.engine.type if aircraft.engine else None
    piston = propulsion == "piston"
    kind = (flaps.type if flaps else None) or {
        "piston": "plain",
        "turboprop": "single-slotted",
    }.get(propulsion, "double-slotted")
    kinds = (
        ("single-slotted", "double-slotted")
        if kind == "mixed-single-double-slotted"
        else (kind,)
    )
    hinged = kind in {"plain", "split"}
    family = "hinged" if hinged else "fowler" if "fowler" in kind else "slotted"

    area = flaps.flapped_area_ratio if flaps else None
    if area is None and flaps and flaps.eta_outboard is not None:
        inner = flaps.eta_inboard
        if inner is None:
            inner = min(fuselage_width_m / wing_span_m, flaps.eta_outboard)
        outer = flaps.eta_outboard
        taper = (aircraft.wing.taper_ratio if aircraft.wing else None) or (
            0.60 if piston else 0.33
        )
        # Integral of a trapezoidal wing chord between the two stations.
        area = (outer - inner) * (2 - (1 - taper) * (outer + inner)) / (1 + taper)
    if area is None:
        area = 0.45 if piston else 0.65
    chord = flaps.chord_ratio if flaps else None
    if chord is None:
        chord = 0.25 if hinged else 0.30
    slatted_area = flaps.slatted_area_ratio if flaps else None
    if slatted_area is None:
        slatted_area = 0.65
    cos_sweep = math.cos(math.radians(wing_sweep_deg))

    # Use the wing area served by the devices for the GASP span and body
    # factors. The station integral already excludes the fuselage cutout.
    drag_chord = np.interp(
        chord,
        _CHORD_RATIOS,
        (0, 0.32, 0.66, 1, 1.32, 1.70) if hinged else (0, 0.24, 0.55, 1, 1.60, 2.20),
    )
    lift_chord = np.interp(chord, _CHORD_RATIOS, _FLAP_LIFT_CHORD[family])
    aspect_ratio = wing_span_m**2 / reference_area_m2
    lift_aspect = np.interp(
        aspect_ratio,
        (0, 0.2, 0.6, 1, 1.4, 2, 2.5, 3, 3.5, 4, 4.3, 5, 7, 9, 10, 11.2, 12),
        (
            0,
            0.1,
            0.24,
            0.33,
            0.41,
            0.50,
            0.56,
            0.61,
            0.66,
            0.70,
            0.72,
            0.77,
            0.88,
            0.95,
            0.97,
            0.99,
            1,
        ),
    )
    lift_thickness = np.interp(
        wing_thickness_to_chord_ratio,
        _THICKNESS_RATIOS,
        (
            1.25,
            1.17,
            1.08,
            1.05,
            1.02,
            1,
            1.02,
            1.05,
            1.20,
            1.36,
            1.60,
            1.87,
            2.02,
            2.12,
            2.18,
            2.20,
        )
        if hinged
        else (
            0.84,
            0.86,
            0.89,
            0.91,
            0.94,
            1,
            1.04,
            1.10,
            1.26,
            1.33,
            1.39,
            1.49,
            1.55,
            1.58,
            1.59,
            1.60,
        ),
    )
    drag_scale = float(drag_chord * area * cos_sweep / len(kinds))
    lift_scale = float(
        lift_chord * lift_aspect * lift_thickness * area * cos_sweep**3 / len(kinds)
    )
    constants = [_FLAP_TYPE_CONSTANTS[entry] for entry in kinds]
    return DeployedDrag(
        gear_parasite=mtow * G0 / reference_area_m2 * GEAR_COEFFICIENT * mtow**-0.215,
        flap_profile_factors=tuple(
            (angle, drag * drag_scale) for angle, drag, _ in constants
        ),
        flap_lift_factor=sum(lift * lift_scale for _, _, lift in constants),
        flap_family=family,
        # G-62 slat chord ratio from Fig. G-24. The sine scaling uses 20 deg,
        # Aviary's default optimum slat angle.
        slat_profile_factor=slatted_area
        * 0.17
        * cos_sweep
        / math.sin(math.radians(20)),
    )
