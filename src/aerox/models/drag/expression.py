"""The geometry drag model as a closed-form expression for any math backend.

A built drag model holds constants that depend only on the aircraft. At a
flight condition, the drag coefficient is a short formula of the lift
coefficient, the Mach number, and the Reynolds number per metre:

    Re_i  = Re_per_m * L_i
    Cf_i  = 0.455 / (log10(Re_i)^2.58 * (1 + 0.144 M^2)^0.65)    (turbulent)
    Cf_i  = f_i * 1.328 / sqrt(Re_i) + (1 - f_i) * Cf_i          (if laminar)
    CD0   = dCD0 + sum(w_i * Cf_i)
    CD    = CD0 + k_span * cl^2 + k_profile * CD0 * cl^2
          + 20 * max(M - (Mcrit0 + Mcrit_slope * cl), 0)^4       (wave drag)

With flaps deployed by ``delta`` radians and the landing gear extended by a
fraction ``g``:

    k_span -> k_span / (1 - r_flap * delta)
    CD    += c_flap * sin(delta)^2 + g * dCD0_gear

``DragExpression`` stores these constants and evaluates the formula with the
functions of a math module passed as ``xp``. NumPy is the default. Any module
or object with ``log``, ``sqrt``, and ``maximum`` works (and ``sin`` when a
flap angle is given), for example
``jax.numpy`` or OpenAP's CasADi backend, so the same drag model can be used
in symbolic and differentiable code.

The expression does not check its inputs. ``drag_coefficient()`` of the drag
model does, and should be used for NumPy evaluation where errors matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_LN10 = math.log(10.0)


@dataclass(frozen=True)
class DragExpression:
    """Constants of a geometry drag model and its closed-form expression.

    One entry per wetted component in ``lengths_m``, ``weights``, and
    ``laminar_fractions``. ``weights`` already include the form factor, the
    interference factor, the wetted to reference area ratio, and the parasite
    scale. The critical Mach terms are ``None`` for models without wave drag.
    ``flap_parasite_factor``, ``flap_efficiency_ratio`` (the loss of span
    efficiency per radian of flap divided by the span efficiency factor), and
    ``gear_parasite`` are zero for models without flap and gear drag.
    """

    lengths_m: tuple[float, ...]
    weights: tuple[float, ...]
    laminar_fractions: tuple[float, ...]
    parasite_increment: float
    span_induced_factor: float
    profile_factor: float
    critical_mach_zero_lift: float | None = None
    critical_mach_slope: float | None = None
    flap_parasite_factor: float = 0.0
    flap_efficiency_ratio: float = 0.0
    gear_parasite: float = 0.0

    @property
    def has_wave_drag(self) -> bool:
        return self.critical_mach_zero_lift is not None

    def parasite(self, mach: Any, reynolds_per_m: Any, xp: Any = np) -> Any:
        """Zero-lift drag coefficient CD0 at a Mach and Reynolds number."""

        compressibility = (1.0 + 0.144 * mach**2) ** 0.65
        total = self.parasite_increment
        for length, weight, laminar in zip(
            self.lengths_m, self.weights, self.laminar_fractions
        ):
            reynolds = reynolds_per_m * length
            log10_re = xp.log(reynolds) / _LN10
            friction = 0.455 / (log10_re**2.58 * compressibility)
            if laminar > 0.0:
                friction = (
                    laminar * 1.328 / xp.sqrt(reynolds) + (1.0 - laminar) * friction
                )
            total = total + weight * friction
        return total

    def critical_mach(self, cl: Any) -> Any:
        """Critical Mach number at a lift coefficient (models with wave drag)."""

        if not self.has_wave_drag:
            raise ValueError("this drag model has no wave drag")
        return self.critical_mach_zero_lift + self.critical_mach_slope * cl

    def wave(self, cl: Any, mach: Any, xp: Any = np) -> Any:
        """Wave drag coefficient; zero for models without wave drag."""

        if not self.has_wave_drag:
            return 0.0 * mach
        excess = xp.maximum(mach - self.critical_mach(cl), 0.0)
        return 20.0 * excess**4

    def drag_coefficient(
        self,
        cl: Any,
        mach: Any,
        reynolds_per_m: Any,
        xp: Any = np,
        *,
        flap_angle_rad: Any = None,
        gear_extension_fraction: Any = None,
    ) -> Any:
        """Total drag coefficient, evaluated with the functions of ``xp``.

        ``flap_angle_rad`` and ``gear_extension_fraction`` default to the clean
        configuration.
        """

        cd0 = self.parasite(mach, reynolds_per_m, xp)
        lift_squared = cl**2
        induced = self.span_induced_factor * lift_squared
        if flap_angle_rad is not None:
            induced = induced / (1.0 - self.flap_efficiency_ratio * flap_angle_rad)
        profile = self.profile_factor * cd0 * lift_squared
        total = cd0 + induced + profile + self.wave(cl, mach, xp)
        if flap_angle_rad is not None:
            total = total + self.flap_parasite_factor * xp.sin(flap_angle_rad) ** 2
        if gear_extension_fraction is not None:
            total = total + self.gear_parasite * gear_extension_fraction
        return total

    def cd(
        self,
        cl: Any,
        mach: Any,
        reynolds_per_m: Any,
        xp: Any = np,
        *,
        flap_angle_rad: Any = None,
        gear_extension_fraction: Any = None,
    ) -> Any:
        """Shorthand for ``drag_coefficient()``."""

        return self.drag_coefficient(
            cl,
            mach,
            reynolds_per_m,
            xp,
            flap_angle_rad=flap_angle_rad,
            gear_extension_fraction=gear_extension_fraction,
        )
