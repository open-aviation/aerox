"""Shared interfaces and value objects for drag-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...aero import reynolds_per_m as isa_reynolds_per_m

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DragCondition:
    """Air-relative flow condition and aircraft configuration.

    Reynolds number is supplied per metre so each component can apply its own
    characteristic length. When it is ``None``, a model that depends on it uses
    its own reference value. Configuration inputs are explicit so
    clean-aircraft models cannot silently evaluate a deployed configuration.
    """

    mach: ArrayLike
    reynolds_per_m: ArrayLike | None = None
    flap_angle_rad: ArrayLike = 0.0
    gear_extension_fraction: ArrayLike = 0.0
    spoiler_deflection_rad: ArrayLike = 0.0

    def with_default_reynolds(self, reynolds_per_m: float) -> DragCondition:
        """Return this condition with a Reynolds number filled in if missing."""

        if self.reynolds_per_m is not None:
            return self
        return replace(self, reynolds_per_m=reynolds_per_m)

    def broadcast_with(
        self, lift_coefficient: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Validate and broadcast the condition with a lift coefficient.

        A missing Reynolds number is returned as NaN.
        """

        reynolds_given = self.reynolds_per_m is not None
        lift, mach, reynolds, flap, gear, spoiler = np.broadcast_arrays(
            np.asarray(lift_coefficient, dtype=float),
            np.asarray(self.mach, dtype=float),
            np.asarray(self.reynolds_per_m if reynolds_given else np.nan, dtype=float),
            np.asarray(self.flap_angle_rad, dtype=float),
            np.asarray(self.gear_extension_fraction, dtype=float),
            np.asarray(self.spoiler_deflection_rad, dtype=float),
        )
        if np.any(~np.isfinite(lift)):
            raise ValueError("lift coefficient (cl) must be finite")
        if np.any(~np.isfinite(mach)) or np.any(mach < 0.0):
            raise ValueError("mach must be finite and non-negative")
        if reynolds_given and (
            np.any(~np.isfinite(reynolds)) or np.any(reynolds <= 0.0)
        ):
            raise ValueError("reynolds_per_m must be finite and positive")
        if np.any(~np.isfinite(flap)) or np.any(flap < 0.0):
            raise ValueError("flap_angle_rad must be finite and non-negative")
        if np.any(~np.isfinite(gear)) or np.any((gear < 0.0) | (gear > 1.0)):
            raise ValueError("gear_extension_fraction must lie between zero and one")
        if np.any(~np.isfinite(spoiler)) or np.any(spoiler < 0.0):
            raise ValueError("spoiler_deflection_rad must be finite and non-negative")
        return lift, mach, reynolds, flap, gear, spoiler


def require_clean(flap: FloatArray, gear: FloatArray, spoiler: FloatArray) -> None:
    """Reject a deployed configuration in a clean-configuration model."""

    if np.any(flap > 0.0) or np.any(gear > 0.0) or np.any(spoiler > 0.0):
        raise ValueError(
            "this drag model is supported only when clean: "
            "flaps, landing gear, and spoilers must be retracted"
        )


def resolve_condition(
    mach: ArrayLike | None,
    reynolds_per_m: ArrayLike | None,
    condition: DragCondition | None,
    *,
    h: ArrayLike | None = None,
    dT: ArrayLike | None = None,
) -> DragCondition:
    """Return the condition given as arguments or as a ``DragCondition``.

    With the altitude ``h`` in metres, and optionally the ISA temperature
    offset ``dT`` in K, the Reynolds number per metre is computed with
    :func:`aerox.aero.reynolds_per_m`.
    """

    if condition is not None:
        if mach is not None or reynolds_per_m is not None or h is not None:
            raise TypeError("give either mach with h or reynolds_per_m, or condition")
        if dT is not None:
            raise TypeError("dT is used only with h")
        return condition
    if mach is None:
        raise TypeError("mach is required")
    if h is not None:
        if reynolds_per_m is not None:
            raise TypeError("give either h or reynolds_per_m, not both")
        reynolds_per_m = isa_reynolds_per_m(mach, h, 0 if dT is None else dT)
    elif dT is not None:
        raise TypeError("dT is used only with h")
    return DragCondition(mach=mach, reynolds_per_m=reynolds_per_m)


class DragBreakdownLike(Protocol):
    """Any drag breakdown: it has at least the total drag coefficient."""

    @property
    def total(self) -> FloatArray: ...


class CdShorthand:
    """Adds ``cd(...)`` as a shorthand for ``drag_coefficient(...)``.

    ``cd`` calls ``self.drag_coefficient`` so that a subclass overriding
    ``drag_coefficient`` also changes ``cd``.
    """

    def cd(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        """Return the drag coefficient; shorthand for ``drag_coefficient``."""

        return self.drag_coefficient(  # type: ignore[attr-defined]
            cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
        )


@runtime_checkable
class DragModel(Protocol):
    """Drag-coefficient model used by the flight mechanics.

    Every model is evaluated as ``drag_coefficient(cl, mach, reynolds_per_m)``,
    as ``drag_coefficient(cl, mach, h=..., dT=...)`` with the Reynolds number
    from the ISA atmosphere, or with ``condition=DragCondition(...)``. The Aerox models also have the
    shorthand ``cd(...)``. ``breakdown`` takes the same arguments and returns
    the drag parts. ``_drag_coefficient_scalar`` is a fast path
    for scalar conditions used inside integrators.
    """

    def breakdown(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> DragBreakdownLike: ...

    def drag_coefficient(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray: ...

    def _drag_coefficient_scalar(
        self, cl: float, condition: DragCondition
    ) -> float: ...


@dataclass(frozen=True)
class DragBreakdown:
    """Common drag-coefficient components returned by compact models."""

    parasite: FloatArray
    induced: FloatArray
    wave: FloatArray
    total: FloatArray
    critical_mach: FloatArray
