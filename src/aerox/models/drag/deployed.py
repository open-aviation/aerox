"""Drag of deployed trailing-edge flaps and landing gear.

Flap profile drag, McCormick (1995), eq. 3.45-3.46, with the flapped wing
area ``S_wf``:

    dCD0_flap = lambda_f * (cf/c)^1.38 * (S_wf/S) * sin^2(delta)

Lift-dependent drag with flaps: the span efficiency factor falls linearly with
the flap angle, by 0.0025 per degree. This matches Roskam (Part I, Table 3.6):
``e - 0.05`` for take-off flaps (about 20 degrees) and ``e - 0.10`` for landing
flaps (about 40 degrees).

Landing gear, Torenbeek's correlation as given by Anderson (1999):

    dCD0_gear = g * (MTOW g0 / S) * K_uc * MTOW^-0.215,   K_uc = 5.81e-5

with the gear extension fraction ``g`` between zero and one. For the A320 it
gives 0.0322; the value identified from flight data is 0.0299 (Deiler and
Fezans, AIAA 2025-1250). Slats add no drag term, and flap-gear interference is
not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from aerox.aircraft import Aircraft

G0 = 9.80665

#: Largest flap angle the model supports, in radians (60 degrees).
MAXIMUM_FLAP_ANGLE_RAD = math.radians(60.0)

#: Loss of span efficiency per radian of flap angle (0.0025 per degree).
SPAN_EFFICIENCY_LOSS_PER_RAD = 0.0025 * 180.0 / math.pi

#: Torenbeek gear coefficient with flaps retracted, in SI units.
GEAR_COEFFICIENT = 5.81e-5

_SLOTTED_EFFECTIVENESS = 0.9
_PLAIN_EFFECTIVENESS = 1.7
_SLOTTED_CHORD_RATIO = 0.30
_PLAIN_CHORD_RATIO = 0.25
_TRANSPORT_FLAPPED_AREA_RATIO = 0.65
_LIGHT_AIRCRAFT_FLAPPED_AREA_RATIO = 0.45


@dataclass(frozen=True)
class DeployedDrag:
    """Constants of the flap and landing gear drag of one aircraft.

    ``flap_parasite_factor`` is ``lambda_f (cf/c)^1.38 S_wf/S``, and
    ``gear_parasite`` is the zero-lift drag coefficient of the fully extended
    landing gear.
    """

    flap_parasite_factor: float
    gear_parasite: float
    span_efficiency_loss_per_rad: float = SPAN_EFFICIENCY_LOSS_PER_RAD

    def __post_init__(self) -> None:
        values = (
            self.flap_parasite_factor,
            self.gear_parasite,
            self.span_efficiency_loss_per_rad,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("deployed drag constants must be finite and non-negative")

    def flap_parasite(self, flap_angle_rad: Any, xp: Any = np) -> Any:
        """Zero-lift drag coefficient of the flaps."""

        return self.flap_parasite_factor * xp.sin(flap_angle_rad) ** 2

    def span_efficiency_loss(self, flap_angle_rad: Any) -> Any:
        """Decrease of the span efficiency factor with flaps."""

        return self.span_efficiency_loss_per_rad * flap_angle_rad


def deployed_drag_from_aircraft(
    aircraft: Aircraft, reference_area_m2: float
) -> DeployedDrag | None:
    """Flap and gear drag constants from an aircraft record.

    The record values of ``flaps.lambda_f``, ``flaps.cf/c``, and ``flaps.bf/b``
    are used when present. Otherwise the flap type sets the defaults: slotted
    flaps ``lambda_f = 0.9`` and ``cf/c = 0.30``, plain or split flaps 1.7 and
    0.25. Without ``bf/b``, the flapped area is 0.45 of the wing area for
    piston aircraft and 0.65 for the others. Returns ``None`` without MTOW.
    """

    mtow = aircraft.mtow
    if mtow is None or not np.isfinite(mtow) or mtow <= 0.0:
        return None
    flaps = aircraft.flaps
    flap_type = str(getattr(flaps, "type", None) or "").lower()
    plain = "plain" in flap_type or "split" in flap_type
    effectiveness = _record_value(flaps, "lambda_f") or (
        _PLAIN_EFFECTIVENESS if plain else _SLOTTED_EFFECTIVENESS
    )
    chord_ratio = _record_value(flaps, "cf_c") or (
        _PLAIN_CHORD_RATIO if plain else _SLOTTED_CHORD_RATIO
    )
    piston = aircraft.engine is not None and aircraft.engine.type == "piston"
    flapped_area_ratio = _record_value(flaps, "bf_b") or (
        _LIGHT_AIRCRAFT_FLAPPED_AREA_RATIO if piston else _TRANSPORT_FLAPPED_AREA_RATIO
    )
    wing_loading = mtow * G0 / reference_area_m2
    return DeployedDrag(
        flap_parasite_factor=effectiveness * chord_ratio**1.38 * flapped_area_ratio,
        gear_parasite=wing_loading * GEAR_COEFFICIENT * mtow**-0.215,
    )


def _record_value(flaps: Any, name: str) -> float | None:
    value = getattr(flaps, name, None)
    if value is None or not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)
