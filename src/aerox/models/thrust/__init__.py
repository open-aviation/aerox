"""Aircraft thrust generation and autothrust models.

The package separates speed-tracking control from the effective N1 operating
envelope, spool response, and QAR-identified thrust map.
"""

from .control import SpeedTrackingAutothrustController
from .engine import (
    N1_ENVELOPE_FEATURE_SETS,
    THRUST_MAP_FEATURE_SETS,
    AsymmetricFirstOrderN1Actuator,
    AutothrustMode,
    DemonstratedClimbN1Capacity,
    EffectiveN1Envelope,
    EffectiveThrustMap,
    MeasuredN1ThrustAdapter,
    fan_speed_fraction,
    n1_envelope_features,
    thrust_map_features,
    total_temperature_ratio,
)
from .planning import (
    ConstantMaximumThrust,
    PressureScaledMinimumThrust,
    QARDemonstratedClimbThrustCapacity,
    QAREffectiveClimbThrustCeiling,
)

__all__ = [
    "N1_ENVELOPE_FEATURE_SETS",
    "THRUST_MAP_FEATURE_SETS",
    "AsymmetricFirstOrderN1Actuator",
    "AutothrustMode",
    "ConstantMaximumThrust",
    "DemonstratedClimbN1Capacity",
    "EffectiveN1Envelope",
    "EffectiveThrustMap",
    "MeasuredN1ThrustAdapter",
    "PressureScaledMinimumThrust",
    "QARDemonstratedClimbThrustCapacity",
    "QAREffectiveClimbThrustCeiling",
    "SpeedTrackingAutothrustController",
    "fan_speed_fraction",
    "n1_envelope_features",
    "thrust_map_features",
    "total_temperature_ratio",
]
