"""aerox aircraft data library."""

from . import aero, fms
from .aircraft import (
    Aircraft,
    Certification,
    CertifiedMassLimits,
    ControlSurface,
    DragModelConfig,
    Engine,
    Flaps,
    Fuselage,
    KinematicParameters,
    MassProperties,
    Pax,
    ReferenceGeometry,
    Tail,
    Wing,
    get_available_aircraft,
    load_aircraft,
    load_aircraft_from_dict,
)

__all__ = [
    "Aircraft",
    "Certification",
    "CertifiedMassLimits",
    "ControlSurface",
    "DragModelConfig",
    "Engine",
    "Flaps",
    "Fuselage",
    "KinematicParameters",
    "MassProperties",
    "Pax",
    "ReferenceGeometry",
    "Tail",
    "Wing",
    "aero",
    "fms",
    "get_available_aircraft",
    "load_aircraft",
    "load_aircraft_from_dict",
]
