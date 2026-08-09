"""Aircraft kinematic-performance parameters stored in aircraft YAML files."""

from __future__ import annotations

from pathlib import Path

from aerox.aircraft import Aircraft, KinematicParameters, load_aircraft


def kinematic_parameters_from_aircraft(aircraft: Aircraft) -> KinematicParameters:
    """Return the populated kinematic section of an aircraft record."""

    if aircraft.kinematic is None:
        raise ValueError(f"aircraft {aircraft.icao!r} has no kinematic parameters")
    return aircraft.kinematic


def load_kinematic_parameters(
    icao: str,
    *,
    data_dir: str | Path | None = None,
) -> KinematicParameters:
    """Load one aircraft's kinematic parameters from the YAML database."""

    aircraft = load_aircraft(icao, data_dir=data_dir)
    return kinematic_parameters_from_aircraft(aircraft)


__all__ = [
    "KinematicParameters",
    "kinematic_parameters_from_aircraft",
    "load_kinematic_parameters",
]
