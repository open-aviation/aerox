"""Aircraft path guidance and inverse flight-mechanics control."""

from .inverse import InverseFlightMechanicsController, PitchBankThrustCommand
from .path import GuidanceTarget, SpatialPathGuidance
from .reference import TrajectoryReference
from .selected_altitude import SelectedAltitudeGuidance, SelectedAltitudeMode
from aerox.navigation import (
    WindCorrectedPath,
    ground_track_to_air_heading,
    wind_corrected_path,
)

__all__ = [
    "GuidanceTarget",
    "InverseFlightMechanicsController",
    "PitchBankThrustCommand",
    "SpatialPathGuidance",
    "SelectedAltitudeGuidance",
    "SelectedAltitudeMode",
    "TrajectoryReference",
    "WindCorrectedPath",
    "ground_track_to_air_heading",
    "wind_corrected_path",
]
