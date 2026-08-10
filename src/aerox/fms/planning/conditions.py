"""Shared mass, atmosphere, wind, and route conditions for spatial planners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerox.navigation import WindCorrectedPath, wind_corrected_path

from ..environment import SpatialEnvironmentProfile
from ..route import FloatArray, SpatialTrajectory


@dataclass(frozen=True)
class PlanningConditions:
    """Node-aligned exogenous conditions used by spatial planning."""

    mass_kg: FloatArray
    static_temperature_k: FloatArray
    wind_north_mps: FloatArray
    wind_east_mps: FloatArray
    wind_down_mps: FloatArray
    ground_track_rad: FloatArray

    def subset(self, selection: slice | FloatArray) -> PlanningConditions:
        """Return conditions restricted to a matching route-grid selection."""

        return PlanningConditions(
            mass_kg=self.mass_kg[selection],
            static_temperature_k=self.static_temperature_k[selection],
            wind_north_mps=self.wind_north_mps[selection],
            wind_east_mps=self.wind_east_mps[selection],
            wind_down_mps=self.wind_down_mps[selection],
            ground_track_rad=self.ground_track_rad[selection],
        )

    def path_motion(
        self,
        *,
        along_path_m: FloatArray,
        altitude_m: FloatArray,
        true_airspeed_mps: FloatArray,
    ) -> WindCorrectedPath:
        """Solve air-relative motion for one altitude/speed profile."""

        ground_path_angle = np.arctan(np.gradient(altitude_m, along_path_m))
        return wind_corrected_path(
            ground_track_rad=self.ground_track_rad,
            ground_path_angle_rad=ground_path_angle,
            true_airspeed_mps=true_airspeed_mps,
            wind_north_mps=self.wind_north_mps,
            wind_east_mps=self.wind_east_mps,
            wind_down_mps=self.wind_down_mps,
        )


def planning_conditions(
    *,
    route: SpatialTrajectory,
    along_path_m: FloatArray,
    mass_kg: float,
    static_temperature_k: float,
    environment_profile: SpatialEnvironmentProfile | None,
    waypoint_mass_kg: FloatArray | None,
) -> PlanningConditions:
    """Interpolate all exogenous planning conditions onto a route grid."""

    if waypoint_mass_kg is None:
        mass = np.full_like(along_path_m, mass_kg)
    else:
        waypoint_mass = np.asarray(waypoint_mass_kg, dtype=float)
        if (
            waypoint_mass.shape != route.waypoint_along_path_m.shape
            or np.any(~np.isfinite(waypoint_mass))
            or np.any(waypoint_mass <= 0.0)
        ):
            raise ValueError("waypoint_mass_kg must match positive route anchors")
        mass = np.interp(along_path_m, route.waypoint_along_path_m, waypoint_mass)
    if environment_profile is None:
        temperature = np.full_like(along_path_m, static_temperature_k)
        wind_north = np.zeros_like(along_path_m)
        wind_east = np.zeros_like(along_path_m)
        wind_down = np.zeros_like(along_path_m)
    else:
        temperature, wind_north, wind_east, wind_down = (
            environment_profile.sample_arrays(along_path_m)
        )
    track = np.array(
        [route.sample(float(distance)).ground_track_rad for distance in along_path_m]
    )
    return PlanningConditions(
        mass_kg=mass,
        static_temperature_k=temperature,
        wind_north_mps=wind_north,
        wind_east_mps=wind_east,
        wind_down_mps=wind_down,
        ground_track_rad=track,
    )


__all__ = ["PlanningConditions", "planning_conditions"]
