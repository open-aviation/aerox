"""Wind-aware conversion between desired ground paths and air-relative motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WindCorrectedPath:
    """Air-relative commands and horizontal progress for a desired ground path."""

    air_heading_rad: FloatArray
    air_flight_path_angle_rad: FloatArray
    horizontal_ground_speed_mps: FloatArray


def wind_corrected_path(
    *,
    ground_track_rad: ArrayLike,
    ground_path_angle_rad: ArrayLike,
    true_airspeed_mps: ArrayLike,
    wind_north_mps: ArrayLike = 0.0,
    wind_east_mps: ArrayLike = 0.0,
    wind_down_mps: ArrayLike = 0.0,
) -> WindCorrectedPath:
    """Solve the 3D wind triangle for a desired ground-velocity direction.

    Ground distance is horizontal route distance. Positive path angle denotes
    climb, while the wind's vertical component follows the NED convention and
    is therefore positive downward.
    """

    track, path_angle, speed, wind_north, wind_east, wind_down = np.broadcast_arrays(
        np.asarray(ground_track_rad, dtype=float),
        np.asarray(ground_path_angle_rad, dtype=float),
        np.asarray(true_airspeed_mps, dtype=float),
        np.asarray(wind_north_mps, dtype=float),
        np.asarray(wind_east_mps, dtype=float),
        np.asarray(wind_down_mps, dtype=float),
    )
    if np.any(
        ~np.isfinite(track + path_angle + speed + wind_north + wind_east + wind_down)
    ) or np.any(speed <= 1.0):
        raise ValueError(
            "path, airspeed, and wind must be finite with positive airspeed"
        )
    if np.any(np.abs(path_angle) >= np.deg2rad(89.0)):
        raise ValueError("ground-path angle must remain below 89 degrees")

    direction_north = np.cos(track)
    direction_east = np.sin(track)
    direction_down = -np.tan(path_angle)
    quadratic = 1.0 + direction_down**2
    projection = (
        direction_north * wind_north
        + direction_east * wind_east
        + direction_down * wind_down
    )
    constant = wind_north**2 + wind_east**2 + wind_down**2 - speed**2
    discriminant = projection**2 - quadratic * constant
    if np.any(discriminant <= 0.0):
        raise ValueError(
            "wind or crosswind makes the requested 3D ground path unattainable"
        )
    ground_speed = (projection + np.sqrt(discriminant)) / quadratic
    if np.any(ground_speed <= 0.0):
        raise ValueError("wind leaves no positive horizontal route progress")

    air_north = ground_speed * direction_north - wind_north
    air_east = ground_speed * direction_east - wind_east
    air_down = ground_speed * direction_down - wind_down
    horizontal_air_speed = np.hypot(air_north, air_east)
    air_heading = np.arctan2(air_east, air_north)
    air_path_angle = np.arctan2(-air_down, horizontal_air_speed)
    return WindCorrectedPath(
        air_heading_rad=np.asarray(air_heading, dtype=float),
        air_flight_path_angle_rad=np.asarray(air_path_angle, dtype=float),
        horizontal_ground_speed_mps=np.asarray(ground_speed, dtype=float),
    )


def ground_track_to_air_heading(
    ground_track_rad: float,
    true_airspeed_mps: float,
    wind_north_mps: float,
    wind_east_mps: float,
    ground_path_angle_rad: float = 0.0,
    wind_down_mps: float = 0.0,
) -> tuple[float, float]:
    """Return air heading and horizontal groundspeed for one desired path."""

    solution = wind_corrected_path(
        ground_track_rad=ground_track_rad,
        ground_path_angle_rad=ground_path_angle_rad,
        true_airspeed_mps=true_airspeed_mps,
        wind_north_mps=wind_north_mps,
        wind_east_mps=wind_east_mps,
        wind_down_mps=wind_down_mps,
    )
    return float(solution.air_heading_rad), float(solution.horizontal_ground_speed_mps)


def horizontal_ground_speed_mps(
    true_airspeed_mps: ArrayLike,
    flight_path_angle_rad: ArrayLike,
    along_track_wind_mps: ArrayLike = 0.0,
) -> FloatArray:
    """Return horizontal progress when air heading is already aligned to track."""

    speed, angle, wind = np.broadcast_arrays(
        np.asarray(true_airspeed_mps, dtype=float),
        np.asarray(flight_path_angle_rad, dtype=float),
        np.asarray(along_track_wind_mps, dtype=float),
    )
    if np.any(~np.isfinite(speed + angle + wind)) or np.any(speed <= 1.0):
        raise ValueError("airspeed, flight-path angle, and wind must be finite")
    if np.any(np.abs(angle) >= np.deg2rad(89.0)):
        raise ValueError("flight-path angle must remain below 89 degrees")
    result = speed * np.cos(angle) + wind
    if np.any(result <= 0.0):
        raise ValueError("wind leaves no positive horizontal route progress")
    return np.asarray(result, dtype=float)


def ground_path_angle_from_air_motion(
    *,
    ground_track_rad: ArrayLike,
    air_flight_path_angle_rad: ArrayLike,
    true_airspeed_mps: ArrayLike,
    wind_north_mps: ArrayLike = 0.0,
    wind_east_mps: ArrayLike = 0.0,
    wind_down_mps: ArrayLike = 0.0,
) -> FloatArray:
    """Return the ground-path angle produced while holding ground track."""

    track, air_angle, speed, wind_north, wind_east, wind_down = np.broadcast_arrays(
        np.asarray(ground_track_rad, dtype=float),
        np.asarray(air_flight_path_angle_rad, dtype=float),
        np.asarray(true_airspeed_mps, dtype=float),
        np.asarray(wind_north_mps, dtype=float),
        np.asarray(wind_east_mps, dtype=float),
        np.asarray(wind_down_mps, dtype=float),
    )
    if np.any(
        ~np.isfinite(track + air_angle + speed + wind_north + wind_east + wind_down)
    ) or np.any(speed <= 1.0):
        raise ValueError(
            "path, airspeed, and wind must be finite with positive airspeed"
        )
    right_north = -np.sin(track)
    right_east = np.cos(track)
    tangent_north = np.cos(track)
    tangent_east = np.sin(track)
    crosswind = wind_north * right_north + wind_east * right_east
    along_wind = wind_north * tangent_north + wind_east * tangent_east
    horizontal_air = speed * np.cos(air_angle)
    if np.any(np.abs(crosswind) >= horizontal_air):
        raise ValueError("crosswind makes the requested air-path angle unattainable")
    ground_speed = np.sqrt(horizontal_air**2 - crosswind**2) + along_wind
    if np.any(ground_speed <= 0.0):
        raise ValueError("wind leaves no positive horizontal route progress")
    altitude_rate = speed * np.sin(air_angle) - wind_down
    return np.asarray(np.arctan2(altitude_rate, ground_speed), dtype=float)


__all__ = [
    "WindCorrectedPath",
    "ground_track_to_air_heading",
    "ground_path_angle_from_air_motion",
    "horizontal_ground_speed_mps",
    "wind_corrected_path",
]
